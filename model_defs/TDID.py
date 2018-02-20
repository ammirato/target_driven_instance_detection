import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import cv2
import numpy as np
import sys

from anchors.proposal_layer import proposal_layer as proposal_layer_py
from anchors.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
from utils import *

class TDID(torch.nn.Module):
    '''
    Target Driven Instance Detection network.

    Detects a single target object in a scene image. Fully convolutional.

    Input parameters:
        cfg: (Config) a config instance from configs/
    '''

    def __init__(self, cfg):
        super(TDID, self).__init__()
        self.cfg = cfg
        self.anchor_scales = cfg.ANCHOR_SCALES

        self.features,self._feat_stride,self.num_feature_channels = \
                                    self.get_feature_net(cfg.FEATURE_NET_NAME)
        self.embedding_conv = self.get_embedding_conv(cfg)
        self.corr_conv = Conv2d(cfg.NUM_TARGETS*self.num_feature_channels,
                              self.num_feature_channels, 3, 
                              relu=True, same_padding=True)
        self.diff_conv = Conv2d(cfg.NUM_TARGETS*self.num_feature_channels,
                                self.num_feature_channels, 3, 
                                relu=True, same_padding=True)
        #for getting output size of score and bbbox convs
        # 3 = number of anchor aspect ratios
        # 2 = number of classes (background, target)
        # 4 = number of bounding box parameters
        self.score_conv = Conv2d(512, len(self.anchor_scales) * 3 * 2, 1, relu=False, same_padding=False)
        self.bbox_conv = Conv2d(512, len(self.anchor_scales) * 3 * 4, 1, relu=False, same_padding=False)

        # loss
        self.class_cross_entropy_loss = None
        self.box_regression_loss = None

    @property
    def loss(self):
        '''
        Get loss of last forward pass through the network     
        '''
        return self.class_cross_entropy_loss + self.box_regression_loss * 10

    def forward(self, target_data, img_data, img_info, gt_boxes=None,
                features_given=False):
        '''
        Forward pass through TDID network.

        B = batch size
        C = number of channels
        H = height
        W = width

        Input parameters:
            target_data: (torch.FloatTensor) (B*2)xCxHxW tensor of target data 
            img_data: (torch.FloatTensor) BxCxHxW tensor of scene image data 
            img_info: (tuple) shape of original scene image
            
            gt_boxes (optional): (ndarray) ground truth bounding boxes for this
                                 scene/target pair. Must be provided for training
                                 not used for testing. Default: None
            features_given (optional): (bool) If True, target_data and img_data
                                       are assumed to be feature maps. The feature
                                       extraction portion of the forward pass
                                       is skipped. Default: False

        Returns:
            scores: (torch.autograd.variable.Variable) Bxcfg.PROPOSAL_BATCH_SIZEx1
            rois: (torch.autograd.variable.Variable) Bxcfg.PROPOSAL_BATCH_SIZEx4

        '''
        if features_given:
            img_features = img_data
            target_features = target_data 
        else:
            img_features = self.features(img_data)
            target_features = self.features(target_data)


        all_corrs = []
        all_diffs = []
        for batch_ind in range(img_features.size()[0]):
            img_ind = np_to_variable(np.asarray([batch_ind]),
                                     is_cuda=True, dtype=torch.LongTensor)
            cur_img_feats = torch.index_select(img_features,0,img_ind)

            cur_diffs = []
            cur_corrs = []
            for target_type in range(self.cfg.NUM_TARGETS):
                target_ind = np_to_variable(np.asarray([batch_ind*
                                            self.cfg.NUM_TARGETS+target_type]),
                                            is_cuda=True,dtype=torch.LongTensor)
                cur_target_feats = torch.index_select(target_features,0,
                                                      target_ind[0])
                cur_target_feats = cur_target_feats.view(-1,1,
                                                     cur_target_feats.size()[2],
                                                     cur_target_feats.size()[3])
                pooled_target_feats = F.max_pool2d(cur_target_feats,
                                         (cur_target_feats.size()[2],
                                          cur_target_feats.size()[3]))

                cur_diffs.append(cur_img_feats -
                    pooled_target_feats.permute(1,0,2,3).expand_as(cur_img_feats))
                if self.cfg.CORR_WITH_POOLED:
                    cur_corrs.append(F.conv2d(cur_img_feats,
                                             pooled_target_feats,
                                             groups=self.num_feature_channels))
                else:
                    target_conv_padding = (max(0,int(
                                          target_features.size()[2]/2)), 
                                           max(0,int(
                                           target_features.size()[3]/2)))
                    cur_corrs.append(F.conv2d(cur_img_feats,cur_target_feats,
                                             padding=target_conv_padding,
                                             groups=self.num_feature_channels))
                

            cur_corrs = torch.cat(cur_corrs,1)
            cur_corrs = self.select_to_match_dimensions(cur_corrs,cur_img_feats)
            all_corrs.append(cur_corrs)
            all_diffs.append(torch.cat(cur_diffs,1))

        corr = self.corr_conv(torch.cat(all_corrs,0))
        diff = self.diff_conv(torch.cat(all_diffs,0))
      
        if self.cfg.USE_IMG_FEATS and self.cfg.USE_DIFF_FEATS:
            if self.cfg.USE_CC_FEATS: 
                concat_feats = torch.cat([corr,img_features, diff],1) 
            else:
                concat_feats = torch.cat([img_features, diff],1) 
        elif self.cfg.USE_IMG_FEATS:
            if self.cfg.USE_CC_FEATS: 
                concat_feats = torch.cat([corr,img_features],1) 
            else:
                concat_feats = torch.cat([img_features],1) 
        elif self.cfg.USE_DIFF_FEATS:
            if self.cfg.USE_CC_FEATS: 
                concat_feats = torch.cat([corr,diff],1) 
            else:
                concat_feats = torch.cat([diff],1) 
        else:
            concat_feats = corr 

        embedding_feats = self.embedding_conv(concat_feats)
        class_score = self.score_conv(embedding_feats)
        class_score_reshape = self.reshape_layer(class_score, 2)
        class_prob = F.softmax(class_score_reshape)
        class_prob_reshape = self.reshape_layer(class_prob, len(self.anchor_scales)*3*2)

        bbox_pred = self.bbox_conv(embedding_feats)

        # proposal layer
        rois, scores, anchor_inds, labels = self.proposal_layer(class_prob_reshape, 
                                                               bbox_pred,
                                                               img_info,
                                                               self.cfg,
                                                               self._feat_stride, 
                                                               self.anchor_scales,
                                                               gt_boxes)
    
        if self.training:
            assert gt_boxes is not None
            anchor_data = self.anchor_target_layer(class_score,gt_boxes, 
                                                img_info, self.cfg,
                                                self._feat_stride, self.anchor_scales)
            self.cross_entropy, self.loss_box = self.build_loss(class_prob_reshape, bbox_pred, anchor_data)

        return scores, rois



    def build_loss(self, class_score_reshape, bbox_pred, anchor_data):
        '''
        Compute loss of a batch from a single forward pass
    
        Input parameters:
            class_score_reshape: (torch.FloatTensor)
            bbox_pred: (torch.FloatTensor)
            anchor_data: (ndarray)

        Returns:
            cross_entropy: (torch.autograd.variable.Variable) classifcation loss
            loss_box: (torch.autograd.variable.Variable) bbox regression loss

        '''
        # classification loss
        class_score = class_score_reshape.permute(0, 2, 3, 1).contiguous().view(-1, 2)

        anchor_label = anchor_data[0].view(-1)
        keep = Variable(anchor_label.data.ne(-1).nonzero().squeeze()).cuda()
        class_score = torch.index_select(class_score, 0, keep)
        anchor_label = torch.index_select(anchor_label, 0, keep)

        fg_cnt = torch.sum(anchor_label.data.ne(0))

        # box loss
        bbox_targets = anchor_data[1]
        bbox_inside_weights = anchor_data[2]
        bbox_outside_weights = anchor_data[3]
        bbox_targets = torch.mul(bbox_targets, bbox_inside_weights)
        bbox_pred = torch.mul(bbox_pred, bbox_inside_weights)

        cross_entropy = F.cross_entropy(class_score,anchor_label, size_average=False)
        loss_box = F.smooth_l1_loss(bbox_pred, bbox_targets, size_average=False) / (fg_cnt + 1e-4)
        return cross_entropy, loss_box


    @staticmethod
    def reshape_layer(x, d):
        '''
        Reshape a tensor to have second dimension d, changing 3rd dimension

        Input parameters:
            x: (torch.autograd.variable.Variable) 
            d: (int)

        Returns:
            (torch.autograd.variable.Variable)

        '''

        input_shape = x.size()
        # b c w h
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x

    @staticmethod
    def select_to_match_dimensions(a,b):
        '''
        Select elements from first tensor so it's size matches second tensor.

        Input parameters:
            a: (torch.autograd.variable.Variable)
            b: (torch.autograd.variable.Variable)

        Returns:
            (torch.autograd.variable.Variable)
        
        '''

        if a.size()[2] > b.size()[2]:
            a = torch.index_select(a, 2, 
                                  np_to_variable(np.arange(0,
                                        b.size()[2]).astype(np.int32),
                                         is_cuda=True,dtype=torch.LongTensor))
        if a.size()[3] > b.size()[3]:
            a = torch.index_select(a, 3, 
                                  np_to_variable(np.arange(0,
                                    b.size()[3]).astype(np.int32),
                                          is_cuda=True,dtype=torch.LongTensor))
        return a 


    @staticmethod
    def proposal_layer(class_prob_reshape, bbox_pred, img_info, cfg, _feat_stride, anchor_scales, gt_boxes=None):
        '''
        Get top scoring detections
 
        Wrapper for proposal_layer_py. 

        Input parameters:
            class_prob_reshape: (torch.autograd.variable.Variable)
            bbox_pred: (torch.autograd.variable.Variable)
            img_info: (tuple)
            cfg: (Config) from ../configs
            _feat_stride:  (int)
            anchor_scales: (list of int)
            
            gt_boxes (optional): (ndarray) Defatul: None
                        
        
        '''
        
        #convert to  numpy
        class_prob_reshape = class_prob_reshape.data.cpu().numpy()
        bbox_pred = bbox_pred.data.cpu().numpy()

        rois, scores, anchor_inds, labels = proposal_layer_py(
                                                       class_prob_reshape,
                                                       bbox_pred,
                                                       img_info, cfg, 
                                                       _feat_stride=_feat_stride,
                                                       anchor_scales=anchor_scales,
                                                       gt_boxes=gt_boxes)
        #convert to pytorch
        rois = np_to_variable(rois, is_cuda=True)
        anchor_inds = np_to_variable(anchor_inds, is_cuda=True,
                                                 dtype=torch.LongTensor)
        labels = np_to_variable(labels, is_cuda=True,
                                             dtype=torch.LongTensor)
        scores = np_to_variable(scores, is_cuda=True)
        return rois, scores, anchor_inds, labels


    @staticmethod
    def anchor_target_layer(cls_score, gt_boxes, img_info,
                            cfg, _feat_stride, anchor_scales):
        ''' 
        Assigns fg/bg label to anchor boxes.      


        Input parameters:
            cls_score:  (torch.autograd.variable.Variable)
            gt_boxes:  (ndarray)
            img_info:  (tuple of int)
            cfg: (Config) from ../configs
            _feat_stride:  (int)
            anchor_scales: (list of int)

        Returns:
            labels: (torch.autograd.variable.Variable)
            bbox_targets: (torch.autograd.variable.Variable)
            bbox_inside_weights:(torch.autograd.variable.Variable)
            bbox_outside_weights:(torch.autograd.variable.Variable)
        ''' 
        cls_score = cls_score.data.cpu().numpy()
        labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
            anchor_target_layer_py(cls_score, gt_boxes, img_info,
                                   cfg, _feat_stride, anchor_scales)

        labels = np_to_variable(labels, is_cuda=True, dtype=torch.LongTensor)
        bbox_targets = np_to_variable(bbox_targets, is_cuda=True)
        bbox_inside_weights = np_to_variable(bbox_inside_weights, is_cuda=True)
        bbox_outside_weights = np_to_variable(bbox_outside_weights, is_cuda=True)

        return labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def get_features(self, img_data):
        img_data = np_to_variable(img_data, is_cuda=True)
        img_data = img_data.permute(0, 3, 1, 2)
        features = self.features(img_data)

        return features


    @staticmethod
    def get_feature_net(net_name):
        '''
        Get the object representing the desired feature extraction network

        Note: only the part of the network considered useful for feature
              extraction is returned. i.e. everythnig but the fully
              connected layers of AlexNet.

        Input parameters:
            net_name: (str) the name of the desired network


        Availble net names:
            vgg16_bn
            squeezenet1_1
            resenet101
            alexnet
        '''
        if net_name == 'vgg16_bn':
            fnet = models.vgg16_bn(pretrained=False)
            return torch.nn.Sequential(*list(fnet.features.children())[:-1]), 16, 512
        elif net_name == 'squeezenet1_1':
            fnet = models.squeezenet1_1(pretrained=False)
            return torch.nn.Sequential(*list(fnet.features.children())[:-1]), 16, 512 
        elif net_name == 'resnet101':
            fnet = models.resnet101(pretrained=False)
            return torch.nn.Sequential(*list(fnet.children())[:-2]), 32, 2048 
        elif net_name == 'alexnet':
            fnet = models.alexnet(pretrained=False)
            return  torch.nn.Sequential(*list(fnet.features.children())), 17, 256
        else:
            raise NotImplementedError
   
    def get_embedding_conv(self,cfg):
        '''
        Get a Conv2D layer for the TDID embedding based on the config paprams

        Input parameters:
            cfg: (Config) from ../configs/
        
        '''
        if cfg.USE_IMG_FEATS and cfg.USE_DIFF_FEATS: 
            if cfg.USE_CC_FEATS:
                return Conv2d(3*self.num_feature_channels,
                                512, 3, relu=False, same_padding=True)
            else:
                return Conv2d(2*self.num_feature_channels,
                                512, 3, relu=False, same_padding=True)
        elif cfg.USE_IMG_FEATS:
            if cfg.USE_CC_FEATS:
                return Conv2d(2*self.num_feature_channels,
                                512, 3, relu=False, same_padding=True)
            else:
                return Conv2d(self.num_feature_channels,
                                512, 3, relu=False, same_padding=True)
        elif cfg.USE_DIFF_FEATS:
            if cfg.USE_CC_FEATS:
                return Conv2d(2*self.num_feature_channels,
                                512, 3, relu=False, same_padding=True)
            else:
                return Conv2d(self.num_feature_channels,
                                512, 3, relu=False, same_padding=True)
        else:
            return Conv2d(self.num_feature_channels,
                            512, 3, relu=False, same_padding=True)

 
