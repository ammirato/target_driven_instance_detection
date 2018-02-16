import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

import cv2
import numpy as np
import sys

from target_driven_instance_detection.model_defs.anchors.proposal_layer import proposal_layer as proposal_layer_py
from target_driven_instance_detection.model_defs.anchors.anchor_target_layer import anchor_target_layer as anchor_target_layer_py

from target_driven_instance_detection.utils import *

class TDID(nn.Module):
    groups=512

    def __init__(self, cfg):
        super(TDID, self).__init__()
        self.cfg = cfg
        self.anchor_scales = cfg.ANCHOR_SCALES

        self.features,self._feat_stride,self.num_feature_channels = \
                                    self.get_feature_net(cfg.FEATURE_NET_NAME)

        self.groups = self.num_feature_channels
        self.conv1 = self.get_conv1(cfg)
        self.cc_conv = Conv2d(cfg.NUM_TARGETS*self.num_feature_channels,
                              self.num_feature_channels, 3, 
                              relu=True, same_padding=True)
        self.diff_conv = Conv2d(cfg.NUM_TARGETS*self.num_feature_channels,
                                self.num_feature_channels, 3, 
                                relu=True, same_padding=True)
        self.score_conv = Conv2d(512, len(self.anchor_scales) * 3 * 2, 1, relu=False, same_padding=False)
        self.bbox_conv = Conv2d(512, len(self.anchor_scales) * 3 * 4, 1, relu=False, same_padding=False)

        # loss
        self.roi_cross_entropy = None
        self.cross_entropy = None
        self.loss_box = None

    @property
    def loss(self):
        return self.cross_entropy + self.loss_box * 10

    def forward(self, target_data, im_data, gt_boxes=None, features_given=False, im_info=None):

        if not features_given:
            #get image features 
            img_features = self.features(im_data)
            target_features = self.features(target_data)
        else:
            img_features = im_data
            target_features = target_data 

        padding = (max(0,int(target_features.size()[2]/2)), 
                         max(0,int(target_features.size()[3]/2)))
        ccs = []
        diffs = []
        for b_ind in range(img_features.size()[0]):
            img_ind = np_to_variable(np.asarray([b_ind]),
                                                is_cuda=True, dtype=torch.LongTensor)
            sample_img = torch.index_select(img_features,0,img_ind)

            diff = []
            cc = []
            for t_ind in range(self.cfg.NUM_TARGETS):
                target_ind = np_to_variable(np.asarray([b_ind*2+t_ind]),
                                                    is_cuda=True, dtype=torch.LongTensor)
                sample_target = torch.index_select(target_features,0,target_ind[0])

                sample_target = sample_target.view(-1,1,sample_target.size()[2], 
                                                   sample_target.size()[3])
                tf_pooled = F.max_pool2d(sample_target,(sample_target.size()[2],
                                                           sample_target.size()[3]))

                diff.append(sample_img - tf_pooled.permute(1,0,2,3).expand_as(sample_img))
                if self.cfg.CORR_WITH_POOLED:
                    cc.append(F.conv2d(sample_img,tf_pooled,groups=self.groups))
                else:
                    cc.append(F.conv2d(sample_img,sample_target,padding=padding,groups=self.groups))
                

            cc = torch.cat(cc,1)
            cc = self.select_to_match_dimensions(cc,sample_img)
            ccs.append(cc)
            diffs.append(torch.cat(diff,1))

        cc = torch.cat(ccs,0)
        cc = self.cc_conv(cc)
        diffs = torch.cat(diffs,0) 
        diffs = self.diff_conv(diffs)
      
        if self.cfg.USE_IMG_FEATS and self.cfg.USE_DIFF_FEATS:
            if self.cfg.USE_CC_FEATS: 
                cc = torch.cat([cc,img_features, diffs],1) 
            else:
                cc = torch.cat([img_features, diffs],1) 
        elif self.cfg.USE_IMG_FEATS:
            if self.cfg.USE_CC_FEATS: 
                cc = torch.cat([cc,img_features,],1) 
            else:
                cc = torch.cat([img_features,],1) 
        elif self.cfg.USE_DIFF_FEATS:
            if self.cfg.USE_CC_FEATS: 
                cc = torch.cat([cc,diffs],1) 
            else:
                cc = torch.cat([diffs],1) 
        else:
            cc = cc 

        rpn_conv1 = self.conv1(cc)
        rpn_cls_score = self.score_conv(rpn_conv1)
        rpn_cls_score_reshape = self.reshape_layer(rpn_cls_score, 2)
        rpn_cls_prob = F.softmax(rpn_cls_score_reshape)
        rpn_cls_prob_reshape = self.reshape_layer(rpn_cls_prob, len(self.anchor_scales)*3*2)

        rpn_bbox_pred = self.bbox_conv(rpn_conv1)

        # proposal layer
        rois,scores, anchor_inds, labels = self.proposal_layer(rpn_cls_prob_reshape, 
                                                               rpn_bbox_pred,
                                                               im_info,
                                                               self.cfg,
                                                               self._feat_stride, 
                                                               self.anchor_scales,
                                                               gt_boxes)
    
        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None
            rpn_data = self.anchor_target_layer(rpn_cls_score,gt_boxes, 
                                                im_info, self.cfg,
                                                self._feat_stride, self.anchor_scales)
            self.cross_entropy, self.loss_box = self.build_loss(rpn_cls_score_reshape, rpn_bbox_pred, rpn_data)

        #return target_features, features, rois, scores
        bbox_pred = []
        for il in range(len(rois)):
            bbox_pred.append(np_to_variable(np.zeros((rois[il].size()[0],8))))
        return scores, bbox_pred, rois



    def build_loss(self, rpn_cls_score_reshape, rpn_bbox_pred, rpn_data):
        # classification loss
        rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(-1, 2)

        rpn_label = rpn_data[0].view(-1)
        rpn_keep = Variable(rpn_label.data.ne(-1).nonzero().squeeze()).cuda()
        rpn_cls_score = torch.index_select(rpn_cls_score, 0, rpn_keep)
        rpn_label = torch.index_select(rpn_label, 0, rpn_keep)

        fg_cnt = torch.sum(rpn_label.data.ne(0))

        # box loss
        rpn_bbox_targets = rpn_data[1]
        rpn_bbox_inside_weights = rpn_data[2]
        rpn_bbox_outside_weights = rpn_data[3]
        rpn_bbox_targets = torch.mul(rpn_bbox_targets, rpn_bbox_inside_weights)
        rpn_bbox_pred = torch.mul(rpn_bbox_pred, rpn_bbox_inside_weights)

        rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label, size_average=False)
        rpn_loss_box = F.smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, size_average=False) / (fg_cnt + 1e-4)
        return rpn_cross_entropy, rpn_loss_box


    @staticmethod
    def reshape_layer(x, d):
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
    def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg, _feat_stride, anchor_scales, gt_boxes=None):
        
        #convert to  numpy
        rpn_cls_prob_reshape = rpn_cls_prob_reshape.data.cpu().numpy()
        rpn_bbox_pred = rpn_bbox_pred.data.cpu().numpy()

        rois, scores, anchor_inds, labels = proposal_layer_py(rpn_cls_prob_reshape,
                                                               rpn_bbox_pred,
                                                      im_info, cfg, 
                                                      _feat_stride=_feat_stride,
                                                      anchor_scales=anchor_scales,
                                                      gt_boxes=gt_boxes)
        rois = np_to_variable(rois, is_cuda=True)
        anchor_inds = np_to_variable(anchor_inds, is_cuda=True,
                                                 dtype=torch.LongTensor)
        labels = np_to_variable(labels, is_cuda=True,
                                             dtype=torch.LongTensor)
        #just get fg scores, make bg scores 0 
        scores = np_to_variable(scores, is_cuda=True)
        return rois, scores, anchor_inds, labels


    @staticmethod
    def anchor_target_layer(rpn_cls_score, gt_boxes, im_info,
                            cfg, _feat_stride, anchor_scales):
        """
        rpn_cls_score: for pytorch (1, Ax2, H, W) bg/fg scores of previous conv layer
        gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
        gt_ishard: (G, 1), 1 or 0 indicates difficult or not
        dontcare_areas: (D, 4), some areas may contains small objs but no labelling. D may be 0
        im_info: a list of [image_height, image_width, scale_ratios]
        _feat_stride: the downsampling ratio of feature map to the original input image
        anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
        ----------
        Returns
        ----------
        rpn_labels : (1, 1, HxA, W), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
        rpn_bbox_targets: (1, 4xA, H, W), distances of the anchors to the gt_boxes(may contains some transform)
                        that are the regression objectives
        rpn_bbox_inside_weights: (1, 4xA, H, W) weights of each boxes, mainly accepts hyper param in cfg
        rpn_bbox_outside_weights: (1, 4xA, H, W) used to balance the fg/bg,
        beacuse the numbers of bgs and fgs mays significiantly different
        """
        rpn_cls_score = rpn_cls_score.data.cpu().numpy()
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = \
            anchor_target_layer_py(rpn_cls_score, gt_boxes, im_info,
                                   cfg, _feat_stride, anchor_scales)

        rpn_labels = np_to_variable(rpn_labels, is_cuda=True, dtype=torch.LongTensor)
        rpn_bbox_targets = np_to_variable(rpn_bbox_targets, is_cuda=True)
        rpn_bbox_inside_weights = np_to_variable(rpn_bbox_inside_weights, is_cuda=True)
        rpn_bbox_outside_weights = np_to_variable(rpn_bbox_outside_weights, is_cuda=True)

        return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

    def get_features(self, im_data):
        im_data = np_to_variable(im_data, is_cuda=True)
        im_data = im_data.permute(0, 3, 1, 2)
        features = self.features(im_data)

        return features


    @staticmethod
    def get_feature_net(net_name):
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
            print('feature net type not supported!')
            sys.exit() 
   
    def get_conv1(self,cfg):
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
         
        Conv2d(3*self.num_feature_channels,
                            512, 3, relu=False, same_padding=True)
