import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

import cv2
import numpy as np
import sys

from utils.timer import Timer
#from rpn_msr.anchor_target_layer_HN import anchor_target_layer as anchor_target_layer_py

import network
from network import Conv2d, FC


class TDID(nn.Module):
    groups =256

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


        #self.fc1 = FC(9216*2,4096)
        #self.fc2 = FC(4096,4096)
        #self.fc3 = FC(4096,2)
        self.fc1 = torch.nn.Linear(9216*2,4096)
        self.fc2 = torch.nn.Linear(4096,4096)
        self.fc3 = torch.nn.Linear(4096,2)

        #self.class_conv = Conv2d(self.conv1.conv.out_channels,
        #                        2,13,relu=False,same_padding=False)
        # loss
        self.roi_cross_entropy = None
        self.cross_entropy = None
        self.loss_box = None

    @property
    def loss(self):
        return self.cross_entropy

    def forward(self, target_data, im_data, gt_cids=None, features_given=False, im_info=None, batch_test=False):


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
            img_ind = network.np_to_variable(np.asarray([b_ind]),
                                                is_cuda=True, dtype=torch.LongTensor)
            sample_img = torch.index_select(img_features,0,img_ind)

            diff = []
            cc = []
            for t_ind in range(self.cfg.NUM_TARGETS):
                target_ind = network.np_to_variable(np.asarray([b_ind*2+t_ind]),
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
        rpn_conv1 = rpn_conv1.view(rpn_conv1.size()[0], -1)
        
        fc = F.dropout(rpn_conv1, training=self.training) 
        fc = F.relu(self.fc1(fc))
        fc = F.dropout(fc, training=self.training) 
        fc = F.relu(self.fc2(fc))
        scores = self.fc3(fc) 

 
        #scores = self.class_conv(rpn_conv1)
        #scores = scores.squeeze(2).squeeze(2)
    
        if self.training:
            assert gt_cids is not None
            self.cross_entropy = F.cross_entropy(scores, gt_cids, size_average=False)

        scores = F.softmax(scores)
        return scores



    def forward_test_batch(self, target_data, im_data, gt_cids=None, features_given=False, im_info=None):
        '''
        Gives output for many different targets on one scene image at once
        '''

        assert features_given
        assert self.cfg.CORR_WITH_POOLED

        img_features = im_data
        target_features = target_data 

        ccs = []
        diffs = []
        #for b_ind in range(img_features.size()[0]):
        img_ind = network.np_to_variable(np.asarray([0]),
                                            is_cuda=True, dtype=torch.LongTensor)
        sample_img = torch.index_select(img_features,0,img_ind)

        diff = []
        cc = []
        for t_ind in range(self.cfg.NUM_TARGETS):
            target_ind = network.np_to_variable(np.asarray([b_ind*2+t_ind]),
                                                is_cuda=True, dtype=torch.LongTensor)
            sample_target = torch.index_select(target_features,0,target_ind[0])

            sample_target = sample_target.view(-1,1,sample_target.size()[2], 
                                               sample_target.size()[3])
            tf_pooled = F.max_pool2d(sample_target,(sample_target.size()[2],
                                                       sample_target.size()[3]))

            diff.append(sample_img - tf_pooled.permute(1,0,2,3).expand_as(sample_img))
            cc.append(F.conv2d(sample_img,tf_pooled,groups=self.groups))
            

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
        scores = self.class_conv(rpn_conv1)
        scores = scores.squeeze(2).squeeze(2)
    
        if self.training:
            assert gt_cids is not None
            self.cross_entropy = F.cross_entropy(scores, gt_cids, size_average=False)

        return scores











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
                                  network.np_to_variable(np.arange(0,
                                        b.size()[2]).astype(np.int32),
                                         is_cuda=True,dtype=torch.LongTensor))
        if a.size()[3] > b.size()[3]:
            a = torch.index_select(a, 3, 
                                  network.np_to_variable(np.arange(0,
                                    b.size()[3]).astype(np.int32),
                                          is_cuda=True,dtype=torch.LongTensor))
        return a 




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
            #return  torch.nn.Sequential(*list(fnet.features.children())[:-1]), 32, 256 
            return  torch.nn.Sequential(*list(fnet.features.children())), 64, 256 
        else:
            print 'feature net type not supported!'
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
        

 
