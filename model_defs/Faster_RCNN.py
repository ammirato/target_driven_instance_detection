import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
import cv2
import numpy as np
import sys


from .roi_pooling.modules.roi_pool import RoIPool
from .anchors.proposal_layer import proposal_layer as proposal_layer_py
from .anchors.anchor_target_layer import anchor_target_layer as anchor_target_layer_py
from utils import *

class RPN(torch.nn.Module):
    '''
    Target Driven Instance Detection network.

    Detects a single target object in a scene image. Fully convolutional.

    Input parameters:
        cfg: (Config) a config instance from configs/
    '''

    def __init__(self, cfg):
        super(RPN, self).__init__()
        self.cfg = cfg
        self.anchor_scales = cfg.ANCHOR_SCALES

        self.features,self._feat_stride,self.num_feature_channels = \
                                    self.get_feature_net(cfg.FEATURE_NET_NAME)

        self.conv1 = Conv2d(512,512,3,same_padding=True)
        self.score_conv = Conv2d(512, len(self.anchor_scales) * 3 * 2, 1, relu=False, same_padding=False)
        self.bbox_conv = Conv2d(512, len(self.anchor_scales) * 3 * 4, 1, relu=False, same_padding=False)

        # loss
        self.class_cross_entropy_loss = None
        self.box_regression_loss = None
        self.roi_cross_entropy_loss = None

    @property
    def loss(self):
        '''
        Get loss of last forward pass through the network     
        '''
        return self.class_cross_entropy_loss + self.box_regression_loss * 10

    def forward(self, img_data, img_info, gt_boxes=None,
                features_given=False):
        img_features = self.features(img_data)
        rpn_conv1 = self.conv1(img_features)
        rpn_cls_score = self.score_conv(rpn_conv1)


        # rpn score
        rpn_cls_score_reshape = self.reshape_layer(rpn_cls_score, 2)
        rpn_cls_prob = F.softmax(rpn_cls_score_reshape)
        rpn_cls_prob_reshape = self.reshape_layer(rpn_cls_prob, len(self.anchor_scales)*3*2)

        # rpn boxes
        rpn_bbox_pred = self.bbox_conv(rpn_conv1)

        # proposal layer
        rois = self.proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, img_info,
                                   self.cfg, self._feat_stride, self.anchor_scales,
                                   gt_boxes=gt_boxes)

        # generating training labels and build the rpn loss
        if self.training:
            assert gt_boxes is not None
            rpn_data = self.anchor_target_layer(rpn_cls_score, gt_boxes,
                                                img_info,
                                             self._feat_stride, self.anchor_scales)
            self.cross_entropy, self.loss_box = self.build_loss(rpn_cls_score_reshape, rpn_bbox_pred, rpn_data)

        return img_features, rois



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
    def anchor_target_layer(class_score, gt_boxes, img_info,
                            cfg, _feat_stride, anchor_scales):
        ''' 
        Assigns fg/bg label to anchor boxes.      


        Input parameters:
            class_score:  (torch.autograd.variable.Variable)
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
        class_score = class_score.data.cpu().numpy()
        labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
            anchor_target_layer_py(class_score, gt_boxes, img_info,
                                   cfg, _feat_stride, anchor_scales)

        labels = np_to_variable(labels, is_cuda=True, dtype=torch.LongTensor)
        bbox_targets = np_to_variable(bbox_targets, is_cuda=True)
        bbox_inside_weights = np_to_variable(bbox_inside_weights, is_cuda=True)
        bbox_outside_weights = np_to_variable(bbox_outside_weights, is_cuda=True)

        return labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


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
   












class FasterRCNN(nn.Module):
    PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
    SCALES = (600,)
    MAX_SIZE = 1000

    def __init__(self, cfg, classes=None, debug=False):
        super(FasterRCNN, self).__init__()

        self.n_classes = len(cfg.TRAIN_OBJ_IDS)
        self.cfg = cfg
        self.rpn = RPN(cfg)
        self.roi_pool = RoIPool(7, 7, 1.0/16)
        self.fc6 = FC(512 * 7 * 7, 4096)
        self.fc7 = FC(4096, 4096)
        self.score_fc = FC(4096, self.n_classes, relu=False)
        self.bbox_fc = FC(4096, self.n_classes * 4, relu=False)

        # loss
        self.cross_entropy = None
        self.loss_box = None

        # for log
        self.debug = debug

    @property
    def loss(self):
        # print self.cross_entropy
        # print self.loss_box
        # print self.rpn.cross_entropy
        # print self.rpn.loss_box
        return self.cross_entropy + self.loss_box * 10

    def forward(self, im_data, im_info, gt_boxes=None, gt_ishard=None, dontcare_areas=None):
        features, rois = self.rpn(im_data, im_info, gt_boxes, gt_ishard, dontcare_areas)

        if self.training:
            roi_data = self.proposal_target_layer(rois, gt_boxes, gt_ishard, dontcare_areas, self.n_classes)
            rois = roi_data[0]

        # roi pool
        pooled_features = self.roi_pool(features, rois)
        x = pooled_features.view(pooled_features.size()[0], -1)
        x = self.fc6(x)
        x = F.dropout(x, training=self.training)
        x = self.fc7(x)
        x = F.dropout(x, training=self.training)

        cls_score = self.score_fc(x)
        cls_prob = F.softmax(cls_score)
        bbox_pred = self.bbox_fc(x)

        if self.training:
            self.cross_entropy, self.loss_box = self.build_loss(cls_score, bbox_pred, roi_data)

        return cls_prob, bbox_pred, rois

    def build_loss(self, cls_score, bbox_pred, roi_data):
        # classification loss
        label = roi_data[1].squeeze()
        fg_cnt = torch.sum(label.data.ne(0))
        bg_cnt = label.data.numel() - fg_cnt

        # for log
        if self.debug:
            maxv, predict = cls_score.data.max(1)
            self.tp = torch.sum(predict[:fg_cnt].eq(label.data[:fg_cnt])) if fg_cnt > 0 else 0
            self.tf = torch.sum(predict[fg_cnt:].eq(label.data[fg_cnt:]))
            self.fg_cnt = fg_cnt
            self.bg_cnt = bg_cnt

        ce_weights = torch.ones(cls_score.size()[1])
        ce_weights[0] = float(fg_cnt) / bg_cnt
        ce_weights = ce_weights.cuda()
        cross_entropy = F.cross_entropy(cls_score, label, weight=ce_weights)

        # bounding box regression L1 loss
        bbox_targets, bbox_inside_weights, bbox_outside_weights = roi_data[2:]
        bbox_targets = torch.mul(bbox_targets, bbox_inside_weights)
        bbox_pred = torch.mul(bbox_pred, bbox_inside_weights)

        loss_box = F.smooth_l1_loss(bbox_pred, bbox_targets, size_average=False) / (fg_cnt + 1e-4)

        return cross_entropy, loss_box

    @staticmethod
    def proposal_target_layer(rpn_rois, gt_boxes, gt_ishard, dontcare_areas, num_classes):
        """
        ----------
        rpn_rois:  (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        gt_boxes: (G, 5) [x1 ,y1 ,x2, y2, class] int
        # gt_ishard: (G, 1) {0 | 1} 1 indicates hard
        dontcare_areas: (D, 4) [ x1, y1, x2, y2]
        num_classes
        ----------
        Returns
        ----------
        rois: (1 x H x W x A, 5) [0, x1, y1, x2, y2]
        labels: (1 x H x W x A, 1) {0,1,...,_num_classes-1}
        bbox_targets: (1 x H x W x A, K x4) [dx1, dy1, dx2, dy2]
        bbox_inside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
        bbox_outside_weights: (1 x H x W x A, Kx4) 0, 1 masks for the computing loss
        """
        rpn_rois = rpn_rois.data.cpu().numpy()
        rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = \
            proposal_target_layer_py(rpn_rois, gt_boxes, gt_ishard, dontcare_areas, num_classes)
        # print labels.shape, bbox_targets.shape, bbox_inside_weights.shape
        rois = network.np_to_variable(rois, is_cuda=True)
        labels = network.np_to_variable(labels, is_cuda=True, dtype=torch.LongTensor)
        bbox_targets = network.np_to_variable(bbox_targets, is_cuda=True)
        bbox_inside_weights = network.np_to_variable(bbox_inside_weights, is_cuda=True)
        bbox_outside_weights = network.np_to_variable(bbox_outside_weights, is_cuda=True)

        return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    def interpret_faster_rcnn(self, cls_prob, bbox_pred, rois, im_info, im_shape, nms=True, clip=True, min_score=0.0):
        # find class
        scores, inds = cls_prob.data.max(1)
        scores, inds = scores.cpu().numpy(), inds.cpu().numpy()

        keep = np.where((inds > 0) & (scores >= min_score))
        scores, inds = scores[keep], inds[keep]

        # Apply bounding-box regression deltas
        keep = keep[0]
        box_deltas = bbox_pred.data.cpu().numpy()[keep]
        box_deltas = np.asarray([
            box_deltas[i, (inds[i] * 4): (inds[i] * 4 + 4)] for i in range(len(inds))
        ], dtype=np.float)
        boxes = rois.data.cpu().numpy()[keep, 1:5] / im_info[0][2]
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        if clip:
            pred_boxes = clip_boxes(pred_boxes, im_shape)

        # nms
        if nms and pred_boxes.shape[0] > 0:
            pred_boxes, scores, inds = nms_detections(pred_boxes, scores, 0.3, inds=inds)

        return pred_boxes, scores, self.classes[inds]

    def detect(self, image, thr=0.3):
        im_data, im_scales = self.get_image_blob(image)
        im_info = np.array(
            [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
            dtype=np.float32)

        cls_prob, bbox_pred, rois = self(im_data, im_info)
        pred_boxes, scores, classes = \
            self.interpret_faster_rcnn(cls_prob, bbox_pred, rois, im_info, image.shape, min_score=thr)
        return pred_boxes, scores, classes

    def get_image_blob_noscale(self, im):
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= self.PIXEL_MEANS

        processed_ims = [im]
        im_scale_factors = [1.0]

        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

    def get_image_blob(self, im):
        """Converts an image into a network input.
        Arguments:
            im (ndarray): a color image in BGR order
        Returns:
            blob (ndarray): a data blob holding an image pyramid
            im_scale_factors (list): list of image scales (relative to im) used
                in the image pyramid
        """
        im_orig = im.astype(np.float32, copy=True)
        im_orig -= self.PIXEL_MEANS

        im_shape = im_orig.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        processed_ims = []
        im_scale_factors = []

        for target_size in self.SCALES:
            im_scale = float(target_size) / float(im_size_min)
            # Prevent the biggest axis from being more than MAX_SIZE
            if np.round(im_scale * im_size_max) > self.MAX_SIZE:
                im_scale = float(self.MAX_SIZE) / float(im_size_max)
            im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                            interpolation=cv2.INTER_LINEAR)
            im_scale_factors.append(im_scale)
            processed_ims.append(im)

        # Create a blob to hold the input images
        blob = im_list_to_blob(processed_ims)

        return blob, np.array(im_scale_factors)

