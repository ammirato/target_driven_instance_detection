# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np
import yaml

from .generate_anchors import generate_anchors

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from ..fast_rcnn.config import cfg
from ..fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from ..fast_rcnn.nms_wrapper import nms



from ..utils.cython_bbox import bbox_overlaps, bbox_intersections


# <<<< obsolete


DEBUG = False
"""
Outputs object detection proposals by applying estimated bounding-box
transformations to a set of regular boxes (called "anchors").
"""


def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, _feat_stride=[16, ],
                   anchor_scales=[2, 4, 8],gt_boxes=None):
    """
    Parameters
    ----------
    rpn_cls_prob_reshape: (1 , H , W , Ax2) outputs of RPN, prob of bg or fg
                         NOTICE: the old version is ordered by (1, H, W, 2, A) !!!!
    rpn_bbox_pred: (1 , H , W , Ax4), rgs boxes output of RPN
    im_info: a list of [image_height, image_width, scale_ratios]
    cfg_key: 'TRAIN' or 'TEST'
    _feat_stride: the downsampling ratio of feature map to the original input image
    anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
    ----------
    Returns
    ----------
    rpn_rois : (1 x H x W x A, 5) e.g. [0, x1, y1, x2, y2]

    # Algorithm:
    #
    # for each (H, W) location i
    #   generate A anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the A anchors
    # clip predicted boxes to image
    # remove predicted boxes with either height or width < threshold
    # sort all (proposal, score) pairs by score from highest to lowest
    # take top pre_nms_topN proposals before NMS
    # apply NMS with threshold 0.7 to remaining proposals
    # take after_nms_topN proposals after NMS
    # return the top proposals (-> RoIs top, scores top)
    #layer_params = yaml.load(self.param_str_)

    """
    _anchors = generate_anchors(scales=np.array(anchor_scales))
    _num_anchors = _anchors.shape[0]
    # rpn_cls_prob_reshape = np.transpose(rpn_cls_prob_reshape,[0,3,1,2]) #-> (1 , 2xA, H , W)
    # rpn_bbox_pred = np.transpose(rpn_bbox_pred,[0,3,1,2])              # -> (1 , Ax4, H , W)

    # rpn_cls_prob_reshape = np.transpose(np.reshape(rpn_cls_prob_reshape,[1,rpn_cls_prob_reshape.shape[0],rpn_cls_prob_reshape.shape[1],rpn_cls_prob_reshape.shape[2]]),[0,3,2,1])
    # rpn_bbox_pred = np.transpose(rpn_bbox_pred,[0,3,2,1])
    im_info = im_info[0]

    assert rpn_cls_prob_reshape.shape[0] == 1, \
        'Only single item batches are supported'
    # cfg_key = str(self.phase) # either 'TRAIN' or 'TEST'
    # cfg_key = 'TEST'
    pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
    nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
    min_size = cfg[cfg_key].RPN_MIN_SIZE

    # the first set of _num_anchors channels are bg probs
    # the second set are the fg probs, which we want
    scores = rpn_cls_prob_reshape[:, _num_anchors:, :, :]
    bbox_deltas = rpn_bbox_pred
    # im_info = bottom[2].data[0, :]

    if DEBUG:
        print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
        print 'scale: {}'.format(im_info[2])

    # 1. Generate proposals from bbox deltas and shifted anchors
    height, width = scores.shape[-2:]

    if DEBUG:
        print 'score map size: {}'.format(scores.shape)

    # Enumerate all shifts
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors
    K = shifts.shape[0]
    anchors = _anchors.reshape((1, A, 4)) + \
              shifts.reshape((1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4))

    # Transpose and reshape predicted bbox transformations to get them
    # into the same order as the anchors:
    #
    # bbox deltas will be (1, 4 * A, H, W) format
    # transpose to (1, H, W, 4 * A)
    # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
    # in slowest to fastest order
    bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

    # Same story for the scores:
    #
    # scores are (1, A, H, W) format
    # transpose to (1, H, W, A)
    # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
    scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

    # Convert anchors into proposals via bbox transformations
    proposals = bbox_transform_inv(anchors, bbox_deltas)

    # 2. clip predicted boxes to image
    proposals = clip_boxes(proposals, im_info[:2])
    #orig_props = proposals

    # 3. remove predicted boxes with either height or width < threshold
    # (NOTE: convert min_size to input image scale stored in im_info[2])
    keep = _filter_boxes(proposals, min_size * im_info[2])
    proposals = proposals[keep, :]
    scores = scores[keep]

    # # remove irregular boxes, too fat too tall
    # keep = _filter_irregular_boxes(proposals)
    # proposals = proposals[keep, :]
    # scores = scores[keep]

    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top pre_nms_topN (e.g. 6000)
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        order = order[:pre_nms_topN]
    proposals = proposals[order, :]
    scores = scores[order]
    anchor_inds = keep[order]

    # 6. apply nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)
    keep = nms(np.hstack((proposals, scores)), nms_thresh)
    if post_nms_topN > 0:
        keep = keep[:post_nms_topN]

    proposals = proposals[keep, :]
    scores = scores[keep]
    anchor_inds = anchor_inds[keep]

    assert(anchor_inds.shape[0] == scores.shape[0])

    if proposals.shape[0] == 0:
        print 'NOOOOOOOOOOOOOOOOOOOO'
        proposals = np.zeros((1,4))
        scores = np.zeros((1,1))
        anchor_inds = np.zeros(1)
    # Output rois blob
    # Our RPN implementation only supports a single input image, so all
    # batch inds are 0
    batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))



    #match anchor inds with gt boxes
    labels = -1*np.ones(anchor_inds.size)


    if gt_boxes is not None:
        
        labels.fill(0)

         #get rid of background gt_boxes
        gt_boxes = np.delete(gt_boxes,np.where(gt_boxes[:,4]==0),0)
        if gt_boxes.shape[0] == 0:
            labels.fill(0)
        else:
            # overlaps between the anchors and the gt boxes
            # overlaps (ex, gt), shape is A x G
            overlaps = bbox_overlaps(
                np.ascontiguousarray(proposals, dtype=np.float),
                np.ascontiguousarray(gt_boxes, dtype=np.float))
            argmax_overlaps = overlaps.argmax(axis=1)  # (A)
            max_overlaps = overlaps[np.arange(len(anchor_inds)), argmax_overlaps]
            gt_argmax_overlaps = overlaps.argmax(axis=0)  # G 
            gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                       np.arange(overlaps.shape[1])]
            gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

            if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                # assign bg labels first so that positive labels can clobber them
                #labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0 
                labels[max_overlaps < .2] = 0 

            # fg label: for each gt, anchor with highest overlap
            labels[gt_argmax_overlaps] = 1 
            # fg label: above threshold IOU
            #labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1 
            labels[max_overlaps >= .5] = 1 

            if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
                # assign bg labels last so that negative labels can clobber positives
                #labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0 
                labels[max_overlaps < .2] = 0 


        ## subsample negative labels if we have too many
        #num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
        #bg_inds = np.where(labels == 0)[0]
        #if len(bg_inds) > num_bg:
        #    #Phil
        #    #hard mine negatives
        #    hard_negs = np.where((rpn_scores[:,1]>1) & (labels==0))[0]

        #    disable_inds = npr.choice(
        #        bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        #    labels[disable_inds] = -1
        #    # print "was %s inds, disabling %s, now %s inds" % (
        #    # len(bg_inds), len(disable_inds), np.sum(labels == 0))

        #    #Phil
        #    #hard mine negatives
        #    #assert( np.all(labels[hard_negs] <1))
        #    #labels[hard_negs] = 0
   


    

    return blob, scores, anchor_inds, labels


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep


def _filter_irregular_boxes(boxes, min_ratio=0.2, max_ratio=5):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    rs = ws / hs
    keep = np.where((rs <= max_ratio) & (rs >= min_ratio))[0]
    return keep
