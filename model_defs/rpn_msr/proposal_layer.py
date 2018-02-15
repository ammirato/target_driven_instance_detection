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
from ..fast_rcnn.bbox_transform_batch import bbox_transform_inv, clip_boxes
from ..fast_rcnn.nms_wrapper import nms



from ..utils.cython_bbox import bbox_overlaps, bbox_intersections


# <<<< obsolete


DEBUG = False
"""
Outputs object detection proposals by applying estimated bounding-box
transformations to a set of regular boxes (called "anchors").
"""


def proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg, _feat_stride=[16, ],
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
    batch_size = rpn_cls_prob_reshape.shape[0]
    _anchors = generate_anchors(scales=np.array(anchor_scales))
    _num_anchors = _anchors.shape[0]
    # rpn_cls_prob_reshape = np.transpose(rpn_cls_prob_reshape,[0,3,1,2]) #-> (1 , 2xA, H , W)
    # rpn_bbox_pred = np.transpose(rpn_bbox_pred,[0,3,1,2])              # -> (1 , Ax4, H , W)

    # rpn_cls_prob_reshape = np.transpose(np.reshape(rpn_cls_prob_reshape,[1,rpn_cls_prob_reshape.shape[0],rpn_cls_prob_reshape.shape[1],rpn_cls_prob_reshape.shape[2]]),[0,3,2,1])
    # rpn_bbox_pred = np.transpose(rpn_bbox_pred,[0,3,2,1])
    #im_info = im_info[0]

#    assert rpn_cls_prob_reshape.shape[0] == 1, \
#        'Only single item batches are supported'


    pre_nms_topN = cfg.PRE_NMS_TOP_N
    post_nms_topN = cfg.POST_NMS_TOP_N
    nms_thresh = cfg.NMS_THRESH
    min_size = cfg.PROPOSAL_MIN_BOX_SIZE

    # the first set of _num_anchors channels are bg probs
    # the second set are the fg probs, which we want
    scores = rpn_cls_prob_reshape[:, _num_anchors:, :, :]
    bbox_deltas = rpn_bbox_pred
    # im_info = bottom[2].data[0, :]

    if DEBUG:
        print( 'im_size: ({}, {})'.format(im_info[0], im_info[1]) )
        print( 'scale: {}'.format(im_info[2]))

    # 1. Generate proposals from bbox deltas and shifted anchors
    height, width = scores.shape[-2:]

    if DEBUG:
        print('score map size: {}'.format(scores.shape) )

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
    anchors = np.tile(anchors, (batch_size,1,1))

    # Transpose and reshape predicted bbox transformations to get them
    # into the same order as the anchors:
    #
    # bbox deltas will be (1, 4 * A, H, W) format
    # transpose to (1, H, W, 4 * A)
    # reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
    # in slowest to fastest order
    bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((batch_size,-1, 4))

    # Same story for the scores:
    #
    # scores are (1, A, H, W) format
    # transpose to (1, H, W, A)
    # reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
    #scores = scores.transpose((0, 2, 3, 1)).reshape((batch_size,-1, 1))
    scores = scores.transpose((0, 2, 3, 1)).reshape((batch_size,-1))

    # Convert anchors into proposals via bbox transformations
    proposals = bbox_transform_inv(anchors, bbox_deltas)

    # 2. clip predicted boxes to image
    proposals = clip_boxes(proposals, im_info[:2])
    #orig_props = proposals

    prop_info = []


    # 3. remove predicted boxes with either height or width < threshold
    # (NOTE: convert min_size to input image scale stored in im_info[2])
    #keep = _filter_boxes(proposals, min_size * im_info[2])
    #proposals = proposals[keep, :]
    #scores = scores[keep]
    lose = _filter_boxes(proposals, min_size * im_info[2])
    proposals[lose[0],lose[1],:] = 0
    scores[lose[0],lose[1]] = 0

    # # remove irregular boxes, too fat too tall
    # keep = _filter_irregular_boxes(proposals)
    # proposals = proposals[keep, :]
    # scores = scores[keep]

    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top pre_nms_topN (e.g. 6000)
    #order = scores.ravel().argsort()[::-1]
    order = scores.argsort(1)[:,::-1]
    anchor_inds = np.tile(np.arange(order.shape[1]),(batch_size,1))
    if pre_nms_topN > 0:
        order = order[:,:pre_nms_topN]
    #proposals = proposals[order, :]
    #scores = scores[order]
    b_select = np.arange(batch_size)
    proposals = np.take(proposals,order,axis=1)[b_select,b_select,:,:]
    scores = np.take(scores,order,axis=1)[b_select,b_select,:]
    anchor_inds = np.take(anchor_inds,order,axis=1)[b_select,b_select,:]
    #anchor_inds = keep[order]
    #anchor_inds = np.arange[order]



    all_proposals = None
    all_scores = None
    all_anchor_inds = None
    all_labels = None

    for batch_ind in range(batch_size):

        b_proposals = proposals[batch_ind,:,:]
        b_scores = np.expand_dims(scores[batch_ind,:], 1)
        b_anchor_inds = (np.expand_dims(anchor_inds[batch_ind,:],1) + 
                         batch_ind*anchors.shape[1])

        # 6. apply nms (e.g. threshold = 0.7)
        # 7. take after_nms_topN (e.g. 300)
        # 8. return the top proposals (-> RoIs top)
        keep = nms(np.hstack((b_proposals, b_scores)), nms_thresh)
        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]

        b_proposals = b_proposals[keep, :]
        b_scores = b_scores[keep]
        b_anchor_inds = b_anchor_inds[keep]

        assert(b_anchor_inds.shape[0] == b_scores.shape[0])

        if b_proposals.shape[0] == 0:
            b_proposals = np.zeros((1,4))
            b_scores = np.zeros((1,1))
            b_anchor_inds = np.zeros(1)

        #batch_inds = np.zeros((b_proposals.shape[0], 1), dtype=np.float32) + batch_ind
        #blob = np.hstack((batch_inds, b_proposals.astype(np.float32, copy=False)))



        #match anchor inds with gt boxes
        b_labels = -1*np.ones(b_anchor_inds.size)

        if gt_boxes is not None:
            
            b_labels.fill(0)

             #get rid of background gt_boxes
            gt_box = np.expand_dims(gt_boxes[batch_ind,:],axis=0)
            if gt_box[0,-1] == 0:#this is a bg box
                b_labels.fill(0)
            else:
                # overlaps between the anchors and the gt boxes
                # overlaps (ex, gt), shape is A x G
                overlaps = bbox_overlaps(
                    np.ascontiguousarray(b_proposals, dtype=np.float),
                    np.ascontiguousarray(gt_box, dtype=np.float))
                argmax_overlaps = overlaps.argmax(axis=1)  # (A)
                max_overlaps = overlaps[np.arange(len(b_anchor_inds)), argmax_overlaps]
                gt_argmax_overlaps = overlaps.argmax(axis=0)  # G 
                gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                           np.arange(overlaps.shape[1])]
                gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

                if not cfg.PROPOSAL_CLOBBER_POSITIVES:
                    # assign bg labels first so that positive labels can clobber them
                    #labels[max_overlaps < cfg.TRAIN.PROPOSAL_NEGATIVE_OVERLAP] = 0 
                    b_labels[max_overlaps < .2] = 0 

                # fg label: for each gt, anchor with highest overlap
                b_labels[gt_argmax_overlaps] = 1 
                # fg label: above threshold IOU
                #labels[max_overlaps >= cfg.TRAIN.PROPOSAL_POSITIVE_OVERLAP] = 1 
                b_labels[max_overlaps >= .5] = 1 

                if True:#cfg.TRAIN.PROPOSAL_CLOBBER_POSITIVES:
                    # assign bg labels last so that negative labels can clobber positives
                    #labels[max_overlaps < cfg.TRAIN.PROPOSAL_NEGATIVE_OVERLAP] = 0 
                    b_labels[max_overlaps < .2] = 0 
        if all_proposals is None:
            all_proposals = np.expand_dims(b_proposals, axis=0)
            all_scores = np.expand_dims(b_scores, axis=0)
            all_anchor_inds = np.expand_dims(b_anchor_inds, axis=0)
            all_labels = np.expand_dims(b_labels, axis=0)
        else:
            all_proposals = _append_and_pad(all_proposals,b_proposals)
            all_scores = _append_and_pad(all_scores,b_scores)
            all_anchor_inds = _append_and_pad(all_anchor_inds,b_anchor_inds)
            all_labels = _append_and_pad(all_labels,b_labels)



    return all_proposals, all_scores,all_anchor_inds,all_labels 



def _append_and_pad(all_batches, single_batch):
    """ appends a1 to a2 at axis 0, padding the shorter of a1,a2"""
    if all_batches.shape[1] < single_batch.shape[0]:
        num_to_add = single_batch.shape[0] - all_batches.shape[1]
        all_batches = _pad_to_match(all_batches, num_to_add, axis=1)
    elif all_batches.shape[1] > single_batch.shape[0]:
        num_to_add = all_batches.shape[1] - single_batch.shape[0]
        single_batch = _pad_to_match(single_batch,num_to_add, axis=0)
    single_batch = np.expand_dims(single_batch,0)

    return np.concatenate((all_batches,single_batch))


def _pad_to_match(to_pad, num_to_add, axis=0):
        pad_dims = []
        for dim, dim_size in enumerate(to_pad.shape):
            if (dim==axis):
                pad_dims.append(num_to_add)
            else: 
                pad_dims.append(dim_size)
        padding = np.zeros(pad_dims)
        return np.concatenate((to_pad,padding),axis=axis)
    


def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    #ws = boxes[:, 2] - boxes[:, 0] + 1
    #hs = boxes[:, 3] - boxes[:, 1] + 1
    #keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    ws = boxes[:,:, 2] - boxes[:,:, 0] + 1
    hs = boxes[:,:, 3] - boxes[:,:, 1] + 1
    lose = np.where((ws < min_size) & (hs < min_size))
    return lose 


def _filter_irregular_boxes(boxes, min_ratio=0.2, max_ratio=5):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    rs = ws / hs
    keep = np.where((rs <= max_ratio) & (rs >= min_ratio))[0]
    return keep


