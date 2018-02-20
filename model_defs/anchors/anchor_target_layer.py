# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------
# Edited by Phil Ammirato, UNC-Chapel Hill

import os
import yaml
import numpy as np
import numpy.random as npr

from .generate_anchors import generate_anchors
from .cython_bbox import bbox_overlaps, bbox_intersections
from .bbox_transform import bbox_transform

def anchor_target_layer(cls_score, gt_boxes, im_info, cfg, _feat_stride=[16, ],
                        anchor_scales=[4, 8, 16, 32]):
    ''' 
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    Parameters
    ----------
    cls_score: for pytorch (1, Ax2, H, W) bg/fg scores of previous conv layer
    gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
    im_info: a list of [image_height, image_width, scale_ratios]
    _feat_stride: the downsampling ratio of feature map to the original input image
    anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
    ----------
    Returns
    ----------
    labels : (HxWxA, 1), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
    bbox_targets: (HxWxA, 4), distances of the anchors to the gt_boxes(may contains some transform)
                            that are the regression objectives
    bbox_inside_weights: (HxWxA, 4) weights of each boxes, mainly accepts hyper param in cfg
    bbox_outside_weights: (HxWxA, 4) used to balance the fg/bg,
                            beacuse the numbers of bgs and fgs mays significiantly different
    ''' 
    # Algorithm:
    #
    # for each (H, W) location i
    #   generate 9 anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the 9 anchors
    # filter out-of-image anchors
    # measure GT overlap

    batch_size = cls_score.shape[0]

    _anchors = generate_anchors(scales=np.array(anchor_scales))
    _num_anchors = _anchors.shape[0]

    # allow boxes to sit over the edge by a small amount
    _allowed_border = 0

    # map of shape (..., H, W)
    # pytorch (bs, c, h, w)
    height, width = cls_score.shape[2:4]

    # 1. Generate proposals from bbox deltas and shifted anchors
    shift_x = np.arange(0, width) * _feat_stride
    shift_y = np.arange(0, height) * _feat_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)  # in W H order
    # K is H x W
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors
    K = shifts.shape[0]
    all_anchors = (_anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))
    total_anchors = int(K * A)

    # only keep anchors inside the image
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
    )[0]

    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]
    #add one set of anchors for each batch
    #anchors = np.tile(anchors, (batch_size,1,1))
    #scores = scores[inds_inside,:]


    all_labels = None 
    all_bbox_targets = None 
    all_bbox_inside_weights = None
    all_bbox_outside_weights = None 

    for batch_ind in range(batch_size):

        # label: 1 is positive, 0 is negative, -1 is dont care
        # (A)
        labels = np.empty((len(inds_inside),), dtype=np.float32)
        labels.fill(-1)

        #get rid of background gt_boxes
        gt_box = np.expand_dims(gt_boxes[batch_ind,:], 0)
        if gt_box[0,-1] == 0:
            labels.fill(0)
        else:

            # overlaps between the anchors and the gt boxes
            # overlaps (ex, gt), shape is A x G
            overlaps = bbox_overlaps(
                np.ascontiguousarray(anchors, dtype=np.float),
                np.ascontiguousarray(gt_box, dtype=np.float))
            argmax_overlaps = overlaps.argmax(axis=1)  # (A)
            max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
            gt_argmax_overlaps = overlaps.argmax(axis=0)  # G
            gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                       np.arange(overlaps.shape[1])]
            gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

            if not cfg.PROPOSAL_CLOBBER_POSITIVES:
                # assign bg labels first so that positive labels can clobber them
                labels[max_overlaps < cfg.PROPOSAL_NEGATIVE_OVERLAP] = 0

            # fg label: for each gt, anchor with highest overlap
            labels[gt_argmax_overlaps] = 1
            # fg label: above threshold IOU
            labels[max_overlaps >= cfg.PROPOSAL_POSITIVE_OVERLAP] = 1

            if cfg.PROPOSAL_CLOBBER_POSITIVES:
                # assign bg labels last so that negative labels can clobber positives
                labels[max_overlaps < cfg.PROPOSAL_NEGATIVE_OVERLAP] = 0


        # subsample positive labels if we have too many
        num_fg = int(cfg.PROPOSAL_FG_FRACTION * cfg.PROPOSAL_BATCH_SIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        # subsample negative labels if we have too many
        num_bg = cfg.PROPOSAL_BATCH_SIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:

            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1

        #Phil add
        if gt_box[0,-1] == 0:
            bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
        else:
            bbox_targets = _compute_targets(anchors, gt_box[argmax_overlaps, :])

        bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array(cfg.PROPOSAL_BBOX_INSIDE_WEIGHTS)

        bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
        if cfg.PROPOSAL_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            positive_weights = np.ones((1, 4))
            negative_weights = np.zeros((1, 4))
        else:
            assert ((cfg.PROPOSAL_POSITIVE_WEIGHT > 0) &
                    (cfg.PROPOSAL_POSITIVE_WEIGHT < 1))
            positive_weights = (cfg.PROPOSAL_POSITIVE_WEIGHT /
                                (np.sum(labels == 1)) + 1)
            negative_weights = ((1.0 - cfg.PROPOSAL_POSITIVE_WEIGHT) /
                                (np.sum(labels == 0)) + 1)
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights

        # map up to original set of anchors
        labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
        bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

        # labels
        labels = labels.reshape((1, height, width, A))
        labels = labels.transpose(0, 3, 1, 2)
        labels = labels.reshape((1, 1, A * height, width)).transpose(0, 2, 3, 1)
        #labels = labels

        # bbox_targets
        bbox_targets = bbox_targets \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)

        bbox_targets = bbox_targets
        # bbox_inside_weights
        bbox_inside_weights = bbox_inside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)

        bbox_inside_weights = bbox_inside_weights

        # bbox_outside_weights
        bbox_outside_weights = bbox_outside_weights \
            .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)

        bbox_outside_weights = bbox_outside_weights

        if all_labels is None:
            all_labels = labels
            all_bbox_targets = bbox_targets
            all_bbox_inside_weights = bbox_inside_weights
            all_bbox_outside_weights = bbox_outside_weights
        else:
            all_labels = np.concatenate((all_labels,labels),0)
            all_bbox_targets = np.concatenate((all_bbox_targets,
                                                  bbox_targets), 0)
            all_bbox_inside_weights = np.concatenate((all_bbox_inside_weights,
                                                          bbox_inside_weights),0)
            all_bbox_outside_weights = np.concatenate((all_bbox_outside_weights,
                                                           bbox_outside_weights),0)

    return all_labels, all_bbox_targets, all_bbox_inside_weights, all_bbox_outside_weights


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    if len(data.shape) == 1:
        ret = np.empty((count,), dtype=np.float32)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)
