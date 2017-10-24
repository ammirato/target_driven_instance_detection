# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import yaml
import numpy as np
import numpy.random as npr

from .generate_anchors import generate_anchors
from ..utils.cython_bbox import bbox_overlaps, bbox_intersections

# TODO: make fast_rcnn irrelevant
# >>>> obsolete, because it depends on sth outside of this project
from ..fast_rcnn.config import cfg
from ..fast_rcnn.bbox_transform import bbox_transform

# <<<< obsolete

DEBUG = False


def rois_target_layer(rois, gt_boxes, gt_ishard, dontcare_areas, im_info, _feat_stride=[16, ],
                        anchor_scales=[4, 8, 16, 32]):
    """
    Assign rois to ground-truth targets. Produces rois classification
    labels and bounding-box regression targets.
    Parameters
    ----------
    rois: (N x 4) list of rois 
    gt_boxes: (G, 5) vstack of [x1, y1, x2, y2, class]
    gt_ishard: (G, 1), 1 or 0 indicates difficult or not
    dontcare_areas: (D, 4), some areas may contains small objs but no labelling. D may be 0
    im_info: a list of [image_height, image_width, scale_ratios]
    _feat_stride: the downsampling ratio of feature map to the original input image
    anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
    ----------
    Returns
    ----------
    rpn_labels : (HxWxA, 1), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
    rpn_bbox_targets: (HxWxA, 4), distances of the anchors to the gt_boxes(may contains some transform)
                            that are the regression objectives
    rpn_bbox_inside_weights: (HxWxA, 4) weights of each boxes, mainly accepts hyper param in cfg
    rpn_bbox_outside_weights: (HxWxA, 4) used to balance the fg/bg,
                            beacuse the numbers of bgs and fgs mays significiantly different
    """

    # Algorithm:
    #
    # for each roi
    #   get iou with gt_box to assign label 
    #   get target box offsets 

    rois = rois.data.cpu().numpy()
    rois = rois[:,1:]

    # label: 1 is positive, 0 is negative, -1 is dont care
    labels = np.empty((rois.shape[0],), dtype=np.float32)
    labels.fill(-1)

    #get rid of background gt_boxes
    gt_boxes = np.delete(gt_boxes,np.where(gt_boxes[:,4]==0),0)
    if gt_boxes.shape[0] == 0:
        labels.fill(0)
    else:

        # overlaps between the anchors and the gt boxes
        # overlaps (ex, gt), shape is A x G
        overlaps = bbox_overlaps(
            np.ascontiguousarray(rois, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))
        argmax_overlaps = overlaps.argmax(axis=1)  # (A)
        max_overlaps = overlaps[np.arange(rois.shape[0]), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)  # G
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]

        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1
        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    #    # preclude dontcare areas
    #    if dontcare_areas is not None and dontcare_areas.shape[0] > 0:
    #        # intersec shape is D x A
    #        intersecs = bbox_intersections(
    #            np.ascontiguousarray(dontcare_areas, dtype=np.float),  # D x 4
    #            np.ascontiguousarray(rois, dtype=np.float)  # A x 4
    #        )
    #        intersecs_ = intersecs.sum(axis=0)  # A x 1
    #        labels[intersecs_ > cfg.TRAIN.DONTCARE_AREA_INTERSECTION_HI] = -1

    #    # preclude hard samples that are highly occlusioned, truncated or difficult to see
    #    if cfg.TRAIN.PRECLUDE_HARD_SAMPLES and gt_ishard is not None and gt_ishard.shape[0] > 0:
    #        assert gt_ishard.shape[0] == gt_boxes.shape[0]
    #        gt_ishard = gt_ishard.astype(int)
    #        gt_hardboxes = gt_boxes[gt_ishard == 1, :]
    #        if gt_hardboxes.shape[0] > 0:
    #            # H x A
    #            hard_overlaps = bbox_overlaps(
    #                np.ascontiguousarray(gt_hardboxes, dtype=np.float),  # H x 4
    #                np.ascontiguousarray(anchors, dtype=np.float))  # A x 4
    #            hard_max_overlaps = hard_overlaps.max(axis=0)  # (A)
    #            labels[hard_max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = -1
    #            max_intersec_label_inds = hard_overlaps.argmax(axis=1)  # H x 1
    #            labels[max_intersec_label_inds] = -1  #

    # subsample positive labels if we have too many
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    bg_inds = np.where(labels == 0)[0]
    if len(bg_inds) > num_bg:
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        labels[disable_inds] = -1
        # print "was %s inds, disabling %s, now %s inds" % (
        # len(bg_inds), len(disable_inds), np.sum(labels == 0))

    #Phil add
    if gt_boxes.shape[0] == 0:
        bbox_targets = np.zeros((rois.shape[0], 4), dtype=np.float32)
    else:
        # bbox_targets = np.zeros((rois.shape[0], 4), dtype=np.float32)
        bbox_targets = _compute_targets(rois, gt_boxes[argmax_overlaps, :])

    bbox_inside_weights = np.zeros((rois.shape[0], 4), dtype=np.float32)
    bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

    bbox_outside_weights = np.zeros((rois.shape[0], 4), dtype=np.float32)
    if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
        # uniform weighting of examples (given non-uniform sampling)
        # num_examples = np.sum(labels >= 0) + 1
        # positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        # negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        positive_weights = np.ones((1, 4))
        negative_weights = np.zeros((1, 4))
    else:
        assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                            (np.sum(labels == 1)) + 1)
        negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                            (np.sum(labels == 0)) + 1)
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    if DEBUG:
        _sums += bbox_targets[labels == 1, :].sum(axis=0)
        _squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts += np.sum(labels == 1)
        means = _sums / _counts
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        print 'means:'
        print means
        print 'stdevs:'
        print stds

    # map up to original set of anchors
    #labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    #bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    #bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    #bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    if DEBUG:
        print 'rpn: max max_overlap', np.max(max_overlaps)
        print 'rpn: num_positive', np.sum(labels == 1)
        print 'rpn: num_negative', np.sum(labels == 0)
        _fg_sum += np.sum(labels == 1)
        _bg_sum += np.sum(labels == 0)
        _count += 1
        print 'rpn: num_positive avg', _fg_sum / _count
        print 'rpn: num_negative avg', _bg_sum / _count

    # labels
    # pdb.set_trace()
#    labels = labels.reshape((1, height, width, A))
#    labels = labels.transpose(0, 3, 1, 2)
#    rpn_labels = labels.reshape((1, 1, A * height, width)).transpose(0, 2, 3, 1)
#
#    # bbox_targets
#    bbox_targets = bbox_targets \
#        .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
#
#    rpn_bbox_targets = bbox_targets
#    # bbox_inside_weights
#    bbox_inside_weights = bbox_inside_weights \
#        .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
#    # assert bbox_inside_weights.shape[2] == height
#    # assert bbox_inside_weights.shape[3] == width
#
#    rpn_bbox_inside_weights = bbox_inside_weights
#
#    # bbox_outside_weights
#    bbox_outside_weights = bbox_outside_weights \
#        .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
#    # assert bbox_outside_weights.shape[2] == height
#    # assert bbox_outside_weights.shape[3] == width
#
#    rpn_bbox_outside_weights = bbox_outside_weights
#
#    return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights
    return labels, bbox_targets, bbox_inside_weights, bbox_outside_weights


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
