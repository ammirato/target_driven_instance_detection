# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from faster_rcnn.nms.cpu_nms import cpu_nms
from faster_rcnn.nms.gpu_nms import gpu_nms
# from ..nms import cpu_nms
# from ..nms import gpu_nms


def nms(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if True: #cfg.USE_GPU_NMS and not force_cpu:
        return gpu_nms(dets, thresh, device_id=0)
    else:
        return cpu_nms(dets, thresh)
