from .cocoapi.PythonAPI.pycocotools.coco import COCO
from .cocoapi.PythonAPI.pycocotools.cocoeval import COCOeval
import numpy as np


def coco_det_eval(gt_path, det_path, catIds,
                  iouThrs=.5,
                  maxDets=[1,10,100]):
    ''' 
    Performs coco detection mAP evaluation

    Example:
        coco_det_eval('/path/to/ground_truth.json','/path/to/detection.json')

    Input parameters:
        gt_path: (str) path to ground truth bounding box json file
        det_path: (str) path to detection output json file
        catIds: (list of int) class ids to evaluate

        iouThrs: (int) iou threshold for a correct detection Default: .5
        maxDets (optional): (list of int) Default: [1,10,100]

    Returns:
        (float) m_ap result
    ''' 

    #initialize COCO ground truth api
    cocoGt=COCO(gt_path)
    #initialize COCO detections api
    cocoDt=cocoGt.loadRes(det_path)

    # setup parameters 
    annType = 'bbox' 
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.params.iouThrs = np.array([iouThrs])
    cocoEval.params.maxDets = maxDets 
    cocoEval.params.catIds = catIds 
    #Areas as defined by AVD dataset
    cocoEval.params.areaRng = [[0, 10000000000.0], [416, 10000000000.0 ], [0, 416], [416, 1250], [1250, 3750], [3750, 7500], [7500,10000000000.0]]
    cocoEval.params.areaRngLbl = ['all', 'valid', 'l0', 'l1', 'l2', 'l3', 'l4']
    cocoEval.params.useSegs = [0]

    #run evaluation
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    return cocoEval.stats[1]


