from target_driven_instance_detection.evaluation.pycocotools.coco import COCO
from target_driven_instance_detection.evaluation.pycocotools.cocoeval import COCOeval
import numpy as np
#import skimage.io as io
#import pylab
#pylab.rcParams['figure.figsize'] = (10.0, 8.0)






def coco_det_eval(gt_path, det_path,
                  iouThrs=.5,
                  maxDets=[1,10,100],
                  catIds=[0,1]):
    """
    Performs coco detection mAP evaluation


    ex) coco_det_eval('/path/to/ground_truth.json','/path/to/detection.json')

    """

    #initialize COCO ground truth api
    cocoGt=COCO(gt_path)
    #initialize COCO detections api
    cocoDt=cocoGt.loadRes(det_path)

    # setup parameters 
    annType = ['segm','bbox','keypoints']
    annType = annType[1]      #specify type here
    prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
    print 'Running demo for *%s* results.'%(annType)

    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.params.iouThrs = np.array([iouThrs])
    cocoEval.params.maxDets = maxDets 
    cocoEval.params.catIds = catIds 
    cocoEval.params.areaRng = [[0, 10000000000.0], [416, 10000000000.0 ], [0, 416], [416, 1250], [1250, 3750], [3750, 7500], [7500,10000000000.0]]
    cocoEval.params.areaRngLbl = ['all', 'valid', 'l0', 'l1', 'l2', 'l3', 'l4']
    cocoEval.params.useSegs = [0]

    #run evaluation
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    return cocoEval.stats[2]
