import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np


annType ='bbox'

#initialize COCO ground truth api
#cocoGt=COCO('/net/bvisionserver3/playpen10/ammirato/Data/RohitCOCOgt/avd_all.json')
cocoGt=COCO('./Data/GT/all_gmu2.json')
#initialize COCO detections api
det_bp = '/net/bvisionserver3/playpen10/ammirato/Data/Detection/recorded_models_and_meta/test_outputs/'
#cocoDt=cocoGt.loadRes(det_bp + 'TDID_GMUsynth2AVD_05_12.json')
cocoDt=cocoGt.loadRes('./Data/TestOutputs/TDID_GEN4GMU_04_80000.json')

#catIds =[1050, 1052, 1053, 1054, 1055, 1270, 1143, 1243, 1244, 1245, 1247, 1252, 1255, 1256, 1257, 1004, 1005, 1007, 1140, 1142,1271, 1272]
#catIds =[1050, 1052, 1053, 1054, 1055, 1270, 1143, 1243, 1244, 1245, 1247, 1252, 1255, 1256, 1257, 1004, 1005, 1007, 1140, 1142,1271, 1272]
#catIds =  [1270,1271,1272,1140,1142,1143,1004,1005,1007,1252,1255,1256,1257,1243,1244,1245,1247,1050,1052,1053,1054,1055]
catIds = [5,50,10,12,14,79,28,94,96,18,21]
#catIds = [5,10,12,14,21,28]


# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
#cocoEval.params.imgIds  = imgIds
cocoEval.params.iouThrs = np.array([0.5])
cocoEval.params.areaRng = [[0, 10000000000.0], [416, 10000000000.0 ], [0, 416], [416, 3700], [3700, 3750], [3750, 7500], [7500,10000000000.0]]
cocoEval.params.areaRngLbl = ['all', 'valid', 'l0', 'l1', 'l2', 'l3', 'l4']
cocoEval.params.maxDets = [1, 100, 500]
cocoEval.params.useSegs = [0]

for cid in catIds:
    print('\n\n {}'.format(cid))
    cocoEval.params.catIds = [cid]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

