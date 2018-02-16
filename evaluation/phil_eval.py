import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
#import skimage.io as io
#import pylab
#pylab.rcParams['figure.figsize'] = (10.0, 8.0)

annType = ['segm','bbox','keypoints']
annType = annType[1]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print 'Running demo for *%s* results.'%(annType)

#initialize COCO ground truth api
#dataDir='../'
#dataType='val2014'
#annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
#cocoGt=COCO('/net/bvisionserver3/playpen10/ammirato/Data/RohitCOCOgt/all_uw_scenes.json')
#cocoGt=COCO('/net/bvisionserver3/playpen10/ammirato/Data/RohitCOCOgt/uw_201.json')
cocoGt=COCO('/net/bvisionserver3/playpen10/ammirato/Data/RohitCOCOgt/avd_split3.json')
#cocoGt=COCO('/net/bvisionserver3/playpen10/ammirato/Data/RohitCOCOgt/avd_split2.json')
#cocoGt=COCO('/net/bvisionserver3/playpen10/ammirato/Data/RohitCOCOgt/avd_split1.json')
#cocoGt=COCO('/net/bvisionserver3/playpen10/ammirato/Data/RohitCOCOgt/avd_all.json')
#cocoGt=COCO('/net/bvisionserver3/playpen10/ammirato/Data/RohitCOCOgt/gmu_split1.json')
#cocoGt=COCO('/net/bvisionserver3/playpen10/ammirato/Data/RohitCOCOgt/gmu9.json')
#cocoGt=COCO('/net/bvisionserver3/playpen10/ammirato/Data/RohitCOCOgt/all_gmu.json')
#cocoGt=COCO('/net/bvisionserver3/playpen10/ammirato/Data/RohitCOCOgt/gmu2avd_withbox.json')

#initialize COCO detections api
#resFile='%s/results/%s_%s_fake%s100_results.json'
#resFile = resFile%(dataDir, prefix, dataType, annType)
#cocoDt=cocoGt.loadRes('/net/bvisionserver3/playpen/ammirato/Data/Detections/COCO_det_results/TDID_COMB_AVD2_archDmtDIFFbn_ROI_0_15.json')
#cocoDt=cocoGt.loadRes('/net/bvisionserver3/playpen/ammirato/Data/Detections/COCO_det_results/GMU2AVD_chengy_ms_5_15.json')
det_bp = '/net/bvisionserver3/playpen10/ammirato/Data/Detection/TestOutputs/'
det_bp = '/net/bvisionserver3/playpen10/ammirato/Data/Detection/recorded_models_and_meta/test_outputs/'
#cocoDt=cocoGt.loadRes(det_bp + 'AVD2_D_2_10.json')
#cocoDt=cocoGt.loadRes(det_bp + 'AVD2_IMG_0_12.json')
#cocoDt=cocoGt.loadRes(det_bp + 'AVD2_DIFF_0_16.json')
#cocoDt=cocoGt.loadRes(det_bp + 'TDID_COMB_AVD2_archDmtDIFFbn_ROI_0_15.json')
#cocoDt=cocoGt.loadRes(det_bp + 'GMU2AVD_0_7.json')
#cocoDt=cocoGt.loadRes(det_bp + 'GMU1_0_2_1500.json')
#cocoDt=cocoGt.loadRes(det_bp + 'GEN4GMU_0_1_4500.json')
#cocoDt=cocoGt.loadRes(det_bp + 'GEN4GUW_0_2_1500.json')
#cocoDt=cocoGt.loadRes(det_bp + 'AVD2_0_12_01nms.json')
#cocoDt=cocoGt.loadRes(det_bp + 'AVD3_0_12.json')
#cocoDt=cocoGt.loadRes(det_bp + 'AVD1_GEN.json')

#cocoDt=cocoGt.loadRes(det_bp + 'TDIDS_AVD2_34.json')
#cocoDt=cocoGt.loadRes(det_bp + 'TDIDS_AVD1_01.json')
cocoDt=cocoGt.loadRes(det_bp + 'TDIDS_AVD3_01.json')



# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
#cocoEval.params.imgIds  = imgIds
cocoEval.params.iouThrs = np.array([0.5])
#cocoEval.params.areaRng = [[0, 10000000000.0], [1500, 10000000000.0 ], [0, 1500], [1500, 5000], [5000, 15000], [15000, 30000], [30000,10000000000.0]]
#cocoEval.params.areaRng = [[0, 10000000000.0], [1250, 10000000000.0 ], [0, 416], [416, 1250], [1250, 3750], [3750, 7500], [7500,10000000000.0]]
cocoEval.params.areaRng = [[416, 10000000000.0], [3700, 10000000000.0 ], [0, 416], [416, 3700], [3700, 3750], [3750, 7500], [7500,10000000000.0]]
cocoEval.params.areaRngLbl = ['all', 'valid', 'l0', 'l1', 'l2', 'l3', 'l4']
#cocoEval.params.areaRng = [[416, 10000000000.0 ]]
#cocoEval.params.areaRngLbl = ['all']
cocoEval.params.maxDets = [1, 100, 500]
cocoEval.params.useSegs = [0]
#cocoEval.params.maxDets = [100]
#cocoEval.params.catIds =  [5,10,12,14,21,28]
#cocoEval.params.catIds = [96] #[18,50,79,94,96]
#cocoEval.params.catIds = [5,10,12,14,21,28,18,50,79,94,96]
#cocoEval.params.catIds = [28]
cocoEval.params.catIds = [1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,19,20,21,22,23,24,25,26,27,28,29,30,31]
#cocoEval.params.catIds =[1050, 1052, 1053, 1054, 1055, 1270, 1143, 1243, 1244, 1245, 1247, 1252, 1255, 1256, 1257, 1004, 1005, 1007, 1140, 1142,1271, 1272]
#cocoEval.params.catIds =[1055] #, 1052, 1053, 1054, 1055, 1270, 1143, 1243, 1244, 1245, 1247, 1252, 1255, 1256, 1257, 1004, 1005, 1007, 1140, 1142,1271, 1272]
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
