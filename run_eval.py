from evaluation.coco_det_eval import coco_det_eval
import numpy as np
import os

det_file_names= [

'TDID_GEN4GUW_02TO_50000_2.json',
#'TDID_GEN4GMU_06TO_150000.json',
#'TDID_GEN4GMUONLY_05TO_40000.json',
#'TDID_GEN4GMUONLY_03TO_50000.json'

                ]
#gt_name = 'all_gmu2.json'
gt_name = 'all_uw_scenes.json'
outfile_name = 'evals'
#catIds = [5,50,10,12,14,79,28,94,96,18,21]
catIds = [1050, 1052, 1053, 1054, 1055, 1270, 1143, 1243, 1244, 1245, 1247, 1252, 1255, 1256, 1257, 1004, 1005, 1007, 1140, 1142, 1271, 1272]
#catIds = [50,79,94,96,18]
bp = './Data/'
gt_path =  os.path.join(bp,'GT', gt_name)
det_bp =  os.path.join(bp,'TestOutputs')

total_counter = 300
counter = 0
outfid = open(outfile_name+str(total_counter)+'.txt', 'w')


for det_fn in det_file_names:

    outfid.write('{}: '.format(det_fn.replace('.json','')))
    det_path = os.path.join(det_bp,det_fn)

    m_ap = coco_det_eval(gt_path, det_path, catIds)
    outfid.write(' {} '.format(m_ap))

    print('Done!')

    for cat_id in catIds:
        m_ap = coco_det_eval(gt_path,det_path,[cat_id])
        outfid.write(' {} '.format(m_ap))
    outfid.write('\n\n')

    if counter > 10:
        outfid.close()
        total_counter +=1
        counter = 0
        outfid = open(outfile_name+str(total_counter)+'.txt', 'w')
    counter += 1
outfid.close()
