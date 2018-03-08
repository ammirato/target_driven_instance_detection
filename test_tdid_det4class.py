import os
import torch
import torchvision.models as models
import cv2
import cPickle
import numpy as np
import importlib
import json

from model_defs.TDID import TDID
from model_defs.nms.nms_wrapper import nms
from utils import * 
from model_defs.anchors.bbox_transform import bbox_transform_inv, clip_boxes

import active_vision_dataset_processing.data_loading.active_vision_dataset as AVD  





def im_classify(net, target_data,im_data, im_info, features_given=True):
    """
    Gives classifcation score for image/target pair 

    """

    cls_prob = net(target_data, im_data, 
                   features_given=features_given, im_info=im_info)
    scores = cls_prob.data.cpu().numpy()[0,:,:]
    return scores.max()


def test_net(model_name, net, dataloader, id_to_name, target_images, chosen_ids, cfg,
             max_dets_per_target=5, score_thresh=0.1,
             output_dir=None,):
    """Test a TDID network on an image dataset."""
    #num images in test set
    num_images = len(dataloader)
   
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    
    #pre compute features for all targets
    target_features_dict = {}
    target_data_dict = {}
    for id_ind,t_id in enumerate(chosen_ids):
        target_name = id_to_name[t_id]
        if target_name == 'background':
            continue
        target_data = []
        for t_type,_ in enumerate(target_images[target_name]):
            img_ind = np.random.choice(np.arange(
                                  len(target_images[target_name][t_type])))
            target_img = cv2.imread(target_images[target_name][t_type][img_ind])
            target_img = normalize_image(target_img,cfg)
            target_data.append(target_img)

        target_data = match_and_concat_images_list(target_data)
        target_data = np_to_variable(target_data, is_cuda=True)
        target_data = target_data.permute(0, 3, 1, 2)
        if cfg.TEST_ONE_AT_A_TIME:
            target_data_dict[target_name] = target_data
        else:
            target_features_dict[target_name] = net.features(target_data)

    print('Hi')

    num_correct = 0
    num_total = 0
    total_score = 0
    total_run = 0
    for i,batch in enumerate(dataloader):
        im_data= batch[0]
        im_info = im_data.shape[:]
        im_data=normalize_image(im_data,cfg)
        im_data = np_to_variable(im_data, is_cuda=True)
        im_data = im_data.unsqueeze(0)
        im_data = im_data.permute(0, 3, 1, 2)

        #get image name and index
        img_name = batch[1][1]
        img_ind = int(img_name[:-4])

        gt_id = batch[1][0][0][4]

        max_score = 0
        max_id = 0
        tos = 0
        #get image features
        if not cfg.TEST_ONE_AT_A_TIME:
            img_features = net.features(im_data)

        for id_ind,t_id in enumerate(chosen_ids):
            target_name = id_to_name[t_id]
            if target_name == 'background':
                continue

            if cfg.TEST_ONE_AT_A_TIME:
                target_data = target_data_dict[target_name]
                _t['im_detect'].tic()
                score = im_detect(net, target_data, im_data, im_info,
                                          features_given=False)
                detect_time = _t['im_detect'].toc(average=False)
            else:
                target_features = target_features_dict[target_name]
                _t['im_detect'].tic()
                score = im_detect(net, target_features, img_features, im_info)
                detect_time = _t['im_detect'].toc(average=False)

            _t['misc'].tic()

            total_score += score
            total_run += 1
            if score>max_score:
                max_score = score
                max_id = t_id    
            if t_id == gt_id:
                tos = score            
        if max_id == gt_id:
            num_correct += 1
        num_total += 1


    print num_correct
    print num_total
    print float(total_score)/float(total_run)
    return float(num_correct)/float(num_total)






if __name__ == '__main__':

    #load config file
    cfg_file = 'configGEN4UWC' #NO EXTENSTION!
    cfg = importlib.import_module('configs.'+cfg_file)
    cfg = cfg.get_config()

    ##prepare target images (gather paths to the images)
    target_images ={}
    if cfg.PYTORCH_FEATURE_NET:
        target_images = get_target_images(cfg.TARGET_IMAGE_DIR,cfg.NAME_TO_ID.keys())
    else:
        print 'Must use pytorch pretrained model, others not supported'
        #would need to add new normaliztion to get_target_images, and elsewhere

    #make sure only targets that have ids, and have target images are chosen
    test_ids = check_object_ids(cfg.TEST_OBJ_IDS, cfg.ID_TO_NAME,target_images)
    #print test_ids
    if test_ids==-1:
        print 'Invalid IDS!'
        sys.exit()

    testset = get_AVD_dataset(cfg.DATA_BASE_DIR,
                              cfg.TEST_LIST,
                              test_ids,
                              max_difficulty=6,#cfg.MAX_OBJ_DIFFICULTY,
                              fraction_of_no_box=1)#cfg.TEST_FRACTION_OF_NO_BOX_IMAGES)

    #create train/test loaders, with CUSTOM COLLATE function
    testloader = torch.utils.data.DataLoader(testset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=cfg.NUM_WORKERS,
                                              collate_fn=AVD.collate)

    load_names = [
                'TDID_GEN4UWC_20_16_20000_230.62730_0.52458.h5',
                'TDID_GEN4UWC_20_8_10000_291.84813_0.48088.h5',
                'TDID_GEN4UWC_20_16_20000_230.62730_0.52458.h5', 
#                'TDID_GEN4UWC_17_1_2800_971.88680_0.45155.h5', 
#                'TDID_GEN4UWC_18_2_2300_693.64961_0.51149.h5',
#                'TDID_GEN4UWC_18_3_3400_694.39169_0.44955.h5',
#                'TDID_GEN4UWC_17_2_2900_863.00803_0.45954.h5',
#                'TDID_GEN4UWC_15_1_1000_476.61535_0.45654.h5',
#                'TDID_GEN4UWC_16_2_1700_345.35398_0.30070.h5',
#                'TDID_GEN4UWC_15_2_1800_338.26621_0.38262.h5',
                ]
    for load_name in load_names:

        # load net
        #print('Loading ' + cfg.FULL_MODEL_LOAD_NAME + ' ...')
        net = TDID(cfg)
        load_net(cfg.FULL_MODEL_LOAD_DIR + load_name, net)
        net.features.eval()#freeze batchnorms layers?
        print('load model successfully!')
        
        net.cuda()
        net.eval()
        
        # evaluation
        acc = test_net(cfg.MODEL_BASE_SAVE_NAME, net, testloader, cfg.ID_TO_NAME, 
             target_images,test_ids,cfg, 
             max_dets_per_target=cfg.MAX_DETS_PER_TARGET,
             score_thresh=cfg.SCORE_THRESH, 
             output_dir=cfg.TEST_OUTPUT_DIR)

        print '{}  {}'.format(acc, load_name)


