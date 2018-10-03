import os
import torch
import torchvision.models as models
import cv2
#import cPickle
import numpy as np
import importlib
import json

from model_defs.TDID import TDID
from model_defs.nms.nms_wrapper import nms
from utils import * 

import active_vision_dataset_processing.data_loading.active_vision_dataset as AVD  


def im_detect(net, target_data,im_data, im_info, features_given=True):
    """
    Detect single target object in a single scene image.

    Input Parameters:
        net: (TDID) the network
        target_data: (torch Variable) target images
        im_data: (torch Variable) scene_image
        im_info: (tuple) (height,width,channels) of im_data
        
        features_given(optional): (bool) if true, target_data and im_data
                                  are feature maps from net.features,
                                  not images. Default: True
                                    

    Returns:
        scores (ndarray): N x 2 array of class scores
                          (N boxes, classes={background,target})
        boxes (ndarray): N x 4 array of predicted bounding boxes
    """

    cls_prob, rois = net(target_data, im_data, im_info,
                                    features_given=features_given)
    scores = cls_prob.data.cpu().numpy()[0,:,:]
    zs = np.zeros((scores.size, 1))
    scores = np.concatenate((zs,scores),1)
    boxes = rois.data.cpu().numpy()[0,:, :]

    return scores, boxes


def test_net(model_name, net, dataloader, num_images,cfg, 
             max_dets_per_target=1, score_thresh=0.1,
             output_dir=None):
    """
    Test a TDID network.

    Input Parameters:
        model_name: (string) name of model for saving results
        net: (TDID) the network
        dataloader:  (torch DataLoader) dataloader for test set
        cfg: (Config) config file
        
        max_dets_per_target (optional): (int) maximum number of detections 
                                        outputted for a single target/scene 
                                        image pair. Default: 5.
        score_thresh (optional): (float) minimum score a box must have to be 
                                 outputted. Default: .1
        output_dir (optional): (str) full path of directory to save results in
                               if None, nothing will be saved. 
                               Default: None. 
         

    """
   
    iou50_count = 0
    iou75_count = 0
    total_count = 0

    for step in range(1,num_images+1):

        batch_scene_imgs,batch_gt_boxes,batch_target_imgs = dataloader.get_batch(1) 
        batch_target_imgs = resize_target_images(batch_target_imgs[:2])

        #prep data for input to network
        #batch_scene_imgs = match_and_concat_images_list(batch_scene_imgs)
        batch_scene_imgs = batch_scene_imgs[0]
        batch_scene_imgs = np.expand_dims(batch_scene_imgs,axis=0)
        batch_scene_imgs = normalize_image(batch_scene_imgs, cfg)
        batch_target_imgs = normalize_image(batch_target_imgs, cfg)
        gt_box = np.asarray(batch_gt_boxes[0]) 
        im_info = batch_scene_imgs.shape[1:]
        batch_scene_imgs = np_to_variable(batch_scene_imgs, is_cuda=True)
        batch_scene_imgs = batch_scene_imgs.permute(0, 3, 1, 2)
        batch_target_imgs = np_to_variable(batch_target_imgs, is_cuda=True)
        batch_target_imgs = batch_target_imgs.permute(0, 3, 1, 2)


        scores, boxes = im_detect(net, batch_target_imgs, batch_scene_imgs,
                                  im_info, features_given=False)

        if cfg.TEST_RESIZE_BOXES_FACTOR > 0:
            boxes *= cfg.TEST_RESIZE_BOXES_FACTOR

        #get scores for foreground, non maximum supression
        inds = np.where(scores[:, 1] > score_thresh)[0]
        fg_scores = scores[inds, 1]
        fg_boxes = boxes[inds,:]
        fg_dets = np.hstack((fg_boxes, fg_scores[:, np.newaxis])) \
            .astype(np.float32, copy=False)
        keep = nms(fg_dets, cfg.TEST_NMS_OVERLAP_THRESH)
        fg_dets = fg_dets[keep, :]

        # Limit to max_per_target detections *over all classes*
        if max_dets_per_target > 0:
            image_scores = np.hstack([fg_dets[:, -1]])
            if len(image_scores) > max_dets_per_target:
                image_thresh = np.sort(image_scores)[-max_dets_per_target]
                keep = np.where(fg_dets[:, -1] >= image_thresh)[0]
                fg_dets = fg_dets[keep, :]

        total_count += 1
        if len(fg_dets) >0:
            pred_box = fg_dets[0]
            iou = get_boxes_iou(pred_box,gt_box)
            if iou >=.5:
                iou50_count +=1
            if iou >=.75:
                iou75_count +=1
        
    return iou50_count/float(total_count), iou75_count/float(total_count) 



if __name__ == '__main__':

    print('no')
#    #load config file
#    cfg_file = 'configAVD2' #NO EXTENSTION!
#    cfg = importlib.import_module('configs.'+cfg_file)
#    cfg = cfg.get_config()
#
#    ##prepare target images (gather paths to the images)
#    target_images ={}
#    if cfg.PYTORCH_FEATURE_NET:
#        target_images = get_target_images(cfg.TARGET_IMAGE_DIR, 
#                                          cfg.NAME_TO_ID.keys())
#    else:
#        raise NotImplementedError
#        #would need to add new normaliztion to get_target_images, and elsewhere
#
#    #make sure only targets that have ids, and have target images are chosen
#    test_ids = check_object_ids(cfg.TEST_OBJ_IDS, cfg.ID_TO_NAME,target_images)
#    if test_ids==-1:
#        print('Invalid IDS!')
#        sys.exit()
#
#    testset = get_AVD_dataset(cfg.AVD_ROOT_DIR,
#                              cfg.TEST_LIST,
#                              test_ids,
#                              max_difficulty=cfg.MAX_OBJ_DIFFICULTY,
#                              fraction_of_no_box=cfg.TEST_FRACTION_OF_NO_BOX_IMAGES)
#
#    #create train/test loaders, with CUSTOM COLLATE function
#    testloader = torch.utils.data.DataLoader(testset,
#                                              batch_size=1,
#                                              shuffle=True,
#                                              num_workers=cfg.NUM_WORKERS,
#                                              collate_fn=AVD.collate)
#
#    # load net
#    print('Loading ' + cfg.FULL_MODEL_LOAD_NAME + ' ...')
#    net = TDID(cfg)
#    load_net(cfg.FULL_MODEL_LOAD_DIR + cfg.FULL_MODEL_LOAD_NAME, net)
#    net.features.eval()#freeze batchnorms layers?
#    print('load model successfully!')
#    
#    net.cuda()
#    net.eval()
#    
#    # evaluation
#    test_net(cfg.MODEL_BASE_SAVE_NAME, net, testloader, 
#    	 target_images,test_ids,cfg, 
#    	 max_dets_per_target=cfg.MAX_DETS_PER_TARGET,
#    	 score_thresh=cfg.SCORE_THRESH, 
#    	 output_dir=cfg.TEST_OUTPUT_DIR)




