import torch
import torch.utils.data
import torchvision.models as models
import os
import sys
import importlib
import numpy as np
from datetime import datetime
import cv2
import time

#from model_defs.TDID import TDID 
from model_defs.TDID_sim import TDID 
from utils import *
from evaluation.coco_det_eval import coco_det_eval 
from AVD_extra_loader import AVD_Extra_Loader
from ILSVRC_VID_loader import VID_Loader

import active_vision_dataset_processing.data_loading.active_vision_dataset as AVD  

# load config
cfg_file = 'configPRE' #NO FILE EXTENSTION!
cfg = importlib.import_module('configs.'+cfg_file)
cfg = cfg.get_config()

if cfg.DET4CLASS:
    test_net = importlib.import_module('test_tdid_det4class').test_net
else:
    #test_net = importlib.import_module('test_tdid').test_net
    test_net = importlib.import_module('test_tdid_classification').test_net


def validate_and_save(cfg,net,total_iterations, avg_loss, valloader):
    '''
    Test on validation data, and save a snapshot of model
    '''
##################################################################r##
    net.eval()
    test_acc = test_net(cfg.MODEL_BASE_SAVE_NAME, net, valloader,
                            1000, cfg,
                            max_dets_per_target=cfg.MAX_DETS_PER_TARGET,
                            output_dir=cfg.TEST_OUTPUT_DIR,
                            score_thresh=cfg.SCORE_THRESH)
    save_name = os.path.join(cfg.SNAPSHOT_SAVE_DIR, 
                             (cfg.MODEL_BASE_SAVE_NAME+
                              '_{}_{:1.5f}_{:1.5f}_{:1.5f}.h5').format(
                             total_iterations,avg_loss, test_acc, test_acc))
    save_net(save_name, net)
    print('save model: {}'.format(save_name))
    net.train()
    #net.features.eval() #freeze batch norm layers?


avd_extra_loader = AVD_Extra_Loader(cfg.AVD_ROOT_DIR, cfg.AVD_EXTRA_FILE)
vid_loader = VID_Loader('/net/bvisionserver3/playpen10/ammirato/Data/ILSVRC/',True)

#prepare target images (gather paths to the images)
target_images ={} 
if True: #cfg.PYTORCH_FEATURE_NET:
    target_images = get_target_images(cfg.TARGET_IMAGE_DIR,cfg.NAME_TO_ID.keys())
else:
    raise NotImplementedError
    #would need to add new normalization to get_target_images, and utilts, etc 

#make sure only targets that have ids, and have target images are chosen
train_ids = check_object_ids(cfg.TRAIN_OBJ_IDS, cfg.ID_TO_NAME,target_images) 
cfg.TRAIN_OBJ_IDS = train_ids
val_ids = check_object_ids(cfg.VAL_OBJ_IDS, cfg.ID_TO_NAME,target_images) 
cfg.VAL_OBJ_IDS = val_ids
if train_ids==-1 or val_ids==-1:
    print('Invalid IDS!')
    sys.exit()


print('Setting up training data...')
train_set = get_AVD_dataset(cfg.AVD_ROOT_DIR,
                            cfg.TRAIN_LIST,
                            train_ids,
                            max_difficulty=cfg.MAX_OBJ_DIFFICULTY,
                            fraction_of_no_box=cfg.FRACTION_OF_NO_BOX_IMAGES)
val_set = get_AVD_dataset(cfg.AVD_ROOT_DIR,
                            cfg.VAL_LIST,
                            train_ids,
                            max_difficulty=cfg.MAX_OBJ_DIFFICULTY,
                            fraction_of_no_box=cfg.FRACTION_OF_NO_BOX_IMAGES)
print('Loading network...')
net = TDID(cfg)
if cfg.LOAD_FULL_MODEL:
    load_net(cfg.FULL_MODEL_LOAD_DIR + cfg.FULL_MODEL_LOAD_NAME, net)
else:
    weights_normal_init(net, dev=0.01)
    if cfg.USE_PRETRAINED_WEIGHTS:
        net.features = load_pretrained_weights(cfg.FEATURE_NET_NAME) 
#net.features.eval()#freeze batchnorms layers?

if not os.path.exists(cfg.SNAPSHOT_SAVE_DIR):
    os.makedirs(cfg.SNAPSHOT_SAVE_DIR)
if not os.path.exists(cfg.META_SAVE_DIR):
    os.makedirs(cfg.META_SAVE_DIR)

#put net on gpu
net.cuda()
net.train()

#setup optimizer
params = list(net.parameters())
optimizer = torch.optim.SGD(params, lr=cfg.LEARNING_RATE,
                                    momentum=cfg.MOMENTUM, 
                                    weight_decay=cfg.WEIGHT_DECAY)
# things to keep track of during training
train_loss = 0
t = Timer()
t.tic()
total_iterations = 1 

save_training_meta_data(cfg,net)
dims = [32,200]

print('Begin Training...')
for step in range(1,cfg.MAX_NUM_ITERATIONS+1):
    total_iterations += 1

    rand_num = np.random.rand()
    if cfg.USE_AVD_EXTRA > rand_num:
####################################################
        #print ('EXTRA')
        batch_scene_imgs,batch_gt_boxes,batch_target_imgs = avd_extra_loader.get_batch(int(cfg.BATCH_SIZE/2),classification=True,dims=dims)
        batch_target_imgs = resize_target_images(batch_target_imgs)
        batch_target_imgs =  batch_target_imgs.astype(np.uint8)
        #prep data for input to network
        batch_scene_imgs = match_and_concat_images_list(batch_scene_imgs)
        batch_scene_imgs = normalize_image(batch_scene_imgs, cfg)
        batch_target_imgs = normalize_image(batch_target_imgs, cfg)
        gt_boxes = np.asarray(batch_gt_boxes)
        im_info = batch_scene_imgs.shape[1:]
        batch_scene_imgs = np_to_variable(batch_scene_imgs, is_cuda=True)
        batch_scene_imgs = batch_scene_imgs.permute(0, 3, 1, 2)
        batch_target_imgs = np_to_variable(batch_target_imgs, is_cuda=True)
        batch_target_imgs = batch_target_imgs.permute(0, 3, 1, 2)

####################################################
    elif cfg.USE_VID + cfg.USE_AVD_EXTRA > rand_num:
        #print ('VID')
        batch_scene_imgs,batch_gt_boxes,batch_target_imgs = vid_loader.get_batch(int(cfg.BATCH_SIZE/2),classification=True, dims=dims)
        batch_target_imgs = resize_target_images(batch_target_imgs)
        batch_target_imgs =  batch_target_imgs.astype(np.uint8)
        #prep data for input to network
        batch_scene_imgs = match_and_concat_images_list(batch_scene_imgs)
        batch_scene_imgs = normalize_image(batch_scene_imgs, cfg)
        batch_target_imgs = normalize_image(batch_target_imgs, cfg)
        gt_boxes = np.asarray(batch_gt_boxes)
        im_info = batch_scene_imgs.shape[1:]
        batch_scene_imgs = np_to_variable(batch_scene_imgs, is_cuda=True)
        batch_scene_imgs = batch_scene_imgs.permute(0, 3, 1, 2)
        batch_target_imgs = np_to_variable(batch_target_imgs, is_cuda=True)
        batch_target_imgs = batch_target_imgs.permute(0, 3, 1, 2)

####################################################
    else:
        #print ('AVD')
        b_ind = np.random.choice(len(train_set)-cfg.BATCH_SIZE)
        batch = train_set[b_ind:b_ind+cfg.BATCH_SIZE] 

        batch_im_data = []
        batch_target_data = []
        batch_gt_boxes = []
        for batch_ind in range(cfg.BATCH_SIZE):
            im_data=batch[0][batch_ind]
            im_data=normalize_image(im_data,cfg)
            gt_boxes = np.asarray(batch[1][batch_ind][0],dtype=np.float32) 

            if np.random.rand() < cfg.RESIZE_IMG:
                im_data = cv2.resize(im_data,(0,0),fx=cfg.RESIZE_IMG_FACTOR,
                                     fy=cfg.RESIZE_IMG_FACTOR)
                if gt_boxes.shape[0] >0:
                    gt_boxes[:,:4] *= cfg.RESIZE_IMG_FACTOR

            #if there are no boxes for this image, add a dummy background box
            if gt_boxes.shape[0] == 0:
                gt_boxes = np.asarray([[100,100,200,200,0]])
            objects_present = gt_boxes[:,4]
            objects_present = objects_present[np.where(objects_present!=0)[0]]
            not_present = np.asarray([ind for ind in train_ids 
                                      if ind not in objects_present and 
                                      ind != 0]) 

            #pick a target 
            if ((np.random.rand() < cfg.CHOOSE_PRESENT_TARGET or 
                    not_present.shape[0]==0) and 
                    objects_present.shape[0]!=0):
                target_ind = int(np.random.choice(objects_present))
                gt_boxes = gt_boxes[np.where(gt_boxes[:,4]==target_ind)[0],:-1] 
                gt_boxes[0,4] = 1
            else:#the target is not in the image, give a dummy background box
                target_ind = int(np.random.choice(not_present))
                gt_boxes = np.asarray([[100,100,200,200,0]])
            
            #get target images
            target_name = cfg.ID_TO_NAME[target_ind]
            target_data = []
            for t_type,_ in enumerate(target_images[target_name]):
                img_ind = np.random.choice(np.arange(
                                      len(target_images[target_name][t_type])))
                target_img = cv2.imread(target_images[target_name][t_type][img_ind])
                if np.random.rand() < cfg.AUGMENT_TARGET_IMAGES:
                    target_img = augment_image(target_img, 
                                           do_illum=cfg.AUGMENT_TARGET_ILLUMINATION)
                #target_img = normalize_image(target_img,cfg)
                batch_target_data.append(target_img)

            im_data = im_data[int(gt_boxes[0][1]):int(gt_boxes[0][3]), int(gt_boxes[0][0]):int(gt_boxes[0][2]),:]
            im_data = resize_image(im_data,dims[0],dims[1])
            batch_im_data.append(im_data)
            batch_gt_boxes.extend(gt_boxes)

        #prep data for input to network
        target_data = resize_target_images(batch_target_data)
        target_data = normalize_image(target_data,cfg)
        im_data = match_and_concat_images_list(batch_im_data)
        gt_boxes = np.asarray(batch_gt_boxes) 
        im_info = im_data.shape[1:]
        im_data = np_to_variable(im_data, is_cuda=True)
        batch_scene_imgs = im_data.permute(0, 3, 1, 2)
        target_data = np_to_variable(target_data, is_cuda=True)
        batch_target_imgs = target_data.permute(0, 3, 1, 2)

    #print(batch_scene_imgs.size())
    # forward
    net(batch_target_imgs, batch_scene_imgs, im_info, gt_boxes=gt_boxes)
   # loss = net.roi_cross_entropy_loss
    loss = net.class_cross_entropy_loss

    train_loss += loss.data[0]

    # backprop and parameter update
    optimizer.zero_grad()
    loss.backward()
    #clip_gradient(net, 10.)
    optimizer.step()
####################################################

    #print out training info
    if step % cfg.DISPLAY_INTERVAL == 0:
        duration = t.toc(average=False)
        fps = step+1.0 / duration

        log_text = 'step %d,  ' \
                   'loss: %.4f tot_avg_loss: %.4f %s' % (
            total_iterations,  
             loss.data[0],train_loss/(step+1), cfg.MODEL_BASE_SAVE_NAME)
        print(log_text)

        
    ######################################################
    if step% cfg.SAVE_FREQ == 0:
        validate_and_save(cfg,net,total_iterations, train_loss/(step+1),vid_loader)

