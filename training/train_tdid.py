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

from instance_detection.model_defs import network
from instance_detection.model_defs.TDID import TDID 

from instance_detection.utils.timer import Timer
from instance_detection.utils.utils import *
from instance_detection.utils.ILSVRC_VID_loader import VID_Loader

from instance_detection.testing.test_tdid import test_net, im_detect
from instance_detection.evaluation.COCO_eval.coco_det_eval import coco_det_eval 

import active_vision_dataset_processing.data_loading.active_vision_dataset_pytorch as AVD  

# load config
cfg_file = 'configAVD2' #NO EXTENSTION!
cfg = importlib.import_module('instance_detection.utils.configs.'+cfg_file)
cfg = cfg.get_config()

##prepare target images (gather paths to the images)
target_images ={} 
if cfg.PYTORCH_FEATURE_NET:
    target_images = get_target_images(cfg.TARGET_IMAGE_DIR,cfg.NAME_TO_ID.keys())
else:
    'Must use pytorch pretrained model, others not supported'
    #would need to add new normaliztion to get_target_images, and elsewhere

#make sure only targets that have ids, and have target images are chosen
train_ids = check_object_ids(cfg.TRAIN_OBJ_IDS, cfg.ID_TO_NAME,target_images) 
val_ids = check_object_ids(cfg.VAL_OBJ_IDS, cfg.ID_TO_NAME,target_images) 
if train_ids==-1 or val_ids==-1:
    print 'Invalid IDS!'
    sys.exit()


print('Setting up training data...')
train_set = get_AVD_dataset(cfg.DATA_BASE_DIR,
                            cfg.TRAIN_LIST,
                            train_ids,
                            max_difficulty=cfg.MAX_OBJ_DIFFICULTY,
                            fraction_of_no_box=cfg.FRACTION_OF_NO_BOX_IMAGES)
valset = get_AVD_dataset(cfg.DATA_BASE_DIR,
                         cfg.VAL_LIST,
                         val_ids, 
                         max_difficulty=cfg.MAX_OBJ_DIFFICULTY,
                         fraction_of_no_box=cfg.VAL_FRACTION_OF_NO_BOX_IMAGES)

trainloader = torch.utils.data.DataLoader(train_set,
                                          batch_size=cfg.BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=cfg.NUM_WORKERS,
                                          collate_fn=AVD.collate)
if cfg.USE_VID:
    vid_train_set = VID_Loader(cfg.VID_DATA_DIR,cfg.VID_SUBSET, 
                           target_size=cfg.VID_MAX_MIN_TARGET_SIZE, 
                           multiple_targets=True, 
                           batch_size=cfg.BATCH_SIZE)



#write meta data out
#meta_fid = open(os.path.join(text_out_dir,save_name_base+'.txt'),'w')
#meta_fid.write('save name: {}\n'.format(save_name_base))
#meta_fid.write('batch norm: {}\n'.format(use_batch_norm))
#meta_fid.write('torch vgg: {}\n'.format(use_torch_vgg))
#meta_fid.write('pretrained vgg: {}\n'.format(use_pretrained_vgg))
#meta_fid.write('batch_size: {}\n'.format(batch_size))
#meta_fid.write('vary images: {}\n'.format(vary_images))
#meta_fid.write('chosen_ids: {}\n'.format(chosen_ids))
#meta_fid.write('val chosen_ids: {}\n'.format(val_chosen_ids))
#meta_fid.write('train_list: {}\n'.format(train_list))
#meta_fid.write('val_lists: {}\n'.format(val_lists))
#meta_fid.write('target_path: {}\n'.format(target_path))
#if use_VID:
#    meta_fid.write('VID_target_size: {}\n'.format(target_size))
#    meta_fid.write('vid_set: {}\n'.format('train_single'))
#meta_fid.write('learing rate: {}\n'.format(lr))
#if load_trained_model:
#    meta_fid.write('start from: {}\n'.format(trained_model_name))
#meta_fid.close()


print('Loading network...')
net = TDID(cfg)
vgg16_bn = models.vgg16_bn(pretrained=False)
net.features = torch.nn.Sequential(*list(vgg16_bn.features.children())[:-1])
net.features.eval()#freeze batchnorms layers?

if cfg.LOAD_FULL_MODEL:
    #load a previously trained model
    network.load_net(trained_model_path + trained_model_name, net)
else:
    #load pretrained vgg weights, and init everything else randomly
    network.weights_normal_init(net, dev=0.01)
    if cfg.USE_PRETRAINED_WEIGHTS: 
        vgg16_bn = models.vgg16_bn(pretrained=True)
        net.features = torch.nn.Sequential(*list(vgg16_bn.features.children())[:-1])
        net.features.eval()


#put net on gpu
net.cuda()
net.train()

#setup optimizer
params = list(net.parameters())
optimizer = torch.optim.SGD(params, lr=cfg.LEARNING_RATE,
                                    momentum=cfg.MOMENTUM, 
                                    weight_decay=cfg.WEIGHT_DECAY)

#make sure dir for saving model checkpoints exists
if not os.path.exists(cfg.SNAPSHOT_SAVE_DIR):
    os.mkdir(cfg.SNAPSHOT_SAVE_DIR)

# things to print out during training training
train_loss = 0
t = Timer()
t.tic()


print('Begin Training...')
for epoch in range(cfg.MAX_NUM_EPOCHS):
    targets_cnt = {}#how many times a target is used(visible, total)
    for cid in train_ids:
        targets_cnt[cid] = [0,0]
    epoch_loss = 0
    epoch_step_cnt = 0
    for step,batch in enumerate(trainloader):

        if cfg.BATCH_SIZE == 1:
            batch[0] = [batch[0]]
            batch[1] = [batch[1]]
        if type(batch[0]) is not list or len(batch[0]) < cfg.BATCH_SIZE:
            continue

        batch_im_data = []
        batch_target_data = []
        batch_gt_boxes = []
        for sample_ind in range(cfg.BATCH_SIZE):
            im_data=batch[0][sample_ind]
            im_data=normalize_image(im_data,cfg)
            gt_boxes = np.asarray(batch[1][sample_ind][0],dtype=np.float32) 
            #if there are no boxes for this image, add a dummy background box
            if gt_boxes.shape[0] == 0:
                gt_boxes = np.asarray([[0,0,1,1,0]])

            objects_present = gt_boxes[:,4]
            objects_present = objects_present[np.where(objects_present!=0)[0]]
            not_present = np.asarray([ind for ind in train_ids 
                                              if ind not in objects_present and 
                                                 ind != 0]) 

            #pick a random target, with a bias towards choosing a target that 
            #is in the image. Also pick just that object's gt_box
            if (np.random.rand() < .8 or not_present.shape[0]==0) and objects_present.shape[0]!=0:
                target_ind = int(np.random.choice(objects_present))
                gt_boxes = gt_boxes[np.where(gt_boxes[:,4]==target_ind)[0],:-1] 
                gt_boxes[0,4] = 1
                targets_cnt[target_ind][0] += 1 
            else:#the target is not in the image, give a dummy background box
                target_ind = int(np.random.choice(not_present))
                gt_boxes = np.asarray([[0,0,1,1,0]])
            targets_cnt[target_ind][1] += 1 
            
            #get target images
            target_name = cfg.ID_TO_NAME[target_ind]
            target_data = []
            for t_type,_ in enumerate(target_images[target_name]):
                img_ind = np.random.choice(np.arange(
                                      len(target_images[target_name][t_type])))
                target_img = cv2.imread(target_images[target_name][t_type][img_ind])

                if np.random.rand() < .9 and cfg.AUGMENT_TARGET_IMAGES:
                    target_img = vary_image(target_img)
                target_img = normalize_image(target_img,cfg)
                batch_target_data.append(target_img)

            batch_im_data.append(im_data)
            batch_gt_boxes.extend(gt_boxes)

        #prep data for input to network
        target_data = match_and_concat_images_list(batch_target_data,
                                                   min_size=cfg.MIN_TARGET_SIZE)
        im_data = match_and_concat_images_list(batch_im_data)
        gt_boxes = np.asarray(batch_gt_boxes) 

        im_info = im_data.shape[1:]
        im_data = network.np_to_variable(im_data, is_cuda=True)
        im_data = im_data.permute(0, 3, 1, 2)
        target_data = network.np_to_variable(target_data, is_cuda=True)
        target_data = target_data.permute(0, 3, 1, 2)

        # forward
        net(target_data, im_data, gt_boxes=gt_boxes, im_info=im_info)
        loss = net.loss

        train_loss += loss.data[0]
        epoch_step_cnt += 1
        epoch_loss += loss.data[0]

        # backprop and parameter update
        optimizer.zero_grad()
        loss.backward()
        network.clip_gradient(net, 10.)
        optimizer.step()

        if cfg.USE_VID:
            batch = vid_train_set[0]
            gt_boxes = np.asarray(batch[1])
            im_data = match_and_concat_images_list(batch[0])
            im_data =  normalize_image(im_data,cfg)

            target_data = match_and_concat_images_list(batch[2]) 
            target_data = normalize_image(target_data,cfg)

            net(target_data, im_data, gt_boxes)
            loss = net.loss
            optimizer.zero_grad()
            loss.backward()
            network.clip_gradient(net, 10.)
            optimizer.step()

        #print out training info
        if step % cfg.DISPLAY_INTERVAL == 0:
            duration = t.toc(average=False)
            fps = step+1.0 / duration

            log_text = 'step %d, epoch_avg_loss: %.4f, fps: %.2f (%.2fs per batch) ' \
                       'epoch:%d loss: %.4f tot_avg_loss: %.4f %s' % (
                step,  epoch_loss/epoch_step_cnt, fps, 1./fps, 
                epoch, loss.data[0],train_loss/(step+1), cfg.MODEL_BASE_SAVE_NAME)
            print log_text
            print targets_cnt

    ######################################################
    #epoch over
    #test validation set, save a checkpoint
    if epoch % cfg.SAVE_FREQ == 0:
        valloader = torch.utils.data.DataLoader(valset,
                                              batch_size=1,
                                              shuffle=True,
                                              collate_fn=AVD.collate)
        net.eval()
        model_name = cfg.MODEL_BASE_SAVE_NAME + '_{}'.format(epoch)
        all_results = test_net(model_name, net, valloader, cfg.NAME_TO_ID, 
                               target_images,cfg.VAL_OBJ_IDS,cfg, 
                               max_dets_per_target=cfg.MAX_DETS_PER_TARGET,
                               output_dir=cfg.TEST_OUTPUT_DIR,
                               score_thresh=cfg.SCORE_THRESH)

        m_ap = coco_det_eval(cfg.GROUND_TRUTH_BOXES,
                             cfg.TEST_OUTPUT_DIR+model_name+'.json',
                             catIds=cfg.VAL_OBJ_IDS)

        save_name = os.path.join(cfg.SNAPSHOT_SAVE_DIR, 
                                 (cfg.MODEL_BASE_SAVE_NAME+
                                  '_{}_{}_{:1.5f}_{:1.5f}.h5').format(
                                 epoch, step, epoch_loss/epoch_step_cnt, m_ap))
        network.save_net(save_name, net)
        print('save model: {}'.format(save_name))

        net.train()
        net.features.eval() #freeze batch norm layers?




