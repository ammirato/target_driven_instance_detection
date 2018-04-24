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
import matplotlib.pyplot as plt

from model_defs.TDID_sim import TDID 
from utils import *
from evaluation.coco_det_eval import coco_det_eval 

#import active_vision_dataset_processing.data_loading.active_vision_dataset as AVD  
from ILSVRC_VID_loader import VID_Loader


# load config
cfg_file = 'configVID' #NO FILE EXTENSTION!
cfg = importlib.import_module('configs.'+cfg_file)
cfg = cfg.get_config()

max_iterations = 50000

test_net = importlib.import_module('test_tdid_VID').test_net


dataloader = VID_Loader('/net/bvisionserver3/playpen10/ammirato/Data/ILSVRC/',True)
valloader = VID_Loader('/net/bvisionserver3/playpen10/ammirato/Data/ILSVRC/',False)


def validate_and_save(cfg,net,valloader, total_iterations):
    '''
    Test on validation data, and save a snapshot of model
    '''
    model_name = cfg.MODEL_BASE_SAVE_NAME + '_{}'.format(total_iterations)
    net.eval()

    acc50, acc75 = test_net(model_name,net,valloader,5000,cfg)

    save_name = os.path.join(cfg.SNAPSHOT_SAVE_DIR, 
                             (model_name+
                              '_{:1.5f}_{:1.5f}.h5').format(acc50,acc75))
     
     
    save_net(save_name, net)
    print('save model: {}'.format(save_name))
    net.train()
    net.features.eval() #freeze batch norm layers?



print('Loading network...')
net = TDID(cfg)
if cfg.LOAD_FULL_MODEL:
    load_net(cfg.FULL_MODEL_LOAD_DIR + cfg.FULL_MODEL_LOAD_NAME, net)
else:
    weights_normal_init(net, dev=0.01)
    if cfg.USE_PRETRAINED_WEIGHTS:
        net.features = load_pretrained_weights(cfg.FEATURE_NET_NAME) 
net.features.eval()#freeze batchnorms layers?

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
loss_history_num = 1000
last_few_losses = np.zeros(loss_history_num)


save_training_meta_data(cfg,net)

print('Begin Training...')
for step in range(1,2*max_iterations):


    if step == max_iterations:
        params = list(net.parameters())
        optimizer = torch.optim.SGD(params, lr=cfg.LEARNING_RATE*.1,
                                            momentum=cfg.MOMENTUM, 
                                            weight_decay=cfg.WEIGHT_DECAY)
    
    batch_scene_imgs,batch_gt_boxes,batch_target_imgs = dataloader.get_batch(cfg.BATCH_SIZE) 
    batch_target_imgs = resize_target_images(batch_target_imgs)
#    batch_target_imgs =  batch_target_imgs.astype(np.uint8)
#    for b_ind,img in enumerate(batch_scene_imgs):
#        print img.shape
#        cv2.imshow('scene',img)
#        cv2.imshow('target_image 1',batch_target_imgs[2*b_ind])
#        cv2.imshow('target_image 2',batch_target_imgs[2*b_ind+1])
#        print batch_gt_boxes[b_ind]
#        abc = cv2.waitKey(0) 
#        print abc
#        if abc == 113:
#            sys.exit(0)

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

    # forward
    net(batch_target_imgs, batch_scene_imgs, im_info, gt_boxes=gt_boxes)
    loss = net.loss

    train_loss += loss.data[0]
    last_few_losses[step % loss_history_num] = loss.data[0]
    

    # backprop and parameter update
    optimizer.zero_grad()
    loss.backward()
    clip_gradient(net, 10.)
    optimizer.step()


    if step % cfg.DISPLAY_INTERVAL ==0: 
        print('Step: {}   last {} avg Loss: {}'.format(step, loss_history_num,
                                last_few_losses.sum()/min(step,loss_history_num)))




    if (not cfg.SAVE_BY_EPOCH) and  step % cfg.SAVE_FREQ==0:
        validate_and_save(cfg,net,valloader, step)
        

