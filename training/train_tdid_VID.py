import os
import sys
import torch
import torch.utils.data
import numpy as np
from datetime import datetime
import cv2

import matplotlib.pyplot as plt

from instance_detection.model_defs import network
#from instance_detection.model_defs.tdid import TDID 
from instance_detection.model_defs.tdid_depthwise_batch import TDID 
#from instance_detection.model_defs.tdid_many_measures import TDID 
from instance_detection.model_defs.utils.timer import Timer
from instance_detection.model_defs.fast_rcnn.config import cfg, cfg_from_file

from instance_detection.utils.get_data import get_target_images, match_and_concat_images
from instance_detection.utils.ILSVRC_VID_loader import VID_Loader 

from instance_detection.testing.test_tdid_VID import test_net, im_detect

import active_vision_dataset_processing.data_loading.active_vision_dataset_pytorch as AVD  
import active_vision_dataset_processing.data_loading.transforms as AVD_transforms
import exploring_pytorch.basic_examples.GetDataSet as GetDataSet
from exploring_pytorch.basic_examples.DetecterEvaluater import DetectorEvaluater


#TODO make target image to gt_box index(id) more robust,clean, better


try:
    from termcolor import cprint
except ImportError:
    cprint = None

def log_print(text, color=None, on_color=None, attrs=None):
    if cprint is not None:
        cprint(text, color=color, on_color=on_color, attrs=attrs)
    else:
        print(text)



# hyper-parameters
# ------------
cfg_file = '../utils/config.yml'
#pretrained_model = '/net/bvisionserver3/playpen/ammirato/Data/Detections/pretrained_models/VGG_imagenet.npy'
pretrained_model = '/playpen/ammirato/Data/Detections/pretrained_models/VGG_imagenet.npy'
#output_dir = ('/net/bvisionserver3/playpen/ammirato/Data/Detections/' + 
#             '/saved_models/')
output_dir = ('/playpen/ammirato/Data/Detections/' + 
             '/saved_models/')
#save_name_base = 'TDID_archMM_10'
save_name_base = 'TDID_VID_archD_3'
save_freq = 1 

#trained_model_path = ('/net/bvisionserver3/playpen/ammirato/Data/Detections/' +
#                     '/saved_models/')
trained_model_path = ('/playpen/ammirato/Data/Detections/' +
                     '/saved_models/')
trained_model_name = 'TDID_VID_archD_1_98000_30.15133.h5'
load_trained_model = False 
trained_step = 0 

dumb_acc_fid = open('./VID_acc_things.txt', 'w')
dumb_acc_fid.close()

preload_target_images =  False

max_steps = 300000 

rand_seed = 1024

# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)
lr = cfg.TRAIN.LEARNING_RATE 
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval =10# cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS

# load data
data_path = '/playpen/ammirato/Downloads/ILSVRC/'


#CREATE TRAIN/TEST splits
train_set = VID_Loader(data_path,'train_single')
val_set = VID_Loader(data_path,'val_single')

#load net definition and init parameters
net = TDID()
if load_trained_model:
    #load a previously trained model
    network.load_net(trained_model_path + trained_model_name, net)
else:
    #load pretrained vgg weights, and init everything else randomly
    network.weights_normal_init(net, dev=0.01)
    network.load_pretrained_tdid(net, pretrained_model)

#put net on gpu
net.cuda()
net.train()

#setup optimizer
params = list(net.parameters())
optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

#make sure dir for saving model checkpoints exists
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# things to print out during training 
train_loss = 0
step_cnt = 0
t = Timer()
t.tic()

window_loss = 0
window_steps = 0

for step in range(max_steps):

    #gets a random batch
    batch = train_set[0]

    #get first and second images
    im_data = match_and_concat_images(batch[0],batch[3]) -127
    #get target images
    target_data = match_and_concat_images(batch[2][0],batch[2][1])  -127

    #get gt box, add 1 for fg label
    gt_box = batch[1]
    gt_box.append(1)
    gt_box = np.asarray(gt_box,dtype=np.float32)
    #1 gt_box for each image, the second is a dummy box since there is no fg
    gt_boxes = np.asarray([gt_box, [0,0,1,1,0]])

    # forward
    net(target_data, im_data, gt_boxes)
    loss = net.loss
    #loss = net.loss*10

    #keep track of loss for print outs
    train_loss += loss.data[0]
    step_cnt += 1
    window_loss += loss.data[0]
    window_steps += 1.0

    # backprop and parameter update
    optimizer.zero_grad()
    loss.backward()
    network.clip_gradient(net, 10.)
    optimizer.step()

    #print out training info
    if step % disp_interval == 0:
        duration = t.toc(average=False)
        fps = step_cnt / duration

        log_text = 'step %d, epoch_avg_loss: %.4f, fps: %.2f (%.2fs per batch) tv_cnt:%d' \
                   'epoch:%d loss: %.4f tot_avg_loss: %.4f' % (
            step,  window_loss/window_steps, fps, 1./fps, 0, 0, loss.data[0],train_loss/step_cnt)
        log_print(log_text, color='green', attrs=['bold'])

       # log_print('\tcls: %.4f, box: %.4f' % (
       #     net.cross_entropy.data.cpu().numpy()[0], net.loss_box.data.cpu().numpy()[0])
       # )



    if step % 1000 == 0:

        net.eval()
        model_name = save_name_base + '_{}'.format(step)
        acc, acc_ish,correct,correct_ish = test_net(model_name, net, val_set,num_images=100)
        #acc, acc_ish = [0,0] 

        dumb_acc_fid = open('./VID_acc_things.txt', 'a')
        dumb_acc_fid.write('-----------\n')
        dumb_acc_fid.write('step: {}\n'.format(step))
        dumb_acc_fid.write('acc: {}\n'.format(acc))
        dumb_acc_fid.write('acc_ish: {}\n'.format(acc_ish))
        dumb_acc_fid.write('-----------\n')
        dumb_acc_fid.close()
        net.train()



    if step % 1000 == 0:

        save_name = os.path.join(output_dir, save_name_base+'_{}_{:1.5f}_{:1.5f}.h5'.format(step + trained_step, window_loss/window_steps, correct))
        network.save_net(save_name, net)

        window_loss = 0
        window_steps = 0






