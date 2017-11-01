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
from instance_detection.model_defs.tdid_depthwise import TDID 
#from instance_detection.model_defs.tdid_many_measures import TDID 
from instance_detection.model_defs.utils.timer import Timer
from instance_detection.model_defs.fast_rcnn.config import cfg, cfg_from_file

from instance_detection.utils.get_data import get_target_images, match_and_concat_images
from instance_detection.utils.ILSVRC_VID_loader import VID_Loader 

from instance_detection.testing.test_tdid_VID_single_batch import test_net, im_detect

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
save_name_base = 'TDID_VID_archDs_0'
save_freq = 1 

#trained_model_path = ('/net/bvisionserver3/playpen/ammirato/Data/Detections/' +
#                     '/saved_models/')
trained_model_path = ('/playpen/ammirato/Data/Detections/' +
                     '/saved_models/')
trained_model_name = 'TDID_VID_archD_0_27000_0.07397.h5'
load_trained_model = False 
trained_step =  0

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
lr = cfg.TRAIN.LEARNING_RATE * 10 
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval =10# cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS

# load data
data_path = '/playpen/ammirato/Downloads/ILSVRC/'


#CREATE TRAIN/TEST splits
train_set = VID_Loader(data_path,'val_single')
val_set = VID_Loader(data_path,'val2_single')

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

group_loss = 0
group_steps = 0 


for step in range(max_steps):

    #gets a random batch
    batch = train_set[0]

    #get first and second images
    im_data = np.expand_dims(batch[0]-127, 0)
    im_info = [im_data.shape[1:]]
    #get target images
    target_data = [np.expand_dims(batch[2][0]-127,0),np.expand_dims(batch[2][1]-127,0)]

    #get gt box, add 1 for fg label
    gt_box = batch[1]
    gt_box.append(1)
    gt_box = np.asarray(gt_box,dtype=np.float32)
    #1 gt_box for each image, the second is a dummy box since there is no fg
    gt_boxes = np.asarray([gt_box])

    # forward
    net(target_data, im_data,im_info, gt_boxes)
    loss = net.loss
    #loss = net.loss*10
    group_loss += loss.data[0]
    group_steps +=1


    #keep track of loss for print outs
    train_loss += loss.data[0]
    step_cnt += 1

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
            step,  0, fps, 1./fps, 0, 0, loss.data[0],train_loss/step_cnt)
        log_print(log_text, color='green', attrs=['bold'])

       # log_print('\tcls: %.4f, box: %.4f' % (
       #     net.cross_entropy.data.cpu().numpy()[0], net.loss_box.data.cpu().numpy()[0])
       # )



    if step % 500 == 0:

        net.eval()
        model_name = save_name_base + '_{}'.format(step)
        acc, acc_ish = test_net(model_name, net, val_set,num_images=200)

        dumb_acc_fid = open('./VID_acc_things.txt', 'a')
        dumb_acc_fid.write('-----------\n')
        dumb_acc_fid.write('step: {}\n'.format(step))
        dumb_acc_fid.write('acc: {}\n'.format(acc))
        dumb_acc_fid.write('acc_ish: {}\n'.format(acc_ish))
        dumb_acc_fid.write('-----------\n')
        dumb_acc_fid.close()
        net.train()



    if step % 2500 == 0:

        save_name = os.path.join(output_dir, save_name_base+'_{}_{:1.5f}{:1.5f}.h5'.format(step + trained_step, acc, group_loss/ float(group_steps)))
        network.save_net(save_name, net)
        group_loss = 0
        group_steps = 0


    ######################################################
    #epoch over
    #test validation set, save a checkpoint
    #if epoch % save_freq == 0:
    #    
    #    m_aps = []
    #    for vci in val_chosen_ids:
    #    #test net on some val data
    #        data_path = '/net/bvisionserver3/playpen/ammirato/Data/HalvedRohitData/'
    #        scene_list=[
    #                 'Home_003_1',
    #                 #'Gen_002_1',
    #                 #'Home_014_1',
    #                 #'Home_003_2',
    #                 #'test',
    #                 #'Office_001_1'
    #                 ]
    #        #CREATE TRAIN/TEST splits
    #        valset = GetDataSet.get_fasterRCNN_AVD(data_path,
    #                                                scene_list,
    #                                                preload=False,
    #                                                #chosen_ids=val_chosen_ids, 
    #                                                chosen_ids=vci, 
    #                                                by_box=False,
    #                                                max_difficulty=max_difficulty,
    #                                                fraction_of_no_box=0)

    #        #create train/test loaders, with CUSTOM COLLATE function
    #        valloader = torch.utils.data.DataLoader(valset,
    #                                                  batch_size=1,
    #                                                  shuffle=True,
    #                                                  collate_fn=AVD.collate)
    #        print 'Testing...'

    #        net.eval()
    #        max_per_target = 5
    #        model_name = save_name_base + '_{}'.format(epoch)
    #        t_output_dir='/net/bvisionserver3/playpen/ammirato/Data/Detections/FasterRCNN_AVD/'
    #        all_results = test_net(model_name, net, valloader, name_to_id, 
    #                               #val_target_images,val_chosen_ids, 
    #                               val_target_images,vci, 
    #                               max_per_target=max_per_target, output_dir=t_output_dir)

    #        gt_labels= valset.get_original_bboxes()
    #        evaluater = DetectorEvaluater(score_thresholds=np.linspace(0,1,111),
    #                                      recall_thresholds=np.linspace(0,1,11))
    #        m_ap = evaluater.run(
    #                    #all_results,gt_labels,chosen_ids,
    #                    all_results,gt_labels,vci,
    #                    max_difficulty=max_difficulty,
    #                    difficulty_classifier=valset.get_box_difficulty)
    #        m_aps.append(m_ap)
    #    net.train()


    #if epoch % save_freq == 0:
    #    save_epoch = epoch
    #    if load_trained_model:
    #        save_epoch = epoch+trained_epoch+1

    #    #save_name = os.path.join(output_dir, save_name_base+'_{}_{:1.5f}_{:1.5f}_{:1.5f}.h5'.format(save_epoch, epoch_loss/epoch_step_cnt, m_aps[0], m_aps[0]))
    #    save_name = os.path.join(output_dir, save_name_base+'_{}_{:1.5f}_{:1.5f}.h5'.format(save_epoch, epoch_loss/epoch_step_cnt, m_aps[0]))
    #    network.save_net(save_name, net)
    #    print('save model: {}'.format(save_name))





