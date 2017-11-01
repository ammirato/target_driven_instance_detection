import os
import sys
import torch
import torch.utils.data
import numpy as np
from datetime import datetime
import cv2

from instance_detection.model_defs import network
#from instance_detection.model_defs.tdid import TDID 
from instance_detection.model_defs.tdid_depthwise import TDID 
#from instance_detection.model_defs.tdid_many_measures import TDID 
from instance_detection.model_defs.utils.timer import Timer
from instance_detection.model_defs.fast_rcnn.config import cfg, cfg_from_file

from instance_detection.utils.get_data import get_target_images

from instance_detection.testing.test_tdid import test_net, im_detect

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
pretrained_model = '/net/bvisionserver3/playpen/ammirato/Data/Detections/pretrained_models/VGG_imagenet.npy'
output_dir = ('/net/bvisionserver3/playpen/ammirato/Data/Detections/' + 
             '/saved_models/')
#save_name_base = 'TDID_archMM_10'
save_name_base = 'TDID_archD_0'
save_freq = 1 

trained_model_path = ('/net/bvisionserver3/playpen/ammirato/Data/Detections/' +
                     '/saved_models/')
trained_model_name = 'TDID_VID_archD_1_98000_30.15133.h5'
load_trained_model = True 
trained_epoch = 98000 

preload_target_images =  False

num_epochs = 50 

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
data_path = '/net/bvisionserver3/playpen/ammirato/Data/HalvedRohitData/'
train_list=[
             'Home_001_1',
             'Home_001_2',
             'Home_002_1',
             'Home_004_1',
             'Home_004_2',
             'Home_005_1',
             'Home_005_2',
             'Home_006_1',
             'Home_008_1',
             'Home_014_1',
             'Home_014_2',
#              'Gen_002_1',
#              'Gen_003_1',
#              'Gen_003_2',
#              'Gen_003_3',
            ]

#pick which objects to include
#will be further refined by the name_to_id_map loaded later
#excluded_cids = [53,76,78,79,82,86,16,   1,2,18,21,25]
excluded_cids = [53,76,78,79,82,86,16,33,32,   1,2,18,21,25]
chosen_ids =  [x for x in range(0,28) if x not in excluded_cids]
val_chosen_ids = [chosen_ids]# [[1,2,18,21,25]] #, [5,10,17]]#chosen_ids #range(0,28)#for validation testing

max_difficulty = 4 

#get a map from instance name to id, and back
id_to_name = GetDataSet.get_class_id_to_name_dict(data_path)
name_to_id = {}
for cid in id_to_name.keys():
    name_to_id[id_to_name[cid]] = cid


##prepare target images (gather paths to the images)
#
target_images ={} 
#means to subtract from each channel of target image
means = np.array([[[102.9801, 115.9465, 122.7717]]])

#path that holds dirs of all targets
#i.e. target_path/target_0/* has one type of target image for each object
#     target_path/target_1/* has another type of target image
#type of target image can mean different things, 
#probably different type is different view
#each type can have multiple images, 
#i.e. target_0/* can have multiple images per object
#target_path = '/net/bvisionserver3/playpen/ammirato/Data/instance_detection_targets/AVD_single_bb_targets/'
#target_path = '/net/bvisionserver3/playpen/ammirato/Data/instance_detection_targets/sygen_many_bb_similar_targets/'
target_path = '/net/bvisionserver3/playpen/ammirato/Data/instance_detection_targets/AVD_BB_exact_few/'
#target_path = '/net/bvisionserver3/playpen/ammirato/Data/instance_detection_targets/AVD_BB_exact_few_and_other_BB_gen/'
target_images = get_target_images(target_path,name_to_id.keys(),
                                  preload_images=preload_target_images)
val_target_images = get_target_images(target_path,name_to_id.keys(),
                                      for_testing=True, means=means)




#make sure only targets that have ids, and have target images are chosen
chosen_ids = list(set(set(chosen_ids) & set(name_to_id.values())))
for vci in val_chosen_ids:
    vci = list(set(set(vci) & set(name_to_id.values())))
for cid in chosen_ids:
    if cid == 0:
        continue
    if ((len(target_images[id_to_name[cid]]) < 1) or 
            (len(target_images[id_to_name[cid]][0])) < 1):
        print('Missing target images for {}!'.format(id_to_name[cid]))
        sys.exit()

#CREATE TRAIN/TEST splits
train_set = GetDataSet.get_fasterRCNN_AVD(data_path,
                                          train_list,
                                          #test_list,
                                          max_difficulty=max_difficulty,
                                          chosen_ids=chosen_ids,
                                          by_box=False,
                                          fraction_of_no_box=0.02)

#create train/test loaders, with CUSTOM COLLATE function
trainloader = torch.utils.data.DataLoader(train_set,
                                          batch_size=1,
                                          shuffle=True,
                                          collate_fn=AVD.collate)

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
# optimizer = torch.optim.Adam(params[-8:], lr=lr)
#optimizer = torch.optim.SGD(params[8:], lr=lr, momentum=momentum, weight_decay=weight_decay)
optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

#make sure dir for saving model checkpoints exists
if not os.path.exists(output_dir):
    os.mkdir(output_dir)


# things to print out during training training
train_loss = 0
step_cnt = 0
t = Timer()
t.tic()



for epoch in range(num_epochs):
    #more things to print out later
    tv_cnt = 0
    ir_cnt = 0
    targets_cnt = {}#how many times a target is used(visible, total)
    for cid in chosen_ids:
        targets_cnt[cid] = [0,0]
    epoch_loss = 0
    epoch_step_cnt = 0
    for step,batch in enumerate(trainloader):

        # get one batch, image and bounding boxes
        im_data=batch[0].unsqueeze(0).numpy()
        im_data=np.transpose(im_data,(0,2,3,1))
        gt_boxes = np.asarray(batch[1][0],dtype=np.float32)
        #if there are no boxes for this image, add a dummy background box
        if gt_boxes.shape[0] == 0:
            gt_boxes = np.asarray([[0,0,1,1,0]])
 
        #get the gt inds that are in this image, not counting 0(background)
        objects_present = gt_boxes[:,4]
        objects_present = objects_present[np.where(objects_present!=0)[0]]
        #get the ids of objects that are not in this image
        not_present = np.asarray([ind for ind in chosen_ids 
                                          if ind not in objects_present and 
                                             ind != 0]) 

        #pick a random target, with a bias towards choosing a target that 
        #is in the image. Also pick just that object's gt_box
        if (np.random.rand() < .6 or not_present.shape[0]==0) and objects_present.shape[0]!=0:
            target_ind = int(np.random.choice(objects_present))
            gt_boxes = gt_boxes[np.where(gt_boxes[:,4]==target_ind)[0],:-1]
            gt_boxes[0,4] = 1

            tv_cnt += 1
            targets_cnt[target_ind][0] += 1 
        else:#the target is not in the image, give a dummy background box
            target_ind = int(np.random.choice(not_present))
            gt_boxes = np.asarray([[0,0,1,1,0]])

        #get the target images
        targets_cnt[target_ind][1] += 1 
        target_name = id_to_name[target_ind]
        target_data = []
        #get one image for each target type
        for t_type,_ in enumerate(target_images[target_name]):
            #pick one image of this type
            img_ind = np.random.choice(np.arange(
                                  len(target_images[target_name][t_type])))
            if preload_target_images:
                target_img = target_images[target_name][t_type][img_ind]
            else:
                target_img = cv2.imread(target_images[target_name][t_type][img_ind])

            #subtract means, give batch dimension, add to list
            target_img = target_img - means
            target_img = np.expand_dims(target_img,axis=0)
            target_data.append(target_img)
        

        #TODO: lose this stuff
        im_info = np.zeros((1,3))
        im_info[0,:] = [im_data.shape[1],im_data.shape[2],1]
        gt_ishard = np.zeros(gt_boxes.shape[0])
        dontcare_areas = np.zeros((0,4))

        # forward
        ir_cnt +=1
        net(target_data, im_data, im_info, gt_boxes, gt_ishard, dontcare_areas)
        loss = net.loss
        loss = net.loss*10

        #keep track of loss for print outs
        train_loss += loss.data[0]
        step_cnt += 1
        epoch_step_cnt += 1
        epoch_loss += loss.data[0]

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
                       'ir_cnt:%d epoch:%d loss: %.4f tot_avg_loss: %.4f' % (
                step,  epoch_loss/epoch_step_cnt, fps, 1./fps, tv_cnt, ir_cnt, epoch, loss.data[0],train_loss/step_cnt)
            log_print(log_text, color='green', attrs=['bold'])
            print(targets_cnt)

            log_print('\tcls: %.4f, box: %.4f' % (
                net.cross_entropy.data.cpu().numpy()[0], net.loss_box.data.cpu().numpy()[0])
            )

    ######################################################
    #epoch over
    #test validation set, save a checkpoint
    if epoch % save_freq == 0:
        
        m_aps = []
        for vci in val_chosen_ids:
        #test net on some val data
            data_path = '/net/bvisionserver3/playpen/ammirato/Data/HalvedRohitData/'
            scene_list=[
                     'Home_003_1',
                     #'Gen_002_1',
                     #'Home_014_1',
                     #'Home_003_2',
                     #'test',
                     #'Office_001_1'
                     ]
            #CREATE TRAIN/TEST splits
            valset = GetDataSet.get_fasterRCNN_AVD(data_path,
                                                    scene_list,
                                                    preload=False,
                                                    #chosen_ids=val_chosen_ids, 
                                                    chosen_ids=vci, 
                                                    by_box=False,
                                                    max_difficulty=max_difficulty,
                                                    fraction_of_no_box=0)

            #create train/test loaders, with CUSTOM COLLATE function
            valloader = torch.utils.data.DataLoader(valset,
                                                      batch_size=1,
                                                      shuffle=True,
                                                      collate_fn=AVD.collate)
            print 'Testing...'

            net.eval()
            max_per_target = 5
            model_name = save_name_base + '_{}'.format(epoch)
            t_output_dir='/net/bvisionserver3/playpen/ammirato/Data/Detections/FasterRCNN_AVD/'
            all_results = test_net(model_name, net, valloader, name_to_id, 
                                   #val_target_images,val_chosen_ids, 
                                   val_target_images,vci, 
                                   max_per_target=max_per_target, output_dir=t_output_dir)

            gt_labels= valset.get_original_bboxes()
            evaluater = DetectorEvaluater(score_thresholds=np.linspace(0,1,111),
                                          recall_thresholds=np.linspace(0,1,11))
            m_ap = evaluater.run(
                        #all_results,gt_labels,chosen_ids,
                        all_results,gt_labels,vci,
                        max_difficulty=max_difficulty,
                        difficulty_classifier=valset.get_box_difficulty)
            m_aps.append(m_ap)
        net.train()


    if epoch % save_freq == 0:
        save_epoch = epoch
        if load_trained_model:
            save_epoch = epoch+trained_epoch+1

        #save_name = os.path.join(output_dir, save_name_base+'_{}_{:1.5f}_{:1.5f}_{:1.5f}.h5'.format(save_epoch, epoch_loss/epoch_step_cnt, m_aps[0], m_aps[0]))
        save_name = os.path.join(output_dir, save_name_base+'_{}_{:1.5f}_{:1.5f}.h5'.format(save_epoch, epoch_loss/epoch_step_cnt, m_aps[0]))
        network.save_net(save_name, net)
        print('save model: {}'.format(save_name))





