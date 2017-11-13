import os
import sys
import torch
import torch.utils.data
import torchvision.models as models

import numpy as np
from datetime import datetime
import cv2

from instance_detection.model_defs import network
#from instance_detection.model_defs.tdid import TDID 
#from instance_detection.model_defs.tdid_depthwise_batch import TDID 
#from instance_detection.model_defs.tdid_depthwise_mtargets_batch import TDID 
#from instance_detection.model_defs.tdid_depthwise_mtargets_sim_batch import TDID 
from instance_detection.model_defs.tdid_depthwise_mtargets_simSep_batch import TDID 
#from instance_detection.model_defs.tdid_depthwise_mtargets_simMisha_batch import TDID 
#from instance_detection.model_defs.tdid_depthwise_mtargets_diff_batch import TDID
#from instance_detection.model_defs.tdid_depthwise_mtargets_scales_batch import TDID 
#from instance_detection.model_defs.tdid_mtargets_split_batch import TDID 
#from instance_detection.model_defs.tdid_depthwise_mtargets_bn_batch import TDID 
#from instance_detection.model_defs.tdid_depthwise_plus_bn_batch import TDID 
#from instance_detection.model_defs.tdid_depthwise_sim_batch import TDID 
#from instance_detection.model_defs.tdid_depthwise_scales_batch import TDID 
#from instance_detection.model_defs.tdid_depthwise_plus_batch import TDID 
#from instance_detection.model_defs.tdid_many_measures import TDID 
from instance_detection.model_defs.utils.timer import Timer
from instance_detection.model_defs.fast_rcnn.config import cfg, cfg_from_file

from instance_detection.utils.get_data import get_target_images,match_and_concat_images
from instance_detection.utils.get_data import match_and_concat_images_list, vary_image
#from instance_detection.utils.get_data import * 
from instance_detection.utils.ILSVRC_VID_loader import VID_Loader

from instance_detection.testing.test_tdid_batch_eff import test_net, im_detect

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
text_out_dir = ('/net/bvisionserver3/playpen/ammirato/Data/Detections/' + 
             '/saved_models_meta/')
#save_name_base = 'TDID_archMM_10'
save_name_base = 'TDID_COMB_GEN4GMU_archDmtSimSepbn_ROI_1'

save_freq = 1500

use_batch_norm = True 
use_torch_vgg= True 
use_pretrained_vgg = True
batch_size=6
loss_mult = 1
vary_images = False
#id_map_fname = 'all_instance_id_map.txt'
id_map_fname = 'hybrid_instance_id_map.txt'
#id_map_fname = 'instance_id_map.txt'



trained_model_path = ('/net/bvisionserver3/playpen/ammirato/Data/Detections/' +
                     '/saved_models/')
trained_model_name = 'TDID_COMB_archDmtSim_ROI_1_0_223.66245_0.22533_0.12992.h5'
load_trained_model = False 
trained_epoch = 0 

preload_target_images =  False

num_epochs = 50 



# load config
cfg_from_file(cfg_file)
lr = cfg.TRAIN.LEARNING_RATE 
momentum = cfg.TRAIN.MOMENTUM
weight_decay = cfg.TRAIN.WEIGHT_DECAY
disp_interval =10# cfg.TRAIN.DISPLAY
log_interval = cfg.TRAIN.LOG_IMAGE_ITERS

# load data
data_path = '/net/bvisionserver3/playpen10/ammirato/Data/HalvedRohitData/'
train_list=[
             #'Home_001_1',
#             #'Home_001_2',
             'Home_002_1',
             'Home_003_1',
             'Home_003_2',
             'Home_004_1',
#             'Home_004_2',
             'Home_005_1',
#             'Home_005_2',
             'Home_006_1',
             'Home_008_1',
             'Home_014_1',
#             'Home_014_2',
             'Office_001_1',
##              'Gen_002_1',
              'Gen_003_1',
         'Gen_004_1'
#              'Gen_003_2',
##              'Gen_003_3',
#              'Gen_004_2',
#              'Gen_005_2',
              'Gen_006_2',
#              'Gen_006_2',
#              'Gen_007_2',
#              'Gen_004_3',
#              'Gen_004_3',

#             'Home_101_1',
#             'Home_102_1',
#             'Home_103_1',
#             'Home_104_1',
#             'Home_105_1',
            ]



#train_list=[
#             'Home_101_1',
#             #'Home_102_1',
#             'Home_103_1',
#             'Home_104_1',
#             'Home_105_1',
#             #'Home_106_1',
#             'Home_107_1',
#             'Home_108_1',
#             'Home_109_1',
##             'Home_109_1',
##              'Gen_002_1',
#              'Gen_003_1',
#              'Gen_003_2',
##              'Gen_003_3',
#              'Gen_004_2',
#              'Gen_005_2',
#            ]


val_lists = [[
             'Home_102_1',
             'Home_104_1',
             'Home_105_1',
            ],

            [
             #'Home_001_1',
             #'Home_001_2',
             #'Home_008_1',
            ]]

#pick which objects to include
#will be further refined by the name_to_id_map loaded later
#excluded_cids = [53,76,78,79,82,86,16,   1,2,18,21,25]
excluded_cids = [53,76,78,79,82,86,16,33,32,   50,79,94,96,]#5,10,12,14,18,21,28]
#excluded_cids = []
chosen_ids =  [x for x in range(0,2111) if x not in excluded_cids]
#val_chosen_ids = [[4,5,17,19,23],[18,50,79,94,96]] #, [5,10,17]]#chosen_ids #range(0,28)#for validation testing
#val_chosen_ids = [[18,50,79,94,96,5,10,12,14,21],[5,10,12,14,21]] #, [5,10,17]]#chosen_ids #range(0,28)#for validation testing
val_chosen_ids = [[50,79,94,96],[5]] #, [5,10,17]]#chosen_ids #range(0,28)#for validation testing

max_difficulty = 4 

#get a map from instance name to id, and back
id_to_name = GetDataSet.get_class_id_to_name_dict(data_path, file_name=id_map_fname)
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
#target_path = '/net/bvisionserver3/playpen/ammirato/Data/instance_detection_targets/AVD_BB_exact_few/'
#target_path = '/net/bvisionserver3/playpen/ammirato/Data/instance_detection_targets/AVD_BB_exact_few_and_other_BB_gen/'
#target_path = '/net/bvisionserver3/playpen10/ammirato/Data/instance_detection_targets/AVD_BB_exact_few_and_other_BB_gen_160_varied/'
#target_path = '/net/bvisionserver3/playpen10/ammirato/Data/instance_detection_targets/AVD_BB_exact_few_and_other_BB_gen_160/'
#target_path = '/net/bvisionserver3/playpen10/ammirato/Data/instance_detection_targets/AVD_BB_exact_few_and_other_BB_gen_and_AVD_ns_BB_160/'
target_path = '/net/bvisionserver3/playpen10/ammirato/Data/instance_detection_targets/AVD_BB_exact_few_and_other_BB_gen_and_AVD_ns_BB_and_UW_80/'
val_target_path = '/net/bvisionserver3/playpen10/ammirato/Data/instance_detection_targets/AVD_BB_exact_few_and_other_BB_gen_and_AVD_ns_BB_80/'
#target_path = '/net/bvisionserver3/playpen10/ammirato/Data/instance_detection_targets/AVD_BB_exact_few_and_other_BB_gen_and_AVD_ns_BB_80/'
#val_target_path = '/net/bvisionserver3/playpen10/ammirato/Data/instance_detection_targets/AVD_BB_exact_few_and_other_BB_gen_160/'
val_target_path = target_path
target_images = get_target_images(target_path,name_to_id.keys(),
                                  preload_images=preload_target_images,means=None)
if use_torch_vgg:
    val_target_images = get_target_images(target_path,name_to_id.keys(),
                                          for_testing=True, bn_normalize=True)
else:
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
                                          fraction_of_no_box=0.1,
                                          bn_normalize=use_torch_vgg,
                                          to_tensor=False)

#create train/test loaders, with CUSTOM COLLATE function
trainloader = torch.utils.data.DataLoader(train_set,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          collate_fn=AVD.collate)



print 'train_set_good'


if save_freq > len(train_set)/batch_size:
    save_freq = len(train_set)/batch_size - 5*batch_size
    print save_freq


use_VID = True 
VID_data_path = '/net/bvisionserver3/playpen10/ammirato/Data/ILSVRC/'
target_size = [200,16]
##CREATE TRAIN/TEST splits
vid_train_set = VID_Loader(VID_data_path,'train_single', target_size=target_size, multiple_targets=True, batch_size=batch_size)



#write meta data out
meta_fid = open(os.path.join(text_out_dir,save_name_base+'.txt'),'w')
meta_fid.write('save name: {}\n'.format(save_name_base))
meta_fid.write('batch norm: {}\n'.format(use_batch_norm))
meta_fid.write('torch vgg: {}\n'.format(use_torch_vgg))
meta_fid.write('pretrained vgg: {}\n'.format(use_pretrained_vgg))
meta_fid.write('batch_size: {}\n'.format(batch_size))
meta_fid.write('vary images: {}\n'.format(vary_images))
meta_fid.write('chosen_ids: {}\n'.format(chosen_ids))
meta_fid.write('val chosen_ids: {}\n'.format(val_chosen_ids))
meta_fid.write('train_list: {}\n'.format(train_list))
meta_fid.write('val_lists: {}\n'.format(val_lists))
meta_fid.write('target_path: {}\n'.format(target_path))
meta_fid.write('val target_path: {}\n'.format(val_target_path))
if use_VID:
    meta_fid.write('VID_target_size: {}\n'.format(target_size))
    meta_fid.write('vid_set: {}\n'.format('train_single'))
meta_fid.write('learing rate: {}\n'.format(lr))
if load_trained_model:
    meta_fid.write('start from: {}\n'.format(trained_model_name))
meta_fid.close()


#load net definition and init parameters
net = TDID()
if load_trained_model:
    #load a previously trained model
    if use_batch_norm:
        vgg16_bn = models.vgg16_bn(pretrained=False)
        net.features = torch.nn.Sequential(*list(vgg16_bn.features.children())[:-1])
        net.features.eval()
    elif use_torch_vgg:
        vgg16 = models.vgg16(pretrained=False)
        net.features = torch.nn.Sequential(*list(vgg16.features.children())[:-1])
    network.load_net(trained_model_path + trained_model_name, net)
else:
    #load pretrained vgg weights, and init everything else randomly
    network.weights_normal_init(net, dev=0.01)
    if use_batch_norm:
        if use_pretrained_vgg:
            vgg16_bn = models.vgg16_bn(pretrained=True)
        else:
            vgg16_bn = models.vgg16_bn(pretrained=False)
        net.features = torch.nn.Sequential(*list(vgg16_bn.features.children())[:-1])
        net.features.eval()
    elif use_torch_vgg:
        if use_pretrained_vgg:
            vgg16 = models.vgg16(pretrained=True)
        else:
            vgg16 = models.vgg16(pretrained=False)
        net.features = torch.nn.Sequential(*list(vgg16.features.children())[:-1])
    else:
        if use_pretrained_vgg:
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
        step = step+1
        if type(batch[0]) is not list or len(batch[0]) < batch_size:
            continue

        batch_im_data = []
        batch_target_data = []
        batch_gt_boxes = []
        for sample_ind in range(batch_size):
            # get one batch, image and bounding boxes
            ##im_data=batch[0].unsqueeze(0).numpy()
            #im_data=batch[0][sample_ind].numpy()
            #im_data=np.transpose(im_data,(1,2,0)) 
            im_data=batch[0][sample_ind]


            gt_boxes = np.asarray(batch[1][sample_ind][0],dtype=np.float32) 
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

                if np.random.rand() < .9 and vary_images:
                    target_img = vary_image(target_img)
                #subtract means, give batch dimension, add to list
                if use_torch_vgg:
                    target_img = ((target_img/255.0) - [0.485, 0.456, 0.406])/[0.229, 0.224, 0.225] 
                else:
                    target_img = target_img - means

                #target_img = np.expand_dims(target_img,axis=0)
                batch_target_data.append(target_img)

            batch_im_data.append(im_data)
            #batch_gt_boxes.append(gt_boxes)
            batch_gt_boxes.extend(gt_boxes)



        #target_data = match_and_concat_images(target_data[0][0,:],target_data[1][0,:])
        target_data = match_and_concat_images_list(batch_target_data, min_size=32)
        im_data = match_and_concat_images_list(batch_im_data)
        #gt_boxes = np.concatenate((batch_gt_boxes[0],batch_gt_boxes[1])) 
        gt_boxes = np.asarray(batch_gt_boxes) 

        # forward
        ir_cnt +=1
        net(target_data, im_data,gt_boxes=gt_boxes)
        loss = net.loss
        loss = net.loss*loss_mult

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












        if use_VID:

            #gets a random batch
            batch = vid_train_set[0]

            #get first and second images
            #im_data = match_and_concat_images(batch[0],batch[3]) -127
            im_data = match_and_concat_images_list(batch[0])
            if use_torch_vgg:
                im_data = ((im_data/255.0) - [0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]
            else:
                im_data = im_data - means
            #get target images
            target_data = match_and_concat_images_list(batch[2]) 
            if use_torch_vgg:
                target_data = ((target_data/255.0) - [0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]
            else:
                target_data = target_data-means

            #get gt box, add 1 for fg label
            gt_boxes = np.asarray(batch[1])




            # forward
            net(target_data, im_data, gt_boxes)
            loss = net.loss
            loss = net.loss * loss_mult

            #keep track of loss for print outs
            #train_loss += loss.data[0]
            #step_cnt += 1
            #window_loss += loss.data[0]
            #window_steps += 1.0 

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
                       'ir_cnt:%d epoch:%d loss: %.4f tot_avg_loss: %.4f %s' % (
                step,  epoch_loss/epoch_step_cnt, fps, 1./fps, tv_cnt, ir_cnt, epoch, loss.data[0],train_loss/step_cnt, save_name_base)
            log_print(log_text, color='green', attrs=['bold'])
            print(targets_cnt)

            log_print('\tcls: %.4f, box: %.4f' % (
                net.cross_entropy.data.cpu().numpy()[0], net.loss_box.data.cpu().numpy()[0])
            )

        ######################################################
        #epoch over
        #test validation set, save a checkpoint
        if step % save_freq == 0:
            
            m_aps = []
            for val_ind, vci in enumerate(val_chosen_ids):
            #test net on some val data
                data_path = '/net/bvisionserver3/playpen10/ammirato/Data/HalvedRohitData/'
         #       scene_list=[
         #                #'Home_003_1',
         #                #'Gen_002_1',
         #                #'Home_014_1',
         #                #'Home_003_2',
         #                #'test',
         #                #'Office_001_1'

         #                'Home_101_1',
         #                'Home_102_1',


         #                ]

                scene_list = val_lists[val_ind]
                if scene_list == []:
                    m_aps.append(-1)
                    continue
                #CREATE TRAIN/TEST splits
                valset = GetDataSet.get_fasterRCNN_AVD(data_path,
                                                        scene_list,
                                                        preload=False,
                                                        #chosen_ids=val_chosen_ids, 
                                                        chosen_ids=vci, 
                                                        by_box=False,
                                                        max_difficulty=max_difficulty,
                                                        fraction_of_no_box=.01,
                                                        bn_normalize=use_torch_vgg)

                #create train/test loaders, with CUSTOM COLLATE function
                valloader = torch.utils.data.DataLoader(valset,
                                                          batch_size=1,
                                                          shuffle=True,
                                                          collate_fn=AVD.collate)
                print 'Testing...'

                net.eval()
                max_per_target = 15
                thresh = 0
                model_name = save_name_base + '_{}'.format(epoch)
                t_output_dir='/net/bvisionserver3/playpen/ammirato/Data/Detections/FasterRCNN_AVD/'
                all_results = test_net(model_name, net, valloader, name_to_id, 
                                       #val_target_images,val_chosen_ids, 
                                       val_target_images,vci, 
                                       max_per_target=max_per_target,
                                       output_dir=t_output_dir,
                                       thresh=thresh)

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
            if use_batch_norm:
                net.features.eval()

        #if  % save_freq == 0:
            save_epoch = epoch
            if load_trained_model:
                save_epoch = epoch+trained_epoch+1

            #save_name = os.path.join(output_dir, save_name_base+'_{}_{:1.5f}_{:1.5f}_{:1.5f}.h5'.format(save_epoch, epoch_loss/epoch_step_cnt, m_aps[0], m_aps[0]))
            save_name = os.path.join(output_dir, save_name_base+'_{}_{}_{:1.5f}_{:1.5f}_{:1.5f}.h5'.format(save_epoch,step, epoch_loss/epoch_step_cnt, m_aps[0], m_aps[1]))
            network.save_net(save_name, net)
            print('save model: {}'.format(save_name))





