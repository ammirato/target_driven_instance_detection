import torch
import torchvision.models as models
import os
import cv2
import numpy as np
import math
import sys

import active_vision_dataset_processing.data_loading.active_vision_dataset_pytorch as AVD
import active_vision_dataset_processing.data_loading.transforms as AVD_transforms



def create_illumination_pattern(rows, cols, xCenter,yCenter,minI=.1,maxI=1,radius=None):
    if radius is None:
        radius = float(int(20000 + (30000)*np.random.rand(1)))

    pattern = np.zeros((rows, cols));
    for row in range(rows):
        for col in range(cols):
            dy = row - yCenter;
            dx = col - xCenter;
            pattern[row,col] = (minI + (maxI -minI) 
                                * math.exp(-(.5)*(dx*dx + dy*dy)/ radius))

    return pattern



illum_patterns = []
for il in range(100):
    xc,yc = np.random.choice(160,2)
    illum_patterns.append(create_illumination_pattern(160,160,xc,yc))






def get_target_images(target_path, target_names,preload_images=False, for_testing=False,
                      means=None, pytorch_normalize=False):
    """
    returns dict with path to each target image, or loaded image

    Ex) get_target_images('target_path', ['possible','target','names'])

    ARGS:
        target_path -path that holds directories of all targets.
                     i.e. target_path/target_0/* has one type 
                     of target image for each object.
                     target_path/target_1/* has another type 
                     of target image for each object.
                     Each type can have multiple images, 
                     i.e. target_0/* can have multiple images per object
        target_names - 

    KWARGS:
        preload_images - 
        for_testing  -  loads 1 image per object per type
        pytorch_normalize -  
    """


    #type of target image can mean different things, 
    #probably different type is different view
    target_dirs = os.listdir(target_path)
    target_dirs.sort()
    target_images = {}
    #each target gets a list of lists, one for each type dir
    for name in target_names:
        target_images[name] = []

    for type_ind, t_dir in enumerate(target_dirs):
        for name in os.listdir(os.path.join(target_path,t_dir)):

            if name.find('N') == -1:
                obj_name = name[:name.rfind('_')]
            else:
                obj_name = name[:name.find('N')-1]

            #make sure object is valid, and load the image or store path
            if obj_name in target_names:
                #make sure this type has a list
                if len(target_images[obj_name]) <= type_ind:
                    target_images[obj_name].append([])
                if preload_images:
                    target_images[obj_name][type_ind].append(cv2.imread(
                                            os.path.join(target_path,t_dir,name)))
                else:
                    target_images[obj_name][type_ind].append(
                                            os.path.join(target_path,t_dir,name))


    #for testing, only 1 image per type, and must be loaded
    if for_testing:
        for target_name in target_images.keys():
            cur_images = target_images[target_name]
            all_types = []
            for t_type,_ in enumerate(cur_images):
                #just grab the first image of each type
                if preload_images:
                    img = cur_images[t_type][0]
                else:
                    img = cv2.imread(cur_images[t_type][0])
                if means is not None:
                    img = img-means
                if pytorch_normalize:
                    img = img / 255.0
                    img = (img-[0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]

                all_types.append(np.expand_dims(img, axis=0))
            target_images[target_name] = all_types

    return target_images








def match_and_concat_images(img1, img2, min_size=None):
    """
    Returns both images stacked and padded with zeros

    """
    max_rows = max(img1.shape[0], img2.shape[0])
    max_cols = max(img1.shape[1], img2.shape[1])

    if min_size is not None:
        max_rows = max(max_rows,min_size)
        max_cols = max(max_cols,min_size)

    resized_img1 = np.zeros((max_rows,max_cols,img1.shape[2]))
    resized_img2 = np.zeros((max_rows,max_cols,img2.shape[2]))

    resized_img1[0:img1.shape[0],0:img1.shape[1],:] = img1
    resized_img2[0:img2.shape[0],0:img2.shape[1],:] = img2

    return np.stack((resized_img1,resized_img2),axis=0) 

def match_and_concat_images_list(img_list, min_size=None):
    """
    Returns both images stacked and padded with zeros

    """
    max_rows = 0
    max_cols = 0
    for img in img_list:
        max_rows = max(img.shape[0], max_rows)
        max_cols = max(img.shape[1], max_cols)

    if min_size is not None:
        max_rows = max(max_rows,min_size)
        max_cols = max(max_cols,min_size)


    for il,img in enumerate(img_list):

        resized_img = np.zeros((max_rows,max_cols,img.shape[2]))
        resized_img[0:img.shape[0],0:img.shape[1],:] = img
        img_list[il] = resized_img

    return np.stack(img_list,axis=0) 






def vary_image(img, crop_max=5, rotate_max=30,blur_max=9, do_illum=True):

    #crop
    crops = np.random.choice(crop_max,4)   
    start_row = 0 + crops[0]
    end_row = img.shape[0] - crops[1] 
    start_col = 0 + crops[2]
    end_col = img.shape[1] - crops[3]
    img = img[start_row:end_row, start_col:end_col,:]



    #rotate
    rot_angle = np.random.choice(rotate_max*2,1) - rotate_max
    M = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),rot_angle,1)
    img = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))


    #change illumination
    if do_illum:
        pattern = illum_patterns[int(np.random.choice(len(illum_patterns),1))]
        pattern = pattern[0:img.shape[0],0:img.shape[1]]
        img = img* np.tile(np.expand_dims(pattern,2),(1,1,3)) 



    return img







def check_object_ids(chosen_ids,id_to_name,target_images):
    """
    Picks only chosen ids that have a target object and target image.

    ex) check_object_ids(chosen_ids,id_to_name,target_images)
        Returns only ids in chosen ids that exist in id_to_name dict, and 
        returns -1 if any id does not have a target image 
    """

    
    ids_with_name = list(set(set(chosen_ids) & set(id_to_name.keys())))
    for cid in ids_with_name:
        if cid == 0:#skip background
            continue
    
        if ((len(target_images[id_to_name[cid]]) < 1) or  
                (len(target_images[id_to_name[cid]][0])) < 1): 
            print('Missing target images for {}!'.format(id_to_name[cid]))
            return -1
    return ids_with_name




def normalize_image(image,cfg):
    """
    Noramlizes image according to config parameters
    
    ex) normalize_image(image,config)
    """
    if cfg.PYTORCH_FEATURE_NET:
        return ((image/255.0) - [0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]
    else:
        print('only pytorch feature nets supported at this time!')
        return -1





def get_class_id_to_name_dict(root,file_name='instance_id_map.txt'):
    """
    Returns a dict from integer class id to string name
    """
    map_file = open(os.path.join(root,file_name),'r')
    id_to_name_dict = {}
    for line in map_file:
        line = str.split(line)
        id_to_name_dict[int(line[1])] = line[0]
    return id_to_name_dict



def get_AVD_dataset(root, scene_list, chosen_ids,
                       max_difficulty=4,
                       fraction_of_no_box=.1,
                       instance_fname=None,
                      ):
    """
    Returns a dataloader for the AVD dataset.

    dataset = get_AVD_dataset('/path/to/data', ['scene1','scene2,...], [chosen_ids])


    ARGS:
        root: path to data. Parent of all scene directories
        scene_list: scenes to include
        chosen_ids: list of object ids to keep labels for
                    (other labels discarded) 
    KEYWORD ARGS:
        max_difficulty(int=4): max bbox difficulty to use 
    """
    ##initialize transforms for the labels
    #only consider boxes from the chosen classes
    pick_trans = AVD_transforms.PickInstances(chosen_ids,
                                              max_difficulty=max_difficulty)
    #compose the transforms in a specific order, first to last
    target_trans = AVD_transforms.Compose([
                                           pick_trans,
                                          ])
    if instance_fname is None:
        id_to_name_dict = get_class_id_to_name_dict(root)
    else:
        id_to_name_dict = get_class_id_to_name_dict(root,instance_fname)

    dataset = AVD.AVD(root=root,
                         scene_list=scene_list,
                         target_transform=target_trans,
                         classification=False,
                         class_id_to_name=id_to_name_dict,
                         fraction_of_no_box=fraction_of_no_box)
    return dataset





def write_training_meta(cfg,net):
    """
    Writes a text file that describes model and paramters.
    
    ex) write_training_meta(cfg,net)
    """
    meta_fid = open(os.path.join(cfg.META_SAVE_DIR, cfg.MODEL_BASE_SAVE_NAME + '.txt'),'w')
   
    config_params = [attr for attr in dir(cfg)
                     if not callable(getattr(cfg, attr)) 
                     and not attr.startswith("__")] 
    for param in config_params:
        if param == 'ID_TO_NAME' or param == 'NAME_TO_ID':
            continue
        meta_fid.write('{}: {}\n'.format(param, str(getattr(cfg,param))))

    meta_fid.write(net.__str__())
    meta_fid.close()



def load_pretrained_weights(model_name):
    if model_name == 'vgg16_bn':
        vgg16_bn = models.vgg16_bn(pretrained=True)
        return torch.nn.Sequential(*list(vgg16_bn.features.children())[:-1])
    elif model_name == 'squeezenet1_1':
        fnet = models.squeezenet1_1(pretrained=True)
        return torch.nn.Sequential(*list(fnet.features.children())[:-1])
    elif model_name == 'resnet101':
        fnet = models.resnet101(pretrained=True)
        return torch.nn.Sequential(*list(fnet.children())[:-2])
    else:
        print 'model name {} not supported!'.format(model_name)
        sys.exit()



