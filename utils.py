import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import os
import cv2
import numpy as np
import math
import sys
import h5py

import active_vision_dataset_processing.data_loading.active_vision_dataset as AVD
import active_vision_dataset_processing.data_loading.transforms as AVD_transforms

#TODO check gradient clipping



def get_target_images(target_path, target_names,preload_images=False):
    """
    Returns dict with path to each target image, or loaded image

    Ex) get_target_images('target_path', ['possible','target','names'])

    Input parameters:
        target_path: (str) path that holds directories of all targets.
                     i.e. target_path/target_0/* has one type 
                     of target image for each object.
                     target_path/target_1/* has another type 
                     of target image for each object.
                     Each type can have multiple images, 
                     i.e. target_0/* can have multiple images per object
        target_names: (list) list of str, each element is a target name
        
        preload_images (optional): (bool) If True, the return dict will have 
                                   images(as ndarrays) as values. If False,
                                   the values will be full paths to the images.
                                           
     Returns:
        (dict) key=target_name, value=list of lists
               Parent list has one list for each target type.
               Elments in child list are either path to image, or loaded image 

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
    return target_images




def match_and_concat_images_list(img_list, min_size=None):
    """
    Stacks image in a list into a single ndarray 

    Input parameters:
        img_list: (list) list of ndarrays, images to be stacked. If images
                  are not the same shape, zero padding will be used to make
                  them the same size. 

        min_size (optional): (int) If not None, ensures images are at least
                             min_size x min_size. Default: None 

    Returns:
        (ndarray) a single ndarray with first dimension equal to the 
        number of elements in the inputted img_list    
    """
    #find size all images will be
    max_rows = 0
    max_cols = 0
    for img in img_list:
        max_rows = max(img.shape[0], max_rows)
        max_cols = max(img.shape[1], max_cols)
    if min_size is not None:
        max_rows = max(max_rows,min_size)
        max_cols = max(max_cols,min_size)

    #resize and stack the images
    for il,img in enumerate(img_list):
        resized_img = np.zeros((max_rows,max_cols,img.shape[2]))
        resized_img[0:img.shape[0],0:img.shape[1],:] = img
        img_list[il] = resized_img
    return np.stack(img_list,axis=0) 




def create_illumination_pattern(rows, cols, center_row,center_col,minI=.1,maxI=1,radius=None):
    '''
    Creates a random illumination pattern mask

    Input parameters:
        rows: (int) number of rows in returned pattern
        cols: (int) number of cols in returned pattern
        center_row: (int) row of center of illumination
        center_col: (int) col of center of illumination

        minI (optional): min illumination change. Default: .1
        maxI (optional): (float) max illum change. Default: 1
        radius (optional): (int) radius of illumination thing. If None
                           a random radius is chosen. Default: None

    Returns:
        (ndarray) array to be pixel-wise multiplied with an image to change
        the images illumination
    
    '''
    if radius is None:
        radius = float(int(20000 + (30000)*np.random.rand(1)))
    pattern = np.zeros((rows, cols));
    for row in range(rows):
        for col in range(cols):
            dy = row - center_row;
            dx = col - center_col;
            pattern[row,col] = (minI + (maxI -minI) 
                                * math.exp(-(.5)*(dx*dx + dy*dy)/ radius))
    return pattern


def augment_image(img, crop_max=5, rotate_max=30, do_illum=.5):
    '''
    Alters an image with some common data augmentation techniques
       
    Imput parameters:
        img: (ndarray) the image

        crop_max (optional): (int) max length that can be "cropped" from 
                             each side. Cropping does not change image shape,
                             but sets "cropped" region to 0. Default: 5 
        rotate_max (optional): (int) max degrees for in-plane rotation
                               Default: 30
        do_illum (optional): (float) chance that a random illumination
                             change will be applied. Set to 0 if no 
                             illumination change is desired. Default: .5 

    Returns:
        (ndarray) the augmented image
    '''
    #crop
    crops = np.random.choice(crop_max,4)   
    start_row = 0 + crops[0]
    end_row = img.shape[0] - crops[1] 
    start_col = 0 + crops[2]
    end_col = img.shape[1] - crops[3]
    img[0:start_row,:,:] = 0
    img[:,0:start_col,:] = 0
    img[end_row:,:,:] = 0
    img[:,end_col:,:] = 0

    #rotate
    rot_angle = np.random.choice(rotate_max*2,1) - rotate_max
    M = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2),rot_angle,1)
    img = cv2.warpAffine(img,M,(img.shape[1],img.shape[0]))


    #change illumination
    if np.random.rand() < do_illum:
        max_side = max(img.shape[:2])
        xc,yc = np.random.choice(max_side,2)
        pattern = create_illumination_pattern(max_side,max_side,xc,yc)
        pattern = pattern[0:img.shape[0],0:img.shape[1]]
        img = img* np.tile(np.expand_dims(pattern,2),(1,1,3)) 

    return img



def check_object_ids(chosen_ids,id_to_name,target_images):
    """
    Picks only chosen ids that have a target object and target image.

    ex) check_object_ids(chosen_ids,id_to_name,target_images)


    Input Parameters:
        chosen_ids: (list) list of ints, each int is a class id
        id_to_name: (dict) key=class_id(int), value=target_name(str)
        target_images: (dict) same as returned from get_target_images function

    Returns:
        (list) ids in chosen ids that exist in id_to_name dict, and 
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




def normalize_image(img,cfg):
    """
    Noramlizes image according to config parameters
    
    ex) normalize_image(image,config)

    Input Parameters:
        img: (ndarray) numpy array, the image to be normalized
        cfg: (Config) config instance from configs/

    Returns: 
        (ndarray) noralized image
    """
    if cfg.PYTORCH_FEATURE_NET:
        return ((img/255.0) - [0.485, 0.456, 0.406])/[0.229, 0.224, 0.225]
    else:
        raise NotImplementedError





def get_class_id_to_name_dict(root,file_name='instance_id_map.txt'):
    """
    Get dict from integer class id to string name

    Input Parameters:
        root: (str) directory that holds .txt file with class names and ids
    
        file_name (optional): (str) name of file with class names and ids
                              Default: 'instance_id_map.txt'

                              Format: each line has: target_name id
                              where id is an integer character

    Returns:
        (dict) dict with key=id, value=target_name
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
                       classification=False,
                      ):
    """
    Returns a loader for the AVD dataset.

    dataset = get_AVD_dataset('/path/to/data',['scene1','scene2], [chosen_ids])


    Input Parameters:
        root: (str) path to data. Parent of all scene directories
        scene_list: (list) scenes to include
        chosen_ids: (list) list of object ids to keep labels for
                    (other labels discarded) 

        max_difficulty (optional): (int) max bbox difficulty to use Default: 4
        instance_fname (optional): (str) name of file with class ids and names
                                   If none, uses default in get_class_id_to_name
                                   Default: None
        classification (opitional): (bool) Whether or not data is for
                                    classification. Default: False

    Returns:
        an instance of AVD class from the AVD data_loading code 

    """
    ##initialize transforms for the labels
    #only consider boxes from the chosen classes
    pick_trans = AVD_transforms.PickInstances(chosen_ids,
                                              max_difficulty=max_difficulty)
    #compose the transforms in a specific order, first to last
    if instance_fname is None:
        id_to_name_dict = get_class_id_to_name_dict(root)
    else:
        id_to_name_dict = get_class_id_to_name_dict(root,instance_fname)

    dataset = AVD.AVD(root=root,
                         scene_list=scene_list,
                         target_transform=pick_trans,
                         classification=classification,
                         class_id_to_name=id_to_name_dict,
                         fraction_of_no_box=fraction_of_no_box)
    return dataset





def save_training_meta_data(cfg,net):
    """
    Writes a text file that describes model and paramters.
    
    ex) save_training_meta_data(cfg,net)

    Input parameters:
        cfg: (Config) a config isntance from configs/ 
        net: (torch Module) a pytorch network   
  
    Returns:
        None 
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
    '''
    Load weights of a pretrained pytorch model for feature extraction

    Example: For Alexnet, a torch.nn.Sequential model with everything
             but the fully connected layers is returned

    Input parameters:
        model_name: name of the model to load. Options:
            vgg16_bn
            squeezenet1_1
            resnet101
            alexnet

    Returns:
        (torch.nn.Sequential) The first N layers of the pretrained model
        that are useful for feature extraction. N depends on which model 
    '''
    if model_name == 'vgg16_bn':
        vgg16_bn = models.vgg16_bn(pretrained=True)
        return torch.nn.Sequential(*list(vgg16_bn.features.children())[:-1])
    elif model_name == 'squeezenet1_1':
        fnet = models.squeezenet1_1(pretrained=True)
        return torch.nn.Sequential(*list(fnet.features.children())[:-1])
    elif model_name == 'resnet101':
        fnet = models.resnet101(pretrained=True)
        return torch.nn.Sequential(*list(fnet.children())[:-2])
    elif model_name == 'alexnet':
        fnet = models.alexnet(pretrained=True)
        return torch.nn.Sequential(*list(fnet.features.children()))
    else:
        raise NotImplementedError
        sys.exit()





# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import time

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0 
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff





class Conv2d(nn.Module):
    '''
        A wrapper for a 2D pytorch conv layer. 
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False, bn=False):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class FC(nn.Module):
    '''
        A wrapper for a pytorch fully connected layer. 
    '''
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


def save_net(fname, net):
    '''
    Saves a network using h5py
    
    Input parameters:
        fname: (str) full path of file to save model
        net: (torch.nn.Module) network to save
    '''
    h5f = h5py.File(fname, mode='w')
    for k, v in net.state_dict().items():
        h5f.create_dataset(k, data=v.cpu().numpy())


def load_net(fname, net):
    '''
    Loads a network using h5py
    
    Input parameters:
        fname: (str) full path of file to load model from
        net: (torch.nn.Module) network to load weights to
    '''
    h5f = h5py.File(fname, mode='r')
    for k, v in net.state_dict().items():
        param = torch.from_numpy(np.asarray(h5f[k]))
        v.copy_(param)


def np_to_variable(np_var, is_cuda=True, dtype=torch.FloatTensor):
    '''
    Converts numpy array to pytorch Variable

    Input parameters:
        np_var: (ndarray) numpy variable

        is_cuda (optional): (bool) If True, torch variable's .cuda() is
                           applied. If false nothing happens. Default: True
        dtype (optional):  (type) desired type of returned torch variable.
                            Default: torch.FloatTensor

    Returns:
        (torch.autograd.Variable) a torch variable version of the np_var
    '''
    pytorch_var = Variable(torch.from_numpy(np_var).type(dtype))
    if is_cuda:
        pytorch_var = pytorch_var.cuda()
    return pytorch_var 


def weights_normal_init(model, dev=0.01):
    '''
    Initialize weights of model randomly according to a normal distribution

    Input parameters:
        model: (torch.nn.Module) pytorch model
        
        dev (optional): (float) standard deviation of the normal distribution
                        Default: .01
    '''
    if isinstance(model, list):
        for m in model:
            weights_normal_init(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, dev)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)


def clip_gradient(model, clip_norm):
    '''
    Computes a gradient clipping coefficient based on gradient norm.
    ''' 
    totalnorm = 0 
    for p in model.parameters():
#    for name,p in model.named_parameters():
        if p.requires_grad:
            #print name
            try:
                modulenorm = p.grad.data.norm()
                totalnorm += modulenorm ** 2
            except:
                continue
    totalnorm = np.sqrt(totalnorm)
    norm = clip_norm / max(totalnorm, clip_norm)
    for p in model.parameters():
        if p.requires_grad:
            try:
                p.grad.mul_(norm)
            except:
                continue


