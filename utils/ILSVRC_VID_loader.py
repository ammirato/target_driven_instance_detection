import os
import numpy as np
import sys
import json
import cv2
import torch
import collections
import glob
import xml.etree.ElementTree as ET




class VID_Loader(object):
    """
    ***ASSUMES each video only has one object annotated, and that
        object is in every frame*******
    """
    
    ann_path_insert = 'Annotations/VID/'
    data_path_insert = 'Data/VID/'

    def __init__(self, root, data_subset, transform=None, target_transform=None, 
                 class_id_to_name=None, batch_random_sampling=True, batch_size=2,
                 target_size=[80,16]):
        """
        Create instance of class

        Ex) traindata = AVD('/path/to/data/')

        INPUTS:
          root: root directory of ILSVRC root dir i.e. '/some/path/ILSVRC/' 
          data_subset: 'val' or 'val2'  

        KEYWORD INPUTS(default value):
          transform(None): function to apply to images before 
                           returning them(i.e. normalization)
          target_transform(None): function to apply to labels 
                                  before returning them
          class_id_to_name(None): dict with keys=class ids, values = names
                                  Assumes original class ids, any changes
                                  to ids via a target transform will 
                                  be applied by this object.  
          batch_random_sampling(True): __getitem__ always returns a batch 
                                       with randomly sampled items 
          target_size([int,int]=[80,16]): max,min size of target image, 
                                          set to None to skip resizing  
        """

        self.root = root
        self.data_subset = data_subset
        self.transform = transform
        self.target_transform = target_transform
        self.class_id_to_name = class_id_to_name 
        self.ann_path = os.path.join(self.root, self.ann_path_insert, data_subset)
        self.data_path = os.path.join(self.root, self.data_path_insert, data_subset)
        self.batch_random_sampling = batch_random_sampling 
        self.batch_size = 2
        self.target_size = target_size

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.')


        #get list of all possible videos
        self.video_data_paths = glob.glob(os.path.join(self.root,
                                                       self.data_path_insert,
                                                       self.data_subset,'*'))
        self.video_ann_paths = glob.glob(os.path.join(self.root,
                                                       self.ann_path_insert,
                                                       self.data_subset,'*'))



    def __getitem__(self, index):
        """ 
        Gets desired image and label   
        """
        if not self.batch_random_sampling:
            print 'Only batch random sampling currently supported!'
            return -1

        
        #pick two random videos
        inds = np.random.choice(len(self.video_data_paths),2)
        v1_path = self.video_data_paths[inds[0]]
        v2_path = self.video_data_paths[inds[1]]
        #pick three random frames from the first video
        #get the annotations from these 3 frames
        image_paths = glob.glob(os.path.join(v1_path,'*.JPEG'))
        inds = np.random.choice(len(image_paths),3)
        v1_img = cv2.imread(image_paths[inds[0]])
        v1_bbox = self._get_bbox_from_data_path(image_paths[0])
        t_imgs = []
        for ind in inds[1:]:
            full_img = cv2.imread(image_paths[ind])
            bbox = self._get_bbox_from_data_path(image_paths[ind])
            #crop the second and third frame around the object of interest
            t_img = full_img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
            if self.target_size is not None:
                large_side = np.max(t_img.shape)
                scale_factor = float(self.target_size[0])/large_side
                t_img = cv2.resize(t_img,(int(t_img.shape[1]*scale_factor),
                                              int(t_img.shape[0]*scale_factor)))

                if np.min(t_img.shape[:2]) < self.target_size[1]:
                    if t_img.shape[0] < self.target_size[1]:
                        blank_img = np.zeros((self.target_size[1], t_img.shape[1],t_img.shape[2]))
                    else:
                        blank_img = np.zeros((t_img.shape[0],self.target_size[1],t_img.shape[2]))
                    blank_img[0:t_img.shape[0],0:t_img.shape[1],:] = t_img
                    t_img = blank_img

            t_imgs.append(t_img)

        #pick one random frame from the second video 
        v2_image_paths = glob.glob(os.path.join(v2_path,'*.JPEG'))
        inds = np.random.choice(len(v2_image_paths),1)
        v2_img = cv2.imread(v2_image_paths[inds[0]])
       

        return [v1_img, v1_bbox, t_imgs, v2_img]



    def _get_bbox_from_data_path(self, data_path):

        ann_path = data_path.replace('Data','Annotations').replace('JPEG','xml')
        root = ET.parse(ann_path) 
        obj = root.findall('object')[0]
        box = obj.findall('bndbox')
        xmin = int(box[0].findall('xmin')[0].text) 
        xmax = int(box[0].findall('xmax')[0].text)
        ymin = int(box[0].findall('ymin')[0].text)
        ymax = int(box[0].findall('ymax')[0].text)
        return [xmin, ymin, xmax, ymax]


    def __len__(self):
        """ 
        """
        return 0   

    def _check_integrity(self):
        """ 
        """
        root = self.root
        if (os.path.isdir(self.ann_path) and os.path.isdir(self.data_path)):
            return True
        else:
            return False 
            

    def get_count_by_class(self):
        """
        Returns a count of how many labels there are per class

        Assumes class id is still 5th element of each target
        even after target transform
        """
        return 0 

    def get_num_classes(self):
        return 0 



    def get_class_names(self):
        return self.class_id_to_name.values()
        

    def transform_id_to_name_dict(self):
        """
        Changes id->name to reflect changes made to ids from target transform
        """
        
        #get a list of ids after transform
        self.get_count_by_class()
        ids_after = self.count_by_class.keys()

        #for each original id, make a dummy box, transform it,
        #and make a new key,value in a new dict with new_id,name
        new_dict = {}
        ids_before = self.class_id_to_name.keys()
        for old_id in ids_before:
            dummy_box = [0,0,0,0,old_id,0]
            transformed_box = self.target_transform([dummy_box])
            if len(transformed_box) > 1:
                transformed_box = [transformed_box[-1]]
            #check to see if box is None of empty
            if transformed_box is None or not transformed_box:   
                continue
            if sum(transformed_box[0][0:4]) > 0:
                continue
            new_id = transformed_box[0][4]
            new_dict[new_id] = self.class_id_to_name[old_id]
     

        self.class_id_to_name = new_dict

        return None



    def get_box_difficulty(self,box):
        """
        Returns box difficulty measure, as defined on dataset website
        """
        box_dims = np.array([box[2]-box[0], box[3]-box[1]])
        maxd = box_dims.max()
        mind = box_dims.min()

        if maxd>=300 and mind>=100:
            return 1
        elif maxd>=200 and mind>=75:
            return 2
        elif maxd>=100 and mind>=50:
            return 3 
        elif maxd>=50 and mind>=30:
            return 4 
        else:
            return 5



    #http://pytorch.org/docs/_modules/torch/utils/data/dataloader.html#DataLoader
    def collate(batch):
        
        return batch 

