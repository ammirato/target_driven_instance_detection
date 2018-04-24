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

    def __init__(self, root, for_training, 
                 target_size=[100,100]):
        """
        INPUTS:
          root: root directory of ILSVRC root dir i.e. '/some/path/ILSVRC/' 
          for_training: bool  

        KEYWORD INPUTS(default value):
          target_size([int,int]=[100,100]): max,min size of target image, 
                                          set to None to skip resizing  
        """

        self.root = root
        self.for_training = for_training
        if for_training:
            data_subset = 'train'
        else:
            data_subset = 'val'
        self.data_subset = data_subset
        self.ann_path = os.path.join(self.root, self.ann_path_insert, data_subset)
        self.data_path = os.path.join(self.root, self.data_path_insert, data_subset)
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



    def get_batch(self, batch_size):
        """ 
        Gets desired image and label

        """
        batch_scene_imgs = []
        batch_target_imgs = []
        batch_gt_boxes = []

        for batch_ind in range(batch_size):

            #pick two random videos
            vid_paths = np.random.choice(len(self.video_data_paths),2)

            #from the first video, get 2 target images, and one scene image
            v1_path = self.video_data_paths[vid_paths[0]]
            v1_img_paths = glob.glob(os.path.join(v1_path,'*.JPEG'))
            v1_img_paths.sort()
            #get first target image from first frame of the video
            v1_first_img = cv2.imread(v1_img_paths[0])
            bbox = self._get_bbox_from_data_path(v1_img_paths[0])
            v1_first_target_img = v1_first_img[bbox[1]:bbox[3], bbox[0]:bbox[2],:]
            #get the second target image from another random frame
            #(possibly the first frame, but not the last)
            second_frame = np.random.choice(len(v1_img_paths)-1)
            v1_second_img = cv2.imread(v1_img_paths[second_frame])
            bbox = self._get_bbox_from_data_path(v1_img_paths[second_frame])
            v1_second_target_img = v1_second_img[bbox[1]:bbox[3], bbox[0]:bbox[2],:]
            #get the scene image from within 3 frames of the second target image
            third_frame = np.random.choice(range(second_frame+1,
                                                 min(len(v1_img_paths),
                                                     second_frame+4)))
            v1_scene_img = cv2.imread(v1_img_paths[third_frame])
            #get the gt bounding box in this scene image
            v1_gt_bbox = self._get_bbox_from_data_path(v1_img_paths[third_frame])

            #from the second video, get 1 scene image
            v2_path = self.video_data_paths[vid_paths[1]]
            v2_img_paths = glob.glob(os.path.join(v2_path,'*.JPEG'))
            v2_scene_img = cv2.imread(v2_img_paths[np.random.choice(len(v2_img_paths))])
            #put dummy background bounding box for second scene image
            v2_gt_bbox = [0,0,0,0,0]

            #consolidate images for the batch
            batch_scene_imgs.append(v1_scene_img)
            batch_scene_imgs.append(v2_scene_img)
            batch_gt_boxes.append(v1_gt_bbox) 
            batch_gt_boxes.append(v2_gt_bbox) 
            batch_target_imgs.append(v1_first_target_img) 
            batch_target_imgs.append(v1_second_target_img) 
            batch_target_imgs.append(v1_first_target_img) 
            batch_target_imgs.append(v1_second_target_img) 

        return [batch_scene_imgs,batch_gt_boxes,batch_target_imgs]





    def _get_bbox_from_data_path(self, data_path):

        ann_path = data_path.replace(self.data_path_insert,self.ann_path_insert).replace('JPEG','xml')
        root = ET.parse(ann_path) 
        obj = root.findall('object')[0]
        box = obj.findall('bndbox')
        xmin = int(box[0].findall('xmin')[0].text) 
        xmax = int(box[0].findall('xmax')[0].text)
        ymin = int(box[0].findall('ymin')[0].text)
        ymax = int(box[0].findall('ymax')[0].text)
        return [xmin, ymin, xmax, ymax,1]

    def _check_integrity(self):
        """ 
        """
        root = self.root
        if (os.path.isdir(self.ann_path) and os.path.isdir(self.data_path)):
            return True
        else:
            return False 
            

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



