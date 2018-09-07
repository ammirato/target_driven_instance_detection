import os
import numpy as np
import sys
import json
import cv2
import random

from utils import *

class AVD_Extra_Loader(object):
    """
    """
    
    def __init__(self, avd_root, pair_file):
        """
        INPUTS:
          root: root directory of ILSVRC root dir i.e. '/some/path/ILSVRC/' 
          for_training: bool  

        KEYWORD INPUTS(default value):
          target_size([int,int]=[100,100]): max,min size of target image, 
                                          set to None to skip resizing  
        """

        self.avd_root = avd_root
       
        pair_fid = open(pair_file,'r') 
        all_pairs = []
        for line in pair_fid:
            all_pairs.append(line)
        #self.pairs_list = random.shuffle(all_pairs)
        self.pairs_list = all_pairs


    def get_batch(self, batch_size, classification=False, dims=[16,200]):
        """ 
        Gets desired image and label

        """
        batch_scene_imgs = []
        batch_target_imgs = []
        batch_gt_boxes = []

        for batch_ind in range(batch_size):
            ind = -1
            while(ind <0):
                ind =  np.random.choice(range(1,len(self.pairs_list)-1))
                info = self.get_pair_info(self.pairs_list[ind])
                info1 = self.get_pair_info(self.pairs_list[ind+1])
                info2 = self.get_pair_info(self.pairs_list[np.random.choice(len(self.pairs_list))])
                if info1[0] == info[0]:
                    break
                info1 = self.get_pair_info(self.pairs_list[ind-1])
                if info1[0] == info[0]:
                    ind = ind -1
                    break
                ind = -1 

            img1 = self.get_image_from_name(info[0],self.avd_root) 
            img2 = self.get_image_from_name(info[2],self.avd_root) 
            img3 = self.get_image_from_name(info1[2],self.avd_root) 
            img4 = self.get_image_from_name(info2[0],self.avd_root)

            box1 = info[1]/2
            box2 = info[3]/2
            box3 = info1[3]/2
            box4 = info2[1]/2
            if info2[0] == info[0] and box4[0]==box1[0] and box4[1] == box1[1]:
                box4 = [box4[0],box4[1],box4[2],box4[3],1]
            else:
                box4 = [box4[0],box4[1],box4[2],box4[3],0]

            box3 = [box3[0],box3[1],box3[2],box3[3],1]
            target1 = img1[box1[1]:box1[3], box1[0]:box1[2],:]
            target2 = img2[box2[1]:box2[3], box2[0]:box2[2],:]
            if classification:
                img3 = img3[box3[1]:box3[3], box3[0]:box3[2],:]
                img4 = img4[box4[1]:box4[3], box4[0]:box4[2],:]
                img3 = resize_image(img3,dims[0],dims[1])           
                img4 = resize_image(img4,dims[0],dims[1])           
 
            #consolidate images for the batch
            batch_scene_imgs.append(img3)
            batch_gt_boxes.append(box3) 
            batch_target_imgs.append(target1) 
            batch_target_imgs.append(target2) 
            ##################################
            batch_scene_imgs.append(img4)
            batch_gt_boxes.append(box4) 
            batch_target_imgs.append(target1) 
            batch_target_imgs.append(target2) 

        return [batch_scene_imgs,batch_gt_boxes,batch_target_imgs]

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

    def get_pair_info(self,line):
        line = line.split('#')
        return line[0],self.convert_string_to_numpy(line[1]),line[2],self.convert_string_to_numpy(line[3])


#     img1 = get_image_from_name(line[0],avd_root_dir)
#     img2 = get_image_from_name(line[2],avd_root_dir)
#     img1 = cv2.resize(img1,(0,0), fx=.5,fy=.5)
#     img2 = cv2.resize(img2,(0,0), fx=.5,fy=.5)
# 
#     box1 = convert_string_to_numpy(line[1])
#     box2 = convert_string_to_numpy(line[3])

#
    def get_image_from_name(self,img_name, avd_root):
        if img_name[0] == '0':
            scene_type = 'Home'
        elif img_name[0] == '1':
            scene_type = 'Office'
        else:
            return []
        scene_name = scene_type + '_' + img_name[1:4] + '_' + img_name[4]
        return cv2.imread(os.path.join(avd_root,scene_name,'jpg_rgb',img_name))
# 
    def convert_string_to_numpy(self,str_array):
     split = str_array.split()
     return np.asarray([int(split[0][1:-1]), int(split[1][:-1]), int(split[2][:-1]), int(split[3][:-1])])
# 
# all_lines = []
# for line in pair_fid:
#     all_lines.append(line)
# random.shuffle(all_lines)
# 
# for line in all_lines:
#     line = line.split('#')
#     img1 = get_image_from_name(line[0],avd_root_dir)
#     img2 = get_image_from_name(line[2],avd_root_dir)
#     img1 = cv2.resize(img1,(0,0), fx=.5,fy=.5)
#     img2 = cv2.resize(img2,(0,0), fx=.5,fy=.5)
# 
#     box1 = convert_string_to_numpy(line[1])
#     box2 = convert_string_to_numpy(line[3])
# 
#     cv2.rectangle(img1,(box1[0]/2,box1[1]/2),(box1[2]/2,box1[3]/2),(255,0,0),4)
#     cv2.rectangle(img2,(box2[0]/2,box2[1]/2),(box2[2]/2,box2[3]/2),(255,0,0),4)
# 
#     cv2.imshow("org_image", img1)
#     cv2.imshow("new_image", img2)
#     k =  cv2.waitKey(0)
#     if k == ord('q'):
#         break
# 


