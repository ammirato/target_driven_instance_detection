import os
import glob
import shutil
import cv2
import numpy as np
import xml.etree.ElementTree as ET

def resize_box_from_path(ann_path, scale_factor):

    root = ET.parse(ann_path) 
    obj = root.findall('object')[0]
    box = obj.findall('bndbox')
    xmin = int(int(box[0].findall('xmin')[0].text)* scale_factor) 
    xmax = int(int(box[0].findall('xmax')[0].text)* scale_factor)
    ymin = int(int(box[0].findall('ymin')[0].text)* scale_factor)
    ymax = int(int(box[0].findall('ymax')[0].text)* scale_factor)

    box[0][0].text = str(xmax)
    box[0][1].text = str(xmin)
    box[0][2].text = str(ymax)
    box[0][3].text = str(ymin)

    root.write(ann_path)



if __name__ == '__main__':

    max_size = 600

    base_path =  '/net/bvisionserver3/playpen10/ammirato/Data/ILSVRC/'

    ann_path_insert = 'Annotations/VID'
    data_path_insert  = 'Data/VID'
    set_name = 'train_single'


    all_vids = glob.glob(os.path.join(base_path,data_path_insert,set_name, '*'))

    for il, vid_data_path in enumerate(all_vids):

        print '{} / {}'.format(il, len(all_vids))
        img_paths = glob.glob(os.path.join(vid_data_path,'*'))
        
        for img_path in img_paths:
            img = cv2.imread(img_path)

            large_side = np.max(img.shape)
            if large_side > max_size:

                scale_factor = float(max_size)/large_side
                img = cv2.resize(img,(int(img.shape[1]*scale_factor),
                                          int(img.shape[0]*scale_factor)))



                #resize bouning box
                ann_path = img_path.replace(data_path_insert,ann_path_insert).replace('JPEG','xml')
                
                resize_box_from_path(ann_path,scale_factor)
                cv2.imwrite(img_path, img)

