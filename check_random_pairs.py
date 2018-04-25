import os
import cv2
import numpy as np

avd_root_dir = '/playpen/ammirato/Data/RohitData/'
pair_fid = open('/playpen/ammirato/AVD_extra.txt', 'r')

def get_image_from_name(img_name, avd_root):
    if img_name[0] == '0':
        scene_type = 'Home'
    elif img_name[0] == '1':
        scene_type = 'Office'
    else:
        return []
    scene_name = scene_type + '_' + img_name[1:4] + '_' + img_name[4]
    return cv2.imread(os.path.join(avd_root,scene_name,'jpg_rgb',img_name))

def convert_string_to_numpy(str_array):
    split = str_array.split()
    return np.asarray([int(split[0][1:-1]), int(split[1][:-1]), int(split[2][:-1]), int(split[3][:-1])])
    



for line in pair_fid:
    line = line.split('#')
    img1 = get_image_from_name(line[0],avd_root_dir) 
    img2 = get_image_from_name(line[2],avd_root_dir) 

    box1 = convert_string_to_numpy(line[1])
    box2 = convert_string_to_numpy(line[3])

pair_fid.close()

