import os
import cv2
import numpy as np
import random

avd_root_dir = '/playpen/ammirato/Data/RohitData/'
pair_fid = open('/playpen/ammirato/AVD_extra_all.txt', 'r')

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
    
all_lines = []
for line in pair_fid:
    all_lines.append(line)
random.shuffle(all_lines)

for line in all_lines:
    line = line.split('#')
    img1 = get_image_from_name(line[0],avd_root_dir) 
    img2 = get_image_from_name(line[2],avd_root_dir) 
    img1 = cv2.resize(img1,(0,0), fx=.5,fy=.5)
    img2 = cv2.resize(img2,(0,0), fx=.5,fy=.5)

    box1 = convert_string_to_numpy(line[1])
    box2 = convert_string_to_numpy(line[3])

    cv2.rectangle(img1,(box1[0]/2,box1[1]/2),(box1[2]/2,box1[3]/2),(255,0,0),4)
    cv2.rectangle(img2,(box2[0]/2,box2[1]/2),(box2[2]/2,box2[3]/2),(255,0,0),4)

    cv2.imshow("org_image", img1)
    cv2.imshow("new_image", img2)
    k =  cv2.waitKey(0)
    if k == ord('q'):
        break

pair_fid.close()

