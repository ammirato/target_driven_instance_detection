import os
import cv2
import numpy as np
import glob

from instance_detection.utils.get_data import vary_image





target_path = '/net/bvisionserver3/playpen10/ammirato/Data/instance_detection_targets/AVD_BB_exact_few_and_other_BB_gen_160/'
dest_path = '/net/bvisionserver3/playpen10/ammirato/Data/instance_detection_targets/AVD_BB_exact_few_and_other_BB_gen_160_varied/'


target_types = ['target_0','target_1']


for t_type in target_types:

    b_path = os.path.join(target_path,t_type)
    dp_path = os.path.join(dest_path,t_type)


    img_names = os.listdir(b_path)

    for img_name in img_names:
        img = cv2.imread(os.path.join(b_path,img_name))

        sep_ind = img_name.rfind('_')
        ext_ind = img_name.rfind('.')
        obj_name = img_name[:sep_ind]
        obj_index = int(img_name[sep_ind+1:ext_ind])


        cv2.imwrite(os.path.join(dp_path,img_name),img)


        for il in range(1,101):
            new_img = vary_image(img)

            new_index = str(obj_index+il+100)
            new_name = obj_name + '_' + new_index + '.jpg'
            
            cv2.imwrite(os.path.join(dp_path,new_name),new_img)



