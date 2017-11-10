import os
import cv2
import numpy as np
import math



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
                      means=None, bn_normalize=False):



    #path that holds dirs of all targets
    #i.e. target_path/target_0/* has one type of target image for each object
    #     target_path/target_1/* has another type of target image
    #type of target image can mean different things, 
    #probably different type is different view
    #each type can have multiple images, 
    #i.e. target_0/* can have multiple images per object
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
                if bn_normalize:
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














