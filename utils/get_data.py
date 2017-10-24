import os
import cv2
import numpy as np









def get_target_images(target_path, target_names,preload_images=False, for_testing=False,
                      means=None):



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
                all_types.append(np.expand_dims(img-means, axis=0))
            target_images[target_name] = all_types

    return target_images
