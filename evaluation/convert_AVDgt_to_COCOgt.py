import os
import json

'''
Convert AVD annotations into MSCOCO format for evaluation.

Combines annotations from multiple scenes into a single file for use 
with the MSCOCO evaluation code. 
'''

#AVD_root_path = '/net/bvisionserver3/playpen10/ammirato/Data/RohitData/'
AVD_root_path = '/playpen/ammirato/Data/RohitData/'
save_path = '../Data/GT/'
save_name = 'home0031.json'
scene_list = [
             #'Home_001_1',
             #'Home_001_2',
             #'Home_002_1',
             'Home_003_1',
             #'Home_003_2',
             #'Home_004_1',
             #'Home_004_2',
             #'Home_005_1',
             #'Home_005_2',
             #'Home_006_1',
             #'Home_008_1',
             #'Home_014_1',
             #'Home_014_2',
             #'Office_001_1',

             #    'Home_101_1',
             #    'Home_102_1',
             #    'Home_103_1',
             #    'Home_104_1',
             #    'Home_105_1',
             #    'Home_106_1',
             #    'Home_107_1',
             #    'Home_108_1',
             #    'Home_109_1',


             ]


if not(os.path.isdir(save_path)):
    os.makedirs(save_path)

#first make categories dict
map_file = open(os.path.join(AVD_root_path,'instance_id_map.txt'),'r')
categories = []
for line in map_file:
    line = str.split(line)
    cid = int(line[1])
    name = line[0]
    categories.append({'id':cid, 'name':name})

img_anns = []
box_anns = []
box_id_counter = 0

cids = []
for scene in scene_list:
    scene_path = os.path.join(AVD_root_path,scene)
    annotations = json.load(open(os.path.join(scene_path,'annotations.json')))

    for img_name in annotations.keys():

        img_ind = int(img_name[:-4])

        pre_boxid_counter = box_id_counter
        boxes = annotations[img_name]['bounding_boxes']
        for box in boxes:
             
            xmin = box[0]
            ymin = box[1]
            width = box[2]-box[0] +1
            height = box[3]-box[1] +1
            iscrowd = 0
            if max(width, height) <= 25 or min(width,height) <= 15:
               iscrowd=1
            if box[5] >= 5:
                iscrowd=1

            area = width*height
            cid = box[4]

            cids.append(cid)            
            box_anns.append({'area':area,'bbox':[xmin,ymin,width,height],
                             'category_id':cid,'image_id':img_ind,
                             'iscrowd':iscrowd,'segmentation':[], 
                             'id':box_id_counter})
            box_id_counter += 1
        img_anns.append({'file_name':img_name, 'id':img_ind, 'height':540, 'width':960})
coco_anns = {'images':img_anns, 'annotations':box_anns,'categories':categories}

with open(os.path.join(save_path,save_name), 'w') as outfile:
    json.dump(coco_anns, outfile)



