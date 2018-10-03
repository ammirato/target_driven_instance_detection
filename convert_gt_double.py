import json


gt = json.load(open('./Data/GT/all_gmu.json'))


anns = gt['annotations']

new_anns = []
for ann in anns:
    box = ann['bbox']
    box[0] = box[0]*2
    box[1] = box[1]*2
    box[2] = box[2]*2
    box[3] = box[3]*2
    ann['bbox'] = box
    new_anns.append(ann)


gt['annotations'] = new_anns


imgs = gt['images']
new_imgs = []
for img in imgs:
    img['height'] = img['height']*2   
    img['width'] = img['width']*2   
    new_imgs.append(img)


gt['images'] =  new_imgs


json.dump(gt,open('./Data/GT/all_gmu2.json','w'))


