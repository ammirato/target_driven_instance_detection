import json


gt = json.load(open('./Data/TestOutputs/TDID_GEN4GUW_03TO_50000.json'))


anns = gt

new_anns = []
for ann in anns:
    box = ann['bbox']
    box[0] = box[0]/2
    box[1] = box[1]/2
    box[2] = box[2]/2
    box[3] = box[3]/2
    ann['bbox'] = box
    new_anns.append(ann)




json.dump(new_anns,open('./Data/TestOutputs/TDID_GEN4GUW_02TO_50000_2.json','w'))


