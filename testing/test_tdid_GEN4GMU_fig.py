import os
import torch
import torchvision.models as models
import cv2
import cPickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc

from instance_detection.model_defs import network
#from instance_detection.model_defs.tdid import TDID 
#from instance_detection.model_defs.tdid_depthwise_batch import TDID 
#from instance_detection.model_defs.tdid_depthwise_plus_batch import TDID 
#from instance_detection.model_defs.tdid_depthwise_mtargets_batch import TDID
#from instance_detection.model_defs.tdid_depthwise_mtargets_img_batch import TDID
#from instance_detection.model_defs.tdid_depthwise_mtargets_diff_batch import TDID
#from instance_detection.model_defs.tdid_depthwise_mtargets_diff_batch_ms import TDID
from instance_detection.model_defs.TDID_final import TDID
#from instance_detection.model_defs.tdid_depthwise_mtargets_sim_batch import TDID 
#from instance_detection.model_defs.tdid_depthwise_mtargets_simSep_batch import TDID 
#from instance_detection.model_defs.tdid_depthwise_sim_batch import TDID 
from instance_detection.model_defs.utils.timer import Timer
from instance_detection.model_defs.fast_rcnn.nms_wrapper import nms

from instance_detection.utils.get_data import get_target_images,match_and_concat_images


from instance_detection.model_defs.fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from instance_detection.model_defs.fast_rcnn.config import cfg, cfg_from_file, get_output_dir

import active_vision_dataset_processing.data_loading.active_vision_dataset_pytorch as AVD  
import active_vision_dataset_processing.data_loading.transforms as AVD_transforms
import exploring_pytorch.basic_examples.GetDataSet as GetDataSet

#import matplotlib.pyplot as plt
import json

# hyper-parameters
# ------------
cfg_file = '../utils/config.yml'
trained_model_path = ('/net/bvisionserver3/playpen/ammirato/Data/Detections/' + 
                     'saved_models/')
trained_model_names=[


#                    'TDID_final_GEN4GMU_0_0_6000_131.28783_0.41657_-1.00000',
#                    'TDID_final_GEN4GMU_0_0_4500_138.53446_0.40704_-1.00000',
#            'TDID_final_GEN4GMU_0_1_3000_110.62451_0.59036_-1.00000',
        'TDID_final_GEN4GMU_0_1_4500_108.49101_0.62820_-1.00000',

                    ########################################################
                    #####           AVD ABLATION STUDY             #########
                    ########################################################
#                    'TDID_COMB_AVD2_archDmtbn_ROI_2_10_1461_50.26356_0.36411_0.40334',
#                    'TDID_COMB_AVD2_archDmtIMGbn_ROI_0_12_1465_44.27885_0.37181_0.46335',
#                    'TDID_COMB_AVD2_archDmtDIFFbn_ROI_0_15_1460_36.55347_0.60418_0.64426',

                    ]
use_batch_norm =True
use_torch_vgg=True
rand_seed =None 
max_per_target = 15 
thresh = 0
vis = False 
means = np.array([[[102.9801, 115.9465, 122.7717]]])
if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)




def im_detect(net, target_data,im_data, im_info, features_given=True):
    """Detect object classes in an image given object proposals.
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """


    cls_prob, bbox_pred, rois, layers = net(target_data, im_data, 
                                    features_given=features_given, im_info=im_info)
    scores = cls_prob.data.cpu().numpy()[0,:,:]
    zs = np.zeros((scores.size, 1))
    scores = np.concatenate((zs,scores),1)
    #boxes = rois.data.cpu().numpy()[:, 1:5] / im_info[0][2]
    boxes = rois.data.cpu().numpy()[0,:, :] #/ im_info[0][2]

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred[0].data.cpu().numpy()
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        #pred_boxes = clip_boxes(pred_boxes, im_data.shape[1:])
        pred_boxes = clip_boxes(pred_boxes, im_info)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes , layers


def test_net(model_name, net, dataloader, name_to_id, target_images, chosen_ids,
             max_per_target=5, thresh=0, vis=False,
             output_dir=None,):
    """Test a Fast R-CNN network on an image database."""

    #get map from target id to name
    id_to_name = {}
    for name in name_to_id.keys():
        id_to_name[name_to_id[name]] =name 
    #num images in test set
    num_images = len(dataloader)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(dataloader.dataset.get_num_classes())]
    #array of result dicts
    all_results = {} 
    
    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    
    if output_dir is not None:
        det_file = os.path.join(output_dir, model_name+'.json')
        print det_file


    #pre compute features for all targets
    target_features_dict = {}
    for id_ind,t_id in enumerate(chosen_ids):
        t_name = id_to_name[t_id]
        if t_name == 'background':
            continue
        target_data = target_images[t_name]
        target_data = match_and_concat_images(target_data[0][0,:,:,:], target_data[1][0,:,:,:])
        target_data = network.np_to_variable(target_data, is_cuda=True)
        target_data = target_data.permute(0, 3, 1, 2)
        target_features_dict[t_name] = net.features(target_data)




    #for i in range(num_images):
    for i,batch in enumerate(dataloader):
        #if i<100:
        #    continue

        im_data=batch[0].unsqueeze(0).numpy()
        im_data=np.transpose(im_data,(0,2,3,1))
        im_info = im_data.shape[1:]
        #im_info = np.zeros((1,3))
        #im_info[0,:] = [im_data.shape[1],im_data.shape[2],1]
        dontcare_areas = np.zeros((0,4))       



        #get image features

        im_data = network.np_to_variable(im_data, is_cuda=True)
        im_data = im_data.permute(0, 3, 1, 2)
        img_features = net.features(im_data)


        all_image_dets = np.zeros((0,6)) 
        for id_ind,t_id in enumerate(chosen_ids):
            target_name = id_to_name[t_id]
            if target_name == 'background':
                continue



            target_features = target_features_dict[target_name]

            if (target_data is None) or len(target_data) < 1:
                print 'Empty target data: {}'.format(target_name)
                continue

            #target_data = match_and_concat_images(target_data[0][0,:,:,:], target_data[1][0,:,:,:])


            _t['im_detect'].tic()
            scores, boxes, layers = im_detect(net, target_features, img_features, im_info)
            detect_time = _t['im_detect'].toc(average=False)

            if scores[0][1] > .8:
                bp = '/net/bvisionserver3/playpen10/ammirato/Data/feat_maps/'
                imf = layers[0]            
                cc = layers[1]               
                df = layers[2]               
                rpn = layers[3] 

                #imf = (imf/imf.max() + 1) * 100
                #cc = (cc/cc.max() + 1) * 100
                #df = (df/df.max() + 1) * 100
                rpn = (rpn*-1 + rpn.max())*10 


                plt.imshow(imf)
                plt.savefig(fname=bp+'imf_' + target_name + '_' + batch[1][1] + '.jpg')
                plt.imshow(cc)
                plt.savefig(fname=bp+'cc_' + target_name + '_' + batch[1][1] + '.jpg')
                plt.imshow(df)
                plt.savefig(fname=bp+'df_' + target_name + '_' + batch[1][1] + '.jpg')
                plt.imshow(rpn)
                plt.savefig(fname=bp+'rpn_' + target_name + '_' + batch[1][1] + '.jpg')
 #               cv2.imwrite(bp+'imf_' + target_name + '_' + batch[1][1] + '.jpg',imf)
 #               cv2.imwrite(bp+'cc_' + target_name + '_' + batch[1][1] + '.jpg',cc)
 #               cv2.imwrite(bp+'df_' + target_name + '_' + batch[1][1] + '.jpg',df)
 #               cv2.imwrite(bp+'rpn_' + target_name + '_' + batch[1][1] + '.jpg',rpn)
                              
#                breakp = 1

            _t['misc'].tic()

            #get scores for foreground, non maximum supression
            inds = np.where(scores[:, 1] > thresh)[0]
            fg_scores = scores[inds, 1]
            fg_boxes = boxes[inds, 1 * 4:(1 + 1) * 4]
            fg_dets = np.hstack((fg_boxes, fg_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            keep = nms(fg_dets, cfg.TEST.NMS)
            fg_dets = fg_dets[keep, :]

            # Limit to max_per_target detections *over all classes*
            if max_per_target > 0:
                image_scores = np.hstack([fg_dets[:, -1]])
                if len(image_scores) > max_per_target:
                    image_thresh = np.sort(image_scores)[-max_per_target]
                    keep = np.where(fg_dets[:, -1] >= image_thresh)[0]
                    fg_dets = fg_dets[keep, :]
            nms_time = _t['misc'].toc(average=False)

            print 'im_detect: {:d}/{:d} {:.3f}s {:.3f}s' \
                .format(i + 1, num_images, detect_time, nms_time)

            #put class id in the box
            fg_dets = np.insert(fg_dets,4,t_id,axis=1)
            all_image_dets = np.vstack((all_image_dets,fg_dets))

        #record results by image name
        all_results[batch[1][1]] = all_image_dets.tolist()
    if output_dir is not None:
        with open(det_file, 'w') as f:
            json.dump(all_results,f)
    return all_results






if __name__ == '__main__':
    # load data
    data_path = '/net/bvisionserver3/playpen10/ammirato/Data/HalvedRohitData/'
    #data_path = '/net/bvisionserver3/playpen10/ammirato/Data/HalvedGMUData/'
    #target_path = '/net/bvisionserver3/playpen/ammirato/Data/instance_detection_targets/AVD_single_bb_targets/' 
    #target_path = '/net/bvisionserver3/playpen/ammirato/Data/instance_detection_targets/sygen_many_bb_similar_targets/'
    #target_path = '/net/bvisionserver3/playpen/ammirato/Data/instance_detection_targets/AVD_BB_exact_few/'
    #target_path = '/net/bvisionserver3/playpen10/ammirato/Data/instance_detection_targets/AVD_BB_exact_few_and_other_BB_gen_160/'
    #target_path = '/net/bvisionserver3/playpen10/ammirato/Data/instance_detection_targets/AVD_BB_exact_few_and_other_BB_gen_80/'
    target_path = '/net/bvisionserver3/playpen10/ammirato/Data/instance_detection_targets/AVD_BB_exact_few_and_other_BB_gen_and_AVD_ns_BB_80/'
    output_dir='/net/bvisionserver3/playpen/ammirato/Data/Detections/FasterRCNN_AVD/'


    #data_path = '/playpen/ammirato/Data/HalvedRohitData/'
    #target_path = '/playpen/ammirato/Data/Target_Images/AVD_BB_exact_few/'
    #output_dir='/playpen/ammirato/Data/Detections/FasterRCNN_AVD/'


    scene_list=[
             #'Home_001_1',
             #'Home_001_2',
             #'Home_002_1',
             #'Home_003_1',
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

             #'Home_102_1',
             #'Home_104_1',
             #'Home_105_1',


             #'Home_101_1',
             #'Home_102_1',
             'Home_103_1',
#             'Home_104_1',
#             'Home_105_1',
#             'Home_106_1',
#             'Home_107_1',
#             'Home_108_1',
#             'Home_109_1',



             #'test',
             ]
    chosen_ids = [18,50,94,79]#range(28)

    #CREATE TRAIN/TEST splits
    dataset = GetDataSet.get_fasterRCNN_AVD(data_path,
                                            scene_list,
                                            preload=False,
                                            chosen_ids=chosen_ids, 
                                            by_box=False,
                                            fraction_of_no_box=0,
                                            bn_normalize=use_torch_vgg,
                                            max_difficulty=4)



    #CREATE TRAIN/TEST splits
#    dataset = GetDataSet.get_fasterRCNN_GMU(data_path,
#                                            scene_list,
#                                            preload=False,
#                                            chosen_ids=[6],#chosen_ids, 
#                                            by_box=False,
#                                            fraction_of_no_box=0,
#                                            bn_normalize=use_torch_vgg)
#



    batch = dataset[0]


    #create train/test loaders, with CUSTOM COLLATE function
    dataloader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=1,
                                              collate_fn=AVD.collate)

    map_fname = 'all_instance_id_map.txt'
    id_to_name = GetDataSet.get_class_id_to_name_dict(data_path, file_name=map_fname)
    name_to_id = {}
    for cid in id_to_name.keys():
        name_to_id[id_to_name[cid]] = cid 


    if use_torch_vgg:
        target_images = get_target_images(target_path, name_to_id.keys(),
                                          for_testing=True,bn_normalize=True)
    else:
        target_images = get_target_images(target_path, name_to_id.keys(),
                                          for_testing=True,means=means)




    #test multiple trained nets
    for model_name in trained_model_names:
        print model_name
        # load net
        net = TDID()
        #load a previously trained model
        if use_batch_norm:
            vgg16_bn = models.vgg16_bn(pretrained=False)
            net.features = torch.nn.Sequential(*list(vgg16_bn.features.children())[:-1])
            net.features.eval()
        elif use_torch_vgg:
            vgg16 = models.vgg16(pretrained=False)
            net.features = torch.nn.Sequential(*list(vgg16.features.children())[:-1])

        network.load_net(trained_model_path + model_name+'.h5', net)
        print('load model successfully!')

        net.cuda()
        net.eval()

        # evaluation
        test_net(model_name, net, dataloader, name_to_id, target_images,chosen_ids, 
                 max_per_target=max_per_target, thresh=thresh, vis=vis,
                 output_dir=output_dir)




