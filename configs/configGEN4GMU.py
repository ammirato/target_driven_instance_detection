from utils import *
import os

class Config():
    """
    Holds all config parameters for training/testing.
    """

    #Directories - MUST BE CHANGED for your environment
    DATA_BASE_DIR = '/net/bvisionserver3/playpen/ammirato/sandbox/code/target_driven_instance_detection/Data/'
    AVD_ROOT_DIR = '/net/bvisionserver3/playpen10/ammirato/Data/HalvedRohitData/'
    FULL_MODEL_LOAD_DIR= os.path.join(DATA_BASE_DIR, 'Models/')
    SNAPSHOT_SAVE_DIR= os.path.join(DATA_BASE_DIR , 'Models/')
    META_SAVE_DIR = os.path.join(DATA_BASE_DIR, 'ModelsMeta/')
    TARGET_IMAGE_DIR= os.path.join(DATA_BASE_DIR, 'uw_real_and_BB/')
    #TARGET_IMAGE_DIR= os.path.join(DATA_BASE_DIR, 'HR_target/')
    TEST_OUTPUT_DIR = os.path.join(DATA_BASE_DIR, 'TestOutputs/')
    TEST_GROUND_TRUTH_BOXES = os.path.join(DATA_BASE_DIR, 'GT/all_gmu2.json')
    TEST_GROUND_TRUTH_BOXES = os.path.join(DATA_BASE_DIR, 'GT/AVD_split2_test.json')
    #VAL_GROUND_TRUTH_BOXES = os.path.join(DATA_BASE_DIR ,'GT/AVD_part3_val.json')
    #VAL_GROUND_TRUTH_BOXES = os.path.join(DATA_BASE_DIR ,'GT/AVD_split2_test.json')
    VAL_GROUND_TRUTH_BOXES = os.path.join(DATA_BASE_DIR ,'GT/home0031.json')
    AVD_EXTRA_FILE = os.path.join(AVD_ROOT_DIR, 'AVD_extra.txt')

    #Model Loading and saving 
    FEATURE_NET_NAME= 'vgg16_bn'
    PYTORCH_FEATURE_NET= True
    USE_PRETRAINED_WEIGHTS = True
    FULL_MODEL_LOAD_NAME= 'TDID_GEN4GMU_06_740000_7.48426_0.39258_0.46357.h5'
    LOAD_FULL_MODEL = False 
    MODEL_BASE_SAVE_NAME = 'TDID_GEN4GMU_12'
    SAVE_FREQ = 50000
    SAVE_BY_EPOCH = False 


    #Training 
    MAX_NUM_EPOCHS= 500 
    MAX_NUM_ITERATIONS= 600000 
    BATCH_SIZE = 4 
    LEARNING_RATE = .0005
    MOMENTUM = .9
    WEIGHT_DECAY = .0005
    DISPLAY_INTERVAL = 10
    NUM_WORKERS = 4 
    RESIZE_IMG = 0 
    RESIZE_IMG_FACTOR = .5 
    CHOOSE_PRESENT_TARGET = .8
    DET4CLASS = False 
    USE_AVD_EXTRA = 0 
    USE_VID = 0 
    BLACK_OUT_IDS=1

    #Target Images
    PRELOAD_TARGET_IMAGES= False
    AUGMENT_TARGET_IMAGES= .9 
    AUGMENT_TARGET_ILLUMINATION= .3 
    MIN_TARGET_SIZE = 32

    #Training Data
    #ID_MAP_FNAME= 'all_instance_id_map.txt'
    ID_MAP_FNAME= 'hybrid_instance_id_map.txt'
    ID_TO_NAME = {}
    NAME_TO_ID = {}
    OBJ_IDS_TO_EXCLUDE = [8,32,33] + [5,50,10,12,14,79,28,94,96,18,21] 

    TRAIN_OBJ_IDS=[cid for cid in range(1,2111) if cid not in OBJ_IDS_TO_EXCLUDE] 
    FRACTION_OF_NO_BOX_IMAGES = .1 
    MAX_OBJ_DIFFICULTY= 4
    TRAIN_LIST= [
                 'Home_001_1',
#                 'Home_001_2',
                 'Home_002_1',
 #                'Home_003_1',
                 'Home_003_2',
                 'Home_004_1',
  #               'Home_004_2',
                 'Home_005_1',
   #              'Home_005_2',
                 'Home_006_1',
                 'Home_008_1',
                 'Home_014_1',
    #             'Home_014_2',
                 'Office_001_1',

                 #'Home_101_1',
                 #'Home_102_1',
                 #'Home_103_1',
                 #'Home_104_1',
                 #'Home_105_1',
                 #'Home_106_1',
                 #'Home_107_1',
                 #'Home_108_1',
                 #'Home_109_1',

                 'Gen_010_2',
                 'Gen_010_2',
                 'Gen_010_5',
                 'Gen_010_6',
                 'Gen_010_6',
                 #'Gen_010_5',
                 #'Gen_010_6',
                 #'Gen_009_4',

                 'Office_201_1',
                 'Office_201_2',
                 'Office_201_3',
                 'Office_202_1',
                 'Office_203_1',
                 'Office_204_1',
                 'Office_205_1',
                 'Office_205_2',

                 'Office_201_1',
                 'Office_201_2',
                 'Office_201_3',
                 'Office_202_1',
                 'Office_203_1',
                 'Office_204_1',
                 'Office_205_1',
                 'Office_205_2',
                ]

    TEST_ONLY_OBJ_IDS = [5,50,10,12,14,79,28,94,96,18,21]
    #VAL_OBJ_IDS = TRAIN_OBJ_IDS + TEST_ONLY_OBJ_IDS 
    VAL_OBJ_IDS =[1,2,3,4,6,7,9,11,13,15,16,17]# TRAIN_OBJ_IDS 
    VAL_FRACTION_OF_NO_BOX_IMAGES =1
    VAL_LIST=   [
                 'Home_003_1',
                ]
    TEST_ONLY_LIST=   [
                 'Home_101_1',
                 'Home_102_1',
                 'Home_103_1',
                 'Home_104_1',
                 'Home_105_1',
                 'Home_106_1',
                 'Home_107_1',
                 'Home_108_1',
                 'Home_109_1',
                ]

    ##############################################
    #Testing
    TEST_RESIZE_IMG_FACTOR = 0 
    TEST_RESIZE_BOXES_FACTOR = 2 
    MAX_DETS_PER_TARGET = 5
    SCORE_THRESH = .01
    TEST_NMS_OVERLAP_THRESH = .7

    TEST_OBJ_IDS= TEST_ONLY_OBJ_IDS#[1,2,3,4,6,7,9,11,13,15,16,17,19,20,22,23,24,25,26,27,28,29,30,31]#TEST_ONLY_OBJ_IDS
    TEST_FRACTION_OF_NO_BOX_IMAGES = 1 
    TEST_LIST = TEST_ONLY_LIST# [ 
#                'Home_003_1',
#                'Home_003_2',
#                'Office_001_1',
#    #             'Home_101_1',
#    #             'Home_102_1',
#    #             'Home_103_1',
#    #             'Home_104_1',
#    #             'Home_105_1',
#    #             'Home_106_1',
#    #             'Home_107_1',
#    #             'Home_108_1',
#    #             'Home_109_1',
#                ]
    TEST_ONE_AT_A_TIME = False 
    ###############################################
    #Model paramters
    ANCHOR_SCALES = [1,2,4]
    NUM_TARGETS = 2
    CORR_WITH_POOLED = True 
    USE_IMG_FEATS = False 
    USE_DIFF_FEATS = True 
    USE_CC_FEATS = True 

    PRE_NMS_TOP_N = 6000
    POST_NMS_TOP_N = 300
    NMS_THRESH = .7
    PROPOSAL_MIN_BOX_SIZE = 8 
    PROPOSAL_CLOBBER_POSITIVES = False 
    PROPOSAL_NEGATIVE_OVERLAP = .3
    PROPOSAL_POSITIVE_OVERLAP = .6
    PROPOSAL_FG_FRACTION = .5
    PROPOSAL_BATCH_SIZE = 300 
    PROPOSAL_POSITIVE_WEIGHT = -1
    PROPOSAL_BBOX_INSIDE_WEIGHTS = [1,1,1,1]

    EPS = 1e-14



def get_config():

    cfg = Config()
    cfg.ID_TO_NAME = get_class_id_to_name_dict(cfg.AVD_ROOT_DIR,
                                               cfg.ID_MAP_FNAME)
    name_to_id = {}
    for cid in cfg.ID_TO_NAME.keys():
        name_to_id[cfg.ID_TO_NAME[cid]] = cid 
    cfg.NAME_TO_ID = name_to_id
    
    return cfg 
