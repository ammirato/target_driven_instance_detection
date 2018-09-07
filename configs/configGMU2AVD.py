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
    #TARGET_IMAGE_DIR= os.path.join(DATA_BASE_DIR, 'AVD_and_BigBIRD_targets_v1/')
    TARGET_IMAGE_DIR= os.path.join(DATA_BASE_DIR, 'HR_target/')
    TEST_OUTPUT_DIR = os.path.join(DATA_BASE_DIR, 'TestOutputs/')
    TEST_GROUND_TRUTH_BOXES = os.path.join(DATA_BASE_DIR, 'GT/AVD_split2_test.json')
    #VAL_GROUND_TRUTH_BOXES = os.path.join(DATA_BASE_DIR ,'GT/AVD_part3_val.json')
    VAL_GROUND_TRUTH_BOXES = os.path.join(DATA_BASE_DIR ,'GT/AVD_split2_test.json')


    #Model Loading and saving 
    FEATURE_NET_NAME= 'vgg16_bn'
    PYTORCH_FEATURE_NET= True
    USE_PRETRAINED_WEIGHTS = True
    FULL_MODEL_LOAD_NAME= ''
    LOAD_FULL_MODEL= False 
    MODEL_BASE_SAVE_NAME = 'FasterRCNN_GMU2AVD_01'
    SAVE_FREQ  = 25 
    SAVE_BY_EPOCH = True 


    #Training 
    MAX_NUM_EPOCHS= 50 
    BATCH_SIZE = 5 
    LEARNING_RATE = .0001
    MOMENTUM = .9
    WEIGHT_DECAY = .0005
    DISPLAY_INTERVAL = 10
    NUM_WORKERS = 4 
    RESIZE_IMG = 0 
    RESIZE_IMG_FACTOR = .5 
    CHOOSE_PRESENT_TARGET = .6
    DET4CLASS = False 

    #Target Images
    PRELOAD_TARGET_IMAGES= False
    AUGMENT_TARGET_IMAGES= .9 
    AUGMENT_TARGET_ILLUMINATION= .3 
    MIN_TARGET_SIZE = 32

    #Training Data
    ID_MAP_FNAME= 'all_instance_id_map.txt'
    ID_TO_NAME = {}
    NAME_TO_ID = {}
    OBJ_IDS_TO_EXCLUDE = [8,18,32,33]

    TRAIN_OBJ_IDS=[5,10,12,14,21,28]#[cid for cid in range(1,33) if cid not in OBJ_IDS_TO_EXCLUDE] 
    FRACTION_OF_NO_BOX_IMAGES = .01 
    MAX_OBJ_DIFFICULTY= 4
    TRAIN_LIST= [
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

    VAL_OBJ_IDS = TRAIN_OBJ_IDS 
    VAL_FRACTION_OF_NO_BOX_IMAGES = 1 
    VAL_LIST=   [
                 'Home_001_1',
                 'Home_001_2',
                 'Home_002_1',
                 'Home_003_1',
                 'Home_003_2',
                 'Home_004_1',
                 'Home_004_2',
                 'Home_005_1',
                 'Home_005_2',
                 'Home_006_1',
                 'Home_008_1',
                 'Home_014_1',
                 'Home_014_2',
                 'Office_001_1',
                ]

    ##############################################
    #Testing
    TEST_RESIZE_IMG_FACTOR = 0 
    TEST_RESIZE_BOXES_FACTOR = 2
    MAX_DETS_PER_TARGET = 5
    SCORE_THRESH = .01
    TEST_NMS_OVERLAP_THRESH = .7

    TEST_OBJ_IDS= TRAIN_OBJ_IDS
    TEST_FRACTION_OF_NO_BOX_IMAGES =  1 
    TEST_LIST = [ 
                 'Home_003_1',
                 'Home_003_2',
                 'Office_001_1',
                ]
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
