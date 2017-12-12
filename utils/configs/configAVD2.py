from instance_detection.utils.utils import *



class Config():
    """
    Holds all config parameters for training/testing.
    """

    #Model Loading and saving 
    BATCH_NORM= True
    PRETRAINED_MODELS_DIR= ''
    PRETRAINED_MODEL_NAME= 'VGG16'
    PYTORCH_FEATURE_NET= True
    USE_PRETRAINED_WEIGHTS = True
    FULL_MODEL_LOAD_DIR= '/net/bvisionserver3/playpen/ammirato/Data/Detections/saved_models/'
    FULL_MODEL_LOAD_NAME= ''
    LOAD_FULL_MODEL= False
    SNAPSHOT_SAVE_DIR= '/net/bvisionserver3/playpen/ammirato/Data/Detections/saved_models/'
    META_SAVE_DIR = '/net/bvisionserver3/playpen/ammirato/Data/Detections/saved_models_meta/'
    MODEL_BASE_SAVE_NAME = 'TDID_AVD2_0'
    SAVE_FREQ = 1
    SAVE_BY_EPOCH = True  


    #Training 
    MAX_NUM_EPOCHS= 30
    BATCH_SIZE= 5
    LEARNING_RATE = .001
    MOMENTUM = .9
    WEIGHT_DECAY = .0005
    DISPLAY_INTERVAL = 10
    NUM_WORKERS = 4

    #Target Images
    PRELOAD_TARGET_IMAGES= False
    AUGMENT_TARGET_IMAGES= True
    TARGET_IMAGE_DIR= '/net/bvisionserver3/playpen10/ammirato/Data/instance_detection_targets/AVD_BB_exact_few_and_other_BB_gen_and_AVD_ns_BB_80/'
    MIN_TARGET_SIZE = 32

    #Triaing Data
    DATA_BASE_DIR = '/net/bvisionserver3/playpen10/ammirato/Data/HalvedRohitData/'
    ID_MAP_FNAME= 'all_instance_id_map.txt'
    ID_TO_NAME = {}
    NAME_TO_ID = {}
    TRAIN_LIST= [
                 'Home_001_1',
#                 'Home_001_2',
#                 'Home_002_1',
#                 'Home_004_1',
#                 'Home_004_2',
#                 'Home_005_1',
#                 'Home_005_2',
#                 'Home_006_1',
#                 'Home_008_1',
#                 'Home_014_1',
#                 'Home_014_2',
                ]
    VAL_LIST=   [
                 'Home_003_1',
#                 'Home_003_2',
#                 'Office_001_1',
                ]
    FRACTION_OF_NO_BOX_IMAGES = .1
    VAL_FRACTION_OF_NO_BOX_IMAGES = 1
    OBJ_IDS_TO_EXCLUDE = [8,18, 32,33]
    TRAIN_OBJ_IDS= [cid for cid in range(1,2) if cid not in OBJ_IDS_TO_EXCLUDE]
    VAL_OBJ_IDS = TRAIN_OBJ_IDS 
    MAX_OBJ_DIFFICULTY= 4

    #VID dataset
    USE_VID = False
    VID_DATA_DIR = '/net/bvisionserver3/playpen10/ammirato/Data/ILSVRC/'
    VID_MAX_MIN_TARGET_SIZE = [200,16]
    VID_SUBSET = 'train_single' 



    ##############################################
    #Testing
    TEST_OUTPUT_DIR = '/net/bvisionserver3/playpen/ammirato/Data/Detections/coco_dets/'
    MAX_DETS_PER_TARGET = 5
    SCORE_THRESH = .01
    TEST_NMS_OVERLAP_THRESH = .7

    #Evaluation
    GROUND_TRUTH_BOXES = '/net/bvisionserver3/playpen10/ammirato/Data/RohitCOCOgt/avd_split2.json'

    ###############################################
    #Model paramters
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


#
#    IS_MULTISCALE= False
#    NET_NAME= VGGnet
#    ANCHOR_SCALES= [2,4,8]
#    NCLASSES= 2 
#    TRAIN=
#      OHEM= False
#      RPN_BATCHSIZE= 300
#      BATCH_SIZE= 300
#      LOG_IMAGE_ITERS= 100
#      DISPLAY= 10
#      SNAPSHOT_ITERS= 5000
#      HAS_RPN= True
#      LEARNING_RATE= 0.001
#      MOMENTUM= 0.9
#      GAMMA= 0.1
#      STEPSIZE= 60000
#      IMS_PER_BATCH= 1
#      BBOX_NORMALIZE_TARGETS_PRECOMPUTED= False
#      RPN_POSITIVE_OVERLAP= 0.7
#      RPN_BATCHSIZE= 256
#      PROPOSAL_METHOD= gt
#      BG_THRESH_LO= 0.0
#      PRECLUDE_HARD_SAMPLES= True
#      BBOX_INSIDE_WEIGHTS= [1, 1, 1, 1]
#      RPN_BBOX_INSIDE_WEIGHTS= [1, 1, 1, 1]
#      RPN_POSITIVE_WEIGHT= -1.0
#      FG_FRACTION= 0.3
#      WEIGHT_DECAY= 0.0005
#    TEST=
#      HAS_RPN= True




def get_config():

    cfg = Config()
    cfg.ID_TO_NAME = get_class_id_to_name_dict(cfg.DATA_BASE_DIR,
                                               cfg.ID_MAP_FNAME)
    name_to_id = {}
    for cid in cfg.ID_TO_NAME.keys():
        name_to_id[cfg.ID_TO_NAME[cid]] = cid 
    cfg.NAME_TO_ID = name_to_id
    
    #ensures chosen object ids are valid(exist in the name/id map file)
    

    return cfg 
