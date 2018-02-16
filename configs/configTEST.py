from target_driven_instance_detection.utils import *

class Config():
    """
    Holds all config parameters for training/testing.
    """

    #Directories - MUST BE CHANGED for your environment
    BASE_DIR = '/net/bvisionserver3/playpen10/ammirato/Data/'
    DATA_BASE_DIR = '/net/bvisionserver3/playpen10/ammirato/Data/HalvedRohitData/'
    FULL_MODEL_LOAD_DIR= BASE_DIR + 'Detection/recorded_models_and_meta/models/'
    SNAPSHOT_SAVE_DIR= BASE_DIR + 'Detection/Models/'
    META_SAVE_DIR = BASE_DIR + 'Detection/ModelsMeta/'
    TARGET_IMAGE_DIR= BASE_DIR + 'instance_detection_targets/AVD_BB_exact_few_and_other_BB_gen_and_AVD_ns_BB_80/'
    #TARGET_IMAGE_DIR= BASE_DIR + 'instance_detection_targets/AVD_BB_exact_few_and_other_BB_gen_and_AVD_ns_BB_80_t0_copy/'
    TEST_OUTPUT_DIR = BASE_DIR + 'Detection/TestOutputs/'
    GROUND_TRUTH_BOXES = BASE_DIR + 'RohitCOCOgt/avd_split2.json'
    #PRETRAINED_MODELS_DIR= BASE_DIR + ''
    #using VID dataset is not necessary, set USE_VID to false
    VID_DATA_DIR = BASE_DIR + 'ILSVRC/'


    #Model Loading and saving 
    FEATURE_NET_NAME= 'vgg16_bn'
    PYTORCH_FEATURE_NET= True
    USE_PRETRAINED_WEIGHTS = True
    FULL_MODEL_LOAD_NAME= ''
    LOAD_FULL_MODEL= False 
    MODEL_BASE_SAVE_NAME = 'TDID_TEST_00'
    SAVE_FREQ = 1 
    SAVE_BY_EPOCH = True 
    #BATCH_NORM= True


    #Training 
    MAX_NUM_EPOCHS= 2 
    BATCH_SIZE = 4 
    LEARNING_RATE = .001
    MOMENTUM = .9
    WEIGHT_DECAY = .0005
    DISPLAY_INTERVAL = 10
    NUM_WORKERS = 8 
    LOSS_MULT = 1
    IMG_RESIZE = .75 

    #Target Images
    PRELOAD_TARGET_IMAGES= False
    AUGMENT_TARGET_IMAGES= True 
    MIN_TARGET_SIZE = 32

    #Training Data
    ID_MAP_FNAME= 'all_instance_id_map.txt'
    ID_TO_NAME = {}
    NAME_TO_ID = {}
    OBJ_IDS_TO_EXCLUDE = [8,18,32,33]

    TRAIN_OBJ_IDS= [6]
    FRACTION_OF_NO_BOX_IMAGES = .01 
    MAX_OBJ_DIFFICULTY= 4
    TRAIN_LIST= [
                 'Home_003_1',
                ]

    VAL_OBJ_IDS = TRAIN_OBJ_IDS 
    VAL_FRACTION_OF_NO_BOX_IMAGES = .01 
    VAL_LIST=   [
                 'Home_003_1',
                ]

    ##############################################
    #Testing
    TEST_IMG_RESIZE = 0 
    MAX_DETS_PER_TARGET = 5
    SCORE_THRESH = .01
    TEST_NMS_OVERLAP_THRESH = .7

    TEST_OBJ_IDS=[6]#,6,7,9,10,11] # [cid for cid in range(1,33) if cid not in OBJ_IDS_TO_EXCLUDE]
    TEST_FRACTION_OF_NO_BOX_IMAGES = .01
    TEST_LIST = [ 
                 'Home_003_1',
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
    cfg.ID_TO_NAME = get_class_id_to_name_dict(cfg.DATA_BASE_DIR,
                                               cfg.ID_MAP_FNAME)
    name_to_id = {}
    for cid in cfg.ID_TO_NAME.keys():
        name_to_id[cfg.ID_TO_NAME[cid]] = cid 
    cfg.NAME_TO_ID = name_to_id
    
    #ensures chosen object ids are valid(exist in the name/id map file)
    

    return cfg 
