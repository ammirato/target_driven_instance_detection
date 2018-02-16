from instance_detection.utils.utils import *

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
    #TARGET_IMAGE_DIR= BASE_DIR + 'instance_detection_targets/uw_targets_class_80_1/'
    #TARGET_IMAGE_DIR= BASE_DIR + 'instance_detection_targets/most_bb_and_uw_80_min/'
    #TARGET_IMAGE_DDIR= BASE_DIR + 'instance_detection_targets/uw_real_and_BB/'
    TARGET_IMAGE_DIR= BASE_DIR + 'instance_detection_targets/uw_real_and_BB_single/'
    #TARGET_IMAGE_DIR= BASE_DIR + 'instance_detection_targets/uw_real_and_BB_one/'
    #TARGET_IMAGE_DIR= BASE_DIR + 'instance_detection_targets/AVD_BB_exact_few_and_other_BB_gen_and_AVD_ns_BB_and_UW_80/'
    TEST_OUTPUT_DIR = BASE_DIR + 'Detection/TestOutputs/'
    GROUND_TRUTH_BOXES = BASE_DIR + 'RohitCOCOgt/uw_2081.json'
    #PRETRAINED_MODELS_DIR= BASE_DIR + ''
    #using VID dataset is not necessary, set USE_VID to false
    VID_DATA_DIR = BASE_DIR + 'ILSVRC/'


    #Model Loading and saving 
    FEATURE_NET_NAME= 'alexnet'
    PYTORCH_FEATURE_NET= True
    USE_PRETRAINED_WEIGHTS = True
    FULL_MODEL_LOAD_NAME= ''
    LOAD_FULL_MODEL= False 
    MODEL_BASE_SAVE_NAME = 'TDID_GEN4UWC_20'
    SAVE_FREQ = 10000 
    SAVE_BY_EPOCH = False 
    #BATCH_NORM= True


    #Training 
    MAX_NUM_EPOCHS= 100
    BATCH_SIZE = 128
    LEARNING_RATE = .001
    MOMENTUM = .9
    WEIGHT_DECAY = .0005
    DISPLAY_INTERVAL = 10
    NUM_WORKERS = 4 
    LOSS_MULT = 1
    IMG_RESIZE = 0
    AUGMENT_SCENE_IMAGES= True 

    #Target Images
    PRELOAD_TARGET_IMAGES= False
    AUGMENT_TARGET_IMAGES= True 
    MIN_TARGET_SIZE = 64 

    #Training Data
    ID_MAP_FNAME= 'hybrid_instance_id_map.txt'
    ID_TO_NAME = {}
    NAME_TO_ID = {}
    OBJ_IDS_TO_EXCLUDE = [8,32,33,53, 1050,1051,1052,1053,1054,1055,1269,1270,1271,1272,1273,1274,1140,1141,1142,1143,1243,1244,1245,1247,1246,1252,1253,1254,1255,1256,1257,1258,1259,1004,1005,1006,1007,1008]

    TRAIN_OBJ_IDS= [x for x in range(1,2111)  if x not in OBJ_IDS_TO_EXCLUDE] 
    FRACTION_OF_NO_BOX_IMAGES = 0 
    MAX_OBJ_DIFFICULTY= 4
    TRAIN_LIST= [
                 'Office_301_1',
                 'Office_301_2',
                 'Office_301_3',
                 #'Office_209_1',
                 'Office_209_2',
                 'Office_211_1',
                ]

    VAL_OBJ_IDS = [1050,1052,1053,1054,1055,1270,1271,1272,1140,1142,1143,1243,1244,1245,1247,1252,1255,1256,1257,1004,1005,1007] 
    VAL_FRACTION_OF_NO_BOX_IMAGES = 0 
    VAL_LIST=   [
                 'Office_210_1',
                ]

    #VID dataset
    USE_VID = False 
    VID_MAX_MIN_TARGET_SIZE = [200,16]
    VID_SUBSET = 'train_single' 


    ##############################################
    #Testing
    MAX_DETS_PER_TARGET = 5
    SCORE_THRESH = .01
    TEST_NMS_OVERLAP_THRESH = .7

    TEST_OBJ_IDS= VAL_OBJ_IDS 
    #TEST_OBJ_IDS= TRAIN_OBJ_IDS 
    TEST_FRACTION_OF_NO_BOX_IMAGES = 0 
    TEST_LIST = [ 
                #'Office_207_2',
                #'Office_301_1',
                'Office_210_1',
                ]
    TEST_ONE_AT_A_TIME = False 
    ###############################################
    #Model paramters
    ANCHOR_SCALES = [8,12,14]
    NUM_TARGETS = 2 
    CORR_WITH_POOLED = True 
    USE_IMG_FEATS = False
    USE_DIFF_FEATS = True
    USE_CC_FEATS = True 
    #USE_BOX_LOSS = True
    #USE_ROI_LOSS = True

    PRE_NMS_TOP_N = 6000
    POST_NMS_TOP_N = 300
    NMS_THRESH = .7
    PROPOSAL_MIN_BOX_SIZE = 8 
    PROPOSAL_CLOBBER_POSITIVES = False 
    PROPOSAL_NEGATIVE_OVERLAP = .001
    PROPOSAL_POSITIVE_OVERLAP = .1
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
