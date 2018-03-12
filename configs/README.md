# Config Files
Meant to be the only file changed when changing experiments. Defines all needed model training and testing parameters and paths. 

The parameters are defined in the following format:
`name` - text definition. expected type/format




* `ANCHOR_SCALES` - scale of anchor boxes to be used. [int,int,int]
* `AUGMENT_TARGET_ILLUMINATION` - how often to change the illumination of target images. float [0,1]
* `AUGMENT_TARGET_IMAGES` - how often to augment the target images. float [0,1]
* `AVD_ROOT_DIR` - directory that holds all scene directories for the AVD. string
* `BATCH_SIZE` - batch size for training. int
* `CHOOSE_PRESENT_TARGET` - about how often the target object is in the scene image for training. float [0,1]
* `CORR_WITH_POOLED` - whether or not to pool the target features to 1x1 before correlation. bool
* `DATA_BASE_DIR` - optional,  base directory that holds other directories. string
* `DET4CLASS` - whether this is a classification experiment or not. bool
* `DISPLAY_INTERVAL` - how often to print info during training. int
* `EPS`  
* `FEATURE_NET_NAME` - which architeture to use as the backbone network. string
* `FRACTION_OF_NO_BOX_IMAGES` - fraction of images to include from training set that have no objects present. float [0,1]
* `FULL_MODEL_LOAD_DIR` - where to load trained models from. string
* `FULL_MODEL_LOAD_NAME` - 
* `ID_MAP_FNAME`
* `ID_TO_NAME`
* `LEARNING_RATE`
* `LOAD_FULL_MODEL`
* `MAX_DETS_PER_TARGET`
* `MAX_NUM_EPOCHS`
* `MAX_OBJ_DIFFICULTY`
* `META_SAVE_DIR`
* `MIN_TARGET_SIZE`
* `MODEL_BASE_SAVE_NAME`
* `MOMENTUM`
* `NAME_TO_ID`
* `NMS_THRESH`
* `NUM_TARGETS`
* `NUM_WORKERS`
* `OBJ_IDS_TO_EXCLUDE`
* `POST_NMS_TOP_N`
* `PRELOAD_TARGET_IMAGES`
* `PRE_NMS_TOP_N`
* `PROPOSAL_BATCH_SIZE`
* `PROPOSAL_BBOX_INSIDE_WEIGHTS`
* `PROPOSAL_CLOBBER_POSITIVES`
* `PROPOSAL_FG_FRACTION`
* `PROPOSAL_MIN_BOX_SIZE`
* `PROPOSAL_NEGATIVE_OVERLAP`
* `PROPOSAL_POSITIVE_OVERLAP`
* `PROPOSAL_POSITIVE_WEIGHT`
* `PYTORCH_FEATURE_NET`
* `RESIZE_IMG`
* `RESIZE_IMG_FACTOR`
* `SAVE_BY_EPOCH`
* `SAVE_FREQ`
* `SCORE_THRESH`
* `SNAPSHOT_SAVE_DIR`
* `TARGET_IMAGE_DIR`
* `TEST_FRACTION_OF_NO_BOX_IMAGES`
* `TEST_GROUND_TRUTH_BOXES`
* `TEST_LIST`
* `TEST_NMS_OVERLAP_THRESH`
* `TEST_OBJ_IDS`
* `TEST_ONE_AT_A_TIME`
* `TEST_OUTPUT_DIR`
* `TEST_RESIZE_IMG_FACTOR`
* `TRAIN_LIST`
* `TRAIN_OBJ_IDS`
* `USE_CC_FEATS`
* `USE_DIFF_FEATS`
* `USE_IMG_FEATS`
* `USE_PRETRAINED_WEIGHTS`
* `VAL_FRACTION_OF_NO_BOX_IMAGES`
* `VAL_GROUND_TRUTH_BOXES`
* `VAL_LIST`
* `VAL_OBJ_IDS`
* `WEIGHT_DECAY`


