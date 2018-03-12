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
* `EPS` - 
* `FEATURE_NET_NAME` - which architeture to use as the backbone network. string
* `FRACTION_OF_NO_BOX_IMAGES` - fraction of images to include from training set that have no objects present. float [0,1]
* `FULL_MODEL_LOAD_DIR` - where to load trained models from. string
* `FULL_MODEL_LOAD_NAME` - name of saved model to load. string
* `ID_MAP_FNAME` - name of file that has map from instance name to id. string
* `ID_TO_NAME` 
* `LEARNING_RATE` - learning rate. float
* `LOAD_FULL_MODEL` - whether or not to load a saved model. bool
* `MAX_DETS_PER_TARGET` - maximum detections outputted for a single target/scene image pair. int
* `MAX_NUM_EPOCHS` - maximum number of epochs for training. int
* `MAX_OBJ_DIFFICULTY` - max object difficult as defined by AVD data loader. int
* `META_SAVE_DIR` - where to save the meta (config) data used in training. str
* `MIN_TARGET_SIZE` - minimum size of any dimension for a target images. int
* `MODEL_BASE_SAVE_NAME` - name to use for saving model. string
* `MOMENTUM` 
* `NAME_TO_ID`
* `NMS_THRESH` - box score threshold for nms. float [0,1] 
* `NUM_TARGETS` - how many target images to use. int
* `NUM_WORKERS` - how many worker to use when laoding data. int
* `OBJ_IDS_TO_EXCLUDE` - instances to not include as foreground during training. list of ints
* `POST_NMS_TOP_N` - max number of anchor boxes to keep after nms. int 
* `PRELOAD_TARGET_IMAGES` - 
* `PRE_NMS_TOP_N -`max number of anchor boxes to keep after nms. int
* `PROPOSAL_BATCH_SIZE` - max number of anchors boxes to use for loss for one scene images. int
* `PROPOSAL_BBOX_INSIDE_WEIGHTS` - 
* `PROPOSAL_CLOBBER_POSITIVES` - 
* `PROPOSAL_FG_FRACTION` - max fraction of proposals that can be forground. float [0,1]
* `PROPOSAL_MIN_BOX_SIZE` - minimum size of a proposal box after applying regression parameters. int
* `PROPOSAL_NEGATIVE_OVERLAP` - max overlap of anchor box with gt target box s.t. anchor box can be given gt background label. float [0,1]
* `PROPOSAL_POSITIVE_OVERLAP` - min overlap of anchor box with gt target box s.t. anchor box can be given gt foreground label. float [0,1]
* `PROPOSAL_POSITIVE_WEIGHT` - 
* `PYTORCH_FEATURE_NET` - whether or not to use a pytorch implementation of backbone feature extractor. bool
* `RESIZE_IMG` - how often to resize scene images during training. float [0,1]
* `RESIZE_IMG_FACTOR` -scaling factor to resize images during training. float
* `SAVE_BY_EPOCH` - whether SAVE-FREQ refers to epochs(true) or steps(false). bool
* `SAVE_FREQ` - how often to save the model during training. int
* `SCORE_THRESH` - minimum score for outputting a box during inference. float [0,1]
* `SNAPSHOT_SAVE_DIR` - where to save models during training. string
* `TARGET_IMAGE_DIR` - where target images are stored. string
* `TEST_FRACTION_OF_NO_BOX_IMAGES` - fraction of images to include from testing set that have no objects present. float [0,1]
* `TEST_GROUND_TRUTH_BOXES` - location of file that has annotations of the test set. string
* `TEST_LIST` - list of scenes included in the test set. list of string
* `TEST_NMS_OVERLAP_THRESH` - 
* `TEST_OBJ_IDS` - objects ids to include in the test set. list of ints
* `TEST_ONE_AT_A_TIME` - whether to test one target/scene image pair at a time, or use faster testing method. bool
* `TEST_OUTPUT_DIR` - where to save results of testing. string
* `TEST_RESIZE_IMG_FACTOR` - scale for resizing images for testing. float
* `TRAIN_LIST` - list of scenes included in the training set. list of strings
* `TRAIN_OBJ_IDS` - objects ids to include in the train set. list of ints
* `USE_CC_FEATS` - whether to use the CC feats, or not. bool
* `USE_DIFF_FEATS` - whether to use the DIFF feats, or not. bool
* `USE_IMG_FEATS` - whether to use the IMG feats, or not. bool
* `USE_PRETRAINED_WEIGHTS` - whether to use weights from pytorch pretrained network for backbone feature extractor, or not. bool
* `VAL_FRACTION_OF_NO_BOX_IMAGES` - fraction of images to include from validation set that have no objects present. float [0,1]
* `VAL_GROUND_TRUTH_BOXES` - location of file that has annotations of the validation set. string
* `VAL_LIST` - list of scenes included in the validation set. list of strings
* `VAL_OBJ_IDS` - objects ids to include in the test set. list of ints
* `WEIGHT_DECAY` - 


