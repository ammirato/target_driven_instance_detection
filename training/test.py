import torch
import importlib

cfg = importlib.import_module('instance_detection.utils.configs.config')
cfg = cfg.get_config()

print cfg.TRAIN_OBJ_IDS
