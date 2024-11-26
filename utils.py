import os
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import yaml
import codecs
import logging
from datetime import datetime

# Local modules and custom files
from network import MMCDE
import data_provider
from metrics import *

def load_config(config_path="config.yaml"):
    """Load the config from a YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

    
def setup_logging(log_dir='./logs', log_file="training.log", log_level="INFO"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    
    logging.basicConfig(
        filename=log_path,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(module)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO 
    )
    logger = logging.getLogger(__name__)
    return logger

    
def pretrained_model_mapper(backbone='multimodal'):
    huggingface_mapper = {
    'multimodal': 'bert-base-uncased'
    }

    return huggingface_mapper[backbone]


def init_model(model, logger, init_checkpoint=None):
    if init_checkpoint is not None:
        state_dict = torch.load(init_checkpoint, map_location="cpu")
        if init_checkpoint.endswith('pt'):
            model.bert.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict, strict=False)

        logger.debug(model.module) if hasattr(model, 'module') else logger.debug(model)
        
        return model