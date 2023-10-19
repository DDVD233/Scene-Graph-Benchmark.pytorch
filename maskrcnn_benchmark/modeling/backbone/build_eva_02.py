from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
import os
from torch.hub import download_url_to_file
import logging


def build_eva2_model(cfg_parent):
    cfg_path = cfg_parent.MODEL.BACKBONE_CONFIG
    logging.info(f"Loading model from {cfg_path}")
    cfg = LazyConfig.load(cfg_path)
    model = instantiate(cfg.model)
    checkpointer = DetectionCheckpointer(model)
    try:
        ckpt_filename = cfg_parent.MODEL.BACKBONE_WEIGHT
        if not os.path.exists(ckpt_filename):
            print("Checkpoint not loaded")
            return model
        checkpointer.load(ckpt_filename)
    except Exception as e:
        logging.warning(e)
    return model
