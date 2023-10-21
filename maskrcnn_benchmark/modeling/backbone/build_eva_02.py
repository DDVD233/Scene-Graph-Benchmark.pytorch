from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
import os
from torch.hub import download_url_to_file


def build_eva2_model(cfg_parent):
    cfg_path = cfg_parent.MODEL.BACKBONE_CONFIG
    print(f"Loading BACKBONE model from {cfg_path}")
    cfg = LazyConfig.load(cfg_path)
    model = instantiate(cfg.model)
    checkpointer = DetectionCheckpointer(model)
    try:
        ckpt_filename = cfg_parent.MODEL.BACKBONE_WEIGHT
    except Exception as e:
        print(e)
        return model

    if not os.path.exists(ckpt_filename):
        raise FileNotFoundError(f"BACKBONE_WEIGHT {ckpt_filename} not found")
    checkpointer.load(ckpt_filename)
    return model
