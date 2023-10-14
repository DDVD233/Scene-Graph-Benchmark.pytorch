from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
import os
from torch.hub import download_url_to_file


def build_eva2_model(cfg_parent, cfg_path='EVA-02/det/projects/ViTDet/configs/eva2_o365_to_coco/'
                              'eva2_vg.py'):
    cfg = LazyConfig.load(cfg_path)
    model = instantiate(cfg.model)
    checkpointer = DetectionCheckpointer(model)
    try:
        ckpt_filename = cfg_parent.MODEL.BACKBONE_WEIGHT
        if not os.path.exists(ckpt_filename):
            print("Checkpoint not loaded")
            return model
        checkpointer.load(ckpt_filename)
    except:
        pass
    return model
