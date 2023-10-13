from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
import os
from torch.hub import download_url_to_file


def build_eva2_model(cfg_parent, cfg_path='EVA-02/det/projects/ViTDet/configs/eva2_o365_to_coco/'
                              'eva2_vg.py'):
    cfg = LazyConfig.load(cfg_path)
    model = instantiate(cfg.model)
    checkpointer = DetectionCheckpointer(model)
    ckpt_filename = cfg_parent.MODEL.BACKBONE_WEIGHT
    if not os.path.exists(ckpt_filename):
        print("Checkpoint not loaded")
        return model
        # raise FileNotFoundError(f'{ckpt_filename} not found')
        # download_url_to_file('https://huggingface.co/Yuxin-CV/EVA-02/resolve/main/eva02/det/eva02_L_lvis_sys_o365.pth',
        #                         ckpt_filename)
    checkpointer.load(ckpt_filename)
    return model
