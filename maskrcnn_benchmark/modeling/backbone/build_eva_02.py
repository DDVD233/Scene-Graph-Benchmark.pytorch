from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate


def build_eva2_model(cfg_path='EVA-02/det/projects/ViTDet/configs/eva2_o365_to_lvis/'
                              'eva2_o365_to_lvis_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8.py'):
    cfg = LazyConfig.load(cfg_path)
    model = instantiate(cfg.model)
    checkpointer = DetectionCheckpointer(model)
    weights = cfg.train.init_checkpoint
    checkpointer.load(weights)
    return model
