# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch

from .box_head.box_head import build_roi_box_head
from .mask_head.mask_head import build_roi_mask_head
from .attribute_head.attribute_head import build_roi_attribute_head
from .keypoint_head.keypoint_head import build_roi_keypoint_head
from .relation_head.relation_head import build_roi_relation_head


class CombinedROIHeads(torch.nn.ModuleDict):
    """
    Combines a set of individual heads (for box prediction or masks) into a single
    head.
    """

    def __init__(self, cfg, heads):
        super(CombinedROIHeads, self).__init__(heads)
        self.cfg = cfg.clone()
        if cfg.MODEL.MASK_ON and cfg.MODEL.ROI_MASK_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.mask.feature_extractor = self.box.feature_extractor
        if cfg.MODEL.KEYPOINT_ON and cfg.MODEL.ROI_KEYPOINT_HEAD.SHARE_BOX_FEATURE_EXTRACTOR:
            self.keypoint.feature_extractor = self.box.feature_extractor

    def forward(self, features, detections, targets=None, logger=None):
        losses = {}

        if self.cfg.MODEL.RELATION_ON:
            # it may be not safe to share features due to post processing
            # During training, self.box() will return the unaltered proposals as "detections"
            # this makes the API consistent during training and testing
            if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX and targets is not None:
                x, detections, loss_relation = self.relation(features, targets, targets, logger)
            else:
                x, detections, loss_relation = self.relation(features, detections, targets, logger)
            losses.update(loss_relation)

        return x, detections, losses


def build_roi_heads(cfg, in_channels):
    # individually create the heads, that will be combined together
    # afterwards
    roi_heads = []
    if cfg.MODEL.RETINANET_ON:
        return []

    if not cfg.MODEL.RPN_ONLY:
        roi_heads.append(("box", build_roi_box_head(cfg, in_channels)))
    if cfg.MODEL.MASK_ON:
        roi_heads.append(("mask", build_roi_mask_head(cfg, in_channels)))
    if cfg.MODEL.KEYPOINT_ON:
        roi_heads.append(("keypoint", build_roi_keypoint_head(cfg, in_channels)))
    if cfg.MODEL.RELATION_ON:
        roi_heads.append(("relation", build_roi_relation_head(cfg, in_channels)))
    if cfg.MODEL.ATTRIBUTE_ON:
        roi_heads.append(("attribute", build_roi_attribute_head(cfg, in_channels)))

    # combine individual heads in a single module
    if roi_heads:
        roi_heads = CombinedROIHeads(cfg, roi_heads)

    return roi_heads
