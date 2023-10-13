# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from ..preprocessing import Preprocessing
from maskrcnn_benchmark.structures.bounding_box import BoxList
import torch.nn.functional as F


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()
        self.cfg = cfg.clone()
        self.backbone = build_backbone(cfg)
        shapes = self.backbone.backbone.output_shape()
        out_channels = list(shapes.values())[0].channels
        self.roi_heads = build_roi_heads(cfg, out_channels)
        self.preprocess = Preprocessing()
        self.num_features = 512

    def instances_to_boxlist(self, instances, features, filter=True, max_dets=20):
        """
        Convert a list of detectron2 Instances to a list of BoxList

        Args:
            instances (list[Instances]): a list of detectron2 Instances
            filter (bool): filter out instances with score < 0.2
        """

        boxlists = []
        for index, instance_dict in enumerate(instances):
            instance = instance_dict['instances']
            boxes = instance.pred_boxes.tensor
            scores = instance.scores
            labels = instance.pred_classes
            if filter:
                inds = scores > 0.2
                boxes = boxes[inds]
                scores = scores[inds]
                labels = labels[inds]

            boxlist = BoxList(boxes,
                              instance.image_size[::-1], mode="xyxy")
            boxlist.add_field("labels", labels)
            boxlist.add_field("scores", scores)
            boxlist.add_field("features", features[0][index])
            boxlists.append(boxlist)
        if len(boxlists) > max_dets:
            boxlists = boxlists[:max_dets]
        return boxlists

    def forward(self, images, targets=None, logger=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        # image_input = self.preprocess(images.tensors)
        image_input = [dict(image=image, im_info=images.image_sizes) for image in images.tensors]
        with torch.no_grad():
            features, proposals, detections = self.backbone.inference(image_input)
        detections_boxlist = self.instances_to_boxlist(detections, features, filter=False)
        # proposals, proposal_losses = self.rpn(images, features_3d, targets)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(features, detections_boxlist, targets, logger)
            box_len = [len(box) for box in result]
            split_x = torch.split(x, box_len, dim=0)
            for index, box in enumerate(result):
                box.add_field('relation_features', split_x[index])
        else:
            # RPN-only models don't have roi_heads
            x = features
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            # if not self.cfg.MODEL.RELATION_ON:
            #     # During the relationship training stage, the rpn_head should be fixed, and no loss.
            #     losses.update(proposal_losses)
            return losses

        return result

    def get_num_layer(self, var_name=""):
        if var_name in ("cls_token", "mask_token", "pos_embed"):
            return 0
        elif var_name.startswith("patch_embed"):
            return 0
        elif var_name.startswith("rel_pos_bias"):
            return len(self.backbone.backbone.net.blocks) - 1
        elif var_name.startswith("blocks"):
            layer_id = int(var_name.split('.')[1])
            return layer_id + 1
        else:
            return len(self.backbone.backbone.net.blocks)

    @staticmethod
    def process_result_to_features(result):
        backbone_features = torch.stack([box.get_field('features') for box in result])  # (batch_size, 256, 384, 384)
        features_chunk = torch.mean(backbone_features, dim=(2, 3))  # (batch_size, 256)
        relation_features = [box.get_field('relation_features') for box in result]  # (batch_size, n, 4096)
        relation_features = [torch.mean(relation_feature, dim=0) for relation_feature in relation_features]
        input_tensor = torch.stack(relation_features, dim=0)  # (batch_size, 4096)
        relation_chunk = F.avg_pool1d(input_tensor, kernel_size=16, stride=16).squeeze(-1)  # (batch_size, 256)
        # normalize
        features_chunk = F.normalize(features_chunk, dim=-1)
        relation_chunk = F.normalize(relation_chunk, dim=-1)
        features_chunk = torch.cat((features_chunk, relation_chunk), dim=-1)  # (batch_size, 512)
        return features_chunk
