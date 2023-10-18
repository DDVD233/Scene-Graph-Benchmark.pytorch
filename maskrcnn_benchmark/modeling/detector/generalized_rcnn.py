# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from maskrcnn_benchmark.structures.image_list import to_image_list

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..backbone.eva_vit import create_eva_vit_g
from ..roi_heads.roi_heads import build_roi_heads
from ..preprocessing import Preprocessing
from maskrcnn_benchmark.structures.bounding_box import BoxList
import torch.nn.functional as F
import time


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
        self.patch_backbone = create_eva_vit_g()
        shapes = self.backbone.backbone.output_shape()
        out_channels = list(shapes.values())[0].channels
        self.roi_heads = build_roi_heads(cfg, out_channels)
        self.preprocess = Preprocessing()
        self.num_features = 1408
        self.pooling = nn.AdaptiveAvgPool1d(1408)

    def instances_to_boxlist(self, instances, features, patch_features, filter=True, max_dets=20):
        """
        Convert a list of detectron2 Instances to a list of BoxList

        Args:
            instances (list[Instances]): a list of detectron2 Instances
            filter (bool): filter out instances with score < 0.2
        """
        feature = features[-1]
        assert len(feature) == len(instances)
        boxlists = []
        for index, instance_dict in enumerate(instances):
            instance = instance_dict['instances']
            boxes = instance.pred_boxes.tensor
            scores = instance.scores
            labels = instance.pred_classes
            if len(boxes) == 0:  # We need to add a dummy entry for the batch size
                boxes = torch.zeros((1, 4), device=boxes.device)
                scores = torch.zeros((1,), device=scores.device)
                labels = torch.zeros((1,), device=labels.device)
            if filter:
                inds = scores > 0.2
                boxes = boxes[inds]
                scores = scores[inds]
                labels = labels[inds]

            boxlist = BoxList(boxes,
                              instance.image_size[::-1], mode="xyxy")
            boxlist.add_field("labels", labels)
            boxlist.add_field("scores", scores)
            boxlist.add_field("features", feature[index])
            boxlist.add_field("patch_features", patch_features[index])
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
        start_time = time.time()

        images = to_image_list(images)
        # image_input = self.preprocess(images.tensors)
        image_input = [dict(image=image, im_info=images.image_sizes) for image in images.tensors]
        with torch.no_grad():
            features, proposals, detections = self.backbone.inference(image_input)
            # Crop & resize image tensors to bsx224x224 before feeding into patch_backbone
            cropped_image = F.interpolate(images.tensors, size=(224, 224),
                                          mode='bilinear', align_corners=False)
            patch_features = self.patch_backbone(cropped_image)
        detections_boxlist = self.instances_to_boxlist(detections, features, patch_features, filter=False)
        assert len(detections_boxlist) == len(images.tensors)
        x, result, detector_losses = self.roi_heads(features, detections_boxlist, targets, logger)
        assert len(result) == len(images.tensors)
        box_len = [len(box) for box in result]
        split_x = torch.split(x, box_len, dim=0)
        for index, box in enumerate(result):
            box.add_field('relation_features', split_x[index])

        if self.training:
            losses = {}
            losses.update(detector_losses)
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
    def process_relation_features(relation_tensor):
        """
        Extract the first 4 lines from the relation tensor if n >= 4,
        else: zero pad the tensor to 4 lines using nn.functional.pad

        @type relation_tensor: Tensor of shape (n, 4096)
        """

        if relation_tensor.shape[0] >= 4:
            return relation_tensor[:4, :]
        else:
            pad_length = 4 - relation_tensor.shape[0]
            # (padding_left, padding_right, padding_top, padding_bottom)
            pad = nn.functional.pad(relation_tensor, (0, 0, 0, pad_length))
            return pad
    
    def process_result_to_features(self, result):
        backbone_features = torch.stack([box.get_field('features') for box in result])  # (batch_size, 256, 24, 24)

        patch_backbone_result = torch.stack([box.get_field('patch_features') for box in result]) # (batch_size, 257, 1408)
        features_chunk = torch.reshape(backbone_features, (backbone_features.shape[0], -1))  # (batch_size, 256 * 24 * 24)
        features_chunk = self.pooling(features_chunk).unsqueeze(1)  # (batch_size, 1, 1408)

        relation_features = [self.process_relation_features(box.get_field('relation_features'))
                             for box in result]  # (batch_size, 4, 4096)
        relation_chunk = torch.stack(relation_features, dim=0)  # (batch_size, 4, 4096)
        relation_chunk = self.pooling(relation_chunk)  # (batch_size, 4, 1408)
        # normalize
        # features_chunk = F.normalize(features_chunk, dim=-1)
        # relation_chunk = F.normalize(relation_chunk, dim=-1)
        final_chunk = torch.cat((patch_backbone_result, features_chunk, relation_chunk), dim=1)  # (batch_size, 262, 1408)
        return final_chunk
