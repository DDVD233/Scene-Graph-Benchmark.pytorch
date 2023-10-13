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
