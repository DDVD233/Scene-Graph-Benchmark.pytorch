# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import copy
import io
import itertools
import logging

import numpy as np
import os
from collections import OrderedDict, defaultdict
import torch
import detectron2.utils.comm as comm
from detectron2.utils.file_io import PathManager
from .coco_evaluation import COCOEvaluator
import pycocotools.mask as mask_util
from scipy.optimize import linear_sum_assignment
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.modeling.matcher import Matcher
from sklearn.metrics import average_precision_score


def xywh_to_xyxy(bbox):
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


def pad_with_zeros(rows):
    if len(rows) == 0:
        return []
    # Determine the maximum length of the rows
    max_length = max(len(row) for row in rows)

    # Create an array of zeros with the shape (number_of_rows, max_length)
    padded_array = np.zeros((len(rows), max_length))

    # Iterate over the rows and assign each row to the corresponding row in the array of zeros
    for i, row in enumerate(rows):
        padded_array[i, :len(row)] = row

    return padded_array


def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    results = []
    for k in range(num_instance):
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        if instances.has("pred_attributes"):
            result["attribute_id"] = instances.pred_attributes[k].item()
        if instances.has("pred_attributes_prob"):
            result["attribute_score"] = instances.pred_attributes_prob[k].tolist()
        results.append(result)
    return results


class COCOAttributeEvaluator(COCOEvaluator):
    """
    Evaluate COCO-style annotations using the official COCO API.
    It supports attribute evaluation.
    """

    def __init__(
            self,
            dataset_name,
            tasks=None,
            distributed=True,
            output_dir=None,
            *,
            max_dets_per_image=None,
            use_fast_impl=True,
            kpt_oks_sigmas=(),
            allow_cached_coco=True,
            use_custom_ranges=False,
    ):
        """

        @param dataset_name:
        @param tasks:
        @param distributed:
        @param output_dir:
        @param max_dets_per_image:
        @param use_fast_impl:
        @param kpt_oks_sigmas:
        @param allow_cached_coco:
        @param use_custom_ranges:
        """
        super().__init__(dataset_name, tasks, distributed, output_dir, max_dets_per_image=max_dets_per_image,
                         use_fast_impl=use_fast_impl, kpt_oks_sigmas=kpt_oks_sigmas,
                         allow_cached_coco=allow_cached_coco, use_custom_ranges=use_custom_ranges)
        self._logger = logging.getLogger(__name__)

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            if len(prediction) > 1:
                self._predictions.append(prediction)

    def evaluate(self, img_ids=None):
        """
        Args:
            img_ids: a list of image IDs to evaluate on. Default to None for the whole dataset
        """
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._eval_predictions(predictions, img_ids=img_ids)
        self._eval_attributes(predictions, img_ids=img_ids)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _eval_attributes(self, predictions, img_ids=None):
        """
        Args:
            predictions (list[dict]): the output of the model
            img_ids (list[int]): a list of image IDs to evaluate on. Default to None for the whole dataset
        """

        # filter predictions
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        cocoGt = self._coco_api
        cocoDt = cocoGt.loadRes(coco_results)

        total_attributes = defaultdict(int)
        correct_attributes = defaultdict(int)
        matcher = Matcher([0.7], labels=[-1, 1], allow_low_quality_matches=False)
        att_predictions = []
        att_classes = []
        att_scores = []
        att_gts = []

        for img_id in cocoGt.getImgIds():
            # Get ground truth annotations and detections for the current image
            gt_ann_ids = cocoGt.getAnnIds(imgIds=img_id)
            gt_anns = cocoGt.loadAnns(ids=gt_ann_ids)

            dt_ann_ids = cocoDt.getAnnIds(imgIds=img_id)
            dt_anns = cocoDt.loadAnns(ids=dt_ann_ids)

            # Match the ground truth and detections, xywh -> xyxy
            gt_boxes = [[ann['bbox'][0],
                         ann['bbox'][1],
                         ann['bbox'][0] + ann['bbox'][2],
                         ann['bbox'][1] + ann['bbox'][3]] for ann in gt_anns]
            dt_boxes = [[ann['bbox'][0],
                         ann['bbox'][1],
                         ann['bbox'][0] + ann['bbox'][2],
                         ann['bbox'][1] + ann['bbox'][3]] for ann in dt_anns]

            ious = pairwise_iou(Boxes(torch.tensor(gt_boxes)),
                                Boxes(torch.tensor(dt_boxes)))
            matched_idxs = matcher(ious)  # (N, )

            for index, dt_ann in enumerate(dt_anns):
                # Skip unmatched detections
                if matched_idxs[1][index] == -1:
                    continue
                gt_idx = matched_idxs[0][index]
                gt_ann = gt_anns[gt_idx]

                if gt_ann['category_id'] != dt_ann['category_id']:
                    continue

                # Calculate the number of correct attributes
                attr = gt_ann['attribute_id']
                if attr is None or attr == -1:
                    continue

                att_gts.append(attr)
                att_prediction = dt_ann['attribute_id'] + 1
                att_predictions.append(att_prediction)
                att_scores.append(dt_ann['attribute_score'])
                att_classes.append(gt_ann['category_id'])

                total_attributes[gt_ann['category_id']] += 1
                if attr == att_prediction:
                    correct_attributes[gt_ann['category_id']] += 1

        # Convert to numpy arrays
        att_scores = pad_with_zeros(att_scores)
        att_predictions = np.array(att_predictions)
        att_classes = np.array(att_classes)
        att_gts = np.array(att_gts)

        # Calculate the attribute accuracy per category
        attribute_accuracy = {}
        categories = cocoGt.loadCats(cocoGt.getCatIds())
        for cat_id in total_attributes:
            cat_name = categories[cat_id - 1]['name']
            attribute_accuracy[cat_name] = correct_attributes[cat_id] / total_attributes[cat_id]

        # Calculate the attribute accuracy overall
        if sum(total_attributes.values()) > 0:
            attribute_accuracy['overall'] = sum(correct_attributes.values()) / sum(total_attributes.values())

        # Calculate the AP per attribute per category
        attribute_ap = {}
        for cat_id in total_attributes.keys():
            # find attribute ids from the category
            attributes = categories[cat_id - 1]['attributes']
            attribute_ids = [attr['id'] for attr in attributes]
            attribute_names = [attr['name'] for attr in attributes]
            cat_name = categories[cat_id - 1]['name']

            cat_mask = att_classes == cat_id
            this_attr_scores = att_scores[cat_mask, :]
            this_attr_gts = att_gts[cat_mask]
            # For each attribute, calculate the AP if it has at least one ground truth
            for attr_id, attr_name in zip(attribute_ids, attribute_names):
                if attr_id in att_gts:
                    # filter out the predictions for the current attribute
                    attribute_ap[f'{cat_name}_{attr_name}'] = average_precision_score(
                        this_attr_gts == attr_id, this_attr_scores[:, attr_id - 1])

        self._logger.info(f'Attribute Accuracy: {attribute_accuracy}')
        self._logger.info(f'Attribute AP: {attribute_ap}')
