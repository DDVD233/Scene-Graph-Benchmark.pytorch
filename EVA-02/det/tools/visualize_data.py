#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
import argparse
import os
from itertools import chain
import cv2
import tqdm

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.config import LazyConfig, instantiate


def setup(args):
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    # cfg = get_cfg()
    # if args.config_file:
    #     cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    # cfg.DATALOADER.NUM_WORKERS = 0
    # cfg.freeze()
    return cfg


def register_datasets(args, cfg):
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data.datasets import load_coco_json
    from detectron2.data import MetadataCatalog
    from pycocotools.coco import COCO
    base_dir = args.base_dir
    train_dataset = cfg.dataloader.train.dataset.names
    test_dataset = cfg.dataloader.test.dataset.names
    det_dir = base_dir
    image_root = os.path.join(base_dir, 'VG_100K')

    mappings = {
        "vg_train": os.path.join(det_dir, "coco_train.json"),
        "vg_val": os.path.join(det_dir, "coco_test.json"),
    }
    train_file_path = mappings[train_dataset]
    register_coco_instances(train_dataset, {}, train_file_path, image_root)
    test_file_path = mappings[test_dataset]
    register_coco_instances(test_dataset, {}, test_file_path, image_root)
    # # Calculate class_image_count for cba_p1_train
    try:
        dataset_dicts = load_coco_json(train_file_path, base_dir, train_dataset)
    except:
        print(f'Failed to load {train_dataset} with {train_file_path}')
        return
    thing_classes = MetadataCatalog.get(train_dataset).thing_classes
    class_image_count = [{"id": i, "image_count": 0, "name": thing_classes[i-1]}
                         for i in range(1, len(thing_classes) + 1)]
    # for instance in dataset_dicts:
    #     annotations = instance["annotations"]
    #     for annotation in annotations:
    #         cat_id = annotation["category_id"]
    #         class_image_count[cat_id-1]["image_count"] += 1
    # MetadataCatalog.get(train_dataset).set(class_image_count=class_image_count)
    MetadataCatalog.get(train_dataset).set(mask_on=False)
    MetadataCatalog.get(test_dataset).set(mask_on=False)


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    parser.add_argument(
        "--source",
        choices=["annotation", "dataloader"],
        required=True,
        help="visualize the annotations or the data loader (with pre-processing)",
    )
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument("--output-dir", default="./", help="path to output directory")
    parser.add_argument("--show", action="store_true", help="show output in a window")
    parser.add_argument("--base_dir", type=str, default="/home/data/datasets/vg/")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(in_args)


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup(args)

    dirname = args.output_dir
    os.makedirs(dirname, exist_ok=True)
    metadata = MetadataCatalog.get(cfg.dataloader.train.dataset.names)
    register_datasets(args, cfg)

    def output(vis, fname):
        if args.show:
            print(fname)
            cv2.imshow("window", vis.get_image()[:, :, ::-1])
            cv2.waitKey()
        else:
            filepath = os.path.join(dirname, fname)
            print("Saving to {} ...".format(filepath))
            vis.save(filepath)

    scale = 1.0
    if args.source == "dataloader":
        train_data_loader = build_detection_train_loader(cfg)
        for batch in train_data_loader:
            for per_image in batch:
                # Pytorch tensor is in (C, H, W) format
                img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
                img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)

                visualizer = Visualizer(img, metadata=metadata, scale=scale)
                target_fields = per_image["instances"].get_fields()
                labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
                vis = visualizer.overlay_instances(
                    labels=labels,
                    boxes=target_fields.get("gt_boxes", None),
                    masks=target_fields.get("gt_masks", None),
                    keypoints=target_fields.get("gt_keypoints", None),
                )
                output(vis, str(per_image["image_id"]) + ".jpg")
    else:
        dicts = list(chain.from_iterable([DatasetCatalog.get(cfg.dataloader.test.dataset.names)]))
        # if cfg.MODEL.KEYPOINT_ON:
        #     dicts = filter_images_with_few_keypoints(dicts, 1)
        for dic in tqdm.tqdm(dicts):
            img = utils.read_image(dic["file_name"], "RGB")
            visualizer = Visualizer(img, metadata=metadata, scale=scale)
            vis = visualizer.draw_dataset_dict(dic)
            output(vis, os.path.basename(dic["file_name"]))
