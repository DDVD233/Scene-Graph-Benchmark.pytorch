#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Training script using the new "LazyConfig" python config files.

This scripts reads a given python config file and runs the training or evaluation.
It can be used to train any models or dataset as long as they can be
instantiated by the recursive construction defined in the given config file.

Besides lazy construction of models, dataloader, etc., this scripts expects a
few common configuration parameters currently defined in "configs/common/train.py".
To add more complicated training logic, you can easily add other configs
in the config file and implement a new train_net.py to handle them.
"""
import logging
import os

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    AMPTrainer,
    SimpleTrainer,
    default_argument_parser,
    default_setup,
    default_writers,
    hooks,
    launch,
)
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import inference_on_dataset, print_csv_format
from detectron2.utils import comm

from detectron2.modeling import GeneralizedRCNNWithTTA, ema
import torch


logger = logging.getLogger("detectron2")


def do_test_with_tta(cfg, model):
    # may add normal test results for comparison
    if "evaluator" in cfg.dataloader:
        model = GeneralizedRCNNWithTTA(cfg, model, batch_size=1)
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)
        return ret


def do_test(cfg, model, eval_only=False):
    logger = logging.getLogger("detectron2")

    if eval_only:
        logger.info("Run evaluation under eval-only mode")
        if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
            logger.info("Run evaluation with EMA.")
        else:
            logger.info("Run evaluation without EMA.")
        if "evaluator" in cfg.dataloader:
            ret = inference_on_dataset(
                model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
            )
            print_csv_format(ret)
        return ret

    logger.info("Run evaluation without EMA.")
    if "evaluator" in cfg.dataloader:
        ret = inference_on_dataset(
            model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
        )
        print_csv_format(ret)

        if cfg.train.model_ema.enabled:
            logger.info("Run evaluation with EMA.")
            with ema.apply_model_ema_and_restore(model):
                if "evaluator" in cfg.dataloader:
                    ema_ret = inference_on_dataset(
                        model, instantiate(cfg.dataloader.test), instantiate(cfg.dataloader.evaluator)
                    )
                    print_csv_format(ema_ret)
                    ret.update(ema_ret)
        return ret


def do_train(args, cfg):
    """
    Args:
        cfg: an object with the following attributes:
            model: instantiate to a module
            dataloader.{train,test}: instantiate to dataloaders
            dataloader.evaluator: instantiate to evaluator for test set
            optimizer: instantaite to an optimizer
            lr_multiplier: instantiate to a fvcore scheduler
            train: other misc config defined in `configs/common/train.py`, including:
                output_dir (str)
                init_checkpoint (str)
                amp.enabled (bool)
                max_iter (int)
                eval_period, log_period (int)
                device (str)
                checkpointer (dict)
                ddp (dict)
    """
    model = instantiate(cfg.model)
    logger = logging.getLogger("detectron2")
    logger.info("Model:\n{}".format(model))
    model.to(cfg.train.device)

    cfg.optimizer.params.model = model
    lr = cfg.optimizer.lr
    optim = instantiate(cfg.optimizer)

    train_loader = instantiate(cfg.dataloader.train)

    model = create_ddp_model(model, **cfg.train.ddp)
    # build model ema
    ema.may_build_model_ema(cfg, model)

    trainer = (AMPTrainer if cfg.train.amp.enabled else SimpleTrainer)(model, train_loader, optim)
    checkpointer = DetectionCheckpointer(
        model,
        cfg.train.output_dir,
        trainer=trainer,
        # save model ema
        **ema.may_get_ema_checkpointer(cfg, model)
    )
    trainer.register_hooks(
        [
            hooks.IterationTimer(),
            ema.EMAHook(cfg, model) if cfg.train.model_ema.enabled else None,
            hooks.LRScheduler(scheduler=instantiate(cfg.lr_multiplier)),
            hooks.PeriodicCheckpointer(checkpointer, **cfg.train.checkpointer)
            if comm.is_main_process()
            else None,
            hooks.EvalHook(cfg.train.eval_period, lambda: do_test(cfg, model)),
            hooks.PeriodicWriter(
                default_writers(cfg.train.output_dir, cfg.train.max_iter,
                                use_wandb=args.wandb),
                period=cfg.train.log_period,
            )
            if comm.is_main_process()
            else None,
        ]
    )

    checkpointer.resume_or_load(cfg.train.init_checkpoint, resume=args.resume)
    if args.resume and checkpointer.has_checkpoint():
        # The checkpoint stores the training iteration that just finished, thus we start
        # at the next iteration
        start_iter = trainer.iter + 1
    else:
        start_iter = 0
    optim.lr = lr
    trainer.train(start_iter, cfg.train.max_iter)


def register_datasets(args, cfg):
    from detectron2.data.datasets import register_coco_instances
    from detectron2.data.datasets import load_coco_json
    from detectron2.data import MetadataCatalog
    from pycocotools.coco import COCO
    base_dir = args.base_dir
    train_dataset = cfg.dataloader.train.dataset.names
    test_dataset = cfg.dataloader.test.dataset.names
    det_dir = base_dir
    image_root = os.path.join(base_dir, 'VG_100K') if "vg" in train_dataset else base_dir

    mappings = {
        "vg_train": os.path.join(det_dir, "coco_train.json"),
        "vg_val": os.path.join(det_dir, "coco_test.json"),
        "moma_train": os.path.join(det_dir, "coco_train.json"),
        "moma_val": os.path.join(det_dir, "coco_test.json"),
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
    for instance in dataset_dicts:
        annotations = instance["annotations"]
        for annotation in annotations:
            cat_id = annotation["category_id"]
            class_image_count[cat_id-1]["image_count"] += 1
    MetadataCatalog.get(train_dataset).set(class_image_count=class_image_count)
    MetadataCatalog.get(train_dataset).set(mask_on=False)
    MetadataCatalog.get(test_dataset).set(mask_on=False)


def main(args):
    torch.autograd.set_detect_anomaly(True)
    cfg = LazyConfig.load(args.config_file)
    cfg = LazyConfig.apply_overrides(cfg, args.opts)
    default_setup(cfg, args)
    register_datasets(args, cfg)

    if args.eval_only:
        model = instantiate(cfg.model)
        model.to(cfg.train.device)
        model = create_ddp_model(model)

        # using ema for evaluation
        ema.may_build_model_ema(cfg, model)
        DetectionCheckpointer(model, **ema.may_get_ema_checkpointer(cfg, model)).load(cfg.train.init_checkpoint)
        # Apply ema state for evaluation
        if cfg.train.model_ema.enabled and cfg.train.model_ema.use_ema_weights_for_eval_only:
            ema.apply_model_ema(model)
        print(do_test(cfg, model, eval_only=True))
    else:
        do_train(args, cfg)


if __name__ == "__main__":
    argparser = default_argument_parser()
    argparser.add_argument("--debug", action="store_true")
    argparser.add_argument("--base_dir", type=str, default="/home/data/cba/")
    args = argparser.parse_args()
    if args.debug:
        main(args)
    else:
        launch(
            main,
            args.num_gpus,
            num_machines=args.num_machines,
            machine_rank=args.machine_rank,
            dist_url=args.dist_url,
            args=(args,),
        )
