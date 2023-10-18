from detectron2.config import LazyCall as L
from detectron2.data.samplers import RepeatFactorTrainingSampler
from detectron2.data.detection_utils import get_fed_loss_cls_weights

from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads import (
    FastRCNNOutputLayers,
    FastRCNNConvFCHead,
    CascadeROIHeads,
)
from detectron2.evaluation import COCOEvaluator


from ..eva2_o365_to_coco.eva2_o365_to_coco_cascade_mask_rcnn_vitdet_l_8attn_1536_lrd0p8 import (
    dataloader,
    model,
    train,
    lr_multiplier,
    optimizer,
)

dataloader.train.dataset.names = "moma_train"
dataloader.train.mapper.recompute_boxes = False
dataloader.train.sampler = L(RepeatFactorTrainingSampler)(
    repeat_factors=L(
        RepeatFactorTrainingSampler.repeat_factors_from_category_frequency
    )(dataset_dicts="${dataloader.train.dataset}", repeat_thresh=0.001)
)
dataloader.test.dataset.names = "moma_val"
dataloader.evaluator = L(COCOEvaluator)(
    dataset_name="${..test.dataset.names}",
)


model.roi_heads.update(
    _target_=CascadeROIHeads,
    num_classes=253,
    mask_in_features=None,
    box_heads=[
        L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[256, 256, 256, 256],
            fc_dims=[1024],
            conv_norm="LN",
        )
        for _ in range(3)
    ],
    box_predictors=[
        L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=1024),
            box2box_transform=L(Box2BoxTransform)(weights=(w1, w1, w2, w2)),
            num_classes="${...num_classes}",
            test_score_thresh=0.02,
            test_topk_per_image=300,
            cls_agnostic_bbox_reg=True,
            use_sigmoid_ce=True,
            use_fed_loss=True,
            get_fed_loss_cls_weights=lambda: get_fed_loss_cls_weights(
                dataloader.train.dataset.names, 0.5
            ),
        )
        for (w1, w2) in [(10, 5), (20, 10), (30, 15)]
    ],
    proposal_matchers=[
        L(Matcher)(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False)
        for th in [0.5, 0.6, 0.7]
    ],
)

dataloader.test.num_workers = 0
dataloader.train.total_batch_size = 24

optimizer.lr = 1e-6
train.model_ema.enabled = False

# Phase 2: 50 ep = 42836 iters * 32 images/iter / 27415 images/ep
# Phase 2 full: 50 ep = 42836 iters * 32 images/iter / 70000 images/ep
train.max_iter = 50000
train.eval_period = 5000
train.checkpointer.period = 1000
