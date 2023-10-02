import torch
from torch import nn
from torch.nn import functional as F
from detectron2.layers import Conv2d, ShapeSpec
from detectron2.structures import Instances
from typing import List
from logging import getLogger


class AttributeHead(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, num_subclasses: List[int]):
        super().__init__()
        self.num_classes = num_classes
        self.num_subclasses = num_subclasses
        # self.hidden_dim = 256

        self.classifiers = nn.ModuleList()
        self.mapping = {}  # map class index to classifier index
        for index, num_subclass in enumerate(num_subclasses):
            if num_subclass > 1:
                self.classifiers.append(nn.Linear(input_dim, num_subclass))
                self.mapping[len(self.classifiers) - 1] = index  # map classifier index to class index

        class_counts = [
            [78061, 32802],
            [812, 86047, 20450, 2335],
            [9571, 113936, 25109]
        ]

        class_weights = [
            torch.tensor([1.0 / class_counts[0][i] for i in range(len(class_counts[0]))]).cuda(),
            torch.tensor([1.0 / class_counts[1][i] for i in range(len(class_counts[1]))]).cuda(),
            torch.tensor([1.0 / class_counts[2][i] for i in range(len(class_counts[2]))]).cuda(),
        ]

        # normalize weights
        for i in range(len(class_weights)):
            class_weights[i] = class_weights[i] / class_weights[i].sum()

        self.class_weights = class_weights

        # nn.init.normal_(self.mlp[0].weight, std=0.01)
        # nn.init.constant_(self.mlp[0].bias, 0)
        # nn.init.normal_(self.mlp[2].weight, std=0.01)
        # nn.init.constant_(self.mlp[2].bias, 0)

        for classifier in self.classifiers:
            if isinstance(classifier, nn.Linear):
                nn.init.normal_(classifier.weight, std=0.01)
                nn.init.constant_(classifier.bias, 0)

        self.logger = getLogger(__name__)

    def forward(self, x, instances: List[Instances]):
        """
        Args:
            x: input region feature(s) provided by :class:`ROIHeads`.
            instances (list[Instances]): contains the boxes & labels corresponding
                to the input features.
                Exact format is up to its caller to decide.
                Typically, this is the foreground instances in training, with
                "proposal_boxes" field and other gt annotations.
                In inference, it contains boxes that are already predicted.

        Returns:
            A dict of losses in training. The predicted "instances" in inference.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        # attribute_logits = self.mlp(x)  # (N, hidden_dim)
        out = []
        for classifier in self.classifiers:
            out.append(classifier(x))  # (num_classifiers, N, num_subclasses)  (not rectangular)

        if self.training:
            # Assuming `gt_attributes` are provided in `instances` during training
            gt_attributes, gt_classes = [], []
            for instance in instances:
                if hasattr(instance, "gt_attributes"):
                    gt_attributes.extend(instance.gt_attributes)
                if hasattr(instance, "gt_classes"):
                    gt_classes.extend(instance.gt_classes)
                elif hasattr(instance, "pred_classes"):
                    gt_classes.extend(instance.pred_classes)
                else:
                    raise ValueError("Instance does not have gt_classes or pred_classes")
            gt_attributes = torch.tensor(gt_attributes).to(out[0].device)
            gt_classes = torch.tensor(gt_classes)
            losses = 0
            for index in range(len(self.classifiers)):
                # Assign label to -1 if gt_class != self.mapping[index]
                classifier_gt_attributes = gt_attributes.clone()
                classifier_gt_classes = gt_classes.clone()
                out_dim = out[index].shape[0]
                if classifier_gt_attributes.shape[0] != gt_classes.shape[0]:
                    self.logger.warning(f"gt_attributes and gt_classes have different length: "
                                        f"{classifier_gt_attributes.shape[0]} and {gt_classes.shape[0]}")
                    classifier_gt_classes = classifier_gt_classes[:classifier_gt_attributes.shape[0]]
                classifier_gt_attributes[classifier_gt_classes != self.mapping[index]] = -1
                classifier_gt_attributes = classifier_gt_attributes.to(out[index].device)

                loss = F.cross_entropy(out[index],
                                       classifier_gt_attributes[:out_dim],
                                       reduction="mean", ignore_index=-1, weight=self.class_weights[index])
                if torch.isnan(loss) or torch.isinf(loss):  # is nan or is zero
                    continue
                if losses == 0:
                    losses = loss
                else:
                    losses += loss
            if losses == 0:  # no loss, create a dummy loss
                losses = torch.tensor(0, dtype=torch.float32).to(out[0].device)
                # Loop through self parameters
            for param in self.parameters():  # Prevents error when no loss is created
                losses += param.sum() * 0
            return {"loss_attribute": losses / len(self.classifiers)}
        else:
            start_index = 0
            for instance in instances:
                size = len(instance)
                instance.pred_attributes_prob = torch.zeros((size, max(self.num_subclasses))).to(out[0].device)
                instance.pred_attributes = - torch.ones((size,), dtype=torch.long).to(out[0].device)
                for index in range(len(self.classifiers)):
                    cls_output_size = out[index].size(1)
                    out_target = out[index][start_index:start_index + size][instance.pred_classes == self.mapping[index]]
                    probs = F.softmax(out_target, dim=1)
                    predictions = torch.argmax(probs, dim=1)
                    instance.pred_attributes_prob[instance.pred_classes == self.mapping[index], :cls_output_size] = probs
                    instance.pred_attributes[instance.pred_classes == self.mapping[index]] = predictions
                start_index += size
            return instances
