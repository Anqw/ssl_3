# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import torch
from torch import nn
import numpy as np
import math

from fs3c.structures import ImageList
from fs3c.utils.logger import log_first_n

from ..backbone import build_backbone
from ..postprocessing import detector_postprocess
from ..proposal_generator import build_proposal_generator
from ..roi_heads import build_roi_heads, build_jig_heads, build_rot_heads, build_con_heads
from .build import META_ARCH_REGISTRY
import torchvision.transforms as transforms

__all__ = ["GeneralizedRCNN", "ProposalNetwork"]


@META_ARCH_REGISTRY.register()
class GeneralizedRCNN(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.jig_heads = build_jig_heads(cfg, self.backbone.output_shape())
        self.rot_heads = build_rot_heads(cfg, self.backbone.output_shape())
        self.con_heads = build_con_heads(cfg, self.backbone.output_shape())
        #self.in_features = cfg.MODEL.ROI_HEADS.IN_FEATURES

        self.jigsaw = cfg.JIG
        self.rotation = cfg.ROT
        self.contrastive = cfg.CON
        self.ssl = cfg.SSL

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        num_channels = len(cfg.MODEL.PIXEL_MEAN)
        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(num_channels, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(num_channels, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

        if cfg.MODEL.BACKBONE.FREEZE:
            for p in self.backbone.parameters():
                p.requires_grad = False
            print('froze backbone parameters')

        if cfg.MODEL.PROPOSAL_GENERATOR.FREEZE:
            for p in self.proposal_generator.parameters():
                p.requires_grad = False
            print('froze proposal generator parameters')

        if cfg.MODEL.ROI_HEADS.FREEZE_FEAT:
            for p in self.roi_heads.box_head.parameters():
                p.requires_grad = False
            print('froze roi_box_head parameters')

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                    "pred_boxes", "pred_classes", "scores"
        """
        if not self.training:
            return self.inference(batched_inputs[0])

        images = self.preprocess_image(batched_inputs[0])
        if "instances" in batched_inputs[0][0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs[0]]
        elif "targets" in batched_inputs[0][0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs[0]]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        ssl_rot_losses = 0
        ssl_jig_losses = 0
        ssl_con_losses = 0
        if self.jigsaw:
            patches_list = []
            labels_list = []
            for x in batched_inputs[1]:
                patch, label = x["jig"]
                patches_list.append(patch.to(self.device))
                labels_list.append(label)
            labels = torch.Tensor(labels_list).long().to(self.device)
            patches = ImageList.from_tensors(patches_list, self.backbone.size_divisibility)
            patches = patches.tensor
            B, T, C, H, W = patches.size()
            patches = patches.view(B*T, C, H, W)
            features_ssl = self.backbone(patches)
            features_ssl = features_ssl["p6"]
            features_ssl = torch.flatten(features_ssl, start_dim=1).view(B, T, -1)
            features_ssl = features_ssl.transpose(0, 1)
            image = None
            proposal = None
            ssl_jig_losses = self.jig_heads(image, features_ssl, proposal, labels)

        if self.rotation:
            patches_list = []
            labels_list = []
            for x in batched_inputs[1]:
                patch, label = x["rot"]
                patches_list.append(patch.to(self.device))
                labels_list.append(label.to(self.device))
            labels = torch.cat(labels_list)
            patches = ImageList.from_tensors(patches_list, self.backbone.size_divisibility)
            patches = patches.tensor
            B, R, C, H, W = patches.size()
            patches = patches.view(B * R, C, H, W)
            features_ssl = self.backbone(patches)
            features_ssl = features_ssl["p6"]
            image = None
            proposal = None
            ssl_rot_losses = self.rot_heads(image, features_ssl, proposal, labels)

        if self.contrastive:
            patches_1 = []
            patches_2 = []
            coord_1 = []
            coord_2 = []
            for x in batched_inputs[1]:
                patch1, patch2, coord1, coord2 = x["con"]
                patches_1.append(patch1.to(self.device))
                patches_2.append(patch2.to(self.device))
                coord_1.append(coord1.to(self.device))
                coord_2.append(coord2.to(self.device))
            labels = [torch.stack(coord_1, dim=0), torch.stack(coord_2, dim=0)]
            patches_1 = ImageList.from_tensors(patches_1, self.backbone.size_divisibility)
            patches_1 = patches_1.tensor
            patches_2 = ImageList.from_tensors(patches_2, self.backbone.size_divisibility)
            patches_2 = patches_2.tensor

            features_1 = self.backbone(patches_1)
            features_1 = features_1["p5"]
            features_2 = self.backbone(patches_2)
            features_2 = features_2["p5"]

            features_ssl = [features_1, features_2]

            image = None
            proposal = None
            ssl_con_losses = self.con_heads(image, features_ssl, proposal, labels)
        if self.ssl:
            ssl_losses = {"ssl_rot_losses": 0.4 * ssl_rot_losses,
                          "ssl_jig_losses": 0.3 * ssl_jig_losses,
                          "ssl_con_losses": 0.3 * ssl_con_losses
                          }

        if self.proposal_generator:
            proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        else:
            assert "proposals" in batched_inputs[0][0]
            proposals = [x["proposals"].to(self.device) for x in batched_inputs[0]]
            proposal_losses = {}

        _, detector_losses = self.roi_heads(images, features, proposals, gt_instances)

        losses = {}
        losses.update(detector_losses)
        if self.ssl:
            losses.update(ssl_losses)
        losses.update(proposal_losses)
        return losses

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if detected_instances is None:
            if self.proposal_generator:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes
            ):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


@META_ARCH_REGISTRY.register()
class ProposalNetwork(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std
        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        # In training, the proposals are not useful at all but we generate them anyway.
        # This makes RPN-only models about 5% slower.
        if self.training:
            return proposal_losses

        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            proposals, batched_inputs, images.image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"proposals": r})
        return processed_results
