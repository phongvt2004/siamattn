# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops as ops

import numpy as np
from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss, weight_l1_loss, mask_loss_bce, det_loss_smooth_l1
from pysot.models.backbone import get_backbone
from pysot.models.head import get_rpn_head, get_mask_head, get_refine_head
from pysot.models.neck import get_neck
from pysot.models.neck.feature_fusion import FeatureFusionNeck
from pysot.models.neck.enhance import FeatureEnhance
from pysot.models.multi_template_fusion import MultiTemplateFusion

from pysot.models.head.mask import FusedSemanticHead
from pysot.models.head.detection import FCx2DetHead
from pysot.utils.mask_target_builder import _build_proposal_target, _build_mask_target, _convert_loc_to_bbox


class CrossViewModelBuilder(nn.Module):
    """
    Model Builder cho Cross-View Few-Shot Object Detection
    - Input: 3 templates (ground images) + 1 search image (drone frame)
    - Sử dụng MultiTemplateFusion để fuse 3 templates
    """
    def __init__(self, fusion_method='attention'):
        super(CrossViewModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            # Get in_channels and out_channels from config or use defaults
            # AdjustAllLayer needs: in_channels (list) and out_channels (list)
            # ResNet-50 output: [x_(64), p1(256), p2(512), p3(1024), p4(2048)] for used_layers=[0,1,2,3,4]
            # Neck processes zf[2:], so we need channels for layers from index 2 onwards
            adjust_kwargs = {}
            if hasattr(cfg.ADJUST.KWARGS, '__dict__'):
                adjust_kwargs = dict(cfg.ADJUST.KWARGS)
            elif isinstance(cfg.ADJUST.KWARGS, dict):
                adjust_kwargs = cfg.ADJUST.KWARGS.copy()
            
            # Default values based on ResNet-50
            # For used_layers=[0,1,2,3,4], zf[2:] = [p2(512), p3(1024), p4(2048)]
            # Adjust all to 256
            if 'in_channels' not in adjust_kwargs:
                # ResNet-50 channel sizes: [64, 256, 512, 1024, 2048] for [x_, p1, p2, p3, p4]
                # Neck processes from index 2, so [512, 1024, 2048]
                adjust_kwargs['in_channels'] = [512, 1024, 2048]
            if 'out_channels' not in adjust_kwargs:
                # Default: adjust all to 256
                num_layers = len(adjust_kwargs.get('in_channels', [512, 1024, 2048]))
                adjust_kwargs['out_channels'] = [256] * num_layers
            
            self.neck = get_neck(cfg.ADJUST.TYPE, **adjust_kwargs)

        # build multi-template fusion
        fusion_method = getattr(cfg.MODEL, 'FUSION_METHOD', fusion_method) if hasattr(cfg, 'MODEL') else fusion_method
        num_templates = getattr(cfg.MODEL, 'NUM_TEMPLATES', 3) if hasattr(cfg, 'MODEL') else 3
        
        self.multi_template_fusion = MultiTemplateFusion(
            in_channels=256,
            fusion_method=fusion_method,
            num_templates=num_templates
        )

        # build rpn head
        # Get anchor_num and in_channels from config or use defaults
        rpn_kwargs = {}
        if hasattr(cfg.RPN.KWARGS, '__dict__'):
            rpn_kwargs = dict(cfg.RPN.KWARGS)
        elif isinstance(cfg.RPN.KWARGS, dict):
            rpn_kwargs = cfg.RPN.KWARGS.copy()
        
        # Default values
        if 'anchor_num' not in rpn_kwargs:
            rpn_kwargs['anchor_num'] = cfg.ANCHOR.ANCHOR_NUM
        if 'in_channels' not in rpn_kwargs:
            # MultiRPN needs list of in_channels for each layer
            # Based on neck output: [256, 256, 256] for layers 2,3,4
            rpn_kwargs['in_channels'] = [256, 256, 256]
        
        self.rpn_head = get_rpn_head(cfg.RPN.TYPE, **rpn_kwargs)

        # build mask head
        if cfg.MASK.MASK:
            self.feature_enhance = FeatureEnhance(in_channels=256, out_channels=256)
            self.feature_fusion = FeatureFusionNeck(num_ins=5, fusion_level=1,
                                                    in_channels=[64, 256, 256, 256, 256], conv_out_channels=256)
            self.mask_head = FusedSemanticHead(pooling_func=None,
                                               num_convs=4, in_channels=256,
                                               upsample_ratio=(cfg.MASK.MASK_OUTSIZE // cfg.TRAIN.ROIPOOL_OUTSIZE))
            self.bbox_head = FCx2DetHead(pooling_func=None,
                                         in_channels=256 * (cfg.TRAIN.ROIPOOL_OUTSIZE // 4)**2)

    def template(self, z_list):
        """
        Set templates (for inference)
        Args:
            z_list: List of 3 template images [B, C, H, W] each
        """
        with torch.no_grad():
            # Extract features for each template
            zf_list = []
            for z in z_list:
                zf = self.backbone(z)
                if cfg.ADJUST.ADJUST:
                    # Adjust neck for levels 2,3,4
                    zf_adjusted = self.neck(zf[2:])
                    # Reconstruct full feature list
                    if isinstance(zf, (list, tuple)):
                        zf = list(zf[:2]) + list(zf_adjusted)
                    else:
                        zf = [zf] + list(zf_adjusted) if isinstance(zf_adjusted, (list, tuple)) else [zf, zf_adjusted]
                zf_list.append(zf)
            
            # Fuse templates
            zf_fused = self.multi_template_fusion(zf_list)
            self.zf = zf_fused

    def track(self, x):
        """
        Track object in search image
        Args:
            x: Search image [B, C, H, W]
        """
        with torch.no_grad():
            xf = self.backbone(x)
            if cfg.ADJUST.ADJUST:
                # Adjust neck for levels 2,3,4
                xf_adjusted = self.neck(xf[2:])
                # Reconstruct full feature list
                if isinstance(xf, (list, tuple)):
                    xf = list(xf[:2]) + list(xf_adjusted)
                else:
                    xf = [xf] + list(xf_adjusted) if isinstance(xf_adjusted, (list, tuple)) else [xf, xf_adjusted]

            # Ensure self.zf and xf are lists
            if not isinstance(self.zf, (list, tuple)):
                zf_list = [self.zf]
            else:
                zf_list = self.zf
            if not isinstance(xf, (list, tuple)):
                xf_list = [xf]
            else:
                xf_list = xf

            # FeatureEnhance expects list of features (levels 2,3,4)
            zf_enhanced_list, xf_enhanced_list = self.feature_enhance(zf_list[2:], xf_list[2:])
            cls, loc = self.rpn_head(zf_enhanced_list, xf_enhanced_list)
            
            # Combine for mask head if needed
            if cfg.MASK.MASK:
                # Ensure both are lists for FeatureFusionNeck
                # enhanced_zf should be [level0, level1, level2_enhanced, level3_enhanced, level4_enhanced]
                if isinstance(self.zf, (list, tuple)):
                    enhanced_zf = list(self.zf[:2]) + list(zf_enhanced_list)
                else:
                    enhanced_zf = [self.zf] + list(zf_enhanced_list) if isinstance(zf_enhanced_list, (list, tuple)) else [self.zf, zf_enhanced_list]
                
                if not isinstance(xf, (list, tuple)):
                    xf_list = [xf]
                else:
                    xf_list = xf
                
                self.b_fused_features, self.m_fused_features = self.feature_fusion(enhanced_zf, xf_list)
            return {
                'cls': cls,
                'loc': loc
            }

    def mask_refine(self, roi):
        with torch.no_grad():
            mask_pred = self.mask_head(self.m_fused_features, roi)
        return mask_pred

    def bbox_refine(self, roi):
        with torch.no_grad():
            bbox_pred = self.bbox_head(self.b_fused_features, roi)
        return bbox_pred

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2 // 2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """
        Forward pass for training
        Args:
            data: Dict containing:
                - templates: [B, 3, C, H, W] - 3 template images
                - search: [B, C, H, W] - search image
                - label_cls: Classification labels
                - label_loc: Location labels
                - label_loc_weight: Location label weights
                - bbox: Ground truth bbox
                - search_mask: (optional) Mask for search image
                - mask_weight: (optional) Mask weights
                - bbox_weight: (optional) Bbox weights
        """
        templates = data['templates'].cuda()  # [B, 3, C, H, W]
        search = data['search'].cuda()  # [B, C, H, W]
        label_cls = data['label_cls'].cuda()
        label_loc = data['label_loc'].cuda()
        label_loc_weight = data['label_loc_weight'].cuda()
        gt_bboxes = data['bbox'].cuda()

        # Split templates: [B, 3, C, H, W] -> list of [B, C, H, W]
        batch_size = templates.size(0)
        template_list = [templates[:, i, :, :, :] for i in range(3)]

        # Extract features for each template
        zf_list = []
        for template in template_list:
            zf = self.backbone(template)
            if cfg.ADJUST.ADJUST:
                # Adjust neck for levels 2,3,4
                zf_adjusted = self.neck(zf[2:])
                # Reconstruct full feature list
                if isinstance(zf, (list, tuple)):
                    zf = list(zf[:2]) + list(zf_adjusted)
                else:
                    zf = [zf] + list(zf_adjusted) if isinstance(zf_adjusted, (list, tuple)) else [zf, zf_adjusted]
            zf_list.append(zf)

        # Fuse multi-template features
        # Only fuse levels 2,3,4 (after neck adjustment, all have 256 channels)
        # Keep levels 0,1 unchanged or use simple fusion
        if isinstance(zf_list[0], (list, tuple)):
            # Multi-level features
            zf_fused = []
            # Keep first 2 levels unchanged (or use simple mean fusion)
            for level_idx in range(2):
                level_features = [zf[level_idx] for zf in zf_list]
                # Simple mean fusion for early levels
                zf_fused.append(torch.stack(level_features, dim=0).mean(dim=0))
            
            # Fuse levels 2,3,4 (after neck, all 256 channels)
            levels_to_fuse = [zf[2:] for zf in zf_list]  # Extract levels 2,3,4
            fused_levels = self.multi_template_fusion(levels_to_fuse)
            zf_fused.extend(fused_levels)
        else:
            # Single level - should not happen with ResNet
            zf_fused = self.multi_template_fusion(zf_list)

        # Extract search features
        xf = self.backbone(search)
        if cfg.ADJUST.ADJUST:
            # Adjust neck for levels 2,3,4
            xf_adjusted = self.neck(xf[2:])
            # Reconstruct full feature list
            if isinstance(xf, (list, tuple)):
                xf = list(xf[:2]) + list(xf_adjusted)
            else:
                xf = [xf] + list(xf_adjusted) if isinstance(xf_adjusted, (list, tuple)) else [xf, xf_adjusted]

        # Feature enhancement với deformable attention
        # Ensure zf_fused and xf are lists
        if not isinstance(zf_fused, (list, tuple)):
            zf_fused = [zf_fused]
        if not isinstance(xf, (list, tuple)):
            xf = [xf]
        
        zf_enhanced, xf_enhanced = self.feature_enhance(zf_fused[2:], xf[2:])
        
        # RPN
        cls, loc = self.rpn_head(zf_enhanced, xf_enhanced)

        # Get loss
        cls_sm = self.log_softmax(cls)
        cls_loss = select_cross_entropy_loss(cls_sm, label_cls)
        loc_loss = weight_l1_loss(loc, label_loc, label_loc_weight)

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
                                cfg.TRAIN.LOC_WEIGHT * loc_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        if cfg.MASK.MASK:
            # Get optional mask data (nếu có)
            search_mask = data.get('search_mask', None)
            mask_weight = data.get('mask_weight', None)
            bbox_weight = data.get('bbox_weight', None)
            
            # Nếu không có mask data, skip mask loss
            if search_mask is not None and mask_weight is not None:
                search_mask = search_mask.cuda()
                mask_weight = mask_weight.cuda() if mask_weight is not None else None
                bbox_weight = bbox_weight.cuda() if bbox_weight is not None else None
                
                # Convert loc coordinate to (x1,y1,x2,y2)
                loc = loc.detach()
                bbox = _convert_loc_to_bbox(loc)
                rois, cls_ind, regression_target = _build_proposal_target(bbox, gt_bboxes)
                mask_targets, select_roi_list = _build_mask_target(rois, cls_ind, search_mask)

                # for deformable roi pooling
                batch_inds = torch.from_numpy(np.arange(
                    batch_size).repeat(cfg.TRAIN.ROI_PER_IMG).reshape(batch_size*cfg.TRAIN.ROI_PER_IMG, 1)).cuda().float()
                rois = torch.cat((batch_inds, torch.stack(select_roi_list).view(-1, 4)), dim=1)

                # Combine fused template features với search features
                # zf_fused[:2] là levels 0,1 (mean fused), zf_enhanced là levels 2,3,4 (enhanced)
                # Ensure both are lists
                if isinstance(zf_fused, (list, tuple)):
                    enhanced_zf = list(zf_fused[:2]) + list(zf_enhanced)
                else:
                    # Should not happen, but handle gracefully
                    enhanced_zf = [zf_fused] + list(zf_enhanced) if isinstance(zf_enhanced, (list, tuple)) else [zf_fused, zf_enhanced]
                
                # Ensure xf is also a list
                if not isinstance(xf, (list, tuple)):
                    xf = [xf]
                
                b_fused_features, m_fused_features = self.feature_fusion(enhanced_zf, xf)
                
                bbox_pred = self.bbox_head(b_fused_features, rois)
                bbox_pred = bbox_pred.view_as(regression_target)

                mask_pred = self.mask_head(m_fused_features, rois)
                mask_pred = mask_pred.view_as(mask_targets)

                # compute loss
                mask_loss, iou_m, iou_5, iou_7 = mask_loss_bce(mask_pred, mask_targets, mask_weight)
                bbox_loss = det_loss_smooth_l1(bbox_pred, regression_target, bbox_weight)

                outputs['mask_labels'] = mask_targets
                outputs['mask_preds'] = mask_pred
                outputs['total_loss'] += (cfg.TRAIN.MASK_WEIGHT * mask_loss + cfg.TRAIN.BBOX_WEIGHT * bbox_loss)
                outputs['bbox_loss'] = bbox_loss
                outputs['mask_loss'] = mask_loss
                outputs['iou_m'] = iou_m
                outputs['iou_5'] = iou_5
                outputs['iou_7'] = iou_7

        return outputs

