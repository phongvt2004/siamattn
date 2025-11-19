# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
from pysot.models.head.attention import PAM_Module, CAM_Calculate, CAM_Use
from pysot.models.init_weight import kaiming_init


class MultiTemplateFusion(nn.Module):
    """
    Fuse multiple template features (Z₁, Z₂, Z₃) thành Z_fused
    
    Fusion methods:
    1. 'max': Max-pooling across templates
    2. 'mean': Average pooling
    3. 'attention': Attention-weighted fusion
    4. 'self_attention_then_fusion': Apply self-attention on each template, then fuse
    """
    def __init__(self, in_channels=256, fusion_method='attention', num_templates=3):
        super(MultiTemplateFusion, self).__init__()
        self.in_channels = in_channels
        self.fusion_method = fusion_method
        self.num_templates = num_templates
        
        if fusion_method == 'attention':
            # Attention-weighted fusion
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 4, num_templates, 1),
                nn.Softmax(dim=1)
            )
            kaiming_init(self.attention[1])
            kaiming_init(self.attention[3])
            
        elif fusion_method == 'self_attention_then_fusion':
            # Self-attention trên mỗi template trước
            self.self_attn = nn.ModuleList([
                PAM_Module(in_channels) for _ in range(num_templates)
            ])
            # Sau đó attention-weighted fusion
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // 4, num_templates, 1),
                nn.Softmax(dim=1)
            )
            kaiming_init(self.attention[1])
            kaiming_init(self.attention[3])
            
        elif fusion_method == 'learned_weight':
            # Learned weights (không phụ thuộc vào features)
            self.weights = nn.Parameter(torch.ones(num_templates) / num_templates)
            self.softmax = nn.Softmax(dim=0)
            
    def forward(self, zf_list):
        """
        Args:
            zf_list: List of feature maps, mỗi element là [B, C, H, W]
                    hoặc nếu là multi-level features, mỗi element là list of features
                    (assumed to be levels 2,3,4 with 256 channels each after neck)
        
        Returns:
            zf_fused: Fused feature map [B, C, H, W] hoặc list of fused features
        """
        # Check if multi-level features (list of lists)
        if isinstance(zf_list[0], (list, tuple)):
            # Multi-level features: zf_list[i] là list of features cho template i
            # Each zf_list[i] should be [level2, level3, level4] with 256 channels
            num_levels = len(zf_list[0])
            fused_list = []
            
            for level_idx in range(num_levels):
                level_features = [zf[level_idx] for zf in zf_list]
                # Verify all have same channels (should be 256 after neck)
                channels = [f.shape[1] for f in level_features]
                if len(set(channels)) > 1:
                    raise ValueError(f"Level {level_idx} has inconsistent channels: {channels}")
                
                # Only use attention fusion if channels match expected (256)
                if channels[0] == self.in_channels:
                    fused = self._fuse_single_level(level_features)
                else:
                    # Fallback to mean for other channels
                    fused = torch.stack(level_features, dim=0).mean(dim=0)
                
                fused_list.append(fused)
            
            return fused_list
        else:
            # Single level features - should all have same channels
            channels = [f.shape[1] for f in zf_list]
            if len(set(channels)) > 1:
                raise ValueError(f"Inconsistent channels: {channels}")
            
            # Only use attention if channels match
            if channels[0] == self.in_channels:
                return self._fuse_single_level(zf_list)
            else:
                # Fallback to mean
                return torch.stack(zf_list, dim=0).mean(dim=0)
    
    def _fuse_single_level(self, zf_list):
        """Fuse features at single level"""
        # Stack: [num_templates, B, C, H, W]
        zf_stack = torch.stack(zf_list, dim=0)
        
        if self.fusion_method == 'max':
            # Max-pooling across templates
            zf_fused, _ = torch.max(zf_stack, dim=0)
            
        elif self.fusion_method == 'mean':
            # Average pooling
            zf_fused = torch.mean(zf_stack, dim=0)
            
        elif self.fusion_method == 'attention':
            # Attention-weighted fusion
            # Compute attention weights từ first template (hoặc average)
            zf_avg = torch.mean(zf_stack, dim=0)  # [B, C, H, W]
            weights = self.attention(zf_avg)  # [B, num_templates, 1, 1]
            
            # Weighted sum
            zf_fused = torch.zeros_like(zf_list[0])
            for i, zf in enumerate(zf_list):
                zf_fused += weights[:, i:i+1, :, :] * zf
            
        elif self.fusion_method == 'self_attention_then_fusion':
            # Apply self-attention on each template
            zf_attn_list = []
            for i, zf in enumerate(zf_list):
                zf_attn = self.self_attn[i](zf)
                zf_attn_list.append(zf_attn)
            
            # Attention-weighted fusion
            zf_attn_stack = torch.stack(zf_attn_list, dim=0)
            zf_avg = torch.mean(zf_attn_stack, dim=0)
            weights = self.attention(zf_avg)
            
            zf_fused = torch.zeros_like(zf_attn_list[0])
            for i, zf_attn in enumerate(zf_attn_list):
                zf_fused += weights[:, i:i+1, :, :] * zf_attn
                
        elif self.fusion_method == 'learned_weight':
            # Learned weights
            weights = self.softmax(self.weights)  # [num_templates]
            zf_fused = sum(w * zf for w, zf in zip(weights, zf_list))
            
        else:
            raise ValueError("Unknown fusion method: {}".format(self.fusion_method))
        
        return zf_fused

