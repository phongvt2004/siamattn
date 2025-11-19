#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script để verify CrossViewDataset hoạt động đúng
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import os
import numpy as np
import cv2

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pysot.core.config import cfg
from pysot.datasets.cross_view_dataset import CrossViewDataset

def test_dataset():
    """Test CrossViewDataset"""
    print("Testing CrossViewDataset...")
    
    # Load config
    config_file = 'configs/cross_view_config.yaml'
    if os.path.exists(config_file):
        cfg.merge_from_file(config_file)
    else:
        print("Warning: Config file not found, using defaults")
        # Set default values
        cfg.TRAIN = type('obj', (object,), {
            'EXEMPLAR_SIZE': 127,
            'SEARCH_SIZE': 255,
            'BASE_SIZE': 8,
            'OUTPUT_SIZE': 25,
            'EPOCH': 1
        })()
        cfg.ANCHOR = type('obj', (object,), {
            'STRIDE': 8,
            'RATIOS': [0.33, 0.5, 1, 2, 3],
            'SCALES': [8]
        })()
        cfg.DATASET = type('obj', (object,), {
            'TEMPLATE': type('obj', (object,), {
                'SHIFT': 4,
                'SCALE': 0.05,
                'BLUR': 0.0,
                'FLIP': 0.0,
                'COLOR': 0.5
            })(),
            'SEARCH': type('obj', (object,), {
                'SHIFT': 64,
                'SCALE': 0.18,
                'BLUR': 0.2,
                'FLIP': 0.2,
                'COLOR': 0.5
            })(),
            'NEG': 0.2,
            'GRAY': 0.0,
            'OBSERVING': type('obj', (object,), {
                'ROOT': 'training_dataset/observing/train/samples',
                'ANNO': 'training_dataset/observing/train/annotations/annotations.json'
            })()
        })()
    
    # Create dataset
    try:
        dataset = CrossViewDataset(
            root=cfg.DATASET.OBSERVING.ROOT,
            anno_file=cfg.DATASET.OBSERVING.ANNO,
            frame_range=1
        )
        print("✓ Dataset created successfully")
        print("  Number of videos: {}".format(len(dataset.video_list)))
    except Exception as e:
        print("✗ Error creating dataset: {}".format(e))
        return False
    
    # Test __getitem__
    try:
        if len(dataset) > 0:
            sample = dataset[0]
            print("✓ Sample loaded successfully")
            print("  Templates shape: {}".format(sample['templates'].shape))
            print("  Search shape: {}".format(sample['search'].shape))
            print("  Label cls shape: {}".format(sample['label_cls'].shape))
            print("  Label loc shape: {}".format(sample['label_loc'].shape))
            
            # Verify shapes
            assert sample['templates'].shape == (3, 3, 127, 127), \
                "Templates shape should be (3, 3, 127, 127), got {}".format(sample['templates'].shape)
            assert sample['search'].shape == (3, 255, 255), \
                "Search shape should be (3, 255, 255), got {}".format(sample['search'].shape)
            
            print("✓ All shapes are correct")
        else:
            print("✗ Dataset is empty")
            return False
    except Exception as e:
        print("✗ Error loading sample: {}".format(e))
        import traceback
        traceback.print_exc()
        return False
    
    print("\n✓ All tests passed!")
    return True

if __name__ == '__main__':
    success = test_dataset()
    sys.exit(0 if success else 1)

