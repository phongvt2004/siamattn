# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import logging
import os
import sys

import cv2
import numpy as np
from torch.utils.data import Dataset

from pysot.utils.bbox import center2corner, Center
from pysot.datasets.anchor_target import AnchorTarget
from pysot.datasets.augmentation import Augmentation
from pysot.core.config import cfg

logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


class CrossViewDataset(Dataset):
    """
    Dataset cho Cross-View Few-Shot Object Detection
    - Support Set: 3 ground-level images (img_1.jpg, img_2.jpg, img_3.jpg)
    - Query Set: 1 frame từ drone video với bbox annotation
    """
    def __init__(self, root, anno_file, frame_range=1):
        super(CrossViewDataset, self).__init__()
        
        self.root = root
        self.anno_file = anno_file
        self.frame_range = frame_range
        
        # Load annotations
        logger.info("Loading annotations from {}".format(anno_file))
        with open(anno_file, 'r') as f:
            self.annotations = json.load(f)
        
        # Build video index
        self.video_list = []
        self.video_annotations = {}
        
        for item in self.annotations:
            video_id = item['video_id']
            self.video_list.append(video_id)
            
            # Parse bboxes
            bbox_dict = {}
            for ann_group in item['annotations']:
                for bbox_info in ann_group['bboxes']:
                    frame_num = bbox_info['frame']
                    bbox = [bbox_info['x1'], bbox_info['y1'], 
                           bbox_info['x2'], bbox_info['y2']]
                    bbox_dict[frame_num] = bbox
            
            self.video_annotations[video_id] = bbox_dict
        
        self.num = len(self.video_list)
        logger.info("Loaded {} videos".format(self.num))
        
        # Create anchor target
        desired_size = (cfg.TRAIN.SEARCH_SIZE - cfg.TRAIN.EXEMPLAR_SIZE) / \
            cfg.ANCHOR.STRIDE + 1 + cfg.TRAIN.BASE_SIZE
        if desired_size != cfg.TRAIN.OUTPUT_SIZE:
            raise Exception('size not match!')
        
        self.anchor_target = AnchorTarget()
        
        # Data augmentation
        # Template augmentation: nhẹ hơn, không flip
        self.template_aug = Augmentation(
            shift=cfg.DATASET.TEMPLATE.SHIFT,
            scale=cfg.DATASET.TEMPLATE.SCALE,
            blur=cfg.DATASET.TEMPLATE.BLUR,
            flip=0.0,  # Không flip cho ground images
            color=cfg.DATASET.TEMPLATE.COLOR
        )
        
        # Search augmentation: mạnh hơn cho drone images
        self.search_aug = Augmentation(
            shift=cfg.DATASET.SEARCH.SHIFT,
            scale=cfg.DATASET.SEARCH.SCALE,
            blur=cfg.DATASET.SEARCH.BLUR,
            flip=cfg.DATASET.SEARCH.FLIP,
            color=cfg.DATASET.SEARCH.COLOR
        )
        
        # Shuffle
        self.pick = self.shuffle()
        
    def shuffle(self):
        """Shuffle video list"""
        pick = list(range(self.num))
        np.random.shuffle(pick)
        return pick
    
    def _get_bbox(self, image, shape):
        """Convert bbox to corner format"""
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2]-shape[0], shape[3]-shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w+h)
        hc_z = h + context_amount * (w+h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w*scale_z
        h = h*scale_z
        cx, cy = imw//2, imh//2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox
    
    def _load_drone_frame(self, video_id, frame_num):
        """Load frame từ drone video"""
        video_path = os.path.join(self.root, video_id, 'drone_video.mp4')
        if not os.path.exists(video_path):
            raise FileNotFoundError("Video not found: {}".format(video_path))
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError("Cannot read frame {} from {}".format(frame_num, video_path))
        
        return frame
    
    def _load_ground_images(self, video_id):
        """Load 3 ground images"""
        image_dir = os.path.join(self.root, video_id, 'object_images')
        images = []
        
        for i in range(1, 4):  # img_1.jpg, img_2.jpg, img_3.jpg
            img_path = os.path.join(image_dir, 'img_{}.jpg'.format(i))
            if not os.path.exists(img_path):
                raise FileNotFoundError("Image not found: {}".format(img_path))
            
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError("Cannot read image: {}".format(img_path))
            
            images.append(img)
        
        return images
    
    def __len__(self):
        # Mỗi video có thể có nhiều frames, nhưng để đơn giản, 
        # chúng ta sẽ sample một frame mỗi video mỗi lần
        # Có thể tăng số lượng bằng cách repeat
        if hasattr(cfg, 'TRAIN') and hasattr(cfg.TRAIN, 'EPOCH'):
            return self.num * cfg.TRAIN.EPOCH
        return self.num
    
    def __getitem__(self, index):
        index = self.pick[index % len(self.pick)]
        video_id = self.video_list[index]
        
        # Check negative sampling
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()
        
        if neg:
            # Negative pair: ground images từ video A, drone frame từ video B
            template_video_id = video_id
            search_video_id = np.random.choice(self.video_list)
            while search_video_id == template_video_id:
                search_video_id = np.random.choice(self.video_list)
        else:
            # Positive pair: cùng video_id
            template_video_id = video_id
            search_video_id = video_id
        
        # Load 3 ground images
        ground_images = self._load_ground_images(template_video_id)
        
        # Load drone frame với bbox
        search_annotations = self.video_annotations[search_video_id]
        if len(search_annotations) == 0:
            # Fallback: use first video
            search_video_id = self.video_list[0]
            search_annotations = self.video_annotations[search_video_id]
        
        # Sample một frame có annotation
        frame_nums = list(search_annotations.keys())
        if len(frame_nums) == 0:
            raise ValueError("No annotations for video: {}".format(search_video_id))
        
        frame_num = np.random.choice(frame_nums)
        bbox_anno = search_annotations[frame_num]
        
        # Load drone frame
        search_image = self._load_drone_frame(search_video_id, frame_num)
        
        # Get bbox
        search_box = self._get_bbox(search_image, bbox_anno)
        
        # Process 3 ground images
        templates = []
        for ground_img in ground_images:
            # Get bbox cho ground image (assume center crop)
            template_box = self._get_bbox(ground_img, ground_img.shape[:2])
            
            # Augmentation
            gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
            template, _ = self.template_aug(
                ground_img,
                template_box,
                cfg.TRAIN.EXEMPLAR_SIZE,
                gray=gray
            )
            templates.append(template)
        
        # Process search image
        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
        search, bbox = self.search_aug(
            search_image,
            search_box,
            cfg.TRAIN.SEARCH_SIZE,
            gray=gray
        )
        
        # Get labels
        cls, delta, delta_weight, overlap = self.anchor_target(
            bbox, cfg.TRAIN.OUTPUT_SIZE, neg
        )
        
        # Convert to numpy arrays
        templates = [t.transpose((2, 0, 1)).astype(np.float32) for t in templates]
        search = search.transpose((2, 0, 1)).astype(np.float32)
        
        # Stack 3 templates thành [3, C, H, W]
        templates = np.stack(templates, axis=0)
        
        return {
            'templates': templates,  # [3, C, H, W]
            'search': search,  # [C, H, W]
            'label_cls': cls,
            'label_loc': delta,
            'label_loc_weight': delta_weight,
            'bbox': np.array(bbox)
        }

