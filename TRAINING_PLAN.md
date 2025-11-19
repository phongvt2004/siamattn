# Kế Hoạch Training: Cross-View Deformable Siamese Detection (CV-DSD)

## Tổng Quan

Kế hoạch này mô tả cách train model **Cross-View Deformable Siamese Detection** sử dụng dữ liệu trong folder `training_dataset/observing/train/` để giải quyết bài toán **Ground-to-Air Few-Shot Object Detection**.

## 1. Phân Tích Dữ Liệu Hiện Tại

### 1.1 Cấu Trúc Dữ Liệu `observing/train/`

```
observing/train/
├── annotations/
│   └── annotations.json      # Bbox annotations cho các frame trong drone video
└── samples/
    ├── Backpack_0/
    │   ├── drone_video.mp4    # UAV view video
    │   └── object_images/
    │       ├── img_1.jpg      # Ground view 1
    │       ├── img_2.jpg      # Ground view 2
    │       └── img_3.jpg      # Ground view 3
    ├── Backpack_1/
    ├── Jacket_0/
    ├── ...
```

**Đặc điểm:**
- **Support Set (Z):** 3 ảnh ground-level (img_1, img_2, img_3) - các góc nhìn khác nhau của cùng một object
- **Query Set (X):** 1 video drone (drone_video.mp4) - chứa các frame với bbox annotations
- **Annotations:** JSON format với bbox cho mỗi frame trong drone video

### 1.2 Format Annotations

```json
[
  {
    "video_id": "Backpack_0",
    "annotations": [
      {
        "bboxes": [
          {"frame": 3483, "x1": 321, "y1": 0, "x2": 381, "y2": 12},
          ...
        ]
      }
    ]
  }
]
```

## 2. Kiến Trúc Model Cần Modify

### 2.1 Thay Đổi So Với Model Hiện Tại

**Model Hiện Tại (SiamRPN++):**
- Input: 1 template (Z) + 1 search image (X)
- Feature Extraction → Feature Enhancement → RPN → Output

**Model Mới (CV-DSD):**
- Input: **3 templates (Z₁, Z₂, Z₃)** + 1 search image (X)
- Feature Extraction → **Multi-Template Fusion** → **Deformable Cross-Attention** → RPN → Output

### 2.2 Các Module Cần Thêm/Modify

#### 2.2.1 Multi-Template Feature Fusion Module
```python
class MultiTemplateFusion(nn.Module):
    """
    Fuse 3 template features (Z₁, Z₂, Z₃) thành Z_fused
    Options:
    1. Max-Pooling: Z_fused = max(Z₁, Z₂, Z₃)
    2. Attention-weighted: Z_fused = Σ αᵢ * Zᵢ
    3. Self-Attention + Fusion: Apply self-attention trước, sau đó fuse
    """
```

#### 2.2.2 Enhanced Deformable Cross-Attention
- Sử dụng `FeatureEnhance` hiện có nhưng modify để:
  - Nhận Z_fused (thay vì Z đơn)
  - Tăng cường Deformable Convolution offsets để handle viewpoint shift lớn hơn

## 3. Implementation Plan

### Phase 1: Tạo Dataset Mới cho Cross-View Training

#### 3.1.1 File: `pysot/datasets/cross_view_dataset.py`

**Chức năng:**
- Load 3 ground images từ `object_images/`
- Load 1 frame từ `drone_video.mp4` với bbox annotation
- Apply augmentation phù hợp cho cross-view scenario

**Key Methods:**
```python
class CrossViewDataset(Dataset):
    def __init__(self, root, anno_file):
        # Load annotations.json
        # Parse video_id và bboxes
        
    def __getitem__(self, index):
        # Load 3 ground images (img_1, img_2, img_3)
        # Load 1 drone frame với bbox
        # Apply augmentation
        return {
            'templates': [template1, template2, template3],  # 3 templates
            'search': search_image,
            'label_cls': cls_labels,
            'label_loc': loc_labels,
            'bbox': bbox
        }
```

#### 3.1.2 Data Augmentation Strategy

**Cho Ground Images (Templates):**
- Standard augmentation: shift, scale, blur, color jitter
- **Không flip** (giữ nguyên orientation)
- **Perspective transform nhẹ** để simulate slight viewpoint changes

**Cho Drone Images (Search):**
- Standard augmentation: shift, scale, blur, color jitter
- **Perspective transform mạnh hơn** để simulate different drone angles
- **Scale augmentation lớn hơn** (UAV objects thường nhỏ hơn)

### Phase 2: Modify Model Architecture

#### 3.2.1 File: `pysot/models/cross_view_model.py`

**Thay đổi chính:**

1. **Template Processing:**
```python
def forward(self, data):
    templates = data['templates']  # List of 3 templates [B, 3, C, H, W]
    search = data['search']  # [B, C, H, W]
    
    # Extract features cho 3 templates
    zf_list = []
    for template in templates:
        zf = self.backbone(template)
        if cfg.ADJUST.ADJUST:
            zf[2:] = self.neck(zf[2:])
        zf_list.append(zf)
    
    # Multi-template fusion
    zf_fused = self.multi_template_fusion(zf_list)  # NEW MODULE
    
    # Search feature extraction
    xf = self.backbone(search)
    if cfg.ADJUST.ADJUST:
        xf[2:] = self.neck(xf[2:])
    
    # Enhanced deformable cross-attention
    zf_enhanced, xf_enhanced = self.feature_enhance(zf_fused[2:], xf[2:])
    
    # RPN
    cls, loc = self.rpn_head(zf_enhanced, xf_enhanced)
    ...
```

2. **Multi-Template Fusion Module:**
```python
class MultiTemplateFusion(nn.Module):
    def __init__(self, in_channels=256, fusion_method='attention'):
        super().__init__()
        self.fusion_method = fusion_method
        
        if fusion_method == 'attention':
            # Attention-weighted fusion
            self.attention = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, in_channels // 4, 1),
                nn.ReLU(),
                nn.Conv2d(in_channels // 4, 3, 1),  # 3 templates
                nn.Softmax(dim=1)
            )
        elif fusion_method == 'self_attention_then_fusion':
            # Self-attention trên mỗi template trước
            self.self_attn = PAM_Module(in_channels)
            self.attention = ...  # như trên
    
    def forward(self, zf_list):
        # zf_list: list of 3 feature maps [B, C, H, W]
        if self.fusion_method == 'max':
            zf_fused = torch.stack(zf_list).max(dim=0)[0]
        elif self.fusion_method == 'attention':
            # Compute attention weights
            weights = self.attention(zf_list[0])  # [B, 3, 1, 1]
            # Weighted sum
            zf_fused = sum(w * zf for w, zf in zip(weights.split(1, 1), zf_list))
        ...
        return zf_fused
```

#### 3.2.2 Modify FeatureEnhance cho Cross-View

- Tăng số lượng deformable convolution offsets
- Thêm regularization cho offsets để tránh quá lớn

### Phase 3: Training Script

#### 3.3.1 File: `tools/train_cross_view.py`

**Dựa trên `tools/train.py` nhưng modify:**

1. **DataLoader:**
```python
def build_data_loader():
    train_dataset = CrossViewDataset(
        root='training_dataset/observing/train/samples',
        anno_file='training_dataset/observing/train/annotations/annotations.json'
    )
    train_loader = DataLoader(...)
    return train_loader
```

2. **Loss Function:**
- Giữ nguyên: Classification Loss + Regression Loss
- Có thể thêm: **Cross-View Consistency Loss** (khuyến khích features từ 3 templates consistent với nhau)

#### 3.3.2 Training Configuration

**File: `configs/cross_view_config.yaml`**

```yaml
# Dataset
DATASET:
  NAMES: ('OBSERVING',)
  OBSERVING:
    ROOT: 'training_dataset/observing/train/samples'
    ANNO: 'training_dataset/observing/train/annotations/annotations.json'
    NUM_USE: -1  # Use all

# Model
MODEL:
  MULTI_TEMPLATE: True
  FUSION_METHOD: 'attention'  # 'max', 'attention', 'self_attention_then_fusion'
  
# Training
TRAIN:
  BATCH_SIZE: 8  # Smaller vì có 3 templates
  EPOCH: 50
  BASE_LR: 0.001  # Lower learning rate
  ...
```

### Phase 4: Training Strategy

#### 3.4.1 Pre-training Phase (Optional)

1. **Pre-train trên standard tracking datasets** (LaSOT, TrackingNet) với 1 template
2. **Fine-tune** với cross-view dataset

#### 3.4.2 Main Training Phase

**Stage 1: Feature Extraction (Epochs 1-10)**
- Freeze backbone
- Train only fusion module và feature enhancement
- Learning rate: 0.001

**Stage 2: Full Training (Epochs 11-40)**
- Unfreeze backbone layers (layer2, layer3, layer4)
- Train toàn bộ model
- Learning rate: 0.0001 (backbone), 0.001 (other)

**Stage 3: Fine-tuning (Epochs 41-50)**
- Lower learning rate
- Focus on difficult samples

#### 3.4.3 Data Sampling Strategy

**Positive Pairs:**
- 3 ground images từ class A + 1 drone frame từ class A (same video_id)

**Negative Pairs:**
- 3 ground images từ class A + 1 drone frame từ class B (different video_id)
- Ratio: 80% positive, 20% negative

**Frame Sampling từ Video:**
- Sample frames có bbox annotations
- Prefer frames với bbox size > threshold (tránh quá nhỏ)

### Phase 5: Evaluation & Ablation Studies

#### 3.5.1 Metrics

- **Detection Accuracy:** mAP, Precision@IoU=0.5
- **Cross-View Performance:** Compare với single-template baseline

#### 3.5.2 Ablation Studies

1. **Fusion Method:**
   - Max-Pooling vs Attention-weighted vs Self-Attention+Fusion

2. **Number of Templates:**
   - 1 template vs 3 templates vs 5 templates

3. **Deformable vs Standard:**
   - Standard Conv vs Deformable Conv trong FeatureEnhance

## 4. File Structure Cần Tạo

```
pysot/
├── datasets/
│   ├── cross_view_dataset.py      # NEW: Dataset cho cross-view
│   └── ...
├── models/
│   ├── cross_view_model.py         # NEW: Model với multi-template
│   ├── multi_template_fusion.py    # NEW: Fusion module
│   └── ...
tools/
├── train_cross_view.py             # NEW: Training script
└── ...
configs/
└── cross_view_config.yaml          # NEW: Config file
```

## 5. Implementation Steps

### Step 1: Tạo Dataset (Priority: HIGH)
- [ ] Implement `CrossViewDataset`
- [ ] Test data loading
- [ ] Verify augmentation

### Step 2: Implement Fusion Module (Priority: HIGH)
- [ ] Create `MultiTemplateFusion`
- [ ] Test với dummy data
- [ ] Compare các fusion methods

### Step 3: Modify Model (Priority: HIGH)
- [ ] Create `CrossViewModelBuilder`
- [ ] Integrate fusion module
- [ ] Test forward pass

### Step 4: Training Script (Priority: MEDIUM)
- [ ] Create `train_cross_view.py`
- [ ] Create config file
- [ ] Test training loop

### Step 5: Training & Evaluation (Priority: MEDIUM)
- [ ] Run training
- [ ] Monitor metrics
- [ ] Run ablation studies

## 6. Expected Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **Memory**: 3 templates tăng memory | Giảm batch size, sử dụng gradient checkpointing |
| **Convergence**: Model khó converge | Pre-training, learning rate scheduling |
| **Scale mismatch**: UAV objects nhỏ hơn | Adjust anchor scales, data augmentation |
| **Viewpoint gap**: Quá lớn giữa ground và air | Tăng deformable offsets, perspective augmentation |

## 7. Timeline Ước Tính

- **Week 1:** Dataset + Fusion Module
- **Week 2:** Model Integration + Training Script
- **Week 3:** Training + Debugging
- **Week 4:** Evaluation + Ablation Studies

## 8. Next Steps

1. Bắt đầu với **Step 1**: Implement `CrossViewDataset`
2. Test với một vài samples trước
3. Sau đó implement các components khác

