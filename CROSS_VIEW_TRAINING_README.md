# Hướng Dẫn Training Cross-View Siamese Detection

## Tổng Quan

Hệ thống này được thiết kế để train model **Cross-View Few-Shot Object Detection** sử dụng dữ liệu từ folder `training_dataset/observing/train/`.

## Cấu Trúc Files

```
pysot/
├── datasets/
│   └── cross_view_dataset.py          # Dataset loader cho cross-view data
├── models/
│   ├── cross_view_model.py            # Model với multi-template support
│   └── multi_template_fusion.py       # Module fusion 3 templates
tools/
├── train_cross_view.py                # Training script chính
└── test_cross_view_dataset.py         # Test script để verify dataset
configs/
└── cross_view_config.yaml             # Config file cho training
```

## Yêu Cầu

1. **Dữ liệu**: Folder `training_dataset/observing/train/` với cấu trúc:
   ```
   observing/train/
   ├── annotations/
   │   └── annotations.json
   └── samples/
       ├── Backpack_0/
       │   ├── drone_video.mp4
       │   └── object_images/
       │       ├── img_1.jpg
       │       ├── img_2.jpg
       │       └── img_3.jpg
       └── ...
   ```

2. **Pretrained Backbone**: ResNet-50 pretrained weights (đặt tại `pretrained_models/resnet50.model`)

3. **Dependencies**: Xem `requirements.txt`

## Cách Sử Dụng

### 1. Test Dataset (Khuyến nghị chạy trước)

```bash
python tools/test_cross_view_dataset.py
```

Script này sẽ:
- Load dataset và verify cấu trúc
- Test việc load một sample
- Verify shapes của data

### 2. Training

#### Cách 1: Sử dụng script shell (Khuyến nghị)

```bash
chmod +x run_cross_view_training.sh
./run_cross_view_training.sh [config_file] [gpu_ids]
```

Ví dụ:
```bash
# Sử dụng config mặc định, GPU 0
./run_cross_view_training.sh

# Sử dụng config tùy chỉnh, GPU 0,1
./run_cross_view_training.sh configs/cross_view_config.yaml 0,1
```

#### Cách 2: Chạy trực tiếp Python

```bash
python tools/train_cross_view.py \
    --cfg configs/cross_view_config.yaml \
    --seed 123456
```

### 3. Multi-GPU Training

```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=2333 \
    tools/train_cross_view.py \
    --cfg configs/cross_view_config.yaml
```

## Config File

File `configs/cross_view_config.yaml` chứa tất cả các tham số training:

### Key Parameters:

- **TRAIN.BATCH_SIZE**: Batch size (mặc định: 8, nhỏ hơn vì có 3 templates)
- **TRAIN.EPOCH**: Số epochs (mặc định: 50)
- **TRAIN.BASE_LR**: Learning rate (mặc định: 0.001)
- **MODEL.FUSION_METHOD**: Phương pháp fusion templates
  - `'max'`: Max-pooling
  - `'mean'`: Average pooling
  - `'attention'`: Attention-weighted (mặc định, khuyến nghị)
  - `'self_attention_then_fusion'`: Self-attention trước, sau đó fuse
  - `'learned_weight'`: Learned weights

### Thay Đổi Config:

1. **Fusion Method**:
```yaml
MODEL:
  FUSION_METHOD: 'attention'  # Thay đổi ở đây
```

2. **Learning Rate**:
```yaml
TRAIN:
  BASE_LR: 0.001  # Thay đổi ở đây
```

3. **Batch Size**:
```yaml
TRAIN:
  BATCH_SIZE: 8  # Thay đổi ở đây (tùy GPU memory)
```

## Training Strategy

Model được train theo 3 stages:

1. **Stage 1 (Epochs 1-10)**: 
   - Freeze backbone
   - Train fusion module và feature enhancement
   - Learning rate: 0.001

2. **Stage 2 (Epochs 11-40)**:
   - Unfreeze backbone layers (layer2, layer3, layer4)
   - Train toàn bộ model
   - Learning rate: 0.0001 (backbone), 0.001 (other)

3. **Stage 3 (Epochs 41-50)**:
   - Fine-tuning với lower learning rate

## Output

- **Logs**: `./logs/cross_view/`
- **Checkpoints**: `./snapshot/cross_view/checkpoint_e{epoch}.pth`
- **Tensorboard**: Chạy `tensorboard --logdir ./logs/cross_view/`

## Checkpoint và Resume Training

### 1. Resume Training từ Checkpoint

Nếu training bị gián đoạn, bạn có thể resume từ checkpoint:

**Cách 1: Sử dụng config file**
```yaml
# Trong configs/cross_view_config.yaml
TRAIN:
  RESUME: 'snapshot/cross_view/checkpoint_e10.pth'  # Path to checkpoint
  START_EPOCH: 10  # Epoch để resume (thường là epoch trong checkpoint)
```

**Cách 2: Sử dụng pretrained model**
```yaml
# Trong configs/cross_view_config.yaml
TRAIN:
  PRETRAINED: 'snapshot/cross_view/checkpoint_e20.pth'  # Load weights only
```

**Cách 3: Command line (nếu script hỗ trợ)**
```bash
python tools/train_cross_view.py \
    --cfg configs/cross_view_config.yaml \
    --resume snapshot/cross_view/checkpoint_e10.pth
```

### 2. Load Checkpoint để Inference

```python
import torch
from pysot.models.cross_view_model import CrossViewModelBuilder

# Load checkpoint
checkpoint = torch.load('snapshot/cross_view/checkpoint_e50.pth', map_location='cpu')

# Create model
model = CrossViewModelBuilder(fusion_method='attention')

# Load weights
model.load_state_dict(checkpoint['state_dict'])
model.eval()

# Hoặc nếu chỉ có state_dict
# model.load_state_dict(torch.load('snapshot/cross_view/checkpoint_e50.pth'))
```

### 3. Checkpoint Structure

Mỗi checkpoint chứa:
```python
{
    'epoch': 10,                    # Epoch number
    'state_dict': model.state_dict(), # Model weights
    'optimizer': optimizer.state_dict() # Optimizer state (for resume)
}
```

### 4. Best Practices

- **Lưu checkpoint thường xuyên**: Model tự động lưu mỗi epoch
- **Giữ best checkpoint**: Lưu checkpoint có loss thấp nhất
- **Resume đúng epoch**: Set `START_EPOCH` đúng với epoch trong checkpoint
- **Backup checkpoints**: Copy checkpoints quan trọng ra nơi khác

### 5. Ví dụ Resume Training

```yaml
# configs/cross_view_config.yaml
TRAIN:
  RESUME: 'snapshot/cross_view/checkpoint_e25.pth'
  START_EPOCH: 25
  EPOCH: 50  # Continue to epoch 50
```

Khi resume, training sẽ:
- Load model weights từ checkpoint
- Load optimizer state (learning rate, momentum, etc.)
- Tiếp tục từ epoch đã chỉ định

## Troubleshooting

### 1. Lỗi "Video not found"
- Kiểm tra đường dẫn trong config: `DATASET.OBSERVING.ROOT`
- Đảm bảo tất cả videos có tên `drone_video.mp4`

### 2. Lỗi "Image not found"
- Kiểm tra tất cả folders có đủ 3 images: `img_1.jpg`, `img_2.jpg`, `img_3.jpg`

### 3. Out of Memory
- Giảm `TRAIN.BATCH_SIZE` trong config
- Giảm `TRAIN.NUM_WORKERS`

### 4. Dataset empty
- Kiểm tra `annotations.json` có đúng format
- Kiểm tra các video_id trong annotations match với folder names

## Evaluation

Sau khi training, bạn có thể:

### 1. Load Checkpoint để Evaluation

```python
import torch
from pysot.models.cross_view_model import CrossViewModelBuilder

# Load best checkpoint
checkpoint_path = 'snapshot/cross_view/checkpoint_e50.pth'
checkpoint = torch.load(checkpoint_path, map_location='cuda:0')

# Create và load model
model = CrossViewModelBuilder(fusion_method='attention').cuda()
model.load_state_dict(checkpoint['state_dict'])
model.eval()

print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
```

### 2. Inference với Model

```python
# Load 3 template images
templates = [load_image('img_1.jpg'), load_image('img_2.jpg'), load_image('img_3.jpg')]
search_image = load_drone_frame('frame_100.jpg')

# Set templates
model.template(templates)

# Track object
output = model.track(search_image)
bbox = output['loc']  # Predicted bounding box
```

### 3. Evaluate Metrics

- **mAP**: Mean Average Precision
- **Precision@IoU=0.5**: Precision at IoU threshold 0.5
- **Success Rate**: Percentage of successful detections

## Ablation Studies

Để so sánh các fusion methods:

1. Train với `FUSION_METHOD: 'max'`
2. Train với `FUSION_METHOD: 'attention'`
3. Train với `FUSION_METHOD: 'self_attention_then_fusion'`
4. So sánh kết quả

## Notes

- Model sử dụng **Deformable Convolution** trong FeatureEnhance để handle viewpoint shift
- **Multi-Template Fusion** giúp model học được features từ nhiều góc nhìn ground-level
- **Cross-View Attention** giúp align features giữa ground và air views

## Contact

Nếu có vấn đề, vui lòng kiểm tra:
1. Dataset structure
2. Config file
3. Logs trong `./logs/cross_view/`

