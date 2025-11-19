# Quick Start cho Kaggle

## Cách chạy training trên Kaggle

### Option 1: Chạy trực tiếp trong Notebook (Khuyến nghị)

```python
# Cell 1: Setup environment và build DCN extensions
import os
import sys

# Add current directory to path
if '/kaggle/working/siamattn' not in sys.path:
    sys.path.insert(0, '/kaggle/working/siamattn')

# Build DCN extensions (required for training)
print("Building DCN extensions...")
!cd /kaggle/working/siamattn/pysot/models/head/dcn && python setup.py build_ext --inplace
print("✓ DCN extensions built")

# Cell 2: Verify imports
try:
    from pysot.core.config import cfg
    from pysot.models.head.dcn import deform_conv_cuda
    print("✓ pysot imported successfully")
    print("✓ DCN extensions loaded")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure DCN extensions are built")

# Cell 3: Run training
!cd /kaggle/working/siamattn && python tools/train_cross_view.py --cfg configs/cross_view_config.yaml --seed 123456
```

### Option 2: Sử dụng script shell

```bash
# Trong Kaggle notebook
!cd /kaggle/working/siamattn && bash run_cross_view_training.sh
```

### Option 3: Set PYTHONPATH và chạy

```python
import os
os.environ['PYTHONPATH'] = '/kaggle/working/siamattn:' + os.environ.get('PYTHONPATH', '')
!cd /kaggle/working/siamattn && python tools/train_cross_view.py --cfg configs/cross_view_config.yaml
```

## Lưu ý cho Kaggle

1. **Working Directory**: Đảm bảo bạn đang ở đúng thư mục (`/kaggle/working/siamattn`)

2. **PYTHONPATH**: Script đã tự động thêm path, nhưng nếu vẫn lỗi, set thủ công:
   ```python
   import sys
   sys.path.insert(0, '/kaggle/working/siamattn')
   ```

3. **Config Path**: Đảm bảo config file ở đúng vị trí:
   ```python
   import os
   print(os.path.exists('configs/cross_view_config.yaml'))
   ```

4. **Dataset Path**: Kiểm tra dataset:
   ```python
   import os
   print(os.path.exists('training_dataset/observing/train'))
   ```

## Troubleshooting

### Lỗi "No module named 'pysot'"

**Giải pháp 1**: Thêm path vào script (đã được fix trong train_cross_view.py)

**Giải pháp 2**: Chạy từ root directory:
```bash
cd /kaggle/working/siamattn
python tools/train_cross_view.py --cfg configs/cross_view_config.yaml
```

**Giải pháp 3**: Install package (development mode):
```bash
cd /kaggle/working/siamattn
pip install -e .
```

### Lỗi "cannot import name 'deform_conv_cuda'"

**Giải pháp**: Build DCN extensions trước:
```python
# Trong Kaggle notebook
!cd /kaggle/working/siamattn/pysot/models/head/dcn && python setup.py build_ext --inplace
```

Hoặc script sẽ tự động build (nhưng có thể mất vài phút).

### Lỗi "Config file not found"

Đảm bảo bạn đang ở đúng thư mục:
```python
import os
print("Current dir:", os.getcwd())
os.chdir('/kaggle/working/siamattn')
print("Changed to:", os.getcwd())
```

### Lỗi "Dataset not found"

Kiểm tra đường dẫn dataset trong config:
```python
from pysot.core.config import cfg
cfg.merge_from_file('configs/cross_view_config.yaml')
print("Dataset root:", cfg.DATASET.OBSERVING.ROOT)
print("Exists:", os.path.exists(cfg.DATASET.OBSERVING.ROOT))
```

