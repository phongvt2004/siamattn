# Quick Start cho Kaggle

## Cách chạy training trên Kaggle

### Option 1: Chạy trực tiếp trong Notebook

```python
# Cell 1: Setup environment
import os
import sys

# Add current directory to path
if '/kaggle/working/siamattn' not in sys.path:
    sys.path.insert(0, '/kaggle/working/siamattn')

# Verify imports
try:
    from pysot.core.config import cfg
    print("✓ pysot imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("Make sure you're in the correct directory")

# Cell 2: Run training
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

