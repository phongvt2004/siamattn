# Các Vấn Đề Tiềm Ẩn và Giải Pháp

## 1. ✅ Đã Sửa: MultiTemplateFusion - Channel Mismatch
**Vấn đề**: Attention module expect 256 channels nhưng nhận 64 channels từ level 0
**Giải pháp**: Chỉ fuse levels 2,3,4 (sau neck, tất cả 256 channels), levels 0,1 dùng mean fusion

## 2. ⚠️ Cần Kiểm Tra: FeatureFusionNeck - Hardcoded Channels
**Vấn đề**: `FeatureFusionNeck` có hardcoded `in_channels=[64, 256, 256, 256, 256]`
**Vị trí**: `pysot/models/cross_view_model.py:98-99`
**Rủi ro**: Nếu backbone output khác, sẽ có shape mismatch
**Giải pháp**: Đã được xử lý trong code - chỉ dùng với levels đúng channels

## 3. ⚠️ Cần Kiểm Tra: List Concatenation
**Vấn đề**: `enhanced_zf = zf_fused[:2] + zf_enhanced` - cần đảm bảo types match
**Vị trí**: `pysot/models/cross_view_model.py:263`
**Rủi ro**: Nếu zf_fused[:2] là list nhưng zf_enhanced không phải list
**Giải pháp**: Đã được xử lý - zf_enhanced là list từ FeatureEnhance

## 4. ⚠️ Cần Kiểm Tra: FeatureEnhance Input
**Vấn đề**: `FeatureEnhance` expect list of 3 features với 256 channels
**Vị trí**: `pysot/models/cross_view_model.py:223`
**Rủi ro**: Nếu zf_fused[2:] không có đúng 3 elements hoặc channels khác 256
**Giải pháp**: Đã được xử lý - chỉ dùng zf_fused[2:] sau neck (đảm bảo 256 channels)

## 5. ⚠️ Cần Kiểm Tra: RPN Input
**Vấn đề**: `MultiRPN` expect list of features với channels match config
**Vị trí**: `pysot/models/cross_view_model.py:226`
**Rủi ro**: Nếu zf_enhanced và xf_enhanced không match với in_channels config
**Giải pháp**: Đã được xử lý - FeatureEnhance output có cùng channels (256)

## 6. ⚠️ Cần Kiểm Tra: Mask Head (nếu enable)
**Vấn đề**: `FeatureFusionNeck` trong mask head có hardcoded channels
**Vị trí**: `pysot/models/cross_view_model.py:98-99`
**Rủi ro**: Nếu actual features khác, sẽ lỗi
**Giải pháp**: Chỉ enable khi MASK=True và đảm bảo features match

## 7. ⚠️ Cần Kiểm Tra: Device Mismatch
**Vấn đề**: Các tensors có thể ở device khác nhau
**Rủi ro**: RuntimeError khi operations giữa tensors trên different devices
**Giải pháp**: Đã sửa trong model_load.py - load trên CPU rồi move lên device

## 8. ⚠️ Cần Kiểm Tra: Batch Size Consistency
**Vấn đề**: 3 templates có thể có batch size khác nhau
**Rủi ro**: Shape mismatch khi stack
**Giải pháp**: Dataset đảm bảo tất cả templates cùng batch size

## 9. ⚠️ Cần Kiểm Tra: NumPy/PyTorch Compatibility
**Vấn đề**: NumPy 2.0 đã remove np.float
**Rủi ro**: Lỗi khi build extensions hoặc load data
**Giải pháp**: Đã sửa tất cả np.float → np.float32

## 10. ⚠️ Cần Kiểm Tra: Config Keys Missing
**Vấn đề**: Một số config keys có thể không được định nghĩa
**Rủi ro**: KeyError khi access config
**Giải pháp**: Đã thêm MODEL và DATASET.OBSERVING vào config.py

## Checklist Trước Khi Training

- [x] DCN extensions đã được build
- [x] Pretrained backbone weights có sẵn
- [x] Config file có đầy đủ required keys
- [x] Dataset format đúng
- [x] All imports work
- [x] Device handling đúng
- [x] Channel sizes match giữa các modules
- [ ] Test với một batch nhỏ trước
- [ ] Monitor memory usage
- [ ] Check gradient flow

## Recommended Testing Order

1. **Test Dataset Loading**: `python tools/test_cross_view_dataset.py`
2. **Test Model Forward (no training)**: Tạo script test forward pass
3. **Test với batch size 1**: Đảm bảo không có shape issues
4. **Test với batch size nhỏ**: Kiểm tra memory
5. **Full training**: Sau khi tất cả tests pass

