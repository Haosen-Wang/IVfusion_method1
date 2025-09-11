# 图像融合评价指标库

这个库实现了常用的图像融合评价指标，包括有参考和无参考指标。**支持numpy数组和PyTorch Tensor输入，特别针对(b,c,h,w)格式进行了优化。**

## 实现的指标

### 有参考指标 (需要参考图像)
- **MSE** (Mean Square Error): 均方误差
- **PSNR** (Peak Signal-to-Noise Ratio): 峰值信噪比
- **MS-SSIM** (Multi-Scale Structural Similarity Index): 多尺度结构相似性指数
- **CC** (Correlation Coefficient): 相关系数
- **VIF** (Visual Information Fidelity): 视觉信息保真度

### 无参考指标 (不需要参考图像)
- **EN** (Entropy): 信息熵
- **SF** (Spatial Frequency): 空间频率
- **SD** (Standard Deviation): 标准差
- **NABF** (Noise and Artifact Blind/referenceless): 噪声和伪影盲评价

## 支持的输入格式

✅ **PyTorch Tensor**:
- `(b,c,h,w)` - 批量图像，标准深度学习格式
- `(c,h,w)` - 单张图像
- `(h,w)` - 灰度图像

✅ **NumPy 数组**:
- `(h,w,c)` - OpenCV格式
- `(h,w)` - 灰度图像

✅ **自动处理**:
- GPU/CPU张量自动转换
- 彩色到灰度自动转换
- 数值范围自动归一化

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. PyTorch Tensor输入 (推荐)

```python
import torch
from test_metric import FusionMetrics

# 创建评价指标实例
metrics = FusionMetrics()

# 输入Tensor格式 (b,c,h,w)
batch_size, channels, height, width = 4, 1, 256, 256
fused_tensor = torch.randn(batch_size, channels, height, width)
vis_tensor = torch.randn(batch_size, channels, height, width)
ir_tensor = torch.randn(batch_size, channels, height, width)

# 方法1: 批量计算 - 返回平均值和标准差
batch_metrics = metrics.calculate_batch_metrics(fused_tensor, vis_tensor, ir_tensor)
for metric_name, value in batch_metrics.items():
    if not metric_name.endswith('_std'):
        std_value = batch_metrics.get(f"{metric_name}_std", 0)
        print(f"{metric_name}: {value:.6f} ± {std_value:.6f}")

# 方法2: 单张图像计算
single_metrics = metrics.calculate_all_metrics(
    fused_tensor[0:1], vis_tensor[0:1], ir_tensor[0:1]
)
```

### 2. 专用Tensor评价器

```python
from tensor_example import TensorFusionEvaluator

# 创建专用评价器
evaluator = TensorFusionEvaluator(device='cuda')  # 支持GPU

# 批量评价
batch_metrics = evaluator.evaluate_fusion_batch(
    fused_tensor, vis_tensor, ir_tensor
)

# 单图像评价
single_metrics = evaluator.evaluate_single_image(
    fused_tensor[0], vis_tensor[0], ir_tensor[0]
)
```

### 3. 传统numpy数组输入

```python
import cv2
from test_metric import FusionMetrics

metrics = FusionMetrics()

# 加载图像
vis_img = cv2.imread('visible.png', cv2.IMREAD_GRAYSCALE)
ir_img = cv2.imread('infrared.png', cv2.IMREAD_GRAYSCALE)
fusion_img = cv2.imread('fusion.png', cv2.IMREAD_GRAYSCALE)

# 计算指标
all_metrics = metrics.calculate_all_metrics(fusion_img, vis_img, ir_img)
```

### 4. 命令行使用

```bash
# 单张图像评价
python example_usage.py single visible.png infrared.png fusion.png

# 批量评价数据集
python example_usage.py batch /path/to/dataset

# Tensor格式示例
python tensor_example.py
```

## 高级功能

### GPU加速支持

```python
# GPU上的张量会自动转换为CPU进行计算
fused_gpu = torch.randn(4, 1, 512, 512).cuda()
vis_gpu = torch.randn(4, 1, 512, 512).cuda()
ir_gpu = torch.randn(4, 1, 512, 512).cuda()

# 支持GPU张量输入
metrics = FusionMetrics()
results = metrics.calculate_batch_metrics(fused_gpu, vis_gpu, ir_gpu)
```

### 批量统计分析

```python
# 批量计算会返回每个指标的平均值和标准差
batch_metrics = metrics.calculate_batch_metrics(fused_batch, vis_batch, ir_batch)

# 获取统计信息
psnr_mean = batch_metrics['PSNR']
psnr_std = batch_metrics['PSNR_std']
print(f"PSNR: {psnr_mean:.2f} ± {psnr_std:.2f} dB")
```

### 不同输入尺寸处理

```python
# 自动处理不同尺寸
small_img = torch.randn(1, 1, 64, 64)    # 小图像
large_img = torch.randn(1, 1, 1024, 1024)  # 大图像

# 都可以正常计算
small_metrics = metrics.calculate_all_metrics(small_img, small_img, small_img)
large_metrics = metrics.calculate_all_metrics(large_img, large_img, large_img)
```

## 指标说明

### 有参考指标

1. **MSE (均方误差)**
   - 值越小越好
   - 衡量融合图像与参考图像的像素级差异

2. **PSNR (峰值信噪比)**
   - 值越大越好，单位dB
   - 基于MSE计算，常用于图像质量评价

3. **MS-SSIM (多尺度结构相似性)**
   - 值越大越好，范围[0,1]
   - 考虑亮度、对比度和结构信息

4. **CC (相关系数)**
   - 值越大越好，范围[-1,1]
   - 衡量融合图像与参考图像的线性相关性

5. **VIF (视觉信息保真度)**
   - 值越大越好
   - 基于人类视觉系统的信息理论指标

### 无参考指标

1. **EN (信息熵)**
   - 值越大越好
   - 衡量图像包含的信息量

2. **SF (空间频率)**
   - 值越大越好
   - 衡量图像的细节丰富程度

3. **SD (标准差)**
   - 值越大越好
   - 衡量图像的对比度

4. **NABF**
   - 值越小越好
   - 衡量融合图像相对于源图像的噪声和伪影

## 注意事项

1. 所有图像会自动转换为灰度图并归一化到[0,1]范围
2. 如果没有提供参考图像，有参考指标会使用源图像的加权平均作为近似参考
3. 图像尺寸会自动调整以保持一致
4. VIF指标使用简化实现，完整版本需要更复杂的小波变换

## 输出示例

```
图像融合评价指标结果:
----------------------------------------
无参考指标:
  熵 (EN):           7.234567
  空间频率 (SF):      0.123456
  标准差 (SD):        0.234567
  NABF:             0.045678

有参考指标:
  均方误差 (MSE):     0.001234
  峰值信噪比 (PSNR):  29.12 dB
  多尺度SSIM:        0.876543
  相关系数 (CC):      0.934567
  视觉信息保真度:      0.567890
```
