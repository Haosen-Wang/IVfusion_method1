# 像素值分析和亮区域移除工具

## 🎯 功能说明

这个工具可以帮你：
1. ✅ 分析图像的像素值分布
2. ✅ 找出灰白色（高亮）区域的具体像素值
3. ✅ 提供多种方法消除这些区域
4. ✅ 可视化对比不同处理方法的效果

## 📊 分析结果（针对你的红外图像）

### 像素值统计
- **整体范围**: 0.0 ~ 0.8039 (0-255 scale: 0 ~ 205)
- **平均值**: 0.1230 (约 31/255)
- **标准差**: 0.1485

### 灰白色区域特征
根据不同阈值的分析：

| 百分位数 | 像素值(0-1) | 像素值(0-255) | 占比 | 说明 |
|---------|------------|--------------|------|------|
| P90 | 0.2824 | 72 | 10% | 较亮区域 |
| P95 | 0.4392 | 112 | 5% | 明显的亮区域 |
| P98 | 0.6431 | 164 | 2% | 非常亮的区域 |
| P99 | 0.6784 | 173 | 1% | 极亮区域 |

**推荐**：使用 **P95 (0.4392)** 或 **P98 (0.6431)** 作为阈值，可以有效移除灰白色部分而不影响正常的行人和背景。

## 🚀 使用方法

### 方法1: 命令行方式

```bash
# 基础分析（使用P95阈值，移除最亮的5%）
python analyze_pixel_values.py \
    --image /path/to/infrared.jpg \
    --mode L \
    --percentile 95 \
    --save_dir ./output \
    --compare

# 使用固定阈值
python analyze_pixel_values.py \
    --image /path/to/infrared.jpg \
    --mode L \
    --threshold 0.7 \
    --save_dir ./output \
    --method inpaint

# 只使用特定方法处理
python analyze_pixel_values.py \
    --image /path/to/infrared.jpg \
    --mode L \
    --percentile 98 \
    --method zero \
    --save_dir ./output
```

### 方法2: 使用快速脚本

```bash
# 使用预设的脚本进行多种阈值对比
bash run_pixel_analysis.sh
```

### 方法3: Python代码

```python
from analyze_pixel_values import PixelAnalyzer

# 初始化分析器
analyzer = PixelAnalyzer('/path/to/infrared.jpg', convert_mode='L')

# 分析百分位数
analyzer.analyze_percentiles()

# 分析直方图
analyzer.analyze_histogram(save_path='histogram.png')

# 可视化亮区域
analyzer.visualize_bright_regions(percentile=95, save_path='bright_regions.png')

# 移除亮区域
processed = analyzer.remove_bright_regions(
    percentile=95,
    method='inpaint',  # 或 'zero', 'mean', 'median'
    save_path='processed.png'
)

# 比较所有方法
results = analyzer.compare_methods(percentile=95, save_dir='./comparison')
```

## 📋 参数说明

### 命令行参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--image` | 图像路径 | 必需 | `/path/to/image.jpg` |
| `--mode` | 图像模式 | `L` | `L` (灰度) 或 `RGB` |
| `--threshold` | 固定阈值 (0-1) | 0.7 | `0.5`, `0.7`, `0.8` |
| `--percentile` | 百分位数阈值 | None | `90`, `95`, `98` |
| `--method` | 移除方法 | `zero` | 见下表 |
| `--value` | 自定义替换值 | 0.0 | `0.0`, `0.1` |
| `--save_dir` | 保存目录 | `./analysis_output` | `./results` |
| `--compare` | 比较所有方法 | False | 添加此标志 |

### 移除方法说明

| 方法 | 说明 | 适用场景 | 效果 |
|------|------|---------|------|
| `zero` | 设置为黑色 (0) | 完全移除亮区域 | ⭐⭐⭐ 简单直接 |
| `mean` | 设置为图像均值 | 保持整体亮度 | ⭐⭐ 可能有接缝 |
| `median` | 设置为中位数 | 鲁棒性好 | ⭐⭐ 可能有接缝 |
| `value` | 自定义值 | 精确控制 | ⭐⭐ 需要调参 |
| `inpaint` | 图像修复算法 | 自然过渡 | ⭐⭐⭐⭐⭐ **推荐** |

**推荐使用 `inpaint` 方法**，它会使用周围像素智能填充亮区域，效果最自然。

## 📊 输出文件

运行后会生成以下文件：

1. **histogram.png** - 像素值分布直方图
2. **bright_regions.png** - 亮区域可视化（原图、掩码、叠加）
3. **methods_comparison.png** - 不同方法的效果对比
4. **processed_*.png** - 各种方法处理后的图像

## 💡 使用建议

### 针对你的红外图像（行人检测场景）

1. **推荐阈值**：使用 **P95 (值≈0.44)** 或 **P98 (值≈0.64)**
   - P95: 移除最亮的5%，保留大部分正常区域
   - P98: 只移除极亮的2%，更保守

2. **推荐方法**：
   - 首选：`inpaint` - 自然填充
   - 备选：`zero` - 直接清除

3. **完整命令示例**：
```bash
# 推荐配置
python analyze_pixel_values.py \
    --image /data/1024whs_data/DeMMI-RF/Train/infrared/noise50/LLVIP/010066.jpg \
    --mode L \
    --percentile 95 \
    --method inpaint \
    --save_dir ./processed_output
```

### 实际应用场景

**场景1: 快速测试**
```bash
# 比较所有方法，找出最佳效果
python analyze_pixel_values.py --image image.jpg --mode L --percentile 95 --compare
```

**场景2: 批量处理**
```python
from analyze_pixel_values import PixelAnalyzer
import os

image_dir = '/path/to/images'
output_dir = '/path/to/output'

for img_name in os.listdir(image_dir):
    if img_name.endswith('.jpg'):
        analyzer = PixelAnalyzer(os.path.join(image_dir, img_name), 'L')
        processed = analyzer.remove_bright_regions(
            percentile=95,
            method='inpaint',
            save_path=os.path.join(output_dir, f'processed_{img_name}')
        )
```

**场景3: 调整阈值**
```bash
# 如果P95移除太多，尝试P98
python analyze_pixel_values.py --image image.jpg --percentile 98 --method inpaint

# 如果P95移除太少，尝试P90
python analyze_pixel_values.py --image image.jpg --percentile 90 --method inpaint

# 使用固定阈值精确控制
python analyze_pixel_values.py --image image.jpg --threshold 0.6 --method inpaint
```

## 🔍 问题排查

### 问题1: 移除了太多区域
**解决方案**: 提高阈值
```bash
# 从P95改为P98
--percentile 98
# 或使用更高的固定阈值
--threshold 0.75
```

### 问题2: 还有灰白色残留
**解决方案**: 降低阈值
```bash
# 从P95改为P90
--percentile 90
# 或使用更低的固定阈值
--threshold 0.4
```

### 问题3: 处理效果不自然
**解决方案**: 更换方法
```bash
# 尝试inpaint方法
--method inpaint
```

## 📈 性能提示

- 灰度图像处理速度快
- RGB图像需要处理3个通道
- `inpaint` 方法较慢但效果最好
- 建议先用小图测试，找到最佳参数后再批量处理

## 📞 技术细节

### 像素值范围说明
- 工具内部使用 **0-1** 范围（浮点数）
- 显示时也会标注 **0-255** 范围（整数）
- 保存图像时自动转换为0-255范围

### 百分位数解释
- P90 = 90% 的像素值 ≤ 此值，10% > 此值
- P95 = 95% 的像素值 ≤ 此值，5% > 此值
- P99 = 99% 的像素值 ≤ 此值，1% > 此值

## 🎯 总结

对于你的红外图像：
1. **灰白色区域像素值**: 大约在 **0.44-0.80** (112-205/255)
2. **推荐阈值**: **P95 (0.44)** 或 **P98 (0.64)**
3. **推荐方法**: **inpaint**
4. **预期效果**: 移除灰白色高亮区域，保留正常的行人和场景信息
