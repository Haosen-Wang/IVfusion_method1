# 融合图像红外背景移除功能说明

## 🎯 功能概述

已在 `test_single_visualize.py` 中添加了**自动移除红外图像亮背景**的功能。该功能可以：

1. ✅ 检测红外图像中的灰白色亮区域
2. ✅ 从融合图像中像素级减去这些区域的影响
3. ✅ 支持多种移除方法
4. ✅ 自动可视化对比原始和处理后的结果
5. ✅ 计算处理前后的指标对比

## 📊 原理说明

### 检测方法
根据之前的像素值分析，红外图像中的灰白色背景具有以下特征：
- **P95阈值**: 0.44 (112/255) - 移除最亮的5%
- **P98阈值**: 0.64 (164/255) - 移除最亮的2%  
- **P99阈值**: 0.68 (173/255) - 移除最亮的1%

### 处理方法

#### 1. `subtract` (像素级相减) ⭐推荐
- **原理**: 根据红外图像中亮区域的像素值，按比例从融合图像中减去
- **公式**: `processed = fused - (infrared * mask * strength)`
- **优点**: 保留细节，效果自然
- **适用**: 大多数场景

#### 2. `mask` (直接屏蔽)
- **原理**: 将亮区域替换为融合图像的平均值
- **效果**: 完全消除亮区域，可能有明显边界
- **适用**: 需要完全移除背景的场景

#### 3. `blend` (混合处理)
- **原理**: 降低亮区域的强度，保留部分信息
- **效果**: 温和处理，保留更多原始信息
- **适用**: 需要保留部分背景信息的场景

## 🚀 使用方法

### 基础用法

```bash
# 默认启用背景移除（P95阈值，subtract方法）
python test_single_visualize.py \
    --model /path/to/model.pth \
    --i_img /path/to/infrared.jpg \
    --v_img /path/to/visible.jpg \
    --d_img /path/to/degraded.jpg \
    --task dv_i \
    --device cuda:0 \
    --save_dir ./output
```

### 自定义参数

```bash
# 使用P98阈值（更严格）
python test_single_visualize.py \
    --model /path/to/model.pth \
    --i_img /path/to/infrared.jpg \
    --v_img /path/to/visible.jpg \
    --d_img /path/to/degraded.jpg \
    --task dv_i \
    --device cuda:0 \
    --save_dir ./output \
    --bg_percentile 98 \
    --bg_method subtract
```

### 禁用背景移除

```bash
# 如果不想使用背景移除功能
python test_single_visualize.py \
    --model /path/to/model.pth \
    --i_img /path/to/infrared.jpg \
    --v_img /path/to/visible.jpg \
    --d_img /path/to/degraded.jpg \
    --task dv_i \
    --device cuda:0 \
    --save_dir ./output \
    --no_remove_ir_bg
```

## 📋 参数说明

### 新增参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--remove_ir_bg` | flag | True | 启用红外背景移除（默认） |
| `--no_remove_ir_bg` | flag | - | 禁用红外背景移除 |
| `--bg_percentile` | float | 95 | 背景检测阈值（百分位数） |
| `--bg_method` | str | subtract | 背景移除方法 |

### 阈值推荐

| 百分位数 | 阈值(0-1) | 阈值(0-255) | 移除比例 | 适用场景 |
|---------|----------|------------|---------|---------|
| P90 | ~0.28 | ~72 | 10% | 移除较多区域 |
| **P95** | ~0.44 | ~112 | 5% | ⭐推荐，平衡 |
| **P98** | ~0.64 | ~164 | 2% | ⭐保守，精确 |
| P99 | ~0.68 | ~173 | 1% | 只移除极亮区域 |

### 方法对比

| 方法 | 效果 | 速度 | 推荐度 |
|------|------|------|--------|
| `subtract` | 自然，保留细节 | 快 | ⭐⭐⭐⭐⭐ |
| `mask` | 完全移除，可能有边界 | 快 | ⭐⭐⭐ |
| `blend` | 温和，保留部分信息 | 快 | ⭐⭐⭐⭐ |

## 📊 输出文件

运行后会生成以下文件：

### 图像文件
1. **fused_output.png** - 原始融合输出
2. **fused_output_bg_removed.png** - 背景移除后的融合输出 ⭐
3. **clean_output.png** - Clean输出
4. **visualization.png** - 综合可视化（3行4列布局）

### 可视化布局（启用背景移除时）

```
第一行：输入图像
[Infrared] [Visible] [Degraded] [IR Bright Mask]

第二行：输出对比
[Original Fused] [BG Removed Fused] [Clean] [Difference]

第三行：指标对比
[Original Metrics] [BG Removed Metrics] [Clean Metrics] [...]
```

## 💡 实际测试结果

### 测试配置
- 模型: Train_DIV_LLVIP_visible
- 图像: 010066.jpg (LLVIP数据集)
- 任务: dv_i

### P95阈值结果
```
检测到亮区域: 2869/57600 (4.98%)
阈值: 0.4157
方法: subtract
```

### P98阈值结果
```
检测到亮区域: 1134/57600 (1.97%)
阈值: 0.6275
方法: subtract
```

## 🔧 Python代码示例

```python
from test_single_visualize import SingleImageTester

# 创建测试器
tester = SingleImageTester(
    model_path='/path/to/model.pth',
    task='dv_i',
    device='cuda:0'
)

# 测试（启用背景移除）
out, clean, fusion_metrics, clean_metrics, out_processed = tester.test_single(
    i_path='/path/to/infrared.jpg',
    v_path='/path/to/visible.jpg',
    d_path='/path/to/degraded.jpg',
    save_dir='./output',
    remove_ir_background=True,  # 启用背景移除
    bg_percentile=95,           # P95阈值
    bg_method='subtract'        # 使用subtract方法
)

# out_processed 是背景移除后的融合图像
```

## 🎯 推荐配置

### 场景1: 标准处理（推荐）
```bash
--bg_percentile 95 --bg_method subtract
```
- 移除最亮的5%区域
- 平衡效果和保留细节

### 场景2: 保守处理
```bash
--bg_percentile 98 --bg_method subtract
```
- 只移除最亮的2%区域
- 更精确，避免误移除

### 场景3: 温和处理
```bash
--bg_percentile 95 --bg_method blend
```
- 降低亮度而不是完全移除
- 保留更多原始信息

### 场景4: 完全移除
```bash
--bg_percentile 90 --bg_method mask
```
- 移除较多区域（10%）
- 完全屏蔽亮区域

## 📈 效果分析

### 优点
1. ✅ **自动检测**: 无需手动标注亮区域
2. ✅ **像素级处理**: 精确控制每个像素的修改
3. ✅ **保留细节**: subtract方法保留行人等关键信息
4. ✅ **实时可视化**: 自动生成对比图
5. ✅ **灵活调节**: 多种阈值和方法可选

### 注意事项
1. ⚠️ 阈值太低可能移除正常区域（如行人）
2. ⚠️ 阈值太高可能无法完全移除灰白色背景
3. ⚠️ 建议先用P95或P98测试，再根据效果调整

## 🔍 调试技巧

### 如果背景没有完全移除
```bash
# 降低阈值
--bg_percentile 90  # 从95降到90

# 或使用mask方法
--bg_method mask
```

### 如果移除了太多区域
```bash
# 提高阈值
--bg_percentile 98  # 从95提高到98

# 或使用blend方法
--bg_method blend
```

### 查看检测的亮区域
可视化结果中的第一行第四个位置显示了检测到的亮区域掩码（红色热图）

## 📞 技术支持

### 相关文件
- `test_single_visualize.py` - 主测试脚本
- `analyze_pixel_values.py` - 像素分析工具
- `PIXEL_ANALYSIS_README.md` - 像素分析文档

### 相关方法
- `detect_bright_regions()` - 检测亮区域
- `remove_bright_background()` - 移除背景
- `visualize_results()` - 可视化结果

## 🎉 总结

该功能已成功集成到单图测试工具中，可以：
1. **自动识别**红外图像中的灰白色背景（像素值 > P95/P98）
2. **智能移除**融合图像中对应区域的影响
3. **保留细节**使用像素级相减，不影响正常内容
4. **实时对比**生成原始和处理后的结果对比

**推荐配置**: `--bg_percentile 95 --bg_method subtract`

这将移除红外图像中最亮的5%区域，并从融合图像中按比例减去其影响，效果自然且不损失细节！
