# 单图测试和可视化工具

这个工具允许你对单张图像进行测试，并可视化输出结果。

## 文件说明

- `test_single_visualize.py`: 主要的单图测试类
- `example_single_test.py`: Python使用示例
- `run_single_test.sh`: Shell脚本使用示例

## 功能特性

1. ✅ 单张图像推理
2. ✅ 自动计算评估指标（融合指标和Clean指标）
3. ✅ 可视化所有输入输出图像
4. ✅ 单独保存输出图像
5. ✅ 支持两种任务：dv_i 和 di_v
6. ✅ 灵活的图像尺寸设置

## 快速开始

### 方法1: 命令行方式

```bash
python test_single_visualize.py \
    --model /path/to/model.pth \
    --i_img /path/to/infrared.png \
    --v_img /path/to/visible.png \
    --d_img /path/to/degraded.png \
    --task dv_i \
    --device cuda:0 \
    --save_dir ./output
```

### 方法2: 使用Shell脚本

1. 编辑 `run_single_test.sh`，修改路径：
```bash
MODEL_PATH="path/to/your/model.pth"
I_IMAGE="path/to/infrared.png"
V_IMAGE="path/to/visible.png"
D_IMAGE="path/to/degraded.png"
```

2. 运行脚本：
```bash
bash run_single_test.sh
```

### 方法3: Python代码方式

```python
from test_single_visualize import SingleImageTester

# 创建测试器（需要指定task）
tester = SingleImageTester(
    model_path='/path/to/model.pth',
    task='dv_i',  # 必须指定任务类型
    device='cuda:0'
)

# 测试单张图像
out, clean, fusion_metrics, clean_metrics = tester.test_single(
    i_path='/path/to/infrared.png',
    v_path='/path/to/visible.png',
    d_path='/path/to/degraded.png',
    save_dir='./output'
)

# 打印指标
tester.print_metrics(fusion_metrics, clean_metrics)
```

## 参数说明

### 命令行参数

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `--model` | 是 | 模型权重路径 | - |
| `--i_img` | 是 | 红外图像路径 | - |
| `--v_img` | 是 | 可见光图像路径 | - |
| `--d_img` | 是 | 退化图像路径 | - |
| `--task` | 否 | 任务类型 (dv_i/di_v) | dv_i |
| `--device` | 否 | 运行设备 | cuda:0 |
| `--save_dir` | 否 | 保存目录 | ./single_test_output |

### 任务类型说明

#### dv_i 任务
- **输入**：
  - 红外图像 (灰度, L模式)
  - 可见光图像 (彩色, RGB模式)
  - 退化的可见光图像 (彩色, RGB模式)
- **输出**：
  - 融合图像 (彩色)
  - Clean红外图像 (灰度)

#### di_v 任务
- **输入**：
  - 红外图像 (灰度, L模式)
  - 可见光图像 (彩色, RGB模式)
  - 退化的红外图像 (灰度, L模式)
- **输出**：
  - 融合图像 (彩色)
  - Clean可见光图像 (彩色)

## 输出说明

运行后会在指定的保存目录生成以下文件：

1. **visualization.png**: 包含所有输入输出图像和指标的综合可视化图
   - 第一行：红外图像、可见光图像、退化图像
   - 第二行：融合输出、Clean输出、评估指标

2. **fused_output.png**: 单独的融合输出图像

3. **clean_output.png**: 单独的Clean输出图像

## 评估指标

脚本会自动计算以下指标：

### 融合图像指标
- EN (熵)
- SD (标准差)
- SF (空间频率)
- MI (互信息)
- VIF (视觉信息保真度)
- Qabf (基于梯度的融合质量)
- 等等...

### Clean图像指标
- 与退化图像和参考图像的相似度指标

## 高级用法

### 自定义图像尺寸

```python
tester = SingleImageTester(model_path, device='cuda:0')

# 使用自定义尺寸 (480x640)
i_tensor = tester.preprocess_image(i_path, 'L', size=(480, 640))
v_tensor = tester.preprocess_image(v_path, 'RGB', size=(480, 640))
d_tensor = tester.preprocess_image(d_path, 'RGB', size=(480, 640))

# 推理
with torch.no_grad():
    out, clean = tester.model(i_tensor.to(device), d_tensor.to(device), 
                              device, device, device)
```

### 批量测试多对图像

参考 `example_single_test.py` 中的 `example_batch_test()` 函数。

## 示例

查看 `example_single_test.py` 获取更多使用示例：

```bash
python example_single_test.py
```

## 注意事项

1. 确保模型路径正确
2. 图像格式支持：PNG, JPG, JPEG等常见格式
3. 默认会将图像调整为 240x240 尺寸
4. 需要安装依赖：torch, PIL, matplotlib, numpy

## 故障排除

### 问题：CUDA out of memory
**解决方案**：使用CPU或减小图像尺寸
```bash
--device cpu
```

### 问题：找不到模块
**解决方案**：确保在正确的目录运行，或检查Python路径

### 问题：图像格式不支持
**解决方案**：确保图像是标准格式（PNG, JPG等）

## 依赖要求

```
torch >= 1.8.0
torchvision >= 0.9.0
Pillow >= 8.0.0
matplotlib >= 3.3.0
numpy >= 1.19.0
```

## 联系方式

如有问题，请查看项目README或提交Issue。
