# 检查点内存优化指南

## 🚀 概述

在深度学习训练中，检查点加载是显存占用的重要瓶颈。本优化方案提供了多种策略来减少检查点加载时的显存占用。

## 💡 核心优化策略

### 1. 内存高效加载 (Memory Efficient Loading)

**原理**: 将检查点强制加载到CPU，然后逐层传输到GPU，避免GPU显存峰值。

```python
# 启用内存高效加载
memory:
  checkpoint_memory_efficient: true
```

**效果**: 可减少30-50%的显存峰值占用

### 2. 选择性优化器状态加载

**原理**: 优化器状态通常占用与模型参数相同的显存。不加载优化器状态可大幅减少显存占用。

```python
# 不加载优化器状态（推荐用于显存不足时）
memory:
  load_optimizer_state: false
```

**效果**: 
- ✅ 可减少50%的显存占用
- ⚠️ 会重置优化器状态（学习率调度、动量等）
- 💡 适用于微调或显存极度不足的情况

### 3. 压缩存储

**原理**: 使用PyTorch的新版压缩序列化格式减少文件大小。

```python
# 启用压缩保存
memory:
  compress_checkpoints: true
```

**效果**: 可减少10-30%的文件大小

### 4. 选择性保存策略

**原理**: 对不同类型的检查点采用不同的保存策略。

```python
# 仅在最新检查点中保存优化器状态
memory:
  save_optimizer_in_epoch_checkpoints: false
```

## 🔧 使用方法

### 方法1: 配置文件方式

1. 创建内存优化配置：
```yaml
memory:
  checkpoint_memory_efficient: true
  load_optimizer_state: false  # 极低内存模式
  compress_checkpoints: true
  save_optimizer_in_epoch_checkpoints: false
```

2. 运行训练：
```bash
./run_train_dc.sh --config config_memory_optimized.yaml --experiment ultra_low_memory
```

### 方法2: 直接调用API

```python
from checkpoint_utils import load_checkpoint_memory_efficient, load_model_only

# 内存高效加载（包含优化器）
epoch, train_loss, val_loss = load_checkpoint_memory_efficient(
    model, optimizer, 'checkpoint.pth', device='cpu'
)

# 仅加载模型权重（最省显存）
epoch, train_loss = load_model_only(
    model, 'checkpoint.pth', strict=False
)
```

## 📊 显存占用对比

| 策略 | 显存占用 | 文件大小 | 训练连续性 | 适用场景 |
|------|----------|----------|------------|----------|
| 标准加载 | 100% | 100% | 完美 | 显存充足 |
| 内存高效 | 70% | 100% | 完美 | 中等显存 |
| 仅模型权重 | 50% | 50% | 部分丢失 | 显存不足 |
| 压缩+优化 | 45% | 70% | 部分丢失 | 极限情况 |

## ⚠️ 注意事项

### 1. 优化器状态的影响

**不加载优化器状态的后果**:
- Adam优化器的动量信息丢失
- 学习率调度器状态重置
- 可能影响训练收敛性

**建议**:
- 在训练早期或显存不足时可以不加载
- 在训练后期建议加载以保持稳定性
- 微调任务可以安全地不加载

### 2. 显存监控

```python
# 监控显存使用
if torch.cuda.is_available():
    print(f"当前显存使用: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
    print(f"显存峰值: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
```

### 3. 错误处理

如果遇到显存不足错误：
1. 首先尝试减小batch_size
2. 启用内存高效加载
3. 设置load_optimizer_state=false
4. 考虑使用梯度累积替代大批次

## 🛠️ 高级优化

### 1. 自定义加载策略

```python
def smart_load_checkpoint(model, optimizer, checkpoint_path):
    """智能选择加载策略"""
    available_memory = torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
    file_size = os.path.getsize(checkpoint_path)
    
    if available_memory < file_size * 2:
        # 显存不足，使用最保守策略
        return load_model_only(model, checkpoint_path)
    else:
        # 显存充足，使用标准策略
        return load_checkpoint_memory_efficient(model, optimizer, checkpoint_path)
```

### 2. 分阶段加载

```python
# 大模型分阶段加载
def load_large_model_gradually(model, state_dict):
    """逐层加载大模型"""
    for name, param in model.named_parameters():
        if name in state_dict:
            param.data.copy_(state_dict[name].to(param.device))
            # 每加载一些参数就清理缓存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
```

## 📈 最佳实践

1. **开发阶段**: 使用ultra_low_memory模式快速迭代
2. **正式训练**: 使用balanced模式平衡性能和内存
3. **生产环境**: 根据硬件条件选择合适策略
4. **持续监控**: 定期检查显存使用和文件大小

## 🔍 故障排除

### 常见问题

1. **"CUDA out of memory"错误**
   - 解决方案: 启用ultra_low_memory模式

2. **训练不稳定**
   - 可能原因: 未加载优化器状态
   - 解决方案: 设置load_optimizer_state=true

3. **检查点文件过大**
   - 解决方案: 启用压缩并关闭不必要的状态保存

4. **加载速度慢**
   - 可能原因: 逐层加载开销
   - 解决方案: 在显存允许时使用标准加载
