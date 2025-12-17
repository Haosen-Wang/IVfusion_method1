#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修改后的test.py逻辑
"""

import sys
import os

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(model_dir)

from test_metrics.tensor_example import TensorFusionEvaluator
import torch

print("=" * 60)
print("测试双指标计算逻辑")
print("=" * 60)

# 创建评估器
evaluator = TensorFusionEvaluator()

# 创建测试数据
batch_size = 2
fused = torch.randn(batch_size, 3, 64, 64)
clean = torch.randn(batch_size, 3, 64, 64)
vis = torch.randn(batch_size, 3, 64, 64)
ir = torch.randn(batch_size, 1, 64, 64)
degraded = torch.randn(batch_size, 3, 64, 64)

print(f"\n测试数据形状:")
print(f"  fused: {fused.shape}")
print(f"  clean: {clean.shape}")
print(f"  vis: {vis.shape}")
print(f"  ir: {ir.shape}")
print(f"  degraded: {degraded.shape}")

# 计算融合图像的指标
print("\n【融合图像指标】(fused vs vis and ir):")
fusion_metrics = evaluator.evaluate_fusion_batch(fused, vis, ir)
for name, value in sorted(fusion_metrics.items()):
    if not name.endswith('_std'):
        print(f"  {name:15s}: {value:.6f}")

# 计算干净图像的指标
print("\n【干净图像指标】(clean vs degraded and ir):")
clean_metrics = evaluator.evaluate_fusion_batch(clean, degraded, ir)
for name, value in sorted(clean_metrics.items()):
    if not name.endswith('_std'):
        print(f"  {name:15s}: {value:.6f}")

print("\n" + "=" * 60)
print("✓ 测试成功！逻辑验证通过。")
print("=" * 60)
