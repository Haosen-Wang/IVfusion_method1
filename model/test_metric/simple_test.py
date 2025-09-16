#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试TensorFusionEvaluator
"""

import torch
import sys
import os

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
sys.path.append(project_root)

# 导入TensorFusionEvaluator
from test_metric.tensor_example import TensorFusionEvaluator

def simple_test():
    """简单测试"""
    print("🧪 TensorFusionEvaluator 简单测试")
    print("=" * 50)
    
    # 检查CUDA可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建评估器
    evaluator = TensorFusionEvaluator(device=device)
    
    # 创建小尺寸测试数据
    batch_size = 2
    height, width = 64, 64
    
    print(f"创建测试数据: batch_size={batch_size}, size={height}x{width}")
    
    # 创建测试张量
    visible = torch.rand(batch_size, 3, height, width, device=device)
    infrared = torch.rand(batch_size, 1, height, width, device=device)
    
    # 扩展红外图像到3通道
    infrared_3ch = infrared.repeat(1, 3, 1, 1)
    
    # 创建融合图像
    fusion = 0.7 * visible + 0.3 * infrared_3ch
    
    print(f"✓ 测试数据创建完成")
    print(f"  可见光图像: {visible.shape}")
    print(f"  红外图像: {infrared.shape}")
    print(f"  融合图像: {fusion.shape}")
    
    try:
        # 计算MSE
        print("\n计算MSE...")
        mse_values = evaluator.mse_batch(fusion, visible)
        print(f"✓ MSE: {mse_values}")
        
        # 计算PSNR
        print("\n计算PSNR...")
        psnr_values = evaluator.psnr_batch(fusion, visible)
        print(f"✓ PSNR: {psnr_values}")
        
        # 计算熵
        print("\n计算熵...")
        entropy_values = evaluator.entropy_batch(fusion)
        print(f"✓ 熵: {entropy_values}")
        
        # 计算空间频率
        print("\n计算空间频率...")
        sf_values = evaluator.spatial_frequency_batch(fusion)
        print(f"✓ 空间频率: {sf_values}")
        
        # 批量计算所有指标
        print("\n批量计算所有指标...")
        all_metrics = evaluator.calculate_metrics_batch(
            fusion, visible, infrared_3ch,
            metrics=['mse', 'psnr', 'entropy', 'sf']
        )
        
        print("✓ 批量计算结果:")
        for metric, values in all_metrics.items():
            print(f"  {metric}: {values}")
        
        # 计算平均指标
        avg_metrics = evaluator.calculate_average_metrics(
            fusion, visible, infrared_3ch,
            metrics=['mse', 'psnr', 'entropy', 'sf']
        )
        
        print("\n✓ 平均指标:")
        for metric, value in avg_metrics.items():
            print(f"  {metric}: {value:.6f}")
        
        print("\n🎉 所有测试通过！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_test()
