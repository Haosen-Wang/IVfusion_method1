#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TensorFusionEvaluator 使用示例 - 针对训练中的评估
"""

import torch
import sys
import os

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, '..', '..')
test_metric_dir = os.path.join(current_dir, '..', 'test_metric')
sys.path.append(project_root)
sys.path.append(test_metric_dir)

try:
    from tensor_example import TensorFusionEvaluator
    print("✓ 成功导入 TensorFusionEvaluator")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    # 尝试直接导入
    try:
        import sys
        sys.path.append('/home/user/1024_whs/IVfusion_method1/model/test_metric')
        from tensor_example import TensorFusionEvaluator
        print("✓ 成功导入 TensorFusionEvaluator (直接路径)")
    except ImportError as e2:
        print(f"❌ 直接导入也失败: {e2}")
        exit(1)

def test_evaluator():
    """测试评估器"""
    print("🧪 TensorFusionEvaluator 测试")
    print("=" * 50)
    
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 创建评估器
    evaluator = TensorFusionEvaluator(device=device)
    
    # 创建测试数据
    batch_size = 2
    height, width = 64, 64
    
    print(f"\n创建测试数据:")
    print(f"  批次大小: {batch_size}")
    print(f"  图像尺寸: {height}x{width}")
    
    # 模拟训练中的数据
    visible = torch.rand(batch_size, 3, height, width, device=device)
    infrared = torch.rand(batch_size, 1, height, width, device=device)
    
    # 扩展红外图像到3通道以匹配可见光
    infrared_3ch = infrared.repeat(1, 3, 1, 1)
    
    # 模拟融合结果
    fusion = 0.6 * visible + 0.4 * infrared_3ch
    
    print(f"✓ 数据创建完成")
    print(f"  可见光图像: {visible.shape}, device: {visible.device}")
    print(f"  红外图像: {infrared.shape}, device: {infrared.device}")
    print(f"  融合图像: {fusion.shape}, device: {fusion.device}")
    
    # 模拟训练过程中的评估
    print(f"\n🔥 模拟训练评估:")
    
    try:
        # 1. 计算单个指标
        print("1. 计算MSE...")
        mse_values = evaluator.mse_batch(fusion, visible)
        print(f"   MSE: {mse_values}")
        
        print("2. 计算PSNR...")
        psnr_values = evaluator.psnr_batch(fusion, visible)
        print(f"   PSNR: {psnr_values}")
        
        print("3. 计算熵...")
        entropy_values = evaluator.entropy_batch(fusion)
        print(f"   熵: {entropy_values}")
        
        print("4. 计算空间频率...")
        sf_values = evaluator.spatial_frequency_batch(fusion)
        print(f"   空间频率: {sf_values}")
        
        # 2. 批量计算所有指标
        print("\n5. 批量计算所有指标...")
        batch_metrics = evaluator.calculate_metrics_batch(
            fusion, visible, infrared_3ch,
            metrics=['mse', 'psnr', 'entropy', 'sf']
        )
        
        print("   批量结果:")
        for metric, values in batch_metrics.items():
            print(f"     {metric}: {values}")
        
        # 3. 计算平均值（用于日志记录）
        print("\n6. 计算平均指标（用于训练日志）...")
        avg_metrics = evaluator.calculate_average_metrics(
            fusion, visible, infrared_3ch,
            metrics=['mse', 'psnr', 'entropy', 'sf']
        )
        
        print("   平均指标:")
        for metric, value in avg_metrics.items():
            print(f"     {metric}: {value:.6f}")
        
        # 4. 模拟训练时的使用方式
        print("\n7. 模拟训练时使用...")
        model_output = {'fusion': fusion}
        input_data = {'visible': visible, 'infrared': infrared_3ch}
        
        training_metrics = evaluator.evaluate_during_training(model_output, input_data)
        
        print("   训练评估结果:")
        for metric, value in training_metrics.items():
            print(f"     {metric}: {value:.6f}")
        
        print("\n🎉 所有测试通过！")
        print("\n💡 在训练代码中的使用方法:")
        print("""
# 在训练循环中添加：
from test_metric.tensor_example import TensorFusionEvaluator

evaluator = TensorFusionEvaluator(device='cuda:0')

# 在每个epoch或每隔几个batch评估：
def evaluate_batch(fused, visible, infrared):
    metrics = evaluator.calculate_average_metrics(
        fused, visible, infrared,
        metrics=['mse', 'psnr', 'entropy', 'sf']
    )
    return metrics

# 在训练循环中：
if batch_idx % eval_interval == 0:
    eval_metrics = evaluate_batch(fusion_output, vis_input, ir_input)
    wandb.log(eval_metrics)
        """)
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_evaluator()
