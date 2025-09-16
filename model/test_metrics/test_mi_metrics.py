#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试互信息(MI)指标的实现
"""
import sys
import os
import numpy as np
import torch
import cv2
from test_metric import FusionMetrics
from tensor_example import TensorFusionEvaluator

def test_mi_with_synthetic_data():
    """使用合成数据测试MI指标"""
    print("=== 测试互信息指标 - 合成数据 ===")
    
    # 创建合成数据
    height, width = 256, 256
    
    # 可见光图像 - 包含纹理信息
    vis_img = np.random.rand(height, width) * 0.3 + 0.3
    x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
    pattern = 0.4 * np.sin(10 * x) * np.cos(10 * y)
    vis_img = np.clip(vis_img + pattern, 0, 1)
    
    # 红外图像 - 包含边缘信息
    ir_img = np.zeros((height, width))
    ir_img[height//4:3*height//4, width//4:3*width//4] = 0.8
    ir_img[height//3:2*height//3, width//3:2*width//3] = 0.6
    # 添加噪声
    noise = np.random.randn(height, width) * 0.1
    ir_img = np.clip(ir_img + noise, 0, 1)
    
    # 融合图像 - 不同融合策略
    # 策略1: 简单平均
    fusion1 = 0.5 * vis_img + 0.5 * ir_img
    
    # 策略2: 保留更多可见光信息
    fusion2 = 0.7 * vis_img + 0.3 * ir_img
    
    # 策略3: 保留更多红外信息  
    fusion3 = 0.3 * vis_img + 0.7 * ir_img
    
    # 策略4: 最大值融合
    fusion4 = np.maximum(vis_img, ir_img)
    
    # 初始化评估器
    evaluator = FusionMetrics()
    
    print(f"图像尺寸: {height}x{width}")
    print(f"可见光图像统计: 均值={vis_img.mean():.3f}, 方差={vis_img.var():.3f}")
    print(f"红外图像统计: 均值={ir_img.mean():.3f}, 方差={ir_img.var():.3f}")
    print()
    
    # 测试不同融合策略的MI指标
    strategies = [
        ("简单平均", fusion1),
        ("偏向可见光", fusion2), 
        ("偏向红外", fusion3),
        ("最大值融合", fusion4)
    ]
    
    for name, fusion in strategies:
        mi = evaluator.mutual_information(fusion, vis_img, ir_img)
        nmi = evaluator.normalized_mutual_information(fusion, vis_img, ir_img)
        
        print(f"{name}:")
        print(f"  MI = {mi:.4f}")
        print(f"  NMI = {nmi:.4f}")
        print()

def test_mi_with_tensor():
    """使用PyTorch Tensor测试MI指标"""
    print("=== 测试互信息指标 - PyTorch Tensor ===")
    
    # 创建Tensor评估器
    evaluator = TensorFusionEvaluator(device='cpu')
    
    # 创建测试数据
    batch_size, channels, height, width = 2, 1, 128, 128
    device = 'cpu'
    
    # 可见光图像
    vis_tensor = torch.rand(batch_size, channels, height, width, device=device) * 0.5 + 0.3
    
    # 红外图像
    ir_tensor = torch.zeros(batch_size, channels, height, width, device=device)
    ir_tensor[:, :, height//4:3*height//4, width//4:3*width//4] = 0.8
    ir_tensor += torch.randn_like(ir_tensor) * 0.1
    ir_tensor = torch.clamp(ir_tensor, 0, 1)
    
    # 融合图像
    fusion_tensor = 0.6 * vis_tensor + 0.4 * ir_tensor
    
    # 计算指标
    metrics = evaluator.evaluate_fusion_batch(fusion_tensor, vis_tensor, ir_tensor)
    
    print(f"Tensor形状: {fusion_tensor.shape}")
    print("互信息指标结果:")
    if 'MI' in metrics:
        print(f"  MI = {metrics['MI']:.4f}")
    if 'NMI' in metrics:
        print(f"  NMI = {metrics['NMI']:.4f}")
    
    # 显示所有指标
    print("\n所有评估指标:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

def test_mi_edge_cases():
    """测试MI指标的边界情况"""
    print("=== 测试互信息指标 - 边界情况 ===")
    
    evaluator = FusionMetrics()
    height, width = 64, 64
    
    # 情况1: 完全相同的图像
    img_same = np.random.rand(height, width)
    mi_same = evaluator.mutual_information(img_same, img_same, img_same)
    nmi_same = evaluator.normalized_mutual_information(img_same, img_same, img_same)
    print(f"完全相同图像: MI={mi_same:.4f}, NMI={nmi_same:.4f}")
    
    # 情况2: 完全不相关的图像
    img1 = np.random.rand(height, width)
    img2 = np.random.rand(height, width) 
    img3 = np.random.rand(height, width)
    mi_random = evaluator.mutual_information(img1, img2, img3)
    nmi_random = evaluator.normalized_mutual_information(img1, img2, img3)
    print(f"随机不相关图像: MI={mi_random:.4f}, NMI={nmi_random:.4f}")
    
    # 情况3: 线性相关的图像
    base_img = np.random.rand(height, width)
    img_a = base_img * 0.8 + 0.1
    img_b = base_img * 0.6 + 0.2
    fusion_linear = base_img * 0.7 + 0.15
    mi_linear = evaluator.mutual_information(fusion_linear, img_a, img_b)
    nmi_linear = evaluator.normalized_mutual_information(fusion_linear, img_a, img_b)
    print(f"线性相关图像: MI={mi_linear:.4f}, NMI={nmi_linear:.4f}")
    
    # 情况4: 常数图像
    img_const = np.ones((height, width)) * 0.5
    mi_const = evaluator.mutual_information(img_const, img_const, img_const)
    nmi_const = evaluator.normalized_mutual_information(img_const, img_const, img_const)
    print(f"常数图像: MI={mi_const:.4f}, NMI={nmi_const:.4f}")

if __name__ == "__main__":
    print("开始测试互信息(MI)指标...")
    print()
    
    # 测试合成数据
    test_mi_with_synthetic_data()
    
    # 测试Tensor格式
    test_mi_with_tensor()
    
    # 测试边界情况
    test_mi_edge_cases()
    
    print("所有测试完成!")
