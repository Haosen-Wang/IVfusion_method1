#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch Tensor格式图像融合评价指标使用示例
支持(b,c,h,w)格式的输入
"""
import sys
import os
import importlib.util
# 添加model目录到Python路径
model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(model_dir)
import torch
import torch.nn.functional as F
import numpy as np
from test_metrics.test_metric import FusionMetrics


class TensorFusionEvaluator:
    """
    专门处理PyTorch Tensor的图像融合评价器
    """
    
    def __init__(self, device='cpu'):
        self.metrics = FusionMetrics()
        self.device = device
    
    def evaluate_fusion_batch(self, fused_tensor, vis_tensor, ir_tensor, reference_tensor=None):
        """
        评价一个batch的融合结果
        
        Args:
            fused_tensor: 融合图像 Tensor (b,c,h,w)
            vis_tensor: 可见光图像 Tensor (b,c,h,w)
            ir_tensor: 红外图像 Tensor (b,c,h,w)
            reference_tensor: 参考图像 Tensor (可选) (b,c,h,w)
        
        Returns:
            评价指标字典
        """
        # 确保所有tensor在CPU上进行评价
        if fused_tensor.is_cuda:
            fused_tensor = fused_tensor.cpu()
        if vis_tensor.is_cuda:
            vis_tensor = vis_tensor.cpu()
        if ir_tensor.is_cuda:
            ir_tensor = ir_tensor.cpu()
        if reference_tensor is not None and reference_tensor.is_cuda:
            reference_tensor = reference_tensor.cpu()
        
        # 使用批量评价方法
        metrics_result = self.metrics.calculate_batch_metrics(
            fused_tensor, vis_tensor, ir_tensor, reference_tensor
        )
        
        return metrics_result
    
    def evaluate_single_image(self, fused_tensor, vis_tensor, ir_tensor, reference_tensor=None):
        """
        评价单张图像
        
        Args:
            fused_tensor: 融合图像 Tensor (1,c,h,w) 或 (c,h,w)
            vis_tensor: 可见光图像 Tensor (1,c,h,w) 或 (c,h,w)
            ir_tensor: 红外图像 Tensor (1,c,h,w) 或 (c,h,w)
            reference_tensor: 参考图像 Tensor (可选)
            
        Returns:
            评价指标字典
        """
        # 确保是4D tensor
        if len(fused_tensor.shape) == 3:
            fused_tensor = fused_tensor.unsqueeze(0)
        if len(vis_tensor.shape) == 3:
            vis_tensor = vis_tensor.unsqueeze(0)
        if len(ir_tensor.shape) == 3:
            ir_tensor = ir_tensor.unsqueeze(0)
        if reference_tensor is not None and len(reference_tensor.shape) == 3:
            reference_tensor = reference_tensor.unsqueeze(0)
        
        # 移到CPU
        if fused_tensor.is_cuda:
            fused_tensor = fused_tensor.cpu()
        if vis_tensor.is_cuda:
            vis_tensor = vis_tensor.cpu()
        if ir_tensor.is_cuda:
            ir_tensor = ir_tensor.cpu()
        if reference_tensor is not None and reference_tensor.is_cuda:
            reference_tensor = reference_tensor.cpu()
        
        # 计算指标
        metrics_result = self.metrics.calculate_all_metrics(
            fused_tensor, vis_tensor, ir_tensor, reference_tensor
        )
        
        return metrics_result


def create_sample_data(batch_size=4, channels=1, height=256, width=256, device='cpu'):
    """
    创建示例数据
    """
    # 创建有特征的测试数据
    # 可见光图像 - 更多纹理
    vis_base = torch.randn(batch_size, channels, height, width, device=device) * 0.2 + 0.5
    
    # 添加一些结构特征
    y, x = torch.meshgrid(torch.linspace(0, 1, height, device=device), 
                         torch.linspace(0, 1, width, device=device), indexing='ij')
    pattern = 0.3 * torch.sin(10 * x) * torch.cos(10 * y)
    pattern = pattern.unsqueeze(0).unsqueeze(0).repeat(batch_size, channels, 1, 1)
    vis_images = torch.clamp(vis_base + pattern, 0, 1)
    
    # 红外图像 - 更多边缘信息
    ir_base = torch.zeros(batch_size, channels, height, width, device=device)
    # 添加一些矩形区域
    ir_base[:, :, height//4:3*height//4, width//4:3*width//4] = 0.8
    ir_base[:, :, height//3:2*height//3, width//3:2*width//3] = 0.6
    # 添加噪声
    noise = torch.randn_like(ir_base) * 0.1
    ir_images = torch.clamp(ir_base + noise, 0, 1)
    
    # 融合图像 - 加权组合
    fusion_images = 0.6 * vis_images + 0.4 * ir_images
    
    # 理想参考图像
    reference_images = torch.maximum(vis_images, ir_images) * 0.7 + \
                      torch.minimum(vis_images, ir_images) * 0.3
    
    return fusion_images, vis_images, ir_images, reference_images


def demo_batch_evaluation():
    """
    批量评价示例
    """
    print("=" * 60)
    print("批量图像融合评价示例")
    print("=" * 60)
    
    # 创建评价器
    evaluator = TensorFusionEvaluator()
    
    # 创建测试数据
    batch_size = 8
    fusion_imgs, vis_imgs, ir_imgs, ref_imgs = create_sample_data(
        batch_size=batch_size, height=128, width=128
    )
    
    print(f"输入数据形状:")
    print(f"  融合图像: {fusion_imgs.shape}")
    print(f"  可见光图像: {vis_imgs.shape}")
    print(f"  红外图像: {ir_imgs.shape}")
    print(f"  参考图像: {ref_imgs.shape}")
    
    # 批量评价
    print(f"\n批量评价结果 (batch_size={batch_size}):")
    print("-" * 50)
    
    batch_metrics = evaluator.evaluate_fusion_batch(
        fusion_imgs, vis_imgs, ir_imgs, ref_imgs
    )
    
    # 打印结果
    for metric_name, value in batch_metrics.items():
        if not metric_name.endswith('_std'):
            std_name = f"{metric_name}_std"
            std_value = batch_metrics.get(std_name, 0)
            print(f"{metric_name:15s}: {value:8.6f} ± {std_value:.6f}")


def demo_single_evaluation():
    """
    单图像评价示例
    """
    print("\n" + "=" * 60)
    print("单图像融合评价示例")
    print("=" * 60)
    
    # 创建评价器
    evaluator = TensorFusionEvaluator()
    
    # 创建单张图像数据
    fusion_img, vis_img, ir_img, ref_img = create_sample_data(
        batch_size=1, height=256, width=256
    )
    
    print(f"输入数据形状:")
    print(f"  融合图像: {fusion_img.shape}")
    print(f"  可见光图像: {vis_img.shape}")
    print(f"  红外图像: {ir_img.shape}")
    print(f"  参考图像: {ref_img.shape}")
    
    # 单图像评价
    print(f"\n单图像评价结果:")
    print("-" * 40)
    
    single_metrics = evaluator.evaluate_single_image(
        fusion_img, vis_img, ir_img, ref_img
    )
    
    # 打印结果
    print("无参考指标:")
    no_ref_metrics = ['EN', 'SF', 'SD', 'NABF']
    for metric in no_ref_metrics:
        if metric in single_metrics:
            print(f"  {metric:10s}: {single_metrics[metric]:8.6f}")
    
    print("\n有参考指标:")
    ref_metrics = ['MSE', 'PSNR', 'MS-SSIM', 'CC', 'VIF']
    for metric in ref_metrics:
        if metric in single_metrics:
            value = single_metrics[metric]
            if metric == 'PSNR':
                print(f"  {metric:10s}: {value:8.2f} dB")
            else:
                print(f"  {metric:10s}: {value:8.6f}")


def demo_gpu_evaluation():
    """
    GPU评价示例
    """
    if not torch.cuda.is_available():
        print("\nGPU不可用，跳过GPU示例")
        return
    
    print("\n" + "=" * 60)
    print("GPU图像融合评价示例")
    print("=" * 60)
    
    device = 'cuda'
    evaluator = TensorFusionEvaluator(device=device)
    
    # 在GPU上创建数据
    fusion_imgs, vis_imgs, ir_imgs, ref_imgs = create_sample_data(
        batch_size=4, height=512, width=512, device=device
    )
    
    print(f"GPU设备: {device}")
    print(f"输入数据设备: {fusion_imgs.device}")
    print(f"输入数据形状: {fusion_imgs.shape}")
    
    # GPU批量评价
    print(f"\nGPU批量评价结果:")
    print("-" * 40)
    
    batch_metrics = evaluator.evaluate_fusion_batch(
        fusion_imgs, vis_imgs, ir_imgs, ref_imgs
    )
    
    # 打印关键指标
    key_metrics = ['EN', 'SF', 'PSNR', 'MS-SSIM', 'NABF']
    for metric in key_metrics:
        if metric in batch_metrics:
            value = batch_metrics[metric]
            std_value = batch_metrics.get(f"{metric}_std", 0)
            print(f"{metric:10s}: {value:8.6f} ± {std_value:.6f}")


def demo_different_input_formats():
    """
    不同输入格式示例
    """
    print("\n" + "=" * 60)
    print("不同输入格式示例")
    print("=" * 60)
    
    evaluator = TensorFusionEvaluator()
    
    # 格式1: (b,c,h,w) - 标准格式
    print("格式1: (b,c,h,w)")
    fusion_4d = torch.randn(2, 1, 64, 64) * 0.5 + 0.5
    vis_4d = torch.randn(2, 1, 64, 64) * 0.5 + 0.5
    ir_4d = torch.randn(2, 1, 64, 64) * 0.5 + 0.5
    
    metrics_4d = evaluator.evaluate_fusion_batch(fusion_4d, vis_4d, ir_4d)
    print(f"  PSNR: {metrics_4d['PSNR']:.2f} dB")
    print(f"  EN: {metrics_4d['EN']:.6f}")
    
    # 格式2: (c,h,w) - 单张图像
    print("\n格式2: (c,h,w)")
    fusion_3d = torch.randn(1, 64, 64) * 0.5 + 0.5
    vis_3d = torch.randn(1, 64, 64) * 0.5 + 0.5
    ir_3d = torch.randn(1, 64, 64) * 0.5 + 0.5
    
    metrics_3d = evaluator.evaluate_single_image(fusion_3d, vis_3d, ir_3d)
    print(f"  PSNR: {metrics_3d['PSNR']:.2f} dB")
    print(f"  EN: {metrics_3d['EN']:.6f}")
    
    # 格式3: (b,c,h,w) RGB图像 - 会自动转换为灰度
    print("\n格式3: (b,c,h,w) RGB图像")
    fusion_rgb = torch.randn(1, 3, 64, 64) * 0.5 + 0.5
    vis_rgb = torch.randn(1, 3, 64, 64) * 0.5 + 0.5
    ir_rgb = torch.randn(1, 3, 64, 64) * 0.5 + 0.5
    
    metrics_rgb = evaluator.evaluate_single_image(fusion_rgb, vis_rgb, ir_rgb)
    print(f"  PSNR: {metrics_rgb['PSNR']:.2f} dB")
    print(f"  EN: {metrics_rgb['EN']:.6f}")


if __name__ == "__main__":
    # 运行所有示例
    demo_batch_evaluation()
    demo_single_evaluation()
    demo_gpu_evaluation()
    demo_different_input_formats()
    
    print("\n" + "=" * 60)
    print("所有示例运行完成！")
    print("=" * 60)
