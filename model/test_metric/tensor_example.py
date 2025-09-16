#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
支持PyTorch Tensor输入的图像融合评价指标

专门为深度学习训练中的批量评估设计
输入格式: (B, C, H, W) PyTorch Tensor
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple, Dict, List
import sys
import os

# 添加test_metric目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
test_metric_dir = os.path.join(os.path.dirname(current_dir), 'test_metric')
sys.path.append(test_metric_dir)

from test_metric import FusionMetrics


class TensorFusionEvaluator:
    """
    专门处理PyTorch Tensor的图像融合评价指标
    支持批量处理和GPU加速计算
    """
    
    def __init__(self, device: str = 'cpu'):
        """
        初始化评价器
        
        Args:
            device: 计算设备 ('cpu' 或 'cuda:x')
        """
        self.device = device
        self.base_metrics = FusionMetrics()
        
    def _validate_tensor_format(self, tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
        """
        验证并标准化张量格式
        
        Args:
            tensor: 输入张量
            name: 张量名称（用于错误提示）
            
        Returns:
            标准化后的张量 (B, C, H, W)
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} 必须是 torch.Tensor 类型，当前类型: {type(tensor)}")
        
        # 确保在正确的设备上
        if tensor.device != torch.device(self.device):
            tensor = tensor.to(self.device)
        
        # 处理不同的输入维度
        if tensor.dim() == 2:  # (H, W)
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # -> (1, 1, H, W)
        elif tensor.dim() == 3:  # (C, H, W) 或 (B, H, W)
            if tensor.shape[0] <= 4:  # 假设是 (C, H, W)
                tensor = tensor.unsqueeze(0)  # -> (1, C, H, W)
            else:  # 假设是 (B, H, W)
                tensor = tensor.unsqueeze(1)  # -> (B, 1, H, W)
        elif tensor.dim() == 4:  # (B, C, H, W)
            pass  # 已经是正确格式
        else:
            raise ValueError(f"{name} 的维度不正确: {tensor.shape}")
        
        # 确保数据类型为float32
        if tensor.dtype != torch.float32:
            tensor = tensor.float()
        
        # 归一化到[0,1]范围
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        
        return tensor
    
    def _tensor_to_numpy_batch(self, tensor: torch.Tensor) -> List[np.ndarray]:
        """
        将张量批次转换为numpy数组列表
        
        Args:
            tensor: 输入张量 (B, C, H, W)
            
        Returns:
            numpy数组列表，每个数组为 (H, W)
        """
        # 转移到CPU并转换为numpy
        tensor_cpu = tensor.detach().cpu()
        
        # 转换为灰度图（如果是多通道）
        if tensor_cpu.shape[1] == 3:  # RGB
            # 使用加权平均转换为灰度
            weights = torch.tensor([0.299, 0.587, 0.114], dtype=tensor_cpu.dtype)
            tensor_cpu = torch.sum(tensor_cpu * weights.view(1, 3, 1, 1), dim=1, keepdim=True)
        elif tensor_cpu.shape[1] > 1:  # 多通道但不是RGB
            # 取第一个通道
            tensor_cpu = tensor_cpu[:, 0:1, :, :]
        
        # 压缩通道维度并转换为numpy
        tensor_cpu = tensor_cpu.squeeze(1)  # (B, H, W)
        
        # 转换为列表
        numpy_list = []
        for i in range(tensor_cpu.shape[0]):
            numpy_list.append(tensor_cpu[i].numpy())
        
        return numpy_list
    
    def mse_batch(self, fused: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """
        批量计算MSE
        
        Args:
            fused: 融合图像 (B, C, H, W)
            reference: 参考图像 (B, C, H, W)
            
        Returns:
            MSE值 (B,)
        """
        fused = self._validate_tensor_format(fused, "fused")
        reference = self._validate_tensor_format(reference, "reference")
        
        # 确保尺寸匹配
        if fused.shape != reference.shape:
            reference = F.interpolate(reference, size=fused.shape[2:], mode='bilinear', align_corners=False)
        
        # 计算MSE
        mse = torch.mean((fused - reference) ** 2, dim=[1, 2, 3])
        return mse
    
    def psnr_batch(self, fused: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """
        批量计算PSNR
        
        Args:
            fused: 融合图像 (B, C, H, W)
            reference: 参考图像 (B, C, H, W)
            
        Returns:
            PSNR值 (B,)
        """
        mse = self.mse_batch(fused, reference)
        
        # 避免除零
        mse = torch.clamp(mse, min=1e-10)
        
        # 计算PSNR
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr
    
    def ssim_batch(self, fused: torch.Tensor, reference: torch.Tensor, 
                   window_size: int = 11) -> torch.Tensor:
        """
        批量计算SSIM
        
        Args:
            fused: 融合图像 (B, C, H, W)
            reference: 参考图像 (B, C, H, W)
            window_size: 窗口大小
            
        Returns:
            SSIM值 (B,)
        """
        fused = self._validate_tensor_format(fused, "fused")
        reference = self._validate_tensor_format(reference, "reference")
        
        # 确保尺寸匹配
        if fused.shape != reference.shape:
            reference = F.interpolate(reference, size=fused.shape[2:], mode='bilinear', align_corners=False)
        
        # 转换为灰度图
        if fused.shape[1] == 3:
            weights = torch.tensor([0.299, 0.587, 0.114], device=fused.device, dtype=fused.dtype)
            fused = torch.sum(fused * weights.view(1, 3, 1, 1), dim=1, keepdim=True)
        if reference.shape[1] == 3:
            weights = torch.tensor([0.299, 0.587, 0.114], device=reference.device, dtype=reference.dtype)
            reference = torch.sum(reference * weights.view(1, 3, 1, 1), dim=1, keepdim=True)
        
        return self._ssim_tensor(fused, reference, window_size)
    
    def _ssim_tensor(self, img1: torch.Tensor, img2: torch.Tensor, 
                     window_size: int = 11) -> torch.Tensor:
        """
        张量版本的SSIM计算
        """
        # 创建高斯窗口
        def gaussian_kernel(size: int, sigma: float = 1.5) -> torch.Tensor:
            coords = torch.arange(size, dtype=torch.float32) - size // 2
            g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
            g = g / g.sum()
            return g.outer(g).unsqueeze(0).unsqueeze(0).to(img1.device)
        
        window = gaussian_kernel(window_size)
        
        # SSIM常数
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # 计算均值
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=1)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=1)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        # 计算方差和协方差
        sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=1) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=1) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=1) - mu1_mu2
        
        # 计算SSIM
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        
        ssim_map = numerator / (denominator + 1e-8)
        
        # 返回每个样本的平均SSIM
        return ssim_map.mean(dim=[1, 2, 3])
    
    def entropy_batch(self, images: torch.Tensor, bins: int = 256) -> torch.Tensor:
        """
        批量计算图像熵
        
        Args:
            images: 图像张量 (B, C, H, W)
            bins: 直方图分箱数
            
        Returns:
            熵值 (B,)
        """
        images = self._validate_tensor_format(images, "images")
        
        # 转换为灰度图
        if images.shape[1] == 3:
            weights = torch.tensor([0.299, 0.587, 0.114], device=images.device, dtype=images.dtype)
            images = torch.sum(images * weights.view(1, 3, 1, 1), dim=1, keepdim=True)
        
        # 量化到指定范围
        images_quantized = (images * (bins - 1)).long().clamp(0, bins - 1)
        
        entropies = []
        for i in range(images.shape[0]):
            img = images_quantized[i].flatten()
            # 计算直方图
            hist = torch.histc(img.float(), bins=bins, min=0, max=bins-1)
            # 归一化
            hist = hist / hist.sum()
            # 去除零值
            hist = hist[hist > 0]
            # 计算熵
            entropy = -torch.sum(hist * torch.log2(hist + 1e-8))
            entropies.append(entropy)
        
        return torch.stack(entropies)
    
    def spatial_frequency_batch(self, images: torch.Tensor) -> torch.Tensor:
        """
        批量计算空间频率
        
        Args:
            images: 图像张量 (B, C, H, W)
            
        Returns:
            空间频率值 (B,)
        """
        images = self._validate_tensor_format(images, "images")
        
        # 转换为灰度图
        if images.shape[1] == 3:
            weights = torch.tensor([0.299, 0.587, 0.114], device=images.device, dtype=images.dtype)
            images = torch.sum(images * weights.view(1, 3, 1, 1), dim=1, keepdim=True)
        
        # 计算行频率和列频率
        rf = torch.sqrt(torch.mean((images[:, :, :, 1:] - images[:, :, :, :-1]) ** 2, dim=[1, 2, 3]))
        cf = torch.sqrt(torch.mean((images[:, :, 1:, :] - images[:, :, :-1, :]) ** 2, dim=[1, 2, 3]))
        
        # 空间频率
        sf = torch.sqrt(rf ** 2 + cf ** 2)
        return sf
    
    def calculate_metrics_batch(self, fused: torch.Tensor, img_a: torch.Tensor, 
                               img_b: torch.Tensor, reference: torch.Tensor = None,
                               metrics: List[str] = None) -> Dict[str, torch.Tensor]:
        """
        批量计算多个指标
        
        Args:
            fused: 融合图像 (B, C, H, W)
            img_a: 源图像A (B, C, H, W)
            img_b: 源图像B (B, C, H, W)
            reference: 参考图像 (B, C, H, W)，可选
            metrics: 要计算的指标列表，None表示计算所有支持的指标
            
        Returns:
            指标字典，每个值为 (B,) 张量
        """
        if metrics is None:
            metrics = ['mse', 'psnr', 'ssim', 'entropy', 'sf']
        
        results = {}
        
        # 准备参考图像
        if reference is None:
            # 使用源图像的加权平均作为近似参考
            reference = 0.5 * img_a + 0.5 * img_b
        
        # 计算指标
        if 'mse' in metrics:
            results['mse'] = self.mse_batch(fused, reference)
        
        if 'psnr' in metrics:
            results['psnr'] = self.psnr_batch(fused, reference)
        
        if 'ssim' in metrics:
            results['ssim'] = self.ssim_batch(fused, reference)
        
        if 'entropy' in metrics:
            results['entropy'] = self.entropy_batch(fused)
        
        if 'sf' in metrics:
            results['sf'] = self.spatial_frequency_batch(fused)
        
        return results
    
    def calculate_average_metrics(self, fused: torch.Tensor, img_a: torch.Tensor, 
                                 img_b: torch.Tensor, reference: torch.Tensor = None,
                                 metrics: List[str] = None) -> Dict[str, float]:
        """
        计算批次的平均指标
        
        Args:
            fused: 融合图像 (B, C, H, W)
            img_a: 源图像A (B, C, H, W)
            img_b: 源图像B (B, C, H, W)
            reference: 参考图像 (B, C, H, W)，可选
            metrics: 要计算的指标列表
            
        Returns:
            平均指标字典
        """
        batch_results = self.calculate_metrics_batch(fused, img_a, img_b, reference, metrics)
        
        # 计算平均值
        avg_results = {}
        for metric_name, values in batch_results.items():
            avg_results[metric_name] = float(values.mean().cpu())
        
        return avg_results
    
    def evaluate_during_training(self, model_output: Dict[str, torch.Tensor], 
                                input_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        训练过程中的评估函数
        
        Args:
            model_output: 模型输出字典，包含 'fusion' 等键
            input_data: 输入数据字典，包含 'visible', 'infrared' 等键
            
        Returns:
            评估指标字典
        """
        fused = model_output.get('fusion')
        visible = input_data.get('visible')
        infrared = input_data.get('infrared')
        
        if fused is None or visible is None or infrared is None:
            raise ValueError("缺少必要的输入数据")
        
        # 计算关键指标
        metrics = ['mse', 'psnr', 'ssim', 'entropy', 'sf']
        return self.calculate_average_metrics(fused, visible, infrared, metrics=metrics)


def demo_tensor_evaluation():
    """
    演示张量评估的用法
    """
    print("PyTorch Tensor 图像融合评价指标演示")
    print("=" * 50)
    
    # 创建评估器
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    evaluator = TensorFusionEvaluator(device=device)
    print(f"使用设备: {device}")
    
    # 创建模拟数据
    batch_size = 4
    height, width = 256, 256
    
    # 模拟可见光图像 (RGB)
    visible = torch.rand(batch_size, 3, height, width, device=device)
    
    # 模拟红外图像 (灰度)
    infrared = torch.rand(batch_size, 1, height, width, device=device)
    
    # 模拟融合图像
    # 先将红外图像扩展到3通道
    infrared_rgb = infrared.repeat(1, 3, 1, 1)
    fusion = 0.6 * visible + 0.4 * infrared_rgb
    
    print(f"批次大小: {batch_size}")
    print(f"可见光图像形状: {visible.shape}")
    print(f"红外图像形状: {infrared.shape}")
    print(f"融合图像形状: {fusion.shape}")
    
    # 批量计算指标
    print("\n计算批量指标...")
    batch_metrics = evaluator.calculate_metrics_batch(
        fusion, visible, infrared_rgb, 
        metrics=['mse', 'psnr', 'ssim', 'entropy', 'sf']
    )
    
    print("\n批量结果:")
    for metric_name, values in batch_metrics.items():
        print(f"{metric_name:10s}: {values}")
    
    # 计算平均指标
    avg_metrics = evaluator.calculate_average_metrics(
        fusion, visible, infrared_rgb,
        metrics=['mse', 'psnr', 'ssim', 'entropy', 'sf']
    )
    
    print("\n平均指标:")
    print("-" * 30)
    for metric_name, value in avg_metrics.items():
        print(f"{metric_name:10s}: {value:.6f}")
    
    # 演示训练时评估
    print("\n训练时评估演示:")
    model_output = {'fusion': fusion}
    input_data = {'visible': visible, 'infrared': infrared_rgb}
    
    training_metrics = evaluator.evaluate_during_training(model_output, input_data)
    
    print("训练指标:")
    for metric_name, value in training_metrics.items():
        print(f"  {metric_name}: {value:.6f}")


if __name__ == "__main__":
    demo_tensor_evaluation()
