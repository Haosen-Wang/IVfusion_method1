#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度学习训练中的图像融合评价指标集成示例
演示如何在训练循环中使用评价指标
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from test_metric import FusionMetrics
from tensor_example import TensorFusionEvaluator
import time
import numpy as np


class SimpleFusionNet(nn.Module):
    """
    简单的图像融合网络示例
    """
    def __init__(self):
        super(SimpleFusionNet, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 1, 3, padding=1)
        self.relu = nn.ReLU()
        
    def forward(self, vis, ir):
        # 拼接可见光和红外图像
        x = torch.cat([vis, ir], dim=1)  # (b, 2, h, w)
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        fused = torch.sigmoid(self.conv4(x))  # 输出范围[0,1]
        
        return fused


def create_synthetic_dataset(num_samples=100, img_size=128):
    """
    创建合成数据集用于训练示例
    """
    vis_images = []
    ir_images = []
    target_images = []
    
    for i in range(num_samples):
        # 创建有特征的可见光图像
        vis = torch.randn(1, img_size, img_size) * 0.2 + 0.5
        
        # 添加纹理
        y, x = torch.meshgrid(torch.linspace(0, 1, img_size), 
                             torch.linspace(0, 1, img_size), indexing='ij')
        texture = 0.2 * torch.sin(10 * x + i) * torch.cos(10 * y + i)
        vis = torch.clamp(vis + texture, 0, 1)
        
        # 创建红外图像 (更多边缘信息)
        ir = torch.zeros(1, img_size, img_size)
        # 随机矩形区域
        h1, h2 = sorted([torch.randint(10, img_size-10, (1,)).item() for _ in range(2)])
        w1, w2 = sorted([torch.randint(10, img_size-10, (1,)).item() for _ in range(2)])
        ir[:, h1:h2, w1:w2] = 0.8
        
        # 添加噪声
        noise = torch.randn_like(ir) * 0.1
        ir = torch.clamp(ir + noise, 0, 1)
        
        # 创建理想融合目标
        target = torch.maximum(vis, ir) * 0.7 + torch.minimum(vis, ir) * 0.3
        
        vis_images.append(vis)
        ir_images.append(ir)
        target_images.append(target)
    
    vis_tensor = torch.stack(vis_images)
    ir_tensor = torch.stack(ir_images)
    target_tensor = torch.stack(target_images)
    
    return vis_tensor, ir_tensor, target_tensor


def train_with_metrics(device='cpu', num_epochs=5):
    """
    带有评价指标监控的训练示例
    """
    print("="*60)
    print("深度学习训练中的图像融合评价指标示例")
    print("="*60)
    
    # 创建数据集
    print("创建合成数据集...")
    vis_data, ir_data, target_data = create_synthetic_dataset(num_samples=200, img_size=64)
    
    # 创建数据加载器
    dataset = TensorDataset(vis_data, ir_data, target_data)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 创建模型
    model = SimpleFusionNet().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 创建评价指标实例
    evaluator = TensorFusionEvaluator(device=device)
    
    print(f"训练设备: {device}")
    print(f"数据集大小: {len(dataset)}")
    print(f"批量大小: {train_loader.batch_size}")
    print(f"训练轮数: {num_epochs}")
    print("-" * 60)
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_metrics = []
        
        start_time = time.time()
        
        for batch_idx, (vis_batch, ir_batch, target_batch) in enumerate(train_loader):
            vis_batch = vis_batch.to(device)
            ir_batch = ir_batch.to(device)
            target_batch = target_batch.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            fused_batch = model(vis_batch, ir_batch)
            
            # 计算损失
            loss = criterion(fused_batch, target_batch)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 每几个batch计算一次评价指标 (为了节省时间)
            if batch_idx % 5 == 0:
                with torch.no_grad():
                    # 计算评价指标
                    batch_metrics = evaluator.evaluate_fusion_batch(
                        fused_batch, vis_batch, ir_batch, target_batch
                    )
                    epoch_metrics.append(batch_metrics)
        
        # 计算平均指标
        avg_loss = epoch_loss / len(train_loader)
        
        if epoch_metrics:
            # 计算所有batch的平均指标
            metric_names = epoch_metrics[0].keys()
            avg_metrics = {}
            for metric_name in metric_names:
                if not metric_name.endswith('_std'):
                    values = [m[metric_name] for m in epoch_metrics]
                    avg_metrics[metric_name] = np.mean(values)
        else:
            avg_metrics = {}
        
        epoch_time = time.time() - start_time
        
        # 打印训练进度
        print(f"Epoch [{epoch+1}/{num_epochs}] - {epoch_time:.2f}s")
        print(f"  Loss: {avg_loss:.6f}")
        
        if avg_metrics:
            print(f"  PSNR: {avg_metrics.get('PSNR', 0):.2f} dB")
            print(f"  EN:   {avg_metrics.get('EN', 0):.6f}")
            print(f"  SF:   {avg_metrics.get('SF', 0):.6f}")
            print(f"  NABF: {avg_metrics.get('NABF', 0):.6f}")
        
        print("-" * 40)
    
    print("训练完成！")
    
    # 最终评价
    print("\n最终模型评价:")
    print("-" * 30)
    
    model.eval()
    with torch.no_grad():
        # 在测试数据上评价
        test_vis, test_ir, test_target = create_synthetic_dataset(num_samples=10, img_size=128)
        test_vis = test_vis.to(device)
        test_ir = test_ir.to(device)
        test_target = test_target.to(device)
        
        test_fused = model(test_vis, test_ir)
        
        final_metrics = evaluator.evaluate_fusion_batch(
            test_fused, test_vis, test_ir, test_target
        )
        
        print("测试集指标 (平均值 ± 标准差):")
        key_metrics = ['EN', 'SF', 'SD', 'PSNR', 'MS-SSIM', 'CC', 'NABF']
        for metric in key_metrics:
            if metric in final_metrics:
                mean_val = final_metrics[metric]
                std_val = final_metrics.get(f"{metric}_std", 0)
                if metric == 'PSNR':
                    print(f"  {metric:8s}: {mean_val:7.2f} ± {std_val:.2f} dB")
                else:
                    print(f"  {metric:8s}: {mean_val:7.6f} ± {std_val:.6f}")


def validation_with_metrics(device='cpu'):
    """
    验证阶段使用评价指标的示例
    """
    print("\n" + "="*60)
    print("验证阶段评价指标计算示例")
    print("="*60)
    
    # 模拟预训练模型
    model = SimpleFusionNet().to(device)
    model.eval()
    
    # 创建评价器
    evaluator = TensorFusionEvaluator(device=device)
    
    # 创建验证数据
    val_vis, val_ir, val_target = create_synthetic_dataset(num_samples=50, img_size=256)
    val_loader = DataLoader(
        TensorDataset(val_vis, val_ir, val_target), 
        batch_size=4, shuffle=False
    )
    
    all_metrics = []
    
    print("在验证集上计算评价指标...")
    
    with torch.no_grad():
        for batch_idx, (vis_batch, ir_batch, target_batch) in enumerate(val_loader):
            vis_batch = vis_batch.to(device)
            ir_batch = ir_batch.to(device)
            target_batch = target_batch.to(device)
            
            # 推理
            fused_batch = model(vis_batch, ir_batch)
            
            # 计算指标
            batch_metrics = evaluator.evaluate_fusion_batch(
                fused_batch, vis_batch, ir_batch, target_batch
            )
            
            all_metrics.append(batch_metrics)
            
            if batch_idx < 3:  # 只打印前几个batch的详细信息
                print(f"\nBatch {batch_idx + 1}:")
                print(f"  PSNR: {batch_metrics['PSNR']:.2f} dB")
                print(f"  EN:   {batch_metrics['EN']:.6f}")
                print(f"  NABF: {batch_metrics['NABF']:.6f}")
    
    # 计算整个验证集的统计
    print(f"\n验证集整体统计 ({len(all_metrics)} batches):")
    print("-" * 40)
    
    metric_names = ['EN', 'SF', 'SD', 'PSNR', 'MS-SSIM', 'CC', 'NABF', 'VIF']
    for metric_name in metric_names:
        values = [m[metric_name] for m in all_metrics if metric_name in m]
        if values:
            mean_val = np.mean(values)
            std_val = np.std(values)
            if metric_name == 'PSNR':
                print(f"{metric_name:8s}: {mean_val:7.2f} ± {std_val:.2f} dB")
            else:
                print(f"{metric_name:8s}: {mean_val:7.6f} ± {std_val:.6f}")


if __name__ == "__main__":
    # 检测可用设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 运行训练示例
    train_with_metrics(device=device, num_epochs=3)
    
    # 运行验证示例
    validation_with_metrics(device=device)
    
    print("\n" + "="*60)
    print("所有示例运行完成！")
    print("="*60)
