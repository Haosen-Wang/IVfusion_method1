#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像融合评价指标使用示例
"""

import cv2
import numpy as np
import os
import sys
from test_metric import FusionMetrics


def load_image(image_path):
    """
    加载图像
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取图像: {image_path}")
    
    return image


def evaluate_fusion_result(vis_path, ir_path, fusion_path, reference_path=None):
    """
    评价融合结果
    
    Args:
        vis_path: 可见光图像路径
        ir_path: 红外图像路径
        fusion_path: 融合图像路径
        reference_path: 参考图像路径（可选）
    """
    # 加载图像
    vis_img = load_image(vis_path)
    ir_img = load_image(ir_path)
    fusion_img = load_image(fusion_path)
    
    reference_img = None
    if reference_path and os.path.exists(reference_path):
        reference_img = load_image(reference_path)
    
    # 确保图像尺寸一致
    height, width = fusion_img.shape
    vis_img = cv2.resize(vis_img, (width, height))
    ir_img = cv2.resize(ir_img, (width, height))
    
    if reference_img is not None:
        reference_img = cv2.resize(reference_img, (width, height))
    
    # 创建评价指标实例
    metrics = FusionMetrics()
    
    # 计算所有指标
    print(f"正在评价融合结果: {fusion_path}")
    print("=" * 60)
    
    all_metrics = metrics.calculate_all_metrics(
        fusion_img, vis_img, ir_img, reference_img
    )
    
    # 打印结果
    print("评价指标结果:")
    print("-" * 40)
    
    # 无参考指标
    print("无参考指标:")
    print(f"  熵 (EN):           {all_metrics['EN']:.6f}")
    print(f"  空间频率 (SF):      {all_metrics['SF']:.6f}")
    print(f"  标准差 (SD):        {all_metrics['SD']:.6f}")
    print(f"  NABF:             {all_metrics['NABF']:.6f}")
    
    print("\n有参考指标:")
    print(f"  均方误差 (MSE):     {all_metrics['MSE']:.6f}")
    print(f"  峰值信噪比 (PSNR):  {all_metrics['PSNR']:.2f} dB")
    print(f"  多尺度SSIM:        {all_metrics['MS-SSIM']:.6f}")
    print(f"  相关系数 (CC):      {all_metrics['CC']:.6f}")
    print(f"  视觉信息保真度:      {all_metrics['VIF']:.6f}")
    
    return all_metrics


def batch_evaluation(dataset_dir):
    """
    批量评价数据集
    
    Args:
        dataset_dir: 数据集目录，应包含vis, ir, fusion子目录
    """
    vis_dir = os.path.join(dataset_dir, "vis")
    ir_dir = os.path.join(dataset_dir, "ir")
    fusion_dir = os.path.join(dataset_dir, "fusion")
    
    if not all(os.path.exists(d) for d in [vis_dir, ir_dir, fusion_dir]):
        print("错误: 数据集目录结构不正确")
        print("期望结构:")
        print("  dataset_dir/")
        print("    ├── vis/")
        print("    ├── ir/")
        print("    └── fusion/")
        return
    
    # 获取图像列表
    fusion_files = [f for f in os.listdir(fusion_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if not fusion_files:
        print("未找到融合图像文件")
        return
    
    # 初始化指标统计
    metrics_sum = {}
    valid_count = 0
    
    print(f"开始批量评价，共 {len(fusion_files)} 张图像")
    print("=" * 60)
    
    for i, fusion_file in enumerate(fusion_files):
        try:
            # 构建文件路径
            fusion_path = os.path.join(fusion_dir, fusion_file)
            vis_path = os.path.join(vis_dir, fusion_file)
            ir_path = os.path.join(ir_dir, fusion_file)
            
            # 检查对应的源图像是否存在
            if not os.path.exists(vis_path):
                print(f"警告: 未找到可见光图像 {vis_path}")
                continue
            if not os.path.exists(ir_path):
                print(f"警告: 未找到红外图像 {ir_path}")
                continue
            
            # 评价当前图像
            print(f"\n[{i+1}/{len(fusion_files)}] {fusion_file}")
            metrics = evaluate_fusion_result(vis_path, ir_path, fusion_path)
            
            # 累加指标
            if valid_count == 0:
                metrics_sum = {k: v for k, v in metrics.items()}
            else:
                for k, v in metrics.items():
                    metrics_sum[k] += v
            
            valid_count += 1
            
        except Exception as e:
            print(f"处理 {fusion_file} 时出错: {e}")
            continue
    
    # 计算平均指标
    if valid_count > 0:
        print(f"\n{'='*60}")
        print(f"批量评价完成，有效图像数量: {valid_count}")
        print(f"{'='*60}")
        print("平均指标:")
        print("-" * 40)
        
        for metric_name, total_value in metrics_sum.items():
            avg_value = total_value / valid_count
            print(f"{metric_name:15s}: {avg_value:.6f}")
    else:
        print("没有成功处理的图像")


def main():
    """
    主函数 - 命令行接口
    """
    if len(sys.argv) < 2:
        print("使用方法:")
        print("  单张图像评价:")
        print("    python example_usage.py single <vis_path> <ir_path> <fusion_path> [reference_path]")
        print("  批量评价:")
        print("    python example_usage.py batch <dataset_dir>")
        print("")
        print("示例:")
        print("    python example_usage.py single vis.png ir.png fusion.png")
        print("    python example_usage.py batch /path/to/dataset")
        return
    
    mode = sys.argv[1]
    
    if mode == "single":
        if len(sys.argv) < 5:
            print("错误: 单张图像评价需要至少4个参数")
            return
        
        vis_path = sys.argv[2]
        ir_path = sys.argv[3]
        fusion_path = sys.argv[4]
        reference_path = sys.argv[5] if len(sys.argv) > 5 else None
        
        try:
            evaluate_fusion_result(vis_path, ir_path, fusion_path, reference_path)
        except Exception as e:
            print(f"评价失败: {e}")
    
    elif mode == "batch":
        if len(sys.argv) < 3:
            print("错误: 批量评价需要数据集目录参数")
            return
        
        dataset_dir = sys.argv[2]
        batch_evaluation(dataset_dir)
    
    else:
        print(f"未知模式: {mode}")
        print("支持的模式: single, batch")


if __name__ == "__main__":
    main()
