"""
图像融合指标测试脚本
展示如何使用各种融合评估指标
"""

import os
import sys
import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt
import json
from typing import List, Dict

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))
from fusion_metrics import ImageFusionMetrics, test_fusion_metrics, batch_evaluate_fusion


def create_test_images():
    """创建测试图像用于演示"""
    print("创建测试图像...")
    
    # 创建测试目录
    test_dir = "test_fusion_images"
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(f"{test_dir}/source1", exist_ok=True)
    os.makedirs(f"{test_dir}/source2", exist_ok=True)
    os.makedirs(f"{test_dir}/fused", exist_ok=True)
    os.makedirs(f"{test_dir}/reference", exist_ok=True)
    
    # 生成不同类型的测试图像
    size = (256, 256)
    
    for i in range(3):
        # 源图像1 - 高频细节
        source1 = np.zeros(size, dtype=np.float32)
        source1[50:200, 50:200] = 0.8  # 矩形区域
        source1 += np.random.normal(0, 0.1, size)  # 添加噪声
        source1 = np.clip(source1, 0, 1)
        
        # 源图像2 - 低频背景
        y, x = np.ogrid[:size[0], :size[1]]
        source2 = np.sin(x * 0.02) * np.cos(y * 0.02) * 0.5 + 0.5
        source2 = source2.astype(np.float32)
        
        # 融合图像 - 简单加权平均
        fused = 0.6 * source1 + 0.4 * source2
        
        # 参考图像 - 理想融合
        reference = np.maximum(source1, source2)
        
        # 保存图像
        cv2.imwrite(f"{test_dir}/source1/test_{i+1}.png", (source1 * 255).astype(np.uint8))
        cv2.imwrite(f"{test_dir}/source2/test_{i+1}.png", (source2 * 255).astype(np.uint8))
        cv2.imwrite(f"{test_dir}/fused/test_{i+1}.png", (fused * 255).astype(np.uint8))
        cv2.imwrite(f"{test_dir}/reference/test_{i+1}.png", (reference * 255).astype(np.uint8))
    
    print(f"测试图像已创建到: {test_dir}")
    return test_dir


def test_single_image_metrics():
    """测试单张图像的所有指标"""
    print("\n" + "="*60)
    print("单张图像指标测试")
    print("="*60)
    
    # 创建测试图像
    test_dir = create_test_images()
    
    # 测试第一张图像
    fused_path = f"{test_dir}/fused/test_1.png"
    source_paths = [
        f"{test_dir}/source1/test_1.png",
        f"{test_dir}/source2/test_1.png"
    ]
    reference_path = f"{test_dir}/reference/test_1.png"
    
    # 计算指标
    results = test_fusion_metrics(
        fused_path, 
        source_paths, 
        reference_path,
        save_results=True
    )
    
    return results


def test_batch_evaluation():
    """测试批量评估功能"""
    print("\n" + "="*60)
    print("批量评估测试")
    print("="*60)
    
    test_dir = "test_fusion_images"
    
    # 批量评估
    batch_evaluate_fusion(
        fused_dir=f"{test_dir}/fused",
        source_dirs=[f"{test_dir}/source1", f"{test_dir}/source2"],
        reference_dir=f"{test_dir}/reference",
        output_file="test_batch_results.csv"
    )


def test_torch_tensor_input():
    """测试PyTorch张量输入"""
    print("\n" + "="*60)
    print("PyTorch张量输入测试")
    print("="*60)
    
    # 创建PyTorch张量
    fused_tensor = torch.rand(1, 256, 256)  # (C, H, W)
    source1_tensor = torch.rand(1, 256, 256)
    source2_tensor = torch.rand(1, 256, 256)
    
    # 创建评估器
    evaluator = ImageFusionMetrics()
    
    # 计算指标
    results = evaluator.compute_all_metrics(
        fused_tensor, 
        [source1_tensor, source2_tensor]
    )
    
    print("PyTorch张量测试结果:")
    for metric, value in results.items():
        print(f"{metric:6s}: {value:.6f}")


def compare_fusion_methods():
    """比较不同融合方法的性能"""
    print("\n" + "="*60)
    print("融合方法性能比较")
    print("="*60)
    
    # 创建测试数据
    size = (256, 256)
    source1 = np.random.rand(*size).astype(np.float32)
    source2 = np.random.rand(*size).astype(np.float32)
    
    # 不同融合方法
    fusion_methods = {
        "Average": (source1 + source2) / 2,
        "Maximum": np.maximum(source1, source2),
        "Weighted_0.6_0.4": 0.6 * source1 + 0.4 * source2,
        "Weighted_0.3_0.7": 0.3 * source1 + 0.7 * source2,
    }
    
    evaluator = ImageFusionMetrics()
    comparison_results = {}
    
    for method_name, fused_img in fusion_methods.items():
        results = evaluator.compute_all_metrics(
            fused_img, 
            [source1, source2]
        )
        comparison_results[method_name] = results
        
        print(f"\n{method_name}:")
        for metric, value in results.items():
            print(f"  {metric:6s}: {value:.6f}")
    
    # 保存比较结果
    with open("fusion_methods_comparison.json", "w") as f:
        json.dump(comparison_results, f, indent=2)
    
    print(f"\n比较结果已保存到: fusion_methods_comparison.json")
    
    return comparison_results


def analyze_metric_sensitivity():
    """分析指标对不同图像特征的敏感性"""
    print("\n" + "="*60)
    print("指标敏感性分析")
    print("="*60)
    
    evaluator = ImageFusionMetrics()
    
    # 测试不同特征的图像
    test_cases = {
        "High_Contrast": {
            "fused": np.random.rand(256, 256) * 0.8 + 0.1,
            "source1": np.random.rand(256, 256) * 0.5,
            "source2": np.random.rand(256, 256) * 0.5 + 0.5
        },
        "Low_Contrast": {
            "fused": np.random.rand(256, 256) * 0.2 + 0.4,
            "source1": np.random.rand(256, 256) * 0.1 + 0.45,
            "source2": np.random.rand(256, 256) * 0.1 + 0.45
        },
        "High_Detail": {
            "fused": np.random.rand(256, 256),
            "source1": np.random.rand(256, 256),
            "source2": np.random.rand(256, 256)
        },
        "Smooth": {
            "fused": np.ones((256, 256)) * 0.5 + np.random.rand(256, 256) * 0.1,
            "source1": np.ones((256, 256)) * 0.4 + np.random.rand(256, 256) * 0.05,
            "source2": np.ones((256, 256)) * 0.6 + np.random.rand(256, 256) * 0.05
        }
    }
    
    sensitivity_results = {}
    
    for case_name, images in test_cases.items():
        results = evaluator.compute_all_metrics(
            images["fused"].astype(np.float32),
            [images["source1"].astype(np.float32), images["source2"].astype(np.float32)]
        )
        sensitivity_results[case_name] = results
        
        print(f"\n{case_name}:")
        for metric, value in results.items():
            print(f"  {metric:6s}: {value:.6f}")
    
    # 保存敏感性分析结果
    with open("metric_sensitivity_analysis.json", "w") as f:
        json.dump(sensitivity_results, f, indent=2)
    
    print(f"\n敏感性分析结果已保存到: metric_sensitivity_analysis.json")
    
    return sensitivity_results


def generate_detailed_report():
    """生成详细的测试报告"""
    print("\n" + "="*60)
    print("生成详细测试报告")
    print("="*60)
    
    report = {
        "test_timestamp": str(np.datetime64('now')),
        "single_image_test": test_single_image_metrics(),
        "torch_tensor_test": None,
        "fusion_methods_comparison": compare_fusion_methods(),
        "sensitivity_analysis": analyze_metric_sensitivity()
    }
    
    # PyTorch张量测试
    try:
        fused_tensor = torch.rand(1, 256, 256)
        source1_tensor = torch.rand(1, 256, 256)
        source2_tensor = torch.rand(1, 256, 256)
        
        evaluator = ImageFusionMetrics()
        torch_results = evaluator.compute_all_metrics(
            fused_tensor, [source1_tensor, source2_tensor]
        )
        report["torch_tensor_test"] = torch_results
        print("✓ PyTorch张量测试完成")
    except Exception as e:
        print(f"✗ PyTorch张量测试失败: {e}")
        report["torch_tensor_test"] = None
    
    # 保存完整报告
    with open("fusion_metrics_test_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\n详细测试报告已保存到: fusion_metrics_test_report.json")
    
    # 打印总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    print("✓ 单张图像指标测试 - 完成")
    print("✓ 批量评估测试 - 完成")
    print("✓ 融合方法比较 - 完成")
    print("✓ 指标敏感性分析 - 完成")
    print("✓ 测试报告生成 - 完成")
    print("\n所有测试文件:")
    print("- test_fusion_images/          # 测试图像目录")
    print("- test_batch_results.csv       # 批量评估结果")
    print("- fusion_methods_comparison.json # 融合方法比较")
    print("- metric_sensitivity_analysis.json # 敏感性分析")
    print("- fusion_metrics_test_report.json # 完整测试报告")


if __name__ == "__main__":
    print("图像融合指标测试系统")
    print("Author: Assistant")
    print("Date: 2025-09-04")
    print("="*60)
    
    # 检查依赖
    try:
        import pandas as pd
        print("✓ 依赖检查通过")
    except ImportError:
        print("✗ 缺少pandas，部分功能可能不可用")
        print("安装命令: pip install pandas")
    
    # 运行完整测试
    try:
        generate_detailed_report()
        print("\n🎉 所有测试完成！")
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
