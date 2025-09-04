"""
快速测试图像融合指标
"""

import numpy as np
import cv2
import sys
import os

# 添加当前目录到路径
sys.path.append(os.path.dirname(__file__))

def quick_test():
    """快速测试所有指标"""
    from fusion_metrics import ImageFusionMetrics
    
    print("🚀 快速测试图像融合指标")
    print("="*50)
    
    # 创建测试图像
    size = (256, 256)
    
    # 模拟可见光图像（高频细节）
    visible = np.zeros(size, dtype=np.float32)
    visible[64:192, 64:192] = 0.8  # 主要物体
    visible += np.random.normal(0, 0.05, size)  # 纹理细节
    visible = np.clip(visible, 0, 1)
    
    # 模拟红外图像（热源信息）
    infrared = np.zeros(size, dtype=np.float32)
    # 创建圆形热源
    y, x = np.ogrid[:size[0], :size[1]]
    center = (128, 128)
    radius = 60
    mask = (x - center[0])**2 + (y - center[1])**2 <= radius**2
    infrared[mask] = 0.9
    infrared += np.random.normal(0, 0.02, size)  # 噪声
    infrared = np.clip(infrared, 0, 1)
    
    # 不同融合策略
    fusion_strategies = {
        "平均融合": (visible + infrared) / 2,
        "最大值融合": np.maximum(visible, infrared),
        "加权融合": 0.7 * visible + 0.3 * infrared,
        "细节保持融合": visible + 0.3 * (infrared - np.mean(infrared))
    }
    
    # 创建评估器
    evaluator = ImageFusionMetrics()
    
    print(f"测试图像尺寸: {size}")
    print(f"源图像数量: 2 (可见光 + 红外)")
    print(f"融合策略数量: {len(fusion_strategies)}")
    print("-"*50)
    
    results_summary = {}
    
    for strategy_name, fused_img in fusion_strategies.items():
        print(f"\n📊 {strategy_name}")
        print("-"*30)
        
        # 计算所有指标
        results = evaluator.compute_all_metrics(
            fused_img, 
            [visible, infrared]
        )
        
        results_summary[strategy_name] = results
        
        # 打印结果
        for metric, value in results.items():
            print(f"{metric:6s}: {value:8.4f}")
    
    # 找出最佳策略
    print("\n🏆 最佳融合策略分析")
    print("="*50)
    
    metrics_higher_better = ['EN', 'SF', 'SD', 'AG', 'MI', 'SCD', 'Qabf']
    
    for metric in metrics_higher_better:
        if metric in results_summary[list(results_summary.keys())[0]]:
            best_strategy = max(results_summary.keys(), 
                              key=lambda x: results_summary[x][metric])
            best_value = results_summary[best_strategy][metric]
            print(f"{metric:6s} 最佳: {best_strategy} ({best_value:.4f})")
    
    # 保存可视化图像
    print(f"\n💾 保存测试图像...")
    
    cv2.imwrite("test_visible.png", (visible * 255).astype(np.uint8))
    cv2.imwrite("test_infrared.png", (infrared * 255).astype(np.uint8))
    
    for strategy_name, fused_img in fusion_strategies.items():
        filename = f"test_fused_{strategy_name.replace(' ', '_')}.png"
        cv2.imwrite(filename, (fused_img * 255).astype(np.uint8))
    
    print("✅ 测试图像已保存到当前目录")
    
    return results_summary


def test_with_real_images(visible_path, infrared_path, output_dir="fusion_test_results"):
    """使用真实图像进行测试"""
    from fusion_metrics import ImageFusionMetrics
    
    print(f"🔍 使用真实图像测试")
    print(f"可见光图像: {visible_path}")
    print(f"红外图像: {infrared_path}")
    print("="*50)
    
    # 检查文件是否存在
    if not os.path.exists(visible_path):
        print(f"❌ 找不到可见光图像: {visible_path}")
        return None
    if not os.path.exists(infrared_path):
        print(f"❌ 找不到红外图像: {infrared_path}")
        return None
    
    # 加载图像
    visible = cv2.imread(visible_path, cv2.IMREAD_GRAYSCALE)
    infrared = cv2.imread(infrared_path, cv2.IMREAD_GRAYSCALE)
    
    if visible is None or infrared is None:
        print("❌ 图像加载失败")
        return None
    
    # 确保图像尺寸一致
    if visible.shape != infrared.shape:
        print(f"⚠️ 图像尺寸不一致，将调整大小")
        min_h = min(visible.shape[0], infrared.shape[0])
        min_w = min(visible.shape[1], infrared.shape[1])
        visible = cv2.resize(visible, (min_w, min_h))
        infrared = cv2.resize(infrared, (min_w, min_h))
    
    # 归一化到[0,1]
    visible = visible.astype(np.float32) / 255.0
    infrared = infrared.astype(np.float32) / 255.0
    
    print(f"图像尺寸: {visible.shape}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 融合策略
    fusion_strategies = {
        "average": (visible + infrared) / 2,
        "maximum": np.maximum(visible, infrared),
        "weighted_vis": 0.7 * visible + 0.3 * infrared,
        "weighted_ir": 0.3 * visible + 0.7 * infrared,
    }
    
    # 评估器
    evaluator = ImageFusionMetrics()
    results_summary = {}
    
    for strategy_name, fused_img in fusion_strategies.items():
        print(f"\n📊 评估 {strategy_name}")
        
        # 计算指标
        results = evaluator.compute_all_metrics(
            fused_img, 
            [visible, infrared]
        )
        
        results_summary[strategy_name] = results
        
        # 保存融合图像
        output_path = os.path.join(output_dir, f"fused_{strategy_name}.png")
        cv2.imwrite(output_path, (fused_img * 255).astype(np.uint8))
        
        # 打印结果
        for metric, value in results.items():
            print(f"  {metric}: {value:.4f}")
    
    # 保存结果
    import json
    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\n✅ 结果已保存到: {output_dir}")
    return results_summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="图像融合指标快速测试")
    parser.add_argument("--visible", type=str, help="可见光图像路径")
    parser.add_argument("--infrared", type=str, help="红外图像路径")
    parser.add_argument("--output", type=str, default="fusion_test_results", help="输出目录")
    
    args = parser.parse_args()
    
    if args.visible and args.infrared:
        # 使用指定的真实图像
        test_with_real_images(args.visible, args.infrared, args.output)
    else:
        # 使用模拟图像进行快速测试
        print("🎯 未指定输入图像，使用模拟图像进行快速测试")
        print("📝 使用方法: python quick_test_metrics.py --visible path/to/vis.jpg --infrared path/to/ir.jpg")
        print()
        quick_test()
