#!/usr/bin/env python3
"""
计算数据集的均值和标准差
支持分别计算红外和可见光图像的统计信息
"""

import os
import numpy as np
import cv2
from tqdm import tqdm
import argparse

def load_image(image_path):
    """加载图像并转换为RGB格式"""
    try:
        # 使用OpenCV加载图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"警告: 无法加载图像 {image_path}")
            return None
        
        # 转换BGR到RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 归一化到[0,1]
        img = img.astype(np.float32) / 255.0
        
        return img
    except Exception as e:
        print(f"错误: 加载图像 {image_path} 失败: {e}")
        return None

def calculate_dataset_stats(image_dir, image_type=""):
    """计算数据集的均值和标准差"""
    
    # 获取所有图像文件
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        import glob
        image_files.extend(glob.glob(os.path.join(image_dir, ext)))
        image_files.extend(glob.glob(os.path.join(image_dir, ext.upper())))
    
    if not image_files:
        print(f"错误: 在 {image_dir} 中未找到图像文件")
        return None
    
    print(f"\n{'='*50}")
    print(f"计算{image_type}图像统计信息")
    print(f"图像目录: {image_dir}")
    print(f"图像数量: {len(image_files)}")
    print(f"{'='*50}")
    
    # 用于累积统计的变量
    pixel_sum = np.zeros(3, dtype=np.float64)  # RGB三个通道的像素值总和
    pixel_squared_sum = np.zeros(3, dtype=np.float64)  # RGB三个通道像素值平方的总和
    total_pixels = 0
    
    # 用于记录图像尺寸信息
    image_sizes = []
    
    # 逐张图像处理
    valid_images = 0
    for image_path in tqdm(image_files, desc=f"处理{image_type}图像"):
        img = load_image(image_path)
        if img is None:
            continue
            
        valid_images += 1
        h, w, c = img.shape
        image_sizes.append((h, w))
        
        # 累积像素值
        img_flat = img.reshape(-1, 3)  # 展平为 (H*W, 3)
        pixel_sum += np.sum(img_flat, axis=0)
        pixel_squared_sum += np.sum(img_flat ** 2, axis=0)
        total_pixels += h * w
        
        # 每处理100张图像显示一次进度
        if valid_images % 100 == 0:
            current_mean = pixel_sum / total_pixels
            print(f"已处理 {valid_images} 张图像, 当前均值: R={current_mean[0]:.4f}, G={current_mean[1]:.4f}, B={current_mean[2]:.4f}")
    
    if valid_images == 0:
        print(f"错误: 没有成功加载任何{image_type}图像")
        return None
    
    # 计算最终统计信息
    mean = pixel_sum / total_pixels
    variance = (pixel_squared_sum / total_pixels) - (mean ** 2)
    std = np.sqrt(variance)
    
    # 图像尺寸统计
    if image_sizes:
        heights, widths = zip(*image_sizes)
        min_size = (min(heights), min(widths))
        max_size = (max(heights), max(widths))
        avg_size = (np.mean(heights), np.mean(widths))
    else:
        min_size = max_size = avg_size = (0, 0)
    
    return {
        'image_type': image_type,
        'total_images': len(image_files),
        'valid_images': valid_images,
        'total_pixels': total_pixels,
        'mean': mean,
        'std': std,
        'min_size': min_size,
        'max_size': max_size,
        'avg_size': avg_size
    }

def print_stats(stats):
    """打印统计信息"""
    if stats is None:
        return
    
    print(f"\n{'='*60}")
    print(f"{stats['image_type']}图像统计结果")
    print(f"{'='*60}")
    print(f"总图像数量: {stats['total_images']}")
    print(f"有效图像数量: {stats['valid_images']}")
    print(f"总像素数量: {stats['total_pixels']:,}")
    print(f"\n📊 像素值统计 (范围: 0-1)")
    print(f"均值 (Mean):")
    print(f"  R: {stats['mean'][0]:.6f}")
    print(f"  G: {stats['mean'][1]:.6f}")
    print(f"  B: {stats['mean'][2]:.6f}")
    print(f"标准差 (Std):")
    print(f"  R: {stats['std'][0]:.6f}")
    print(f"  G: {stats['std'][1]:.6f}")
    print(f"  B: {stats['std'][2]:.6f}")
    
    print(f"\n📏 图像尺寸统计:")
    print(f"最小尺寸 (H, W): {stats['min_size']}")
    print(f"最大尺寸 (H, W): {stats['max_size']}")
    print(f"平均尺寸 (H, W): ({stats['avg_size'][0]:.1f}, {stats['avg_size'][1]:.1f})")
    
    # 转换为PyTorch常用的格式 (0-255范围)
    mean_255 = stats['mean'] * 255
    std_255 = stats['std'] * 255
    
    print(f"\n🔧 PyTorch归一化参数 (0-255范围):")
    print(f"mean = [{mean_255[0]:.2f}, {mean_255[1]:.2f}, {mean_255[2]:.2f}]")
    print(f"std = [{std_255[0]:.2f}, {std_255[1]:.2f}, {std_255[2]:.2f}]")
    
    print(f"\n🔧 PyTorch transforms.Normalize参数:")
    print(f"transforms.Normalize(mean=[{stats['mean'][0]:.4f}, {stats['mean'][1]:.4f}, {stats['mean'][2]:.4f}], ")
    print(f"                     std=[{stats['std'][0]:.4f}, {stats['std'][1]:.4f}, {stats['std'][2]:.4f}])")

def main():
    parser = argparse.ArgumentParser(description='计算数据集的均值和标准差')
    parser.add_argument('--data_root', type=str, 
                        default='/data/1024whs_data/DeMMI-RF/Train_fusion/DroneRGBT',
                        help='数据集根目录')
    parser.add_argument('--modality', type=str, choices=['infrared', 'visible', 'both'], 
                        default='both', help='计算哪种模态的统计信息')
    
    args = parser.parse_args()
    
    print("🔥 数据集统计计算工具")
    print(f"数据根目录: {args.data_root}")
    
    # 检查目录是否存在
    if not os.path.exists(args.data_root):
        print(f"错误: 数据目录不存在: {args.data_root}")
        return
    
    # 计算统计信息
    if args.modality in ['infrared', 'both']:
        infrared_dir = os.path.join(args.data_root, 'infrared')
        if os.path.exists(infrared_dir):
            infrared_stats = calculate_dataset_stats(infrared_dir, "红外")
            print_stats(infrared_stats)
        else:
            print(f"警告: 红外图像目录不存在: {infrared_dir}")
    
    if args.modality in ['visible', 'both']:
        visible_dir = os.path.join(args.data_root, 'visible')
        if os.path.exists(visible_dir):
            visible_stats = calculate_dataset_stats(visible_dir, "可见光")
            print_stats(visible_stats)
        else:
            print(f"警告: 可见光图像目录不存在: {visible_dir}")
    
    # 如果计算了两种模态，还可以计算联合统计
    if args.modality == 'both':
        print(f"\n{'='*60}")
        print("💡 数据集使用建议")
        print(f"{'='*60}")
        print("1. 红外和可见光图像可能具有不同的统计特性")
        print("2. 建议为每种模态使用不同的归一化参数")
        print("3. 或者可以计算联合统计信息用于融合模型")

if __name__ == "__main__":
    main()
