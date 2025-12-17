#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像噪声添加工具
支持添加高斯噪声、椒盐噪声等
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path


def add_gaussian_noise(image, mean=0, sigma=5):
    """
    为图像添加高斯噪声
    
    参数:
        image: 输入图像 (numpy array)
        mean: 噪声均值，默认为0
        sigma: 噪声标准差，控制噪声强度
        
    返回:
        noisy_image: 添加噪声后的图像
        noise: 噪声本身（用于可视化）
    """
    # 生成与图像相同形状的高斯噪声
    gaussian_noise = np.random.normal(mean, sigma, image.shape)
    
    # 添加噪声到图像
    noisy_image = image.astype(np.float32) + gaussian_noise
    
    # 裁剪到有效范围 [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image, gaussian_noise


def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    """
    为图像添加椒盐噪声
    
    参数:
        image: 输入图像
        salt_prob: 盐噪声概率（白点）
        pepper_prob: 椒噪声概率（黑点）
        
    返回:
        noisy_image: 添加噪声后的图像
    """
    noisy_image = image.copy()
    
    # 添加盐噪声（白点）
    salt_mask = np.random.random(image.shape[:2]) < salt_prob
    noisy_image[salt_mask] = 255
    
    # 添加椒噪声（黑点）
    pepper_mask = np.random.random(image.shape[:2]) < pepper_prob
    noisy_image[pepper_mask] = 0
    
    return noisy_image


def add_poisson_noise(image):
    """
    为图像添加泊松噪声（适用于模拟光子计数噪声）
    
    参数:
        image: 输入图像
        
    返回:
        noisy_image: 添加噪声后的图像
    """
    # 归一化到 [0, 1]
    normalized = image.astype(np.float32) / 255.0
    
    # 添加泊松噪声
    noisy = np.random.poisson(normalized * 255.0) / 255.0
    
    # 还原到 [0, 255]
    noisy_image = np.clip(noisy * 255.0, 0, 255).astype(np.uint8)
    
    return noisy_image


def add_stripe_noise(image, stripe_intensity=20, stripe_width=1, direction='vertical'):
    """
    为红外图像添加条纹噪声（stripe noise）
    
    条纹噪声是红外传感器中常见的噪声类型，通常表现为垂直或水平的条纹。
    这种噪声是由于传感器列或行的增益不一致造成的。
    
    参数:
        image: 输入图像 (numpy array)
        stripe_intensity: 条纹强度，控制条纹的明显程度 (默认: 20)
        stripe_width: 条纹宽度，单位为像素 (默认: 1)
        direction: 条纹方向，'vertical' 或 'horizontal' (默认: 'vertical')
        
    返回:
        noisy_image: 添加条纹噪声后的图像
        stripe_noise: 条纹噪声数组（用于可视化）
    """
    h, w = image.shape[:2]
    
    # 创建条纹噪声
    if direction == 'vertical':
        # 垂直条纹（每列不同）
        num_stripes = w // stripe_width
        stripe_pattern = np.random.randn(num_stripes) * stripe_intensity
        
        # 扩展到图像尺寸
        stripe_noise = np.repeat(stripe_pattern, stripe_width)[:w]
        stripe_noise = np.tile(stripe_noise, (h, 1))
        
    elif direction == 'horizontal':
        # 水平条纹（每行不同）
        num_stripes = h // stripe_width
        stripe_pattern = np.random.randn(num_stripes) * stripe_intensity
        
        # 扩展到图像尺寸
        stripe_noise = np.repeat(stripe_pattern, stripe_width)[:h]
        stripe_noise = np.tile(stripe_noise[:, np.newaxis], (1, w))
        
    else:
        raise ValueError(f"不支持的方向: {direction}，请使用 'vertical' 或 'horizontal'")
    
    # 如果是彩色图像，扩展到3通道
    if len(image.shape) == 3:
        stripe_noise = np.stack([stripe_noise] * image.shape[2], axis=2)
    
    # 添加条纹噪声到图像
    noisy_image = image.astype(np.float32) + stripe_noise
    
    # 裁剪到有效范围 [0, 255]
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image, stripe_noise


def add_mixed_stripe_noise(image, stripe_intensity=20, stripe_width=1, 
                           gaussian_sigma=5, stripe_ratio=0.7):
    """
    为红外图像添加混合噪声：条纹噪声 + 高斯噪声
    
    这更符合真实红外图像的噪声特性，既有传感器列增益不一致导致的条纹，
    又有随机的高斯噪声。
    
    参数:
        image: 输入图像
        stripe_intensity: 条纹噪声强度
        stripe_width: 条纹宽度
        gaussian_sigma: 高斯噪声标准差
        stripe_ratio: 条纹噪声所占比例 (0-1)
        
    返回:
        noisy_image: 添加混合噪声后的图像
        mixed_noise: 混合噪声数组
    """
    # 添加条纹噪声
    stripe_noisy, stripe_noise = add_stripe_noise(
        image, stripe_intensity, stripe_width, direction='vertical'
    )
    
    # 添加高斯噪声
    gaussian_noise = np.random.normal(0, gaussian_sigma, image.shape)
    
    # 混合噪声
    mixed_noise = stripe_noise * stripe_ratio + gaussian_noise * (1 - stripe_ratio)
    
    # 应用到图像
    noisy_image = image.astype(np.float32) + mixed_noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image, mixed_noise


def calculate_psnr(original, noisy):
    """
    计算PSNR (Peak Signal-to-Noise Ratio)
    
    参数:
        original: 原始图像
        noisy: 噪声图像
        
    返回:
        psnr: PSNR值（dB）
    """
    mse = np.mean((original.astype(np.float32) - noisy.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10(255**2 / mse)
    return psnr


def calculate_snr(image):
    """
    计算SNR (Signal-to-Noise Ratio)
    
    参数:
        image: 输入图像
        
    返回:
        snr: SNR值（dB）
    """
    mean_signal = np.mean(image)
    std_noise = np.std(image)
    
    if std_noise == 0:
        return float('inf')
    
    snr = 20 * np.log10(mean_signal / std_noise)
    return snr


def visualize_noise(noise, output_path):
    """
    可视化噪声并保存（灰度图）
    
    参数:
        noise: 噪声数组
        output_path: 输出路径
    """
    # 将噪声归一化到 [0, 255] 用于显示
    # 噪声可能是负值，所以需要特殊处理
    noise_normalized = noise.copy()
    
    # 如果是彩色噪声，转换为灰度
    if len(noise.shape) == 3:
        noise_normalized = np.mean(noise_normalized, axis=2)
    
    # 归一化到 [0, 255]
    # 将噪声映射到 [0, 255]，中性灰为128（表示0噪声）
    noise_min = noise_normalized.min()
    noise_max = noise_normalized.max()
    
    if noise_max - noise_min > 0:
        # 将噪声范围映射到 [0, 255]，0值映射到128（中灰色）
        noise_vis = ((noise_normalized - noise_min) / (noise_max - noise_min) * 255).astype(np.uint8)
    else:
        noise_vis = np.ones_like(noise_normalized, dtype=np.uint8) * 128
    
    # 保存纯灰度噪声图
    cv2.imwrite(output_path, noise_vis)
    
    return noise_vis


def process_image(input_path, output_dir, noise_type='gaussian', **kwargs):
    """
    处理单张图像，添加噪声
    
    参数:
        input_path: 输入图像路径
        output_dir: 输出目录
        noise_type: 噪声类型 ('gaussian', 'salt_pepper', 'poisson')
        **kwargs: 噪声参数
    """
    # 读取图像
    print(f"\n正在处理: {input_path}")
    image = cv2.imread(input_path)
    
    if image is None:
        print(f"❌ 错误: 无法读取图像 {input_path}")
        return False
    
    print(f"✓ 图像大小: {image.shape[1]} x {image.shape[0]} pixels")
    
    # 根据噪声类型添加噪声
    noise_array = None  # 用于保存噪声数组
    
    if noise_type == 'gaussian':
        sigma = kwargs.get('sigma', 25)
        mean = kwargs.get('mean', 0)
        noisy_image, noise_array = add_gaussian_noise(image, mean, sigma)
        suffix = f"gaussian_sigma{sigma}"
        print(f"✓ 添加高斯噪声 (μ={mean}, σ={sigma})")
        
    elif noise_type == 'salt_pepper':
        salt_prob = kwargs.get('salt_prob', 0.01)
        pepper_prob = kwargs.get('pepper_prob', 0.01)
        noisy_image = add_salt_pepper_noise(image, salt_prob, pepper_prob)
        # 椒盐噪声通过差值计算
        noise_array = noisy_image.astype(np.float32) - image.astype(np.float32)
        suffix = f"salt_pepper_{salt_prob}_{pepper_prob}"
        print(f"✓ 添加椒盐噪声 (salt={salt_prob}, pepper={pepper_prob})")
        
    elif noise_type == 'poisson':
        noisy_image = add_poisson_noise(image)
        # 泊松噪声通过差值计算
        noise_array = noisy_image.astype(np.float32) - image.astype(np.float32)
        suffix = "poisson"
        print(f"✓ 添加泊松噪声")
        
    elif noise_type == 'stripe':
        stripe_intensity = kwargs.get('stripe_intensity', 20)
        stripe_width = kwargs.get('stripe_width', 1)
        direction = kwargs.get('direction', 'vertical')
        noisy_image, noise_array = add_stripe_noise(image, stripe_intensity, stripe_width, direction)
        suffix = f"stripe_i{stripe_intensity}_w{stripe_width}_{direction}"
        print(f"✓ 添加条纹噪声 (强度={stripe_intensity}, 宽度={stripe_width}, 方向={direction})")
        
    elif noise_type == 'mixed_stripe':
        stripe_intensity = kwargs.get('stripe_intensity', 20)
        stripe_width = kwargs.get('stripe_width', 1)
        gaussian_sigma = kwargs.get('gaussian_sigma', 5)
        stripe_ratio = kwargs.get('stripe_ratio', 0.7)
        noisy_image, noise_array = add_mixed_stripe_noise(
            image, stripe_intensity, stripe_width, gaussian_sigma, stripe_ratio
        )
        suffix = f"mixed_stripe_i{stripe_intensity}_g{gaussian_sigma}"
        print(f"✓ 添加混合噪声 (条纹强度={stripe_intensity}, 高斯σ={gaussian_sigma}, 条纹比例={stripe_ratio})")
        
    else:
        print(f"❌ 错误: 不支持的噪声类型 '{noise_type}'")
        return False
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成输出文件名
    base_name = Path(input_path).stem
    output_name = f"{base_name}_noisy_{suffix}.png"
    output_path = os.path.join(output_dir, output_name)
    
    # 保存噪声图像
    cv2.imwrite(output_path, noisy_image)
    print(f"✓ 已保存噪声图像: {output_path}")
    
    # 保存噪声可视化
    if noise_array is not None:
        noise_vis_path = os.path.join(output_dir, f"{base_name}_noise_{suffix}.png")
        visualize_noise(noise_array, noise_vis_path)
        print(f"✓ 已保存噪声可视化: {noise_vis_path}")
    
    # 计算并显示质量指标
    psnr = calculate_psnr(image, noisy_image)
    print(f"✓ PSNR: {psnr:.2f} dB")
    
    # 创建对比图（包含噪声可视化）
    comparison_path = os.path.join(output_dir, f"{base_name}_comparison_{suffix}.png")
    create_comparison(image, noisy_image, noise_array, comparison_path, noise_type, kwargs)
    print(f"✓ 已保存对比图: {comparison_path}")
    
    return True


def create_comparison(original, noisy, noise_array, output_path, noise_type, params):
    """
    创建原图、噪声、噪声图的三图对比
    
    参数:
        original: 原始图像
        noisy: 噪声图像
        noise_array: 噪声数组
        output_path: 输出路径
        noise_type: 噪声类型
        params: 噪声参数
    """
    # 调整尺寸以便显示
    max_width = 2400
    h, w = original.shape[:2]
    
    if w > max_width // 3:
        scale = (max_width // 3) / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        original = cv2.resize(original, (new_w, new_h))
        noisy = cv2.resize(noisy, (new_w, new_h))
    else:
        new_w, new_h = w, h
    
    # 创建噪声可视化
    if noise_array is not None:
        # 将噪声归一化为灰度图
        noise_vis_temp = noise_array.copy()
        if len(noise_vis_temp.shape) == 3:
            noise_vis_temp = np.mean(noise_vis_temp, axis=2)
        
        # 调整噪声图尺寸
        if noise_vis_temp.shape[:2] != (new_h, new_w):
            noise_vis_temp = cv2.resize(noise_vis_temp, (new_w, new_h))
        
        # 归一化到 [0, 255]
        noise_min = noise_vis_temp.min()
        noise_max = noise_vis_temp.max()
        if noise_max - noise_min > 0:
            noise_vis = ((noise_vis_temp - noise_min) / (noise_max - noise_min) * 255).astype(np.uint8)
        else:
            noise_vis = np.ones_like(noise_vis_temp, dtype=np.uint8) * 128
        
        # 转换为3通道灰度图以便拼接
        noise_gray = cv2.cvtColor(noise_vis, cv2.COLOR_GRAY2BGR)
        
        # 水平拼接三张图
        comparison = np.hstack([original, noise_gray, noisy])
    else:
        # 如果没有噪声数组，只拼接原图和噪声图
        comparison = np.hstack([original, noisy])
    
    # 添加文字标注
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    color = (0, 255, 0)
    
    # 原图标注
    cv2.putText(comparison, 'Original', (20, 40), font, font_scale, color, thickness)
    
    # 中间图标注（如果有噪声数组）
    if noise_array is not None:
        cv2.putText(comparison, 'Noise (colormap)', (new_w + 20, 40), 
                    font, font_scale, color, thickness)
        third_col_offset = new_w * 2
    else:
        third_col_offset = new_w
    
    # 噪声图标注
    if noise_type == 'gaussian':
        text = f'Noisy (sigma={params.get("sigma", 25)})'
    elif noise_type == 'salt_pepper':
        text = f'Noisy (s={params.get("salt_prob", 0.01)}, p={params.get("pepper_prob", 0.01)})'
    elif noise_type == 'poisson':
        text = 'Noisy (Poisson)'
    else:
        text = 'Noisy'
    
    cv2.putText(comparison, text, (third_col_offset + 20, 40), 
                font, font_scale, color, thickness)
    
    # 保存对比图
    cv2.imwrite(output_path, comparison)


def batch_process(input_dir, output_dir, noise_type='gaussian', **kwargs):
    """
    批量处理目录中的所有图像
    
    参数:
        input_dir: 输入目录
        output_dir: 输出目录
        noise_type: 噪声类型
        **kwargs: 噪声参数
    """
    # 支持的图像格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    # 查找所有图像文件
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    if not image_files:
        print(f"❌ 在 {input_dir} 中未找到图像文件")
        return
    
    print(f"\n找到 {len(image_files)} 张图像")
    print("=" * 60)
    
    # 处理每张图像
    success_count = 0
    for image_file in image_files:
        if process_image(str(image_file), output_dir, noise_type, **kwargs):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"✓ 处理完成！成功: {success_count}/{len(image_files)}")
    print("=" * 60)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='图像噪声添加工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 添加高斯噪声（默认 sigma=25）
  python add_noise.py input.jpg -o output/
  
  # 指定噪声强度
  python add_noise.py input.jpg -o output/ --sigma 50
  
  # 添加椒盐噪声
  python add_noise.py input.jpg -o output/ --type salt_pepper --salt 0.02 --pepper 0.02
  
  # 添加条纹噪声（适合红外图像）
  python add_noise.py infrared.jpg -o output/ --type stripe --stripe_intensity 20 --stripe_width 1
  
  # 添加混合条纹噪声（条纹+高斯）
  python add_noise.py infrared.jpg -o output/ --type mixed_stripe --stripe_intensity 20 --gaussian_sigma 5
  
  # 批量处理目录
  python add_noise.py input_dir/ -o output_dir/ --batch --sigma 30
  
噪声强度参考:
  高斯噪声 sigma=10-20  : 轻度噪声
  高斯噪声 sigma=25-35  : 中度噪声
  高斯噪声 sigma=50+    : 重度噪声
  条纹噪声 intensity=10-20 : 轻度条纹
  条纹噪声 intensity=20-40 : 中度条纹
        """
    )
    
    parser.add_argument('input', nargs='?', 
                        default='/data/1024whs_data/DeMMI-RF/Test/Multi-Task/alltask/LLVIP/visible/260009.jpg',
                        help='输入图像或目录路径')
    parser.add_argument('-o', '--output', default='./noisy_output', 
                        help='输出目录 (默认: ./noisy_output)')
    parser.add_argument('--type', choices=['gaussian', 'salt_pepper', 'poisson', 'stripe', 'mixed_stripe'],
                        default='gaussian', help='噪声类型 (默认: gaussian)')
    parser.add_argument('--batch', action='store_true',
                        help='批量处理目录中的所有图像')
    
    # 高斯噪声参数
    parser.add_argument('--sigma', type=float, default=25,
                        help='高斯噪声标准差 (默认: 25)')
    parser.add_argument('--mean', type=float, default=0,
                        help='高斯噪声均值 (默认: 0)')
    
    # 椒盐噪声参数
    parser.add_argument('--salt', type=float, default=0.01,
                        help='盐噪声概率 (默认: 0.01)')
    parser.add_argument('--pepper', type=float, default=0.01,
                        help='椒噪声概率 (默认: 0.01)')
    
    # 条纹噪声参数
    parser.add_argument('--stripe_intensity', type=float, default=20,
                        help='条纹噪声强度 (默认: 20)')
    parser.add_argument('--stripe_width', type=int, default=1,
                        help='条纹宽度，单位像素 (默认: 1)')
    parser.add_argument('--direction', choices=['vertical', 'horizontal'],
                        default='vertical',
                        help='条纹方向 (默认: vertical)')
    
    # 混合条纹噪声参数
    parser.add_argument('--gaussian_sigma', type=float, default=5,
                        help='混合噪声中的高斯噪声标准差 (默认: 5)')
    parser.add_argument('--stripe_ratio', type=float, default=0.7,
                        help='混合噪声中条纹噪声的比例 0-1 (默认: 0.7)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("图像噪声添加工具")
    print("=" * 60)
    
    # 检查输入路径
    if not os.path.exists(args.input):
        print(f"❌ 错误: 路径不存在 {args.input}")
        sys.exit(1)
    
    # 准备参数
    kwargs = {
        'sigma': args.sigma,
        'mean': args.mean,
        'salt_prob': args.salt,
        'pepper_prob': args.pepper,
        'stripe_intensity': args.stripe_intensity,
        'stripe_width': args.stripe_width,
        'direction': args.direction,
        'gaussian_sigma': args.gaussian_sigma,
        'stripe_ratio': args.stripe_ratio
    }
    
    # 处理
    if args.batch or os.path.isdir(args.input):
        batch_process(args.input, args.output, args.type, **kwargs)
    else:
        process_image(args.input, args.output, args.type, **kwargs)
        print("\n✓ 完成！")


if __name__ == "__main__":
    main()
