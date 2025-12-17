"""
像素值分析工具
用于分析图像中特定区域的像素值分布，并提供消除处理选项
"""

import sys
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


class PixelAnalyzer:
    def __init__(self, image_path, convert_mode='L'):
        """
        初始化像素分析器
        
        Args:
            image_path: 图像路径
            convert_mode: 转换模式 ('L' 为灰度, 'RGB' 为彩色)
        """
        self.image_path = image_path
        self.image = Image.open(image_path).convert(convert_mode)
        self.img_array = np.array(self.image).astype(np.float32) / 255.0
        self.convert_mode = convert_mode
        
        print(f"图像加载成功: {image_path}")
        print(f"图像尺寸: {self.image.size}")
        print(f"像素值范围: [{self.img_array.min():.4f}, {self.img_array.max():.4f}]")
        print(f"平均像素值: {self.img_array.mean():.4f}")
        print(f"标准差: {self.img_array.std():.4f}")
    
    def analyze_histogram(self, bins=256, save_path=None):
        """
        分析像素值直方图
        """
        plt.figure(figsize=(12, 5))
        
        if len(self.img_array.shape) == 2:  # 灰度图
            plt.subplot(1, 2, 1)
            hist, bin_edges = np.histogram(self.img_array, bins=bins, range=(0, 1))
            plt.bar(bin_edges[:-1], hist, width=1.0/bins, color='gray', alpha=0.7)
            plt.xlabel('Pixel Value (0-1)')
            plt.ylabel('Frequency')
            plt.title('Pixel Value Distribution')
            plt.grid(True, alpha=0.3)
            
            # 累积分布
            plt.subplot(1, 2, 2)
            cumsum = np.cumsum(hist)
            plt.plot(bin_edges[:-1], cumsum / cumsum[-1], color='blue')
            plt.xlabel('Pixel Value (0-1)')
            plt.ylabel('Cumulative Probability')
            plt.title('Cumulative Distribution')
            plt.grid(True, alpha=0.3)
        else:  # 彩色图
            colors = ['red', 'green', 'blue']
            for i, color in enumerate(colors):
                plt.subplot(1, 3, i+1)
                hist, bin_edges = np.histogram(self.img_array[:, :, i], bins=bins, range=(0, 1))
                plt.bar(bin_edges[:-1], hist, width=1.0/bins, color=color, alpha=0.7)
                plt.xlabel('Pixel Value (0-1)')
                plt.ylabel('Frequency')
                plt.title(f'{color.upper()} Channel')
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"直方图已保存到: {save_path}")
        
        plt.show()
    
    def find_bright_regions(self, threshold=0.7, percentile=None):
        """
        找到亮区域（高像素值区域）
        
        Args:
            threshold: 像素值阈值（0-1）
            percentile: 使用百分位数代替固定阈值
        
        Returns:
            mask: 亮区域的二值掩码
        """
        if percentile is not None:
            threshold = np.percentile(self.img_array, percentile)
            print(f"使用百分位数 {percentile}%: 阈值 = {threshold:.4f}")
        else:
            print(f"使用固定阈值: {threshold:.4f}")
        
        if len(self.img_array.shape) == 2:  # 灰度图
            mask = self.img_array > threshold
        else:  # 彩色图，使用平均值
            gray = self.img_array.mean(axis=2)
            mask = gray > threshold
        
        bright_pixels = np.sum(mask)
        total_pixels = mask.size
        percentage = (bright_pixels / total_pixels) * 100
        
        print(f"亮区域像素数: {bright_pixels} / {total_pixels} ({percentage:.2f}%)")
        
        # 分析亮区域的统计信息
        if len(self.img_array.shape) == 2:
            bright_values = self.img_array[mask]
        else:
            bright_values = self.img_array[mask, :]
        
        if len(bright_values) > 0:
            print(f"亮区域像素值统计:")
            print(f"  最小值: {bright_values.min():.4f}")
            print(f"  最大值: {bright_values.max():.4f}")
            print(f"  平均值: {bright_values.mean():.4f}")
            print(f"  标准差: {bright_values.std():.4f}")
            print(f"  中位数: {np.median(bright_values):.4f}")
        
        return mask
    
    def analyze_percentiles(self):
        """
        分析不同百分位数的像素值
        """
        percentiles = [50, 75, 80, 85, 90, 95, 98, 99, 99.5, 99.9]
        
        print("\n像素值百分位数分析:")
        print("-" * 50)
        
        if len(self.img_array.shape) == 2:  # 灰度图
            for p in percentiles:
                value = np.percentile(self.img_array, p)
                count = np.sum(self.img_array >= value)
                percentage = (count / self.img_array.size) * 100
                print(f"P{p:5.1f}: {value:.4f}  (>= 此值的像素占 {percentage:.2f}%)")
        else:  # 彩色图
            print("RGB通道:")
            for i, channel in enumerate(['R', 'G', 'B']):
                print(f"\n{channel} 通道:")
                for p in percentiles:
                    value = np.percentile(self.img_array[:, :, i], p)
                    count = np.sum(self.img_array[:, :, i] >= value)
                    percentage = (count / self.img_array[:, :, i].size) * 100
                    print(f"  P{p:5.1f}: {value:.4f}  (>= 此值的像素占 {percentage:.2f}%)")
    
    def visualize_bright_regions(self, threshold=0.7, percentile=None, save_path=None):
        """
        可视化亮区域
        """
        mask = self.find_bright_regions(threshold=threshold, percentile=percentile)
        
        fig = plt.figure(figsize=(15, 5))
        
        # 原始图像
        plt.subplot(1, 3, 1)
        if len(self.img_array.shape) == 2:
            plt.imshow(self.img_array, cmap='gray', vmin=0, vmax=1)
        else:
            plt.imshow(self.img_array)
        plt.title('Original Image')
        plt.axis('off')
        
        # 亮区域掩码
        plt.subplot(1, 3, 2)
        plt.imshow(mask, cmap='hot')
        plt.title(f'Bright Regions (threshold={threshold:.2f})')
        plt.axis('off')
        
        # 叠加显示
        plt.subplot(1, 3, 3)
        if len(self.img_array.shape) == 2:
            plt.imshow(self.img_array, cmap='gray', vmin=0, vmax=1)
        else:
            plt.imshow(self.img_array)
        plt.imshow(mask, cmap='Reds', alpha=0.5)
        plt.title('Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
            print(f"可视化结果已保存到: {save_path}")
        
        plt.show()
    
    def remove_bright_regions(self, threshold=0.7, percentile=None, 
                             method='zero', replacement_value=0.0, 
                             save_path=None):
        """
        移除亮区域
        
        Args:
            threshold: 像素值阈值
            percentile: 使用百分位数
            method: 移除方法
                - 'zero': 设置为0（黑色）
                - 'mean': 设置为图像平均值
                - 'median': 设置为图像中位数
                - 'value': 设置为指定值
                - 'inpaint': 使用图像修复算法
            replacement_value: 当method='value'时使用的替换值
            save_path: 保存路径
        
        Returns:
            处理后的图像数组
        """
        mask = self.find_bright_regions(threshold=threshold, percentile=percentile)
        img_processed = self.img_array.copy()
        
        if method == 'zero':
            replacement = 0.0
        elif method == 'mean':
            if len(self.img_array.shape) == 2:
                replacement = self.img_array[~mask].mean()
            else:
                replacement = self.img_array[~mask].mean(axis=0)
        elif method == 'median':
            if len(self.img_array.shape) == 2:
                replacement = np.median(self.img_array[~mask])
            else:
                replacement = np.median(self.img_array[~mask], axis=0)
        elif method == 'value':
            replacement = replacement_value
        elif method == 'inpaint':
            # 使用OpenCV的修复算法
            mask_uint8 = (mask * 255).astype(np.uint8)
            img_uint8 = (self.img_array * 255).astype(np.uint8)
            
            if len(self.img_array.shape) == 2:
                img_inpainted = cv2.inpaint(img_uint8, mask_uint8, 3, cv2.INPAINT_TELEA)
            else:
                img_inpainted = cv2.inpaint(img_uint8, mask_uint8, 3, cv2.INPAINT_TELEA)
            
            img_processed = img_inpainted.astype(np.float32) / 255.0
            print(f"使用修复算法处理亮区域")
            
            if save_path:
                self._save_image(img_processed, save_path)
            
            return img_processed
        else:
            raise ValueError(f"不支持的方法: {method}")
        
        # 应用替换
        img_processed[mask] = replacement
        
        print(f"使用方法 '{method}' 处理亮区域，替换值: {replacement}")
        
        if save_path:
            self._save_image(img_processed, save_path)
        
        return img_processed
    
    def _save_image(self, img_array, save_path):
        """保存图像"""
        img_uint8 = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
        
        if len(img_array.shape) == 2:
            img_pil = Image.fromarray(img_uint8, mode='L')
        else:
            img_pil = Image.fromarray(img_uint8, mode='RGB')
        
        img_pil.save(save_path)
        print(f"处理后的图像已保存到: {save_path}")
    
    def compare_methods(self, threshold=0.7, percentile=None, save_dir='./analysis_output'):
        """
        比较不同的移除方法
        """
        os.makedirs(save_dir, exist_ok=True)
        
        methods = ['zero', 'mean', 'median', 'inpaint']
        results = {}
        
        for method in methods:
            print(f"\n处理方法: {method}")
            results[method] = self.remove_bright_regions(
                threshold=threshold, 
                percentile=percentile,
                method=method
            )
        
        # 可视化比较
        fig = plt.figure(figsize=(20, 8))
        
        # 原始图像
        plt.subplot(2, 3, 1)
        if len(self.img_array.shape) == 2:
            plt.imshow(self.img_array, cmap='gray', vmin=0, vmax=1)
        else:
            plt.imshow(self.img_array)
        plt.title('Original')
        plt.axis('off')
        
        # 亮区域掩码
        mask = self.find_bright_regions(threshold=threshold, percentile=percentile)
        plt.subplot(2, 3, 2)
        plt.imshow(mask, cmap='hot')
        plt.title('Bright Regions Mask')
        plt.axis('off')
        
        # 各种方法的结果
        for idx, (method, img) in enumerate(results.items(), start=3):
            plt.subplot(2, 3, idx)
            if len(img.shape) == 2:
                plt.imshow(img, cmap='gray', vmin=0, vmax=1)
            else:
                plt.imshow(img)
            plt.title(f'Method: {method}')
            plt.axis('off')
        
        plt.tight_layout()
        
        compare_path = os.path.join(save_dir, 'methods_comparison.png')
        plt.savefig(compare_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n方法比较图已保存到: {compare_path}")
        
        plt.show()
        
        return results


def main():
    """
    示例用法
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='像素值分析和亮区域移除工具')
    parser.add_argument('--image', type=str, required=True, help='图像路径')
    parser.add_argument('--mode', type=str, default='L', choices=['L', 'RGB'],
                       help='图像模式: L(灰度) 或 RGB')
    parser.add_argument('--threshold', type=float, default=0.7,
                       help='亮区域阈值 (0-1)')
    parser.add_argument('--percentile', type=float, default=None,
                       help='使用百分位数代替固定阈值 (例如: 95)')
    parser.add_argument('--method', type=str, default='zero',
                       choices=['zero', 'mean', 'median', 'value', 'inpaint'],
                       help='移除方法')
    parser.add_argument('--value', type=float, default=0.0,
                       help='当method=value时的替换值')
    parser.add_argument('--save_dir', type=str, default='./analysis_output',
                       help='保存目录')
    parser.add_argument('--compare', action='store_true',
                       help='比较所有方法')
    
    args = parser.parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 初始化分析器
    print("="*60)
    print("像素值分析工具")
    print("="*60)
    analyzer = PixelAnalyzer(args.image, convert_mode=args.mode)
    
    # 分析百分位数
    print("\n" + "="*60)
    analyzer.analyze_percentiles()
    
    # 分析直方图
    print("\n" + "="*60)
    print("生成像素值直方图...")
    hist_path = os.path.join(args.save_dir, 'histogram.png')
    analyzer.analyze_histogram(save_path=hist_path)
    
    # 可视化亮区域
    print("\n" + "="*60)
    print("可视化亮区域...")
    viz_path = os.path.join(args.save_dir, 'bright_regions.png')
    analyzer.visualize_bright_regions(
        threshold=args.threshold,
        percentile=args.percentile,
        save_path=viz_path
    )
    
    # 比较或单独处理
    if args.compare:
        print("\n" + "="*60)
        print("比较不同的处理方法...")
        analyzer.compare_methods(
            threshold=args.threshold,
            percentile=args.percentile,
            save_dir=args.save_dir
        )
    else:
        print("\n" + "="*60)
        print(f"使用 {args.method} 方法移除亮区域...")
        output_path = os.path.join(args.save_dir, f'processed_{args.method}.png')
        analyzer.remove_bright_regions(
            threshold=args.threshold,
            percentile=args.percentile,
            method=args.method,
            replacement_value=args.value,
            save_path=output_path
        )
    
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)


if __name__ == '__main__':
    main()
