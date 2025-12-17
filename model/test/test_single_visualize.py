import sys
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(model_dir)

from stage3.model import DIV_fusion_model
from test_metrics.tensor_example import TensorFusionEvaluator

class SingleImageTester:
    def __init__(self, model_path, task='dv_i', device='cuda:0'):
        """
        初始化单图测试器
        
        Args:
            model_path: 模型权重路径
            task: 任务类型 ('dv_i' 或 'di_v')
            device: 运行设备
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.task = task
        print(f"使用设备: {self.device}")
        print(f"任务类型: {self.task}")
        
        # 加载模型
        self.model = DIV_fusion_model(task=task)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        print(f"模型加载完成: {model_path}")
        
        # 初始化评估器
        self.evaluator = TensorFusionEvaluator()
    
    def detect_bright_regions(self, infrared_tensor, percentile=95):
        """
        检测红外图像中的亮区域（灰白色背景）
        
        Args:
            infrared_tensor: 红外图像tensor (1, 1, H, W)
            percentile: 百分位数阈值
        
        Returns:
            mask: 亮区域掩码 (1, 1, H, W)
            threshold: 使用的阈值
        """
        # 转换为numpy进行分析
        ir_array = infrared_tensor.detach().cpu().numpy().squeeze()
        
        # 计算阈值
        threshold = np.percentile(ir_array, percentile)
        
        # 创建掩码
        mask_array = (ir_array > threshold).astype(np.float32)
        
        # 转换回tensor
        mask = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0).to(infrared_tensor.device)
        
        bright_pixels = np.sum(mask_array)
        total_pixels = mask_array.size
        percentage = (bright_pixels / total_pixels) * 100
        
        print(f"检测到亮区域: {bright_pixels}/{total_pixels} ({percentage:.2f}%)")
        print(f"阈值 (P{percentile}): {threshold:.4f}")
        
        return mask, threshold
    
    def remove_bright_background(self, fused_tensor, infrared_tensor, percentile=95, method='subtract'):
        """
        从融合图像中移除红外图像的亮背景
        
        Args:
            fused_tensor: 融合图像tensor (1, 3, H, W)
            infrared_tensor: 红外图像tensor (1, 1, H, W)
            percentile: 百分位数阈值
            method: 移除方法
                - 'subtract': 像素级相减
                - 'mask': 直接屏蔽
                - 'blend': 混合处理
        
        Returns:
            processed_tensor: 处理后的融合图像
        """
        # 检测亮区域
        mask, threshold = self.detect_bright_regions(infrared_tensor, percentile)
        
        # 扩展mask到3通道以匹配RGB图像
        mask_rgb = mask.repeat(1, 3, 1, 1)
        
        processed = fused_tensor.clone()
        
        if method == 'subtract':
            # 方法1: 像素级相减
            # 将红外图像扩展到3通道
            ir_rgb = infrared_tensor.repeat(1, 3, 1, 1)
            
            # 只在亮区域进行相减
            # 相减的强度与像素值成正比
            ir_array = infrared_tensor.detach().cpu().numpy().squeeze()
            
            # 计算相减强度（归一化到0-1）
            subtract_strength = torch.from_numpy(
                np.clip((ir_array - threshold) / (1.0 - threshold + 1e-6), 0, 1)
            ).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1).to(fused_tensor.device)
            
            # 执行加权相减
            processed = fused_tensor - (ir_rgb * mask_rgb * subtract_strength * 0.5)
            processed = torch.clamp(processed, 0, 1)
            
            print(f"使用 subtract 方法处理 (强度: 0.5)")
            
        elif method == 'mask':
            # 方法2: 直接屏蔽（设置为较暗的值）
            # 使用融合图像的平均值或中位值
            mean_val = fused_tensor.mean()
            processed = fused_tensor * (1 - mask_rgb) + mean_val * mask_rgb
            
            print(f"使用 mask 方法处理 (替换值: {mean_val:.4f})")
            
        elif method == 'blend':
            # 方法3: 混合处理 - 降低亮区域的强度
            blend_factor = 0.3  # 保留30%的原始值
            processed = fused_tensor * (1 - mask_rgb * (1 - blend_factor))
            processed = torch.clamp(processed, 0, 1)
            
            print(f"使用 blend 方法处理 (保留因子: {blend_factor})")
        
        else:
            raise ValueError(f"不支持的方法: {method}")
        
        return processed, mask_rgb
        
    def preprocess_image(self, image_path, convert_mode, size=(240, 240)):
        """
        预处理单张图像
        
        Args:
            image_path: 图像路径
            convert_mode: 转换模式 ('L' 为灰度, 'RGB' 为彩色)
            size: 调整大小
        
        Returns:
            tensor: 预处理后的tensor (1, C, H, W)
        """
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
        ])
        
        image = Image.open(image_path).convert(convert_mode)
        tensor = transform(image).unsqueeze(0)  # 添加batch维度
        return tensor
    
    def tensor_to_image(self, tensor):
        """
        将tensor转换为可显示的numpy数组
        
        Args:
            tensor: (1, C, H, W) 或 (C, H, W)
        
        Returns:
            numpy array: (H, W, C) 或 (H, W)
        """
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)  # 移除batch维度
        
        # 将tensor移到CPU并转换为numpy
        img = tensor.detach().cpu().numpy()
        
        # 调整维度顺序: (C, H, W) -> (H, W, C)
        if img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0))
        
        # 如果是单通道，去掉最后一维
        if img.shape[-1] == 1:
            img = img.squeeze(-1)
        
        # 确保值在[0, 1]范围内
        img = np.clip(img, 0, 1)
        
        return img
    
    def test_single(self, i_path, v_path, d_path, save_dir='./single_test_output',
                   remove_ir_background=True, bg_percentile=95, bg_method='subtract'):
        """
        测试单对图像并可视化
        
        Args:
            i_path: 红外图像路径
            v_path: 可见光图像路径
            d_path: 退化图像路径
            save_dir: 保存结果的目录
            remove_ir_background: 是否移除红外图像的亮背景
            bg_percentile: 背景检测的百分位数阈值
            bg_method: 背景移除方法 ('subtract', 'mask', 'blend')
        
        Note: task类型在初始化时已指定
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print("\n开始处理图像...")
        
        # 加载和预处理图像
        if self.task == 'dv_i':
            # dv_i: 红外(L), 可见光(RGB), 退化(RGB)
            i_tensor = self.preprocess_image(i_path, 'L').to(self.device)
            v_tensor = self.preprocess_image(v_path, 'RGB').to(self.device)
            d_tensor = self.preprocess_image(d_path, 'RGB').to(self.device)
            
            # 推理
            with torch.no_grad():
                out, clean = self.model(i_tensor, d_tensor, self.device, self.device, self.device)
            
            # 移除红外背景（如果启用）
            if remove_ir_background:
                print("\n" + "="*60)
                print("移除红外图像的亮背景...")
                out_processed, bg_mask = self.remove_bright_background(
                    out, i_tensor, percentile=bg_percentile, method=bg_method
                )
                print("="*60)
            else:
                out_processed = out
                bg_mask = None
            
            # 计算指标（原始融合图像）
            fusion_metrics = self.evaluator.evaluate_fusion_batch(out, v_tensor, i_tensor)
            clean_metrics = self.evaluator.evaluate_fusion_batch(clean, d_tensor, i_tensor)
            
            # 计算指标（处理后的融合图像）
            if remove_ir_background:
                fusion_metrics_processed = self.evaluator.evaluate_fusion_batch(out_processed, v_tensor, i_tensor)
            else:
                fusion_metrics_processed = None
            
        elif self.task == 'di_v':
            # di_v: 红外(L), 可见光(RGB), 退化(L)
            i_tensor = self.preprocess_image(i_path, 'L').to(self.device)
            v_tensor = self.preprocess_image(v_path, 'RGB').to(self.device)
            d_tensor = self.preprocess_image(d_path, 'L').to(self.device)
            
            # 推理
            with torch.no_grad():
                out, clean = self.model(d_tensor, v_tensor, self.device, self.device, self.device)
            
            # 移除红外背景（如果启用）
            if remove_ir_background:
                print("\n" + "="*60)
                print("移除红外图像的亮背景...")
                out_processed, bg_mask = self.remove_bright_background(
                    out, i_tensor, percentile=bg_percentile, method=bg_method
                )
                print("="*60)
            else:
                out_processed = out
                bg_mask = None
            
            # 计算指标（原始融合图像）
            fusion_metrics = self.evaluator.evaluate_fusion_batch(out, v_tensor, i_tensor)
            clean_metrics = self.evaluator.evaluate_fusion_batch(clean, d_tensor, v_tensor)
            
            # 计算指标（处理后的融合图像）
            if remove_ir_background:
                fusion_metrics_processed = self.evaluator.evaluate_fusion_batch(out_processed, v_tensor, i_tensor)
            else:
                fusion_metrics_processed = None
        
        else:
            raise ValueError(f"不支持的任务类型: {self.task}")
        
        print("\n推理完成！")
        
        # 可视化结果
        self.visualize_results(i_tensor, v_tensor, d_tensor, out, clean, 
                             fusion_metrics, clean_metrics, self.task, save_dir,
                             out_processed=out_processed if remove_ir_background else None,
                             bg_mask=bg_mask,
                             fusion_metrics_processed=fusion_metrics_processed)
        
        # 保存单独的输出图像
        self.save_output_images(out, clean, save_dir, 
                               out_processed=out_processed if remove_ir_background else None)
        
        return out, clean, fusion_metrics, clean_metrics, out_processed if remove_ir_background else out
    
    def enhance_contrast(self, img, percentile_low=2, percentile_high=98):
        """
        增强对比度，通过拉伸像素值范围
        
        Args:
            img: 输入图像 (numpy array)
            percentile_low: 低百分位数
            percentile_high: 高百分位数
        
        Returns:
            对比度增强后的图像
        """
        img_enhanced = img.copy()
        
        if len(img.shape) == 2:  # 灰度图
            p_low, p_high = np.percentile(img, (percentile_low, percentile_high))
            img_enhanced = np.clip((img - p_low) / (p_high - p_low), 0, 1)
        else:  # 彩色图
            for c in range(img.shape[2]):
                p_low, p_high = np.percentile(img[:, :, c], (percentile_low, percentile_high))
                img_enhanced[:, :, c] = np.clip((img[:, :, c] - p_low) / (p_high - p_low), 0, 1)
        
        return img_enhanced
    
    def visualize_results(self, i_tensor, v_tensor, d_tensor, out, clean,
                         fusion_metrics, clean_metrics, task, save_dir,
                         out_processed=None, bg_mask=None, fusion_metrics_processed=None):
        """
        可视化所有结果
        
        Args:
            out_processed: 背景移除后的融合图像（可选）
            bg_mask: 背景掩码（可选）
            fusion_metrics_processed: 处理后图像的指标（可选）
        """
        # 转换所有tensor为可显示的图像
        i_img = self.tensor_to_image(i_tensor)
        v_img = self.tensor_to_image(v_tensor)
        d_img = self.tensor_to_image(d_tensor)
        out_img = self.tensor_to_image(out)
        clean_img = self.tensor_to_image(clean)
        
        # 如果有处理后的图像，创建扩展布局
        if out_processed is not None:
            out_processed_img = self.tensor_to_image(out_processed)
            bg_mask_img = self.tensor_to_image(bg_mask) if bg_mask is not None else None
            
            # 创建图形 (3行4列)
            fig = plt.figure(figsize=(24, 18), facecolor='white')
            
            # 第一行：输入图像
            plt.subplot(3, 4, 1)
            if len(i_img.shape) == 2:
                plt.imshow(i_img, cmap='gray', vmin=0, vmax=1)
            else:
                plt.imshow(i_img)
            plt.title('Infrared Image', fontsize=14, fontweight='bold')
            plt.axis('off')
            
            plt.subplot(3, 4, 2)
            plt.imshow(v_img)
            plt.title('Visible Image', fontsize=14, fontweight='bold')
            plt.axis('off')
            
            plt.subplot(3, 4, 3)
            if len(d_img.shape) == 2:
                plt.imshow(d_img, cmap='gray', vmin=0, vmax=1)
            else:
                plt.imshow(d_img)
            plt.title('Degraded Image', fontsize=14, fontweight='bold')
            plt.axis('off')
            
            # 背景掩码
            plt.subplot(3, 4, 4)
            if bg_mask_img is not None:
                if len(bg_mask_img.shape) == 3:
                    plt.imshow(bg_mask_img.mean(axis=2), cmap='hot', vmin=0, vmax=1)
                else:
                    plt.imshow(bg_mask_img, cmap='hot', vmin=0, vmax=1)
                plt.title('IR Bright Regions Mask', fontsize=14, fontweight='bold')
            plt.axis('off')
            
            # 第二行：输出图像
            plt.subplot(3, 4, 5)
            plt.imshow(out_img)
            plt.title('Fused Output (Original)', fontsize=14, fontweight='bold')
            plt.axis('off')
            
            plt.subplot(3, 4, 6)
            plt.imshow(out_processed_img)
            plt.title('Fused Output (BG Removed)', fontsize=14, fontweight='bold', color='red')
            plt.axis('off')
            
            plt.subplot(3, 4, 7)
            if len(clean_img.shape) == 2:
                plt.imshow(clean_img, cmap='gray', vmin=0, vmax=1)
            else:
                plt.imshow(clean_img)
            plt.title('Clean Output', fontsize=14, fontweight='bold')
            plt.axis('off')
            
            # 对比差异
            plt.subplot(3, 4, 8)
            diff = np.abs(out_img - out_processed_img)
            if len(diff.shape) == 3:
                diff_gray = diff.mean(axis=2)
            else:
                diff_gray = diff
            plt.imshow(diff_gray, cmap='hot', vmin=0, vmax=0.5)
            plt.title('Difference (Original - Processed)', fontsize=14, fontweight='bold')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.axis('off')
            
            # 第三行：指标对比
            ax_metrics1 = plt.subplot(3, 4, 9)
            ax_metrics1.axis('off')
            metrics_text1 = "=== Original Fusion ===\n"
            for key, value in fusion_metrics.items():
                if not key.endswith('_std'):
                    metrics_text1 += f"{key}: {value:.4f}\n"
            ax_metrics1.text(0.05, 0.5, metrics_text1, fontsize=10, family='monospace',
                           verticalalignment='center', transform=ax_metrics1.transAxes)
            
            ax_metrics2 = plt.subplot(3, 4, 10)
            ax_metrics2.axis('off')
            if fusion_metrics_processed:
                metrics_text2 = "=== BG Removed Fusion ===\n"
                for key, value in fusion_metrics_processed.items():
                    if not key.endswith('_std'):
                        # 标记改进的指标
                        if key in fusion_metrics:
                            orig_val = fusion_metrics[key]
                            # 一般来说，EN, SF, MI, VIF越大越好，MSE越小越好
                            if key in ['EN', 'SF', 'MI', 'VIF', 'PSNR', 'CC', 'MS-SSIM'] and value > orig_val:
                                metrics_text2 += f"{key}: {value:.4f} ↑\n"
                            elif key == 'MSE' and value < orig_val:
                                metrics_text2 += f"{key}: {value:.4f} ↓\n"
                            else:
                                metrics_text2 += f"{key}: {value:.4f}\n"
                        else:
                            metrics_text2 += f"{key}: {value:.4f}\n"
                ax_metrics2.text(0.05, 0.5, metrics_text2, fontsize=10, family='monospace',
                               verticalalignment='center', transform=ax_metrics2.transAxes)
            
            ax_metrics3 = plt.subplot(3, 4, 11)
            ax_metrics3.axis('off')
            metrics_text3 = "=== Clean Metrics ===\n"
            for key, value in clean_metrics.items():
                if not key.endswith('_std'):
                    metrics_text3 += f"{key}: {value:.4f}\n"
            ax_metrics3.text(0.05, 0.5, metrics_text3, fontsize=10, family='monospace',
                           verticalalignment='center', transform=ax_metrics3.transAxes)
            
            plt.suptitle(f'Task: {task.upper()} - Background Removal Applied', 
                        fontsize=16, fontweight='bold')
            
        else:
            # 原始布局（2行3列）
            fig = plt.figure(figsize=(20, 12), facecolor='white')
            
            # 第一行：输入图像
            ax1 = plt.subplot(2, 3, 1)
            if len(i_img.shape) == 2:
                plt.imshow(i_img, cmap='gray', vmin=0, vmax=1)
            else:
                plt.imshow(i_img)
            plt.title('Infrared Image', fontsize=14, fontweight='bold')
            plt.axis('off')
            
            ax2 = plt.subplot(2, 3, 2)
            plt.imshow(v_img)
            plt.title('Visible Image', fontsize=14, fontweight='bold')
            plt.axis('off')
            
            ax3 = plt.subplot(2, 3, 3)
            if len(d_img.shape) == 2:
                plt.imshow(d_img, cmap='gray', vmin=0, vmax=1)
            else:
                plt.imshow(d_img)
            plt.title('Degraded Image', fontsize=14, fontweight='bold')
            plt.axis('off')
            
            # 第二行：输出图像
            ax4 = plt.subplot(2, 3, 4)
            plt.imshow(out_img)
            plt.title('Fused Output', fontsize=14, fontweight='bold')
            plt.axis('off')
            
            ax5 = plt.subplot(2, 3, 5)
            if len(clean_img.shape) == 2:
                plt.imshow(clean_img, cmap='gray', vmin=0, vmax=1)
            else:
                plt.imshow(clean_img)
            plt.title('Clean Output', fontsize=14, fontweight='bold')
            plt.axis('off')
            
            # 第三个位置显示指标
            ax6 = plt.subplot(2, 3, 6)
            ax6.axis('off')
            
            # 格式化指标文本
            metrics_text = "=== Fusion Metrics ===\n"
            for key, value in fusion_metrics.items():
                if not key.endswith('_std'):
                    metrics_text += f"{key}: {value:.4f}\n"
            
            metrics_text += "\n=== Clean Metrics ===\n"
            for key, value in clean_metrics.items():
                if not key.endswith('_std'):
                    metrics_text += f"{key}: {value:.4f}\n"
            
            ax6.text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                    verticalalignment='center', transform=ax6.transAxes)
            
            plt.suptitle(f'Task: {task.upper()}', fontsize=16, fontweight='bold')
        
        fig.patch.set_facecolor('white')
        plt.tight_layout()
        
        # 保存可视化结果
        save_path = os.path.join(save_dir, 'visualization.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"\n可视化结果已保存到: {save_path}")
        
        # 显示图像
        plt.show()
    
    def save_output_images(self, out, clean, save_dir, out_processed=None):
        """
        单独保存输出图像
        
        Args:
            out_processed: 背景移除后的融合图像（可选）
        """
        # 转换为PIL图像并保存
        out_img = self.tensor_to_image(out)
        clean_img = self.tensor_to_image(clean)
        
        # 转换为0-255范围
        out_img = (out_img * 255).astype(np.uint8)
        clean_img = (clean_img * 255).astype(np.uint8)
        
        # 保存融合输出
        if len(out_img.shape) == 3:
            out_pil = Image.fromarray(out_img, mode='RGB')
        else:
            out_pil = Image.fromarray(out_img, mode='L')
        out_path = os.path.join(save_dir, 'fused_output.png')
        out_pil.save(out_path)
        print(f"融合输出已保存到: {out_path}")
        
        # 保存clean输出
        if len(clean_img.shape) == 3:
            clean_pil = Image.fromarray(clean_img, mode='RGB')
        else:
            clean_pil = Image.fromarray(clean_img, mode='L')
        clean_path = os.path.join(save_dir, 'clean_output.png')
        clean_pil.save(clean_path)
        print(f"Clean输出已保存到: {clean_path}")
        
        # 保存处理后的融合输出（如果有）
        if out_processed is not None:
            out_processed_img = self.tensor_to_image(out_processed)
            out_processed_img = (out_processed_img * 255).astype(np.uint8)
            
            if len(out_processed_img.shape) == 3:
                out_processed_pil = Image.fromarray(out_processed_img, mode='RGB')
            else:
                out_processed_pil = Image.fromarray(out_processed_img, mode='L')
            out_processed_path = os.path.join(save_dir, 'fused_output_bg_removed.png')
            out_processed_pil.save(out_processed_path)
            print(f"融合输出（背景移除）已保存到: {out_processed_path}")
    
    def print_metrics(self, fusion_metrics, clean_metrics):
        """
        打印指标
        """
        print("\n" + "="*50)
        print("融合图像指标:")
        print("="*50)
        for key, value in fusion_metrics.items():
            print(f"{key:20s}: {value:.6f}")
        
        print("\n" + "="*50)
        print("Clean图像指标:")
        print("="*50)
        for key, value in clean_metrics.items():
            print(f"{key:20s}: {value:.6f}")
        print("="*50 + "\n")


def main():
    """
    示例用法
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='单图测试和可视化')
    parser.add_argument('--model', type=str, required=True, help='模型权重路径')
    parser.add_argument('--i_img', type=str, required=True, help='红外图像路径')
    parser.add_argument('--v_img', type=str, required=True, help='可见光图像路径')
    parser.add_argument('--d_img', type=str, required=True, help='退化图像路径')
    parser.add_argument('--task', type=str, default='dv_i', choices=['dv_i', 'di_v'],
                       help='任务类型: dv_i 或 di_v')
    parser.add_argument('--device', type=str, default='cuda:0', help='运行设备')
    parser.add_argument('--save_dir', type=str, default='./single_test_output',
                       help='保存结果的目录')
    parser.add_argument('--remove_ir_bg', action='store_true', default=True,
                       help='移除红外图像的亮背景（默认启用）')
    parser.add_argument('--no_remove_ir_bg', action='store_false', dest='remove_ir_bg',
                       help='不移除红外图像的亮背景')
    parser.add_argument('--bg_percentile', type=float, default=95,
                       help='背景检测的百分位数阈值（默认95）')
    parser.add_argument('--bg_method', type=str, default='subtract',
                       choices=['subtract', 'mask', 'blend'],
                       help='背景移除方法（默认subtract）')
    
    args = parser.parse_args()
    
    # 创建测试器（传入task参数）
    tester = SingleImageTester(args.model, task=args.task, device=args.device)
    
    # 测试单对图像
    out, clean, fusion_metrics, clean_metrics, out_processed = tester.test_single(
        args.i_img, args.v_img, args.d_img, 
        save_dir=args.save_dir,
        remove_ir_background=args.remove_ir_bg,
        bg_percentile=args.bg_percentile,
        bg_method=args.bg_method
    )
    
    # 打印指标
    tester.print_metrics(fusion_metrics, clean_metrics)


if __name__ == '__main__':
    main()
