import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from scipy.ndimage import sobel, generic_gradient_magnitude
from scipy.signal import convolve2d
import math
from typing import Union, Tuple
import torch


class FusionMetrics:
    """
    图像融合评价指标类
    包含MSE, PSNR, NABF, MS-SSIM, CC, EN, SF, SD, VIF等指标
    支持numpy数组和PyTorch Tensor输入
    """
    
    def __init__(self):
        pass
    
    @staticmethod
    def _convert_tensor_to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        将PyTorch Tensor转换为numpy数组
        支持(b,c,h,w)和(h,w)格式
        
        Args:
            tensor: 输入张量或数组
            
        Returns:
            numpy数组，格式为(h,w)
        """
        if isinstance(tensor, torch.Tensor):
            # 转换为numpy数组
            if tensor.requires_grad:
                tensor = tensor.detach()
            if tensor.is_cuda:
                tensor = tensor.cpu()
            array = tensor.numpy()
        else:
            array = tensor
        
        # 处理不同维度
        if len(array.shape) == 4:  # (b,c,h,w)
            # 取第一个batch和第一个通道
            array = array[0, 0, :, :]
        elif len(array.shape) == 3:  # (c,h,w) or (h,w,c)
            if array.shape[0] <= 3:  # 假设是(c,h,w)
                array = array[0, :, :]
            else:  # 假设是(h,w,c)
                if array.shape[2] == 3:
                    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
                elif array.shape[2] == 1:
                    array = array.squeeze()
        elif len(array.shape) == 2:  # (h,w)
            pass  # 已经是正确格式
        else:
            raise ValueError(f"不支持的张量维度: {array.shape}")
        
        return array
    
    @staticmethod
    def _check_image_format(image: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        检查并标准化图像格式
        支持PyTorch Tensor和numpy数组输入
        """
        # 首先转换为numpy数组
        image = FusionMetrics._convert_tensor_to_numpy(image)
        
        if len(image.shape) == 3:
            # 如果是彩色图像，转为灰度
            if image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[2] == 1:
                image = image.squeeze()
        
        # 确保数据类型为float
        if image.dtype != np.float64:
            image = image.astype(np.float64)
            
        # 归一化到[0,1]范围
        if image.max() > 1.0:
            image = image / 255.0
            
        return image
    
    def mse(self, fused: Union[torch.Tensor, np.ndarray], 
            reference: Union[torch.Tensor, np.ndarray]) -> float:
        """
        计算均方误差 (Mean Square Error)
        
        Args:
            fused: 融合图像，支持(b,c,h,w)或(h,w)格式的Tensor/numpy数组
            reference: 参考图像，支持(b,c,h,w)或(h,w)格式的Tensor/numpy数组
            
        Returns:
            MSE值
        """
        fused = self._check_image_format(fused)
        reference = self._check_image_format(reference)
        
        return np.mean((fused - reference) ** 2)
    
    def psnr(self, fused: Union[torch.Tensor, np.ndarray], 
             reference: Union[torch.Tensor, np.ndarray]) -> float:
        """
        计算峰值信噪比 (Peak Signal-to-Noise Ratio)
        
        Args:
            fused: 融合图像，支持(b,c,h,w)或(h,w)格式的Tensor/numpy数组
            reference: 参考图像，支持(b,c,h,w)或(h,w)格式的Tensor/numpy数组
            
        Returns:
            PSNR值 (dB)
        """
        mse_value = self.mse(fused, reference)
        if mse_value == 0:
            return float('inf')
        
        max_pixel = 1.0  # 归一化后的最大像素值
        return 20 * math.log10(max_pixel / math.sqrt(mse_value))
    
    def nabf(self, fused: Union[torch.Tensor, np.ndarray], 
             img_a: Union[torch.Tensor, np.ndarray], 
             img_b: Union[torch.Tensor, np.ndarray]) -> float:
        """
        计算Nabf指标 (Noise and Artifact Blind/referenceless image spatial Quality)
        
        Args:
            fused: 融合图像，支持(b,c,h,w)或(h,w)格式的Tensor/numpy数组
            img_a: 源图像A，支持(b,c,h,w)或(h,w)格式的Tensor/numpy数组
            img_b: 源图像B，支持(b,c,h,w)或(h,w)格式的Tensor/numpy数组
            
        Returns:
            NABF值
        """
        fused = self._check_image_format(fused)
        img_a = self._check_image_format(img_a)
        img_b = self._check_image_format(img_b)
        
        # 计算梯度
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        # 计算各图像的梯度强度
        gx_f = convolve2d(fused, sobel_x, mode='same', boundary='symm')
        gy_f = convolve2d(fused, sobel_y, mode='same', boundary='symm')
        g_f = np.sqrt(gx_f**2 + gy_f**2)
        
        gx_a = convolve2d(img_a, sobel_x, mode='same', boundary='symm')
        gy_a = convolve2d(img_a, sobel_y, mode='same', boundary='symm')
        g_a = np.sqrt(gx_a**2 + gy_a**2)
        
        gx_b = convolve2d(img_b, sobel_x, mode='same', boundary='symm')
        gy_b = convolve2d(img_b, sobel_y, mode='same', boundary='symm')
        g_b = np.sqrt(gx_b**2 + gy_b**2)
        
        # 计算NABF
        max_gradient = np.maximum(g_a, g_b)
        
        # 避免除零
        denominator = np.sum(max_gradient) + 1e-10
        numerator = np.sum(np.abs(g_f - max_gradient))
        
        return numerator / denominator
    
    def ms_ssim(self, fused: Union[torch.Tensor, np.ndarray], 
                reference: Union[torch.Tensor, np.ndarray], 
                levels: int = 5, weights: list = None) -> float:
        """
        计算多尺度结构相似性指数 (Multi-Scale Structural Similarity Index)
        
        Args:
            fused: 融合图像，支持(b,c,h,w)或(h,w)格式的Tensor/numpy数组
            reference: 参考图像，支持(b,c,h,w)或(h,w)格式的Tensor/numpy数组
            levels: 尺度级别
            weights: 各级别权重
            
        Returns:
            MS-SSIM值
        """
        fused = self._check_image_format(fused)
        reference = self._check_image_format(reference)
        
        if weights is None:
            weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]
        
        if len(weights) != levels:
            weights = [1.0/levels] * levels
        
        # 多尺度计算
        mssim = 1.0
        
        for i in range(levels):
            # 检查图像尺寸，确保足够大
            min_size = min(fused.shape)
            if min_size < 7:  # SSIM需要至少7x7的窗口
                # 如果图像太小，只计算普通SSIM
                try:
                    ssim_val = ssim(fused, reference, data_range=1.0, win_size=min_size if min_size % 2 == 1 else min_size-1)
                except:
                    ssim_val = ssim(fused, reference, data_range=1.0, win_size=3)
                mssim *= ssim_val ** weights[i]
                break
            
            # 计算当前尺度的SSIM
            try:
                ssim_val = ssim(fused, reference, data_range=1.0)
            except ValueError:
                # 如果默认窗口大小不适用，使用较小的窗口
                win_size = min(7, min(fused.shape))
                if win_size % 2 == 0:
                    win_size -= 1
                ssim_val = ssim(fused, reference, data_range=1.0, win_size=win_size)
            
            # 对于多尺度，通常取SSIM的幂次
            if i == levels - 1:
                mssim *= ssim_val ** weights[i]
            else:
                # 下采样前检查尺寸
                if fused.shape[0] > 4 and fused.shape[1] > 4:
                    fused = cv2.resize(fused, (fused.shape[1]//2, fused.shape[0]//2))
                    reference = cv2.resize(reference, (reference.shape[1]//2, reference.shape[0]//2))
                    mssim *= ssim_val ** weights[i]
                else:
                    # 图像太小，无法继续下采样
                    mssim *= ssim_val ** weights[i]
                    break
        
        return mssim
    
    def cc(self, fused: Union[torch.Tensor, np.ndarray], 
           reference: Union[torch.Tensor, np.ndarray]) -> float:
        """
        计算相关系数 (Correlation Coefficient)
        
        Args:
            fused: 融合图像，支持(b,c,h,w)或(h,w)格式的Tensor/numpy数组
            reference: 参考图像，支持(b,c,h,w)或(h,w)格式的Tensor/numpy数组
            
        Returns:
            CC值
        """
        fused = self._check_image_format(fused)
        reference = self._check_image_format(reference)
        
        # 展平图像
        fused_flat = fused.flatten()
        reference_flat = reference.flatten()
        
        # 计算相关系数
        correlation_matrix = np.corrcoef(fused_flat, reference_flat)
        return correlation_matrix[0, 1]
    
    def entropy(self, image: Union[torch.Tensor, np.ndarray]) -> float:
        """
        计算图像熵 (Entropy)
        
        Args:
            image: 输入图像，支持(b,c,h,w)或(h,w)格式的Tensor/numpy数组
            
        Returns:
            熵值
        """
        image = self._check_image_format(image)
        
        # 量化到256级
        image_quantized = (image * 255).astype(np.uint8)
        
        # 计算直方图
        hist, _ = np.histogram(image_quantized, bins=256, range=(0, 256))
        
        # 归一化直方图得到概率分布
        hist = hist / hist.sum()
        
        # 去除零值避免log(0)
        hist = hist[hist > 0]
        
        # 计算熵
        return -np.sum(hist * np.log2(hist))
    
    def sf(self, image: Union[torch.Tensor, np.ndarray]) -> float:
        """
        计算空间频率 (Spatial Frequency)
        
        Args:
            image: 输入图像，支持(b,c,h,w)或(h,w)格式的Tensor/numpy数组
            
        Returns:
            SF值
        """
        image = self._check_image_format(image)
        
        # 行频率
        rf = np.sqrt(np.mean((image[:, 1:] - image[:, :-1]) ** 2))
        
        # 列频率
        cf = np.sqrt(np.mean((image[1:, :] - image[:-1, :]) ** 2))
        
        # 空间频率
        return np.sqrt(rf**2 + cf**2)
    
    def sd(self, image: Union[torch.Tensor, np.ndarray]) -> float:
        """
        计算标准差 (Standard Deviation)
        
        Args:
            image: 输入图像，支持(b,c,h,w)或(h,w)格式的Tensor/numpy数组
            
        Returns:
            SD值
        """
        image = self._check_image_format(image)
        return np.std(image)
    
    def vif(self, fused: Union[torch.Tensor, np.ndarray], 
            reference: Union[torch.Tensor, np.ndarray]) -> float:
        """
        计算视觉信息保真度 (Visual Information Fidelity)
        简化版本实现
        
        Args:
            fused: 融合图像，支持(b,c,h,w)或(h,w)格式的Tensor/numpy数组
            reference: 参考图像，支持(b,c,h,w)或(h,w)格式的Tensor/numpy数组
            
        Returns:
            VIF值
        """
        fused = self._check_image_format(fused)
        reference = self._check_image_format(reference)
        
        # 简化的VIF计算，基于互信息
        # 这是一个简化版本，完整的VIF需要更复杂的小波变换
        
        # 计算局部方差
        def local_variance(img, window_size=3):
            kernel = np.ones((window_size, window_size)) / (window_size ** 2)
            mu = convolve2d(img, kernel, mode='same', boundary='symm')
            mu_sq = convolve2d(img**2, kernel, mode='same', boundary='symm')
            return mu_sq - mu**2
        
        var_ref = local_variance(reference)
        var_fused = local_variance(fused)
        
        # 计算协方差
        kernel = np.ones((3, 3)) / 9
        mu_ref = convolve2d(reference, kernel, mode='same', boundary='symm')
        mu_fused = convolve2d(fused, kernel, mode='same', boundary='symm')
        
        cov = convolve2d(reference * fused, kernel, mode='same', boundary='symm') - mu_ref * mu_fused
        
        # 计算简化的VIF
        numerator = np.sum(np.log2(1 + var_ref / (var_fused + 1e-10)))
        denominator = np.sum(np.log2(1 + var_ref / 1e-10))
        
        return numerator / (denominator + 1e-10)
    
    def calculate_all_metrics(self, fused: Union[torch.Tensor, np.ndarray], 
                            img_a: Union[torch.Tensor, np.ndarray], 
                            img_b: Union[torch.Tensor, np.ndarray], 
                            reference: Union[torch.Tensor, np.ndarray] = None) -> dict:
        """
        计算所有指标
        
        Args:
            fused: 融合图像，支持(b,c,h,w)或(h,w)格式的Tensor/numpy数组
            img_a: 源图像A (可见光)，支持(b,c,h,w)或(h,w)格式的Tensor/numpy数组
            img_b: 源图像B (红外)，支持(b,c,h,w)或(h,w)格式的Tensor/numpy数组
            reference: 参考图像 (可选，用于有参考指标)，支持(b,c,h,w)或(h,w)格式的Tensor/numpy数组
            
        Returns:
            包含所有指标的字典
        """
        metrics = {}
        
        # 无参考指标
        metrics['EN'] = self.entropy(fused)
        metrics['SF'] = self.sf(fused)
        metrics['SD'] = self.sd(fused)
        metrics['NABF'] = self.nabf(fused, img_a, img_b)
        
        # 有参考指标 (如果提供了参考图像)
        if reference is not None:
            metrics['MSE'] = self.mse(fused, reference)
            metrics['PSNR'] = self.psnr(fused, reference)
            metrics['MS-SSIM'] = self.ms_ssim(fused, reference)
            metrics['CC'] = self.cc(fused, reference)
            metrics['VIF'] = self.vif(fused, reference)
        else:
            # 如果没有参考图像，可以用源图像的加权平均作为近似参考
            approx_ref = 0.5 * img_a + 0.5 * img_b
            metrics['MSE'] = self.mse(fused, approx_ref)
            metrics['PSNR'] = self.psnr(fused, approx_ref)
            metrics['MS-SSIM'] = self.ms_ssim(fused, approx_ref)
            metrics['CC'] = self.cc(fused, approx_ref)
            metrics['VIF'] = self.vif(fused, approx_ref)
        
        return metrics


    def calculate_batch_metrics(self, fused: torch.Tensor, 
                               img_a: torch.Tensor, 
                               img_b: torch.Tensor, 
                               reference: torch.Tensor = None) -> dict:
        """
        批量计算指标 - 处理整个batch的Tensor
        
        Args:
            fused: 融合图像 Tensor，形状为(b,c,h,w)
            img_a: 源图像A Tensor，形状为(b,c,h,w)  
            img_b: 源图像B Tensor，形状为(b,c,h,w)
            reference: 参考图像 Tensor (可选)，形状为(b,c,h,w)
            
        Returns:
            包含所有指标平均值的字典
        """
        batch_size = fused.shape[0]
        all_metrics = []
        
        for i in range(batch_size):
            # 取出第i个样本
            fused_i = fused[i:i+1]  # 保持4D格式
            img_a_i = img_a[i:i+1]
            img_b_i = img_b[i:i+1]
            reference_i = reference[i:i+1] if reference is not None else None
            
            # 计算单个样本的指标
            metrics_i = self.calculate_all_metrics(fused_i, img_a_i, img_b_i, reference_i)
            all_metrics.append(metrics_i)
        
        # 计算平均指标
        avg_metrics = {}
        metric_names = all_metrics[0].keys()
        
        for metric_name in metric_names:
            values = [metrics[metric_name] for metrics in all_metrics]
            avg_metrics[metric_name] = np.mean(values)
            avg_metrics[f'{metric_name}_std'] = np.std(values)  # 同时返回标准差
        
        return avg_metrics


def demo_tensor_usage():
    """
    PyTorch Tensor使用示例
    """
    import torch
    
    # 创建评价指标实例
    metrics = FusionMetrics()
    
    # 示例：创建测试Tensor (batch_size=2, channels=1, height=256, width=256)
    batch_size, channels, height, width = 2, 1, 256, 256
    
    # 模拟源图像 Tensor
    img_a = torch.randn(batch_size, channels, height, width) * 0.5 + 0.5  # [0,1]范围
    img_b = torch.randn(batch_size, channels, height, width) * 0.5 + 0.5
    
    # 模拟融合图像 Tensor  
    fused = 0.6 * img_a + 0.4 * img_b
    
    print("PyTorch Tensor 图像融合评价指标示例:")
    print("=" * 50)
    print(f"输入Tensor形状: {fused.shape}")
    
    # 方法1: 单张图像计算 (取第一个batch)
    print("\n方法1: 单张图像计算")
    print("-" * 30)
    single_metrics = metrics.calculate_all_metrics(fused[0:1], img_a[0:1], img_b[0:1])
    for metric_name, value in single_metrics.items():
        print(f"{metric_name:15s}: {value:.6f}")
    
    # 方法2: 批量计算 (整个batch的平均值)
    print("\n方法2: 批量计算 (平均值 ± 标准差)")
    print("-" * 40)
    batch_metrics = metrics.calculate_batch_metrics(fused, img_a, img_b)
    for metric_name, value in batch_metrics.items():
        if not metric_name.endswith('_std'):
            std_name = f"{metric_name}_std"
            std_value = batch_metrics.get(std_name, 0)
            print(f"{metric_name:15s}: {value:.6f} ± {std_value:.6f}")


def demo_usage():
    """
    使用示例 (numpy数组)
    """
    # 创建评价指标实例
    metrics = FusionMetrics()
    
    # 示例：创建测试图像
    height, width = 256, 256
    
    # 模拟源图像
    img_a = np.random.rand(height, width) * 255
    img_b = np.random.rand(height, width) * 255
    
    # 模拟融合图像
    fused = 0.6 * img_a + 0.4 * img_b
    
    # 计算所有指标
    all_metrics = metrics.calculate_all_metrics(fused, img_a, img_b)
    
    print("图像融合评价指标结果:")
    print("-" * 40)
    for metric_name, value in all_metrics.items():
        print(f"{metric_name:10s}: {value:.6f}")


if __name__ == "__main__":
    # numpy数组示例
    demo_usage()
    
    print("\n" + "="*60 + "\n")
    
    # PyTorch Tensor示例
    demo_tensor_usage()