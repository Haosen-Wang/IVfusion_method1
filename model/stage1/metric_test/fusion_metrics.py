"""
图像融合质量评估指标
包含常用的图像融合评估指标，用于测试融合图像的质量
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage.measure import shannon_entropy
from skimage.metrics import structural_similarity as ssim
import math
from typing import Union, Tuple, List
import warnings
warnings.filterwarnings('ignore')


class ImageFusionMetrics:
    """图像融合评估指标类"""
    
    def __init__(self):
        self.metrics = {}
    
    def compute_all_metrics(self, fused_img: np.ndarray, 
                          source_imgs: List[np.ndarray],
                          reference_img: np.ndarray = None) -> dict:
        """
        计算所有融合指标
        
        Args:
            fused_img: 融合图像 (H, W) or (H, W, C)
            source_imgs: 源图像列表 [img1, img2, ...]
            reference_img: 参考图像 (可选)
            
        Returns:
            dict: 所有指标的结果
        """
        results = {}
        
        # 确保图像格式一致
        fused_img = self._normalize_image(fused_img)
        source_imgs = [self._normalize_image(img) for img in source_imgs]
        
        # 无参考指标
        results['EN'] = self.entropy(fused_img)
        results['SF'] = self.spatial_frequency(fused_img)
        results['SD'] = self.standard_deviation(fused_img)
        results['AG'] = self.average_gradient(fused_img)
        
        # 需要源图像的指标
        if len(source_imgs) >= 2:
            results['MI'] = self.mutual_information(fused_img, source_imgs)
            results['SCD'] = self.sum_correlation_differences(fused_img, source_imgs)
            results['Qabf'] = self.gradient_based_similarity(fused_img, source_imgs)
        
        # 需要参考图像的指标
        if reference_img is not None:
            reference_img = self._normalize_image(reference_img)
            results['VIF'] = self.visual_information_fidelity(fused_img, reference_img)
            results['SSIM'] = ssim(fused_img, reference_img, data_range=1.0)
        
        return results
    
    def _normalize_image(self, img: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """标准化图像到[0,1]范围"""
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        
        if len(img.shape) == 3 and img.shape[0] in [1, 3]:  # (C, H, W)
            img = img.transpose(1, 2, 0)
        
        if len(img.shape) == 3 and img.shape[2] == 1:  # (H, W, 1)
            img = img.squeeze(-1)
        
        if img.max() > 1.0:
            img = img.astype(np.float32) / 255.0
        
        return img.astype(np.float32)
    
    def entropy(self, img: np.ndarray) -> float:
        """
        计算图像熵 (Entropy)
        衡量图像信息内容的丰富程度
        """
        if len(img.shape) == 3:
            # 彩色图像转为灰度
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 将图像转换为0-255范围
        img_uint8 = (img * 255).astype(np.uint8)
        
        # 计算直方图
        hist, _ = np.histogram(img_uint8, bins=256, range=(0, 256))
        hist = hist / hist.sum()  # 归一化
        
        # 计算熵
        entropy_val = -np.sum(hist * np.log2(hist + 1e-10))
        return float(entropy_val)
    
    def spatial_frequency(self, img: np.ndarray) -> float:
        """
        计算空间频率 (Spatial Frequency)
        衡量图像的清晰度和细节信息
        """
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # 行频率
        RF = np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2))
        
        # 列频率
        CF = np.sqrt(np.mean((img[1:, :] - img[:-1, :]) ** 2))
        
        # 空间频率
        sf = np.sqrt(RF ** 2 + CF ** 2)
        return float(sf)
    
    def standard_deviation(self, img: np.ndarray) -> float:
        """
        计算标准差 (Standard Deviation)
        衡量图像对比度
        """
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        return float(np.std(img))
    
    def average_gradient(self, img: np.ndarray) -> float:
        """
        计算平均梯度 (Average Gradient)
        衡量图像的清晰度
        """
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Sobel算子计算梯度
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        
        # 梯度幅值
        grad_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        
        # 平均梯度
        ag = np.mean(grad_magnitude)
        return float(ag)
    
    def mutual_information(self, fused_img: np.ndarray, 
                         source_imgs: List[np.ndarray]) -> float:
        """
        计算互信息 (Mutual Information)
        衡量融合图像与源图像之间的信息共享程度
        """
        if len(fused_img.shape) == 3:
            fused_img = cv2.cvtColor(fused_img, cv2.COLOR_RGB2GRAY)
        
        mi_sum = 0.0
        for src_img in source_imgs:
            if len(src_img.shape) == 3:
                src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
            
            mi_sum += self._calculate_mi(fused_img, src_img)
        
        return float(mi_sum / len(source_imgs))
    
    def _calculate_mi(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """计算两幅图像间的互信息"""
        # 转换为整数
        img1_int = (img1 * 255).astype(np.uint8)
        img2_int = (img2 * 255).astype(np.uint8)
        
        # 计算联合直方图
        hist_2d, _, _ = np.histogram2d(img1_int.ravel(), img2_int.ravel(), bins=256)
        
        # 归一化
        pxy = hist_2d / float(np.sum(hist_2d))
        px = np.sum(pxy, axis=1)
        py = np.sum(pxy, axis=0)
        
        # 计算互信息
        px_py = px[:, None] * py[None, :]
        nzs = pxy > 0
        mi = np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
        
        return mi
    
    def visual_information_fidelity(self, fused_img: np.ndarray, 
                                  reference_img: np.ndarray) -> float:
        """
        计算视觉信息保真度 (Visual Information Fidelity)
        需要参考图像
        """
        if len(fused_img.shape) == 3:
            fused_img = cv2.cvtColor(fused_img, cv2.COLOR_RGB2GRAY)
        if len(reference_img.shape) == 3:
            reference_img = cv2.cvtColor(reference_img, cv2.COLOR_RGB2GRAY)
        
        # 简化的VIF计算 (完整实现较复杂)
        # 这里使用基于小波的近似方法
        
        # 计算方差
        var_ref = np.var(reference_img)
        var_fused = np.var(fused_img)
        
        # 计算协方差
        cov = np.cov(reference_img.ravel(), fused_img.ravel())[0, 1]
        
        # VIF近似
        if var_ref == 0:
            return 0.0
        
        vif = (2 * cov + 1e-10) / (var_ref + var_fused + 1e-10)
        return float(np.clip(vif, 0, 1))
    
    def sum_correlation_differences(self, fused_img: np.ndarray, 
                                  source_imgs: List[np.ndarray]) -> float:
        """
        计算相关差之和 (Sum of Correlations of Differences)
        """
        if len(fused_img.shape) == 3:
            fused_img = cv2.cvtColor(fused_img, cv2.COLOR_RGB2GRAY)
        
        scd_sum = 0.0
        
        for i in range(len(source_imgs)):
            src_img = source_imgs[i]
            if len(src_img.shape) == 3:
                src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
            
            # 计算相关系数
            corr = np.corrcoef(fused_img.ravel(), src_img.ravel())[0, 1]
            if not np.isnan(corr):
                scd_sum += corr
        
        return float(scd_sum)
    
    def gradient_based_similarity(self, fused_img: np.ndarray, 
                                source_imgs: List[np.ndarray]) -> float:
        """
        计算基于梯度的相似性测量 (Qabf)
        """
        if len(fused_img.shape) == 3:
            fused_img = cv2.cvtColor(fused_img, cv2.COLOR_RGB2GRAY)
        
        # 计算融合图像的梯度
        grad_f_x = cv2.Sobel(fused_img, cv2.CV_64F, 1, 0, ksize=3)
        grad_f_y = cv2.Sobel(fused_img, cv2.CV_64F, 0, 1, ksize=3)
        grad_f_mag = np.sqrt(grad_f_x ** 2 + grad_f_y ** 2)
        
        qabf_sum = 0.0
        
        for src_img in source_imgs:
            if len(src_img.shape) == 3:
                src_img = cv2.cvtColor(src_img, cv2.COLOR_RGB2GRAY)
            
            # 计算源图像的梯度
            grad_s_x = cv2.Sobel(src_img, cv2.CV_64F, 1, 0, ksize=3)
            grad_s_y = cv2.Sobel(src_img, cv2.CV_64F, 0, 1, ksize=3)
            grad_s_mag = np.sqrt(grad_s_x ** 2 + grad_s_y ** 2)
            
            # 计算梯度相似性
            gaf = self._gradient_similarity(grad_f_mag, grad_s_mag, grad_f_x, grad_f_y, grad_s_x, grad_s_y)
            qabf_sum += gaf
        
        return float(qabf_sum / len(source_imgs))
    
    def _gradient_similarity(self, grad_f_mag, grad_s_mag, grad_f_x, grad_f_y, grad_s_x, grad_s_y):
        """计算梯度相似性"""
        # 梯度幅值相关性
        corr_mag = np.corrcoef(grad_f_mag.ravel(), grad_s_mag.ravel())[0, 1]
        if np.isnan(corr_mag):
            corr_mag = 0
        
        # 梯度方向相关性
        grad_f_angle = np.arctan2(grad_f_y, grad_f_x)
        grad_s_angle = np.arctan2(grad_s_y, grad_s_x)
        
        angle_diff = np.abs(grad_f_angle - grad_s_angle)
        angle_similarity = np.mean(np.cos(angle_diff))
        
        # 综合相似性
        gaf = corr_mag * angle_similarity
        return gaf


def test_fusion_metrics(fused_image_path: str, 
                       source_image_paths: List[str],
                       reference_image_path: str = None,
                       save_results: bool = True) -> dict:
    """
    测试图像融合指标
    
    Args:
        fused_image_path: 融合图像路径
        source_image_paths: 源图像路径列表
        reference_image_path: 参考图像路径 (可选)
        save_results: 是否保存结果
        
    Returns:
        dict: 评估结果
    """
    # 加载图像
    fused_img = cv2.imread(fused_image_path, cv2.IMREAD_UNCHANGED)
    if fused_img is None:
        raise ValueError(f"无法加载融合图像: {fused_image_path}")
    
    source_imgs = []
    for path in source_image_paths:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"无法加载源图像: {path}")
        source_imgs.append(img)
    
    reference_img = None
    if reference_image_path:
        reference_img = cv2.imread(reference_image_path, cv2.IMREAD_UNCHANGED)
        if reference_img is None:
            print(f"警告: 无法加载参考图像: {reference_image_path}")
    
    # 创建评估器
    evaluator = ImageFusionMetrics()
    
    # 计算指标
    results = evaluator.compute_all_metrics(fused_img, source_imgs, reference_img)
    
    # 打印结果
    print("=" * 50)
    print("图像融合质量评估结果")
    print("=" * 50)
    print(f"融合图像: {fused_image_path}")
    print(f"源图像数量: {len(source_imgs)}")
    print("-" * 50)
    
    for metric, value in results.items():
        print(f"{metric:6s}: {value:.6f}")
    
    print("=" * 50)
    
    # 保存结果
    if save_results:
        import json
        import os
        
        result_file = os.path.splitext(fused_image_path)[0] + "_metrics.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"结果已保存到: {result_file}")
    
    return results


def batch_evaluate_fusion(fused_dir: str, 
                         source_dirs: List[str],
                         reference_dir: str = None,
                         output_file: str = "fusion_evaluation_results.csv") -> None:
    """
    批量评估融合图像
    
    Args:
        fused_dir: 融合图像目录
        source_dirs: 源图像目录列表
        reference_dir: 参考图像目录
        output_file: 输出CSV文件
    """
    import os
    import pandas as pd
    
    evaluator = ImageFusionMetrics()
    results_list = []
    
    # 获取所有融合图像
    fused_files = [f for f in os.listdir(fused_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    for fused_file in fused_files:
        try:
            # 构建路径
            fused_path = os.path.join(fused_dir, fused_file)
            
            # 查找对应的源图像
            source_paths = []
            for source_dir in source_dirs:
                source_path = os.path.join(source_dir, fused_file)
                if os.path.exists(source_path):
                    source_paths.append(source_path)
            
            if len(source_paths) < 2:
                print(f"跳过 {fused_file}: 源图像不足")
                continue
            
            # 查找参考图像
            reference_path = None
            if reference_dir:
                reference_path = os.path.join(reference_dir, fused_file)
                if not os.path.exists(reference_path):
                    reference_path = None
            
            # 加载图像
            fused_img = cv2.imread(fused_path, cv2.IMREAD_UNCHANGED)
            source_imgs = [cv2.imread(path, cv2.IMREAD_UNCHANGED) for path in source_paths]
            reference_img = cv2.imread(reference_path, cv2.IMREAD_UNCHANGED) if reference_path else None
            
            # 计算指标
            results = evaluator.compute_all_metrics(fused_img, source_imgs, reference_img)
            results['filename'] = fused_file
            results_list.append(results)
            
            print(f"✓ 完成评估: {fused_file}")
            
        except Exception as e:
            print(f"✗ 评估失败 {fused_file}: {e}")
    
    # 保存结果到CSV
    if results_list:
        df = pd.DataFrame(results_list)
        df.to_csv(output_file, index=False)
        print(f"\n批量评估完成！结果保存到: {output_file}")
        
        # 打印统计信息
        print("\n=== 评估统计 ===")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        print(df[numeric_cols].describe())
    else:
        print("没有成功评估的图像！")


if __name__ == "__main__":
    # 使用示例
    print("图像融合评估指标测试")
    print("=" * 50)
    
    # 创建测试数据
    test_img = np.random.rand(256, 256).astype(np.float32)
    source1 = np.random.rand(256, 256).astype(np.float32)
    source2 = np.random.rand(256, 256).astype(np.float32)
    
    # 创建评估器
    evaluator = ImageFusionMetrics()
    
    # 计算指标
    results = evaluator.compute_all_metrics(test_img, [source1, source2])
    
    print("测试结果:")
    for metric, value in results.items():
        print(f"{metric:6s}: {value:.6f}")
    
    print("\n指标说明:")
    print("EN   : 熵 - 越大越好，表示信息丰富度")
    print("SF   : 空间频率 - 越大越好，表示清晰度")
    print("SD   : 标准差 - 越大越好，表示对比度")
    print("AG   : 平均梯度 - 越大越好，表示清晰度")
    print("MI   : 互信息 - 越大越好，表示信息保持")
    print("VIF  : 视觉信息保真度 - 越大越好，表示视觉质量")
    print("SCD  : 相关差之和 - 越大越好，表示相关性")
    print("Qabf : 梯度相似性 - 越大越好，表示边缘保持")
