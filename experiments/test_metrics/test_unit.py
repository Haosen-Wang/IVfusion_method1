#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像融合评价指标单元测试
"""

import unittest
import numpy as np
import cv2
from test_metric import FusionMetrics


class TestFusionMetrics(unittest.TestCase):
    """
    图像融合评价指标单元测试类
    """
    
    def setUp(self):
        """
        测试前准备
        """
        self.metrics = FusionMetrics()
        
        # 创建测试图像
        self.height, self.width = 128, 128
        
        # 创建不同特征的测试图像
        # 可见光图像 - 更多纹理细节
        self.vis_img = self._create_texture_image()
        
        # 红外图像 - 更多边缘信息
        self.ir_img = self._create_edge_image()
        
        # 融合图像 - 简单加权融合
        self.fusion_img = 0.6 * self.vis_img + 0.4 * self.ir_img
        
        # 理想融合图像 - 作为参考
        self.reference_img = self._create_ideal_fusion()
    
    def _create_texture_image(self):
        """创建纹理丰富的图像"""
        np.random.seed(42)
        texture = np.random.rand(self.height, self.width) * 0.3
        
        # 添加一些结构
        y, x = np.ogrid[:self.height, :self.width]
        structure = 0.5 + 0.3 * np.sin(x * 0.1) * np.cos(y * 0.1)
        
        return np.clip(texture + structure, 0, 1)
    
    def _create_edge_image(self):
        """创建边缘丰富的图像"""
        img = np.zeros((self.height, self.width))
        
        # 添加一些矩形边缘
        img[30:60, 30:90] = 0.8
        img[70:100, 40:80] = 0.6
        
        # 添加噪声
        np.random.seed(123)
        noise = np.random.rand(self.height, self.width) * 0.1
        
        return np.clip(img + noise, 0, 1)
    
    def _create_ideal_fusion(self):
        """创建理想融合图像"""
        # 结合两个图像的优势
        return np.maximum(self.vis_img, self.ir_img) * 0.8 + \
               np.minimum(self.vis_img, self.ir_img) * 0.2
    
    def test_image_format_conversion(self):
        """测试图像格式转换"""
        # 测试彩色图像转换
        color_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        converted = self.metrics._check_image_format(color_img)
        
        self.assertEqual(len(converted.shape), 2)  # 应该是灰度图
        self.assertEqual(converted.dtype, np.float64)  # 应该是float类型
        self.assertTrue(converted.max() <= 1.0)  # 应该归一化
        
        # 测试已经是正确格式的图像
        correct_img = np.random.rand(64, 64)
        converted = self.metrics._check_image_format(correct_img)
        self.assertTrue(np.allclose(converted, correct_img))
    
    def test_mse(self):
        """测试MSE指标"""
        # 相同图像的MSE应该为0
        mse_same = self.metrics.mse(self.vis_img, self.vis_img)
        self.assertAlmostEqual(mse_same, 0.0, places=10)
        
        # 不同图像的MSE应该大于0
        mse_diff = self.metrics.mse(self.fusion_img, self.vis_img)
        self.assertGreater(mse_diff, 0)
        
        # MSE应该在合理范围内
        self.assertLess(mse_diff, 1.0)
    
    def test_psnr(self):
        """测试PSNR指标"""
        # 相同图像的PSNR应该是无穷大
        psnr_same = self.metrics.psnr(self.vis_img, self.vis_img)
        self.assertEqual(psnr_same, float('inf'))
        
        # 不同图像的PSNR应该是有限正值
        psnr_diff = self.metrics.psnr(self.fusion_img, self.reference_img)
        self.assertGreater(psnr_diff, 0)
        self.assertLess(psnr_diff, 100)  # 通常不会超过100dB
    
    def test_entropy(self):
        """测试熵指标"""
        # 随机图像应该有较高的熵
        random_img = np.random.rand(self.height, self.width)
        entropy_random = self.metrics.entropy(random_img)
        
        # 均匀图像应该有较低的熵
        uniform_img = np.ones((self.height, self.width)) * 0.5
        entropy_uniform = self.metrics.entropy(uniform_img)
        
        self.assertGreater(entropy_random, entropy_uniform)
        self.assertGreater(entropy_random, 5.0)  # 随机图像熵应该较高
        self.assertLess(entropy_uniform, 2.0)    # 均匀图像熵应该较低
    
    def test_spatial_frequency(self):
        """测试空间频率指标"""
        # 边缘丰富的图像应该有更高的空间频率
        sf_edge = self.metrics.sf(self.ir_img)
        sf_smooth = self.metrics.sf(np.ones((self.height, self.width)) * 0.5)
        
        self.assertGreater(sf_edge, sf_smooth)
        self.assertGreater(sf_edge, 0)
    
    def test_standard_deviation(self):
        """测试标准差指标"""
        # 对比度高的图像应该有更高的标准差
        high_contrast = np.zeros((self.height, self.width))
        high_contrast[:self.height//2] = 1.0
        
        low_contrast = np.ones((self.height, self.width)) * 0.5
        
        sd_high = self.metrics.sd(high_contrast)
        sd_low = self.metrics.sd(low_contrast)
        
        self.assertGreater(sd_high, sd_low)
        self.assertAlmostEqual(sd_low, 0.0, places=10)
    
    def test_correlation_coefficient(self):
        """测试相关系数指标"""
        # 相同图像的相关系数应该为1
        cc_same = self.metrics.cc(self.vis_img, self.vis_img)
        self.assertAlmostEqual(cc_same, 1.0, places=10)
        
        # 不同图像的相关系数应该在[-1,1]范围内
        cc_diff = self.metrics.cc(self.fusion_img, self.vis_img)
        self.assertGreaterEqual(cc_diff, -1.0)
        self.assertLessEqual(cc_diff, 1.0)
    
    def test_nabf(self):
        """测试NABF指标"""
        nabf = self.metrics.nabf(self.fusion_img, self.vis_img, self.ir_img)
        
        # NABF应该是非负值
        self.assertGreaterEqual(nabf, 0)
        
        # 理想情况下NABF应该较小
        self.assertLess(nabf, 2.0)
    
    def test_ms_ssim(self):
        """测试多尺度SSIM指标"""
        # 相同图像的MS-SSIM应该接近1
        ms_ssim_same = self.metrics.ms_ssim(self.vis_img, self.vis_img)
        self.assertGreater(ms_ssim_same, 0.99)
        
        # 不同图像的MS-SSIM应该在(0,1)范围内
        ms_ssim_diff = self.metrics.ms_ssim(self.fusion_img, self.reference_img)
        self.assertGreater(ms_ssim_diff, 0)
        self.assertLessEqual(ms_ssim_diff, 1.0)
    
    def test_vif(self):
        """测试VIF指标"""
        vif = self.metrics.vif(self.fusion_img, self.reference_img)
        
        # VIF应该是正值
        self.assertGreater(vif, 0)
    
    def test_calculate_all_metrics(self):
        """测试计算所有指标功能"""
        # 测试无参考图像情况
        metrics_no_ref = self.metrics.calculate_all_metrics(
            self.fusion_img, self.vis_img, self.ir_img
        )
        
        expected_metrics = ['EN', 'SF', 'SD', 'NABF', 'MSE', 'PSNR', 'MS-SSIM', 'CC', 'VIF']
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics_no_ref)
            self.assertIsInstance(metrics_no_ref[metric], (int, float))
            self.assertFalse(np.isnan(metrics_no_ref[metric]))
        
        # 测试有参考图像情况
        metrics_with_ref = self.metrics.calculate_all_metrics(
            self.fusion_img, self.vis_img, self.ir_img, self.reference_img
        )
        
        for metric in expected_metrics:
            self.assertIn(metric, metrics_with_ref)
    
    def test_edge_cases(self):
        """测试边界情况"""
        # 测试全零图像
        zero_img = np.zeros((32, 32))
        
        # 测试全一图像
        one_img = np.ones((32, 32))
        
        # 这些不应该引发异常
        try:
            self.metrics.entropy(zero_img)
            self.metrics.sf(zero_img)
            self.metrics.sd(zero_img)
            self.metrics.mse(zero_img, one_img)
            self.metrics.psnr(zero_img, one_img)
        except Exception as e:
            self.fail(f"边界情况测试失败: {e}")
    
    def test_different_image_sizes(self):
        """测试不同尺寸图像的处理"""
        small_img = np.random.rand(32, 32)
        large_img = np.random.rand(128, 128)
        
        # 不同尺寸的图像计算应该不会出错
        try:
            entropy_small = self.metrics.entropy(small_img)
            entropy_large = self.metrics.entropy(large_img)
            
            self.assertIsInstance(entropy_small, float)
            self.assertIsInstance(entropy_large, float)
        except Exception as e:
            self.fail(f"不同尺寸图像测试失败: {e}")


def run_performance_test():
    """
    性能测试
    """
    import time
    
    print("\n运行性能测试...")
    print("=" * 50)
    
    metrics = FusionMetrics()
    
    # 创建大尺寸测试图像
    sizes = [(256, 256), (512, 512), (1024, 1024)]
    
    for height, width in sizes:
        print(f"\n测试图像尺寸: {height}x{width}")
        print("-" * 30)
        
        # 创建测试图像
        vis_img = np.random.rand(height, width)
        ir_img = np.random.rand(height, width)
        fusion_img = 0.5 * vis_img + 0.5 * ir_img
        
        # 测试各个指标的计算时间
        start_time = time.time()
        result = metrics.calculate_all_metrics(fusion_img, vis_img, ir_img)
        end_time = time.time()
        
        print(f"总计算时间: {end_time - start_time:.3f}秒")
        
        # 单独测试耗时较长的指标
        start_time = time.time()
        _ = metrics.ms_ssim(fusion_img, vis_img)
        ms_ssim_time = time.time() - start_time
        
        start_time = time.time()
        _ = metrics.vif(fusion_img, vis_img)
        vif_time = time.time() - start_time
        
        print(f"MS-SSIM时间: {ms_ssim_time:.3f}秒")
        print(f"VIF时间: {vif_time:.3f}秒")


if __name__ == "__main__":
    # 运行单元测试
    print("运行图像融合评价指标单元测试...")
    unittest.main(verbosity=2, exit=False)
    
    # 运行性能测试
    run_performance_test()
