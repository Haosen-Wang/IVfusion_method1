"""
单图测试使用示例

这个脚本展示了如何使用 SingleImageTester 进行单张图像的测试和可视化
"""

import sys
import os

# 添加项目路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from test_single_visualize import SingleImageTester


def example_dv_i():
    """
    示例1: dv_i 任务测试
    输入: 红外(L) + 退化可见光(RGB)
    输出: 融合图像 + Clean红外图像
    """
    print("\n" + "="*60)
    print("示例1: dv_i 任务测试")
    print("="*60)
    
    # 设置路径 - 请修改为你的实际路径
    model_path = "/path/to/your/model_dv_i.pth"
    i_img = "/path/to/infrared.png"
    v_img = "/path/to/visible.png"
    d_img = "/path/to/degraded_visible.png"
    
    # 创建测试器（指定task）
    tester = SingleImageTester(model_path, task='dv_i', device='cuda:0')
    
    # 测试单对图像
    out, clean, fusion_metrics, clean_metrics = tester.test_single(
        i_path=i_img,
        v_path=v_img,
        d_path=d_img,
        save_dir='./output_dv_i'
    )
    
    # 打印指标
    tester.print_metrics(fusion_metrics, clean_metrics)
    
    print("\n结果已保存到: ./output_dv_i")


def example_di_v():
    """
    示例2: di_v 任务测试
    输入: 退化红外(L) + 可见光(RGB)
    输出: 融合图像 + Clean可见光图像
    """
    print("\n" + "="*60)
    print("示例2: di_v 任务测试")
    print("="*60)
    
    # 设置路径 - 请修改为你的实际路径
    model_path = "/path/to/your/model_di_v.pth"
    i_img = "/path/to/infrared.png"
    v_img = "/path/to/visible.png"
    d_img = "/path/to/degraded_infrared.png"
    
    # 创建测试器（指定task）
    tester = SingleImageTester(model_path, task='di_v', device='cuda:0')
    
    # 测试单对图像
    out, clean, fusion_metrics, clean_metrics = tester.test_single(
        i_path=i_img,
        v_path=v_img,
        d_path=d_img,
        save_dir='./output_di_v'
    )
    
    # 打印指标
    tester.print_metrics(fusion_metrics, clean_metrics)
    
    print("\n结果已保存到: ./output_di_v")


def example_batch_test():
    """
    示例3: 批量测试多对图像
    """
    print("\n" + "="*60)
    print("示例3: 批量测试多对图像")
    print("="*60)
    
    model_path = "/path/to/your/model.pth"
    
    # 定义多对图像
    image_pairs = [
        {
            'i': '/path/to/infrared_1.png',
            'v': '/path/to/visible_1.png',
            'd': '/path/to/degraded_1.png',
            'name': 'pair_1'
        },
        {
            'i': '/path/to/infrared_2.png',
            'v': '/path/to/visible_2.png',
            'd': '/path/to/degraded_2.png',
            'name': 'pair_2'
        },
        # 添加更多图像对...
    ]
    
    # 创建测试器（指定task）
    tester = SingleImageTester(model_path, task='dv_i', device='cuda:0')
    
    # 批量测试
    for pair in image_pairs:
        print(f"\n处理图像对: {pair['name']}")
        save_dir = f"./output_batch/{pair['name']}"
        
        out, clean, fusion_metrics, clean_metrics = tester.test_single(
            i_path=pair['i'],
            v_path=pair['v'],
            d_path=pair['d'],
            save_dir=save_dir
        )
        
        print(f"完成 {pair['name']}，结果保存到: {save_dir}")


def example_custom_size():
    """
    示例4: 使用自定义图像尺寸
    """
    print("\n" + "="*60)
    print("示例4: 自定义图像尺寸测试")
    print("="*60)
    
    model_path = "/path/to/your/model.pth"
    i_img = "/path/to/infrared.png"
    v_img = "/path/to/visible.png"
    d_img = "/path/to/degraded.png"
    
    # 创建测试器（指定task）
    tester = SingleImageTester(model_path, task='dv_i', device='cuda:0')
    
    # 手动预处理图像（使用自定义尺寸）
    custom_size = (480, 640)  # 自定义尺寸
    i_tensor = tester.preprocess_image(i_img, 'L', size=custom_size).to(tester.device)
    v_tensor = tester.preprocess_image(v_img, 'RGB', size=custom_size).to(tester.device)
    d_tensor = tester.preprocess_image(d_img, 'RGB', size=custom_size).to(tester.device)
    
    # 推理
    import torch
    with torch.no_grad():
        out, clean = tester.model(i_tensor, d_tensor, tester.device, tester.device, tester.device)
    
    # 计算指标
    fusion_metrics = tester.evaluator.evaluate_fusion_batch(out, v_tensor, i_tensor)
    clean_metrics = tester.evaluator.evaluate_fusion_batch(clean, d_tensor, i_tensor)
    
    # 可视化
    tester.visualize_results(i_tensor, v_tensor, d_tensor, out, clean,
                            fusion_metrics, clean_metrics, 'dv_i', './output_custom_size')
    
    print(f"\n自定义尺寸 {custom_size} 的结果已保存")


if __name__ == '__main__':
    """
    运行示例
    
    使用方法：
    1. 修改上面示例中的路径为你的实际路径
    2. 取消注释你想运行的示例
    3. 运行: python example_single_test.py
    """
    
    # 运行示例1: dv_i任务
    # example_dv_i()
    
    # 运行示例2: di_v任务
    # example_di_v()
    
    # 运行示例3: 批量测试
    # example_batch_test()
    
    # 运行示例4: 自定义尺寸
    # example_custom_size()
    
    print("\n" + "="*60)
    print("提示: 请取消注释上面的示例代码以运行")
    print("示例1: example_dv_i() - dv_i任务测试")
    print("示例2: example_di_v() - di_v任务测试")
    print("示例3: example_batch_test() - 批量测试")
    print("示例4: example_custom_size() - 自定义尺寸")
    print("="*60)
