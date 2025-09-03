#!/usr/bin/env python3
"""
示例：如何使用 ImageDataset 类获取文件夹内所有图像的路径
"""

import os
from dataset import ImageDataset
from torchvision import transforms

def main():
    # 示例1：基本用法
    image_dir = "/data/1024whs_data/DeMMI-RF/Test"  # 修改为你的图像文件夹路径
    
    # 创建数据集实例
    dataset = ImageDataset(image_dir)
    
    print(f"找到 {len(dataset)} 张图像")
    print("\n前10张图像的路径:")
    
    # 获取所有图像路径
    all_paths = dataset.get_all_paths()
    for i, path in enumerate(all_paths[:10]):  # 只显示前10个
        print(f"{i+1}: {path}")
    
    # 示例2：使用数据变换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    dataset_with_transform = ImageDataset(image_dir, transform=transform)
    
    # 获取第一张图像
    if len(dataset_with_transform) > 0:
        image, path = dataset_with_transform[0]
        print(f"\n第一张图像:")
        print(f"路径: {path}")
        print(f"图像shape: {image.shape}")
    
    # 示例3：按文件夹分类获取路径
    print("\n按文件夹分类的图像:")
    paths_by_folder = {}
    for path in all_paths:
        folder = os.path.dirname(path)
        if folder not in paths_by_folder:
            paths_by_folder[folder] = []
        paths_by_folder[folder].append(path)
    
    for folder, paths in list(paths_by_folder.items())[:5]:  # 只显示前5个文件夹
        print(f"{folder}: {len(paths)} 张图像")

if __name__ == "__main__":
    main()
