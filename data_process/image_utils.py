#!/usr/bin/env python3
"""
简化版本：获取文件夹内所有图像路径的函数
"""

import os
from pathlib import Path

def get_image_paths(folder_path, recursive=True, supported_formats=None):
    """
    获取文件夹内所有图像文件的路径
    
    Args:
        folder_path (str): 图像文件夹路径
        recursive (bool): 是否递归搜索子文件夹，默认True
        supported_formats (list): 支持的图像格式列表，默认为常见格式
    
    Returns:
        list: 图像文件路径列表
    """
    if supported_formats is None:
        supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.webp']
    
    # 转换为小写以便比较
    supported_formats = [fmt.lower() for fmt in supported_formats]
    
    image_paths = []
    folder_path = Path(folder_path)
    
    if not folder_path.exists():
        print(f"错误：文件夹 {folder_path} 不存在")
        return image_paths
    
    # 选择搜索模式
    if recursive:
        # 递归搜索所有子文件夹
        pattern = "**/*"
        files = folder_path.glob(pattern)
    else:
        # 只搜索当前文件夹
        files = folder_path.iterdir()
    
    # 筛选图像文件
    for file_path in files:
        if file_path.is_file() and file_path.suffix.lower() in supported_formats:
            image_paths.append(str(file_path.absolute()))
    
    return sorted(image_paths)

def get_images_by_subfolder(folder_path):
    """
    按子文件夹分组获取图像路径
    
    Args:
        folder_path (str): 根文件夹路径
    
    Returns:
        dict: {子文件夹名: [图像路径列表]}
    """
    images_by_folder = {}
    folder_path = Path(folder_path)
    
    # 遍历所有子文件夹
    for subfolder in folder_path.iterdir():
        if subfolder.is_dir():
            subfolder_name = subfolder.name
            images = get_image_paths(subfolder, recursive=False)
            if images:  # 只添加包含图像的文件夹
                images_by_folder[subfolder_name] = images
    
    return images_by_folder

def print_image_summary(folder_path):
    """
    打印文件夹图像统计信息
    
    Args:
        folder_path (str): 文件夹路径
    """
    print(f"正在扫描文件夹: {folder_path}")
    
    # 获取所有图像
    all_images = get_image_paths(folder_path)
    print(f"总共找到 {len(all_images)} 张图像")
    
    # 按子文件夹分组
    images_by_folder = get_images_by_subfolder(folder_path)
    print(f"包含图像的子文件夹数量: {len(images_by_folder)}")
    
    print("\n各子文件夹图像数量:")
    for folder_name, images in images_by_folder.items():
        print(f"  {folder_name}: {len(images)} 张")
    
    return all_images, images_by_folder

# 使用示例
if __name__ == "__main__":
    # 修改为你的文件夹路径
    test_folder = "/data/1024whs_data/DeMMI-RF/Test"
    
    # 获取所有图像路径
    all_paths = get_image_paths(test_folder)
    
    print("示例用法:")
    print(f"1. 递归获取所有图像: {len(all_paths)} 张")
    
    # 显示前5张图像路径
    if all_paths:
        print("\n前5张图像路径:")
        for i, path in enumerate(all_paths[:5]):
            print(f"  {i+1}: {path}")
    
    # 打印统计信息
    print("\n" + "="*50)
    print_image_summary(test_folder)
