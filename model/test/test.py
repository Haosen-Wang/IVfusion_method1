import sys
import os
import importlib.util
# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 添加model目录到Python路径
model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(model_dir)

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

stage1_dir = os.path.join(current_dir, "stage3")
from test_metrics.tensor_example import TensorFusionEvaluator
import torch
from stage3.model import DIV_fusion_model
import torch
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import torch.nn as nn
from data_process.dataset import ImageDataset
from PIL import Image
import math
import argparse
import yaml
import torchvision.transforms as transforms
import json
from datetime import datetime

def get_device(config):
    """根据配置获取设备"""
    device_config = config['test']['device']
    if device_config == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        return device_config

class PairedDataset(Dataset):
        def __init__(self, i_dataset, v_dataset,d_dataset):
            assert len(i_dataset) == len(v_dataset)==len(d_dataset), "两个数据集长度必须相等"
            self.i_dataset = i_dataset
            self.v_dataset = v_dataset
            self.d_dataset = d_dataset

        def __len__(self):
            return len(self.i_dataset)
        
        def __getitem__(self, idx):
            i_image= self.i_dataset[idx][0]
            v_image = self.v_dataset[idx][0]
            d_image = self.d_dataset[idx][0]
            return i_image, v_image, d_image
def check_data(i_dataset,v_dataset,d_dataset):
    """根据配置获取设备"""
    device_config = config['test']['device']
    if device_config == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        return device_config

def filter_metrics_by_config(metrics, config):
    """根据配置过滤需要的指标"""
    if 'evaluation' not in config or 'metrics' not in config['evaluation']:
        return metrics
    
    required_metrics = config['evaluation']['metrics']
    filtered_metrics = {}
    
    for metric_name, value in metrics.items():
        # 检查基础指标名（去掉_std后缀）
        base_metric = metric_name.replace('_std', '')
        if base_metric in required_metrics or metric_name in required_metrics:
            filtered_metrics[metric_name] = value
    
    return filtered_metrics

def save_results(config, all_results, overall_stats=None):
    """保存测试结果"""
    if not config['output']['save_results']:
        return
        
    results_dir = config['output']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    # 创建结果文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(results_dir, f"test_results_{timestamp}.json")
    
    # 准备保存的数据
    save_data = {
        'config': config,
        'test_results': all_results,
        'overall_stats': overall_stats,
        'timestamp': timestamp
    }
    
    # 保存到JSON文件
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已保存到: {result_file}")

def calculate_overall_stats(all_results):
    """计算多次测试的总体统计"""
    overall_stats = {}
    
    # 获取所有指标名称
    metric_names = set()
    for result in all_results:
        metric_names.update(result.keys())
    
    for metric_name in metric_names:
        if not metric_name.endswith('_std'):
            values = [result[metric_name] for result in all_results if metric_name in result]
            if values:
                values_tensor = torch.tensor(values)
                overall_stats[metric_name] = {
                    'mean': values_tensor.mean().item(),
                    'std': values_tensor.std().item(),
                    'min': values_tensor.min().item(),
                    'max': values_tensor.max().item()
                }
    
    return overall_stats

def check_data(i_dataset, v_dataset, d_dataset):
    if len(i_dataset) != len(v_dataset):
        print(f"警告: 红外图像数量({len(i_dataset)})与可见光图像数量({len(v_dataset)})不相等与退化图像数量({len(d_dataset)})不相等")
        # 取较小的数量，确保配对
        min_len = min(len(i_dataset), len(v_dataset), len(d_dataset))
        i_dataset.image_paths = i_dataset.image_paths[:min_len]
        v_dataset.image_paths = v_dataset.image_paths[:min_len]
        d_dataset.image_paths = d_dataset.image_paths[:min_len]
        print(f"已调整为相同数量: {min_len}")
    
    # 确保文件名排序一致
    i_dataset.image_paths.sort()
    v_dataset.image_paths.sort()
    d_dataset.image_paths.sort()
    
    # 验证文件名对应关系（假设文件名格式相同，只是扩展名或前缀不同）
    i_names = [os.path.splitext(os.path.basename(path))[0] for path in i_dataset.image_paths]
    v_names = [os.path.splitext(os.path.basename(path))[0] for path in v_dataset.image_paths]
    d_names = [os.path.splitext(os.path.basename(path))[0] for path in d_dataset.image_paths]
    if i_names != v_names or i_names != d_names or v_names != d_names:
        print("警告: 文件名不完全对应")
        # 找到共同的文件名
        common_names = set(i_names) & set(v_names) & set(d_names)
        print(f"共同文件数量: {len(common_names)}")
        
        # 重新筛选对应的文件
        i_filtered = []
        v_filtered = []
        d_filtered = []
        for name in sorted(common_names):
            i_idx = i_names.index(name)
            v_idx = v_names.index(name)
            d_idx = d_names.index(name)
            i_filtered.append(i_dataset.image_paths[i_idx])
            v_filtered.append(v_dataset.image_paths[v_idx])
            d_filtered.append(d_dataset.image_paths[d_idx])
        
        i_dataset.image_paths = i_filtered
        v_dataset.image_paths = v_filtered
        d_dataset.image_paths = d_filtered
        print(f"已筛选出对应的图像对: {len(i_dataset.image_paths)}")

    print(f"最终配对数量: 红外图像{len(i_dataset)}, 可见光图像{len(v_dataset)}, 退化图像{len(d_dataset)}")
def test_model(model, dataloader, device, task="dv_i", config=None):
    Evaluator = TensorFusionEvaluator()
    all_metrics = {}
    with torch.no_grad():
        for i_batch, (i_data, v_data, d_data) in enumerate(tqdm(dataloader, desc="Testing")):
            # 将数据移到设备上
            i_data = i_data.to(device)
            v_data = v_data.to(device)
            d_data = d_data.to(device)
            
            if task == 'dv_i':
                out, clean = model(i_data, d_data, device, device, device)
                batch_metrics = Evaluator.evaluate_fusion_batch(out, v_data, i_data)
            elif task == 'di_v':
                out, clean = model(d_data, v_data)
                batch_metrics = Evaluator.evaluate_fusion_batch(out, v_data, i_data)
            
            # 根据配置过滤指标
            if config:
                batch_metrics = filter_metrics_by_config(batch_metrics, config)
                
            for metric_name, value in batch_metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
    
    # 计算每个指标的平均值和标准差
    averaged_metrics = {}
    for metric_name, values in all_metrics.items():
        values_tensor = torch.tensor(values)
        averaged_metrics[metric_name] = values_tensor.mean().item()
        averaged_metrics[f"{metric_name}_std"] = values_tensor.std().item()
    
    return averaged_metrics
def load_datasets(config):
    """根据配置加载数据集"""
    # 从配置获取路径
    i_dir = config['data']['i_dir']
    v_dir = config['data']['v_dir']
    d_dir = config['data']['d_dir']
    batch_size = config['test']['batch_size']
    num_workers = config['test']['num_workers']
    pin_memory = config['test']['pin_memory']
    
    # 创建变换 - 使用固定的240x240尺寸
    transform_i = transforms.Compose([
        transforms.Resize((240,240)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.253], std=[0.191]) #LLVIP
    ])
    
    transform_v = transforms.Compose([
        transforms.Resize((240,240)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.188, 0.186, 0.154], std=[0.183, 0.190, 0.197]) #LLVIP
    ])
    
    transform_d = transforms.Compose([
        transforms.Resize((240,240)),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.188, 0.186, 0.154], std=[0.183, 0.190, 0.197]) #LLVIP
    ])
    
    # 红外图像使用灰度模式 (L)
    i_dataset = ImageDataset(i_dir, transform=transform_i, convert='L')
    # 可见光图像使用RGB模式
    v_dataset = ImageDataset(v_dir, transform=transform_v, convert='RGB')
    # 退化图像使用RGB模式
    d_dataset = ImageDataset(d_dir, transform=transform_d, convert='RGB')
    
    check_data(i_dataset, v_dataset, d_dataset)
    paired_dataset = PairedDataset(i_dataset, v_dataset, d_dataset)
    print(f"联合数据集创建完成，包含 {len(paired_dataset)} 对图像")

    # 创建数据加载器
    data_loader = DataLoader(
        paired_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=pin_memory
    )
    print(f"数据加载器创建完成，批次大小:{batch_size}")
    return data_loader
def main(config):
    """主函数，根据配置进行测试"""
    print(f"开始实验: {config['experiment']['name']}")
    print(f"描述: {config['experiment']['description']}")
    print(f"版本: {config['experiment']['version']}")
    print()
    
    # 加载数据集
    data_loader = load_datasets(config)
    
    # 获取设备
    device = get_device(config)
    print(f"使用设备: {device}")
    
    # 加载模型
    task = config['model']['task']
    checkpoint_path = config['model']['checkpoint']
    
    model = DIV_fusion_model(task=task).to(device)
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"成功加载模型: {checkpoint_path}")
    else:
        print(f"错误: 找不到检查点文件 {checkpoint_path}")
        return
    
    # 进行测试
    test_times = config['test']['test_times']
    all_test_results = []
    
    for test_time in range(test_times):
        print(f"\n=== 第 {test_time + 1}/{test_times} 次测试 ===")
        with torch.no_grad():
            averaged_metrics = test_model(model, data_loader, device, task=task, config=config)
        
        all_test_results.append(averaged_metrics)
        
        print("本次测试结果:")
        filtered_for_display = filter_metrics_by_config(averaged_metrics, config) if config else averaged_metrics
        
        for metric_name, value in filtered_for_display.items(): 
            if not metric_name.endswith('_std'):
                std_name = f"{metric_name}_std"
                std_value = filtered_for_display.get(std_name, 0)
                print(f"  {metric_name:15s}: {value:8.6f} ± {std_value:.6f}")
    
    # 计算多次测试的总体统计
    overall_stats = None
    if test_times > 1:
        print(f"\n=== {test_times}次测试总体统计 ===")
        overall_stats = calculate_overall_stats(all_test_results)
        for metric_name, stats in overall_stats.items():
            if not metric_name.endswith('_std'):
                print(f"  {metric_name:15s}: {stats['mean']:8.6f} ± {stats['std']:.6f} (范围: {stats['min']:.6f} - {stats['max']:.6f})")
    
    # 保存结果
    save_results(config, all_test_results, overall_stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test LLVIP fusion model with YAML config')
    parser.add_argument('--config', type=str, default='test_llvip_config.yaml', 
                       help='Path to config file')
    parser.add_argument('--i_dir', type=str, help='Infrared image directory (overrides config)')
    parser.add_argument('--v_dir', type=str, help='Visible image directory (overrides config)')
    parser.add_argument('--d_dir', type=str, help='Degraded image directory (overrides config)')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint path (overrides config)')
    parser.add_argument('--task', type=str, help='Task type (overrides config)')
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--device', type=str, help='Device to use (overrides config)')
    
    args = parser.parse_args()
    
    # 加载YAML配置文件
    config_path = args.config
    if not os.path.exists(config_path):
        print(f"错误: 找不到配置文件 {config_path}")
        print("请确保配置文件存在或使用 --config 参数指定正确路径")
        sys.exit(1)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"成功加载配置文件: {config_path}")
    except Exception as e:
        print(f"错误: 无法加载配置文件 {config_path}: {e}")
        sys.exit(1)
    
    # 命令行参数覆盖配置文件设置
    if args.i_dir:
        config['data']['i_dir'] = args.i_dir
    if args.v_dir:
        config['data']['v_dir'] = args.v_dir
    if args.d_dir:
        config['data']['d_dir'] = args.d_dir
    if args.checkpoint:
        config['model']['checkpoint'] = args.checkpoint
    if args.task:
        config['model']['task'] = args.task
    if args.batch_size:
        config['test']['batch_size'] = args.batch_size
    if args.device:
        config['test']['device'] = args.device
    
    # 验证必要的配置项
    required_paths = [
        config['data']['i_dir'],
        config['data']['v_dir'], 
        config['data']['d_dir']
    ]
    
    for path in required_paths:
        if not os.path.exists(path):
            print(f"错误: 路径不存在 {path}")
            sys.exit(1)
    
    if not os.path.exists(config['model']['checkpoint']):
        print(f"错误: 检查点文件不存在 {config['model']['checkpoint']}")
        sys.exit(1)
    
    print("配置验证通过，开始测试...")
    main(config)