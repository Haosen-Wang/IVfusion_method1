import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))  # 添加项目根目录到路径
import torch
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import wandb
import torch.nn as nn
from model import Degrad_restore_model
from loss import Loss
from data_process.dataset import ImageDataset
from PIL import Image
import math
import argparse
import torchvision.transforms as transforms
import yaml
import gc
from checkpoint_utils import (
    load_checkpoint_memory_efficient, 
    load_model_only, 
    save_checkpoint_compressed,
    get_checkpoint_info
)

class PairedDataset(Dataset):
        def __init__(self, i_dataset, v_dataset):
            assert len(i_dataset) == len(v_dataset), "两个数据集长度必须相等"
            self.i_dataset = i_dataset
            self.v_dataset = v_dataset
        
        def __len__(self):
            return len(self.i_dataset)
        
        def __getitem__(self, idx):
            i_image= self.i_dataset[idx][0]
            v_image = self.v_dataset[idx][0]
            return i_image, v_image

def load_config(config_path, experiment=None):
    """
    加载YAML配置文件
    
    Args:
        config_path: YAML配置文件路径
        experiment: 实验名称，如果指定则会合并对应实验的配置
    
    Returns:
        config: 配置字典
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    # 如果指定了实验名称，合并实验配置
    if experiment and 'experiments' in config and experiment in config['experiments']:
        exp_config = config['experiments'][experiment]
        config = merge_configs(config, exp_config)
    
    return config

def merge_configs(base_config, override_config):
    """
    递归合并配置字典
    """
    result = base_config.copy()
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result

def config_to_args(config):
    """
    将配置字典转换为参数字典，用于传递给main函数
    """
    args = {}
    
    # 数据配置
    if 'data' in config:
        args.update({
            'd_data_dir': config['data'].get('d_data_dir'),
            'c_data_dir': config['data'].get('c_data_dir'),
            'mode': config['data'].get('mode', 'L')
        })
    
    # 训练配置
    if 'training' in config:
        args.update({
            'project_name': config['training'].get('project_name'),
            'batch_size': config['training'].get('batch_size', 2),
            'num_epochs': config['training'].get('num_epochs', 5),
            'resume_from_checkpoint': config['training'].get('resume_from_checkpoint', False)
        })
    
    # 设备配置
    if 'device' in config:
        args.update({
            'device_1': config['device'].get('device_1', 'cuda:0'),
            'device_2': config['device'].get('device_2', 'cuda:1'),
            'device_3': config['device'].get('device_3', 'cuda:2')
        })
    
    # 模型配置
    if 'model' in config:
        args.update({
            'i_block_num': config['model'].get('i_block_num', 2),
            'v_block_num': config['model'].get('v_block_num', 2),
            'i_expert_num': config['model'].get('i_expert_num', 3),
            'v_expert_num': config['model'].get('v_expert_num', 3),
            'i_topk_expert': config['model'].get('i_topk_expert', 2),
            'v_topk_expert': config['model'].get('v_topk_expert', 2),
            'i_alpha': config['model'].get('i_alpha', 1.0),
            'v_alpha': config['model'].get('v_alpha', 1.0),
            'f_block_num': config['model'].get('f_block_num', 3)
        })
    
    # 内存优化配置
    if 'memory' in config:
        args.update({
            'memory_efficient': config['memory'].get('checkpoint_memory_efficient', True),
            'load_optimizer_state': config['memory'].get('load_optimizer_state', True),
            'compress_checkpoints': config['memory'].get('compress_checkpoints', True),
            'save_optimizer_in_epoch_checkpoints': config['memory'].get('save_optimizer_in_epoch_checkpoints', False)
        })
    else:
        # 默认内存优化设置
        args.update({
            'memory_efficient': True,
            'load_optimizer_state': True,
            'compress_checkpoints': True,
            'save_optimizer_in_epoch_checkpoints': False
        })
    
    return args
def train_epoch_model(model, train_loader, criterion, optimizer, device_1, device_2, val_loader, pbar):
    model.train()
    running_loss = 0.0
    epoch_loss = 0.0
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    
    for batch_idx, (d_image, c_image) in enumerate(pbar):
        try:
            Ic_image, n, mu_n, sigma2_n =model(d_image,device_1, device_2)

            # 计算损失
            criterion=criterion.to(device_2)
            # 将模型输出移动到device_3
            d_image=d_image.to(device_2)
            c_image=c_image.to(device_2)
            Ic_image = Ic_image.to(device_2)
            n = n.to(device_2)
            mu_n = mu_n.to(device_2)
            sigma2_n = sigma2_n.to(device_2)
            torch.cuda.empty_cache()
            loss_all= criterion(Ic_image, d_image, c_image, n, mu_n, sigma2_n)
            del d_image,c_image,Ic_image,n,mu_n,sigma2_n
            loss=loss_all["total_loss"]
            
            # 检测 NaN 和 Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  批次 {batch_idx} 检测到 NaN/Inf 损失，跳过此批次...")
                print(f"损失详情: {loss_all}")
                del loss_all
                torch.cuda.empty_cache()
                continue

            # 反向传播
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            wandb.log(loss_all)
            torch.cuda.empty_cache()
            
            # 统计损失
            running_loss += loss.item()
            epoch_loss += loss.item()

                
                
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"⚠️  批次 {batch_idx} 内存不足，跳过...")
                torch.cuda.empty_cache()
                continue
            else:
                raise e
                
    return epoch_loss
def train_model(model, train_loader, criterion, optimizer, device_1, device_2, project_name, num_epochs=10, val_loader=None, checkpoint_dir="./checkpoints"):
    """
    训练函数，集成wandb监控和检查点保存
    
    Args:
        model: 训练模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device_1, device_2, device_3: 设备
        num_epochs: 训练轮数
        val_loader: 验证数据加载器
        project_name: wandb项目名称
        checkpoint_dir: 检查点保存目录
    """
    # 创建检查点目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 初始化wandb
    wandb.init(project=project_name)
    
    # 初始化最佳性能指标
    best_val_loss = float('inf')
    best_train_loss = float('inf') 
    best_epoch = 0
    
    model.train()
    
    for epoch in range(num_epochs):
        
        # 使用tqdm显示进度条
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        # 学习率调度器 - 随着训练减小学习率
        if epoch == 0:
            # 在第一个epoch创建学习率调度器
            #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
            # 或者使用余弦退火调度器
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
            # 或者使用指数衰减
            # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        # 在每个epoch结束后更新学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f'当前学习率: {current_lr:.6f}')
        
        # 在每个epoch结束后更新学习率
        epoch_loss = train_epoch_model(model, train_loader, criterion, optimizer, device_1, device_2, val_loader, pbar)
        scheduler.step()
        # 计算epoch平均损失
        avg_loss = epoch_loss / len(train_loader)

        # 验证阶段
        val_metrics = {}
        val_loss = None
        if val_loader is not None:
            val_loss, val_acc = validate_model(model, val_loader, criterion, device_1, device_2)
            val_metrics = {"val_loss": val_loss}
        
        # 记录epoch指标到wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            **val_metrics
        })
        # 保存当前epoch检查点
        # 创建检查点字典
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_loss,
            'val_loss': val_loss,
            **val_metrics
        }
        
        # 使用压缩保存减少文件大小
        # 保存最新检查点
        latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        save_checkpoint_compressed(
            model, optimizer, epoch + 1, avg_loss, val_loss,
            latest_checkpoint_path, compress=True, save_optimizer=True
        )
        
        # 保存每个epoch的检查点（可选择不保存优化器状态以节省空间）
        epoch_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        save_checkpoint_compressed(
            model, optimizer, epoch + 1, avg_loss, val_loss,
            epoch_checkpoint_path, compress=True, save_optimizer=False  # 不保存优化器减少文件大小
        )
        
        # 判断是否为最佳模型
        is_best = False
        current_metric = val_loss if val_loss is not None else avg_loss
        best_metric = best_val_loss if val_loss is not None else best_train_loss
        
        if current_metric < best_metric:
            is_best = True
            if val_loss is not None:
                best_val_loss = val_loss
            else:
                best_train_loss = avg_loss
            best_epoch = epoch + 1
            
            # 保存最佳模型 - 使用压缩保存
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            save_checkpoint_compressed(
                model, None, epoch + 1, avg_loss, val_loss,  # 不保存优化器状态
                best_checkpoint_path, compress=True, save_optimizer=False
            )
            print(f"✓ 新的最佳模型已保存！ (Epoch {epoch+1})")
        
        # 清理内存
        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # 打印epoch损失
        print(f'Epoch [{epoch+1}/{num_epochs}],Train Loss: {avg_loss:.4f}' + 
              (f', Val Loss: {val_metrics.get("val_loss", 0):.4f}, Val Acc: {val_metrics.get("val_accuracy", 0):.4f}' if val_metrics else '') +
              (f' {"🏆" if is_best else ""}'))
    
    print(f"\n训练完成！最佳模型来自 Epoch {best_epoch}")
    print(f"检查点保存路径: {checkpoint_dir}")
    return best_epoch, best_val_loss if val_loader else best_train_loss

def validate_model(model, val_loader, criterion, device_1, device_2, device_3):
    """
    验证函数
    """
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for i,v in tqdm(val_loader, desc='Validating'):
            output,l, g, mu_l, sigma2_l, mu_g, sigma2_g= model(i,v, device_1, device_2, device_3)
            loss_all= criterion(output, i,v, l, g, mu_l, sigma2_l, mu_g, sigma2_g)
            val_loss=loss_all["total_loss"]
            
    model.train()
    return val_loss / len(val_loader)


def load_checkpoint(model, optimizer, checkpoint_path, memory_efficient=True, load_optimizer=True):
    """
    加载检查点 - 支持内存优化
    """
    if memory_efficient:  # 修复逻辑错误
        # 内存高效的加载方式
        print(f"正在使用内存优化模式加载检查点: {checkpoint_path}")
        
        if not os.path.exists(checkpoint_path):
            print(f"检查点文件不存在: {checkpoint_path}")
            return 0, float('inf'), None
            
        # 直接加载到CPU，避免GPU显存占用
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 确保模型在CPU上
        model = model.to('cpu')
        
        # 直接加载状态字典，无需额外拷贝
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and load_optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        #if optimizer is not None and load_optimizer and 'optimizer_state_dict' in checkpoint:
            #optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint.get('val_loss', None)
        
        # 立即清理检查点数据
        del checkpoint
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"✓ 检查点加载成功！Epoch: {epoch}, Train Loss: {train_loss:.4f}" + 
              (f", Val Loss: {val_loss:.4f}" if val_loss else ""))
        
        return epoch, train_loss, val_loss
    else:
        # 使用其他加载函数
        if load_optimizer:
            epoch, train_loss, val_loss = load_checkpoint_memory_efficient(
                model, optimizer, checkpoint_path, device='cpu'
            )
        else:
            epoch, train_loss = load_model_only(
                model, checkpoint_path, device='cpu', strict=False
            )
            val_loss = None
        
        return epoch, train_loss, val_loss


def resume_training(model, optimizer, checkpoint_dir="/data/1024whs_checkpoint/Degradclean", memory_efficient=True, load_optimizer=True):
    """
    从最新检查点恢复训练 - 支持内存优化
    
    Args:
        model: 模型实例
        optimizer: 优化器实例
        checkpoint_dir: 检查点目录
        memory_efficient: 是否使用内存高效模式
        load_optimizer: 是否加载优化器状态
    
    Returns:
        start_epoch: 恢复训练的起始epoch
    """
    latest_checkpoint = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    if os.path.exists(latest_checkpoint):
        print("🔄 发现检查点，恢复训练...")
        
        # 先查看检查点信息
        info = get_checkpoint_info(latest_checkpoint)
        if info:
            print(f"📋 检查点信息: Epoch {info['epoch']}, "
                  f"Train Loss: {info['train_loss']:.4f}, "
                  f"文件大小: {info['file_size_mb']:.1f}MB")
        
        # 内存高效加载
        epoch, _, _ = load_checkpoint(model, optimizer, latest_checkpoint, 
                                    memory_efficient=memory_efficient, 
                                    load_optimizer=load_optimizer)
        
        # 清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return 0  # 下一个epoch
    else:
        print("🆕 未找到最新检查点，从头开始训练")
        return 0


def check_data(d_dataset,c_dataset):
     if len(d_dataset) != len(c_dataset):
        print(f"警告: 退化数量({len(d_dataset)})与可见光图像数量({len(c_dataset)})不相等")
        # 取较小的数量，确保配对
        min_len = min(len(d_dataset), len(c_dataset))
        d_dataset.image_paths = d_dataset.image_paths[:min_len]
        c_dataset.image_paths = c_dataset.image_paths[:min_len]
        print(f"已调整为相同数量: {min_len}")
    
    # 确保文件名排序一致
     d_dataset.image_paths.sort()
     c_dataset.image_paths.sort()
    
    # 验证文件名对应关系（假设文件名格式相同，只是扩展名或前缀不同）
     d_names = [os.path.splitext(os.path.basename(path))[0] for path in d_dataset.image_paths]
     c_names = [os.path.splitext(os.path.basename(path))[0] for path in c_dataset.image_paths]

     if d_names != c_names:
        print("警告: 退化和干净图像文件名不完全对应")
        # 找到共同的文件名
        common_names = set(d_names) & set(c_names)
        print(f"共同文件数量: {len(common_names)}")
        
        # 重新筛选对应的文件
        d_filtered = []
        c_filtered = []
        for name in sorted(common_names):
            d_idx = d_names.index(name)
            c_idx = c_names.index(name)
            d_filtered.append(d_dataset.image_paths[d_idx])
            c_filtered.append(c_dataset.image_paths[c_idx])

        d_dataset.image_paths = d_filtered
        c_dataset.image_paths = c_filtered
        print(f"已筛选出对应的图像对: {len(d_dataset.image_paths)}")

     print(f"最终配对数量: 退化图像{len(d_dataset)}, 干净图像{len(c_dataset)}")

def main(d_data_dir, c_data_dir, project_name, batch_size, num_epochs=10, device_1="cuda:0", device_2="cuda:1", device_3="cuda:2", resume_from_checkpoint=True,i_block_num=2,v_block_num=2,i_expert_num=3,v_expert_num=3,i_topk_expert=2,v_topk_expert=2,i_alpha=1.0,v_alpha=1.0,f_block_num=3,mode="L", memory_efficient=True, load_optimizer_state=True, compress_checkpoints=True, save_optimizer_in_epoch_checkpoints=False):
    if mode=="L":
        transform = transforms.Compose([
            transforms.Resize((224,224)),  # 调整图像大小
            transforms.ToTensor(),          # 转换为张量 [0,1]
            transforms.Normalize(mean=[0.5], std=[0.5])  # 单通道标准化
        ])
    if mode=="RGB":
        transform = transforms.Compose([
            transforms.Resize((224,224)),  # 调整图像大小
            transforms.ToTensor(),          # 转换为张量 [0,1]
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # RGB标准化
        ])
    
    d_dataset = ImageDataset(d_data_dir, transform=transform, convert=mode)
    c_dataset = ImageDataset(c_data_dir, transform=transform, convert=mode)
    
    # 检查数据集
    check_data(d_dataset, c_dataset)
    
    # 创建联合数据集
    dc_dataset = PairedDataset(d_dataset, c_dataset)
    
    print(f"联合数据集创建完成，包含 {len(dc_dataset)} 对图像")

    # 创建数据加载器 - 优化内存使用
    train_loader = DataLoader(dc_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=2, pin_memory=False)  # 减少workers和禁用pin_memory
    print(f"数据加载器创建完成，批次大小:{batch_size}")
    
    # 内存优化设置
    torch.backends.cudnn.benchmark = False  # 禁用cudnn benchmark以节省内存
    torch.cuda.empty_cache()  # 清理GPU缓存
    
    # 初始化模型、损失函数、优化器
    model = Degrad_restore_model(i_block_num=2,v_block_num=2,i_expert_num=3,v_expert_num=3,i_topk_expert=2,v_topk_expert=2,i_alpha=1.0,v_alpha=1.0,f_block_num=3,mode=mode)  # 你的模型
    
    criterion = Loss()  # 你的损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 降低学习率
    
    # 参数检查
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数总数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 权重初始化检查
    has_nan_weights = any(torch.isnan(p).any() for p in model.parameters())
    has_inf_weights = any(torch.isinf(p).any() for p in model.parameters())
    if has_nan_weights or has_inf_weights:
        print("⚠️  警告：模型权重包含 NaN 或 Inf 值！")
    else:
        print("✓ 模型权重初始化正常")
    
    # 检查点目录
    checkpoint_dir = f"/data/1024whs_checkpoint/Degradclean/{project_name}"
    
    # 是否从检查点恢复训练
    start_epoch = 0
    if resume_from_checkpoint:
        start_epoch = resume_training(
            model, optimizer, checkpoint_dir, 
            memory_efficient=memory_efficient, 
            load_optimizer=load_optimizer_state
        )
    
    # 调整训练轮数
    remaining_epochs = max(0, num_epochs - start_epoch)
    if remaining_epochs == 0:
        print("训练已完成！")
        return
    
    print(f"开始训练，从 epoch {start_epoch + 1} 到 epoch {num_epochs}")
    
    # 开始训练
    best_epoch, best_loss = train_model(
        model=model, 
        train_loader=train_loader, 
        criterion=criterion, 
        optimizer=optimizer, 
        device_1=device_1, 
        device_2=device_2,  
        project_name=project_name,
        num_epochs=remaining_epochs, 
        val_loader=None,  # 如果有验证数据，在这里传入
        checkpoint_dir=checkpoint_dir
    )
    
    print(f"\n🎉 训练完成！")
    print(f"📊 最佳模型: Epoch {best_epoch}, Loss: {best_loss:.4f}")
    print(f"💾 模型检查点保存在: {checkpoint_dir}")
    
    # 加载最佳模型进行测试（示例）
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        print(f"✓ 最佳模型路径: {best_model_path}")
        # 这里可以加载最佳模型进行推理或测试
        # load_checkpoint(model, None, best_model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练退化恢复模型')
    
    # 添加配置文件选项
    parser.add_argument('--config', type=str, help='YAML配置文件路径')
    parser.add_argument('--experiment', type=str, help='实验名称（可选）')
    
    # 保留原有的命令行参数作为备选
    parser.add_argument('--d_data_dir', type=str, default="/data/1024whs_data/DeMMI-RF/Train/degrad/Stripe/DroneRGBT", help='退化图像数据目录')
    parser.add_argument('--c_data_dir', type=str, default="/data/1024whs_data/DeMMI-RF/Train/infrared/Stripe/DroneRGBT", help='干净图像数据目录')
    parser.add_argument('--project_name', type=str, default="Train_Degradclean_DroneRGBT_stripe", help='项目名称')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=5, help='训练轮数')
    parser.add_argument('--device_1', type=str, default="cuda:0", help='第一个GPU设备')
    parser.add_argument('--device_2', type=str, default="cuda:1", help='第二个GPU设备')
    parser.add_argument('--resume_from_checkpoint', default=True, action='store_true', help='是否从检查点恢复训练')
    parser.add_argument('--i_block_num', type=int, default=2, help='红外分支块数量')
    parser.add_argument('--v_block_num', type=int, default=2, help='可见光分支块数量')
    parser.add_argument('--i_expert_num', type=int, default=3, help='红外专家数量')
    parser.add_argument('--v_expert_num', type=int, default=3, help='可见光专家数量')
    parser.add_argument('--i_topk_expert', type=int, default=2, help='红外topk专家')
    parser.add_argument('--v_topk_expert', type=int, default=2, help='可见光topk专家')
    parser.add_argument('--i_alpha', type=float, default=1.0, help='红外alpha参数')
    parser.add_argument('--v_alpha', type=float, default=1.0, help='可见光alpha参数')
    parser.add_argument('--f_block_num', type=int, default=3, help='融合块数量')
    parser.add_argument('--mode', type=str, default="L", choices=["L", "RGB"], help='图像模式')
    
    args = parser.parse_args()
    
    # 如果指定了配置文件，则从配置文件加载参数
    if args.config:
        print(f"📄 从配置文件加载参数: {args.config}")
        if args.experiment:
            print(f"🧪 使用实验配置: {args.experiment}")
        
        try:
            config = load_config(args.config, args.experiment)
            config_dic = config_to_args(config)
            
            # 打印配置信息
            print("📋 配置文件内容:")
            for section, content in config.items():
                if section != 'experiments' and isinstance(content, dict):
                    print(f"  {section}:")
                    for key, value in content.items():
                        print(f"    {key}: {value}")
            
        except FileNotFoundError:
            print(f"❌ 错误: 配置文件不存在: {args.config}")
            exit(1)
        except yaml.YAMLError as e:
            print(f"❌ 错误: YAML文件格式错误: {e}")
            exit(1)
        except Exception as e:
            print(f"❌ 错误: 加载配置文件失败: {e}")
            exit(1)
    else:
        # 使用命令行参数
        print("⌨️  使用命令行参数")
        config_dic = {
            "d_data_dir": args.d_data_dir,
            "c_data_dir": args.c_data_dir,
            "project_name": args.project_name,
            "batch_size": args.batch_size,
            "num_epochs": args.num_epochs,
            "device_1": args.device_1,
            "device_2": args.device_2,
            "device_3": getattr(args, 'device_3', 'cuda:2'),
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "i_block_num": args.i_block_num,
            "v_block_num": args.v_block_num,
            "i_expert_num": args.i_expert_num,
            "v_expert_num": args.v_expert_num,
            "i_topk_expert": args.i_topk_expert,
            "v_topk_expert": args.v_topk_expert,
            "i_alpha": args.i_alpha,
            "v_alpha": args.v_alpha,
            "f_block_num": args.f_block_num,
            "mode": args.mode,
            "memory_efficient": getattr(args, 'memory_efficient', True),
            "load_optimizer_state": getattr(args, 'load_optimizer_state', True),
            "compress_checkpoints": getattr(args, 'compress_checkpoints', True),
            "save_optimizer_in_epoch_checkpoints": getattr(args, 'save_optimizer_in_epoch_checkpoints', False)
        }
    
    print("训练配置:")
    for key, value in config_dic.items():
        print(f"  {key}: {value}")
    
    main(**config_dic)