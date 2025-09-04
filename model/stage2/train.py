import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))  # 添加项目根目录到路径
import torch
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import wandb
import torch.nn as nn
from model import IV_fusion_model
from loss import Loss
from data_process.dataset import ImageDataset
from PIL import Image
import math
import torchvision.transforms as transforms

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
def train_epoch_model(model, train_loader, criterion, optimizer, device_1, device_2, device_3, val_loader, pbar):
    model.train()
    running_loss = 0.0
    epoch_loss = 0.0
    
    # 清理GPU缓存
    torch.cuda.empty_cache()
    
    for batch_idx, (i_image, v_image) in enumerate(pbar):
        try:
            output, l, g, mu_l, sigma2_l, mu_g, sigma2_g = model(i_image, v_image, device_1, device_2, device_3)
            
            # 计算损失
            criterion=criterion.to(device_3)
            # 将模型输出移动到device_3
            i_image=i_image.to(device_3)
            v_image=v_image.to(device_3)
            output = output.to(device_3)
            l = l.to(device_3)
            g = g.to(device_3)
            mu_l = mu_l.to(device_3)
            sigma2_l = sigma2_l.to(device_3)
            mu_g = mu_g.to(device_3)
            sigma2_g = sigma2_g.to(device_3)
            torch.cuda.empty_cache()
            loss_all= criterion(output, i_image, v_image, l, g, mu_l, sigma2_l, mu_g, sigma2_g)
            del i_image,v_image,output,l,g,mu_l,sigma2_l,mu_g,sigma2_g
            loss=loss_all["total_loss"]
            
            # 检测 NaN 和 Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  批次 {batch_idx} 检测到 NaN/Inf 损失，跳过此批次...")
                print(f"损失详情: {loss_all}")
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
def train_model(model, train_loader, criterion, optimizer, device_1, device_2, device_3, project_name, num_epochs=10, val_loader=None, checkpoint_dir="./checkpoints"):
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
        epoch_loss = train_epoch_model(model, train_loader, criterion, optimizer, device_1, device_2, device_3, val_loader, pbar)
        scheduler.step()
        # 计算epoch平均损失
        avg_loss = epoch_loss / len(train_loader)

        # 验证阶段
        val_metrics = {}
        val_loss = None
        if val_loader is not None:
            val_loss, val_acc = validate_model(model, val_loader, criterion, device_1, device_2, device_3)
            val_metrics = {"val_loss": val_loss}
        
        # 记录epoch指标到wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
            **val_metrics
        })
        
        # 保存当前epoch检查点
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_loss,
            'val_loss': val_loss,
            **val_metrics
        }
        
        # 保存最新检查点
        latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
        torch.save(checkpoint, latest_checkpoint_path)
        
        # 保存每个epoch的检查点
        epoch_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
        torch.save(checkpoint, epoch_checkpoint_path)
        
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
            
            # 保存最佳模型
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_checkpoint_path)
            print(f"✓ 新的最佳模型已保存！ (Epoch {epoch+1})")
        
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


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    加载检查点
    
    Args:
        model: 模型实例
        optimizer: 优化器实例
        checkpoint_path: 检查点文件路径
    
    Returns:
        epoch: 检查点对应的epoch
        loss: 检查点对应的损失
    """
    if os.path.exists(checkpoint_path):
        print(f"正在加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        train_loss = checkpoint['train_loss']
        val_loss = checkpoint.get('val_loss', None)
        
        print(f"✓ 检查点加载成功！Epoch: {epoch}, Train Loss: {train_loss:.4f}" + 
              (f", Val Loss: {val_loss:.4f}" if val_loss else ""))
        
        return epoch, train_loss, val_loss
    else:
        print(f"检查点文件不存在: {checkpoint_path}")
        return 0, None, None


def resume_training(model, optimizer, checkpoint_dir="/data/1024whs_checkpoint/iv_fusion"):
    """
    从最新检查点恢复训练
    
    Args:
        model: 模型实例
        optimizer: 优化器实例
        checkpoint_dir: 检查点目录
    
    Returns:
        start_epoch: 恢复训练的起始epoch
    """
    latest_checkpoint = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    if os.path.exists(latest_checkpoint):
        epoch, _, _ = load_checkpoint(model, optimizer, latest_checkpoint)
        return epoch
    else:
        print("未找到最新检查点，从头开始训练")
        return 0


def check_data(i_dataset,v_dataset):
     if len(i_dataset) != len(v_dataset):
        print(f"警告: 红外图像数量({len(i_dataset)})与可见光图像数量({len(v_dataset)})不相等")
        # 取较小的数量，确保配对
        min_len = min(len(i_dataset), len(v_dataset))
        i_dataset.image_paths = i_dataset.image_paths[:min_len]
        v_dataset.image_paths = v_dataset.image_paths[:min_len]
        print(f"已调整为相同数量: {min_len}")
    
    # 确保文件名排序一致
     i_dataset.image_paths.sort()
     v_dataset.image_paths.sort()
    
    # 验证文件名对应关系（假设文件名格式相同，只是扩展名或前缀不同）
     i_names = [os.path.splitext(os.path.basename(path))[0] for path in i_dataset.image_paths]
     v_names = [os.path.splitext(os.path.basename(path))[0] for path in v_dataset.image_paths]
    
     if i_names != v_names:
        print("警告: 红外和可见光图像文件名不完全对应")
        # 找到共同的文件名
        common_names = set(i_names) & set(v_names)
        print(f"共同文件数量: {len(common_names)}")
        
        # 重新筛选对应的文件
        i_filtered = []
        v_filtered = []
        for name in sorted(common_names):
            i_idx = i_names.index(name)
            v_idx = v_names.index(name)
            i_filtered.append(i_dataset.image_paths[i_idx])
            v_filtered.append(v_dataset.image_paths[v_idx])
        
        i_dataset.image_paths = i_filtered
        v_dataset.image_paths = v_filtered
        print(f"已筛选出对应的图像对: {len(i_dataset.image_paths)}")
    
     print(f"最终配对数量: 红外图像{len(i_dataset)}, 可见光图像{len(v_dataset)}")

def main(i_data_dir, v_data_dir, project_name, batch_size, num_epochs=10, device_1="cuda:0", device_2="cuda:1", device_3="cuda:2", resume_from_checkpoint=True,i_block_num=2,v_block_num=2,i_expert_num=3,v_expert_num=3,i_topk_expert=2,v_topk_expert=2,i_alpha=1.0,v_alpha=1.0,f_block_num=3):
    # 为红外图像（单通道）创建变换
    transform_i = transforms.Compose([
        transforms.Resize((224,224)),  # 调整图像大小
        transforms.ToTensor(),          # 转换为张量 [0,1]
        transforms.Normalize(mean=[0.5], std=[0.5])  # 单通道标准化
    ])
    
    # 为可见光图像（3通道）创建变换
    transform_v = transforms.Compose([
        transforms.Resize((224,224)),  # 调整图像大小
        transforms.ToTensor(),          # 转换为张量 [0,1]
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # RGB标准化
    ])
    
    i_dataset = ImageDataset(i_data_dir, transform=transform_i, convert="L")
    v_dataset = ImageDataset(v_data_dir, transform=transform_v, convert="RGB")
    
    # 检查数据集
    check_data(i_dataset, v_dataset)
    
    # 创建联合数据集
    iv_dataset = PairedDataset(i_dataset, v_dataset)
    
    print(f"联合数据集创建完成，包含 {len(iv_dataset)} 对图像")

    # 创建数据加载器 - 优化内存使用
    train_loader = DataLoader(iv_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=2, pin_memory=False)  # 减少workers和禁用pin_memory
    print(f"数据加载器创建完成，批次大小:{batch_size}")
    
    # 内存优化设置
    torch.backends.cudnn.benchmark = False  # 禁用cudnn benchmark以节省内存
    torch.cuda.empty_cache()  # 清理GPU缓存
    
    # 初始化模型、损失函数、优化器
    model = IV_fusion_model(i_block_num=2,v_block_num=2,i_expert_num=3,v_expert_num=3,i_topk_expert=2,v_topk_expert=2,i_alpha=1.0,v_alpha=1.0,f_block_num=3)  # 你的模型
    
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
    checkpoint_dir = f"/data/1024whs_checkpoint/iv_fusion/{project_name}"
    
    # 是否从检查点恢复训练
    start_epoch = 0
    if resume_from_checkpoint:
        start_epoch = resume_training(model, optimizer, checkpoint_dir)
    
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
        device_3=device_3, 
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
    config_dic = {
        "i_data_dir": "/data/1024whs_data/DeMMI-RF/Train_fusion/DroneVehicle/infrared",
        "v_data_dir": "/data/1024whs_data/DeMMI-RF/Train_fusion/DroneVehicle/visible",
        "project_name": "Train_IVfusion_DroneVehicle",
        "batch_size": 2,  # 减小批次大小从2到1
        "num_epochs": 5,
        "device_1": "cuda:0", 
        "device_2": "cuda:1",
        "device_3": "cuda:2",
        "resume_from_checkpoint": False  # 设置为True来从检查点恢复训练
    }
    main(**config_dic)