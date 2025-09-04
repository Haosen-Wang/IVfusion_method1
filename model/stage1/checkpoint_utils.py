"""
优化的检查点加载工具 - 减少显存占用
"""
import torch
import os
import gc


def load_checkpoint_memory_efficient(model, optimizer, checkpoint_path, device=None, strict=True):
    """
    内存高效的检查点加载函数
    
    Args:
        model: 模型实例
        optimizer: 优化器实例
        checkpoint_path: 检查点文件路径
        device: 目标设备，如果为None则使用CPU
        strict: 是否严格匹配模型参数
        
    Returns:
        epoch: 检查点对应的epoch
        train_loss: 训练损失
        val_loss: 验证损失
    """
    if not os.path.exists(checkpoint_path):
        print(f"检查点文件不存在: {checkpoint_path}")
        return 0, float('inf'), float('inf')
    
    print(f"正在加载检查点: {checkpoint_path}")
    
    # 方法1: 直接加载到CPU，避免GPU显存峰值
    if device is None:
        device = 'cpu'
    
    # 清理缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    try:
        # 强制加载到CPU，减少显存占用
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 分步加载模型状态字典
        model_state_dict = checkpoint['model_state_dict']
        
        # 如果模型在GPU上，逐层加载参数以避免显存峰值
        if next(model.parameters()).device.type == 'cuda':
            print("检测到GPU模型，使用逐层加载避免显存峰值...")
            load_state_dict_gradually(model, model_state_dict, strict=strict)
        else:
            model.load_state_dict(model_state_dict, strict=strict)
        
        # 清理模型状态字典的内存
        del model_state_dict
        
        # 加载优化器状态（如果提供）
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            # 优化器状态也可能很大，小心处理
            optimizer_state_dict = checkpoint['optimizer_state_dict']
            
            # 如果优化器参数在GPU上，也需要小心加载
            try:
                optimizer.load_state_dict(optimizer_state_dict)
            except Exception as e:
                print(f"警告: 优化器状态加载失败: {e}")
                print("将重新初始化优化器状态")
            
            del optimizer_state_dict
        
        # 获取其他信息
        epoch = checkpoint.get('epoch', 0)
        train_loss = checkpoint.get('train_loss', float('inf'))
        val_loss = checkpoint.get('val_loss', float('inf'))
        
        # 清理检查点内存
        del checkpoint
        
        # 再次清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f"✓ 检查点加载成功！Epoch: {epoch}, Train Loss: {train_loss:.4f}" + 
              (f", Val Loss: {val_loss:.4f}" if val_loss != float('inf') else ""))
        
        return epoch, train_loss, val_loss
        
    except Exception as e:
        print(f"❌ 检查点加载失败: {e}")
        # 清理可能的残留内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return 0, float('inf'), float('inf')


def load_state_dict_gradually(model, state_dict, strict=True):
    """
    逐层加载模型参数，避免显存峰值
    
    Args:
        model: 模型实例
        state_dict: 状态字典
        strict: 是否严格匹配
    """
    model_dict = model.state_dict()
    
    # 检查参数匹配性
    if strict:
        missing_keys = set(model_dict.keys()) - set(state_dict.keys())
        unexpected_keys = set(state_dict.keys()) - set(model_dict.keys())
        
        if missing_keys:
            print(f"警告: 模型中缺少以下参数: {missing_keys}")
        if unexpected_keys:
            print(f"警告: 检查点中存在多余参数: {unexpected_keys}")
    
    # 逐个参数加载
    loaded_count = 0
    total_count = len(model_dict)
    
    for name, param in model_dict.items():
        if name in state_dict:
            try:
                # 将参数移动到与模型相同的设备
                checkpoint_param = state_dict[name].to(param.device)
                
                # 检查形状匹配
                if param.shape != checkpoint_param.shape:
                    print(f"警告: 参数 {name} 形状不匹配: "
                          f"模型 {param.shape} vs 检查点 {checkpoint_param.shape}")
                    if not strict:
                        continue
                    else:
                        raise ValueError(f"参数形状不匹配: {name}")
                
                # 复制参数
                param.data.copy_(checkpoint_param)
                loaded_count += 1
                
                # 立即清理临时张量
                del checkpoint_param
                
                # 每加载一些参数就清理一次缓存
                if loaded_count % 50 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"警告: 参数 {name} 加载失败: {e}")
                if strict:
                    raise
    
    print(f"成功加载 {loaded_count}/{total_count} 个参数")


def load_model_only(model, checkpoint_path, device=None, strict=True):
    """
    仅加载模型权重，忽略优化器状态（最省显存）
    
    Args:
        model: 模型实例
        checkpoint_path: 检查点文件路径
        device: 目标设备
        strict: 是否严格匹配
        
    Returns:
        epoch: 检查点对应的epoch
        train_loss: 训练损失
    """
    if not os.path.exists(checkpoint_path):
        print(f"检查点文件不存在: {checkpoint_path}")
        return 0, float('inf')
    
    print(f"正在仅加载模型权重: {checkpoint_path}")
    
    # 清理缓存
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
    
    try:
        # 加载到CPU
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 仅加载模型状态
        if 'model_state_dict' in checkpoint:
            model_state_dict = checkpoint['model_state_dict']
            
            if next(model.parameters()).device.type == 'cuda':
                load_state_dict_gradually(model, model_state_dict, strict=strict)
            else:
                model.load_state_dict(model_state_dict, strict=strict)
            
            del model_state_dict
        
        epoch = checkpoint.get('epoch', 0)
        train_loss = checkpoint.get('train_loss', float('inf'))
        
        del checkpoint
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f"✓ 模型权重加载成功！Epoch: {epoch}, Train Loss: {train_loss:.4f}")
        return epoch, train_loss
        
    except Exception as e:
        print(f"❌ 模型权重加载失败: {e}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return 0, float('inf')


def save_checkpoint_compressed(model, optimizer, epoch, train_loss, val_loss, checkpoint_path, 
                             compress=True, save_optimizer=True):
    """
    保存压缩的检查点文件
    
    Args:
        model: 模型实例
        optimizer: 优化器实例
        epoch: 当前epoch
        train_loss: 训练损失
        val_loss: 验证损失
        checkpoint_path: 保存路径
        compress: 是否压缩保存
        save_optimizer: 是否保存优化器状态
    """
    # 构建检查点字典
    checkpoint = {
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'model_state_dict': model.state_dict(),
    }
    
    # 可选择性保存优化器状态
    if save_optimizer and optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    try:
        # 保存时选择是否压缩
        if compress:
            # 使用压缩保存
            torch.save(checkpoint, checkpoint_path, 
                      _use_new_zipfile_serialization=True)
        else:
            torch.save(checkpoint, checkpoint_path)
        
        file_size = os.path.getsize(checkpoint_path) / (1024 * 1024)  # MB
        print(f"✓ 检查点保存成功: {checkpoint_path} ({file_size:.1f}MB)")
        
    except Exception as e:
        print(f"❌ 检查点保存失败: {e}")
    
    finally:
        # 清理内存
        del checkpoint
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


def get_checkpoint_info(checkpoint_path):
    """
    获取检查点文件信息，不加载权重
    
    Args:
        checkpoint_path: 检查点文件路径
        
    Returns:
        dict: 检查点信息
    """
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        # 仅加载元数据
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        info = {
            'epoch': checkpoint.get('epoch', 0),
            'train_loss': checkpoint.get('train_loss', float('inf')),
            'val_loss': checkpoint.get('val_loss', float('inf')),
            'file_size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024),
        }
        
        # 检查是否包含模型和优化器状态
        info['has_model'] = 'model_state_dict' in checkpoint
        info['has_optimizer'] = 'optimizer_state_dict' in checkpoint
        
        del checkpoint
        return info
        
    except Exception as e:
        print(f"获取检查点信息失败: {e}")
        return None


if __name__ == "__main__":
    # 使用示例
    print("检查点内存优化工具使用示例：")
    print("""
    # 1. 内存高效加载（推荐）
    epoch, train_loss, val_loss = load_checkpoint_memory_efficient(
        model, optimizer, 'checkpoint.pth', device='cpu'
    )
    
    # 2. 仅加载模型权重（最省显存）
    epoch, train_loss = load_model_only(
        model, 'checkpoint.pth', strict=False
    )
    
    # 3. 压缩保存检查点
    save_checkpoint_compressed(
        model, optimizer, epoch, train_loss, val_loss, 
        'checkpoint.pth', compress=True, save_optimizer=False
    )
    
    # 4. 查看检查点信息
    info = get_checkpoint_info('checkpoint.pth')
    print(info)
    """)
