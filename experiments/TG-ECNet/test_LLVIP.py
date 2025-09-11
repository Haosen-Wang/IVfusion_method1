import sys
import os
import importlib.util
# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# 添加model目录到Python路径
model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(model_dir)

# 添加当前TG-ECNet目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 添加stage2和stage2/net目录到Python路径
stage1_dir = os.path.join(current_dir, "stage1")
stage1_net_dir = os.path.join(stage1_dir, "net")
stage2_dir = os.path.join(current_dir, "stage2")
stage2_net_dir = os.path.join(stage2_dir, "net")
sys.path.append(stage1_dir)
sys.path.append(stage1_net_dir)
sys.path.append(stage2_dir)
sys.path.append(stage2_net_dir)
from test_metrics.tensor_example import TensorFusionEvaluator
import torch
# 导入stage1和stage2的TG_ECNet
from stage1.net.TG_ECNet import TG_ECNet as TG_ECNet_Stage1
from stage2.net.TG_ECNet import TG_ECNet as TG_ECNet_Stage2
import torch
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
import torch.nn as nn
from data_process.dataset import ImageDataset
from PIL import Image
import math
import torchvision.transforms as transforms

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
def test_model(model_noise,model_iv,dataloader,device):
    Evaluator=TensorFusionEvaluator()
    all_metrics = {}
    with torch.no_grad():
        for i_batch, (i_data, v_data,d_data) in enumerate(tqdm(dataloader, desc="Testing")):
            # 将三通道RGB转为单通道灰度图
            if v_data.shape[1] == 3:
                v_data = torch.mean(v_data, dim=1, keepdim=True)
            if i_data.shape[1] == 3:
                i_data = torch.mean(i_data, dim=1, keepdim=True)
            if d_data.shape[1] == 3:
                d_data = torch.mean(d_data, dim=1, keepdim=True)
            i_data = i_data.to(device)
            v_data = v_data.to(device)
            d_data = d_data.to(device)
            cv,_=model_noise(d_data)
            out,_=model_iv(cv,i_data)
            batch_metrics=Evaluator.evaluate_fusion_batch(out,v_data,i_data)
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
def load_datasets(i_dir,v_dir,d_dir,transform_i,transform_v,transform_d,batch_size):
    i_dataset = ImageDataset(i_dir, transform=transform_i)
    v_dataset = ImageDataset(v_dir, transform=transform_v)
    d_dataset = ImageDataset(d_dir, transform=transform_d)
    check_data(i_dataset,v_dataset,d_dataset)
    paired_dataset = PairedDataset(i_dataset, v_dataset, d_dataset)
    print(f"联合数据集创建完成，包含 {len(paired_dataset)} 对图像")

    # 创建数据加载器 - 优化内存使用
    data_loader = DataLoader(paired_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=2, pin_memory=False)  # 减少workers和禁用pin_memory
    print(f"数据加载器创建完成，批次大小:{batch_size}")
    return data_loader
def main():
    transform_i = transforms.Compose([
        transforms.Resize((240,240)),  # 调整图像大小
        transforms.ToTensor(),          # 转换为张量 [0,1]
        #transforms.Normalize(mean=[0.253], std=[0.191]) #LLVIP
    ])
    
    # 为可见光图像（3通道）创建变换
    # 为可见光图像（3通道）创建变换 - 提供多种标准化选择
    # 选项1: 通用标准化 [-1, 1]
    transform_v = transforms.Compose([
        transforms.Resize((240,240)),  # 调整图像大小
        transforms.ToTensor(),          # 转换为张量 [0,1]
        #transforms.Normalize(mean=[0.188, 0.186, 0.154], std=[0.183, 0.190, 0.197]) #LLVIP
    ])
    transform_d = transforms.Compose([
        transforms.Resize((240,240)),  # 调整图像大小
        transforms.ToTensor(),          # 转换为张量 [0,1]
        #transforms.Normalize(mean=[0.188, 0.186, 0.154], std=[0.183, 0.190, 0.197]) #LLVIP
    ])
    data_loader=load_datasets('/home/user/whs/1024whs_data/DeMMI-RF/Test/Multi-Task/alltask/LLVIP/infrared',
                              '/home/user/whs/1024whs_data/DeMMI-RF/Test/Multi-Task/alltask/LLVIP/visible',
                              '/home/user/whs/1024whs_data/DeMMI-RF/Test/Multi-Task/alltask/LLVIP/input',
                              transform_i,transform_v,transform_d,batch_size=6)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model_noise=TG_ECNet_Stage1()
    modelpre_noise = torch.load('/home/user/whs/IVfusion_method1/experiments/TG-ECNet/ckpt/epoch=4-step=16365.ckpt', map_location=torch.device('cuda'), weights_only=True)
    conv_dict_noise = {k: v for k, v in modelpre_noise.items()}
    model_noise.load_state_dict(conv_dict_noise, strict=False)
    model_noise=model_noise.to(device)
    model_iv=TG_ECNet_Stage2()
    modelpre_iv = torch.load('/home/user/whs/IVfusion_method1/experiments/TG-ECNet/ckpt_ori/epoch=4-step=33290.ckpt', map_location=torch.device('cuda'), weights_only=True)
    conv_dict_iv = {k: v for k, v in  modelpre_iv.items()}
    model_iv.load_state_dict(conv_dict_iv, strict=False)
    model_iv=model_iv.to(device)
    for test_time in range(5):
        with torch.no_grad():
            averaged_metrics=test_model(model_noise,model_iv,data_loader,device)
        print("测试集上的平均指标:")
        for metric_name, value in averaged_metrics.items(): 
            if not metric_name.endswith('_std'):
                std_name = f"{metric_name}_std"
                std_value = averaged_metrics.get(std_name, 0)
                print(f"测试次数{test_time+1} {metric_name:15s}: {value:8.6f} ± {std_value:.6f}")

if __name__ == "__main__":
    main()
    '''
    
    dv=torch.randn(2,1,256,256).to('cuda:0')
    i=torch.randn(2,1,256,256).to('cuda:0')
    with torch.no_grad():
        model_noise=TG_ECNet_Stage1()
        modelpre_noise = torch.load('/home/user/whs/IVfusion_method1/experiments/TG-ECNet/ckpt/epoch=4-step=16365.ckpt', map_location=torch.device('cuda'), weights_only=True)
        conv_dict_noise = {k: v for k, v in modelpre_noise.items()}
        model_noise.load_state_dict(conv_dict_noise, strict=False)
        model_noise=model_noise.to('cuda:0')
        model_iv=TG_ECNet_Stage2()
        modelpre_iv = torch.load('/home/user/whs/IVfusion_method1/experiments/TG-ECNet/ckpt_ori/epoch=4-step=33290.ckpt', map_location=torch.device('cuda'), weights_only=True)
        conv_dict_iv = {k: v for k, v in  modelpre_iv.items()}
        model_iv.load_state_dict(conv_dict_iv, strict=False)
        model_iv=model_noise.to('cuda:0')
        cv,_=model_noise(dv)
        out,_=model_iv(cv,i)
        print(out.shape)
    Evaluator=TensorFusionEvaluator()

    batch_metrics=Evaluator.evaluate_fusion_batch(out,cv,i)
    for metric_name, value in batch_metrics.items():
        if not metric_name.endswith('_std'):
            std_name = f"{metric_name}_std"
            std_value = batch_metrics.get(std_name, 0)
            print(f"{metric_name:15s}: {value:8.6f} ± {std_value:.6f}")'''