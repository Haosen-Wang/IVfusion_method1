import os
from torch.utils.data import Dataset,DataLoader
import torch
from PIL import Image
import torchvision.transforms as transforms
class ImageDataset(Dataset):
    def __init__(self, image_dir,transform=None,convert="RGB"):
        self.image_dir = image_dir
        self.image_paths = self._get_image_paths()
        self.transform = transform
        self.convert=convert        
    def _get_image_paths(self):
        """获取文件夹内所有图像的路径"""
        image_paths = []
        # 支持的图像格式
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif'}
        
        # 遍历目录及子目录
        for root, dirs, files in os.walk(self.image_dir):
            for file in files:
                # 检查文件扩展名
                if os.path.splitext(file)[1].lower() in image_extensions:
                    image_paths.append(os.path.join(root, file))
        
        return sorted(image_paths)  # 排序以确保一致性
    
    def _load_data(self, data_path):
        """加载图像数据"""
        try:
            image = Image.open(data_path).convert(self.convert)
            return image
        except Exception as e:
            print(f"Error loading image {data_path}: {e}")
            return None

    def __len__(self):
        return len(self.image_paths)



    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = self._load_data(image_path)
        
        if image is None:
            # 如果图像加载失败，返回空的tensor
            return torch.zeros((3, 224, 224)), image_path
        
        if self.transform:
            image = self.transform(image)
            
        return image, image_path
    
    def get_all_paths(self):
        """返回所有图像路径的列表"""
        return self.image_paths
    
if __name__=="__main__":
    v_data_dir="/data/1024whs_data/DeMMI-RF/Train/visible/noise15/M3FD"
    transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor(),          # 转换为张量 [0,1]
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 可选的标准化
])

    v_data=ImageDataset(v_data_dir,transform=transform,convert="RGB")
    v_dataloader=DataLoader(dataset=v_data,shuffle=True,batch_size=2)
    for i in v_dataloader:
        print(i[0])
        print(i[0].shape)
        break
    i_data_dir="/data/1024whs_data/DeMMI-RF/Train/infrared/noise15/M3FD"
    i_data=ImageDataset(i_data_dir,transform=transform,convert="L")
    i_dataloader=DataLoader(dataset=i_data,shuffle=True,batch_size=2)
    for i in i_dataloader:
        print(i[0])
        print(i[0].shape)
        break