from restormer.restormer_arch import Restormer
from PIL import Image
import requests
import torch
import torchvision.transforms as transforms

# 创建模型
model = Restormer()
model.eval()  # 设置为评估模式

# 计算模型参数量
def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

total_params, trainable_params = count_parameters(model)
print(f"模型总参数量: {total_params:,}")
print(f"可训练参数量: {trainable_params:,}")
print(f"模型大小估计: {total_params * 4 / 1024 / 1024:.2f} MB (假设float32)")
print("-" * 50)

# 下载图像
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
print(f"原始图像尺寸: {image.size}")

# 定义图像预处理
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小
    transforms.ToTensor(),          # 转换为张量 [0,1]
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 可选的标准化
])

# 预处理图像
input_tensor = transform(image).unsqueeze(0)  # 添加batch维度，形状变为[1, 3, H, W]
print(f"输入张量形状: {input_tensor.shape}")

# 将图像输入模型
with torch.no_grad():  # 推理时不需要计算梯度
    output = model(input_tensor)
    
print(f"输出张量形状: {output.shape}")
print("模型推理完成！")