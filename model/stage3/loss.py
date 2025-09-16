import torch
from torchvision.models import vgg16
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

class FusionLoss_F(nn.Module):
    def __init__(self, pixel_weight=1.0, gradient_weight=1.0, ssim_weight=1.0, 
                 l1_reg_weight=0.0, l2_reg_weight=0.0, learnable_reg=True):
        super(FusionLoss_F, self).__init__()
        self.pixel_weight = pixel_weight
        self.gradient_weight = gradient_weight
        self.ssim_weight = ssim_weight
        
        # 可学习的正则化权重
        if learnable_reg:
            # 使用Parameter将正则化权重变为可训练参数
            # 初始化为给定值，并使用log空间确保正值
            self.log_l1_reg_weight = nn.Parameter(torch.log(torch.tensor(l1_reg_weight + 1e-8)))
            self.log_l2_reg_weight = nn.Parameter(torch.log(torch.tensor(l2_reg_weight + 1e-8)))
            self.learnable_reg = True
        else:
            # 固定的正则化权重
            self.register_buffer('l1_reg_weight', torch.tensor(l1_reg_weight))
            self.register_buffer('l2_reg_weight', torch.tensor(l2_reg_weight))
            self.learnable_reg = False
        
        # VGG16 for perceptual loss
        #vgg = vgg16(pretrained=True)
        #self.vgg_features = nn.Sequential(*list(vgg.features)[:16]).eval()
        #for param in self.vgg_features.parameters():
            #param.requires_grad = False
    
    def get_reg_weights(self):
        """获取当前的正则化权重"""
        if self.learnable_reg:
            # 使用exp确保权重为正值
            l1_weight = torch.exp(self.log_l1_reg_weight)
            l2_weight = torch.exp(self.log_l2_reg_weight)
        else:
            l1_weight = self.l1_reg_weight
            l2_weight = self.l2_reg_weight
        return l1_weight, l2_weight
    
    def pixel_loss(self, fused, visible, infrared):
        """Pixel-level L1 loss with learnable regularization"""
        infrared=infrared.expand_as(fused)
        visible=visible.expand_as(fused)
        
        # 基础L1损失
        base_loss = 0.5*(F.l1_loss(fused, visible) + F.l1_loss(fused, infrared))
        
        # 获取当前正则化权重
        l1_reg_weight, l2_reg_weight = self.get_reg_weights()
        
        # 添加正则化项
        regularization = 0.0
        
        # L1正则化 (促进稀疏性)
        if l1_reg_weight > 1e-8:
            l1_reg = l1_reg_weight * torch.sum(torch.abs(fused))
            regularization += l1_reg
        
        # L2正则化 (防止过拟合)
        if l2_reg_weight > 1e-8:
            l2_reg = l2_reg_weight * torch.sum(fused ** 2)
            regularization += l2_reg
        
        return base_loss + regularization 
    
    def gradient_loss(self, fused, visible, infrared):
        """Gradient-level loss using Sobel operator with regularization"""
        def sobel_gradient(img):
            # Get number of channels
            channels = img.size(1)
            
            # Create Sobel kernels for all channels
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            
            # Repeat for all channels
            sobel_x = sobel_x.repeat(channels, 1, 1, 1)
            sobel_y = sobel_y.repeat(channels, 1, 1, 1)
            
            # Move to the same device as input image
            sobel_x = sobel_x.to(img.device)
            sobel_y = sobel_y.to(img.device)
            
            grad_x = F.conv2d(img, sobel_x, padding=1, groups=channels)
            grad_y = F.conv2d(img, sobel_y, padding=1, groups=channels)
            # 确保平方和为非负，避免 sqrt(负数)
            grad_magnitude_sq = grad_x**2 + grad_y**2
            grad_magnitude_sq = torch.clamp(grad_magnitude_sq, min=0.0)
            return torch.sqrt(grad_magnitude_sq)
        
        grad_fused = sobel_gradient(fused)
        grad_visible = sobel_gradient(visible)
        grad_infrared = sobel_gradient(infrared)
        grad_infrared = grad_infrared.expand_as(grad_fused)
        grad_visible = grad_visible.expand_as(grad_fused)
        
        # 基础梯度损失
        base_loss = 0.5*(F.l1_loss(grad_fused, grad_visible) + F.l1_loss(grad_fused, grad_infrared))
        
        # 获取当前正则化权重
        l1_reg_weight, l2_reg_weight = self.get_reg_weights()
        
        # 添加梯度正则化项
        regularization = 0.0
        
        # 梯度L1正则化 (促进梯度稀疏性，减少噪声)
        if l1_reg_weight > 1e-8:
            grad_l1_reg = l1_reg_weight * 0.1 * torch.sum(torch.abs(grad_fused))
            regularization += grad_l1_reg
        
        # 梯度L2正则化 (平滑梯度)
        if l2_reg_weight > 1e-8:
            grad_l2_reg = l2_reg_weight * 0.1 * torch.sum(grad_fused ** 2)
            regularization += grad_l2_reg
        
        return base_loss + regularization
    
    def ssim_loss(self, fused, visible, infrared):
        """Structural Similarity Index loss with numerical stability"""
        def ssim(img1, img2, window_size=11, window_sigma=1.5):
            # 增加常数以提高数值稳定性
            C1 = (0.01 * 255)**2  # 增大常数
            C2 = (0.03 * 255)**2  
            
            # 确保输入在合理范围内
            img1 = torch.clamp(img1, 0, 1)
            img2 = torch.clamp(img2, 0, 1)
            
            mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
            mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
            
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
            sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
            sigma12 = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu1_mu2
            
            # 确保分母不为零
            denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
            denominator = torch.clamp(denominator, min=1e-8)
            
            numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
            ssim_map = numerator / denominator
            
            # 确保SSIM值在合理范围内
            ssim_map = torch.clamp(ssim_map, -1, 1)
            result = ssim_map.mean()
            
            # 检查NaN
            if torch.isnan(result) or torch.isinf(result):
                print(f"⚠️ SSIM计算出现NaN/Inf，返回安全值")
                return torch.tensor(0.0, device=img1.device, requires_grad=True)
            
            return result
        
        # 确保输入维度一致
        if infrared.size(1) != fused.size(1):
            infrared = infrared.expand_as(fused)
        if visible.size(1) != fused.size(1):
            visible = visible.expand_as(fused)
            
        ssim_vis = ssim(fused, visible)
        ssim_ir = ssim(fused, infrared)
        
        # 安全地计算损失
        loss = 1 - torch.clamp((ssim_vis + ssim_ir) / 2, 0, 1)
        return loss 
    
    def perceptual_loss(self, fused, visible, infrared):
        """Perceptual loss using VGG16 features with stability checks"""
        try:
            # 确保输入在正确范围内[0,1]
            fused = torch.clamp(fused, 0, 1)
            visible = torch.clamp(visible, 0, 1)
            infrared = torch.clamp(infrared, 0, 1)
            
            # Convert to 3-channel if grayscale
            if fused.size(1) == 1:
                fused = fused.repeat(1, 3, 1, 1)
            if visible.size(1) == 1:
                visible = visible.repeat(1, 3, 1, 1)
            if infrared.size(1) == 1:
                infrared = infrared.repeat(1, 3, 1, 1)
            
            # 确保所有图像尺寸一致
            min_size = min(fused.size(2), fused.size(3))
            if min_size < 32:  # VGG需要足够大的输入
                print(f"⚠️ 图像尺寸过小 ({min_size}x{min_size})，跳过感知损失")
                return torch.tensor(0.0, device=fused.device, requires_grad=True)
            
            # 检查输入是否包含NaN/Inf
            if torch.isnan(fused).any() or torch.isnan(visible).any() or torch.isnan(infrared).any():
                print(f"⚠️ 感知损失输入包含NaN，返回安全值")
                return torch.tensor(0.0, device=fused.device, requires_grad=True)
                
            if torch.isinf(fused).any() or torch.isinf(visible).any() or torch.isinf(infrared).any():
                print(f"⚠️ 感知损失输入包含Inf，返回安全值")
                return torch.tensor(0.0, device=fused.device, requires_grad=True)
            
            # 将VGG移动到正确设备
            if next(self.vgg_features.parameters()).device != fused.device:
                self.vgg_features = self.vgg_features.to(fused.device)
            
            fused_features = self.vgg_features(fused)
            visible_features = self.vgg_features(visible)
            infrared_features = self.vgg_features(infrared)
            
            # 检查特征是否包含NaN/Inf
            if torch.isnan(fused_features).any() or torch.isnan(visible_features).any() or torch.isnan(infrared_features).any():
                print(f"⚠️ VGG特征包含NaN，返回安全值")
                return torch.tensor(0.0, device=fused.device, requires_grad=True)
            
            loss = F.l1_loss(fused_features, visible_features) + F.l1_loss(fused_features, infrared_features)
            
            # 最终检查损失值
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ 感知损失计算出现NaN/Inf，返回安全值")
                return torch.tensor(0.0, device=fused.device, requires_grad=True)
                
            return loss
            
        except Exception as e:
            print(f"⚠️ 感知损失计算出错: {e}，返回安全值")
            return torch.tensor(0.0, device=fused.device, requires_grad=True) 
    def forward(self, fused, visible, infrared):
        """前向传播，包含完整的数值稳定性检查"""
        try:
            # 输入检查
            inputs = [fused, visible, infrared]
            input_names = ['fused', 'visible', 'infrared']
            
            for i, (inp, name) in enumerate(zip(inputs, input_names)):
                if torch.isnan(inp).any():
                    print(f"⚠️ {name} 图像包含NaN值")
                    return self._safe_loss_dict(fused.device)
                if torch.isinf(inp).any():
                    print(f"⚠️ {name} 图像包含Inf值")
                    return self._safe_loss_dict(fused.device)
                if inp.max() > 10 or inp.min() < -10:
                    print(f"⚠️ {name} 图像值范围异常: [{inp.min():.3f}, {inp.max():.3f}]")
                    # 限制到合理范围
            
            fused, visible, infrared = inputs
            
            # 计算各项损失
            pixel_loss = self.pixel_loss(fused, visible, infrared)
            gradient_loss = self.gradient_loss(fused, visible, infrared)
            ssim_loss = self.ssim_loss(fused, visible, infrared)
            
            # 检查每项损失
            losses = {
                'pixel_loss': pixel_loss,
                'gradient_loss': gradient_loss, 
                'ssim_loss': ssim_loss,
            }
            
            for loss_name, loss_val in losses.items():
                if torch.isnan(loss_val) or torch.isinf(loss_val):
                    print(f"⚠️ {loss_name} 出现NaN/Inf: {loss_val}")
                    return self._safe_loss_dict(fused.device)
            
            # 计算总损失，使用较小的权重避免溢出
            total_loss = (self.pixel_weight * pixel_loss + 
                         self.gradient_weight * gradient_loss+
                         self.ssim_weight * ssim_loss)
            #total_loss = (self.pixel_weight * pixel_loss + 
                         #self.gradient_weight * gradient_loss)
            
            # 最终检查
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"⚠️ 总损失出现NaN/Inf: {total_loss}")
                return self._safe_loss_dict(fused.device)

            return {
                'total_loss_f': total_loss,
                'pixel_loss_f': pixel_loss,
                'gradient_loss_f': gradient_loss,
                #'ssim_loss_f': ssim_loss
            }
            
        except Exception as e:
            print(f"⚠️ FusionLoss计算出错: {e}")
            return self._safe_loss_dict(fused.device)
    
    def _safe_loss_dict(self, device):
        """返回安全的损失字典"""
        safe_loss = torch.tensor(1.0, device=device, requires_grad=True)
        return {
            'total_loss_f': safe_loss,
            'pixel_loss': safe_loss * 0.1,
            'gradient_loss': safe_loss * 0.1, 
            'ssim_loss_f': safe_loss * 0.1
        }
class FusionLoss_N(nn.Module):
    def __init__(self, pixel_weight=1.0, gradient_weight=1.0, ssim_weight=1.0,
                 l1_reg_weight=0.0, l2_reg_weight=0.0, learnable_reg=True):
        super(FusionLoss_N, self).__init__()
        self.pixel_weight = pixel_weight
        self.gradient_weight = gradient_weight
        self.ssim_weight = ssim_weight
        
        # 可学习的正则化权重
        if learnable_reg:
            # 使用Parameter将正则化权重变为可训练参数
            self.log_l1_reg_weight = nn.Parameter(torch.log(torch.tensor(l1_reg_weight + 1e-8)))
            self.log_l2_reg_weight = nn.Parameter(torch.log(torch.tensor(l2_reg_weight + 1e-8)))
            self.learnable_reg = True
        else:
            # 固定的正则化权重
            self.register_buffer('l1_reg_weight', torch.tensor(l1_reg_weight))
            self.register_buffer('l2_reg_weight', torch.tensor(l2_reg_weight))
            self.learnable_reg = False
    
    def get_reg_weights(self):
        """获取当前的正则化权重"""
        if self.learnable_reg:
            # 使用exp确保权重为正值
            l1_weight = torch.exp(self.log_l1_reg_weight)
            l2_weight = torch.exp(self.log_l2_reg_weight)
        else:
            l1_weight = self.l1_reg_weight
            l2_weight = self.l2_reg_weight
        return l1_weight, l2_weight
        
        # VGG16 for perceptual loss
        #try:
           # from torchvision.models import VGG16_Weights
            #vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        #except ImportError:
            # 兼容旧版本
            #vgg = vgg16(pretrained=True)
        #self.vgg_features = nn.Sequential(*list(vgg.features)[:16]).eval()
        #for param in self.vgg_features.parameters():
            #param.requires_grad = False
    
    def pixel_loss(self, clean_pre, clean):
        """Pixel-level L1 loss with learnable regularization"""
        # 基础L1损失
        base_loss = F.l1_loss(clean_pre, clean)
        
        # 获取当前正则化权重
        l1_reg_weight, l2_reg_weight = self.get_reg_weights()
        
        # 添加正则化项
        regularization = 0.0
        
        # L1正则化 (促进稀疏性)
        if l1_reg_weight > 1e-8:
            l1_reg = l1_reg_weight * torch.sum(torch.abs(clean_pre))
            regularization += l1_reg
        
        # L2正则化 (防止过拟合)
        if l2_reg_weight > 1e-8:
            l2_reg = l2_reg_weight * torch.sum(clean_pre ** 2)
            regularization += l2_reg
        
        return base_loss + regularization 
    
    def gradient_loss(self, clean_pre, clean):
        """Gradient-level loss using Sobel operator with regularization"""
        def sobel_gradient(img):
            # Get number of channels
            channels = img.size(1)
            
            # Create Sobel kernels for all channels
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
            
            # Repeat for all channels
            sobel_x = sobel_x.repeat(channels, 1, 1, 1)
            sobel_y = sobel_y.repeat(channels, 1, 1, 1)
            
            # Move to the same device as input image
            sobel_x = sobel_x.to(img.device)
            sobel_y = sobel_y.to(img.device)
            
            grad_x = F.conv2d(img, sobel_x, padding=1, groups=channels)
            grad_y = F.conv2d(img, sobel_y, padding=1, groups=channels)
            # 确保平方和为非负，避免 sqrt(负数)
            grad_magnitude_sq = grad_x**2 + grad_y**2
            grad_magnitude_sq = torch.clamp(grad_magnitude_sq, min=0.0)
            return torch.sqrt(grad_magnitude_sq)
        
        grad_clean_pre = sobel_gradient(clean_pre)
        grad_clean = sobel_gradient(clean)
        
        # 基础梯度损失
        base_loss = F.l1_loss(grad_clean_pre, grad_clean)
        
        # 获取当前正则化权重
        l1_reg_weight, l2_reg_weight = self.get_reg_weights()
        
        # 添加梯度正则化项
        regularization = 0.0
        
        # 梯度L1正则化 (促进梯度稀疏性)
        if l1_reg_weight > 1e-8:
            grad_l1_reg = l1_reg_weight * 0.1 * torch.sum(torch.abs(grad_clean_pre))
            regularization += grad_l1_reg
        
        # 梯度L2正则化 (平滑梯度)
        if l2_reg_weight > 1e-8:
            grad_l2_reg = l2_reg_weight * 0.1 * torch.sum(grad_clean_pre ** 2)
            regularization += grad_l2_reg
        
        return base_loss + regularization

    def ssim_loss(self, clean_pre, clean):
        """Structural Similarity Index loss"""
        def ssim(img1, img2, window_size=11, window_sigma=1.5):
            C1 = 0.01**2
            C2 = 0.03**2
            
            mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
            mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
            
            mu1_sq = mu1.pow(2)
            mu2_sq = mu2.pow(2)
            mu1_mu2 = mu1 * mu2
            
            sigma1_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu1_sq
            sigma2_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu2_sq
            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma1_sq + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
            result = ssim_map.mean()
            return result if not torch.isnan(result) else torch.zeros_like(result, requires_grad=True)
        ssim_dc = ssim(clean_pre, clean)
        loss = 1 - ssim_dc
        return loss

    def perceptual_loss(self, clean_pre, clean):
        """Perceptual loss using VGG16 features"""
        # Convert to 3-channel if grayscale
        if clean_pre.size(1) == 1:
            clean_pre = clean_pre.repeat(1, 3, 1, 1)
        if clean.size(1) == 1:
            clean = clean.repeat(1, 3, 1, 1)

        degrad_features = self.vgg_features(clean_pre)
        clean_features = self.vgg_features(clean)

        loss = F.l1_loss(degrad_features, clean_features)
        return loss
    def forward(self, clean_pre, clean):
        pixel_loss = self.pixel_loss(clean_pre, clean)
        gradient_loss = self.gradient_loss(clean_pre, clean)
        #ssim_loss = self.ssim_loss(clean_pre, clean)

        total_loss = (self.pixel_weight * pixel_loss +
                     self.gradient_weight * gradient_loss)
                     #self.ssim_weight * ssim_loss)

        return {
            'total_loss_d': total_loss,
            'pixel_loss_d': pixel_loss,
            'gradient_loss_d': gradient_loss,
            #'ssim_loss_d': ssim_loss
        }
class Loss(nn.Module):
     def __init__(self, f_weight=1.0, vi_weight=1.0, pixel_weight=1.0, gradient_weight=1.0, 
                  ssim_weight=1.0, rec_loss_weight=1.0, KL_loss_weight=0.01,
                  l1_reg_weight=0.0, l2_reg_weight=0.0, learnable_reg=False):
          super().__init__()
          self.f_loss = FusionLoss_F(pixel_weight, gradient_weight, ssim_weight, 
                                   l1_reg_weight, l2_reg_weight, learnable_reg)
          self.d_loss = FusionLoss_N(pixel_weight, gradient_weight, ssim_weight,
                                   l1_reg_weight, l2_reg_weight, learnable_reg)
          self.f_weight = f_weight
          self.d_weight = vi_weight
          self.learnable_reg = learnable_reg
     
     def get_regularization_weights(self):
         """获取当前的正则化权重（用于监控）"""
         f_l1, f_l2 = self.f_loss.get_reg_weights()
         d_l1, d_l2 = self.d_loss.get_reg_weights()
         return {
             'fusion_l1_reg': f_l1.item() if hasattr(f_l1, 'item') else float(f_l1),
             'fusion_l2_reg': f_l2.item() if hasattr(f_l2, 'item') else float(f_l2),
             'degrad_l1_reg': d_l1.item() if hasattr(d_l1, 'item') else float(d_l1),
             'degrad_l2_reg': d_l2.item() if hasattr(d_l2, 'item') else float(d_l2)
         }
     def forward(self,task,f,v,i,c):
         f_losses = self.f_loss(f, v, i)
         if task=="dv_i":
             d_losses = self.d_loss(c, v)
         if task=="di_v":
             d_losses = self.d_loss(c, i)
         total_loss=5*self.f_weight*f_losses['total_loss_f']+self.d_weight*d_losses['total_loss_d']
         
         # 如果使用可学习正则化，添加正则化权重信息到输出
         result = {**f_losses, **d_losses, 'total_loss': total_loss}
         if self.learnable_reg:
             reg_weights = self.get_regularization_weights()
             result.update(reg_weights)
         
         return result
if __name__=="__main__":
    # 测试代码
   fused=torch.rand(size=(2, 3, 224, 224))
   visible=torch.rand(size=(2, 3, 224, 224))
   infrared=torch.rand(size=(2, 1, 224, 224))
   loss_fn=Loss(rec_loss_weight=1.0,KL_loss_weight=0.01)
   losses=loss_fn(fused,visible, infrared)
   print(losses)

