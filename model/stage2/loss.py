import torch
from torchvision.models import vgg16
import numpy as np

import torch.nn as nn
import torch.nn.functional as F

class FusionLoss(nn.Module):
    def __init__(self, pixel_weight=1.0, gradient_weight=1.0, ssim_weight=1.0, perceptual_weight=1.0):
        super(FusionLoss, self).__init__()
        self.pixel_weight = pixel_weight
        self.gradient_weight = gradient_weight
        self.ssim_weight = ssim_weight
        self.perceptual_weight = perceptual_weight
        
        # VGG16 for perceptual loss
        vgg = vgg16(pretrained=True)
        self.vgg_features = nn.Sequential(*list(vgg.features)[:16]).eval()
        for param in self.vgg_features.parameters():
            param.requires_grad = False
    
    def pixel_loss(self, fused, visible, infrared):
        """Pixel-level L1 loss"""
        infrared=infrared.expand_as(fused)
        visible=visible.expand_as(fused)
        loss = 0.5*(F.l1_loss(fused, visible) + F.l1_loss(fused, infrared))
        return loss 
    
    def gradient_loss(self, fused, visible, infrared):
        """Gradient-level loss using Sobel operator"""
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
        loss = 0.5*(F.l1_loss(grad_fused, grad_visible) + F.l1_loss(grad_fused, grad_infrared))
        return loss
    
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
                    # 限制到合理范围
            
            fused, visible, infrared = inputs
            
            # 计算各项损失
            pixel_loss = self.pixel_loss(fused, visible, infrared)
            gradient_loss = self.gradient_loss(fused, visible, infrared)
            ssim_loss = self.ssim_loss(fused, visible, infrared)
            perceptual_loss = self.perceptual_loss(fused, visible, infrared)
            
            # 检查每项损失
            losses = {
                'pixel_loss': pixel_loss,
                'gradient_loss': gradient_loss, 
                'ssim_loss': ssim_loss,
                'perceptual_loss': perceptual_loss
            }
            
            for loss_name, loss_val in losses.items():
                if torch.isnan(loss_val) or torch.isinf(loss_val):
                    print(f"⚠️ {loss_name} 出现NaN/Inf: {loss_val}")
                    return self._safe_loss_dict(fused.device)
            
            # 计算总损失，使用较小的权重避免溢出
            total_loss = (self.pixel_weight * pixel_loss + 
                         self.gradient_weight * gradient_loss +
                         self.ssim_weight * ssim_loss +
                         self.perceptual_weight * perceptual_loss)
            #total_loss = (self.pixel_weight * pixel_loss + 
                         #self.gradient_weight * gradient_loss)
            
            # 最终检查
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"⚠️ 总损失出现NaN/Inf: {total_loss}")
                return self._safe_loss_dict(fused.device)

            return {
                'total_loss_f': total_loss,
                'pixel_loss': pixel_loss,
                'gradient_loss': gradient_loss,
                'perceptual_loss': perceptual_loss
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
            'perceptual_loss': safe_loss * 0.1
        }
    
class VI_Loss(nn.Module):
    def __init__(self, rec_loss_weight=1.0,KL_loss_weight=0.01):
        super(VI_Loss, self).__init__()
        self.rec_loss_weight = rec_loss_weight
        self.KL_loss_weight = KL_loss_weight
    def forward(self, i, v, g, l, mu_i, sigma2_i, mu_v, sigma2_v):
        """VI损失前向传播，包含完整的数值稳定性检查"""
        try:
            # 输入检查
            inputs = [i, v, g, l, mu_i, sigma2_i, mu_v, sigma2_v]
            input_names = ['i', 'v', 'g', 'l', 'mu_i', 'sigma2_i', 'mu_v', 'sigma2_v']
            
            for inp, name in zip(inputs, input_names):
                if torch.isnan(inp).any():
                    print(f"⚠️ VI_Loss输入 {name} 包含NaN值")
                    return self._safe_vi_loss_dict(i.device)
                if torch.isinf(inp).any():
                    print(f"⚠️ VI_Loss输入 {name} 包含Inf值")
                    return self._safe_vi_loss_dict(i.device)
    
            
            # 计算目标值
            l_tru = torch.max(i, v)
            l = l.expand_as(l_tru)
            
            # 重建损失 - 添加数值稳定性检查
            rec_l = F.mse_loss(l, l_tru)
            if torch.isnan(rec_l) or torch.isinf(rec_l):
                print(f"⚠️ rec_l 出现NaN/Inf")
                rec_l = torch.tensor(1.0, device=i.device, requires_grad=True)
            
            # 梯度损失计算
            try:
                grad_visible = self.sobel_gradient(v)
                grad_infrared = self.sobel_gradient(i)
                g_tru = torch.max(grad_visible, grad_infrared)
                g = g.expand_as(g_tru)
                rec_g = F.mse_loss(g, g_tru)
                
                if torch.isnan(rec_g) or torch.isinf(rec_g):
                    print(f"⚠️ rec_g 出现NaN/Inf")
                    rec_g = torch.tensor(1.0, device=i.device, requires_grad=True)
                    
            except Exception as e:
                print(f"⚠️ 梯度损失计算出错: {e}")
                rec_g = torch.tensor(1.0, device=i.device, requires_grad=True)
            
            # KL散度损失 - 大幅提高数值稳定性
            sigma2_i_safe = torch.clamp(sigma2_i, min=1e-6, max=100)  # 限制上界
            sigma2_v_safe = torch.clamp(sigma2_v, min=1e-6, max=100)
            
            # 限制mu的范围，避免平方后溢出
            mu_i_safe = torch.clamp(mu_i, -5, 5)
            mu_v_safe = torch.clamp(mu_v, -5, 5)
            
            # 安全的KL计算
            try:
                log_sigma2_i = torch.log(sigma2_i_safe)
                log_sigma2_v = torch.log(sigma2_v_safe)
                
                # 检查对数是否产生NaN
                if torch.isnan(log_sigma2_i).any() or torch.isinf(log_sigma2_i).any():
                    print(f"⚠️ log(sigma2_i) 出现问题")
                    kl_loss_i = torch.tensor(0.1, device=i.device, requires_grad=True)
                else:
                    kl_loss_i = -0.5 * torch.mean(1 + log_sigma2_i - mu_i_safe.pow(2) - sigma2_i_safe)
                    kl_loss_i = torch.clamp(kl_loss_i, -10, 10)  # 限制KL损失范围
                
                if torch.isnan(log_sigma2_v).any() or torch.isinf(log_sigma2_v).any():
                    print(f"⚠️ log(sigma2_v) 出现问题")
                    kl_loss_g = torch.tensor(0.1, device=i.device, requires_grad=True)
                else:
                    kl_loss_g = -0.5 * torch.mean(1 + log_sigma2_v - mu_v_safe.pow(2) - sigma2_v_safe)
                    kl_loss_g = torch.clamp(kl_loss_g, -10, 10)  # 限制KL损失范围
                    
            except Exception as e:
                print(f"⚠️ KL损失计算出错: {e}")
                kl_loss_i = torch.tensor(0.1, device=i.device, requires_grad=True)
                kl_loss_g = torch.tensor(0.1, device=i.device, requires_grad=True)

            #total_loss = 0.5 * (self.rec_loss_weight * rec_l + self.KL_loss_weight * kl_loss_i + self.rec_loss_weight * rec_g+self.KL_loss_weight * kl_loss_g)
            total_loss = 0.5 * (self.rec_loss_weight * rec_l + self.rec_loss_weight * rec_g)
            # 最终检查
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"⚠️ VI总损失出现NaN/Inf: {total_loss}")
                return self._safe_vi_loss_dict(i.device)
            
            return {
                'total_loss_vi': total_loss,
                'rec_i': rec_l,
                'rec_g': rec_g,
                'kl_loss_i': kl_loss_i,
                'kl_loss_g': kl_loss_g
            }
            
        except Exception as e:
            print(f"⚠️ VI_Loss计算出错: {e}")
            return self._safe_vi_loss_dict(i.device)
    
    def _safe_vi_loss_dict(self, device):
        """返回安全的VI损失字典"""
        safe_loss = torch.tensor(1.0, device=device, requires_grad=True)
        return {
            'total_loss_vi': safe_loss,
            'rec_i': safe_loss * 0.5,
            'rec_g': safe_loss * 0.5,
            'kl_loss_i': safe_loss * 0.1,
            'kl_loss_g': safe_loss * 0.1
        }
    
    def sobel_gradient(self,img):
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
    def calculate_cross_entropy(self,img1, img2):#self,true,pred
                # Convert to numpy and flatten
                img1_np = img1.detach().cpu().numpy()
                img2_np = img2.detach().cpu().numpy()
                img1_flat = img1_np.flatten()
                img2_flat = img2_np.flatten()
                
                # Calculate histograms
                hist1, _ = np.histogram(img1_flat, bins=256, range=(0, 1))
                hist2, _ = np.histogram(img2_flat, bins=256, range=(0, 1))
                hist1 = hist1 + 1e-7  # Add small value to avoid log(0)
                hist2 = hist2 + 1e-7
                
                # Normalize to get probabilities
                prob1 = hist1 / hist1.sum()
                prob2 = hist2 / hist2.sum()
                
                # Calculate cross entropy: H(p,q) = -sum(p(x) * log(q(x)))
                cross_entropy = -np.sum(prob1 * np.log2(prob2 + 1e-7))
                return cross_entropy
class Loss(nn.Module):
     def __init__(self, f_weight=1.0,vi_weight=1.0,pixel_weight=1.0, gradient_weight=1.0, ssim_weight=1.0, perceptual_weight=1.0,rec_loss_weight=1.0,KL_loss_weight=0.01):
          super().__init__()
          self.fusion_loss=FusionLoss(pixel_weight, gradient_weight, ssim_weight, perceptual_weight)
          self.vi_loss=VI_Loss(rec_loss_weight,KL_loss_weight)
          self.f_weight=f_weight
          self.vi_weight=vi_weight
     def forward(self, f,i, v, g, l, mu_l, sigma2_l, mu_g, sigma2_g):
         fusion_losses = self.fusion_loss(f, v, i)
         vi_losses = self.vi_loss(i, v, g, l, mu_l, sigma2_l, mu_g, sigma2_g)
         total_loss=self.f_weight*fusion_losses['total_loss_f']+self.vi_weight*vi_losses['total_loss_vi']
         return {**fusion_losses, **vi_losses, 'total_loss': total_loss}
if __name__=="__main__":
    # 测试代码
   fused=torch.rand(size=(2, 3, 224, 224))
   visible=torch.rand(size=(2, 3, 224, 224))
   infrared=torch.rand(size=(2, 1, 224, 224))
   g=torch.rand(size=(2, 3, 224, 224))
   l=torch.rand(size=(2, 1, 224, 224))
   mu=torch.rand(size=(2, 3, 224, 224))
   sigma=torch.rand(size=(2, 3, 224, 224))
   loss_fn=Loss(rec_loss_weight=1.0,KL_loss_weight=0.01)
   losses=loss_fn(fused,visible, infrared, g, l, mu, sigma)
   print(losses)

