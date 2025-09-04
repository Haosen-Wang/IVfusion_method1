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
        ssim_vis = ssim(fused, visible)
        ssim_ir = ssim(fused, infrared)
        loss = 1 - (ssim_vis + ssim_ir) / 2
        return loss 
    
    def perceptual_loss(self, fused, visible, infrared):
        """Perceptual loss using VGG16 features"""
        # Convert to 3-channel if grayscale
        if fused.size(1) == 1:
            fused = fused.repeat(1, 3, 1, 1)
        if visible.size(1) == 1:
            visible = visible.repeat(1, 3, 1, 1)
        if infrared.size(1) == 1:
            infrared = infrared.repeat(1, 3, 1, 1)
        
        fused_features = self.vgg_features(fused)
        visible_features = self.vgg_features(visible)
        infrared_features = self.vgg_features(infrared)
        
        loss = F.l1_loss(fused_features, visible_features) + F.l1_loss(fused_features, infrared_features)
        return loss 
    def forward(self, fused, visible, infrared):
        pixel_loss = self.pixel_loss(fused, visible, infrared)
        gradient_loss = self.gradient_loss(fused, visible, infrared)
        ssim_loss = self.ssim_loss(fused, visible, infrared)
        perceptual_loss = self.perceptual_loss(fused, visible, infrared)
        
        total_loss = (self.pixel_weight * pixel_loss + 
                     self.gradient_weight * gradient_loss +
                     self.ssim_weight * ssim_loss +
                     self.perceptual_weight * perceptual_loss)

        return {
            'total_loss_f': total_loss,
            'pixel_loss': pixel_loss,
            'gradient_loss': gradient_loss,
            'perceptual_loss': perceptual_loss
        }
    
class VI_Loss(nn.Module):
    def __init__(self, rec_loss_weight=1.0,KL_loss_weight=0.01):
        super(VI_Loss, self).__init__()
        self.rec_loss_weight = rec_loss_weight
        self.KL_loss_weight = KL_loss_weight
    def forward(self,i,v,g,l,mu_i, sigma2_i,mu_v, sigma2_v):
        # Calculate entropy for images i and v
        l_tru=torch.max(i,v)
        l = l.expand_as(l_tru)
        # Use MSE loss instead of cross entropy to preserve gradients
        rec_l = F.mse_loss(l, l_tru)
        # Calculate reconstruction loss (you can adjust this based on your needs)
        grad_visible = self.sobel_gradient(v)
        grad_infrared = self.sobel_gradient(i)
        g_tru=torch.max(grad_visible,grad_infrared)
        g=g.expand_as(g_tru)
        # Use MSE loss instead of cross entropy to preserve gradients
        rec_g = F.mse_loss(g, g_tru)
        # Calculate reconstruction loss and KL loss
        # 确保 sigma2 值为正，避免 log(0) 或 log(负数)
        sigma2_i_safe = torch.clamp(sigma2_i, min=1e-8)
        sigma2_v_safe = torch.clamp(sigma2_v, min=1e-8)
        
        kl_loss_i = -0.5 * torch.mean(1 + torch.log(sigma2_i_safe) - mu_i.pow(2) - sigma2_i_safe)
        kl_loss_g = -0.5 * torch.mean(1 + torch.log(sigma2_v_safe) - mu_v.pow(2) - sigma2_v_safe)
        # Combine losses
        total_loss = 0.5*(self.rec_loss_weight * rec_l + self.KL_loss_weight * kl_loss_i + self.rec_loss_weight * rec_g + self.KL_loss_weight * kl_loss_g)
        #total_loss = 0.5*(self.rec_loss_weight * rec_l +  self.rec_loss_weight * rec_g)
        return {
            'total_loss_vi': total_loss,
            'rec_i': rec_l,
            'rec_g': rec_g,
            'kl_loss_i': kl_loss_i,
            'kl_loss_g': kl_loss_g
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

