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
        try:
            from torchvision.models import VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        except ImportError:
            # 兼容旧版本
            vgg = vgg16(pretrained=True)
        self.vgg_features = nn.Sequential(*list(vgg.features)[:16]).eval()
        for param in self.vgg_features.parameters():
            param.requires_grad = False
    
    def pixel_loss(self, clean_pre, clean):
        """Pixel-level L1 loss"""
        loss = F.l1_loss(clean_pre, clean)
        return loss 
    
    def gradient_loss(self, clean_pre, clean):
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
        
        grad_clean_pre = sobel_gradient(clean_pre)
        grad_clean = sobel_gradient(clean)
        loss = F.l1_loss(grad_clean_pre, grad_clean)
        return loss

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
        ssim_loss = self.ssim_loss(clean_pre, clean)
        perceptual_loss = self.perceptual_loss(clean_pre, clean)

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
    def forward(self, degrad, clean, n, mu_n, sigma2_n):
        # Calculate entropy for images i and v
        n_true=degrad-clean
        # Use MSE loss instead of cross entropy to preserve gradients
        rec_n = self.calculate_cross_entropy(n_true, n)

        sigma2_n_safe = torch.clamp(sigma2_n, min=1e-8)

        kl_loss_n = -0.5 * torch.mean(1 + torch.log(sigma2_n_safe) - mu_n.pow(2) - sigma2_n_safe)
        # Combine losses
        total_loss = 0.5*(self.rec_loss_weight * rec_n + self.KL_loss_weight * kl_loss_n)
        return {
            'total_loss_vi': total_loss,
            'rec_n': rec_n,
            'kl_loss_n': kl_loss_n
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
                # Clamp values to valid range and add small epsilon
                img1 = torch.clamp(img1, min=0.0, max=1.0)
                img2 = torch.clamp(img2, min=0.0, max=1.0)
                
                # Flatten tensors
                img1_flat = img1.view(-1)
                img2_flat = img2.view(-1)
                
                # Calculate histograms using PyTorch
                bins = 256
                hist1 = torch.histc(img1_flat, bins=bins, min=0.0, max=1.0)
                hist2 = torch.histc(img2_flat, bins=bins, min=0.0, max=1.0)
                
                # Add small value to avoid log(0) and ensure numerical stability
                eps = 1e-8
                hist1 = hist1 + eps
                hist2 = hist2 + eps
                
                # Normalize to get probabilities
                prob1 = hist1 / torch.sum(hist1)
                prob2 = hist2 / torch.sum(hist2)
                
                # Calculate cross entropy: H(p,q) = -sum(p(x) * log(q(x)))
                # Use log instead of log2 for better numerical stability
                cross_entropy = -torch.sum(prob1 * torch.log(prob2 + eps))
                
                # Ensure the result is finite and not NaN
                cross_entropy = torch.where(torch.isfinite(cross_entropy), cross_entropy, torch.tensor(0.0, device=img1.device))
                
                return cross_entropy
class Loss(nn.Module):
     def __init__(self, f_weight=1.0,vi_weight=1.0,pixel_weight=1.0, gradient_weight=1.0, ssim_weight=1.0, perceptual_weight=1.0,rec_loss_weight=1.0,KL_loss_weight=0.01):
          super().__init__()
          self.fusion_loss=FusionLoss(pixel_weight, gradient_weight, ssim_weight, perceptual_weight)
          self.vi_loss=VI_Loss(rec_loss_weight,KL_loss_weight)
          self.f_weight=f_weight
          self.vi_weight=vi_weight
     def forward(self, clean_pre,degrad,clean,n,mu_n, sigma2_n):
         fusion_losses = self.fusion_loss(clean_pre, clean)
         vi_losses = self.vi_loss(degrad, clean, n, mu_n, sigma2_n)
         total_loss=self.f_weight*fusion_losses['total_loss_f']+self.vi_weight*vi_losses['total_loss_vi']
         return {**fusion_losses, **vi_losses, 'total_loss': total_loss}
if __name__=="__main__":
    # 测试代码
   
   '''clean_pre=torch.rand(size=(2, 3, 224, 224))
   clean=torch.rand(size=(2, 3, 224, 224))
   degrad=torch.rand(size=(2, 3, 224, 224))
   n=torch.rand(size=(2, 3, 224, 224))
   mu_n=torch.rand(size=(2, 3, 224, 224))
   sigma2_n=torch.rand(size=(2, 3, 224, 224))
   loss_fn=Loss()
   losses=loss_fn(clean_pre,degrad,clean,n,mu_n, sigma2_n)
   print(losses)'''

