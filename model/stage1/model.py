import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from restormer.restormer_arch import Restormer
import torch
import transformers
import torch.nn as nn
from component import VI_Encoder, VI_Decoder, Noise_Multi_Expert_Decoder
from component import Fusion_Net,VI_Z
class Noise_encoder_decoder(nn.Module):
    def __init__(self,block_num=2,expert_num=4,topk_expert=2,alpha=1.0,mode="L"):
        super(Noise_encoder_decoder,self).__init__()
        if mode=="L":
            self.encoder=VI_Encoder(block_num=block_num,mode="L")
            self.VI_Z=VI_Z(alpha=alpha,mode="L")
            self.decoder=Noise_Multi_Expert_Decoder(expert_num=expert_num,topk_expert=topk_expert,mode="L")
        if mode=="RGB":
            self.encoder=VI_Encoder(block_num=block_num,mode="RGB")
            self.VI_Z=VI_Z(alpha=alpha,mode="RGB")
            self.decoder=Noise_Multi_Expert_Decoder(expert_num=expert_num,topk_expert=topk_expert,mode="RGB")
    def forward(self,x):
        x=self.encoder(x)
        x,mu,sigma2=self.VI_Z(x)
        n=self.decoder(x)
        del x
        return n,mu,sigma2


class Degrad_restore_model(nn.Module):
    def __init__(self,i_block_num=2,v_block_num=2,i_expert_num=3,v_expert_num=3,i_topk_expert=2,v_topk_expert=2,i_alpha=1.0,v_alpha=1.0,f_block_num=3,mode="L"):
        super(Degrad_restore_model,self).__init__()
        self.noise_encoder_decoder=Noise_encoder_decoder(block_num=i_block_num,expert_num=i_expert_num,topk_expert=i_topk_expert,alpha=i_alpha,mode=mode)
        self.fusion_net=Fusion_Net(block_num=f_block_num,mode=mode)
    def forward(self,image,device_1="cuda:0",device_2="cuda:1",device_3="cuda:2"):
        image_cpu = image.clone().cpu()
        image= image.to(device_1)
        self.noise_encoder_decoder=self.noise_encoder_decoder.to(device_1)
        n,mu_n,sigma2_n = self.noise_encoder_decoder(image)
        del image
        torch.cuda.empty_cache() 
        n= n.to(device_2)
        image=image_cpu.to(device_2)
        self.fusion_net.to(device_2)
        Ic_image = image-n
        fused_input = Ic_image
        del Ic_image
        torch.cuda.empty_cache() 
        out= self.fusion_net(fused_input)
        return out, n, mu_n, sigma2_n

if __name__ == "__main__":
    # 测试代码
    '''model=Degrad_restore_model(i_block_num=2,v_block_num=2,i_expert_num=4,v_expert_num=4,i_topk_expert=2,v_topk_expert=2,i_alpha=1.0,v_alpha=1.0,f_block_num=3,mode="L")
   i=torch.rand(size=(2, 1, 224, 224))
   v=torch.rand(size=(2, 3, 224, 224))
   fusion, n, mu_n, sigma2_n = model(i)
   print(fusion.shape)'''
   