import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from restormer.restormer_arch import Restormer
import torch
import transformers
import torch.nn as nn
from component import VI_Encoder, VI_Decoder, Local_Multi_Expert_Decoder
from component import Global_Multi_Expert_Decoder,Fusion_Net,VI_Z
class I_encoder_decoder(nn.Module):
    def __init__(self,block_num=2,expert_num=4,topk_expert=2,alpha=1.0):
        super(I_encoder_decoder,self).__init__()
        self.encoder=VI_Encoder(block_num=block_num,mode="L")
        self.VI_Z=VI_Z(alpha=alpha,mode="L")
        self.decoder=Local_Multi_Expert_Decoder(expert_num=expert_num,topk_expert=topk_expert,mode="L")
    def forward(self,x):
        x=self.encoder(x)
        x,mu,sigma2=self.VI_Z(x)
        l=self.decoder(x)
        return l,mu,sigma2

class V_encoder_decoder(nn.Module):
    def __init__(self,block_num=2,expert_num=4,topk_expert=2,alpha=1.0):
        super(V_encoder_decoder,self).__init__()
        self.encoder=VI_Encoder(block_num=block_num,mode="RGB")
        self.VI_Z=VI_Z(alpha=alpha,mode="RGB")
        self.decoder=Local_Multi_Expert_Decoder(expert_num=expert_num,topk_expert=topk_expert,mode="RGB")
    def forward(self,x):
        x=self.encoder(x)
        x,mu,sigma2=self.VI_Z(x)
        g=self.decoder(x)
        return g,mu,sigma2

class IV_fusion_model(nn.Module):
    def __init__(self,i_block_num=2,v_block_num=2,i_expert_num=3,v_expert_num=3,i_topk_expert=2,v_topk_expert=2,i_alpha=1.0,v_alpha=1.0,f_block_num=3):
        super(IV_fusion_model,self).__init__()
        self.I_encoder_decoder=I_encoder_decoder(block_num=i_block_num,expert_num=i_expert_num,topk_expert=i_topk_expert,alpha=i_alpha)
        self.V_encoder_decoder=V_encoder_decoder(block_num=v_block_num,expert_num=v_expert_num,topk_expert=v_topk_expert,alpha=v_alpha)
        self.fusion_net=Fusion_Net(block_num=f_block_num)
    def forward(self,i,v,device_1="cuda:0",device_2="cuda:1",device_3="cuda:2"):
        i_cpu = i.clone().cpu()
        i = i.to(device_1)
        self.I_encoder_decoder=self.I_encoder_decoder.to(device_1)
        l,mu_l,sigma2_l = self.I_encoder_decoder(i)
        del i
        v_cpu=v.clone().cpu()
        v = v.to(device_2)
        self.V_encoder_decoder=self.V_encoder_decoder.to(device_2)
        g,mu_g,sigma2_g = self.V_encoder_decoder(v)
        del v
        l = l.to(device_3)
        g = g.to(device_3)
        i = i_cpu.to(device_3)
        v = v_cpu.to(device_3)
        self.fusion_net.to(device_3)
        Ic_i = i + l
        Ic_v = v + g
        fused_input = Ic_i+Ic_v
        del Ic_i, Ic_v, i, v
        fusion = self.fusion_net(fused_input)
        return fusion, l,g,mu_l, sigma2_l, mu_g, sigma2_g

if __name__ == "__main__":
    # 测试代码
   model=IV_fusion_model(i_block_num=2,v_block_num=2,i_expert_num=4,v_expert_num=4,i_topk_expert=2,v_topk_expert=2,i_alpha=1.0,v_alpha=1.0,f_block_num=4)
   i=torch.rand(size=(2, 1, 224, 224))
   v=torch.rand(size=(2, 3, 224, 224))
   fusion, l,g,mu_l, sigma2_l, mu_g, sigma2_g=model(i,v)
   print(fusion.shape)