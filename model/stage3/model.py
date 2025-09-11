import sys
import os
import importlib.util

# 添加model目录到Python路径
model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(model_dir)

import torch
import torch.nn as nn
from stage1.component import VI_Encoder as stage1_VI_Encoder, VI_Decoder as stage1_VI_Decoder, Noise_Multi_Expert_Decoder
from stage1.component import Fusion_Net as stage1_Fusion_Net,VI_Z as stage1_VI_Z
from stage1.model import Noise_encoder_decoder as stage1_Noise_encoder_decoder
from stage1.model import Degrad_restore_model as stage1_Degrad_restore_model


from stage2.component import VI_Encoder as stage2_VI_Encoder, VI_Decoder as stage2_VI_Decoder, Local_Multi_Expert_Decoder as stage2_Local_Multi_Expert_Decoder
from stage2.component import Global_Multi_Expert_Decoder as stage2_Global_Multi_Expert_Decoder,Fusion_Net as stage2_Fusion_Net,VI_Z as stage2_VI_Z
from stage2.model import IV_fusion_model as stage2_IV_fusion_model
from stage2.model import I_encoder_decoder as stage2_I_encoder_decoder
from stage2.model import V_encoder_decoder as stage2_V_encoder_decoder


from loss import FusionLoss
class Degrad_restore_model(nn.Module):
    def __init__(self,i_block_num=2,v_block_num=2,i_expert_num=4,v_expert_num=4,i_topk_expert=2,v_topk_expert=2,i_alpha=1.0,v_alpha=1.0,f_block_num=2,mode="L"):
        super(Degrad_restore_model,self).__init__()
        self.noise_encoder_decoder=stage1_Noise_encoder_decoder(block_num=i_block_num,expert_num=i_expert_num,topk_expert=i_topk_expert,alpha=i_alpha,mode=mode)
        self.fusion_net=stage1_Fusion_Net(block_num=f_block_num,mode=mode)
    def forward(self,image):
        with torch.no_grad():
            n,_,_ = self.noise_encoder_decoder(image)
            Ic_image = image-n
            fused_input = Ic_image
        out= self.fusion_net(fused_input)
        return out

class IV_fusion_model(nn.Module):
    def __init__(self,i_block_num=2,v_block_num=2,i_expert_num=4,v_expert_num=4,i_topk_expert=2,v_topk_expert=2,i_alpha=1.0,v_alpha=1.0,f_block_num=2):
        super(IV_fusion_model,self).__init__()
        self.I_encoder_decoder=stage2_I_encoder_decoder(block_num=i_block_num,expert_num=i_expert_num,topk_expert=i_topk_expert,alpha=i_alpha)
        self.V_encoder_decoder=stage2_V_encoder_decoder(block_num=v_block_num,expert_num=v_expert_num,topk_expert=v_topk_expert,alpha=v_alpha)
        self.fusion_net=stage2_Fusion_Net(block_num=f_block_num)
    def forward(self,i,v):
        with torch.no_grad():
            l,_,_= self.I_encoder_decoder(i)
            g,_,_ = self.V_encoder_decoder(v)
            Ic_i = i + l
            Ic_v = v + g
            fused_input = 0.5*Ic_i+0.5*Ic_v
        fusion = self.fusion_net(fused_input)
        return fusion

class DIV_fusion_model(nn.Module):
    def __init__(self,task):
        super(DIV_fusion_model,self).__init__()
        self.task=task
        if task=="dv_i":
            self.idr=Degrad_restore_model(i_block_num=2,v_block_num=2,i_expert_num=4,v_expert_num=4,i_topk_expert=2,v_topk_expert=2,i_alpha=1.0,v_alpha=1.0,f_block_num=2,mode="L")
            self.civ=IV_fusion_model(i_block_num=2,v_block_num=2,i_expert_num=4,v_expert_num=4,i_topk_expert=2,v_topk_expert=2,i_alpha=1.0,v_alpha=1.0,f_block_num=2)
        if task=="dv_i":
            self.vdr=Degrad_restore_model(i_block_num=2,v_block_num=2,i_expert_num=4,v_expert_num=4,i_topk_expert=2,v_topk_expert=2,i_alpha=1.0,v_alpha=1.0,f_block_num=2,mode="RGB")
            self.civ=IV_fusion_model(i_block_num=2,v_block_num=2,i_expert_num=4,v_expert_num=4,i_topk_expert=2,v_topk_expert=2,i_alpha=1.0,v_alpha=1.0,f_block_num=2)
    def forward(self,di,dv):
        if self.task=="dv_i":
            clean_v= self.vdr(dv)
            clean_i= di
            clean_fusion= self.civ(clean_i,clean_v)
        return clean_fusion

if __name__ == "__main__":
    model=DIV_fusion_model(task="dv_i")
    checkpoint_iv = torch.load("/data/1024whs_checkpoint/iv_fusion/Train_IVfusion_DroneVehicle/best_model.pth")
    checkpoint_dv = torch.load("/data/1024whs_checkpoint/Degradclean/Train_Degradclean_DroneRGBT_visible/best_model.pth")
    model.vdr.load_state_dict(checkpoint_dv['model_state_dict'])
    model.civ.load_state_dict(checkpoint_iv['model_state_dict'])
    print(model)
    