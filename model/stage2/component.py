import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from restormer.restormer_arch import Restormer
import torch
import transformers
import torch.nn as nn
class VI_Encoder(nn.Module):
    def __init__(self,block_num=2,mode="RGB"):
        super(VI_Encoder,self).__init__()
        self.model=nn.Sequential()
        if mode=="L":
            for i in range(block_num):
                self.model.add_module(f"restormer_block_{i+1}",Restormer(inp_channels=1,out_channels=1))
        if mode=="RGB":
            for i in range(block_num):
                self.model.add_module(f"restormer_block_{i+1}",Restormer(inp_channels=3,out_channels=3))
    def forward(self,x):
        x=self.model(x)
        return x

class VI_Decoder(nn.Module):
    def __init__(self,mode="RGB"):
        super(VI_Decoder,self).__init__()
        if mode=="L":
            self.model=Restormer(inp_channels=1,out_channels=1)
        if mode=="RGB":
            self.model=Restormer(inp_channels=3,out_channels=3)
    def forward(self,x):
        x=self.model(x)
        return x

class Local_Multi_Expert_Decoder(nn.Module):
    def __init__(self, expert_num=4, topk_expert=2,mode="L"):
        super(Local_Multi_Expert_Decoder, self).__init__()
        self.expert_num = expert_num
        self.mode = mode
        self.topk_expert=topk_expert
        
        # 创建多个VI_Decoder专家
        self.experts = nn.ModuleList()
        for i in range(expert_num):
            expert = VI_Decoder(mode=mode)
            self.experts.append(expert)
        
        # 门控网络，用于选择专家的权重
        channels = 1 if mode == "L" else 3
        self.gate_network = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(channels, expert_num * 2),
            nn.ReLU(),
            nn.Linear(expert_num * 2, expert_num * 2),
            nn.ReLU(),
            nn.Linear(expert_num * 2, expert_num * 2),
            nn.ReLU(),
            nn.Linear(expert_num * 2, expert_num * 2),
            nn.ReLU(),
            nn.Linear(expert_num * 2, expert_num),
            nn.Softmax(dim=1)
        )

    def topk_gate(self, gate_weights):
        # gate_weights: [batch_size, expert_num]
        topk_vals, topk_idx = torch.topk(gate_weights, self.topk_expert, dim=1)
        mask = torch.zeros_like(gate_weights)
        mask.scatter_(1, topk_idx, 1)
        gated = gate_weights * mask
        
        # 重新归一化非零部分
        gated_sum = gated.sum(dim=1, keepdim=True) + 1e-8
        gated = gated / gated_sum
        return gated
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 获取门控权重
        gate_weights = self.gate_network(x)  # [batch_size, expert_num]
        #print(f"原始门控权重: {gate_weights}")
        
        # 应用topk门控
        gated_weights = self.topk_gate(gate_weights)  # [batch_size, expert_num]
        #print(f"TopK门控后权重: {gated_weights}")
        
        # 所有专家的输出
        expert_outputs = []
        for expert in self.experts:
            output = expert(x)
            expert_outputs.append(output)
        
        # 堆叠专家输出 [expert_num, batch_size, channels, height, width]
        expert_outputs = torch.stack(expert_outputs, dim=0)
        
        # 重塑门控权重以便于广播 [expert_num, batch_size, 1, 1, 1]
        gated_weights = gated_weights.transpose(0, 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # 加权组合专家输出
        weighted_output = expert_outputs * gated_weights
        final_output = torch.sum(weighted_output, dim=0)
        
        return final_output

class Global_Multi_Expert_Decoder(nn.Module):
    def __init__(self, expert_num=4, topk_expert=2,mode="RGB"):
        super(Global_Multi_Expert_Decoder, self).__init__()
        self.expert_num = expert_num
        self.mode = mode
        self.topk_expert=topk_expert
        
        # 创建多个VI_Decoder专家
        self.experts = nn.ModuleList()
        for i in range(expert_num):
            expert = VI_Decoder(mode=mode)
            self.experts.append(expert)
        
        # 门控网络，用于选择专家的权重
        channels = 1 if mode == "L" else 3
        self.gate_network = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Flatten(),
            nn.Linear(channels, expert_num * 2),
            nn.ReLU(),
            nn.Linear(expert_num * 2, expert_num * 2),
            nn.ReLU(),
            nn.Linear(expert_num * 2, expert_num * 2),
            nn.ReLU(),
            nn.Linear(expert_num * 2, expert_num * 2),
            nn.ReLU(),
            nn.Linear(expert_num * 2, expert_num),
            nn.Softmax(dim=1)
        )

    def topk_gate(self, gate_weights):
        # gate_weights: [batch_size, expert_num]
        topk_vals, topk_idx = torch.topk(gate_weights, self.topk_expert, dim=1)
        mask = torch.zeros_like(gate_weights)
        mask.scatter_(1, topk_idx, 1)
        gated = gate_weights * mask
        
        # 重新归一化非零部分
        gated_sum = gated.sum(dim=1, keepdim=True) + 1e-8
        gated = gated / gated_sum
        return gated
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 获取门控权重
        gate_weights = self.gate_network(x)  # [batch_size, expert_num]
        #print(f"原始门控权重: {gate_weights}")
        
        # 应用topk门控
        gated_weights = self.topk_gate(gate_weights)  # [batch_size, expert_num]
        #print(f"TopK门控后权重: {gated_weights}")
        
        # 所有专家的输出
        expert_outputs = []
        for expert in self.experts:
            output = expert(x)
            expert_outputs.append(output)
        
        # 堆叠专家输出 [expert_num, batch_size, channels, height, width]
        expert_outputs = torch.stack(expert_outputs, dim=0)
        
        # 重塑门控权重以便于广播 [expert_num, batch_size, 1, 1, 1]
        gated_weights = gated_weights.transpose(0, 1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        # 加权组合专家输出
        weighted_output = expert_outputs * gated_weights
        final_output = torch.sum(weighted_output, dim=0)
        
        return final_output

class Fusion_Net(nn.Module):
    def __init__(self,block_num=4):
        super(Fusion_Net,self).__init__()
        self.model=nn.Sequential()
        for i in range(block_num):
                self.model.add_module(f"restormer_block_{i+1}",Restormer(inp_channels=3,out_channels=3))
    def forward(self,x):
        x=self.model(x)
        return x

class VI_Z(nn.Module):
    def __init__(self,alpha,mode="RGB"):
        super().__init__()
        self.alpha=alpha
        if mode=="RGB":
            self.mmu=nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),nn.ReLU(),nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.ReLU(),nn.Conv2d(64, 3, kernel_size=3, padding=1))
            self.msigma2=nn.Sequential(nn.Conv2d(3, 64, kernel_size=3, padding=1),nn.ReLU(),nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.ReLU(),nn.Conv2d(64, 3, kernel_size=3, padding=1))
        if mode=="L":
            self.mmu=nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1),nn.ReLU(),nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.ReLU(),nn.Conv2d(64, 1, kernel_size=3, padding=1))
            self.msigma2=nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, padding=1),nn.ReLU(),nn.Conv2d(64, 64, kernel_size=3, padding=1),nn.ReLU(),nn.Conv2d(64, 1, kernel_size=3, padding=1))
        
    def get_mu_sigma2(self,x):
        mu=self.mmu(x)
        sigma2=self.msigma2(x)
        return mu,sigma2
    def forward(self,z):
        self.mu,self.sigma2=self.get_mu_sigma2(z)
        epsil=torch.randn_like(self.sigma2)
        self.out=self.mu+self.alpha*torch.sqrt(torch.abs(self.sigma2))*epsil
        return self.out,self.mu,self.sigma2

        
        

if __name__=="__main__":
    # 创建测试数据
    Ic_i = torch.rand(size=(2, 1, 224, 224)).to('cuda:0')
    Ic_v = torch.rand(size=(2, 3, 224, 224)).to('cuda:1')
    
    # 测试编码器
    print("=== 测试编码器 ===")
    v_encoder = VI_Encoder(block_num=2, mode="RGB").to('cuda:1')
    i_encoder = VI_Encoder(block_num=2, mode="L").to('cuda:0')
    
    v = v_encoder(Ic_v)
    i = i_encoder(Ic_i)
    print(f"RGB编码器输出形状: {v.shape}")
    print(f"红外编码器输出形状: {i.shape}")
    # 测试VIZ
    print("\n=== 测试VIZ ===")
    v_z = VI_Z(alpha=0.5, mode="RGB").to('cuda:1')
    i_z = VI_Z(alpha=0.5, mode="L").to('cuda:0')

    v_out = v_z(v)
    i_out = i_z(i)

    print(f"RGB VIZ输出形状: {v_out.shape}")
    print(f"红外 VIZ输出形状: {i_out.shape}")

    # 测试单个解码器
    print("\n=== 测试单个解码器 ===")
    v_decoder = VI_Decoder(mode="RGB").to('cuda:1')
    i_decoder = VI_Decoder(mode="L").to('cuda:0')


    v_decoded = v_decoder(v_out)
    i_decoded = i_decoder(i_out)
    print(f"RGB解码器输出形状: {v_decoded.shape}")
    print(f"红外解码器输出形状: {i_decoded.shape}")
    


    # 测试多专家解码器
    print("\n=== 测试多专家解码器 ===")
    multi_expert_v = Global_Multi_Expert_Decoder(expert_num=4, mode="RGB").to('cuda:1')
    multi_expert_i = Local_Multi_Expert_Decoder(expert_num=4, mode="L").to('cuda:0')
    
    v_multi_decoded = multi_expert_v(v)
    i_multi_decoded = multi_expert_i(i)
    print(f"RGB多专家解码器输出形状: {v_multi_decoded.shape}")
    print(f"红外多专家解码器输出形状: {i_multi_decoded.shape}")
    
    # 显示专家数量
    print(f"\nRGB多专家解码器包含 {multi_expert_v.expert_num} 个专家")
    print(f"红外多专家解码器包含 {multi_expert_i.expert_num} 个专家")

    fusion_net=Fusion_Net().to("cuda:2")
    v_multi_decoded=v_multi_decoded.to("cuda:2")
    i_multi_decoded=i_multi_decoded.to("cuda:2")
    Ic_i = Ic_i.to('cuda:2')
    Ic_v = Ic_v.to('cuda:2')
    Ici=Ic_i+i_multi_decoded
    Icv=Ic_v+v_multi_decoded
    # 在通道维度（dim=1）拼接两个张量
    fused_input = torch.cat([Ici, Icv], dim=1)
    fusion=fusion_net(fused_input)
    print(f"拼接后的张量形状: {fusion.shape}")

    
