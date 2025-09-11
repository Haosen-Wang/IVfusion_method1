import sys
import os
import importlib.util
# 添加model目录到Python路径
model_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(model_dir)
from test_metrics.tensor_example import TensorFusionEvaluator
import torch
if __name__ == "__main__":
    Evaluator=TensorFusionEvaluator()
    fuse=torch.randn(2,3,256,256)
    vis=torch.randn(2,3,256,256)
    ir=torch.randn(2,1,256,256)
    batch_metrics=Evaluator.evaluate_fusion_batch(fuse,vis,ir)
    for metric_name, value in batch_metrics.items():
        if not metric_name.endswith('_std'):
            std_name = f"{metric_name}_std"
            std_value = batch_metrics.get(std_name, 0)
            print(f"{metric_name:15s}: {value:8.6f} ± {std_value:.6f}")
