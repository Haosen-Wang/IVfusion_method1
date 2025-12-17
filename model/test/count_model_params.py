import sys
import os
# 确保项目根路径在 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
# repository root is two levels up from model/test
project_root = os.path.dirname(os.path.dirname(current_dir))
# Ensure both project root and the top-level `model` directory are on sys.path
model_dir = os.path.join(project_root, 'model')
print(f"DEBUG: project_root={project_root}")
print(f"DEBUG: model_dir={model_dir}")
print("DEBUG: initial sys.path[0:5]=", sys.path[0:5])
if project_root not in sys.path:
    sys.path.append(project_root)
if model_dir not in sys.path:
    sys.path.append(model_dir)
print("DEBUG: updated sys.path[0:8]=", sys.path[0:8])
print(f"DEBUG: exists model_dir? {os.path.isdir(model_dir)}")
print(f"DEBUG: exists stage3? {os.path.isdir(os.path.join(model_dir, 'stage3'))}")

import torch
from stage3.model import DIV_fusion_model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def per_submodule_counts(model):
    counts = {}
    for name, module in model.named_children():
        t, tr = count_params(module)
        counts[name] = {'total': t, 'trainable': tr}
    return counts


def human(n):
    for unit in ['','K','M','B']:
        if n < 1000.0:
            return f"{n:.2f}{unit}"
        n /= 1000.0
    return f"{n:.2f}T"


def main(task='dv_i', device='cpu'):
    device = torch.device(device)
    print(f"Instantiating DIV_fusion_model(task={task}) on {device}")
    model = DIV_fusion_model(task=task)
    model.to(device)
    model.eval()

    total, trainable = count_params(model)
    print(f"Total params    : {total} ({human(total)})")
    print(f"Trainable params: {trainable} ({human(trainable)})")

    print("\nPer-submodule parameter counts:")
    counts = per_submodule_counts(model)
    for k,v in counts.items():
        print(f"  {k}: total={v['total']}, trainable={v['trainable']}")

    # optionally, list top-level param counts for large modules
    print('\nTop-level modules with param counts:')
    for name, module in model.named_modules():
        # only show top-level or direct children
        if '.' not in name and name != '':
            t = sum(p.numel() for p in module.parameters())
            if t>0:
                print(f"  {name}: {t}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='dv_i')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    main(task=args.task, device=args.device)
