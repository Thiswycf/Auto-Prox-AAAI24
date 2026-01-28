#!/usr/bin/env python
"""简单测试er measure是否能正常工作"""
import torch
import sys
sys.path.insert(0, '.')

from pycls.core.config import cfg, load_cfg
from pycls.models.build import MODEL
from pycls.predictor.pruners.predictive import find_measures
import torch.nn.functional as F
import pycls.datasets.loader as data_loader

# 加载配置
load_cfg('configs/auto/pit/pit-ti_c100_base.yaml')
cfg.PROXY_DATASET = 'cifar100'

# 创建数据加载器
data_loader_instance = data_loader.construct_proxy_loader()

# 创建模型
model = MODEL.get(cfg.MODEL.TYPE)(
    arch_config={
        'base_dim': 40,
        'mlp_ratio': 2,
        'depth': [2, 4, 6],
        'num_heads': [2, 4, 8]
    },
    num_classes=cfg.MODEL.NUM_CLASSES
)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 测试er measure
print("Testing er measure...")
try:
    dataload_info = ['random', 1, cfg.MODEL.NUM_CLASSES]
    er_score = find_measures(
        model,
        data_loader_instance,
        dataload_info=dataload_info,
        device=device,
        loss_fn=F.cross_entropy,
        measure_names=['er']
    )
    print(f"ER score: {er_score}")
    print("ER measure test passed!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
