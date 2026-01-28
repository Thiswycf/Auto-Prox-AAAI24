import torch
import torch.nn as nn
import scipy.stats
import numpy as np
from collections import *
from torch.nn.init import trunc_normal_

from . import measure

def kaiming_normal_fanin_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        if m.affine:
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

# 适合AutoFormer和PiT搜索空间的Transformer初始化方法
def init_method(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Conv2d):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.ones_(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.zeros_(m.bias)

def get_effective_rank(matrix):
    s = torch.linalg.svdvals(matrix)
    s /= torch.sum(s)
    erank = torch.e ** scipy.stats.entropy(s.detach().cpu())
    return np.nan_to_num(erank)

def getFeat(model, inputs):
    feats = []
    
    # Check if model is a Transformer (has features attribute)
    if hasattr(model, 'features'):
        # Clear existing features
        model.features.clear()
        # Forward pass to capture features
        model(inputs)
        org_feats = model.features
    else:
        # Original CNN processing
        model.topo_forward(inputs)
        _, _, edges = model.getTopo()
        org_feats = [_data for _, _, _data in edges]
            
    for _data in org_feats:
        dim = _data.dim()
        if dim == 4:
            b,c,h,w = _data.shape
            if h*w > 1024: # Use pooling at large resolutions
                _data = torch.nn.functional.adaptive_avg_pool2d(_data, (32, 32))
                h, w = 32, 32
            _data = _data.permute(0,2,3,1).contiguous().view(b,h*w,c)
        elif dim == 2: # Linear layers are not processed
            continue
        else:
            raise ValueError(f"Invalid dimension: {dim}")
        feats.append(_data)

    return feats

@measure('er', bn=True, mode='param')
def compute_er_score(net, inputs, targets, mode, loss_fn, split_data=1):
    device = inputs.device
    inputs = inputs.to(device)
    inputs = torch.randn_like(inputs)
    # first dimension better less than 32
    if inputs.size(0) > 32:
        inputs = inputs[:32]
    
    # 使用适合Transformer的初始化方法
    net.apply(init_method)
    feats = getFeat(net, inputs)
    
    score = 0.0
    for i, feat in enumerate(feats):
        if len(feat) == 0:
            continue
        def get_effective_rank(x):
            ############ 1.Covariance matrix ############
            if x.shape[1] < x.shape[2]: # modify
                x = x.transpose(2, 1)
            mean = x.mean(dim=[0,1], keepdim=True)
            std = x.std(dim=[0,1], keepdim=True)
            x = (x - mean) / (std + 1e-8)
            x = torch.bmm(x.transpose(2, 1), x) / (x.shape[1] - 1)
            ############################################

            ############# 2.Effective Rank #############
            try:
                s = torch.linalg.svdvals(x)
            except Exception as e:
                print(f'{i}/{len(feats)} activation.shape = {feat.shape}')
                raise e
            s /= torch.sum(s, dim=1, keepdim=True)
            entropy = scipy.stats.entropy(s.detach().cpu(), axis=1)
            erank = torch.e ** entropy
            erank = np.nan_to_num(erank)
            ############################################

            return erank.mean()
        score += get_effective_rank(feat)

    torch.cuda.empty_cache()
    
    return score