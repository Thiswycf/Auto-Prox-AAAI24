"""
Auto-Prox structures for combining zero-cost proxies.
Supports TreeStructure, LinearStructure, and GraphStructure.
"""
import torch
import torch.nn.functional as F
import numpy as np
from pycls.predictor.pruners.predictive import find_measures

# Mapping from input_geno identifiers to ZCP measure names
ZCP_MAPPING = {
    't2': 'synflow',  # Typical mapping, may need adjustment
    't4g': 'grasp',   # Typical mapping, may need adjustment
    'er': 'er',       # Effective Rank
    'ntk': 'ntk',
    'snip': 'snip',
    'fisher': 'fisher',
    'dss': 'dss',
    'nwot': 'nwot',
    'size': 'size',
    'epe_nas': 'epe_nas',
}


class BaseStructure:
    """Base class for Auto-Prox structures."""
    
    def __init__(self):
        self._genotype = {
            'input_geno': None,
            'op_geno': None,
            'repr_geno': None
        }
    
    def _get_zcp_scores(self, model, data_loader, input_geno, device):
        """Compute ZCP scores for given input_geno identifiers."""
        scores = {}
        
        # Get num_classes from model or config
        try:
            num_classes = model.num_classes if hasattr(model, 'num_classes') else 100
        except:
            num_classes = 100
        
        for geno_id in input_geno:
            if geno_id not in ZCP_MAPPING:
                raise ValueError(f"Unknown input_geno identifier: {geno_id}. Available: {list(ZCP_MAPPING.keys())}")
            
            zcp_name = ZCP_MAPPING[geno_id]
            
            # Set dataload info based on proxy method
            if zcp_name == 'grasp':
                dataload_info = ['grasp', 1, num_classes]
            else:
                dataload_info = ['random', 1, num_classes]
            
            # Calculate ZCP score
            try:
                score = find_measures(
                    model,
                    data_loader,
                    dataload_info=dataload_info,
                    device=device,
                    loss_fn=F.cross_entropy,
                    measure_names=[zcp_name]
                )
                scores[geno_id] = score
            except Exception as e:
                import traceback
                print(f"Error computing {zcp_name} (geno_id: {geno_id}): {e}")
                traceback.print_exc()
                scores[geno_id] = 0.0
        
        return scores
    
    def __call__(self, inputs, targets, model):
        """Compute the combined ZCP score."""
        raise NotImplementedError("Subclasses must implement __call__")


class TreeStructure(BaseStructure):
    """Tree structure for combining ZCPs."""
    
    def __init__(self):
        super().__init__()
    
    def _normalize(self, x, method='min_max'):
        """Normalize values."""
        if method == 'min_max':
            x_min, x_max = x.min(), x.max()
            if x_max - x_min < 1e-8:
                return torch.ones_like(x)
            return (x - x_min) / (x_max - x_min)
        elif method == 'z_score':
            mean, std = x.mean(), x.std()
            if std < 1e-8:
                return torch.ones_like(x)
            return (x - mean) / std
        return x
    
    def _apply_op(self, x, op_name):
        """Apply operation to tensor."""
        if op_name == 'min_max_normalize':
            return self._normalize(x, 'min_max')
        elif op_name == 'z_score_normalize':
            return self._normalize(x, 'z_score')
        elif op_name == 'frobenius_norm':
            return torch.norm(x, p='fro')
        elif op_name == 'sigmoid':
            return torch.sigmoid(x)
        elif op_name == 'element_wise_sqrt':
            return torch.sqrt(torch.clamp(x, min=1e-8))
        elif op_name == 'element_wise_square':
            return x ** 2
        elif op_name == 'log':
            return torch.log(torch.clamp(x, min=1e-8))
        elif op_name == 'abs':
            return torch.abs(x)
        else:
            return x
    
    def _binary_op(self, x1, x2, op_name):
        """Apply binary operation."""
        if op_name == 'element_wise_sum':
            return x1 + x2
        elif op_name == 'element_wise_mul':
            return x1 * x2
        elif op_name == 'element_wise_sub':
            return x1 - x2
        elif op_name == 'element_wise_div':
            return x1 / (x2 + 1e-8)
        else:
            return x1 + x2
    
    def __call__(self, inputs, targets, model):
        """Compute tree-structured ZCP score."""
        device = inputs.device if isinstance(inputs, torch.Tensor) else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Get input_geno from genotype
        input_geno = self._genotype.get('input_geno', [])
        if not input_geno:
            raise ValueError("input_geno not set in genotype")
        
        # Create a dummy data loader for ZCP computation
        # In practice, this should use the actual data loader
        from pycls.datasets.loader import construct_proxy_loader
        data_loader = construct_proxy_loader()
        
        # Get ZCP scores - compute once per model
        zcp_scores = self._get_zcp_scores(model, data_loader, input_geno, device)
        
        # Convert scores to tensors
        score_tensors = []
        for geno_id in input_geno:
            score = zcp_scores[geno_id]
            if isinstance(score, (int, float)):
                score = torch.tensor(float(score), device=device)
            elif isinstance(score, torch.Tensor):
                score = score.to(device)
            else:
                score = torch.tensor(float(score), device=device)
            score_tensors.append(score)
        
        # Apply tree operations based on repr_geno
        repr_geno = self._genotype.get('repr_geno', '')
        
        # Simple implementation: if repr_geno is provided, parse it
        # Otherwise, use default operations
        if repr_geno and 'TREE:' in repr_geno:
            # Parse repr_geno (simplified version)
            # Format: INPUT:(t2, t4g)TREE:(op1|op2|...)BINARY:(binary_op)
            try:
                # Extract operations
                tree_part = repr_geno.split('TREE:')[1].split('BINARY:')[0]
                ops = [op.strip() for op in tree_part.strip('()').split('|')]
                
                binary_part = repr_geno.split('BINARY:')[1].strip('()')
                binary_op = binary_part.strip()
                
                # Apply tree operations
                result = score_tensors[0]
                for op in ops:
                    result = self._apply_op(result, op)
                
                # Apply binary operation if multiple inputs
                if len(score_tensors) > 1:
                    result2 = score_tensors[1]
                    for op in ops:
                        result2 = self._apply_op(result2, op)
                    result = self._binary_op(result, result2, binary_op)
            except:
                # Fallback: simple sum
                result = sum(score_tensors)
        else:
            # Default: weighted sum
            result = sum(score_tensors)
        
        # Return scalar value
        if isinstance(result, torch.Tensor):
            return result.item() if result.numel() == 1 else result.sum().item()
        return float(result)


class LinearStructure(BaseStructure):
    """Linear structure for combining ZCPs."""
    
    def __init__(self):
        super().__init__()
    
    def __call__(self, inputs, targets, model):
        """Compute linear-structured ZCP score."""
        device = inputs.device
        
        input_geno = self._genotype.get('input_geno', [])
        if not input_geno:
            raise ValueError("input_geno not set in genotype")
        
        from pycls.datasets.loader import construct_proxy_loader
        data_loader = construct_proxy_loader()
        
        zcp_scores = self._get_zcp_scores(model, data_loader, input_geno, device)
        
        # Linear combination: weighted sum
        op_geno = self._genotype.get('op_geno', None)
        if op_geno and isinstance(op_geno, list):
            weights = op_geno[:len(input_geno)]
        else:
            weights = [1.0] * len(input_geno)
        
        result = sum(w * zcp_scores[geno_id] for w, geno_id in zip(weights, input_geno))
        
        if isinstance(result, torch.Tensor):
            return result.item() if result.numel() == 1 else result.sum().item()
        return float(result)


class GraphStructure(BaseStructure):
    """Graph structure for combining ZCPs."""
    
    def __init__(self):
        super().__init__()
    
    def __call__(self, inputs, targets, model):
        """Compute graph-structured ZCP score."""
        # For now, use similar to TreeStructure
        # Full graph implementation would require more complex graph operations
        tree = TreeStructure()
        tree._genotype = self._genotype
        return tree(inputs, targets, model)
