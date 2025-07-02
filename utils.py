from typing import Dict, List, Union
import subprocess, operator

import torch
from torch_geometric.data import Data



def create_node_mask(data: Data, node_types: Union[int, List[int], str]) -> torch.Tensor:
    """
    Creates a boolean mask for specified node types in the data.
    
    Args:
        data: PyG Data object containing node_type attribute
        node_types: Single node type or list of node types to mask
        
    Returns:
        torch.Tensor: Boolean mask indicating nodes of specified types
    """
    if isinstance(node_types, list):
        mask = torch.zeros_like(data.node_type, dtype=torch.bool)
        for node_type in node_types:
            mask |= (data.node_type == node_type)
    elif isinstance(node_types, int) or isinstance(node_types, float):
        mask = data.node_type == node_types
    elif node_types == 'all':
        mask = torch.ones(data.size(0)).bool()
    elif node_types == 'none':
        mask = torch.zeros(data.size(0)).bool()
    return mask.squeeze()


def add_noise_(data: Data, metadata: Dict) -> None:
    """
    Adds Gaussian noise to input (and takes it away from target if target is diff)
    to specified attributes based on node types.
    
    Args:
        data: PyG Data object to modify
        metadata: Dictionary containing noise configuration:
            - noise_mask: Dict mapping attributes to node types
            - noise_std: Dict mapping attributes to noise standard deviations
    """
    for attr in metadata['noise_mask']:
        if attr not in metadata['fields']:
            continue
        mask = create_node_mask(data, metadata['noise_mask'][attr])
        noise = torch.randn_like(data[attr][mask]) * metadata['noise_std'][attr]
        data[attr][mask] += noise
        if metadata['diff_t0_t1'].get(attr, False):
            data[f'target_{attr}'][mask] -= noise


def move_graph_(graph: Data, device: torch.device) -> None:
    """
    Moves graph to the device in place.

    Args:
        graph (Data): The PyTorch Geometric data object to move.
        device (torch.device): The target device.
    """
    tensor_attrs = {k: v for k, v in graph.items() if isinstance(v, torch.Tensor)}
    transferred_tensors = {k: v.to(device, non_blocking=True) for k, v in tensor_attrs.items()}
    for k, v in transferred_tensors.items():
        graph[k] = v
    if device.type == "cuda":
        torch.cuda.synchronize()



# def get_freest_gpu():
#     result = subprocess.check_output(
#         ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
#         encoding='utf-8'
#     )
#     # List of free memory per GPU
#     memory_free = [int(x) for x in result.strip().split('\n')]
#     best_gpu = max(range(len(memory_free)), key=lambda i: memory_free[i])
#     return f'cuda:{best_gpu}'


# def get_device(metadata):
#     if torch.cuda.is_available():
#         if metadata['device'] == 'most_free':
#             try:
#                 device = get_freest_gpu()
#             except:
#                 device = 'cuda:2'  # default on cuda:2 if the process fails
#         else:
#             davice = metadata['device']
#     else:
#         device = 'cpu'
#     return device

