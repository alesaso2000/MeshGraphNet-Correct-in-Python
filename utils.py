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



def get_freest_gpu():
    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
        encoding='utf-8'
    )
    # List of free memory per GPU
    memory_free = [int(x) for x in result.strip().split('\n')]
    best_gpu = max(range(len(memory_free)), key=lambda i: memory_free[i])
    return f'cuda:{best_gpu}'


def get_device(metadata):
    if torch.cuda.is_available():
        if metadata['device'] == 'most_free':
            try:
                device = get_freest_gpu()
            except:
                device = 'cuda:2'  # default on cuda:2 if the process fails
        else:
            davice = metadata['device']
    else:
        device = 'cpu'
    return device




# def get_most_free_gpu_index():
#     """
#     Finds the index of the NVIDIA GPU with the most free memory using nvidia-smi.

#     Requires the nvidia-smi command-line tool to be installed and in the system's PATH.

#     Returns:
#         int: The index (0, 1, 2, ...) of the GPU with the most free memory.
#         Returns -1 if nvidia-smi fails, no GPUs are found, or output cannot be parsed.
#     """
#     # Command to query free memory for all GPUs, formatted as CSV, no header, no units (gives MiB)
#     command = "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits"
#     try:
#         # Execute the command
#         result = subprocess.run(
#             command,
#             shell=True,        # Allows running the command through the shell
#             check=True,        # Raises CalledProcessError if command returns non-zero exit code
#             stdout=subprocess.PIPE, # Capture standard output
#             stderr=subprocess.PIPE, # Capture standard error
#             universal_newlines=True # Decode stdout/stderr as text (UTF-8 by default)
#         )

#         # Get the output and strip leading/trailing whitespace
#         output = result.stdout.strip()

#         # If the output is empty, something went wrong or no GPUs found
#         if not output:
#             print("Error: nvidia-smi returned empty output. No GPUs detected or driver issue?")
#             return -1

#         # Split the output into lines (one per GPU) and convert memory to integers
#         free_memory_list = [int(mem) for mem in output.split('\n')]

#         # Check if we actually got memory values
#         if not free_memory_list:
#              print("Error: Could not parse free memory values from nvidia-smi output.")
#              return -1

#         # Find the index of the GPU with the maximum free memory
#         # enumerate provides (index, value) pairs
#         # max() finds the pair with the largest value (memory) using the key
#         # operator.itemgetter(1) tells max to compare based on the second element of the pair (the memory value)
#         gpu_index, max_free_memory = max(enumerate(free_memory_list), key=operator.itemgetter(1))

#         # Alternatively, without operator module:
#         # gpu_index = max(range(len(free_memory_list)), key=lambda i: free_memory_list[i])
#         # max_free_memory = free_memory_list[gpu_index] # If you need the value too

#         print(f"GPU Free Memory (MiB): {free_memory_list}") # Optional: print memory for debugging
#         return gpu_index

#     except FileNotFoundError:
#         print("Error: 'nvidia-smi' command not found.")
#         print("Please ensure NVIDIA drivers and the toolkit are properly installed and 'nvidia-smi' is in your system's PATH.")
#         return -1
#     except subprocess.CalledProcessError as e:
#         print(f"Error executing nvidia-smi: {e}")
#         print(f"stderr: {e.stderr}")
#         return -1
#     except ValueError:
#         print(f"Error: Could not convert nvidia-smi memory output to integers.")
#         print(f"Raw output:\n{output}")
#         return -1
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#         return -1