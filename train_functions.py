import os

import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import numpy as np
import matplotlib.pyplot as plt

import wandb

from typing import Dict, List, Union, Optional

from utils import create_node_mask
from config import node_types_to_train


def define_optimizer_scheduler(model: Module, initial_lr: float = 1e-4, 
                               decay_rate: float = 0.1**(1/15)) -> tuple[Adam, LambdaLR]:
    """
    Create an Adam optimizer with an exponentially decaying learning rate scheduler.

    This function sets up an optimization strategy with a configurable initial learning 
    rate and decay rate. By default, it reduces the learning rate by a factor of 10 
    every 15 epochs.

    Args:
        model (Module): The neural network model whose parameters will be optimized.
        initial_lr (float, optional): Starting learning rate for the optimizer. 
            Defaults to 1e-4.
        decay_rate (float, optional): Exponential decay rate for learning rate. 
            Defaults to reducing by factor of 10 every 15 epochs.

    Returns:
        tuple[Adam, LambdaLR]: A tuple containing the Adam optimizer and LambdaLR scheduler.
    """
    optimizer = Adam(model.parameters(), lr=initial_lr)
    lr_lambda = lambda epoch: decay_rate**epoch
    
    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler



def training_step(model: Module, data: Data,
                  masks: Dict[str, torch.Tensor], optimizer: Adam) -> Dict[str, float]:
    """
    Perform a single training step for a graph neural network model.

    Computes predictions, calculates mean squared error losses for each field,
    performs backpropagation, and updates model parameters.

    Args:
        model (Module): The neural network model to train.
        data (Data): The input graph data for training.
        masks (Dict[str, torch.Tensor]): Masks for each field indicating which nodes to use.
        optimizer (Adam): The optimizer for updating model parameters.

    Returns:
        Dict[str, float]: A dictionary of losses, including per-field and total losses.

    Note:
        - Expects target fields to be named 'target_{field}'
    """
    optimizer.zero_grad()
    
    preds = model(data)
    losses = {}
    losses['total'] = 0
    
    for i, field in enumerate(model.metadata['fields']):
        losses[field] = F.mse_loss(preds[masks[field], i], data[f'target_{field}'][masks[field]].squeeze())
        losses['total'] += losses[field]
    
    losses['total'] /= len(model.metadata['fields'])
    losses['total'].backward()
    optimizer.step()
    
    losses = {key: losses[key].item() for key in losses.keys()}
    
    return losses 


@torch.no_grad()
def get_val_loss(model: Module, dataloader: DataLoader) -> float:
    """
    Calculate validation loss for a model across all batches in the dataloader.

    Computes mean squared error for each field and logs results to Weights & Biases.

    Args:
        model (Module): The neural network model to evaluate.
        dataloader (DataLoader): Dataloader containing validation data.

    Returns:
        float: The total average validation loss across all fields.

    Note:
        - Uses node types specified in node_types_to_train
        - Logs individual and total validation losses to wandb
    """
    losses: Dict[str, List[float]] = {field: [] for field in model.metadata['fields']}
    
    for batch in dataloader:
        batch = batch.to(model.metadata['device'])
        preds = model(batch)
        
        masks = {field: create_node_mask(batch, node_types_to_train[field]) for field in model.metadata['fields']}
        
        for i, field in enumerate(model.metadata['fields']):
            losses[field].append(
                F.mse_loss(preds[masks[field], i], batch[f'target_{field}'][masks[field]].squeeze()).item()
            )
    
    losses_mean = {key: np.mean(losses[key]) for key in losses.keys()}
    losses_mean['total'] = np.mean(list(losses_mean.values()))
    
    wandb.log({f'val_loss_{field}': losses_mean[field] for field in losses_mean.keys()})
    
    return losses_mean['total']



def plot_losses(train_losses: List[float], val_losses: List[float], metadata: Dict,
                show: bool = False, path: Optional[str] = None) -> None:
    """
    Visualize training and validation losses with log-scale plotting.

    Creates a two-panel plot showing training and validation losses on log scales.

    Args:
        train_losses (List[float]): List of training losses.
        val_losses (List[float]): List of validation losses.
        show (bool, optional): Whether to display the plot. Defaults to False.
        path (Optional[str], optional): File path to save the plot. Defaults to None.

    Note:
        - Uses batch size from metadata for x-axis scaling
        - Supports both displaying and saving the plot
        - Automatically clears the plot after saving/showing to prevent memory issues
    """
    if not show and not path:
        return
    
    if path:
        filename = os.path.join(path, 'losses.pdf')
    
    x = np.arange(1, len(train_losses) + 1) * metadata['batch_size']
    x_val = np.linspace(
        len(train_losses) / len(val_losses), 
        len(train_losses), 
        len(val_losses)
    ).astype(int) * metadata['batch_size']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 12))
    
    # Training and validation loss plot
    ax1.plot(x, train_losses, label='train')
    ax1.scatter(x_val, val_losses, label='val')
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_title('Losses (Log Scale)')
    ax1.set_xlabel('Batch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Placeholder for potential second plot visualization
    ax2.axis('off')
    
    fig.suptitle('Training Metrics', fontsize=16)
    plt.tight_layout()
    
    if show:
        plt.show()
    
    if path:
        plt.savefig(filename)
    
    plt.clf()