import torch
import torch.nn as nn
import os

from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

from typing import List, Dict, Any, Union



class Normalizer:
    """
    A class for normalizing and denormalizing batched graph data using rolling statistics.
    """
    def __init__(self, fields: List[str]):
        self.fields = fields
        self.attributes = self.fields + ['edge_attr'] + [f'target_{field}' for field in self.fields]
        
        self.rolling_means = {attr: 0 for attr in self.attributes}
        self.rolling_means2 = {attr: 0 for attr in self.attributes}
        self.means = {}
        self.means2 = {}
        self.stds = {}
        self.num_rec = 0

    def record(self, batch: Data):
        """
        Records the mean and variance statistics from a batch.

        Args:
            batch (Data): A PyTorch Geometric Data object containing graph data.
        """
        
        self.num_rec += 1

        if self.num_rec <= 1e6:
            for attr in self.attributes:
                self.rolling_means[attr] += batch[attr].mean(dim=0)
                self.rolling_means2[attr] += (batch[attr]**2).mean(dim=0)
            
            for attr in self.attributes:
                self.means[attr] = self.rolling_means[attr] / self.num_rec
                self.means2[attr] = self.rolling_means2[attr] / self.num_rec
                self.stds[attr] = torch.sqrt(self.means2[attr] - self.means[attr]**2 + 1e-8)  # Avoid division by zero

    def to_device_(self, device: Union[torch.device, str]):
        """
        Moves all recorded statistics to the specified device.
        
        Args:
            device (torch.device): Target device (CPU or GPU).
        """
        for attr in self.attributes:
            self.rolling_means[attr] = self.rolling_means[attr].to(device, non_blocking=True)
            self.rolling_means2[attr] = self.rolling_means2[attr].to(device, non_blocking=True)
            self.means[attr] = self.means[attr].to(device, non_blocking=True)
            self.means2[attr] = self.means2[attr].to(device, non_blocking=True)
            self.stds[attr] = self.stds[attr].to(device, non_blocking=True)


        if isinstance(device, str):
            if 'cuda' in device:
                torch.cuda.synchronize()
        
        elif device.type == "cuda":
            torch.cuda.synchronize()
            

    def normalize(self, batch: Data) -> Data:
        """
        Normalizes a batch using the stored statistics.

        Args:
            batch (Data): The batch to normalize.
        
        Returns:
            Data: The normalized batch.
        """
        assert self.num_rec > 0, 'Cannot normalize without recorded statistics.'
        return Data(**{attr: (batch[attr] - self.means[attr]) / self.stds[attr] for attr in self.attributes},
                    **{k: v for k, v in batch.items() if k not in self.attributes})
    
    def normalize_(self, batch: Data):
        """
        In-place normalization of a batch.

        Args:
            batch (Data): The batch to normalize.
        """
        assert self.num_rec > 0, 'Cannot normalize without recorded statistics.'
        for attr in self.attributes:
            batch[attr] = (batch[attr] - self.means[attr]) / self.stds[attr]

    def de_normalize(self, batch: Data) -> Data:
        """
        Denormalizes a batch using stored statistics.

        Args:
            batch (Data): The batch to denormalize.
        
        Returns:
            Data: The denormalized batch.
        """
        if hasattr(batch, 'preds'):
            unnorm_data =  Data(**{attr: batch[attr] * self.stds[attr] + self.means[attr] for attr in self.attributes},
                         **{k: v for k, v in batch.items() if k not in self.attributes} )
            preds_mean = torch.cat([self.means[f'target_{field}'] for field in self.fields], dim=-1)
            preds_stds = torch.cat([self.stds[f'target_{field}'] for field in self.fields], dim=-1)
            preds = batch.preds * preds_stds + preds_mean
            unnorm_data.preds = preds
            return unnorm_data
        
        return Data(**{attr: batch[attr] * self.stds[attr] + self.means[attr] for attr in self.attributes},
                    **{k: v for k, v in batch.items() if k not in self.attributes})
    
    def de_normalize_(self, batch: Data):
        """
        In-place denormalization of a batch.

        Args:
            batch (Data): The batch to denormalize.
        """
        for attr in self.attributes:
            batch[attr] = batch[attr] * self.stds[attr] + self.means[attr]
        if hasattr(batch, 'preds'):
            preds_mean = torch.cat([self.means[f'target_{field}'] for field in self.fields], dim=-1)
            preds_stds = torch.cat([self.stds[f'target_{field}'] for field in self.fields], dim=-1)
            batch.preds = batch.preds * preds_stds + preds_mean



def warm_up_normalizer(dataset: Dataset, model: nn.Module, n_batches=1000):
    """
    Warms up the normalizer "inside" of model

    Args:
        dataset (Dataset): The full initialized dataset
        model (nn): A model with normalizer as attribute
    """
    dataloader = DataLoader(dataset, batch_size=20, num_workers=max(1, os.cpu_count()//4), shuffle=True)
    for i, batch in enumerate(dataloader):
        model.normalizer.record(batch)
        if i==n_batches-1:
            break
