import os, pickle
from typing import List, Dict

import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from plot_utils import plot_field
import pandas as pd

import copy

from models import MeshGraphNet, MeshGraphNetCorrect, MyMeshGraphNet
from dataset_definition import MyOwnDataset
from utils import create_node_mask, get_freest_gpu
from config import node_types_to_train
from plot_utils import get_ordered_body_coords

device = get_freest_gpu()

def do_rollout(model, dataset, geom=50):
    graph = dataset.get_graph(which=(geom,0)).to(device)
    model = model.to(device)


    true = {field: [graph[field].clone().cpu()] for field in model.metadata['fields']}
    preds = {field: [graph[field].clone().cpu()] for field in model.metadata['fields']}

    with torch.no_grad():
        for t in range(1,dataset.num_timesteps-1):
            graph = graph.to('cuda:2')
            model.evolve_(graph)
            graph = graph.to('cpu')
            for field in model.metadata['fields']:
                preds[field].append(graph[field].clone().cpu())
            true_graph = dataset.get_graph(which=(geom,t))
            for field in model.metadata['fields']:
                true[field].append(true_graph[field].clone())
    return true, preds, graph.cpu()

def get_rollout(epoch, model_type, integration_type, file, geom=50, dataset='valid'):
    if file is None:
        saved = torch.load(f'savings/cylinder_flow/{integration_type}_{model_type}/epoch{epoch}/model_and_losses.pt', weights_only=False)
    else:
        saved = torch.load(file, weights_only=False)
    metadata = saved['metadata']
    if file is None:
        metadata['noise_when'] = 'before'
    else:
        metadata['noise_when'] = 'after'
    dataset =  MyOwnDataset(metadata, split=dataset)
    sample_graph = dataset.get(0)
    model = MeshGraphNet(metadata, saved['normalizer'], sample_graph) if model_type == 'code' else MeshGraphNetCorrect(metadata, saved['normalizer'], sample_graph) if model_type == 'paper' else MyMeshGraphNet(metadata, saved['normalizer'], sample_graph) if model_type == 'mine' else None
    if not model:
        raise ValueError(f"Invalid model type: {saved['model']}. Must be 'code' 'paper' or 'mine'.")
    model.load_state_dict(saved['model_state_dict'])
    return do_rollout(model, dataset, geom)


def plot_all(true, preds: Dict, graph, field, folder_name):
    
    for t in range(len(true['velocity_x'])):
        print(t)
        to_df = {'X': graph.pos[:,0], 'Y': graph.pos[:,1], f'{field}_true':true[field][t].squeeze(), f'{field}_pred': preds[field][t].squeeze()}
        df = pd.DataFrame(to_df)
        plot_field(df, variable=[f'{field}_true', f'{field}_pred'], save_to_disk=True, 
                   file_name=os.path.join(folder_name, f't={t}.png'), grid_resolution=1000,
                   body_coords=get_ordered_body_coords(graph), body_color='black')
        


geom_idx = 50
model_type = 'paper'
epoch = 15
when = 'before'
diff = 'diff'
file = f'savings/cylinder_flow/{when}_{model_type}_{diff}/epoch{epoch}/model_and_losses.pt'
true, preds, graph = get_rollout(epoch, model_type, diff, file, geom_idx)

field = 'velocity_x'
folder_name = f'./images_damiano/geom_idx{geom_idx}/{field}/'
os.makedirs(folder_name, exist_ok=True)
plot_all(true, preds, graph, field, folder_name)