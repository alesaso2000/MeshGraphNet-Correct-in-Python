import os, yaml
import numpy as np
import torch
import wandb
from torch_geometric.loader import DataLoader

from models import MeshGraphNet, MeshGraphNetCorrect, MyMeshGraphNet
from dataset_definition import MyOwnDataset
from normalization import Normalizer, warm_up_normalizer
from train_functions import define_optimizer_scheduler, training_step, get_val_loss, plot_losses
from utils import create_node_mask, get_device
from config import node_types_to_train, parse_args, create_saving_filename


def setup_experiment(metadata) -> tuple:
    """
    Set up the entire experiment environment including datasets, model, and logging.

    Args:
        metadata (dict): Configuration dictionary containing experiment parameters

    Returns:
        tuple: (model, dataloader, val_dataloader, optimizer, scheduler)
    """
    # Initialize Weights & Biases
    wandb.init(project=f'reproduce_meshgraphnet_{metadata["dataset_name"]}_pytorch', config=metadata, name=metadata['run'])

    # Create savings directory
    general_savings_dir = f"./savings/{metadata['dataset_name']}/{metadata['run']}"
    os.makedirs(general_savings_dir, exist_ok=False)  # CHANGE TO FALSE!!!!!!!!!!

    # Load datasets
    print('loading_datasets')
    dataset = MyOwnDataset(metadata, split='train')
    val_dataset = MyOwnDataset(metadata, split='valid')
    print('dataset loaded')
    
    # Get sample graph for model initialization
    sample_graph = dataset.get(0)
    print(sample_graph, sample_graph.node_type.unique())

    # Initialize model
    torch.manual_seed(metadata['model_seed'])
    if metadata['model'] == 'code':
        model = MeshGraphNet(metadata, Normalizer(metadata['fields']), sample_graph)
    elif metadata['model'] == 'paper':
        model = MeshGraphNetCorrect(metadata, Normalizer(metadata['fields']), sample_graph)
    elif metadata['model'] == 'mine':
        model = MyMeshGraphNet(metadata, Normalizer(metadata['fields']), sample_graph)
    else:
        raise ValueError(f"Invalid model type: {metadata['model']}. Must be 'code' or 'paper'.")

    # Save metadata to quickly access the hyperparameters
    with open(os.path.join(general_savings_dir, 'metadata.yaml'), 'w') as f:
        yaml.dump(metadata, f, default_flow_style=False, sort_keys= False)

    # Warm up normalizer
    print('warming up normalizer')
    warm_up_normalizer(dataset, model, n_batches=metadata['warmup_batches'])
    print('normalizer warmed up')

    # Move model to device and compile
    print('defining and compiling model')
    model = model.to(metadata['device'])
    # model = torch.compile(model)
    # # Perform initial forward pass to actually compile
    # sample_graph = sample_graph.to(metadata['device'])
    # model(sample_graph)
    print('model compiled')

    # Define optimizer and scheduler
    optimizer, scheduler = define_optimizer_scheduler(
        model, 
        initial_lr=metadata['initial_lr'], 
        decay_rate=metadata['decay_rate']
    )

    print('keys to exclude:', [
            key for key in sample_graph.keys() 
            if not any(item in key for item in metadata['fields'] + 
                       ['edge_attr', 'node_type_one_hot', 'node_type', 'edge_index'])
        ])

    # Create dataloaders
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=metadata['batch_size'],
        shuffle=True,
        exclude_keys=[
            key for key in sample_graph.keys() 
            if not any(item in key for item in metadata['fields'] + 
                       ['edge_attr', 'node_type_one_hot', 'node_type', 'edge_index'])
        ],
        num_workers=max(1, os.cpu_count()//4),
        pin_memory=True,
        prefetch_factor=2,
        generator = torch.Generator().manual_seed(metadata['dataloader_seed'])
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=metadata['val_batch_size'],
        exclude_keys=[
            key for key in sample_graph.keys() 
            if not any(item in key for item in metadata['fields'] + 
                       ['edge_attr', 'node_type_one_hot', 'node_type', 'edge_index'])
        ],
        num_workers=max(1, os.cpu_count()//4),
        pin_memory=True,
        prefetch_factor=2
    )

    return model, dataloader, val_dataloader, optimizer, scheduler, general_savings_dir

def train_model(
    model, 
    dataloader, 
    val_dataloader, 
    optimizer, 
    scheduler, 
    savings_dir,
    metadata
    ):
    """
    Main training loop for the MeshGraphNet model.

    Args:
        model (torch.nn.Module): Neural network model
        dataloader (DataLoader): Training data loader
        val_dataloader (DataLoader): Validation data loader
        optimizer (torch.optim.Optimizer): Model optimizer
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler
        savings_dir (str): Directory to save model checkpoints
    """
    print('starting training')
    record_every = 1_000  
    avgd_train_losses = {field: [] for field in metadata['fields']}
    avgd_train_losses['total'] = []
    train_losses = []
    val_losses = []

    for epoch in range(metadata['n_epochs']):
        save_dir = os.path.join(savings_dir, f'epoch{epoch}')
        os.makedirs(save_dir, exist_ok=True)

        model.train()
        for batch_idx, batch in enumerate(dataloader, start=1):
            batch = batch.to(metadata['device'])
            masks = {field: create_node_mask(batch, node_types_to_train[field]) for field in metadata['fields']}
            
            losses = training_step(model, batch, masks, optimizer)

            for key in losses.keys():
                avgd_train_losses[key].append(losses[key])

            if batch_idx % record_every == 0:
                avgd_train_losses = {key: np.mean(avgd_train_losses[key]) for key in avgd_train_losses.keys()}
                wandb.log(avgd_train_losses)
                avgd_train_losses = {key: [] for key in avgd_train_losses.keys()}

        # Validation 
        model.eval()
        val_loss = get_val_loss(model, val_dataloader)
        val_losses.append(val_loss)

        # Plot and save
        # plot_losses(train_losses, val_losses, metadata, show=False, path=save_dir)

        # Save model checkpoint
        checkpoint = {
            'model_state_dict': model.state_dict(), 
            'normalizer': model.normalizer, 
            'optimizer_state_dict': optimizer.state_dict(), 
            'scheduler_state_dict': scheduler.state_dict(),
            'metadata': metadata, 
            'train_losses': train_losses, 
            'val_losses': val_losses
        }
        torch.save(checkpoint, os.path.join(save_dir, 'model_and_losses.pt'))
        scheduler.step()

def main():
    """
    Main entry point for the MeshGraphNet training script.
    Parses configuration and runs the training process.
    """

    # Parse configuration (to make it "universal")
    metadata = parse_args()
    print('I am using', metadata['device'], 'as the device')

    # Setup experiment
    model, dataloader, val_dataloader, optimizer, scheduler, savings_dir = setup_experiment(metadata)

    # Train model
    train_model(
        model, 
        dataloader, 
        val_dataloader, 
        optimizer, 
        scheduler, 
        savings_dir,
        metadata
        )


if __name__ == '__main__':
    main()


