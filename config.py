import argparse
import ast
from utils import get_device

def parse_args():

    parser = argparse.ArgumentParser(description="Parse metadata configuration.")
    
    # Required arguments
    parser.add_argument('--dataset_name', type=str, required=True,
                        choices=['cylinder_flow', 'airfoil'],
                        help="The name of the dataset (one of 'cylinder_flow', 'airfoil').")
    
    parser.add_argument('--fields', type=str, required=True,
                        help="Comma-separated list of fields to use.")
    
    parser.add_argument('--diff_t0_t1', type=str, required=True,
                        help="Dictionary specifying whether to predict differences or direct values.")
    
    parser.add_argument('--run', type=str, required=True,
                        help="The number (id) or name of the training run")
    
    parser.add_argument('--noise_when', type=str, required=True, choices=['before', 'after'])
    
    # Optional arguments with explicit values
    parser.add_argument('--device', type=str, default='most_free', choices=['cuda', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3', 'cpu', 'most_free'])
    parser.add_argument('--warmup_batches', type=int, default=1_000)   # Change here
    parser.add_argument('--num_layers', type=int, default=15)
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--initial_lr', type=float, default=0.001)  # Change here
    parser.add_argument('--decay_rate', type=float, default=10**(-1/15))
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--val_batch_size', type=int, default=20)
    parser.add_argument('--dataloader_seed', type=int, default=23)
    parser.add_argument('--model_seed', type=int, default=123)
    parser.add_argument('--model', type=str, choices=['paper', 'code', 'mine'], default='paper')
    parser.add_argument('--miscellaneous', type=str, default=None)
    parser.add_argument('--noise_mask', type=str, default="{'velocity_x': [0,5], 'velocity_y': [0,5], 'pressure': 'all', 'density': 'all'}",
                        help="Dictionary {field: [node_types] or all or node_type} specifying which node types to add noise to.")
    parser.add_argument('--loss_on', type=str, default="{'velocity_x': [0,5], 'velocity_y': [0,5], 'pressure': 'all', 'density': 'all'}",
                        help="Dictionary {field: [node_types] or all or node_type} specifying which node types to apply loss to.")
    parser.add_argument('--noise_std', type=str, default="{'velocity_x': 2e-2, 'velocity_y': 2e-2, 'pressure': 0, 'density': 1e-2}",  # For airfoil change velocity* to 1e-1
                        help="Dictionary {field: float} specifying noise standard deviations.")

    args = parser.parse_args()
    
    # Convert string representations of dicts/lists to actual Python objects
    parsed_metadata = {key: ast.literal_eval(value) if key in ['fields', 'loss_on', 'noise_mask', 'noise_std', 'diff_t0_t1'] else value for key, value in vars(args).items()}
    
    if parsed_metadata['dataset_name']=='cylinder_flow':
        parsed_metadata['object_node_type'] = 6
    elif parsed_metadata['dataset_name']=='airfoil':
        parsed_metadata['object_node_type'] = 2
    else:
        print("I can't recognize the name of the dataset")

    parsed_metadata['device'] = get_device(parsed_metadata)

    return parsed_metadata


def create_saving_filename(metadata):
    return f"./savings/{metadata['dataset_name']}/run{metadata['run']}"


node_types_to_train = {'velocity_x': [0,5], 'velocity_y': [0,5], 'pressure': 'all', 'density': 'all'}


metadata = {}

def validate_metadata(meta):
    valid_fields = {
        'cylinder_flow': {'velocity_x', 'velocity_y', 'pressure'},
        'airfoil': {'velocity_x', 'velocity_y', 'pressure', 'density'}
    }
    
    # Validate fields
    if not set(meta['fields']).issubset(valid_fields[meta['dataset_name']]):
        raise ValueError(f"Invalid fields {meta['fields']} for dataset {meta['dataset_name']}. Must be a subset of {valid_fields[meta['dataset_name']]}.")

    
    # Validate noise_std
    if set(meta['noise_std'].keys()) != set(meta['noise_mask'].keys()):
        raise ValueError(f"Keys in noise_std {meta['noise_std'].keys()} must match noise_mask {meta['noise_mask'].keys()}.")
    
    # Validate diff_t0_t1
    if set(meta['diff_t0_t1'].keys()) != set(meta['fields']):
        raise ValueError(f"Keys in diff_t0_t1 {meta['diff_t0_t1'].keys()} must match fields {meta['fields']}.")
    
