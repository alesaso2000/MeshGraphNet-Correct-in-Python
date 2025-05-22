import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy.interpolate import griddata
import numpy as np
from matplotlib import path
import os

import matplotlib.pyplot as plt

import pandas as pd

from typing import List, Dict

import torch  
from torch import Tensor

import wandb


@torch.no_grad()
def plot_train_val_loss(train_loss, val_loss, step, show=False, save_path=None):
    x_train = np.linspace(0, step, len(train_loss))
    dx = step//len(val_loss)
    x_val = [i*dx for i in range(1,len(val_loss)+1)]

    fig, (ax1,ax2) = plt.subplots(1,2, figsize=(20,8))

    ax1.plot(x_train, train_loss, label='train loss', color='blue')
    ax1.scatter(x_val, val_loss, label='validation loss', color='red')
    ax1.set_title('Train and Val Loss')
    ax1.set_xlabel('step')
    ax1.set_ylabel('Loss (MSE)')

    ax2.plot(x_train, train_loss, label='train loss', color='blue')
    ax2.scatter(x_val, val_loss, label='validation loss', color='red')
    ax2.set_title('Train and Val Loss loglog')
    ax2.set_xlabel('step')
    ax2.set_ylabel('Loss (MSE)')
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    if save_path:
        plt.savefig(os.path.join(save_path, 'Loss.pdf'))

    if show:
        plt.show()

    plt.close()


def get_body_coords(graph):
    y_min, y_max = graph.pos[:,1].min().item(), graph.pos[:,1].max().item()
    dy = y_max - y_min
    wall_mask = (graph.node_type==6).squeeze()
    body_mask = torch.logical_and(graph.pos[:,1]>y_min+dy*.01, graph.pos[:,1]<y_max-dy*.01)
    mask = wall_mask * body_mask
    body_coords = graph.pos[mask]
    return body_coords

def get_ordered_body_coords(graph):
    body_coords = get_body_coords(graph)
    center = body_coords.mean(dim=0)
    shifted_cody_coords = body_coords - center
    angles = torch.atan2(shifted_cody_coords[:,1], shifted_cody_coords[:,0])
    return body_coords[torch.argsort(angles)]

@torch.no_grad()
def plot_meshgraphnet(graph, model, which_fields: List[str], normalizer, grid_resolution=1000, save_to_disk=False, save_path='./', 
                      show=False, body_color='white', wandb_log=True, train_or_val='train'):

    graph1 = model.evolve(graph, normalizer)

    graph, graph1 = graph.cpu(), graph1.cpu()

    body_coords = get_body_coords(graph)   # change to body_coords = graph.pos[graph.node_type==2]

    to_df = {'X': graph.pos[:,0], 'Y': graph.pos[:,1]}
    for field in model.metadata['fields']:
        to_df[f'{field}_init'] = graph[field].squeeze()
        to_df[f'{field}_pred'] = graph1[field].squeeze()
        to_df[f'{field}_true'] = graph[f'y_{field}'].squeeze()

    df = pd.DataFrame(to_df)
    file_name = os.path.join(save_path, f'{train_or_val}_{which_fields}.pdf')
    
    plot_field(df, variable=which_fields, body_coords=body_coords.numpy(), grid_resolution=grid_resolution, save_to_disk=save_to_disk, 
               file_name=file_name, show=show, body_color=body_color, wandb_log=wandb_log)



def plot_field(df, variable='U', method='cubic', x_range=None, y_range=None, grid_resolution=100, body_coords=None,
               body_color='red', save_to_disk=False, file_name='./forgot_file_name.pdf', show=True, wandb_log=False):
    """
    Function to visualize a velocity field from a CFD dataframe using interpolation.
    Zeros out the velocity inside a specified body.

    Parameters:
    - df: Pandas DataFrame containing the columns 'X', 'Y', and the variable(s) to plot.
    - variable: Name of the variable to plot (default 'U'). Can also be a list of variables.
    - method: Interpolation method (default 'cubic'). Options: 'linear', 'cubic', 'nearest'.
    - x_range: Tuple defining the range of X values (min, max). Default is based on data.
    - y_range: Tuple defining the range of Y values (min, max). Default is based on data.
    - grid_resolution: The number of grid points along each axis (default 100).
    - body_coords: NumPy array of shape (N, 2) with the coordinates of the body to plot.
    - body_color: Color for the body patch (default 'red').
    - file_path: Path to save the file (default './').
    """


    # Determine the range for X and Y
    if x_range is None:
        x_range = (df['X'].min(), df['X'].max())
    if y_range is None:
        y_range = (df['Y'].min(), df['Y'].max())

    # Create a meshgrid for X, Y based on the specified range and resolution
    x_vals = np.linspace(x_range[0], x_range[1], grid_resolution)
    y_vals = np.linspace(y_range[0], y_range[1], grid_resolution)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Ensure variable is a list, even if a single string is provided
    if isinstance(variable, str):
        variable = [variable]

    num_vars = len(variable)

    # Calculate global min and max for consistent coloring
    cmax = df[variable].values.ravel().max()
    cmin = df[variable].values.ravel().min()

    # Create a figure with subplots
    fig, axes = plt.subplots(1, num_vars, figsize=(10 * num_vars, 8))

    # Ensure axes is a list for consistent handling
    if num_vars == 1:
        axes = [axes]

    for i, var in enumerate(variable):
        ax = axes[i]

        # Interpolate the variable values across the grid
        Z = griddata((df['X'], df['Y']), df[var], (X, Y), method=method)

        # If body_coords is provided, zero out values inside the body
        if body_coords is not None:
            xv, yv = body_coords[:, 0], body_coords[:, 1]
            # Calculate the inside_body mask for the flattened grid
            inside_body = inpolygon(X, Y, xv, yv).reshape(X.shape)
            Z[inside_body] = 0

        # Plot the velocity field using imshow
        cax = ax.imshow(Z, extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
                        origin='lower', aspect='auto', cmap='jet', vmax=cmax, vmin=cmin)

        # Add a unique colorbar for each axis
        fig.colorbar(cax, ax=ax, label=f'{var}')

        # Title and axis labels
        ax.set_title(f'{var} Field (Interpolated, method={method})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')

        # Plot the body if coordinates are provided
        if body_coords is not None:
            body_patch = Polygon(body_coords, closed=True, color=body_color)
            ax.add_patch(body_patch)

        # Set axis equal to maintain aspect ratio
        ax.set_aspect('equal', adjustable='box')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure to disk if the flag is set
    if save_to_disk:
        plt.savefig(file_name)
        # print(f'Plot saved to {file_name}')

    if wandb_log:
        wandb.log({f'{variable} plot': wandb.Image(plt)})

    # Show the plot
    if show:
        plt.show()

    plt.close()




def inpolygon(xq, yq, xv, yv):
    """
    Efficiently determine whether the query points (xq, yq) are inside the polygon defined by (xv, yv).

    Parameters:
    - xq, yq: Coordinates of the query points (can be any shape).
    - xv, yv: Coordinates of the vertices of the polygon (1D arrays).

    Returns:
    - Boolean array with the same shape as (xq, yq), indicating whether each point is inside the polygon.
    """

    # Ensure inputs are numpy arrays and flatten the query points
    xq = np.asarray(xq).ravel()
    yq = np.asarray(yq).ravel()

    # Convert polygon vertices into a list of tuples
    polygon = np.column_stack((xv, yv))

    # Create the Path object for the polygon
    p = path.Path(polygon)

    # Check containment and reshape the result to match the original input shape
    contained = p.contains_points(np.column_stack((xq, yq)))

    return contained
