import torch
import torch.nn.functional as F
import torch.nn as nn

import copy

from torch_geometric.data import Data
from torch_geometric.utils import scatter

from utils import create_node_mask, move_graph_, add_noise_
from normalization import Normalizer
from config import node_types_to_train



class MLP(nn.Module):
    """
    Multi-layer perceptron with 2 hidden layers and ReLU activations.
    
    As specified in the paper, all MLPs have 2 hidden layers with ReLU activations
    and hidden dimension of 128.
    
    Args:
        input_dim (int): Input feature dimension
        output_dim (int): Output feature dimension
        DoLayerNorm (bool): Whether to apply layer normalization to output
        hidden_dim (int, optional): Hidden layer dimension. Defaults to 128
    """
    def __init__(self, input_dim: int, output_dim: int, DoLayerNorm=True, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.do_layer_norm = DoLayerNorm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MLP.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        if self.do_layer_norm:
            x = self.layer_norm(x)
        return x


class Encoder(nn.Module):
    """
    Encodes node and edge features into embedding space.
    
    Uses two separate MLPs to project node and edge features into
    a common embedding dimension.
    
    Args:
        node_feature_dim (int): Input node feature dimension
        edge_feature_dim (int): Input edge feature dimension
        embedding_dim (int, optional): Output embedding dimension. Defaults to 128
    """
    def __init__(self, node_feature_dim: int, edge_feature_dim: int, embedding_dim: int = 128):
        super().__init__()
        self.node_encoder = MLP(node_feature_dim, embedding_dim)
        self.edge_encoder = MLP(edge_feature_dim, embedding_dim)

    def forward(self, node_features: torch.Tensor, edge_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode node and edge features.
        
        Args:
            node_features (torch.Tensor): Input node features
            edge_features (torch.Tensor): Input edge features
            
        Returns:
            tuple: (encoded_nodes, encoded_edges)
        """
        h_nodes = self.node_encoder(node_features)
        h_edges = self.edge_encoder(edge_features)
        return h_nodes, h_edges


class ProcessorBlock(nn.Module):
    """
    Single message passing block in the processor.
    
    Implements edge and node updates using MLPs and message passing.
    This follows the message passing scheme from the MeshGraphNet paper:
    1. Edge update: e'_ij = e_ij + MLP(e_ij, v_i, v_j)
    2. Node update: v'_i = v_i + MLP(v_i, sum_{j∈N(i)} e'_ji)
    
    Args:
        embedding_dim (int, optional): Embedding dimension. Defaults to 128
    """
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.edge_update_mlp = MLP(3 * embedding_dim, embedding_dim)  # 3 because node_i, node_j, edge_ij
        self.node_update_mlp = MLP(2 * embedding_dim, embedding_dim)  # 2 because node_i, aggregated_edges_{*i}

    # FOR SOME REASON, IN THE CODE BY DEEPMIND THE NODE UPDATE TAKES AS INPUT THE AGGREGATED EDGE UPDATE,
    #  NOT THE AGGREGATED UPDATED EDGES (as done in the commented forward block at the end of the file) 
    def forward(self, h_nodes: torch.Tensor, h_edges: torch.Tensor, 
                edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process one message passing step.
        
        Args:
            h_nodes (torch.Tensor): Node embeddings [num_nodes, embedding_dim]
            h_edges (torch.Tensor): Edge embeddings [num_edges, embedding_dim]
            edge_index (torch.Tensor): Graph connectivity as [2, num_edges] tensor
                                    where edge_index[0] = sender nodes, edge_index[1] = receiver nodes
            
        Returns:
            tuple: (updated_nodes, updated_edges)
        """
        # Unpack edge indices for sender (i) and receiver (j) nodes
        sender, receiver = edge_index

        # Edge update: e'_ij = e_ij + MLP(e_ij, v_i, v_j)
        # Concatenate edge features with sender and receiver node features
        edge_inputs = torch.cat([h_edges, h_nodes[sender], h_nodes[receiver]], dim=-1)
        edge_updates = self.edge_update_mlp(edge_inputs)
        
        # Node update: v'_i = v_i + MLP(v_i, sum_{j∈N(i)} e^{update}_ji)
        # First aggregate incoming messages to each node using sum reduction
        # This computes the sum of edge features for each receiving node
        aggregated_edges = scatter(edge_updates, receiver, dim=0, dim_size=h_nodes.size(0), reduce='sum')
        # Concatenate node features with aggregated edge features
        node_inputs = torch.cat([h_nodes, aggregated_edges], dim=-1)
        node_updates = self.node_update_mlp(node_inputs)

        # Residual connections
        h_edges += edge_updates
        h_nodes += node_updates  # Residual connection

        return h_nodes, h_edges


class Processor(nn.Module):
    """
    Multi-layer processor implementing message passing.
    
    Stacks multiple ProcessorBlocks sequentially to perform multiple rounds
    of message passing between nodes.
    
    Args:
        num_layers (int): Number of message passing layers
        embedding_dim (int, optional): Embedding dimension. Defaults to 128
    """
    def __init__(self, num_layers: int = 15, embedding_dim: int = 128):
        super().__init__()
        self.layers = nn.ModuleList([ProcessorBlock(embedding_dim) for _ in range(num_layers)])

    def forward(self, h_nodes: torch.Tensor, h_edges: torch.Tensor, 
                edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process through all layers sequentially.
        
        Args:
            h_nodes (torch.Tensor): Node embeddings
            h_edges (torch.Tensor): Edge embeddings
            edge_index (torch.Tensor): Graph connectivity
            
        Returns:
            tuple: (final_nodes, final_edges)
        """
        for layer in self.layers:
            h_nodes, h_edges = layer(h_nodes, h_edges, edge_index)
        return h_nodes, h_edges


class Decoder(nn.Module):
    """
    Decodes node embeddings to output features.
    
    Uses an MLP without layer normalization for final prediction.
    
    Args:
        embedding_dim (int): Input embedding dimension
        output_dim (int): Output feature dimension
    """
    def __init__(self, embedding_dim: int, output_dim: int):
        super().__init__()
        self.mlp = MLP(embedding_dim, output_dim, DoLayerNorm=False)

    def forward(self, h_nodes: torch.Tensor) -> torch.Tensor:
        """
        Decode node embeddings to output features.
        
        Args:
            h_nodes (torch.Tensor): Node embeddings
            
        Returns:
            torch.Tensor: Output predictions
        """
        return self.mlp(h_nodes)


class MeshGraphNet(nn.Module):
    """
    Implementation of MeshGraphNet architecture.
    
    MeshGraphNet is a graph neural network for mesh-based simulations as described in:
    "Learning Mesh-Based Simulation with Graph Networks" by Pfaff et al.
    
    The network has three main components:
    1. Encoder: Maps raw node/edge features to latent embeddings
    2. Processor: Performs multiple rounds of message passing to propagate information
    3. Decoder: Maps final node embeddings to output predictions
    """

    def __init__(self, metadata, normalizer: Normalizer, sample_graph: Data):
        super().__init__()
        self.metadata = metadata
        node_feature_dim = len(self.metadata['fields']) + sample_graph.node_type_one_hot.shape[1]
        edge_feature_dim = sample_graph.edge_attr.shape[1]
        self.encoder = Encoder(node_feature_dim, edge_feature_dim, self.metadata['embedding_dim'])
        self.processor = Processor(num_layers=self.metadata['num_layers'], embedding_dim=self.metadata['embedding_dim'])
        self.decoder = Decoder(self.metadata['embedding_dim'], len(self.metadata['fields']))
        self.normalizer = normalizer


    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            data (Data): Input graph data containing node fields, type, and edge information
            
        Returns:
            torch.Tensor: Output predictions
        """

        # Record data stats in the normalizer
        self.normalizer.record(data)

        # In the implementation of DeepMind, the noise is added before the normalization
        if self.metadata['noise_when']=='before' and self.training:
            add_noise_(data, self.metadata)

        # Normalizer data
        self.normalizer.normalize_(data)  # In place so that also the data "outside" gets normalized

        # I think the noise should be added after normalization
        if self.metadata['noise_when']=='after' and self.training:
            add_noise_(data, self.metadata)

        # Extract node features from data object based on field names from self.metadata
        node_fields = [data[field] for field in self.metadata['fields']]
        
        # Get one-hot encoded node type
        node_type_one_hot = data.node_type_one_hot
        
        # Get edge features and connectivity
        edge_features = data.edge_attr
        edge_index = data.edge_index

        # Concatenate node features with node type
        # This combines all available node information into a single feature vector
        node_features = torch.cat(node_fields + [node_type_one_hot], dim=1)

        # Process through network
        # 1. Encode: Transform raw features to latent embeddings
        h_nodes, h_edges = self.encoder(node_features, edge_features)
        
        # 2. Process: Apply message passing to propagate information
        h_nodes, h_edges = self.processor(h_nodes, h_edges, edge_index)
        
        # 3. Decode: Transform final node embeddings to output predictions
        output = self.decoder(h_nodes)

        return output
    

    def to(self, device):
        """
        Moves both the model and the normalizer on "device".
        Args:
            device (torch.device or str): device on which to move the data
        """
        self.normalizer.to_device_(device)
        return super().to(device)
    

    def evolve(self, graph0: Data, keep_on_device: bool = False) -> Data:
        """
        Evolve the graph state by one timestep.
        Takes an unnormalized graph as input and outputs the evolved unnormalized graph
        
        This method:
        1. Normalizes the input graph
        2. Applies the model to predict updates
        3. Updates the graph state with predictions
        4. De-normalizes the result
        
        Note: For repeated calls, manual memory management is recommended:
        ```python
        graph = dataset.get_sample(normalize=False)
        for i in range(200):
            prev_graph = graph
            graph = mesh_graph_net.evolve(graph)
            del prev_graph
            torch.cuda.empty_cache()
        ```
        
        Args:
            graph (Data): Input graph (unnormalized)
            keep_on_device (bool): Whether to keep output on same device as input
            
        Returns:
            Data: Updated graph
        """
        # Put model in evaluation mode
        self.eval()

        graph = copy.deepcopy(graph0)  # Protect the graph outside from the in-place normalization of the forward

        # Store original devices
        graph_device = graph[self.metadata['fields'][0]].device
        model_device = next(self.parameters()).device
        
        # If mismatch between graph and model devices, move graph to model device
        if graph_device != model_device:
            graph = graph.to(model_device)

        # Calculate and add predictions as graph attribute
        graph.preds = self(graph) # the forward method already normalizes the graph in place
        graph = self.normalizer.de_normalize(graph)

        #  Update node fields based on predictions
        # Create masks to identify which nodes should be updated
        masks = {field: create_node_mask(graph, self.metadata['loss_on'][field]) 
                for field in self.metadata['fields']}

        # Update each field according to self.metadata configuration
        for dim, field in enumerate(self.metadata['fields']):
            if self.metadata['diff_t0_t1'][field]:
                # Add prediction as delta if field uses differential updates
                graph[field][masks[field]] += graph.preds[masks[field], dim].unsqueeze(-1)
            else:
                # Replace with prediction if field uses absolute updates
                graph[field][masks[field]] = graph.preds[masks[field], dim].unsqueeze(-1)

        # Move graph back to original device if requested
        if not keep_on_device and graph_device != model_device:
            graph = graph.to(graph_device)

        self.train()

        return graph

    def evolve_(self, graph: Data, keep_on_device: bool = False) -> None:
        """
        In-place version of evolve method for memory efficiency.
        
        This performs the same operations as evolve() but modifies the graph in-place
        to avoid creating a new Data object.
        
        Args:
            graph (Data): Input graph to modify (unnormalized)
            keep_on_device (bool): Whether to keep tensors on same device as input
        """
        # Put model in evaluation mode
        self.eval()

        # Store original device
        model_device = next(self.parameters()).device
        graph_device = graph[self.metadata['fields'][0]].device
        
        # Move to model device if needed
        if graph_device != model_device:
            move_graph_(graph, model_device)

        # Calculate and add predictions as graph attribute
        graph.preds = self(graph) # the forward method already normalizes the graph
        self.normalizer.de_normalize_(graph)

        # Update node fields based on predictions
        # Create masks to identify which nodes should be updated
        masks = {field: create_node_mask(graph, self.metadata['loss_on'][field]) 
                for field in self.metadata['fields']}
        
        # Update each field according to self.metadata configuration
        for dim, field in enumerate(self.metadata['fields']):
            if self.metadata['diff_t0_t1'][field]:
                # Add prediction as delta if field uses differential updates
                graph[field][masks[field]] += graph.preds[masks[field], dim].unsqueeze(-1)
            else:
                # Replace with prediction if field uses absolute updates
                graph[field][masks[field]] = graph.preds[masks[field], dim].unsqueeze(-1)

        # Move back to original device if requested
        if not keep_on_device and graph_device != model_device:
            move_graph_(graph, graph_device)






######################## "CORRECTED" VERSION (i.e. with edge residual before node update)  ############################


class MeshGraphNetCorrect(MeshGraphNet):
    def __init__(self, metadata, normalizer: Normalizer, sample_graph: Data):
        super().__init__(metadata, normalizer, sample_graph)

        self.processor = ProcessorCorrect(self.metadata['num_layers'], self.metadata['embedding_dim'])


class ProcessorCorrect(Processor):
    """
    Multi-layer processor implementing message passing with correct residual connection.
    
    Stacks multiple ProcessorBlocks sequentially to perform multiple rounds
    of message passing between nodes.
    
    Args:
        num_layers (int): Number of message passing layers
        embedding_dim (int, optional): Embedding dimension. Defaults to 128
    """
    def __init__(self, num_layers: int = 15, embedding_dim: int = 128):
        super().__init__()
        self.layers = nn.ModuleList([ProcessorBlockCorrect(embedding_dim) for _ in range(num_layers)])


class ProcessorBlockCorrect(nn.Module):
    """
    Single message passing block in the processor.
    
    Implements edge and node updates using MLPs and message passing.
    This follows the message passing scheme from the MeshGraphNet paper:
    1. Edge update: e'_ij = e_ij + MLP(e_ij, v_i, v_j)
    2. Node update: v'_i = v_i + MLP(v_i, sum_{j∈N(i)} e'_ji)
    
    Args:
        embedding_dim (int, optional): Embedding dimension. Defaults to 128
    """
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.edge_update_mlp = MLP(3 * embedding_dim, embedding_dim)  # 3 because node_i, node_j, edge_ij
        self.node_update_mlp = MLP(2 * embedding_dim, embedding_dim)  # 2 because node_i, aggregated_edges_{*i}

    def forward(self, h_nodes: torch.Tensor, h_edges: torch.Tensor, 
                edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Process one message passing step.
        
        Args:
            h_nodes (torch.Tensor): Node embeddings [num_nodes, embedding_dim]
            h_edges (torch.Tensor): Edge embeddings [num_edges, embedding_dim]
            edge_index (torch.Tensor): Graph connectivity as [2, num_edges] tensor
                                        where edge_index[0] = sender nodes, edge_index[1] = receiver nodes
            
        Returns:
            tuple: (updated_nodes, updated_edges)
        """
        # Unpack edge indices for sender (i) and receiver (j) nodes
        sender, receiver = edge_index

        # Edge update: e'_ij = e_ij + MLP(e_ij, v_i, v_j)
        # Concatenate edge features with sender and receiver node features
        edge_inputs = torch.cat([h_edges, h_nodes[sender], h_nodes[receiver]], dim=-1)
        edge_updates = self.edge_update_mlp(edge_inputs)
        h_edges = h_edges + edge_updates  # Residual connection
        
        # Node update: v'_i = v_i + MLP(v_i, sum_{j∈N(i)} e'_ji)
        # First aggregate incoming messages to each node using sum reduction
        # This computes the sum of edge features for each receiving node
        aggregated_edges = scatter(h_edges, receiver, dim=0, dim_size=h_nodes.size(0), reduce='sum')
        # Concatenate node features with aggregated edge features
        node_inputs = torch.cat([h_nodes, aggregated_edges], dim=-1)
        node_updates = self.node_update_mlp(node_inputs)
        h_nodes = h_nodes + node_updates  # Residual connection

        return h_nodes, h_edges
    




############### My idea simlar to Message Passing Neural Networks ######################

"""
m_{ij} = MLP(h_j, e_{ij})
h_i <-- h_i + MLP(h_j, \sum_j m_{ij})
e_{ij} <-- e_{ij} + MLP(e_{ij}, \sum_k e_{jk})
"""

class MyMeshGraphNet(MeshGraphNet):
    def __init__(self, metadata, normalizer: Normalizer, sample_graph: Data):
        super().__init__(metadata, normalizer, sample_graph)

        self.processor = MyProcessor(self.metadata['num_layers'], self.metadata['embedding_dim'])




class MyProcessorBlock(nn.Module):
    """
    My custom message passing block 
    m_{ij} = MLP(h_j, e_{ij})
    h_i <-- MLP(h_j, \sum_j m_{ij})
    e_{ij} <-- MLP(e_{ij}, \sum_{k\neq i} e_{jk})
    """
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.message_mlp = MLP(2*embedding_dim, embedding_dim)
        self.node_mlp = MLP(2*embedding_dim, embedding_dim)
        self.edge_mlp = MLP(2*embedding_dim, embedding_dim)


    def forward(self, h_nodes: torch.Tensor, h_edges: torch.Tensor, 
                edge_index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        senders, receivers = edge_index

        # Compute messages
        message_arg = torch.cat([h_nodes[senders], h_edges], dim=-1)
        messages = self.message_mlp(message_arg)

        # Update nodes
        aggregated_messages = scatter(messages, receivers)
        h_nodes += self.node_mlp(torch.cat([h_nodes, aggregated_messages], dim=-1))

        # Update edges
        # The last edge update is wasted but alas
        aggregated_edges_j = scatter(h_edges, receivers)  # a_g1[j] = \sum_m e_{jk}
        aggregated_edges_i = aggregated_edges_j[senders]  # a_g2[i] = ag1[sender_to_i]
        aggregated_edges = aggregated_edges_i - h_edges 
        h_edges += self.edge_mlp(torch.cat([h_edges, aggregated_edges], dim=-1))

        return h_nodes, h_edges
    

class MyProcessor(Processor):
    def __init__(self, num_layers: int = 15, embedding_dim: int = 128):
        super().__init__()
        self.layers = nn.ModuleList([MyProcessorBlock(embedding_dim) for _ in range(num_layers)])

    

