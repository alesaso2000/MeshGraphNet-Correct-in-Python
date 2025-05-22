import os, json, functools, copy, re, random
from contextlib import contextmanager
from typing import Dict, Union, List

import torch
import torch.nn.functional as F
from torch_geometric.data import Dataset, download_url, Data
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
import numpy as np

from tfrecord.torch.dataset import TFRecordDataset

class TransposeFace:
    def __call__(self, data):
        data['face'] = data['face'].T  # Transpose face indices
        return data
    
class DecomposeVelocity:
    def __call__(self, data):
        data['velocity_x'] = data['velocity'][:,0].unsqueeze(-1)
        data['velocity_y'] = data['velocity'][:,1].unsqueeze(-1)
        del data.velocity
        return data  

class ToLong:
    def __call__(self, data):
        data['edge_index'] = data['edge_index'].long()
        return data
    
class ToOneHot:
    def __call__(self, data):
        unique = torch.unique(data['node_type'])
        # mapping = {u: i for i,u in enumerate(unique)}
        # node_type_mapped = torch.tensor([mapping[val.item()] for val in data['node_type']])
        node_type_mapped = torch.searchsorted(unique, data['node_type'])  # Vectorized mapping
        data['node_type_one_hot'] = F.one_hot(node_type_mapped.long(), num_classes=len(unique)).squeeze(1)
        return data

pre_transforms = T.Compose([
    TransposeFace(),
    T.FaceToEdge(),     
    T.Cartesian(norm=False),  # False because I will normalize it on the fly during training
    T.Distance(norm=False),   # """"
    DecomposeVelocity(),
    ToLong(),
    ToOneHot()
]) 


class MyOwnDataset(Dataset):

    def __init__(self, metadata, root='./data', split='train', delta_t=1,
                 pre_transform = pre_transforms, transform=None, pre_filter=None,
                 num_timesteps=598, num_geometries=99):
        """
        Args:
            root (str): the directory in which the raw and processed data will be downloaded
            split (Union['train', 'valid', 'test']): self explanatory
            pre_transforms: self explanatory
            transform: 
            pre_filter: the downloaded data to skip saving
            num_timesteps (int): the number of timesteps
        """                 
        self.metadata = metadata
        self.delta_t = delta_t
        # Create a root directory for every dataset
        root = os.path.join(root, self.metadata['dataset_name']) 

        self.split = split
        root = os.path.join(root, split)
        self.root = root
        print('root:', os.path.abspath(root))

        self.dictionary = {'mesh_pos': 'pos', 'cells': 'face'}  # Some fields are stored with a "weird name" in the .tfrecord file, so I am changing them

        # Either read or initialize with best guess the number of geometries and time steps 
        # The super().__init__() will read the files present in the root directory and if they match those given by self.processed_file_names 
        # (which depends on self.num_geometries) it just moves on, otherwise it triggers the process method (so it is important that they are a good guess)
        self.num_geometries, self.num_timesteps = self.read_num_geometries_timesteps(num_geometries, num_timesteps)

        # this at the end since it triggers both download and process
        # super().__init__(root, transform, pre_transform, pre_filter)
        super().__init__(root, transform, pre_transform, pre_filter)


    
    def read_num_geometries_timesteps(self, geom_guess: int, time_guess: int): 
        """
        Read the number of geometries and timesteps or return guess
        """
        file_path = os.path.join(self.root, 'description.txt')
        if not os.path.exists(file_path):
            return geom_guess, time_guess
        
        pattern = re.compile(rf"^{self.split}: num_geometries (\d+), num_timesteps (\d+)", re.MULTILINE)

        with open(file_path, 'r') as f:
            content = f.read()

        match = pattern.search(content)
        if match:
            num_geometries = int(match.group(1))
            num_timesteps = int(match.group(2))
            return num_geometries, num_timesteps
        else:
            return geom_guess, time_guess



    @property
    def raw_file_names(self):
        return ["meta.json", f"{self.split}.tfrecord"]

    @property
    def processed_file_names(self):
        return [f'geometry_{geom_idx}_t_{t}.pt' for t in range(self.num_timesteps) for geom_idx in range(self.num_geometries)]


    def download(self):
        print('downloading raw dataset')
        base_url = f"https://storage.googleapis.com/dm-meshgraphnets/{self.metadata['dataset_name']}/"
        files = ["meta.json", f"{self.split}.tfrecord"]

        for file in files:
            url = base_url + file
            download_url(url, self.raw_dir, filename=file)


    def process(self):
        print('processing dataset')
        def _parse(record, meta):
            """Parses a trajectory from TFRecord."""
            out = {}
            for key, field in meta['features'].items():
                data = np.frombuffer(record[key], dtype=np.dtype(field['dtype']))
                data = data.reshape(field['shape'])
                
                if field['type'] == 'static':
                    data = np.tile(data, (meta['trajectory_length'], 1, 1))
                elif field['type'] == 'dynamic_varlen':
                    length = np.frombuffer(record['length_' + key], dtype=np.int32)
                    data = [data[i : i + l] for i, l in enumerate(length)]
                elif field['type'] != 'dynamic':
                    raise ValueError('Invalid data format')
                
                out[key] = data
            
            return out

        def load_dataset(path, split):
            """Load dataset."""
            with open(os.path.join(path, 'meta.json'), 'r') as fp:
                meta = json.load(fp)
            
            feature_description = {k: "byte" for k in meta['field_names']}
            dataset = TFRecordDataset(os.path.join(path, f"{split}.tfrecord"), index_path=None, description=feature_description)
            
            parsed_dataset = map(functools.partial(_parse, meta=meta), dataset)
            
            return parsed_dataset, meta


        def timestep_tensor(field_trajectory, t):
            field_trajectory = field_trajectory  #.numpy()
            field_timestep = field_trajectory[t]
            # return torch.from_numpy(field_timestep)
            return torch.from_numpy(field_timestep.copy())
        
        ds, meta = load_dataset(self.raw_dir, self.split)
        for geom_idx, geom in enumerate(ds):
            for time in range(meta['trajectory_length']):
                graph = Data(**{self.dictionary.get(key,key): timestep_tensor(geom[key], time) 
                                for key in geom.keys()})
                if self.pre_transform is not None:
                    graph = self.pre_transform(graph)
                if self.pre_filter is not None and not self.pre_filter(graph):
                    continue  # Skip saving this graph if it doesn't pass the filter (I will probably never use it)
                torch.save(graph, os.path.join(self.processed_dir, f'geometry_{geom_idx}_t_{time}.pt'))

        # Write the number of geometries 
        self.num_geometries = geom_idx + 1
        self.num_timesteps = time + 1
        with open(os.path.join(self.root, 'description.txt'), 'a+') as f:
            f.write(f'{self.split}: num_geometries {geom_idx+1}, num_timesteps {time+1} \n')      

    # For how I have defined processed_file_name, the last timesteps should be at the end of the list so by
    # giving the dataset (hence the dataloader) an appropiately smaller lenght, the timesteps for which I don't have a 
    # target should not compare at all among the idx in the get method called by the dataloader
    def len(self):
        return len(self.processed_file_names) - self.num_geometries * self.delta_t
    

    def extract_geom_idx_time(self, filename: str):
        """
        Extracts geom_idx and time from the filename, which is defined as f'geometry_{geom_idx}_t_{t}.pt'
        """
        parts = filename[:-3].split('_')  # Remove ".pt" and split by "_"
        geom_idx = int(parts[1])  # Extract geom_idx
        time = int(parts[3])  # Extract time
        return geom_idx, time


    def add_target(self, data_t0, data_t1):
        """
        Adds fields of data_t1 as target of data_t0 (either directly the field or the difference as indicated by self.metadata['to_diff'])

        Args:
            data_t0 (Data): graph with data at time t0
            data_t1 (Data): graph with data at time t0 + self.delta_t (i.e. the time one would like to predict)
        """
        for field in self.metadata['fields']:  # add the targets only to the fields I am actually using in this run (that are specified in the self.metadata)
            data_t0[f'target_{field}'] = data_t1[field]
        for field, do_diff in self.metadata['diff_t0_t1'].items():
            if do_diff:
                data_t0[f'target_{field}'] -= data_t0[field] 
        return data_t0
    
    def get(self, idx):
        """
        Gets idx_th graph from the processed files and adds its target. Is implicitly called by the dataloader

        Args:
            idx: index of the graph to get as stored in the self.processed_file_names list
        """
        filename = self.processed_file_names[idx]
        geom_idx, time = self.extract_geom_idx_time(filename)  
        filename_dt = f"geometry_{geom_idx}_t_{time+self.delta_t}.pt"
        data_t0 = torch.load(os.path.join(self.processed_dir, filename), weights_only=False)
        data_t1 = torch.load(os.path.join(self.processed_dir, filename_dt), weights_only=False)
        data = self.add_target(data_t0, data_t1)
        return data
    
    def get_without_target(self, idx):
        """
        Gets the idx_th graph from the processed files

        Args:
            idx: index of the graph to get as stored in the self.processed_file_names list
        """
        filename = self.processed_file_names[idx]
        return torch.load(os.path.join(self.processed_dir, filename), weights_only=False)


    def _get_graph(self, which, verbose=False, with_target=True):
        if which is None:
            raise ValueError("which cannot be None, specify (geom_idx, time_idx).")

        geom_idx, time = which
        idx = time * self.num_geometries + geom_idx  

        if verbose:
            print(f'Extracting geometry {geom_idx}, time {time}. Adding target? {with_target}')
        
        return self.get(idx) if with_target else self.get_without_target(idx)

    def get_graph(self, which=None, delta_t=None, verbose=False, with_target=True):
        """
        Returns a graph.

        Args:
            which (Tuple[int, int] or None): Specifies which graph to get (geometry, time). If None, an error is raised.
            delta_t (int or None): If specified, temporarily overrides `self.delta_t` before retrieving the graph.
            verbose (bool): If True, prints information about the selected graph.
            with_target (bool): If True, includes target attributes.

        Returns:
            Graph object.
        """
        if which is None:
            which = (random.randint(0,self.num_geometries-1), 
                        random.randint(0,self.num_timesteps-1-(self.delta_t if delta_t is None else delta_t)))
            
        if delta_t is not None:
            original_delta_t = self.delta_t
            self.delta_t = delta_t
            graph = self._get_graph(which, verbose, with_target=True)
            self.delta_t = original_delta_t
            return graph

        return self._get_graph(which, verbose, with_target)


# Run this before training on a purely cpu node as it doesn't use gpu to save gpu time (change dataset_name)

# import time

# if __name__ == '__main__':
#     t0 = time.time()
#     dataset = MyOwnDataset(split='train')
#     print('to download and process the dataset it took', time.time()-t0)
#     val_dataset = MyOwnDataset(split='valid')
#     test_dataset = MyOwnDataset(split='test')

#     # Benchmark dataloader speed
#     dataloader = DataLoader(dataset, batch_size=20, shuffle=True, num_workers=max(1,os.cpu_count()//4))
#     t0 = time.time()
#     times = []
#     for i, batch in enumerate(dataloader):
#         if i%10 == 0:
#             times.append(time.time()-t0)
#             print(times[-1])
#             t0 = time.time()
#         if i == 100:
#             break
#     print('10 batches take', np.mean(times), '+-', np.std(times))