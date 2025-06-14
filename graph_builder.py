import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
from Tower import Tower
from torch_geometric.data import Data

def encode_generation_onehot(generation):
    onehot = [0, 0, 0, 0]
    if 0 <= generation < 4:
        onehot[generation] = 1
    return onehot

def build_tower_graph(towers, region_width=1000, region_height=1000, k_neighbors=10):
    node_features = []
    coords = []

    for tower in towers:
        x_norm = tower.x / region_width
        y_norm = tower.y / region_height
        tower.sigma = 5
        cost = np.log1p(tower.cost)
        radius = tower.calculate_radius() / 1000.0
        gen_onehot = encode_generation_onehot(tower.generation)
        
        

        node_features.append([x_norm, y_norm, cost, radius] + gen_onehot)
        coords.append([tower.x, tower.y])

    x_tensor = torch.tensor(node_features, dtype=torch.float)
    coords_np = np.array(coords)

    # k-NN edge construction
    nbrs = NearestNeighbors(n_neighbors=k_neighbors).fit(coords_np)
    _, indices = nbrs.kneighbors(coords_np)

    edge_index = []
    for src, neighbors in enumerate(indices):
        for dst in neighbors:
            edge_index.append([src, dst])
    edge_index = torch.tensor(edge_index, dtype=torch.long).T.contiguous()

    return Data(x=x_tensor, edge_index=edge_index)
