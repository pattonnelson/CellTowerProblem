import torch
from torch_geometric.data import Data, Dataset
import json
import numpy as np
from sklearn.neighbors import NearestNeighbors
from Tower import Tower
from torch_geometric.loader import DataLoader

class TowerGraphDataset(Dataset):
    def __init__(self, json_path, sigma_func=5):
        super().__init__()
        self.sigma_func = sigma_func
        with open(json_path, 'r') as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        towers = sample['towers']  # List of (x, y, gen)
        deadzone = sample['deadzone']
        cost = sample['cost']

        node_features = []
        coords = []

        for x, y, gen in towers:
            tower = Tower(x, y, gen)
            tower.sigma = self.sigma_func
            radius = tower.calculate_radius()
            tower_cost = tower.cost  # Don't recompute
            
            radius /= 1000.0
            log_cost = np.log1p(tower_cost)
            

            # Normalize coordinates (assuming region is 1000x1000)
            x_norm = x / 1000
            y_norm = y / 1000

            # One-hot encode generation
            gen_onehot = [0, 0, 0, 0]
            if 0 <= gen < 4:
                gen_onehot[gen] = 1

            # 8D feature: [x_norm, y_norm, cost, gen0, gen1, gen2, gen3]
            node_features.append([x_norm, y_norm, log_cost, radius] + gen_onehot)
            coords.append([x, y])

        x = torch.tensor(node_features, dtype=torch.float)
        # print("Feature vector:", [x_norm, y_norm, tower_cost] + gen_onehot)
        # print("Feature length:", len([x_norm, y_norm, tower_cost] + gen_onehot))
        # print(f"gen={gen}, onehot={gen_onehot}")


        # k-NN edges (undirected)
        coords = np.array(coords)
        nbrs = NearestNeighbors(n_neighbors=10).fit(coords)
        _, indices = nbrs.kneighbors(coords)

        edge_index = []
        for src, neighbors in enumerate(indices):
            for dst in neighbors:
                edge_index.append([src, dst])
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # shape [2, num_edges]

        y = torch.tensor([[np.log1p(deadzone), np.log1p(cost)]], dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, y=y)



dataset = TowerGraphDataset("tower_training_data.json", sigma_func=5)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

