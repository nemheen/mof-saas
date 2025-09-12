# model_definition.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import CGConv, global_mean_pool
from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from torch_geometric.data import Data

# class CGCNN(nn.Module):
#     def __init__(self, node_dim=1, edge_dim=1, hidden_dim=64, output_dim=1):
#         super().__init__()
#         self.conv1 = CGConv(channels=node_dim, dim=edge_dim)
#         self.conv2 = CGConv(channels=node_dim, dim=edge_dim)
#         self.fc1 = nn.Linear(node_dim, hidden_dim)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, data):
#         x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
#         x = self.conv1(x, edge_index, edge_attr)
#         x = F.relu(x)
#         x = self.conv2(x, edge_index, edge_attr)
#         x = F.relu(x)
#         x = global_mean_pool(x, batch)
#         x = F.relu(self.fc1(x))
#         return self.fc2(x)

    
TARGET_PROPERTIES = [
    'LCD', 'PLD', 'LFPD', 'cm3_g', 'ASA_m2_cm3', 'ASA_m2_g',
    'AV_VF', 'AV_cm3_g', 'Has_OMS'
]
OUTPUT_DIM = len(TARGET_PROPERTIES)

class CGCNN(torch.nn.Module):
    def __init__(self, node_dim=1, edge_dim=1, hidden_dim=64, output_dim=OUTPUT_DIM):
        super().__init__()
        self.conv1 = CGConv(channels=node_dim, dim=edge_dim)
        self.conv2 = CGConv(channels=node_dim, dim=edge_dim)
        self.fc1 = nn.Linear(node_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
