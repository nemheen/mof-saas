from pymatgen.core import Structure
from pymatgen.analysis.local_env import CrystalNN
from torch_geometric.data import Data
import torch
import os

def cif_to_pyg_graph(cif_path, label=None):
    structure = Structure.from_file(cif_path)
    atom_features = []
    pos = []
    edge_index = []
    edge_attr = []

    cnn = CrystalNN()

    for i, site in enumerate(structure):
        atom_features.append(site.specie.Z)  # atomic number as node feature
        pos.append(site.coords)

        neighbors = cnn.get_nn_info(structure, i)
        for neighbor in neighbors:
            j = neighbor['site_index']
            edge_index.append((i, j))
            edge_attr.append([neighbor['weight']])

    # Convert to tensor
    x = torch.tensor(atom_features, dtype=torch.float).unsqueeze(1)  # shape [num_atoms, 1]
    pos = torch.tensor(pos, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    data = Data(x=x, pos=pos, edge_index=edge_index, edge_attr=edge_attr)

    # Add label if provided, else None
    if label is not None:
        data.y = torch.tensor([label], dtype=torch.float)
    else:
        data.y = None

    return data
