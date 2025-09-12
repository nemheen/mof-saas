import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F


def get_loader(dataset, indices, batch_size=16):
    subset = [dataset[i] for i in indices]
    return DataLoader(subset, batch_size=batch_size, shuffle=True)


def evaluate_loss(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            out = model(batch)
            loss = criterion(out.view(-1), batch.y.view(-1))
            total_loss += loss.item() * batch.num_graphs
    return total_loss / len(data_loader.dataset)


def get_uncertainty_scores(model, dataset, indices, device):
    model.eval()
    uncertainties = []
    criterion = torch.nn.MSELoss(reduction='none')
    with torch.no_grad():
        for idx in indices:
            data = dataset[idx].to(device)
            out = model(data)
            loss = criterion(out.view(-1), data.y.view(-1)).item()
            uncertainties.append((idx, loss))
    return sorted(uncertainties, key=lambda x: x[1], reverse=True)


# utils.py
import matplotlib.pyplot as plt
from io import BytesIO

def generate_co2_adsorption_plot(pressures: list, amounts: list) -> bytes:
    """
    Generates a CO2 adsorption plot as a PNG image in memory.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(pressures, amounts, marker='o', linestyle='-')
    ax.set_title("CO₂ Adsorption Isotherm")
    ax.set_xlabel("Pressure (bar)")
    ax.set_ylabel("Adsorption Amount (mol/kg)")
    ax.grid(True)
    
    # Save the plot to an in-memory buffer
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)  # Close the plot to free up memory
    buf.seek(0)
    
    return buf.getvalue()