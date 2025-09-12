import os
import torch
from model_definition import CGCNN, TARGET_PROPERTIES, OUTPUT_DIM
from cif_graph import cif_to_pyg_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Path to model checkpoint
MODEL_PATH = os.path.join(os.path.dirname(__file__), "saved_models/cgcnn_finetuned_with_norm.pth")

# Load checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device)

# Load model
model = CGCNN(output_dim=OUTPUT_DIM)
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# Load normalization stats
mean = checkpoint['mean'].to(device)  # Tensor of shape [9]
std = checkpoint['std'].to(device)

# def predict_from_cif(cif_path: str) -> dict:
#     """
#     Predicts 9 MOF properties from a given CIF file.
#     Applies inverse normalization to return values in original units.
#     """
#     data = cif_to_pyg_graph(cif_path)
#     data = data.to(device)

#     with torch.no_grad():
#         output = model(data)  # normalized predictions

#     # Inverse normalization
#     output_physical = output * std + mean

#     prediction_values = output_physical.squeeze().cpu().tolist()
#     results = {prop: float(val) for prop, val in zip(TARGET_PROPERTIES, prediction_values)}

#     # Binarize Has_OMS
#     results["Has_OMS"] = int(results["Has_OMS"] >= 0.5)


#     print(f"Predicted properties for {cif_path}: {results}")
#     return results

# predictor.py
# ... (existing imports and model loading code)

def predict_from_cif(cif_path: str) -> tuple[dict, dict]:
    """
    Predicts 9 MOF properties and provides placeholder CO2 adsorption data.
    """
    data = cif_to_pyg_graph(cif_path)
    data = data.to(device)

    with torch.no_grad():
        output = model(data)  # normalized predictions

    # Inverse normalization
    output_physical = output * std + mean

    prediction_values = output_physical.squeeze().cpu().tolist()
    results = {prop: float(val) for prop, val in zip(TARGET_PROPERTIES, prediction_values)}

    # Binarize Has_OMS
    results["Has_OMS"] = int(results["Has_OMS"] >= 0.5)

    # NEW: Generate placeholder CO2 adsorption data
    adsorption_data = {
        'pressures': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'adsorption_amounts': [0.0, 0.05, 0.1, 0.15, 0.25, 0.4, 0.5, 0.65, 0.8, 0.9, 0.95]
    }

    print(f"Predicted properties for {cif_path}: {results}")

    # Return both the prediction results and the adsorption data
    return results, adsorption_data
