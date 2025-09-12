import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy.orm import Session
from sqlalchemy import or_


from typing import List, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import or_

from models import CoreMOF, User
from auth import hash_pw, verify_pw


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- User Management Functions ---

def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def create_user(db: Session, email: str, password: str) -> User:
    user = User(email=email, hashed_password=hash_pw(password))
    db.add(user); db.commit(); db.refresh(user)
    return user

# --- Core MOF Database Functions ---
from typing import List, Optional
from sqlalchemy.orm import Session
from models import CoreMOF

def get_mofs_by_filters(db: Session, *, min_asa: float = 0.0, min_pld: float = 0.0, limit: int = 500):
    q = db.query(CoreMOF)
    if min_asa is not None:
        q = q.filter(CoreMOF.ASA_m2_g != None, CoreMOF.ASA_m2_g >= float(min_asa))  # noqa: E711
    if min_pld is not None:
        q = q.filter(CoreMOF.PLD != None, CoreMOF.PLD >= float(min_pld))  # noqa: E711
    q = q.order_by(CoreMOF.ASA_m2_g.desc())
    return q.limit(limit).all()



def search_coremofs_by_properties(db: Session, search_params: Dict[str, Any]) -> List[CoreMOF]:
    """
    Dynamically searches for the 5 closest CoreMOF instances based on a dictionary of properties.
    For numerical properties, it calculates a 'closeness score' based on the sum of absolute differences.
    """
    # Get all MOFs from the database
    all_mofs = db.query(CoreMOF).all()
    
    # Initialize a list to store MOFs that match string properties and their closeness scores
    scored_mofs = []
    
    # Define a list of numerical properties for scoring
    numerical_properties = [
        "LCD", "PLD", "LFPD", "cm3_g", "ASA_m2_cm3", "ASA_m2_g",
        "AV_VF", "AV_cm3_g", "Has_OMS"
    ]
    
    # Iterate through all MOFs to find the best matches
    for mof in all_mofs:
        score = 0.0
        is_match = True
        
        for key, value in search_params.items():
            db_value = getattr(mof, key, None)
            
            if db_value is None:
                # Property not found on the model, skip
                continue

            if key in numerical_properties:
                # Calculate the absolute difference for numerical properties
                if isinstance(value, (int, float)):
                    score += abs(db_value - value)
            else:
                # For string properties, check for a match
                if isinstance(value, str):
                    if value.lower() not in str(db_value).lower():
                        is_match = False
                        break # No match for this MOF, move to the next one
                elif db_value != value:
                    is_match = False
                    break
        
        if is_match:
            scored_mofs.append({"mof": mof, "score": score})

    # Sort the results by the calculated score (lowest score is best)
    scored_mofs.sort(key=lambda x: x["score"])

    # Return the MOF objects from the top 5 results
    return [item["mof"] for item in scored_mofs[:5]]


def get_all_coremofs(db: Session, limit: int = 100):
    return db.query(CoreMOF).limit(limit).all()

def get_coremof_by_filename(db: Session, filename: str):
    return db.query(CoreMOF).filter(CoreMOF.filename == filename).first()

def get_high_surface_area_mofs(db: Session, min_asa: float = 1000.0):
    return db.query(CoreMOF).filter(CoreMOF.ASA_m2_g > min_asa).all()

def search_coremofs(db: Session, search_params: dict):
    """
    A generalized search function that filters MOFs by one or more properties.
    search_params: a dictionary like {"filename": "ABEXEM", "LCD": 10.5}
    """
    query = db.query(CoreMOF)

    # Iterate through the search parameters and apply filters
    for key, value in search_params.items():
        if key == "filename":
            # Use ilike for case-insensitive keyword search
            query = query.filter(CoreMOF.filename.ilike(f"%{value}%"))
        else:
            # For numerical properties, filter based on exact match
            try:
                numeric_value = float(value)
                query = query.filter(getattr(CoreMOF, key) == numeric_value)
            except (ValueError, AttributeError):
                # Handle cases where value is not a number or the column does not exist
                pass

    return query.all()

def get_top_mofs(db: Session, limit: int = 10, min_asa: float = None, min_pld: float = None):
    query = db.query(CoreMOF)
    if min_asa is not None:
        query = query.filter(CoreMOF.ASA_m2_g >= min_asa)
    if min_pld is not None:
        query = query.filter(CoreMOF.PLD >= min_pld)
    return query.limit(limit).all()

# --- Prediction & Plotting Functions ---

def generate_adsorption_plot(model, mof_data):
    """Generates a CO2 adsorption plot for a single MOF prediction."""
    try:
        pressures = np.linspace(0.0, 1.0, 50).reshape(-1, 1) # 50 points from 0 to 1
        
        # Ensure mof_data has the correct columns for the model
        mof_data_columns = [
            "LCD", "PLD", "LFPD", "cm3_g", "ASA_m2_cm3",
            "ASA_m2_g", "AV_VF", "AV_cm3_g", "Has_OMS"
        ]
        
        # Check if any required data is missing
        missing_data = [col for col in mof_data_columns if col not in mof_data.columns or mof_data[col].isnull().iloc[0]]
        if missing_data:
            print(f"Missing data for adsorption plot: {missing_data}")
            return None

        features = mof_data[mof_data_columns].values
        
        # Repeat MOF features for each pressure point
        features_repeated = np.repeat(features, len(pressures), axis=0)
        
        # Combine features and pressures
        input_data = np.hstack((features_repeated, pressures))
        
        # Convert to tensor and get predictions
        import torch
        with torch.no_grad():
            input_tensor = torch.tensor(input_data, dtype=torch.float32)
            predictions = model(input_tensor).numpy()
            
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(pressures, predictions, marker='o', linestyle='-', color='b')
        ax.set_title("Predicted CO₂ Adsorption Isotherm")
        ax.set_xlabel("Pressure (bar)")
        ax.set_ylabel("Adsorption (mmol/g)")
        ax.grid(True)
        
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    except Exception as e:
        print(f"Error generating adsorption plot: {e}")
        return None
