# mof-application/mof-api/main.py
from dotenv import load_dotenv
load_dotenv()

import os
import re
import json
import base64
import shutil
import asyncio
import tempfile
from tempfile import NamedTemporaryFile
from typing import List, Dict, Any, Optional
from collections.abc import Mapping, Sequence

import numpy as np
import firebase_admin
from firebase_admin import credentials

from fastapi import (
    FastAPI, Depends, UploadFile, File, HTTPException, status,
    Response, Request, Query
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, PlainTextResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, ValidationError
from sqlalchemy.orm import Session
from fpdf import FPDF
from firebase_init import init_firebase_app

# Our modules
import crud  # <-- keep module import; call as crud.func()
from crud import get_mofs_by_filters 
from database import SessionLocal
from schemas import UserCreate, UserOut, TokenOut, CoreMOFSchema
from models import User, CoreMOF
from predictor import predict_from_cif
from gemini_rag import ask_gemini_rag, ask_gemini_recommendation
from gemini_vision import ask_gemini_vision
from utils import generate_co2_adsorption_plot
from auth import (
    get_current_user, create_access_token, create_refresh_token,
    set_refresh_cookie, clear_refresh_cookie, verify_pw, decode_token
)


# --- START: New imports for serving frontend ---
from fastapi.staticfiles import StaticFiles

# ---- Initialize Firebase Admin safely ----
init_firebase_app()

import firebase_crud


app = FastAPI(
    title="MOF Properties Prediction API",
    description="Predict MOF material properties from CIF files and retrieve data from CoRE MOF database.",
)

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://.*\.hf\.space$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------- Request Schemas -------------------
class RAGQuery(BaseModel):
    query: str

class RecommendationQuery(BaseModel):
    query: str

class FilteredRecommendationQuery(BaseModel):
    requirement: str
    min_asa: float = 0.0
    min_pld: float = 0.0

class ExportPayload(BaseModel):
    filename: str
    prediction: Dict[str, Any]
    adsorption_plot: Optional[str] = None

class AdsorptionData(BaseModel):
    pressures: list[float]
    adsorption_amounts: list[float]


# ------------------- DB Dependency -------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- add near the top ---

def _mk_pdf(prediction: dict, plot_b64: str, title: str = "MOF Prediction Report") -> bytes:
    """Create a simple PDF with a table and embedded plot image (base64 PNG)."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=12)

    # Table
    pdf.cell(0, 8, "Predicted Properties:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    for k, v in prediction.items():
        pdf.cell(80, 7, str(k))
        pdf.cell(0, 7, f"{v:.4f}" if isinstance(v, (int, float)) else str(v), new_x="LMARGIN", new_y="NEXT")

    # Plot
    if plot_b64:
        try:
            img_bytes = base64.b64decode(plot_b64)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                tmp.write(img_bytes)
                img_path = tmp.name
            pdf.ln(5)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "CO₂ Adsorption Plot:", new_x="LMARGIN", new_y="NEXT")
            # width ~170mm keeps margins on A4
            pdf.image(img_path, w=170)
        except Exception:
            pass

    return bytes(pdf.output(dest="S"))


# ------------------- Filter → Rank → Recommend/Explain -------------------
import re
import traceback
from decimal import Decimal
from typing import Optional, Dict, Any, List

from fastapi import Query
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

# --- Logging ---
import logging, time

logger = logging.getLogger("mof")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,  # change to DEBUG if you want verbose
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )

def _get_field(obj, key):
    """
    Robustly read a field from SQLAlchemy ORM instance, Row, or dict.
    """
    if obj is None:
        return None
    # ORM attribute
    if hasattr(obj, key):
        return getattr(obj, key)
    # SQLAlchemy Row: obj._mapping behaves like a read-only dict
    m = getattr(obj, "_mapping", None)
    if m and key in m:
        return m[key]
    # plain dict
    if isinstance(obj, dict) and key in obj:
        return obj[key]
    return None


def _as_float(x):
    try:
        if x is None:
            return None
        if isinstance(x, (float, int)):
            return float(x)
        if isinstance(x, Decimal):
            return float(x)
        # numpy, strings, etc.
        return float(str(x))
    except Exception:
        return None

def _parse_requirement_weights(text: str) -> Dict[str, float]:
    t = (text or "").lower()
    w = {"ASA_m2_g":0.8, "AV_cm3_g":0.4, "AV_VF":0.3, "PLD":0.2, "LCD":0.2, "Has_OMS":0.2}
    if re.search(r"\blow[-\s]?pressure|\bdilute|\bppm|\bppmv|\bflue gas\b", t):
        w["AV_VF"]+=0.3; w["Has_OMS"]+=0.3; w["PLD"]-=0.3
    if re.search(r"\b(high|max(imum)?)\s+(capacity|uptake)|\buptake\b", t):
        w["ASA_m2_g"]+=0.4; w["AV_cm3_g"]+=0.3
    if re.search(r"\b(large|bulky|big)\s+(molecule|sorbate)|\bC\d{2,}\b", t):
        w["PLD"]+=0.4; w["LCD"]+=0.3
    if re.search(r"\bselectiv(ity|e)\b|\bco2\b", t):
        w["Has_OMS"]+=0.4; w["PLD"]-=0.1
    total = sum(abs(v) for v in w.values()) or 1.0
    return {k: v/total for k, v in w.items()}

def _min_max_norm(values: List[Optional[float]]) -> List[float]:
    xs = [v for v in values if isinstance(v, (int, float)) and v is not None]
    if not xs:
        return [0.5 for _ in values]
    vmin, vmax = min(xs), max(xs)
    if vmax <= vmin + 1e-12:
        return [0.5 for _ in values]
    def norm(v):
        if v is None or not isinstance(v, (int, float)):
            return 0.5
        return (v - vmin) / (vmax - vmin)
    return [norm(v) for v in values]

def _score_candidates(cands, weights: Dict[str, float]) -> List[Dict[str, Any]]:
    # Pull raw values as floats/ints (or None)
    ASA_m2_g_raw = [_as_float(_get_field(m, "ASA_m2_g")) for m in cands]
    AV_cm3_g_raw = [_as_float(_get_field(m, "AV_cm3_g")) for m in cands]
    AV_VF_raw    = [_as_float(_get_field(m, "AV_VF"))    for m in cands]
    PLD_raw      = [_as_float(_get_field(m, "PLD"))      for m in cands]
    LCD_raw      = [_as_float(_get_field(m, "LCD"))      for m in cands]
    Has_OMS_raw  = [1.0 if _get_field(m, "Has_OMS") in (1, True) else 0.0 for m in cands]

    # Normalize numeric columns
    n_ASA_m2_g = _min_max_norm(ASA_m2_g_raw)
    n_AV_cm3_g = _min_max_norm(AV_cm3_g_raw)
    n_AV_VF    = _min_max_norm(AV_VF_raw)
    n_PLD      = _min_max_norm(PLD_raw)
    n_LCD      = _min_max_norm(LCD_raw)

    out = []
    for i, m in enumerate(cands):
        terms = {
            "ASA_m2_g": n_ASA_m2_g[i],
            "AV_cm3_g": n_AV_cm3_g[i],
            "AV_VF":    n_AV_VF[i],
            "PLD":      n_PLD[i],
            "LCD":      n_LCD[i],
            "Has_OMS":  Has_OMS_raw[i],
        }
        score, contribs = 0.0, {}
        for k, w in weights.items():
            val = terms.get(k, 0.5)
            val_eff = (1.0 - val) if w < 0 else val
            part = abs(w) * val_eff
            score += part
            contribs[k] = round(part, 4)

        props = {
            "ASA_m2_g": ASA_m2_g_raw[i],
            "AV_cm3_g": AV_cm3_g_raw[i],
            "AV_VF":    AV_VF_raw[i],
            "PLD":      PLD_raw[i],
            "LCD":      LCD_raw[i],
            "Has_OMS":  1 if Has_OMS_raw[i] >= 0.5 else 0,
        }

        filename = _get_field(m, "filename")
        bits = []
        if props["ASA_m2_g"] is not None: bits.append(f"ASA {props['ASA_m2_g']:.0f} m²/g")
        if props["AV_cm3_g"] is not None: bits.append(f"AV {props['AV_cm3_g']:.2f} cm³/g")
        if props["PLD"] is not None:      bits.append(f"PLD {props['PLD']:.2f} Å")
        if props["Has_OMS"]:              bits.append("OMS present")

        out.append({
            "filename": filename,
            "score": round(float(score), 4),
            "contribs": contribs,
            "properties": props,
            "reason": ", ".join(bits) if bits else "meets filters",
        })

    out.sort(key=lambda x: x["score"], reverse=True)
    return out


import traceback

def _truncate(s: str, max_chars: int = 6000) -> str:
    s = s or ""
    return s[:max_chars]

def _compact_top_items_for_prompt(top_items):
    # Keep the LLM input small and numeric-only (no huge blobs)
    lines = []
    for it in top_items:
        p = it.get("properties", {})
        lines.append(
            f"- {it.get('filename','?')}"
            f" | score={it.get('score')}"
            f" | ASA={p.get('ASA_m2_g')}"
            f" | AV={p.get('AV_cm3_g')}"
            f" | PLD={p.get('PLD')}"
            f" | LCD={p.get('LCD')}"
            f" | OMS={p.get('Has_OMS')}"
        )
    return "\n".join(lines)

def _call_gemini_summary(prompt: str) -> str:
    """
    Use the same path that works for /rag_query. Never raise.
    Always return a short string (possibly empty).
    """
    try:
        text = ask_gemini_rag(prompt)  # SAME function that works for /rag_query
        if isinstance(text, str) and text.strip():
            return text.strip()
        return str(text) if text is not None else ""
    except Exception as e:
        print("[recommend_filtered] LLM summary failed:", e)
        traceback.print_exc()
        return ""




# --- JSON/Firestore safety helpers ---

def _to_py_scalar(x):
    # Convert numpy / torch scalars to plain Python
    try:
        import torch
        if isinstance(x, torch.Tensor) and x.ndim == 0:
            return x.item()
    except Exception:
        pass
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    return x

def json_safe(obj):
    """
    Recursively convert numpy types, torch tensors, ndarrays, and sequences
    into Firestore/JSON-safe Python types.
    """
    # torch tensor -> list or scalar
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
    except Exception:
        pass

    # numpy array -> list
    if isinstance(obj, np.ndarray):
        return obj.tolist()

    # mapping
    if isinstance(obj, Mapping):
        return {str(k): json_safe(v) for k, v in obj.items()}

    # sequence but not str/bytes
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        return [json_safe(v) for v in obj]

    # numpy / torch scalar
    return _to_py_scalar(obj)



# ------------------- ROUTES -------------------

@app.post("/auth/signup", response_model=UserOut, status_code=201)
def signup(payload: UserCreate, db: Session = Depends(get_db)):
    if crud.get_user_by_email(db, payload.email):
        raise HTTPException(status_code=409, detail="Email already registered")
    user = crud.create_user(db, payload.email, payload.password)
    return user

@app.post("/auth/login", response_model=TokenOut)
def login(payload: UserCreate, response: Response, db: Session = Depends(get_db)):
    user = crud.get_user_by_email(db, payload.email)
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not verify_pw(payload.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    access = create_access_token(sub=user.email)
    refresh = create_refresh_token(sub=user.email)
    set_refresh_cookie(response, refresh)
    return {"access_token": access, "token_type": "bearer"}

@app.post("/auth/refresh", response_model=TokenOut)
def refresh(request: Request, response: Response, db: Session = Depends(get_db)):
    from jose import JWTError
    refresh_token = request.cookies.get("refresh_token")
    if not refresh_token:
        raise HTTPException(status_code=401, detail="No refresh cookie")
    try:
        payload = decode_token(refresh_token)
        if payload.get("scope") != "refresh":
            raise HTTPException(status_code=401, detail="Invalid refresh token")
        email = payload["sub"]
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid refresh token")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    user = crud.get_user_by_email(db, email)
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not active")
    
    new_access = create_access_token(email)
    new_refresh = create_refresh_token(email)
    set_refresh_cookie(response, new_refresh)
    return {"access_token": new_access, "token_type": "bearer"}

@app.post("/auth/logout", status_code=204)
def logout(response: Response):
    clear_refresh_cookie(response)
    return Response(status_code=204)

@app.get("/auth/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return user

# ------------------- MOF SPECIFIC ROUTES -------------------

@app.get("/coremofs/search", response_model=List[CoreMOFSchema])
async def search_mofs(
    filename: Optional[str] = None,
    LCD: Optional[float] = None,
    PLD: Optional[float] = None,
    LFPD: Optional[float] = None,
    cm3_g: Optional[float] = None,
    ASA_m2_cm3: Optional[float] = None,
    ASA_m2_g: Optional[float] = None,
    AV_VF: Optional[float] = None,
    AV_cm3_g: Optional[float] = None,
    Has_OMS: Optional[int] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Search CoRE MOFs by any combination of properties."""
    search_params = {
        "filename": filename,
        "LCD": LCD,
        "PLD": PLD,
        "LFPD": LFPD,
        "cm3_g": cm3_g,
        "ASA_m2_cm3": ASA_m2_cm3,
        "ASA_m2_g": ASA_m2_g,
        "AV_VF": AV_VF,
        "AV_cm3_g": AV_cm3_g,
        "Has_OMS": Has_OMS
    }
    
    # Filter out None values to build the query
    filtered_params = {k: v for k, v in search_params.items() if v is not None}
    
    # If no parameters are provided, return an error or all results
    if not filtered_params:
        raise HTTPException(status_code=400, detail="Please provide at least one search parameter.")

    results = crud.search_coremofs_by_properties(db, filtered_params)

    # Convert the values in filtered_params to strings for logging
    str_filtered_params = {k: str(v) for k, v in filtered_params.items()}

    await firebase_crud.add_user_activity(
        user_id=current_user.id,
        activity_type="Property Search",
        details={"search_params": str_filtered_params, "num_results": str(len(results))}
    )
    return results

@app.get("/coremofs/surface_area", response_model=List[CoreMOFSchema])
async def read_high_surface_area_mofs(min_asa: float = 1000.0, db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieve CoRE MOFs with high surface area."""
    results = crud.get_high_surface_area_mofs(db, min_asa)
    await firebase_crud.add_user_activity(
        user_id=current_user.id,
        activity_type="Surface Area Search",
        details={"min_asa": str(min_asa), "num_results": str(len(results))}
    )
    return results


@app.get("/coremofs/{filename}/plot")
async def get_mof_plot_by_id(filename: str, db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieve a CO2 adsorption plot for a specific MOF."""
    mof_data = crud.get_coremof_by_id(db, filename)
    if not mof_data:
        raise HTTPException(status_code=404, detail="MOF not found")
    
    if not hasattr(mof_data, 'adsorption_data') or mof_data.adsorption_data is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Adsorption data not found for this MOF. Cannot generate plot."
        )
    
    adsorption_data = mof_data.adsorption_data
    
    if not hasattr(adsorption_data, 'pressures') or not hasattr(adsorption_data, 'adsorption_amounts'):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Incomplete adsorption data for this MOF (missing pressures or amounts). Cannot generate plot."
        )
    
    plot_bytes = generate_co2_adsorption_plot(
        adsorption_data.pressures, adsorption_data.adsorption_amounts
    )
    
    import base64
    plot_base64 = base64.b64encode(plot_bytes).decode('utf-8')

    await firebase_crud.add_user_activity(
        user_id=current_user.id,
        activity_type="View MOF Plot",
        details={"filename": filename}
    )
    return {"filename": filename, "adsorption_plot": plot_base64}



@app.get("/coremofs/{filename}", response_model=CoreMOFSchema)
async def read_single_coremof(filename: str, db: Session = Depends(get_db)):
    result = crud.get_coremof_by_id(db, filename)
    if not result:
        raise HTTPException(status_code=404, detail="MOF not found")

    try:
        validated_result = CoreMOFSchema.model_validate(result)
        print(f"Successfully validated: {validated_result.filename}")
        return validated_result
    except ValidationError as e:
        print(f"Pydantic Validation Error for filename '{filename}':")
        print(e.errors())
        raise HTTPException(status_code=500, detail=f"Data validation error: {e.errors()}")
    
@app.get("/coremofs", response_model=List[CoreMOFSchema])
async def read_coremofs(limit: int = 100, db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Retrieve all CoRE MOFs."""
    results = crud.get_all_coremofs(db, limit)
    await firebase_crud.add_user_activity(
        user_id=current_user.id,
        activity_type="Browse All MOFs",
        details={"limit": str(limit), "num_results": str(len(results))}
    )
    return results


@app.post("/predict/")
async def predict_endpoint(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    """
    Predict properties from uploaded CIF file. Requires authentication.
    """
    if not file.filename.lower().endswith(".cif"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only .cif files are accepted."
        )

    tmp_path = None
    try:
        # 1) Save temp file
        with NamedTemporaryFile(delete=False, suffix=".cif") as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # 2) Run prediction
        prediction, adsorption_data = predict_from_cif(tmp_path)

        # 3) Coerce to plain Python types (avoid Firestore type errors)
        safe_prediction = {str(k): float(_to_py_scalar(v)) for k, v in dict(prediction).items()}
        # adsorption_data may contain numpy arrays / tensors
        # Ensure expected keys exist and convert to floats
        pressures = adsorption_data.get("pressures") if isinstance(adsorption_data, dict) else None
        amounts   = adsorption_data.get("adsorption_amounts") if isinstance(adsorption_data, dict) else None
        if pressures is None or amounts is None:
            raise ValueError("Predictor did not return 'pressures' and 'adsorption_amounts' in adsorption_data.")

        pressures = [float(_to_py_scalar(x)) for x in json_safe(pressures)]
        amounts   = [float(_to_py_scalar(x)) for x in json_safe(amounts)]
        safe_adsorption = {"pressures": pressures, "adsorption_amounts": amounts}

        # 4) Generate plot
        plot_bytes = generate_co2_adsorption_plot(pressures, amounts)
        plot_base64 = base64.b64encode(plot_bytes).decode("utf-8")

        # 5) Best-effort Firestore writes (don’t fail the API if these fail)
        save_warning = None
        try:
            await firebase_crud.add_prediction_to_my_files(
                user_id=current_user.id,
                filename=file.filename,
                prediction=json_safe(safe_prediction),
                plot=plot_base64,
                adsorption_data=json_safe(safe_adsorption),
            )
        except Exception as fe:
            # Don’t 500 the whole request—just bubble a warning to the client.
            save_warning = f"Saved prediction locally but failed to write to history/storage: {fe}"

        try:
            await firebase_crud.add_user_activity(
                user_id=current_user.id,
                activity_type="CIF Prediction",
                details={
                    "filename": file.filename,
                    "predicted_property_example": next(iter(safe_prediction.keys()), "N/A"),
                },
            )
        except Exception as fe:
            # Not fatal either
            save_warning = (save_warning or "") + f" | Failed to log history: {fe}"

        # 6) Return success payload
        resp = {
            "filename": file.filename,
            "prediction": safe_prediction,
            "adsorption_plot": plot_base64,
        }
        if save_warning:
            resp["warning"] = save_warning
        return resp

    except HTTPException:
        raise
    except Exception as e:
        # Surface the real reason—it’s usually a Firestore/serialization issue
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {e}"
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ------------------- FAVORITES -------------------

@app.post("/user/favorites/{mof_id}")
async def add_to_favorites_endpoint(
    mof_id: str,
    filename: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
):
    try:
        await firebase_crud.add_to_favorites(str(current_user.id), mof_id, filename=filename)
        # Optional: log in history
        await firebase_crud.add_user_activity(str(current_user.id), "Add Favorite", {"mof_id": mof_id, "filename": filename or mof_id})
        return {"message": f"Added {mof_id} to favorites"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add favorite: {e}")

@app.delete("/user/favorites/{mof_id}")
async def remove_favorite_endpoint(
    mof_id: str,
    current_user: User = Depends(get_current_user),
):
    try:
        await firebase_crud.remove_from_favorites(str(current_user.id), mof_id)
        await firebase_crud.add_user_activity(str(current_user.id), "Remove Favorite", {"mof_id": mof_id})
        return {"message": f"Removed {mof_id} from favorites"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to remove favorite: {e}")

@app.get("/user/favorites")
async def list_favorites_endpoint(
    limit: int = 50,
    cursor: Optional[str] = None,
    current_user: User = Depends(get_current_user),
):
    try:
        return await firebase_crud.list_favorites(str(current_user.id), limit=limit, cursor=cursor)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list favorites: {e}")


@app.get("/user/history")  # no response_model so we can return a dict
async def get_user_history(
    limit: int = 20,
    current_user: User = Depends(get_current_user),
):
    """
    Return recent activity in the shape the UI expects:
    { "items": [...], "next_cursor_ts": null }
    """
    try:
        # IMPORTANT: cast user id to str to avoid Firestore path errors
        data = await firebase_crud.get_user_activities(str(current_user.id), limit=limit)
        # firebase_crud.get_user_activities returns {"items":[...]}
        items = data.get("items", []) if isinstance(data, dict) else (data or [])
        return {"items": items, "next_cursor_ts": None}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load history: {e}")



@app.post("/image_search")
async def image_search_endpoint(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user)
):
    if not (file.content_type or "").startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Please upload an image file."
        )

    try:
        # 1) Vision
        image_bytes = await file.read()
        response_text = await ask_gemini_vision(image_bytes)

        # 2) Fire-and-forget history log (cannot break the request)
        async def _log_activity(u_id: Any, details: Dict[str, Any]):
            try:
                await firebase_crud.add_user_activity(
                    user_id=str(u_id),  # ensure string
                    activity_type="Image Search (Gemini Vision)",
                    details=_json_safe(details),  # ensure JSON/Firestore safe
                )
            except Exception as e:
                # Keep it quiet to avoid 500s; print if you want server-side diagnostics
                # print(f"[image_search] history log failed: {e}")
                pass

        asyncio.create_task(_log_activity(
            current_user.id,
            {
                "filename": file.filename or "",
                "mime_type": file.content_type or "",
                "response_summary": (response_text[:100] + "...") if isinstance(response_text, str) and len(response_text) > 100 else response_text
            }
        ))

        # 3) Return the vision result (regardless of logging outcome)
        return {"response": response_text}

    except HTTPException:
        raise
    except Exception as e:
        # Anything else bubbles as a 500 with a clear message for the UI
        raise HTTPException(status_code=500, detail=f"Image search failed: {e}")


# ------------------- RAG & Recommendation -------------------


@app.post("/rag_query")
async def rag_query_endpoint(query: RAGQuery,
    current_user: User = Depends(get_current_user)
):
    try:
        answer = ask_gemini_rag(query.query)
        # NEW: Log this activity
        await firebase_crud.add_user_activity(
            user_id=current_user.id,
            activity_type="RAG Query",
            details={"query": query.query, "answer_summary": answer[:100] + "..." if len(answer) > 100 else answer}
        )
        return {"response": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG query failed: {e}")

@app.post("/recommend")
async def recommend_materials(query: RecommendationQuery,
    current_user: User = Depends(get_current_user)
):
    """LLM-based material similarity/synthesis recommendations (Gemini + RAG)."""
    try:
        recommendation = ask_gemini_recommendation(query.query)
        # NEW: Log this activity
        await firebase_crud.add_user_activity(
            user_id=current_user.id,
            activity_type="Material Recommendation",
            details={"query": query.query, "recommendation_summary": recommendation[:100] + "..." if len(recommendation) > 100 else recommendation}
        )
        return {"recommendation": recommendation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {e}")
    
@app.post("/recommend_filtered")
async def recommend_filtered_materials(
    query: FilteredRecommendationQuery,
    top_k: int = Query(5, ge=1, le=20),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    debug: bool = Query(False),  # optional: ?debug=1 to return a tiny debug section
):
    t0 = time.perf_counter()
    try:
        # 0) sanitize filters
        min_asa = max(0.0, _as_float(query.min_asa) or 0.0)
        min_pld = max(0.0, _as_float(query.min_pld) or 0.0)
        logger.info(
            "[recommend_filtered] user=%s filters min_asa=%s min_pld=%s top_k=%s",
            str(current_user.id), min_asa, min_pld, top_k
        )

        # 1) FILTER (materialize)
        try:
            t_db0 = time.perf_counter()
            candidate_mofs = list(get_mofs_by_filters(db, min_asa=min_asa, min_pld=min_pld))
            t_db1 = time.perf_counter()
            logger.info(
                "[recommend_filtered] DB returned %d candidates in %.3fs",
                len(candidate_mofs), t_db1 - t_db0
            )
        except Exception as db_err:
            logger.exception("[recommend_filtered] DB filter failed")
            raise HTTPException(status_code=500, detail=f"DB filter failed: {db_err}")

        if not candidate_mofs:
            payload = {
                "filters": {"min_asa": min_asa, "min_pld": min_pld},
                "candidates_count": 0,
                "items": [],
                "llm_recommendation": None,
                "recommendation": "No MOFs met the filter criteria.",
            }
            if debug:
                payload["debug"] = {"elapsed_s": time.perf_counter() - t0}
            return JSONResponse(content=jsonable_encoder(payload), status_code=200)

        # 2) WEIGHTS
        weights = _parse_requirement_weights(query.requirement)
        logger.info("[recommend_filtered] weights=%s", weights)

        # 3) SCORE / RANK
        t_rank0 = time.perf_counter()
        ranked = _score_candidates(candidate_mofs, weights)
        top_items = ranked[:top_k]
        t_rank1 = time.perf_counter()
        logger.info(
            "[recommend_filtered] ranked %d items in %.3fs; top: %s",
            len(ranked), t_rank1 - t_rank0,
            [it["filename"] for it in top_items]
        )

        # 4) LLM summary (best-effort)
        llm_expl = None
        try:
            tab = _compact_top_items_for_prompt(top_items)
            prompt = (
                "Produce a concise 120–180 word recommendation explaining the top MOF picks and key trade-offs. "
                "Do NOT invent numbers; only use the fields provided below.\n\n"
                f"User requirement: {query.requirement}\n\n"
                f"Top candidates (filename | score | ASA | AV | PLD | LCD | OMS):\n{tab}"
            )
            prompt = _truncate(prompt, 6000)
            t_llm0 = time.perf_counter()
            llm_expl = _call_gemini_summary(prompt) or None
            t_llm1 = time.perf_counter()
            logger.info(
                "[recommend_filtered] LLM summary len=%s in %.3fs",
                (len(llm_expl) if llm_expl else 0),
                t_llm1 - t_llm0
            )
        except Exception as e:
            logger.exception("[recommend_filtered] LLM summary crashed: %s", e)
            llm_expl = None  # never fatal

        # Legacy text
        legacy_text = "\n".join(
            f"{i+1}. {it['filename']} — {it['reason']} (score {it['score']})"
            for i, it in enumerate(top_items)
        )
        if llm_expl:
            legacy_text = f"{legacy_text}\n\nSummary:\n{llm_expl}"

        # 5) LOG HISTORY (best-effort)
        try:
            await firebase_crud.add_user_activity(
                user_id=str(current_user.id),
                activity_type="Filtered Recommendation",
                details={
                    "query": query.requirement,
                    "filters": {"min_asa": str(min_asa), "min_pld": str(min_pld)},
                    "top_files": [it["filename"] for it in top_items],
                },
            )
        except Exception as e:
            logger.warning("[recommend_filtered] history log failed: %s", e)

        elapsed = time.perf_counter() - t0
        logger.info("[recommend_filtered] done in %.3fs", elapsed)

        payload = {
            "filters": {"min_asa": min_asa, "min_pld": min_pld},
            "weights": weights,
            "candidates_count": len(candidate_mofs),
            "top_k": top_k,
            "items": top_items,
            "llm_recommendation": llm_expl,
            "recommendation": legacy_text,
        }
        if debug:
            payload["debug"] = {"elapsed_s": elapsed}
        return JSONResponse(content=jsonable_encoder(payload), status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[recommend_filtered] unhandled error: %s", e)
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {e}")




# ------------------- CO2 Adsorption Analysis -------------------

@app.post("/analyze_adsorption_data")
async def analyze_adsorption_data_endpoint(data: AdsorptionData,
    current_user: User = Depends(get_current_user)
):
    """
    Takes CO2 adsorption data, converts it to an image, and uses Gemini Vision to analyze it.
    """
    try:
        # Step 1: Generate the image from the provided data
        image_bytes = generate_co2_adsorption_plot(data.pressures, data.adsorption_amounts)
        
        # Step 2: Pass the generated image bytes to your Gemini Vision function
        response = await ask_gemini_vision(image_bytes)
        
        # NEW: Log this activity
        await firebase_crud.add_user_activity(
            user_id=current_user.id,
            activity_type="CO2 Adsorption Analysis",
            details={"analysis_summary": response[:100] + "..." if len(response) > 100 else response}
        )
        return {"analysis": response}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {e}"
        )

# --- Export model & endpoints ---

@app.post("/export/pdf")
async def export_pdf(payload: ExportPayload, current_user: User = Depends(get_current_user)):
    pdf_bytes = _mk_pdf(payload.prediction, payload.adsorption_plot or "", f"Report · {payload.filename}")
    headers = {"Content-Disposition": f'attachment; filename="{payload.filename}.pdf"'}
    return StreamingResponse(iter([pdf_bytes]), media_type="application/pdf", headers=headers)

@app.post("/export/html")
async def export_html(payload: ExportPayload, current_user: User = Depends(get_current_user)):
    rows = "\n".join(
        f"<tr><td>{k}</td><td>{(f'{v:.4f}' if isinstance(v,(int,float)) else v)}</td></tr>"
        for k, v in payload.prediction.items()
    )
    img_tag = f'<img alt="CO2 plot" style="max-width:100%" src="data:image/png;base64,{payload.adsorption_plot}"/>' if payload.adsorption_plot else ""
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{payload.filename} · MOF Report</title>
<style>
body{{font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding:20px}}
table{{border-collapse:collapse;width:100%}} th,td{{border:1px solid #ddd;padding:8px}} th{{background:#fafafa;text-align:left}}
h1{{margin-bottom:6px}} .sub{{color:#666;margin-top:0}}
</style></head><body>
<h1>{payload.filename}</h1>
<p class="sub">Carbon-Capture MOF Prediction Snapshot</p>
<h2>Predicted Properties</h2>
<table><thead><tr><th>Property</th><th>Value</th></tr></thead><tbody>{rows}</tbody></table>
{('<h2>CO₂ Adsorption Plot</h2>'+img_tag) if img_tag else ''}
</body></html>"""
    headers = {"Content-Disposition": f'attachment; filename="{payload.filename}.html"'}
    return HTMLResponse(html, headers=headers)

@app.post("/export/json")
async def export_json(payload: ExportPayload, current_user: User = Depends(get_current_user)):
    doc = payload.model_dump()
    buf = json.dumps(doc, indent=2)
    headers = {"Content-Disposition": f'attachment; filename="{payload.filename}.json"'}
    return PlainTextResponse(buf, headers=headers, media_type="application/json")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
            await firebase_crud.add_user_activity(
                user_id=str(u_id),  # ensure string
                activity_type="Image Search (Gemini Vision)",
                details=_json_safe(details),  # ensure JSON/Firestore safe
            )
            except Exception as e:
                # Keep it quiet to avoid 500s; print if you want server-side diagnostics
                # print(f"[image_search] history log failed: {e}")
                pass

        asyncio.create_task(_log_activity(
            current_user.id,
            {
                "filename": file.filename or "",
                "mime_type": file.content_type or "",
                "response_summary": (response_text[:100] + "...") if isinstance(response_text, str) and len(response_text) > 100 else response_text
            }
        ))

        # 3) Return the vision result (regardless of logging outcome)
        return {"response": response_text}

    except HTTPException:
        raise
    except Exception as e:
        # Anything else bubbles as a 500 with a clear message for the UI
        raise HTTPException(status_code=500, detail=f"Image search failed: {e}")

