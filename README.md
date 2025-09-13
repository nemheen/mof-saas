# MOF SaaS Platform

This repository contains a starting point for a SaaS platform focused on material optimization for carbon capture using Metal-Organic Frameworks (MOFs) and related materials. The project uses **FastAPI** as the backend framework and will eventually incorporate machine learning models and a rich frontend.

## Project structure

- `backend/app/` - FastAPI application
- `tests/` - unit tests
- `requirements.txt` - Python dependencies

## Running locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the API server:
   ```bash
   uvicorn backend.app.main:app --reload
   ```
3. Visit `http://localhost:8000` to see the welcome message and `http://localhost:8000/docs` for the automatically generated API documentation.

## Future work

This is a minimal starting point. The plan is to expand the API with:

- Endpoints for material property prediction and recommendations
- Integration with a vector database (e.g., FAISS) for RAG-based search
- Authentication and user management
- Dashboard and visualization features
- Frontend built with React and Plotly

Contributions and suggestions are welcome!
