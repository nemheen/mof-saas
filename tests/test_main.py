import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the MOF SaaS Platform"}


def test_predict_default():
    response = client.get("/predict")
    assert response.status_code == 200
    assert response.json()["property"] == "adsorption"
