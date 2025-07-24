from fastapi import FastAPI

app = FastAPI(title="MOF SaaS Platform")

@app.get("/")
def read_root():
    return {"message": "Welcome to the MOF SaaS Platform"}

@app.get("/predict")
def predict(property: str = "adsorption"):
    """Dummy prediction endpoint.
    In the future this will use machine learning models
    to predict material properties.
    """
    return {"property": property, "value": 0.0}
