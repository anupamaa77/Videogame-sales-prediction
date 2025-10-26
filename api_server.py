import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor

# Load the trained model and feature names
model = joblib.load("random_forest_model.joblib")
with open("feature_names.txt", "r") as f:
    feature_names = [line.strip() for line in f.readlines()]


app = FastAPI()

# Define Pydantic model for request data validation
class SalesPredictionRequest(BaseModel):
    Platform: str
    Year: int
    Genre: str
    NA_Sales: float
    EU_Sales: float
    JP_Sales: float
    Other_Sales: float


@app.post("/predict")
def predict_sales(request: SalesPredictionRequest):
    # Convert request data into a DataFrame with the correct columns
    input_data = pd.DataFrame([request.dict()])

    # Reorder columns to match the trained model
    input_data = input_data[feature_names]

    # Predict with the trained model
    prediction = model.predict(input_data)[0]
    
    # Return the prediction result
    return {"predicted_sales": prediction}
