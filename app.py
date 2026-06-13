# app.py
from fastapi import FastAPI, HTTPException
import pydantic
from pydantic import BaseModel, model_validator
import pandas as pd
import joblib

app = FastAPI(title="Telco Churn Batch Prediction API")

# Load model artifacts
try:
    artifacts = joblib.load('model_artifacts.joblib')
    model = artifacts['model']
    model_features = artifacts['features']
except FileNotFoundError:
    raise RuntimeError("Model file 'model_artifacts.joblib' not found. Run train.py first.")

# Define the expected JSON payload with automated placeholder validation
class CustomerData(BaseModel):
    gender: str                  # "Male" or "Female"
    SeniorCitizen: int          # 0 or 1
    Partner: str                # "Yes" or "No"
    Dependents: str             # "Yes" or "No"
    tenure: int
    PhoneService: str           # "Yes" or "No"
    MultipleLines: str          # "Yes", "No", "No phone service"
    InternetService: str        # "DSL", "Fiber optic", "No"
    OnlineSecurity: str         # "Yes", "No", "No internet service"
    OnlineBackup: str           # "Yes", "No", "No internet service"
    DeviceProtection: str       # "Yes", "No", "No internet service"
    TechSupport: str            # "Yes", "No", "No internet service"
    StreamingTV: str            # "Yes", "No", "No internet service"
    StreamingMovies: str        # "Yes", "No", "No internet service"
    Contract: str               # "Month-to-month", "One year", "Two year"
    PaperlessBilling: str       # "Yes" or "No"
    PaymentMethod: str          # "Electronic check", "Mailed check", etc.
    MonthlyCharges: float
    TotalCharges: float

    @model_validator(mode='after')
    def validate_placeholders(self):
        # Loop through all input fields sent by the user
        for field_name, value in self.__dict__.items():
            if isinstance(value, str):
                cleaned_value = value.strip().lower()
                
                # Reject default Swagger UI "string" placeholders
                if cleaned_value == "string":
                    raise ValueError(
                        f"Field '{field_name}' contains default placeholder 'string'. Please provide actual customer data."
                    )
                
                # Reject empty strings or plain whitespaces
                if cleaned_value == "":
                    raise ValueError(
                        f"Field '{field_name}' cannot be empty. Please provide a valid value."
                    )
                    
        return self

@app.post("/predict")
def predict_churn(customers: list[CustomerData]):
    # 1. Guard rail for empty list requests
    if not customers:
        raise HTTPException(status_code=400, detail="The request body must contain at least one customer record.")
        
    # 2. Convert list of incoming data items to a single multi-row Pandas DataFrame
    input_df = pd.DataFrame([customer.model_dump() for customer in customers])
    
    # 3. Replicate the data cleaning steps from train.py (vectorized across all rows)
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        input_df[col] = input_df[col].map({'Yes': 1, 'No': 0})
    input_df['gender'] = input_df['gender'].map({'Male': 1, 'Female': 0})
    
    # One-hot encode the categorical variables
    categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 
                        'Contract', 'PaymentMethod']
    input_df = pd.get_dummies(input_df, columns=categorical_cols)
    
    # Reindex columns to match the exact order and structure expected by the model
    input_df = input_df.reindex(columns=model_features, fill_value=0)
    
    # 4. Generate Batch Predictions
    predictions = model.predict(input_df)
    probabilities = model.predict_proba(input_df)[:, 1]
    
    # 5. Map outputs back into a clean structured list response
    batch_results = []
    for pred, prob in zip(predictions, probabilities):
        pred_int = int(pred)
        prob_float = float(prob)
        batch_results.append({
            "churn_prediction": pred_int,
            "churn_probability": round(prob_float, 4),
            "status": "High Churn Risk" if pred_int == 1 else "Low Churn Risk"
        })
        
    return {"predictions": batch_results}