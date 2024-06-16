from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import List

# Define the FastAPI app
app = FastAPI()

# Load the trained model
model = joblib.load('stroke_prediction_model.pkl')


# Define input data schema using Pydantic BaseModel
class InputData(BaseModel):
    id: int
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    Residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str


# Define output data schema
class OutputData(BaseModel):
    id: int
    stroke_probability: float


# Endpoint to predict stroke
@app.get("/predict_stroke", response_model=OutputData)
async def predict_stroke(data: InputData):
    try:
        # Create a DataFrame from input data for prediction
        input_data = pd.DataFrame([data.dict()])

        # Perform any necessary preprocessing (e.g., encoding categorical variables)
        # Ensure the preprocessing steps match those used during training

        # Make predictions
        probabilities = model.predict_proba(input_data.drop(['id'], axis=1))

        # Get the probability of stroke (class 1)
        stroke_probability = probabilities[0][1]  # probabilities[0] returns [prob_class_0, prob_class_1]

        # Return the prediction probability
        return {"id": data.id, "stroke_probability": stroke_probability}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the FastAPI server with uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
