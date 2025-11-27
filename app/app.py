# file: app.py
from fastapi import FastAPI
from pydantic import BaseModel
from utils.predict_utils import predict_user_input
from utils.encode_utils import decode_value

app = FastAPI(title="ML Prediction API")

class UserInput(BaseModel):
    Age: float
    Height: float
    Weight: float
    Sex_encoded: str
    Hypertension_encoded: str
    Diabetes_encoded: str
    Fitness_Goal_encoded: str

def to_python_type(val):
    try:
        return val.item() 
    except AttributeError:
        return val

@app.post("/predict")
def predict(user_input: UserInput):
    user_input_dict = user_input.dict()
    predictions = predict_user_input(user_input_dict)

    predictions_converted = {}
    decoded_predictions = {}
    for col, val in predictions.items():
        val_native = to_python_type(val)
        predictions_converted[col] = val_native
        decoded_predictions[col] = decode_value(col, val_native)

    return {
        "predictions_encoded": predictions_converted,
        "predictions_decoded": decoded_predictions
    }
