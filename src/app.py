from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from catboost import CatBoostRegressor

app = FastAPI(
    title="Demand Forecasting API",
    description="API для прогнозирования спроса на основе модели CatBoost",
    version="1.0.0"
)

MODEL_PATH = "models/catboost_demand_model.cbm"

model = CatBoostRegressor()
model.load_model(MODEL_PATH)


class PredictionRequest(BaseModel):
    rolling_mean_7: float
    rolling_mean_14: float
    day_of_week: int
    item: int
    month: int
    store: int
    is_weekend: int
    year: int
    sales_lag_1: float
    sales_lag_7: float


@app.get("/")
def read_root():
    return {"status": "healthy", "message": "API is up and running"}


@app.post("/predict")
def predict_demand(request: PredictionRequest):
    try:
        features = [
            request.store,
            request.item,
            request.day_of_week,
            request.month,
            request.year,
            request.is_weekend,
            request.sales_lag_1,
            request.sales_lag_7,
            request.rolling_mean_7,
            request.rolling_mean_14
        ]

        prediction = model.predict([features])[0]

        final_prediction = max(0, float(prediction))

        return {
            "store": request.store,
            "item": request.item,
            "predicted_sales": round(final_prediction, 2)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")