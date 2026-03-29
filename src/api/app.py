from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load model at startup
model = joblib.load("models/xgb_model.pkl")

FEATURES = [
    'store', 'item',
    'day_of_week', 'month', 'year', 'day_of_month',
    'lag_1', 'lag_7', 'lag_14', 'lag_28', 'lag_56', 'lag_84',
    'rolling_mean_7', 'rolling_std_7',
    'rolling_mean_14', 'rolling_mean_28',
    'rolling_max_7'
]


@app.get("/")
def home():
    return {"message": "Demand Forecast API is running"}



def build_features_from_history(data):
    import numpy as np
    import pandas as pd

    sales = data["recent_sales"]

    df = pd.DataFrame()

    df["store"] = [data["store"]]
    df["item"] = [data["item"]]

    # Use last date assumption (you can improve later)
    df["day_of_week"] = [0]
    df["month"] = [1]
    df["year"] = [2017]
    df["day_of_month"] = [1]

    # Lag features
    df["lag_1"] = [sales[-1]]
    df["lag_7"] = [sales[-7]]
    df["lag_14"] = [sales[-14]]
    df["lag_28"] = [sales[-28]]
    df["lag_56"] = [sales[-56]]
    df["lag_84"] = [sales[-84]]

    # Rolling features
    df["rolling_mean_7"] = [np.mean(sales[-7:])]
    df["rolling_std_7"] = [np.std(sales[-7:])]

    df["rolling_mean_14"] = [np.mean(sales[-14:])]
    df["rolling_mean_28"] = [np.mean(sales[-28:])]

    df["rolling_max_7"] = [np.max(sales[-7:])]

    return df



@app.post("/predict")
def predict(data: dict):

    df = build_features_from_history(data)

    prediction = model.predict(df[FEATURES])[0]

    return {
        "prediction": float(prediction)
    }