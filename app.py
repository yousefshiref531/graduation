from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from sklearn.base import BaseEstimator, TransformerMixin

# =========================== #
#  Custom Feature Transformer #
# =========================== #
class FeatureEngineer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.temp_cols = ['Temp1','Temp2','BigF_T','Side_T','Center_T']
        self.pres_cols = ['Pres1','Pres2','BigF_P','Side_P','Center_P']

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Basic
        df["Temp_mean"] = df[self.temp_cols].mean(axis=1)
        df["Pres_mean"] = df[self.pres_cols].mean(axis=1)

        df["Temp_max"] = df[self.temp_cols].max(axis=1)
        df["Temp_min"] = df[self.temp_cols].min(axis=1)
        df["Temp_range"] = df["Temp_max"] - df["Temp_min"]

        df["Pres_max"] = df[self.pres_cols].max(axis=1)
        df["Pres_min"] = df[self.pres_cols].min(axis=1)
        df["Pres_range"] = df["Pres_max"] - df["Pres_min"]

        # Interaction features
        for t, p in zip(self.temp_cols, self.pres_cols):
            df[f"{t}_x_{p}"] = df[t] * df[p]
            df[f"{t}_div_{p}"] = df[t] / (df[p] + 1)

        # Differences
        df["Pres_diff_big_center"] = df["BigF_P"] - df["Center_P"]
        df["Pres_diff_side_center"] = df["Side_P"] - df["Center_P"]

        df["Temp_diff_big_center"] = df["BigF_T"] - df["Center_T"]
        df["Temp_diff_side_center"] = df["Side_T"] - df["Center_T"]

        # Total pressure
        df["Total_pressure"] = df[self.pres_cols].sum(axis=1)

        return df


# ================= #
# FIX PICKLE LOADING
# ================= #

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'FeatureEngineer':       # <- مهم
            return FeatureEngineer
        return super().find_class(module, name)

with open("medical_model.pkl", "rb") as f:
    model = CustomUnpickler(f).load()


# ============ #
# FastAPI App  #
# ============ #
app = FastAPI()

@app.get("/")
def root():
    return {"message": "🚀 API is Running Successfully!"}


class InputData(BaseModel):
    Temp1: float
    Temp2: float
    BigF_T: float
    Side_T: float
    Center_T: float
    Pres1: float
    Pres2: float
    BigF_P: float
    Side_P: float
    Center_P: float


@app.post("/predict")
def predict(data: InputData):
    try:
        df = pd.DataFrame([data.dict()])
        prediction = model.predict(df)
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
