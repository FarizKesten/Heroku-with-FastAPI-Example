import pytest
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import inference, compute_model_metrics

@pytest.fixture
def data():
    return pd.read_csv("data/census_clean.csv", skipinitialspace=True)

@pytest.fixture
def sample(data):
    _, sample = train_test_split(data, test_size=0.20)
    return sample

@pytest.fixture
def model():
    encoder = joblib.load("model/encoder.joblib")
    lb = joblib.load("model/lb.joblib")
    model = joblib.load("model/model.joblib")
    return {"lb":lb, "encoder":encoder, "model":model}


def run_inference(sample, model):
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, y, encoder, lb = process_data(sample, categorical_features=cat_features,
        label="salary", training=False, encoder=model["encoder"], lb=model["lb"])
    preds = inference(model["model"], X)
    return (y, preds)

def test_can_run_inference(sample, model):
    run_inference(sample, model)

def test_pred_range(sample, model):
    y, preds = run_inference(sample, model)
    assert min(preds) >= 0 , "min value has to be greater equal to 0"
    assert max(preds) <= 1 , "max value has to be smaller equal to 1"

def test_pred_precision(sample, model):
    y, preds = run_inference(sample, model)
    precision, _, _ = compute_model_metrics(y, preds)
    print(precision)
    assert precision > 0.9, "precision must be higher then 0.9"



