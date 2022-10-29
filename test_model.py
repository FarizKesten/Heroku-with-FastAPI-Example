import pytest
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from ml.model import compute_model_metrics
from ml.pipeline import run_pipeline


@pytest.fixture
def data():
    return pd.read_csv("data/census_clean.csv", skipinitialspace=True)

@pytest.fixture
def sample(data):
    _, sample = train_test_split(data, test_size=0.20)
    return sample

@pytest.fixture
def model():
    return joblib.load("model/model.joblib")

@pytest.fixture
def lb():
    return joblib.load("model/lb.joblib")

@pytest.fixture
def encoder():
    return joblib.load("model/encoder.joblib")

def test_can_run_inference(sample, model, lb, encoder):
    run_pipeline(sample, model, lb, encoder)

def test_pred_range(sample, model, lb, encoder):
    y, preds = run_pipeline(sample, model, lb, encoder)
    assert min(preds) >= 0 , "min value has to be greater equal to 0"
    assert max(preds) <= 1 , "max value has to be smaller equal to 1"

def test_pred_precision(sample, model, lb, encoder):
    y, preds = run_pipeline(sample, model, lb, encoder)
    precision, _, _ = compute_model_metrics(y, preds)
    print(precision)
    assert precision > 0.9, "precision must be higher then 0.9"



