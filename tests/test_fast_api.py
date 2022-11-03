import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    api_client = TestClient(app)
    return api_client


def test_get(client):
    request = client.get("/")
    assert request.status_code == 200
    assert request.json() == {"greeting": "Welcome!!"}


def test_post_high(client):

    request = client.post("/", json={
                                     'age': 40,
                                     'workclass': 'Private',
                                     'fnlgt': 140000,
                                     'education': 'Doctorate',
                                     'marital-status': 'Never-married',
                                     'occupation': 'Prof-specialty',
                                     'relationship': 'Not-in-family',
                                     'race': 'White',
                                     'sex': 'Male',
                                     'hours-per-week': 60,
                                     'native-country': 'United-States'
                                     })
    assert request.status_code == 200
    assert request.json() == {"prediction": ">50K"}

def test_post_low(client):

    request = client.post("/", json={
                                     'age': 20,
                                     'workclass': 'Never-worked',
                                     'fnlgt': 140000,
                                     'education': 'Preschool',
                                     'marital-status': 'Never-married',
                                     'occupation': 'Prof-specialty',
                                     'relationship': 'Not-in-family',
                                     'race': 'White',
                                     'sex': 'Male',
                                     'hours-per-week': 60,
                                     'native-country': 'United-States'
                                     })
    assert request.status_code == 200
    assert request.json() == {"prediction": "<=50K"}