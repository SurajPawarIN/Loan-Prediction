import pytest
import sys
from flask import Flask, render_template, request, jsonify
import joblib

# Add your app code import here
sys.path.append("..")  # Adjust the path to your app code
from .. import app  # Assuming your Flask app instance is named 'app'

# Load the loan prediction model
model = joblib.load("loan_prediction_model.pkl")

@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Loan Prediction" in response.data

def test_predict(client):
    data = {
        "gender": "Male",
        "married": "Yes",
        "dependents": "0",
        "education": "Graduate",
        "self_employed": "No",
        "property_area": "Urban",
        "applicant_income": 5000,
        "coapplicant_income": 2000,
        "loan_amount": 150,
        "loan_amount_term": 360,
        "credit_history": 1,
    }

    response = client.post("/predict", data=data)
    assert response.status_code == 200
    assert b"Prediction Results:" in response.data
    assert b"Predicted Probability:" in response.data
    assert b"Predicted Class:" in response.data

    # Clean up any modifications to the app context after the test
    with app.app_context():
        pass
