import os
import joblib
from data_loader import load_data
from model import get_model
import mlflow
import numpy as nppython src/train.py

X = np.array([1,2,3,4,5]).reshape(-1,1)
y = np.array([2,4,6,8,10])

with mlflow.start_run():
    model = get_model()
    model.fit(X.reshape(-1, 1), y)
    # Load dataset
    X, y = load_data()

    # Train model
    clf = get_model()
    clf.fit(X.reshape(-1, 1), y)  # reshape for dummy model


    # Log parameters
    mlflow.log_param("dataset_size", len(X))

    # Log metrics
    mlflow.log_metric("accuracy", 0.5)

    # Log model
    mlflow.sklearn.log_model(model, "model")
    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, "models/model.pkl")
    print("Model saved to models/model.pkl")
