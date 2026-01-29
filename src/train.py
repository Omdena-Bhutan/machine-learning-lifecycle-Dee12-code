import os
import joblib
from data_loader import load_data
from model import get_model

# Load dataset
X, y = load_data()

# Train model
clf = get_model()
clf.fit(X.reshape(-1, 1), y)  # reshape for dummy model

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(clf, "models/model.pkl")
print("Model saved to models/model.pkl")
