import joblib
from data_loader import load_data
from sklearn.metrics import accuracy_score

# Load dataset and model
X, y = load_data()
model = joblib.load("models/model.pkl")

# Evaluate
y_pred = model.predict(X.reshape(-1, 1))
acc = accuracy_score(y, y_pred)
print(f"Accuracy: {acc}")
