# tests/test_train.py
import numpy as np
from sklearn.linear_model import LinearRegression

def test_model_training():
    # Sample data
    X = np.array([1,2,3,4,5]).reshape(-1,1)
    y = np.array([2,4,6,8,10])
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Check model coefficients
    assert model.coef_.shape == (1,)
    
    # Check model predicts numbers
    predictions = model.predict(X)
    assert all(isinstance(p, float) for p in predictions)
