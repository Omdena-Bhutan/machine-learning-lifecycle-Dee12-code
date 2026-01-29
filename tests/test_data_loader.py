# tests/test_data_loader.py
import os
import numpy as np
from src.data import load_data

def test_load_data_returns_arrays():
    # Adjust path if your dataset is elsewhere
    file_path = os.path.join("data", "IMDB Dataset.csv")
    X, y = load_data(file_path)
    
    # Check types
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    
    # Check lengths match
    assert len(X) == len(y)
    
    # Optional: check for non-empty arrays
    assert len(X) > 0
    assert len(y) > 0
