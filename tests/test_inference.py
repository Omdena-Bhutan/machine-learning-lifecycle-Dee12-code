# tests/test_inference.py
from src.inference import predict  # Your predict() function

def test_predict_output():
    text = "This movie was amazing!"
    result = predict(text)
    
    # Replace with the labels your model actually outputs
    valid_labels = ["Positive", "Negative"]
    
    assert result in valid_labels
