import joblib

def predict_review(review):
    model = joblib.load("models/model.pkl")
    return model.predict([[review]])

if __name__ == "__main__":
    sample_review = "This movie was fantastic!"
    print(predict_review(sample_review))
