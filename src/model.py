from sklearn.dummy import DummyClassifier

def get_model():
    model = DummyClassifier(strategy="most_frequent")
    return model
