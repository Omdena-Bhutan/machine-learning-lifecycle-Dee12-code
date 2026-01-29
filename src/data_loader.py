import pandas as pd

def load_data(path="data/raw/IMDB Dataset.csv"):
    data = pd.read_csv(path)
    X = data['review'].values
    y = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0).values
    return X, y
