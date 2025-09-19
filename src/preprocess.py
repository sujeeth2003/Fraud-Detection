import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path="../data/transactions.csv"):
    return pd.read_csv(path)

def preprocess_data(df):
    X = df.drop("is_fraud", axis=1)
    y = df["is_fraud"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
