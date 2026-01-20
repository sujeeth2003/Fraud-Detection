from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from preprocess import load_data, preprocess_data

def train():
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Training Complete ✅")
    print(classification_report(y_test, y_pred))
    return model

if __name__ == "__main__":
    train()
