# Fraud Detection Notebook - Day 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ---- Generate Synthetic Data ----
X, y = make_classification(n_samples=2000, n_features=6, 
                           n_classes=2, weights=[0.95, 0.05], 
                           random_state=42)

columns = ["amount", "time", "location_id", "merchant_id", "device_id", "user_id"]
df = pd.DataFrame(X, columns=columns)
df["is_fraud"] = y
df.to_csv("../data/transactions.csv", index=False)
print("✅ Synthetic dataset saved to data/transactions.csv")

# ---- Train/Test Split ----
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop("is_fraud", axis=1),
                                                    df["is_fraud"],
                                                    test_size=0.2,
                                                    random_state=42)

# ---- Model Training ----
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---- Evaluation ----
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues")
plt.show()
