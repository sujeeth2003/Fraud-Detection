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

