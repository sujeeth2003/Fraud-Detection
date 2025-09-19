# 🏦 Fraud Detection System (FinTech Project)

## 📌 Overview
This project demonstrates a **fraud detection pipeline** for financial transactions using **machine learning**.  
It generates synthetic data, preprocesses it, trains a model, and evaluates fraud classification performance.

## ⚙️ Tech Stack
- Python (Pandas, NumPy, Scikit-learn, Matplotlib)
- Jupyter Notebook
- Random Forest Classifier

## 📂 Project Structure
Fraud detection/
│── data/ # transaction dataset
│── notebooks/ # exploratory data analysis
│── src/ # preprocessing, training, evaluation
│── requirements.txt # dependencies
│── README.md # overview

## 🚀 How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Run notebook
jupyter notebook notebooks/fraud_detection.ipynb

# Or run training script
cd src
python train_model.py
