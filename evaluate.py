"""
Evaluate the Smart Expense Categorizer model performance.
Tests accuracy, per-class precision/recall, and confusion matrix.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from categorize import build_training_data, train_model, preprocess

def evaluate():
    print("=" * 60)
    print("  SMART EXPENSE CATEGORIZER — MODEL EVALUATION")
    print("=" * 60)

    texts, labels = build_training_data()
    texts_clean = [preprocess(t) for t in texts]

    model = train_model()

    # Cross-validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, texts_clean, labels, cv=skf, scoring="accuracy")

    print(f"\n📊 5-Fold Cross-Validation Accuracy:")
    print(f"   Mean:  {scores.mean():.4f}")
    print(f"   Std:   {scores.std():.4f}")
    print(f"   Scores: {[round(s, 4) for s in scores]}")

    # Full fit evaluation
    preds = model.predict(texts_clean)
    print("\n📋 Classification Report (full training set):")
    print(classification_report(labels, preds, zero_division=0))

    # Sample predictions
    print("\n🔍 Sample Predictions:")
    test_cases = [
        "zomato food delivery",
        "uber cab ride",
        "amazon order shoes",
        "electricity board payment",
        "apollo pharmacy medicine",
        "emi home loan sbi",
        "bigbasket vegetables",
        "pvr movie ticket",
        "udemy machine learning course",
        "atm cash withdrawal",
    ]
    for tc in test_cases:
        pred = model.predict([preprocess(tc)])[0]
        proba = model.predict_proba([preprocess(tc)]).max()
        print(f"  {tc:<40} → {pred:<25} ({proba:.2%})")

if __name__ == "__main__":
    evaluate()
