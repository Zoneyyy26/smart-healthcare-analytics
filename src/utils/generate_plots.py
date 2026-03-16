import os
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils.preprocess import load_dataset, split_features_target, normalize_feature_frame

def main():
    docs_dir = Path("docs/images")
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Data & Correlation Matrix
    df = load_dataset("data/diabetes.csv")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.savefig(docs_dir / "correlation_matrix.png")
    plt.close()
    
    # Data distribution of Outcome
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Outcome", data=df, palette="Set2")
    plt.title("Distribution of Target Variable (Outcome)")
    plt.tight_layout()
    plt.savefig(docs_dir / "target_distribution.png")
    plt.close()

    # 2. Get evaluation Data
    x, y = split_features_target(df, "Outcome")
    x = normalize_feature_frame(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Load Model
    model_artifact = joblib.load("models/diabetes_model.joblib")
    model = model_artifact["model"]
    
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]
    
    # 4. Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Negative", "Positive"], 
                yticklabels=["Negative", "Positive"])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(docs_dir / "confusion_matrix.png")
    plt.close()
    
    # 5. ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(docs_dir / "roc_curve.png")
    plt.close()
    
    print("All plots generated successfully in docs/images/")

if __name__ == "__main__":
    main()
