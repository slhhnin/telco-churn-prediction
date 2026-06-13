# train.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import joblib

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    
    # Drop Customer ID as it is a unique identifier
    df.drop(columns=['customerID'], inplace=True)
    
    # Clean TotalCharges missing strings
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'].replace(" ", np.nan), errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Map Binary Text to 1/0
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
            
    # Map Gender
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    
    # One-hot encode remaining multi-category text features
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df

def train_pipeline():
    # Load and clean
    df = load_and_clean_data('Dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # CRUCIAL: Handle Class Imbalance using XGBoost's scale_pos_weight
    # Formula: count(negative cases) / count(positive cases)
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_weight = num_neg / num_pos
    
    print(f"Calculated scale_pos_weight for imbalance: {scale_weight:.2f}")
    
    # Train Model
    model = XGBClassifier(
        scale_pos_weight=scale_weight,
        random_state=42,
        eval_metric='logloss',
        n_estimators=600,
        max_depth=4
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    
    print("\n--- Evaluation Metrics ---")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, probs):.4f}")
    
    # Save the trained model and feature alignment columns
    model_artifacts = {
        'model': model,
        'features': X.columns.tolist()
    }
    joblib.dump(model_artifacts, 'model_artifacts.joblib')
    print("\nModel saved successfully as 'model_artifacts.joblib'!")

if __name__ == "__main__":
    train_pipeline()