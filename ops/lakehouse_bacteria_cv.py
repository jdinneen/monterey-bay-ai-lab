import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
import sys

def main():
    file_path = 'data/lakehouse/master_bacteria.parquet'
    df = pd.read_parquet(file_path)

    # Ensure sample_date is datetime
    df['sample_date'] = pd.to_datetime(df['sample_date'])

    # Extract temporal features
    df['month'] = df['sample_date'].dt.month
    df['day_of_year'] = df['sample_date'].dt.dayofyear

    # Create target
    df['target'] = (df['bacteria_result'] > 104).astype(int)

    # Features
    features = ['latitude', 'longitude', 'month', 'day_of_year']

    # Chronological holdout
    train_mask = df['sample_date'] < '2022-01-01'
    test_mask = df['sample_date'] >= '2022-01-01'

    X_train = df.loc[train_mask, features]
    y_train = df.loc[train_mask, 'target']
    
    X_test = df.loc[test_mask, features]
    y_test = df.loc[test_mask, 'target']

    if len(X_test) == 0:
        print("Error: No data in test set.")
        sys.exit(1)
        
    if len(X_train) == 0:
        print("Error: No data in train set.")
        sys.exit(1)

    # Calculate majority-class baseline on the Test set
    majority_class = y_train.mode()[0]
    baseline_preds = np.full(len(y_test), majority_class)
    baseline_accuracy = accuracy_score(y_test, baseline_preds)
    
    test_majority_class = y_test.mode()[0]
    test_baseline_preds = np.full(len(y_test), test_majority_class)
    test_baseline_accuracy = accuracy_score(y_test, test_baseline_preds)
    
    print(f"Test Set Total Samples: {len(y_test)}")
    print(f"Test Set Positive Samples (target > 104): {y_test.sum()} ({y_test.mean()*100:.2f}%)")
    print(f"Majority-Class Baseline Accuracy (Test Majority): {test_baseline_accuracy*100:.2f}%")

    # Train model
    # scale_pos_weight
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos_weight = num_neg / num_pos if num_pos > 0 else 1.0

    model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train, y_train)

    # Predict
    preds_proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    # Evaluate
    # Note: if y_test only has one class, ROC AUC will fail, but we assume it has both
    try:
        roc_auc = roc_auc_score(y_test, preds_proba)
    except ValueError:
        roc_auc = np.nan
        
    try:
        pr_auc = average_precision_score(y_test, preds_proba)
    except ValueError:
        pr_auc = np.nan
        
    accuracy = accuracy_score(y_test, preds)

    print("-" * 30)
    print("MODEL METRICS ON TEST SET")
    print("-" * 30)
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC (AP): {pr_auc:.4f}")
    print(f"Accuracy: {accuracy*100:.2f}%")
    
    print("-" * 30)
    if accuracy > test_baseline_accuracy:
        print(f"SUCCESS: Model Accuracy ({accuracy*100:.2f}%) BEATS the Baseline ({test_baseline_accuracy*100:.2f}%)!")
    elif accuracy == test_baseline_accuracy:
        print(f"TIE: Model Accuracy ({accuracy*100:.2f}%) EQUALS the Baseline ({test_baseline_accuracy*100:.2f}%).")
    else:
        print(f"FAILURE: Model Accuracy ({accuracy*100:.2f}%) FAILS TO BEAT the Baseline ({test_baseline_accuracy*100:.2f}%).")

if __name__ == "__main__":
    main()
