"""
Workflow Structure: 
- Import Libraries
- Prepare Data and load it
- Split Data (Train, Test, and Validation)
- Function to find the optimal "Threshold" --> Why would you need that? How does the right threshold contribute to better training / model performance?
- Train the model based on the stratified training data
- Test the trained model based on predicting labels for the unseen test data
- Calculate accuracy metrics of the model (Precision, Recall, F2-Score, PR-AUC)
- Save the created artifacts such as the trained model and the scores in folder of the parent path
"""


import argparse
import json
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, fbeta_score,
    precision_score, recall_score
)

# Importing functions from 'piplines.py' and 'utils.py'
from pipelines import get_pipeline
from utils import set_seed, save_threshold

def load_and_prepare_data(path: str):
    """Lead CSV and prepare features/target."""
    df = pd.read_csv(path)
    
    # Convert Total Charges to numeric (handle strings)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # Convert Monthly Charges to numeric
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce')
    
    # Convert Churn to binary
    df['Churn'] = (df['Churn']== 'Yes').astype(int)
    
    # Drop customer ID
    df = df.drop(columns='customerID', axis=1)
    
    # Separate features and target
    X = df.drop(columns='Churn', axis=1)
    y = df['Churn']    
    
    return X, y

def split_data(X,y, seed=42):
    """Stratified 60/20/20 split."""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=seed
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def find_optimal_threshold(y_true, y_scores, min_precision=0.35):
    """
    Find threshold that maximizes F2 score while maintaining min_precision.
    
    Returns:
        threshold: optimal threshold
        metrics: dict with precision, recall, f2 at threshold
    """
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
    
    # Calculate F2 for each threshold
    f2_scores = []
    for p, r in zip(precisions[:-1], recalls[:-1]):
        if p >= min_precision:
            f2 = fbeta_score([0,1], [0,1], beta=2, average='binary',
                             sample_weight=[1-r, r] if r>0 else None,
                             zero_division=0)
            # Proper F2 calculation
            f2 = (5*p*r) / (4*p+r) if (4*p+r)>0 else 0
            f2_scores.append(f2)
        else:
            f2_scores.append(0)
            
    # Find best threshold
    if max(f2_scores) == 0:
        # Fallback: use threshold that gives min_precision
        valid_idx = np.where(precisions[:1] >= min_precision)[0]
        if len(valid_idx) > 0:
            best_idx = valid_idx[np.argmax(recalls[:-1][valid_idx])]
        else:
            best_idx = 0
            
    else:
        best_idx = np.argmax(f2_scores)
        
    threshold = thresholds[best_idx]
    
    return threshold, {
        'precision': precisions[best_idx],
        'recall': recalls[best_idx],
        'f2': f2_scores[best_idx]        
    }
    
def train_and_evaluate(data_path, model_type='hgb', seed=42, outdir='.'):
    """Main training pipeline."""
    set_seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} model (seed={seed})")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading data...")
    X, y = load_and_prepare_data(data_path)
    print(f"✓ Dataset shape: {X.shape}, Churn rate: {y.mean():.2%}")
    
    # Split
    print("\nSplitting data (60/20/20)...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, seed)
    print(f"✓ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train
    print(f"\nTraining {model_type} pipeline...")
    pipeline = get_pipeline(model_type)
    pipeline.fit(X_train, y_train)
    print("✓ Training complete")
    
    # Validate
    print("\nValidation metrics:")
    y_val_scores = pipeline.predict_proba(X_val)[:, 1]
    pr_auc_val = average_precision_score(y_val, y_val_scores)
    roc_auc_val = roc_auc_score(y_val, y_val_scores)
    print(f"  PR-AUC:  {pr_auc_val:.4f}")
    print(f"  ROC-AUC: {roc_auc_val:.4f}")
    
    # Find optimal threshold
    print("\nFinding optimal threshold (Precision >= 0.35)...")
    threshold, th_metrics = find_optimal_threshold(y_val, y_val_scores)
    print(f"✓ Optimal threshold: {threshold:.4f}")
    print(f"  Precision: {th_metrics['precision']:.4f}")
    print(f"  Recall:    {th_metrics['recall']:.4f}")
    print(f"  F2:        {th_metrics['f2']:.4f}")
    
    # Test
    print("\nTest set evaluation:")
    y_test_scores = pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_scores >= threshold).astype(int)
    
    precision_test = precision_score(y_test, y_test_pred, zero_division=0)
    recall_test = recall_score(y_test, y_test_pred, zero_division=0)
    f2_test = fbeta_score(y_test, y_test_pred, beta=2, zero_division=0)
    roc_auc_test = roc_auc_score(y_test, y_test_scores)
    
    print(f"  Precision: {precision_test:.4f}")
    print(f"  Recall:    {recall_test:.4f}")
    print(f"  F2:        {f2_test:.4f}")
    print(f"  ROC-AUC:   {roc_auc_test:.4f}")
    
    # Save artifacts
    print("\nSaving artifacts...")
    models_dir = Path(outdir) / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = models_dir / "model.pkl"
    joblib.dump(pipeline, model_path)
    print(f"✓ Model saved: {model_path}")
    
    save_threshold(threshold, models_dir / "threshold.json")
    
    # Log metrics
    exp_dir = Path(outdir) / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics_log = exp_dir / "metrics.csv"
    
    log_entry = {
        'model': model_type,
        'seed': seed,
        'pr_auc_val': pr_auc_val,
        'threshold': threshold,
        'precision_test': precision_test,
        'recall_test': recall_test,
        'f2_test': f2_test,
        'roc_auc_test': roc_auc_test
    }
    
    log_df = pd.DataFrame([log_entry])
    if metrics_log.exists():
        log_df.to_csv(metrics_log, mode='a', header=False, index=False)
    else:
        log_df.to_csv(metrics_log, index=False)
    print(f"✓ Metrics logged: {metrics_log}")
    
    # Summary JSON
    summary = {
        'model': model_type,
        'seed': seed,
        'validation': {
            'pr_auc': float(pr_auc_val),
            'roc_auc': float(roc_auc_val),
            'threshold': float(threshold)
        },
        'test': {
            'precision': float(precision_test),
            'recall': float(recall_test),
            'f2': float(f2_test),
            'roc_auc': float(roc_auc_test)
        },
        'acceptance_check': {
            'pr_auc_val >= 0.75': pr_auc_val >= 0.75,
            'recall_test >= 0.80': recall_test >= 0.80,
            'precision_test >= 0.35': precision_test >= 0.35
        }
    }
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(json.dumps(summary, indent=2))
    print(f"{'='*60}\n")
    
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train telco churn model')
    parser.add_argument('--data', required=True, help='Path to telco.csv')
    parser.add_argument('--model', default='hgb', choices=['dummy', 'logreg', 'hgb'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--outdir', default='.', help='Output directory')
    
    args = parser.parse_args()
    train_and_evaluate(args.data, args.model, args.seed, args.outdir)
