"""Batch scoring script for telco churn MVP."""
import argparse
import pandas as pd
import joblib
from pathlib import Path
from utils import load_threshold


def prepare_scoring_data(df):
    """Prepare data for scoring (same as training prep)."""
    # Convert TotalCharges to numeric
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'].fillna(0, inplace=True)
    
    # Keep customerID if present
    customer_ids = None
    if 'customerID' in df.columns:
        customer_ids = df['customerID'].copy()
        df = df.drop('customerID', axis=1)
    
    # Drop Churn if accidentally present
    if 'Churn' in df.columns:
        df = df.drop('Churn', axis=1)
    
    return df, customer_ids


def batch_predict(input_path, output_path, model_path='models/model.pkl', 
                  threshold_path='models/threshold.json'):
    """
    Run batch predictions on CSV file.
    
    Args:
        input_path: Path to input CSV (without Churn column)
        output_path: Path to save predictions CSV
        model_path: Path to trained model
        threshold_path: Path to threshold JSON
    """
    print(f"\n{'='*60}")
    print("BATCH SCORING")
    print(f"{'='*60}\n")
    
    # Load model and threshold
    print(f"Loading model from {model_path}...")
    pipeline = joblib.load(model_path)
    threshold = load_threshold(threshold_path)
    print(f"✓ Model loaded, threshold: {threshold:.4f}")
    
    # Load data
    print(f"\nLoading data from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"✓ Loaded {len(df)} rows")
    
    # Prepare
    X, customer_ids = prepare_scoring_data(df.copy())
    
    # Predict
    print("\nGenerating predictions...")
    y_scores = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_scores >= threshold).astype(int)
    
    # Create output
    output_df = pd.DataFrame({
        'churn_score': y_scores,
        'churn_pred': y_pred
    })
    
    if customer_ids is not None:
        output_df.insert(0, 'customerID', customer_ids)
    
    # Save
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    
    print(f"✓ Predictions saved: {output_path}")
    print(f"\nSummary:")
    print(f"  Total customers:     {len(output_df)}")
    print(f"  Predicted churners:  {y_pred.sum()} ({y_pred.mean():.1%})")
    print(f"  Mean churn score:    {y_scores.mean():.4f}")
    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch predict telco churn')
    parser.add_argument('--input', required=True, help='Input CSV path')
    parser.add_argument('--output', required=True, help='Output CSV path')
    parser.add_argument('--model', default='models/model.pkl', help='Model path')
    parser.add_argument('--threshold', default='models/threshold.json', help='Threshold path')
    
    args = parser.parse_args()
    batch_predict(args.input, args.output, args.model, args.threshold)