"""
Find a real loan from the test set that the model REJECTED (predicted as default).

This gives us a realistic example with all features properly populated.
"""

import joblib
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "artifacts" / "loan_classifier_final.pkl"
DATA_PATH = BASE_DIR / "data" / "loan.csv"  # Original data


def main():
    print("Loading model...")
    artifacts = joblib.load(MODEL_PATH)
    model = artifacts["model"]
    threshold = artifacts["threshold"]
    feature_names = artifacts["features"]
    
    print(f"Model threshold: {threshold:.3f}")
    print(f"Expected features: {len(feature_names)}")
    
    # Load a sample of the original data
    print("\nLoading data sample...")
    df = pd.read_csv(DATA_PATH, nrows=50000, low_memory=False)
    
    # Apply same preprocessing as in test_2.ipynb
    # (This is simplified - you'd need to replicate the exact preprocessing pipeline)
    print("\nNote: This script needs the exact preprocessing from test_2.ipynb")
    print("For now, let's use the saved model's feature list to find a rejected example.")
    
    # Better approach: Load from test_2 notebook's X_test, y_test if available
    # Or use the cleaned dataset if it has the same features
    
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("="*60)
    print("Run this in test_2.ipynb after training:")
    print("""
# Find a rejected example
test_probs = xgb_proper.predict_proba(X_test)[:, 1]
test_preds = (test_probs >= best_threshold).astype(int)

# Find loans the model REJECTED (predicted as default)
rejected_mask = (test_preds == 1)
rejected_examples = X_test[rejected_mask].copy()
rejected_probs = test_probs[rejected_mask]

# Pick one with high probability (clearly risky)
idx = rejected_probs.argmax()
rejected_loan = rejected_examples.iloc[idx:idx+1]

print(f"Example index: {rejected_examples.index[idx]}")
print(f"Predicted default prob: {rejected_probs[idx]:.3f}")
print(f"Threshold: {best_threshold:.3f}")
print(f"Decision: REJECTED")

# Convert to dict for API
feature_dict = rejected_loan.iloc[0].to_dict()
print("\\nFeature dict (for /predict):")
print(json.dumps({k: float(v) if pd.notna(v) else None for k, v in feature_dict.items()}, indent=2))
    """)


if __name__ == "__main__":
    main()

