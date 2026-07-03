"""Train a linear model for Battlesnake move selection.

Usage:
    python train.py experiments/data/battlesnake_data_*.csv
    python train.py experiments/data/battlesnake_data_*.parquet

The script reads per-move feature rows, fits a linear model that predicts
whether a move was chosen (binary classification), and writes the resulting
checkpoint to ``model.json`` in the project root.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


FEATURE_COLS = [
    "space_capped",
    "open_space",
    "voronoi",
    "reaches_tail",
    "escape",
    "h2h_danger",
    "near_bigger_head",
    "near_enemy_head",
    "wall_dist",
    "food_score",
    "food_delta",
    "is_food",
    "dist_to_center",
    "adjacent_food",
    "avg_enemy_dist",
]

# Map from unprefixed names to possible prefixed column names
_FEAT_PREFIX = "feat_"


def load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        print(f"❌ File not found: {path}")
        sys.exit(1)
    if p.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df


def _resolve_feature_cols(df: pd.DataFrame) -> list:
    """Resolve feature column names, handling both prefixed and unprefixed."""
    cols = []
    for col in FEATURE_COLS:
        if col in df.columns:
            cols.append(col)
        elif f"{_FEAT_PREFIX}{col}" in df.columns:
            cols.append(f"{_FEAT_PREFIX}{col}")
        else:
            print(f"⚠️  Missing feature column: {col}")
    return cols


def train(df: pd.DataFrame) -> dict:
    """Fit a logistic-regression model and return a checkpoint dict."""
    # Resolve feature columns (handle feat_ prefix)
    feature_cols = _resolve_feature_cols(df)
    
    if not feature_cols:
        print("❌ No feature columns found in data.")
        sys.exit(1)

    # Keep only rows where we have all features
    df = df.dropna(subset=feature_cols)

    if len(df) == 0:
        print("❌ No data available after dropping missing features.")
        sys.exit(1)

    X = df[feature_cols].values
    y = df["chosen"].astype(int).values

    # Use class_weight='balanced' because chosen moves are rare
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    model.fit(X, y)

    # Extract coefficients and intercept from the pipeline
    lr = model.named_steps["logisticregression"]
    scaler = model.named_steps["standardscaler"]

    coef = lr.coef_[0].tolist()
    intercept = float(lr.intercept_[0])
    mean = scaler.mean_.tolist()
    std = scaler.scale_.tolist()

    # Compute top-1 accuracy on training data
    proba = model.predict_proba(X)
    top1 = (proba.argmax(axis=1) == y).mean()

    checkpoint = {
        "feature_names": FEATURE_COLS,
        "mean": mean,
        "std": std,
        "coef": coef,
        "intercept": intercept,
        "top1_accuracy": float(top1),
    }

    print(f"✅ Trained on {len(df)} samples")
    print(f"   Top-1 accuracy: {top1:.4f}")
    print(f"   Chosen moves: {y.sum()} / {len(y)}")
    return checkpoint


def save_checkpoint(checkpoint: dict, out_path: str = "model.json"):
    with open(out_path, "w") as f:
        json.dump(checkpoint, f, indent=2)
    print(f"✅ Checkpoint saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Train Battlesnake move model")
    parser.add_argument("data", help="Path to CSV or Parquet data file")
    parser.add_argument("-o", "--output", default="model.json", help="Output checkpoint path")
    args = parser.parse_args()

    df = load_data(args.data)
    checkpoint = train(df)
    save_checkpoint(checkpoint, args.output)


if __name__ == "__main__":
    main()
