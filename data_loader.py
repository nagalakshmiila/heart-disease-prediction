"""
data_loader.py
--------------
Automatically downloads the UCI Cleveland Heart Disease dataset
(303 patients, 13 clinical features + target) and cleans it.

Download strategy (in order, first one that works is used):
  1. Official 'ucimlrepo' package (dataset id=45) -> straight from UCI servers.
  2. A mirrored raw CSV on GitHub (used as backup if UCI servers are down /
     blocked on your network).
  3. If both fail (e.g. no internet), a small synthetic fallback dataset with
     the same 13 columns is generated so the rest of the pipeline can still
     be demoed/tested. (Never use this fallback for your real report numbers.)

Run directly to test:
    python data_loader.py
"""

import os
import numpy as np
import pandas as pd

RAW_PATH = "data/heart_raw.csv"
CLEAN_PATH = "data/heart_processed.csv"

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

BACKUP_CSV_URL = (
    "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/"
    "master/heart.csv"
)


def _download_via_ucimlrepo() -> pd.DataFrame:
    from ucimlrepo import fetch_ucirepo
    heart_disease = fetch_ucirepo(id=45)
    X = heart_disease.data.features.copy()
    y = heart_disease.data.targets.copy()
    df = pd.concat([X, y], axis=1)
    df.columns = COLUMNS
    return df


def _download_via_backup_csv() -> pd.DataFrame:
    df = pd.read_csv(BACKUP_CSV_URL)
    # this mirror already uses a binarized 'target' column and same order
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"num": "target"})
    return df[COLUMNS]


def _generate_synthetic(n=303) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "age": rng.integers(29, 78, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(0, 4, n),
        "trestbps": rng.integers(94, 201, n),
        "chol": rng.integers(126, 565, n),
        "fbs": rng.integers(0, 2, n),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.integers(71, 203, n),
        "exang": rng.integers(0, 2, n),
        "oldpeak": np.round(rng.uniform(0, 6.2, n), 1),
        "slope": rng.integers(0, 3, n),
        "ca": rng.integers(0, 4, n),
        "thal": rng.integers(0, 3, n),
    })
    score = (df.age > 54).astype(int) + (df.cp >= 2) + (df.chol > 240) + \
            (df.thalach < 140) + (df.exang == 1) + (df.oldpeak > 1.5)
    df["target"] = (score >= 3).astype(int)
    return df


def download_dataset() -> pd.DataFrame:
    os.makedirs("data", exist_ok=True)
    try:
        print("[data_loader] Trying official UCI repo (ucimlrepo)...")
        df = _download_via_ucimlrepo()
        print("[data_loader] Success via ucimlrepo.")
    except Exception as e1:
        print(f"[data_loader] ucimlrepo failed ({e1}). Trying backup CSV mirror...")
        try:
            df = _download_via_backup_csv()
            print("[data_loader] Success via backup CSV mirror.")
        except Exception as e2:
            print(f"[data_loader] Backup mirror failed too ({e2}).")
            print("[data_loader] No internet reachable -> generating synthetic "
                  "placeholder data so you can still test the pipeline.")
            df = _generate_synthetic()
    df.to_csv(RAW_PATH, index=False)
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # UCI files mark missing values with '?'
    df = df.replace("?", np.nan)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # impute numeric NaNs with column median (ca, thal usually have a few)
    df = df.fillna(df.median(numeric_only=True))
    # target in raw UCI data is 0-4 severity -> binarize: 0 = no disease, 1 = disease
    df["target"] = (df["target"] > 0).astype(int)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def load_and_prepare() -> pd.DataFrame:
    if os.path.exists(CLEAN_PATH):
        print(f"[data_loader] Found cached cleaned dataset at {CLEAN_PATH}")
        return pd.read_csv(CLEAN_PATH)
    raw = download_dataset()
    clean = clean_dataset(raw)
    clean.to_csv(CLEAN_PATH, index=False)
    print(f"[data_loader] Saved cleaned dataset -> {CLEAN_PATH} "
          f"({clean.shape[0]} rows, {clean.shape[1]} columns)")
    return clean


if __name__ == "__main__":
    data = load_and_prepare()
    print(data.head())
    print("\nClass balance:\n", data["target"].value_counts())
