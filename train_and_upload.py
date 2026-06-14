"""
Train a fraud detection model and upload it to S3.
Run with: python train_and_upload.py
"""

import pandas as pd
import numpy as np
import joblib
import boto3
import os
import duckdb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score

CSV_PATH   = "C:/projects/credit_card_transactions.csv"
MODEL_PATH = "fraud_model.joblib"
S3_BUCKET  = "fraud-detector-rishabh"
S3_KEY     = "model/fraud_model.joblib"

# ---------------------------------------------------------------------------
# 1. Load & feature engineer
# ---------------------------------------------------------------------------
print("Loading data...")
df = duckdb.query(f"SELECT * FROM read_csv_auto('{CSV_PATH}')").fetchdf()
df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])

print(f"  {len(df):,} rows loaded")

# Features we'll use
df["hour"]       = df["trans_date_trans_time"].dt.hour
df["day_of_week"] = df["trans_date_trans_time"].dt.dayofweek
df["month"]      = df["trans_date_trans_time"].dt.month

# Encode category
le = LabelEncoder()
df["category_enc"] = le.fit_transform(df["category"])

FEATURES = ["amt", "hour", "day_of_week", "month", "category_enc",
            "city_pop", "lat", "long", "merch_lat", "merch_long"]

X = df[FEATURES].fillna(0)
y = df["is_fraud"]

print(f"  Fraud rate: {y.mean()*100:.2f}%")

# ---------------------------------------------------------------------------
# 2. Train
# ---------------------------------------------------------------------------
print("\nTraining Random Forest...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ---------------------------------------------------------------------------
# 3. Evaluate
# ---------------------------------------------------------------------------
print("\nEvaluation on test set:")
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

# Feature importance
print("\nTop features by importance:")
for feat, imp in sorted(zip(FEATURES, model.feature_importances_),
                         key=lambda x: -x[1]):
    print(f"  {feat:<20} {imp:.4f}")

# ---------------------------------------------------------------------------
# 4. Save model + encoder locally
# ---------------------------------------------------------------------------
print(f"\nSaving model to {MODEL_PATH}...")
joblib.dump({"model": model, "label_encoder": le, "features": FEATURES}, MODEL_PATH)
print("  Done.")

# ---------------------------------------------------------------------------
# 5. Upload to S3
# ---------------------------------------------------------------------------
print(f"\nUploading to s3://{S3_BUCKET}/{S3_KEY}...")
s3 = boto3.client("s3", region_name="us-east-2")
s3.upload_file(MODEL_PATH, S3_BUCKET, S3_KEY)
print("  Upload complete.")
print(f"\n✅ Model live at: s3://{S3_BUCKET}/{S3_KEY}")
