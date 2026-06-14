"""
AWS Lambda handler — Fraud Detection Inference
------------------------------------------------
Loads model from S3 on cold start, then serves predictions.

Request body (JSON):
{
    "amt": 150.0,
    "hour": 22,
    "day_of_week": 4,
    "month": 6,
    "category": "shopping_net",
    "city_pop": 45000,
    "lat": 37.77,
    "long": -122.41,
    "merch_lat": 37.80,
    "merch_long": -122.43
}

Response:
{
    "fraud_probability": 0.87,
    "prediction": "FRAUD",
    "risk_level": "HIGH"
}
"""

import json
import os
import io
import boto3
import joblib
import numpy as np

# Loaded once on cold start, reused across warm invocations
_MODEL_BUNDLE = None

S3_BUCKET = os.environ.get("S3_BUCKET", "fraud-detector-rishabh")
S3_KEY    = os.environ.get("S3_KEY",    "model/fraud_model.joblib")


def load_model():
    global _MODEL_BUNDLE
    if _MODEL_BUNDLE is None:
        print(f"Cold start: loading model from s3://{S3_BUCKET}/{S3_KEY}")
        s3  = boto3.client("s3")
        buf = io.BytesIO()
        s3.download_fileobj(S3_BUCKET, S3_KEY, buf)
        buf.seek(0)
        _MODEL_BUNDLE = joblib.load(buf)
        print("Model loaded.")
    return _MODEL_BUNDLE


def lambda_handler(event, context):
    try:
        # Parse input
        body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event

        bundle = load_model()
        model  = bundle["model"]
        le     = bundle["label_encoder"]

        # Encode category
        category = body.get("category", "misc_pos")
        if category in le.classes_:
            category_enc = int(le.transform([category])[0])
        else:
            category_enc = 0  # fallback for unknown categories

        features = np.array([[
            float(body.get("amt",         0)),
            int(body.get("hour",          12)),
            int(body.get("day_of_week",   0)),
            int(body.get("month",         1)),
            category_enc,
            float(body.get("city_pop",    50000)),
            float(body.get("lat",         37.0)),
            float(body.get("long",        -95.0)),
            float(body.get("merch_lat",   37.0)),
            float(body.get("merch_long",  -95.0)),
        ]])

        prob       = float(model.predict_proba(features)[0][1])
        prediction = "FRAUD" if prob >= 0.5 else "LEGIT"
        risk       = "HIGH" if prob >= 0.7 else "MEDIUM" if prob >= 0.4 else "LOW"

        result = {
            "fraud_probability": round(prob, 4),
            "prediction":        prediction,
            "risk_level":        risk,
        }

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result),
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": str(e)}),
        }
