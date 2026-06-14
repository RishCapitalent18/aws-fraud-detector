# 🚨 Fraud Detection API — AWS Lambda + S3 + CloudWatch

> End-to-end MLOps pipeline: train a fraud detection model, deploy it serverlessly on AWS Lambda, and serve real-time predictions via API.

---

## Architecture

```
CSV Data (1.3M transactions)
        │
        ▼
  train_and_upload.py
  (RandomForest, scikit-learn)
        │
        ▼
  S3 Bucket (model artifact)
  fraud-detector-rishabh/model/fraud_model.joblib
        │
        ▼
  AWS Lambda (Python 3.13)
  - Cold start: loads model from S3
  - Warm: serves predictions in ~270ms
        │
        ▼
  JSON Response
  { fraud_probability, prediction, risk_level }
```

---

## Why This Matters

Building a model is only half the job. This project demonstrates the full production loop:
- **Model training** on 1.3M real transactions with class imbalance handling
- **Artifact management** via S3 (versioned, durable, decoupled from compute)
- **Serverless inference** via Lambda (zero idle cost, auto-scaling)
- **Monitoring** via CloudWatch (latency, memory, invocation logs)

---

## Model Performance

Trained on [synthetic credit card transactions](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset) (1,296,675 rows, 0.58% fraud rate).

| Metric | Value |
|--------|-------|
| ROC-AUC | **0.9943** |
| Fraud Recall | 94% |
| Fraud Precision | 24% |
| Top feature | `amt` (65% importance) |
| 2nd feature | `hour` (21% importance) |

Class imbalance handled with `class_weight="balanced"`. High recall prioritized over precision — better to flag a legitimate transaction than miss fraud.

---

## API Usage

**Invoke via AWS CLI:**

```bash
aws lambda invoke \
  --function-name fraud-detector \
  --payload '{"amt": 25000.0, "hour": 23, "day_of_week": 4, "month": 6, "category": "shopping_net", "city_pop": 45000, "lat": 37.77, "long": -122.41, "merch_lat": 37.80, "merch_long": -122.43}' \
  --region us-east-2 \
  --cli-binary-format raw-in-base64-out \
  output.json && cat output.json
```

**Response:**

```json
{
  "statusCode": 200,
  "body": {
    "fraud_probability": 0.1682,
    "prediction": "LEGIT",
    "risk_level": "LOW"
  }
}
```

**Request fields:**

| Field | Type | Description |
|-------|------|-------------|
| `amt` | float | Transaction amount ($) |
| `hour` | int | Hour of day (0–23) |
| `day_of_week` | int | 0=Monday, 6=Sunday |
| `month` | int | 1–12 |
| `category` | string | Merchant category |
| `city_pop` | int | Cardholder city population |
| `lat`, `long` | float | Cardholder location |
| `merch_lat`, `merch_long` | float | Merchant location |

**Risk levels:**

| Fraud Probability | Prediction | Risk Level |
|-------------------|------------|------------|
| ≥ 0.70 | FRAUD | HIGH |
| 0.40 – 0.69 | FRAUD | MEDIUM |
| < 0.40 | LEGIT | LOW |

---

## AWS Infrastructure

| Component | Service | Detail |
|-----------|---------|--------|
| Model storage | S3 | `fraud-detector-rishabh/model/` |
| Inference | Lambda | Python 3.13, 512MB, 60s timeout |
| Monitoring | CloudWatch | Latency, memory, error logs |
| Permissions | IAM | Least-privilege execution role |

**Lambda performance (warm invocation):**
- Duration: ~270ms
- Memory used: ~261MB
- Cold start: ~700ms (model load from S3)

---

## Project Structure

```
aws-fraud-detector/
├── train_and_upload.py    # Train model + upload to S3
├── lambda_function.py     # Lambda inference handler
├── requirements.txt
└── README.md
```

---

## Setup & Deployment

```bash
# 1. Train and upload model
pip install -r requirements.txt
python train_and_upload.py

# 2. Build deployment package (run in AWS CloudShell)
mkdir package
pip install scikit-learn==1.6.1 --no-deps --target ./package
pip install numpy joblib threadpoolctl scipy narwhals pandas --target ./package
cp lambda_function.py package/
cd package && zip -r ../lambda_package.zip . && cd ..
aws s3 cp lambda_package.zip s3://YOUR_BUCKET/lambda/lambda_package.zip

# 3. Deploy to Lambda
aws lambda update-function-code \
  --function-name fraud-detector \
  --s3-bucket YOUR_BUCKET \
  --s3-key lambda/lambda_package.zip \
  --region us-east-2
```

---

## Key Findings from Training Data

- **Fraud peaks at 10–11pm** (2.8–2.9% rate vs 0.09% during business hours)
- **shopping_net** and **misc_net** have highest fraud rates by category
- **Amount** is the strongest predictor (65% feature importance)
- **High-value transactions** (>2× customer average) disproportionately flag fraud

---

## Limitations & Next Steps

- Model trained on synthetic data — real-world performance may differ
- Cold start latency (~700ms) can be reduced with Lambda SnapStart
- Add API Gateway for REST endpoint with rate limiting
- Add model versioning via S3 prefixes for A/B testing
- Retrain pipeline with SageMaker for full MLOps automation

---

## Author

**Rishabh Karthik Ramesh** — MS Computer Engineering, Virginia Tech  
[LinkedIn](https://www.linkedin.com/in/rishabh-karthik-ramesh/) · [GitHub](https://github.com/RishCapitalent18) · rishabhkramesh@gmail.com
