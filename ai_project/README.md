# Car Price Predictor (Multiple Linear Regression)

This project trains and evaluates a multiple linear regression model to predict car prices from:
- `mileage`
- `engine_capacity`
- `vehicle_age`
- `fuel_type`
- `transmission`
- `assembly`

Dataset file expected at workspace root:
- `pakwheels_pakistan_automobile_dataset.csv`

## Quick Start

```powershell
python -m pip install -r requirements.txt
python main.py
```

## What `main.py` does

1. Loads and cleans the dataset.
2. Applies outlier capping (1st-99th percentile).
3. Builds a preprocessing pipeline (imputation + scaling + one-hot encoding).
4. Compares two multiple linear regression variants:
   - Raw target (`price`)
   - Log-transformed target (`log1p(price)`)
5. Selects the better variant using 5-fold CV RMSE.
6. Reports MAE, RMSE, and R2 on CV and test set.
7. Saves artifacts to `artifacts/`.

## Output Artifacts

After running, these files are created:
- `artifacts/car_price_linear_model.pkl`
- `artifacts/metrics.json`
- `artifacts/test_predictions.csv`
