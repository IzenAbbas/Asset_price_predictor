# Asset Advisor + Car/House Predictors

This project now includes a CSV-backed voice advisor that can inspect and compare:
- used cars from `pakwheels_pakistan_automobile_dataset.csv`
- house/property listings from `House_Details.csv`

The advisor in `advisor.py` uses a detailed system prompt and direct CSV tools to answer valuation and comparison questions with grounded data.

## Advisor behavior

- Use CSV-backed tools before answering factual questions.
- Ask for missing asset details instead of guessing.
- Return comparable listings and price summaries when possible.
- Stay within the car/property advisory domain.

## Data sources

- `pakwheels_pakistan_automobile_dataset.csv`
- `House_Details.csv`

Dataset file expected at project root:
- `pakwheels_pakistan_automobile_dataset.csv`

## Quick Start

```powershell
python -m pip install -r requirements.txt
python main.py
```

To run the voice advisor, launch `advisor.py` in the same environment after setting your LiveKit / Gemini credentials.

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

