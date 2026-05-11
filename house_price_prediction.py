from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

RANDOM_STATE = 42
DATASET_PATH = Path("House_Details.csv")
ARTIFACTS_DIR = Path("artifacts")

NUMERIC_FEATURES = ["Total_Area", "bedrooms", "baths", "latitude", "longitude", "listing_year", "listing_month"]
CATEGORICAL_FEATURES = ["property_type", "location", "city", "province_name", "purpose"]
TARGET = "price"


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return "unknown"
    clean = str(value).strip().lower()
    return clean if clean else "unknown"


def parse_listing_date(value: object) -> pd.Timestamp:
    if pd.isna(value):
        return pd.NaT
    return pd.to_datetime(str(value), errors="coerce")


def load_data(csv_path: Path) -> pd.DataFrame:
    required_columns = [
        "property_id",
        "property_type",
        "price",
        "location",
        "city",
        "province_name",
        "latitude",
        "longitude",
        "baths",
        "purpose",
        "bedrooms",
        "date_added",
        "Total_Area",
    ]

    df = pd.read_csv(str(csv_path))
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df[required_columns].copy().drop_duplicates(subset=["property_id"])
    data = data[
        (data["purpose"].astype(str).str.lower() == "for sale")
        & (data[TARGET] > 0)
        & (data["Total_Area"] > 0)
        & (data["bedrooms"] > 0)
        & (data["baths"] > 0)
        & (data["latitude"].between(20, 38))
        & (data["longitude"].between(60, 78))
    ]

    data["date_added"] = data["date_added"].apply(parse_listing_date)
    data["listing_year"] = data["date_added"].dt.year.fillna(2019).astype(int)
    data["listing_month"] = data["date_added"].dt.month.fillna(1).astype(int)

    for col in CATEGORICAL_FEATURES:
        data[col] = data[col].apply(normalize_text)

    return data.drop(columns=["property_id", "date_added"]).dropna()


house = load_data(DATASET_PATH)
print(f"Rows used for training: {len(house):,}")


def build_gradient_boosting_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", RobustScaler()),
                    ]
                ),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                TransformedTargetRegressor(
                    regressor=HistGradientBoostingRegressor(
                        random_state=RANDOM_STATE
                    ),
                    func=np.log1p,
                    inverse_func=np.expm1,
                ),
            ),
        ]
    )


all_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
X = house[all_features]
y = house[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
)

selected_model_name = "gradient_boosting"
selected_model = build_gradient_boosting_model()
selected_model.fit(X_train, y_train)


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import json
import base64
from io import BytesIO

predictions = np.asarray(selected_model.predict(X_test), dtype=float)
y_true = np.asarray(y_test, dtype=float)

mae = float(mean_absolute_error(y_true, predictions))
mse = float(mean_squared_error(y_true, predictions))
rmse = float(np.sqrt(mse))
r2 = float(r2_score(y_true, predictions))

print("-" * 60)
print("HOUSE PRICE REGRESSION REPORT")
print("-" * 60)
print(f"Selected model variant: {selected_model_name}")
print("Holdout test set:")
print(f"  MAE : Rs. {mae:,.2f}")
print(f"  MSE : {mse:,.2f}")
print(f"  RMSE: Rs. {rmse:,.2f}")
print(f"  R2  : {r2:.4f}")

def plot_to_base64():
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return img_str

plt.figure(figsize=(8, 6))
plt.scatter(y_true, predictions, alpha=0.3)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
plt.xlabel('Actual Value')
plt.ylabel('Predicted Value')
plt.title('Predicted vs. Actual Plot')
graph_pred_vs_actual = plot_to_base64()

residuals = y_true - predictions
plt.figure(figsize=(8, 6))
plt.scatter(predictions, residuals, alpha=0.3)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Value')
plt.ylabel('Residuals')
plt.title('Residual Plot')
graph_residuals = plot_to_base64()

from sklearn.inspection import permutation_importance

graph_feature_importance = ""
try:
    final_estimator = selected_model.steps[-1][1]
    if hasattr(final_estimator, 'regressor_'):
        reg = final_estimator.regressor_
        if hasattr(reg, 'coef_'):
            coefs = reg.coef_
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(coefs)), coefs)
            plt.title('Feature Coefficients')
            graph_feature_importance = plot_to_base64()
        else:
            print("Calculating permutation importance (this may take a moment)...")
            result = permutation_importance(
                selected_model, X_test.iloc[:1000], y_test.iloc[:1000], 
                n_repeats=3, random_state=RANDOM_STATE, n_jobs=-1
            )
            importances = result.importances_mean
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(importances)), importances)
            plt.xticks(range(len(importances)), all_features, rotation=45)
            plt.title('Feature Importance (Permutation)')
            graph_feature_importance = plot_to_base64()
except Exception as e:
    print(f"Error calculating feature importance: {e}")

from sklearn.model_selection import learning_curve

print("Generating learning curves...")
train_sizes, train_scores, test_scores = learning_curve(
    selected_model, X, y, cv=3, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
)
train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training error")
plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation error")
plt.xlabel("Training examples")
plt.ylabel("MSE")
plt.title("Learning Curves")
plt.legend(loc="best")
ARTIFACTS_DIR.mkdir(exist_ok=True)
plt.savefig(ARTIFACTS_DIR / "house_learning_curve.png", bbox_inches='tight')
graph_learning_curve = plot_to_base64()

model_path = ARTIFACTS_DIR / "house_price_best_model.pkl"
metrics_path = ARTIFACTS_DIR / "house_metrics.json"

import joblib
joblib.dump(selected_model, model_path)

payload = {
    "selected_model": selected_model_name,
    "test_mae": mae,
    "test_mse": mse,
    "test_rmse": rmse,
    "test_r2": r2,
    "features": {
        "numeric": NUMERIC_FEATURES,
        "categorical": CATEGORICAL_FEATURES,
    },
    "artifacts": {
        "model_path": str(model_path.resolve()),
    },
    "graphs": {
        "predicted_vs_actual": graph_pred_vs_actual,
        "residual_plot": graph_residuals,
        "feature_importance": graph_feature_importance,
        "learning_curve": graph_learning_curve
    }
}
metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

print(f"Saved model to: {model_path}")
print(f"Saved metrics to: {metrics_path}")
