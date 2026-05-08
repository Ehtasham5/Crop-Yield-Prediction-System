import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


RAW_DATA_PATH = Path("data/raw/crop_yield_dataset.csv")
PROCESSED_DATA_PATH = Path("data/processed/cleaned_crop_yield_data.csv")
MODEL_PATH = Path("models/crop_yield_model.joblib")
METRICS_PATH = Path("models/model_metrics.json")
TARGET_COLUMN = "Yield_ton_per_ha"


def load_and_clean_data() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset not found. Run `python src/download_data.py` first."
        )

    df = pd.read_csv(RAW_DATA_PATH)
    df = df.drop_duplicates()
    df.columns = [column.strip() for column in df.columns]

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column `{TARGET_COLUMN}` not found in dataset.")

    numeric_columns = df.columns
    numeric_columns = [
        "Soil_pH",
        "Rainfall_mm",
        "Temperature_C",
        "Humidity_pct",
        "Fertilizer_Used_kg",
        "Pesticides_Used_kg",
        "Planting_Density",
        TARGET_COLUMN,
    ]
    categorical_columns = ["Crop", "Region", "Soil_Type", "Irrigation", "Previous_Crop"]

    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce")
    for column in categorical_columns:
        df[column] = df[column].fillna("Unknown").astype(str).str.strip()

    df = df.dropna(subset=[TARGET_COLUMN])

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    return df


def train_model(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    feature_columns = [column for column in df.columns if column != TARGET_COLUMN]
    numeric_features = df[feature_columns].select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [
        column for column in feature_columns if column not in numeric_features
    ]
    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "regressor",
                RandomForestRegressor(
                    n_estimators=200,
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "r2_score": r2_score(y_test, predictions),
        "mean_absolute_error": mean_absolute_error(y_test, predictions),
        "root_mean_squared_error": mean_squared_error(
            y_test, predictions,
        )
        ** 0.5,
        "feature_columns": feature_columns,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "target_column": TARGET_COLUMN,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
    }

    return model, metrics


def save_outputs(model: Pipeline, metrics: dict) -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    with METRICS_PATH.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)


def main() -> None:
    df = load_and_clean_data()
    model, metrics = train_model(df)
    save_outputs(model, metrics)

    print(f"Cleaned dataset saved to {PROCESSED_DATA_PATH}")
    print(f"Model saved to {MODEL_PATH}")
    print(f"Metrics saved to {METRICS_PATH}")
    print(f"R2 score: {metrics['r2_score']:.4f}")
    print(f"MAE: {metrics['mean_absolute_error']:.2f} tons/ha")


if __name__ == "__main__":
    main()
