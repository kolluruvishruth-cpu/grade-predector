from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "student_data.csv"
MODEL_PATH = BASE_DIR / "model.pkl"
FEATURES = [
    "attendance",
    "study_hours",
    "previous_marks",
    "assignments_completed",
    "sleep_hours",
]
TARGET = "final_grade"


def main() -> None:
    data = pd.read_csv(DATA_PATH)

    X = data[FEATURES]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "r2_score": round(r2_score(y_test, predictions), 4),
        "mean_absolute_error": round(mean_absolute_error(y_test, predictions), 4),
    }

    bundle = {
        "model": model,
        "features": FEATURES,
        "target": TARGET,
        "metrics": metrics,
    }
    joblib.dump(bundle, MODEL_PATH)

    print("Model trained successfully.")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"R2 Score: {metrics['r2_score']}")
    print(f"Mean Absolute Error: {metrics['mean_absolute_error']}")


if __name__ == "__main__":
    main()
