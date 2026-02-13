import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

LOG_PATH = "data/training_data.csv"
MODEL_PATH = "models/fps_model.joblib"

def main():
    if not os.path.exists(LOG_PATH):
        print("Missing data/training_data.csv (we will create it after FPS merging).")
        return

    os.makedirs("models", exist_ok=True)
    df = pd.read_csv(LOG_PATH)

    y = df["avg_fps"].astype(float)

    cat_cols = ["game","mode","resolution","texture","shadows","aa","render_scale"]
    num_cols = ["cpu_usage_pct","ram_usage_pct"]

    X = df[cat_cols + num_cols].copy()

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ])

    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    pipe = Pipeline([("pre", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, preds)

    joblib.dump(pipe, MODEL_PATH)
    print(f"Saved model: {MODEL_PATH}")
    print(f"MAE(FPS): {mae:.2f}")

if __name__ == "__main__":
    main()
