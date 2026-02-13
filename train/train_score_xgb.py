import argparse
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


DEFAULT_RUNS_TABLE = Path("data") / "runs" / "runs_table.csv"
DEFAULT_MODEL_PATH = Path("models") / "score_model_xgb.joblib"
DEFAULT_REPORT_PATH = Path("models") / "score_model_report.json"
DEFAULT_PRED_PATH = Path("models") / "score_model_candidates.csv"


QUALITY_LEVELS = ["Lowest", "Low", "Medium", "High", "Highest"]
AA_OPTIONS = ["Off", "FSR 2", "SMAA"]
AF_OPTIONS = ["1X", "2X", "4X", "8X", "16X"]


def compute_score(df: pd.DataFrame, w_fps=0.60, w_cpu=0.25, w_ram=0.15) -> pd.Series:
    max_fps = df["avg_fps"].max()
    if max_fps <= 0:
        raise ValueError("avg_fps must be > 0 to compute score.")
    fps_score = df["avg_fps"] / max_fps
    cpu_penalty = df["cpu_percent_avg"] / 100.0
    ram_penalty = df["ram_percent_avg"] / 100.0
    return (w_fps * fps_score) - (w_cpu * cpu_penalty) - (w_ram * ram_penalty)


def load_runs_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Runs table not found: {path}")
    return pd.read_csv(path)


def normalize_runs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "shadow_quality" not in df.columns and "quality" in df.columns:
        df["shadow_quality"] = df["quality"]
    if "shadow_quality" in df.columns and "quality" in df.columns:
        df["shadow_quality"] = df["shadow_quality"].fillna(df["quality"])

    for col in ["anti_aliasing", "anisotropic_filtering"]:
        if col not in df.columns:
            df[col] = None

    for col in ["render_scale", "avg_fps", "cpu_percent_avg", "ram_percent_avg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = ["avg_fps", "cpu_percent_avg", "ram_percent_avg", "render_scale"]
    df = df.dropna(subset=[c for c in required if c in df.columns])

    if "score" not in df.columns:
        df["score"] = compute_score(df)
    else:
        df["score"] = pd.to_numeric(df["score"], errors="coerce")
        missing = df["score"].isna()
        if missing.any():
            df.loc[missing, "score"] = compute_score(df.loc[missing])

    return df


def prepare_features(df: pd.DataFrame):
    cat_cols = [
        col
        for col in ["game", "resolution", "shadow_quality", "anti_aliasing", "anisotropic_filtering"]
        if col in df.columns
    ]
    num_cols = [col for col in ["render_scale", "avg_fps", "cpu_percent_avg", "ram_percent_avg"] if col in df.columns]

    for col in cat_cols:
        if df[col].isna().all():
            df[col] = "Unknown"
        else:
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    return cat_cols, num_cols


def train_model(df: pd.DataFrame, cat_cols, num_cols, random_state=42):
    X = df[cat_cols + num_cols].copy()
    y = df["score"].astype(float)

    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=random_state,
        n_jobs=-1,
    )

    pipe = Pipeline([("pre", pre), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)

    feature_names = pipe.named_steps["pre"].get_feature_names_out()
    importances = pipe.named_steps["model"].feature_importances_
    importance_df = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    return pipe, r2, rmse, importance_df


def build_candidate_grid(df: pd.DataFrame, cat_cols, num_cols):
    options = {}
    for col in cat_cols:
        values = sorted({str(v) for v in df[col].dropna().unique()})
        if not values:
            if col == "shadow_quality":
                values = QUALITY_LEVELS
            elif col == "anti_aliasing":
                values = AA_OPTIONS
            elif col == "anisotropic_filtering":
                values = AF_OPTIONS
            else:
                values = ["Unknown"]
        options[col] = values

    if "render_scale" in df.columns:
        observed = sorted({round(float(v), 2) for v in df["render_scale"].dropna().tolist()})
        grid = [0.7, 0.75, 0.8, 0.85, 0.9, 1.0, 1.1]
        min_rs = min(observed) if observed else 0.7
        max_rs = max(observed) if observed else 1.1
        grid = [v for v in grid if min_rs - 0.05 <= v <= max_rs + 0.05]
        options["render_scale"] = sorted(set(observed + grid))

    game_value = None
    if "game" in cat_cols:
        mode = df["game"].mode(dropna=True)
        game_value = mode.iloc[0] if not mode.empty else "Unknown"

    settings_cols = [c for c in ["resolution", "shadow_quality", "anti_aliasing", "anisotropic_filtering", "render_scale"] if c in options]
    group_medians = (
        df.groupby(settings_cols)[["avg_fps", "cpu_percent_avg", "ram_percent_avg"]]
        .median()
        .reset_index()
    )
    median_map = {
        tuple(row[settings_cols]): row[["avg_fps", "cpu_percent_avg", "ram_percent_avg"]].to_dict()
        for _, row in group_medians.iterrows()
    }

    global_medians = {
        "avg_fps": float(df["avg_fps"].median()),
        "cpu_percent_avg": float(df["cpu_percent_avg"].median()),
        "ram_percent_avg": float(df["ram_percent_avg"].median()),
    }

    product_cols = [c for c in settings_cols if c != "render_scale"]
    render_scales = options.get("render_scale", [float(df["render_scale"].median())])

    candidates = []
    for combo in itertools.product(*[options[c] for c in product_cols]):
        base = dict(zip(product_cols, combo))
        for rs in render_scales:
            row = base.copy()
            row["render_scale"] = float(rs)
            key = tuple(row[c] for c in settings_cols)
            stats = median_map.get(key, global_medians)
            row.update(stats)
            if game_value is not None:
                row["game"] = game_value
            candidates.append(row)

    return pd.DataFrame(candidates)


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost model to predict score and recommend best settings.")
    parser.add_argument("--runs-table", default=str(DEFAULT_RUNS_TABLE))
    parser.add_argument("--model-out", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--report-out", default=str(DEFAULT_REPORT_PATH))
    parser.add_argument("--candidates-out", default=str(DEFAULT_PRED_PATH))
    args = parser.parse_args()

    runs_table_path = Path(args.runs_table)
    model_out = Path(args.model_out)
    report_out = Path(args.report_out)
    candidates_out = Path(args.candidates_out)

    df = load_runs_table(runs_table_path)
    df = normalize_runs(df)
    if df.empty:
        raise ValueError("No usable rows found after cleaning. Check runs_table.csv.")

    cat_cols, num_cols = prepare_features(df)
    pipe, r2, rmse, importance_df = train_model(df, cat_cols, num_cols)

    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, model_out)

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "rows_used": int(len(df)),
        "features_categorical": cat_cols,
        "features_numeric": num_cols,
        "r2": float(r2),
        "rmse": float(rmse),
        "top_features": importance_df.head(15).to_dict(orient="records"),
    }
    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    candidates = build_candidate_grid(df, cat_cols, num_cols)
    pred_scores = pipe.predict(candidates[cat_cols + num_cols])
    candidates["pred_score"] = pred_scores
    candidates = candidates.sort_values("pred_score", ascending=False)

    candidates_out.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(candidates_out, index=False)

    best = candidates.iloc[0].to_dict()
    print("=== XGBoost Score Model ===")
    print(f"Rows used: {len(df)}")
    print(f"R2: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Saved model: {model_out}")
    print(f"Saved report: {report_out}")
    print(f"Saved candidates: {candidates_out}")
    print("")
    print("=== Best Predicted Settings ===")
    for k in ["game", "resolution", "shadow_quality", "anti_aliasing", "anisotropic_filtering", "render_scale"]:
        if k in best:
            print(f"{k}: {best[k]}")
    print(f"pred_score: {best['pred_score']:.4f}")


if __name__ == "__main__":
    main()
