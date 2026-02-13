import itertools
import json
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "data" / "runs"
RUNS_TABLE = RUNS_DIR / "runs_table.csv"
RUN_PLAN_PATH = RUNS_DIR / "run_plan.json"

MIN_RUNS_FOR_MODEL = 5
DEFAULT_TARGET_RUNS = 10

QUALITY_LEVELS = ["Lowest", "Low", "Medium", "High", "Highest"]
AA_OPTIONS = ["Off", "FSR 2", "SMAA"]
AF_OPTIONS = ["1X", "2X", "4X", "8X", "16X"]

# Quality preference weights (favor visuals while keeping FPS strong)
QUALITY_WEIGHTS = {
    "fps": 0.60,
    "visual": 0.35,
    "penalty": 0.05,
}
CPU_SOFT_CAP = 95.0
RAM_SOFT_CAP = 95.0


def safe_read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def compute_score(df: pd.DataFrame, w_fps=0.60, w_cpu=0.25, w_ram=0.15) -> pd.DataFrame:
    """
    Score = (w_fps * normalized_fps) - (w_cpu * cpu_penalty) - (w_ram * ram_penalty)
    Higher score = better FPS-to-stability tradeoff.
    """
    out = df.copy()

    # Ensure required columns exist
    needed = ["avg_fps", "cpu_percent_avg", "ram_percent_avg"]
    for c in needed:
        if c not in out.columns:
            out[c] = None

    # Drop rows without essentials
    out = out.dropna(subset=["avg_fps", "cpu_percent_avg", "ram_percent_avg"])
    if out.empty:
        return out

    max_fps = out["avg_fps"].max()
    if max_fps <= 0:
        return pd.DataFrame()

    out["fps_score"] = out["avg_fps"] / max_fps
    out["cpu_penalty"] = out["cpu_percent_avg"] / 100.0
    out["ram_penalty"] = out["ram_percent_avg"] / 100.0

    out["score"] = (w_fps * out["fps_score"]) - (w_cpu * out["cpu_penalty"]) - (w_ram * out["ram_penalty"])
    return out


def _normalize(value, min_v, max_v):
    if max_v <= min_v:
        return 0.0
    return (value - min_v) / (max_v - min_v)


def quality_score_from_settings(row: pd.Series) -> float:
    quality_map = {name: idx / (len(QUALITY_LEVELS) - 1) for idx, name in enumerate(QUALITY_LEVELS)}
    aa_map = {"Off": 0.0, "SMAA": 0.6, "FSR 2": 1.0}
    af_map = {"1X": 0.0, "2X": 0.25, "4X": 0.5, "8X": 0.75, "16X": 1.0}

    shadow = quality_map.get(str(row.get("shadow_quality", "Lowest")), 0.0)
    aa = aa_map.get(str(row.get("anti_aliasing", "Off")), 0.0)
    af = af_map.get(str(row.get("anisotropic_filtering", "1X")), 0.0)
    rs = float(row.get("render_scale", 0.8))
    rs_norm = _normalize(rs, 0.7, 1.1)

    # Visual quality weighted blend
    return (0.4 * shadow) + (0.25 * aa) + (0.25 * af) + (0.10 * rs_norm)


def compute_balanced_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    max_fps = out["avg_fps"].max()
    if max_fps <= 0:
        return pd.DataFrame()

    out["fps_score"] = out["avg_fps"] / max_fps
    out["visual_score"] = out.apply(quality_score_from_settings, axis=1)

    # Soft penalty only when exceeding caps
    cpu_over = (out["cpu_percent_avg"] - CPU_SOFT_CAP).clip(lower=0.0) / 100.0
    ram_over = (out["ram_percent_avg"] - RAM_SOFT_CAP).clip(lower=0.0) / 100.0
    out["stability_penalty"] = (cpu_over + ram_over) / 2.0

    out["score"] = (
        QUALITY_WEIGHTS["fps"] * out["fps_score"]
        + QUALITY_WEIGHTS["visual"] * out["visual_score"]
        - QUALITY_WEIGHTS["penalty"] * out["stability_penalty"]
    )
    return out


def load_run_plan(path: Path) -> int:
    if not path.exists():
        return DEFAULT_TARGET_RUNS
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return max(MIN_RUNS_FOR_MODEL, int(payload.get("target_runs", DEFAULT_TARGET_RUNS)))
    except Exception:
        return DEFAULT_TARGET_RUNS


def normalize_runs_for_model(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "shadow_quality" not in out.columns and "quality" in out.columns:
        out["shadow_quality"] = out["quality"]
    if "shadow_quality" in out.columns and "quality" in out.columns:
        out["shadow_quality"] = out["shadow_quality"].fillna(out["quality"])

    for col in ["anti_aliasing", "anisotropic_filtering"]:
        if col not in out.columns:
            out[col] = None

    for col in ["render_scale", "avg_fps", "cpu_percent_avg", "ram_percent_avg"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    required = ["avg_fps", "cpu_percent_avg", "ram_percent_avg", "render_scale"]
    out = out.dropna(subset=[c for c in required if c in out.columns])
    return out


def prepare_settings_features(df: pd.DataFrame):
    cat_cols = [
        col
        for col in ["game", "resolution", "shadow_quality", "anti_aliasing", "anisotropic_filtering"]
        if col in df.columns
    ]
    num_cols = [col for col in ["render_scale"] if col in df.columns]

    for col in cat_cols:
        if df[col].isna().all():
            df[col] = "Unknown"
        else:
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

    return cat_cols, num_cols


def train_xgb_regressor(df: pd.DataFrame, cat_cols, num_cols, target_col: str):
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    try:
        from xgboost import XGBRegressor
    except Exception as e:
        raise RuntimeError(
            "XGBoost is not installed. Install it with: pip install xgboost"
        ) from e

    X = df[cat_cols + num_cols].copy()
    y = df[target_col].astype(float)

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
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline([("pre", pre), ("model", model)])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipe.fit(X_train, y_train)
    return pipe


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
    product_cols = [c for c in settings_cols if c != "render_scale"]
    render_scales = options.get("render_scale", [float(df["render_scale"].median())])

    candidates = []
    for combo in itertools.product(*[options[c] for c in product_cols]):
        base = dict(zip(product_cols, combo))
        for rs in render_scales:
            row = base.copy()
            row["render_scale"] = float(rs)
            if game_value is not None:
                row["game"] = game_value
            candidates.append(row)

    return pd.DataFrame(candidates)


st.set_page_config(page_title="Results", page_icon="📊", layout="wide")
st.title("📊 Results (Run Analysis + Best Settings)")

st.write(
    "This page reads your saved calibration runs from `data/runs/runs_table.csv`, "
    "computes a **performance + visual quality score**, and recommends the best settings."
)

st.divider()

if not RUNS_TABLE.exists():
    st.warning(
        "No runs found yet. Go to **Calibrate** → do a run → **Save Run to CSV**.\n\n"
        f"Expected file: `{RUNS_TABLE}`"
    )
    st.stop()

df = safe_read_csv(RUNS_TABLE)
if df.empty:
    st.warning("`runs_table.csv` exists but is empty or unreadable. Save at least 1 run from Calibrate.")
    st.stop()

# Sidebar controls
# Scoring is now balanced toward FPS + visual quality (see below)

st.sidebar.header("Filters")
min_fps = st.sidebar.number_input("Minimum FPS", min_value=0.0, value=0.0, step=1.0)
max_cpu = st.sidebar.number_input("Max CPU avg (%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0)
max_ram = st.sidebar.number_input("Max RAM avg (%)", min_value=0.0, max_value=100.0, value=100.0, step=1.0)

target_runs = load_run_plan(RUN_PLAN_PATH)
st.caption(f"Run target: {target_runs} (minimum {MIN_RUNS_FOR_MODEL}, recommended 10).")

# Compute score (balanced for FPS + visual quality)
scored = compute_balanced_score(df)

if scored.empty:
    st.error(
        "Not enough usable data to score runs.\n\n"
        "Make sure your runs_table.csv has columns:\n"
        "- avg_fps\n- cpu_percent_avg\n- ram_percent_avg\n\n"
        "Do 1–2 new runs from Calibrate and save again."
    )
    st.dataframe(df, use_container_width=True)
    st.stop()

# Apply filters
f = scored.copy()
f = f[f["avg_fps"] >= min_fps]
f = f[f["cpu_percent_avg"] <= max_cpu]
f = f[f["ram_percent_avg"] <= max_ram]

if f.empty:
    st.warning("No runs match your current filters. Loosen the filter limits in the sidebar.")
    st.stop()

# Best run
best = f.sort_values("score", ascending=False).iloc[0]

st.subheader("🏆 Best run (recommended settings from your data)")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Score", f"{best['score']:.3f}")
c2.metric("Avg FPS", f"{best['avg_fps']:.1f}")
c3.metric("CPU avg", f"{best['cpu_percent_avg']:.1f}%")
c4.metric("RAM avg", f"{best['ram_percent_avg']:.1f}%")
c5.metric("Run ID", str(best.get("run_id", "—")))
st.caption(
    "Score emphasizes FPS + visual quality, with a small penalty only when CPU/RAM exceed soft caps."
)

st.markdown("### Recommended settings (from best run)")
shadow_quality = best.get("shadow_quality", best.get("quality", "--"))
r1, r2, r3, r4, r5, r6 = st.columns(6)
r1.metric("Resolution", str(best.get("resolution", "--")))
r2.metric("Shadow Quality", str(shadow_quality))
r3.metric("Anti-Aliasing", str(best.get("anti_aliasing", "--")))
r4.metric("Anisotropic Filtering", str(best.get("anisotropic_filtering", "--")))
r5.metric("Render Scale", str(best.get("render_scale", "--")))
r6.metric("Game", str(best.get("game", "--")))

st.divider()

st.subheader("Model-predicted best settings (XGBoost)")
st.caption(
    "We train XGBoost on your completed runs to learn how settings affect FPS/CPU/RAM. "
    "Then we predict those metrics for candidate settings (including ones you haven't tried), "
    "compute a performance+visual score, and recommend the highest predicted score."
)

df_ml = normalize_runs_for_model(df)
usable_runs = len(df_ml)

if usable_runs < MIN_RUNS_FOR_MODEL:
    st.warning(
        f"Need at least {MIN_RUNS_FOR_MODEL} runs to train the model. "
        f"Current usable runs: {usable_runs}."
    )
elif usable_runs < target_runs:
    st.info(
        f"Model will auto-train after {target_runs} runs. "
        f"Current usable runs: {usable_runs}."
    )
else:
    try:
        if df_ml.empty:
            raise ValueError("No usable rows after cleaning.")

        cat_cols, num_cols = prepare_settings_features(df_ml)
        fps_model = train_xgb_regressor(df_ml, cat_cols, num_cols, "avg_fps")
        cpu_model = train_xgb_regressor(df_ml, cat_cols, num_cols, "cpu_percent_avg")
        ram_model = train_xgb_regressor(df_ml, cat_cols, num_cols, "ram_percent_avg")

        candidates = build_candidate_grid(df_ml, cat_cols, num_cols)
        if candidates.empty:
            raise ValueError("No candidate settings generated.")

        Xc = candidates[cat_cols + num_cols]
        candidates["pred_avg_fps"] = fps_model.predict(Xc)
        candidates["pred_cpu_avg"] = cpu_model.predict(Xc)
        candidates["pred_ram_avg"] = ram_model.predict(Xc)

        max_pred_fps = candidates["pred_avg_fps"].max()
        if max_pred_fps <= 0:
            raise ValueError("Predicted FPS is non-positive.")
        candidates["visual_score"] = candidates.apply(quality_score_from_settings, axis=1)
        cpu_over = (candidates["pred_cpu_avg"] - CPU_SOFT_CAP).clip(lower=0.0) / 100.0
        ram_over = (candidates["pred_ram_avg"] - RAM_SOFT_CAP).clip(lower=0.0) / 100.0
        candidates["stability_penalty"] = (cpu_over + ram_over) / 2.0
        candidates["pred_score"] = (
            QUALITY_WEIGHTS["fps"] * (candidates["pred_avg_fps"] / max_pred_fps)
            + QUALITY_WEIGHTS["visual"] * candidates["visual_score"]
            - QUALITY_WEIGHTS["penalty"] * candidates["stability_penalty"]
        )
        candidates = candidates.sort_values("pred_score", ascending=False)
        best_pred = candidates.iloc[0]

        observed_settings = df_ml[
            ["resolution", "shadow_quality", "anti_aliasing", "anisotropic_filtering", "render_scale"]
        ].drop_duplicates()
        is_observed = not observed_settings[
            (observed_settings["resolution"] == best_pred.get("resolution"))
            & (observed_settings["shadow_quality"] == best_pred.get("shadow_quality"))
            & (observed_settings["anti_aliasing"] == best_pred.get("anti_aliasing"))
            & (observed_settings["anisotropic_filtering"] == best_pred.get("anisotropic_filtering"))
            & (observed_settings["render_scale"].astype(float) == float(best_pred.get("render_scale")))
        ].empty

        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Resolution", str(best_pred.get("resolution", "--")))
        m2.metric("Shadow Quality", str(best_pred.get("shadow_quality", "--")))
        m3.metric("Anti-Aliasing", str(best_pred.get("anti_aliasing", "--")))
        m4.metric("Anisotropic Filtering", str(best_pred.get("anisotropic_filtering", "--")))
        m5.metric("Render Scale", str(best_pred.get("render_scale", "--")))
        m6.metric("Predicted Score", f"{best_pred.get('pred_score', 0.0):.3f}")

        if is_observed:
            st.caption("This combination exists in your runs.")
        else:
            st.caption("This combination was not directly tested in your runs.")
    except Exception as e:
        st.error(f"Model training failed: {e}")

st.subheader("All runs (scored)")
quality_col = "shadow_quality" if "shadow_quality" in f.columns else ("quality" if "quality" in f.columns else None)
show_cols = [
    "run_id", "saved_at_local", "game",
    "resolution", quality_col, "anti_aliasing", "anisotropic_filtering", "render_scale",
    "avg_fps", "cpu_percent_avg", "ram_percent_avg",
    "score"
]
show_cols = [c for c in show_cols if c in f.columns and c is not None]
st.dataframe(f.sort_values("score", ascending=False)[show_cols], use_container_width=True)

st.divider()

st.subheader("Charts (quick demo)")
# Simple charts that look good in demo
chart_cols = [c for c in ["avg_fps", "cpu_percent_avg", "ram_percent_avg", "score"] if c in f.columns]
chart_df = f.sort_values("saved_at_local") if "saved_at_local" in f.columns else f.copy()

if "saved_at_local" in chart_df.columns:
    # Use saved_at_local as index if possible
    try:
        chart_df["saved_at_local"] = pd.to_datetime(chart_df["saved_at_local"], errors="coerce")
        chart_df = chart_df.sort_values("saved_at_local")
        chart_df = chart_df.set_index("saved_at_local")
    except Exception:
        pass

st.line_chart(chart_df[chart_cols])

st.divider()

st.download_button(
    "⬇️ Download runs_table.csv",
    data=RUNS_TABLE.read_bytes(),
    file_name="runs_table.csv",
    mime="text/csv",
    use_container_width=True
)

st.info(
    "Tip: Do 10 runs across different gameplay moments (city, combat, domain). "
    "More variety = more reliable recommendation."
)
