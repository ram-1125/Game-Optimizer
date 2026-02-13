import time
import uuid
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil
import streamlit as st
import subprocess


# -------------------------------
# Settings options
# -------------------------------
QUALITY_LEVELS = ["Lowest", "Low", "Medium", "High", "Highest"]
AA_OPTIONS = ["Off", "FSR 2", "SMAA"]
AF_OPTIONS = ["1X", "2X", "4X", "8X", "16X"]


# -------------------------------
# Paths
# -------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Game_Optimizer/
RUNS_DIR = PROJECT_ROOT / "data" / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
RUN_PLAN_PATH = RUNS_DIR / "run_plan.json"
TOOLS_DIR = PROJECT_ROOT / "tools"
PRESENTMON_EXE = TOOLS_DIR / "presentmon.exe"
FPS_DIR = PROJECT_ROOT / "data" / "fps_logs"
FPS_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------
# Helpers
# -------------------------------
def sample_metrics():
    """
    One sample of system metrics.
    CPU/RAM are the key baselines. Disk IO is optional, but useful.
    """
    cpu = psutil.cpu_percent(interval=0.2)  # short interval for a stable sample
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage("/") if hasattr(psutil, "disk_usage") else None

    return {
        "ts": datetime.utcnow().isoformat(),
        "cpu_percent": float(cpu),
        "ram_percent": float(vm.percent),
        "ram_used_gb": round(vm.used / (1024**3), 3),
        "ram_total_gb": round(vm.total / (1024**3), 3),
        "disk_percent": float(disk.percent) if disk else None,
    }


def run_monitor(duration_sec: int, sample_every_sec: float = 1.0):
    """
    Monitor for duration_sec and return a DataFrame of samples.
    """
    rows = []
    start = time.time()
    next_sample = start

    while True:
        now = time.time()
        if now >= start + duration_sec:
            break

        if now >= next_sample:
            rows.append(sample_metrics())
            next_sample += sample_every_sec

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame):
    """
    Simple aggregates for your ML dataset later.
    """
    if df.empty:
        return {}

    def agg(col):
        return {
            f"{col}_avg": float(df[col].mean()),
            f"{col}_min": float(df[col].min()),
            f"{col}_max": float(df[col].max()),
        }

    out = {}
    out.update(agg("cpu_percent"))
    out.update(agg("ram_percent"))

    # disk_percent may have None values on some systems
    if "disk_percent" in df.columns and df["disk_percent"].notna().any():
        out.update(agg("disk_percent"))

    return out


def run_presentmon_capture(process_name: str, duration_sec: int, out_csv: Path) -> subprocess.Popen:
    """
    Starts PresentMon capture and returns the process handle.
    Uses v2 metrics by default (PresentMon 2.x).
    """
    if not PRESENTMON_EXE.exists():
        raise FileNotFoundError(f"PresentMon not found at: {PRESENTMON_EXE}")

    cmd = [
        str(PRESENTMON_EXE),
        "--process_name", process_name,
        "--output_file", str(out_csv),
        "--timed", str(duration_sec),
        "--terminate_after_timed",
        "--restart_as_admin",
        "--v2_metrics",
    ]
    # Create a new console-less process (Windows)
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def compute_avg_fps_from_presentmon(csv_path: Path) -> float:
    """
    Compute average FPS from PresentMon CSV.
    Uses MsBetweenPresents column (ms/frame).
    Avg FPS = mean(1000 / msBetweenPresents).
    """
    import pandas as pd

    if not csv_path.exists():
        raise FileNotFoundError(f"PresentMon CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # PresentMon can emit different column names depending on version/flags
    candidates = [
        "MsBetweenPresents",   # PresentMon 2.x (when enabled)
        "FrameTime",           # PresentMon 2.x default
        "msBetweenPresents",   # older builds (case variant)
        "MsBetweenDisplayChange",
    ]
    ft_col = next((c for c in candidates if c in df.columns), None)
    if not ft_col:
        cols = ", ".join(df.columns)
        raise ValueError(f"No frame-time column found. Available columns: {cols}")

    ft = pd.to_numeric(df[ft_col], errors="coerce").dropna()
    ft = ft[ft > 0]

    if ft.empty:
        raise ValueError("No valid frame-time samples found (MsBetweenPresents).")

    fps_series = 1000.0 / ft
    return float(fps_series.mean())


# -------------------------------
# Plan helpers
# -------------------------------
def save_run_plan(path: Path, target_runs: int):
    payload = {
        "target_runs": int(target_runs),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    path.write_text(pd.Series(payload).to_json(), encoding="utf-8")


# -------------------------------
# Recommendation helpers
# -------------------------------
def compute_score(df: pd.DataFrame, w_fps=0.60, w_cpu=0.25, w_ram=0.15) -> pd.DataFrame:
    """
    Score = (w_fps * normalized_fps) - (w_cpu * cpu_penalty) - (w_ram * ram_penalty)
    Higher score = better FPS-to-stability tradeoff.
    """
    out = df.copy()

    needed = ["avg_fps", "cpu_percent_avg", "ram_percent_avg"]
    for c in needed:
        if c not in out.columns:
            out[c] = None

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


def _clean_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _coerce_choice(value, options, default):
    value = _clean_value(value)
    if value is None:
        return default
    value_str = str(value).strip().lower()
    for opt in options:
        if str(opt).strip().lower() == value_str:
            return opt
    return default


def _coerce_render_scale(value, default, min_v=0.3, max_v=1.5):
    value = _clean_value(value)
    if value is None:
        return default
    try:
        v = float(value)
    except Exception:
        return default
    return min(max(v, min_v), max_v)


def get_best_run(runs_table_path: Path):
    if not runs_table_path.exists():
        return None
    try:
        df = pd.read_csv(runs_table_path)
    except Exception:
        return None
    if df.empty:
        return None

    scored = compute_score(df)
    if scored.empty:
        return None
    best = scored.sort_values("score", ascending=False).iloc[0]
    return best


def build_recommended_settings(best_row, defaults):
    shadow_quality = best_row.get("shadow_quality") if best_row is not None else None
    if shadow_quality is None and best_row is not None:
        shadow_quality = best_row.get("quality")

    return {
        "game_name": str(_clean_value(best_row.get("game")) or defaults["game_name"]),
        "resolution": str(_clean_value(best_row.get("resolution")) or defaults["resolution"]),
        "shadow_quality": _coerce_choice(shadow_quality, QUALITY_LEVELS, defaults["shadow_quality"]),
        "anti_aliasing": _coerce_choice(best_row.get("anti_aliasing"), AA_OPTIONS, defaults["anti_aliasing"]),
        "anisotropic_filtering": _coerce_choice(
            best_row.get("anisotropic_filtering"), AF_OPTIONS, defaults["anisotropic_filtering"]
        ),
        "render_scale": _coerce_render_scale(best_row.get("render_scale"), defaults["render_scale"]),
    }


# -------------------------------
# UI
# -------------------------------
st.set_page_config(page_title="Calibrate", page_icon="🎯", layout="wide")

st.title("🎯 Calibrate")
st.write(
    "This step records **60 seconds of system metrics** while you play, "
    "then you enter your **average FPS**. We save it as a run for training."
)

st.info(
    "How to use:\n"
    "1) Open your game (Genshin Impact) with fixed settings.\n"
    "2) Come back here and click **Start 60s Monitoring**.\n"
    "3) Immediately Alt+Tab to the game and play normally for 60 seconds.\n"
    "4) After 60 seconds, return and enter your **average FPS**."
)

st.subheader("Run Plan")
target_runs = st.number_input(
    "How many runs do you want to take?",
    min_value=5,
    max_value=100,
    value=10,
    step=1
)
st.caption("Minimum 5 runs required. 10 runs recommended for optimal recommendation.")
save_run_plan(RUN_PLAN_PATH, int(target_runs))

st.divider()

# Auto-fill settings from best run so far (or safe defaults)
default_settings = {
    "game_name": "Genshin Impact",
    "resolution": "1280x720",
    "shadow_quality": "Lowest",
    "anti_aliasing": "Off",
    "anisotropic_filtering": "1X",
    "render_scale": 0.8,
}

use_test_plan = st.session_state.get("use_test_plan", False)

pending_prefill = st.session_state.pop("pending_prefill", None)
if pending_prefill:
    for key, value in pending_prefill.items():
        st.session_state[key] = value

for key, value in default_settings.items():
    if key not in st.session_state:
        st.session_state[key] = value

if not use_test_plan:
    best_row = get_best_run(RUNS_DIR / "runs_table.csv")
    if best_row is not None:
        best_id = str(best_row.get("run_id", "")).strip()
        if st.session_state.get("prefill_source_run_id") != best_id:
            recommended = build_recommended_settings(best_row, default_settings)
            st.session_state["pending_prefill"] = recommended
            st.session_state["prefill_source_run_id"] = best_id
            try:
                st.rerun()
            except AttributeError:
                st.experimental_rerun()

if st.session_state.get("just_saved_run_id"):
    st.success(f"Saved run {st.session_state['just_saved_run_id']}")
    st.session_state.pop("just_saved_run_id", None)

if st.session_state.get("next_run_updated"):
    st.info("Next run settings updated — apply these in game.")
    st.session_state.pop("next_run_updated", None)

st.subheader("Next-run plan")
use_test_plan = st.toggle(
    "Use 10-run diversity plan (auto-cycles settings after each save)",
    value=st.session_state.get("use_test_plan", False),
)
st.session_state["use_test_plan"] = use_test_plan
st.caption("You can still edit settings manually before starting each run.")

def build_test_plan(defaults):
    return [
        {"shadow_quality": "Lowest", "anti_aliasing": "Off", "anisotropic_filtering": "1X", "render_scale": 0.70},
        {"shadow_quality": "Lowest", "anti_aliasing": "SMAA", "anisotropic_filtering": "2X", "render_scale": 0.75},
        {"shadow_quality": "Low", "anti_aliasing": "Off", "anisotropic_filtering": "2X", "render_scale": 0.80},
        {"shadow_quality": "Low", "anti_aliasing": "FSR 2", "anisotropic_filtering": "4X", "render_scale": 0.80},
        {"shadow_quality": "Medium", "anti_aliasing": "Off", "anisotropic_filtering": "4X", "render_scale": 0.85},
        {"shadow_quality": "Medium", "anti_aliasing": "SMAA", "anisotropic_filtering": "8X", "render_scale": 0.90},
        {"shadow_quality": "High", "anti_aliasing": "Off", "anisotropic_filtering": "8X", "render_scale": 0.90},
        {"shadow_quality": "High", "anti_aliasing": "FSR 2", "anisotropic_filtering": "16X", "render_scale": 1.00},
        {"shadow_quality": "Highest", "anti_aliasing": "SMAA", "anisotropic_filtering": "16X", "render_scale": 1.00},
        {"shadow_quality": "Highest", "anti_aliasing": "FSR 2", "anisotropic_filtering": "8X", "render_scale": 1.10},
    ]

def build_plan_settings(plan_entry, defaults):
    out = defaults.copy()
    out.update(plan_entry)
    return out

if use_test_plan and "test_plan" not in st.session_state:
    st.session_state["test_plan"] = build_test_plan(default_settings)
    st.session_state["test_plan_index"] = 0
    st.session_state["pending_prefill"] = build_plan_settings(
        st.session_state["test_plan"][0], default_settings
    )
    st.session_state["next_run_updated"] = True
    try:
        st.rerun()
    except AttributeError:
        st.experimental_rerun()

with st.expander("Game settings used for this run (edit if needed)", expanded=True):
    st.caption("Auto-filled from your best run so far. You can edit before testing.")
    col1, col2, col3 = st.columns(3)
    with col1:
        game_name = st.text_input("Game", key="game_name")
        resolution = st.text_input("Resolution", key="resolution")
    with col2:
        shadow_quality = st.selectbox("Shadow Quality", QUALITY_LEVELS, key="shadow_quality")
        anti_aliasing = st.selectbox("Anti-Aliasing", AA_OPTIONS, key="anti_aliasing")
    with col3:
        anisotropic_filtering = st.selectbox("Anisotropic Filtering", AF_OPTIONS, key="anisotropic_filtering")
        render_scale = st.number_input(
            "Render Resolution Scale",
            min_value=0.3,
            max_value=1.5,
            step=0.05,
            key="render_scale",
        )

st.divider()

duration = st.selectbox("Monitoring duration (seconds)", [30, 60, 90, 120], index=1)
sample_every = st.selectbox("Sample every (seconds)", [0.5, 1.0, 2.0], index=1)

st.subheader("FPS Capture (PresentMon)")

process_name = st.text_input(
    "Game process name (EXE)",
    value="GenshinImpact.exe",
    help="Must match the running game exe name (e.g., GenshinImpact.exe)."
)

use_presentmon = st.toggle("Auto-detect FPS using PresentMon", value=True)

if use_presentmon and not PRESENTMON_EXE.exists():
    st.error(f"PresentMon exe not found at: {PRESENTMON_EXE}\n\nPut presentmon.exe inside Game_Optimizer/tools/")

colA, colB = st.columns([1, 1])
with colA:
    start_btn = st.button("▶️ Start Monitoring", use_container_width=True)
with colB:
    clear_btn = st.button("🧹 Clear current run (not saved)", use_container_width=True)

if clear_btn:
    st.session_state.pop("calib_df", None)
    st.session_state.pop("calib_summary", None)
    st.session_state.pop("calib_run_id", None)
    st.success("Cleared current run in memory.")

if start_btn:
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
    st.session_state["calib_run_id"] = run_id
    presentmon_csv = FPS_DIR / f"{run_id}_presentmon.csv"
    pm_proc = None

    if use_presentmon:
        try:
            pm_proc = run_presentmon_capture(process_name, int(duration), presentmon_csv)
            st.caption("PresentMon started (FPS will be computed automatically after the run).")
        except Exception as e:
            st.warning(f"PresentMon could not start. You can still enter FPS manually.\nReason: {e}")
            pm_proc = None

    st.warning("Monitoring started. Alt+Tab to the game NOW and play normally.")
    progress = st.progress(0)

    # Monitor with a simple progress bar
    rows = []
    start_time = time.time()
    total = float(duration)

    while True:
        elapsed = time.time() - start_time
        if elapsed >= total:
            break

        rows.append(sample_metrics())
        progress.progress(min(1.0, elapsed / total))
        time.sleep(float(sample_every))

    df = pd.DataFrame(rows)
    st.session_state["calib_df"] = df
    st.session_state["calib_summary"] = summarize(df)

    auto_fps = None
    if use_presentmon and pm_proc is not None:
        # Ensure PresentMon finishes
        try:
            pm_proc.wait(timeout=int(duration) + 10)
        except Exception:
            pass

        try:
            auto_fps = compute_avg_fps_from_presentmon(presentmon_csv)
            st.session_state["auto_fps"] = auto_fps
            st.success(f"✅ Detected Average FPS (PresentMon): {auto_fps:.1f}")
        except Exception as e:
            st.warning(f"PresentMon ran but FPS could not be computed.\nReason: {e}")

    st.success("Monitoring complete. Now enter average FPS below and save the run.")

st.divider()

df = st.session_state.get("calib_df")
summary = st.session_state.get("calib_summary", {})
run_id = st.session_state.get("calib_run_id")

if df is None:
    st.caption("No monitoring data yet. Click **Start Monitoring** above.")
else:
    st.subheader("Captured samples")
    st.dataframe(df, use_container_width=True)

    st.subheader("Summary (what we’ll use for ML later)")
    st.json(summary)

    st.divider()
    st.subheader("Enter FPS + Save this run")

    fps = st.number_input("Average FPS during the 60s", min_value=1.0, max_value=500.0, value=60.0, step=1.0)

    save_btn = st.button("💾 Save Run to CSV", type="primary", use_container_width=True)

    if save_btn:
        # Build a single-row summary record (good for training dataset)
        record = {
            "run_id": run_id,
            "saved_at_local": datetime.now().isoformat(),
            "game": game_name,
            "resolution": resolution,
            "shadow_quality": shadow_quality,
            "anti_aliasing": anti_aliasing,
            "anisotropic_filtering": anisotropic_filtering,
            "render_scale": float(render_scale),
            "duration_sec": int(duration),
            "sample_every_sec": float(sample_every),
            "avg_fps": float(fps),
            **summary,
        }

        # Save metrics samples file
        metrics_path = RUNS_DIR / f"{run_id}_metrics.csv"
        df.to_csv(metrics_path, index=False)

        # Append to master runs table
        runs_table_path = RUNS_DIR / "runs_table.csv"
        row_df = pd.DataFrame([record])

        if runs_table_path.exists():
            old = pd.read_csv(runs_table_path)
            new = pd.concat([old, row_df], ignore_index=True)
            new.to_csv(runs_table_path, index=False)
        else:
            row_df.to_csv(runs_table_path, index=False)

        st.success("Saved.")
        st.write("Saved files:")
        st.code(str(metrics_path))
        st.code(str(runs_table_path))

        st.info(
            "Next: Do **10–20 runs** (same fixed settings) at different situations "
            "(cool laptop vs warm laptop, different areas in game), then we’ll train the recommender."
        )

        # Update auto-fill settings to the current best run and refresh UI
        if st.session_state.get("use_test_plan") and st.session_state.get("test_plan"):
            plan = st.session_state["test_plan"]
            idx = int(st.session_state.get("test_plan_index", 0))
            idx = (idx + 1) % len(plan)
            st.session_state["test_plan_index"] = idx
            next_settings = build_plan_settings(plan[idx], default_settings)
            st.session_state["pending_prefill"] = next_settings
            st.session_state["next_run_updated"] = True
        else:
            best_row = get_best_run(runs_table_path)
            if best_row is not None:
                recommended = build_recommended_settings(best_row, default_settings)
                st.session_state["pending_prefill"] = recommended
                st.session_state["prefill_source_run_id"] = str(best_row.get("run_id", "")).strip()
                st.session_state["next_run_updated"] = True

        st.session_state["just_saved_run_id"] = run_id
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()
