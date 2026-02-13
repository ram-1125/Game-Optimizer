import json
import os
import shutil
import subprocess
import traceback
from datetime import datetime
from pathlib import Path

import pandas as pd
import psutil
import streamlit as st

APP_NAME = "Game Optimizer"
RUNS_DIR = Path("runs")
MODEL_PATH = Path("models") / "fps_model.joblib"


def record_error(context):
    st.session_state["last_error"] = {
        "context": context,
        "traceback": traceback.format_exc(),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }


def ensure_runs_dir():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return RUNS_DIR


def new_run_id():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path):
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_metadata(run_id, metadata):
    run_dir = RUNS_DIR / run_id
    save_json(run_dir / "metadata.json", metadata)
    return run_dir


def save_result(run_id, avg_fps=None, skipped=False):
    run_dir = RUNS_DIR / run_id
    payload = {
        "run_id": run_id,
        "avg_fps": None if avg_fps is None else float(avg_fps),
        "skipped": bool(skipped),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    save_json(run_dir / "result.json", payload)

    summary_path = run_dir / "summary.csv"
    df = pd.DataFrame([payload])
    df.to_csv(summary_path, index=False)
    return payload


def compute_system_averages(log_path):
    if not log_path.exists():
        return None, None
    try:
        df = pd.read_csv(log_path)
        if df.empty:
            return None, None
        return float(df["cpu_usage_pct"].mean()), float(df["ram_usage_pct"].mean())
    except Exception:
        return None, None


def format_profile(metadata):
    if not metadata:
        return "Unknown"
    res = metadata.get("resolution", "Unknown")
    texture = metadata.get("texture", "Unknown")
    shadows = metadata.get("shadows", "Unknown")
    aa = metadata.get("aa", "Unknown")
    scale = metadata.get("render_scale", "Unknown")
    return f"{res} | texture {texture}, shadows {shadows}, AA {aa}, scale {scale}%"


def load_runs():
    runs = []
    if not RUNS_DIR.exists():
        return runs
    for run_dir in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        metadata = load_json(run_dir / "metadata.json")
        result = load_json(run_dir / "result.json")
        log_path = run_dir / "system_log.csv"
        avg_cpu, avg_ram = compute_system_averages(log_path)
        runs.append(
            {
                "run_id": run_dir.name,
                "profile": format_profile(metadata),
                "avg_cpu": avg_cpu,
                "avg_ram": avg_ram,
                "avg_fps": result.get("avg_fps"),
                "metadata": metadata,
                "result": result,
                "log_path": log_path,
                "run_dir": run_dir,
            }
        )
    return runs


def get_gpu_name():
    try:
        result = subprocess.check_output(
            ["wmic", "path", "win32_VideoController", "get", "name"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        lines = [
            line.strip()
            for line in result.splitlines()
            if line.strip() and line.strip().lower() != "name"
        ]
        if lines:
            return ", ".join(lines)
    except Exception:
        return "Unknown"
    return "Unknown"


def get_hardware_info():
    cpu_name = os.getenv("PROCESSOR_IDENTIFIER", "").strip()
    if not cpu_name:
        cpu_name = os.getenv("PROCESSOR_ARCHITECTURE", "").strip()
    ram_gb = psutil.virtual_memory().total / (1024**3)
    cores = psutil.cpu_count(logical=False) or psutil.cpu_count(logical=True) or 0
    gpu_name = get_gpu_name()
    return {
        "cpu_name": cpu_name or "Unknown CPU",
        "ram_gb": ram_gb,
        "cores": cores,
        "gpu_name": gpu_name,
    }


def classify_tier(info):
    ram_gb = info["ram_gb"]
    cores = info["cores"]
    if ram_gb < 8 or cores <= 4:
        return "low"
    if ram_gb < 16 or cores <= 8:
        return "medium"
    return "high"


def baseline_settings_for_tier(tier):
    if tier == "low":
        return {
            "resolution": "1280x720",
            "texture": "low",
            "shadows": "low",
            "aa": "off",
            "render_scale": 80,
            "rationale": "Designed to maximize FPS on entry-level hardware.",
        }
    if tier == "medium":
        return {
            "resolution": "1600x900",
            "texture": "medium",
            "shadows": "medium",
            "aa": "taa",
            "render_scale": 90,
            "rationale": "Balances image quality and smoothness for mid-tier rigs.",
        }
    return {
        "resolution": "1920x1080",
        "texture": "high",
        "shadows": "high",
        "aa": "taa",
        "render_scale": 100,
        "rationale": "A quality-forward baseline for high-tier hardware.",
    }


def normalize(values):
    if not values:
        return []
    min_v = min(values)
    max_v = max(values)
    if max_v == min_v:
        return [1.0 for _ in values]
    return [(v - min_v) / (max_v - min_v) for v in values]


def score_runs(runs, mode):
    fps_values = [r["avg_fps"] for r in runs]
    cpu_values = [r["avg_cpu"] for r in runs]
    ram_values = [r["avg_ram"] for r in runs]

    fps_norm = normalize(fps_values)
    cpu_norm = normalize(cpu_values)
    ram_norm = normalize(ram_values)

    scored = []
    for idx, run in enumerate(runs):
        fps = fps_norm[idx]
        cpu = cpu_norm[idx]
        ram = ram_norm[idx]
        if mode == "Max FPS":
            score = fps
        elif mode == "Stable (recommended)":
            stability = 1.0 - 0.5 * (cpu + ram)
            score = 0.7 * fps + 0.3 * stability
        else:
            balance = (1.0 - cpu + 1.0 - ram) / 2.0
            score = 0.6 * fps + 0.4 * balance
        scored.append((score, run))
    scored.sort(key=lambda item: item[0], reverse=True)
    return scored


def list_tree(root, max_depth=2):
    root = Path(root)
    if not root.exists():
        return []
    lines = []
    for path in root.rglob("*"):
        depth = len(path.relative_to(root).parts)
        if depth > max_depth:
            continue
        prefix = "  " * (depth - 1)
        label = f"{prefix}- {path.name}"
        lines.append(label)
    return lines


def reset_runs_folder():
    if RUNS_DIR.exists():
        shutil.rmtree(RUNS_DIR)

