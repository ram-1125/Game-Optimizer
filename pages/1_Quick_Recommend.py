import platform
import subprocess
from pathlib import Path

import psutil
import streamlit as st


def get_gpu_name_windows():
    """Best-effort GPU name on Windows without extra libraries."""
    try:
        # WMIC works on many Windows installs; if not, it will fail gracefully.
        out = subprocess.check_output(
            ["wmic", "path", "win32_VideoController", "get", "name"],
            text=True,
            stderr=subprocess.DEVNULL
        )
        lines = [l.strip() for l in out.splitlines() if l.strip() and l.strip().lower() != "name"]
        return lines[0] if lines else "Unknown GPU"
    except Exception:
        return "Unknown GPU"


def bytes_to_gb(b: int) -> float:
    return round(b / (1024**3), 2)


def recommend_from_hardware(ram_gb: float, cpu_cores: int, gpu_name: str):
    """
    Hardware-only heuristics: conservative baseline recommendations.
    For Intel iGPU / low VRAM systems, prioritize stability.
    """
    gpu_low = ("intel" in gpu_name.lower()) or ("uhd" in gpu_name.lower()) or ("iris" in gpu_name.lower())

    # Baseline defaults
    res = "1280x720"
    quality = "Lowest"
    render_scale = 0.8
    target_fps = "30–45"

    # Adjust based on RAM and CPU
    if ram_gb >= 16 and cpu_cores >= 8 and not gpu_low:
        res = "1600x900"
        quality = "Medium"
        render_scale = 1.0
        target_fps = "50–60"
    elif ram_gb >= 16 and cpu_cores >= 8 and gpu_low:
        res = "1280x720"
        quality = "Low"
        render_scale = 0.9
        target_fps = "45–60"
    elif ram_gb >= 8 and cpu_cores >= 4 and gpu_low:
        res = "1280x720"
        quality = "Lowest"
        render_scale = 0.8
        target_fps = "30–45"
    else:
        res = "1280x720"
        quality = "Lowest"
        render_scale = 0.7
        target_fps = "25–35"

    reasoning = [
        f"RAM detected: ~{ram_gb} GB (more RAM allows higher textures/stability).",
        f"CPU cores: {cpu_cores} (more cores helps background + game threads).",
        f"GPU detected: {gpu_name} (integrated GPUs need lower render scale).",
        "This is a **hardware-only** baseline. Use **Calibrate** to get personalized recommendations."
    ]

    return {
        "resolution": res,
        "quality": quality,
        "render_scale": render_scale,
        "expected_fps": target_fps,
        "reasoning": reasoning,
    }


st.set_page_config(page_title="Quick Recommend", page_icon="⚡", layout="wide")
st.title("⚡ Quick Recommend (Hardware-only)")

st.write(
    "This page gives an **instant baseline** recommendation using your system hardware only. "
    "For best accuracy, do 10+ runs in **Calibrate**."
)

st.divider()

# Hardware snapshot
cpu_name = platform.processor() or "Unknown CPU"
cpu_cores = psutil.cpu_count(logical=True) or 0
ram_total_gb = bytes_to_gb(psutil.virtual_memory().total)
os_name = f"{platform.system()} {platform.release()}"
gpu_name = get_gpu_name_windows() if platform.system().lower() == "windows" else "Unknown GPU"

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("OS", os_name)
    st.metric("CPU", cpu_name)
with col2:
    st.metric("CPU Cores (logical)", str(cpu_cores))
    st.metric("RAM (GB)", str(ram_total_gb))
with col3:
    st.metric("GPU", gpu_name)

st.divider()

rec = recommend_from_hardware(ram_total_gb, cpu_cores, gpu_name)

st.subheader("Recommended baseline settings")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Resolution", rec["resolution"])
c2.metric("Graphics Quality", rec["quality"])
c3.metric("Render Scale", str(rec["render_scale"]))
c4.metric("Expected FPS", rec["expected_fps"])

st.markdown("### Why these settings?")
for r in rec["reasoning"]:
    st.write("•", r)

st.info(
    "Next step: go to **Calibrate** and do ~10 runs (60s each) while playing. "
    "Then **Results** will compute the best settings based on real performance."
)