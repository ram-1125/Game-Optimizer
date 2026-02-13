import base64
import platform
import subprocess
from pathlib import Path

import psutil
import streamlit as st


# -----------------------------
# Paths
# -----------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
ASSETS = PROJECT_ROOT / "assets"


# -----------------------------
# Helpers
# -----------------------------
def b64_image(path: Path) -> str:
    if not path.exists():
        return ""
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def pick_asset(stem: str) -> Path | None:
    """Return assets/<stem>.jpg or .jpeg if present."""
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        p = ASSETS / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def get_gpu_name_windows() -> str:
    try:
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


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Game Optimizer", page_icon="🎮", layout="wide")

#
# Theme (fixed to Dark)
#

# -----------------------------
# Load collage images (supports .jpg/.jpeg mix)
# Your files: cs2(.jpg), fortnite(.jpg), genshin(.jpeg), valorant(.jpg), gta5(.jpeg)
# -----------------------------
img_val = b64_image(pick_asset("valorant") or Path(""))
img_for = b64_image(pick_asset("fortnite") or Path(""))
img_cs = b64_image(pick_asset("cs2") or Path(""))
img_gen = b64_image(pick_asset("genshin") or Path(""))
img_gta = b64_image(pick_asset("gta5") or Path(""))

# -----------------------------
# System badge data (2)
# -----------------------------
os_name = f"{platform.system()} {platform.release()}"
cpu_cores = psutil.cpu_count(logical=True) or 0
ram_total_gb = bytes_to_gb(psutil.virtual_memory().total)
gpu_name = get_gpu_name_windows() if platform.system().lower() == "windows" else "Unknown GPU"


# -----------------------------
# Futuristic CSS + Animations (1) + Hover glow (3)
# -----------------------------
is_light = False

# CSS color variables for theme
if is_light:
    bg_css = """
    background:
      radial-gradient(1100px 650px at 15% 15%, rgba(145, 70, 255, 0.10), transparent 60%),
      radial-gradient(900px 600px at 85% 20%, rgba(0, 255, 198, 0.10), transparent 55%),
      linear-gradient(180deg, #f6f7fb 0%, #eef1f7 100%);
    """
    text_main = "rgba(10, 15, 25, 0.92)"
    text_muted = "rgba(10, 15, 25, 0.70)"
    glass_bg = "rgba(255,255,255,0.55)"
    glass_border = "rgba(10,15,25,0.10)"
    hero_overlay = "linear-gradient(90deg, rgba(245,247,251,0.92) 0%, rgba(245,247,251,0.65) 55%, rgba(245,247,251,0.92) 100%)"
else:
    bg_css = """
    background:
      radial-gradient(1200px 700px at 10% 10%, rgba(145, 70, 255, 0.18), transparent 60%),
      radial-gradient(900px 600px at 90% 20%, rgba(0, 255, 198, 0.12), transparent 55%),
      radial-gradient(700px 500px at 60% 90%, rgba(0, 153, 255, 0.10), transparent 55%),
      linear-gradient(180deg, #0b0f19 0%, #070a12 100%);
    """
    text_main = "rgba(255,255,255,0.92)"
    text_muted = "rgba(255,255,255,0.74)"
    glass_bg = "rgba(255,255,255,0.06)"
    glass_border = "rgba(255,255,255,0.12)"
    hero_overlay = "linear-gradient(90deg, rgba(10,14,26,0.88) 0%, rgba(10,14,26,0.55) 55%, rgba(10,14,26,0.85) 100%)"


st.markdown(
    f"""
<style>
:root {{
  --scan-duration: 3.8s;
  /* 5 slides * scan duration */
  --slide-duration: 19s;
  --slide-2-delay: 3.8s;
  --slide-3-delay: 7.6s;
  --slide-4-delay: 11.4s;
  --slide-5-delay: 15.2s;
}}

/* Background */
.stApp {{
  {bg_css}
  color: {text_main};
}}

.block-container {{
  padding-top: 2rem;
  padding-bottom: 3rem;
}}

/* HERO */
.hero {{
  position: relative;
  border-radius: 26px;
  padding: 34px 34px 26px 34px;
  overflow: hidden;
  border: 1px solid {glass_border};
  background: {glass_bg};
  box-shadow: 0 20px 80px rgba(0,0,0,0.35);
}}

/* Hero slideshow */
.hero-slideshow {{
  position: absolute;
  inset: 0;
  z-index: 0;
  pointer-events: none;
}}
.hero-slide {{
  position: absolute;
  inset: 0;
  opacity: 0;
  background-size: cover;
  background-position: center;
  filter: saturate(1.2) contrast(1.1) blur(0.2px);
  transform: scale(1.02);
  animation: heroSlide var(--slide-duration) linear infinite;
}}
.hero-slide.slide-1 {{ background-image: url("data:image/jpeg;base64,{img_val}"); animation-delay: 0s; }}
.hero-slide.slide-2 {{ background-image: url("data:image/jpeg;base64,{img_for}"); animation-delay: var(--slide-2-delay); }}
.hero-slide.slide-3 {{ background-image: url("data:image/jpeg;base64,{img_cs}"); animation-delay: var(--slide-3-delay); }}
.hero-slide.slide-4 {{ background-image: url("data:image/jpeg;base64,{img_gen}"); animation-delay: var(--slide-4-delay); }}
.hero-slide.slide-5 {{ background-image: url("data:image/jpeg;base64,{img_gta}"); animation-delay: var(--slide-5-delay); }}

@keyframes heroSlide {{
  0%   {{ opacity: 0; }}
  4%   {{ opacity: 0.28; }}
  16%  {{ opacity: 0.28; }}
  20%  {{ opacity: 0; }}
  100% {{ opacity: 0; }}
}}

/* Readability overlay */
.hero::after {{
  content: "";
  position: absolute;
  inset: 0;
  background: {hero_overlay};
  z-index: 1;
  pointer-events: none;
}}

/* Neon scan line animation (1) */
.scanline {{
  position: absolute;
  left: -10%;
  right: -10%;
  top: 10%;
  height: 2px;
  background: linear-gradient(90deg,
    rgba(145,70,255,0),
    rgba(145,70,255,0.85),
    rgba(0,255,198,0.75),
    rgba(0,153,255,0.85),
    rgba(0,153,255,0)
  );
  filter: blur(0.2px);
  opacity: 0.9;
  animation: scan var(--scan-duration) linear infinite;
  z-index: 2;
}}

@keyframes scan {{
  0%   {{ transform: translateY(0px); opacity: 0.0; }}
  10%  {{ opacity: 0.95; }}
  50%  {{ transform: translateY(260px); opacity: 0.80; }}
  90%  {{ opacity: 0.95; }}
  100% {{ transform: translateY(520px); opacity: 0.0; }}
}}

.hero-inner {{
  position: relative;
  z-index: 3;
}}

.kicker {{
  display: inline-flex;
  gap: 10px;
  align-items: center;
  font-size: 0.9rem;
  padding: 6px 12px;
  border-radius: 999px;
  border: 1px solid {glass_border};
  background: {glass_bg};
  color: {text_main};
}}

.h-title {{
  margin: 14px 0 8px 0;
  font-size: 2.35rem;
  line-height: 1.15;
  font-weight: 850;
  letter-spacing: -0.02em;
  color: {text_main};
}}

.h-sub {{
  margin: 0 0 12px 0;
  font-size: 1.05rem;
  color: {text_muted};
  max-width: 980px;
}}

.glowline {{
  height: 1px;
  margin: 14px 0 18px 0;
  background: linear-gradient(90deg,
    rgba(145,70,255,0.0),
    rgba(145,70,255,0.7),
    rgba(0,255,198,0.55),
    rgba(0,153,255,0.6),
    rgba(0,153,255,0.0)
  );
}}

/* Glass panels */
.glass {{
  border-radius: 18px;
  padding: 18px 18px;
  border: 1px solid {glass_border};
  background: {glass_bg};
  box-shadow: 0 12px 40px rgba(0,0,0,0.18);
}}

/* Cards (3) */
.card {{
  border-radius: 18px;
  padding: 18px 18px;
  border: 1px solid {glass_border};
  background: {glass_bg};
  transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
  height: 100%;
}}
.card:hover {{
  transform: translateY(-4px);
  border-color: rgba(145,70,255,0.45);
  box-shadow: 0 16px 60px rgba(145,70,255,0.10), 0 14px 40px rgba(0,0,0,0.18);
}}

.card-title {{
  font-size: 1.05rem;
  font-weight: 780;
  margin-bottom: 6px;
  color: {text_main};
}}
.card-desc {{
  color: {text_muted};
  font-size: 0.95rem;
  margin-bottom: 10px;
}}
.small {{
  color: {text_muted};
  font-size: 0.88rem;
}}

/* Button glow (3) - applies to st.page_link and st.button */
.stButton > button, a[data-testid="stPageLink-NavLink"] {{
  border-radius: 14px !important;
  border: 1px solid {glass_border} !important;
  background: {glass_bg} !important;
  color: {text_main} !important;
  transition: transform 160ms ease, box-shadow 160ms ease, border-color 160ms ease !important;
}}
.stButton > button:hover, a[data-testid="stPageLink-NavLink"]:hover {{
  transform: translateY(-2px);
  border-color: rgba(0,255,198,0.55) !important;
  box-shadow: 0 0 0 2px rgba(0,255,198,0.12), 0 14px 40px rgba(0,0,0,0.18);
}}

/* System badge */
.sysbadge {{
  display: inline-flex;
  gap: 14px;
  flex-wrap: wrap;
  align-items: center;
  padding: 10px 12px;
  border-radius: 16px;
  border: 1px solid {glass_border};
  background: {glass_bg};
}}
.sysitem {{
  display: inline-flex;
  gap: 8px;
  align-items: baseline;
}}
.syslabel {{
  font-size: 0.82rem;
  color: {text_muted};
}}
.sysval {{
  font-size: 0.92rem;
  font-weight: 720;
  color: {text_main};
}}
</style>
""",
    unsafe_allow_html=True
)

# -----------------------------
# HERO content
# -----------------------------
st.markdown(
    f"""
<div class="hero">
  <div class="hero-slideshow">
    <span class="hero-slide slide-1"></span>
    <span class="hero-slide slide-2"></span>
    <span class="hero-slide slide-3"></span>
    <span class="hero-slide slide-4"></span>
    <span class="hero-slide slide-5"></span>
  </div>
  <div class="scanline"></div>
  <div class="hero-inner">
    <div class="kicker">⚡ AI-Assisted Gaming Performance Optimizer</div>
    <div class="h-title">Futuristic tuning for maximum FPS stability on <span style="color: rgba(0,255,198,0.92);">your</span> system</div>
    <div class="h-sub">
      Hardware-aware recommendations + calibration runs that learn which settings stay stable on your laptop.
      Built for integrated graphics systems and real-world gameplay.
    </div>
    <div class="sysbadge" style="margin-top: 10px;">
      <div class="sysitem"><span class="syslabel">OS</span> <span class="sysval">{os_name}</span></div>
      <div class="sysitem"><span class="syslabel">CPU cores</span> <span class="sysval">{cpu_cores}</span></div>
      <div class="sysitem"><span class="syslabel">RAM</span> <span class="sysval">{ram_total_gb} GB</span></div>
      <div class="sysitem"><span class="syslabel">GPU</span> <span class="sysval">{gpu_name}</span></div>
    </div>
    <div class="glowline"></div>
  </div>
</div>
""",
    unsafe_allow_html=True
)

st.write("")

# -----------------------------
# Summary panels
# -----------------------------
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        """
<div class="glass">
  <div class="card-title">📌 What it does</div>
  <div class="small">Recommends resolution / quality / render scale using performance + system stress.</div>
</div>
""",
        unsafe_allow_html=True
    )
with c2:
    st.markdown(
        """
<div class="glass">
  <div class="card-title">🧪 How it learns</div>
  <div class="small">You run ~10 calibrations → we score FPS vs CPU/RAM stability → select best profile.</div>
</div>
""",
        unsafe_allow_html=True
    )
with c3:
    st.markdown(
        """
<div class="glass">
  <div class="card-title">🧠 Why it’s reliable</div>
  <div class="small">Empirical data from your own laptop (not online benchmarks). Perfect for iGPU systems.</div>
</div>
""",
        unsafe_allow_html=True
    )

st.write("")
st.subheader("Choose a path")

# -----------------------------
# Navigation cards
# -----------------------------
colA, colB, colC = st.columns(3)

with colA:
    st.markdown(
        """
<div class="card">
  <div class="card-title">⚡ Quick Recommend</div>
  <div class="card-desc">Instant baseline from hardware-only analysis.</div>
  <div class="small">Best for: first-time users, fast demo.</div>
</div>
""",
        unsafe_allow_html=True
    )
    st.page_link("pages/1_Quick_Recommend.py", label="Open Quick Recommend", use_container_width=True)

with colB:
    st.markdown(
        """
<div class="card">
  <div class="card-title">🎯 Calibrate</div>
  <div class="card-desc">Monitor 60s while playing + enter average FPS.</div>
  <div class="small">Best for: accurate personalized recommendation.</div>
</div>
""",
        unsafe_allow_html=True
    )
    st.page_link("pages/2_Calibrate.py", label="Open Calibrate", use_container_width=True)

with colC:
    st.markdown(
        """
<div class="card">
  <div class="card-title">📊 Results</div>
  <div class="card-desc">Score runs (FPS vs stability) and select best settings.</div>
</div>
""",
        unsafe_allow_html=True
    )
    st.page_link("pages/3_Results.py", label="Open Results", use_container_width=True)

st.write("")
st.caption("Assets loaded locally from /assets (semi-transparent collage). Toggle Light mode from the sidebar.")
