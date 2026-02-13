import argparse
import csv
import os
import time
from datetime import datetime
import psutil

def ensure_csv(path, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/system_log.csv")
    ap.add_argument("--game", required=True)
    ap.add_argument("--mode", default="competitive")
    ap.add_argument("--resolution", default="1920x1080")
    ap.add_argument("--texture", default="high")
    ap.add_argument("--shadows", default="med")
    ap.add_argument("--aa", default="taa")
    ap.add_argument("--render_scale", default="100")
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--interval", type=int, default=2)
    ap.add_argument("--run_id", default="")
    args = ap.parse_args()

    if not args.run_id:
        args.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    fields = [
        "timestamp","run_id",
        "game","mode","resolution","texture","shadows","aa","render_scale",
        "cpu_usage_pct","ram_usage_pct"
    ]
    ensure_csv(args.out, fields)

    start = time.time()
    print("Logging system metrics... (Ctrl+C to stop)")
    try:
        while time.time() - start < args.duration:
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "run_id": args.run_id,
                "game": args.game,
                "mode": args.mode,
                "resolution": args.resolution,
                "texture": args.texture,
                "shadows": args.shadows,
                "aa": args.aa,
                "render_scale": args.render_scale,
                "cpu_usage_pct": psutil.cpu_percent(interval=None),
                "ram_usage_pct": psutil.virtual_memory().percent
            }
            with open(args.out, "a", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=fields).writerow(row)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")
    print(f"Saved: {args.out}")
    print(f"RUN_ID: {args.run_id}")

if __name__ == "__main__":
    main()
