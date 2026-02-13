import argparse
import os
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--avg_fps", required=True, type=float)
    ap.add_argument("--system_log", default="data/system_log.csv")
    ap.add_argument("--out", default="data/training_data.csv")
    args = ap.parse_args()

    sys_df = pd.read_csv(args.system_log)
    run_df = sys_df[sys_df["run_id"] == args.run_id].copy()
    if run_df.empty:
        raise ValueError(f"No rows found for run_id={args.run_id} in {args.system_log}")

    row = {
        "run_id": args.run_id,
        "game": run_df["game"].iloc[0],
        "mode": run_df["mode"].iloc[0],
        "resolution": run_df["resolution"].iloc[0],
        "texture": run_df["texture"].iloc[0],
        "shadows": run_df["shadows"].iloc[0],
        "aa": run_df["aa"].iloc[0],
        "render_scale": run_df["render_scale"].iloc[0],
        "cpu_usage_pct": run_df["cpu_usage_pct"].mean(),
        "ram_usage_pct": run_df["ram_usage_pct"].mean(),
        "avg_fps": float(args.avg_fps)
    }

    out_df = pd.DataFrame([row])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if os.path.exists(args.out):
        existing = pd.read_csv(args.out)
        combined = pd.concat([existing, out_df], ignore_index=True)
        combined.to_csv(args.out, index=False)
    else:
        out_df.to_csv(args.out, index=False)

    print(f"✅ Added {args.run_id} with avg_fps={args.avg_fps:.2f} to {args.out}")

if __name__ == "__main__":
    main()
