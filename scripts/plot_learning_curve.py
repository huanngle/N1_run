import os, glob
import pandas as pd
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

RUN_DIR = "/home/huanngle/Huan/IsaacLab/logs/rsl_rl/PJ2/2026-04-24_04-22-55_walk_run_v6"
OUT_DIR = os.path.join(RUN_DIR, "report_plots")
os.makedirs(OUT_DIR, exist_ok=True)

print("RUN_DIR =", RUN_DIR)
print("Files in RUN_DIR:", os.listdir(RUN_DIR))

event_files = glob.glob(os.path.join(RUN_DIR, "*.tfevents.*"))
if not event_files:
    raise FileNotFoundError("No TensorBoard event files found under run directory.")

# pick the largest event file (usually main)
event_file = max(event_files, key=os.path.getsize)
ea = EventAccumulator(event_file, size_guidance={"scalars": 0})
ea.Reload()

scalar_tags = ea.Tags().get("scalars", [])
print("Found scalar tags:", scalar_tags)

def load_scalar(tag):
    s = ea.Scalars(tag)
    return pd.DataFrame({"step":[x.step for x in s], "value":[x.value for x in s]})

# ---- choose tags that exist in your run ----
# Adjust these based on printout above.
CANDIDATES = {
    "reward": [
        "Train/mean_reward", "train/mean_reward", "Episode/rew_mean", "Mean/episode_reward",
        "Episode_Reward/total", "Episode_Reward/reward"
    ],
    "ep_len": [
        "Train/mean_episode_length", "train/mean_episode_length", "Episode/len_mean",
        "Mean/episode_length", "Episode_Length/mean"
    ],
    "fallen": [
        "Episode_Termination/fallen", "Episode_Termination/fall", "Termination/fallen"
    ],
    "bad_orientation": [
        "Episode_Termination/bad_orientation", "Termination/bad_orientation"
    ],
    "time_out": [
        "Episode_Termination/time_out", "Termination/time_out"
    ],
}

def first_existing(tags):
    for t in tags:
        if t in scalar_tags:
            return t
    return None

reward_tag = first_existing(CANDIDATES["reward"])
len_tag    = first_existing(CANDIDATES["ep_len"])
fallen_tag = first_existing(CANDIDATES["fallen"])

print("Using reward_tag:", reward_tag)
print("Using len_tag:", len_tag)
print("Using fallen_tag:", fallen_tag)

def plot_tag(tag, title, fname):
    df = load_scalar(tag)
    plt.figure()
    plt.plot(df["step"], df["value"])
    plt.title(title)
    plt.xlabel("iteration/step")
    plt.ylabel("value")
    plt.grid(True)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, fname)
    plt.savefig(out, dpi=200)
    print("Saved:", out)

if reward_tag:
    plot_tag(reward_tag, "Learning curve: Episode reward", "curve_reward.png")
if len_tag:
    plot_tag(len_tag, "Learning curve: Episode length", "curve_episode_length.png")
if fallen_tag:
    plot_tag(fallen_tag, "Learning curve: Fall rate (fallen termination)", "curve_fall_rate.png")

# also export reward breakdown terms if present
for t in scalar_tags:
    if t.startswith("Episode_Reward/"):
        safe = t.replace("/", "_")
        plot_tag(t, f"Reward term: {t}", f"{safe}.png")
