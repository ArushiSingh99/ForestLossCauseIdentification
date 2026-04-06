"""
map of patches
bar chart
confidence graph
"""

import argparse
import json
from pathlib import Path
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


# colours for each cause
col_map = {
    "Agricultural Expansion": "#e67e22",
    "Agricultural Expansion / Grazing": "#f39c12",
    "Urbanization": "#c0392b",
    "Urbanization / Infrastructure": "#e74c3c",
    "Infrastructure Development": "#8e44ad",
    "Degradation / Natural Transition": "#7f8c8d",
    "Natural Feature (not loss)": "#2ecc71",
    "Still Forested": "#27ae60",
    "Unknown": "#bdc3c7",
}


def make_map(res, out_path):
    lon = []
    lat = []
    col = []
    cause_list = []

    for r in res:
        if r["lon"] is None or r["lat"] is None:
            continue

        c = r.get("cause", "Unknown")

        lon.append(r["lon"])
        lat.append(r["lat"])
        col.append(col_map.get(c, "#bdc3c7"))
        cause_list.append(c)

    if len(lon) == 0:
        print("no points for map")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(lon, lat, c=col, s=30, alpha=0.7)

    # legend
    unique_causes = sorted(list(set(cause_list)))
    leg = []
    for c in unique_causes:
        leg.append(patches.Patch(color=col_map.get(c, "#bdc3c7"), label=c))

    ax.legend(handles=leg, fontsize=8)

    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title("patch map")

    ax.grid(True)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close()

    print("map saved:", out_path)


def make_bar(sum_data, out_path):
    causes = list(sum_data["cause_breakdown"].keys())
    counts = list(sum_data["cause_breakdown"].values())

    cols = []
    for c in causes:
        cols.append(col_map.get(c, "#bdc3c7"))

    order = np.argsort(counts)

    causes = [causes[i] for i in order]
    counts = [counts[i] for i in order]
    cols   = [cols[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(causes, counts, color=cols)

    for i in range(len(bars)):
        ax.text(counts[i], i, str(counts[i]))

    ax.set_xlabel("count")
    ax.set_title("cause distribution")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close()

    print("bar saved:", out_path)


def make_hist(res, out_path):
    conf = []

    for r in res:
        if r["confidence"] is not None:
            conf.append(r["confidence"])

    if len(conf) == 0:
        print("no confidence data")
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(conf, bins=20)

    avg = np.mean(conf)
    ax.axvline(avg, linestyle="--")

    ax.set_xlabel("confidence")
    ax.set_ylabel("count")
    ax.set_title("confidence graph")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close()

    print("hist saved:", out_path)


def show_summary(s):
    print("\nsummary")

    print("patches:", s["total"])
    print("loss:", s["loss"])
    print("non-loss:", s["non_loss"])

    print("main cause:", s["dominant_cause"], s["dominant_pct"], "%")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results_dir", type=str, default="./results")
    p.add_argument("--output_dir", type=str, default="./results/plots")

    a = p.parse_args()

    res_dir = Path(a.results_dir)
    out_dir = Path(a.output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(res_dir / "patch_predictions.json") as f:
        res = json.load(f)

    with open(res_dir / "summary.json") as f:
        summ = json.load(f)

    show_summary(summ)

    print("making plots...")

    make_map(res, out_dir / "map.png")
    make_bar(summ, out_dir / "bar.png")
    make_hist(res, out_dir / "hist.png")

    print("done, check:", out_dir)


if __name__ == "__main__":
    main()