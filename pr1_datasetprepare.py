import os
import argparse
import shutil
import json
from pathlib import Path
import pandas as pd


# class mapping
cause_map = {
    "AnnualCrop": "Agricultural Expansion",
    "PermanentCrop": "Agricultural Expansion",
    "Pasture": "Agricultural Expansion / Grazing",
    "Residential": "Urbanization",
    "Industrial": "Urbanization / Infrastructure",
    "Highway": "Infrastructure Development",
    "HerbaceousVegetation": "Degradation / Natural Transition",
    "River": "Natural Feature (not loss)",
    "SeaLake": "Natural Feature (not loss)",
    "Forest": "Still Forested",
}


def read_split_csv(csv_path):
    df = pd.read_csv(csv_path)


    df.columns = df.columns.str.strip()

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "filename":   str(row["Filename"]),
            "label":      int(row["Label"]),
            "class_name": str(row["ClassName"]),
        })

    return rows


def copy_data(rows, data_dir, out_dir, split_name):
    missing = 0
    copied = 0
    class_count = {}

    for r in rows:
        cls = r["class_name"]
        file_name = r["filename"]

        src = data_dir / file_name

        if not src.exists():
            src = data_dir / cls / Path(file_name).name

        if not src.exists():
            missing += 1
            continue

        dst_dir = out_dir / split_name / cls
        dst_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(src, dst_dir / src.name)

        if cls not in class_count:
            class_count[cls] = 0
        class_count[cls] += 1

        copied += 1

    print("\n", split_name, "done")
    print("copied:", copied, "missing:", missing)

    for k in sorted(class_count.keys()):
        v = class_count[k]
        cause = cause_map.get(k, "Unknown")
        print(k, v, "->", cause)


def show_label_map(data_dir):
    p = data_dir / "label_map.json"

    if p.exists():
        f = open(p)
        data = json.load(f)
        f.close()

        print("\nlabel map:")
        for k in sorted(data.keys()):
            print(k, "->", data[k])
    else:
        print("no label_map.json found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./EuroSAT")
    parser.add_argument("--output_dir", type=str, default="./EuroSAT_split")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)

    if not data_dir.exists():
        print("data dir not found")
        return

    show_label_map(data_dir)

    if out_dir.exists():
        print("output folder already exists, delete and rerun")
        return

    train_x = data_dir / "train.csv"
    val_x = data_dir / "validation.csv"
    test_x = data_dir / "test.csv"

    splits = {
        "train": train_x,
        "validation": val_x,
        "test": test_x
    }

    for name in splits:
        csv = splits[name]

        if not csv.exists():
            print("missing file:", csv)
            continue

        print("\nreading", csv)
        rows = read_split_csv(csv)
        print("entries:", len(rows))

        copy_data(rows, data_dir, out_dir, name)

    print("\ndone")
    print("saved at:", out_dir)
    print("run next script now")


if __name__ == "__main__":
    main()