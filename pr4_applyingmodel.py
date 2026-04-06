import argparse
import json
from pathlib import Path
from collections import Counter

import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

# mapping classes → causes
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

non_loss_classes = {"Forest", "River", "SeaLake"}

mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


def load_model(model_dir, num_classes, device):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(
        torch.load(model_dir / "best_model.pth", map_location=device)
    )

    model.to(device)
    model.eval()

    return model


def predict(model, img_path, idx_to_class, device):
    try:
        img = Image.open(img_path).convert("RGB")
    except:
        print("could not read:", img_path.name)
        return None, None, None

    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(x)
        probs = F.softmax(out, dim=1)[0]

    conf, idx = probs.max(0)

    cls = idx_to_class[str(idx.item())]
    cause = cause_map.get(cls, "Unknown")

    return cls, conf.item(), cause


def make_summary(results):
    class_count = Counter()
    cause_count = Counter()

    loss_list = []
    non_loss_list = []

    for r in results:
        if r["predicted_class"] is None:
            continue

        class_count[r["predicted_class"]] += 1
        cause_count[r["cause"]] += 1

        if r["predicted_class"] in non_loss_classes:
            non_loss_list.append(r)
        else:
            loss_list.append(r)

    total = len(results)
    loss_total = len(loss_list)

    filtered = {
        k: v for k, v in cause_count.items()
        if k not in {"Natural Feature (not loss)", "Still Forested"}
    }

    if len(filtered) > 0:
        top = sorted(filtered.items(), key=lambda x: -x[1])[0]
    else:
        top = ("None", 0)

    summary = {
        "total": total,
        "loss": loss_total,
        "non_loss": total - loss_total,
        "dominant_cause": top[0],
        "dominant_count": top[1],
        "dominant_pct": round(top[1] / max(loss_total, 1) * 100, 1),
        "cause_breakdown": dict(cause_count),
        "class_breakdown": dict(class_count),
    }

    return summary


def print_summary(summary, region):
    print("\n" + "=" * 50)
    print("FOREST LOSS ANALYSIS:", region)
    print("=" * 50)

    print("total patches:", summary["total"])
    print("loss patches:", summary["loss"])
    print("non-loss patches:", summary["non_loss"])

    print("\ndominant cause:", summary["dominant_cause"])
    print("count:", summary["dominant_count"],
          "(", summary["dominant_pct"], "% )")

    print("\ncauses:")
    for k, v in summary["cause_breakdown"].items():
        print(" ", k, ":", v)

    print("=" * 50 + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patches_dir", type=str, default="./target_patches")
    parser.add_argument("--model_dir", type=str, default="./model_output")
    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--region", type=str, default="Target Region")

    args = parser.parse_args()

    patches_dir = Path(args.patches_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # load class names
    with open(model_dir / "class_names.json") as f:
        data = json.load(f)

    idx_to_class = data["idx_to_class"]
    num_classes = len(idx_to_class)

    model = load_model(model_dir, num_classes, device)
    print("model loaded\n")

    # load metadata if exists
    meta_file = patches_dir / "patch_metadata.json"
    if meta_file.exists():
        with open(meta_file) as f:
            meta_data = {m["file"]: m for m in json.load(f)}
    else:
        meta_data = {}

    # get images
    imgs = list(patches_dir.glob("*.png")) + \
           list(patches_dir.glob("*.jpg")) + \
           list(patches_dir.glob("*.tif"))

    imgs = sorted(imgs)

    print("total patches:", len(imgs))

    results = []

    for i, img_path in enumerate(imgs):
        cls, conf, cause = predict(model, img_path, idx_to_class, device)

        meta = meta_data.get(img_path.name, {})

        results.append({
            "file": img_path.name,
            "lon": meta.get("lon"),
            "lat": meta.get("lat"),
            "predicted_class": cls,
            "confidence": round(conf, 4) if conf is not None else None,
            "cause": cause,
        })

        if (i + 1) % 20 == 0 or (i + 1) == len(imgs):
            print("processed:", i + 1, "/", len(imgs))

    summary = make_summary(results)

    print_summary(summary, args.region)

    with open(output_dir / "patch_predictions.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("saved results to:", output_dir)


if __name__ == "__main__":
    main()