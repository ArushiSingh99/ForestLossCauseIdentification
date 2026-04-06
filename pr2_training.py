import os
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# seed
seed_val = 42
torch.manual_seed(seed_val)

# normalisation
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]


def get_tf(split):
    if split == "train":
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def make_model(n_cls, dev):
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_f = m.fc.in_features
    m.fc = nn.Linear(in_f, n_cls)
    return m.to(dev)


def train_one(m, ld, loss_fn, opt, dev):
    m.train()

    tot_loss = 0
    correct = 0
    total = 0

    for imgs, labs in ld:
        imgs = imgs.to(dev)
        labs = labs.to(dev)

        opt.zero_grad()

        out = m(imgs)
        loss = loss_fn(out, labs)

        loss.backward()
        opt.step()

        tot_loss += loss.item() * imgs.size(0)

        _, pred = out.max(1)
        correct += pred.eq(labs).sum().item()
        total += labs.size(0)

    return tot_loss / total, correct / total


def eval_one(m, ld, loss_fn, dev):
    m.eval()

    tot_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labs in ld:
            imgs = imgs.to(dev)
            labs = labs.to(dev)

            out = m(imgs)
            loss = loss_fn(out, labs)

            tot_loss += loss.item() * imgs.size(0)

            _, pred = out.max(1)
            correct += pred.eq(labs).sum().item()
            total += labs.size(0)

    return tot_loss / total, correct / total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="./EuroSAT_split")
    p.add_argument("--output_dir", type=str, default="./model_output")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)

    a = p.parse_args()

    data_dir = Path(a.data_dir)
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # device
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", dev)

    # datasets
    data = {}
    for s in ["train", "validation", "test"]:
        data[s] = datasets.ImageFolder(
            root=data_dir / s,
            transform=get_tf(s)
        )

    loaders = {}
    for s in data:
        loaders[s] = DataLoader(
            data[s],
            batch_size=a.batch_size,
            shuffle=(s == "train"),
            num_workers=2,
            pin_memory=True
        )

    classes = data["train"].classes
    n_cls = len(classes)

    print("classes:", classes)

    # save class map
    c2i = data["train"].class_to_idx
    i2c = {}
    for k in c2i:
        i2c[c2i[k]] = k

    with open(out_dir / "class_names.json", "w") as f:
        json.dump({"class_to_idx": c2i, "idx_to_class": i2c}, f)

    # model stuff
    m = make_model(n_cls, dev)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(m.parameters(), lr=a.lr)
    sched = optim.lr_scheduler.StepLR(opt, step_size=7, gamma=0.1)

    best_acc = 0
    hist = []

    for ep in range(1, a.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = train_one(m, loaders["train"], loss_fn, opt, dev)
        v_loss, v_acc = eval_one(m, loaders["validation"], loss_fn, dev)

        sched.step()

        t1 = time.time() - t0

        print(ep, "train:", round(tr_loss,4), round(tr_acc,3),
              "val:", round(v_loss,4), round(v_acc,3),
              "time:", round(t1,1))

        hist.append({
            "ep": ep,
            "tr_loss": tr_loss,
            "tr_acc": tr_acc,
            "v_loss": v_loss,
            "v_acc": v_acc
        })

        if v_acc > best_acc:
            best_acc = v_acc
            torch.save(m.state_dict(), out_dir / "best_model.pth")
            print("saved best")

    # test
    print("\ntesting...")
    m.load_state_dict(torch.load(out_dir / "best_model.pth", map_location=dev))

    t_loss, t_acc = eval_one(m, loaders["test"], loss_fn, dev)
    print("test:", t_loss, t_acc)

    # save history
    with open(out_dir / "training_history.json", "w") as f:
        json.dump({"hist": hist, "test_acc": t_acc}, f)

    print("done")


if __name__ == "__main__":
    main()