import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

ROOT = Path.cwd()
OLD_DATASET = ROOT / "dataset"
NEW_DATASET = ROOT / "dataset_visual_split"

CITIES        = ["istanbul", "nyc", "tokyo"]
SPLITS        = ["train", "val", "test"]
TARGET_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

BATCH_SIZE          = 64
NUM_WORKERS         = 4
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_EPS         = 0.15
DEFAULT_MIN_SAMPLES = 2


class ImagePathDataset(Dataset):
    _tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    def __init__(self, paths: list[Path]):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img  = Image.open(path).convert("RGB")
        return self._tfm(img), str(path)


def build_encoder() -> nn.Module:
    weights   = models.ResNet50_Weights.IMAGENET1K_V2
    model     = models.resnet50(weights=weights)
    model.fc  = nn.Identity()
    model.eval()
    return model.to(DEVICE)


@torch.no_grad()
def compute_embeddings(paths: list[Path], model: nn.Module) -> np.ndarray:
    dataset  = ImagePathDataset(paths)
    loader   = DataLoader(dataset, batch_size=BATCH_SIZE,
                          num_workers=NUM_WORKERS, pin_memory=True)
    all_embs = []
    for imgs, _ in tqdm(loader, desc="  Embedding", leave=False):
        imgs = imgs.to(DEVICE)
        embs = model(imgs).cpu().numpy()
        all_embs.append(embs)
    embs = np.concatenate(all_embs, axis=0).astype(np.float32)
    return normalize(embs, norm="l2")


def cluster_images(embeddings: np.ndarray, eps: float,
                   min_samples: int) -> np.ndarray:
    db     = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean",
                    algorithm="ball_tree", n_jobs=-1)
    labels = db.fit_predict(embeddings)
    next_id = labels.max() + 1
    for i, lbl in enumerate(labels):
        if lbl == -1:
            labels[i] = next_id
            next_id  += 1
    return labels


def assign_clusters_to_splits(
    labels: np.ndarray,
    target: dict[str, float],
) -> dict[int, str]:
    total       = len(labels)
    targets_abs = {s: target[s] * total for s in SPLITS}
    clusters: dict[int, list[int]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        clusters[int(lbl)].append(i)
    sorted_clusters = sorted(clusters.items(), key=lambda kv: -len(kv[1]))
    counts      = {s: 0 for s in SPLITS}
    assignment: dict[int, str] = {}
    for cid, members in sorted_clusters:
        deficit         = {s: targets_abs[s] - counts[s] for s in SPLITS}
        chosen          = max(deficit, key=deficit.__getitem__)
        assignment[cid] = chosen
        counts[chosen] += len(members)
    return assignment


def process_city(city: str, model: nn.Module,
                 eps: float, min_samples: int) -> dict[str, int]:
    print(f"\n{'='*60}")
    print(f"  City: {city.upper()}")
    print(f"{'='*60}")

    paths: list[Path] = []
    for split in SPLITS:
        city_dir = OLD_DATASET / split / city
        if not city_dir.exists():
            print(f"  [warn] {city_dir} does not exist, skipping.")
            continue
        found = (sorted(city_dir.glob("*.jpg")) +
                 sorted(city_dir.glob("*.jpeg")) +
                 sorted(city_dir.glob("*.png")))
        paths.extend(found)

    if not paths:
        print(f"  [warn] No images found for {city}. Skipping.")
        return {}

    print(f"  Total images : {len(paths)}")
    print("  Computing embeddings …")
    embeddings = compute_embeddings(paths, model)

    print(f"  Clustering  (eps={eps}, min_samples={min_samples}) …")
    labels     = cluster_images(embeddings, eps=eps, min_samples=min_samples)
    n_clusters = len(set(labels))
    print(f"  Clusters formed: {n_clusters}")

    assignment = assign_clusters_to_splits(labels, TARGET_RATIOS)

    counts: dict[str, int] = {s: 0 for s in SPLITS}
    for img_path, lbl in zip(paths, labels):
        split   = assignment[int(lbl)]
        dst_dir = NEW_DATASET / split / city
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, dst_dir / img_path.name)
        counts[split] += 1

    for s in SPLITS:
        pct = 100 * counts[s] / len(paths) if paths else 0
        print(f"    {s:5s}: {counts[s]:5d} images  ({pct:.1f}%)")

    return counts


def main():
    parser = argparse.ArgumentParser(
        description="Re-split dataset by visual similarity via DBSCAN clustering.")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS,
                        help="DBSCAN eps on L2-normalised embeddings. Default: %(default)s")
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES,
                        help="DBSCAN min_samples. Default: %(default)s")
    parser.add_argument("--root", type=Path, default=None,
                        help="Override project root (default: cwd)")
    args = parser.parse_args()

    global ROOT, OLD_DATASET, NEW_DATASET
    if args.root:
        ROOT        = args.root
        OLD_DATASET = ROOT / "dataset"
        NEW_DATASET = ROOT / "dataset_visual_split"

    print(f"Project root : {ROOT}")
    print(f"Source       : {OLD_DATASET}")
    print(f"Destination  : {NEW_DATASET}")
    print(f"Device       : {DEVICE}")

    if not OLD_DATASET.exists():
        sys.exit(f"[error] Source dataset not found: {OLD_DATASET}")

    if NEW_DATASET.exists():
        print(f"\n[warn] {NEW_DATASET} already exists — removing it.")
        shutil.rmtree(NEW_DATASET)

    model = build_encoder()

    grand_totals: dict[str, int] = {s: 0 for s in SPLITS}
    for city in CITIES:
        counts = process_city(city, model, eps=args.eps,
                              min_samples=args.min_samples)
        for s in SPLITS:
            grand_totals[s] += counts.get(s, 0)

    print(f"\n{'='*60}")
    print("  GRAND TOTALS")
    print(f"{'='*60}")
    total_imgs = sum(grand_totals.values())
    for s in SPLITS:
        pct = 100 * grand_totals[s] / total_imgs if total_imgs else 0
        print(f"  {s:5s}: {grand_totals[s]:6d} images  ({pct:.1f}%)")
    print(f"  {'total':5s}: {total_imgs:6d} images")
    print(f"\nNew dataset written to: {NEW_DATASET}")


if __name__ == "__main__":
    main()
