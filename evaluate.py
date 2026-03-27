import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATASET_DIR = "dataset"
CHECKPOINT_PATH = "best_model_resnet50.pth"
NUM_CLASSES = 3
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model():
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    return model.to(DEVICE)


def main():
    dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"), transform=EVAL_TRANSFORMS)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    classes = dataset.classes
    print(f"Classes: {classes}")
    print(f"Test images: {len(dataset)}\n")

    model = build_model()
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            preds = model(inputs).argmax(1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    # Overall accuracy
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    overall_acc = correct / len(all_labels)
    random_baseline = 1 / len(classes)
    print(f"Overall accuracy: {correct}/{len(all_labels)} ({overall_acc:.4f})")
    print(f"Random baseline:  {random_baseline:.4f}  (improvement: {overall_acc - random_baseline:+.4f})\n")

    # Per-class metrics
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 56)
    for i, cls in enumerate(classes):
        tp = sum(p == i and l == i for p, l in zip(all_preds, all_labels))
        fp = sum(p == i and l != i for p, l in zip(all_preds, all_labels))
        fn = sum(p != i and l == i for p, l in zip(all_preds, all_labels))
        support = sum(l == i for l in all_labels)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"{cls:<12} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10}")

    # Confusion matrix
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    header = f"{'':>12}" + "".join(f"{c:>12}" for c in classes)
    print(header)
    for i, row_cls in enumerate(classes):
        row = f"{row_cls:>12}" + "".join(
            f"{sum(p == j and l == i for p, l in zip(all_preds, all_labels)):>12}"
            for j in range(len(classes))
        )
        print(row)


if __name__ == "__main__":
    main()
