import os
import json
import copy
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.feature_extraction.text import TfidfVectorizer


def get_image_paths_from_imagefolder(dataset):
    return [path for path, _ in dataset.samples]


def run_or_load_ocr(image_paths: List[str], cache_path: Path) -> Dict[str, str]:
    if cache_path.exists():
        print(f"Loading OCR cache from {cache_path}")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print("OCR cache not found. Running EasyOCR...")

    try:
        import easyocr
    except ImportError:
        raise ImportError("easyocr is not installed. Run: pip install easyocr")

    # EasyOCR compatibility note:
    # Japanese can only be combined with English, not Turkish.
    # Turkish uses Latin script, so English OCR still captures many Turkish signs.
    reader = easyocr.Reader(["en", "ja"], gpu=torch.cuda.is_available())

    ocr_results = {}

    for i, img_path in enumerate(image_paths):
        try:
            results = reader.readtext(img_path, detail=0)
            text = " ".join(results)
        except Exception as e:
            print(f"OCR failed for {img_path}: {e}")
            text = ""

        ocr_results[img_path] = text

        if (i + 1) % 500 == 0:
            print(f"OCR processed {i + 1}/{len(image_paths)} images")

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(ocr_results, f, indent=2, ensure_ascii=False)

    print(f"Saved OCR cache to {cache_path}")
    return ocr_results


class ImageTextDataset(torch.utils.data.Dataset):
    def __init__(self, imagefolder_dataset, text_features):
        self.imagefolder_dataset = imagefolder_dataset
        self.text_features = text_features

    def __len__(self):
        return len(self.imagefolder_dataset)

    def __getitem__(self, idx):
        image, label = self.imagefolder_dataset[idx]
        img_path, _ = self.imagefolder_dataset.samples[idx]

        text_vec = self.text_features[img_path]
        text_vec = torch.tensor(text_vec, dtype=torch.float32)

        return image, text_vec, label


class LateFusionResNetText(nn.Module):
    def __init__(self, num_classes, text_dim):
        super().__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        image_dim = resnet.fc.in_features
        resnet.fc = nn.Identity()

        self.image_encoder = resnet

        self.text_projector = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(image_dim + 256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, images, text_vecs):
        image_features = self.image_encoder(images)
        text_features = self.text_projector(text_vecs)
        fused = torch.cat([image_features, text_features], dim=1)
        return self.classifier(fused)


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, text_vecs, labels in loader:
            images = images.to(device, non_blocking=True)
            text_vecs = text_vecs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images, text_vecs)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)

            running_loss += loss.item() * images.size(0)
            running_correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    loss = running_loss / total if total > 0 else 0.0
    acc = running_correct / total if total > 0 else 0.0
    return loss, acc, all_preds, all_labels


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, text_vecs, labels in loader:
        images = images.to(device, non_blocking=True)
        text_vecs = text_vecs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images, text_vecs)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    loss = running_loss / total if total > 0 else 0.0
    acc = running_correct / total if total > 0 else 0.0
    return loss, acc


def per_class_accuracy(preds, labels, class_names):
    results = {}

    for class_idx, class_name in enumerate(class_names):
        cls_total = 0
        cls_correct = 0

        for p, y in zip(preds, labels):
            if y == class_idx:
                cls_total += 1
                if p == y:
                    cls_correct += 1

        results[class_name] = cls_correct / cls_total if cls_total > 0 else 0.0

    return results


def main():
    project_dir = Path("/share/sablab/nfs02/users/ac2492/projects/geog/GeoGuesserAI")

    data_dir = project_dir / "dataset_visual_split"
    output_dir = project_dir / "outputs_late_fusion_ocr"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    test_dir = data_dir / "test"

    batch_size = 32
    num_epochs = 10
    learning_rate = 1e-4
    weight_decay = 1e-4
    num_workers = 8
    image_size = 224
    max_text_features = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_imagefolder = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_imagefolder = datasets.ImageFolder(val_dir, transform=eval_transforms)
    test_imagefolder = datasets.ImageFolder(test_dir, transform=eval_transforms)

    class_names = train_imagefolder.classes
    num_classes = len(class_names)

    print("Classes:", class_names)
    print("Class to index:", train_imagefolder.class_to_idx)
    print("Train size:", len(train_imagefolder))
    print("Val size:", len(val_imagefolder))
    print("Test size:", len(test_imagefolder))

    all_image_paths = (
        get_image_paths_from_imagefolder(train_imagefolder)
        + get_image_paths_from_imagefolder(val_imagefolder)
        + get_image_paths_from_imagefolder(test_imagefolder)
    )

    ocr_cache_path = output_dir / "ocr_cache.json"
    ocr_text = run_or_load_ocr(all_image_paths, ocr_cache_path)

    train_paths = get_image_paths_from_imagefolder(train_imagefolder)
    val_paths = get_image_paths_from_imagefolder(val_imagefolder)
    test_paths = get_image_paths_from_imagefolder(test_imagefolder)

    train_texts = [ocr_text.get(p, "") for p in train_paths]
    val_texts = [ocr_text.get(p, "") for p in val_paths]
    test_texts = [ocr_text.get(p, "") for p in test_paths]

    vectorizer = TfidfVectorizer(
        max_features=max_text_features,
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
    )

    print("Fitting TF-IDF vectorizer on training OCR text only...")
    train_tfidf = vectorizer.fit_transform(train_texts).toarray()
    val_tfidf = vectorizer.transform(val_texts).toarray()
    test_tfidf = vectorizer.transform(test_texts).toarray()

    text_dim = train_tfidf.shape[1]
    print(f"Text feature dimension: {text_dim}")

    train_text_features = {
        path: train_tfidf[i]
        for i, path in enumerate(train_paths)
    }
    val_text_features = {
        path: val_tfidf[i]
        for i, path in enumerate(val_paths)
    }
    test_text_features = {
        path: test_tfidf[i]
        for i, path in enumerate(test_paths)
    }

    train_dataset = ImageTextDataset(train_imagefolder, train_text_features)
    val_dataset = ImageTextDataset(val_imagefolder, val_text_features)
    test_dataset = ImageTextDataset(test_imagefolder, test_text_features)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = LateFusionResNetText(
        num_classes=num_classes,
        text_dim=text_dim,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_val_acc = 0.0
    best_epoch = -1
    best_model_wts = copy.deepcopy(model.state_dict())

    history = []

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_model_wts = copy.deepcopy(model.state_dict())

            checkpoint = {
                "epoch": best_epoch,
                "model_state_dict": best_model_wts,
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "class_names": class_names,
                "class_to_idx": train_imagefolder.class_to_idx,
                "vectorizer_vocabulary": vectorizer.vocabulary_,
                "text_dim": text_dim,
                "max_text_features": max_text_features,
            }

            torch.save(
                checkpoint,
                output_dir / "best_late_fusion_ocr_resnet50.pth"
            )

            print("Saved new best late-fusion OCR model.")

    print(f"\nBest validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")

    with open(output_dir / "train_history_late_fusion_ocr.json", "w") as f:
        json.dump(history, f, indent=2)

    model.load_state_dict(best_model_wts)

    test_loss, test_acc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Acc:  {test_acc:.4f}")

    class_accs = per_class_accuracy(test_preds, test_labels, class_names)

    print("\nPer-class test accuracy:")
    for cls, acc in class_accs.items():
        print(f"{cls}: {acc:.4f}")

    results = {
        "best_val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "per_class_test_accuracy": class_accs,
        "class_names": class_names,
        "data_dir": str(data_dir),
        "model_type": "ResNet50 + OCR TF-IDF late fusion",
        "ocr_languages": ["en", "ja"],
    }

    with open(output_dir / "final_results_late_fusion_ocr.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()
