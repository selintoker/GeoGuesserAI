import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

DATASET_DIR = "dataset"
NUM_CLASSES = 3
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_TYPE = "resnet50"
CHECKPOINT_PATH = f"best_model_{MODEL_TYPE}.pth"

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

EVAL_TRANSFORMS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model():
    if MODEL_TYPE == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.encoder.layers[-1].parameters():
            param.requires_grad = True
        model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    else:
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in model.parameters():
            param.requires_grad = False
        for param in model.layer4.parameters():
            param.requires_grad = True
        model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(DEVICE)


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, total_correct = 0.0, 0

    with torch.set_grad_enabled(training):
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * inputs.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()

    n = len(loader.dataset)
    return total_loss / n, total_correct / n


def main():
    print(f"Using device: {DEVICE}")

    datasets_ = {
        split: datasets.ImageFolder(
            os.path.join(DATASET_DIR, split),
            transform=TRAIN_TRANSFORMS if split == "train" else EVAL_TRANSFORMS,
        )
        for split in ("train", "val")
    }

    loaders = {
        split: DataLoader(ds, batch_size=BATCH_SIZE, shuffle=(split == "train"),
                          num_workers=NUM_WORKERS, pin_memory=True)
        for split, ds in datasets_.items()
    }

    print(f"Classes: {datasets_['train'].classes}")
    print(f"Train: {len(datasets_['train'])}  Val: {len(datasets_['val'])}\n")

    model = build_model()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(trainable_params, lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0.0
    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, loaders["train"], criterion, optimizer)
        val_loss, val_acc = run_epoch(model, loaders["val"], criterion)
        scheduler.step()

        print(f"Epoch {epoch:>2}/{NUM_EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights, CHECKPOINT_PATH)
            print(f"  => Saved best model (val_acc={best_val_acc:.4f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Model saved to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
