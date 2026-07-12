import csv
from pathlib import Path
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
RESNET_DIR = ROOT / "tree" / "image_classification" / "convolutional_networks" / "resnet"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RESNET_DIR))

from tree.image_classification.convolutional_networks.resnet.resnet import resnet18

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
EPOCHS = 20
LEARNING_RATE = 0.001

OUTPUT_DIR = ROOT / "tree" / "outputs" / "classification" / "resnet18"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def validate(model, loader, criterion):
    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

    return total_loss / total_samples, total_correct / total_samples


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)

    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [45000, 5000], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = resnet18(num_classes=10).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    history = []
    best_val_accuracy = 0

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_accuracy = validate(model, val_loader, criterion)

        epoch_time = time.time() - start_time

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "time_seconds": epoch_time
        })

        torch.save(model.state_dict(), OUTPUT_DIR / "last.pt")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), OUTPUT_DIR / "best.pt")

        print(f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} | train_acc={train_accuracy:.4f} | val_loss={val_loss:.4f} | val_acc={val_accuracy:.4f} | time={epoch_time:.1f}s")

    with open(OUTPUT_DIR / "history.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    print("Best validation accuracy:", best_val_accuracy)


if __name__ == "__main__":
    main()
