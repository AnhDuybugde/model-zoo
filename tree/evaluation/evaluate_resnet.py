import csv
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
RESNET_DIR = ROOT / "tree" / "image_classification" / "convolutional_networks" / "resnet"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RESNET_DIR))

from tree.image_classification.convolutional_networks.resnet.resnet import resnet18

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 128
OUTPUT_DIR = ROOT / "tree" / "outputs" / "classification" / "resnet18"
CHECKPOINT_PATH = OUTPUT_DIR / "best.pt"


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    test_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = resnet18(num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)

            logits = model(images)
            predictions = logits.argmax(dim=1).cpu()

            all_labels.extend(labels.tolist())
            all_predictions.extend(predictions.tolist())

    metrics = {
        "model": "resnet18",
        "dataset": "cifar10",
        "accuracy": accuracy_score(all_labels, all_predictions),
        "macro_precision": precision_score(all_labels, all_predictions, average="macro", zero_division=0),
        "macro_recall": recall_score(all_labels, all_predictions, average="macro", zero_division=0),
        "macro_f1": f1_score(all_labels, all_predictions, average="macro", zero_division=0),
        "parameters": sum(parameter.numel() for parameter in model.parameters())
    }

    with open(OUTPUT_DIR / "test_metrics.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=metrics.keys())
        writer.writeheader()
        writer.writerow(metrics)

    for name, value in metrics.items():
        print(f"{name}: {value}")


if __name__ == "__main__":
    main()
