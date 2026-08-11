"""
CIFAR-10 画像分類プログラム
シンプルな CNN (畳み込みニューラルネットワーク) を構築し、CIFAR-10 データセットで
学習・評価を行う。

必要なライブラリ:
    pip install torch torchvision

実行方法:
    python analysis.py
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------
# 設定
# ---------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./data"
BATCH_SIZE = 128
NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
NUM_CLASSES = 10
CLASSES = (
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)


# ---------------------------------------------------------
# データセットの準備
# ---------------------------------------------------------
def get_dataloaders(batch_size: int = BATCH_SIZE):
    # CIFAR-10 の各チャンネルの平均・標準偏差 (よく使われる値)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_set = datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=train_transform
    )
    test_set = datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return train_loader, test_loader


# ---------------------------------------------------------
# CNN モデル定義
# ---------------------------------------------------------
class SimpleCNN(nn.Module):
    """
    3 つの畳み込みブロック (Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU -> MaxPool)
    + 全結合層 2 層 からなるシンプルな CNN。
    """

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16 -> 8x8
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8 -> 4x4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------
# 学習・評価ループ
# ---------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate_per_class(model, loader, device, num_classes: int = NUM_CLASSES):
    model.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels)

        for i in range(labels.size(0)):
            label = labels[i].item()
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    for i, class_name in enumerate(CLASSES):
        acc = 100.0 * class_correct[i] / class_total[i]
        print(f"  {class_name:>6s}: {acc:5.1f}% ({class_correct[i]}/{class_total[i]})")


# ---------------------------------------------------------
# メイン処理
# ---------------------------------------------------------
def main():
    print(f"使用デバイス: {DEVICE}")

    train_loader, test_loader = get_dataloaders()

    model = SimpleCNN(NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_acc = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, DEVICE)
        scheduler.step()

        elapsed = time.time() - start
        print(
            f"Epoch [{epoch:2d}/{NUM_EPOCHS}] "
            f"train_loss: {train_loss:.4f} train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} test_acc: {test_acc:.4f} "
            f"({elapsed:.1f}s)"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_cifar10_cnn.pth")

    print(f"\nベスト test_acc: {best_acc:.4f}")
    print("\nクラスごとの精度 (最終エポック時点のモデル):")
    evaluate_per_class(model, test_loader, DEVICE)


if __name__ == "__main__":
    main()
