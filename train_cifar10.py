# train_cifar10.py
import os
import time
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_top1(logits, targets):
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total


class MLPHead(nn.Module):
    """2-layer head: 512 -> hidden -> 10"""
    def __init__(self, in_dim=512, hidden_dim=2048, out_dim=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class ResNet18WithHead(nn.Module):
    def __init__(self, head: nn.Module):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_dim = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Identity()
        self.head = head
        assert hasattr(self.head, "forward")

    def forward(self, x):
        h = self.backbone(x)       # (B,512)
        logits = self.head(h)      # (B,10)
        return logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--hidden_dim", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_path", type=str, default="./best.pt")
    args = ap.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # CIFAR-10 transforms
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_ds = datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=train_tf)
    test_ds  = datasets.CIFAR10(root=args.data_dir, train=False, download=True, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    # Build model
    head = MLPHead(in_dim=512, hidden_dim=args.hidden_dim, out_dim=10)
    model = ResNet18WithHead(head).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    head_params = sum(p.numel() for p in model.head.parameters())
    print(f"Params total: {total_params:,} | head: {head_params:,}")

    # Optim
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()

        running_loss = 0.0
        running_acc = 0.0
        n = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            running_loss += loss.item() * bs
            running_acc += accuracy_top1(logits.detach(), y) * bs
            n += bs

        scheduler.step()

        train_loss = running_loss / n
        train_acc = running_acc / n
        test_acc = evaluate(model, test_loader, device)
        dt = time.time() - t0

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d} | {dt:.1f}s | lr {lr_now:.4f} | "
            f"loss {train_loss:.4f} | train {train_acc*100:.2f}% | test {test_acc*100:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "best_acc": best_acc, "args": vars(args)},
                args.save_path
            )
            print(f"✅ Saved best checkpoint: {args.save_path} (acc={best_acc*100:.2f}%)")

    print(f"Done. Best test acc: {best_acc*100:.2f}%")

if __name__ == "__main__":
    main()
