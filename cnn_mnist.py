import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from dataclasses import dataclass
import random
import numpy as np
from typing import Literal

_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class Config:
    conv_filters: tuple = (32, 64)
    fc_hidden: int = 128
    optimizer: Literal["sgd","adam","adamw"] = "adam"
    lr: float = 1e-3
    momentum: float = 0.9
    batch_size: int = 128
    epochs: int = 5
    dropout: float = 0.25
    weight_decay: float = 0.0
    device: str = _device
    val_split: float = 0.1
    num_workers: int = 0
    pin_memory: bool = False

def get_dataloaders(cfg: Config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_full = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    n_train = int((1 - cfg.val_split) * len(train_full))
    n_val = len(train_full) - n_train
    train, val = random_split(train_full, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    val_loader   = DataLoader(val,   batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    test_loader  = DataLoader(test,  batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    return train_loader, val_loader, test_loader

class CNN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        c1, c2 = cfg.conv_filters
        self.conv1 = nn.Conv2d(1, c1, 3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten_dim = c2 * 7 * 7
        self.fc1 = nn.Linear(self.flatten_dim, cfg.fc_hidden)
        self.fc2 = nn.Linear(cfg.fc_hidden, 10)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_optimizer(model: nn.Module, cfg: Config):
    if cfg.optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    raise ValueError("optimizer debe ser 'sgd', 'adam' o 'adamw'")

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        loss_sum += loss.item() * y.size(0)
    return loss_sum / total, correct / total

def train_one_experiment(cfg: Config):
    set_seed(42)
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    model = CNN(cfg).to(cfg.device)
    optimizer = make_optimizer(model, cfg)
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_state = None
    print(f"device={cfg.device} params={count_params(model):,}")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for x, y in train_loader:
            x, y = x.to(cfg.device), y.to(cfg.device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            preds = logits.argmax(dim=1)
            running_correct += (preds == y).sum().item()
            running_total += y.size(0)
            running_loss += loss.item() * y.size(0)
        train_loss = running_loss / running_total
        train_acc = running_correct / running_total
        val_loss, val_acc = evaluate(model, val_loader, cfg.device)
        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_acc = evaluate(model, test_loader, cfg.device)
    print(f"[TEST] loss={test_loss:.4f} acc={test_acc:.4f}")
    return test_acc, model

if __name__ == "__main__":
    cfg = Config(conv_filters=(32, 64), fc_hidden=128, optimizer="adam", lr=1e-3, epochs=5, batch_size=128, dropout=0.25, weight_decay=0.0)
    train_one_experiment(cfg)
