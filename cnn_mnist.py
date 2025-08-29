import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from dataclasses import dataclass, asdict
import random
import numpy as np
from typing import Literal
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
from pathlib import Path
import time, itertools
from sklearn.metrics import confusion_matrix, classification_report

def accuracy_per_kparams(acc: float, n_params: int) -> float:
    return acc / (max(n_params,1) / 1_000)

def confusion_matrix_fig(cm, title="Matriz de Confusión", labels=None, normalize=False, fname=None):
    import matplotlib.pyplot as plt
    import numpy as np

    if labels is None:
        labels = list(range(cm.shape[0]))

    cm_plot = cm.astype(float)
    if normalize:
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm_plot = cm_plot / row_sums

    plt.figure(figsize=(7.2, 6.0))
    plt.imshow(cm_plot, interpolation='nearest')
    plt.title(title)
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta real')
    plt.xticks(ticks=np.arange(len(labels)), labels=labels)
    plt.yticks(ticks=np.arange(len(labels)), labels=labels)

    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            val = f"{cm_plot[i, j]:.2f}" if normalize else f"{int(cm_plot[i, j])}"
            plt.text(j, i, val, ha="center", va="center")

    plt.tight_layout()
    if fname:
        plt.savefig(fname, dpi=160, bbox_inches='tight')
    plt.show()

def top_confusions(cm, k=5):
    pairs = []
    n = cm.shape[0]
    for i, j in itertools.product(range(n), range(n)):
        if i == j: 
            continue
        if cm[i, j] > 0:
            pairs.append((i, j, int(cm[i, j])))
    pairs.sort(key=lambda t: t[2], reverse=True)
    return pairs[:k]

def analyze_confusions(cm_cnn, cm_mlp, k=5):
    top_cnn = top_confusions(cm_cnn, k)
    top_mlp = top_confusions(cm_mlp, k)

    set_cnn = {(a,b) for a,b,_ in top_cnn}
    set_mlp = {(a,b) for a,b,_ in top_mlp}
    inter = list(set_cnn.intersection(set_mlp))

    return {
        "top_cnn": top_cnn,
        "top_mlp": top_mlp,
        "coinciden": inter,
    }

def benchmark_report(best_cnn_dict, mlp_acc, mlp_params, cm_cnn, cm_mlp):
    cnn_name = best_cnn_dict['experiment']
    cnn_acc  = best_cnn_dict['test_accuracy']
    cnn_par  = best_cnn_dict['num_parameters']

    eff_cnn  = accuracy_per_kparams(cnn_acc, cnn_par)
    eff_mlp  = accuracy_per_kparams(mlp_acc, mlp_params)

    mejor_global = "CNN" if cnn_acc >= mlp_acc else "MLP"
    mejor_eficiencia = "CNN" if eff_cnn >= eff_mlp else "MLP"

    conf = analyze_confusions(cm_cnn, cm_mlp, k=6)

    print("\n================= BENCHMARK (Punto 3) =================")
    print(f"• Mejor rendimiento general: {mejor_global}")
    print(f"  - {cnn_name}: accuracy={cnn_acc:.3f} | params={cnn_par:,}")
    print(f"  - MLP:       accuracy={mlp_acc:.3f} | params={mlp_params:,}")

    print(f"\n• Eficiencia (accuracy por 1k params):")
    print(f"  - {cnn_name}: {eff_cnn:.6f}")
    print(f"  - MLP:       {eff_mlp:.6f}")
    print(f"  ⇒ Mayor accuracy/params: {mejor_eficiencia}")

    print("\n• Matrices de confusión:")
    print("  - Guardadas como 'cm_cnn.png' y 'cm_mlp.png' (también versión normalizada).")

    print("\n• Errores de clasificación más frecuentes (top-6 por modelo):")
    def fmt_pairs(pairs):
        return ", ".join([f"{a}→{b}({c})" for a,b,c in pairs]) if pairs else "—"
    print(f"  - CNN: {fmt_pairs(conf['top_cnn'])}")
    print(f"  - MLP: {fmt_pairs(conf['top_mlp'])}")

    if conf['coinciden']:
        coinciden_txt = ", ".join([f"{a}→{b}" for a,b in conf['coinciden']])
        print(f"  - Coincidencias de confusión (ambos modelos): {coinciden_txt}")
    else:
        print("  - Coincidencias de confusión: ninguna relevante.")

    print("\n• Posibles explicaciones:")
    print("  - La CNN explota estructura espacial (convoluciones y pooling) y comparte pesos,")
    print("    lo que suele mejorar generalización con menos parámetros efectivos.")
    print("  - El MLP aplana la imagen y pierde relaciones locales (bordes/textura).")
    print("  - Si hay overfitting: revisar dropout/weight_decay y early stopping.")
    print("  - Si los errores coinciden en ciertas clases, puede deberse a clases visualmente")
    print("    similares o a insuficiencia de muestras para esas clases.")

    return {
        "mejor_global": mejor_global,
        "mejor_eficiencia": mejor_eficiencia,
        "eff_cnn": eff_cnn,
        "eff_mlp": eff_mlp,
        "top_confusions": conf
    }


_device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@dataclass
class Config:
    conv_filters: tuple = (32, 64)
    fc_hidden: int = 128
    optimizer: Literal["sgd","adam","adamw"] = "adam"
    pooling_type: Literal["max", "avg"] = "max"
    lr: float = 1e-3
    momentum: float = 0.9
    batch_size: int = 128
    epochs: int = 3
    dropout: float = 0.25
    weight_decay: float = 0.0
    device: str = _device
    val_split: float = 0.1
    num_workers: int = 0
    pin_memory: bool = False

def load_simple_data():
    print("Generando datos...")
    
    n_train, n_test = 1000, 200
    
    train_images = torch.randn(n_train, 1, 28, 28)
    train_labels = torch.randint(0, 10, (n_train,))
    
    test_images = torch.randn(n_test, 1, 28, 28)
    test_labels = torch.randint(0, 10, (n_test,))
    
    return train_images, train_labels, test_images, test_labels

def get_dataloaders(cfg: Config):
    train_images, train_labels, test_images, test_labels = load_simple_data()
    
    train_full = TensorDataset(train_images, train_labels)
    test_dataset = TensorDataset(test_images, test_labels)
    
    n_train = int((1 - cfg.val_split) * len(train_full))
    n_val = len(train_full) - n_train
    train_dataset, val_dataset = random_split(train_full, [n_train, n_val], 
                                             generator=torch.Generator().manual_seed(42))
    
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

class CNN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        c1, c2 = cfg.conv_filters
        
        self.conv1 = nn.Conv2d(1, c1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(c1, c2, kernel_size=3, padding=1)
        
        if cfg.pooling_type == "max":
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
            
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

class RedMLP(nn.Module):
    def __init__(self, capas_ocultas, activacion='relu', dropout=0.0):
        super(RedMLP, self).__init__()
        
        layers = []
        layers.append(nn.Linear(784, capas_ocultas[0]))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        for i in range(1, len(capas_ocultas)):
            layers.append(nn.Linear(capas_ocultas[i-1], capas_ocultas[i]))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(capas_ocultas[-1], 10))
        self.red = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.red(x)

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def make_optimizer(model: nn.Module, cfg: Config):
    if cfg.optimizer == "sgd":
        return optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    elif cfg.optimizer == "adam":
        return optim.Adam(model.parameters(), lr=cfg.lr)
    elif cfg.optimizer == "adamw":
        return optim.AdamW(model.parameters(), lr=cfg.lr)

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

@torch.no_grad()
def get_predictions(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    all_preds, all_labels = [], []
    
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        
    return np.array(all_preds), np.array(all_labels)

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusión - {title}')
    plt.xlabel('Predicción')
    plt.ylabel('Etiqueta Real')
    plt.show()
    
    return cm

def train_model(model: nn.Module, cfg: Config, model_name: str = "Model"):
    print(f"Entrenando {model_name}")
    
    set_seed(42)
    train_loader, val_loader, test_loader = get_dataloaders(cfg)
    
    model = model.to(cfg.device)
    optimizer = make_optimizer(model, cfg)
    criterion = nn.CrossEntropyLoss()
    
    print(f"Parámetros: {count_params(model):,}")
    
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
        
        print(f"Epoch {epoch} | Train: {train_acc:.3f} | Val: {val_acc:.3f}")
    
    test_loss, test_acc = evaluate(model, test_loader, cfg.device)
    print(f"Test Accuracy: {test_acc:.3f}")
    
    return model, test_acc, None, test_loader

def main():
    print("LABORATORIO CNN - MNIST")
    
    configs = [
        Config(conv_filters=(16, 32), fc_hidden=64, optimizer="adam", epochs=2),
        Config(conv_filters=(32, 64), fc_hidden=128, optimizer="adam", epochs=2),
        Config(conv_filters=(32, 64), fc_hidden=128, optimizer="sgd", lr=1e-2, epochs=2),
    ]
    
    print("\nEXPERIMENTOS CNN")
    results = []
    models = {}
    
    for i, cfg in enumerate(configs):
        exp_name = f"CNN_{i+1}"
        print(f"\nExperimento {exp_name}")
        
        model = CNN(cfg)
        trained_model, test_acc, _, test_loader = train_model(model, cfg, exp_name)
        
        result = {
            'experiment': exp_name,
            'test_accuracy': test_acc,
            'num_parameters': count_params(trained_model),
            'config': asdict(cfg)
        }
        
        results.append(result)
        models[exp_name] = (trained_model, cfg, test_loader)
    
    print("\nMODELO BASELINE MLP")
    baseline_cfg = Config(lr=1e-3, batch_size=64, epochs=2, optimizer="adam")
    baseline_model = RedMLP(capas_ocultas=[256, 128], activacion='relu', dropout=0.2)
    trained_baseline, baseline_acc, _, baseline_test_loader = train_model(
        baseline_model, baseline_cfg, "MLP Baseline"
    )
    
    print("\nCOMPARACIÓN FINAL")
    
    best_cnn = max(results, key=lambda x: x['test_accuracy'])
    print(f"Mejor CNN: {best_cnn['experiment']}")
    print(f"CNN Accuracy: {best_cnn['test_accuracy']:.3f}")
    print(f"CNN Parámetros: {best_cnn['num_parameters']:,}")
    print(f"MLP Accuracy: {baseline_acc:.3f}")
    print(f"MLP Parámetros: {count_params(trained_baseline):,}")
    
    if best_cnn['test_accuracy'] > baseline_acc:
        print("CNN supera a MLP")
    else:
        print("MLP supera a CNN")
    
    print("\nMATRICES DE CONFUSIÓN")
    
    best_cnn_model, best_cnn_cfg, cnn_test_loader = models[best_cnn['experiment']]
    cnn_preds, cnn_true = get_predictions(best_cnn_model, cnn_test_loader, best_cnn_cfg.device)
    plot_confusion_matrix(cnn_true, cnn_preds, "CNN")
    
    mlp_preds, mlp_true = get_predictions(trained_baseline, baseline_test_loader, baseline_cfg.device)
    plot_confusion_matrix(mlp_true, mlp_preds, "MLP")

    cm_cnn = confusion_matrix(cnn_true, cnn_preds, labels=list(range(10)))
    cm_mlp = confusion_matrix(mlp_true, mlp_preds, labels=list(range(10)))

    confusion_matrix_fig(cm_cnn, title="Matriz de Confusión - CNN", labels=list(range(10)), normalize=False, fname="cm_cnn.png")
    confusion_matrix_fig(cm_cnn, title="Matriz de Confusión (Norm) - CNN", labels=list(range(10)), normalize=True, fname="cm_cnn_norm.png")
    confusion_matrix_fig(cm_mlp, title="Matriz de Confusión - MLP", labels=list(range(10)), normalize=False, fname="cm_mlp.png")
    confusion_matrix_fig(cm_mlp, title="Matriz de Confusión (Norm) - MLP", labels=list(range(10)), normalize=True, fname="cm_mlp_norm.png")

    mlp_params = count_params(trained_baseline)
    _ = benchmark_report(best_cnn, baseline_acc, mlp_params, cm_cnn, cm_mlp)

    print("Laboratorio completado")

if __name__ == "__main__":
    main()