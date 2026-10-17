"""Deney 1: MNIST üzerinde temel MLP modelinin sıfırdan eğitilmesi ve değerlendirilmesi."""
import csv
import json
import platform
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST

from model import MLP

FOLDER = Path(__file__).resolve().parent
DATA_DIR = FOLDER / "veri"
OUT_DIR = FOLDER / "sonuclar" / "deney1"
MODEL_PATH = FOLDER / "modeller" / "mlp_deney1.pt"

HPARAMS = {
    "hidden_sizes": [256, 128],
    "activation": "ReLU",
    "loss": "CrossEntropyLoss",
    "optimizer": "Adam",
    "learning_rate": 0.001,
    "batch_size": 128,
    "epochs": 20,
    "validation_size": 10_000,
    "seed": 42,
}


def load_mnist(train):
    """MNIST'i indirir; görüntüleri (N, 1, 28, 28) boyutlu, [0, 1] aralığında float olarak döndürür."""
    dataset = MNIST(DATA_DIR, train=train, download=True)
    return dataset.data.float().div(255).unsqueeze(1), dataset.targets


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """Optimizer verilirse bir eğitim epoch'u, verilmezse değerlendirme yapar. (loss, accuracy) döndürür."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(dim=1) == y).sum().item()
            count += len(y)
    return total_loss / count, correct / count


def class_counts(labels):
    return torch.bincount(labels, minlength=10).tolist()


def load_train_val(seed, validation_size):
    """MNIST'in 60 000 görüntülük eğitim kısmını train ve validation olarak ikiye böler.

    Ortalama ve standart sapma yalnızca train kısmından hesaplanır. Resmî test kümesi burada yüklenmez.
    """
    images, labels = load_mnist(train=True)
    perm = torch.randperm(len(labels), generator=torch.Generator().manual_seed(seed))
    val_idx, train_idx = perm[:validation_size], perm[validation_size:]
    assert len(set(train_idx.tolist()) & set(val_idx.tolist())) == 0
    mean = images[train_idx].mean().item()
    std = images[train_idx].std().item()
    train_ds = TensorDataset((images[train_idx] - mean) / std, labels[train_idx])
    val_ds = TensorDataset((images[val_idx] - mean) / std, labels[val_idx])
    return train_ds, val_ds, mean, std


def fit(model, train_ds, val_ds, optimizer, hp, device, log=True):
    """Modeli eğitir ve her epoch sonunda validation kümesini ölçer.

    Epoch geçmişini ve validation loss'un en düşük olduğu epoch'taki ağırlıkları döndürür.
    """
    train_loader = DataLoader(train_ds, batch_size=hp["batch_size"], shuffle=True,
                              generator=torch.Generator().manual_seed(hp["seed"]))
    val_loader = DataLoader(val_ds, batch_size=1000)
    loss_fn = nn.CrossEntropyLoss()
    history, best_state = [], None
    for epoch in range(1, hp["epochs"] + 1):
        epoch_start = time.perf_counter()
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, loss_fn, device)
        epoch_time = time.perf_counter() - epoch_start
        if best_state is None or val_loss < min(r["val_loss"] for r in history):
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc, "epoch_time_s": epoch_time})
        if log:
            print(f"Epoch {epoch:2d}/{hp['epochs']} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.4f} | {epoch_time:.2f} s")
    return history, best_state


def plot_history(history, path):
    epochs = [row["epoch"] for row in history]
    colors = {"train": "#2a78d6", "val": "#eb6834"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), layout="constrained")
    panels = [("loss", "Loss (cross-entropy)", 1), ("acc", "Accuracy (%)", 100)]
    for ax, (key, ylabel, scale) in zip(axes, panels):
        ax.plot(epochs, [r[f"train_{key}"] * scale for r in history], "o-", color=colors["train"], lw=2, ms=4, label="Training")
        ax.plot(epochs, [r[f"val_{key}"] * scale for r in history], "o-", color=colors["val"], lw=2, ms=4, label="Validation")
        ax.set(xlabel="Epoch", ylabel=ylabel, xticks=range(0, len(epochs) + 1, 2))
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    axes[0].set_title("Training ve validation loss")
    axes[1].set_title("Training ve validation accuracy")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    seed = HPARAMS["seed"]
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Veri ayrımı ve normalizasyon. Resmî 10 000 görüntülük test kümesi eğitim boyunca hiç yüklenmez.
    train_ds, val_ds, mean, std = load_train_val(seed, HPARAMS["validation_size"])
    print(f"Cihaz: {device}  |  train: {len(train_ds)}  validation: {len(val_ds)}  (test eğitimden sonra)")
    print(f"Normalizasyon: mean={mean:.4f}, std={std:.4f}")

    # 2) Model ve optimizer. Kayıp fonksiyonu CrossEntropyLoss (fit içinde).
    model = MLP(HPARAMS["hidden_sizes"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=HPARAMS["learning_rate"])
    n_params = sum(p.numel() for p in model.parameters())
    print(model)
    print(f"Eğitilebilir parametre: {n_params:,}")

    # 3) Eğitim: her epoch sonunda validation kümesi ölçülür. Deney 1 son epoch'taki modeli kullanır.
    start = time.perf_counter()
    history, _ = fit(model, train_ds, val_ds, optimizer, HPARAMS, device)
    training_time = time.perf_counter() - start

    # 4) Test: eğitim bittikten sonra test kümesi ilk ve tek kez kullanılır.
    test_images, test_labels = load_mnist(train=False)
    test_loader = DataLoader(TensorDataset((test_images - mean) / std, test_labels), batch_size=1000)
    test_loss, test_acc = run_epoch(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"Test loss {test_loss:.4f}  |  test accuracy {test_acc:.4f}  |  eğitim süresi {training_time:.1f} s")

    torch.save({"state_dict": model.state_dict(), "hidden_sizes": HPARAMS["hidden_sizes"],
                "mean": mean, "std": std}, MODEL_PATH)

    with open(OUT_DIR / "egitim_gecmisi.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)

    final = history[-1]
    results = {
        "hyperparameters": HPARAMS,
        "trainable_parameters": n_params,
        "normalization": {"mean": mean, "std": std},
        "split": {
            "train": {"size": len(train_ds), "class_counts": class_counts(train_ds.tensors[1])},
            "validation": {"size": len(val_ds), "class_counts": class_counts(val_ds.tensors[1])},
            "test": {"size": len(test_labels), "class_counts": class_counts(test_labels)},
        },
        "final_epoch": {k: final[k] for k in ("train_loss", "val_loss", "train_acc", "val_acc")},
        "test": {"loss": test_loss, "accuracy": test_acc},
        "training_time_s": training_time,
        "environment": {
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or "CPU",
            "python": platform.python_version(),
            "torch": torch.__version__,
            "os": platform.platform(),
        },
    }
    (OUT_DIR / "metrikler.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_history(history, OUT_DIR / "loss_accuracy.png")
    print(f"Çıktılar: {OUT_DIR.relative_to(FOLDER)}  |  model: {MODEL_PATH.relative_to(FOLDER)}")


if __name__ == "__main__":
    main()
