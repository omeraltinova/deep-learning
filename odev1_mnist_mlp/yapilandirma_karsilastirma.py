"""Model yapılandırmalarını validation kümesinde karşılaştırır ve en iyisini seçer.

Her yapılandırma, Deney 1'in ayarlarından yalnızca birini değiştirir ve Deney 1'in kurallarına uyar:
en az 2 gizli katman, ReLU, CrossEntropyLoss, Adam veya SGD. Her yapılandırma 20 epoch eğitilir ve
validation loss'un en düşük olduğu epoch alınır. Seçim yalnızca validation loss'a göre yapılır.
Test kümesi yalnızca seçilen model için, bir kez kullanılır.
"""
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from deney1_temel_mlp import HPARAMS, fit, load_mnist, load_train_val, run_epoch
from model import MLP

FOLDER = Path(__file__).resolve().parent
OUT_DIR = FOLDER / "sonuclar" / "karsilastirma"
MODEL_PATH = FOLDER / "modeller" / "mlp_secilen.pt"

BASELINE = "Deney 1 (256-128, Adam 0.001, batch 128)"
CONFIGS = [  # (grup, ad, Deney 1'e göre değişen ayar)
    ("Temel", BASELINE, {}),
    ("Gizli katmanlar", "128-64", {"hidden_sizes": [128, 64]}),
    ("Gizli katmanlar", "512-256", {"hidden_sizes": [512, 256]}),
    ("Gizli katmanlar", "256-128-64", {"hidden_sizes": [256, 128, 64]}),
    ("Gizli katmanlar", "512-256-128", {"hidden_sizes": [512, 256, 128]}),
    ("Learning rate", "Adam lr 0.0001", {"learning_rate": 0.0001}),
    ("Learning rate", "Adam lr 0.01", {"learning_rate": 0.01}),
    ("Optimizer", "SGD lr 0.1", {"optimizer": "SGD", "learning_rate": 0.1}),
    ("Optimizer", "SGD lr 0.01, momentum 0.9", {"optimizer": "SGD", "learning_rate": 0.01, "momentum": 0.9}),
    ("Batch size", "Batch 32", {"batch_size": 32}),
    ("Batch size", "Batch 512", {"batch_size": 512}),
]
# Aynı temel ayar farklı seed'lerle de eğitilir; validation sonucunun seed'e bağlı oynaması böyle ölçülür.
EXTRA_SEEDS = [43, 44]
GROUPS = ["Gizli katmanlar", "Learning rate", "Optimizer", "Batch size"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]


def make_optimizer(model, hp):
    if hp["optimizer"] == "Adam":
        return torch.optim.Adam(model.parameters(), lr=hp["learning_rate"])
    return torch.optim.SGD(model.parameters(), lr=hp["learning_rate"], momentum=hp.get("momentum", 0.0))


def train_config(change, seed, train_ds, val_ds, device):
    hp = {**HPARAMS, **change, "seed": seed}
    torch.manual_seed(seed)
    model = MLP(hp["hidden_sizes"]).to(device)
    start = time.perf_counter()
    history, best_state = fit(model, train_ds, val_ds, make_optimizer(model, hp), hp, device, log=False)
    elapsed = time.perf_counter() - start
    return hp, history, best_state, elapsed


def summarize(group, name, seed, history, elapsed):
    best = min(history, key=lambda r: r["val_loss"])
    return {"grup": group, "yapilandirma": name, "seed": seed, "en_iyi_epoch": best["epoch"],
            "val_loss": best["val_loss"], "val_acc": best["val_acc"], "train_acc": best["train_acc"],
            "son_epoch_val_loss": history[-1]["val_loss"], "son_epoch_val_acc": history[-1]["val_acc"],
            "egitim_suresi_s": elapsed}


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(histories, path):
    fig, axes = plt.subplots(2, len(GROUPS), figsize=(16, 7), sharex=True, layout="constrained")
    for col, group in enumerate(GROUPS):
        names = [BASELINE] + [name for g, name, _ in CONFIGS if g == group]
        for color, name in zip(COLORS, names):
            epochs = [r["epoch"] for r in histories[name]]
            label = "Deney 1" if name == BASELINE else name
            axes[0, col].plot(epochs, [r["val_loss"] for r in histories[name]], color=color, lw=2, label=label)
            axes[1, col].plot(epochs, [100 * r["val_acc"] for r in histories[name]], color=color, lw=2, label=label)
        axes[0, col].set_title(group)
        axes[0, col].set_ylim(0.05, 0.25)
        axes[1, col].set_ylim(95, 98.8)
        axes[1, col].set_xlabel("Epoch")
        axes[0, col].legend(frameon=False, fontsize=8)
        for ax in axes[:, col]:
            ax.grid(alpha=0.25)
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Validation loss")
    axes[1, 0].set_ylabel("Validation accuracy (%)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Deney 1 ile aynı train/validation ayrımı ve aynı normalizasyon.
    train_ds, val_ds, mean, std = load_train_val(HPARAMS["seed"], HPARAMS["validation_size"])

    rows, histories, models = [], {}, {}
    for group, name, change in CONFIGS:
        hp, history, best_state, elapsed = train_config(change, HPARAMS["seed"], train_ds, val_ds, device)
        histories[name], models[name] = history, (hp, best_state)
        rows.append(summarize(group, name, HPARAMS["seed"], history, elapsed))
        r = rows[-1]
        print(f"{name:42s} en iyi epoch {r['en_iyi_epoch']:2d} | val loss {r['val_loss']:.4f} "
              f"acc {r['val_acc']:.4f} | {elapsed:.1f} s")

    seed_rows = [rows[0]]
    for seed in EXTRA_SEEDS:
        _, history, _, elapsed = train_config({}, seed, train_ds, val_ds, device)
        seed_rows.append(summarize("Temel", BASELINE, seed, history, elapsed))
        r = seed_rows[-1]
        print(f"{'Deney 1, seed ' + str(seed):42s} en iyi epoch {r['en_iyi_epoch']:2d} | val loss {r['val_loss']:.4f} "
              f"acc {r['val_acc']:.4f}")

    # Seçim: en düşük validation loss. Seçilen epoch'taki ağırlıklar fit() içinde saklandı.
    chosen = min(rows, key=lambda r: r["val_loss"])
    hp, best_state = models[chosen["yapilandirma"]]
    model = MLP(hp["hidden_sizes"]).to(device)
    model.load_state_dict(best_state)

    # Test: yalnızca seçilen model için, bir kez.
    test_images, test_labels = load_mnist(train=False)
    test_loader = DataLoader(TensorDataset((test_images - mean) / std, test_labels), batch_size=1000)
    test_loss, test_acc = run_epoch(model, test_loader, nn.CrossEntropyLoss(), device)
    print(f"Seçilen: {chosen['yapilandirma']}, epoch {chosen['en_iyi_epoch']} | "
          f"test loss {test_loss:.4f} | test accuracy {test_acc:.4f}")

    torch.save({"state_dict": model.state_dict(), "hidden_sizes": hp["hidden_sizes"],
                "mean": mean, "std": std}, MODEL_PATH)
    write_csv(OUT_DIR / "ozet.csv", rows)
    write_csv(OUT_DIR / "seed_kontrolu.csv", seed_rows)
    write_csv(OUT_DIR / "epoch_gecmisi.csv",
              [{"yapilandirma": name, **r} for name, history in histories.items() for r in history])
    selected = {"yapilandirma": chosen["yapilandirma"], "epoch": chosen["en_iyi_epoch"],
                "hyperparameters": {**hp, "epochs": chosen["en_iyi_epoch"]},
                "validation": {"loss": chosen["val_loss"], "accuracy": chosen["val_acc"]},
                "test": {"loss": test_loss, "accuracy": test_acc}}
    (OUT_DIR / "secilen_model.json").write_text(json.dumps(selected, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_curves(histories, OUT_DIR / "validation_egrileri.png")
    print(f"Çıktılar: {OUT_DIR.relative_to(FOLDER)}  |  model: {MODEL_PATH.relative_to(FOLDER)}")


if __name__ == "__main__":
    main()
