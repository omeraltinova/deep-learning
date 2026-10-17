"""el_yazisi_hazirla.py ile hazırlanan 28 × 28 rakamları Deney 1 modeliyle sınıflandırır."""
import argparse
import csv
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import MLP

FOLDER = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--girdi", type=Path, default=FOLDER / "el_yazisi" / "islenmis")
    parser.add_argument("--model", type=Path, default=FOLDER / "modeller" / "mlp_deney1.pt")
    parser.add_argument("--cikti", type=Path, default=FOLDER / "sonuclar" / "el_yazisi")
    args = parser.parse_args()
    args.cikti.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=True)
    model = MLP(checkpoint["hidden_sizes"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with open(args.girdi / "etiketler.csv", encoding="utf-8") as f:
        records = list(csv.DictReader(f))
    if not records:
        raise SystemExit("Hazırlanmış rakam yok. Önce el_yazisi_hazirla.py çalıştırılmalı.")
    images = np.stack([cv2.imdecode(np.fromfile(args.girdi / r["dosya"], np.uint8), cv2.IMREAD_GRAYSCALE)
                       for r in records])
    assert images.shape[1:] == (28, 28)

    # Normalizasyon: eğitimdeki gibi [0, 1] aralığına ölçeklenir, train kümesinin mean/std değeri kullanılır.
    x = torch.from_numpy(images).float().div(255).unsqueeze(1)
    x = (x - checkpoint["mean"]) / checkpoint["std"]
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
    confidence, predicted = probs.max(dim=1)

    for record, pred, conf in zip(records, predicted.tolist(), confidence.tolist()):
        record["tahmin"] = str(pred)
        record["olasilik"] = round(conf, 4)
        record["dogru"] = record["etiket"] == record["tahmin"] if record["etiket"] else ""
    with open(args.cikti / "tahminler.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    lines = {}
    for record in records:
        lines.setdefault((record["foto"], record["satir"]), []).append(record)
    labeled = [r for r in records if r["etiket"]]
    correct = sum(r["dogru"] for r in labeled)
    summary = {
        "rakam_sayisi": len(records),
        "etiketli_rakam": len(labeled),
        "dogru": correct,
        "accuracy": correct / len(labeled) if labeled else None,
        "ortalama_olasilik": round(confidence.mean().item(), 4),
        "satirlar": [{"foto": foto, "satir": satir,
                      "gercek": "".join(r["etiket"] for r in rows),
                      "tahmin": "".join(r["tahmin"] for r in rows)} for (foto, satir), rows in lines.items()],
    }
    (args.cikti / "ozet.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    for line in summary["satirlar"]:
        print(f"{line['foto']} satır {line['satir']}:  gerçek {line['gercek'] or '?'}  |  tahmin {line['tahmin']}")
    if labeled:
        print(f"Doğru: {correct}/{len(labeled)}  |  accuracy {summary['accuracy']:.4f}")

    columns = max(len(rows) for rows in lines.values())
    fig, axes = plt.subplots(len(lines), columns, figsize=(1.25 * columns, 1.75 * len(lines)),
                             squeeze=False, layout="constrained")
    index = {r["dosya"]: i for i, r in enumerate(records)}
    for row_axes, rows in zip(axes, lines.values()):
        for ax in row_axes:
            ax.axis("off")
        for ax, record in zip(row_axes, rows):
            ax.imshow(images[index[record["dosya"]]], cmap="gray", vmin=0, vmax=255)
            wrong = record["dogru"] is False
            title = f"tahmin {record['tahmin']}\n%{100 * record['olasilik']:.0f}"
            if record["etiket"]:
                title = f"gerçek {record['etiket']}\n" + title
            ax.set_title(title + ("  ✗" if wrong else ""), fontsize=8, color="#c0392b" if wrong else "#0b0b0b")
    fig.suptitle("Kendi el yazısı rakamlarda Deney 1 modelinin tahminleri", fontsize=10)
    fig.savefig(args.cikti / "tahminler.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
