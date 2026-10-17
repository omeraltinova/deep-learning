"""Kendi el yazısı fotoğraflarını MNIST biçiminde 28 × 28 rakam görüntülerine dönüştürür.

Adımlar: gri tonlama → kâğıttaki ışık farkının giderilmesi → renklerin ters çevrilmesi
(siyah arka plan, beyaz rakam) → eşikleme → rakamların ayrılması → kalem kalınlığının MNIST'e
yaklaştırılması → 20 × 20 kutuya sığdırma → 28 × 28 alanda ağırlık merkezine göre ortalama.
Normalizasyon, eğitimdeki ortalama ve standart sapmayla el_yazisi_tahmin.py içinde yapılır.

Her fotoğrafta öğrenci numarası bir satır olarak yazılmalıdır. Birden fazla satır varsa
her satır ayrı bir numara örneği sayılır.
"""
import argparse
import csv
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FOLDER = Path(__file__).resolve().parent
RAW_DIR = FOLDER / "el_yazisi" / "ham"
OUT_DIR = FOLDER / "el_yazisi" / "islenmis"
FIG_DIR = FOLDER / "sonuclar" / "el_yazisi"
STUDENT_NO = "23120205043"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# MNIST eğitim görüntülerinden ölçüldü: kalem kalınlığı, rakamın uzun kenarının yaklaşık 0.117'si;
# ağırlık merkezi ortalaması (14, 14) pikseli.
MNIST_STROKE_RATIO = 0.117
MNIST_CENTER = 14.0


def read_gray(path, max_side=1600):
    """1) Gri tonlama. cv2.imread Türkçe karakterli yolları okuyamadığı için dosya bayt olarak okunur."""
    gray = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise SystemExit(f"Görüntü okunamadı: {path}")
    scale = max_side / max(gray.shape)
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return gray


def ink_image(gray):
    """2) Kâğıdı siyaha, mürekkebi beyaza çevirir; MNIST'teki renk düzeni budur."""
    size = max(gray.shape) // 15 | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)  # mürekkep silinir, kâğıdın ışığı kalır
    flat = cv2.divide(gray, background, scale=255)  # gölge ve ışık farkı giderilir; kâğıt ≈ 255
    return 255 - flat


def ink_mask(ink):
    """3) Otsu eşiği ile mürekkep piksellerini seçer. Çok düşük eşikler kâğıt dokusunu alacağı için alt sınır 40."""
    blurred = cv2.GaussianBlur(ink, (5, 5), 0)
    otsu, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(blurred, max(otsu, 40), 255, cv2.THRESH_BINARY)
    return mask


def overlap(a1, a2, b1, b2):
    return min(a2, b2) - max(a1, b1)


def find_digits(mask):
    """4) Rakamları ayırır. Satır satır ve soldan sağa sıralı [x1, y1, x2, y2, bileşen_listesi] döndürür."""
    h, w = mask.shape
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    parts = []
    for i, (x, y, bw, bh, area) in enumerate(stats[1:], start=1):
        touches_border = x == 0 or y == 0 or x + bw == w or y + bh == h
        if area >= 0.00005 * h * w and not touches_border and area < 0.2 * h * w:
            parts.append([x, y, x + bw, y + bh, [i]])

    rows = []  # dikey aralıkları çakışan parçalar aynı satırdadır
    for part in sorted(parts, key=lambda p: p[1]):
        for row in rows:
            if overlap(part[1], part[3], row["y1"], row["y2"]) > 0.5 * min(part[3] - part[1], row["y2"] - row["y1"]):
                row["parts"].append(part)
                row["y1"], row["y2"] = min(row["y1"], part[1]), max(row["y2"], part[3])
                break
        else:
            rows.append({"y1": part[1], "y2": part[3], "parts": [part]})

    result = []
    for row in sorted(rows, key=lambda r: r["y1"]):
        digits = []  # yatayda büyük ölçüde çakışan parçalar aynı rakamdır (ör. 5'in üst çizgisi)
        for part in sorted(row["parts"], key=lambda p: p[0]):
            last = digits[-1] if digits else None
            if last and overlap(part[0], part[2], last[0], last[2]) > 0.5 * min(part[2] - part[0], last[2] - last[0]):
                last[:4] = [min(last[0], part[0]), min(last[1], part[1]), max(last[2], part[2]), max(last[3], part[3])]
                last[4] += part[4]
            else:
                digits.append(list(part[:4]) + [list(part[4])])
        heights = np.median([d[3] - d[1] for d in digits])
        digits = [d for d in digits if d[3] - d[1] >= 0.4 * heights]  # nokta ve leke gibi küçük parçalar
        if digits:
            result.append(digits)
    return result, labels


def to_mnist(labels, digit):
    """5) Tek rakamı MNIST biçimine getirir: kalınlık, 20 × 20 kutu, 28 × 28 alan, ağırlık merkezi."""
    x1, y1, x2, y2, ids = digit
    mask = np.isin(labels[y1:y2, x1:x2], ids).astype(np.uint8) * 255

    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    stroke = 2 * np.count_nonzero(mask) / perimeter  # alan / (çevre / 2) ≈ çizgi kalınlığı
    grow = int(round(MNIST_STROKE_RATIO * max(mask.shape) - stroke))
    if grow >= 1:  # kalem MNIST'e göre inceyse çizgi kalınlaştırılır; kalın kalem inceltilmez
        mask = cv2.copyMakeBorder(mask, grow, grow, grow, grow, cv2.BORDER_CONSTANT, value=0)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow + 1, grow + 1)))
        ys, xs = np.nonzero(mask)
        mask = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]

    # MNIST'teki gibi en-boy oranı korunarak 20 × 20 kutuya sığdırılır; INTER_AREA kenarlarda gri tonlar üretir.
    h, w = mask.shape
    scale = 20 / max(h, w)
    small = cv2.resize(mask.astype(np.float32), (max(1, round(w * scale)), max(1, round(h * scale))),
                       interpolation=cv2.INTER_AREA)
    canvas = np.zeros((28, 28), np.float32)
    top, left = (28 - small.shape[0]) // 2, (28 - small.shape[1]) // 2
    canvas[top:top + small.shape[0], left:left + small.shape[1]] = small

    m = cv2.moments(canvas)
    dx, dy = round(MNIST_CENTER - m["m10"] / m["m00"]), round(MNIST_CENTER - m["m01"] / m["m00"])
    canvas = cv2.warpAffine(canvas, np.float32([[1, 0, dx], [0, 1, dy]]), (28, 28))
    return np.clip(np.rint(canvas), 0, 255).astype(np.uint8)


def save_png(path, image):
    cv2.imencode(".png", image)[1].tofile(path)


def plot_steps(gray, ink, rows, digit_images, row_labels, path):
    fig = plt.figure(figsize=(12, 4 + 1.3 * len(rows)), layout="constrained")
    grid = fig.add_gridspec(1 + len(rows), 2, height_ratios=[4] + [1.3] * len(rows))
    ax = fig.add_subplot(grid[0, 0])
    ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
    ax.set_title("1) Gri tonlamalı fotoğraf")
    ax.axis("off")
    ax = fig.add_subplot(grid[0, 1])
    ax.imshow(ink, cmap="gray", vmin=0, vmax=255)
    for r, row in enumerate(rows):
        for j, (x1, y1, x2, y2, _) in enumerate(row):
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="#eb6834", lw=1.5))
            ax.text(x1, y1 - 6, str(j + 1), color="#eb6834", fontsize=8)
    ax.set_title("2-4) Ters çevrilmiş renkler ve bulunan rakamlar")
    ax.axis("off")
    for r, (images, label) in enumerate(zip(digit_images, row_labels)):
        ax = fig.add_subplot(grid[1 + r, :])
        gap = np.full((28, 2), 128, np.uint8)
        strip = np.hstack([part for image in images for part in (image, gap)][:-1])
        ax.imshow(strip, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        ax.set_xticks([15 + 30 * j for j in range(len(images))], list(label) if label else [""] * len(images))
        ax.set_yticks([])
        ax.set_title(f"5) Satır {r + 1}: 28 × 28 MNIST biçimi" + ("" if label else " (etiket eşleşmedi)"), fontsize=10)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("girdiler", nargs="*", type=Path, help="fotoğraflar (varsayılan: el_yazisi/ham içindekiler)")
    parser.add_argument("--etiket", default=STUDENT_NO, help="her satırda yazan sayı (varsayılan: öğrenci numarası)")
    parser.add_argument("--cikti", type=Path, default=OUT_DIR, help="28 × 28 PNG'lerin yazılacağı klasör")
    parser.add_argument("--sekil", type=Path, default=FIG_DIR, help="ara adım görsellerinin klasörü")
    args = parser.parse_args()

    photos = args.girdiler or sorted(p for p in RAW_DIR.glob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not photos:
        raise SystemExit(f"Fotoğraf bulunamadı. Öğrenci numaranın fotoğrafını {RAW_DIR} içine koy.")
    args.cikti.mkdir(parents=True, exist_ok=True)
    args.sekil.mkdir(parents=True, exist_ok=True)
    for old in args.cikti.glob("*.png"):
        old.unlink()

    records = []
    for photo in photos:
        gray = read_gray(photo)
        ink = ink_image(gray)
        rows, labels = find_digits(ink_mask(ink))
        if not rows:
            print(f"UYARI: {photo.name} içinde rakam bulunamadı.")
        digit_images, row_labels = [], []
        for r, row in enumerate(rows, start=1):
            images = [to_mnist(labels, digit) for digit in row]
            label = args.etiket if len(images) == len(args.etiket) else ""
            if not label:
                print(f"UYARI: {photo.name} satır {r}: {len(images)} rakam bulundu, '{args.etiket}' "
                      f"{len(args.etiket)} hane. Bu satır etiketsiz kaydedildi.")
            for j, image in enumerate(images, start=1):
                name = f"{photo.stem}_s{r}_{j:02d}.png"
                save_png(args.cikti / name, image)
                records.append({"dosya": name, "foto": photo.name, "satir": r, "sira": j,
                                "etiket": label[j - 1] if label else ""})
            digit_images.append(images)
            row_labels.append(label)
        plot_steps(gray, ink, rows, digit_images, row_labels, args.sekil / f"on_isleme_{photo.stem}.png")
        print(f"{photo.name}: {len(rows)} satır, {sum(map(len, rows))} rakam")

    with open(args.cikti / "etiketler.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["dosya", "foto", "satir", "sira", "etiket"])
        writer.writeheader()
        writer.writerows(records)
    print(f"{len(records)} rakam {args.cikti} klasörüne kaydedildi.")


if __name__ == "__main__":
    main()
