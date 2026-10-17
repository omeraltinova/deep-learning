# Derin öğrenme ödevleri

Derin öğrenme dersinin ödevleri bu depoda duruyor. Her ödevin kodu, sonuçları ve raporu kendi klasöründe. Bütün ödevler kökteki ortak `.venv` Python ortamını kullanır.

| Ödev | Konu | Klasör |
| --- | --- | --- |
| Ödev 1 | MNIST veri setiyle MLP tabanlı el yazısı rakam tanıma | [odev1_mnist_mlp](odev1_mnist_mlp/README.md) |

## Python ortamı

Ortam Python 3.12 ile kuruldu. Paket sürümleri `requirements.txt` içinde sabitlendi. `.venv` klasörü Git'e girmez, her bilgisayarda yeniden kurulur.

Windows'ta, PowerShell ile proje kökünde:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\Activate.ps1
```

macOS veya Linux'ta:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows'ta PyPI'dan gelen PyTorch yalnızca CPU kullanır. Ödevler CPU'da da çalışır. NVIDIA ekran kartı kullanmak istersen `requirements.txt` kurulumundan önce PyTorch'un CUDA derlemesini kur:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
```

Ortamı etkinleştirdikten sonra `python` ve `pip` komutları `.venv` içinden çalışır. Etkinleştirmeden `.\.venv\Scripts\python.exe` yolu da doğrudan kullanılabilir. Her ödevin çalıştırma komutları kendi README dosyasında.
