# Derin öğrenme ödevleri

Derin öğrenme dersinin ödevleri bu repoda. Her ödevin kodu, sonuçları ve raporu kendi klasöründe. Ödevlerin hepsi kökteki ortak `.venv` ortamını kullanıyor.

| Ödev | Konu | Klasör |
| --- | --- | --- |
| Ödev 1 | MNIST ile MLP tabanlı el yazısı rakam tanıma | [odev1_mnist_mlp](odev1_mnist_mlp/README.md) |

## Python ortamı

Ortam Python 3.12 ile kuruldu. Paket sürümleri `requirements.txt` içinde sabit. `.venv` Git'e girmiyor, her bilgisayarda yeniden kurulması gerekiyor.

Windows'ta, proje kökünde PowerShell ile:

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

Windows'ta PyPI'dan gelen PyTorch yalnızca CPU'yu kullanıyor. Ödevler CPU'da da çalışıyor. NVIDIA ekran kartı kullanmak için `requirements.txt` kurulumundan önce PyTorch'un CUDA sürümünün kurulması gerekiyor:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
```

Ortam etkinken `python` ve `pip` komutları `.venv` içinden çalışır. Etkinleştirmeden `.\.venv\Scripts\python.exe` yolu da kullanılabilir. Çalıştırma komutları her ödevin kendi README dosyasında.
