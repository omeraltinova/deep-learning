from torch import nn


class MLP(nn.Module):
    """28 × 28 görüntüyü 784 boyutlu vektöre açan, ReLU aktivasyonlu tam bağlantılı ağ.

    Evrişim (CNN) katmanı yoktur. Çıkış katmanı 10 sınıf için ham skor (logit) üretir;
    softmax, CrossEntropyLoss içinde uygulanır.
    """

    def __init__(self, hidden_sizes=(256, 128), num_classes=10):
        super().__init__()
        layers = [nn.Flatten()]
        in_features = 28 * 28
        for size in hidden_sizes:
            layers += [nn.Linear(in_features, size), nn.ReLU()]
            in_features = size
        layers.append(nn.Linear(in_features, num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)
