
---

## 📄 WEEK2-CNN-LAYERS/README.md

```markdown
# Week 2 - Convolutional Neural Networks from Scratch

Implémentation from scratch des couches convolutives, pooling et entraînement sur MNIST.

##  Objectifs Réalisés

- ✅ Implémentation des couches `Conv2D`, `MaxPool2D`, `Flatten`
- ✅ Forward/backward propagation pour les convolutions
- ✅ Entraînement CNN sur MNIST avec **98.2% d'accuracy**
- ✅ Visualisation des filtres et feature maps

##  Architecture CNN

```python
Sequential([
    Conv2D(1, 8, 3, padding=1), ReLU(),
    MaxPool2D(2),
    Conv2D(8, 16, 3, padding=1), ReLU(), 
    MaxPool2D(2),
    Flatten(),
    Dense(16*7*7, 128), ReLU(),
    Dense(128, 10), Softmax()
])