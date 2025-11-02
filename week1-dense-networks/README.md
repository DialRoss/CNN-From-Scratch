
---

## 📄 WEEK1-DENSE-NETWORKS/README.md

```markdown
# Week 1 - Fully Connected Networks from Scratch

Implémentation from scratch des couches dense, fonctions d'activation et algorithmes d'optimisation.

##  Objectifs Réalisés

- ✅ Implémentation des couches `Dense`, `ReLU`, `Softmax`
- ✅ Algorithmes d'optimisation `SGD` et `Adam` 
- ✅ Backpropagation et calcul des gradients
- ✅ Entraînement sur MNIST avec **94.4% d'accuracy**

##  Architecture du Modèle

```python
Sequential([
    Flatten(),
    Dense(784, 512), ReLU(),
    Dense(512, 256), ReLU(), 
    Dense(256, 10), Softmax()
])