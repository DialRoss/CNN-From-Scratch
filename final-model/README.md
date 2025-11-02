# 🧠 CNN From Scratch - Framework Deep Learning from Scratch

Un framework de Deep Learning implémenté from scratch en NumPy, avec comparaison détaillée des optimiseurs, architectures et techniques d'entraînement.

## 📊 Résultats Principaux

| Architecture | MNIST Accuracy | Paramètres | Temps d'entraînement |
|--------------|----------------|------------|---------------------|
| MLP Baseline | 94.4%          | ~500K      | 2 min               |
| CNN Simple   | 98.2%          | ~150K      | 5 min               |
| CNN Avancé   | **99.1%**      | ~200K      | 8 min               |

## 🗂️ Structure du Projet

- **[week1-dense-networks/](week1-dense-networks/)** - Réseaux fully-connected from scratch
- **[week2-cnn-layers/](week2-cnn-layers/)** - Implémentation des couches convolutives  
- **[week3-advanced-training/](week3-advanced-training/)** - Techniques avancées d'entraînement
- **[final-model/](final-model/)** - Version finale et optimisée

## 🚀 Utilisation Rapide

```python
from final-model.src.model import Sequential
from final-model.src.layers import Conv2D, Dense, ReLU, Softmax
from final-model.src.optimizers import Adam

# Charger le modèle pré-entraîné
model = Sequential([...])  # Architecture finale
model.load_weights('final-model/best_weights.npy')