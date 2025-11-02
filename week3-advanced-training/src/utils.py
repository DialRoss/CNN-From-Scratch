import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import urllib.request
import gzip
import pickle
import os
from time import time
import pandas as pd
from callbacks import LearningRateScheduler

def load_mnist():
    url = 'https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz'
    filename = 'mnist.pkl.gz'
    if not os.path.exists(filename):
        print("Téléchargement des données MNIST...")
        urllib.request.urlretrieve(url, filename)
    
    with gzip.open(filename, 'rb') as f:
        train, val, test = pickle.load(f, encoding='latin1')
    
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test
    
    X_train = X_train.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    X_val   = X_val.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    X_test  = X_test.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
    
    y_train_oh = np.eye(10)[y_train.astype(int)]
    y_val_oh = np.eye(10)[y_val.astype(int)]
    y_test_oh = np.eye(10)[y_test.astype(int)]
    
    return (X_train, y_train, y_train_oh), (X_val, y_val, y_val_oh), (X_test, y_test, y_test_oh)

def augment_data(X, y, max_aug=1):
    X_aug = []
    y_aug = []
    
    for i in range(len(X)):
        image = X[i]
        label = y[i]
        
        X_aug.append(image)
        y_aug.append(label)
        
        aug_count = 0
        
        for shift in [-1, 1]:
            if aug_count >= max_aug:
                break
            shifted = np.roll(image, shift, axis=2)
            X_aug.append(shifted)
            y_aug.append(label)
            aug_count += 1
        
        for shift in [-1, 1]:
            if aug_count >= max_aug:
                break
            shifted = np.roll(image, shift, axis=1)
            X_aug.append(shifted)
            y_aug.append(label)
            aug_count += 1
    
    return np.array(X_aug), np.array(y_aug)

def train_model(model, X_train, y_train_oh, X_val, y_val, y_val_oh, 
                optimizer, scheduler, epochs=50, batch_size=32, callbacks=[], 
                save_feature_maps=False, save_filters=False):
    
    from losses import CrossEntropyLoss
    loss_fn = CrossEntropyLoss()
    n_batches = len(X_train) // batch_size
    history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
    
    for epoch in range(epochs):
        current_lr = scheduler.step(epoch) if scheduler else getattr(optimizer, 'lr', 0.001)
        if hasattr(optimizer, 'lr'):
            optimizer.lr = current_lr
        
        indices = np.random.permutation(len(X_train))
        X_train_shuf = X_train[indices]
        y_train_shuf = y_train_oh[indices]
        
        epoch_loss, epoch_acc = 0, 0
        
        for b in range(n_batches):
            model.zero_grad()
            
            start, end = b*batch_size, (b+1)*batch_size
            X_batch = X_train_shuf[start:end]
            y_batch = y_train_shuf[start:end]
            
            logits = model.forward(X_batch, training=True)
            loss = loss_fn.forward(logits, y_batch)
            
            grad = loss_fn.backward()
            model.backward(grad)
            
            params = model.get_params_and_grads()
            optimizer.step(params)
            
            preds = np.argmax(logits, axis=1)
            labels = np.argmax(y_batch, axis=1)
            acc = np.mean(preds == labels)
            
            epoch_loss += loss
            epoch_acc += acc
        
        val_logits = model.forward(X_val, training=False)
        val_loss = loss_fn.forward(val_logits, y_val_oh)
        val_preds = np.argmax(val_logits, axis=1)
        val_accuracy = np.mean(val_preds == y_val)
        
        epoch_loss /= n_batches
        epoch_acc /= n_batches
        
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_accuracy)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f} | LR: {current_lr:.6f}")
        
        stop = False
        for cb in callbacks:
            if hasattr(cb, 'on_epoch_end'):
                if cb.on_epoch_end(epoch, val_loss, model):
                    stop = True
        if stop:
            break
    
    feature_maps = save_all_feature_maps(model, X_val[:1]) if save_feature_maps else {}
    filters = save_all_filters(model) if save_filters else {}
    
    return history, feature_maps, filters

def evaluate_model(model, X_test, y_test, show_errors=True):
    predictions = model.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=range(10), yticklabels=range(10))
    plt.title('Matrice de Confusion')
    plt.ylabel('Vraie classe')
    plt.xlabel('Prédite')
    plt.show()
    
    if show_errors:
        analyze_errors(X_test, y_test, predictions)
    
    return accuracy, predictions

def analyze_errors(X_test, y_test, predictions, num_examples=5):
    errors = np.where(predictions != y_test)[0]
    print(f"\nAnalyse des erreurs: {len(errors)} erreurs sur {len(y_test)} échantillons")
    
    error_pairs = [(y_test[i], predictions[i]) for i in errors]
    error_counts = pd.DataFrame(error_pairs, columns=['Vraie', 'Prédite'])
    error_freq = error_counts.groupby(['Vraie', 'Prédite']).size().reset_index(name='Count')
    error_freq = error_freq.sort_values('Count', ascending=False).head(10)
    
    print("\nTop 10 des erreurs les plus fréquentes:")
    print(error_freq)
    
    plt.figure(figsize=(15, 5))
    for i, idx in enumerate(errors[:num_examples]):
        plt.subplot(1, num_examples, i+1)
        plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f"Vraie: {y_test[idx]}, Prédit: {predictions[idx]}")
        plt.axis('off')
    plt.suptitle("Exemples d'images mal classées")
    plt.tight_layout()
    plt.show()

def plot_history(history, title="Courbes d'apprentissage", save_path=None):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Courbes de perte')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Courbes de précision')
    plt.xlabel('Époque')
    plt.ylabel('Précision')
    plt.legend()
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def save_all_filters(model):
    filters = {}
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'W') and len(layer.W.shape) == 4:
            filters[f'conv_{i}'] = layer.W
    return filters

def visualize_filters(filters, save_path=None):
    for layer_name, weights in filters.items():
        print(f"Visualisation des filtres de {layer_name}")
        n_filters = weights.shape[0]
        
        plt.figure(figsize=(15, 8))
        for i in range(min(n_filters, 16)):
            plt.subplot(4, 4, i+1)
            plt.imshow(weights[i, 0], cmap='viridis')
            plt.title(f"Filtre {i}")
            plt.axis('off')
        
        plt.suptitle(f"Filtres de {layer_name} (premier canal)")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_{layer_name}.png")
        plt.show()

def save_all_feature_maps(model, X_sample):
    feature_maps = {}
    activations = X_sample.copy()
    
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'W') and len(layer.W.shape) == 4:
            activations = layer.forward(activations)
            feature_maps[f'conv_{i}'] = activations.copy()
        elif hasattr(layer, 'input') and not hasattr(layer, 'W'):
            activations = layer.forward(activations)
            feature_maps[f'relu_{i}'] = activations.copy()
        elif hasattr(layer, 'pool_size'):
            activations = layer.forward(activations)
            feature_maps[f'pool_{i}'] = activations.copy()
        elif hasattr(layer, 'input_shape'):
            break
    
    return feature_maps

def visualize_feature_maps(feature_maps, save_path=None):
    for layer_name, maps in feature_maps.items():
        print(f"Visualisation des feature maps de {layer_name}")
        n_maps = maps.shape[1]
        
        plt.figure(figsize=(15, 8))
        for i in range(min(n_maps, 16)):
            plt.subplot(4, 4, i+1)
            plt.imshow(maps[0, i], cmap='viridis')
            plt.title(f"Map {i}")
            plt.axis('off')
        
        plt.suptitle(f"Feature maps de {layer_name}")
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}_{layer_name}.png")
        plt.show()

def plot_comparison(histories, save_path=None):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    for name, history in histories.items():
        plt.plot(history['train_loss'], label=f'{name} Train')
    plt.title('Perte d\'entraînement')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    for name, history in histories.items():
        plt.plot(history['val_loss'], label=f'{name} Val')
    plt.title('Perte de validation')
    plt.xlabel('Époque')
    plt.ylabel('Perte')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    for name, history in histories.items():
        plt.plot(history['train_acc'], label=f'{name} Train')
    plt.title('Précision d\'entraînement')
    plt.xlabel('Époque')
    plt.ylabel('Précision')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    for name, history in histories.items():
        plt.plot(history['val_acc'], label=f'{name} Val')
    plt.title('Précision de validation')
    plt.xlabel('Époque')
    plt.ylabel('Précision')
    plt.legend()
    
    plt.suptitle("Comparaison des modèles")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def build_baseline_cnn():
    from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense, Softmax
    from model import Sequential
    
    return Sequential([
        Conv2D(1, 8, 3, padding=1), 
        ReLU(),
        MaxPool2D(2, 2),
        Conv2D(8, 16, 3, padding=1), 
        ReLU(),
        MaxPool2D(2, 2),
        Flatten(),
        Dense(16*7*7, 128),
        ReLU(),
        Dense(128, 10), 
        Softmax()
    ])

def build_improved_cnn():
    from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense, Dropout, Softmax
    from model import Sequential
    
    return Sequential([
        Conv2D(1, 32, 3, padding=1), 
        ReLU(),
        MaxPool2D(2, 2),
        Conv2D(32, 64, 3, padding=1), 
        ReLU(),
        MaxPool2D(2, 2),
        Conv2D(64, 128, 3, padding=1), 
        ReLU(),
        MaxPool2D(2, 2),
        Flatten(),
        Dense(128 * 3 * 3, 256),
        ReLU(),
        Dropout(0.5),
        Dense(256, 10), 
        Softmax()
    ])

def build_no_dropout_cnn():
    from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense, Softmax
    from model import Sequential
    
    return Sequential([
        Conv2D(1, 32, 3, padding=1), 
        ReLU(),
        MaxPool2D(2, 2),
        Conv2D(32, 64, 3, padding=1), 
        ReLU(),
        MaxPool2D(2, 2),
        Conv2D(64, 128, 3, padding=1), 
        ReLU(),
        MaxPool2D(2, 2),
        Flatten(),
        Dense(128 * 3 * 3, 256),
        ReLU(),
        Dense(256, 10), 
        Softmax()
    ])

def build_smaller_cnn():
    from layers import Conv2D, ReLU, MaxPool2D, Flatten, Dense, Dropout, Softmax
    from model import Sequential
    
    return Sequential([
        Conv2D(1, 32, 3, padding=1), 
        ReLU(),
        MaxPool2D(2, 2),
        Conv2D(32, 64, 3, padding=1), 
        ReLU(),
        MaxPool2D(2, 2),
        Flatten(),
        Dense(64 * 7 * 7, 128),
        ReLU(),
        Dropout(0.5),
        Dense(128, 10), 
        Softmax()
    ])

def compare_optimizers(X_train, y_train_oh, X_val, y_val, y_val_oh, X_test, y_test, 
                      epochs=20, batch_size=32):

    from optimizers import SGD, Adam, RMSprop
    from callbacks import EarlyStopping
    
    optimizers = {
        "SGD": SGD(lr=0.1, momentum=0.9),
        "Adam": Adam(lr=0.001),
        "RMSprop": RMSprop(lr=0.001)
    }
    
    results = {}
    histories = {}
    
    for opt_name, optimizer in optimizers.items():
        print(f"\n{'='*50}")
        print(f"Entraînement avec {opt_name}")
        print(f"{'='*50}")
        
        model = build_improved_cnn()
        scheduler = LearningRateScheduler(optimizer.lr, decay_rate=1.0, decay_steps=999)
        early_stopper = EarlyStopping(patience=5, min_delta=0.001)
        
        start_time = time()
        history, _, _ = train_model(
            model, 
            X_train, y_train_oh, 
            X_val, y_val, y_val_oh,
            optimizer, 
            scheduler, 
            epochs=epochs,   
            batch_size=batch_size,
            callbacks=[early_stopper]
        )
        training_time = time() - start_time
        
        test_acc, _ = evaluate_model(model, X_test, y_test, show_errors=False)
        
        results[opt_name] = {
            'test_accuracy': test_acc,
            'training_time': training_time,
            'best_val_acc': max(history['val_acc']),
            'best_epoch': np.argmax(history['val_acc']) + 1
        }
        histories[opt_name] = history
    
    plot_comparison(histories, save_path="optimizer_comparison.png")
    
    print("\n" + "="*50)
    print("RÉSUMÉ DE LA COMPARAISON DES OPTIMISEURS")
    print("="*50)
    print(f"{'Optimiseur':<10} | {'Précision Test':<15} | {'Temps (s)':<10} | {'Meilleure Val Acc':<18} | {'Époque Meilleure':<15}")
    print("-"*80)
    for opt_name, res in results.items():
        print(f"{opt_name:<10} | {res['test_accuracy']:<15.4f} | {res['training_time']:<10.2f} | {res['best_val_acc']:<18.4f} | {res['best_epoch']:<15}")
    
    return results, histories

def run_ablation_studies(X_train, y_train_oh, X_val, y_val, y_val_oh, X_test, y_test, 
                        epochs=20, batch_size=32):
    from optimizers import Adam
    from callbacks import EarlyStopping
    
    models = {
        'Baseline': build_baseline_cnn(),
        'Amélioré': build_improved_cnn(),
        'Sans Dropout': build_no_dropout_cnn(),
        'Moins de Couches': build_smaller_cnn()
    }
    
    results = {}
    histories = {}
    
    for model_name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Entraînement du modèle: {model_name}")
        print(f"{'='*50}")
        
        optimizer = Adam(lr=0.001)
        scheduler = LearningRateScheduler(0.001, decay_rate=0.5, decay_steps=5)
        early_stopper = EarlyStopping(patience=5, min_delta=0.001)
        
        start_time = time()
        history, _, _ = train_model(
            model, 
            X_train, y_train_oh, 
            X_val, y_val, y_val_oh,
            optimizer, 
            scheduler, 
            epochs=epochs, 
            batch_size=batch_size,
            callbacks=[early_stopper]
        )
        training_time = time() - start_time
        
        test_acc, _ = evaluate_model(model, X_test, y_test, show_errors=False)
        
        results[model_name] = {
            'test_accuracy': test_acc,
            'training_time': training_time,
            'best_val_acc': max(history['val_acc']),
            'best_epoch': np.argmax(history['val_acc']) + 1,
            'n_params': count_parameters(model)
        }
        histories[model_name] = history
    
    plot_comparison(histories, save_path="ablation_study.png")
    
    print("\n" + "="*50)
    print("RÉSUMÉ DE L'ÉTUDE D'ABLATION")
    print("="*50)
    print(f"{'Modèle':<15} | {'Précision Test':<15} | {'Temps (s)':<10} | {'Meilleure Val Acc':<18} | {'Époque Meilleure':<15} | {'Paramètres':<10}")
    print("-"*100)
    for model_name, res in results.items():
        print(f"{model_name:<15} | {res['test_accuracy']:<15.4f} | {res['training_time']:<10.2f} | {res['best_val_acc']:<18.4f} | {res['best_epoch']:<15} | {res['n_params']:<10}")
    
    return results, histories

def count_parameters(model):
    total = 0
    for layer in model.layers:
        if hasattr(layer, 'W'):
            total += layer.W.size
        if hasattr(layer, 'b'):
            total += layer.b.size
    return total