import numpy as np
import matplotlib.pyplot as plt
from src import (
    load_mnist, augment_data, train_model, evaluate_model, plot_history,
    visualize_filters, visualize_feature_maps, plot_comparison,
    build_baseline_cnn, build_improved_cnn, build_no_dropout_cnn, build_smaller_cnn,
    compare_optimizers, run_ablation_studies,
    Adam, LearningRateScheduler, EarlyStopping
)

def main():
    print("=== SEMAIRE 3 - CNN AVANCÉ POUR MNIST ===\n")
    
    # Chargement des données
    print("1. Chargement des données MNIST...")
    (X_train, y_train, y_train_oh), (X_val, y_val, y_val_oh), (X_test, y_test, y_test_oh) = load_mnist()
    
    print(f"Taille des données:")
    print(f"  - Entraînement: {X_train.shape}")
    print(f"  - Validation: {X_val.shape}")
    print(f"  - Test: {X_test.shape}")
    
    # Augmentation des données
    print("\n2. Augmentation des données...")
    X_train_aug, y_train_aug = augment_data(X_train, y_train, max_aug=2)
    y_train_aug_oh = np.eye(10)[y_train_aug.astype(int)]
    
    print(f"  - Données après augmentation: {X_train_aug.shape}")
    
    # Entraînement du modèle amélioré
    print("\n3. Entraînement du modèle CNN amélioré...")
    model = build_improved_cnn()
    
    optimizer = Adam(lr=0.001)
    scheduler = LearningRateScheduler(0.001, decay_rate=0.5, decay_steps=5)
    early_stopper = EarlyStopping(patience=5, min_delta=0.001)
    
    history, feature_maps, filters = train_model(
        model, 
        X_train_aug, y_train_aug_oh, 
        X_val, y_val, y_val_oh,
        optimizer, 
        scheduler, 
        epochs=30, 
        batch_size=64,
        callbacks=[early_stopper],
        save_feature_maps=True,
        save_filters=True
    )
    
    # Évaluation finale
    print("\n4. Évaluation finale du modèle...")
    test_accuracy, predictions = evaluate_model(model, X_test, y_test)
    
    # Visualisation des courbes d'apprentissage
    plot_history(history, title="Modèle CNN Amélioré")
    
    # Visualisation des filtres et feature maps
    if filters:
        print("\n5. Visualisation des filtres...")
        visualize_filters(filters)
    
    if feature_maps:
        print("\n6. Visualisation des feature maps...")
        visualize_feature_maps(feature_maps)
    
    # Comparaison des optimiseurs
    print("\n7. Comparaison des optimiseurs...")
    optimizer_results, optimizer_histories = compare_optimizers(
        X_train_aug, y_train_aug_oh, X_val, y_val, y_val_oh, X_test, y_test,
        epochs=20, batch_size=64
    )
    
    # Étude d'ablation
    print("\n8. Étude d'ablation...")
    ablation_results, ablation_histories = run_ablation_studies(
        X_train_aug, y_train_aug_oh, X_val, y_val, y_val_oh, X_test, y_test,
        epochs=20, batch_size=64
    )
    
    # Résumé final
    print("\n" + "="*60)
    print("RÉSUMÉ FINAL")
    print("="*60)
    print(f"Meilleur modèle (CNN Amélioré): {test_accuracy:.4f}")
    
    best_optimizer = max(optimizer_results.items(), key=lambda x: x[1]['test_accuracy'])
    print(f"Meilleur optimiseur: {best_optimizer[0]} ({best_optimizer[1]['test_accuracy']:.4f})")
    
    best_ablation = max(ablation_results.items(), key=lambda x: x[1]['test_accuracy'])
    print(f"Meilleur modèle ablation: {best_ablation[0]} ({best_ablation[1]['test_accuracy']:.4f})")
    
    # Sauvegarde des résultats
    import pickle
    results = {
        'final_model_accuracy': test_accuracy,
        'optimizer_comparison': optimizer_results,
        'ablation_study': ablation_results
    }
    
    with open('week3_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\nRésultats sauvegardés dans 'week3_results.pkl'")
    print("=== FIN DE LA SEMAINE 3 ===")

if __name__ == "__main__":
    main()