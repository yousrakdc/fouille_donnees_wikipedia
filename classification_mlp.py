"""
Classification supervisée par MLP (Perceptron Multi-Couche)
============================================================
Ce script s'intègre dans le pipeline existant du projet de fouille de données.
Il réutilise la même matrice TF-IDF que le clustering, mais cette fois
on entraîne un réseau de neurones supervisé (cours IA - Chapitre 2).

Objectif : comparer l'approche non-supervisée (K-Means, ARI = 0.67)
avec un classifieur supervisé (MLP) sur les mêmes données.

Concepts du cours d'IA utilisés :
- Perceptron multi-couche (MLP) avec couches cachées (section 2.4)
- Fonctions d'activation ReLU et Softmax (section 2.3)
- Rétropropagation et descente de gradient (section 2.5)
- Fonction de coût : entropie croisée (section 2.5.1)
- Optimiseur Adam (variante de la descente de gradient)
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# 1. CHARGEMENT DES DONNÉES (identique à vectorisation.py)
# ============================================================
print("=" * 55)
print("CLASSIFICATION SUPERVISÉE PAR MLP (cours IA)")
print("=" * 55)

with open("corpus_propre.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

titres = [article['titre'] for article in corpus]
themes = [article['theme'] for article in corpus]
tokens_list = [article['tokens'] for article in corpus]

# Textes sous forme de chaînes pour TF-IDF
textes = [" ".join(tokens) for tokens in tokens_list]

# ============================================================
# 2. VECTORISATION TF-IDF (même paramétrage que le projet)
# ============================================================
tfidf = TfidfVectorizer(max_features=500)
X = tfidf.fit_transform(textes).toarray()  # matrice 50 x 500

print(f"\nDonnées : {X.shape[0]} articles, {X.shape[1]} features TF-IDF")
print(f"Thèmes : {sorted(set(themes))}")

# ============================================================
# 3. ENCODAGE DES LABELS
# ============================================================
# On transforme les noms de thèmes en nombres (0, 1, 2, 3, 4)
# puis en one-hot encoding pour la sortie softmax du MLP
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(themes)  # ex: "biologie" -> 0

print(f"\nCorrespondance thème -> label :")
for i, classe in enumerate(label_encoder.classes_):
    nb = sum(1 for y in y_encoded if y == i)
    print(f"  {classe} -> {i}  ({nb} articles)")

# ============================================================
# 4. CONSTRUCTION DU MODÈLE MLP
# ============================================================
# Architecture inspirée de l'exercice MNIST du cours (section 2.6)
# mais adaptée à notre petit corpus de 50 articles :
#   - Couche d'entrée : 500 features (vecteurs TF-IDF)
#   - Couche cachée 1 : 64 neurones, activation ReLU
#   - Dropout : 30% (pour limiter le surapprentissage)
#   - Couche cachée 2 : 32 neurones, activation ReLU
#   - Couche de sortie : 5 neurones, activation Softmax (5 thèmes)
#
# On utilise moins de neurones que l'exercice MNIST (qui en avait 512)
# parce qu'on a seulement 50 exemples. Un réseau trop gros ferait
# du surapprentissage (overfitting, cf. cours section 1.10).

def construire_mlp(input_dim, num_classes):
    """
    Construit un MLP adapté à notre problème.
    
    Paramètres :
    - input_dim : nombre de features en entrée (500 pour TF-IDF)
    - num_classes : nombre de catégories en sortie (5 thèmes)
    
    Retourne : le modèle Keras compilé
    """
    model = Sequential()
    
    # Couche cachée 1 : 64 neurones avec activation ReLU
    # ReLU : f(x) = max(0, x) - cf. cours section 2.3.4
    model.add(Dense(64, activation='relu', input_shape=(input_dim,)))
    
    # Dropout : désactive aléatoirement 30% des neurones pendant
    # l'entraînement pour éviter le surapprentissage
    model.add(Dropout(0.3))
    
    # Couche cachée 2 : 32 neurones avec activation ReLU
    model.add(Dense(32, activation='relu'))
    
    # Couche de sortie : 5 neurones (1 par thème) avec Softmax
    # Softmax transforme les sorties en probabilités (somme = 1)
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compilation du modèle :
    # - Fonction de coût : entropie croisée catégorielle (cf. cours 2.5.1)
    # - Optimiseur : Adam (variante avancée de la descente de gradient)
    # - Métrique : accuracy (précision de classification)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    
    return model

# ============================================================
# 5. ÉVALUATION PAR VALIDATION CROISÉE (5-fold)
# ============================================================
# Avec seulement 50 articles, on ne peut pas se permettre un simple
# split train/test (on perdrait trop de données d'entraînement).
# La validation croisée stratifiée à 5 folds est plus robuste :
# chaque article est utilisé exactement 1 fois comme test.

print("\n" + "=" * 55)
print("ENTRAÎNEMENT ET ÉVALUATION (validation croisée 5-fold)")
print("=" * 55)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accuracies = []
predictions_globales = np.zeros(len(y_encoded), dtype=int)

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y_encoded)):
    # Séparation train/test pour ce fold
    X_train, X_test = X[train_idx], X[test_idx]
    y_train = to_categorical(y_encoded[train_idx], num_classes=5)
    y_test = to_categorical(y_encoded[test_idx], num_classes=5)
    
    # Construction du modèle (réinitialisé à chaque fold)
    model = construire_mlp(input_dim=500, num_classes=5)
    
    # Entraînement avec early stopping
    # (arrête l'entraînement si la loss ne s'améliore plus)
    early_stop = EarlyStopping(
        monitor='loss',
        patience=20,
        restore_best_weights=True
    )
    
    model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=8,
        verbose=0,
        callbacks=[early_stop]
    )
    
    # Évaluation sur le fold de test
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    accuracies.append(accuracy)
    
    # Récupération des prédictions pour analyse détaillée
    y_pred = model.predict(X_test, verbose=0)
    predictions_globales[test_idx] = np.argmax(y_pred, axis=1)
    
    print(f"  Fold {fold+1}/5 : accuracy = {accuracy:.2%}")

# ============================================================
# 6. RÉSULTATS ET COMPARAISON AVEC K-MEANS
# ============================================================
print("\n" + "=" * 55)
print("RÉSULTATS")
print("=" * 55)

accuracy_moyenne = np.mean(accuracies)
accuracy_std = np.std(accuracies)

print(f"\nAccuracy moyenne du MLP : {accuracy_moyenne:.2%} (± {accuracy_std:.2%})")

# Calcul de l'ARI pour comparer avec K-Means
from sklearn.metrics import adjusted_rand_score, classification_report

ari_mlp = adjusted_rand_score(y_encoded, predictions_globales)
print(f"Adjusted Rand Index du MLP : {ari_mlp:.4f}")

print(f"\n--- Comparaison avec les méthodes non-supervisées ---")
print(f"  K-Means TF-IDF (non-supervisé) : ARI = 0.6723")
print(f"  SOM TF-IDF (non-supervisé)      : ARI = 0.3294")
print(f"  MLP supervisé (ce script)       : ARI = {ari_mlp:.4f}")

# ============================================================
# 7. ANALYSE DÉTAILLÉE DES ERREURS
# ============================================================
print("\n" + "=" * 55)
print("ANALYSE DÉTAILLÉE")
print("=" * 55)

# Articles bien classés vs mal classés
print(f"\nArticles mal classés par le MLP :")
nb_erreurs = 0
for i in range(len(y_encoded)):
    if predictions_globales[i] != y_encoded[i]:
        theme_reel = label_encoder.inverse_transform([y_encoded[i]])[0]
        theme_predit = label_encoder.inverse_transform([predictions_globales[i]])[0]
        print(f"  - '{titres[i]}' : réel={theme_reel}, prédit={theme_predit}")
        nb_erreurs += 1

if nb_erreurs == 0:
    print("  Aucune erreur ! Tous les articles sont correctement classés.")
else:
    print(f"\n  Total : {nb_erreurs} erreur(s) sur {len(y_encoded)} articles")

# Rapport de classification par thème
print(f"\nRapport de classification par thème :")
print(classification_report(
    y_encoded, predictions_globales,
    target_names=label_encoder.classes_
))

# ============================================================
# 8. CONCLUSION
# ============================================================
print("=" * 55)
print("CONCLUSION")
print("=" * 55)
if ari_mlp > 0.6723:
    print(f"\nLe MLP supervisé (ARI={ari_mlp:.4f}) surpasse le K-Means")
    print(f"non-supervisé (ARI=0.6723). C'est attendu : le réseau de")
    print(f"neurones bénéficie des labels pendant l'entraînement.")
    print(f"La rétropropagation ajuste les poids pour minimiser l'erreur")
    print(f"de classification, contrairement au K-Means qui ne cherche")
    print(f"qu'à minimiser la distance intra-cluster.")
else:
    print(f"\nLe MLP supervisé (ARI={ari_mlp:.4f}) ne surpasse pas nettement")
    print(f"le K-Means (ARI=0.6723). Cela peut s'expliquer par la très")
    print(f"petite taille du corpus (50 articles) : le réseau n'a pas assez")
    print(f"d'exemples pour bien apprendre les frontières de décision,")
    print(f"ce qui illustre le risque de surapprentissage (overfitting)")
    print(f"mentionné dans le cours (section 1.10).")