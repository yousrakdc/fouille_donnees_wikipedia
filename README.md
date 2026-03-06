# Fouille de Données Wikipedia : Science & Technologie

## Description

Ce projet explore la fouille de données sur des articles Wikipedia en français, autour de cinq grands thèmes scientifiques : **intelligence artificielle**, **informatique**, **physique**, **biologie** et **espace**. Il applique un pipeline complet : collecte, prétraitement, vectorisation, clustering et évaluation.

## Structure du projet

- **collecte.py** : Récupère les articles Wikipedia selon des sous-thèmes définis, en utilisant l’API Wikipedia. Les articles sont stockés dans `corpus.json`.
- **pretraitement.py** : Nettoie les textes, lemmatise avec spaCy, filtre les tokens, et génère un corpus propre (`corpus_propre.json`).
- **vectorisation.py** : Vectorise les textes avec TF-IDF et Word2Vec, sauvegarde les matrices (`matrice_tfidf.npy`, `matrice_w2v.npy`) et le modèle Word2Vec (`modele_w2v.model`).
- **clustering.py** : Applique des méthodes de clustering (K-Means, SOM) sur les matrices, réduit la dimension avec PCA, génère des labels (`labels_tfidf.npy`, `labels_w2v.npy`, `labels_som.npy`) et des visualisations (dans `resultats/`).
- **evaluation.py** : Évalue la qualité des clusters avec des métriques (Silhouette Score, Adjusted Rand Index), compare les méthodes, affiche les résultats.

## Fichiers de données

- **corpus.json** : Articles bruts collectés.
- **corpus_propre.json** : Corpus nettoyé et lemmatisé.
- **meta.json** : Métadonnées (titres, thèmes).
- **matrice_tfidf.npy** / **matrice_w2v.npy** : Matrices de vecteurs.
- **modele_w2v.model** : Modèle Word2Vec entraîné.

## Dossier `resultats/`

- **labels_tfidf.npy**, **labels_w2v.npy**, **labels_som.npy** : Labels de clustering.
- **kmeans_tfidf.png**, **kmeans_w2v.png**, **som_tfidf.png**, **comparaison_methodes.png** : Visualisations des clusters.

## Prérequis

- Python 3.11+
- Packages : `wikipedia-api`, `spacy`, `scikit-learn`, `gensim`, `matplotlib`, `minisom`, etc.
- Modèle spaCy français : `fr_core_news_sm`

## Pipeline d’exécution

1. **Collecte** : `python collecte.py`
2. **Prétraitement** : `python pretraitement.py`
3. **Vectorisation** : `python vectorisation.py`
4. **Clustering** : `python clustering.py`
5. **Évaluation** : `python evaluation.py`

## Résultats

- Analyse comparative des méthodes de clustering sur des articles scientifiques.
- Visualisation des clusters par thème.
- Évaluation quantitative (cohérence, pertinence).

## Contact

Projet réalisé par Yousra Kerdouchi.  
Date : Mars 2026.
