# WikiCluster

**Clustering thématique non supervisé d'articles scientifiques Wikipedia**

Projet réalisé dans le cadre de la L3 Informatique — Université Paris 8 Vincennes  
Cours : Fouille de données, Ingénierie des langues, Développement de logiciel libre

## Description

WikiCluster est une application interactive qui collecte, prétraite, vectorise et regroupe automatiquement des articles encyclopédiques Wikipedia en français. L'objectif est de déterminer, sans supervision, quelle méthode de clustering regroupe le mieux les articles selon leur thème scientifique.

### Corpus
- 50 articles Wikipedia en français
- 5 thèmes : Intelligence artificielle, Informatique, Physique, Biologie, Espace
- 10 articles par thème, tronqués aux 3 000 premiers caractères

### Méthodes comparées
| Méthode | Vectorisation | Algorithme |
|---------|--------------|------------|
| 1 | TF-IDF (500 features) | K-Means (k=5) |
| 2 | Word2Vec (dim=100) | K-Means (k=5) |
| 3 | TF-IDF (500 features) | SOM (grille 5×5) |

### Évaluation
- **Silhouette Score** : cohérence géométrique interne des clusters
- **Adjusted Rand Index (ARI)** : comparaison avec les thèmes réels (gold standard Wikipedia)

## Installation

### Prérequis
- Python >= 3.9
- pip

### Installation rapide

```bash
# Cloner le dépôt
git clone https://github.com/yousrakerdouchi/fouille_donnees_wikipedia.git
cd wikicluster

# Installer les dépendances
pip install -r requirements.txt

# Télécharger le modèle spaCy français
python -m spacy download fr_core_news_sm
```

### Installation via le paquet .deb (Debian/Ubuntu)

```bash
sudo dpkg -i wikicluster_1.0.0_all.deb
sudo apt-get install -f  # résoudre les dépendances si nécessaire
```

## Utilisation

### Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre dans votre navigateur à l'adresse `http://localhost:8501`.

### Pipeline

1. **Collecte** : collecte des articles via l'API Wikipedia (ou chargement d'un fichier JSON existant)
2. **Prétraitement** : nettoyage, lemmatisation et suppression des mots vides avec spaCy
3. **Vectorisation** : calcul des matrices TF-IDF et Word2Vec
4. **Clustering** : K-Means et SOM avec visualisation PCA 2D
5. **Évaluation** : Silhouette Score et Adjusted Rand Index, graphiques comparatifs

### Paramètres configurables (sidebar)

- Nombre de caractères par article (1000–5000)
- Nombre de features TF-IDF (100–1000)
- Dimensions Word2Vec (50–300)
- Fenêtre de contexte Word2Vec (2–10)
- Nombre de clusters (2–10)
- Taille de la grille SOM (3–10)
- Nombre d'itérations SOM (100–2000)

## Structure du projet

```
wikicluster/
├── app.py                  # Application Streamlit principale
├── requirements.txt        # Dépendances Python
├── setup.py               # Configuration du paquet
├── LICENSE                # Licence GPL-3.0
├── README.md              # Ce fichier
└── data/                  # Données (corpus, matrices, résultats)
```

## Dépendances

| Bibliothèque | Usage |
|--------------|-------|
| streamlit | Interface web interactive |
| wikipedia-api | Collecte des articles Wikipedia |
| spacy | Lemmatisation et traitement linguistique |
| scikit-learn | TF-IDF, K-Means, PCA, métriques d'évaluation |
| gensim | Word2Vec |
| minisom | Self-Organizing Maps |
| matplotlib | Visualisations |
| numpy / pandas | Manipulation de données |

## Auteur

**Yousra Kerdouchi**  
L3 Informatique — Université Paris 8  
ID : 82507301

## Licence

Ce logiciel est distribué sous licence [GPL-3.0](LICENSE).