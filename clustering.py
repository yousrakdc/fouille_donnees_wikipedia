import json
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from minisom import MiniSom
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# Chargement des donnees
matrice_tfidf = np.load("matrice_tfidf.npy")
matrice_w2v = np.load("matrice_w2v.npy")

with open("meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

titres = meta['titres']
themes = meta['themes']
themes_uniques = list(set(themes))
couleurs_themes = {theme: i for i, theme in enumerate(themes_uniques)}

os.makedirs("resultats", exist_ok=True)

# Nombre de clusters = nombre de themes
N_CLUSTERS = 5

# Couleurs pour les graphiques
palette = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']
couleur_par_theme = {theme: palette[i] for i, theme in enumerate(themes_uniques)}

def reduire_dimensions(matrice, n_components=2):
    pca = PCA(n_components=n_components)
    return pca.fit_transform(matrice)

def afficher_clusters(coords, labels, titres, themes, titre_graphique, nom_fichier):
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, (x, y) in enumerate(coords):
        couleur = palette[labels[i] % len(palette)]
        ax.scatter(x, y, c=couleur, s=100, alpha=0.7)
        ax.annotate(
            titres[i].split()[0],
            (x, y),
            fontsize=7,
            alpha=0.8
        )

    legende = [
        mpatches.Patch(color=palette[i], label=f"Cluster {i}")
        for i in range(N_CLUSTERS)
    ]
    ax.legend(handles=legende, loc='upper right')
    ax.set_title(titre_graphique)
    ax.set_xlabel("Composante 1")
    ax.set_ylabel("Composante 2")
    plt.tight_layout()
    plt.savefig(f"resultats/{nom_fichier}")
    plt.close()
    print(f"Graphique sauvegarde : resultats/{nom_fichier}")

print("=" * 50)
print("CLUSTERING K-MEANS SUR TF-IDF")
print("=" * 50)

kmeans_tfidf = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
labels_tfidf = kmeans_tfidf.fit_predict(matrice_tfidf)

coords_tfidf = reduire_dimensions(matrice_tfidf)
afficher_clusters(coords_tfidf, labels_tfidf, titres, themes,
                  "K-Means sur TF-IDF", "kmeans_tfidf.png")

print("\nRepartition des articles par cluster (TF-IDF) :")
for cluster_id in range(N_CLUSTERS):
    articles_cluster = [titres[i] for i, l in enumerate(labels_tfidf) if l == cluster_id]
    print(f"\n  Cluster {cluster_id} ({len(articles_cluster)} articles) :")
    for titre in articles_cluster:
        print(f"    - {titre}")

print("\n" + "=" * 50)
print("CLUSTERING K-MEANS SUR WORD2VEC")
print("=" * 50)

kmeans_w2v = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
labels_w2v = kmeans_w2v.fit_predict(matrice_w2v)

coords_w2v = reduire_dimensions(matrice_w2v)
afficher_clusters(coords_w2v, labels_w2v, titres, themes,
                  "K-Means sur Word2Vec", "kmeans_w2v.png")

print("\nRepartition des articles par cluster (Word2Vec) :")
for cluster_id in range(N_CLUSTERS):
    articles_cluster = [titres[i] for i, l in enumerate(labels_w2v) if l == cluster_id]
    print(f"\n  Cluster {cluster_id} ({len(articles_cluster)} articles) :")
    for titre in articles_cluster:
        print(f"    - {titre}")

print("\n" + "=" * 50)
print("CLUSTERING SOM SUR TF-IDF")
print("=" * 50)

# Normalisation pour le SOM
matrice_tfidf_norm = matrice_tfidf / (np.linalg.norm(matrice_tfidf, axis=1, keepdims=True) + 1e-10)

som = MiniSom(x=5, y=5, input_len=matrice_tfidf.shape[1], sigma=1.0, learning_rate=0.5)
som.random_weights_init(matrice_tfidf_norm)
som.train(matrice_tfidf_norm, num_iteration=500)

# Attribution de chaque article a une cellule du SOM
positions_som = np.array([som.winner(x) for x in matrice_tfidf_norm])

# Conversion en labels de cluster via K-Means sur les positions
positions_flat = positions_som.reshape(-1, 2)
kmeans_som = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
labels_som = kmeans_som.fit_predict(positions_flat)

coords_som = reduire_dimensions(matrice_tfidf)
afficher_clusters(coords_som, labels_som, titres, themes,
                  "SOM sur TF-IDF", "som_tfidf.png")

print("\nRepartition des articles par cluster (SOM) :")
for cluster_id in range(N_CLUSTERS):
    articles_cluster = [titres[i] for i, l in enumerate(labels_som) if l == cluster_id]
    print(f"\n  Cluster {cluster_id} ({len(articles_cluster)} articles) :")
    for titre in articles_cluster:
        print(f"    - {titre}")

# Sauvegarde des labels
np.save("resultats/labels_tfidf.npy", labels_tfidf)
np.save("resultats/labels_w2v.npy", labels_w2v)
np.save("resultats/labels_som.npy", labels_som)

print("\nLabels sauvegardes dans le dossier resultats/")
