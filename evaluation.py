import json
import numpy as np
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Chargement des donnees
matrice_tfidf = np.load("matrice_tfidf.npy")
matrice_w2v = np.load("matrice_w2v.npy")

labels_tfidf = np.load("resultats/labels_tfidf.npy")
labels_w2v = np.load("resultats/labels_w2v.npy")
labels_som = np.load("resultats/labels_som.npy")

with open("meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

themes = meta['themes']
themes_uniques = list(set(themes))

# Conversion des themes en labels numeriques
theme_to_id = {theme: i for i, theme in enumerate(themes_uniques)}
labels_reels = np.array([theme_to_id[t] for t in themes])

print("=" * 50)
print("EVALUATION DES CLUSTERS")
print("=" * 50)

print("\n1. Silhouette Score (entre -1 et 1, plus c'est proche de 1, mieux c'est)")
print("   Mesure la coherence interne des clusters\n")

score_tfidf = silhouette_score(matrice_tfidf, labels_tfidf)
score_w2v = silhouette_score(matrice_w2v, labels_w2v)
score_som = silhouette_score(matrice_tfidf, labels_som)

print(f"  K-Means TF-IDF  : {score_tfidf:.4f}")
print(f"  K-Means Word2Vec: {score_w2v:.4f}")
print(f"  SOM TF-IDF      : {score_som:.4f}")

print("\n2. Adjusted Rand Index (entre 0 et 1, plus c'est proche de 1, mieux c'est)")
print("   Compare les clusters trouves avec les themes reels\n")

ari_tfidf = adjusted_rand_score(labels_reels, labels_tfidf)
ari_w2v = adjusted_rand_score(labels_reels, labels_w2v)
ari_som = adjusted_rand_score(labels_reels, labels_som)

print(f"  K-Means TF-IDF  : {ari_tfidf:.4f}")
print(f"  K-Means Word2Vec: {ari_w2v:.4f}")
print(f"  SOM TF-IDF      : {ari_som:.4f}")

print("\n3. Analyse detaillee : articles mal classes (K-Means TF-IDF)")
print("   On compare le cluster trouve avec le theme reel\n")

with open("meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)
titres = meta['titres']

# Pour chaque cluster TF-IDF, quel theme domine ?
for cluster_id in range(5):
    indices = [i for i, l in enumerate(labels_tfidf) if l == cluster_id]
    themes_cluster = [themes[i] for i in indices]
    theme_dominant = max(set(themes_cluster), key=themes_cluster.count)
    mal_classes = [titres[i] for i in indices if themes[i] != theme_dominant]
    print(f"  Cluster {cluster_id} - theme dominant : {theme_dominant}")
    if mal_classes:
        print(f"    Articles hors theme : {', '.join(mal_classes)}")
    else:
        print(f"    Cluster parfaitement homogene")

# Graphique comparatif des scores
print("\n4. Graphique comparatif")

methodes = ['K-Means\nTF-IDF', 'K-Means\nWord2Vec', 'SOM\nTF-IDF']
silhouette_scores = [score_tfidf, score_w2v, score_som]
ari_scores = [ari_tfidf, ari_w2v, ari_som]

x = np.arange(len(methodes))
largeur = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
barres1 = ax.bar(x - largeur/2, silhouette_scores, largeur, label='Silhouette Score', color='steelblue')
barres2 = ax.bar(x + largeur/2, ari_scores, largeur, label='Adjusted Rand Index', color='darkorange')

ax.set_ylabel('Score')
ax.set_title('Comparaison des methodes de clustering')
ax.set_xticks(x)
ax.set_xticklabels(methodes)
ax.legend()
ax.set_ylim(0, 1)

for barre in barres1:
    ax.text(barre.get_x() + barre.get_width()/2, barre.get_height() + 0.01,
            f'{barre.get_height():.2f}', ha='center', va='bottom', fontsize=10)
for barre in barres2:
    ax.text(barre.get_x() + barre.get_width()/2, barre.get_height() + 0.01,
            f'{barre.get_height():.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig("resultats/comparaison_methodes.png")
plt.close()
print("Graphique sauvegarde : resultats/comparaison_methodes.png")

print("\n" + "=" * 50)
print("CONCLUSION")
print("=" * 50)
meilleure_methode = methodes[np.argmax(ari_scores)].replace('\n', ' ')
print(f"Meilleure methode (ARI) : {meilleure_methode}")
print(f"Meilleure methode (Silhouette) : {methodes[np.argmax(silhouette_scores)].replace(chr(10), ' ')}")
