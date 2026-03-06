import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

# Chargement du corpus propre
with open("corpus_propre.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

titres = [article['titre'] for article in corpus]
themes = [article['theme'] for article in corpus]
tokens_list = [article['tokens'] for article in corpus]

# Textes sous forme de chaines pour TF-IDF
textes = [" ".join(tokens) for tokens in tokens_list]

print("=" * 50)
print("VECTORISATION TF-IDF")
print("=" * 50)

# TF-IDF
tfidf = TfidfVectorizer(max_features=500)
matrice_tfidf = tfidf.fit_transform(textes)

print(f"Dimensions de la matrice TF-IDF : {matrice_tfidf.shape}")
print(f"  -> {matrice_tfidf.shape[0]} articles x {matrice_tfidf.shape[1]} mots")

# Affichage des mots les plus importants pour le premier article
feature_names = tfidf.get_feature_names_out()
premier_article = matrice_tfidf[0].toarray()[0]
indices_tries = premier_article.argsort()[::-1][:10]
print(f"\nMots les plus importants pour '{titres[0]}' :")
for i in indices_tries:
    print(f"  {feature_names[i]} : {premier_article[i]:.4f}")

print("\n" + "=" * 50)
print("VECTORISATION WORD2VEC")
print("=" * 50)

# Word2Vec
w2v_model = Word2Vec(
    sentences=tokens_list,
    vector_size=100,
    window=5,
    min_count=1,
    workers=4,
    epochs=10
)

print(f"Vocabulaire Word2Vec : {len(w2v_model.wv)} mots")

# Verification : mots proches de "neurone"
if "neurone" in w2v_model.wv:
    print(f"\nMots proches de 'neurone' :")
    for mot, score in w2v_model.wv.most_similar("neurone", topn=5):
        print(f"  {mot} : {score:.4f}")

# Pour le clustering, on moyenne les vecteurs Word2Vec de chaque article
def vectoriser_article_w2v(tokens, model):
    vecteurs = [model.wv[mot] for mot in tokens if mot in model.wv]
    if vecteurs:
        return np.mean(vecteurs, axis=0)
    else:
        return np.zeros(model.vector_size)

matrice_w2v = np.array([
    vectoriser_article_w2v(tokens, w2v_model)
    for tokens in tokens_list
])

print(f"Dimensions de la matrice Word2Vec : {matrice_w2v.shape}")
print(f"  -> {matrice_w2v.shape[0]} articles x {matrice_w2v.shape[1]} dimensions")

# Sauvegarde des matrices
np.save("matrice_tfidf.npy", matrice_tfidf.toarray())
np.save("matrice_w2v.npy", matrice_w2v)
w2v_model.save("modele_w2v.model")

with open("meta.json", "w", encoding="utf-8") as f:
    json.dump({"titres": titres, "themes": themes}, f, ensure_ascii=False, indent=2)

print("\nMatrices sauvegardees : matrice_tfidf.npy, matrice_w2v.npy")
