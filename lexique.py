import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict

# ─── Chargement ───────────────────────────────────────────────────────────────
with open("corpus_propre.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

themes = [article["theme"] for article in corpus]
titres = [article["titre"] for article in corpus]
tokens_list = [article["tokens"] for article in corpus]
textes = [" ".join(tokens) for tokens in tokens_list]

themes_uniques = sorted(set(themes))

# ─── TF-IDF ───────────────────────────────────────────────────────────────────
tfidf = TfidfVectorizer(max_features=1000)
matrice = tfidf.fit_transform(textes)
feature_names = tfidf.get_feature_names_out()

# ─── Lexique par thème ────────────────────────────────────────────────────────
# Pour chaque thème, on calcule le score TF-IDF moyen de chaque mot
# sur tous les articles du thème, puis on prend les 15 premiers.

print("=" * 60)
print("ANALYSE LEXICALE PAR THÈME")
print("=" * 60)

lexique_par_theme = {}

for theme in themes_uniques:
    # Indices des articles de ce thème
    indices = [i for i, t in enumerate(themes) if t == theme]

    # Sous-matrice de ce thème
    sous_matrice = matrice[indices].toarray()

    # Score moyen TF-IDF par mot sur ce thème
    scores_moyens = sous_matrice.mean(axis=0)

    # Top 15 mots
    top_indices = scores_moyens.argsort()[::-1][:15]
    top_mots = [(feature_names[i], round(float(scores_moyens[i]), 4)) for i in top_indices]

    lexique_par_theme[theme] = top_mots

    print(f"\nThème : {theme.upper()}")
    print(f"{'Mot':<25} {'Score TF-IDF moyen':>20}")
    print("-" * 47)
    for mot, score in top_mots:
        print(f"{mot:<25} {score:>20.4f}")

# ─── Analyse des chevauchements ───────────────────────────────────────────────
print("\n" + "=" * 60)
print("MOTS PARTAGÉS ENTRE PLUSIEURS THÈMES (top 15 de chaque thème)")
print("=" * 60)

# On collecte tous les mots du top 15 de chaque thème
mots_par_theme = {theme: set(m for m, _ in mots) for theme, mots in lexique_par_theme.items()}

for theme in themes_uniques:
    for autre_theme in themes_uniques:
        if theme >= autre_theme:
            continue
        communs = mots_par_theme[theme] & mots_par_theme[autre_theme]
        if communs:
            print(f"\n{theme} ∩ {autre_theme} : {', '.join(sorted(communs))}")

# ─── Mots exclusifs à un seul thème ──────────────────────────────────────────
print("\n" + "=" * 60)
print("MOTS EXCLUSIFS À CHAQUE THÈME (dans le top 15)")
print("=" * 60)

for theme in themes_uniques:
    autres = set()
    for autre_theme, mots in mots_par_theme.items():
        if autre_theme != theme:
            autres |= mots
    exclusifs = mots_par_theme[theme] - autres
    print(f"\n{theme.upper()} : {', '.join(sorted(exclusifs)) if exclusifs else '(aucun mot exclusif dans le top 15)'}")

# ─── Sauvegarde JSON ─────────────────────────────────────────────────────────
with open("lexique_themes.json", "w", encoding="utf-8") as f:
    json.dump(lexique_par_theme, f, ensure_ascii=False, indent=2)

print("\n\nLexique sauvegardé dans lexique_themes.json")