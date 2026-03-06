import json
import re
import spacy

# Chargement du modèle français
nlp = spacy.load("fr_core_news_sm")

# Chargement du corpus brut
with open("corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)

def nettoyer_texte(texte):
    # Mise en minuscules
    texte = texte.lower()
    # Suppression des chiffres et caractères spéciaux
    texte = re.sub(r'[^a-zàâäéèêëîïôùûüç\s]', ' ', texte)
    # Suppression des espaces multiples
    texte = re.sub(r'\s+', ' ', texte).strip()
    return texte

def lemmatiser(texte):
    doc = nlp(texte)
    # On garde uniquement les mots significatifs :
    # - pas les stopwords (le, de, et...)
    # - pas la ponctuation
    # - longueur minimale de 3 caractères
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and len(token.lemma_) >= 3
    ]
    return tokens

# Traitement de chaque article
corpus_propre = []

for i, article in enumerate(corpus):
    print(f"Traitement {i+1}/{len(corpus)} : {article['titre']}")
    
    texte_nettoye = nettoyer_texte(article['texte'])
    tokens = lemmatiser(texte_nettoye)
    
    corpus_propre.append({
        "titre": article['titre'],
        "theme": article['theme'],
        "texte_original": article['texte'],
        "texte_nettoye": texte_nettoye,
        "tokens": tokens
    })

# Sauvegarde
with open("corpus_propre.json", "w", encoding="utf-8") as f:
    json.dump(corpus_propre, f, ensure_ascii=False, indent=2)

# Resume
print(f"\n{'='*50}")
print(f"Pretraitement termine : {len(corpus_propre)} articles")
print(f"\nExemple avec '{corpus_propre[0]['titre']}' :")
print(f"  Tokens (20 premiers) : {corpus_propre[0]['tokens'][:20]}")
print(f"Corpus propre sauvegarde dans corpus_propre.json")
