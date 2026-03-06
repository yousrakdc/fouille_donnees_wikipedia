import wikipediaapi
import json
import time

# Initialisation de l'API Wikipedia en français
wiki = wikipediaapi.Wikipedia(
    language='fr',
    user_agent='ProjetFouilleDonnees/1.0'
)

# Sous-thèmes science & technologie
categories = {
    "intelligence_artificielle": [
        "Intelligence artificielle", "Apprentissage automatique",
        "Réseau de neurones artificiels", "Traitement automatique du langage naturel",
        "Vision par ordinateur", "Deep learning", "Algorithme",
        "Reconnaissance vocale", "Robotique", "ChatGPT"
    ],
    "informatique": [
        "Informatique", "Programmation informatique", "Système d'exploitation",
        "Internet", "Cryptographie", "Base de données", "Logiciel libre",
        "Cybersécurité", "Cloud computing", "Blockchain"
    ],
    "physique": [
        "Physique quantique", "Relativité restreinte", "Thermodynamique",
        "Mécanique classique", "Électromagnétisme", "Physique des particules",
        "Matière noire", "Trou noir", "Supraconductivité", "Fusion nucléaire"
    ],
    "biologie": [
        "Biologie cellulaire", "Génétique", "Évolution", "ADN", "Protéine",
        "Microbiome", "CRISPR", "Neurosciences", "Immunologie", "Épigénétique"
    ],
    "espace": [
        "Exploration spatiale", "Station spatiale internationale", "Mars",
        "Exoplanète", "Télescope James-Webb", "SpaceX", "NASA",
        "Cosmologie", "Big Bang", "Satellite artificiel"
    ]
}

# Collecte des articles
corpus = []
articles_non_trouves = []

for theme, articles in categories.items():
    print(f"\nCollecte du theme : {theme}")
    for titre in articles:
        page = wiki.page(titre)
        if page.exists():
            texte = page.text[:3000]
            corpus.append({
                "titre": titre,
                "theme": theme,
                "texte": texte
            })
            print(f"  OK : {titre} ({len(texte)} caracteres)")
        else:
            articles_non_trouves.append(titre)
            print(f"  Introuvable : {titre}")
        time.sleep(0.5)

# Sauvegarde en JSON
with open("corpus.json", "w", encoding="utf-8") as f:
    json.dump(corpus, f, ensure_ascii=False, indent=2)

# Resume
print(f"\n{'='*50}")
print(f"Articles collectes : {len(corpus)}")
print(f"Articles non trouves : {len(articles_non_trouves)}")
if articles_non_trouves:
    print(f"   -> {', '.join(articles_non_trouves)}")
print(f"Corpus sauvegarde dans corpus.json")
