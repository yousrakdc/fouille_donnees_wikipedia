"""
WikiCluster - Clustering thématique d'articles Wikipedia
Application Streamlit pour la fouille de données et l'ingénierie des langues
Auteur : Yousra Kerdouchi
L3 Informatique - Université Paris 8
Licence : GPL-3.0
"""

import streamlit as st
import json
import re
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from io import StringIO

# ========================================================================
# Configuration de la page
# ========================================================================
st.set_page_config(
    page_title="WikiCluster",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================================================================
# CSS personnalisé
# ========================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    .stApp {
        font-family: 'Source Sans 3', sans-serif;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 1.1rem;
        color: #6c757d;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px;
        padding: 1.2rem;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
    }

    .metric-card h3 {
        font-size: 0.85rem;
        font-weight: 400;
        opacity: 0.9;
        margin: 0;
    }

    .metric-card .value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0.3rem 0;
    }

    .cluster-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.15rem;
        color: white;
    }

    .step-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a1a2e;
        border-left: 4px solid #667eea;
        padding-left: 0.8rem;
        margin: 1.5rem 0 1rem 0;
    }

    .info-box {
        background: #f0f4ff;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
        font-size: 0.95rem;
    }

    code {
        font-family: 'JetBrains Mono', monospace;
    }

    div[data-testid="stSidebar"] {
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

# ========================================================================
# Définition des catégories (identique à ton collecte.py)
# ========================================================================
CATEGORIES = {
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

N_CLUSTERS = 5
PALETTE = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

# ========================================================================
# Fonctions utilitaires (basées sur tes scripts)
# ========================================================================

def nettoyer_texte(texte):
    """Nettoyage du texte (identique à ton pretraitement.py)"""
    texte = texte.lower()
    texte = re.sub(r'[^a-zàâäéèêëïîôùûüç\s]', ' ', texte)
    texte = re.sub(r'\s+', ' ', texte).strip()
    return texte


def lemmatiser(texte, nlp):
    """Lemmatisation avec spaCy (identique à ton pretraitement.py)"""
    doc = nlp(texte)
    tokens = [
        token.lemma_
        for token in doc
        if not token.is_stop
        and not token.is_punct
        and len(token.lemma_) >= 3
    ]
    return tokens


def collecter_articles(progress_bar, status_text):
    """Collecte des articles via l'API Wikipedia (basé sur ton collecte.py)"""
    import wikipediaapi
    
    wiki = wikipediaapi.Wikipedia(
        language='fr',
        user_agent='WikiCluster/1.0'
    )
    
    corpus = []
    articles_non_trouves = []
    total = sum(len(v) for v in CATEGORIES.values())
    count = 0
    
    for theme, articles in CATEGORIES.items():
        for titre in articles:
            page = wiki.page(titre)
            if page.exists():
                texte = page.text[:3000]
                corpus.append({
                    "titre": titre,
                    "theme": theme,
                    "texte": texte
                })
                status_text.text(f"✓ {titre} ({len(texte)} caractères)")
            else:
                articles_non_trouves.append(titre)
                status_text.text(f"✗ Introuvable : {titre}")
            count += 1
            progress_bar.progress(count / total)
            time.sleep(0.5)
    
    return corpus, articles_non_trouves


def pretraiter_corpus(corpus, progress_bar, status_text):
    """Prétraitement du corpus (basé sur ton pretraitement.py)"""
    import spacy
    nlp = spacy.load("fr_core_news_sm")
    
    corpus_propre = []
    for i, article in enumerate(corpus):
        status_text.text(f"Traitement {i+1}/{len(corpus)} : {article['titre']}")
        
        texte_nettoye = nettoyer_texte(article['texte'])
        tokens = lemmatiser(texte_nettoye, nlp)
        
        corpus_propre.append({
            "titre": article['titre'],
            "theme": article['theme'],
            "texte_original": article['texte'],
            "texte_nettoye": texte_nettoye,
            "tokens": tokens
        })
        progress_bar.progress((i + 1) / len(corpus))
    
    return corpus_propre


def vectoriser_tfidf(corpus_propre):
    """Vectorisation TF-IDF (basé sur ton vectorisation.py)"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    textes = [" ".join(article['tokens']) for article in corpus_propre]
    tfidf = TfidfVectorizer(max_features=500)
    matrice = tfidf.fit_transform(textes)
    feature_names = tfidf.get_feature_names_out()
    
    return matrice, feature_names, tfidf


def vectoriser_word2vec(corpus_propre):
    """Vectorisation Word2Vec (basé sur ton vectorisation.py)"""
    from gensim.models import Word2Vec
    
    tokens_list = [article['tokens'] for article in corpus_propre]
    
    model = Word2Vec(
        sentences=tokens_list,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        seed=42
    )
    
    matrice = np.array([
        np.mean([model.wv[mot] for mot in tokens if mot in model.wv] 
                 or [np.zeros(100)], axis=0)
        for tokens in tokens_list
    ])
    
    return matrice, model


def faire_clustering_kmeans(matrice, n_clusters=N_CLUSTERS):
    """K-Means clustering (basé sur ton clustering.py)"""
    from sklearn.cluster import KMeans
    
    if hasattr(matrice, 'toarray'):
        matrice_dense = matrice.toarray()
    else:
        matrice_dense = matrice
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(matrice_dense)
    
    return labels


def faire_clustering_som(matrice, grid_size=5, n_iterations=500):
    """SOM clustering (basé sur ton clustering.py)"""
    from minisom import MiniSom
    
    if hasattr(matrice, 'toarray'):
        matrice_dense = matrice.toarray()
    else:
        matrice_dense = matrice
    
    som = MiniSom(grid_size, grid_size, matrice_dense.shape[1],
                  sigma=1.0, learning_rate=0.5, random_seed=42)
    som.random_weights_init(matrice_dense)
    som.train_random(matrice_dense, n_iterations)
    
    # Attribution des clusters via K-Means sur les positions SOM
    from sklearn.cluster import KMeans
    winners = np.array([som.winner(x) for x in matrice_dense])
    positions = winners[:, 0] * grid_size + winners[:, 1]
    
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
    labels = kmeans.fit_predict(positions.reshape(-1, 1))
    
    return labels


def reduire_dimensions(matrice, n_components=2):
    """Réduction PCA (basé sur ton clustering.py)"""
    from sklearn.decomposition import PCA
    
    if hasattr(matrice, 'toarray'):
        matrice_dense = matrice.toarray()
    else:
        matrice_dense = matrice
    
    pca = PCA(n_components=n_components)
    return pca.fit_transform(matrice_dense)


def calculer_scores(matrice, labels_dict, themes):
    """Calcul des scores d'évaluation (basé sur ton evaluation.py)"""
    from sklearn.metrics import silhouette_score, adjusted_rand_score
    
    if hasattr(matrice, 'toarray'):
        matrice_dense = matrice.toarray()
    else:
        matrice_dense = matrice
    
    themes_uniques = list(set(themes))
    theme_to_id = {theme: i for i, theme in enumerate(themes_uniques)}
    labels_reels = np.array([theme_to_id[t] for t in themes])
    
    resultats = {}
    for nom, labels in labels_dict.items():
        sil = silhouette_score(matrice_dense, labels)
        ari = adjusted_rand_score(labels_reels, labels)
        resultats[nom] = {"silhouette": sil, "ari": ari}
    
    return resultats


# ========================================================================
# Sidebar
# ========================================================================
with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("---")
    
    st.markdown("### Collecte")
    max_chars = st.slider("Caractères par article", 1000, 5000, 3000, 500)
    
    st.markdown("### Vectorisation")
    max_features_tfidf = st.slider("Features TF-IDF", 100, 1000, 500, 50)
    w2v_vector_size = st.slider("Dimensions Word2Vec", 50, 300, 100, 50)
    w2v_window = st.slider("Fenêtre Word2Vec", 2, 10, 5)
    
    st.markdown("### Clustering")
    n_clusters = st.slider("Nombre de clusters", 2, 10, 5)
    som_grid = st.slider("Grille SOM (n×n)", 3, 10, 5)
    som_iterations = st.slider("Itérations SOM", 100, 2000, 500, 100)
    
    st.markdown("---")
    st.markdown("### À propos")
    st.markdown("""
    **WikiCluster** v1.0  
    Projet L3 Informatique  
    Université Paris 8  
    
    Cours :
    - Fouille de données
    - Ingénierie des langues  
    - Développement logiciel libre
    
    Licence GPL-3.0
    """)


# ========================================================================
# Page principale
# ========================================================================
st.markdown('<div class="main-title">WikiCluster</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Clustering thématique non supervisé d\'articles scientifiques Wikipedia</div>', unsafe_allow_html=True)

# Tabs principaux
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Collecte", 
    "Prétraitement", 
    "Vectorisation",
    "Clustering", 
    "Évaluation"
])

# ========================================================================
# TAB 1 - Collecte
# ========================================================================
with tab1:
    st.markdown('<div class="step-header">Étape 1 - Collecte du corpus Wikipedia</div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-box">
    Le corpus comprend <strong>50 articles</strong> Wikipedia en français, répartis en 
    <strong>5 thèmes</strong> de 10 articles chacun. Chaque article est tronqué aux 
    <strong>{max_chars} premiers caractères</strong>. La collecte utilise l'API 
    <code>wikipedia-api</code>.
    </div>
    """, unsafe_allow_html=True)
    
    # Afficher les catégories
    cols = st.columns(5)
    theme_labels = {
        "intelligence_artificielle": "🤖 IA",
        "informatique": "Informatique",
        "physique": "Physique",
        "biologie": "Biologie",
        "espace": "Espace"
    }
    
    for i, (theme, articles) in enumerate(CATEGORIES.items()):
        with cols[i]:
            st.markdown(f"**{theme_labels[theme]}**")
            for a in articles:
                st.markdown(f"- {a}")
    
    st.markdown("---")
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        if st.button("Lancer la collecte", type="primary", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()
            
            with st.spinner("Collecte en cours..."):
                corpus, non_trouves = collecter_articles(progress, status)
            
            st.session_state['corpus'] = corpus
            st.session_state['non_trouves'] = non_trouves
            
            st.success(f"{len(corpus)} articles collectés avec succès !")
            if non_trouves:
                st.warning(f"{len(non_trouves)} articles non trouvés : {', '.join(non_trouves)}")
    
    with col_btn2:
        uploaded = st.file_uploader("Ou charger un corpus.json existant", type="json")
        if uploaded:
            corpus = json.load(uploaded)
            st.session_state['corpus'] = corpus
            st.success(f"Corpus chargé : {len(corpus)} articles")
    
    # Aperçu du corpus
    if 'corpus' in st.session_state:
        st.markdown("### Aperçu du corpus")
        corpus = st.session_state['corpus']
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Articles collectés", len(corpus))
        col2.metric("Thèmes", len(set(a['theme'] for a in corpus)))
        col3.metric("Caractères/article", f"{max_chars}")
        
        # Tableau
        df_corpus = pd.DataFrame([
            {"Titre": a['titre'], "Thème": a['theme'], 
             "Longueur": len(a['texte']), "Extrait": a['texte'][:100] + "..."}
            for a in corpus
        ])
        st.dataframe(df_corpus, use_container_width=True, height=300)
        
        # Téléchargement
        json_str = json.dumps(corpus, ensure_ascii=False, indent=2)
        st.download_button(
            "Télécharger corpus.json",
            json_str,
            "corpus.json",
            "application/json"
        )


# ========================================================================
# TAB 2 - Prétraitement
# ========================================================================
with tab2:
    st.markdown('<div class="step-header">Étape 2 - Prétraitement linguistique</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Le prétraitement comprend trois étapes : <strong>mise en minuscules</strong>, 
    <strong>suppression des caractères spéciaux et chiffres</strong>, puis 
    <strong>lemmatisation avec spaCy</strong> (modèle <code>fr_core_news_sm</code>) 
    en filtrant les mots vides, la ponctuation et les mots de moins de 3 caractères.
    </div>
    """, unsafe_allow_html=True)
    
    if 'corpus' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord collecter ou charger un corpus dans l'onglet Collecte.")
    else:
        if st.button("Lancer le prétraitement", type="primary", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()
            
            with st.spinner("Prétraitement en cours (lemmatisation spaCy)..."):
                corpus_propre = pretraiter_corpus(st.session_state['corpus'], progress, status)
            
            st.session_state['corpus_propre'] = corpus_propre
            st.success(f"Prétraitement terminé : {len(corpus_propre)} articles traités")
        
        if 'corpus_propre' in st.session_state:
            corpus_propre = st.session_state['corpus_propre']
            
            # Statistiques
            all_tokens = [t for a in corpus_propre for t in a['tokens']]
            unique_tokens = set(all_tokens)
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Articles traités", len(corpus_propre))
            col2.metric("Tokens totaux", f"{len(all_tokens):,}")
            col3.metric("Vocabulaire unique", f"{len(unique_tokens):,}")
            col4.metric("Tokens/article (moy.)", f"{len(all_tokens)//len(corpus_propre)}")
            
            # Aperçu article par article
            st.markdown("### Aperçu par article")
            article_choisi = st.selectbox(
                "Choisir un article :",
                [a['titre'] for a in corpus_propre]
            )
            
            article = next(a for a in corpus_propre if a['titre'] == article_choisi)
            
            col_orig, col_clean = st.columns(2)
            with col_orig:
                st.markdown("**Texte original (extrait) :**")
                st.text_area("", article['texte_original'][:500], height=200, 
                           disabled=True, key="orig")
            with col_clean:
                st.markdown(f"**Tokens ({len(article['tokens'])} mots) :**")
                st.text_area("", " ".join(article['tokens'][:80]) + "...", height=200,
                           disabled=True, key="clean")
            
            # Distribution des tokens par thème
            st.markdown("### Distribution des tokens par thème")
            theme_tokens = {}
            for a in corpus_propre:
                theme_tokens.setdefault(a['theme'], []).extend(a['tokens'])
            
            fig, ax = plt.subplots(figsize=(10, 4))
            themes_names = list(theme_tokens.keys())
            counts = [len(theme_tokens[t]) for t in themes_names]
            bars = ax.bar(themes_names, counts, color=PALETTE[:len(themes_names)])
            ax.set_ylabel("Nombre de tokens")
            ax.set_title("Tokens par thème après prétraitement")
            for bar, c in zip(bars, counts):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                       str(c), ha='center', fontsize=10)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ========================================================================
# TAB 3 - Vectorisation
# ========================================================================
with tab3:
    st.markdown('<div class="step-header">Étape 3 - Vectorisation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Deux méthodes de vectorisation sont comparées : <strong>TF-IDF</strong> (méthode 
    statistique, matrice creuse) et <strong>Word2Vec</strong> (méthode par embeddings, 
    vecteurs denses). TF-IDF fonctionne bien sur les petits corpus, tandis que Word2Vec 
    nécessite de grandes quantités de données.
    </div>
    """, unsafe_allow_html=True)
    
    if 'corpus_propre' not in st.session_state:
        st.warning("Veuillez d'abord effectuer le prétraitement dans l'onglet précédent.")
    else:
        if st.button("Lancer la vectorisation", type="primary", use_container_width=True):
            corpus_propre = st.session_state['corpus_propre']
            
            with st.spinner("Vectorisation TF-IDF..."):
                matrice_tfidf, feature_names, tfidf_model = vectoriser_tfidf(corpus_propre)
                st.session_state['matrice_tfidf'] = matrice_tfidf
                st.session_state['feature_names'] = feature_names
            
            with st.spinner("Vectorisation Word2Vec..."):
                matrice_w2v, w2v_model = vectoriser_word2vec(corpus_propre)
                st.session_state['matrice_w2v'] = matrice_w2v
                st.session_state['w2v_model'] = w2v_model
            
            st.success("Vectorisation terminée !")
        
        if 'matrice_tfidf' in st.session_state:
            matrice_tfidf = st.session_state['matrice_tfidf']
            matrice_w2v = st.session_state['matrice_w2v']
            corpus_propre = st.session_state['corpus_propre']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### TF-IDF")
                st.metric("Dimensions", f"{matrice_tfidf.shape[0]} × {matrice_tfidf.shape[1]}")
                
                # Top mots par article
                st.markdown("**Mots les plus importants (1er article) :**")
                premier = matrice_tfidf[0].toarray()[0]
                feature_names = st.session_state['feature_names']
                top_indices = premier.argsort()[::-1][:10]
                
                df_top = pd.DataFrame({
                    "Mot": [feature_names[i] for i in top_indices],
                    "Score TF-IDF": [round(premier[i], 4) for i in top_indices]
                })
                st.dataframe(df_top, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("### Word2Vec")
                st.metric("Dimensions", f"{matrice_w2v.shape[0]} × {matrice_w2v.shape[1]}")
                
                w2v_model = st.session_state['w2v_model']
                st.metric("Vocabulaire appris", f"{len(w2v_model.wv)} mots")
                
                # Mots similaires
                st.markdown("**Mots proches de « neurone » :**")
                try:
                    similaires = w2v_model.wv.most_similar("neurone", topn=5)
                    df_sim = pd.DataFrame(similaires, columns=["Mot", "Similarité"])
                    df_sim["Similarité"] = df_sim["Similarité"].round(4)
                    st.dataframe(df_sim, use_container_width=True, hide_index=True)
                except KeyError:
                    st.info("Le mot 'neurone' n'est pas dans le vocabulaire.")


# ========================================================================
# TAB 4 - Clustering
# ========================================================================
with tab4:
    st.markdown('<div class="step-header">Étape 4 - Clustering</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Trois combinaisons sont comparées : <strong>K-Means sur TF-IDF</strong>, 
    <strong>K-Means sur Word2Vec</strong>, et <strong>SOM sur TF-IDF</strong>. 
    Le nombre de clusters est fixé à k=5, correspondant aux 5 thèmes du corpus. 
    Les résultats sont visualisés après réduction PCA en 2D.
    </div>
    """, unsafe_allow_html=True)
    
    if 'matrice_tfidf' not in st.session_state:
        st.warning("⚠️ Veuillez d'abord effectuer la vectorisation dans l'onglet précédent.")
    else:
        if st.button("Lancer le clustering", type="primary", use_container_width=True):
            matrice_tfidf = st.session_state['matrice_tfidf']
            matrice_w2v = st.session_state['matrice_w2v']
            corpus_propre = st.session_state['corpus_propre']
            titres = [a['titre'] for a in corpus_propre]
            themes = [a['theme'] for a in corpus_propre]
            
            with st.spinner("K-Means sur TF-IDF..."):
                labels_tfidf = faire_clustering_kmeans(matrice_tfidf, n_clusters)
                st.session_state['labels_tfidf'] = labels_tfidf
            
            with st.spinner("K-Means sur Word2Vec..."):
                labels_w2v = faire_clustering_kmeans(matrice_w2v, n_clusters)
                st.session_state['labels_w2v'] = labels_w2v
            
            with st.spinner("SOM sur TF-IDF..."):
                labels_som = faire_clustering_som(matrice_tfidf, som_grid, som_iterations)
                st.session_state['labels_som'] = labels_som
            
            st.session_state['titres'] = titres
            st.session_state['themes'] = themes
            
            st.success("Clustering terminé !")
        
        if 'labels_tfidf' in st.session_state:
            titres = st.session_state['titres']
            themes = st.session_state['themes']
            matrice_tfidf = st.session_state['matrice_tfidf']
            matrice_w2v = st.session_state['matrice_w2v']
            
            methode_affichee = st.selectbox(
                "Méthode à afficher :",
                ["K-Means sur TF-IDF", "K-Means sur Word2Vec", "SOM sur TF-IDF"]
            )
            
            if methode_affichee == "K-Means sur TF-IDF":
                labels = st.session_state['labels_tfidf']
                coords = reduire_dimensions(matrice_tfidf)
            elif methode_affichee == "K-Means sur Word2Vec":
                labels = st.session_state['labels_w2v']
                coords = reduire_dimensions(matrice_w2v)
            else:
                labels = st.session_state['labels_som']
                coords = reduire_dimensions(matrice_tfidf)
            
            # Graphique scatter
            col_graph, col_detail = st.columns([3, 2])
            
            with col_graph:
                fig, ax = plt.subplots(figsize=(10, 7))
                
                for i, (x, y) in enumerate(coords):
                    couleur = PALETTE[labels[i] % len(PALETTE)]
                    ax.scatter(x, y, c=couleur, s=100, alpha=0.7, edgecolors='white', linewidth=0.5)
                    ax.annotate(
                        titres[i].split()[0],
                        (x, y),
                        fontsize=7,
                        alpha=0.8
                    )
                
                legende = [
                    mpatches.Patch(color=PALETTE[i], label=f"Cluster {i}")
                    for i in range(n_clusters)
                ]
                ax.legend(handles=legende, loc='upper right')
                ax.set_title(methode_affichee, fontsize=14, fontweight='bold')
                ax.set_xlabel("Composante 1")
                ax.set_ylabel("Composante 2")
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
            
            with col_detail:
                st.markdown("### Répartition par cluster")
                
                for cluster_id in range(n_clusters):
                    articles_cluster = [titres[i] for i, l in enumerate(labels) if l == cluster_id]
                    themes_cluster = [themes[i] for i, l in enumerate(labels) if l == cluster_id]
                    
                    if themes_cluster:
                        theme_dominant = max(set(themes_cluster), key=themes_cluster.count)
                    else:
                        theme_dominant = "-"
                    
                    with st.expander(f"Cluster {cluster_id} ({len(articles_cluster)} articles) - {theme_dominant}"):
                        for titre in articles_cluster:
                            idx = titres.index(titre)
                            theme_reel = themes[idx]
                            emoji = "✅" if theme_reel == theme_dominant else "⚠️"
                            st.markdown(f"{emoji} {titre} *({theme_reel})*")


# ========================================================================
# TAB 5 - Évaluation
# ========================================================================
with tab5:
    st.markdown('<div class="step-header">Étape 5 - Évaluation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    L'évaluation repose sur deux métriques : le <strong>Silhouette Score</strong> (cohérence 
    géométrique interne, entre -1 et 1) et l'<strong>Adjusted Rand Index</strong> (comparaison 
    avec les thèmes réels Wikipedia servant de gold standard, entre 0 et 1).
    </div>
    """, unsafe_allow_html=True)
    
    if 'labels_tfidf' not in st.session_state:
        st.warning("Veuillez d'abord effectuer le clustering dans l'onglet précédent.")
    else:
        matrice_tfidf = st.session_state['matrice_tfidf']
        themes = st.session_state['themes']
        
        labels_dict = {
            "K-Means TF-IDF": st.session_state['labels_tfidf'],
            "K-Means Word2Vec": st.session_state['labels_w2v'],
            "SOM TF-IDF": st.session_state['labels_som']
        }
        
        # Calculer les scores
        resultats = calculer_scores(matrice_tfidf, labels_dict, themes)
        
        # Cartes métriques
        cols = st.columns(3)
        for i, (nom, scores) in enumerate(resultats.items()):
            with cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{nom}</h3>
                    <div class="value">{scores['ari']:.4f}</div>
                    <h3>Adjusted Rand Index</h3>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="metric-card" style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);">
                    <h3>{nom}</h3>
                    <div class="value">{scores['silhouette']:.4f}</div>
                    <h3>Silhouette Score</h3>
                </div>
                """, unsafe_allow_html=True)
        
        # Graphique comparatif (basé sur ton evaluation.py)
        st.markdown("### Comparaison des méthodes")
        
        methodes = list(resultats.keys())
        silhouette_scores = [resultats[m]['silhouette'] for m in methodes]
        ari_scores = [resultats[m]['ari'] for m in methodes]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(methodes))
        largeur = 0.35
        
        barres1 = ax.bar(x - largeur/2, silhouette_scores, largeur, 
                         label='Silhouette Score', color='steelblue')
        barres2 = ax.bar(x + largeur/2, ari_scores, largeur,
                         label='Adjusted Rand Index', color='darkorange')
        
        ax.set_ylabel('Score')
        ax.set_title('Comparaison des méthodes de clustering', fontsize=14, fontweight='bold')
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
        st.pyplot(fig)
        plt.close()
        
        # Conclusion automatique
        meilleur_ari = max(resultats, key=lambda m: resultats[m]['ari'])
        meilleur_sil = max(resultats, key=lambda m: resultats[m]['silhouette'])
        
        st.markdown("### Conclusion")
        st.success(f"""
        **Meilleure méthode (ARI)** : {meilleur_ari} ({resultats[meilleur_ari]['ari']:.4f})  
        **Meilleure méthode (Silhouette)** : {meilleur_sil} ({resultats[meilleur_sil]['silhouette']:.4f})
        """)
        
        # Analyse détaillée des articles mal classés
        st.markdown("### Analyse détaillée - Articles mal classés (K-Means TF-IDF)")
        
        labels_tfidf = st.session_state['labels_tfidf']
        titres = st.session_state['titres']
        
        for cluster_id in range(n_clusters):
            indices = [i for i, l in enumerate(labels_tfidf) if l == cluster_id]
            themes_cluster = [themes[i] for i in indices]
            
            if themes_cluster:
                theme_dominant = max(set(themes_cluster), key=themes_cluster.count)
                mal_classes = [titres[i] for i in indices if themes[i] != theme_dominant]
                
                if mal_classes:
                    st.markdown(f"**Cluster {cluster_id}** - thème dominant : *{theme_dominant}*")
                    st.markdown(f"   Articles hors thème : {', '.join(mal_classes)}")
                else:
                    st.markdown(f"**Cluster {cluster_id}** - thème dominant : *{theme_dominant}* Parfaitement homogène")