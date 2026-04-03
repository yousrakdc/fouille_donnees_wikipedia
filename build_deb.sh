#!/bin/bash
# ==========================================================
# Script de construction du paquet .deb pour WikiCluster
# Usage : ./build_deb.sh
# Produit : wikicluster_1.0.0_all.deb
# ==========================================================

set -e

VERSION="1.0.0"
PACKAGE="wikicluster"
BUILD_DIR="build_deb"

echo "=========================================="
echo "  Construction de ${PACKAGE}_${VERSION}_all.deb"
echo "=========================================="

# Nettoyage
rm -rf "$BUILD_DIR"

# -------------------------------------------------------
# 1. Création de l'arborescence du paquet
# -------------------------------------------------------
echo "[1/5] Création de l'arborescence..."

mkdir -p "$BUILD_DIR/DEBIAN"
mkdir -p "$BUILD_DIR/opt/wikicluster/resultats"
mkdir -p "$BUILD_DIR/usr/local/bin"
mkdir -p "$BUILD_DIR/usr/share/doc/wikicluster"
mkdir -p "$BUILD_DIR/usr/share/applications"

# -------------------------------------------------------
# 2. Copie des fichiers de contrôle Debian
# -------------------------------------------------------
echo "[2/5] Copie des fichiers de contrôle..."

cp debian/control  "$BUILD_DIR/DEBIAN/control"
cp debian/postinst "$BUILD_DIR/DEBIAN/postinst"
cp debian/prerm    "$BUILD_DIR/DEBIAN/prerm"
chmod 755 "$BUILD_DIR/DEBIAN/postinst"
chmod 755 "$BUILD_DIR/DEBIAN/prerm"

# -------------------------------------------------------
# 3. Copie des fichiers de l'application
# -------------------------------------------------------
echo "[3/5] Copie des fichiers de l'application..."

# === Scripts Python ===
cp app.py                "$BUILD_DIR/opt/wikicluster/"
cp collecte.py           "$BUILD_DIR/opt/wikicluster/"
cp pretraitement.py      "$BUILD_DIR/opt/wikicluster/"
cp vectorisation.py      "$BUILD_DIR/opt/wikicluster/"
cp clustering.py         "$BUILD_DIR/opt/wikicluster/"
cp evaluation.py         "$BUILD_DIR/opt/wikicluster/"

# === Fichiers optionnels (copiés s'ils existent) ===
[ -f lexique.py ] && cp lexique.py "$BUILD_DIR/opt/wikicluster/"
[ -f classification_mlp.py ] && cp classification_mlp.py "$BUILD_DIR/opt/wikicluster/"

# === Données pré-calculées ===
[ -f corpus.json ]         && cp corpus.json         "$BUILD_DIR/opt/wikicluster/"
[ -f corpus_propre.json ]  && cp corpus_propre.json  "$BUILD_DIR/opt/wikicluster/"
[ -f meta.json ]           && cp meta.json           "$BUILD_DIR/opt/wikicluster/"
[ -f matrice_tfidf.npy ]   && cp matrice_tfidf.npy   "$BUILD_DIR/opt/wikicluster/"
[ -f matrice_w2v.npy ]     && cp matrice_w2v.npy     "$BUILD_DIR/opt/wikicluster/"
[ -f modele_w2v.model ]    && cp modele_w2v.model    "$BUILD_DIR/opt/wikicluster/"
[ -f lexique_themes.json ] && cp lexique_themes.json "$BUILD_DIR/opt/wikicluster/"

# === Résultats (graphiques) ===
if [ -d resultats ]; then
    cp resultats/*.png "$BUILD_DIR/opt/wikicluster/resultats/" 2>/dev/null || true
    cp resultats/*.npy "$BUILD_DIR/opt/wikicluster/resultats/" 2>/dev/null || true
fi

# === Configuration ===
cp requirements.txt    "$BUILD_DIR/opt/wikicluster/"
cp setup.py            "$BUILD_DIR/opt/wikicluster/"

# === Lanceur ===
cp debian/wikicluster.sh "$BUILD_DIR/usr/local/bin/wikicluster"
chmod 755 "$BUILD_DIR/usr/local/bin/wikicluster"

# -------------------------------------------------------
# 4. Copie de la documentation
# -------------------------------------------------------
echo "[4/5] Copie de la documentation..."

cp README.md  "$BUILD_DIR/usr/share/doc/wikicluster/"
cp LICENSE    "$BUILD_DIR/usr/share/doc/wikicluster/copyright"

cp debian/changelog "$BUILD_DIR/usr/share/doc/wikicluster/changelog"
gzip -9 "$BUILD_DIR/usr/share/doc/wikicluster/changelog"

cat > "$BUILD_DIR/usr/share/applications/wikicluster.desktop" << 'EOF'
[Desktop Entry]
Name=WikiCluster
Comment=Clustering thématique d'articles Wikipedia
Exec=/usr/local/bin/wikicluster
Type=Application
Categories=Science;Education;
Terminal=true
EOF

# -------------------------------------------------------
# 5. Construction du .deb
# -------------------------------------------------------
echo "[5/5] Construction du paquet .deb..."

find "$BUILD_DIR" -type d -exec chmod 755 {} \;
find "$BUILD_DIR/opt" -type f -exec chmod 644 {} \;
chmod 755 "$BUILD_DIR/usr/local/bin/wikicluster"
chmod 755 "$BUILD_DIR/DEBIAN/postinst"
chmod 755 "$BUILD_DIR/DEBIAN/prerm"

dpkg-deb --build "$BUILD_DIR" "${PACKAGE}_${VERSION}_all.deb"

# -------------------------------------------------------
# Résultat
# -------------------------------------------------------
SIZE=$(du -h "${PACKAGE}_${VERSION}_all.deb" | cut -f1)

echo ""
echo "=========================================="
echo "  Paquet créé : ${PACKAGE}_${VERSION}_all.deb ($SIZE)"
echo "=========================================="
echo ""
echo "Pour installer :"
echo "  sudo dpkg -i ${PACKAGE}_${VERSION}_all.deb"
echo "  sudo apt-get install -f"
echo ""
echo "Pour lancer :"
echo "  wikicluster"
echo ""
echo "Pour désinstaller :"
echo "  sudo dpkg -r wikicluster"

# Nettoyage
rm -rf "$BUILD_DIR"

echo ""
echo "Construction terminée."