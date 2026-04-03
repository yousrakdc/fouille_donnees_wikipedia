#!/bin/bash
# ==========================================================
# Script de construction du paquet source pour WikiCluster
# Usage : ./build_source.sh
# Produit : wikicluster_1.0.0_source.tar.gz
# ==========================================================

set -e

VERSION="1.0.0"
PACKAGE="wikicluster"
SOURCE_DIR="${PACKAGE}-${VERSION}"

echo "=========================================="
echo "  Construction du paquet source"
echo "=========================================="

# Nettoyage
rm -rf "$SOURCE_DIR"
rm -f "${PACKAGE}_${VERSION}_source.tar.gz"

# Création du dossier source
mkdir -p "$SOURCE_DIR"

# Copie des fichiers source
cp app.py              "$SOURCE_DIR/"
cp requirements.txt    "$SOURCE_DIR/"
cp setup.py            "$SOURCE_DIR/"
cp README.md           "$SOURCE_DIR/"
cp LICENSE             "$SOURCE_DIR/"
cp build_deb.sh        "$SOURCE_DIR/"

# Copie du dossier debian
cp -r debian           "$SOURCE_DIR/"

# Création de l'archive
tar -czf "${PACKAGE}_${VERSION}_source.tar.gz" "$SOURCE_DIR"

# Nettoyage
rm -rf "$SOURCE_DIR"

echo ""
echo "=========================================="
echo "  Paquet source créé :"
echo "  ${PACKAGE}_${VERSION}_source.tar.gz"
echo "=========================================="
echo ""
echo "Pour extraire et construire :"
echo "  tar -xzf ${PACKAGE}_${VERSION}_source.tar.gz"
echo "  cd ${PACKAGE}-${VERSION}"
echo "  chmod +x build_deb.sh"
echo "  ./build_deb.sh"