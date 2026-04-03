#!/bin/bash
# Lanceur WikiCluster
# Usage : wikicluster [--port PORT]

INSTALL_DIR="/opt/wikicluster"
VENV_DIR="$INSTALL_DIR/venv"
APP="$INSTALL_DIR/app.py"

# Vérification de l'installation
if [ ! -f "$APP" ]; then
    echo "Erreur : WikiCluster n'est pas correctement installé."
    echo "Essayez : sudo dpkg -i wikicluster_1.0.0_all.deb"
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Erreur : L'environnement virtuel n'existe pas."
    echo "Essayez : sudo dpkg --configure wikicluster"
    exit 1
fi

# Port par défaut
PORT=${1:-8501}

echo "=========================================="
echo "  WikiCluster v1.0.0"
echo "  http://localhost:$PORT"
echo "  Ctrl+C pour arrêter"
echo "=========================================="

# Lancement de Streamlit
"$VENV_DIR/bin/streamlit" run "$APP" \
    --server.port "$PORT" \
    --server.headless true \
    --browser.gatherUsageStats false