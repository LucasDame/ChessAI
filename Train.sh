#!/bin/bash

# Définition des couleurs (Similaire à COLOR 0B)
CYAN='\033[0;36m'
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${CYAN}========================================================${NC}"
echo -e "${CYAN}  PIPELINE OPTIMISE : 1 PREPROCESS, 3 ENTRAINEMENTS${NC}"
echo -e "${CYAN}========================================================${NC}"
echo ""

# Activation automatique de l'environnement virtuel si présent (optionnel mais recommandé)
if [ -d ".venv" ]; then
    echo "Activation de l'environnement virtuel..."
    source .venv/bin/activate
fi

# :: 1. PRETRAITEMENT UNIVERSEL
echo -e "${GREEN}[ETAPE 1/5] Generation du Dataset Universel (Moves + Results)...${NC}"
cd DeepLearning/src || { echo -e "${RED}Dossier introuvable${NC}"; exit 1; }

# On lance le script qui extrait TOUT
python3 preprocess_universal.py

# Vérification du code de retour ($?)
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERREUR] Le preprocess a echoue.${NC}"
    exit 1
fi

echo "Dataset genere avec succes."
cd ../..
echo ""

# :: 2. ENTRAINEMENT CNN (Simple)
echo -e "${GREEN}[ETAPE 2/5] Entrainement CNN (Utilise seulement les Coups)...${NC}"
cd DeepLearning/src/CNN || exit 1
python3 train_CNN.py
if [ $? -ne 0 ]; then echo -e "${RED}[ATTENTION] Erreur CNN.${NC}"; fi
cd ../../..
echo ""

# :: 3. ENTRAINEMENT RESNET (Avance)
echo -e "${GREEN}[ETAPE 3/5] Entrainement ResNet (Utilise seulement les Coups)...${NC}"
cd DeepLearning/src/ResNet || exit 1
python3 train_ResNet.py
if [ $? -ne 0 ]; then echo -e "${RED}[ATTENTION] Erreur ResNet.${NC}"; fi
cd ../../..
echo ""

# :: 4. ENTRAINEMENT SERESNET (Avance)
echo -e "${GREEN}[ETAPE 4/5] Entrainement SEResNet (Utilise seulement les Coups)...${NC}"
cd DeepLearning/src/ResNet || exit 1
# Assure-toi que train_SEResNet.py existe bien
python3 train_SEResNet.py
if [ $? -ne 0 ]; then echo -e "${RED}[ATTENTION] Erreur SEResNet.${NC}"; fi
cd ../../..
echo ""

# :: 5. ENTRAINEMENT ALPHAZERO (Expert)
echo -e "${GREEN}[ETAPE 5/5] Entrainement AlphaZero (Utilise Coups + Resultats)...${NC}"
cd DeepLearning/src/AlphaZero || exit 1
python3 train_AlphaZero.py
if [ $? -ne 0 ]; then echo -e "${RED}[ATTENTION] Erreur AlphaZero.${NC}"; fi
cd ../../..
echo ""

echo -e "${CYAN}========================================================${NC}"
echo -e "${CYAN}                 TRAVAIL TERMINE !${NC}"
echo -e "${CYAN}========================================================${NC}"

# Pause (read attend une entrée utilisateur)
read -p "Appuyez sur Entree pour quitter..."