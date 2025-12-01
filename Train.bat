@echo off
TITLE Chess AI - Optimized Pipeline
COLOR 0B

echo ========================================================
echo   PIPELINE OPTIMISE : 1 PREPROCESS, 3 ENTRAINEMENTS
echo ========================================================
echo.

:: 1. PRETRAITEMENT UNIVERSEL
echo [ETAPE 1/5] Generation du Dataset Universel (Moves + Results)...
cd DeepLearning/src
:: On lance le script qui extrait TOUT (Input + Move + Result)
:: Assure-toi d'avoir renommé preprocess2.py en preprocess_universal.py
python3 preprocess2.py

if %ERRORLEVEL% NEQ 0 (
    echo [ERREUR] Le preprocess a echoue.
    pause
    exit /b
)
echo Dataset genere avec succes.
cd ../..
echo.

:: 2. ENTRAINEMENT CNN (Simple)
echo [ETAPE 2/5] Entrainement CNN (Utilise seulement les Coups)...
cd DeepLearning/src/CNN
python3 train_CNN.py
if %ERRORLEVEL% NEQ 0 echo [ATTENTION] Erreur CNN.
cd ../../..
echo.

:: 3. ENTRAINEMENT RESNET (Avance)
echo [ETAPE 3/5] Entrainement ResNet (Utilise seulement les Coups)...
cd DeepLearning/src/ResNet
:: Vérifie le nom du fichier python3 ici (ex: train_ResNet.py)
python3 train_ResNet.py
if %ERRORLEVEL% NEQ 0 echo [ATTENTION] Erreur ResNet.
cd ../../..
echo.

:: 3. ENTRAINEMENT SERESNET (Avance)
echo [ETAPE 4/5] Entrainement SEResNet (Utilise seulement les Coups)...
cd DeepLearning/src/ResNet
:: Vérifie le nom du fichier python3 ici (ex: train_ResNet.py)
python3 train_SEResNet.py
if %ERRORLEVEL% NEQ 0 echo [ATTENTION] Erreur SEResNet.
cd ../../..
echo.

:: 5. ENTRAINEMENT ALPHAZERO (Expert)
echo [ETAPE 5/5] Entrainement AlphaZero (Utilise Coups + Resultats)...
cd DeepLearning/src/AlphaZero
python3 train_AlphaZero.py
if %ERRORLEVEL% NEQ 0 echo [ATTENTION] Erreur AlphaZero.
cd ../../..
echo.

echo ========================================================
echo                 TRAVAIL TERMINE !
echo ========================================================
pause