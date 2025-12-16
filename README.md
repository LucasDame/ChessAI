# ♟️ Projet de Moteur d'Échecs Haute Performance

> **Moteur C ultra-rapide · UI Python/Pygame · Deep Learning PyTorch · Pipeline automatisé**

Ce projet implémente un **moteur d'échecs moderne et modulaire**, combinant :

* un **cœur de calcul en C** (bitboards, Negamax, Alpha-Beta),
* une **interface graphique Python** (Pygame),
* plusieurs **architectures d'IA Deep Learning** (CNN, ResNet, SE‑ResNet),
* des approches avancées (**AlphaZero**, **neuro‑évolution génétique**).

---

## 📚 Sommaire

1. [Moteur de Jeu Haute Performance (C)](#-moteur-de-jeu-haute-performance-c-engine)
2. [Interface Graphique (Python / Pygame)](#-interface-graphique-ui)
3. [Pipeline d'Entraînement Automatisé](#-pipeline-dentraînement-automatisé-deep-learning)
4. [Instructions Linux & macOS](#-instructions-spécifiques-linux--macos)
5. [Configuration Git & LFS](#️-configuration-git--large-file-storage-lfs)
6. [Intelligence Artificielle (Deep Learning)](#-intelligence-artificielle-deep-learning)
7. [Modèles Avancés](#-modèles-avancés-renforcement--évolution)

---

## ⚡ Moteur de Jeu Haute Performance (C Engine)

Le cœur de la logique du jeu d'échecs est implémenté en **C pur**, afin de garantir des performances maximales pour :

* la génération de coups,
* la validation des règles,
* l'exploration de l'arbre de recherche.

Le moteur fonctionne comme un **serveur TCP autonome**, complètement découplé de l'interface graphique.

### 🏗️ Architecture du Code C

| Fichier    | Rôle                                                 |
| ---------- | ---------------------------------------------------- |
| `main.c`   | Serveur TCP, API, boucle principale                  |
| `board.c`  | Représentation du plateau (Bitboards), sérialisation |
| `move.c`   | Génération des coups, règles, fins de partie         |
| `search.c` | IA classique (Negamax + Alpha-Beta)                  |
| `defs.h`   | Structures globales, macros, types                   |

**Points clés :**

* Représentation par **Bitboards (uint64)**
* Règles spéciales : roque, en passant, promotion
* Détection : mat, pat, répétition

---

### 🚀 Performances & Optimisations

* ⚙️ **Bitboards** : opérations bit-à-bit ultra‑rapides
* 🔌 **Serveur TCP** : UI non bloquante
* 🚀 **Compilation `-O3`** via GCC

---

### 🛠️ Compilation (Build)

**Prérequis :** `gcc`, `make`

```bash
make
```

L'exécutable généré est : **`API_negamax`**

> 🔁 Après modification du code C :

```bash
make clean
make
```

---

### 🔌 Protocole de Communication TCP

* **Port :** `12345`

#### 📤 Requête (Client → Serveur)

```text
e2e4
```

#### 📥 Réponse (Serveur → Client)

```text
board:rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR
```

Ou messages d'erreur :

* `illegal_move_rules`
* `illegal_move_king_check`
* `game_over:checkmate`

---

## 🎨 Interface Graphique (UI)

L'interface est développée en **Python** avec **Pygame**.

### Fonctionnalités

* 🎥 Affichage fluide du plateau
* 🔌 Client TCP du moteur C
* 🎮 Modes de jeu :

  * Humain vs IA (C / Minimax)
  * Humain vs Deep Learning
  * IA vs IA (spectateur)

### ▶️ Lancement

```bash
cd DeepLearning/src
python UI.py
```

Le moteur C est lancé automatiquement en arrière‑plan.

---

## 🤖 Pipeline d'Entraînement Automatisé (Deep Learning)

Le projet inclut un **pipeline entièrement automatisé** pour entraîner plusieurs modèles PyTorch.

### 🔄 Étapes du Pipeline

1. **Preprocessing** (`preprocess2.py`)

   * Lecture des fichiers `.pgn`
   * Conversion Bitboards → tenseurs
   * Génération des datasets `.pt`

2. **Entraînement CNN**

3. **Entraînement ResNet**

4. **Entraînement SE‑ResNet**

---

### 🚀 Lancer l'Entraînement

#### 🖥️ Windows

```cmd
Train.bat
```

#### 🐧 Linux / macOS

```bash
chmod +x Train.sh
./Train.sh
```

---

## ⚙️ Configuration Git & Large File Storage (LFS)

### 🚫 Fichiers Ignorés

* Binaires C (`API_negamax`, `*.o`)
* Environnements Python (`.venv`, `__pycache__`)
* Fichiers système

### 📦 Git LFS

Extensions suivies :

* `*.pt`, `*.pth` (modèles)
* `*.pgn` (datasets)

```bash
git lfs install
git lfs pull
```

---

## 🧠 Intelligence Artificielle (Deep Learning)

Les modèles sont implémentés avec **PyTorch**.

### 📂 Architectures Disponibles

#### 1️⃣ CNN

* Léger, rapide
* Idéal pour tests ou machines modestes

#### 2️⃣ ResNet

* Connexions résiduelles
* Excellente compréhension stratégique

#### 3️⃣ SE‑ResNet

* Attention par canaux (*Squeeze‑and‑Excitation*)
* Précision positionnelle accrue

---

### 📊 Traitement des Données

* Entrée : fichiers `.pgn`
* Sortie : tenseurs `C×8×8` (ex : `14×8×8`)
* Paires `(Position, Coup)` ou `(Position, Résultat)`

---

### 📂 Structure du Dossier DeepLearning

```text
DeepLearning/
├── src/
│   ├── CNN/
│   ├── ResNet/
│   ├── Genetic/
│   ├── AlphaZero/
│   ├── dataset.py
│   ├── preprocess2.py
│   └── UI.py
```

---

## 🧬 Modèles Avancés (Renforcement & Évolution)

### ♟️ AlphaZero

* Self‑Play + MCTS
* Réseau Policy + Value
* Apprentissage sans connaissance humaine

### 🧬 Genetic TinyNet

* Neuro‑évolution
* Sélection naturelle
* Confrontation contre Stockfish

---

## 📈 Comparatif des Modèles

| Modèle    | Apprentissage | Force    | Vitesse | Usage         |
| --------- | ------------- | -------- | ------- | ------------- |
| CNN       | Supervisé     | ⭐⭐       | ⭐⭐⭐⭐⭐   | Tests rapides |
| ResNet    | Supervisé     | ⭐⭐⭐⭐     | ⭐⭐⭐     | Standard      |
| SE‑ResNet | Supervisé     | ⭐⭐⭐⭐⭐    | ⭐⭐      | Précision     |
| AlphaZero | RL            | ♾️       | ⭐       | Recherche     |
| Genetic   | Évolution     | Variable | ⭐⭐⭐⭐    | Exploration   |

---

✨ *Projet conçu pour l'expérimentation, la performance et la recherche en IA appliquée aux échecs.*
