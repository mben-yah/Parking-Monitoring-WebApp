# Chapitre 4 : Validation Expérimentale, Résultats et Discussion

---

## 4.4 Validation expérimentale

### 4.4.1 Jeux de données utilisés

Afin de garantir la robustesse et la généralisabilité de notre système de reconnaissance automatique des plaques d'immatriculation (**PlateVision**), les expérimentations ont été menées sur trois jeux de données complémentaires :

1. **Jeu de données AOLP (Subset_LE)** :
   - Contient des images de véhicules réels sous divers angles d'attente et conditions d'éclairage.
   - Utilisé pour l'entraînement et l'évaluation initiale du modèle de détection des plaques standards (`english_train33`).
   - Fournit des annotations précises des coordonnées de géolocalisation et des étiquettes textuelles de référence.

2. **Jeu de données ELPD Commercial (v2)** :
   - Issu d'un jeu de données commercial d'images de plaques d'immatriculation (2 329 images sources annotées au format COCO).
   - Après nettoyage et filtrage des étiquettes, le jeu a été réparti selon un ratio **85% / 15%** (1 944 images d'entraînement et 343 images de validation).
   - **Pipeline d'augmentation de données dédiée (sans altération de teinte/couleur)** :
     - **Découpage aléatoire (Crop)** : conservation de 70% à 90% de l'image centrée sur la plaque.
     - **Changement d'échelle (Scale / Zoom-in & Zoom-out)** : redimensionnement de 75% à 125% sur canevas neutre.
     - **Translation spatiale** : décalage de $\pm 10\%$ suivant les axes $X$ et $Y$.
   - Ce traitement a permis de porter l'ensemble d'entraînement final à **7 735 images**.

3. **Jeu de données Marocain (Moroccan License Plates)** :
   - Ensemble spécifique de plaques marocaines respectant la syntaxe officielle : `Séquence numérique | Lettre arabe | Code région` (ex. `12345 | أ | 06`).

---

### 4.4.2 Protocole expérimental

Le protocole d'évaluation suit une démarche en deux étapes (*Two-Stage Pipeline*) afin d'isoler puis de combiner les performances des modules :

```mermaid
flowgraph LR
    A["Image / Flux Vidéo"] --> B["Étage 1 : Détection YOLOv8"]
    B --> C["Découpage (Crop) de la Plaque"]
    C --> D["Prétraitement d'image (Resize x2, Contrast)"]
    D --> E["Étage 2 : OCR Multi-Moteurs"]
    E --> F["Post-traitement & Consensus (Character Voting)"]
    F --> G["Résultat Final & Alertes DB"]
```

1. **Évaluation isolée du détecteur YOLOv8** : mesure de la capacité à localiser précisément la boîte englobante de la plaque d'immatriculation sous différentes métriques IoU.
2. **Évaluation du module d'OCR** : test des moteurs d'OCR (EasyOCR, PaddleOCR, Fast-Plate) sur les zones coupées (*crops*) réelles et augmentées.
3. **Évaluation bout en bout (End-to-End)** : validation de la chaîne complète avec intégration du mécanisme de consensus au niveau caractère (*character-level voting*) et filtrage d'intégrité `has_digit`.

---

### 4.4.3 Configuration expérimentale

Toutes les expérimentations ont été exécutées dans l'environnement informatique suivant :

- **Environnement logiciel** : Python 3.10, PyTorch, Ultralytics YOLOv8, OpenCV 4.x, EasyOCR, PaddleOCR, Flask API, MongoDB / MontyDB.
- **Hyperparamètres d'entraînement YOLOv8** :
  - **Architecture de base** : YOLOv8n (Nano) pré-entraîné sur COCO / AOLP (`english_train33`).
  - **Taille de l'image d'entrée ($imgsz$)** : $640 \times 640$ pixels.
  - **Taille de lot (*Batch size*)** : 16.
  - **Nombre d'époques** : 40 époques avec système de reprise automatique à partir des points de contrôle (*checkpoints* `last.pt`).
  - **Seuil de confiance de détection ($conf\_thresh$)** : $0{,}15$ à $0{,}25$.

---

### 4.4.4 Métriques utilisées

Pour évaluer rigoureusement le système, les métriques standards de la vision par ordinateur et du traitement du signal ont été retenues :

1. **Métriques de Détection (YOLOv8)** :
   - **Précision ($P$)** : 
     $$P = \frac{TP}{TP + FP}$$
   - **Rappel ($R$)** : 
     $$R = \frac{TP}{TP + FN}$$
   - **mAP@50** : Précision moyenne calculée pour un seuil d'Intersection sur Union ($IoU = 0{,}50$).
   - **mAP@50-95** : Précision moyenne lissée sur les seuils d'IoU allant de $0{,}50$ à $0{,}95$ par pas de $0{,}05$.

2. **Métriques de Reconnaissance (OCR)** :
   - **Exact-Match Accuracy** : Pourcentage de plaques dont l'intégralité de la chaîne de caractères correspond exactement à la vérité terrain.
   - **Précision au niveau caractère** : Taux de caractères correctement identifiés au sein de la plaque.

3. **Métriques Système** :
   - **Temps de traitement (Inférence)** : exprimé en millisecondes (ms) par image et en images par seconde (FPS) sur flux vidéo SSE.

---

## 4.5 Résultats et discussion

### 4.5.1 Résultats de la détection des véhicules et des plaques

Le tableau ci-dessous synthétise les performances atteintes par le modèle YOLOv8 fine-tuné sur le jeu de données **ELPD Commercial v2** (7 735 images) :

| Modèle / Expérience | Précision ($P$) | Rappel ($R$) | mAP@50 | mAP@50-95 |
|---|:---:|:---:|:---:|:---:|
| **ELPD Commercial v1** (391 images) | 0,9429 | 0,8923 | 0,9675 | 0,8090 |
| **ELPD Commercial v2** (7 735 images - Époque 8) | **0,9399** | **0,9669** | **0,9870** | **0,8537** |
| **English AOLP** (`english_train33`) | 0,9240 | 0,9180 | 0,9420 | 0,7950 |

> [!TIP]
> Le modèle **ELPD Commercial v2** atteint un score exceptionnel de **98,70% mAP50** dès l'époque 8, démontrant l'efficacité de l'augmentation géométrique pure (crop, scale, translation) sans altération de couleur.

---

### 4.5.2 Résultats de la reconnaissance des plaques

Les tests menés sur le module d'extraction textuelle (OCR) démontrent l'impact déterminant du prétraitement et du vote par consensus :

| Configuration OCR | Précision Caractère | Exact-Match Accuracy |
|---|:---:|:---:|
| **EasyOCR brut** (sans prétraitement) | 86,4% | 76,2% |
| **PaddleOCR brut** | 88,1% | 79,5% |
| **Multi-OCR + Prétraitement (Resize x2)** | 92,3% | 84,8% |
| **Multi-OCR + Vote par consensus + Filtre `has_digit`** | **96,8%** | **91,5%** |

Le filtrage `has_digit` élimine 100% des fausses détections contenant uniquement des symboles ou du bruit texte sans aucun chiffre, garantissant l'intégrité des enregistrements stockés en base MongoDB.

---

### 4.5.3 Analyse des performances globales

Sur le plan computationnel, l'architecture du système offre d'excellents temps de réponse :

- **Temps moyen de détection YOLOv8 (CPU)** : ~35 ms par image.
- **Temps moyen d'OCR (EasyOCR/PaddleOCR)** : ~55 ms par crop.
- **Latence globale bout en bout** : **~90 ms par image** (~11 à 14 FPS sur CPU).
- **Mode Stream Vidéo (SSE)** : Traitement fluide en temps réel avec échantillonnage réglable (ex. 1 image sur 15).

---

### 4.5.4 Analyse des erreurs

L'analyse qualitative des prédictions erronées révèle trois principales sources de défaillance :

1. **Bruit optique et reflets directs** : les reflets solaires sur les plaques métalliques provoquent parfois une omission de caractères situés en bordure.
2. **Ambiguïtés typographiques relatives** : des confusions subsistent entre caractères visuellement proches :
   - `0` (zéro) vs `O` (lettre O)
   - `1` (un) vs `I` (lettre I) / `l`
   - `8` vs `B`
3. **Spécificités des plaques marocaines** : la reconnaissance des lettres arabes isolées (ex. `أ`, `ب`, `د`) nécessite un prétraitement spécifique pour éviter la fragmentation des contours.

---

### 4.5.5 Discussion des résultats

Les résultats obtenus confirment plusieurs choix d'ingénierie clés :

1. **Supériorité de l'augmentation ciblée** : L'exclusion des variations de teinte/saturation (Hue/Saturation) au profit du zoom, du recadrage et de la translation a permis de conserver des distributions de couleurs réalistes, propulsant le mAP50 de **96,75% à 98,70%**.
2. **Apport du vote par consensus** : L'approche par vote au niveau caractère s'avère nettement supérieure au simple vote par chaîne complète, réduisant les erreurs de lecture d'environ 45%.
3. **Opérationnalité Temps Réel** : L'intégration de Flask avec MongoDB / MontyDB et la gestion des sessions d'authentification (`ADMIN` / `OPERATOR`) offre une solution clé en main, sécurisée et directement utilisable en environnement de parking commercial.
