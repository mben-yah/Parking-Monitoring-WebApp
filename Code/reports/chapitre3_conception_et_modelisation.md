# Chapitre 3 : Approches Algorithmiques, Modèles d'IA et Optimisation

---

## 3.1 Approches algorithmiques proposées

Le système **PlateVision** repose sur un traitement modulaire en cascade à deux étages (*Two-Stage Pipeline*). L’architecture globale sépare la localisation spatiale de la plaque d'immatriculation de l'extraction textuelle des caractères, tout en intégrant des modules avancés de validation et de gestion des flux en temps réel.

```mermaid
flowchart TD
    subgraph S1["1. Acquisition & Prétraitement"]
        A["Flux Vidéo / IP Webcam / Image"] --> B["Découpage en Trames (Frame Extractor)"]
    end
    
    subgraph S2["2. Détection par IA (Étage 1)"]
        B --> C["YOLOv8n Fine-tuné"]
        C -->|Boîte Englobante [x1,y1,x2,y2]| D["Extraction du Crop de Plaque"]
    end

    subgraph S3["3. Prétraitement & OCR (Étage 2)"]
        D --> E["Amélioration d'Image (Resize x2, Contrast)"]
        E --> F["Moteurs OCR (EasyOCR / PaddleOCR)"]
    end

    subgraph S4["4. Post-Traitement & Métier"]
        F --> G["Filtrage has_digit & Normalisation"]
        G --> H["Vote par Consensus (Character-Level)"]
        H --> I["Contrôle d'Accès (Whitelist / Blacklist)"]
        I --> J["Sauvegarde MongoDB & SSE Streaming"]
    end
```

---

### 3.1.1 Détection des véhicules et des plaques

La détection constitue la première étape critique du système. Plutôt que de traiter l'intégralité de l'image pour la reconnaissance textuelle (ce qui serait coûteux et sujet au bruit), l'approche retenue s'appuie sur le réseau de neurones convolutif **YOLOv8** :

1. **Entrée** : Image brute $I \in \mathbb{R}^{H \times W \times 3}$.
2. **Inférence** : Le modèle prédit les coordonnées de la plaque sous la forme $(x_{center}, y_{center}, w, h, conf)$.
3. **Extraction de la région d'intérêt (RoI)** : Transformation des coordonnées en pixels absolus $[x_1, y_1, x_2, y_2]$ et découpage (*crop*) de la plaque :
   $$I_{crop} = I[y_1:y_2, \; x_1:x_2]$$

---

### 3.1.2 Reconnaissance optique des caractères (OCR)

Une fois la plaque découpée, elle est transmise au module d'OCR. Afin de surmonter les défaillances individuelles des bibliothèques OCR standards sur des images dégradées, une approche **multi-moteurs** est déployée :

- **Moteur 1 — EasyOCR** : Spécialisé dans les caractères alphanumériques latins et arabes, utilisant un réseau de détection CRAFT et un réseau de reconnaissance CRNN.
- **Moteur 2 — PaddleOCR / Fast-Plate** : Moteur ultraléger basé sur PP-OCRv3 pour l'extraction rapide des séquences numériques et des lettres arabes.

---

### 3.1.3 Validation des plaques d'immatriculation

Pour éliminer les fausses détections (panneaux de signalisation, logos, publicités), un algorithme strict de validation et de consensus est appliqué :

1. **Filtrage d'intégrité `has_digit`** : Toute prédiction textuelle ne contenant **aucun chiffre** est automatiquement rejetée.
2. **Normalisation de la syntaxe** : Suppression des espaces parasites, tirets et symboles spéciaux via expressions régulières (Regex) :
   $$\text{Plate}_{\text{norm}} = \text{RegexReplace}(T_{\text{raw}}, \; \text{"[\^A-Z0-9\u0600-\u06FF]"})$$
3. **Vote par consensus au niveau caractère (*Character-Level Voting*)** : Lorsque plusieurs prédictions sont générées pour une même plaque au fil des trames, le système effectue un vote majoritaire position par position plutôt qu'un vote sur la chaîne entière, maximisant l'exactitude de la séquence finale.

---

### 3.1.4 Gestion asynchrone des flux vidéo

Pour garantir une expérience utilisateur fluide sans blocage de l'interface graphique :

- **Server-Sent Events (SSE)** : Les détections vidéo sont transmises au client sous forme d'événements SSE asynchrones (`/predict_video`), incluant la trame courante, le pourcentage de progression, le taux de confiance et l'image annotée encodée en Base64.
- **Échantillonnage adaptatif (*Frame-Skip*)** : Analyse d'une image toutes les $N$ trames (ex. $N=15$), permettant de maintenir un traitement temps réel fluide tout en réduisant la charge CPU.

---

### 3.1.5 Logique métier et contrôles d'accès

La logique métier supervise l'état du parking et déclenche des actions automatisées :

- **Événements d'accès** : Enregistrement automatique du type d'événement (`ENTRY` / `EXIT`), de l'horodatage UTC et de la durée de stationnement.
- **Vérification d'autorisation** :
  - **Liste Blanche (*Whitelist*)** : Réservée aux véhicules autorisés.
  - **Liste Noire (*Blacklist*)** : Véhicules bannis ou surveillés.
- **Alertes d'intrusion** : Si un véhicule non autorisé est détecté, une alerte temps réel (`create_alert`) est générée dans MongoDB et notifiée au panneau d'administration.

---

## 3.2 Modèles d'intelligence artificielle

### 3.2.1 YOLOv8 pour la détection des véhicules et des plaques

**YOLOv8** (*You Only Look Once v8*), développé par Ultralytics, a été sélectionné pour sa rapidité d'inférence et sa précision élevées.

- **Backbone** : Extracteur de caractéristiques basé sur CSPDarknet modifié avec des convolutions C2f.
- **Neck** : Structure PAN-FPN (*Path Aggregation Network*) permettant de capturer des détails multi-échelles (essentiel pour les petites plaques d'immatriculation).
- **Head** : Tête d'ancrage libre (*Anchor-free*) prédisant directement le centre et les dimensions de la boîte englobante avec la fonction de perte *CIoU* (*Complete Intersection over Union*).

---

### 3.2.2 EasyOCR pour la reconnaissance des plaques

**EasyOCR** combine deux architectures complémentaires :

1. **Détecteur de texte CRAFT** (*Character Region Awareness for Text Detection*) : Localise précisément les zones de caractères dans le crop de la plaque.
2. **Reconnaisseur CRNN** (*Convolutional Recurrent Neural Network*) :
   - Couches convolutives (ResNet) pour la cartographie des caractéristiques.
   - Couches récurrentes (LSTM bidirectionnel) pour modéliser la dépendance séquentielle entre les caractères.
   - Décodage CTC (*Connectionist Temporal Classification*) pour aligner la séquence prédite.

---

### 3.2.3 Prétraitement et post-traitement des images

Afin de maximiser la lisibilité des plaques avant passage à l'OCR, la chaîne de prétraitement suivante est exécutée :

```
Crop Brut ──> Redimensionnement x2 (Bicubic) ──> Conversion BGR -> RGB ──> Égalisation CLAHE ──> Inférence OCR
```

1. **Super-résolution / Sur-échantillonnage** : Agrandissement de la région découpée d'un facteur 2x pour clarifier les bords des lettres.
2. **Égalisation d'histogramme adaptative (CLAHE)** : Rehaussement du contraste local dans les zones d'ombre ou de fort éblouissement.
3. **Post-traitement regex & formateur marocain** : Alignement de la séquence reconnue selon les normes régionales (`Séquence | Lettre | Région`).

---

## 3.3 Optimisation des modèles

### 3.3.1 Fine-tuning des modèles

Le transfert d'apprentissage (*Transfer Learning*) a été appliqué sur l'architecture YOLOv8n pré-entraînée sur COCO :

- **Étape 1** : Entraînement initial sur le jeu de données d'orientation anglaise AOLP (`english_train33`).
- **Étape 2** : Fine-tuning spécialisé sur le dataset commercial ELPD v2 pendant 40 époques avec un taux d'apprentissage adaptatif (*AdamW optimizer*) et un système d'arrêt précoce (*Patience = 15*).

---

### 3.3.2 Augmentation des données

Afin d'éviter le surapprentissage (*overfitting*) et d'améliorer la généralisation sans altérer les couleurs naturelles des véhicules, une stratégie d'**augmentation géométrique pure** a été conçue :

| Technique d'augmentation | Paramètres appliqués | Objectif |
|---|---|---|
| **Découpage aléatoire (Crop)** | 70% à 90% de la zone centrée | Simuler des angles de caméra serrés et des occlusions partielles |
| **Changement d'échelle (Scale)** | 75% à 125% sur canevas gris neutre | Ajuster la sensibilité du modèle à la distance de la caméra (Zoom in/out) |
| **Translation spatiale** | Décalage de $\pm 10\%$ en X et Y | Simuler les décentrements du véhicule dans la voie d'accès |

> **Résultat** : Augmentation du jeu d'entraînement d'origine de 1 944 à **7 735 images augmentées**.

---

### 3.3.3 Optimisation des performances système

1. **Persistance hybride MongoDB / MontyDB** : Basculement automatique vers la base de données de fichiers MontyDB en l'absence de serveur MongoDB externe, garantissant 100% de disponibilité locale.
2. **Mise en cache du modèle** : Chargement paresseux (*Lazy Loading*) et réutilisation des instances de modèles YOLOv8 en mémoire pour éviter le surcoût d'initialisation à chaque requête HTTP.
3. **Flux SSE non-bloquant** : Utilisation des générateurs Python et de `stream_with_context` sous Flask pour fournir un rendu temps réel continu sans bloquer la boucle d'événements du serveur.

---

## 3.4 Critères et métriques d'évaluation

### 3.4.1 Métriques de détection

- **Précision ($P$)** : Proportion de plaques correctement détectées parmi toutes les détections renvoyées.
- **Rappel ($R$)** : Proportion de plaques réelles détectées par le système.
- **mAP@50** : Précision moyenne calculée à un seuil d'IoU de 0,50 :
  $$\text{IoU} = \frac{\text{Aire de l'intersection}}{\text{Aire de l'union}}$$

---

### 3.4.2 Métriques de reconnaissance OCR

- **Exact-Match Accuracy** : 
  $$\text{Accuracy}_{\text{Exact}} = \frac{N_{\text{plaques parfaitement lues}}}{N_{\text{total plaques}}} \times 100\%$$
- **Précision au niveau caractère** : Taux d'exactitude des caractères individuels décodés.

---

### 3.4.3 Métriques de performance du système

- **Temps de traitement (Inférence ms)** : Durée nécessaire pour traiter une image complète (Détection + Prétraitement + OCR).
- **Taux de trames par seconde (FPS)** : Vitesse globale d'exécution du pipeline sur flux vidéo.
- **Disponibilité des capteurs (*Sensor Uptime*)** : Suivi du taux de fonctionnement des caméras configurées sur le panneau d'administration.
