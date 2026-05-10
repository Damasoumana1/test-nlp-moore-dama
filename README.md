# Système de Traduction Automatique Moore → Français

Ce projet a été réalisé dans le cadre d'un test technique pour le poste de Spécialiste en Intelligence Artificielle. Il propose un modèle de traduction automatique (NMT) pour une langue à faibles ressources (le Moore) vers le Français.

## Structure du Projet

- **Google Colab** : [Accéder au Notebook en direct](https://drive.google.com/file/d/1dgU4B6I1DlYNWUwmfmUOUn1W8hbPDQZk/view?usp=sharing)
- **Modèle et Corpus (Google Drive)** : [Dossier complet sur Drive](https://drive.google.com/drive/folders/1ttcez3Wx6AtpzvCgv45DZEbBM8jXlhqW?usp=sharing)
- `traduction_moore_francais.ipynb` : Le notebook principal contenant tout le processus (exploration, préparation, entraînement, évaluation et test interactif).
- `corpus/` : Contient les fichiers sources `moore.txt` et `francais.txt`.
- `requirements.txt` : Liste des dépendances Python.
- `modele_traduction.keras` : Le modèle entraîné (optionnel, généré après exécution).

## Approche Technique

Le modèle est basé sur une architecture **Sequence-to-Sequence (Seq2Seq)** utilisant des réseaux de neurones récurrents de type **LSTM (Long Short-Term Memory)**.

### Justification du choix :
1. **Low-Resource Handling** : Pour un corpus de petite taille (env. 5330 paires), les modèles LSTM sont souvent plus stables et moins sujets au surapprentissage que les Transformers sans pré-entraînement massif.
2. **Interprétabilité** : L'architecture encodeur-décodeur permet de bien comprendre le flux d'information.
3. **Performance** : Avec une dimension d'embedding de 64 et 128 unités LSTM, le modèle capture efficacement les relations lexicales pour des phrases courtes.

## Instructions d'installation

1. Cloner le dépôt :
   ```bash
   git clone <url-du-depot>
   cd test-nlp-moore-dama
   ```
2. Installer les dépendances :
   ```bash
   pip install -r requirements.txt
   ```
3. Lancer le notebook :
   Ouvrez `traduction_moore_francais.ipynb` avec Jupyter Notebook ou Google Colab et exécutez toutes les cellules.

## Instructions Spécifiques pour Google Colab

Si vous utilisez Google Colab, suivez ces étapes pour garantir le bon chargement des données :

1. Importez le dossier `corpus` (contenant `moore.txt` et `francais.txt`) dans l'onglet **Fichiers** de la barre latérale gauche.
2. Assurez-vous que le dossier est nommé `corpus`. 
3. Le code utilise des chemins relatifs (`corpus/moore.txt`). Si vous placez le dossier ailleurs (par exemple dans `sample_data`), veuillez ajuster la cellule "Définition des chemins du corpus" en conséquence (ex: `/content/sample_data/corpus/moore.txt`).
4. Allez dans le menu **Exécution > Tout exécuter**.

## Évaluation

Le modèle est évalué via le score BLEU et une analyse qualitative d'exemples dans la section 5 du notebook. Une cellule interactive finale (Section 6) permet de tester des phrases personnalisées.

---
**Auteur** : Soumana Dama
**Date** : 10 Mai 2026
