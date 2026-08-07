# One-to-One SkewGRAM : de la Biologie au Traitement du Langage Naturel (NLP)

**Institut Supérieur du Numérique (SupNum), Nouakchott, Mauritanie**  
**Équipe :** Bechir Mady, Abderahmane Abderrahmane, El Moustapha Mohamed El Moustapha, Mohamedou Yahya Cheikh Mohamed Vall  
**Superviseurs :** Dr. Mohamed Lemine Ahmed Sidi, Dr. Hafedh Mohamed Babou  

---

## 📌 À propos de ce projet

Ce dépôt contient le code et les données liés à notre article de recherche, intitulé **"One-to-one SkewGRAM: from biology to NLP"**, soumis à la conférence **I2COMSAPP'26** (Track 11: Other Relevant Topics in AI).

Le projet s'attaque à un problème d'optimisation combinatoire complexe issu de la **bio-informatique**, connu sous le nom de **problème One-to-One SkewGRAM**. Ce problème vise à identifier des signatures métaboliques et génomiques conservées entre différentes espèces. Pour le résoudre, nous proposons une approche novatrice qui transpose des techniques modernes issues du **Traitement du Langage Naturel (NLP)**, plus spécifiquement le modèle **Skip-Gram**, au domaine des graphes.

## 🧬 Le Problème (D,G)-consistant

L'objectif du problème One-to-One SkewGRAM est de trouver le plus long chemin qui respecte à la fois un ordre causal et une cohérence topologique à travers deux réseaux biologiques distincts mais partageant les mêmes éléments :

1. **Un graphe orienté acyclique $D = (V, A)$** : qui modélise des processus biologiques dirigés (par exemple, des voies métaboliques).
2. **Un graphe non-orienté $G = (V, E)$** : qui modélise des associations fonctionnelles ou physiques (par exemple, la proximité des gènes ou les interactions protéiques).

Un chemin $P = (v_1, v_2, \dots, v_k)$ dans $D$ est dit **(D,G)-consistant** si :
- C'est un chemin valide dans le graphe orienté $D$.
- Le sous-graphe induit par les sommets de ce chemin dans le graphe non-orienté $G$ est **connexe**.

## 🚀 Notre Contribution : Skew-GRAM

Des travaux antérieurs (notamment basés sur les Graph Neural Networks - GNN) ont montré des limites en termes de longueur de chemin trouvé ou de temps d'exécution. Nous améliorons l'approche GNN avec une méthode NLP beaucoup plus rapide.

Notre méthode, **Skew-GRAM**, repose sur :
- **L'Apprentissage de Représentations de Nœuds (Node Embeddings)** : En utilisant des marches aléatoires sur le graphe $D$ (à l'instar de DeepWalk/node2vec) traitées comme des "phrases", et les nœuds comme des "mots".
- **Un Échantillonnage Négatif Structurel** : Au lieu de tirer des exemples négatifs au hasard, nous ciblons les non-successeurs dans le graphe $D$ pour renforcer le respect de l'ordre topologique.
- **Un Décodage en Deux Étapes** : 
  1. Génération de chemins candidats via *beam search* guidé par la similarité des embeddings dans $D$.
  2. Vérification de la (D,G)-consistance et extraction du plus long sous-chemin valide.

## 📊 Résultats Clés

Notre méthode permet de trouver des chemins (D,G)-consistants valides de manière **extrêmement rapide**. Comparé à un solveur exact (Programmation Linéaire en Nombres Entiers - ILP2), notre heuristique est en moyenne **68 fois plus rapide**, ce qui ouvre la voie au passage à l'échelle sur de très grands réseaux biologiques, tout en surpassant la vitesse d'inférence et la complexité des modèles GNN existants.

## ⚙️ Reproductibilité

Le code complet, incluant l'entraînement du modèle Skew-GRAM et l'évaluation, est disponible dans le notebook Jupyter :
- `one_to_one_skewgram.ipynb`

### Organisation du dépôt
- `data/` : Contient les instances de graphes (fichiers `graphD.txt`, `graphG.txt` et solutions ILP2).
- `figures/` : Visualisations générées (embeddings PCA/t-SNE, topologie des graphes).
- `results/` : Tableaux de résultats bruts (.csv).
- `paper_en/` : Code source LaTeX de notre article (version anglaise) pour I2COMSAPP'26.
- `paper_fr/` : Code source LaTeX de notre article (version française).
- `Microsoft+Word+Proceedings+Templates/` : Template Springer fourni.

---
*Ce projet est réalisé dans le cadre de nos études à SupNum et de notre publication pour la conférence I2COMSAPP'26.*
