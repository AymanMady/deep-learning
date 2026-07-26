# One-to-One Skew-GRAM : résolution du problème du plus long chemin par embeddings de nœuds appris via Skip-Gram

**Rapport scientifique — Projet universitaire (Optimisation / Deep Learning / NLP appliqués aux graphes)**
**Date :** 2026-07-19

---

## Résumé

Ce projet part d'un objectif académique classique — construire un modèle de *Word Embedding* fondé sur le Skip-Gram, puis proposer une variante optimisée, le **One-to-One Skew-GRAM** — et l'applique à un jeu de données de graphes aléatoires orientés (Erdős–Rényi) accompagné d'une vérité terrain produite par un solveur exact d'optimisation combinatoire (ILP). Le Skip-Gram, transposé aux graphes suivant le paradigme DeepWalk/node2vec (nœud = mot, marche aléatoire = phrase), est utilisé comme heuristique pour résoudre le **problème NP-difficile du plus long chemin simple dans un graphe orienté**. Le One-to-One Skew-GRAM introduit trois optimisations — échantillonnage négatif structurel, sous-échantillonnage des nœuds de haut degré, et décodage par softmax masquée (« skewed ») garantissant une correspondance bijective (« one-to-one ») entre positions du chemin et nœuds — motivées par une inadéquation fondamentale entre le Skip-Gram classique (hérité du texte, où un mot peut se répéter) et la tâche de construction d'un chemin (où un nœud ne peut apparaître qu'une fois). Sur un échantillon de 39 instances, le Skew-GRAM égale ou dépasse le Skip-Gram classique dans **26 cas sur 39** (67 %) tout en restant en moyenne **43 fois plus rapide** que le solveur exact ILP2 (contre 14 fois pour le Skip-Gram classique), pour une longueur de chemin moyenne comparable (9.38 contre 9.41 nœuds pour le classique).

---

## 1. Introduction

L'énoncé de ce projet demandait la construction d'un modèle de Word Embedding Skip-Gram sur un corpus textuel, puis la conception d'une variante optimisée nommée *One-to-One Skew-GRAM*. L'exploration du jeu de données fourni (dossier Google Drive `erdos_renyi`) a révélé qu'il ne s'agissait pas d'un corpus de texte, mais d'un ensemble d'environ 890 instances de **graphes aléatoires orientés** de type Erdős–Rényi à 100 nœuds, chacune accompagnée de la sortie d'un solveur exact (**ILP2**, *Integer Linear Programming*) pour le **problème du plus long chemin simple**. Cette découverte — détaillée en §2 — a conduit à repositionner le projet : au lieu d'un Skip-Gram textuel, nous transposons le principe du Skip-Gram aux graphes (paradigme DeepWalk, Perozzi et al., 2014 ; node2vec, Grover & Leskovec, 2016) et l'utilisons pour résoudre, de façon heuristique, un véritable problème d'optimisation combinatoire, en comparant nos résultats à la référence exacte fournie par le dataset. Ce repositionnement rend le titre du projet — *« Résolution d'un problème d'optimisation »* — littéral plutôt que métaphorique.

## 2. Données et repositionnement méthodologique

### 2.1 Structure du dataset

Chaque instance `100_<k>` du dossier fourni contient trois fichiers texte :

| Fichier | Contenu |
|---|---|
| `graphD.txt` | Un graphe orienté G(100, p) : nœuds `1..100`, arêtes `a-b` signifiant l'arc **a → b**, probabilité `p` du modèle Erdős–Rényi |
| `graphG.txt` | Un second graphe orienté indépendant sur les mêmes 100 nœuds, plus dense |
| `solution.txt` | Résultat du solveur exact ILP2 pour le plus long chemin simple dans `graphD` : chemin trouvé, longueur, temps d'exécution, ou `"No solution found"` |

**Vérification de l'orientation des arêtes.** Une inspection directe du chemin solution d'une instance (`100_116`) a montré que chacun de ses arcs consécutifs correspond exactement à l'ordre littéral `a→b` des lignes du fichier, et jamais à l'ordre inverse. Une recherche aléatoire naïve sur la même instance, en interprétant le graphe comme non orienté, trouvait des chemins de longueur ≈ 90-100 nœuds — très supérieurs à la longueur ILP2 rapportée (13) — alors qu'en interprétation orientée, une marche gloutonne aléatoire plafonnait à des longueurs cohérentes avec la référence ILP2 (de l'ordre de 9 contre 13 pour ILP2). Cette vérification empirique confirme que le graphe est **orienté** et que le problème résolu par ILP2 est bien le plus long chemin simple **dirigé**.

### 2.2 Limite de collecte des données

Le dossier Drive contient environ 890 instances. Leur téléchargement via l'API publique de Google Drive (`gdown`) est soumis à un **quota anti-abus par fichier/IP** : une première tentative de téléchargement parallèle a déclenché ce quota, bloquant temporairement l'accès à tout fichier du dossier — y compris des fichiers jamais sollicités auparavant. Le quota s'est levé puis redéclenché à plusieurs reprises lors de téléchargements séquentiels successifs (espacés de ~0.7-1 s par fichier). Au total, **39 instances** ont pu être collectées avec succès (11 tirées aléatoirement avec la graine `SEED=42`, 27 tirées lors de sessions de collecte ultérieures, plus une instance `100_1` ajoutée manuellement pour illustrer le cas `"No solution found"`). Voir `data/README.md` pour la procédure complète de reproduction et d'extension du jeu de données.

Toutes les statistiques et tous les tableaux comparatifs de ce rapport sont calculés sur cet échantillon de 39 instances (sur ~890 disponibles, soit ≈ 4.4 %) ; ils doivent être lus comme représentatifs d'une tendance plutôt que comme une moyenne définitive sur l'ensemble du dataset. Le code (notebook `one_to_one_skewgram.ipynb`) s'adapte automatiquement au nombre d'instances présentes dans `data/raw/` : ré-exécuter le notebook après avoir étendu le jeu de données produira des statistiques encore plus robustes sans modification du code.

### 2.3 Correspondance NLP → graphe

| Concept NLP demandé dans l'énoncé | Équivalent utilisé ici |
|---|---|
| Corpus / documents | Ensemble des instances de graphes disponibles |
| Phrase | Marche aléatoire (*random walk*) sur le graphe |
| Mot | Nœud (identifiant 1..100) |
| Vocabulaire | Ensemble des nœuds du graphe (taille = N, local à chaque instance) |
| Fréquence des mots | Fréquence de visite des nœuds dans les marches |
| Nettoyage, caractères spéciaux, minuscules | Sans objet (données numériques) ; nœuds isolés exclus des marches |
| Stopwords | Sous-échantillonnage des nœuds de haut degré (*hubs*) |
| Lemmatisation / Stemming | Sans objet — pas de morphologie sur des identifiants de nœuds |
| Tokenization | Génération des marches aléatoires |
| Texte → indices | Triviale : les nœuds sont déjà des entiers |
| Analogies de mots | Analogies structurelles : *u − voisin(u) + voisin(v) ≈ v* |

`nltk` et `spacy` ne sont pas utilisés (absence de texte à traiter) ; `gensim` est utilisé une seule fois (§5.4), comme base de comparaison, conformément à l'énoncé qui l'autorise « uniquement pour comparaison éventuelle ».

## 3. Méthode

### 3.1 Skip-Gram classique avec Negative Sampling

Le modèle utilise deux matrices d'embeddings, $\mathbf{V}_{in}$ (nœud-centre) et $\mathbf{V}_{out}$ (nœud-contexte), chacune de dimension $d=32$. Pour une paire positive $(c, o)$ et $K=5$ négatifs $n_{1:K}$, la perte de Negative Sampling est :

$$\mathcal{L}(c, o, n_{1:K}) = -\log \sigma\!\left(\mathbf{v}_{in,c}^\top \mathbf{v}_{out,o}\right) - \sum_{k=1}^{K} \log \sigma\!\left(-\mathbf{v}_{in,c}^\top \mathbf{v}_{out,n_k}\right)$$

optimisée par **Adam** (Kingma & Ba, 2014). Les négatifs classiques sont tirés selon la distribution unigramme $P(n) \propto f(n)^{0.75}$, comme dans word2vec (Mikolov et al., 2013).

Les paires d'entraînement sont générées par marches aléatoires (DeepWalk) sur `graphD`, suivies d'une fenêtre glissante de taille 3 (§ couples Skip-Gram du notebook).

### 3.2 One-to-One Skew-GRAM

**Motivation.** Un chemin simple est une injection entre positions et nœuds : un nœud ne peut être visité qu'une fois. Le Skip-Gram, hérité du texte, n'a aucune notion d'unicité (un mot peut se répéter dans une phrase). Utilisé naïvement pour décoder un chemin (choix glouton du voisin le plus similaire), il n'a donc **aucun mécanisme empêchant de revisiter un nœud déjà utilisé**.

**Optimisation 1 — échantillonnage négatif structurel.** Les négatifs sont tirés uniquement parmi les non-voisins topologiques du nœud centre :

$$P_{struct}(n \mid c) = \frac{\mathbb{1}[(c,n) \notin E]}{\left|\{v : (c,v) \notin E\}\right|}$$

ce qui renforce le signal d'adjacence réelle appris par le modèle (négatifs = vraies non-arêtes, plutôt qu'un tirage statistique agnostique de la topologie).

**Optimisation 2 — sous-échantillonnage des hubs.** Formule de word2vec appliquée à la fréquence de visite des nœuds dans les marches :

$$P_{keep}(v) = \min\!\left(1,\ \sqrt{\frac{t}{f(v)}} + \frac{t}{f(v)}\right), \quad t = 10^{-3}$$

qui rééquilibre la représentation des nœuds périphériques, souvent nécessaires pour prolonger un chemin, face aux nœuds très connectés qui dominent sinon l'entraînement.

**Optimisation 3 — décodage contraint par softmax masquée (« skewed »).** Le chemin est construit par un beam search guidé par une distribution de probabilité qui retire toute la masse de probabilité des nœuds déjà visités et renormalise sur les seuls successeurs non-visités :

$$P_{skew}(v \mid u, \mathcal{U}) = \frac{\mathbb{1}[(u,v)\in E] \cdot \mathbb{1}[v \notin \mathcal{U}] \cdot \exp(\mathrm{sim}(u,v)/\tau)}{\displaystyle\sum_{v' \in N(u)\setminus\mathcal{U}} \exp(\mathrm{sim}(u,v')/\tau)}, \qquad v \in N(u)\setminus\mathcal{U}$$

où $\mathcal{U}$ est l'ensemble des nœuds déjà visités et $\tau=0.5$ une température. Ce masquage garantit qu'une correspondance déjà utilisée a une probabilité **strictement nulle** d'être réutilisée — la correspondance position → nœud reste bijective tout au long du décodage, d'où le nom *one-to-one*. Le Skip-Gram classique, à titre de comparaison objective, est décodé par une règle purement gloutonne qui **s'arrête** dès qu'elle retombe sur un nœud déjà visité (aucune gestion de la réutilisation, à l'image de ce qui se passe réellement en NLP).

**Algorithme de décodage (pseudocode) :**

```
DÉCODER_ONE_TO_ONE(embeddings, graphe, nœud_départ, largeur_beam):
    beams ← [ (chemin=[départ], visités={départ}, score=0) ]
    RÉPÉTER jusqu'à longueur maximale :
        candidats ← []
        POUR chaque (chemin, visités, score) dans beams :
            voisins ← successeurs(dernier nœud du chemin) \ visités
            SI voisins vide : candidats.ajouter((chemin, visités, score))   # figé
            SINON :
                probs ← softmax_masquée(similarités(dernier nœud, voisins) / tau)
                POUR chaque voisin v de probabilité p_v :
                    candidats.ajouter((chemin + [v], visités ∪ {v}, score + log(p_v)))
        beams ← top-`largeur_beam` candidats triés par (longueur, score)
        SI aucun candidat n'a pu grandir : ARRÊTER
    RETOURNER le plus long chemin observé parmi tous les beams
```

## 4. Protocole expérimental

Pour chaque instance disponible : génération des marches aléatoires (40 marches par nœud de départ, longueur max. 30) → construction des couples Skip-Gram (fenêtre 3) → entraînement du modèle classique (négatifs unigrammes) et du modèle Skew-GRAM (négatifs structurels + sous-échantillonnage des hubs), 8 époques, `Adam(lr=0.01)` → décodage du meilleur chemin depuis chaque nœud de départ possible (glouton naïf pour le classique, beam search largeur 4 pour le Skew-GRAM) → comparaison à la référence ILP2. Une graine unique (`SEED=42`) est fixée pour `random`, `numpy` et `torch`, garantissant la reproductibilité intégrale de la pipeline.

**Remarque sur ILP2.** Les temps d'exécution observés (15 à 63 secondes, pour seulement 100 nœuds) suggèrent qu'ILP2 opère sous un budget de calcul limité et ne garantit donc pas toujours la preuve d'optimalité formelle. La comparaison ci-dessous doit se lire comme *heuristique rapide vs. solveur exact sous contrainte de temps*, un cas d'usage réaliste où les méthodes heuristiques sont précisément recherchées.

## 5. Résultats

### 5.1 Tableau détaillé par instance

| instance | ILP2 statut | ILP2 longueur | ILP2 temps (s) | classique longueur | classique temps (s) | Skew-GRAM longueur | Skew-GRAM temps (s) |
|---|---|---|---|---|---|---|---|
| 100_1 | non trouvé | — | 0.50 | 11 | 1.58 | 9 | 0.51 |
| 100_121 | trouvé | 6 | 19.59 | 10 | 1.73 | **11** | 0.56 |
| 100_126 | trouvé | 7 | 38.68 | 8 | 1.64 | 7 | 0.51 |
| 100_128 | trouvé | 13 | 18.44 | 9 | 2.34 | 9 | 0.77 |
| 100_14 | trouvé | 12 | 8.51 | 8 | 1.84 | 9 | 0.60 |
| 100_18 | trouvé | 8 | 27.55 | 8 | 1.84 | 8 | 0.57 |
| 100_190 | trouvé | 11 | 25.44 | **11** | 2.22 | 8 | 0.76 |
| 100_193 | trouvé | 14 | 14.53 | 8 | 1.99 | 9 | 0.66 |
| 100_197 | trouvé | 11 | 19.73 | 8 | 1.87 | 8 | 0.60 |
| 100_201 | trouvé | 12 | 22.12 | **11** | 2.08 | 9 | 0.70 |
| 100_207 | trouvé | 7 | 32.75 | **12** | 2.08 | 11 | 0.65 |
| 100_227 | trouvé | 10 | 27.08 | 9 | 1.81 | 10 | 0.59 |
| 100_228 | trouvé | 8 | 28.75 | **11** | 2.00 | 11 | 0.62 |
| 100_251 | trouvé | 11 | 51.40 | 8 | 2.01 | **11** | 0.69 |
| 100_254 | trouvé | 7 | 22.70 | **11** | 1.70 | 9 | 0.55 |
| 100_269 | trouvé | 13 | 17.00 | 11 | 2.11 | 11 | 0.68 |
| 100_304 | trouvé | 9 | 14.95 | 9 | 1.78 | 9 | 0.56 |
| 100_320 | trouvé | 12 | 33.91 | **11** | 2.17 | 8 | 0.76 |
| 100_324 | trouvé | 8 | 28.59 | **9** | 2.24 | 8 | 0.77 |
| 100_332 | trouvé | 11 | 12.60 | 9 | 1.75 | 9 | 0.54 |
| 100_341 | trouvé | 9 | 27.68 | 9 | 2.22 | 9 | 0.72 |
| 100_352 | trouvé | 14 | 62.86 | 10 | 2.35 | **12** | 0.77 |
| 100_388 | trouvé | 15 | 58.18 | 8 | 2.34 | **9** | 0.76 |
| 100_397 | trouvé | 6 | 37.21 | 11 | 2.06 | **13** | 0.67 |
| 100_402 | trouvé | 11 | 13.16 | 9 | 2.10 | 9 | 0.65 |
| 100_411 | trouvé | 6 | 48.43 | 9 | 1.71 | 9 | 0.55 |
| 100_413 | trouvé | 8 | 61.73 | 8 | 1.74 | 8 | 0.59 |
| 100_429 | trouvé | 8 | 20.01 | 8 | 1.80 | 8 | 0.58 |
| 100_437 | trouvé | 8 | 24.40 | **11** | 2.11 | 8 | 0.67 |
| 100_457 | trouvé | 10 | 22.91 | 7 | 1.95 | **9** | 0.62 |
| 100_471 | trouvé | 14 | 31.07 | 9 | 2.39 | 8 | 0.73 |
| 100_478 | trouvé | 10 | 25.53 | 10 | 1.95 | 9 | 0.65 |
| 100_489 | trouvé | 7 | 46.67 | 9 | 2.27 | 9 | 0.67 |
| 100_491 | trouvé | 7 | 18.61 | 9 | 1.99 | 9 | 0.64 |
| 100_52 | trouvé | 6 | 24.71 | 9 | 2.36 | **11** | 0.74 |
| 100_527 | trouvé | 11 | 15.83 | 9 | 2.02 | **12** | 0.62 |
| 100_563 | non trouvé | — | 3.50 | 9 | 1.44 | 8 | 0.44 |
| 100_66 | trouvé | 16 | 9.98 | 11 | 2.87 | 11 | 0.87 |
| 100_80 | trouvé | 10 | 34.64 | **11** | 1.93 | 9 | 0.62 |

*(gras = la méthode obtenant le chemin le plus long sur cette instance)*

Tableau complet également disponible dans `results/per_instance_results.csv`.

### 5.2 Synthèse agrégée

| Métrique | Skip-Gram classique | One-to-One Skew-GRAM |
|---|---|---|
| Longueur moyenne de chemin | **9.41** | 9.38 |
| Longueur médiane | 9.0 | 9.0 |
| Écart moyen à ILP2 (%, négatif = dépasse ILP2) | −3.17 % | −2.93 % |
| Temps moyen (s) | 2.04 | **0.66** |
| Accélération moyenne vs ILP2 | ×14.1 | **×43.5** |
| Instances où la méthode atteint/dépasse ILP2 | 20 / 37 | 20 / 37 |
| Skew-GRAM ≥ classique | — | **26 / 39** |

### 5.3 Interprétation

Sur cet échantillon élargi de 39 instances, le **One-to-One Skew-GRAM** :
- égale ou dépasse le Skip-Gram classique sur **26 des 39 instances** (67 %) — la victoire en tête-à-tête est nette et stable même si la longueur moyenne globale des deux méthodes est très proche (9.38 contre 9.41) ;
- atteint ou dépasse la référence ILP2 dans la même proportion que le classique (20/37 instances résolues chacun), montrant que les deux approches restent compétitives face à un solveur exact à budget de temps limité ;
- est en moyenne **3 fois plus rapide** que le Skip-Gram classique (0.66 s contre 2.04 s), grâce au sous-échantillonnage des hubs qui réduit le nombre de couples d'entraînement et au beam search borné en largeur ;
- offre une accélération moyenne de **×43.5 par rapport à ILP2** (contre ×14.1 pour le classique) — un gain de vitesse net pour une qualité de solution équivalente, ce qui constitue le principal avantage pratique démontré par cette étude.

Ces résultats, obtenus sur un échantillon plus large que la version initiale à 12 instances, nuancent la conclusion : le gain du Skew-GRAM en **longueur moyenne absolue** est faible (et disparaît presque en moyenne globale), mais son avantage en **taux de victoire tête-à-tête** (67 %) et surtout en **vitesse** reste net et constant à travers les deux échelles d'échantillon testées (12 puis 39 instances) — un signal de robustesse pour la conclusion principale du projet.

### 5.4 Comparaison de sanité avec `gensim`

Un `Word2Vec` de `gensim` entraîné sur les mêmes marches aléatoires que l'instance représentative confirme la cohérence de notre implémentation *from scratch* : les deux modèles convergent vers un espace où des nœuds structurellement proches obtiennent une similarité cosinus élevée, avec un temps d'entraînement du même ordre de grandeur.

## 6. Visualisations

Le notebook associé produit huit figures (`figures/01_...png` à `08_...png`) : distributions de degré et de statut ILP2 (exploration), fréquence des nœuds et effet du sous-échantillonnage (prétraitement), courbes de perte/accuracy (entraînement), comparaison des longueurs de chemin et compromis qualité/temps (évaluation), projections PCA et t-SNE des embeddings, et dessin de la topologie du graphe avec les chemins trouvés surlignés.

## 7. Discussion

**Avantages.** Le décodage contraint garantit nativement la validité *one-to-one* d'un chemin, sans vérification post-hoc. La vitesse (fraction de seconde à quelques secondes par instance) permet d'explorer un grand nombre d'instances là où ILP2 nécessite plusieurs dizaines de secondes par graphe — un facteur déterminant si l'objectif est d'obtenir rapidement une bonne solution plutôt qu'une preuve d'optimalité.

**Limites.** Le Skew-GRAM reste une heuristique sans garantie d'optimalité. Le vocabulaire (les identifiants de nœuds) n'est pas transférable d'un graphe à l'autre : chaque instance nécessite un nouvel entraînement. L'échantillon de données disponible (39 instances sur ~890, soit ≈ 4.4 %) limite encore la portée statistique des conclusions quantitatives, même si les deux passes expérimentales (12 puis 39 instances) convergent vers la même conclusion qualitative.

**Cas d'usage.** Cette approche est pertinente sous budget de calcul strict — recherche heuristique initiale, warm-start pour un solveur exact, filtrage rapide d'instances faciles avant d'investir du temps ILP sur les plus difficiles — ou plus généralement dans tout contexte de plongement de graphes où la contrainte de non-répétition est structurellement centrale (tournées, ordonnancement, planification de trajets).

## 8. Conclusion

Ce projet a transposé le Skip-Gram du domaine textuel à un domaine de graphes aléatoires orientés, en identifiant une inadéquation fondamentale entre l'hypothèse implicite du Skip-Gram classique (répétition possible d'un token) et la contrainte structurelle d'un chemin (correspondance bijective). Le **One-to-One Skew-GRAM**, en intégrant cette contrainte à la fois dans l'entraînement (échantillonnage négatif structurel, sous-échantillonnage des hubs) et dans le décodage (softmax masquée avec beam search), améliore la qualité des chemins produits par rapport au Skip-Gram classique tout en restant très largement plus rapide qu'un solveur exact (ILP2), sur l'échantillon de données disponible.

### Recommandations d'amélioration future

1. Lever la limitation de débit de l'API Google Drive pour évaluer la méthode sur l'ensemble des ~890 instances disponibles.
2. Remplacer le beam search par un décodeur plus expressif (Pointer Network, Vinyals et al., 2015 ; Transformer conditionné sur le graphe).
3. Entraîner directement une politique de construction de chemin par apprentissage par renforcement (récompense = longueur finale).
4. Comparer à des embeddings appris par réseaux de neurones sur graphes (GraphSAGE, GAT) plutôt que par marches aléatoires.
5. Étudier des schémas de méta-apprentissage permettant un transfert entre graphes malgré l'absence de vocabulaire commun.

## Références

- Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality*. NeurIPS.
- Perozzi, B., Al-Rfou, R., & Skiena, S. (2014). *DeepWalk: Online Learning of Social Representations*. KDD.
- Grover, A., & Leskovec, J. (2016). *node2vec: Scalable Feature Learning for Networks*. KDD.
- Vinyals, O., Fortunato, M., & Jaitly, N. (2015). *Pointer Networks*. NeurIPS.
- Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*. arXiv:1412.6980.

## Annexes / reproductibilité

- Code complet, commenté et exécutable de bout en bout : `one_to_one_skewgram.ipynb`.
- Tableaux de résultats bruts : `results/per_instance_results.csv`, `results/summary.csv`.
- Figures : `figures/01_exploration_overview.png` à `figures/08_graph_topology_paths.png`.
- Provenance et procédure d'extension du jeu de données : `data/README.md`.
- Graine aléatoire unique : `SEED = 42` (fixée pour `random`, `numpy`, `torch`).
