# One-to-One Skew-GRAM : plus long chemin (D,G)-consistant par embeddings de nœuds

**Rapport scientifique — Projet universitaire (Optimisation / Deep Learning / NLP appliqués aux graphes)**
**Date :** 2026-08-05 — *version corrigée après retour du professeur*

---

## Résumé

Ce rapport documente une **version corrigée** du projet. Une version antérieure
supposait à tort que le chemin cherché dans le graphe orienté `D` devait *aussi* être
un chemin dans le graphe `G` (mêmes arêtes consécutives dans les deux). C'est faux :
la seule contrainte que `G` impose est que l'**ensemble** de sommets du chemin induise
un **sous-graphe connexe** dans `G` — ce sous-graphe peut être un arbre, pas
nécessairement un chemin. Cette version corrige la définition du problème, retire
toute comparaison avec un Skip-Gram classique (le projet repose désormais
uniquement sur le **One-to-One Skew-GRAM**), et remplace le décodeur par un
algorithme en deux étapes fidèle à la vraie contrainte. Les résultats, recalculés sur
38 instances, sont honnêtes quant aux limites de cette première heuristique : elle
reste, en l'état, nettement en dessous d'ILP2 en longueur de chemin (moyenne 3.42
contre 9.97 nœuds) mais 66 à 170 fois plus rapide.

---

## 1. Définitions formelles

1. **Graphe orienté `D`** : couple $(V, A)$, $A \subseteq V \times V$ un ensemble
   d'arcs (couples ordonnés de sommets).
2. **Graphe non-orienté `G`** : couple $(V, E)$, $E$ un ensemble de paires non
   ordonnées de sommets.
3. **Graphe non-orienté connexe `G`** : pour toute paire $u,v \in V$, il existe une
   chaîne de $u$ à $v$ dans $G$.
4. **Sous-graphe de `G` induit par $S \subseteq V$, connexe** : $G[S] = (S, E \cap
   (S\times S))$ est connexe au sens du point 3. $G[S]$ peut être un arbre — un
   sous-graphe connexe n'est pas nécessairement un chemin ni aussi dense que $G$.
5. **Chemin dans un graphe orienté `D`** : séquence de sommets distincts
   $p=(v_1,\dots,v_k)$ telle que $(v_i,v_{i+1}) \in A$ pour tout $i$.
6. **Graphe orienté acyclique (DAG) `D`** : graphe orienté sans circuit dirigé.
7. **Chemin $(D,G)$-consistant** : un chemin $p=(v_1,\dots,v_k)$ dans `D` (point 5)
   dont l'ensemble de sommets $\{v_1,\dots,v_k\}$ induit dans `G` un sous-graphe
   connexe (point 4).

**Problème résolu** : trouver le plus long chemin $(D,G)$-consistant. Le seul lien
entre `D` et `G` est qu'ils partagent le même ensemble de sommets $V$ ; `D` est un
DAG, `G` est non-orienté connexe.

## 2. Correction par rapport à la version précédente

**Erreur identifiée par le professeur.** La version précédente décodait un chemin en
exigeant, à chaque étape, que l'arête choisie existe *aussi* dans `G` — c'est-à-dire
qu'elle traitait implicitement le chemin cherché dans `D` comme devant être également
un chemin dans `G`. Ce n'est pas la définition du problème : la seule exigence
portant sur `G` est que l'**ensemble final** de sommets visités induise un sous-graphe
connexe, qui peut être un arbre.

**Vérification empirique de la nouvelle définition** (sur les 39 instances locales,
via `networkx`) :
- `D` est un DAG dans 39/39 instances testées (point 6 vérifié).
- `G`, interprété comme **non-orienté** (les lignes `a-b` de `graphG.txt` sont
  symétrisées, le sens d'écriture est ignoré puisque `G` n'a pas d'orientation par
  définition), est connexe dans 38/39 instances (1 exception, `100_126`, exclue de
  l'évaluation).
- Les 37 chemins solutions du solveur exact **ILP2** sont, dans 37/37 cas, des
  chemins valides dans `D` **et** des ensembles de sommets induisant un sous-graphe
  connexe dans `G` — confirmant qu'ILP2 résout bien le problème (D,G)-consistant tel
  que défini ci-dessus, et que nos vérifications de chargement des données sont
  correctes.
- **Constat déterminant pour l'algorithme** : la connexité du **préfixe visité pas à
  pas** (dans l'ordre de parcours du chemin) n'est vérifiée que sur 1 des 37
  solutions ILP2. Autrement dit, il est impossible d'imposer la connexité de façon
  incrémentale pendant la construction du chemin — elle ne peut être vérifiée que sur
  l'ensemble final (ou un sous-ensemble contigu). C'est précisément la conséquence de
  l'avertissement du professeur : le sous-graphe induit peut être un arbre, formé par
  des arêtes de `G` reliant des sommets du chemin dans un ordre complètement
  indépendant de leur ordre de visite dans `D`.

## 3. Repositionnement du projet (rappel)

Le dataset fourni (~890 instances Erdős–Rényi `100_<k>`, 39 collectées localement,
voir `data/README.md`) contient pour chaque instance `graphD.txt` (orienté),
`graphG.txt` (chargé ici comme **non-orienté**, correction par rapport à la lecture
précédente qui le traitait comme un second graphe orienté) et `solution.txt` (sortie
du solveur ILP2). Le Skip-Gram est transposé aux graphes suivant DeepWalk/node2vec
(nœud = mot, marche aléatoire sur `D` = phrase) et sert d'heuristique pour le
problème (D,G)-consistant défini ci-dessus.

## 4. Méthode

### 4.1 One-to-One Skew-GRAM (seule méthode entraînée)

Deux matrices d'embeddings $\mathbf V_{in}, \mathbf V_{out} \in \mathbb R^{N\times d}$
($d=32$), perte de Negative Sampling optimisée par Adam :

$$\mathcal L(c,o,n_{1:K}) = -\log\sigma(\mathbf v_{in,c}^\top \mathbf v_{out,o}) -
\sum_{k=1}^K \log\sigma(-\mathbf v_{in,c}^\top \mathbf v_{out,n_k})$$

- **Négatifs structurels** : tirés parmi les non-successeurs de $c$ dans `D`
  ($P_{struct}(n\mid c) \propto \mathbb 1[(c,n)\notin A]$), au lieu d'une distribution
  unigramme agnostique de la topologie.
- **Sous-échantillonnage des hubs** : formule word2vec appliquée à la fréquence de
  visite des nœuds dans les marches sur `D`, pour rééquilibrer nœuds périphériques et
  nœuds très connectés.

Aucun Skip-Gram classique n'est entraîné dans cette version — la consigne du projet
est de reposer uniquement sur le One-to-One Skew-GRAM.

### 4.2 Décodage en deux étapes (le cœur de la correction)

Puisque la connexité ne peut pas être imposée de façon incrémentale (§2), le décodage
sépare génération et vérification :

**Étape A — génération d'un chemin candidat dans `D`.** Beam search (largeur 4) sur
les successeurs de `D`, masquant les nœuds déjà visités (softmax « skewed », garantit
l'unicité position → nœud) :

$$P_{skew}(v\mid u,\mathcal U) = \frac{\mathbb 1[(u,v)\in A]\cdot\mathbb 1[v\notin\mathcal U]\cdot
\exp(\mathrm{sim}(u,v)/\tau)}{\sum_{v'\in N^+(u)\setminus\mathcal U}\exp(\mathrm{sim}(u,v')/\tau)},
\quad \tau=0.5$$

Aucune contrainte liée à `G` n'intervient à ce stade — on cherche un long chemin dans
`D` guidé par la similarité d'embeddings.

**Étape B — extraction du plus long sous-chemin (D,G)-consistant.** Un sous-chemin
**contigu** d'un chemin dans `D` est toujours lui-même un chemin dans `D`. On cherche
donc, par fenêtre glissante sur le chemin candidat de longueur $L$, le plus long
intervalle `[i..j]` dont l'ensemble de sommets induit un sous-graphe connexe dans `G`
(Union-Find incrémental par fenêtre, $O(L^2\cdot\alpha(L))$, négligeable pour
$L\le 100$). On retient le meilleur intervalle sur l'ensemble des nœuds de départ
testés.

Ce décodage en deux temps remplace l'ancienne hypothèse erronée (arête aussi
présente dans `G` à chaque étape) par la vraie contrainte : connexité de l'ensemble
final, vérifiée a posteriori plutôt qu'imposée pas à pas.

## 5. Protocole expérimental

Pour chaque instance valide (`D` DAG, `G` connexe, 38/39 instances) : marches
aléatoires sur `D` (40 par nœud de départ, longueur max. 30) → sous-échantillonnage
des hubs → couples Skip-Gram (fenêtre 3) → entraînement Skew-GRAM (négatifs
structurels, 8 époques, Adam lr=0.01) → décodage (étape A depuis chaque nœud de
départ possible, étape B sur chaque candidat) → comparaison à ILP2. Graine unique
`SEED=42`.

## 6. Résultats

### 6.1 Tableau détaillé par instance

| instance | ILP2 statut | ILP2 longueur | ILP2 temps (s) | Skew-GRAM brut (D, étape A) | Skew-GRAM (D,G)-consistant (étape A+B) | Skew-GRAM temps (s) | accélération vs ILP2 |
|---|---|---|---|---|---|---|---|
| 100_1 | non trouvé | — | 0.50 | 3 | 2 | 0.30 | ×1.6 |
| 100_121 | trouvé | 6 | 19.59 | 3 | 3 | 0.32 | ×61.6 |
| 100_128 | trouvé | 13 | 18.44 | 5 | 3 | 0.46 | ×40.4 |
| 100_14 | trouvé | 12 | 8.51 | 5 | 2 | 0.34 | ×24.9 |
| 100_18 | trouvé | 8 | 27.55 | 7 | 3 | 0.36 | ×76.8 |
| 100_190 | trouvé | 11 | 25.44 | 6 | 5 | 0.44 | ×57.3 |
| 100_193 | trouvé | 14 | 14.53 | 5 | 3 | 0.38 | ×37.9 |
| 100_197 | trouvé | 11 | 19.73 | 4 | 4 | 0.38 | ×51.3 |
| 100_201 | trouvé | 12 | 22.12 | 7 | 5 | 0.41 | ×54.3 |
| 100_207 | trouvé | 7 | 32.75 | 5 | 4 | 0.40 | ×81.0 |
| 100_227 | trouvé | 10 | 27.08 | 8 | 6 | 0.35 | ×77.8 |
| 100_228 | trouvé | 8 | 28.75 | 5 | 3 | 0.40 | ×71.7 |
| 100_251 | trouvé | 11 | 51.40 | 7 | 6 | 0.44 | ×116.3 |
| 100_254 | trouvé | 7 | 22.70 | 6 | 3 | 0.40 | ×56.6 |
| 100_269 | trouvé | 13 | 17.00 | 8 | 3 | 0.45 | ×38.1 |
| 100_304 | trouvé | 9 | 14.95 | 9 | 4 | 0.39 | ×38.8 |
| 100_320 | trouvé | 12 | 33.91 | 6 | 3 | 0.56 | ×60.1 |
| 100_324 | trouvé | 8 | 28.59 | 6 | 2 | 0.56 | ×51.2 |
| 100_332 | trouvé | 11 | 12.60 | 7 | 4 | 0.35 | ×36.0 |
| 100_341 | trouvé | 9 | 27.68 | 4 | 3 | 0.45 | ×61.0 |
| 100_352 | trouvé | 14 | 62.86 | 7 | 4 | 0.49 | ×128.5 |
| 100_388 | trouvé | 15 | 58.18 | 7 | 6 | 0.45 | ×128.8 |
| 100_397 | trouvé | 6 | 37.21 | 9 | 3 | 0.41 | ×90.5 |
| 100_402 | trouvé | 11 | 13.16 | 8 | 2 | 0.41 | ×32.0 |
| 100_411 | trouvé | 6 | 48.43 | 4 | 4 | 0.34 | ×140.7 |
| 100_413 | trouvé | 8 | 61.73 | 6 | 2 | 0.36 | ×169.9 |
| 100_429 | trouvé | 8 | 20.01 | 7 | 2 | 0.35 | ×56.5 |
| 100_437 | trouvé | 8 | 24.40 | 7 | 4 | 0.41 | ×59.5 |
| 100_457 | trouvé | 10 | 22.91 | 6 | 4 | 0.40 | ×56.9 |
| 100_471 | trouvé | 14 | 31.07 | 8 | 3 | 0.44 | ×70.6 |
| 100_478 | trouvé | 10 | 25.53 | 10 | 3 | 0.41 | ×63.0 |
| 100_489 | trouvé | 7 | 46.67 | 6 | 3 | 0.39 | ×121.2 |
| 100_491 | trouvé | 7 | 18.61 | 11 | 2 | 0.40 | ×46.3 |
| 100_52 | trouvé | 6 | 24.71 | 6 | 3 | 0.46 | ×53.5 |
| 100_527 | trouvé | 11 | 15.83 | 5 | 2 | 0.38 | ×41.5 |
| 100_563 | non trouvé | — | 3.50 | 4 | 3 | 0.27 | ×13.2 |
| 100_66 | trouvé | 16 | 9.98 | 12 | 7 | 0.71 | ×14.0 |
| 100_80 | trouvé | 10 | 34.64 | 6 | 2 | 0.37 | ×92.6 |

Tableau complet également disponible dans `results/per_instance_results.csv`
(38 instances valides sur 39 — `100_126` exclue car `G` n'y est pas connexe).

### 6.2 Synthèse agrégée

| Métrique | Valeur |
|---|---|
| Instances évaluées | 38 (sur 39, `100_126` exclue) |
| Longueur moyenne ILP2 (36 instances résolues) | 9.97 |
| Longueur moyenne chemin brut Skew-GRAM (étape A, dans `D` seul) | 6.45 |
| Longueur moyenne chemin (D,G)-consistant (étape A+B) | **3.42** |
| Écart moyen à ILP2 (%) | −63.6 % |
| Temps moyen Skew-GRAM (s) | 0.41 |
| Temps moyen ILP2 (s) | 28.04 (moyenne biaisée par les temps d'échec) |
| Accélération moyenne vs ILP2 | ×68.3 |
| Instances où Skew-GRAM atteint/dépasse ILP2 | 0 / 36 |
| Instances où le chemin brut est déjà (D,G)-consistant (aucune perte à l'étape B) | 3 / 38 |

### 6.3 Interprétation

**L'écart entre chemin brut (étape A) et chemin (D,G)-consistant (étape A+B) est
important et systématique** : sur seulement 3 des 38 instances, le chemin brut généré
dans `D` induisait déjà un sous-graphe connexe dans `G` sans aucune troncature. Dans
tous les autres cas, l'étape B doit réduire fortement la longueur pour retrouver la
connexité — la longueur moyenne passe de 6.45 (étape A) à 3.42 (étape A+B). **C'est
exactement la manifestation concrète de l'erreur corrigée dans ce rapport** : un long
chemin dans `D`, choisi sans tenir compte de `G`, n'a qu'une faible probabilité que
son ensemble de sommets soit connexe dans `G`, car les embeddings de l'étape A ne sont
entraînés que sur la structure de `D`.

**Comparaison à ILP2.** Sur cette première version de l'algorithme, le Skew-GRAM en
deux étapes ne dépasse ni n'égale ILP2 sur aucune des 36 instances résolues (longueur
moyenne 3.42 contre 9.97). Il reste néanmoins **très largement plus rapide** (×68.3 en
moyenne, jusqu'à ×170 sur certaines instances), ce qui en fait un candidat pertinent
comme *filtre rapide* ou *warm-start*, mais pas encore comme substitut de qualité à
ILP2 en l'état.

**Cause principale de l'écart de qualité.** L'étape A optimise uniquement la
similarité d'embeddings appris sur les marches de `D`, sans aucun signal sur `G`. La
contrainte de connexité dans `G` n'intervient qu'a posteriori (étape B), qui ne peut
que *retrancher* des nœuds au chemin généré, jamais en ajouter ni en réordonner. Une
recherche qui intégrerait un signal de connectivité dans `G` dès l'étape A (par
exemple un bonus de score pour les successeurs déjà adjacents à des sommets visités,
sans que ce soit un filtre dur — puisqu'on a montré que la connexité incrémentale
n'est pas nécessaire pour un ensemble final connexe) devrait réduire cet écart ; c'est
la piste d'amélioration prioritaire (voir §8).

## 7. Visualisations

Le notebook associé (`one_to_one_skewgram.ipynb`) produit six figures
(`figures/01_...png` à `figures/06_...png`) : effet du sous-échantillonnage des hubs,
courbe de perte d'entraînement, comparaison des longueurs de chemin (ILP2 / brut /
(D,G)-consistant) et compromis qualité-temps, projections PCA et t-SNE des
embeddings, et topologie comparée du chemin brut dans `D` et du sous-graphe induit
dans `G` par le chemin (D,G)-consistant final.

## 8. Discussion

**Avantages.** Le décodage en deux étapes est fidèle à la définition corrigée du
problème (D,G)-consistant, contrairement à la version précédente. La séparation
explicite entre génération (étape A) et vérification de connexité (étape B) rend
directement mesurable, pour chaque instance, le coût en longueur de la contrainte de
`G` — un diagnostic qui n'existait pas dans la version précédente.

**Limites.** Cette première implémentation du décodage en deux étapes est
volontairement simple : l'étape B ne considère que des intervalles **contigus** du
chemin produit par l'étape A, sans réordonnancement ni recherche jointe. Les
résultats montrent que ceci pénalise fortement la longueur finale par rapport à ILP2.
L'échantillon (38 instances sur ~890 disponibles) limite par ailleurs la portée
statistique des conclusions quantitatives.

**Cas d'usage réaliste avec l'état actuel de l'heuristique.** Filtre rapide
(élimination d'instances triviales avant d'investir du temps ILP), ou point de départ
(warm-start) pour un solveur exact — pas encore un substitut de qualité comparable à
ILP2.

## 9. Conclusion

Ce projet corrige une erreur de modélisation identifiée par le professeur : le chemin
cherché dans le DAG `D` ne doit pas être un chemin dans `G`, mais son ensemble de
sommets doit seulement induire un sous-graphe **connexe** (potentiellement un arbre)
dans le graphe non-orienté connexe `G`. Le **One-to-One Skew-GRAM** — désormais seule
méthode d'embedding du projet — est associé à un décodeur en deux étapes (génération
dans `D`, puis extraction du plus long sous-chemin contigu (D,G)-consistant) fidèle à
cette définition. Les résultats, recalculés honnêtement sur les données disponibles,
montrent que cette première version de l'heuristique reste loin de la qualité d'ILP2
en longueur de chemin, mais conserve un avantage de vitesse très net — un point de
départ pour les améliorations décrites ci-dessous plutôt qu'une solution aboutie.

### Recommandations d'amélioration future

1. Intégrer un signal de connectivité dans `G` directement dans le score de décodage
   de l'étape A (recherche jointe D+G) plutôt qu'une vérification a posteriori
   uniquement soustractive.
2. Autoriser l'étape B à considérer des réordonnancements ou des sous-ensembles non
   contigus compatibles avec un ordre topologique de `D`, plutôt que les seuls
   intervalles contigus du chemin brut.
3. Étendre l'évaluation à l'ensemble des ~890 instances disponibles (lever la
   limitation de débit de l'API Google Drive, voir `data/README.md`).
4. Entraîner des embeddings joints sur `D` et `G` plutôt que sur les seules marches de
   `D`, pour que l'étape A dispose déjà d'un signal de connectivité dans `G`.
5. Comparer à un décodeur par Pointer Network ou par apprentissage par renforcement,
   conditionné conjointement sur `D` et `G`.

## Références

- Mikolov, T., Sutskever, I., Chen, K., Corrado, G., & Dean, J. (2013). *Distributed Representations of Words and Phrases and their Compositionality*. NeurIPS.
- Perozzi, B., Al-Rfou, R., & Skiena, S. (2014). *DeepWalk: Online Learning of Social Representations*. KDD.
- Grover, A., & Leskovec, J. (2016). *node2vec: Scalable Feature Learning for Networks*. KDD.
- Kingma, D. P., & Ba, J. (2014). *Adam: A Method for Stochastic Optimization*. arXiv:1412.6980.

## Annexes / reproductibilité

- Code complet, commenté et exécutable de bout en bout : `one_to_one_skewgram.ipynb`.
- Tableaux de résultats bruts : `results/per_instance_results.csv`, `results/summary.csv`.
- Figures : `figures/01_hub_subsampling.png` à `figures/06_graph_topology_paths.png`.
- Provenance et procédure d'extension du jeu de données : `data/README.md`.
- Graine aléatoire unique : `SEED = 42` (fixée pour `random`, `numpy`, `torch`).
