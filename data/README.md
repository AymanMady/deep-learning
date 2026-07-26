# Provenance et reproductibilité du dataset

## Source

Dossier Google Drive fourni pour le projet : `erdos_renyi`
https://drive.google.com/drive/folders/18meW_x2uaSTFxhaZagfzLYe5p-RwvOHE

Le dossier contient environ **890 instances** nommées `100_<k>`, chacune avec trois fichiers :

- `graphD.txt` — un graphe orienté Erdős–Rényi G(100, p) (arêtes `a-b` = arc a → b)
- `graphG.txt` — un second graphe orienté indépendant sur les mêmes 100 nœuds
- `solution.txt` — résultat d'un solveur exact ILP2 pour le plus long chemin simple dans `graphD`

Voir `REPORT.md` (§2) pour le détail de l'analyse ayant permis d'identifier cette structure et de vérifier l'orientation des arêtes.

## Échantillon effectivement téléchargé dans `data/raw/`

**39 instances**, collectées en plusieurs vagues : 11 tirées aléatoirement avec la graine `SEED=42` parmi les instances disposant d'une `solution.txt` exploitable, 27 tirées lors de sessions de collecte séquentielles ultérieures (une fois le quota Drive temporairement levé), plus l'instance `100_1` ajoutée manuellement pour illustrer le cas `"No solution found"`.

```
100_1    100_121  100_126  100_128  100_14   100_18   100_190  100_193
100_197  100_201  100_207  100_227  100_228  100_251  100_254  100_269
100_304  100_320  100_324  100_332  100_341  100_352  100_388  100_397
100_402  100_411  100_413  100_429  100_437  100_457  100_471  100_478
100_489  100_491  100_52   100_527  100_563  100_66   100_80
```

## Pourquoi seulement 39 instances sur ~890 ?

Le téléchargement programmatique via `gdown` (API publique Google Drive) est soumis à un **quota anti-abus par fichier/IP** (« Cannot retrieve the public link of the file [...] have had many accesses »). Une première tentative de téléchargement **parallèle** (12 workers simultanés) a déclenché ce quota après ~100 fichiers, bloquant ensuite l'accès à **tout** fichier du dossier — y compris des fichiers jamais sollicités auparavant. Le quota s'est ensuite levé et redéclenché à plusieurs reprises au cours de sessions de téléchargement **séquentielles** : chaque vague permettait de récupérer 15 à 25 instances supplémentaires avant de retomber en blocage, sans qu'on puisse prédire précisément la durée de chaque cycle de blocage/déblocage.

**Leçon retenue : ne pas paralléliser les téléchargements `gdown` sur un même dossier Drive public**, et s'attendre à devoir répéter plusieurs vagues de collecte séquentielle (espacées de ~0.7-1 s entre fichiers) séparées par des pauses, pour un dataset de cette taille.

## Comment étendre le dataset

Une fois le quota Google Drive levé (attendre quelques heures, puis tester avec la commande ci-dessous), relancer la collecte :

```bash
# 1. Tester si le quota est levé (doit réussir sans message "many accesses")
gdown 1zpZqmu14CH_Hw72cpWb3THdaDF_Uk2Ne -O /tmp/test.txt

# 2. Reconstruire la liste complète des fichiers du dossier (mapping instance -> IDs Drive)
gdown --folder "https://drive.google.com/drive/folders/18meW_x2uaSTFxhaZagfzLYe5p-RwvOHE" -O data/raw --dry-run
# (interrompre dès que la liste est suffisante ; NE PAS lancer le téléchargement réel en parallèle)

# 3. Télécharger séquentiellement, avec une pause entre chaque fichier, en réutilisant
#    le même schéma d'échantillonnage aléatoire (SEED=42) que celui utilisé dans le notebook
#    pour sélectionner de nouvelles instances jamais tirées.
```

Le notebook `one_to_one_skewgram.ipynb` (fonction `list_instances()`, §1) **détecte automatiquement** toutes les instances complètes présentes dans `data/raw/` — il suffit de ré-exécuter le notebook après avoir ajouté de nouveaux dossiers `100_<k>/{graphD.txt,graphG.txt,solution.txt}` pour obtenir des statistiques et un tableau comparatif portant sur un échantillon plus large, sans modifier une seule ligne de code.

## Format des fichiers

`graphD.txt` / `graphG.txt` :
```
N:100        <- nombre de nœuds
P:0.08       <- probabilité d'arête du modèle Erdős–Rényi (absent de graphG.txt)
1-9          <- arc orienté 1 → 9
1-30
...
```

`solution.txt` (cas résolu) :
```
Result of ILP2 for 100_121:
Path length: 6
Path : [60, 4, 93, 69, 77, 43]
Execution time: 19.5942 seconds
```

`solution.txt` (cas non résolu dans le temps imparti) :
```
Result of ILP2 for 100_1:
No solution found.
Execution time: 0.4958 seconds
```
