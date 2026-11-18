# Projet MLOps — Prévision des ventes

# Auteurs

- Ju-Heon HWANG
- Néma Elisée KOUROUMA
- Ahmed CHAUDHRY

## Données : provenance et conditions d’utilisation

### Provenance

Le projet utilise le jeu de données **Online Retail**, publié par le **UCI Machine Learning
Repository** (identifiant UCI `352`). La fiche officielle est disponible à l’adresse :

- [UCI Machine Learning Repository — Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)
- DOI : [10.24432/C5BW33](https://doi.org/10.24432/C5BW33)

La citation recommandée par UCI est :

> Chen, D. (2015). *Online Retail* [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C5BW33.

La fiche attribue le jeu à **Daqing Chen**, de la School of Engineering de London South Bank
University. Elle décrit des transactions d’un commerce en ligne britannique sans magasin physique,
principalement consacré à des cadeaux, dont une partie de la clientèle est constituée de grossistes.

Les données couvrent les transactions du **1er décembre 2010 au 9 décembre 2011** et comprennent
**541 909 lignes**. La copie fournie avec ce projet se trouve dans `online+retail.zip` ; l’archive
contient le fichier `Online Retail.xlsx` (23 715 344 octets). Elle est conservée comme copie locale
du fichier distribué par UCI. Pour reproduire une récupération, utiliser la fiche UCI et comparer
les empreintes du fichier avant de réutiliser des résultats existants.

### Contenu du fichier

Le fichier décrit des lignes de transaction avec les champs suivants :

| Champ UCI | Signification |
| --- | --- |
| `InvoiceNo` | Numéro de facture ; un numéro commençant par `C` indique une annulation. |
| `StockCode` | Code du produit. |
| `Description` | Nom du produit. |
| `Quantity` | Quantité d’articles sur la ligne. |
| `InvoiceDate` | Date et heure de création de la transaction. |
| `UnitPrice` | Prix unitaire en livres sterling. |
| `CustomerID` | Identifiant client fourni dans le jeu. |
| `Country` | Pays de résidence du client. |

Ces observations sont des lignes de transactions historiques. Elles ne sont pas directement des
prévisions futures : le projet devra expliquer comment les lignes sont regroupées dans le temps et
quelle variable représente les ventes à prévoir. Les annulations et les valeurs atypiques doivent
être examinées et documentées selon le protocole retenu, plutôt que supprimées sans justification.

### Licence et attribution

La fiche UCI indique une licence **Creative Commons Attribution 4.0 International (CC BY 4.0)**.
Elle permet de partager et d’adapter le jeu, y compris pour tout usage, sous réserve de donner le
crédit approprié. Consulter le texte de la licence sur
[creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

Pour toute copie ou version transformée conservée avec le projet :

- citer Chen et le UCI Machine Learning Repository avec le DOI ci-dessus ;
- conserver le lien vers la fiche et la licence ;
- indiquer les transformations effectuées (nettoyage, agrégation, filtrage ou création de variables)
  et préciser qu’elles sont celles du projet ;
- ne pas laisser entendre que l’auteur ou UCI approuve le projet ou ses conclusions.

La licence du jeu ne remplace pas les conditions des logiciels utilisés pour le traiter. Le dépôt
ne doit pas contenir de secrets. Si une version volumineuse des données n’est pas nécessaire à la
reproduction, privilégier une commande de téléchargement documentée et une empreinte plutôt que
d’ajouter le fichier décompressé au dépôt.


## Exploration des données de ventes

### Données :

Le jeu Online Retail de l’UCI contient des lignes d’articles vendus : une ligne correspond à un produit sur une facture, identifiée par InvoiceNo et StockCode. Il inclut notamment la quantité, la date et l’heure, le prix et le client. Les données couvrent décembre 2010 à décembre 2011.

### Choix retenus :

- Prédiction : quantité vendue brute d’un produit pour le lendemain, au niveau produit × jour.
- Quantité brute : somme des quantités positives, en excluant les factures dont le numéro commence par C (= Canceled ou annulé)
- Jours sans vente : quantité égale à zéro, à partir de la première transaction observée du produit. Ce zéro ne prouve pas que le produit était disponible.
- Produits étudiés : les dix plus vendus jusqu’au 31 août 2011 inclus. Septembre est réservé au test et n’influence donc pas cette sélection.
- Découpage temporel du notebook : entraînement jusqu’au 31 août 2011 ; test chronologique du 1er au 30 septembre.
- Baseline : moyenne des quantités des quatre mêmes jours de semaine précédents.

Le notebook `exploration_donnees.ipynb` sert à comprendre le dataset. Le notebook `entrainement_modele.ipynb` contient un premier prototype LSTM/Transformer et son test temporel. Ils servent à l’apprentissage et à l’expérimentation ; le code final devra fonctionner dans des modules Python indépendants des notebooks.

### Séparation du fichier historique

Le fichier source couvre la période du 1er décembre 2010 au 9 décembre 2011. Il a été séparé au 1er octobre 2011 :

- `online_retail_actuel.xlsx` contient les dates antérieures au 1er octobre, du 1er décembre 2010 au 30 septembre 2011 : **370 931 lignes**.
- `online_retail_future.xlsx` contient les dates à partir du 1er octobre : **170 978 lignes**, du 2 octobre au 9 décembre. La source ne contient aucune transaction datée du 1er octobre.

Cette séparation représente un scénario où le modèle connaît les ventes actuelles et où les ventes d’un nouveau jour arrivent plus tard. **Le notebook d’entraînement ne lit que `online_retail_actuel.xlsx`**. Le fichier `online_retail_future.xlsx` est réservé à une future démonstration de rejeu : un producteur pourra envoyer les lignes jour par jour comme si elles venaient d’arriver. Il ne sera pas fourni au modèle avant leur arrivée simulée.

PUSH signifie que notre API reçoit les nouvelles ventes. PULL signifie que notre application va les chercher auprès d’une source. Online Retail est un fichier historique, pas une API de ventes en direct : le rejeu de ce fichier sera présenté comme une simulation.

### Prototype de modèles : LSTM et Transformer

Le notebook entraîne deux modèles partagés entre les dix produits : un LSTM et un Transformer Encoder avec un seul bloc. Chaque exemple représente un produit et ses 100 quantités quotidiennes précédentes. Un embedding apprend une représentation de son `StockCode`.

Le LSTM est un réseau conçu pour traiter des séquences. Il lit les 100 jours dans l’ordre et garde une mémoire des informations qui lui semblent utiles pour prévoir le lendemain.

Le Transformer examine les jours de la séquence à l’aide d’un mécanisme d’attention. Il peut apprendre quels jours passés sont les plus utiles pour la prévision. Dans notre notebook, il est volontairement petit : une seule couche d’encodeur.

| Donnée | Dimension | Signification |
|---|---|---|
| Ventes passées | `(batch_size, 100, 1)` | 100 quantités journalières par exemple |
| Indice du produit | `(batch_size,)` | Un produit associé à chaque série |
| Embedding du produit | `(batch_size, 4)` | Quatre valeurs apprises qui représentent chaque produit |
| Prévision | `(batch_size,)` | Une quantité prévue pour le lendemain par exemple |

Le test enregistré porte sur septembre 2011 et compare les deux modèles à la baseline. Les résultats sont exploratoires : ils ne constituent pas une garantie de performance sur de nouvelles ventes.

## Progression de l’industrialisation

Les éléments déjà réalisés sont la provenance documentée, l’exploration, la séparation temporelle, le prototype des deux modèles dans les notebooks, ainsi que les étapes 1 à 3 de validation, préparation et définition des architectures. Chaque module aura des tests ciblés avant d’être intégré à la suite. Les notebooks ne seront pas une dépendance à l’exécution du service.

### Étape 1 — Définir les contrats et valider les données — réalisée

Le contrat concerne les **lignes brutes de facture**, avant leur agrégation en ventes quotidiennes. Il est décrit dans `src/retail_forecast/schemas.py` et contrôlé par `validate_raw_data` dans `src/retail_forecast/validation.py`. La fonction retourne le DataFrame inchangé si les règles passent ; sinon, elle lève `DataValidationError` avec les problèmes détectés.

| Colonne | Règle actuelle |
| --- | --- |
| `InvoiceNo`, `StockCode` | Obligatoires, non vides. |
| `Quantity` | Obligatoire, renseigné, numérique, fini et entier. Les valeurs négatives sont acceptées : leur traitement appartient à l’agrégation ultérieure. |
| `InvoiceDate` | Obligatoire et convertible en date. |
| `UnitPrice` | Facultatif et conservé dans les lignes brutes, mais non validé : le prix n’est pas utilisé pour la prévision des quantités. |
| `Description`, `CustomerID`, `Country` | Facultatives ; leurs valeurs absentes sont autorisées. |

Les tests de `tests/test_validation.py` couvrent les données conformes, les colonnes ou valeurs manquantes, les identifiants vides, les quantités non entières, les dates illisibles et le fait que les valeurs de `UnitPrice` ne bloquent pas ce projet. La suite complète compte actuellement **37 tests réussis** ; elle est lancée avec `uv run pytest -q`.

### Étape 2 — Extraire l’ingestion et la préparation des données — implémentée et testée

`src/retail_forecast/ingestion.py` contient `load_current_data(file_path)`. Cette fonction lit le fichier Excel indiqué, applique la validation de l’étape 1 et retourne les lignes brutes sans les transformer. Le futur `main(config.json)` fournira le chemin ; l’ingestion ne lit donc pas elle-même la configuration.

`src/retail_forecast/preprocessing.py` contient deux transformations :

- `prepare_daily_gross_sales` exclut les factures commençant par `C` et les quantités non positives, agrège les ventes par produit et par jour, sélectionne les dix produits les plus vendus jusqu’au 31 août, puis complète les jours sans vente par zéro. Les séries peuvent continuer jusqu’au 30 septembre pour préparer les cibles du test.
- `build_sequences` crée les entrées de 100 jours et la cible du jour suivant. Il conserve le code produit et la date cible pour permettre le découpage chronologique ultérieur. Pour 100 jours d’historique, une entrée de ventes a la forme `(N, 100, 1)` et la cible `(N,)`.

`tests/test_preprocessing.py` vérifie avec de petits DataFrames : l’exclusion des factures annulées, l’exclusion des quantités négatives, l’ajout d’un jour à zéro, le classement des produits sans utiliser les jours après la date de sélection et l’association correcte entre fenêtre et cible.

`tests/test_ingestion.py` vérifie que l’absence du fichier produit une erreur claire, que le lecteur Excel est appelé avec le chemin fourni et que les données lues passent par la validation. La lecture Excel est simulée dans ces tests unitaires : ils ne dépendent donc pas du gros fichier réel.

Commande de test : `uv run pytest -q`. Dernière vérification de la suite complète avec `./.venv/bin/pytest -q` : **17 tests réussis** (9 validation, 3 ingestion, 5 préparation). Pandas affiche un avertissement non bloquant lors du test d’une date volontairement illisible.

### Étape 3 — Déplacer les architectures dans des modules — implémentée et testée

`src/retail_forecast/models.py` contient `SmallLSTM` et `SmallTransformer`, repris du prototype du notebook. Les deux modèles sont partagés entre les dix produits et prédisent une quantité pour le jour suivant. Ils ne chargent pas les données et ne réalisent pas l’entraînement.

Pour un batch de `B` exemples, chacun reçoit les ventes `x` de forme `(B, 100, 1)` et les indices entiers des produits `product_ids` de forme `(B,)`. L’embedding associe une représentation apprise à chaque indice produit. Chaque modèle retourne une prévision de forme `(B,)`. Le LSTM utilise une taille cachée de 16 ; le Transformer utilise un encodeur à un seul bloc avec `d_model=16`, deux têtes et un embedding produit de taille 4.

`tests/test_models.py` vérifie pour les deux architectures qu’un batch de deux exemples produit deux valeurs finies, qu’un batch de taille 1 conserve une sortie de forme `(1,)`, et qu’une séquence de longueur incorrecte est rejetée. Un test vérifie aussi le rejet d’un nombre d’indices produit différent du nombre d’exemples. Le code texte `StockCode` doit être converti en indice avant l’appel au modèle ; le refus d’un code produit inconnu sera testé dans le composant de mapping/inférence, pas dans le réseau lui-même.

Commande : `uv run pytest -q`. Dernière vérification de la suite complète avec `./.venv/bin/pytest -q` : **24 tests réussis**, avec le même avertissement Pandas non bloquant sur une date volontairement invalide.

### Étape 4 — Rendre l’entraînement et l’évaluation reproductibles

Cette étape entraîne les deux modèles de l’étape 3 sur les séquences produites à l’étape 2, puis les compare à une baseline simple. Elle ne sauvegarde pas encore de modèle sur disque et ne lance pas encore toute la chaîne depuis une commande unique.

`config.json` rassemble le fichier source, les paramètres de préparation, les dates du découpage et les paramètres d’entraînement. Les produits sont sélectionnés sur la période allant jusqu’au **31 août 2011** ; les cibles jusqu’au 31 août servent à l’entraînement, et celles du **1er au 30 septembre 2011** servent au test. Le modèle prédit toujours une cible à partir des 100 jours précédents. Une fenêtre de test peut donc contenir des jours antérieurs au test : c’est la date de la cible, et non celle du début de la fenêtre, qui décide du split.

`src/retail_forecast/training.py` réalise ce split chronologique, attribue un indice stable à chaque `StockCode`, puis entraîne le LSTM et le Transformer avec les paramètres de `config.json` (optimiseur Adam, erreur quadratique moyenne et graine définie). Il retourne les modèles, les indices train/test et l’historique de perte. Les deux modèles partagent leurs poids entre produits et reçoivent l’indice du produit en plus de sa séquence de ventes.

`src/retail_forecast/evaluation.py` calcule une baseline qui moyenne les ventes des quatre mêmes jours de semaine précédents (retards de 7, 14, 21 et 28 jours). Elle calcule la MAE et la RMSE pour la baseline et les modèles, globalement et par produit, et conserve une table des prédictions avec la date, le produit et la valeur réelle.

Les tests de `tests/test_training.py` vérifient que les dates limites séparent bien train et test et qu’un entraînement court des deux modèles fonctionne avec une correspondance produit-indice stable. Ceux de `tests/test_evaluation.py` vérifient le calcul de la baseline, les métriques sur un exemple connu et le rejet de tableaux de tailles incompatibles.

Pour lancer toute la suite :

```bash
uv run pytest -q
```

Résultat vérifié : **29 tests réussis**. Un avertissement Pandas non bloquant est attendu dans un test qui vérifie le rejet d’une date invalide. Ces tests incluent un entraînement miniature d’une époque ; ils ne prouvent pas encore qu’un entraînement complet sur le fichier réel a été exécuté. L’exécution complète et reproductible sera raccordée au pipeline dans une étape ultérieure.

### Étape 5 — Sauvegarder et recharger un modèle complet

Cette étape sépare l’apprentissage de l’utilisation du modèle. `training.py` produit les réseaux ajustés ; les modules de cette étape sauvegardent ce qu’il faut pour les réutiliser plus tard, sans garder les objets Python de l’entraînement en mémoire.

`src/retail_forecast/artifacts.py` contient `save_model_artifact` et `load_model_artifact`. Pour chaque modèle, la sauvegarde comprend :

- `model_state.pt` : les poids appris du réseau ;
- `manifest.json` : l’architecture et ses paramètres, la fenêtre de 100 jours, le mapping `StockCode → indice` et le contrat d’entrée/sortie.

Le contrat précise que l’entrée est constituée des ventes quotidiennes brutes, dans l’ordre chronologique, sans normalisation. Les poids sont chargés avec `weights_only=True`. L’artefact est un dossier contenant ces deux fichiers.

`src/retail_forecast/inference.py` contient `predict_next_day`. Cette fonction reçoit un artefact chargé, le code d’un produit et ses 100 dernières quantités journalières. Elle refuse un produit absent du mapping, un historique de mauvaise longueur ou des quantités négatives/non finies. Elle renvoie une prévision continue, ramenée à zéro si le réseau prédit une valeur négative ; elle ne l’arrondit pas.

`tests/test_artifacts.py` vérifie que :

- LSTM et Transformer donnent la même prédiction avant et après sauvegarde/rechargement ;
- le mapping produit est conservé et le modèle rechargé est en mode évaluation ;
- un produit inconnu, un historique trop court, une quantité négative ou non finie sont rejetés ;
- un artefact peut être chargé et utilisé dans un processus Python séparé.

Lancer la suite complète avec :

```bash
uv run pytest -q
```

Résultat vérifié : **34 tests réussis**. Un avertissement Pandas non bloquant reste associé au test d’une date invalide. Les tests d’artefacts utilisent de petits modèles ; ils valident la sauvegarde et le rechargement, mais ne signifient pas qu’un entraînement complet sur le fichier Excel a été exécuté. Le pipeline qui entraînera sur les données réelles sera construit à l’étape suivante.

### Étape 6 — Construire le pipeline d’entraînement

`src/retail_forecast/pipeline.py` enchaîne les modules précédents sans recopier leur logique : lecture et validation, agrégation quotidienne, création des fenêtres, entraînement, évaluation sur les seules cibles de test, puis sauvegarde des deux modèles avec les fonctions de l’étape 5. Si une étape lève une exception, elle remonte à l’appelant et les étapes suivantes ne sont pas lancées. Les modèles et les métadonnées sont sauvegardés après la réussite de l’évaluation.

`src/retail_forecast/train.py` est le point d’entrée en ligne de commande. Il lit `config.json`, résout le chemin des données relativement au fichier de configuration et crée un dossier de sortie horodaté sous `runs/`. On peut aussi choisir explicitement ce dossier avec `--output-dir`. Il affiche les métriques globales et les chemins des artefacts LSTM et Transformer.

`src/retail_forecast/run_metadata.py` ajoute les preuves de l’exécution dans ce dossier :

- `config.json` et une copie de `pyproject.toml`/`uv.lock` ;
- `run_manifest.json`, avec le SHA-256 et l’emplacement du fichier source, ses dimensions et colonnes, les dates, les versions logicielles et l’état Git ;
- `daily_sales.csv`, le snapshot compact des séries quotidiennes réellement préparées, avec son empreinte ; le gros fichier Excel n’est pas recopié ;
- `split.csv`, qui identifie chaque exemple par produit et date cible, ses 100 jours d’historique, sa cible et son appartenance au train, au test ou à une période exclue ;
- `metrics.json`, `test_predictions.csv` et `training_losses.json` ;
- les empreintes SHA-256 des manifestes et poids des deux modèles.

Le fichier source original doit rester disponible à l’emplacement indiqué et avoir la même empreinte pour reproduire l’exécution à partir des lignes brutes. Le snapshot quotidien sauvegardé permet toutefois de retrouver les séries agrégées qui ont alimenté la création des exemples.

`tests/test_pipeline.py` contient deux tests :

- le test nominal vérifie l’ordre ingestion → préparation → séquences → entraînement → évaluation → sauvegardes ; il vérifie aussi que l’évaluation ne reçoit que les indices de test ;
- le test d’échec simule une `DataValidationError` à l’ingestion et vérifie que préparation, entraînement, évaluation et sauvegarde ne sont pas appelés.

Lancer tous les tests depuis la racine du projet :

```bash
uv run pytest -q
```

Résultat vérifié : **38 tests réussis**, avec un avertissement Pandas non bloquant associé au test d’une date invalide.

Pour exécuter manuellement un entraînement complet sur le fichier réel, depuis la racine du projet :

```bash
PYTHONPATH=src uv run python -m retail_forecast.train \
  --config config.json \
  --output-dir runs/test-etape-6
```

Cette commande lit le fichier indiqué dans `config.json`, entraîne le LSTM et le Transformer selon les paramètres configurés (65 époques sur CPU par défaut), évalue leurs résultats sur septembre 2011 et écrit leurs artefacts dans `runs/test-etape-6/lstm` et `runs/test-etape-6/transformer`. `PYTHONPATH=src` rend le package `retail_forecast` importable lors de ce lancement ; le réglage `pythonpath = ["src"]` de `pyproject.toml` ne concerne que pytest. Le dossier d’entraînement réel n’a pas été généré pendant les tests unitaires : ceux-ci simulent les étapes coûteuses du pipeline.

### Étape 7 — Ajouter la traçabilité MLflow

#### Rôle de MLflow et de Docker

MLflow est lancé dans un conteneur Docker défini dans `compose.yml`, avec l’image
`ghcr.io/mlflow/mlflow:v3.16.1`. Docker doit être installé et démarré sur le poste pour utiliser
`docker compose`. Le serveur MLflow est accessible depuis le navigateur et depuis le programme
Python à l’adresse `http://localhost:5100`.

Seul le serveur MLflow tourne dans Docker pour cette étape : **l’entraînement reste exécuté sur le
poste avec `uv`**. Les données ne sont donc pas copiées dans le conteneur d’entraînement. Le volume
Docker `mlflow_data` conserve les métadonnées MLflow et les artefacts du serveur entre ses
redémarrages. Ne pas utiliser `docker compose down -v` si l’on veut conserver ces données.

#### Configuration locale

Le fichier `.env.example` fournit les URI de suivi et du registre. Pour créer la configuration
locale, ignorée par Git, copier ce fichier une seule fois depuis la racine du projet :

```bash
cp -n .env.example .env
```

`-n` évite d’écraser un éventuel `.env` déjà personnalisé. Compose lit ce fichier pour sa
configuration ; pour Python, utiliser `uv run --env-file .env ...`. Cette option transmet les
variables au processus sans commande `export` ou `source`. Ne jamais committer `.env` ; seul
`.env.example` doit être partagé.

#### Ce qui est enregistré pour un run

`src/retail_forecast/tracking.py` ouvre un run MLflow au début du pipeline. Après la préparation,
l’entraînement, l’évaluation et la création du manifeste local, le pipeline envoie au serveur :

- les paramètres de `config.json` sous forme de paramètres MLflow ;
- les métadonnées du jeu quotidien préparé, avec `mlflow.data.from_pandas` et `mlflow.log_input` ;
- le snapshot exact `daily_sales.csv`, le split, les prédictions, les métriques, le manifeste, la
  configuration, l’environnement et les artefacts locaux, avec `mlflow.log_artifacts` ;
- les MAE et RMSE du LSTM, du Transformer et de la baseline quatre semaines ;
- les deux modèles PyTorch dans le registre, sous les noms stables
  `retail-forecast-lstm` et `retail-forecast-transformer`.

Le `run_manifest.json` conserve les SHA-256 du fichier Excel source et du snapshot quotidien. Ces
empreintes identifient le contenu des fichiers ; le digest calculé par MLflow pour ses métadonnées
de dataset est une valeur distincte. Le fichier Excel original n’est pas téléversé par
`log_artifacts` : le snapshot quotidien effectivement utilisé, lui, est conservé comme artefact.

La qualification enregistrée est une règle pédagogique : `candidate` si la MAE est finie et
strictement meilleure que celle de la baseline ; sinon `rejected`. `candidate` ne signifie pas
« prêt pour la production ». Cette décision et sa politique sont également conservées dans les
informations de la version et dans le rapport du run.

#### Tests

Les tests unitaires simulent les fonctions MLflow : ils vérifient le déroulement et la gestion d’un
échec sans contacter le serveur ni créer de vrai run. Depuis la racine du projet :

```bash
uv run --locked pytest -q
```

#### Vérification réelle avec le serveur MLflow

1. Installer et démarrer Docker, puis vérifier les versions des commandes :

   ```bash
   docker --version
   docker compose version
   ```

2. Depuis la racine du projet, démarrer MLflow :

   ```bash
   docker compose up -d mlflow
   docker compose ps mlflow
   curl --fail http://localhost:5100/health
   ```

3. Vérifier que `.env` existe (le créer une seule fois depuis `.env.example` si nécessaire),
   vérifier que le client atteint le serveur, puis lancer l’entraînement réel :

   ```bash
   cp -n .env.example .env
   uv run --env-file .env --locked python -c "import mlflow; print(mlflow.__version__, mlflow.get_tracking_uri())"
   PYTHONPATH=src uv run --env-file .env --locked python -m retail_forecast.train \
     --config config.json \
     --output-dir runs/mlflow_step7_01
   ```

   Le fichier de configuration actuel demande 30 époques sur CPU. Le dossier de sortie est
   volontairement explicite et doit être nouveau pour chaque entraînement réel.

4. Ouvrir `http://localhost:5100`, choisir l’expérience `retail-forecast`, puis vérifier que le
   dernier run est `FINISHED` et contient les paramètres, métriques, données et artefacts listés
   ci-dessus. Dans le registre, vérifier les versions LSTM et Transformer ainsi que leurs tags de
   qualification. Une erreur de pipeline doit produire un run `FAILED`, et non un faux succès.

Chaque nouvelle exécution réelle ouvre un **nouveau run**. Si elle atteint l’étape d’enregistrement,
elle crée aussi une nouvelle version de chacun des modèles ; les versions précédentes restent
disponibles. Les appels `mlflow.log_*` et `mlflow.pytorch.log_model` sont les opérations qui
effectuent ces sauvegardes : le simple fait de démarrer MLflow ou de calculer un modèle ne les
enregistre pas automatiquement.

### Étape 8 — Orchestrer chaque étape avec Prefect

`src/retail_forecast/flows.py` expose le flow `retail-sales-training` et cinq tâches lisibles :
`ingestion-and-validation`, `prepare-daily-sales-cached`, `train-lstm-and-transformer`,
`evaluate-models`, puis `save-artifacts-and-publish-to-mlflow`. Elles appellent les fonctions métier
extraites dans `pipeline.py` : l’algorithme n’est donc pas recopié dans le flow. Leurs résultats
forment les dépendances et le pipeline reste séquentiel.

Le run MLflow est ouvert une seule fois pour le flow. Son `run_id` est transmis à la tâche de
publication et le `flow_run_id` Prefect est conservé comme tag MLflow. En cas d’échec terminal, le
flow marque ce même run MLflow `FAILED` puis laisse l’exception remonter. L’évaluation et la
publication n’ont aucun retry : les relancer pourrait refaire un entraînement ou créer une seconde
version de modèle.

#### Retry d’ingestion

Seule l’ingestion dispose de deux retries, soit au maximum trois tentatives, espacées de 2 puis
5 secondes. La condition `retry_transient_ingestion` accepte uniquement les erreurs réseau ou
d’E/S temporaires (`TimeoutError`, `ConnectionError` et quelques codes `OSError`). Un fichier
absent, une erreur de validation ou un classeur mal formé échoue sans retry. Le fichier utilisé
étant local, ce retry couvre surtout les incidents de lecture temporaires ; ce n’est pas un retry
d’appel à une API.

#### Cache de préparation

Seule la tâche pure `prepare-daily-sales-cached` est mise en cache. Sa clé contient l’empreinte
SHA-256 du fichier source, `top_n`, `selection_end`, `lookback` et la constante
`PREPARATION_VERSION`. Elle ne contient ni le `flow_run_id` ni un `run_id` MLflow : deux flows sur
le même contenu et les mêmes paramètres peuvent réutiliser ce calcul. Un changement de fichier,
paramètre ou version invalide la clé. Le calcul ne journalise rien dans MLflow ; chaque nouveau
flow continue donc à entraîner et publier normalement.

Prefect stocke les résultats de cette tâche dans `.prefect-cache/task-results` sur le poste, pas dans
le conteneur du serveur. Le dossier est ignoré par Git. Au début du flow,
`ensure_cache_storage_block()` vérifie que le block `local-file-system/retail-forecast-cache` existe
et le crée si nécessaire. Cette vérification se fait avant l’ingestion : un problème de stockage
échoue rapidement et ne lit pas inutilement le gros fichier Excel. Le serveur Prefect doit donc
être joignable dès le début du flow.

`tests/test_flows.py` vérifie la chaîne des cinq tâches, la propagation d’un échec, le filtrage des
retries et l’invalidation de la clé de cache selon le contenu source, les paramètres et la version
de préparation. Les tests n’appellent ni services distants ni entraînement réel.

#### Tester Prefect réellement

Depuis la racine de ce projet, démarrer le serveur Prefect et MLflow. Le Compose publie Prefect sur
le port hôte `4300` et MLflow sur `5100` :

```bash
docker compose up -d --wait prefect mlflow
docker compose ps prefect mlflow
```

Les URI Prefect et MLflow sont déjà définies dans le fichier local `.env`. Le shell ne charge pas
automatiquement ce fichier ; `uv run --env-file .env` transmet ses variables au processus Python.
Il n’est donc pas nécessaire de lancer des commandes `export`.

Lancer d’abord les tests unitaires, puis tous les tests :

```bash
uv run --env-file .env --locked pytest -q tests/test_flows.py
uv run --env-file .env --locked pytest -q
```

L’enregistrement du block est automatique au premier flow. Pour le créer explicitement avant le
lancement (facultatif), cette commande nécessite le serveur Prefect actif :

```bash
PYTHONPATH=src uv run --env-file .env --locked python -m retail_forecast.prefect_setup
```

Pour un entraînement réel, avec le fichier indiqué dans `config.json` présent :

```bash
PYTHONPATH=src uv run --env-file .env --locked python -m retail_forecast.train \
  --config config.json \
  --output-dir runs/prefect_step8_02
```

`PYTHONPATH=src` rend le package `retail_forecast` importable depuis le dossier `src`. Cette
commande exécute le flow dans le processus Python local ; elle n’envoie pas l’entraînement dans le
conteneur Prefect. Ouvrir `http://localhost:4300` et vérifier un flow `retail-sales-training` et
ses cinq tasks dans l’ordre. Une première exécution doit calculer la préparation ; une seconde
exécution avec le même fichier et les mêmes paramètres doit afficher la tâche de préparation comme
mise en cache, mais réentraîne et publie quand même les modèles. Utiliser un nouveau dossier de
sortie pour chaque entraînement. Ouvrir ensuite `http://localhost:5100` pour vérifier le run MLflow,
son tag `prefect.flow_run_id` et les modèles publiés.

Pour observer un échec contrôlé sans toucher au fichier de configuration réel : copier
`config.json` vers `/tmp/prefect-fail.json`, puis modifier dans cette copie `data.current_file` pour
indiquer `/tmp/fichier-qui-nexiste-pas.xlsx`. Lancer ensuite :

```bash
PYTHONPATH=src uv run --env-file .env --locked python -m retail_forecast.train \
  --config /tmp/prefect-fail.json \
  --output-dir /tmp/prefect-step8-failure
```

La tâche et le flow doivent apparaître en échec dans Prefect. L’ingestion étant invalide, aucun
modèle ne doit être publié ; le run MLflow doit conserver l’échec du pipeline.

#### Déclencher depuis l’interface Prefect

Le lancement précédent est suivi par Prefect mais est demandé depuis le terminal. Le fichier
`src/retail_forecast/deploy.py` ajoute un **déploiement local** : il enregistre le flow auprès du
serveur et garde un processus `serve` disponible pour exécuter les demandes créées depuis
l’interface Prefect. Il ne contient pas une seconde version du pipeline : il importe simplement
`training_flow`.

Dans un terminal dédié, depuis la racine du projet, avec les services Docker déjà démarrés :

```bash
PYTHONPATH=src uv run --env-file .env --locked python -m retail_forecast.deploy \
  --config config.json
```

Cette commande ne lance pas immédiatement un entraînement. Elle crée ou actualise le déploiement
`retail-sales-training-local`, puis reste active. Tant que ce terminal est ouvert, aller sur
`http://localhost:4300`, ouvrir ce déploiement et utiliser **Run**. Prefect envoie alors la demande
au processus `serve`, qui démarre un sous-processus local pour exécuter le flow. Le calcul reste
donc sur le PC ; le conteneur Prefect coordonne et conserve l’historique.

Le paramètre par défaut `config_path` est un chemin absolu vers `config.json`. Il est modifiable dans
l’interface avant le lancement. Aucun `output_dir` par défaut n’est imposé : chaque run crée son
propre dossier horodaté sous `runs/`, ce qui évite d’écraser les artefacts antérieurs. La limite
`limit=1` empêche ce processus `serve` d’exécuter deux entraînements simultanément.

Pour ajouter une planification locale de test toutes les 15 minutes, arrêter le processus précédent
avec `Ctrl+C`, puis relancer :

```bash
PYTHONPATH=src uv run --env-file .env --locked python -m retail_forecast.deploy \
  --config config.json \
  --interval-minutes 15
```

Le planning reste actif tant que `serve` tourne. Pour éviter des entraînements inutiles, commencer
sans planning et déclencher une seule exécution depuis l’interface. Les tests de
`tests/test_deploy.py` vérifient la configuration envoyée à `serve`, le chemin absolu, la limite de
concurrence et le planning optionnel sans contacter un serveur Prefect :

```bash
uv run --env-file .env --locked pytest -q tests/test_deploy.py
```

Dernière suite exécutée après le découpage en tâches, l'ajout du précontrôle de cache et du
déploiement local : **54 tests réussis**. Un avertissement Pandas
non bloquant est attendu dans le test d’une date invalide. Les tests vérifient le prédicat de retry
et la clé de cache ; le cache persistant et l’interface doivent aussi être vérifiés avec le serveur
Prefect démarré. Pour tester l’invalidation manuelle, relancer avec un autre fichier source ou
modifier un paramètre de préparation dans une copie de configuration ; garder un nouveau dossier de
sortie, car un flow réussi publie toujours une nouvelle version MLflow.

### Étape 9 — Créer l’API de prédiction et de réception des ventes

Créer `api/main.py` et `api/schemas.py` pour exposer la prédiction unitaire et par lot, ainsi qu’une route de réception des ventes PUSH. Les modèles seront chargés une fois au démarrage du service. Ajouter des tests unitaires de validation des schémas et des routes avec un modèle simulé ; vérifier que les lots invalides ne produisent aucune prédiction partielle.

### Étape 10 — Persister les ventes, prédictions et feedbacks

Créer `api/storage.py` et les migrations SQL nécessaires pour enregistrer les requêtes, les nouvelles ventes, les prédictions, les versions de modèles utilisées et les feedbacks. Ajouter des tests du dépôt sur une base de test pour l’idempotence, les doublons, les transactions et les échecs d’insertion ; vérifier également que les écritures sont annulées ensemble en cas d’erreur.

### Étape 11 — Simuler les arrivées PULL/PUSH

Créer `scripts/replay_sales.py` pour rejouer séparément les jours du fichier réservé à cette démonstration. Chaque jour sera envoyé ou récupéré chronologiquement ; le modèle prédira avec les seules observations déjà reçues. Ce script jouera le rôle du producteur de données et ne fera pas partie de l’entraînement initial. Ajouter des tests unitaires vérifiant l’ordre des dates, le format des messages et l’absence de retry qui dupliquerait une arrivée ; simuler les appels réseau dans ces tests.

### Étape 12 — Mesurer la capacité et superviser l’API

Ajouter des scénarios Locust pour mesurer latence, débit et erreurs selon le nombre de workers. Tester le format et le comportement des scénarios Locust avec un client simulé, puis réaliser des essais de charge contrôlés pour mesurer la capacité. Construire ensuite un dashboard Grafana basé sur les traces stockées, avec les performances, versions servies, feedbacks et erreurs ; vérifier les requêtes SQL et les panneaux sur une petite campagne connue.

### Étape 13 — Vérifier le parcours complet et documenter l’exploitation

Compléter les tests de bout en bout — en plus des tests unitaires précédents — et les essais d’incident : données invalides, base indisponible, modèle inconnu et redémarrage. Documenter l’installation, les commandes, le rejeu des données, le déploiement local, les limites et les résultats de charge dans ce README.
