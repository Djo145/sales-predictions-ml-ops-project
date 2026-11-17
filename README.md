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
| `UnitPrice` | Facultatif ; les valeurs absentes sont acceptées. Si renseigné, le prix doit être numérique, fini et supérieur ou égal à zéro. |
| `Description`, `CustomerID`, `Country` | Facultatives ; leurs valeurs absentes sont autorisées. |

Les tests de `tests/test_validation.py` couvrent les données conformes, les colonnes ou valeurs manquantes, les identifiants vides, les quantités non entières, les dates illisibles et les prix invalides. Résultat vérifié : **9 tests réussis** avec `uv run pytest -q`.

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

Créer `src/retail_forecast/training.py` pour entraîner les modèles avec la configuration du projet et `src/retail_forecast/evaluation.py` pour calculer la baseline, la MAE et la RMSE, globalement et par produit. Mettre les dates, la fenêtre de 100 jours et les hyperparamètres dans `config.json`. Ajouter des tests pour le split chronologique et les métriques.

### Étape 5 — Sauvegarder et recharger un modèle complet

Créer `src/retail_forecast/artifacts.py` pour conserver les poids, l’architecture, le dictionnaire `StockCode → indice`, la normalisation et le contrat d’entrée. Créer `src/retail_forecast/inference.py` pour charger cet ensemble et prévoir le lendemain sans relancer l’entraînement. Tester le rechargement dans un nouveau processus et vérifier que les prédictions correspondent.

### Étape 6 — Construire le pipeline d’entraînement

Créer `src/retail_forecast/pipeline.py` pour enchaîner acquisition, validation, préparation, entraînement et évaluation. Garder `train.py` comme point d’entrée simple. Le pipeline devra échouer avant l’entraînement si la validation échoue. Ajouter des tests unitaires vérifiant l’ordre des étapes et prouvant qu’une validation rejetée empêche entraînement et évaluation.

### Étape 7 — Ajouter la traçabilité MLflow

Créer `src/retail_forecast/tracking.py` pour enregistrer les données et leur empreinte, la configuration, les métriques, les artefacts et les versions des modèles. Définir une règle explicite de qualification et vérifier qu’une ancienne version peut être retrouvée et rechargée. Ajouter des tests unitaires avec un client MLflow simulé pour vérifier les informations journalisées et les erreurs ; compléter par un test d’intégration contre le serveur MLflow local.

### Étape 8 — Orchestrer le pipeline avec Prefect

Créer `src/retail_forecast/flows.py` pour suivre les étapes du pipeline, leurs durées et leurs erreurs. Configurer des reprises limitées pour les pannes temporaires, sans rejouer automatiquement une donnée invalide ni créer une publication MLflow en double. Ajouter des tests unitaires des tâches et des règles de retry, puis vérifier un flow nominal et un échec contrôlé avec le serveur Prefect.

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
