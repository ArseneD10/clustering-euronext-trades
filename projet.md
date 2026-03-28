L'objectif de ce projet est d'utiliser les flow de trades afin de trouver des relations entre plusieurs actifs, l'idée est d'entrainer un encoder \
sur plusieurs tâches et d'utiliser ensuite les embeddings produits pour chaque entrée afin de determiner différent régime de marché 

J'ai à ma disposition un fichier csv comportant tout les trades sur l'euronext du 10 mars 2026 au 20 mars 2026, je commencerais donc par lire, filtrer et traiter ces données, le nombre de données étant grand j'utiliserais la lib polars qui est optimisé pour ce type de traitement.

Nous testerons dans cette étude une approche basé sur un modèle multi-task, nous décrirons ci-dessous l'architecture ainsi que les tâches qui lui seront demandé : 
Le but principal de l'étude est de determiner si le profil des trades non-reliés au prix de manière direct a un impact sur la volatilité,
en d'autres termes l'objectif est de regarder si l'on peut anticiper la volatilité sans utiliser les variations sur prix donc trouver une fonction de 
la volatilité tel que :
sigma = f(Ω) | Ω ¬ Rt (Rt = Retour à t)

*** ARCHITECTURE ***

(1) == Categorical Encoder (Titre de la companie, Trade type...) == || == Feature Encoder (Log Volume, Temps entre deux trades...) ==
(2) == Encoder Block (Architecture transformer -> Attention résiduelle + mlp) == 

(3) == Tasking Head (Tête de prédiction spécifique pour chaque tâche) ==

*** TASK ***

(A) Variational Auto-Encoder -> Nous passerons dans le modèle des sequences de trades qu'il devra compresser puis restituer,
    l'objectif ici est d'apprendre une représentation 'normal' en utilisant des relations non-lineaire entre les features d'entrée
    des trades selon chaque actifs (que le modèle pourra adapter avec l'embedding du Titre de la companie) 

(B) Volatility Estimator -> Nous passerons dans le modèle des features n'ayant a priori pas de rapport direct avec le prix (Nom, Volume, Delta de temps ...)
    en utilisant ces features le modèle devra prédire l'écart-type des rendements sur la période selectionné. 
    Nous cherchons donc ici à estimer la Volatilité V comme suit :
    V = f(Ω) | Ω -> Features d'entrées

(*) (C) Directionnal -> Ici le but sera de prédire la distribution des rendements sur la première période, pour cette tâche nous utiliserons des encoder
    issues des deux taches précedentes où nous gèlerons ces paramètres, nous testerons aussi en ce modèle en utilisant un encoder vierge afin de voir si 
    l'utilisation d'un modèle multi-task apporte des plus-values


*** APPLICATION ***

(A) Clustering d'asset via l'embedding de titres (K-means ou autres sur les embedding des titres de compagnies)
(B) Étude des corrélation résidus <-> volatilités t+1
(C) Étude sur la concentration des gradients dans certaines zones du modèle après certains ajustements


*** Organisation ***
src:
    analysis.py: analyses des résidus des erreurs du modèles ainsi que la magnitude des gradients sur les premières couches du modèles
    base_encoder.py: modèle commun d'encodage des features d'entrées 
    clustering.py: clustering des embeddings de l'encodeurs et analyse + visualisation 
    config.py: configuration des modèles + optimizers
    directional_nn.py: tête de prédiction de volatilité t+1
    processing.py: pipeline pré-training
    trainer.py: class trainer pour les modèles
    vae.py: Variational Auto-Encodeur 
    volatility_nn.py: tête de prédiction pour la volatilité t

notebook.ipynb: notebook avec tout le pipeline de recherche 

*** WorkFlow ***

Dans notebook.ipynb j'ai lancé l'entrainement des modèles ainsi que le chargement des données, 

*** RESULTATS ***

Note: J'ai testé de remplacer la prédiction brut de la volatilité par une prédiction sous forme V = µ + sigma * epsilon, cette version s'est montré
instable, ce qui m'as amené à garder la version brut dans le modèle final

To do -->  gradients analysis ---