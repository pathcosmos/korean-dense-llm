# Korean Dense LLM — Français

[한국어](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ko.md) · [English](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.en.md) · [简体中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-CN.md) · [繁體中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-TW.md) · [日本語](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ja.md) · [Français](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.fr.md) · [Deutsch](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.de.md) · [Español](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.es.md) · [Português (Brasil)](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.pt-BR.md) · [हिन्दी](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.hi.md) · [العربية](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ar.md) · [עברית](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.he.md) · [Русский](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ru.md)

## Présentation

Ce projet publie le code de préparation des données, du modèle, de l’entraînement, de l’évaluation, de la reprise et des recherches d’optimisation d’un LLM dense centré sur le coréen. Cette édition contient les sources et les expériences ; textes, tokens binaires, bases et poids sont distribués séparément.

La langue de traduction n’indique pas la présence de données d’entraînement dans cette langue. Les langues principales sont le coréen et l’anglais ; web, mathématiques et code peuvent en contenir d’autres.

## Contenu et volume

| | |
|---|---:|
| common/, data/ | Traitement et séparation |
| tokenizer/ | Traitement et séparation |
| model/, train/, eval/ | Modèle et entraînement |
| scripts/ | Optimisation et mesures |
| tests/, examples/ | Vérification |
| experiments/ | 47 / 99 |


## Traitement et séparation

Le web subit nettoyage des espaces, heuristiques de longueur/répétition/langue et masquage de certains motifs de données personnelles. Le hash initial interne aux fragments diffère du SHA256 NFC final. Le code conserve casse, indentation et Unicode original ; chemins, licences déclarées, code généré et certains secrets sont filtrés. Les groupes connectés de textes identiques, de textes web équivalents en NFC et de dépôts de code sont répartis en train/eval.

## Utilisation

Exécuter les commandes à la racine du dépôt. Utiliser de nouveaux dossiers et vérifier périmètre, taille et revision figée. L’exemple CPU synthétique ne télécharge pas de données externes. Choisir les fichiers nécessaires dans les distributions séparées et vérifier taille/SHA256 avec CHECKSUMS.json ou les manifests.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python examples/quickstart_cpu.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python -m pytest -q tests
```


## Modèle et entraînement

Le décodeur dense PyTorch utilise RMSNorm, RoPE, GQA et SwiGLU. Le parcours3B réel emploie FSDP2,8GPU et eager : contexte4096, micro batch4, batch global2,097,152tokens, pic LR3e-4, cible57,000steps. La démonstration CPU micro diffère du distribué. Une reprise exacte exige modèle, optimizer, RNG et contrats des entrées/configurations.

[examples/3b.observed.json](https://github.com/pathcosmos/korean-dense-llm/blob/main/examples/3b.observed.json) · [examples/3b.fresh.json](https://github.com/pathcosmos/korean-dense-llm/blob/main/examples/3b.fresh.json)

```bash
python scripts/hardware_profile.py --output runs/hardware.json
torchrun --standalone --nproc_per_node=8 --module train.deterministic_train --config examples/3b.fresh.json
```

## Optimisation et mesures

Les essais comparent workers, threads, batches, cache SQLite, RAM, stockage local/NFS, eager/compile, précision et mémoire GPU. Temps total, débit, pic mémoire et équivalence de contenu/numérique accompagnent les décisions d’adoption ou de rejet :47 campagnes et99 JSON. Une accélération ne valide pas un candidat numériquement incorrect ; un petit échantillon ne prouve pas l’optimum global.

[47 / 99](https://github.com/pathcosmos/korean-dense-llm/blob/main/experiments/README.md)

## Vérification

La version source GH compte284 tests CPU réussis/9 ignorés ; le nouvel environnement CPU GitHub en compte258/35. Les exclusions nécessitent des traces originales ou un second système de fichiers. Le quickstart synthétique utilise120documents, train100/eval20, vocab272 et108,864paramètres et vérifie la reprise exacte du modèle/optimizer. Hash, séparation et reader des données ne prouvent pas une reproduction3B indépendante complète.

## Sources et licences

Le code original du projet est MIT, le texte explicatif original CC BY4.0. Ces droits ne remplacent pas ceux des contenus sources. Consulter source.json, LICENSE-SCOPE.md et les documents de provenance pour conditions, attributions, notices et points non résolus. Aucune redistribution commerciale illimitée de chaque texte original n’est affirmée.

[NOTICE.md](https://github.com/pathcosmos/korean-dense-llm/blob/main/NOTICE.md) · [LICENSE](https://github.com/pathcosmos/korean-dense-llm/blob/main/LICENSE)

## Limites

La déduplication approximative globale, l’élimination totale des données personnelles/secrets, l’absence de contamination et la qualité finale ne sont pas garanties. L’entraînement principal était en cours au snapshot ; SFT, export et évaluations linguistiques officielles restent inachevés.11 références de code historiques/9hash uniques n’ont pas de snapshot retrouvé. Certains outils exigent les manifests originaux.

## Dépôts associés

[GitHub](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.fr.md) · [README](https://github.com/pathcosmos/korean-dense-llm)
[GH / Utilisation](https://github.com/pathcosmos/korean-dense-llm/blob/main/docs/PUBLIC_GUIDE.md) · [GH / Optimisation et mesures](https://github.com/pathcosmos/korean-dense-llm/blob/main/experiments/README.md) · [GH / Vérification](https://github.com/pathcosmos/korean-dense-llm/blob/main/docs/PUBLIC_VALIDATION.md)

- [korean-pretrain-corpus](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus)
- [korean-pretrain-training-data](https://huggingface.co/datasets/pathcosmos/korean-pretrain-training-data)
- [korean-pretrain-wikipedia](https://huggingface.co/datasets/pathcosmos/korean-pretrain-wikipedia)
- [korean-pretrain-fineweb2](https://huggingface.co/datasets/pathcosmos/korean-pretrain-fineweb2)
- [korean-pretrain-hplt2](https://huggingface.co/datasets/pathcosmos/korean-pretrain-hplt2)
- [korean-pretrain-fineweb-edu](https://huggingface.co/datasets/pathcosmos/korean-pretrain-fineweb-edu)
- [korean-pretrain-math](https://huggingface.co/datasets/pathcosmos/korean-pretrain-math)
- [korean-pretrain-code](https://huggingface.co/datasets/pathcosmos/korean-pretrain-code)

- [wiki_ko/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/wiki_ko)
- [fineweb2_ko/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/fineweb2_ko)
- [hplt_ko/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/hplt_ko)
- [fineweb_edu/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/fineweb_edu)
- [owm/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/owm)
- [github_code_v1/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/github_code_v1)

[한국어](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ko.md) · [English](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.en.md) · [简体中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-CN.md) · [繁體中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-TW.md) · [日本語](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ja.md) · [Français](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.fr.md) · [Deutsch](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.de.md) · [Español](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.es.md) · [Português (Brasil)](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.pt-BR.md) · [हिन्दी](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.hi.md) · [العربية](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ar.md) · [עברית](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.he.md) · [Русский](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ru.md)
