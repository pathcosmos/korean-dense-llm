# Korean Dense LLM — Deutsch

[한국어](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ko.md) · [English](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.en.md) · [简体中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-CN.md) · [繁體中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-TW.md) · [日本語](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ja.md) · [Français](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.fr.md) · [Deutsch](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.de.md) · [Español](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.es.md) · [Português (Brasil)](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.pt-BR.md) · [हिन्दी](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.hi.md) · [العربية](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ar.md) · [עברית](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.he.md) · [Русский](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ru.md)

## Überblick

Dieses Projekt veröffentlicht Datenvorbereitung, Modell-, Trainings-, Evaluations-, Wiederherstellungs- und Optimierungscode für ein koreanisch zentriertes Dense LLM. Die Ausgabe enthält Quellcode und Versuchsberichte; Korpustexte, Tokenbinärdateien, Datenbanken und Gewichte werden separat bereitgestellt.

Übersetzungssprachen bedeuten keine entsprechenden Trainingsdaten. Hauptsprachen sind Koreanisch und Englisch; Web-, Mathematik- und Codeinhalte können weitere Sprachen enthalten.

## Inhalt und Umfang

| | |
|---|---:|
| common/, data/ | Verarbeitung und Aufteilung |
| tokenizer/ | Verarbeitung und Aufteilung |
| model/, train/, eval/ | Modell und Training |
| scripts/ | Optimierung und Benchmarks |
| tests/, examples/ | Validierung |
| experiments/ | 47 / 99 |


## Verarbeitung und Aufteilung

Webverarbeitung bereinigt Leerraum, nutzt Längen-/Wiederholungs-/Sprachheuristiken und maskiert ausgewählte personenbezogene Muster. Der anfängliche shardinterne Hash unterscheidet sich vom abschließenden NFC-SHA256. Code behält Großschreibung, Einrückung und ursprüngliches Unicode; Pfade, Lizenzangaben, generierter Code und ausgewählte Geheimnismuster werden geprüft. Verbundene Gruppen gleicher Texte, NFC-äquivalenter Webtexte und Code-Repositories werden train/eval zugeteilt.

## Verwendung

Die Befehle im Repository-Hauptverzeichnis ausführen. Neue Ausgabe-/Downloadordner verwenden und Umfang, Größe sowie fixierte Revision prüfen. Das synthetische CPU-Beispiel lädt keine externen Daten. Benötigte Dateien aus separaten Veröffentlichungen wählen und Größe/SHA256 anhand CHECKSUMS.json oder Quellmanifests prüfen.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python examples/quickstart_cpu.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python -m pytest -q tests
```


## Modell und Training

Der reine PyTorch-Dense-Decoder verwendet RMSNorm, RoPE, GQA und SwiGLU. Der reale3B-Pfad nutzt FSDP2,8GPUs und eager: Kontext4096, Microbatch4, global2,097,152Tokens, Spitzen-LR3e-4 und Ziel57,000Schritte. Das CPU-Microbeispiel unterscheidet sich vom verteilten Training. Exakte Wiederaufnahme verlangt Modell, Optimizer, RNG und Eingabe-/Konfigurationsverträge.

[examples/3b.observed.json](https://github.com/pathcosmos/korean-dense-llm/blob/main/examples/3b.observed.json) · [examples/3b.fresh.json](https://github.com/pathcosmos/korean-dense-llm/blob/main/examples/3b.fresh.json)

```bash
python scripts/hardware_profile.py --output runs/hardware.json
torchrun --standalone --nproc_per_node=8 --module train.deterministic_train --config examples/3b.fresh.json
```

## Optimierung und Benchmarks

Verglichen werden Worker, Threads, Batches, SQLite-Cache, RAM, lokal/NFS, eager/compile, Präzision und GPU-Speicher. Gesamtzeit, Durchsatz, Speicherspitze und Inhalts-/numerische Gleichwertigkeit begründen Annahme oder Ablehnung:47 Kampagnen,99 Ergebnis-JSONs. Geschwindigkeit rechtfertigt keinen numerisch fehlerhaften Kandidaten; kleine Stichproben beweisen kein Gesamtoptimum.

[47 / 99](https://github.com/pathcosmos/korean-dense-llm/blob/main/experiments/README.md)

## Validierung

Die GH-Quellversion bestand284 CPU-Tests bei9 Überspringungen; frische GitHub-CPU-CI bestand258 bei35 Überspringungen. Gründe sind fehlende Originalprotokolle oder ein zweites Dateisystem. Der synthetische Quickstart nutzt120Dokumente, train100/eval20, Vokabular272 und108,864Parameter und prüft exakte Modell-/Optimizer-Wiederaufnahme. Datenhashes, Splits und Readerprüfungen sind keine vollständige unabhängige3B-Reproduktion.

## Quellen und Lizenzen

Originaler Projektcode steht unter MIT, eigener Erklärungstext unter CC BY4.0. Dies ersetzt nicht die Rechte an Quelldaten. source.json, LICENSE-SCOPE.md und Herkunftsdokumente nennen Bedingungen, Zuschreibungen, Hinweise und offene Punkte. Eine unbeschränkte kommerzielle Weitergabe aller Originaltexte wird nicht behauptet.

[NOTICE.md](https://github.com/pathcosmos/korean-dense-llm/blob/main/NOTICE.md) · [LICENSE](https://github.com/pathcosmos/korean-dense-llm/blob/main/LICENSE)

## Grenzen

Globale Fast-Duplikatentfernung, vollständige PII-/Geheimnisentfernung, kontaminationsfreie Bewertung und finale Qualität werden nicht garantiert. Haupttraining lief beim Snapshot; SFT, Export und offizielle Sprachbenchmarks sind nicht abgeschlossen.11 historische Codeverweise/9einzigartige Hashes haben keinen wiedergefundenen Snapshot. Einige Betriebswerkzeuge benötigen Originalmanifests.

## Verwandte Repositories

[GitHub](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.de.md) · [README](https://github.com/pathcosmos/korean-dense-llm)
[GH / Verwendung](https://github.com/pathcosmos/korean-dense-llm/blob/main/docs/PUBLIC_GUIDE.md) · [GH / Optimierung und Benchmarks](https://github.com/pathcosmos/korean-dense-llm/blob/main/experiments/README.md) · [GH / Validierung](https://github.com/pathcosmos/korean-dense-llm/blob/main/docs/PUBLIC_VALIDATION.md)

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
