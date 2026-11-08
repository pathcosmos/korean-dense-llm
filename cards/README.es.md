# Korean Dense LLM — Español

[한국어](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ko.md) · [English](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.en.md) · [简体中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-CN.md) · [繁體中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-TW.md) · [日本語](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ja.md) · [Français](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.fr.md) · [Deutsch](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.de.md) · [Español](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.es.md) · [Português (Brasil)](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.pt-BR.md) · [हिन्दी](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.hi.md) · [العربية](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ar.md) · [עברית](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.he.md) · [Русский](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ru.md)

## Descripción

Este proyecto publica el código de preparación de datos, modelo, entrenamiento, evaluación, recuperación y optimización de un LLM denso centrado en coreano. La versión contiene fuentes y registros de experimentos; textos, tokens binarios, bases y pesos se distribuyen por separado.

El idioma de traducción no implica datos de entrenamiento en ese idioma. Predominan coreano e inglés; web, matemáticas y código pueden contener otros.

## Contenido y escala

| | |
|---|---:|
| common/, data/ | Procesamiento y división |
| tokenizer/ | Procesamiento y división |
| model/, train/, eval/ | Modelo y entrenamiento |
| scripts/ | Optimización y mediciones |
| tests/, examples/ | Validación |
| experiments/ | 47 / 99 |


## Procesamiento y división

La web pasa por limpieza de espacios, heurísticas de longitud/repetición/idioma y enmascaramiento de algunos patrones personales. El hash inicial por fragmento difiere del SHA256 NFC final. El código conserva mayúsculas, sangría y Unicode original; se filtran rutas, etiquetas de licencia, código generado y algunos secretos. Grupos conectados de texto idéntico, web equivalente en NFC y repositorios de código se asignan a train/eval.

## Uso

Ejecutar desde la raíz del repositorio. Usar directorios nuevos y comprobar alcance, tamaño y revision fijada. El ejemplo CPU sintético no descarga datos externos. Elegir los archivos necesarios de las publicaciones separadas y verificar tamaño/SHA256 con CHECKSUMS.json o manifests.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python examples/quickstart_cpu.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python -m pytest -q tests
```


## Modelo y entrenamiento

El decodificador denso PyTorch emplea RMSNorm, RoPE, GQA y SwiGLU. La ruta3B real usa FSDP2,8GPU y eager: contexto4096, micro batch4, batch global2,097,152tokens, LR máximo3e-4 y objetivo57,000pasos. El ejemplo CPU micro difiere del entrenamiento distribuido; la reanudación exacta exige modelo, optimizer, RNG y contratos de entrada/configuración.

[examples/3b.observed.json](https://github.com/pathcosmos/korean-dense-llm/blob/main/examples/3b.observed.json) · [examples/3b.fresh.json](https://github.com/pathcosmos/korean-dense-llm/blob/main/examples/3b.fresh.json)

```bash
python scripts/hardware_profile.py --output runs/hardware.json
torchrun --standalone --nproc_per_node=8 --module train.deterministic_train --config examples/3b.fresh.json
```

## Optimización y mediciones

Se comparan workers, threads, batch, caché SQLite, RAM, local/NFS, eager/compile, precisión y memoria GPU. Tiempo total, rendimiento, pico de memoria y equivalencia de contenido/números sustentan cada adopción o rechazo:47 campañas y99 JSON. Ser más rápido no justifica fallos numéricos y una muestra pequeña no prueba el óptimo global.

[47 / 99](https://github.com/pathcosmos/korean-dense-llm/blob/main/experiments/README.md)

## Validación

La versión fuente GH pasó284 pruebas CPU con9 omitidas; GitHub CPU nuevo pasó258 con35 omitidas. Se necesitan registros originales o un segundo sistema de archivos para las omitidas. El quickstart sintético usa120documentos, train100/eval20, vocab272 y108,864parámetros, verificando reanudación exacta de modelo/optimizer. Hash, división y reader de datos no equivalen a una reproducción3B independiente completa.

## Fuentes y licencias

El código original es MIT y la explicación original CC BY4.0. No sustituyen los derechos del contenido fuente. Consultar source.json, LICENSE-SCOPE.md y procedencia para condiciones, atribución, avisos y asuntos pendientes. No se afirma redistribución comercial ilimitada de todos los textos.

[NOTICE.md](https://github.com/pathcosmos/korean-dense-llm/blob/main/NOTICE.md) · [LICENSE](https://github.com/pathcosmos/korean-dense-llm/blob/main/LICENSE)

## Limitaciones

No se garantizan deduplicación aproximada global, eliminación total de PII/secretos, evaluación sin contaminación ni calidad final. El entrenamiento principal seguía activo en el snapshot; SFT, exportación y pruebas lingüísticas oficiales no están completados.11 referencias históricas/9hash únicos carecen de snapshot recuperado. Algunas herramientas requieren manifests originales.

## Repositorios relacionados

[GitHub](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.es.md) · [README](https://github.com/pathcosmos/korean-dense-llm)
[GH / Uso](https://github.com/pathcosmos/korean-dense-llm/blob/main/docs/PUBLIC_GUIDE.md) · [GH / Optimización y mediciones](https://github.com/pathcosmos/korean-dense-llm/blob/main/experiments/README.md) · [GH / Validación](https://github.com/pathcosmos/korean-dense-llm/blob/main/docs/PUBLIC_VALIDATION.md)

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
