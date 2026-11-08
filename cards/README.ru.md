# Korean Dense LLM — Русский

[한국어](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ko.md) · [English](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.en.md) · [简体中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-CN.md) · [繁體中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-TW.md) · [日本語](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ja.md) · [Français](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.fr.md) · [Deutsch](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.de.md) · [Español](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.es.md) · [Português (Brasil)](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.pt-BR.md) · [हिन्दी](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.hi.md) · [العربية](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ar.md) · [עברית](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.he.md) · [Русский](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ru.md)

## Обзор

Проект публикует код подготовки данных, модели, обучения, оценки, восстановления и оптимизации плотной LLM с упором на корейский язык. Выпуск содержит исходники и эксперименты, без текстов, бинарных токенов, баз и весов модели. Отдельные публикации данных HF доступны по ссылкам ниже.

Язык перевода не означает наличие обучающих данных на нём. Основные языки —корейский и английский; веб, математика и код могут содержать другие.

## Состав и объём

| | |
|---|---:|
| common/, data/ | Обработка и разделение |
| tokenizer/ | Обработка и разделение |
| model/, train/, eval/ | Модель и обучение |
| scripts/ | Оптимизация и измерения |
| tests/, examples/ | Проверка |
| experiments/ | 47 / 99 |


## Обработка и разделение

Веб-обработка включает очистку пробелов, эвристики длины/повторов/языка и маскирование отдельных персональных шаблонов. Начальный хеш внутри шарда отличается от итогового NFC SHA256. Код сохраняет регистр, отступы и исходный Unicode; проверяются пути, лицензии, сгенерированный код и отдельные секреты. Связные группы одинаковых текстов, NFC-эквивалентных веб-текстов и кодовых репозиториев делятся на train/eval.

## Использование

Команды выполняются из корня репозитория. Используйте новые папки, проверьте объём и фиксированную revision. Синтетический CPU-пример не загружает внешние данные. Выберите нужные файлы в отдельных публикациях и сверьте размер/SHA256 с CHECKSUMS.json или manifest.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python examples/quickstart_cpu.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python -m pytest -q tests
```


## Модель и обучение

Плотный декодер PyTorch использует RMSNorm, RoPE, GQA и SwiGLU. Реальный3B-путь: FSDP2,8GPU,eager, контекст4096, micro batch4, глобально2,097,152токена, пик LR3e-4, цель57,000шагов. CPU micro пример отличается от распределённого обучения. Точное продолжение требует модель, optimizer, RNG и контракты входов/настроек.

[examples/3b.observed.json](https://github.com/pathcosmos/korean-dense-llm/blob/main/examples/3b.observed.json) · [examples/3b.fresh.json](https://github.com/pathcosmos/korean-dense-llm/blob/main/examples/3b.fresh.json)

```bash
python scripts/hardware_profile.py --output runs/hardware.json
torchrun --standalone --nproc_per_node=8 --module train.deterministic_train --config examples/3b.fresh.json
```

## Оптимизация и измерения

Сравниваются worker/thread/batch, кэш SQLite, RAM, local/NFS, eager/compile, точность и память GPU. Полное время, пропускная способность, пик памяти и эквивалентность содержимого/чисел обосновывают принятие или отказ:47 кампаний и99 JSON. Скорость не оправдывает численные ошибки; малая выборка не доказывает глобальный оптимум.

[47 / 99](https://github.com/pathcosmos/korean-dense-llm/blob/main/experiments/README.md)

## Проверка

GH-исходники прошли284 CPU-теста при9 пропусках; новый GitHub CPU —258 при35 пропусках. Пропуски требуют оригинальные записи или вторую файловую систему. Синтетический пример:120документов, train100/eval20, vocab272,108,864параметра; проверяется точное восстановление модели/optimizer. Хеши, разбиение и reader данных не означают полную независимую репродукцию3B.

## Источники и лицензии

Исходный код проекта —MIT, авторские пояснения —CC BY4.0. Это не заменяет права на исходное содержание. См.source.json, LICENSE-SCOPE.md и происхождение: условия, атрибуция, уведомления и нерешённые вопросы. Не заявляется неограниченное коммерческое распространение каждого исходного текста.

[NOTICE.md](https://github.com/pathcosmos/korean-dense-llm/blob/main/NOTICE.md) · [LICENSE](https://github.com/pathcosmos/korean-dense-llm/blob/main/LICENSE)

## Ограничения

Не гарантируются глобальная приблизительная дедупликация, полное удаление PII/секретов, отсутствие загрязнения оценок и итоговое качество. На момент снимка обучение продолжалось; SFT, экспорт и официальные языковые тесты не завершены.11 исторических ссылок на код/9уникальных hash не имеют восстановленного снимка. Некоторым инструментам нужны оригинальные manifest.

## Связанные репозитории

[GitHub](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ru.md) · [README](https://github.com/pathcosmos/korean-dense-llm)
[GH / Использование](https://github.com/pathcosmos/korean-dense-llm/blob/main/docs/PUBLIC_GUIDE.md) · [GH / Оптимизация и измерения](https://github.com/pathcosmos/korean-dense-llm/blob/main/experiments/README.md) · [GH / Проверка](https://github.com/pathcosmos/korean-dense-llm/blob/main/docs/PUBLIC_VALIDATION.md)

- [korean-pretrain-corpus](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/blob/main/cards/README.ru.md)
- [korean-pretrain-training-data](https://huggingface.co/datasets/pathcosmos/korean-pretrain-training-data/blob/main/cards/README.ru.md)
- [korean-pretrain-wikipedia](https://huggingface.co/datasets/pathcosmos/korean-pretrain-wikipedia/blob/main/cards/README.ru.md)
- [korean-pretrain-fineweb2](https://huggingface.co/datasets/pathcosmos/korean-pretrain-fineweb2/blob/main/cards/README.ru.md)
- [korean-pretrain-hplt2](https://huggingface.co/datasets/pathcosmos/korean-pretrain-hplt2/blob/main/cards/README.ru.md)
- [korean-pretrain-fineweb-edu](https://huggingface.co/datasets/pathcosmos/korean-pretrain-fineweb-edu/blob/main/cards/README.ru.md)
- [korean-pretrain-math](https://huggingface.co/datasets/pathcosmos/korean-pretrain-math/blob/main/cards/README.ru.md)
- [korean-pretrain-code](https://huggingface.co/datasets/pathcosmos/korean-pretrain-code/blob/main/cards/README.ru.md)

- [wiki_ko/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/wiki_ko)
- [fineweb2_ko/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/fineweb2_ko)
- [hplt_ko/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/hplt_ko)
- [fineweb_edu/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/fineweb_edu)
- [owm/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/owm)
- [github_code_v1/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/github_code_v1)

[한국어](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ko.md) · [English](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.en.md) · [简体中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-CN.md) · [繁體中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-TW.md) · [日本語](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ja.md) · [Français](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.fr.md) · [Deutsch](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.de.md) · [Español](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.es.md) · [Português (Brasil)](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.pt-BR.md) · [हिन्दी](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.hi.md) · [العربية](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ar.md) · [עברית](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.he.md) · [Русский](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ru.md)
