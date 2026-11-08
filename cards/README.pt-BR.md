# Korean Dense LLM — Português (Brasil)

[한국어](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ko.md) · [English](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.en.md) · [简体中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-CN.md) · [繁體中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-TW.md) · [日本語](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ja.md) · [Français](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.fr.md) · [Deutsch](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.de.md) · [Español](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.es.md) · [Português (Brasil)](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.pt-BR.md) · [हिन्दी](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.hi.md) · [العربية](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ar.md) · [עברית](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.he.md) · [Русский](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ru.md)

## Visão geral

Este projeto publica o código de preparação de dados, modelo, treinamento, avaliação, recuperação e otimização de um LLM denso centrado em coreano. A edição contém fontes e registros de experimentos; textos, tokens binários, bancos e pesos são distribuídos separadamente.

O idioma da tradução não implica dados de treino nesse idioma. As línguas principais são coreano e inglês; web, matemática e código podem conter outras.

## Conteúdo e escala

| | |
|---|---:|
| common/, data/ | Processamento e divisão |
| tokenizer/ | Processamento e divisão |
| model/, train/, eval/ | Modelo e treinamento |
| scripts/ | Otimização e medições |
| tests/, examples/ | Validação |
| experiments/ | 47 / 99 |


## Processamento e divisão

A web passa por limpeza de espaços, heurísticas de tamanho/repetição/idioma e mascaramento de alguns padrões pessoais. O hash inicial por shard difere do SHA256 NFC final. Código preserva maiúsculas, indentação e Unicode original; verificam-se caminhos, rótulos de licença, código gerado e alguns segredos. Grupos conectados de textos iguais, web equivalente em NFC e repositórios de código são divididos em train/eval.

## Uso

Execute na raiz do repositório. Use novas pastas e confira escopo, tamanho e revision fixada. O exemplo CPU sintético não baixa dados externos. Selecione arquivos necessários nas distribuições separadas e confira tamanho/SHA256 em CHECKSUMS.json ou manifests.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python examples/quickstart_cpu.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python -m pytest -q tests
```


## Modelo e treinamento

O decodificador denso PyTorch usa RMSNorm, RoPE, GQA e SwiGLU. O caminho3B real usa FSDP2,8GPU e eager: contexto4096, micro batch4, batch global2,097,152tokens, pico LR3e-4 e meta57,000passos. O exemplo CPU micro difere do distribuído. A retomada exata exige modelo, optimizer, RNG e contratos de entrada/configuração.

[examples/3b.observed.json](https://github.com/pathcosmos/korean-dense-llm/blob/main/examples/3b.observed.json) · [examples/3b.fresh.json](https://github.com/pathcosmos/korean-dense-llm/blob/main/examples/3b.fresh.json)

```bash
python scripts/hardware_profile.py --output runs/hardware.json
torchrun --standalone --nproc_per_node=8 --module train.deterministic_train --config examples/3b.fresh.json
```

## Otimização e medições

Comparam-se workers, threads, batch, cache SQLite, RAM, local/NFS, eager/compile, precisão e memória GPU. Tempo total, vazão, pico de memória e equivalência de conteúdo/números sustentam adoção ou rejeição:47 campanhas e99 JSON. Velocidade não compensa falha numérica, e uma amostra pequena não prova ótimo global.

[47 / 99](https://github.com/pathcosmos/korean-dense-llm/blob/main/experiments/README.md)

## Validação

A edição GH passou284 testes CPU com9 ignorados; o novo GitHub CPU passou258 com35 ignorados. Os ignorados exigem registros originais ou outro sistema de arquivos. O quickstart sintético usa120documentos, train100/eval20, vocab272 e108,864parâmetros, verificando retomada exata de modelo/optimizer. Hash, divisão e reader de dados não equivalem à reprodução3B independente completa.

## Fontes e licenças

O código original é MIT e o texto original CC BY4.0. Essas permissões não substituem direitos do conteúdo fonte. Consulte source.json, LICENSE-SCOPE.md e procedência para condições, atribuição, avisos e pendências. Não se afirma redistribuição comercial irrestrita de todo texto original.

[NOTICE.md](https://github.com/pathcosmos/korean-dense-llm/blob/main/NOTICE.md) · [LICENSE](https://github.com/pathcosmos/korean-dense-llm/blob/main/LICENSE)

## Limitações

Não se garantem deduplicação aproximada global, remoção total de PII/segredos, avaliação sem contaminação ou qualidade final. O treino principal seguia ativo no snapshot; SFT, exportação e benchmarks linguísticos oficiais não estão concluídos.11 referências históricas/9hash únicos não possuem snapshot recuperado. Algumas ferramentas precisam de manifests originais.

## Repositórios relacionados

[GitHub](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.pt-BR.md) · [README](https://github.com/pathcosmos/korean-dense-llm)
[GH / Uso](https://github.com/pathcosmos/korean-dense-llm/blob/main/docs/PUBLIC_GUIDE.md) · [GH / Otimização e medições](https://github.com/pathcosmos/korean-dense-llm/blob/main/experiments/README.md) · [GH / Validação](https://github.com/pathcosmos/korean-dense-llm/blob/main/docs/PUBLIC_VALIDATION.md)

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
