# Korean Dense LLM — 简体中文

[한국어](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ko.md) · [English](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.en.md) · [简体中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-CN.md) · [繁體中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-TW.md) · [日本語](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ja.md) · [Français](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.fr.md) · [Deutsch](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.de.md) · [Español](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.es.md) · [Português (Brasil)](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.pt-BR.md) · [हिन्दी](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.hi.md) · [العربية](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ar.md) · [עברית](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.he.md) · [Русский](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ru.md)

## 概览

本项目公开构建韩语为中心的稠密大语言模型时使用的数据准备、模型、训练、评估、恢复与优化代码。此版本提供源码和实验记录，不包含原文、词元二进制、数据库或模型权重；下方链接提供独立HF数据发布。

文档翻译语言不代表训练数据覆盖该语言。主要自然语言为韩语和英语，网页、数学及代码中可能混有其他语言。

## 组成与规模

| | |
|---|---:|
| common/, data/ | 处理与划分 |
| tokenizer/ | 处理与划分 |
| model/, train/, eval/ | 模型与训练 |
| scripts/ | 优化与基准 |
| tests/, examples/ | 验证 |
| experiments/ | 47 / 99 |


## 处理与划分

网页处理包括空白整理、长度/重复/语言启发式和部分个人信息模式掩码。最初分片内哈希与最终网页NFC SHA256标准不同。代码过滤保留大小写、缩进及原始Unicode，并检查路径、许可标签、生成代码与部分密钥模式。相同正文、NFC等价网页及代码仓库的连通组被分配到train/eval。

## 使用方法

以下命令在仓库根目录执行。使用新输出/下载目录，先核对范围、容量和固定revision。合成CPU示例不下载外部数据。实际数据请从独立发布中选择必要文件，并依据CHECKSUMS.json或来源manifest验证大小/SHA256。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python examples/quickstart_cpu.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python -m pytest -q tests
```


## 模型与训练

模型采用纯PyTorch稠密解码器，包含RMSNorm、RoPE、GQA和SwiGLU。实际3B路径使用FSDP2、8GPU和eager：上下文4096、micro batch4、global batch2,097,152tokens、峰值LR3e-4、目标57,000step。CPU微型示例与分布式训练不同。精确恢复需要模型、optimizer、RNG及输入/配置契约。

[examples/3b.observed.json](https://github.com/pathcosmos/korean-dense-llm/blob/main/examples/3b.observed.json) · [examples/3b.fresh.json](https://github.com/pathcosmos/korean-dense-llm/blob/main/examples/3b.fresh.json)

```bash
python scripts/hardware_profile.py --output runs/hardware.json
torchrun --standalone --nproc_per_node=8 --module train.deterministic_train --config examples/3b.fresh.json
```

## 优化与基准

实验比较worker、thread、batch、SQLite缓存、RAM、本地/NFS、eager/compile、精度和GPU内存，同时记录总时间、吞吐、峰值内存、内容/数值等价性与采纳或拒绝原因。提供47个实验组和99个结果JSON。速度更快不代表数值失败的候选可用，小样本最优也不代表全量最优。

[47 / 99](https://github.com/pathcosmos/korean-dense-llm/blob/main/experiments/README.md)

## 验证

GH源码版独立CPU测试284通过、9跳过；新GitHub CPU环境258通过、35跳过。跳过项需要原始运行记录或第二文件系统。合成示例使用120文档、train100/eval20、vocab272和108,864参数模型，并验证模型/optimizer精确恢复。数据内容哈希、划分和reader验证不等于完整独立3B复现。

## 来源与许可

项目原创代码为MIT，原创说明为CC BY4.0，不替代原始内容权利。请查看source.json、LICENSE-SCOPE.md与来源文档中的特定条件、署名、通知及未决事项。不宣称每篇原文均可无限制商业再分发。

[NOTICE.md](https://github.com/pathcosmos/korean-dense-llm/blob/main/NOTICE.md) · [LICENSE](https://github.com/pathcosmos/korean-dense-llm/blob/main/LICENSE)

## 局限

不保证全局近似去重、彻底清除个人信息/密钥、官方评估无污染或最终模型质量。发布快照时主训练仍在进行；SFT、模型导出及官方语言基准尚未完成。11个历史代码引用/9个唯一hash缺少原始快照。一些运维工具需要原运行manifest。

## 相关仓库

[GitHub](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-CN.md) · [README](https://github.com/pathcosmos/korean-dense-llm)
[GH / 使用方法](https://github.com/pathcosmos/korean-dense-llm/blob/main/docs/PUBLIC_GUIDE.md) · [GH / 优化与基准](https://github.com/pathcosmos/korean-dense-llm/blob/main/experiments/README.md) · [GH / 验证](https://github.com/pathcosmos/korean-dense-llm/blob/main/docs/PUBLIC_VALIDATION.md)

- [korean-pretrain-corpus](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/blob/main/cards/README.zh-CN.md)
- [korean-pretrain-training-data](https://huggingface.co/datasets/pathcosmos/korean-pretrain-training-data/blob/main/cards/README.zh-CN.md)
- [korean-pretrain-wikipedia](https://huggingface.co/datasets/pathcosmos/korean-pretrain-wikipedia/blob/main/cards/README.zh-CN.md)
- [korean-pretrain-fineweb2](https://huggingface.co/datasets/pathcosmos/korean-pretrain-fineweb2/blob/main/cards/README.zh-CN.md)
- [korean-pretrain-hplt2](https://huggingface.co/datasets/pathcosmos/korean-pretrain-hplt2/blob/main/cards/README.zh-CN.md)
- [korean-pretrain-fineweb-edu](https://huggingface.co/datasets/pathcosmos/korean-pretrain-fineweb-edu/blob/main/cards/README.zh-CN.md)
- [korean-pretrain-math](https://huggingface.co/datasets/pathcosmos/korean-pretrain-math/blob/main/cards/README.zh-CN.md)
- [korean-pretrain-code](https://huggingface.co/datasets/pathcosmos/korean-pretrain-code/blob/main/cards/README.zh-CN.md)

- [wiki_ko/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/wiki_ko)
- [fineweb2_ko/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/fineweb2_ko)
- [hplt_ko/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/hplt_ko)
- [fineweb_edu/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/fineweb_edu)
- [owm/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/owm)
- [github_code_v1/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/github_code_v1)

[한국어](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ko.md) · [English](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.en.md) · [简体中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-CN.md) · [繁體中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-TW.md) · [日本語](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ja.md) · [Français](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.fr.md) · [Deutsch](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.de.md) · [Español](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.es.md) · [Português (Brasil)](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.pt-BR.md) · [हिन्दी](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.hi.md) · [العربية](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ar.md) · [עברית](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.he.md) · [Русский](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ru.md)
