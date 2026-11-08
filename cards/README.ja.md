# Korean Dense LLM — 日本語

[한국어](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ko.md) · [English](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.en.md) · [简体中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-CN.md) · [繁體中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-TW.md) · [日本語](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ja.md) · [Français](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.fr.md) · [Deutsch](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.de.md) · [Español](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.es.md) · [Português (Brasil)](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.pt-BR.md) · [हिन्दी](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.hi.md) · [العربية](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ar.md) · [עברית](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.he.md) · [Русский](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ru.md)

## 概要

韓国語中心のDense LLMを実装する際のデータ準備・モデル・学習・評価・復旧・最適化コードを公開します。この版はソースと実験記録であり、本文・トークンバイナリ・DB・モデル重みを含みません。HFの別データ配布は下記リンクから利用できます。

翻訳言語はその言語の学習データを含むことを意味しません。主自然言語は韓国語・英語で、ウェブ・数学・コードには他言語が混在し得ます。

## 構成と規模

| | |
|---|---:|
| common/, data/ | 処理と分割 |
| tokenizer/ | 処理と分割 |
| model/, train/, eval/ | モデルと学習 |
| scripts/ | 最適化とベンチマーク |
| tests/, examples/ | 検証 |
| experiments/ | 47 / 99 |


## 処理と分割

ウェブは空白整理、長さ・反復・言語のヒューリスティック、一部PIIパターンのマスキングを適用します。初期シャード内ハッシュと最終ウェブNFC SHA256は基準が異なります。コードは大文字小文字・字下げ・元Unicodeを保持し、パス・ライセンス表示・生成コード・一部秘密情報パターンを検査します。同一本文、NFC等価ウェブ、コードリポジトリの連結グループをtrain/evalに分けます。

## 使い方

以下はリポジトリルートで実行します。新しい出力/取得先を使い、範囲・容量・固定revisionを確認してください。合成CPU例は外部データを取得しません。実データは別配布から必要なファイルを選び、CHECKSUMS.jsonまたはsource manifestのサイズ/SHA256を検証します。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python examples/quickstart_cpu.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python -m pytest -q tests
```


## モデルと学習

モデルはRMSNorm・RoPE・GQA・SwiGLUを用いる純PyTorch Dense decoderです。実3B経路はFSDP2・8GPU・eager、文脈4096、micro batch4、global batch2,097,152tokens、peak LR3e-4、目標57,000stepです。CPU micro例は分散学習と異なり、正確な再開にはモデル・optimizer・RNG・入力/設定契約が必要です。

[examples/3b.observed.json](https://github.com/pathcosmos/korean-dense-llm/blob/main/examples/3b.observed.json) · [examples/3b.fresh.json](https://github.com/pathcosmos/korean-dense-llm/blob/main/examples/3b.fresh.json)

```bash
python scripts/hardware_profile.py --output runs/hardware.json
torchrun --standalone --nproc_per_node=8 --module train.deterministic_train --config examples/3b.fresh.json
```

## 最適化とベンチマーク

worker/thread/batch、SQLite cache、RAM、local/NFS、eager/compile、精度、GPUメモリを比較しました。総時間・処理量・peakメモリ・内容/数値等価性と採用/不採用理由を記録し、47キャンペーン・99結果JSONを提供します。高速でも数値条件に失敗した候補を推奨せず、小標本の最適値を全体最適としません。

[47 / 99](https://github.com/pathcosmos/korean-dense-llm/blob/main/experiments/README.md)

## 検証

GHソース版の独立CPU検査は284成功/9skip、新GitHub CPU環境は258成功/35skipです。skipは元運用記録または別ファイルシステムが必要な検査です。合成例は120文書・train100/eval20・vocab272・108,864パラメータを使いモデル/optimizerの正確な再開を確認します。データhash・分割・reader検査と完全な独立3B再現は別です。

## 出典とライセンス

プロジェクト独自コードはMIT、独自説明はCC BY4.0です。元データの権利を置き換えません。source.json、LICENSE-SCOPE.md、出典文書でソース別条件・帰属・告知・未解決事項を確認してください。全原文の無制限な商用再配布を主張しません。

[NOTICE.md](https://github.com/pathcosmos/korean-dense-llm/blob/main/NOTICE.md) · [LICENSE](https://github.com/pathcosmos/korean-dense-llm/blob/main/LICENSE)

## 制限

全域近似重複除去、個人情報/secret完全除去、公式評価の無汚染、最終品質は保証しません。公開時点で本学習は進行中で、SFT・モデルexport・公式言語benchmarkは未完了です。過去コード11参照/9固有hashは元snapshot未回収です。一部運用ツールには元実行manifestが必要です。

## 関連リポジトリ

[GitHub](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ja.md) · [README](https://github.com/pathcosmos/korean-dense-llm)
[GH / 使い方](https://github.com/pathcosmos/korean-dense-llm/blob/main/docs/PUBLIC_GUIDE.md) · [GH / 最適化とベンチマーク](https://github.com/pathcosmos/korean-dense-llm/blob/main/experiments/README.md) · [GH / 検証](https://github.com/pathcosmos/korean-dense-llm/blob/main/docs/PUBLIC_VALIDATION.md)

- [korean-pretrain-corpus](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/blob/main/cards/README.ja.md)
- [korean-pretrain-training-data](https://huggingface.co/datasets/pathcosmos/korean-pretrain-training-data/blob/main/cards/README.ja.md)
- [korean-pretrain-wikipedia](https://huggingface.co/datasets/pathcosmos/korean-pretrain-wikipedia/blob/main/cards/README.ja.md)
- [korean-pretrain-fineweb2](https://huggingface.co/datasets/pathcosmos/korean-pretrain-fineweb2/blob/main/cards/README.ja.md)
- [korean-pretrain-hplt2](https://huggingface.co/datasets/pathcosmos/korean-pretrain-hplt2/blob/main/cards/README.ja.md)
- [korean-pretrain-fineweb-edu](https://huggingface.co/datasets/pathcosmos/korean-pretrain-fineweb-edu/blob/main/cards/README.ja.md)
- [korean-pretrain-math](https://huggingface.co/datasets/pathcosmos/korean-pretrain-math/blob/main/cards/README.ja.md)
- [korean-pretrain-code](https://huggingface.co/datasets/pathcosmos/korean-pretrain-code/blob/main/cards/README.ja.md)

- [wiki_ko/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/wiki_ko)
- [fineweb2_ko/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/fineweb2_ko)
- [hplt_ko/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/hplt_ko)
- [fineweb_edu/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/fineweb_edu)
- [owm/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/owm)
- [github_code_v1/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/github_code_v1)

[한국어](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ko.md) · [English](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.en.md) · [简体中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-CN.md) · [繁體中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-TW.md) · [日本語](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ja.md) · [Français](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.fr.md) · [Deutsch](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.de.md) · [Español](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.es.md) · [Português (Brasil)](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.pt-BR.md) · [हिन्दी](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.hi.md) · [العربية](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ar.md) · [עברית](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.he.md) · [Русский](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ru.md)
