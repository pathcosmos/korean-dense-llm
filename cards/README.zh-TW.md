# Korean Dense LLM — 繁體中文

[한국어](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ko.md) · [English](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.en.md) · [简体中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-CN.md) · [繁體中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-TW.md) · [日本語](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ja.md) · [Français](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.fr.md) · [Deutsch](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.de.md) · [Español](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.es.md) · [Português (Brasil)](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.pt-BR.md) · [हिन्दी](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.hi.md) · [العربية](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ar.md) · [עברית](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.he.md) · [Русский](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ru.md)

## 概覽

本專案公開建立韓語為中心的稠密大型語言模型時使用的資料準備、模型、訓練、評估、復原及最佳化程式碼。此版本提供原始碼及實驗紀錄，不包含原文、詞元二進位、資料庫或模型權重；下方連結提供獨立HF資料發布。

文件翻譯語言不代表訓練資料涵蓋該語言。主要自然語言是韓語與英語，網頁、數學及程式碼可能混有其他語言。

## 組成與規模

| | |
|---|---:|
| common/, data/ | 處理與切分 |
| tokenizer/ | 處理與切分 |
| model/, train/, eval/ | 模型與訓練 |
| scripts/ | 最佳化與基準 |
| tests/, examples/ | 驗證 |
| experiments/ | 47 / 99 |


## 處理與切分

網頁處理包含空白整理、長度/重複/語言啟發式與部分個資模式遮罩。最初分片內雜湊和最終網頁NFC SHA256標準不同。程式碼過濾保留大小寫、縮排及原始Unicode，並檢查路徑、授權標籤、產生的程式碼與部分金鑰模式。相同正文、NFC等價網頁及程式碼儲存庫的連通群組分配至train/eval。

## 使用方法

下列命令在儲存庫根目錄執行。使用新的輸出/下載目錄，先確認範圍、容量及固定revision。合成CPU範例不下載外部資料。實際資料請由獨立發布選取必要檔案，依CHECKSUMS.json或來源manifest驗證大小/SHA256。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python examples/quickstart_cpu.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python -m pytest -q tests
```


## 模型與訓練

模型為純PyTorch稠密解碼器，包含RMSNorm、RoPE、GQA及SwiGLU。實際3B路徑採FSDP2、8GPU與eager：上下文4096、micro batch4、global batch2,097,152tokens、峰值LR3e-4、目標57,000step。CPU微型範例不同於分散式訓練；精確復原需模型、optimizer、RNG與輸入/設定契約。

[examples/3b.observed.json](https://github.com/pathcosmos/korean-dense-llm/blob/main/examples/3b.observed.json) · [examples/3b.fresh.json](https://github.com/pathcosmos/korean-dense-llm/blob/main/examples/3b.fresh.json)

```bash
python scripts/hardware_profile.py --output runs/hardware.json
torchrun --standalone --nproc_per_node=8 --module train.deterministic_train --config examples/3b.fresh.json
```

## 最佳化與基準

實驗比較worker、thread、batch、SQLite快取、RAM、本機/NFS、eager/compile、精度與GPU記憶體，同時記錄總時間、吞吐、峰值記憶體、內容/數值等價性及採用或拒絕原因。提供47組實驗與99個結果JSON。較快不能彌補數值條件失敗，小樣本最佳也不代表全量最佳。

[47 / 99](https://github.com/pathcosmos/korean-dense-llm/blob/main/experiments/README.md)

## 驗證

GH原始碼版獨立CPU測試284通過、9略過；新GitHub CPU環境258通過、35略過。略過項需原始執行紀錄或第二檔案系統。合成範例使用120文件、train100/eval20、vocab272及108,864參數模型，驗證模型/optimizer精確復原。資料內容雜湊、切分與reader驗證不等於完整獨立3B重現。

## 來源與授權

專案原創程式碼為MIT，原創說明為CC BY4.0，不取代原始內容權利。請查source.json、LICENSE-SCOPE.md與來源文件的個別條件、姓名標示、告知及未決事項。不宣稱每篇原文都可無限制商業再發布。

[NOTICE.md](https://github.com/pathcosmos/korean-dense-llm/blob/main/NOTICE.md) · [LICENSE](https://github.com/pathcosmos/korean-dense-llm/blob/main/LICENSE)

## 限制

不保證全域近似去重、完全移除個資/金鑰、官方評估無汙染或最終模型品質。發布快照時主訓練仍進行中；SFT、模型匯出及官方語言基準尚未完成。11個歷史程式碼引用/9個唯一hash缺少原始快照。部分維運工具需原執行manifest。

## 相關儲存庫

[GitHub](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-TW.md) · [README](https://github.com/pathcosmos/korean-dense-llm)
[GH / 使用方法](https://github.com/pathcosmos/korean-dense-llm/blob/main/docs/PUBLIC_GUIDE.md) · [GH / 最佳化與基準](https://github.com/pathcosmos/korean-dense-llm/blob/main/experiments/README.md) · [GH / 驗證](https://github.com/pathcosmos/korean-dense-llm/blob/main/docs/PUBLIC_VALIDATION.md)

- [korean-pretrain-corpus](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/blob/main/cards/README.zh-TW.md)
- [korean-pretrain-training-data](https://huggingface.co/datasets/pathcosmos/korean-pretrain-training-data/blob/main/cards/README.zh-TW.md)
- [korean-pretrain-wikipedia](https://huggingface.co/datasets/pathcosmos/korean-pretrain-wikipedia/blob/main/cards/README.zh-TW.md)
- [korean-pretrain-fineweb2](https://huggingface.co/datasets/pathcosmos/korean-pretrain-fineweb2/blob/main/cards/README.zh-TW.md)
- [korean-pretrain-hplt2](https://huggingface.co/datasets/pathcosmos/korean-pretrain-hplt2/blob/main/cards/README.zh-TW.md)
- [korean-pretrain-fineweb-edu](https://huggingface.co/datasets/pathcosmos/korean-pretrain-fineweb-edu/blob/main/cards/README.zh-TW.md)
- [korean-pretrain-math](https://huggingface.co/datasets/pathcosmos/korean-pretrain-math/blob/main/cards/README.zh-TW.md)
- [korean-pretrain-code](https://huggingface.co/datasets/pathcosmos/korean-pretrain-code/blob/main/cards/README.zh-TW.md)

- [wiki_ko/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/wiki_ko)
- [fineweb2_ko/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/fineweb2_ko)
- [hplt_ko/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/hplt_ko)
- [fineweb_edu/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/fineweb_edu)
- [owm/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/owm)
- [github_code_v1/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/github_code_v1)

[한국어](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ko.md) · [English](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.en.md) · [简体中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-CN.md) · [繁體中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-TW.md) · [日本語](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ja.md) · [Français](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.fr.md) · [Deutsch](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.de.md) · [Español](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.es.md) · [Português (Brasil)](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.pt-BR.md) · [हिन्दी](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.hi.md) · [العربية](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ar.md) · [עברית](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.he.md) · [Русский](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ru.md)
