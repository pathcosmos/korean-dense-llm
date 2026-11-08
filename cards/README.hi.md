# Korean Dense LLM — हिन्दी

[한국어](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ko.md) · [English](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.en.md) · [简体中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-CN.md) · [繁體中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-TW.md) · [日本語](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ja.md) · [Français](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.fr.md) · [Deutsch](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.de.md) · [Español](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.es.md) · [Português (Brasil)](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.pt-BR.md) · [हिन्दी](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.hi.md) · [العربية](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ar.md) · [עברית](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.he.md) · [Русский](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ru.md)

## परिचय

यह परियोजना कोरियाई-केंद्रित Dense LLM के डेटा तैयार करने, मॉडल, प्रशिक्षण, मूल्यांकन, पुनर्प्राप्ति और अनुकूलन का कोड प्रकाशित करती है। संस्करण में स्रोत और प्रयोग रिकॉर्ड हैं, मूल पाठ,टोकन बाइनरी,डेटाबेस या मॉडल वज़न नहीं। अलग HF डेटा प्रकाशन नीचे जुड़े हैं।

अनुवाद की भाषा उस भाषा के प्रशिक्षण डेटा की उपस्थिति नहीं दर्शाती। मुख्य प्राकृतिक भाषाएँ कोरियाई और अंग्रेज़ी हैं; वेब,गणित और कोड में दूसरी भाषाएँ हो सकती हैं।

## सामग्री और आकार

| | |
|---|---:|
| common/, data/ | प्रसंस्करण और विभाजन |
| tokenizer/ | प्रसंस्करण और विभाजन |
| model/, train/, eval/ | मॉडल और प्रशिक्षण |
| scripts/ | अनुकूलन और मापन |
| tests/, examples/ | सत्यापन |
| experiments/ | 47 / 99 |


## प्रसंस्करण और विभाजन

वेब प्रसंस्करण में खाली स्थान सुधार, लंबाई/दोहराव/भाषा नियम और कुछ व्यक्तिगत पैटर्न छिपाना शामिल है। शुरुआती shard के भीतर hash अंतिम वेब NFC SHA256 से अलग है। कोड में अक्षरों का case, indentation और मूल Unicode सुरक्षित हैं; पथ, लाइसेंस लेबल, generated code और कुछ secret पैटर्न जाँचे जाते हैं। समान पाठ, NFC-समतुल्य वेब पाठ और code repository के जुड़े समूह train/eval में बाँटे जाते हैं।

## उपयोग

नीचे के आदेश repository root से चलाएँ। नए output/download फ़ोल्डर लें और पहले दायरा, आकार तथा निश्चित revision जाँचें। कृत्रिम CPU उदाहरण बाहरी डेटा डाउनलोड नहीं करता। अलग डेटा प्रकाशन से आवश्यक फ़ाइलें चुनें और CHECKSUMS.json या source manifests से आकार/SHA256 सत्यापित करें।

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python examples/quickstart_cpu.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python -m pytest -q tests
```


## मॉडल और प्रशिक्षण

शुद्ध PyTorch Dense decoder में RMSNorm, RoPE, GQA और SwiGLU हैं। वास्तविक3B पथ FSDP2,8GPU और eager उपयोग करता है:context4096,micro batch4,global batch2,097,152tokens,peak LR3e-4,लक्ष्य57,000steps। CPU micro उदाहरण वितरित प्रशिक्षण से अलग है। सही पुनःआरंभ में model,optimizer,RNG और input/configuration अनुबंध चाहिए।

[examples/3b.observed.json](https://github.com/pathcosmos/korean-dense-llm/blob/main/examples/3b.observed.json) · [examples/3b.fresh.json](https://github.com/pathcosmos/korean-dense-llm/blob/main/examples/3b.fresh.json)

```bash
python scripts/hardware_profile.py --output runs/hardware.json
torchrun --standalone --nproc_per_node=8 --module train.deterministic_train --config examples/3b.fresh.json
```

## अनुकूलन और मापन

worker,thread,batch,SQLite cache,RAM,local/NFS,eager/compile,precision और GPU memory की तुलना की गई। कुल समय, throughput,peak memory और सामग्री/संख्यात्मक समानता के साथ अपनाने या अस्वीकार करने के कारण दिए हैं:47 अभियान और99 परिणाम JSON। तेज़ होना गलत संख्यात्मक परिणाम को सही नहीं बनाता; छोटा नमूना पूरे डेटा का सर्वोत्तम विकल्प नहीं सिद्ध करता।

[47 / 99](https://github.com/pathcosmos/korean-dense-llm/blob/main/experiments/README.md)

## सत्यापन

GH स्रोत संस्करण में284 स्वतंत्र CPU परीक्षण सफल और9 छोड़े गए; नए GitHub CPU वातावरण में258 सफल और35 छोड़े गए। छोड़े गए परीक्षण मूल रन रिकॉर्ड या दूसरे filesystem पर निर्भर हैं। कृत्रिम quickstart में120दस्तावेज़,train100/eval20,vocab272 और108,864पैरामीटर हैं तथा model/optimizer का सटीक पुनःआरंभ जाँचा जाता है। डेटा hash,split और reader की जाँच पूर्ण स्वतंत्र3B पुनरुत्पादन नहीं है।

## स्रोत और लाइसेंस

परियोजना का मूल कोड MIT और मूल व्याख्या CC BY4.0 है। ये स्रोत सामग्री के अधिकारों का स्थान नहीं लेते। स्रोत की शर्तें, श्रेय, सूचनाएँ और अनसुलझे मुद्दे source.json,LICENSE-SCOPE.md तथा provenance दस्तावेज़ों में देखें। प्रत्येक मूल पाठ के असीमित व्यावसायिक पुनर्वितरण का दावा नहीं है।

[NOTICE.md](https://github.com/pathcosmos/korean-dense-llm/blob/main/NOTICE.md) · [LICENSE](https://github.com/pathcosmos/korean-dense-llm/blob/main/LICENSE)

## सीमाएँ

वैश्विक लगभग-समान डुप्लिकेट हटाने,संपूर्ण PII/secret हटाने,दूषणरहित आधिकारिक मूल्यांकन या अंतिम गुणवत्ता की गारंटी नहीं है। प्रकाशन snapshot पर मुख्य प्रशिक्षण चल रहा था; SFT,model export और आधिकारिक भाषा benchmarks पूरे नहीं हैं।11ऐतिहासिक code references/9अद्वितीय hash का मूल snapshot नहीं मिला। कुछ संचालन उपकरणों को मूल run manifests चाहिए।

## संबंधित रिपॉज़िटरी

[GitHub](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.hi.md) · [README](https://github.com/pathcosmos/korean-dense-llm)
[GH / उपयोग](https://github.com/pathcosmos/korean-dense-llm/blob/main/docs/PUBLIC_GUIDE.md) · [GH / अनुकूलन और मापन](https://github.com/pathcosmos/korean-dense-llm/blob/main/experiments/README.md) · [GH / सत्यापन](https://github.com/pathcosmos/korean-dense-llm/blob/main/docs/PUBLIC_VALIDATION.md)

- [korean-pretrain-corpus](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/blob/main/cards/README.hi.md)
- [korean-pretrain-training-data](https://huggingface.co/datasets/pathcosmos/korean-pretrain-training-data/blob/main/cards/README.hi.md)
- [korean-pretrain-wikipedia](https://huggingface.co/datasets/pathcosmos/korean-pretrain-wikipedia/blob/main/cards/README.hi.md)
- [korean-pretrain-fineweb2](https://huggingface.co/datasets/pathcosmos/korean-pretrain-fineweb2/blob/main/cards/README.hi.md)
- [korean-pretrain-hplt2](https://huggingface.co/datasets/pathcosmos/korean-pretrain-hplt2/blob/main/cards/README.hi.md)
- [korean-pretrain-fineweb-edu](https://huggingface.co/datasets/pathcosmos/korean-pretrain-fineweb-edu/blob/main/cards/README.hi.md)
- [korean-pretrain-math](https://huggingface.co/datasets/pathcosmos/korean-pretrain-math/blob/main/cards/README.hi.md)
- [korean-pretrain-code](https://huggingface.co/datasets/pathcosmos/korean-pretrain-code/blob/main/cards/README.hi.md)

- [wiki_ko/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/wiki_ko)
- [fineweb2_ko/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/fineweb2_ko)
- [hplt_ko/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/hplt_ko)
- [fineweb_edu/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/fineweb_edu)
- [owm/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/owm)
- [github_code_v1/](https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/tree/main/github_code_v1)

[한국어](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ko.md) · [English](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.en.md) · [简体中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-CN.md) · [繁體中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-TW.md) · [日本語](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ja.md) · [Français](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.fr.md) · [Deutsch](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.de.md) · [Español](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.es.md) · [Português (Brasil)](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.pt-BR.md) · [हिन्दी](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.hi.md) · [العربية](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ar.md) · [עברית](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.he.md) · [Русский](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ru.md)
