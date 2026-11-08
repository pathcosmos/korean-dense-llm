# Korean Dense LLM — العربية

[한국어](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ko.md) · [English](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.en.md) · [简体中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-CN.md) · [繁體中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-TW.md) · [日本語](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ja.md) · [Français](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.fr.md) · [Deutsch](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.de.md) · [Español](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.es.md) · [Português (Brasil)](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.pt-BR.md) · [हिन्दी](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.hi.md) · [العربية](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ar.md) · [עברית](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.he.md) · [Русский](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ru.md)

## نظرة عامة

ينشر المشروع شيفرة إعداد البيانات والنموذج والتدريب والتقييم والاستعادة وتجارب التحسين لنموذج لغوي كثيف يركز على الكورية. يحتوي الإصدار على المصادر وسجلات التجارب؛ أما النصوص والرموز الثنائية وقواعد البيانات والأوزان فتوزع بشكل منفصل.

لغة الترجمة لا تعني وجود بيانات تدريب بتلك اللغة. اللغتان الأساسيتان الكورية والإنجليزية؛ قد تحتوي الويب والرياضيات والشيفرة لغات أخرى.

## المحتوى والحجم

| | |
|---|---:|
| common/, data/ | المعالجة والتقسيم |
| tokenizer/ | المعالجة والتقسيم |
| model/, train/, eval/ | النموذج والتدريب |
| scripts/ | التحسين والقياسات |
| tests/, examples/ | التحقق |
| experiments/ | 47 / 99 |


## المعالجة والتقسيم

تشمل معالجة الويب تنظيف الفراغات وقواعد الطول/التكرار/اللغة وإخفاء أنماط محددة من المعلومات الشخصية. تختلف بصمة الجزء الأولية عن NFC SHA256 النهائي للويب. تحافظ تصفية الشيفرة على حالة الأحرف والمسافات البادئة وUnicode الأصلي، وتفحص المسارات وتسميات الترخيص والشيفرة المولدة وبعض أنماط الأسرار. توزع المجموعات المتصلة للنصوص المتطابقة ونصوص الويب المكافئة في NFC ومستودعات الشيفرة على train/eval.

## الاستخدام

نفذ الأوامر من جذر المستودع. استخدم مجلدات إخراج وتنزيل جديدة وتحقق أولاً من النطاق والحجم وrevision المثبت. لا ينزل مثال CPU الاصطناعي بيانات خارجية. اختر الملفات اللازمة من الإصدارات المنفصلة وتحقق من الحجم/SHA256 باستخدام CHECKSUMS.json أو manifests المصادر.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python examples/quickstart_cpu.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python -m pytest -q tests
```


## النموذج والتدريب

النموذج مفكك ترميز كثيف بـPyTorch يستخدم RMSNorm وRoPE وGQA وSwiGLU. مسار3B الفعلي يستخدم FSDP2 و8GPU وeager:سياق4096،micro batch4،دفعة عالمية2,097,152tokens،ذروة LR3e-4 وهدف57,000step. مثال CPU الصغير مختلف عن التدريب الموزع؛ الاستئناف الدقيق يتطلب model وoptimizer وRNG وعقود المدخلات والإعدادات.

[examples/3b.observed.json](https://github.com/pathcosmos/korean-dense-llm/blob/main/examples/3b.observed.json) · [examples/3b.fresh.json](https://github.com/pathcosmos/korean-dense-llm/blob/main/examples/3b.fresh.json)

```bash
python scripts/hardware_profile.py --output runs/hardware.json
torchrun --standalone --nproc_per_node=8 --module train.deterministic_train --config examples/3b.fresh.json
```

## التحسين والقياسات

تقارن التجارب worker وthread وbatch وذاكرة SQLite وRAM والتخزين المحلي/NFS وeager/compile والدقة وذاكرة GPU. الوقت الكلي والإنتاجية وذروة الذاكرة وتكافؤ المحتوى والأرقام تدعم أسباب القبول أو الرفض:47 حملة و99 ملف JSON. السرعة لا تبرر فشلاً عددياً، والعينة الصغيرة لا تثبت الحل الأمثل على كامل البيانات.

[47 / 99](https://github.com/pathcosmos/korean-dense-llm/blob/main/experiments/README.md)

## التحقق

اجتاز إصدار GH المصدر284 اختبار CPU مع9 متجاوزة؛ واجتازت بيئة GitHub CPU الجديدة258 مع35 متجاوزة. تحتاج الاختبارات المتجاوزة لسجلات التشغيل الأصلية أو نظام ملفات ثانٍ. يستخدم المثال الاصطناعي120وثيقة وtrain100/eval20 وvocab272 ونموذج108,864معلمة ويتحقق من استعادة model/optimizer بدقة. فحوص بصمات البيانات وتقسيمها وreader ليست إعادة إنتاج3B مستقلة كاملة.

## المصادر والتراخيص

الشيفرة الأصلية للمشروع MIT والشرح الأصلي CC BY4.0. لا تحل هذه المنح محل حقوق المحتوى الأصلي. راجع source.json وLICENSE-SCOPE.md ووثائق المصدر للشروط والنسب والإشعارات والمسائل المعلقة. لا ندعي حرية إعادة التوزيع التجاري غير المقيد لكل نص أصلي.

[NOTICE.md](https://github.com/pathcosmos/korean-dense-llm/blob/main/NOTICE.md) · [LICENSE](https://github.com/pathcosmos/korean-dense-llm/blob/main/LICENSE)

## القيود

لا نضمن إزالة التشابه العالمي أو جميع المعلومات الشخصية/الأسرار أو خلو التقييم الرسمي من التلوث أو الجودة النهائية. كان التدريب الرئيسي جارياً عند اللقطة؛ لم تكتمل SFT أو تصدير النموذج أو المقاييس اللغوية الرسمية. لا توجد لقطة مستعادة لـ11مرجع شيفرة تاريخياً/9بصمات فريدة. تحتاج بعض أدوات التشغيل manifests الأصلية.

## المستودعات المرتبطة

[GitHub](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ar.md) · [README](https://github.com/pathcosmos/korean-dense-llm)
[GH / الاستخدام](https://github.com/pathcosmos/korean-dense-llm/blob/main/docs/PUBLIC_GUIDE.md) · [GH / التحسين والقياسات](https://github.com/pathcosmos/korean-dense-llm/blob/main/experiments/README.md) · [GH / التحقق](https://github.com/pathcosmos/korean-dense-llm/blob/main/docs/PUBLIC_VALIDATION.md)

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
