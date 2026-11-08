# Korean Dense LLM — עברית

[한국어](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ko.md) · [English](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.en.md) · [简体中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-CN.md) · [繁體中文](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.zh-TW.md) · [日本語](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ja.md) · [Français](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.fr.md) · [Deutsch](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.de.md) · [Español](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.es.md) · [Português (Brasil)](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.pt-BR.md) · [हिन्दी](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.hi.md) · [العربية](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ar.md) · [עברית](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.he.md) · [Русский](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.ru.md)

## סקירה

הפרויקט מפרסם קוד להכנת נתונים,מודל,אימון,הערכה,שחזור וניסויי מיטוב של מודל שפה צפוף המתמקד בקוריאנית. גרסה זו כוללת מקורות ורשומות ניסוי;טקסטים,קובצי טוקנים בינריים,מסדי נתונים ומשקלים מופצים בנפרד.

שפת התרגום אינה מעידה על נתוני אימון באותה שפה. השפות העיקריות הן קוריאנית ואנגלית;רשת,מתמטיקה וקוד עשויים להכיל שפות נוספות.

## תוכן והיקף

| | |
|---|---:|
| common/, data/ | עיבוד וחלוקה |
| tokenizer/ | עיבוד וחלוקה |
| model/, train/, eval/ | מודל ואימון |
| scripts/ | מיטוב ומדידות |
| tests/, examples/ | אימות |
| experiments/ | 47 / 99 |


## עיבוד וחלוקה

עיבוד הרשת כולל ניקוי רווחים,כללי אורך/חזרה/שפה ומיסוך דפוסים מסוימים של מידע אישי. הגיבוב הראשוני בתוך מקטע שונה מNFC SHA256 הסופי. סינון הקוד שומר אותיות גדולות/קטנות,הזחה וUnicode מקורי,ובודק נתיבים,תוויות רישיון,קוד שנוצר ודפוסי סודות מסוימים. קבוצות מחוברות של טקסט זהה,טקסט רשת שקול בNFC ומאגרי קוד מחולקות לtrain/eval.

## שימוש

הריצו את הפקודות משורש המאגר. השתמשו בתיקיות פלט/הורדה חדשות ובדקו היקף,גודל וrevision מקובע. דוגמת CPU הסינתטית אינה מורידה נתונים חיצוניים. בחרו קבצים נחוצים מההפצות הנפרדות ואמתו גודל/SHA256 מול CHECKSUMS.json או manifests.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python examples/quickstart_cpu.py
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 RAYON_NUM_THREADS=2 python -m pytest -q tests
```


## מודל ואימון

המודל הוא מפענח צפוף בPyTorch עם RMSNorm,RoPE,GQA וSwiGLU. מסלול3B האמיתי משתמש בFSDP2,8GPU וeager:הקשר4096,micro batch4,אצווה כוללת2,097,152tokens,שיא LR3e-4 ויעד57,000steps. דוגמת CPU micro שונה מאימון מבוזר. המשך מדויק דורש model,optimizer,RNG וחוזי קלט/תצורה.

[examples/3b.observed.json](https://github.com/pathcosmos/korean-dense-llm/blob/main/examples/3b.observed.json) · [examples/3b.fresh.json](https://github.com/pathcosmos/korean-dense-llm/blob/main/examples/3b.fresh.json)

```bash
python scripts/hardware_profile.py --output runs/hardware.json
torchrun --standalone --nproc_per_node=8 --module train.deterministic_train --config examples/3b.fresh.json
```

## מיטוב ומדידות

הניסויים משווים worker,thread,batch,מטמון SQLite,RAM,אחסון מקומי/NFS,eager/compile,דיוק וזיכרון GPU. זמן כולל,תפוקה,שיא זיכרון ושקילות תוכן/מספרים מלווים סיבות לאימוץ או דחייה:47 מסעות ניסוי ו99 קובצי JSON. מהירות אינה מצדיקה כשל מספרי,ומדגם קטן אינו מוכיח מיטוב לכל הנתונים.

[47 / 99](https://github.com/pathcosmos/korean-dense-llm/blob/main/experiments/README.md)

## אימות

גרסת המקור בGH עברה284 בדיקות CPU עם9 דילוגים;סביבת GitHub CPU חדשה עברה258 עם35 דילוגים. הדילוגים דורשים רשומות ריצה מקוריות או מערכת קבצים שנייה. הדוגמה הסינתטית משתמשת ב120מסמכים,train100/eval20,vocab272 ומודל108,864פרמטרים ומאמתת שחזור מדויק של model/optimizer. בדיקות גיבוב,חלוקה וreader של נתונים אינן שחזור3B עצמאי מלא.

## מקורות ורישיונות

קוד הפרויקט המקורי ברישיון MIT וההסבר המקורי CC BY4.0. אלו אינם מחליפים זכויות בתוכן המקורי. תנאים לפי מקור,ייחוס,הודעות וסוגיות פתוחות מופיעים בsource.json,LICENSE-SCOPE.md ומסמכי המקור. אין טענה להיתר מסחרי בלתי מוגבל לכל טקסט.

[NOTICE.md](https://github.com/pathcosmos/korean-dense-llm/blob/main/NOTICE.md) · [LICENSE](https://github.com/pathcosmos/korean-dense-llm/blob/main/LICENSE)

## מגבלות

אין הבטחה להסרת כפילויות מקורבות גלובלית,להסרה מלאה של מידע אישי/סודות,להערכה רשמית נקייה מזיהום או לאיכות סופית. האימון הראשי נמשך בעת תמונת המצב;SFT,ייצוא ומדדי שפה רשמיים לא הושלמו.11הפניות קוד היסטוריות/9גיבובים ייחודיים ללא תמונה מקורית ששוחזרה. חלק מכלי התפעול דורשים manifests מקוריים.

## מאגרים קשורים

[GitHub](https://github.com/pathcosmos/korean-dense-llm/blob/main/cards/README.he.md) · [README](https://github.com/pathcosmos/korean-dense-llm)
[GH / שימוש](https://github.com/pathcosmos/korean-dense-llm/blob/main/docs/PUBLIC_GUIDE.md) · [GH / מיטוב ומדידות](https://github.com/pathcosmos/korean-dense-llm/blob/main/experiments/README.md) · [GH / אימות](https://github.com/pathcosmos/korean-dense-llm/blob/main/docs/PUBLIC_VALIDATION.md)

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
