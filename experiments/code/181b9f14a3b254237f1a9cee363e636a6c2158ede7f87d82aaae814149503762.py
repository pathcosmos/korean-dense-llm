"""Build consistent, translated dataset cards from measured release facts."""
import hashlib,json,shutil,sys
from pathlib import Path
root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))

# Every translation shares the generated numeric/source/license tables.
# Text describes publication state, not an already-hosted full corpus.
LANGS={
'ko':{'name':'한국어','head':['개요','공개 상태','구성 및 규모','정제와 분리','토크나이저와 형식','출처와 라이선스','검증 범위','사용 방법','한계','문의와 인용'],
'overview':'한국어 중심 LLM 사전학습을 위한 코퍼스입니다. 한국어 위키·웹 문서, 영어 교육 웹 문서, 수학, 코드를 모아 추가 정제·정확중복 제거·그룹 분리·토큰화를 수행했습니다. 제작자는 pathcosmos이며 원본 자료의 제작자는 아래 출처에 표시합니다. 모델 가중치나 대화용 SFT 데이터셋은 포함하지 않습니다.',
'status':'현재 공개된 것은 코드 미리보기, 고정 출처 목록, 라이선스 범위와 문의 안내입니다. 전체 텍스트·토큰·분리 정보의 업로드는 준비 중입니다. 아래 수치는 로컬에서 준비·검증한 자료의 규모이며, 현재 이 저장소에서 다운로드할 수 있는 본문 용량을 뜻하지 않습니다.',
'language_note':'이 문서의 번역 언어가 학습 데이터에 해당 언어가 포함된다는 의미는 아닙니다. 주된 자연어는 한국어·영어이며, 코드·수학 및 웹 자료에 다른 언어가 섞일 수 있습니다.',
'process':'웹 자료에는 공백 정리, 길이·반복·언어 관련 휴리스틱, 일부 개인정보 패턴 치환, 파일 내 정확중복 제거를 적용했습니다. 웹 소스 간 중복 제거는 위키→FineWeb2→HPLT→FineWeb-Edu→OpenWebMath 순서의 NFC 본문 SHA256 판정으로 수행했습니다. 코드는 원래 UTF8·대소문자·들여쓰기를 보존하고 경로·언어·길이·라이선스 라벨·일부 생성/secret 패턴을 필터링했습니다.',
'split':'train/eval은 문서 그룹 단위로 분리했습니다. 정확히 같은 본문, 웹의 NFC 동등 본문, 코드 저장소의 연결 그룹을 양쪽에 나누지 않는 정책입니다. 공개 validation은 기존 eval에 대응합니다. 전역 유사중복 제거 또는 모든 형태의 오염 제거를 의미하지 않습니다.',
'tokenizer':'토크나이저는 어휘 65,536개의 raw BPE입니다. BOS=0, EOS=1, PAD=2이며 별도의 UNK 토큰은 없습니다. 원문 Unicode를 보존하는 정책을 사용합니다. 바이너리는 little-endian uint16이며 각 문서 끝에 EOS 하나를 추가합니다. 다른 토크나이저로 학습하려면 공개 텍스트에서 다시 토큰화해야 합니다. AutoTokenizer 호환과 완성된 다운로드/학습 예제는 아직 검증 중입니다.',
'license':'자체 코드는 MIT, 자체 설명 문서는 CC BY 4.0입니다. 이 허락은 원본 코퍼스 전체의 이용 조건을 대체하지 않습니다. 데이터베이스/포장 조건과 개별 본문 권리를 구별하며, 문서별 귀속·예외·코드 고지 작업을 진행 중입니다. 모든 원문이 상업적으로 자유롭게 재배포 가능하다고 보장하지 않습니다.',
'verify':'전체 128,856,258행의 portable split을 읽기 전용 canonical DB에서 내보내고 재읽기 논리 대조를 통과했습니다. 1,452개 token bin의 SHA256·문서 수·EOS 경계를 대조해 122,402,393,813토큰의 문서 위치를 검증했습니다. 위키 455,397문서의 원본 URL 후보를 복원했고, 다중 출처 7건은 모두 보존하며 원래 생존 행을 검증했습니다. 이 결과는 원문 전체의 재배포 권리 확정이나 새 전역 유사중복 검사와 다릅니다.',
'usage':'현재 사용 가능한 코드·문서는 기본 카드의 다운로드 예제로 받을 수 있습니다. 코드 묶음은 실제 3B 학습 계약의 49개 파일과 추가 의존 모듈을 포함한 준비판이며, 독립 환경에서의 전체 학습 재현은 아직 완료되지 않았습니다. 본문 공개 뒤에는 텍스트부터 자신의 토크나이저를 만드는 경로와 제공된 토큰·설정을 사용하는 경로를 제공할 예정입니다.',
'limits':'웹의 편향, 유해 표현, 사실 오류, 개인정보 또는 secret 잔존 가능성이 있습니다. 정제만으로 이를 완전히 제거했다고 주장하지 않습니다. 공식 benchmark 무오염·전역 유사중복 없음·일정한 모델 성능은 입증하지 않았습니다. 평가 데이터는 학습에 섞지 말고, 데이터 양과 실제 sampling weight를 구별해야 합니다.',
'contact':'제작자: pathcosmos. 문의·비공개 삭제 요청: miracle.yang@pathcosmos.com. URL, doc_id 또는 저장소/파일 경로를 함께 알려 주세요. 공개 이슈에 민감한 개인정보를 게시하지 마세요. 이 코퍼스와 함께 실제 사용한 원 데이터 제작자도 인용해 주세요.'},
'en':{'name':'English','head':['Overview','Publication status','Composition and scale','Processing and splits','Tokenizer and format','Sources and licensing','Validation scope','How to use','Limitations','Contact and citation'],
'overview':'A Korean-centric corpus for language-model pretraining, combining Korean encyclopedia/web text, English educational web text, mathematics, and code. pathcosmos performed additional filtering, exact deduplication, grouped splitting, and tokenization. Original data creators are credited below. This release does not contain model weights or an instruction-tuning dataset.',
'status':'Available now: a code preview, pinned source inventory, license-scope notes, and contact policy. Upload of the complete text, tokens, and assignment payloads is still in preparation. Figures below describe locally prepared and validated artifacts, not the amount of corpus content currently hosted in this repository.',
'language_note':'Documentation translations do not imply training-data coverage in those languages. The main natural languages are Korean and English; web, math, and code content may contain other languages.',
'process':'Web processing includes whitespace cleanup, length/repetition/language heuristics, selected PII-pattern masking, and within-shard exact deduplication. Cross-source web deduplication uses NFC body SHA256 in this order: Wikipedia → FineWeb2 → HPLT → FineWeb-Edu → OpenWebMath. Code preserves original UTF8, case, and indentation, with path, language, length, license-label, generated-content, and selected secret-pattern filters.',
'split':'Documents are split by connected groups of identical content, NFC-equivalent web content, and code repositories. Public validation corresponds to the existing eval split. This policy does not establish global near-duplicate removal or freedom from every form of evaluation contamination.',
'tokenizer':'Raw BPE with vocabulary size 65,536; BOS=0, EOS=1, PAD=2, and no separate UNK token. The policy preserves source Unicode. Token binaries use little-endian uint16 with one appended EOS per document. Users choosing a different tokenizer must retokenize the released text. AutoTokenizer compatibility and complete download/training examples remain under validation.',
'license':'Original project code is MIT; original explanatory prose is CC BY 4.0. These grants do not replace upstream corpus conditions. Database/packaging rights and individual content rights are distinct. Document attribution, exceptions, and code notices are still being completed; unrestricted commercial redistribution of every original text is not asserted.',
'verify':'All 128,856,258 portable split rows were exported from the read-only canonical DB and logically compared after decoding. SHA256, document counts, and EOS boundaries of 1,452 token binaries were checked to establish document offsets for 122,402,393,813 tokens. Source URL candidates were recovered for 455,397 Wikipedia documents; all candidates for seven multiple-origin cases were retained and their original surviving rows verified. This is not per-document rights clearance or a new global near-deduplication audit.',
'usage':'Use the download example on the main card for currently available code and documents. The code preview includes 49 files pinned by the actual 3B training contract and additional dependencies. Full clean-room training reproduction is not yet verified. After payload publication, users will be able to start from text with their own tokenizer or reuse the supplied token inputs and recipe.',
'limits':'Web bias, harmful language, factual errors, residual personal information, and secrets may remain. Filtering is not a guarantee of their absence. Official-benchmark decontamination, global absence of near duplicates, and a particular model-quality level have not been established. Keep validation separate from training and distinguish stored token amounts from sampling weights.',
'contact':'Publisher: pathcosmos. Questions and private removal requests: miracle.yang@pathcosmos.com. Include a URL, doc_id, or repository/file path; do not post sensitive personal information in public issues. Cite the upstream datasets actually used as well as this corpus.'},
'zh-CN':{'name':'简体中文','head':['概述','发布状态','组成与规模','处理与划分','分词器与格式','来源与许可','验证范围','使用方式','局限性','联系与引用'],
'overview':'面向韩语中心语言模型预训练的语料库，结合韩语百科与网页、英语教育网页、数学和代码。pathcosmos 进行了额外过滤、精确去重、分组划分及分词。原始数据作者见下方来源。本项目不包含模型权重或指令微调数据集。',
'status':'目前公开的是代码预览、固定版本的来源清单、许可范围与联系说明。完整文本、词元和划分文件仍在准备上传。以下数值表示本地已准备并验证的数据，并非仓库当前托管的正文数量。',
'language_note':'文档提供某种语言的译文，不代表训练数据覆盖该语言。主要自然语言是韩语和英语；网页、数学和代码可能含有其他语言。',
'process':'网页处理包括空白整理、长度/重复/语言启发式过滤、部分个人信息模式遮蔽及分片内精确去重。跨来源网页按 Wikipedia→FineWeb2→HPLT→FineWeb-Edu→OpenWebMath 的顺序，以 NFC 正文 SHA256 去重。代码保留原始 UTF8、大小写和缩进，并按路径、语言、长度、许可标签、生成内容及部分密钥模式过滤。',
'split':'按相同正文、NFC 等价网页和代码仓库形成的连通组划分训练与评估数据。公开 validation 对应现有 eval。该策略不等于已完成全局近似去重或清除了所有评估污染。',
'tokenizer':'原始 BPE，词表大小 65,536；BOS=0、EOS=1、PAD=2，无独立 UNK。保留源文本 Unicode。二进制格式为小端 uint16，每篇文档追加一个 EOS。更换分词器时需重新对文本分词。AutoTokenizer 兼容性和完整下载/训练示例仍在验证。',
'license':'项目原创代码采用 MIT，原创说明文档采用 CC BY 4.0；这些许可不替代上游语料的条件。数据库/封装权利与单篇内容权利不同。逐篇归属、例外及代码声明仍在完善，未声明所有原文均可无限制商业再分发。',
'verify':'已从只读规范数据库导出并回读比较全部 128,856,258 条划分记录。对 1,452 个词元文件核对 SHA256、文档数及 EOS 边界，验证 122,402,393,813 个词元的文档偏移。已恢复 455,397 篇维基文档的来源 URL 候选；7 个多来源案例保留全部候选并验证原始保留行。这不代表逐篇权利已获确认或进行了新的全局近似去重审计。',
'usage':'当前代码与文档可通过主卡片的下载示例获取。代码预览包含实际 3B 训练契约固定的 49 个文件及额外依赖；尚未完成独立全新环境中的完整训练复现。正文发布后将支持自建分词器或复用所提供的词元与配置。',
'limits':'网页偏见、有害内容、事实错误、残留个人信息和密钥仍可能存在。未证明官方评测无污染、全局无近似重复或达到特定模型质量。请分开训练与验证数据，并区分存储词元数量和采样权重。',
'contact':'发布者：pathcosmos。咨询及私密删除请求：miracle.yang@pathcosmos.com。请提供 URL、doc_id 或仓库/文件路径，勿在公开议题发布敏感个人信息。引用本语料时也请引用实际使用的上游数据。'},
'zh-TW':{'name':'繁體中文','head':['概述','發布狀態','組成與規模','處理與切分','分詞器與格式','來源與授權','驗證範圍','使用方式','限制','聯絡與引用'],
'overview':'以韓語為核心的語言模型預訓練語料庫，結合韓語百科與網頁、英語教育網頁、數學及程式碼。pathcosmos 進行額外篩選、精確去重、群組切分與分詞。原始資料作者列於下方。本項目不包含模型權重或指令微調資料集。',
'status':'目前已公開程式碼預覽、固定版本來源清單、授權範圍與聯絡方式。完整文本、詞元及切分檔案仍在準備上傳。以下數值為本機已準備並驗證的資料，並非目前倉庫已託管的正文數量。',
'language_note':'文件的翻譯語言不代表訓練資料涵蓋該語言。主要自然語言是韓語及英語；網頁、數學與程式碼可能包含其他語言。',
'process':'網頁處理包括空白整理、長度/重複/語言啟發式篩選、部分個資模式遮蔽及分片內精確去重。跨來源網頁依 Wikipedia→FineWeb2→HPLT→FineWeb-Edu→OpenWebMath 順序，以 NFC 正文 SHA256 去重。程式碼保留原始 UTF8、大小寫及縮排，並依路徑、語言、長度、授權標籤、生成內容與部分密鑰模式篩選。',
'split':'以相同正文、NFC 等價網頁及程式碼儲存庫的連通群組切分訓練與評估資料。公開 validation 對應原有 eval。這不等於完成全域近似去重或消除所有評估污染。',
'tokenizer':'原始 BPE，詞彙表 65,536；BOS=0、EOS=1、PAD=2，無獨立 UNK。保留來源 Unicode。二進位格式為小端序 uint16，每份文件追加一個 EOS。更換分詞器須重新對文本分詞。AutoTokenizer 相容性及完整下載/訓練範例仍在驗證。',
'license':'專案原創程式碼採 MIT，原創說明文件採 CC BY 4.0；不取代上游語料條件。資料庫/封裝權利與個別內容權利有別。文件歸屬、例外及程式碼聲明仍在補齊，未主張所有原文均可不受限制地商業再散布。',
'verify':'已從唯讀規範資料庫匯出並回讀比對全部 128,856,258 筆切分紀錄。核對 1,452 個詞元檔案的 SHA256、文件數及 EOS 邊界，驗證 122,402,393,813 個詞元的文件位置。已恢復 455,397 篇維基文件的來源 URL 候選；7 個多來源案例保留所有候選並驗證原始保留列。這不代表個別權利已確認或已進行新的全域近似去重稽核。',
'usage':'目前程式碼與文件可由主卡片的下載範例取得。程式碼預覽包含實際 3B 訓練契約固定的 49 個檔案及額外相依模組；完整獨立環境訓練重現尚未驗證。正文發布後可建立自己的分詞器，或沿用提供的詞元與設定。',
'limits':'仍可能有網頁偏見、有害內容、事實錯誤、殘留個資及密鑰。尚未證明官方評測無污染、全域無近似重複或具有特定模型品質。請分開訓練與驗證資料，並區分儲存詞元數與取樣權重。',
'contact':'發布者：pathcosmos。詢問與非公開刪除請求：miracle.yang@pathcosmos.com。請附 URL、doc_id 或儲存庫/檔案路徑，勿在公開議題張貼敏感個資。引用本語料時也請引用實際使用的上游資料。'},
'ja':{'name':'日本語','head':['概要','公開状況','構成と規模','処理と分割','トークナイザーと形式','出典とライセンス','検証範囲','利用方法','制約','連絡先と引用'],
'overview':'韓国語を中心とする言語モデル事前学習用コーパスです。韓国語の百科事典・ウェブ、英語の教育ウェブ、数学、コードを組み合わせ、pathcosmos が追加フィルタリング、完全一致重複除去、グループ分割、トークン化を行いました。原データの作成者は下記に示します。モデル重みや指示学習データは含みません。',
'status':'現在公開済みなのはコードのプレビュー、固定版の出典一覧、ライセンス範囲、連絡案内です。全テキスト・トークン・分割情報のアップロードは準備中です。下記数値はローカルで準備・検証した資料の規模であり、現在ホストされている本文量ではありません。',
'language_note':'説明文の翻訳言語は、その言語の学習データが含まれることを意味しません。主な自然言語は韓国語・英語で、ウェブ・数学・コードには他言語が混在し得ます。',
'process':'ウェブでは空白整理、長さ・反復・言語に関するヒューリスティック、一部の個人情報パターンのマスク、シャード内完全一致重複除去を実施しました。ソース間の除去は Wikipedia→FineWeb2→HPLT→FineWeb-Edu→OpenWebMath の順に NFC 本文 SHA256 で判定します。コードは元の UTF8・大文字小文字・字下げを保持し、パス・言語・長さ・ライセンスラベル・生成コード・一部の秘密情報パターンでフィルタリングします。',
'split':'同一本文、NFC 同等のウェブ本文、コードリポジトリの連結グループ単位で train/eval を分けます。公開 validation は既存 eval に対応します。全体の近似重複除去や、あらゆる評価汚染の排除を示すものではありません。',
'tokenizer':'語彙数65,536の raw BPE。BOS=0、EOS=1、PAD=2、独立 UNK はありません。元の Unicode を保持します。バイナリは little-endian uint16、文書末尾に EOS を1個追加します。別のトークナイザーを使う場合はテキストから再トークン化してください。AutoTokenizer 互換性と完全なダウンロード・学習例は検証中です。',
'license':'独自コードは MIT、独自説明文は CC BY 4.0。上流コーパスの条件を置き換えるものではありません。DB・パッケージの権利と各本文の権利は別です。文書別帰属・例外・コードの告知整備を進めており、すべての原文の無制限な商用再配布を保証しません。',
'verify':'読み取り専用の正規 DB から全128,856,258行の分割情報を出力し、再読込後に論理一致を確認しました。1,452個のトークンファイルの SHA256・文書数・EOS 境界を照合し、122,402,393,813トークンの文書位置を検証しました。Wikipedia 455,397文書の URL 候補を復元し、複数出典7件は全候補を保持して元の生存行を検証しました。本文権利の個別確認や新たな全体近似重複監査を意味しません。',
'usage':'公開済みコード・文書は基本カードのダウンロード例から取得できます。プレビューは実際の3B学習契約で固定した49ファイルと追加依存モジュールを含みます。クリーン環境での学習全体の再現は未検証です。本文公開後は独自トークナイザーの作成、または提供トークン・設定の再利用が可能になる予定です。',
'limits':'ウェブ由来の偏り、有害表現、誤情報、個人情報・秘密情報の残存があり得ます。公式ベンチマーク無汚染、全体の近似重複なし、特定のモデル品質は立証していません。検証データを学習に混ぜず、保有トークン量とサンプリング比率を区別してください。',
'contact':'公開者：pathcosmos。問い合わせ・非公開の削除依頼：miracle.yang@pathcosmos.com。URL、doc_id、リポジトリ/ファイルパスを添え、公開 issue に機微な個人情報を投稿しないでください。本コーパスと併せて実際に用いた上流データも引用してください。'},
'fr':{'name':'Français','head':['Présentation','État de publication','Composition et volume','Traitement et partitions','Tokeniseur et format','Sources et licences','Portée des vérifications','Utilisation','Limites','Contact et citation'],
'overview':'Corpus de préentraînement de modèles de langue centré sur le coréen : encyclopédie et web coréens, web éducatif anglais, mathématiques et code. pathcosmos a effectué le filtrage supplémentaire, la déduplication exacte, le partitionnement par groupes et la tokenisation. Les auteurs des données originales sont crédités ci-dessous. Aucun poids de modèle ni jeu de données de réglage par instructions n’est inclus.',
'status':'Sont disponibles : un aperçu du code, les sources à révisions fixes, la portée des licences et les coordonnées. Le dépôt des textes, tokens et partitions complets est en préparation. Les chiffres indiquent les artefacts préparés et vérifiés localement, pas le contenu intégral actuellement hébergé.',
'language_note':'Les langues de traduction de cette documentation ne constituent pas une déclaration de couverture linguistique des données. Les principales langues naturelles sont le coréen et l’anglais ; le web, les mathématiques et le code peuvent contenir d’autres langues.',
'process':'Le web fait l’objet d’un nettoyage des espaces, de filtres heuristiques de longueur/répétition/langue, du masquage de certains motifs de données personnelles et d’une déduplication exacte par fragment. La déduplication intersources utilise le SHA256 du texte normalisé NFC dans l’ordre Wikipedia→FineWeb2→HPLT→FineWeb-Edu→OpenWebMath. Le code conserve UTF8, casse et indentation ; les filtres portent sur chemin, langue, longueur, étiquette de licence, contenu généré et certains secrets.',
'split':'Les partitions respectent les groupes connexes de contenu identique, de textes web équivalents en NFC et de dépôts de code. La partition publique validation correspond à eval. Cela ne prouve ni une déduplication approximative globale ni l’absence de toute contamination des évaluations.',
'tokenizer':'BPE brut de 65,536 entrées ; BOS=0, EOS=1, PAD=2, sans token UNK distinct. Unicode d’origine est conservé. Les fichiers sont en uint16 petit-boutiste avec un EOS ajouté par document. Un autre tokeniseur nécessite une nouvelle tokenisation du texte. La compatibilité AutoTokenizer et les exemples complets de téléchargement/entraînement sont encore en validation.',
'license':'Code original du projet : MIT ; documentation originale : CC BY 4.0. Ces licences ne remplacent pas les conditions des corpus sources. Les droits sur la base ou son conditionnement diffèrent de ceux des contenus individuels. Attributions, exceptions et notices du code restent en cours ; une redistribution commerciale sans restriction de chaque texte n’est pas affirmée.',
'verify':'Les 128,856,258 affectations ont été exportées depuis la base canonique en lecture seule puis comparées après relecture. SHA256, nombres de documents et frontières EOS ont été vérifiés pour 1,452 fichiers, soit les positions des documents dans 122,402,393,813 tokens. Les URL candidates de 455,397 documents Wikipédia ont été récupérées ; les sept cas à sources multiples conservent toutes les attributions et la ligne initialement retenue a été vérifiée. Ce n’est pas une validation juridique individuelle ni un nouvel audit global de quasi-doublons.',
'usage':'Le code et les documents disponibles se téléchargent avec l’exemple de la carte principale. L’aperçu comprend 49 fichiers fixés par le contrat réel d’entraînement 3B et des dépendances supplémentaires. La reproduction complète dans un environnement neuf n’est pas encore validée. Après publication des données, il sera possible de repartir du texte ou de réutiliser les tokens et la recette.',
'limits':'Biais du web, propos nuisibles, erreurs factuelles, données personnelles ou secrets résiduels peuvent subsister. L’absence de contamination des benchmarks officiels, de quasi-doublons globaux ou un niveau précis de qualité du modèle ne sont pas établis. Séparer validation et entraînement et distinguer le stock de tokens des poids d’échantillonnage.',
'contact':'Éditeur : pathcosmos. Questions et demandes privées de retrait : miracle.yang@pathcosmos.com. Fournir URL, doc_id ou dépôt/chemin de fichier ; ne pas publier de données sensibles dans les tickets publics. Citer aussi les jeux de données sources réellement utilisés.'},
'de':{'name':'Deutsch','head':['Überblick','Veröffentlichungsstand','Zusammensetzung und Umfang','Verarbeitung und Aufteilung','Tokenizer und Format','Quellen und Lizenzen','Prüfumfang','Verwendung','Einschränkungen','Kontakt und Zitation'],
'overview':'Koreanisch ausgerichtetes Korpus zum Vortraining von Sprachmodellen mit koreanischen Enzyklopädie-/Webtexten, englischen Bildungswebtexten, Mathematik und Code. pathcosmos übernahm zusätzliche Filterung, exakte Deduplizierung, gruppenbasierte Aufteilung und Tokenisierung. Die ursprünglichen Datenautoren sind unten genannt. Modellgewichte und Instruction-Tuning-Daten sind nicht enthalten.',
'status':'Veröffentlicht sind eine Code-Vorschau, fest versionierte Quellenangaben, Lizenzumfang und Kontaktinformationen. Vollständige Texte, Tokens und Aufteilungsdateien werden noch vorbereitet. Die Zahlen beschreiben lokal vorbereitete und geprüfte Artefakte, nicht bereits vollständig gehostete Daten.',
'language_note':'Übersetzungen dieser Dokumentation bedeuten keine entsprechende Sprachabdeckung der Trainingsdaten. Die wichtigsten natürlichen Sprachen sind Koreanisch und Englisch; Web-, Mathematik- und Codeinhalte können weitere Sprachen enthalten.',
'process':'Webverarbeitung: Bereinigung von Leerraum, Heuristiken für Länge/Wiederholung/Sprache, Maskierung bestimmter personenbezogener Muster und exakte Deduplizierung je Shard. Quellenübergreifend wird NFC-Text-SHA256 in der Reihenfolge Wikipedia→FineWeb2→HPLT→FineWeb-Edu→OpenWebMath verwendet. Code behält UTF8, Groß-/Kleinschreibung und Einrückung; Filter betreffen Pfad, Sprache, Länge, Lizenzlabel, generierte Inhalte und ausgewählte Geheimnismuster.',
'split':'Zusammenhängende Gruppen identischer Inhalte, NFC-äquivalenter Webtexte und Code-Repositories bleiben jeweils in einer Partition. Öffentliches validation entspricht eval. Dies belegt weder globale Entfernung ähnlicher Duplikate noch vollständige Freiheit von Evaluationskontamination.',
'tokenizer':'Raw-BPE mit 65,536 Vokabulareinträgen; BOS=0, EOS=1, PAD=2, kein separates UNK. Ursprüngliches Unicode bleibt erhalten. Binärformat: Little-Endian uint16, ein angehängtes EOS je Dokument. Bei anderem Tokenizer muss der Text neu tokenisiert werden. AutoTokenizer-Kompatibilität und vollständige Download-/Trainingsbeispiele werden noch geprüft.',
'license':'Eigener Projektcode: MIT; eigene Erläuterungen: CC BY 4.0. Diese Lizenzen ersetzen nicht die Bedingungen der Quelldaten. Datenbank-/Verpackungsrechte und Einzelinhaltsrechte sind getrennt. Attributionen, Ausnahmen und Code-Hinweise werden noch vervollständigt; uneingeschränkte kommerzielle Weiterverteilung aller Originaltexte wird nicht behauptet.',
'verify':'Alle 128,856,258 Zuordnungen wurden aus der schreibgeschützten kanonischen DB exportiert und nach erneutem Lesen logisch verglichen. SHA256, Dokumentzahlen und EOS-Grenzen von 1,452 Token-Dateien wurden für 122,402,393,813 Tokens geprüft. Quellen-URL-Kandidaten für 455,397 Wikipedia-Dokumente wurden rekonstruiert; sieben Mehrfachquellenfälle behalten alle Zuordnungen und die ursprüngliche überlebende Zeile wurde geprüft. Dies ist keine individuelle Rechtefreigabe oder neue globale Ähnlichkeitsprüfung.',
'usage':'Verfügbaren Code und Dokumente über das Download-Beispiel der Hauptkarte beziehen. Die Vorschau enthält 49 Dateien des tatsächlichen 3B-Trainingsvertrags sowie zusätzliche Abhängigkeiten. Eine vollständige Trainingsreproduktion in frischer Umgebung ist noch nicht verifiziert. Nach Veröffentlichung der Nutzdaten kann man mit eigenem Tokenizer beginnen oder Tokens und Rezept wiederverwenden.',
'limits':'Webverzerrungen, schädliche Inhalte, Sachfehler sowie personenbezogene Daten oder Geheimnisse können verbleiben. Kontaminationsfreiheit offizieller Benchmarks, globale Abwesenheit ähnlicher Duplikate und bestimmte Modellqualität sind nicht nachgewiesen. Validierung von Training trennen und Tokenbestand von Sampling-Gewichten unterscheiden.',
'contact':'Herausgeber: pathcosmos. Fragen und private Löschanfragen: miracle.yang@pathcosmos.com. URL, doc_id oder Repository/Dateipfad angeben; keine sensiblen Daten in öffentlichen Issues veröffentlichen. Auch die tatsächlich genutzten Quelldatensätze zitieren.'},
'es':{'name':'Español','head':['Resumen','Estado de publicación','Composición y tamaño','Procesamiento y particiones','Tokenizador y formato','Fuentes y licencias','Alcance de la validación','Uso','Limitaciones','Contacto y cita'],
'overview':'Corpus centrado en coreano para preentrenar modelos de lenguaje: enciclopedia y web en coreano, web educativa en inglés, matemáticas y código. pathcosmos realizó filtrado adicional, deduplicación exacta, división por grupos y tokenización. Se acredita abajo a los autores originales. No incluye pesos de modelo ni datos de ajuste por instrucciones.',
'status':'Ya están disponibles una vista previa del código, fuentes con revisiones fijas, alcance de licencias y contacto. La carga de todos los textos, tokens y particiones está en preparación. Las cifras describen artefactos preparados y verificados localmente, no todo el contenido actualmente alojado.',
'language_note':'Los idiomas de traducción de la documentación no indican cobertura de esos idiomas en el entrenamiento. Los principales idiomas naturales son coreano e inglés; web, matemáticas y código pueden incluir otros.',
'process':'El procesamiento web incluye limpieza de espacios, heurísticas de longitud/repetición/idioma, ocultación de ciertos patrones personales y deduplicación exacta por fragmento. Entre fuentes se usa SHA256 del texto NFC en orden Wikipedia→FineWeb2→HPLT→FineWeb-Edu→OpenWebMath. El código conserva UTF8, mayúsculas y sangría; se filtra por ruta, idioma, longitud, etiqueta de licencia, generación y ciertos patrones de secretos.',
'split':'Se mantienen unidos los grupos conexos de contenido idéntico, textos web equivalentes en NFC y repositorios de código. La partición pública validation corresponde a eval. Esto no demuestra deduplicación aproximada global ni ausencia de toda contaminación de evaluación.',
'tokenizer':'BPE sin normalización Unicode adicional, vocabulario 65,536; BOS=0, EOS=1, PAD=2, sin UNK separado. Binarios uint16 little-endian con un EOS añadido por documento. Otro tokenizador requiere volver a tokenizar el texto. La compatibilidad AutoTokenizer y los ejemplos completos de descarga/entrenamiento siguen en validación.',
'license':'Código original del proyecto: MIT; documentación original: CC BY 4.0. No sustituyen las condiciones del corpus de origen. Los derechos de base de datos/empaquetado difieren de los derechos del contenido individual. Atribuciones, excepciones y avisos del código siguen en preparación; no se afirma redistribución comercial irrestricta de todos los textos.',
'verify':'Se exportaron y compararon tras releer las 128,856,258 asignaciones desde la base canónica de solo lectura. Se comprobaron SHA256, recuentos de documentos y límites EOS de 1,452 archivos para ubicar documentos en 122,402,393,813 tokens. Se recuperaron URL candidatas para 455,397 documentos de Wikipedia; siete casos con múltiples fuentes conservan todas las atribuciones y se verificó la fila originalmente retenida. No equivale a autorización jurídica individual ni a una nueva auditoría global de similitud.',
'usage':'El código y la documentación se obtienen con el ejemplo de la tarjeta principal. La vista previa incluye 49 archivos fijados por el contrato real de entrenamiento 3B y dependencias adicionales. La reproducción completa en un entorno nuevo aún no está verificada. Tras publicar los datos se podrá usar un tokenizador propio o reutilizar tokens y receta.',
'limits':'Pueden persistir sesgos web, lenguaje dañino, errores, información personal o secretos. No se ha demostrado ausencia de contaminación en benchmarks oficiales, de casi duplicados globales ni un nivel concreto de calidad del modelo. Separar validación y entrenamiento y distinguir cantidades almacenadas de pesos de muestreo.',
'contact':'Editor: pathcosmos. Consultas y solicitudes privadas de retirada: miracle.yang@pathcosmos.com. Incluir URL, doc_id o repositorio/ruta; no publicar datos sensibles en incidencias públicas. Citar también las fuentes realmente utilizadas.'},
'pt-BR':{'name':'Português (Brasil)','head':['Visão geral','Estado da publicação','Composição e tamanho','Processamento e divisões','Tokenizador e formato','Fontes e licenças','Escopo da validação','Como usar','Limitações','Contato e citação'],
'overview':'Corpus de pré-treinamento centrado em coreano, combinando enciclopédia/web em coreano, web educacional em inglês, matemática e código. pathcosmos realizou filtragem adicional, deduplicação exata, divisão por grupos e tokenização. Os criadores originais são creditados abaixo. Não inclui pesos de modelo nem dados de ajuste por instruções.',
'status':'Disponíveis: prévia do código, inventário de fontes com revisões fixas, escopo das licenças e contato. O envio de todos os textos, tokens e divisões está em preparação. Os números descrevem artefatos preparados e verificados localmente, não o volume integral já hospedado.',
'language_note':'Os idiomas das traduções não indicam cobertura desses idiomas nos dados de treinamento. Os principais idiomas naturais são coreano e inglês; web, matemática e código podem incluir outros.',
'process':'A web passa por limpeza de espaços, heurísticas de tamanho/repetição/idioma, mascaramento de alguns padrões pessoais e deduplicação exata por fragmento. Entre fontes, usa-se SHA256 do texto NFC na ordem Wikipedia→FineWeb2→HPLT→FineWeb-Edu→OpenWebMath. O código preserva UTF8, maiúsculas e indentação, com filtros de caminho, idioma, tamanho, rótulo de licença, conteúdo gerado e alguns padrões de segredos.',
'split':'Grupos conexos de conteúdo idêntico, texto web equivalente em NFC e repositórios de código ficam na mesma partição. validation pública corresponde a eval. Isso não comprova deduplicação aproximada global nem ausência de toda contaminação de avaliação.',
'tokenizer':'BPE raw com vocabulário de 65,536; BOS=0, EOS=1, PAD=2, sem UNK separado. Preserva Unicode original. Binários uint16 little-endian, com um EOS acrescentado por documento. Outro tokenizador exige retokenizar o texto. Compatibilidade AutoTokenizer e exemplos completos de download/treinamento ainda estão em validação.',
'license':'Código original: MIT; documentação original: CC BY 4.0. Essas licenças não substituem as condições das fontes. Direitos de banco de dados/empacotamento diferem dos direitos do conteúdo individual. Atribuições, exceções e avisos de código estão sendo concluídos; não se afirma redistribuição comercial irrestrita de todos os textos.',
'verify':'Todas as 128,856,258 atribuições foram exportadas da base canônica somente leitura e comparadas após releitura. SHA256, contagens e limites EOS de 1,452 arquivos foram verificados para localizar documentos em 122,402,393,813 tokens. Foram recuperadas URL candidatas de 455,397 documentos da Wikipédia; sete casos com várias origens mantêm todas as atribuições e a linha originalmente preservada foi verificada. Não constitui liberação individual de direitos nem nova auditoria global de quase duplicatas.',
'usage':'Use o exemplo da página principal para baixar código e documentação disponíveis. A prévia inclui 49 arquivos fixados pelo contrato real de treinamento 3B e dependências extras. A reprodução completa em ambiente novo ainda não foi verificada. Após a publicação dos dados, será possível começar do texto ou reutilizar tokens e receita.',
'limits':'Podem persistir vieses da web, conteúdo nocivo, erros factuais, dados pessoais e segredos. Não foram comprovadas ausência de contaminação de benchmarks oficiais, ausência global de quase duplicatas ou qualidade específica do modelo. Separe validação de treinamento e diferencie estoque de tokens de pesos de amostragem.',
'contact':'Publicador: pathcosmos. Dúvidas e pedidos privados de remoção: miracle.yang@pathcosmos.com. Inclua URL, doc_id ou repositório/caminho; não publique dados sensíveis em issues. Cite também as fontes efetivamente utilizadas.'},
'hi':{'name':'हिन्दी','head':['परिचय','प्रकाशन की स्थिति','संरचना और आकार','प्रसंस्करण और विभाजन','टोकनाइज़र और प्रारूप','स्रोत और लाइसेंस','सत्यापन का दायरा','उपयोग','सीमाएँ','संपर्क और उद्धरण'],
'overview':'यह कोरियाई-केंद्रित भाषा मॉडल पूर्व-प्रशिक्षण कॉर्पस है। इसमें कोरियाई विश्वकोश/वेब, अंग्रेज़ी शैक्षिक वेब, गणित और कोड शामिल हैं। pathcosmos ने अतिरिक्त फ़िल्टरिंग, सटीक डुप्लिकेट हटाना, समूह-आधारित विभाजन और टोकनाइज़ेशन किया है। मूल रचनाकार नीचे सूचीबद्ध हैं। मॉडल वज़न या निर्देश-आधारित फ़ाइन-ट्यूनिंग डेटा शामिल नहीं हैं।',
'status':'अभी कोड पूर्वावलोकन, निश्चित संस्करणों की स्रोत सूची, लाइसेंस का दायरा और संपर्क उपलब्ध हैं। पूरा पाठ, टोकन और विभाजन फ़ाइलें अपलोड करने की तैयारी जारी है। नीचे के आँकड़े स्थानीय रूप से तैयार और सत्यापित सामग्री के हैं, अभी होस्ट किए गए पूरे कॉर्पस के नहीं।',
'language_note':'दस्तावेज़ के अनुवाद की भाषा से उस भाषा के प्रशिक्षण डेटा की उपलब्धता सिद्ध नहीं होती। मुख्य प्राकृतिक भाषाएँ कोरियाई और अंग्रेज़ी हैं; वेब, गणित और कोड में अन्य भाषाएँ हो सकती हैं।',
'process':'वेब पर रिक्त-स्थान सफाई, लंबाई/दोहराव/भाषा संबंधी नियम, कुछ निजी जानकारी पैटर्न का मास्किंग और शार्ड के भीतर सटीक डुप्लिकेट हटाना लागू है। विभिन्न स्रोतों में NFC पाठ SHA256 का क्रम Wikipedia→FineWeb2→HPLT→FineWeb-Edu→OpenWebMath है। कोड का मूल UTF8, अक्षरों का केस और इंडेंटेशन सुरक्षित रहता है; पथ, भाषा, लंबाई, लाइसेंस लेबल, जनित सामग्री और कुछ गुप्त-कुंजी पैटर्न फ़िल्टर किए जाते हैं।',
'split':'समान पाठ, NFC-समान वेब पाठ और कोड रिपॉज़िटरी के जुड़े समूहों को एक ही विभाजन में रखा जाता है। सार्वजनिक validation मौजूदा eval के अनुरूप है। यह वैश्विक लगभग-समान डुप्लिकेट हटाने या सभी मूल्यांकन संदूषण के अभाव का प्रमाण नहीं है।',
'tokenizer':'Raw BPE, शब्दावली 65,536; BOS=0, EOS=1, PAD=2, अलग UNK नहीं। मूल Unicode सुरक्षित है। बाइनरी little-endian uint16 है और हर दस्तावेज़ में एक EOS जोड़ा जाता है। अलग टोकनाइज़र के लिए पाठ को फिर टोकनाइज़ करें। AutoTokenizer संगतता और पूरे डाउनलोड/प्रशिक्षण उदाहरण अभी जाँचे जा रहे हैं।',
'license':'मूल परियोजना कोड MIT और मूल व्याख्यात्मक दस्तावेज़ CC BY 4.0 हैं। ये स्रोत कॉर्पस की शर्तों का स्थान नहीं लेते। डेटाबेस/पैकेजिंग और व्यक्तिगत सामग्री के अधिकार अलग हैं। दस्तावेज़वार श्रेय, अपवाद और कोड नोटिस अभी पूरे किए जा रहे हैं; सभी मूल पाठों के असीमित व्यावसायिक पुनर्वितरण का दावा नहीं है।',
'verify':'केवल-पठन canonical DB से सभी 128,856,258 विभाजन पंक्तियाँ निर्यात कर दोबारा पढ़कर तार्किक तुलना की गई। 1,452 टोकन फ़ाइलों के SHA256, दस्तावेज़ संख्या और EOS सीमाएँ जाँचकर 122,402,393,813 टोकन में दस्तावेज़ स्थितियाँ सत्यापित की गईं। 455,397 विकिपीडिया दस्तावेज़ों के स्रोत URL उम्मीदवार मिले; कई स्रोत वाले सात मामलों में सभी श्रेय सुरक्षित हैं और मूल रूप से बची पंक्ति सत्यापित है। यह व्यक्तिगत अधिकार-मंजूरी या नया वैश्विक समानता ऑडिट नहीं है।',
'usage':'उपलब्ध कोड और दस्तावेज़ मुख्य कार्ड के डाउनलोड उदाहरण से लें। पूर्वावलोकन में वास्तविक 3B प्रशिक्षण अनुबंध से निर्धारित 49 फ़ाइलें और अतिरिक्त निर्भरताएँ हैं। नए स्वतंत्र परिवेश में पूरे प्रशिक्षण का पुनरुत्पादन अभी सत्यापित नहीं है। डेटा प्रकाशित होने पर अपना टोकनाइज़र बनाना या दिए टोकन/रेसिपी का उपयोग संभव होगा।',
'limits':'वेब पूर्वाग्रह, हानिकारक भाषा, तथ्यात्मक त्रुटियाँ, निजी जानकारी और रहस्य बच सकते हैं। आधिकारिक बेंचमार्क संदूषण-मुक्त होना, वैश्विक लगभग-डुप्लिकेट का अभाव या निश्चित मॉडल गुणवत्ता सिद्ध नहीं है। सत्यापन को प्रशिक्षण से अलग रखें और संग्रहीत टोकन मात्रा को सैंपलिंग वज़न से अलग समझें।',
'contact':'प्रकाशक: pathcosmos। प्रश्न और निजी हटाने के अनुरोध: miracle.yang@pathcosmos.com। URL, doc_id या रिपॉज़िटरी/फ़ाइल पथ दें; सार्वजनिक मुद्दों में संवेदनशील जानकारी न डालें। उपयोग किए मूल डेटासेट को भी उद्धृत करें।'},
'ar':{'name':'العربية','head':['نظرة عامة','حالة النشر','التركيب والحجم','المعالجة والتقسيم','المجزّئ والصيغة','المصادر والتراخيص','نطاق التحقق','الاستخدام','القيود','التواصل والاستشهاد'],
'overview':'مدوّنة لتدريب النماذج اللغوية مسبقاً مع التركيز على الكورية، تجمع موسوعة وويب بالكورية، وويب تعليمياً بالإنجليزية، ورياضيات وشيفرات. أجرى pathcosmos تصفية إضافية وإزالة للتكرار المطابق وتقسيماً بحسب المجموعات وتحويلاً إلى رموز. يُذكر منشئو البيانات الأصلية أدناه. لا تتضمن أوزان نموذج أو بيانات ضبط بالتعليمات.',
'status':'المتاح حالياً هو معاينة للشيفرة وقائمة مصادر بإصدارات مثبتة ونطاق التراخيص ومعلومات التواصل. لا يزال رفع النصوص والرموز وملفات التقسيم الكاملة قيد الإعداد. الأرقام أدناه تخص مواد أُعدت وفُحصت محلياً، ولا تعني أن كامل المحتوى مستضاف الآن.',
'language_note':'لغات ترجمة الوثائق لا تعني تغطية لغوية مماثلة في بيانات التدريب. اللغتان الطبيعيتان الأساسيتان هما الكورية والإنجليزية؛ وقد تظهر لغات أخرى في الويب والرياضيات والشيفرات.',
'process':'تشمل معالجة الويب تنظيف المسافات وقواعد للطول والتكرار واللغة وإخفاء بعض أنماط المعلومات الشخصية وإزالة التطابق داخل الأجزاء. تستخدم الإزالة بين المصادر SHA256 للنص وفق NFC بترتيب Wikipedia→FineWeb2→HPLT→FineWeb-Edu→OpenWebMath. تحتفظ الشيفرة بـUTF8 وحالة الأحرف والمسافات البادئة؛ وتُصفّى بحسب المسار واللغة والطول ووسم الترخيص والمحتوى المولّد وبعض أنماط الأسرار.',
'split':'تبقى المجموعات المترابطة للنص المتطابق ونصوص الويب المتكافئة وفق NFC ومستودعات الشيفرة في قسم واحد. يقابل validation العام قسم eval الأصلي. لا يثبت ذلك إزالة التشابه القريب عالمياً أو خلوّ جميع التقييمات من التلوث.',
'tokenizer':'BPE خام بمفردات 65,536؛ BOS=0 وEOS=1 وPAD=2، ولا يوجد UNK منفصل. يُحفظ Unicode الأصلي. الصيغة uint16 بترتيب little-endian مع EOS واحد مضاف لكل وثيقة. يلزم إعادة تجزئة النص عند استخدام مجزّئ آخر. لا يزال توافق AutoTokenizer وأمثلة التنزيل والتدريب الكاملة قيد التحقق.',
'license':'شيفرة المشروع الأصلية MIT والوثائق التفسيرية الأصلية CC BY 4.0. لا تستبدل هذه التراخيص شروط المصادر. حقوق قاعدة البيانات أو التغليف تختلف عن حقوق كل محتوى. لا تزال النسب والاستثناءات وإشعارات الشيفرة قيد الاستكمال؛ ولا يُدّعى السماح بإعادة توزيع تجاري غير مقيّد لجميع النصوص.',
'verify':'صُدّرت جميع صفوف التقسيم البالغ عددها 128,856,258 من قاعدة مرجعية للقراءة فقط وقورنت منطقياً بعد إعادة قراءتها. فُحصت SHA256 وأعداد الوثائق وحدود EOS لعدد 1,452 ملفاً لتحديد مواقع الوثائق ضمن 122,402,393,813 رمزاً. استُعيدت روابط مصادر مرشحة لـ455,397 وثيقة ويكيبيديا؛ وحُفظت جميع مصادر الحالات السبع المتعددة مع التحقق من الصف الأصلي المتبقي. هذا ليس إقراراً بحقوق كل وثيقة أو تدقيقاً جديداً شاملاً للتشابه.',
'usage':'استخدم مثال التنزيل في البطاقة الرئيسية للحصول على الشيفرة والوثائق المتاحة. تشمل المعاينة 49 ملفاً مثبتاً بعقد تدريب 3B الفعلي وتبعيات إضافية. لم يُتحقق بعد من إعادة التدريب كاملاً في بيئة جديدة مستقلة. بعد نشر البيانات يمكن البدء من النص أو إعادة استخدام الرموز والوصفة.',
'limits':'قد تبقى انحيازات الويب أو لغة ضارة أو أخطاء واقعية أو معلومات شخصية وأسرار. لم يُثبت خلوّ الاختبارات الرسمية من التلوث أو الغياب الشامل للتكرار القريب أو مستوى محدد لجودة النموذج. افصل التحقق عن التدريب وميّز بين كمية الرموز وأوزان أخذ العينات.',
'contact':'الناشر: pathcosmos. للأسئلة وطلبات الإزالة الخاصة: miracle.yang@pathcosmos.com. أرفق URL أو doc_id أو مسار المستودع/الملف، ولا تنشر معلومات حساسة في القضايا العامة. استشهد أيضاً بمجموعات المصادر المستخدمة فعلياً.'},
'he':{'name':'עברית','head':['סקירה','מצב הפרסום','הרכב והיקף','עיבוד וחלוקה','טוקנייזר ופורמט','מקורות ורישיונות','היקף האימות','שימוש','מגבלות','קשר וציטוט'],
'overview':'קורפוס לאימון מקדים של מודלי שפה, בדגש על קוריאנית: אנציקלופדיה ורשת בקוריאנית, רשת חינוכית באנגלית, מתמטיקה וקוד. pathcosmos ביצע סינון נוסף, הסרת כפילויות מדויקות, חלוקה לפי קבוצות וטוקניזציה. יוצרי הנתונים המקוריים מצוינים להלן. אין כאן משקלי מודל או נתוני כוונון באמצעות הוראות.',
'status':'זמינים כעת תצוגה מוקדמת של הקוד, רשימת מקורות בגרסאות קבועות, תחולת הרישיונות ופרטי קשר. העלאת הטקסטים, הטוקנים וקובצי החלוקה המלאים עדיין בהכנה. המספרים מתארים תוצרים שהוכנו ואומתו מקומית, ולא את כל התוכן המאוחסן כעת במאגר.',
'language_note':'שפות התרגום של התיעוד אינן מעידות על כיסוי של אותן שפות בנתוני האימון. השפות הטבעיות העיקריות הן קוריאנית ואנגלית; תוכן רשת, מתמטיקה וקוד עשוי לכלול שפות נוספות.',
'process':'עיבוד הרשת כולל ניקוי רווחים, כללים לאורך/חזרות/שפה, מיסוך דפוסים מסוימים של מידע אישי והסרת כפילויות מדויקות בתוך מקטע. בין המקורות נעשה שימוש ב-SHA256 של טקסט NFC בסדר Wikipedia→FineWeb2→HPLT→FineWeb-Edu→OpenWebMath. הקוד שומר על UTF8, רישיות והזחה, עם סינון לפי נתיב, שפה, אורך, תווית רישיון, תוכן שנוצר ודפוסים מסוימים של סודות.',
'split':'קבוצות קשירות של תוכן זהה, טקסט רשת שקול ב-NFC ומאגרי קוד נשארות באותה חלוקה. validation הציבורי תואם ל-eval המקורי. אין בכך הוכחה להסרת כפילויות דומות בכל הקורפוס או להיעדר כל זיהום של הערכה.',
'tokenizer':'BPE גולמי, אוצר מילים 65,536; BOS=0, EOS=1, PAD=2 וללא UNK נפרד. Unicode המקורי נשמר. קבצים בינריים בפורמט little-endian uint16 עם EOS אחד שמתווסף לכל מסמך. טוקנייזר אחר מחייב טוקניזציה מחדש של הטקסט. תאימות AutoTokenizer ודוגמאות הורדה/אימון מלאות עדיין בבדיקה.',
'license':'קוד מקורי של הפרויקט: MIT; תיעוד הסברי מקורי: CC BY 4.0. אלה אינם מחליפים את תנאי המקורות. זכויות מסד נתונים/אריזה נפרדות מזכויות התוכן הבודד. ייחוסים, חריגים והודעות קוד עדיין בהשלמה; אין טענה שכל הטקסטים מותרים להפצה מסחרית בלתי מוגבלת.',
'verify':'כל 128,856,258 שורות החלוקה יוצאו ממסד נתונים קנוני לקריאה בלבד ונבדקו לוגית לאחר קריאה חוזרת. SHA256, מספרי המסמכים וגבולות EOS של 1,452 קבצים נבדקו למיקום מסמכים בתוך 122,402,393,813 טוקנים. שוחזרו כתובות מקור אפשריות עבור 455,397 מסמכי ויקיפדיה; בשבעה מקרים מרובי מקורות נשמרו כל הייחוסים ואומתה השורה המקורית שנשארה. אין זה אישור זכויות פרטני או ביקורת גלובלית חדשה של דמיון.',
'usage':'הקוד והתיעוד הזמינים ניתנים להורדה באמצעות הדוגמה בכרטיס הראשי. התצוגה המוקדמת כוללת 49 קבצים שנקבעו בחוזה אימון 3B בפועל ותלויות נוספות. שחזור מלא של האימון בסביבה חדשה טרם אומת. לאחר פרסום הנתונים יהיה אפשר להתחיל מהטקסט או להשתמש בטוקנים ובמתכון הנתונים.',
'limits':'ייתכנו הטיות רשת, שפה מזיקה, טעויות עובדתיות, מידע אישי או סודות שנותרו. לא הוכחו היעדר זיהום במבחנים רשמיים, היעדר גלובלי של כפילויות דומות או איכות מודל מסוימת. יש להפריד אימות מאימון ולהבחין בין כמות הטוקנים לבין משקלי הדגימה.',
'contact':'מפרסם: pathcosmos. שאלות ובקשות הסרה פרטיות: miracle.yang@pathcosmos.com. יש לצרף URL, doc_id או מאגר/נתיב קובץ, ולא לפרסם מידע רגיש בפניות ציבוריות. יש לצטט גם את מקורות הנתונים ששימשו בפועל.'},
'ru':{'name':'Русский','head':['Обзор','Статус публикации','Состав и объём','Обработка и разбиение','Токенизатор и формат','Источники и лицензии','Объём проверок','Использование','Ограничения','Контакты и цитирование'],
'overview':'Корпус для предобучения языковых моделей с акцентом на корейский язык: корейские энциклопедические и веб-тексты, английские образовательные веб-тексты, математика и код. pathcosmos выполнил дополнительную фильтрацию, точную дедупликацию, групповое разбиение и токенизацию. Авторы исходных данных указаны ниже. Веса модели и данные инструкционного дообучения не включены.',
'status':'Сейчас опубликованы предварительная версия кода, список источников с фиксированными ревизиями, условия лицензирования и контакты. Полная загрузка текстов, токенов и разбиений готовится. Числа относятся к локально подготовленным и проверенным материалам, а не ко всему уже размещённому корпусу.',
'language_note':'Языки перевода документации не означают соответствующего языкового покрытия обучающих данных. Основные естественные языки — корейский и английский; веб, математика и код могут содержать другие языки.',
'process':'Веб-обработка включает очистку пробелов, эвристики длины/повторов/языка, маскирование некоторых шаблонов персональных данных и точную дедупликацию внутри шардов. Между источниками используется SHA256 текста в NFC в порядке Wikipedia→FineWeb2→HPLT→FineWeb-Edu→OpenWebMath. Код сохраняет исходные UTF8, регистр и отступы; фильтры учитывают путь, язык, длину, метку лицензии, сгенерированный контент и некоторые шаблоны секретов.',
'split':'Связные группы идентичного содержания, NFC-эквивалентных веб-текстов и репозиториев кода остаются в одной части. Публичная validation соответствует исходной eval. Это не доказывает глобального удаления близких дублей или отсутствия всех видов загрязнения оценки.',
'tokenizer':'Raw BPE, словарь 65,536; BOS=0, EOS=1, PAD=2, отдельного UNK нет. Исходный Unicode сохраняется. Бинарный формат — little-endian uint16, один EOS добавляется в конец документа. Другой токенизатор требует повторной токенизации текста. Совместимость AutoTokenizer и полные примеры загрузки/обучения ещё проверяются.',
'license':'Оригинальный код проекта — MIT, оригинальная пояснительная документация — CC BY 4.0. Эти лицензии не заменяют условия источников. Права на базу/упаковку отличаются от прав на отдельный контент. Атрибуция, исключения и уведомления для кода ещё уточняются; не заявляется неограниченное коммерческое распространение всех исходных текстов.',
'verify':'Все 128,856,258 строк разбиения экспортированы из канонической БД только для чтения и логически сравнены после повторного чтения. Проверены SHA256, числа документов и EOS-границы 1,452 файлов, определены позиции документов для 122,402,393,813 токенов. Восстановлены URL-кандидаты для 455,397 документов Википедии; семь случаев нескольких источников сохраняют все атрибуции, исходная оставленная строка проверена. Это не индивидуальное подтверждение прав и не новый глобальный аудит близких дублей.',
'usage':'Доступные код и документацию можно получить по примеру на основной карточке. Предварительный код включает 49 файлов реального контракта обучения 3B и дополнительные зависимости. Полное воспроизведение обучения в новой независимой среде ещё не проверено. После публикации данных можно будет начать с текста или использовать готовые токены и рецепт.',
'limits':'Могут оставаться веб-предвзятость, вредные высказывания, фактические ошибки, персональные данные и секреты. Не доказаны отсутствие загрязнения официальных бенчмарков, глобальное отсутствие близких дублей и определённое качество модели. Разделяйте обучение и валидацию, различайте объём токенов и веса выборки.',
'contact':'Издатель: pathcosmos. Вопросы и частные запросы на удаление: miracle.yang@pathcosmos.com. Укажите URL, doc_id или репозиторий/путь файла; не публикуйте чувствительные сведения в открытых обсуждениях. Цитируйте также реально использованные исходные наборы данных.'}
}


def build():
    base=root/'runs/dataset_public_release_v1';out=base/'public_index_v4'
    shutil.copytree(base/'public_index_v2',out)
    cards=out/'cards';cards.mkdir()
    inventory=json.loads((base/'inventory_snapshot.json').read_text())
    source=json.loads((root/'docs/data_provenance_v1/source_registry.json').read_text())['sources']
    counts=json.loads((root/'runs/f6_tokens_v1/counts.json').read_text())['source_counts']
    weights=json.loads((root/'data/tokens/final_v1/mix.json').read_text())['sources']
    def nav(prefix=''):
        return ' · '.join(f"[{v['name']}]({prefix}README.{k}.md)" for k,v in LANGS.items())
    stats='''| Metric | Locally prepared value |
|---|---:|
| Documents | 128,856,258 |
| Train documents | 127,565,418 |
| Validation documents | 1,290,840 |
| Train tokens | 121,174,609,758 |
| Validation tokens | 1,227,784,055 |
| All tokens | 122,402,393,813 |
| Text Parquet bytes | 247,751,650,872 |
| Token binary bytes | 244,804,787,626 |
| Portable split bytes | 9,131,189,705 |
| Document offset bytes | 845,171,422 |

GB = 10^9 bytes. Token counts use this project's tokenizer and include appended EOS.
'''
    composition='| Source | Train docs | Validation docs | All tokens | Training sampling weight |\n|---|---:|---:|---:|---:|\n'
    sources='| Source / original publisher | Pinned revision | Selected subset |\n|---|---|---|\n'
    for s in source:
        key=s['source'];c=counts[key]
        composition+=f"| {key} | {c['train_docs']:,} | {c['holdout_docs']:,} | {c['tokens']+c['holdout_tokens']:,} | {weights[key]['weight']*100:.6f}% |\n"
        sources+=f"| [{s['repository']}]({s['pinned_card_url']}) | `{s['pinned_revision']}` | {s['selected_subset']} |\n"
    scopes='''| Material | Upstream condition / scope |
|---|---|
| FineWeb2 / FineWeb-Edu / OpenWebMath | [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/) database rights + [Common Crawl terms](https://commoncrawl.org/terms-of-use); individual content rights remain separate |
| HPLT2 | [CC0](https://creativecommons.org/publicdomain/zero/1.0/) packaging; no blanket ownership of individual texts |
| Wikipedia | Per-article attribution and [Wikimedia terms](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use); HF card's CC BY-SA 3.0/GFDL labels differ from the 2023 transition to CC BY-SA 4.0; exceptions require review |
| GitHub code | Source labels: MIT, Apache-2.0, BSD-3-Clause, CC0-1.0, BSD-2-Clause, Unlicense, ISC; original Git commits and complete LICENSE/NOTICE coverage are not yet established |

'''
    identifiers='''```text
tokenizer_sha256 = 050f9b43cf964f0c3b29ae443a1c2eae46cb2ee0022cc010bcfea3d26acf1706
split_contract_id = c06e2f6a42b6f8ba0d8a6ecd15f3b9d156ea7f44601efe86ad59f38ab0aedfc4
```
'''
    for code,v in LANGS.items():
        h=v['head'];parts=[f"# Korean Pretraining Corpus — {v['name']}",nav(),f"## {h[0]}",v['overview'],v['language_note'],
            f"## {h[1]}",v['status'],f"## {h[2]}",stats,composition,
            f"## {h[3]}",v['process'],v['split'],f"## {h[4]}",v['tokenizer'],identifiers,
            f"## {h[5]}",v['license'],sources,scopes,'[SOURCES.md](../SOURCES.md) · [License scope](../LICENSE-SCOPE.md)',
            f"## {h[6]}",v['verify'],f"## {h[7]}",v['usage'],'[Download example / 다운로드 예제](../README.md#download-available-material)',
            f"## {h[8]}",v['limits'],f"## {h[9]}",v['contact'],
            '[Citation / 인용](../README.md#citation)',
            'Original project prose: CC BY 4.0 · pathcosmos · https://creativecommons.org/licenses/by/4.0/']
        (cards/f'README.{code}.md').write_text('\n\n'.join(parts)+'\n')
    main='''---
pretty_name: Korean Pretraining Corpus
language:
- ko
- en
license: other
license_name: source-specific-licenses-see-scope
license_link: https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus/blob/main/LICENSE-SCOPE.md
task_categories:
- text-generation
tags:
- pretraining
- korean
- code
- mathematics
- reproducibility
---

# Korean Pretraining Corpus

**한국어 중심 LLM 사전학습 코퍼스 · A Korean-centric pretraining corpus**

'''+nav('cards/')+'''

**공개 준비 중 / Publication in preparation.** 현재 코드·출처·문서를 공개했습니다.
전체 텍스트·토큰·분리 자료는 로컬 검증을 마쳤거나 준비 중이며, 아직 업로드되지 않았습니다.
The code/source/documentation preview is available; complete corpus payloads are not yet hosted.

## Overview / 개요

'''+LANGS['ko']['overview']+'\n\n'+LANGS['en']['overview']+'''

문서 번역 13종은 학습 데이터의 언어 지원 목록이 아닙니다. 실제 주된 자연어는 한국어·영어입니다.
Documentation translations do not imply equivalent training-language coverage.

## Availability / 제공 상태

| Artifact | Public now | Preparation status |
|---|---|---|
| Code/source/license/contact preview | Yes | 49 actual training-contract files plus static dependencies; full clean-room training reproduction pending |
| 13 translated dataset cards | Yes | Shared measured facts and source/license tables |
| Cleaned text: six sources | No | Local corpus exists; source attribution and public export ongoing |
| Train/eval token binaries | No | All 1,452 files rehashed; counts and document boundaries checked |
| Portable splits and offsets | No | All 128,856,258 assignments and offsets verified locally |
| Tokenizer distribution | No | Local raw BPE verified; public packaging/loader compatibility pending |
| Per-document provenance | No | All 455,397 Wikipedia documents mapped; remaining source work pending |

## Composition / 구성

'''+stats+'\n'+composition+'''

Training sampling weights sum to Korean 50%, English 30%, code 10%, mathematics 10%.
They are a project recipe, not raw document proportions or a universal optimum.
Stored tokens and sampled exposure differ; sampling uses replacement.

## Sources / 출처

'''+sources+'''

Upstream dataset authors and original content contributors retain their respective attribution.
pathcosmos is responsible for the additional curation and packaging, not authorship of the upstream texts.
The pinned Hugging Face revision is not an original web-page revision or an original Git commit.
Full source notes: [SOURCES.md](SOURCES.md), [sources.json](sources.json).

## Processing, splits and tokenization

'''+LANGS['en']['process']+'\n\n'+LANGS['en']['split']+'\n\n'+LANGS['en']['tokenizer']+'\n\n'+identifiers+'''

The original web/math final schema is `text,h`; code additionally retains
`repo_name,path,license,language,source_row,content_sha256`.
The legacy `h` is a normalized pre-mask xxhash fingerprint, not a source URL or cryptographic provenance ID.
Public provenance sidecars, stable document IDs, assignments, and token offsets are prepared separately.
The final public schema and pinned release manifest will be supplied with payload publication.

## Validation / 검증

'''+LANGS['en']['verify']+'''

Prepared portable assignments: 9,131,189,705 bytes; offsets: 845,171,422 bytes.
Existing canonical group-isolation evidence was reused; a new global leakage audit was not performed during export.
No model-quality improvement over upstream data is established by these infrastructure checks.

## Download available material

This example downloads only the currently public preview and documents, without a login.
It resolves and prints a commit that can be pinned for subsequent downloads.

```python
from huggingface_hub import HfApi, snapshot_download

repo_id = "pathcosmos/korean-pretrain-corpus"
revision = HfApi(token=False).repo_info(repo_id, repo_type="dataset").sha
path = snapshot_download(
    repo_id, repo_type="dataset", revision=revision, token=False,
    allow_patterns=[
        "README.md", "cards/*", "SOURCES.md", "sources.json",
        "LICENSE-*", "CONTACT.md", "CHECKSUMS.json",
        "PUBLICATION_STATUS.json", "code-preview.tar.gz",
    ],
)
print(revision, path)
```

Verify `CHECKSUMS.json` before unpacking the code archive. The preview is not
a one-command training package; it does not yet include the full text/token payloads.
After publication, reuse will support starting from text with a new tokenizer
or using the supplied tokens and recipe with preserved validation splits.

## License scope / 라이선스 범위

'''+LANGS['ko']['license']+'\n\n'+LANGS['en']['license']+'\n\n'+scopes+'''

[LICENSE-SCOPE.md](LICENSE-SCOPE.md) · [MIT code license](LICENSE-CODE) ·
[CC BY 4.0 documentation](https://creativecommons.org/licenses/by/4.0/)

## Limitations and intended use

'''+LANGS['en']['limits']+'''

Intended for language-model pretraining, tokenizer experiments and reproducible data-pipeline research.
This is not a curated factual reference, an instruction dataset, an official evaluation benchmark,
or a guarantee of production suitability. Do not train on the validation partition when reporting held-out results.

## Contact / 문의·삭제 요청

Publisher: **pathcosmos**. Email: **miracle.yang@pathcosmos.com**.
For private removal requests, include a URL, doc_id, or repository/file path.
Avoid posting sensitive personal data in public discussions.
Corrections must account for text, tokens, provenance and affected release manifests together.
Pinned historical versions may remain accessible; see [CONTACT.md](CONTACT.md).

## Citation

```bibtex
@misc{pathcosmos2026koreanpretraincorpus,
  author = {pathcosmos},
  title = {Korean Pretraining Corpus},
  year = {2026},
  url = {https://huggingface.co/datasets/pathcosmos/korean-pretrain-corpus},
  note = {Publication in preparation; cite the exact revision used}
}
```

Also cite and retain the notices of the upstream datasets actually used.
This project citation does not replace upstream attribution or license obligations.
'''
    # Main card: a complete Korean card first, a complete English card second.
    # Other languages are separate documents, linked from the top.
    root_nav='[한국어](#한국어) · [English](#english) · '+' · '.join(
        f"[{v['name']}](cards/README.{k}.md)" for k,v in LANGS.items() if k not in ('ko','en'))
    def main_section(code):
        body=(cards/f'README.{code}.md').read_text().split('\n\n',2)[2]
        body=body.replace('## ','### ').replace('../SOURCES.md','SOURCES.md').replace('../LICENSE-SCOPE.md','LICENSE-SCOPE.md')
        body=body.replace('../README.md#download-available-material','#download-available-material').replace('../README.md#citation','#citation')
        if code=='ko':
            table_terms={'Metric':'항목','Locally prepared value':'로컬 준비·검증 수량','Train documents':'학습 문서','Validation documents':'검증 문서',
                'All tokens':'전체 토큰','Train tokens':'학습 토큰','Validation tokens':'검증 토큰','Documents':'전체 문서','Text Parquet bytes':'정제 텍스트 크기(bytes)',
                'Token binary bytes':'토큰 바이너리 크기(bytes)','Portable split bytes':'공개용 분리 정보(bytes)','Document offset bytes':'문서별 토큰 위치(bytes)',
                'Training sampling weight':'학습 샘플링 비중','Train docs':'학습 문서','Validation docs':'검증 문서','Source / original publisher':'원본 출처·제작 주체',
                'Pinned revision':'고정 revision','Selected subset':'선택 범위','| Source |':'| 소스 |'}
            for a,b in table_terms.items():body=body.replace(a,b)
            body=body.replace("GB = 10^9 bytes. Token counts use this project's tokenizer and include appended EOS.",
                'GB는 10^9bytes입니다. 토큰 수는 이 프로젝트 토크나이저 기준이며 문서 끝에 추가한 EOS를 포함합니다.')
            ko_scope='''| 자료 | 원본 조건과 적용 범위 |
|---|---|
| FineWeb2 / FineWeb-Edu / OpenWebMath | [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/) DB 권리와 [Common Crawl 약관](https://commoncrawl.org/terms-of-use). 개별 본문 권리는 별도입니다. |
| HPLT2 | [CC0](https://creativecommons.org/publicdomain/zero/1.0/)는 packaging에 적용됩니다. 개별 원문의 권리를 일괄 허락하는 것은 아닙니다. |
| Wikipedia | 문서별 귀속과 [Wikimedia 약관](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use)을 따릅니다. HF 카드의 CC BY-SA 3.0/GFDL 표기와 2023년 CC BY-SA 4.0 전환의 차이 및 예외를 검토합니다. |
| GitHub code | 원본 라벨 MIT / Apache-2.0 / BSD-3-Clause / CC0-1.0 / BSD-2-Clause / Unlicense / ISC. 원 Git commit 및 LICENSE/NOTICE 전체 확보는 아직 완료되지 않았습니다. |

'''
            body=body.replace(scopes,ko_scope)
            body=body.replace('Original project prose: CC BY 4.0 · pathcosmos ·','자체 설명 문서: CC BY 4.0 · pathcosmos ·')
        return body
    yaml_header='---'+main.split('---',2)[1]+'---\n\n'
    example=main[main.index('## Download available material'):main.index('## License scope / 라이선스 범위')]
    citation=main[main.index('## Citation'):]
    main=yaml_header+'# Korean Pretraining Corpus\n\n'+root_nav+'\n\n## 한국어\n\n'+main_section('ko')+'\n## English\n\n'+main_section('en')+'\n'+example+'\n'+citation
    assert main.index('## 한국어')<main.index('## English')<main.index('## Download available material')
    (out/'README.md').write_text(main)
    state=json.loads((out/'PUBLICATION_STATUS.json').read_text());state['documentation_languages']={k:v['name'] for k,v in LANGS.items()};state['training_natural_languages']=['ko','en'];state['translations_do_not_imply_training_language_coverage']=True
    (out/'PUBLICATION_STATUS.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
    entries=[{'path':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(out.rglob('*')) if p.is_file() and p.name!='CHECKSUMS.json']
    (out/'CHECKSUMS.json').write_text(json.dumps({'files':entries},indent=2)+'\n')
    report={'translations':len(LANGS),'language_codes':list(LANGS),'files':len(entries)+1,'bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file()),'shared_stats':True,'full_corpus_public':False}
    (base/'multilingual_cards_build.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':build()
