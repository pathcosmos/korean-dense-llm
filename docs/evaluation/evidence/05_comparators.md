# 비교군 사실 수집 (한국어 LLM 사전학습 데이터 처리 코드 평가용)

수집일 2026-11-09. 코드 저장소는 수정하지 않음. 모든 항목은 아래 출처에서 직접 읽은 내용이며, 읽지 못한 것은 "불확실" 또는 "접근 불가"로 표기함.
조회 시점의 main HEAD (short SHA): datatrove 384d81c011 (2026-11-04), Curator db56fbae11 (2026-11-06), data-juicer(modelscope) 65383e6f8f (2026-11-06), dolma 669f534823 (2025-09-18), text-dedup 7538f3ec6a (2026-03-09).
fineweb-2 저장소는 raw 파일만 읽었음(clone 안 함).

접근 방법 메모: gh api는 "GitHub access ... not enabled" 로 막힘. raw.githubusercontent.com 과 `git clone --filter=blob:none --no-checkout`(트리 목록 조회용)은 동작. docs.nvidia.com 은 DNS 해석 실패(세션 허용 도메인 밖). arxiv.org 은 연결 실패(HTTP 000).

---

## 1. 계층별 기능 매트릭스

범례: O = 소스/문서에서 확인된 기본 제공, 부분 = 일부만 / 조건부, X = 확인한 범위에서 없음, ? = 불확실(확인 못함)

| 계층 | datatrove (+fineweb-2 레시피) | NeMo Curator | Data-Juicer | Dolma | text-dedup |
|---|---|---|---|---|---|
| 정제(Heuristic filter) | O: Gopher quality/repetition, FineWeb quality, C4 quality/badwords/paragraph, URL, Regex, Lambda, FastText classifier, UnigramLogProb | O: heuristic 17종(WordCount, NonAlphaNumeric, Symbols, Urls, Bullets, Ellipsis, Punctuation 등) + 반복·코드·histogram·token_count 필터. 영어 중심(`CommonEnglishWordsFilter`, `lang="en"` 기본) | O: filter 연산자 다수(alphanumeric, char/word repetition, special chars, stopwords, flagged words, perplexity, text length, token num 등) | 부분: taggers(gopher, c4, repetitions, length, punctuation, quality, jigsaw, url 등). 필터링은 tagger+mixer 2단계 | X |
| 언어 식별(LID) | O: FT176 / GlotLID (`kor_Hang` 라벨 포함, fineweb-2는 GlotLID) | 부분: `FastTextLangId`(모델 경로 사용자 제공, 기본 min score 0.3) | O: `language_id_score_filter`(fasttext, 예시 min_score 0.8, lang=en) | O: cld2, langdetect, fasttext lid.176 taggers | X |
| 한국어 전용 처리 | O: `kor_Hang.yml` 임계값/불용어 파일, Kiwi 토크나이저 할당 | X(확인 범위): 한국어는 HTML 추출기의 non-spaced 언어 목록, 위키 접두어, 번역 문장분할(spaCy `xx_sent_ud_sm`), 오디오 쪽에서만 grep됨 | ?: 한국어 전용 operator 파일명 없음. stopwords/flagged words 자산에 ko 포함 여부 확인 못함 | X(확인 범위): 한국어 전용 설정 없음 | X: 공백/`\W` 분리만 |
| 중복제거: exact | O: ExactDedup(문서), SentenceDedup(문장 n=3), ExactSubstr(Google deduplicate-text-datasets 래핑) | O: exact dedup 워크플로우(GPU/Ray) | O: `document_deduplicator`, `document_line_deduplicator` (+Ray 버전) | O: Bloom filter 문서(key 기준)/문단 단위 | O: Bloom filter, Suffix array (substring) |
| 중복제거: near-dup | O: MinHash(4단계 파이프라인) | O: fuzzy dedup (MinHash+LSH, GPU) | O: MinHash, SimHash (+Ray BTS MinHash, C++ 변형) | X (MinHash 없음; 문단 n-gram Bloom만) | O: MinHash, SimHash |
| 중복제거: semantic | X | O: embedding+k-means (기본 `google/embeddinggemma-300m`) | ?: `text_embd_similarity_filter` 존재하나 dedup 용도 불확실 | X | X |
| PII | 부분: `PIIFormatter` = 이메일 + 공인 IPv4 치환만 | 부분: 코어 stage 없음. tutorial로 GLiNER-PII(영어 모델) 레다크션 | 부분: `clean_email`, `clean_ip`, `pii_redaction_mapper`(경로/이메일/시크릿/중국 휴대폰·주민증/국제 전화/IP 등), `pii_llm_suspect_mapper` | 부분: Presidio / regex tagger(이메일, 미국식 전화) | X |
| 한국어 PII(주민번호, 휴대폰) | X | X (GLiNER-PII 모델 카드 language: en) | X (중국식 정규식만 확인) | X (미국식 전화 정규식) | X |
| 인코딩/정규화 | O: FTFYFormatter, SymbolLinesFormatter | 부분: unicode_reformatter, newline/quotation/url remover 등 modifiers | O: `fix_unicode_mapper`, `punctuation_normalization_mapper` 등 | ? | 부분: `ftfy_utils.py` 존재(사용 방식 미확인) |
| 토큰화(학습용 직렬화) | O: DocumentTokenizer(HF tokenizer, eos 지정, 셔플/병합), Megatron 변형 | O: Megatron tokenizer writer(HF AutoTokenizer) | 부분: token_num_filter(HF tokenizer) 만 확인. 학습용 직렬화 ? | O: `dolma tokens`(HF tokenizers, bos/eos, memmap 출력) | X |
| 출처관리(provenance) | 부분: Document(`id`,`metadata`), `minhash_cluster_size`, exclusion_writer로 제거 문서 보존 | ?: Task/메타데이터 모델 있으나 출처 스키마 미확인 | ?: 미확인 | O: 문서 스키마 `id`/`source`/`added`/`created`/`metadata` 필수·선택 필드, attributes 파일 분리 | 부분: 클러스터 컬럼/index 컬럼 저장 옵션 |
| 라이선스 | Apache-2.0 | Apache-2.0 | Apache-2.0 | Apache-2.0 | Apache-2.0 |

---

## 2. 도구별 사실과 근거

### 2.1 huggingface/datatrove 및 huggingface/fineweb-2

**파이프라인 순서** (fineweb-2-pipeline.py docstring): 비영어 데이터 출발 -> GlotLID -> 언어별 language_score 필터 -> 언어별 dedup -> 언어별 필터링 -> 마무리(PII, ftfy).
- https://raw.githubusercontent.com/huggingface/fineweb-2/main/fineweb-2-pipeline.py
- https://raw.githubusercontent.com/huggingface/fineweb-2/main/README.md

**한국어(`kor_Hang`) 설정 존재: 있음.** `configs/kor_Hang.yml`:
- language_score 0.842, dup_line_frac 0.374, line_punct_thr 0.062, new_line_ratio 0.196
- max_avg_word_length 6, min_avg_word_length 1, max_non_alpha_words_ratio 0.791
- top_n_grams: (2,0.282) (3,0.228) (4,0.185); dup_n_grams: (5,0.201) (6,0.184) (7,0.168) (8,0.154) (9,0.14) (10,0.127)
- stopwords 21개: 의 다 이 는 에 을 한 년 하 은 를 했 고 에서 가 있 으로 인 월 로 되 (yml 유니코드 이스케이프를 디코딩한 값)
- https://raw.githubusercontent.com/huggingface/fineweb-2/main/configs/kor_Hang.yml

**필터 구성 (fineweb-2 스크립트 기준)**: GopherRepetitionFilter(문단 관련 3종 0으로 비활성, dup_line_frac/top_n_grams/dup_n_grams는 언어 yml), FineWebQualityFilter(`short_line_thr=999`로 비활성, `char_duplicates_ratio=0.1`(영어 FineWeb은 0.01), line_punct_thr/new_line_ratio는 yml), GopherQualityFilter(`min_stop_words=2` 고정, avg word length/non-alpha/stopwords는 yml). C4 필터 미사용 (README: 성능 저하로 제외). 모든 필터에 `language=lang_script` 전달.
- 같은 스크립트, README

**datatrove 필터 기본값** (라이브러리 기본, fineweb-2에서 덮어쓰기 전):
- GopherQualityFilter: min_doc_words 50, max_doc_words 100000, min_avg_word_length 3, max_avg_word_length 10, max_symbol_word_ratio 0.1, max_bullet_lines_ratio 0.9, max_ellipsis_lines_ratio 0.3, max_non_alpha_words_ratio 0.8, min_stop_words 2 -- https://raw.githubusercontent.com/huggingface/datatrove/main/src/datatrove/pipeline/filters/gopher_quality_filter.py
- GopherRepetitionFilter: dup_line_frac 0.3, dup_para_frac 0.3, dup_line_char_frac 0.2, dup_para_char_frac 0.2, top_n_grams ((2,.2),(3,.18),(4,.16)), dup_n_grams ((5,.15),(6,.14),(7,.13),(8,.12),(9,.11),(10,.10)) -- .../filters/gopher_repetition_filter.py
- FineWebQualityFilter: line_punct_thr 0.12, short_line_thr 0.67, short_line_length 30, char_duplicates_ratio 0.01, new_line_ratio 0.3 -- .../filters/fineweb_quality_filter.py
- C4QualityFilter: min_num_sentences 5, min_words_per_line 3, max_word_length 1000, 기타 bool 플래그 -- .../filters/c4_filters.py
- LanguageFilter: 기본 backend ft176, language_threshold 0.65; backend="glotlid" 선택 가능 -- .../filters/language_filter.py
- 필터 클래스 목록: C4BadWordsFilter, C4ParagraphFilter, C4QualityFilter, FastTextClassifierFilter, FineWebQualityFilter, GopherQualityFilter, GopherRepetitionFilter, LambdaFilter, LanguageFilter, RegexFilter, SamplerFilter, UnigramLogProbFilter, URLFilter -- https://raw.githubusercontent.com/huggingface/datatrove/main/src/datatrove/pipeline/filters/__init__.py

**한국어 토크나이저**: `tokenizer_assignment.csv`에 `kor,ko,Hang,Korean,Koreanic,KiwiTokenizer` (proxy=FALSE). `KiwiTokenizer`는 kiwipiepy의 `Kiwi(model_type="sbg")`로 word_tokenize/sent_tokenize 수행(20자 이상 숫자열 제거 전처리).
- https://raw.githubusercontent.com/huggingface/datatrove/main/src/datatrove/assets/tokenizer_assignment.csv
- https://raw.githubusercontent.com/huggingface/datatrove/main/src/datatrove/utils/word_tokenizers.py
- (추론, 불확실 표기) fineweb-2 스크립트 주석이 `language=` 인자가 단어 토크나이저 선택에 쓰인다고 명시하므로, 한국어에서 필터/MinHash의 "word"는 Kiwi 형태소 단위일 가능성이 높음. 이 연결은 코드 실행으로 검증하지 않음.

**MinHash 파라미터**: `MinhashConfig` 기본 n_grams=5, num_buckets=14, hashes_per_bucket=8 (총 112 해시), seed=1. fineweb-2는 `HashConfig(hash_fc="xxhash", precision=64)`, num_buckets=14, hashes_per_bucket=8, n_grams=5 (word n-gram, 토크나이저는 `language` 인자). 4단계: Signature -> Buckets -> Cluster -> Filter. 클러스터 크기를 `minhash_cluster_size`로 보존. 유사도 임계값(Jaccard)을 직접 지정하는 인자는 없고 밴드/행 구조로 결정됨(14x8 에서의 이론적 임계값은 이 문서에서 계산하지 않음).
- https://raw.githubusercontent.com/huggingface/datatrove/main/src/datatrove/pipeline/dedup/minhash.py
- fineweb-2-pipeline.py

**중복제거 범위**: FineWeb(영어)는 CC 스냅샷별, FineWeb 2는 "언어별 전역(globally per language)" 중복제거. 중복 클러스터에서 1개만 남기고 클러스터 크기 업샘플링용 가중치 제안(1->1, 2~3->N, 4->3, 5~99->5, 100~999->8, 1000+->1). -- https://raw.githubusercontent.com/huggingface/fineweb-2/main/README.md
- 그 외 제공: ExactDedup, SentenceDedup(n_sentences 3, min_doc_words 50, min_num_sentences 3, split_sentences 기본 True), ExactSubstr(외부 Google 저장소 필요), Bloom filter -- dedup/__init__.py, sentence_dedup.py, README 예제 목록

**PII**: `PIIFormatter`는 이메일과 IPv4(기본 공인 IP만)만 치환. 전화번호/주민번호/계좌 등 없음. fineweb-2 README "PII Removal: Kept unchanged, emails and ip addresses are anonymized." -- https://raw.githubusercontent.com/huggingface/datatrove/main/src/datatrove/pipeline/formatters/pii.py

**토큰화**: `DocumentTokenizer(tokenizer_name_or_path, eos_token, batch_size=10000, max_tokens_per_file, ...)`, 병합/셔플/Megatron 변형. -- https://raw.githubusercontent.com/huggingface/datatrove/main/src/datatrove/pipeline/tokens/tokenizer.py, tokens/__init__.py. fineweb-2 README는 `ablations/tokenization/tokenize_dataset.py` 언급.

**버전/라이선스**: datatrove pyproject version 0.10.1, Apache-2.0, 최신 원격 태그 v0.10.1 (git ls-remote). -- https://raw.githubusercontent.com/huggingface/datatrove/main/pyproject.toml. fineweb-2 저장소/데이터셋 라이선스는 확인 안 함(불확실).

### 2.2 NVIDIA-NeMo/Curator

- **라이선스**: Apache-2.0 (LICENSE 헤더) -- https://raw.githubusercontent.com/NVIDIA-NeMo/Curator/main/LICENSE
- **버전/API 변경**: main의 `package_info.py` = 1.5.0. 원격 최신 태그 v1.4.0 (커밋 2026-10-08). README Updates: 2026-04 "26.04: Cosmos-Xenna 0.2.0 upgrade, simplified `Resources` API, Ray runtime upgrade", 2026-02 "26.02: Ray-based pipeline architecture for all modalities". CHANGELOG.md는 1.0.0 항목이 최상단: Dask -> Ray 전환, `Task` 기반 처리 모델, `ProcessingStage`/`Pipeline` 재설계, rapidsmpf 셔플 백엔드, `RankingStrategy` 추가. 1.1~1.5 변경 이력은 CHANGELOG에 없고 docs.nvidia.com release notes는 접근 불가 -> **1.0 이후 API 변경은 불확실**.
  - https://raw.githubusercontent.com/NVIDIA-NeMo/Curator/main/README.md, CHANGELOG.md, nemo_curator/package_info.py
- **dedup 종류** (트리 목록 + 소스): exact (`stages/deduplication/exact/`), fuzzy (`fuzzy/`: minhash, lsh, buckets_to_edges, connected_components), semantic (`semantic/`: kmeans, pairwise, ranking). 텍스트용 `stages/text/deduplication/{removal.py, removal_workflow.py, semantic.py}`.
  - fuzzy 워크플로우 기본값: seed 42, **char_ngrams 24**, num_bands 20, minhashes_per_band 13 (총 260 해시), use_64_bit_hash False, bands_per_iteration 5. char_ngrams < 20 이면 "~5% 오탐" 경고. -- https://raw.githubusercontent.com/NVIDIA-NeMo/Curator/main/nemo_curator/stages/deduplication/fuzzy/workflow.py
  - semantic 기본: model `google/embeddinggemma-300m`, n_clusters 100, kmeans_max_iter 300 -- .../stages/text/deduplication/semantic.py
  - exact 워크플로우 세부 파라미터는 파일은 받았으나 정리하지 않음(불확실). substring dedup은 Curator 코어에서 확인 못함(README는 Nemotron-CC 레시피가 "exact/fuzzy/substring dedup"을 쓴다고 하나 해당 레시피는 별도 Nemotron 저장소).
- **필터**: heuristic string 필터 17종 (BoilerPlateString, Bullets, CommonEnglishWords, Ellipsis, LongWord, MeanWordLength, NonAlphaNumeric, Numbers, Parentheses, PornographicUrls, Punctuation, Substring, SymbolsToWords, Urls, WhiteSpace, WordCount, WordsWithoutAlphabets), 그 외 repetition, code, histogram, token_count, fasttext(`FastTextQualityFilter`, `FastTextLangId`), 분류기(aegis, content_type, domain, fineweb_edu, prompt_task_complexity, quality). 기본값 예: WordCount min 50/max 100000, NonAlphaNumeric 0.25, Urls/Bullets 0.9 등. 일부 필터는 `lang="en"` 기본 인자. -- .../stages/text/filters/heuristic/__init__.py, string.py, fasttext/fasttext_filters.py
- **언어 지원/한국어**: `FastTextLangId`는 fastText 모델 파일을 사용자가 제공 (기본 min_langid_score 0.3, `eng_Latn` 같은 lang-script 라벨 지원). `nemo_curator/`에서 "korean|ko|hangul" grep 결과: HTML 추출기 `NON_SPACED_LANGUAGES`(THAI, CHINESE, JAPANESE, KOREAN), jusText 주석, 위키 `ko` 접두어, 번역 문장분할 `ko -> xx_sent_ud_sm`, 오디오 언어맵. 한국어 전용 텍스트 필터/PII/토크나이저는 확인 못함. docs.nvidia.com 접근 불가라 문서상의 언어 지원 표는 **확인 불가**.
- **PII**: 코어 라이브러리에는 PII stage 없음(nemo_curator 하위 grep에서 PII 관련 경로 미발견). `tutorials/text/gliner-pii-redaction/`에 GLiNER-PII 기반 `GlinerPiiRedactor`(threshold 0.5, 55+ 카테고리). 모델 `nvidia/gliner-PII` 카드: language `en`, 라이선스 NVIDIA Open Model License(코드 Apache-2.0와 별개).
  - https://raw.githubusercontent.com/NVIDIA-NeMo/Curator/main/tutorials/text/gliner-pii-redaction/README.md, gliner_pii_redactor.py; https://huggingface.co/nvidia/gliner-PII/raw/main/README.md
- **토큰화/출력**: `stages/text/io/writer/megatron_tokenizer.py` (AutoTokenizer 기반). 리더: jsonl/parquet/lance.
- **기타**: 다운로드 stage (common_crawl, wikipedia, arxiv), HTML 추출기 (justext/resiliparse/trafilatura), modifiers (unicode_reformatter, c4, line_remover 등).

### 2.3 alibaba/data-juicer

- **저장소 위치 주의**: `github.com/alibaba/data-juicer` 는 세션 프록시에서 403, 따라서 clone/ls-tree 는 `modelscope/data-juicer`(현재 README 링크 체계: `datajuicer/data-juicer-hub`)로 수행. alibaba 경로가 modelscope 로 리디렉션되는 동일 저장소인지는 직접 확인하지 못함(불확실). raw 파일은 modelscope 경로에서 읽음.
- **버전/라이선스**: 최신 태그 v1.6.2, Apache-2.0 (LICENSE 헤더). README 배지: Operators 200+, Recipes 50+ (recipe는 data-juicer-hub에 분리). -- https://raw.githubusercontent.com/modelscope/data-juicer/main/README.md
- **operator 범주** (디렉토리 목록에서 확인): `ops/filter/`, `ops/mapper/`, `ops/deduplicator/`. 그 외 범주(selector, grouper, aggregator, formatter, pipeline 등)는 이번에 목록을 확인하지 않음(불확실). 텍스트 외 이미지/오디오/비디오/멀티모달 operator 다수.
  - dedup: document / document_line / document_minhash / document_simhash / image / video + Ray 변형(ray_basic, ray_document, ray_bts_minhash, ray_bts_minhash_cpp 등)
  - MinHash 기본값: tokenization "space" (space|punctuation|character|sentencepiece), window_size 5, lowercase True, num_permutations 256, jaccard_threshold 0.7, num_bands/num_rows_per_band는 None 이면 FP/FN 가중합(각 0.5) 최소화로 자동 산정, sentencepiece는 `tokenizer_model` 경로 필요. SimHash: window_size 6, num_blocks 6, hamming_distance 4. -- .../ops/deduplicator/document_minhash_deduplicator.py, data_juicer/config/config_all.yaml
- **한국어 지원**: 한국어 전용 operator는 파일명상 발견 못함(중국어 전용은 존재: `chinese_convert_mapper`, `remove_non_chinese_character_mapper`, `nlpcda_zh_mapper`). `stopwords_filter`, `flagged_words_filter` 등은 `lang` 인자로 자산 디렉토리의 언어별 단어 목록을 로드하는데, 문서 예시는 en/zh. 한국어(ko) 자산 존재 여부는 확인 못함(불확실). 한국어에는 `tokenization: character` 또는 사용자 sentencepiece 모델을 쓰는 방식이 가능하다는 것이 코드상의 옵션(품질 검증은 안 함). LID는 fastText 기반(`language_id_score_filter`, lang/min_score).
- **PII**: `pii_redaction_mapper`(에이전트/대화 텍스트 대상 설계, 경로/이메일/시크릿/ID/JWT/PEM/MAC/IP/URL, `phone_cn`=`1[3-9]\d{9}`, `phone_intl`=`\+\d{1,4}...`, 중국 18자리 신분증), `clean_email_mapper`, `clean_ip_mapper`, `pii_llm_suspect_mapper`. 한국 주민등록번호/휴대폰(010) 패턴은 확인 못함. -- .../ops/mapper/pii_redaction_mapper.py
- **레시피 구조**: YAML 설정 (`data_juicer/config/config_all.yaml` 전체 operator 파라미터 레퍼런스, `config_min.yaml`, `demos/process_simple/process.yaml` 등), 분석기(analyzer) 설정 별도. 실제 `process:` 리스트 키 구조는 config_all 발췌에서 `- op_name: {params}` 형태로 확인(파일 상단 구조 전체는 읽지 않음).

### 2.4 allenai/dolma

- **버전/라이선스**: pyproject version 1.2.1, 태그 v1.2.1, Apache-2.0. 마지막 커밋 2025-09-18 (조회 시점 기준 약 14개월 전).
- **구성**: Python taggers + Rust(`src/deduper.rs`, `bloom_filter.rs`, `mixer.rs`). CLI: tagger / dedupe / mixer / tokens / analyzer / warc.
- **dedup**: Bloom filter 기반, 문서 단위(JSON path key) 또는 문단 단위(개행 분할). 문단 `by_ngram`(ngram_length, stride, threshold 기본 1.0)은 Unicode word(UAX #29) 분할. `min_length`, `min_words`, `skip_empty`; bloom 크기는 bytes 또는 (estimated_doc_count + desired_false_positive_rate). 오탐 가능성 명시. 중복 제거 실제 삭제는 mixer 단계. MinHash/SimHash 없음. -- https://raw.githubusercontent.com/allenai/dolma/main/docs/deduplication.md
- **필터/태거**: gopher, c4, repetitions, length, punctuation, quality, jigsaw, url, licenses, code, sampling 등. 언어: `cld2_*`, `langdetect`, `ft_lang_id_*` (lid.176.bin). 이름이 `_en_`인 태거는 영어 판정용; 한국어 대상 설정은 확인 못함. -- docs/taggers.md, python/dolma/taggers/language.py
- **PII**: `PiiPresidioV1`, `PiiRegexV1/V2`, `FastPiiRegex`. regex는 이메일, 미국식 3-3-4 전화번호. -- python/dolma/taggers/pii.py
- **토큰화**: `dolma tokens` (HF tokenizers 래퍼, bos/eos id, `segment_before_tokenization` 옵션, memmap writer). -- python/dolma/tokenizer/tokenizer.py
- **출처관리**: 문서 스키마 `id`(필수, 버전 간 안정, blocklist 용), `source`(필수), `added`, `created`, `metadata`; attributes 파일 분리. -- docs/data-format.md
- 한국어 관련 기본 설정: 확인 못함.

### 2.5 ChenghaoMou/text-dedup

- **버전/라이선스**: pyproject 0.4.1, Python 3.12+, Apache-2.0. 원격 태그는 `2023.09.20`, `list` 뿐이라 PyPI 릴리스 현황은 불확실. 마지막 커밋 2026-03-09.
- **제공 방식**: 알고리즘별 모듈 + TOML 설정 (`python -m text_dedup.minhash` 식, README), HF datasets/parquet 입력, 출력에 cluster/index 컬럼 옵션. 알고리즘: MinHash, SimHash, Suffix Array (substring exact), Bloom Filter (exact).
  - MinHash 설정 클래스 기본: hash_bits 64, hash_func xxh3 (sha1 선택 가능), ngram_size 1 (클래스 기본)/예시 TOML은 5, num_perm 필수(예시 240), threshold 필수(예시 0.7), min_length 5, bands/rows 미지정 시 FP/FN 가중(0.5/0.5) 최적화, check_false_positive 옵션.
  - SimHash 기본: ngram_size 3, f 64, bit_diff 3, num_bucket 4, min_length 5, jaccard_threshold 0.5.
  - Suffix array: Google deduplicate-text-datasets를 submodule/외부 repo 경로로 호출, length_threshold 100, merge_strategy "longest".
  - Bloom: max_elements, error_rate (예시 1,000,000 / 0.001).
- **토큰화 방식**: `content.lower()` 후 `re.split(r"\W", ...)`, 이후 단어 n-gram (공백 join). 문자 n-gram/형태소 옵션 없음 -> 한국어는 어절(공백) 단위 shingle이 됨.
- PII/필터/토큰화: 없음.
- 출처: https://github.com/ChenghaoMou/text-dedup (raw: README.md, configs/*.toml, src/text_dedup/config/algorithms/{minhash,simhash,suffix_array,bloom}.py, src/text_dedup/minhash.py)

### 2.6 한국어 특화 논문

- **arXiv 2510.24139 (한국어 CommonCrawl 필터링)**: arxiv.org 연결 실패(HTTP 000), huggingface.co/papers/2510.24139 도 404. **접근 불가, 내용 확인 못함.** (제목/주장도 확인 못했으므로 요약하지 않음)
- **GECKO (arXiv 2405.15640)**: arxiv.org 접근 불가. huggingface.co/papers/2405.15640 페이지에서 초록만 확인: "한국어·영어·코드 이중언어 LLM, LLaMA 아키텍처, 'balanced, high-quality corpus'를 위한 데이터 파이프라인 구축 경험 공유, 모델 https://huggingface.co/kifai/GECKO-7B". 데이터 정제 세부(필터 임계값, dedup 방법, PII)는 본문 PDF 접근 불가로 **확인 못함.** (이 페이지 텍스트에서 dedup/PII/filter 키워드 없음)
  - https://huggingface.co/papers/2405.15640

---

## 3. 불확실 / 확인 불가 항목 목록

1. arXiv 2510.24139 전체(접근 불가), GECKO 본문의 정제 서술(접근 불가; 초록만).
2. NeMo Curator: 1.0 이후(1.1~1.5) 상세 API 변경 이력(CHANGELOG 미갱신, docs.nvidia.com 접근 불가). 문서상 언어 지원 표. exact dedup 세부 파라미터. substring dedup 존재 여부. 한국어 텍스트 필터/PII 지원 여부는 소스 grep 범위 내에서만 "없음".
3. Data-Juicer: alibaba/data-juicer 와 modelscope/data-juicer 의 동일성. operator 범주 전체 목록(filter/mapper/deduplicator 외). 한국어 stopwords/flagged words 자산 존재 여부. 학습용 토큰화/직렬화 기능. 출처관리 기능. recipe 파일 상단 구조 전체.
4. datatrove/fineweb-2: fineweb-2 저장소·데이터셋 라이선스. `kor_Hang`에서 단어 n-gram이 실제로 Kiwi 단위인지(코드 실행 미검증). MinHash 14x8 구성의 실효 Jaccard 임계값(미계산). `HashConfig` 기본값 세부.
5. Dolma: UAX #29 분할이 한국어에서 어절/음절 어느 단위로 동작하는지 미검증. 한국어 대상 언어 태거 설정 없음(확인 범위 내).
6. text-dedup: PyPI 릴리스/버전 정책, `ftfy_utils.py` 사용 경로.
7. 한국어 PII(주민등록번호, 휴대폰 010-xxxx-xxxx 등): 조사한 5개 도구의 확인된 소스에서 **기본 제공 없음**. 단, Curator 문서/Data-Juicer 자산 일부는 읽지 못해 "전체 부재"를 보증하지는 않음.
8. 각 도구의 성능(처리량) 비교, 한국어 품질 평가 결과: 이번 수집 범위 밖, 수집하지 않음.
