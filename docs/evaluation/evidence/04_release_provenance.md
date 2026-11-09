# 수집·출처·계약·배포 계층 평가 (read-only)

대상 repo: /home/user/korean-dense-llm (HEAD aef52f1). 저장소 파일은 수정하지 않았다. 임시 프로브는 evidence/scripts/probe.py, probe2.py에만 있다.
실행한 것: 관련 pytest 11개 파일 (34 passed, 2 skipped), 합성 입력 프로브 2개 (CPU, 네트워크 없음).
실행하지 않은 것: download.py, upload_public_corpus.py, supervise_corpus_upload.py, consolidate_public_corpus.py, finalize_public_corpus*.py (읽기만 함).
라인 번호는 파일 내 줄 번호이고 `파일:줄` 형식이다.

## 1. 점수표

| 축 | 점수 | 한 줄 근거 |
|---|---|---|
| A 기술품질 | 4 (코어) / 2 (운영 래퍼) | 감사·업로드·리더·오프셋은 실패경로와 resume 근거가 있다. finalize/consolidate는 테스트가 건너뛰어지고 문자열 치환에 의존한다. 종합은 3.5이고 보수적으로 4가 아니라 3~4 사이로 본다. |
| B 포트폴리오 | 3 | 한계를 정직하게 서술하는 수준은 상위권이다. 그러나 "권리 미확인"이라는 문서와 "전체 텍스트 공개"라는 스크립트가 충돌하고, 증거 파일이 번들되어 있지 않다. |
| C 재사용성 | 2 (전체) / 4 (프리미티브 3종) | PROJECT_ROOT, `runs/…` 경로, `pathcosmos` 계정, sched_setaffinity에 강하게 묶여 있다. identity/sha256_file은 복사해서 쓸 수 있다. |

### 컴포넌트별 세부

| 컴포넌트 | A | 비고 |
|---|---|---|
| common/data_contracts.py | 3 | identity와 sha256_file은 정확하다. atomic_json에 결함이 있다 (아래 3절). |
| data/download.py | 4 | revision 고정, 크기·LFS SHA256 검증, 선택 변경 거부. 단위 테스트는 select_entries만 다룬다. |
| data/source_audit.py | 4 | contract/receipt/fingerprint 체인, flock, 입력 불변성 재검증. 테스트 124줄에 6개 케이스. |
| data/release_offsets.py, public_corpus_reader.py | 4 | 합성 프로브에서 변조와 불일치를 모두 거부했다. |
| data/release_provenance.py, release_split.py | 3 | 설계는 좋다. 검증을 assert에 의존하고 입력 파일이 repo에 없다. |
| data/sample_snapshot.py | 3 | 예산과 시드가 있는 표본 추출. 전용 테스트를 찾지 못했다 (grep 기준). |
| scripts/upload_public_corpus.py | 4 | receipt가 resume 권한을 가지고, 원격 LFS OID로 검증하고, quota를 latch한다. 단위 테스트 4개는 네트워크 없는 부분만 다룬다. |
| scripts/consolidate_public_corpus.py, finalize_public_corpus(v1/v2), prepare_corpus_upload.py | 2 | 테스트 2개가 SKIPPED다 (아래 근거). v1과 v2는 거의 복제 코드다. |
| scripts/prepare_dataset_release.py | 3 | AST로 import를 따라가는 스냅샷은 독창적이다. 상태 서술이 stale이다. |
| 나머지 소형 스크립트 | 3 | prepare_source_manifest, check_data_readiness, preprocess_followup. |

## 2. 질문별 답

### 2.1 contract 프리미티브의 정확성과 범용성 (common/data_contracts.py)

프로브 결과는 `evidence/scripts/probe.py`로 얻었다.

**identity (data_contracts.py:37-38)**
- 정확하다. sort_keys, 압축 separators, ensure_ascii=False 조합이라 키 순서에 불변이다 (프로브 true).
- 한계 1: tuple과 list를 구분하지 않는다 (같은 해시). int 키와 str 키도 구분하지 않는다 (`{1:..}`와 `{'1':..}`가 같은 해시).
- 한계 2: `1`과 `1.0`은 다른 해시다. NaN은 `allow_nan` 기본값이라 조용히 해시되어 비표준 JSON `NaN`이 들어간다.
- 한계 3: Path와 혼합 타입 키는 TypeError로 실패한다. 이것은 오히려 안전한 쪽이다.
- 정규화 규칙이 문서화되어 있지 않다. 그래도 계약 ID 용도로는 충분하고, 타입 충돌 가능성은 낮다.

**sha256_file (29-34)**
- 정확하다. 16MiB 청크이고 빈 파일 해시도 맞다. 의존성은 hashlib뿐이다.

**atomic_json (41-45)**
- 임시 파일 이름이 항상 `path + '.tmp'`로 고정이다. 동일 경로에 동시 쓰기를 하면 한쪽이 FileNotFoundError를 낸다. 프로브에서 4 스레드로 재현했다 (`concurrent_errors: ["FileNotFoundError"]`).
- fsync를 하지 않는다. 같은 repo의 scripts/upload_public_corpus.py:23-26은 fsync하는 별도 `atomic_json`을 다시 정의한다. 즉 같은 이름의 함수가 두 개이고 내구성 수준이 다르다. consolidate와 finalize는 fsync가 있는 쪽을 쓰고, data/*와 prepare_source_manifest는 없는 쪽을 쓴다.
- `write_text`에 encoding을 지정하지 않으면서 `ensure_ascii=False`를 쓴다. 로케일이 UTF-8이 아닌 환경에서는 깨질 수 있다. 이 환경(UTF-8)에서는 재현하지 못했으므로 불확실로 둔다.
- 디렉터리 fsync가 없고, 이전 .tmp가 남아 있으면 덮어쓴다.

**resolve, relative, artifact_path (12-26)**
- `PROJECT_ROOT`에 모듈 수준으로 묶여 있다. 외부 경로는 절대 경로 문자열로 새는데 (`artifact_path('/etc/passwd')`가 그대로 반환됨), 주석상 "explicit external fixtures"를 위한 의도다.
- 이식성 관점에서는 전역 환경변수 의존이 큰 단점이다.

**source_files (48-80)**
- 도메인 로직이다 (manifest 상태, `.done` 마커, 파일명 유일성).
- `common.source_inventory`를 함수 안에서 지연 import한다 (77행). 이는 순환 import를 피하려는 회피책이고 추출 비용을 올린다.

**범용성 결론**
- 프리미티브 3종(identity, sha256_file, atomic_json)은 약 20줄이라 다른 프로젝트로 거의 비용 없이 복사할 수 있다. 다만 atomic_json은 먼저 고쳐야 한다 (고유 tmp 이름, fsync, encoding).
- `source_files`와 `resolve`는 이 프로젝트의 디렉터리 규약에 종속되어 있어 추출이 어렵다.
- 전용 단위 테스트 파일이 없고, 다른 테스트에서 간접 사용될 뿐이다 (tests/test_split_manifest.py 등 grep 기준).

### 2.2 라이선스 문서와 source_registry.json 사이의 일관성

**일치하는 것 (교차 검증함)**
- 6개 소스의 pinned revision이 문서 `DATA_SOURCES_AND_LICENSES.md:38-43`과 registry의 `pinned_revision`에서 일치한다.
- registry 합계가 문서 수치와 맞는다. 합계 파일 726, 문서 128,856,258, 바이트 247,751,650,872. 코드 라이선스 카운트 합계는 2,148,459로 `final_documents`와 같다. 문서 24행의 "247,751,650,872 bytes"도 일치한다.
- 소스별 라이선스 라벨이 동일하다: wiki cc-by-sa-3.0 + gfdl, FineWeb2와 FineWeb-Edu는 odc-by, HPLT는 cc0-1.0, OWM은 API null이지만 ODC-By 본문 조항, 코드는 other.

**정직성이 높은 부분**
- registry의 모든 소스가 `redistribution_review_status: not_cleared_for_public_text_release`, `underlying_rights_verified: false`, 최상단에 `upload_authorized: false`다.
- 문서는 revision이 원 웹 페이지나 GitHub commit이 아니라는 점을 명시한다 (46행).
- 해시 `h`가 URL이나 암호학적 해시가 아니라는 점, "모든 개인정보나 비밀키의 완전 제거는 보장하지 않는다"(137행)를 명시한다.
- 확인하지 못한 것(GFDL 1.3 원문 접근 실패, 57행)을 완료로 표시하지 않는다.
- 이것은 이 분야에서 드문 수준의 자기 제한이다.

**일관성 결함**
1. 문서와 실제 배포 동작이 충돌한다.
   - registry는 `upload_authorized: false`이고 "not_cleared_for_public_text_release"다. 문서는 "현재 전체 코퍼스를 하나의 자유 라이선스로 공개할 근거는 확보되지 않았다"고 쓴다 (7행).
   - 그러나 `scripts/consolidate_public_corpus.py:262-266`, `scripts/finalize_public_corpus.py:147`, `scripts/finalize_public_corpus_v2.py:178`은 `corpus_text_public=True`를 기록한다. consolidate는 `authorized_existing_repository: True`를 기록하고 (120행) 약 502.6GB를 공개 저장소로 올린다.
   - 문서 서두는 이 문서를 "Historical technical snapshot"이라 표시하지만, 이후 어떤 권리 검토로 상태가 바뀌었는지 가리키는 문서를 찾지 못했다. `PUBLIC_GUIDE.md`와 `PUBLIC_VALIDATION.md`에서는 라이선스·재배포 판단을 찾지 못했다 (grep 기준).
   - 따라서 법적으로 적법한지는 **불확실**이다. 다만 내부 문서끼리의 불일치는 사실이다. README는 "알려진 미확정 항목"을 source.json에 둔다고 안내하므로 완전히 숨기지는 않는다.
2. registry가 가리키는 증거 파일이 repo에 없다. `card_evidence[].path`의 6개 `docs/data_provenance_v1/evidence/*.md`가 모두 존재하지 않는다. `docs/data_provenance_v1/`에는 source_registry.json만 있다. 문서가 링크하는 upstream_evidence.json, file_inventory.json, local_audit.json, legal_evidence.json, collect_local.py, code_repository_labels.jsonl.gz도 마찬가지다. 문서 서두가 "Missing run artifacts are not bundled data"라고 밝히므로 정직하지만, 독립 검증은 불가능하다. 따라서 registry의 SHA256과 bytes는 **측정 불가**다.
3. 코드 라이선스 상태가 모순된다.
   - `LICENSE`(MIT)와 `NOTICE.md`가 존재한다.
   - `scripts/prepare_dataset_release.py:230`은 `original_code_license: 'pending_selection_and_third_party_review'`를 쓰고, 252-253행의 README 문구는 "no original-code open-source license is granted"다.
   - 이 스크립트가 만든 드래프트는 stale이며 현재 MIT와 모순된다.
4. 저장소 레이아웃 서술이 갈린다.
   - consolidate는 단일 저장소 `pathcosmos/korean-pretrain-corpus` v1.1.0이다.
   - `PUBLIC_GUIDE.md`는 데이터를 별도 `korean-pretrain-training-data` v1.0.1에서 받으라고 안내한다.
   - README.md:79-80은 두 저장소를 함께 링크한다.
   - 어느 쪽이 현재 정본인지 문서만으로 단정하기 어렵다. 원격 상태는 확인하지 않았다 (**불확실**).
5. 개인정보 처리의 한계는 문서에서 정직하게 서술된다 (정규식 기반 이메일·전화·주민번호 치환, 완전 제거 비보장). 다만 실제 재현율과 정밀도는 이 계층에서 **측정 불가**다.

**외부 사실에 대한 판단 보류 (불확실)**
- Wikimedia의 2023년 CC BY-SA 4.0 전환 시점과 효력 범위.
- HPLT CC0의 packaging 범위.
- ODC-By 1.0 조항 번호 (§2.4, §4.2~4.4).
- Common Crawl 약관의 갱신일 (2024-03-07).
- HF 카드의 정확한 문구.
- 위 내용은 문서가 인용한 URL을 이번에 열어 확인하지 않았으므로 맞는지 틀린지 판정하지 않는다. 문서의 내부 논리(카드 표기와 공식 안내가 다를 때 카드만 복사하지 말 것)는 타당해 보인다.

### 2.3 과설계 위험과 중복

**방어 가치가 높은 곳**
- `source_audit.py`: mtime·size 변경 재검증 (93-96), receipt와 fingerprint 체인 (101-106), 코드 해시 고정 (61-62, 143), flock (68).
- `upload_public_corpus.py`: receipt를 resume 권한으로 사용 (96-101), 응답 유실 시 원격 재검사 (131-135), quota latch (86), 원격 LFS SHA256 대조 (verify).
- `release_offsets.py:30-31`: EOS 개수와 문서 수가 다르면 거부한다.
- 이 코드들은 수백 GB 규모의 재개 가능한 작업에서 실제 가치가 있다.

**과설계 또는 중복 신호**
- `finalize_public_corpus.py`(184줄)와 `finalize_public_corpus_v2.py`(215줄)는 사실상 포크다.
  - diff 44줄, v1에만 있는 줄 2줄.
  - 둘 다 `FINAL_STATUS`와 `COMPAT`의 13개 언어 딕셔너리, 같은 `whoami`와 `contact`, 같은 `'documents':128856258`을 복제한다.
  - v1은 v2로 대체되었는데도 삭제되지 않았다.
  - `consolidate_public_corpus.py:12`는 v2의 `discovery_metadata`를 import하므로 무거운 의존성(yaml, huggingface_hub)을 끌어온다.
- finalize와 consolidate는 README와 카드를 `str.replace`로 편집한다 (예: consolidate:173, finalize_public_corpus.py:125-135). 원문 문구가 조금만 달라져도 조용히 무변경이 된다. 대응하는 검증이 없고, 테스트는 건너뛰어진다.
- 하드코딩된 숫자(`128856258`, `122402393813`, `726`, `502.6GB`)가 코드와 카드 문자열에 흩어져 있다. `prepare_dataset_release.py:46`은 `assert len(inventory)==726`이다.
- 다국어(13개 언어) 카드 문자열이 로직과 섞여 있다. 포트폴리오로서의 차별점이지만, 코드 유지보수 비용이다.
- `assert`를 입력 검증에 쓴다 (release_provenance.py:83-86 외, release_split.py:104, 146, release_offsets.py:28 외). `python -O`에서 모두 사라진다. 검증이 핵심인 계층에서는 약점이다. 반면 source_audit와 upload는 명시적 raise를 쓴다.
- 세미콜론으로 한 줄에 여러 문장을 이어 쓰는 압축 스타일이 읽기 어렵다.
- 결론: 코어(감사, 업로드, 리더, 오프셋)는 복잡도가 방어 가치로 정당화된다. 문자열 패치형 finalize 계층은 가치 대비 취약하고 중복이다.

### 2.4 하드코딩된 개인 경로와 환경 가정

- `/home/…` 같은 개인 절대 경로는 이 계층 파일에서 발견하지 못했다 (grep 기준). `PROJECT_ROOT` 환경변수 오버라이드가 있다 (paths.py:3).
- 계정과 식별자가 하드코딩되어 있다.
  - `pathcosmos`: consolidate:16, 117, 사이드 스크립트 다수.
  - 개인 이메일: `miracle.yang@pathcosmos.com`이 finalize_public_corpus.py:140, finalize_public_corpus_v2.py:171, prepare_corpus_upload.py:65, 154에 있다. 공개 카드의 연락처로 쓰는 의도된 값이지만 설정이 아니라 코드에 박혀 있다.
  - 반면 `prepare_dataset_release.py:86-87`은 "Private takedown contact is not yet configured"라고 쓴다. 상태 서술이 어긋난다.
- `os.sched_setaffinity`로 마지막 N개 CPU에 고정한다 (release_provenance.py:92, release_split.py:98, upload_public_corpus.py:79, consolidate:299 근처). Linux 전용이다.
- 고정 상대 경로를 가정한다: `runs/dataset_upload_v1`, `runs/dataset_public_release_v1`, `data/datasets/final_v2.split/manifest.json`, `docs/data_provenance_v1/file_inventory.json`. 클린 clone에서는 `runs/`가 `.gitkeep`뿐이고 file_inventory.json과 recovery_bundle.tar.gz가 없으므로 build/prepare 스크립트는 실행할 수 없다.
  - `PUBLIC_GUIDE.md`가 "원래 `runs/`가 없으면 준비 단계가 필요하다"고 안내한다.
- `/proc/self/status`의 VmRSS를 읽는다 (consolidate, supervise). Linux 전용이다.

### 2.5 합성 입력으로 직접 실행한 확인

| 확인 | 결과 |
|---|---|
| pytest 11개 관련 파일 | 34 passed, 2 skipped, 14 subtests passed (2.08초) |
| skip 사유 | tests/test_corpus_finalizer.py와 tests/test_corpus_consolidation.py가 "requires excluded production dataset release/active training manifests"로 건너뛰어진다. 이 두 파일은 `runs/…/upload_spec.json` 같은 원본 실행물이 필요하다. 즉 공개 clone에서는 finalize/consolidate 로직이 사실상 미검증이다. |
| iter_split 합성 프로브 | train과 validation 분리가 맞다. `split='all'`은 거부된다. 한 행의 content_sha256을 바꾸면 "Document body differs from canonical split"로 거부된다. 알 수 없는 split 라벨은 거부된다. |
| scan_boundaries 합성 프로브 | 정상 케이스는 맞다. EOS 개수가 문서 수와 다르면 거부된다. 한계: 빈 문서(EOS만 있는 문서)는 구분할 수 없다. `[1,EOS,EOS,EOS]`가 3문서로 통과한다. 문서가 "문서당 EOS 1개" 계약을 전제하므로 의도된 한계이고, 인증서의 문서 수를 신뢰하는 모델이다. |
| atomic_json 동시성 | 4 스레드에서 FileNotFoundError 재현 |
| identity | tuple과 list, int 키와 str 키 구분 안 함 (위 2.1) |

실행하지 않은 부분: source_audit.run 전체의 대규모 동작, upload/download/consolidate 네트워크 경로. 해당 점수는 테스트 파일과 코드 읽기에 근거한다.

## 3. 핵심 발견 5개

1. **[B, 높음] 권리 미확인 문서와 전체 텍스트 공개 동작의 충돌.**
   registry는 `upload_authorized:false`, `not_cleared_for_public_text_release`이고 문서는 공개 근거가 없다고 말한다. 반면 consolidate_public_corpus.py:262-266과 120은 `corpus_text_public=True`, `authorized_existing_repository=True`를 기록한다. 이후의 권리 판단을 담은 문서는 찾지 못했다. 적법성은 불확실이고 내부 일관성 결함은 확정이다.
2. **[B/A, 중간] 증거 파일 부재로 registry를 독립 검증할 수 없다.**
   registry가 참조하는 evidence/*.md 6개와 문서가 링크한 file_inventory.json, upstream_evidence.json 등이 repo에 없다. 그래서 prepare_dataset_release.py와 prepare_corpus_upload.py, release_provenance.py의 CLI는 클린 clone에서 실행할 수 없다. 숫자 합계는 내부적으로 일치하지만 해시는 측정 불가다.
3. **[A, 중간] common/data_contracts.py의 atomic_json 결함과 중복 정의.**
   고정 `.tmp` 이름 때문에 동시 쓰기 경쟁이 실제로 재현되고, fsync와 encoding 지정이 없다. fsync가 있는 다른 `atomic_json`이 upload_public_corpus.py에 따로 있다. identity는 tuple과 list, int 키와 str 키를 같게 취급한다. 반면 sha256_file은 정확하다.
4. **[A, 중간] 코어는 견고하고 운영 래퍼는 미검증이며 중복이다.**
   source_audit, upload_public_corpus, release_offsets, public_corpus_reader는 변조·불일치·재개 경로 근거가 있다. 반면 finalize_public_corpus.py(v1)와 v2는 사실상 포크이고, finalize와 consolidate는 `str.replace` 패치에 의존한다. 두 테스트는 공개 clone에서 SKIPPED다. 검증 로직 일부가 assert라 `-O`에서 무력화된다.
5. **[C, 중간] 이식성이 낮다.**
   `pathcosmos` 계정, 개인 이메일, `runs/…` 고정 경로, sched_setaffinity와 /proc 의존, 하드코딩된 문서·토큰 수가 코드에 박혀 있다. 추출 가치가 있는 것은 identity, sha256_file(수정한 atomic_json), source_audit의 receipt 체인 패턴, upload의 receipt 기반 재개 패턴이다. 나머지는 이 프로젝트 전용이다.

## 4. 측정 불가 또는 불확실 목록

- 원격 HF 저장소의 현재 상태, 태그 v1.0.0과 v1.1.0 존재 여부 (네트워크를 쓰지 않음).
- 외부 라이선스 조항의 정확성 (Wikimedia, HPLT, ODC-By, Common Crawl). 문서가 인용한 URL을 열지 않았다.
- registry의 evidence SHA256과 bytes (파일 부재).
- 비UTF-8 로케일에서 atomic_json의 실패 여부.
- PII 마스킹의 실제 재현율과 정밀도.
- sample_snapshot.py의 전용 테스트 존재 여부 (파일명 grep 기준으로 못 찾음).
- 학습 파이프라인과의 실제 결합 (이 평가 범위 밖).
