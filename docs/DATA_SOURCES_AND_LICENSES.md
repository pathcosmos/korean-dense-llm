> Historical technical snapshot (2026-11-08). Original paths and earlier status statements are retained. For the public release entrypoint and current verification scope, see [PUBLIC_GUIDE](PUBLIC_GUIDE.md) and [PUBLIC_VALIDATION](PUBLIC_VALIDATION.md). Missing run artifacts are not bundled data.

# 최종 정제 데이터 출처와 재배포 조건

검토 기준일: 2026-11-07 UTC. 대상은 `data/datasets/final_v1.ready.json`에 고정된 여섯 소스, 최종 726개 Parquet, 128,856,258문서다. 학습 입력의 내용 검증과 공개 배포에 필요한 권리 검토를 구분한다.

**현재 전체 코퍼스를 하나의 자유 라이선스로 공개할 근거는 확보되지 않았다.** 데이터셋별 배포 조건은 확인했지만, 웹 원문의 권리, 위키의 문서별 귀속, 코드의 원본 고지 보존은 추가 조치가 필요하다. 아래의 공개 보류는 불법이라고 확정한 판단이 아니라, 현재 산출물로 재배포 조건 충족을 입증하지 못한 상태를 뜻한다. 모델 가중치의 라이선스나 특정 국가의 학습 예외 적용은 이번 텍스트 재배포 검토의 결론으로 대신하지 않는다.

## 출처별 판단

| 최종 소스 | 확인한 데이터셋 조건 | 원문에 관한 핵심 제한 | 현재 공개판에 필요한 조치 |
|---|---|---|---|
| `wiki_ko` | HF 고정 카드 CC BY-SA 3.0 및 GFDL; Wikimedia는 2023년 CC BY-SA 4.0 전환 | 문서별 귀속·변경 표시·동일조건변경허락, 외부 반입물의 별도 고지 | 문서 URL·제목·이력 연결 복원, 4.0 기준 공개 구성 및 예외 확인 |
| `fineweb2_ko` | ODC-By 1.0, Common Crawl 이용약관 | 데이터베이스 권리와 개별 웹 문서 권리는 별도 | 원본 URL 복원, DB 귀속 고지, 원문 재배포 근거 검토 |
| `hplt_ko` | CC0 1.0은 packaging에 적용 | HPLT는 추출 원문의 소유권을 주장하지 않음 | 원본 URL·수집 식별자 복원, 원문 재배포 근거 검토 |
| `fineweb_edu` | ODC-By 1.0, Common Crawl 이용약관 | 교육 점수·필터링이 원문 사용 허락을 대신하지 않음 | 원본 URL 복원, DB 귀속 고지, 원문 재배포 근거 검토 |
| `owm` | ODC-By 1.0, Common Crawl 이용약관 | 원문 라이선스를 변경하지 않는다고 명시 | URL·수집일 복원, 사이트별 귀속·조건 검토 |
| `github_code_v1` | 카드 `other`; 각 저장소 라이선스 라벨 | 허용 라벨만으로 파일별 권리·필수 고지 보존을 입증하지 못함 | 원본 버전 및 LICENSE·NOTICE·저작권 고지 확보와 파일별 대조 |

이 표는 데이터셋 배포자의 카드와 라이선스 원문을 대조한 검토 결과다. 카드에 표시된 라이선스를 원문 전체에 일괄 적용하지 않는다. 공식 근거와 실제 파일 확인 범위는 아래에 명시한다.

## 고정 출처와 사용 범위

전체 압축 Parquet는 **247,751,650,872 bytes, 약230.74GiB**다. 행 수는 최종 보존 문서 수이며 원본 전체 규모가 아니다. 원본 제공자가 붙인 데이터셋 버전과 우리 정제 버전은 다르다.

| 소스와 고정 카드 | 선택 범위 | 최종 파일 | 최종 문서 | 압축 GiB |
|---|---|---:|---:|---:|
| [Wikimedia Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia/blob/b04c8d1ceb2f5cd4588862100d08de323dccfbaa/README.md) | `20231101.ko` 3개 파일 | 3 | 455,397 | 0.493 |
| [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/blob/af9c13333eb981300149d5ca60a8e9d659b276b9/README.md) | `data/kor_Hang/train/*` 25개 파일 | 25 | 58,664,414 | 93.071 |
| [HPLT2.0 cleaned](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned/blob/d1324a5283f762ee62c2a5c81de08fc6450ea540/README.md) | `kor_Hang/*` 464개 파일 | 464 | 27,434,232 | 58.137 |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md) | `sample/100BT/*` 정렬 앞 56개, 100BT 전체가 아님 | 56 | 34,704,836 | 62.705 |
| [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math/blob/fde8ef8de2300f5e778f56261843dab89f230815/README.md) | `data/*` 114개 파일 | 114 | 5,448,920 | 12.992 |
| [CodeParrot GitHub Code](https://huggingface.co/datasets/codeparrot/github-code/blob/b5661e6b17396364b2bcf8e68977b0d28e1ebd19/README.md) | 1,126개 샤드 중 균등 간격으로 선택한 64개 | 64 | 2,148,459 | 3.338 |

원본 식별값은 다음과 같다. 동일한 이름의 최신 데이터셋으로 대체하지 않는다.

```text
wiki_ko        wikimedia/wikipedia          b04c8d1ceb2f5cd4588862100d08de323dccfbaa
fineweb2_ko    HuggingFaceFW/fineweb-2       af9c13333eb981300149d5ca60a8e9d659b276b9
hplt_ko        HPLT/HPLT2.0_cleaned          d1324a5283f762ee62c2a5c81de08fc6450ea540
fineweb_edu    HuggingFaceFW/fineweb-edu     87f09149ef4734204d70ed1d046ddc9ca3f2b8f9
owm            open-web-math/open-web-math  fde8ef8de2300f5e778f56261843dab89f230815
github_code_v1 codeparrot/github-code       b5661e6b17396364b2bcf8e68977b0d28e1ebd19
```

검토 시 HF API에서 확인한 각 저장소의 현재 main SHA는 모두 위 고정 SHA와 같았다. HF 저장소 revision은 **원 웹 페이지의 revision이나 원 GitHub 저장소 commit이 아니다.**

## 기록과 실물 검증

- [소스별 구조화 기록](data_provenance_v1/source_registry.json): 선택 범위·고정 출처·라이선스 적용 범위·공개 전 미해결 항목을 소스별로 기록했다.
- [원본 카드 증거](data_provenance_v1/upstream_evidence.json): 고정 revision과 조회 당시 main의 README를 각각 보존하고 URL·조회 UTC·bytes·SHA256·API 라이선스 값을 기록했다. 저장소 최상위 LICENSE·NOTICE 파일은 조회 목록에 없었으며 README 내 조건을 확인했다.
- [샤드별 출처 목록](data_provenance_v1/file_inventory.json): 최종 726파일 각각을 원본 repo/revision/원격 파일/고정 다운로드 URL/원본 SHA256/최종 SHA256/행수/용량에 연결했다. 원본 SHA는 기존 검증된 복구 목록, 최종 SHA는 기존 전체 내용 감사에서 재사용했다.
- [이번 실물 검사](data_provenance_v1/local_audit.json): 전체 726파일의 footer 행수·크기·스키마와 읽기 전후 stat을 대조했다. 전체 본문을 새로 해시하거나 재감사한 결과는 아니다.
- 코드 **2,148,459행**의 `repo_name/path/license/source_row/content_sha256`를 새로 전수 읽었다. 필수 필드 공란0, 허용 목록 외 라벨0, 저장소 이름 기준530,331개, 같은 저장소의 복수 라벨0. SHA 문자열 형식 검사는 이번 수행, 본문과의 SHA 일치는 기존 내용 감사 범위다.
- [저장소별 코드 라벨 목록](data_provenance_v1/code_repository_labels.jsonl.gz): 530,331개 저장소 URL과 실제 라벨별 행수를 기록했다. GitHub URL의 현재 접근 가능성이나 원본 commit을 검증했다는 뜻은 아니다.
- [원본 스키마 확인](data_provenance_v1/raw_schema_evidence.json): 이전 복구 실험이 보존한 소스당1개 실제 원본 샤드의 컬럼을 확인했다. 대용량 원본을 새로 다운로드하지 않았다.
- [공식 약관과 라이선스 증거](data_provenance_v1/legal_evidence.json): 확보한 원문 사본·SHA·URL과 직접 HTTP 실패를 구분했다. Wikimedia 약관/덤프 지침은 HTTP403으로 파일 사본을 얻지 못했지만 웹 도구로 내용을 읽었다. GFDL 1.3 원문은 직접 HTTP403 및 웹 timeout으로 이번에 확인하지 못했으므로 그 경로의 상세 준수 검토는 완료로 표시하지 않는다.

출처 목록 생성기는 [collect_local.py](data_provenance_v1/collect_local.py)다. CPU affinity92..95, worker1, Arrow CPU/IO 각1, batch16,384행, RSS예산8GiB·시간300초로 실행했다. 최신 실행 약19.80초, peakRSS약0.332GiB. 입력 본문·학습 producer는 변경하지 않았다.

## 소스별 라이선스 검토

### Wikimedia Wikipedia

고정 HF 카드에는 CC BY-SA 3.0/GFDL 및 일부 텍스트는 CC 조건만 적용된다는 설명이 있다. 그러나 Wikimedia의 [2023년 공식 전환 안내](https://diff.wikimedia.org/2023/06/29/stepping-into-the-future-wikimedia-projects-transition-to-creative-commons-4-0-license/)와 [승인된 약관 기록](https://meta.wikimedia.org/wiki/Wikimedia_Foundation/Legal/2023_ToU_updates/Proposed_update)은 2023-06-07 발효와 4.0 전환을 확인한다. 우리 입력은 그 이후인 2023-11-01 덤프다. **HF 카드의 3.0 표기만 복사해 공개 조건을 결정해서는 안 된다.**

[현재 덤프 지침](https://dumps.wikimedia.org/legal.html)은 텍스트의 기본 조건을 CC BY-SA 4.0/GFDL로 설명하면서, 덤프 내부 고지와 이용약관이 우선하고 외부 반입물·인용 등 예외가 있음을 밝힌다. [Wikimedia 약관 7항](https://foundation.wikimedia.org/wiki/Policy:Terms_of_Use#7._Licensing_of_Content)은 원문 페이지 URL 등의 귀속 방법, 외부 반입물의 추가 귀속, 변경 표시와 라이선스 고지를 규정한다.

공개판 준비의 기본 경로는 문서별 URL과 저자 이력을 연결하고, 변경된 위키 텍스트에 CC BY-SA 4.0 조건을 적용하는 것이다. 이는 기본 정책에 따른 준비 방향이며 모든 문서의 예외까지 확인한 결과는 아니다. 3.0 자료의 adaptation에 대한 후속 같은 요소 라이선스 허용은 [CC BY-SA 3.0 §4(b)](https://creativecommons.org/licenses/by-sa/3.0/legalcode.en), 4.0의 귀속·변경·동일조건은 [CC BY-SA 4.0 §3](https://creativecommons.org/licenses/by-sa/4.0/legalcode.en)을 따른다. 원본의 추가 고지와 예외는 별도로 보존해야 한다.

우리 위키 파일은 `text,h`만 있으므로 원래의 `id,url,title`을 복원해야 한다. 단순히 “출처: Wikipedia”라고 쓰는 것만으로 문서별 귀속을 마쳤다고 판단하지 않는다. 혼합 코퍼스 전체나 학습 모델에 자동으로 동일조건이 적용된다고 단정하지 않으며, 자료의 구분과 실제 변형 관계를 기준으로 판단한다. GFDL을 대체 선택하면 모든 문서에 그 선택이 가능한지와 해당 배포 의무부터 별도로 확인해야 한다.

### FineWeb2 한국어

배포 주체는 Hugging Face의 FineWeb 팀이며, 우리는 `kor_Hang` 학습 부분을 추가 정제했다. [고정 카드의 Licensing Information](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/blob/af9c13333eb981300149d5ca60a8e9d659b276b9/README.md)은 ODC-By 1.0과 Common Crawl 이용약관을 지정한다. 카드의 이메일/IP 처리와 삭제 요청 절차는 원본 제공자의 처리 설명이며, 우리 배포본의 개인정보 검증을 대신하지 않는다.

[ODC-By 1.0 전문](https://opendatacommons.org/licenses/by/1-0/)의 서문·§2.4는 DB 권리와 개별 내용의 권리를 구분한다. §4.2~4.4에 따라 DB 전달 시 라이선스 사본/URI와 원 고지를 보존하고, 공개 산출물에는 원 DB의 출처와 라이선스 안내가 필요하다. 출처·버전·변경 내역·ODC-By 링크를 데이터 디렉터리와 카드에 함께 둔다. 이 고지만으로 모든 웹 문서의 재배포 권리가 생기지는 않는다.

[Common Crawl 약관 §2~3](https://commoncrawl.org/terms-of-use)은 원 사이트의 별도 조건과 제3자 권리, 개인정보 보호를 준수하도록 한다. 현재 확인되는 약관의 갱신일은 2024-03-07이다. 공개 접근 가능, robots.txt 준수, 이미 다른 HF 저장소에 있음은 원문 재배포 허락의 대체 증거가 아니다. 우리 최종 파일에서 빠진 `url/id/dump/date/file_path`를 다시 연결해야 한다.

### HPLT2.0 한국어

배포 주체는 HPLT 프로젝트다. [고정 카드의 Terms of Use and Takedown](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned/blob/d1324a5283f762ee62c2a5c81de08fc6450ea540/README.md)은 CC0가 텍스트의 **packaging**에 적용되고, 추출한 원문의 소유권은 갖지 않는다고 명시한다. 이용자의 관련 법률 준수 책임과 정당한 요청 시 다음 배포에서 해당 출처를 제거하는 절차도 설명한다.

[CC0 §2~4](https://creativecommons.org/publicdomain/zero/1.0/legalcode.en)는 권리자가 보유한 권리의 포기와 보완 허락을 다룬다. 제3자의 권리를 대신 포기하지 않으며 특허·상표 등을 일괄 허락하지 않는다. CC0 자체의 귀속 의무와 별개로 재현성 및 원문 추적을 위해 HPLT 출처를 기록한다.

따라서 HPLT 전체 원문을 CC0 또는 상업적 재배포 무제한으로 표시하지 않는다. 원본의 `u` URL, `id`, `ts`, `collection`, `f/o/s` 등 수집 식별자를 복원하고 원문 이용 근거를 검토한다. HPLT의 연락처는 원본의 요청 창구이며 우리 공개판의 운영 연락처로 무단 지정하지 않는다.

### FineWeb-Edu

배포 주체는 Hugging Face의 FineWeb 팀이다. 우리 입력은 100BT 표본의 앞56개 샤드이며, 전체 FineWeb-Edu 또는 100BT 전체라고 표시하지 않는다. [고정 카드](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/blob/87f09149ef4734204d70ed1d046ddc9ca3f2b8f9/README.md)는 ODC-By 1.0과 Common Crawl 이용약관을 지정한다. 위 FineWeb2와 같은 DB 고지 및 원문 권리 구분이 필요하다.

원본 교육 분류 점수는 품질 선택 정보이지 저작권 허락이 아니다. 원본 `url/id/dump/file_path`를 연결하고, 우리의 길이·반복 검사·공백 정리·PII 치환·중복 제거를 파생 처리로 명시한다. 상위 데이터 제작에 사용된 모델의 조건을 이 텍스트 데이터셋의 라이선스라고 임의로 옮겨 붙이지 않는다.

### OpenWebMath

원본 저자는 Keiran Paster, Marco Dos Santos, Zhangir Azerbayev, Jimmy Ba다. [고정 카드 License 절](https://huggingface.co/datasets/open-web-math/open-web-math/blob/fde8ef8de2300f5e778f56261843dab89f230815/README.md)은 ODC-By 1.0, Common Crawl 이용약관, 원문 라이선스 불변을 명시한다. HF API의 최상위 `cardData.license`는 null이지만 README의 `dataset_info.license`와 본문에는 ODC-By가 있다. **API null을 무라이선스 또는 자유 이용으로 해석하지 않는다.**

원 데이터에는 포럼·교육 사이트·블로그 등이 포함된다. 예컨대 Stack Exchange·Wikipedia 같은 출처가 있다는 원본 설명은 우리 정제분의 사이트별 분포나 모든 문서의 동일 라이선스를 증명하지 않는다. `url/date/metadata` 복원 뒤 해당 문서의 조건과 귀속을 확인해야 한다. 우리 수학 전용 필터 적용이나 LaTeX 보존도 원문 권리를 바꾸지 않는다.

### CodeParrot GitHub Code

[고정 카드](https://huggingface.co/datasets/codeparrot/github-code/blob/b5661e6b17396364b2bcf8e68977b0d28e1ebd19/README.md)는 라이선스를 `other`로 표시하고 각 예제에 저장소 라이선스 라벨을 제공한다. 카드의 v1.1 설명은 BigQuery 수집 시각을 2022-03-16 18:23:39 UTC+1로 기록한다. [고정 수집 SQL](https://huggingface.co/datasets/codeparrot/github-code/blob/b5661e6b17396364b2bcf8e68977b0d28e1ebd19/query.sql)은 `repo_name`으로 라이선스 테이블을 연결하며 결과에 원 저장소 commit SHA나 LICENSE·NOTICE 본문을 포함하지 않는다.

우리 정제본 전체를 읽은 라벨 분포는 다음과 같다. 이는 파일별 법적 판정이 아니라 원 데이터 라벨의 전수 집계다.

| 라벨 | 보존 코드 행수 | 재배포 시 확인할 핵심 조건 |
|---|---:|---|
| MIT | 925,474 | 저작권 고지와 허락 문구를 사본/상당 부분에 포함 |
| Apache-2.0 | 860,734 | 라이선스 사본, 관련 원 고지, 해당 NOTICE의 귀속 고지, 변경한 파일의 변경 표시 |
| BSD-3-Clause | 219,397 | 저작권·조건·면책 고지 유지, 이름을 이용한 무단 보증/홍보 금지 |
| CC0-1.0 | 56,749 | 실제 권리자의 유효한 CC0 적용 범위 확인; 제3자 권리는 별도 |
| BSD-2-Clause | 52,643 | 소스 및 바이너리 배포 형식에 따른 저작권·조건·면책 고지 유지 |
| Unlicense | 20,607 | 해당 코드에 실제 적용된 공중영역 헌정·허락 확인; 라벨만으로 권리자 확인 불가 |
| ISC | 12,855 | 저작권 고지와 허락 문구를 사본에 포함 |
| 합계 | 2,148,459 | 530,331개 저장소 이름 |

조건의 공식 근거: [MIT](https://opensource.org/license/mit), [Apache 2.0 §4](https://www.apache.org/licenses/LICENSE-2.0), [BSD 3-Clause](https://opensource.org/license/bsd-3-clause), [BSD 2-Clause](https://opensource.org/license/bsd-2-clause), [ISC](https://opensource.org/license/isc), [CC0](https://creativecommons.org/publicdomain/zero/1.0/legalcode.en), [Unlicense](https://unlicense.org/).

선택한 라이선스들은 조건을 충족한 상업 이용을 일반적으로 허용하며, 우리 allowlist에는 GPL/AGPL 등은 없다. 그렇다고 저장소 안의 모든 파일이 저장소 대표 라이선스를 따른다고 확정할 수는 없다. 복사된 제3자 코드·파일별 헤더·예외 라이선스를 확인해야 한다.

우리 코드는 본문을 바꾸지 않아 파일 안에 있었던 고지는 보존한다. 그러나 확장자 allowlist는 독립 `LICENSE/COPYING/NOTICE` 파일을 수집하는 기능이 아니다. 별도 LICENSE 파일이 없는 것이 곧 위반이라는 결론도 아니며, 필요한 고지가 본문에 이미 있을 수도 있다. 이번에는 고지 본문을 전수 판독하지 않았다. 현재 GitHub의 최신 LICENSE를 가져오는 것만으로 2022년 수집 코드의 조건을 입증할 수도 없다. 원본 버전과 파일/고지의 연결이 필요하다.

## 정제에서 유지되거나 빠진 출처

| 자료 | 원본에서 확인한 필드 | 최종 보존 | 복원 또는 추가 확인 |
|---|---|---|---|
| Wikipedia | `id,url,title,text` | `text,h` | 문서 ID·URL·제목, 귀속/이력·추가 고지 |
| FineWeb2 | `url,id,dump,date,file_path` 등 | `text,h` | 원문 URL·크롤 식별자 |
| HPLT | `u,id,ts,collection,f,o,s` 등 | `text,h` | 원문 URL·수집 위치/시각 |
| FineWeb-Edu | `url,id,dump,file_path,score` 등 | `text,h` | URL·원본 식별자, 필요하면 선택 점수 |
| OpenWebMath | `url,date,metadata,text` | `text,h` | URL·수집일·추출 metadata |
| 코드 | `repo_name,path,license,content,size` | `text,h,repo_name,path,license,language,source_row,content_sha256` | 원 Git commit, 파일별 적용 조건·LICENSE·NOTICE·권리자 |

웹의 `h`는 원문 URL이나 암호학적 내용 SHA가 아니다. `data/filters.py`의 소문자·공백 축약한 **PII 치환 전** 텍스트에 대한 xxhash64에서 2^63을 뺀 값이다. 해시만으로 URL을 역산하거나 단일 원문을 확정할 수 없다. 후속 NFC SHA256 중복 판정도 출처/허락의 증명은 아니다.

웹 정제는 공백 정리, 길이/언어/반복 필터, 파일 내 중복 제거, 이메일·전화·주민번호 패턴 치환을 수행했다. 그 뒤 위키→FineWeb2→HPLT→FineWeb-Edu→OpenWebMath 순서의 샤드/소스 간 NFC 본문 SHA 중복 제거에서 먼저 나온 문서를 남겼다. 코드는 원문 UTF8를 보존하며 언어/경로/라벨/길이/생성 코드/일부 secret 패턴을 필터링했다. 모든 개인정보나 비밀키의 완전 제거는 보장하지 않는다.

### 문서별 출처를 복원하는 경로

1. [샤드 목록](data_provenance_v1/file_inventory.json)의 원본 repo/revision/path/SHA를 사용한다. 원본 전체를 무조건 다시 받지 않고, [복구 절차](DATA_RETENTION_AND_RECOVERY.md)와 기존 보존 원본을 먼저 활용한다.
2. 웹은 당시 고정한 정제 코드로 원본 행을 순서대로 처리하며 `source_row`, 원본 URL·ID·수집 metadata, 변환 전/후 본문 SHA를 별도 sidecar에 남긴다. `h` 단독 join을 금지하고 최종 본문 bytes·SHA·순서·원본 후보 다중성을 검증한다.
3. 파일 내/소스 간 중복 제거의 생존 규칙을 재현하여 final file/row에 대응시킨다. 출처가 여러 개인 같은 문서에 대해 권리를 임의로 가장 느슨한 라벨로 바꾸지 않는다. 후보 URL이 여러 개면 모두 기록하거나 원래 생존 행의 근거를 남긴다.
4. 코드는 현재 `source_row`와 원본 샤드, 원문 SHA로 기존 행을 추적할 수 있다. 별도로 원 Git 버전과 고지 파일을 연결한다. 최신 저장소 링크만 만든 것을 버전 복원 완료로 표시하지 않는다.
5. 매핑 누락·다중 후보·삭제된 사이트·확인 불가능한 원본 버전을 별도 상태로 남긴다. 기준을 충족하지 못한 자료는 공개 후보에서 분리하고, 기존 학습본은 변경하지 않는다.

이번에는 **샤드 단위 연결과 원본 필드 확인까지 완료**했다. 1.29억 문서의 URL 매핑 복원, 53만 저장소의 개별 권리 조사, 개인정보 전수 재검사는 수행한 것으로 표시하지 않는다. 이들은 상세 출처 기록을 기반으로 진행할 별도의 실제 데이터 작업이다.

## 공개판에 적용할 구성

소스별로 파일·설정 또는 저장소를 나누고, 각 부분에 원본 출처·revision·변경 내역·해당 고지를 둔다. 전체 원문을 MIT/Apache/CC0 하나로 재라이선스하지 않는다. 코드 도구의 라이선스와 데이터 원문의 조건도 별도다. HF 카드의 복수 라이선스 태그만으로 이런 범위 구분이 전달된다고 가정하지 않는다.

ODC 대상 세 소스는 각각 원 데이터베이스 이름/링크, ODC-By 1.0 링크, 원문 권리와 Common Crawl 약관의 별도 적용을 명시한다. HPLT는 CC0의 packaging 범위를 명시한다. 위키는 문서별 귀속과 변경 고지, 코드에는 파일/저장소별 적용 라이선스 및 필요한 고지를 연결한다. 데이터셋 저자 인용은 유용하지만 원문 저작권 고지의 대체물은 아니다.

연락 가능한 공개 운영자와 원문 URL/문서 ID 기반 삭제 요청 처리 절차를 준비하고, 새 버전에서 반영한 제외 목록을 기록한다. 개인정보를 더 노출하는 URL query 등은 공개 필드와 내부 증거를 구분해 검토한다. 접근 제한이나 연구 전용 문구, 토큰 ID로의 변환은 부족한 권리 근거를 보충하지 않는다.

**다음 실무 우선순위**는 위키 문서별 귀속 복원과 4.0 고지 정합성, 웹 URL sidecar 복원·재배포 근거 분류, 코드의 버전/고지 확보 가능성 평가다. 결과에 따라 재배포 근거를 확인한 부분만 데이터로 공개하고, 나머지는 출처 목록·정제 코드·재현 방법을 공개하는 구성을 검토한다. 실제 외부 업로드는 기존 사용자 확인 규칙을 따른다.
