# 라벨링 가이드 (실제 샘플 gold)

JUN이 사전 라벨을 검수할 때와 에이전트가 사전 라벨을 달 때 같은 기준을 쓰기 위한 문서다. 기준을 바꾸면 이 문서를 먼저 고치고 라벨을 다시 만든다.

## 무엇을 라벨하는가

문서마다 두 가지를 정한다. 두 질문은 독립이다.

**1. 이 문서를 한국어 사전학습 코퍼스에 넣을 것인가 (keep / reject)**

기준은 "현재 필터가 무엇을 하는가"가 아니라 **원하는 결과**다. 현행 필터 판정은 일부러 보여 주지 않는다(앵커링 방지).

- `keep`: 뉴스, 블로그, 포럼 글, 백과, 법령·판결, 기술 문서, 학술 초록, 상품 설명이라도 실질적인 문장이 있는 글. 의견이 강하거나 논쟁적이거나 비속어가 있어도 keep이다(독성 필터는 이 파이프라인에 없다).
- `reject`: 학습 신호가 거의 없는 문서.
  - 규칙으로 설명되는 경우 해당 사유를 쓴다:
    `short`(공백 정리 후 200자 미만), `few_hangul`(한글이 거의 없음, 영어·일본어 등), `low_hangul_ratio`(한글이 있으나 대부분 비한글), `dup_lines`(같은 줄 반복), `ngram_rep`(같은 문장·구절 반복), `low_alnum`(기호·장식 위주), `bad_encoding`(깨진 문자), `long`(10만 자 초과), `empty`.
  - 규칙으로 설명되지 않는 경우 `semantic_low_quality`: 광고·SEO 스팸, 기계번역 필러, 상품 목록·내비게이션·푸터 위주 페이지, 자동 생성 목록.
- `ambiguous`: 정말 결정할 수 없을 때만 `true`. 정확도 계산에서 제외된다. 남용하지 않는다.

**2. 개인정보 span**

`phone`, `rrn`(주민등록번호), `alien_reg`(외국인등록번호 형태), `email`, `card`, `account`, `ip` 일곱 유형만 span으로 라벨한다.

- 개인이 쓴 연락처(개인 휴대전화, 개인 이메일): `ambiguous: false`.
- 기관·업체의 공개 연락처(대표번호, 고객센터, 협회 문의 번호): span을 달되 `ambiguous: true`, 태그 `org_contact_phone`.
- 데이터셋이 이미 치환한 플레이스홀더(예: `firstname.lastname@example.org`): span을 달되 `ambiguous: true`, 태그 `upstream_email_placeholder`.
- **이름, 주소, 사업자등록번호는 span으로 라벨하지 않는다.** 대신 태그 `has_person_name`, `has_address`, `has_biz_reg`를 단다(정책이 정해지지 않았다).
  - `has_person_name`은 **사적 개인의 실명**(인터뷰이, 후기·댓글 작성자, 업체 대표·담당자, 일반인)이 나올 때만 단다. 공인(정치인·연예인·위키 인물 항목의 주인공)과 기자 바이라인은 달지 않는다.
  - 기관·업체의 공개 이메일은 `ambiguous: true` span으로 라벨한다 (`org_contact_email`).
  - `org_contact_phone`, `org_contact_email`, `upstream_email_placeholder` 태그는 **직접 달지 않는다.** `make_labels`가 ambiguous span에서 자동으로 붙인다(전화 → `org_contact_phone`, 예약 도메인 이메일 → `upstream_email_placeholder`, 그 외 이메일 → `org_contact_email`). 직접 달아도 중복은 합쳐진다.
  - `has_*` 태그는 이 개정 이전 라벨(prelabel100)에서는 에이전트마다 기준이 달라 집계에 쓰지 않는다.
- 긴 문서는 시트에 앞뒤 일부만 보인다. 중간은 전체 텍스트 파일에서 직접 검색해 확인한다.

## 판단이 갈리는 경우

| 상황 | 기본 처리 |
|---|---|
| 짧지만 의미 있는 글 (200자 이상) | keep |
| 상품 상세 설명이 문장으로 서술됨 | keep, 태그 `commerce` |
| 상품명·가격·버튼 텍스트만 나열 | reject `semantic_low_quality` |
| 한국어 글에 코드·표가 섞임 | 읽을 수 있는 설명 문장이 충분하면 keep |
| 기계번역투 시장보고서 등 스팸성 필러 | reject `semantic_low_quality` |
| 같은 문장이 앞부분에 한 번 반복되지만 본문은 정상 | keep, 태그 `repeated_opening_lines` |
| 로그인·메뉴 안내와 실제 글이 섞임 | 실제 글 비중이 절반 이상이면 keep, 아니면 reject `semantic_low_quality`, 정말 모르면 ambiguous |
| 영어 문서 (한국어 아님) | reject `few_hangul` |

## 출력 형식 (에이전트용 spec)

```json
{"agent": "A",
 "docs": {
   "real-xxxxxxxxxxxx": {
     "label": "keep", "reject_reason": null, "ambiguous": false,
     "tags": ["news"], "notes": "원문을 인용하지 않는 짧은 메모",
     "pii": [{"needle": "010-0000-0000", "type": "phone", "ambiguous": false, "occurrence": "all"}]
   }}}
```

- `needle`은 문서 텍스트에 **그대로** 있는 문자열이다. 오프셋은 도구가 계산한다. `occurrence`는 `all`(기본) 또는 `first`.
- `reject`이면 `reject_reason` 필수, `keep`이면 `null`.
- spec 파일에는 실제 개인정보 문자열이 들어간다. **커밋하지 않는다.** 커밋되는 것은 오프셋·유형·라벨뿐이다.
- `notes`와 보고에 원문이나 개인정보를 인용하지 않는다.

도구: `python -m dataqa.make_labels --sheet <jsonl> --specs spec.json --check`로 needle이 모두 해석되는지 먼저 확인한다.

## `short` 라벨에 대한 주의

`short`는 이 가이드가 프로덕션 임계값(200자)과 같게 정의했다. 따라서 "규칙이 `short`를 모두 잡았다"는 결과는 독립적인 증거가 아니라 정의의 반복이다. 토막글(예: 위키 스텁)을 버릴지는 라벨이 아니라 **정책 결정**이다.

## 검수 시 보는 법 (JUN)

1. 에이전트 라벨과 현행 필터 판정이 **어긋난 문서**를 먼저 본다.
2. `ambiguous`와 정책 태그(`has_person_name` 등)가 달린 문서를 본다.
3. 나머지는 무작위로 몇 건만 확인한다.
4. 결과는 커밋된 파일만으로 적용할 수 있다(스크래치패드의 spec이 없어도 된다):

```json
{"reviewer": "jun",
 "reviewed": ["real-...", "..."],          // 실제로 확인한 문서 전부 (동의한 것 포함)
 "random_controls": ["real-...", "..."],   // 그중 검수 시트의 '무작위 대조군'
 "corrections": {"real-...": {"label": "keep", "reject_reason": null, "ambiguous": false}}}
```

```bash
python -m dataqa.make_labels --apply-review review.json --labels dataqa/real/prelabel100_labels_v1.jsonl \
    --out dataqa/real/prelabel100_labels_v2_human.jsonl --report dataqa/real/prelabel100_review_report_v1.json
```

   `label_source`는 `human_review:<검수자>:agreed|corrected`가 된다. 일치율은 전체와 **무작위 대조군만**을 따로 보고한다(전체는 의심스러운 문서를 골라 보았으므로 낮게 편향된다). **span과 태그 수정은 이 경로의 범위 밖이다**(원문이 필요하다). 필요하면 `real_sample materialize`로 텍스트를 받아 spec을 새로 만든다.
