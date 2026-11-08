> Historical technical snapshot (2026-11-08). Original paths and earlier status statements are retained. For the public release entrypoint and current verification scope, see [PUBLIC_GUIDE](PUBLIC_GUIDE.md) and [PUBLIC_VALIDATION](PUBLIC_VALIDATION.md). Missing run artifacts are not bundled data.

# 겪은 문제 / 교훈

## 실제 전체입력 리허설 시작·재개에서 확인한 문제 (2026-11-06)

- `train/rehearsal.py`를 파일 경로로 실행하면 `train/data.py`가 최상위 `data` 패키지를 가릴 수 있다. torchrun의 `--module train.rehearsal` 진입을 사용하며 실제 v3에서 import·8GPU 시작을 확인했다.
- 설치된 DCP의 `get_state_dict`는 optimizer state를 초기화하기 위해 LR0·zero-gradient의 `optimizer.step(closure=None)`을 호출한다. 관측 hook은 이를 학습 update로 세거나 샘플 부재 오류로 거부하면 안 된다. 빈 optimizer state·실제 update/샘플 없음·명시적 closure=None·전 group LR0 조건의 초기화만 별도로 세고, 그 밖의 샘플 없는 step은 원 optimizer 호출 전에 거부한다. v3 실제 step4→main 재개와 후속 학습 진행을 확인했다. 실패한 v2 checkpoint의 계약을 고쳐 재사용하지 않고 코드 snapshot·산출물을 보존했다.

1. **cgroup 메모리 OOM → multiprocessing.Pool 영구 정지**: 컨테이너 memory.max=240GB인데 정제 워커 10개가 각 ~20-25GB(샤드 2.78M문서를 메모리에 모음)를 써서 OOM kill 5회.
   Pool은 죽은 워커의 작업을 잃어버리고 imap이 무한 대기 → 11시간 낭비. 교훈: 워커 수를 `memory.max / 샤드당 메모리`로 제한, 로그를 짧은 주기로 확인, `/sys/fs/cgroup/memory.events` 확인. (현재 clean.py는 샤드 전체를 메모리에 모으는 구조; 큰 샤드는 row-group 단위 스트리밍 쓰기로 개선 권장)
2. **`pkill -f clean.py`가 자기 셸도 종료**(명령줄에 패턴 포함). pid로 종료할 것.
3. **xxhash는 bytes만 받음**(str 불가) → `.encode()`.
4. **HF 접속 정상, 다운로드 매우 빠름**(100GB가 몇 분): raw 데이터는 옮기지 말고 새 환경에서 `data/download.py`로 재다운로드.
5. **정제 속도**: FineWeb-2 샤드(2.78M문서, 4.8GB) 하나가 35~45분(CPU 경합 시). 순수 Python 필터가 병목.
6. **FineWeb-2의 language=kor 라벨에도 영어 문서가 섞임** → 한글 비율 필터 필수.
7. HPLT와 FineWeb-2는 둘 다 CommonCrawl 기반이라 **상당히 겹칠 가능성**. 정확해시로는 안 잡히므로 URL 겹침 측정 → MinHash 필요 여부 판단 (원본 raw에는 url 컬럼이 있음: FW2 `url`, HPLT `u`). clean 결과에는 url이 없음.
8. 정제 필터의 영숫자 비율(low_alnum<0.5)은 **수학 데이터(OWM)에는 부적절**할 수 있음 → OWM은 완화된 필터 사용 검토.

9. **파일 크기·parquet footer만으로 내부 데이터 무결성을 보장할 수 없음**: FW2 `002_00000.parquet`는 크기와 metadata가 정상인데 text 581,632행 후 ZSTD 손상 오류가 재현됨. 새로운 다운로드는 원격 LFS SHA256 검사, 손상 복구는 별도 경로에서 다시 받고 전체 컬럼 읽기까지 확인한 후 적용한다. 이전 hf_transfer partial 재개 파일이었지만 손상 발생 원인은 확정하지 않음.

10. **코드 정제의 원문 보존과 tokenizer 정규화는 별도 검증 대상**: NFC는 코드 literal의 분해형 Unicode를 합성형으로 바꾸어 문자열의 바이트/길이를 바꿀 수 있다. N1 fixture에서 일반 NFC roundtrip은 통과하면서 코드 raw 변화가 확인됐다. 검사기는 needs_review로 분리하고 실제 tokenizer 학습 전에 정책을 결정한다. NFC 성공만으로 코드 원문 보존을 주장하지 않는다.

11. **토크나이저 경계를 따로 encode해 객관식 점수를 계산하지 않음**: prompt+choice의 실제 encoding에서 prompt prefix가 유지되는지 확인한다. 병합 경계가 바뀌면 템플릿을 재검토하고 해당 결과를 점수로 보고하지 않는다. 합계와 토큰 평균도 task 계약에 구분해 기록한다.
12. **SQLite 재개 지점은 데이터와 같은 transaction에 저장**: ingestion의 row·union 결과, assignment의 cursor, finalize phase를 함께 commit해야 중복/누락 없이 재개된다. N7은 os._exit 강제 종료로 rollback과 commit 이후 복구를 구분해 확인했다. 10만 건 성공을 1억 건 처리 준비 완료로 해석하지 않는다.

13. **표본 문서 수와 실제 읽기량은 다름**: Parquet에서 소수 행을 선택해도 row group을 읽는 비용이 발생한다. N11 최초 실행은 읽기 상한에서 영어/코드 표본이 부족해 partial로 보존했다. 파일/row group 구성을 줄인 새 frame으로 네 source를 맞췄으며, 읽기 예산을 늘려 원래 목표를 억지로 채우지 않았다. 파일별 group 수와 상한을 계약에 포함한다.
14. **0건의 분모를 확인**: N11 실제 표본에는 eligible fenced block이 없었다. 일치 0건만 보고 코드 블록 교차중복이 없다고 판단할 수 없다. 검사 대상 자체의 수, 후보 생성 범위/재현율, 잘림 여부를 함께 기록한다.

15. **후보 재현율의 합성/실제 분모를 분리**: 실제 표본의 양성0이면 재현율은 null이다. 합성 양성100% 탐지는 후보 생성의 제한된 회귀 증거이며 실제/전역 recall 증명이 아니다. 실제 후보를 observer로 추적하고 독립 unhashed n-gram 완전 비교와 대조한다.
16. **빠른 worker의 peak도 종료 시 확인**: 부모의 주기 RSS 관측 사이에 worker가 끝날 수 있다. 성공 종료 후 child RUSAGE HWM과 최종 출력 크기/여유공간을 다시 확인해야 제한 초과를 통과시키지 않는다. journal/출력 sampled peak는 정확한 최대값이라고 표시하지 않는다. 큰 split 측정은 source audit와 I/O 경쟁을 피한다.

17. **bf16 compile/eager 차이를 optimizer 문제로 바로 단정하지 않음**: FP32기준·생산parameter그룹·층별같은입력·반올림barrier를분리한다. 이번SwiGLU의중간cast보존으로단일연산차이는사라졌지만전체3B잔여차이는남았다. AdamW update상대오차는작은gradient의부호변화에민감하며전체가중치변화율이아니다. 측정완료와production수치승인을구분한다.

18. **선택rowgroup별고정quota는충분한총용량이있어도목표를못채울수있음**: HPLT확대표본1991/2000으로partial. 남은선택group용량을고려해quota를재배분하고읽기/저장한도는그대로유지한다. 작은tail100/10행fixture로목표100재현/수정검증.
19. **작은SQLite분리성능을선형외삽하지않음**: 100만case는IO대기·반복무결성검증비용으로cache64에서22분,512에서7.7분. 100만행논리동일성을비교했고메모리비용도기록했다. cache에큰부분이들어가는100만DB성공은실제1.29억건의성능보증이아니므로working-set을키운추가측정이필요하다.
# 실제 학습 입력 전용판 배포에서 확인한 점 (2026-11-07)

- `mix.holdout` 전체 목록과 실제 실행의 평가 입력은 같다고 가정하지 않는다. 현재 native trainer는 명시적 `config.holdout`을 우선하므로, mix의726개 fallback 중 config가 고른6개가 실제 본학습 입력이다. 원mix/config bytes를 보존하고 실제 선택 규칙과 제외 범위를 문서화한다.
- 현재 테스트 환경의 `AutoTokenizer.from_pretrained(dataset_id)`는 기본 model endpoint를 조회해401/RepositoryNotFound를 낼 수 있다. 쓰기토큰 문제로 오진하지 않는다. `snapshot_download(..., repo_type="dataset", allow_patterns=["tokenizer/*"])` 후 local경로로AutoTokenizer를 불러온다. 최종카드의예제를 실제익명/pinned배포에서실행한 뒤 공개완료로판단한다. 수정은데이터가같아도새문서patch태그로내고기존태그와증명을보존한다.

