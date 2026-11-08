> Historical technical snapshot (2026-11-08). Original paths and earlier status statements are retained. For the public release entrypoint and current verification scope, see [PUBLIC_GUIDE](PUBLIC_GUIDE.md) and [PUBLIC_VALIDATION](PUBLIC_VALIDATION.md). Missing run artifacts are not bundled data.

## 최신 체크포인트 공간 정책 (2026-11-07)

NFS최신3개+로컬과거최대15개 적응형 보존이 실제 적용됐다. NFS3+작성1·10%·메타1GiB 약159.45GiB, 로컬15+복사1·10%·메타1GiB 약634.80GiB 상한이다. 로컬은 다음 사본 성장분을 예약하고 압박 시 검증된 과거 이력부터 축소한다. 임시1,024GiB 및 파일시스템별200GiB는 유지하며 `/`와`/tmp`는 합산하지 않는다. 아래 이전10+15 수치는 과거 설계다. 실제 가용량은 `logs/resource_usage_latest.json`, 근거는 `runs/retention3_migration_v1/report.json`을 따른다.

## 최신 비용 선택 기준 (2026-11-07 사용자 지시)

외부 저장·전송안은 **업로드·다운로드 비용 무료/최소화 → 저장 용량 비용 → 속도** 순이다. 전송 관련 API/조회/중계 비용도 포함한다. 아래 과거 총시간 최우선 문구보다 이 선택 기준이 우선하며, 무결성·실제 재개·보존/자원 여유는 유지한다. 추가 비용 없는 서버 내부 최적화는 계속한다. 상세 비교는 TRAINING_COST_EXPERIMENT.md를 따른다.

## 최신 우선순위: 전체 완료시간 단축 (2026-11-06 사용자 지시)

## Clean inactive cache를 고려하는 새 작업 준입 (2026-11-06T21:17:47.745009+00:00)

전수토큰화뒤cgroup file cache약1698GiB가채워지면서기존conservative_headroom은0이었으나OOM은없었다. 새1B/hook/관리기는 `scripts/resource_admission.inspect()`로원hardware_profile실측을유지하고별도 `memory.cache_aware_admission`을계산한다: cleancredit=max(0,min(current,inactive_file,file-shmem)-dirty-writeback), available=min(limit-current+credit,hostMemAvailable)-기존reserve. 값은0아래로내리지않고RSS한도와별개다.

하드quota/원conservative지표를바꾸지않으며activefile·shmem·dirty/writeback을여유로이중계산하지않는다. cacheflush/sysctl/mount/GPUclock은변경하지않는다. 현재약1653GiB추가headroom과실제job64/256GiB RSScap·128GiBsystemreserve·200GiBdiskreserve를같이사용한다. 새도구가원지표와별도준입값을함께기록한다. 기존실행중producer와완료증명의codepin은보존했다.

### 닫힌 NFS DB의 읽기 캐시 준비 실측

2026-11-06 실제115.78GB 최종DB의 NFS 임의읽기 소비자가 D/낮은CPU에머물러,4reader·8MiB block·8pending·900초예산으로읽기전용전체SHA/미리읽기를추가했다. 537.33초에canonical SHA일치,원소비자는이후R/1코어CPU활동과캐시읽기(관측구간physical0)로전환했다. `runs/f4_prefetch_v1`에입력/자원/완료/전후활동기록. 종단검증시간은아직미확정이며cache가따뜻한동일DB에매번반복하지않는다. 원producer·SQL검증·data를변경하지않고현재/향후peak·남은RAM·읽기cache비용을구분해예산을잡는다. source종료/예산초과시보조읽기만종료한다.

### 단계 N 실행 중 N+1 설정 선행 탐색

각 단계 시작 시 다음 단계의 입력/코드/장치 조건, 기존 실측, 남은 병목과 후보를 실행 기록에 적는다. 현재 job의 향후 RSS·디스크 peak와 reserve를 먼저 배정하고, 남은 예산에서 독립 입력의 CPU 준비 또는 GPU 검증을 병행한다. 탐색 때문에 진행 중 producer를 변경하거나 후속 단계의 필수 진입 조건을 넘지 않는다.

운영표는 `runs/pretrain_readiness_v1/stage_optimization_queue.json`에 유지한다. 단계 시작·변경·새 병목 발견 시 갱신하며, 후보별로 준비/측정/검증/선택/적용/보류/기각을 구분한다. 각 후보에는 대표 입력·코드·측정 근거, 실제 worker/thread/cache, 현재 작업의 향후 peak를 차감한 CPU/RAM/장치/디스크 예산, 예상 절약 시간과 탐색 비용, 적용 대상의 새 실행 계약을 기록한다. 계획 등록이나 작은 fixture 통과를 적용 완료로 표시하지 않는다.

현재 작업이 필요한 GPU를 모두 점유하면 GPU 탐색은 대기시키고 CPU 입력 준비·분석을 앞당긴다. 병행 전후 현재 작업의 같은 종류 구간 처리량과 메모리/IO pressure를 비교하며, 경합으로 전체 종료가 늦어지면 보조 작업부터 축소·중지한다. 반복 확인은 기존 30초 로컬 감시를 사용하고 매 확인마다 모델을 호출하지 않는다. 다음 단계 진입 전에 선택 근거와 자원 실측을 다시 확인하고 검증된 설정을 새 실행에 반영한다.

한 축씩 후보를 좁혀 같은 검증을 포함한 총시간·RSS·출력 SHA·재개를 비교한다. 작은 차이는 반복 측정하며, 불명확하거나 느린 설정은 채택하지 않는다. 다음 실행에는 선택 근거와 입력/코드/하드웨어 조건, 실제 thread/cache/worker 값, 예산, 재검증할 범위를 연결한다. 기존 결과가 적용되는 경로를 반복 측정하지 않으며, 탐색 비용이 예상 절약을 초과하면 조합을 줄인다.

현재 F5 출발 후보는 BPE16 threads/streaming, F6은 worker24×Rayon2/batch1024/Arrow1/NFS직접과 근접후보16×3이다. 이는512MiB 실험 결과이며 최종tokenizer/전체파일에서 재확인한다. 분포 집계의 group-range 병렬 후보는132k행에서 기존보다 느려 미채택이며, native TEMP 집계와 검증 공유를 별도로 비교한다. 실행 중인 F4의 producer·cache 설정은 고정한다.

작은 fixture의 안전 상한을 생산 기본값으로 승계하지 않는다. CPU96/RAM1920GiB/H2008장/로컬2.5TiB 여유를 바탕으로 큰 캐시·병렬 읽기·계산·배치 후보를 실측하고, 같은 검증을 포함한 elapsed와 처리량을 기준으로 선택한다. 메모리 할당량이나 사용률을 높이는 자체가 목표는 아니다. stage별 실측 없이 최적이라고 부르지 않는다.

- 실제 SHA 검증: 대표32파일 약102GiB에서16worker 중앙7.69초,32worker4.40초. 현재 데이터용 후보32worker, 동일 실행에서 검증한 파일 SHA 재사용으로 이중 전체해시pass 제거. 기존pagecache/shared-host 조건이며cold NFS의보장속도는아니다.
- SQLite: 300만짧은본문fixture에서512MiB→8GiB만으로전체시간개선은관측하지못했다. 실제전체DB는계획상약100GiB여서working-set을담을128GiBcache를검증후full_v2에기동했다(3M상한설정검증,실제full최적성능은미확정). 프로세스RSS상한은cache와정렬·decode·검증을포함해별도로책정한다.
- 전처리/토큰화: workers와각worker의Arrow/Rayon/OMP thread합계를CPUaffinity에맞추고, per-worker실측RAM으로동시성을정한다.4/8/16/32 등후보중처리량이포화되는지확인한다. BPE단일학습기의Rayon과여러shard토큰화worker를같은설정으로다루지않는다.
- GPU: 독립검사는장치별,학습/FSDP는검증된8장구성을활용한다. 수치허용·재개·입력조건을통과한최적화만채택한다.
- 원본보존,본학습/외부업로드확인,로컬유실복구,내용증명은유지한다. 성능변경을위한실행중단은이유·기존산출·코드스냅샷·새계약을남긴다.

# 이 기기의 자원을 활용하는 실행 기준

사용자 지시(2026-11-06): 향후 모든 작업을 이 기기의 전체 자원을 효율적으로 활용하는 방향으로 진행한다. 목표는 단계별 병목과 전체 소요 시간을 줄이는 것이다. 기존 본학습·자료 삭제/덮어쓰기·외부 업로드 확인 규칙은 유지한다.

## 확인한 자원과 실제 병목

- CPU affinity 96개, 관측된 cgroup CPU quota 제한 없음. 호스트 전체 CPU 수 대신 컨테이너 affinity와 quota 중 작은 값을 사용한다.
- 메모리 한도 1,920GiB. 사용량 약 1TiB의 대부분은 파일 캐시이고, 관측 당시 익명 메모리는 약 1GiB였다. 캐시를 실제 프로세스 점유로 오인하지 않는다. 실행기는 보수적으로 limit-current와 MemAvailable 중 작은 값에서 최대128GiB의 여유를 남긴다.
- NVIDIA H200 8장, 카드당 보고 메모리 약140GiB, 두 NUMA 노드와 NVLink 연결. 분산 학습/검증은 검증된 8GPU 경로를 쓰고 독립 GPU 사례들은 장치별로 배치하는 방향으로 설계한다. NUMA별 CPU와 GPU 매핑은 실행 전 확인한다.
- 프로젝트/영구 결과는 NFS, 여유 약938GiB. `/tmp`는 별도 overlay 파일시스템으로 여유 약2,538GiB. `/dev/shm`은1GiB여서 큰 DB 작업 공간으로 사용하지 않는다.
- SQLite 100만 건/cache512 비교: NFS461.61초 → 로컬97.11초, 약4.75배. 생성 입력 SHA와100만 행의 내용·repo·group·split이 일치했다. 단일 shared-host 측정이며 물리 DB 경로/배치는 다르고, 로컬 측정은 별도 NFS300만 작업과 병행했다. 증거 `runs/resource_policy_validation_v1/local_vs_nfs.json`.

## 새 장시간 작업을 시작하는 순서

1. `python scripts/hardware_profile.py --output logs/hardware_profile.json`으로 실제 제한·메모리·파일시스템을 다시 읽는다. 실행기 자체도 매번 새 프로파일을 만든다.
2. 단계별 스레드·작업 수·메모리·GPU·저장소 예산을 정하고, 입력/출력/코드/설정과 함께 기록한다. 실행 중인 비교 작업은 설정을 유지하고 다음 새 run에서 변경 효과를 측정한다.
3. `scripts/run_with_resources.py` 또는 동등한 명시적 설정을 사용한다. 실행기는 스레드 환경, Python 스크립트의 Arrow CPU/IO thread 수, 로컬 TMPDIR/SQLITE_TMPDIR/Inductor cache를 실제 자식 프로세스에 적용한다. 작업 수와 RSS/DB크기 제한은 생산기 실행 인자로 별도 지정해야 한다. 실행기가 임의의 CLI 인자를 추가하거나 자식의 메모리를 자동 제한하지는 않는다.
4. 동일 데이터·단위의 처리량과 RSS/디스크/대기 시간을 비교한다. 개선이 확인될 때 작업 수를2→4→8 등으로 올리고, I/O 대기나 메모리 비용이 커지면 조정한다. 동시에 실행하는 전체 작업의 예산도 합산한다.
5. PID/state/log/완료증명을 독립 job registry에 등록해30초 로컬 감시를 적용한다. 매 poll마다 모델 호출이나 외부 메시지는 없다.

## 단계별 출발 설정

| 단계 | 출발 설정 | 후속 조정 |
|---|---|---|
| 독립 파일 읽기·정제·해시 | 정제16→최대32후보, actual SHA32worker실측선정, worker당 RAM/Arrow 제한 | source/shard 단위로 병렬화하고 결과 순서·내용 ID·재개 증명을 유지. 이미 완료된 작업은 자원 점유를 위해 반복하지 않는다 |
| 토큰화 | shard 작업8개, worker당 Rayon8/Arrow2/OMP1 출발 후보 | producer가 지원하는 실제 worker 인자를 명시. 여러 풀의 스레드 합계를 CPU 예산과 맞춘다 |
| SQLite 그룹 분리 | 단일 writer, hash32, cache최대128GiB/RSS384GiB/Arrow4 | 현재full_v2설정. 실제가용RAM에따라기본값축소,DBworking set성장과실측throughput으로조정. 로컬임시공간과영구SHA/소비검증유지 |
| 단일 BPE 학습 | tokenizer-train preset: Rayon최대64, Arrow8;8/16/32/64 비교후선정 | 여러shard worker와분리. 실제표본으로RAM/전체시간을측정하며무조건64가빠르다고가정하지않음 |
| GPU 연산·학습 | 검증된 8GPU 또는 장치별 독립 검사, OMP2·compile thread4/프로세스 출발 | CPU 입력 준비와 GPU 연산을 병행하고 실제 batch·메모리·재개·수치를 검증. 새 precision/compile 설정은 별도 수치 승인 필요 |

96CPU·1.9TiB에 비해 기존2스레드·작은 SQLite 캐시 설정이 보수적이었다. 다만 SQLite의 단일 writer에 프로세스를 무작정 추가하면 그룹 연결과 트랜잭션 계약을 훼손할 수 있다. 병렬 읽기/해시 → bounded queue → 단일 writer 구조와 로컬 저장소를 우선 검토한다. GPU에 맞는 연산과 CPU/IO 작업은 별도 예산으로 병행한다.

## 로컬 작업 공간의 발행·복구 조건

로컬 공간은 컨테이너 재생성 때 보존을 보장할 수 없다. 원문과 최종 증명·통계는 영구 프로젝트 저장소에 둔다. 실제 대량 SQLite 작업을 로컬로 전환하기 전에는 프로세스 강제 종료 재개, 입력 계약 변경 거부, 로컬 DB → 영구 파일의 원자적 복사/해시/논리 검증, 컨테이너 유실 시 재생성 또는 일관된 백업 복구를 검증한다. 검증되지 않은 로컬 작업 DB를 유일한 최종 결과로 사용하지 않는다.

로컬 발행/강제종료·컨테이너유실 재생성은검증됐고full_v2에서사용중이다. 원격중간DB백업은없으며이전local유실시재생성비용을명시한다. hash32병렬검증은적용,ingest decode/정규화의병렬producer는추가실측대상이다. 100만종단발행은345.9→165.1초/2.09배실측,전체corpus개선율은미확정이다.

```bash
# 새 하드웨어/여유 상태를 기록
python scripts/hardware_profile.py --output logs/hardware_profile.json
# 실제 스레드/로컬 임시 경로를 적용. producer의 worker 옵션은 해당 CLI에 별도로 명시
python scripts/run_with_resources.py --stage tokenization -- python data/tokenize_shards.py <해당 인자>
python scripts/run_with_resources.py --stage sqlite -- python scripts/measure_split.py <해당 인자>
# 실행 계획만 기록/확인
python scripts/run_with_resources.py --stage gpu --dry-run -- torchrun <검증된 인자>
```

실행 계획은 `logs/resource_launches/<timestamp>/plan.json`에 기록한다. 임의의 명령 인자와 환경 전체는 기록하지 않아 토큰·비밀번호 노출을 피한다. source 데이터 검증 조건과 학습 실행 승인은 자원 프리셋과 별개다.


### 검증된 로컬 발행 실행 경로

`publish_local_split.py`는 로컬 DB 생성/transaction 재개, 닫힌 DB SHA·fsync 복사, staging consumer 검증, 영구 디렉터리 원자적 rename, 최종 consumer 검증을 수행한다. `run_local_split.py`는 별도 process group과 attempt별 시간/RSS/양쪽 디스크 상한을 감독한다. 컨테이너 유실 전 중간 DB는 원격 백업하지 않으며, 영구 staging 이전의 유실은 immutable source 재생성으로 복구한다. 강제 종료/유실/충돌/변조 테스트 통과와 실제 100만 cross-filesystem 검증의 최신 완료 여부는 STATUS를 확인한다. 생산 원본을 scratch로 이동하지 않는다.
