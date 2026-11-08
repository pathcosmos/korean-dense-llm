> Historical technical snapshot (2026-11-08). Original paths and earlier status statements are retained. For the public release entrypoint and current verification scope, see [PUBLIC_GUIDE](PUBLIC_GUIDE.md) and [PUBLIC_VALIDATION](PUBLIC_VALIDATION.md). Missing run artifacts are not bundled data.

# 본학습 보존: NFS 최신3개 + 용량에 따른 로컬 과거최대15개

## 현재 적용 상태 (2026-11-07T23:31 UTC)

사용자의 “이동 삭제 정책도 적용하고 실제 작업 하자” 승인으로 실제 적용했다. 관리 대상은 `runs/3b_full_v1/training/ckpt_XXXXXXX`와 기존 승인 로컬 namespace `/tmp/llm-dense-d7135f5715/checkpoint_history/3b_full_v1/0b89d94159d94614bb213d9a46c89d78/`뿐이다. 기존 데이터·리허설·진단 산출물은 삭제 대상이 아니다.

- NFS: 최신 인증3개. 현재3000/3500/4000. 로컬: NFS에서 제외된 과거최대15개. 현재500/1000/1500/2000/2500. 정상 고유 사본 최대18개이며 작성/복사 중 중복은 별도 peak다.
- 로컬은 별도 임시1,024GiB와 안전200GiB를 남기도록 오래된 검증 사본부터 축소한다. 신규 복사 전 및 평상시 공간 압박 시 점검한다. 다른 보존 transaction과 공간 압박이 겹치면 경쟁 삭제 대신 안전 중단한다. 무명의 잔여물이나 다른 자료를 지워 해결하지 않는다.
- 임시 복사→파일 목록/크기/SHA→완료 발행·인증→NFS 정리 순서다. 서로 다른 파일시스템 사이 rename을 쓰지 않는다. 재개 기준은 NFS만이며 로컬 유실은 허용된 추가 이력 손실이다.
- 원 학습 worker425989와 rank426027..426034는 그대로 유지됐다. 감독기만 계약 개정 및 공간 피드백 적용을 위해 두 번 인계했다. 현재 감독기778439. 학습5개 코드·설정·training contract ID·목표57000 불변. 이전 계약과 코드/대기 controller 근거는 `runs/retention3_migration_v1/original_*`, `applied_contract_before_capacity_feedback.json`에 보존했다.
- 현재 NFS 예산은3+작성중1·10%·메타1GiB 합계 약159.45GiB, 로컬 상한 예산은15+복사중1·10%·메타1GiB 약634.80GiB다. 로컬 예약 계산은 적응형 삭제를 반영해 다음 사본1개 성장분을 사용하고, 임시1TiB는 한 번만 예약한다. `/`와`/tmp`는 합산하지 않는다.

검증: 보존/감사34개, 실제 CPU 프로세스 인계4개, 최종평가/복원 공간11개 시험 통과. 실제3B 복사16.69초·전체 CPU 복원 비교60.03초·peakRSS72.96GiB, model/optimizer1,020 tensors 및8rank RNG exact. 실제4000 정상 저장 뒤2500 사본의 로컬 이전과 NFS3개 유지도 확인했다. 전체 SHA 실제 근거 및 개정 후 불변stat/인증서 재사용 범위는 `runs/main_step4000_review_v1/storage_snapshot.json`, `runs/retention3_migration_v1/retained_sha_reuse.json`에 구분한다.

한계: 실제 로컬 사본은 현재5개이므로 생산 환경의15개 초과/공간 압박 삭제는 아직 발동하지 않았다. 이 경로는 CPU 실제 fixture로 검증했다. 새 GPU 복원을 실행한 것은 아니며 최종57000 실제8GPU 복원·native·보존 감사는 여전히 필요하다.

감독기 복구: 살아 있는 worker를 먼저 실제 PID/명령으로 확인한다. worker가 남아 있으면 새 PID/start-time/계약 해시에 묶인 승인된 인계 receipt로 인계한다. 모든 학습 worker가 종료됐을 때만 동일 개정 계약의 `python scripts/supervise_main_training.py --run-dir runs/3b_full_v1 --run`으로 NFS 최신 인증 사본을 fresh 검증 후 재개한다. 관측 timeout만으로 재기동하지 않는다.

최종 대기 경로: observer_v4, native_v5, resume_v4, retention_audit_v3. 실제 값은 `runs/main_training_readiness_v1/progress.json`을 따른다.

## 변경 전 10+15 설계·검증 이력

아래 내용은 이전 정책 기록이며 현재 실행 정책보다 우선하지 않는다.

# 본학습 NFS 10개 + 로컬 과거 15개 보존

현재 상태(2026-11-07 06:03 UTC): 사용자 승인 receipt `runs/3b_full_v1/authorization.json`을 기록하고 검증된 감독기를 기동했다. **해당 새 run의 본학습 시작과 NFS10+로컬15 보존·정리 실행이 승인됐다.** 실제 checkpoint 발행·보관·정리 결과는 이후 운영 증거로 확인하며, 실제크기25개 장기운영 검증은 아직 완료되지 않았다. 아래 미승인/미실행 문구와 용량 수치는 구현 당시 기록이며 최신 계약·자원·상태는 STATUS를 따른다. 기존 데이터·리허설·진단 산출물은 관리 대상에 포함하지 않는다. REF-TRAIN의 실제 저장/재개·내용 증명 원칙을 유지한다.

## 정확한 관리 범위와 실행 계약

검토용 계약은 `runs/pretrain_readiness_v1/tiered_retention_execution_draft_v2.json`이다. 정상 보존 상태는 서로 다른 checkpoint 최대25개이며 작성/복사/검증 중 사본은 별도 peak 예산으로 관리한다.

- 영구 새 run: `runs/3b_full_v1/`. 재개 검색은 그 아래 `training/ckpt_XXXXXXX`만 대상으로 한다.
- NFS 정상 보존: 최신 **10개**. 정리 중 사본은 같은 run의 `retention_trash/`에 둔다. 기존 리허설·진단 run·학습 데이터는 관리 대상이 아니다.
- 로컬 이력: `/tmp/llm-dense-d7135f5715/checkpoint_history/3b_full_v1/0b89d94159d94614bb213d9a46c89d78/`. `checkpoints/`, `staging/`, `certificates/`, `retention_trash/`만 사용한다. 해당 run의 NFS 보존 범위에서 빠진 과거 **15개**가 목표다. 동일 owner/run ID가 아니면 사용을 거부한다.
- 본학습 임시 작업: `/tmp/llm-dense-d7135f5715/3b_full_v1/`. 로컬 이력과 분리하며, `/`와 `/tmp` 용량은 합산하지 않는다.
- 전체 DCP model/optimizer, rank0..7 RNG, meta/config/입력·코드 계약과 완료 marker를 복사한다. source ledger와 인증·운영 journal은 해당 영구 run에 보존한다. 모델만 따로 보관하는 방식이 아니다.
- 학습기 `keep_ckpts=57001`로 자체 정리를 끈다. 감독기가 인증 후 보존을 관리한다. 시작·새 checkpoint 정리 승인값은 모두 **false**다.

F8/최종 native/monitored3B와 최종 생성·곡선 검토가 끝난 뒤 실행할 **준비 명령**:

```bash
python scripts/supervise_main_training.py \
  --run-dir runs/3b_full_v1 --prepare \
  --keep-checkpoints 10 --local-history 15 \
  --retention-plan runs/pretrain_readiness_v1/tiered_retention_execution_draft_v2.json
```

이 명령은 아직 완료되지 않은 실제 최종 보고서가 없으면 새 본학습 디렉터리 생성 전에 거부한다. 실제 준비 시 기존 draft의 정확한 로컬 namespace를 유지하고, 현재 장치·공간·검증 코드·입력을 다시 고정한다. 사전 검증들이 모두 종료된 뒤에는 그 산출물이 이미 free space에 반영되므로 과거 후속 peak를 다시 차감하지 않는다. 생성된 config/contract/복구 명령을 검토한 후 **본학습과 이 새 run의 정리 정책을 별도 확인**한다. 현재 draft만으로 `--run`을 실행할 수 없다.

## 발행·정리 순서

1. 새 NFS checkpoint의 `.complete`, DCP storage map/bounds, 전체 파일 목록·크기·SHA, model/config/계약, 8rank RNG를 인증하고 파일/디렉터리를 fsync한다. 아직 인증되지 않은 사본은 정상 개수에 포함하지 않고 기존 유효 사본을 정리하지 않는다.
2. NFS 최신10개에서 제외되는 사본을 로컬 `staging/<checkpoint>`에 복사한다. source는 읽기만 한다. 파일별 중단 잔여물은 동일 staging에서 검증·재복사하며 불명의 파일이나 링크는 거부한다.
3. 목적지 파일 목록·크기·SHA·meta identity를 전수 확인하고 fsync한다. 로컬 내 rename으로 `checkpoints/<checkpoint>`를 발행한 뒤 fsync된 인증 파일을 쓴다. NFS→로컬 rename을 사용하지 않는다.
4. 최신10개의 NFS 인증이 유지되고 로컬 수신 확인이 완료된 뒤, 오래된 NFS 사본을 **동일 NFS 파일시스템**의 `retention_trash/<checkpoint>/payload`로 rename한다. 영구 intent와 journal을 먼저 fsync한다. 정해진 파일 목록 외 데이터는 지우지 않는다.
5. 로컬의 서로 다른 과거 이력이15개를 넘으면 가장 오래된 로컬 이력을 같은 방식으로 정리한다. NFS 인증 기록은 과거 checkpoint의 내용 증거로 유지한다. 마지막 정상 상태를 `retention_state.json`에 기록한다.

전체 SHA는 복사 목적지 검증과 동일 실행의 인증을 재사용한다. 강제 종료 후에는 재시작 시 남은 삭제 transaction과 필요한 사본을 fresh 검증한다. 오래된 NFS checkpoint 삭제 때문에 공유 worker 학습이 멈추지 않도록 인증/보존 작업은 감독기의 별도 단일 queue에서 처리한다. 메모리·디스크 감독은 계속되며 실패 시 그 새 main worker만 중단한다.

## 공간 예산과 실제 사용량

3B 실제 checkpoint38,666,917,916bytes를 기준으로 한다. 아래 peak는 payload에10% 여유와 run/인증 메타데이터1GiB를 더한 값이다.

| 구분 | 예약/상한 |
|---|---:|
| NFS: 최신10개 + 작성중1개 | **436.74GiB** |
| 로컬: 과거15개 + 복사중1개 | **634.80GiB** |
| 별도 로컬 임시 작업 | **1,024GiB** |
| 안전 여유 | **각 파일시스템200GiB** |

이번 재측정은 NFS free **1,101.28GiB**, 로컬 free **1,960.31GiB**다. 실제 실험 사본 **59.01GiB**는 그대로 보존했고 이미 로컬 free에서 차감됐다. 현재 F8의 아직 쓰이지 않은 checkpoint forecast와 후속 NFS8+1+1GiB만 합산한 미래 사용량 **128.44GiB**를 추가 확보하면, 위 모든 예약 후 NFS **336.10GiB**, 로컬 **101.51GiB**가 더 남는다. CPU/GPU 검증의 휘발성 작업은 로컬1TiB 임시 작업 allowance의 사용처로 두며 이를 같은 작업의 추가 peak로 중복 합산하지 않는다. 본학습 직전 새 실측이 필요하다.

이는 전체25개를 실제 크기로 채운 실측이 아니다. 실제 시험에서 보존한 크기는1B2개+3B1개 **63,361,645,755bytes/59.01GiB**, 실제3B 한 사본 복사+검증 **16.732초**다. CPU 전체 상태 대조 peak RSS **72.99GiB**, 별도 guard의 sampled PSS **72.66GiB**였으며128GiB 한도 이내였다. 생산 copy worker8/hash8, 8MiB 읽기 block을 선택했다. 실제1B 한 번씩의 worker4/8 총시간 **6.489/5.491초** 비교이며 보편적 최적값을 주장하지 않는다.

## 실패·복구와 검증 범위

- 복사 실패/공간 부족: 기존 NFS 사본과 실패 staging을 보존하고 중단한다. 임의로 다른 자료를 삭제하지 않는다. 잔여물도 해당 run의 byte budget에 포함한다.
- 발행 후 인증 파일 기록 전 종료: 아직 NFS source가 있으면 전체 검증 후 인증을 복원한다. 실행 도중 `os._exit(91)`로 실제 프로세스를 끝낸 뒤 새 프로세스에서 lock 회수/재시작을 검증했다.
- 정리 intent/rename/파일 삭제 중 종료: fsync된 intent와 원 인증으로 남은 정확한 파일만 확인하고 이어간다. 새 사본을 생성하며 실패 잔여물을 무한히 쌓지 않는다.
- 로컬 전체 유실: 추가 이력은 사라진 것으로 처리하며 NFS 최신10개로 재개한다. 로컬 사본을 NFS 재개 후보로 선택하지 않는다.
- **동시에 로컬이 유실되고 오래된 NFS 사본 삭제가 일부 진행된 경우**: NFS 최신10개를 보존하고 미완료 trash 정리를 중단한다. 원래 로컬 사본이 있다고 꾸며 계속하지 않으며 해당 새 run의 잔여물에 대한 별도 검토가 필요하다.
- 감독기 중단/자원 위반: 보존 thread에 취소를 전달하고 해당 main worker만 중단한다. 미완료 출력은 보존한다. 같은 감독기 명령으로 재시작하기 전에 기존 worker가 남아 있는지 확인한다.

검증 증거:

- `runs/tiered_retention_validation_v2/`: CPU DCP fixture의10+15/최대25개, copy 손상/누락·실패·중단/재시작, 실제 프로세스 강제 종료, 로컬 유실, NFS·로컬 공간 부족, 소유자/링크/목록 변조, 취소, 시작 전 자체정리 거부. CPU 실제 AdamW는 복원 후 다음 loss/model/optimizer update가 exact다.
- `runs/tiered_retention_copy_v1/`: 실제 NFS3B 전체 DCP를 로컬에 복사·SHA 검증한 후 양쪽에서 CPU로 읽은 **model255+optimizer765=1,020 tensors exact**, 저장된8rank CPU/CUDA RNG exact. source 원본은 변경하지 않았다.
- 실제 데이터의 기존8GPU 수치·재개 증거는 F7에 별도로 있다. **새 보존 정책으로 장기8GPU 본학습을 돌리거나 실제 크기25개를 채워 운영한 검증은 아직 없다.** CPU 전체 상태 복원을 그 검증으로 승격하지 않는다.

구현: `scripts/tiered_checkpoint_retention.py`, `scripts/supervise_main_training.py`. 실측 도구: `scripts/validate_tiered_retention_copy.py`. 외부 업로드·원본 재다운로드·추가 자료 삭제는 없다. 공식 벤치마크 별도 방침과 F8→후속 실증→F9 최종검토 순서를 유지한다.
