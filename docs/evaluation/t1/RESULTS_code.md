seed 20261109 · 단위: row group (cluster); CIs bootstrap over groups

| 항목 | github_code_v1 (276,625문서/30그룹) |
|---|---|
| NFC가 아닌 원문 | 0.04% [0.03, 0.05] (n=108) |
| 분해형(conjoining) 자모 포함 | 0.00% [0.00, 0.00] (n=4) |
| 특수 토큰 문자열 포함 | 0 |
| <|eos|> 문자열 포함 | 0 |
| 문서 길이 중앙값 / p90 (자) | 2276 / 15798 |

"0"은 표본에서 한 건도 관찰되지 않았다는 뜻이다(전체에 없다는 보증이 아니다). 군집 표본에서 0건 관측의 부트스트랩 구간은 퇴화하므로, row group 단위 상한은 "관측 0 / G개 그룹"에 대한 rule of three로 읽는다: 95% 상한 ≈ 3/G (그룹 60개면 5%, 40개면 7.5%, 30개면 10%).

- shards = first N of the sorted listing, N from the registry; equivalence to the original selection is an assumption (불확실)
- document-level rates describe the sampled shards only, not the full corpus
- text is never stored; only counts

