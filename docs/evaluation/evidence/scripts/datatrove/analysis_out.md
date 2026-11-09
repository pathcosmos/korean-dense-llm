
## (a) gold100

Counts by source: {'fineweb2_ko': 40, 'hplt_ko': 40, 'wiki_ko': 20}

| label class | n | repo rejects | FW2 rejects | both | repo only | FW2 only | neither |
|---|---|---|---|---|---|---|---|
| keep | 67 | 2 | 7 | 1 | 1 | 6 | 59 |
| short | 6 | 6 | 6 | 6 | 0 | 0 | 0 |
| semantic_low_quality | 27 | 3 | 10 | 3 | 0 | 7 | 17 |
| ALL | 100 | 11 | 23 | 10 | 1 | 13 | 76 |

Filter-vs-label metrics (positive = label reject):

| filter | scope | TP | FP | FN | TN | precision | recall |
|---|---|---|---|---|---|---|---|
| repo check_ko | all reject labels | 9 | 2 | 24 | 65 | 0.82 | 0.27 |
| FW2 reconstruction | all reject labels | 16 | 7 | 17 | 60 | 0.70 | 0.48 |
| union (either) | all reject labels | 16 | 8 | 17 | 59 | 0.67 | 0.48 |
| repo check_ko | keep + semantic_low_quality only | 3 | 2 | 24 | 65 | 0.60 | 0.11 |
| FW2 reconstruction | keep + semantic_low_quality only | 10 | 7 | 17 | 60 | 0.59 | 0.37 |
| union (either) | keep + semantic_low_quality only | 10 | 8 | 17 | 59 | 0.56 | 0.37 |
| repo check_ko | keep + short only | 6 | 2 | 0 | 65 | 0.75 | 1.00 |
| FW2 reconstruction | keep + short only | 6 | 7 | 0 | 60 | 0.46 | 1.00 |
| union (either) | keep + short only | 6 | 8 | 0 | 59 | 0.43 | 1.00 |

Reasons by label class (repo | FW2 first):

keep:
- repo=-, FW2=-: 59
- repo=-, FW2=fineweb_qual/line_punct_ratio: 4
- repo=-, FW2=gopher_qual/gopher_below_alpha_threshold: 2
- repo=low_hangul_ratio, FW2=gopher_qual/gopher_below_alpha_threshold: 1
- repo=dup_lines, FW2=-: 1

short:
- repo=short, FW2=gopher_rep/top_4_gram: 2
- repo=short, FW2=gopher_qual/gopher_short_doc: 2
- repo=short, FW2=gopher_qual/gopher_below_alpha_threshold: 1
- repo=short, FW2=fineweb_qual/list_ratio: 1

semantic_low_quality:
- repo=-, FW2=-: 17
- repo=-, FW2=fineweb_qual/line_punct_ratio: 3
- repo=-, FW2=gopher_qual/gopher_too_many_end_ellipsis: 2
- repo=dup_lines, FW2=fineweb_qual/line_punct_ratio: 1
- repo=dup_lines, FW2=gopher_qual/gopher_below_alpha_threshold: 1
- repo=-, FW2=gopher_rep/duplicated_5_n_grams: 1
- repo=ngram_rep, FW2=gopher_rep/duplicated_5_n_grams: 1
- repo=-, FW2=gopher_rep/duplicated_10_n_grams: 1

By source (label, repo reject, FW2 reject):
- fineweb2_ko: n=40 label-reject=9 repo-reject=0 FW2-reject=0
- hplt_ko: n=40 label-reject=18 repo-reject=5 FW2-reject=14
- wiki_ko: n=20 label-reject=6 repo-reject=6 FW2-reject=9

Paired recall on label=reject (Wilson 95% CI), discordant counts, exact McNemar p:

| class | n | repo recall | FW2 recall | repo-only catches | FW2-only catches | McNemar p |
|---|---|---|---|---|---|---|
| short | 6 | 6/6 (0.61-1.00) | 6/6 (0.61-1.00) | 0 | 0 | 1.000 |
| semantic_low_quality | 27 | 3/27 (0.04-0.28) | 10/27 (0.22-0.56) | 0 | 7 | 0.016 |
| all reject | 33 | 9/33 (0.15-0.44) | 16/33 (0.33-0.65) | 0 | 7 | 0.016 |
## (b) cross tables (rows: repo check_ko, cols: FW2 reconstruction)

### fineweb2_ko: 20000 docs, 20 row groups

| | FW2 pass | FW2 reject | total |
|---|---|---|---|
| repo pass | 19204 | 49 | 19253 |
| repo reject | 737 | 10 | 747 |

| metric | rate | 95% CI (group bootstrap) |
|---|---|---|
| both_reject | 0.1% | 0.0% - 0.1% |
| both_pass | 96.0% | 95.6% - 96.4% |
| repo_only | 3.7% | 3.3% - 4.1% |
| fw2_only | 0.2% | 0.2% - 0.3% |
| repo_reject | 3.7% | 3.4% - 4.1% |
| fw2_reject | 0.3% | 0.2% - 0.4% |

Reasons, repo only rejected (n=737; median len 178, median hangul ratio 0.81):

| reason | n | share |
|---|---|---|
| short | 506 | 68.7% |
| dup_lines | 180 | 24.4% |
| low_hangul_ratio | 44 | 6.0% |
| ngram_rep | 4 | 0.5% |
| bad_encoding | 3 | 0.4% |

Reasons, FW2 only rejected (n=49; median len 689, median hangul ratio 0.75):

| reason | n | share |
|---|---|---|
| gopher_qual/gopher_below_alpha_threshold | 21 | 42.9% |
| fineweb_qual/list_ratio | 13 | 26.5% |
| gopher_rep/duplicated_5_n_grams | 5 | 10.2% |
| gopher_rep/duplicated_7_n_grams | 3 | 6.1% |
| gopher_qual/gopher_enough_stop_words | 2 | 4.1% |
| gopher_rep/top_4_gram | 1 | 2.0% |
| gopher_rep/top_3_gram | 1 | 2.0% |
| gopher_rep/duplicated_10_n_grams | 1 | 2.0% |
| gopher_rep/duplicated_8_n_grams | 1 | 2.0% |
| gopher_rep/duplicated_6_n_grams | 1 | 2.0% |

Both reject: top reason pairs (repo | FW2 first):

- short | gopher_qual/gopher_short_doc: 3
- short | gopher_qual/gopher_below_alpha_threshold: 2
- dup_lines | gopher_qual/gopher_below_alpha_threshold: 1
- dup_lines | fineweb_qual/char_dup_ratio: 1
- short | gopher_rep/duplicated_6_n_grams: 1
- short | gopher_rep/top_4_gram: 1
- short | fineweb_qual/list_ratio: 1

FW2 per-stage standalone reject rate (each stage evaluated on every doc, not only survivors):
- gopher_rep: 0.1%
- fineweb_qual: 0.1%
- gopher_qual: 0.1%

### hplt_ko: 20000 docs, 20 row groups

| | FW2 pass | FW2 reject | total |
|---|---|---|---|
| repo pass | 13839 | 4622 | 18461 |
| repo reject | 146 | 1393 | 1539 |

| metric | rate | 95% CI (group bootstrap) |
|---|---|---|
| both_reject | 7.0% | 5.8% - 8.1% |
| both_pass | 69.2% | 61.0% - 74.5% |
| repo_only | 0.7% | 0.5% - 1.0% |
| fw2_only | 23.1% | 17.7% - 31.9% |
| repo_reject | 7.7% | 6.4% - 8.9% |
| fw2_reject | 30.1% | 24.8% - 38.3% |

Reasons, repo only rejected (n=146; median len 1938, median hangul ratio 0.75):

| reason | n | share |
|---|---|---|
| dup_lines | 120 | 82.2% |
| low_hangul_ratio | 21 | 14.4% |
| bad_encoding | 4 | 2.7% |
| ngram_rep | 1 | 0.7% |

Reasons, FW2 only rejected (n=4622; median len 1813, median hangul ratio 0.78):

| reason | n | share |
|---|---|---|
| fineweb_qual/line_punct_ratio | 1543 | 33.4% |
| gopher_qual/gopher_below_alpha_threshold | 1111 | 24.0% |
| gopher_qual/gopher_too_many_end_ellipsis | 452 | 9.8% |
| gopher_rep/duplicated_5_n_grams | 435 | 9.4% |
| gopher_rep/duplicated_9_n_grams | 195 | 4.2% |
| fineweb_qual/char_dup_ratio | 172 | 3.7% |
| gopher_rep/duplicated_10_n_grams | 157 | 3.4% |
| gopher_rep/duplicated_8_n_grams | 122 | 2.6% |
| gopher_rep/duplicated_6_n_grams | 110 | 2.4% |
| gopher_rep/duplicated_7_n_grams | 107 | 2.3% |
| gopher_rep/top_4_gram | 87 | 1.9% |
| gopher_qual/gopher_too_many_bullets | 79 | 1.7% |
| gopher_rep/top_3_gram | 23 | 0.5% |
| gopher_qual/gopher_too_many_hashes | 12 | 0.3% |
| fineweb_qual/list_ratio | 9 | 0.2% |
| gopher_rep/top_2_gram | 4 | 0.1% |
| gopher_qual/gopher_too_many_ellipsis | 3 | 0.1% |
| gopher_qual/gopher_enough_stop_words | 1 | 0.0% |

Both reject: top reason pairs (repo | FW2 first):

- dup_lines | gopher_rep/dup_line_frac: 627
- ngram_rep | gopher_rep/duplicated_5_n_grams: 343
- dup_lines | gopher_rep/duplicated_5_n_grams: 107
- dup_lines | fineweb_qual/line_punct_ratio: 73
- dup_lines | gopher_qual/gopher_below_alpha_threshold: 37
- ngram_rep | gopher_rep/top_4_gram: 30
- dup_lines | fineweb_qual/char_dup_ratio: 19
- low_hangul_ratio | gopher_rep/duplicated_5_n_grams: 19

FW2 per-stage standalone reject rate (each stage evaluated on every doc, not only survivors):
- gopher_rep: 12.4%
- fineweb_qual: 18.0%
- gopher_qual: 15.5%

### wiki_ko: 20000 docs, 20 row groups

| | FW2 pass | FW2 reject | total |
|---|---|---|---|
| repo pass | 8353 | 5716 | 14069 |
| repo reject | 948 | 4983 | 5931 |

| metric | rate | 95% CI (group bootstrap) |
|---|---|---|
| both_reject | 24.9% | 18.9% - 31.7% |
| both_pass | 41.8% | 35.9% - 48.0% |
| repo_only | 4.7% | 3.9% - 5.6% |
| fw2_only | 28.6% | 25.0% - 32.0% |
| repo_reject | 29.7% | 23.3% - 36.4% |
| fw2_reject | 53.5% | 47.5% - 59.6% |

Reasons, repo only rejected (n=948; median len 166, median hangul ratio 0.81):

| reason | n | share |
|---|---|---|
| short | 912 | 96.2% |
| low_hangul_ratio | 33 | 3.5% |
| few_hangul | 1 | 0.1% |
| long | 1 | 0.1% |
| dup_lines | 1 | 0.1% |

Reasons, FW2 only rejected (n=5716; median len 468, median hangul ratio 0.74):

| reason | n | share |
|---|---|---|
| fineweb_qual/line_punct_ratio | 2918 | 51.0% |
| gopher_qual/gopher_below_alpha_threshold | 1969 | 34.4% |
| fineweb_qual/list_ratio | 524 | 9.2% |
| gopher_rep/top_4_gram | 96 | 1.7% |
| gopher_rep/duplicated_5_n_grams | 89 | 1.6% |
| gopher_rep/top_3_gram | 42 | 0.7% |
| gopher_rep/top_2_gram | 22 | 0.4% |
| fineweb_qual/char_dup_ratio | 17 | 0.3% |
| gopher_rep/dup_line_frac | 8 | 0.1% |
| gopher_rep/duplicated_9_n_grams | 8 | 0.1% |
| gopher_rep/duplicated_6_n_grams | 6 | 0.1% |
| gopher_qual/gopher_short_doc | 5 | 0.1% |
| gopher_rep/duplicated_10_n_grams | 5 | 0.1% |
| gopher_rep/duplicated_7_n_grams | 4 | 0.1% |
| gopher_rep/duplicated_8_n_grams | 3 | 0.1% |

Both reject: top reason pairs (repo | FW2 first):

- short | fineweb_qual/list_ratio: 1513
- short | gopher_qual/gopher_short_doc: 1401
- short | gopher_rep/top_4_gram: 726
- short | gopher_rep/top_3_gram: 517
- short | gopher_qual/gopher_below_alpha_threshold: 307
- short | gopher_rep/top_2_gram: 148
- short | fineweb_qual/line_punct_ratio: 100
- low_hangul_ratio | gopher_qual/gopher_below_alpha_threshold: 60

FW2 per-stage standalone reject rate (each stage evaluated on every doc, not only survivors):
- gopher_rep: 9.0%
- fineweb_qual: 29.4%
- gopher_qual: 40.3%

fineweb2_ko language_score >= 0.842: 20000/20000 (min 0.848)
