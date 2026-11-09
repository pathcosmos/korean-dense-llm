"""check_ko with explicit thresholds: a thin wrapper over the production policy object.

Since data/filters.py exposes KoPolicy, the harness no longer keeps a copy of the rule logic, so there
is no copy to drift. What remains to guard is the harness' own entry point against the legacy
`check_ko(t, min_chars=, max_chars=)` entry point (`parity_mismatches`); regression protection of the
defaults themselves is tests/golden/filters_oracle_v1.json.

The per-document sequence of data/clean.py is: norm_ws -> check -> exact_hash dedup -> mask_pii.
`filter_reason` reproduces norm_ws -> check (dedup is cross-document and not measured here).
"""
from dataclasses import replace

from data import filters as F

KoParams = F.KoPolicy
DEFAULT = F.KO_DEFAULT


def check_ko_param(t, p=DEFAULT):
    return F.check_ko(t, policy=p)


def filter_reason(text, p=DEFAULT):
    """norm_ws -> check, the first two per-document steps of data/clean.py."""
    return check_ko_param(F.norm_ws(text), p)


def parity_mismatches(texts, p=DEFAULT):
    """[(index, legacy_reason, harness_reason)] where the two entry points disagree.

    Only defined for production-equivalent parameters (defaults plus the two length knobs the legacy
    signature supports)."""
    only_len = replace(DEFAULT, min_chars=p.min_chars, max_chars=p.max_chars)
    if p != only_len:
        raise ValueError('parity gate only applies to production-equivalent parameters')
    bad = []
    for i, t in enumerate(texts):
        legacy = F.check_ko(t, min_chars=p.min_chars, max_chars=p.max_chars)
        mine = check_ko_param(t, p)
        if legacy != mine:
            bad.append((i, legacy, mine))
    return bad
