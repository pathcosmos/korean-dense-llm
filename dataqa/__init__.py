"""T0 data-processing QA harness: gold-set based measurement of filter and PII rules.

New, self-contained module. It never edits data/ or common/ and never promotes artifacts.
Rule copies in this package (parameterised check_ko, PII candidates) are measurement
instruments and are guarded by parity gates against the production implementation in
data/filters.py. See dataqa/README.md.
"""
