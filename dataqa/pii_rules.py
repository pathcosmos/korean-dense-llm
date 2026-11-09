"""PII detection as measurement instrument: production baseline + candidate variants.

Production `data.filters.mask_pii` applies EMAIL, then RRN, then PHONE, each on the output of
the previous step. `apply` reproduces exactly that order and maps every match back to offsets in
the ORIGINAL text, so spans from different steps never double count. The baseline config must
reproduce `mask_pii(text)` byte for byte (`parity_mismatches`); tests and the CLI enforce it.

Candidate variants are NOT proposals to change production. They exist so the gold set can show
what a change would trade. Several encode factual claims about Korean identifiers; those are
marked UNCERTAIN (불확실) and must be verified against an official source before adoption:
  * area-code / mobile-prefix allowlist (`intl_prefix`): written from the author's general
    knowledge of the Korean numbering plan, not checked against an official plan.
  * resident-registration digit ranges: the baseline accepts 7th digit 1-4. The candidates
    accept 1-8 on the assumption that 5-8 are foreign-resident registration numbers.
    UNCERTAIN; not verified. Birth-date validity is only month 01-12 / day 01-31.
  * RRN check digit is deliberately NOT implemented: whether it still holds for numbers
    issued after the 2020 numbering change is UNCERTAIN, and a wrong validator would silently
    drop real PII.
"""
import re
from dataclasses import dataclass, asdict

from data import filters as F

MASKS = {'email': '<email>', 'rrn': '<id>', 'phone': '<phone>'}
IMPLEMENTED_TYPES = ('phone', 'rrn', 'alien_reg', 'email')
FAMILY = {'phone': ('phone',), 'rrn': ('rrn', 'alien_reg'), 'email': ('email',)}

PHONE_INTL = re.compile(
    r'(?<!\d)(?:\+82[- ]?\(?0?\)?[- ]?|82[- ]?0|0)\d{1,2}[- ]?\d{3,4}[- ]?\d{4}(?!\d)'
    r'|(?<!\d)\(0\d{1,2}\)\s?\d{3,4}[- ]?\d{4}(?!\d)')
PHONE_PREFIX = re.compile(r'^(?:01[016789]|02|0[3-6][1-5]|070|080|050)')
RRN_WIDE = re.compile(r'(?<!\d)(\d{2})(\d{2})(\d{2})([- ]?)([1-8])\d{6}(?!\d)')


@dataclass(frozen=True)
class PiiConfig:
    phone: str = 'baseline'   # baseline | intl | intl_prefix
    rrn: str = 'baseline'     # baseline | range_date | range_date_sep
    email: str = 'baseline'   # baseline | no_git

    def as_dict(self):
        return asdict(self)


PRESETS = {
    'baseline': PiiConfig(),
    'phone_intl': PiiConfig(phone='intl'),
    'phone_intl_prefix': PiiConfig(phone='intl_prefix'),
    'rrn_range_date': PiiConfig(rrn='range_date'),
    'rrn_range_date_sep': PiiConfig(rrn='range_date_sep'),
    'email_no_git': PiiConfig(email='no_git'),
    'all_candidates_loose': PiiConfig('intl_prefix', 'range_date', 'no_git'),
    'all_candidates_strict': PiiConfig('intl_prefix', 'range_date_sep', 'no_git'),
}


def _email(text, mode):
    out = []
    for m in F.EMAIL.finditer(text):
        if mode == 'no_git' and text.startswith('git@', m.start()) and text[m.end():m.end() + 1] == ':':
            continue  # ssh remote such as git@host:org/repo.git, not a mailbox
        out.append((m.start(), m.end()))
    return out


def _rrn(text, mode):
    if mode == 'baseline':
        return [(m.start(), m.end()) for m in F.RRN.finditer(text)]
    out = []
    for m in RRN_WIDE.finditer(text):
        month, day = int(m.group(2)), int(m.group(3))
        if not (1 <= month <= 12 and 1 <= day <= 31):
            continue
        if mode == 'range_date_sep' and not m.group(4):
            continue
        out.append((m.start(), m.end()))
    return out


def _phone(text, mode):
    if mode == 'baseline':
        return [(m.start(), m.end()) for m in F.PHONE.finditer(text)]
    out = []
    for m in PHONE_INTL.finditer(text):
        if mode == 'intl_prefix':
            digits = re.sub(r'\D', '', m.group())
            if m.group().lstrip().startswith(('+82', '82')):
                digits = digits[2:]
                digits = digits if digits.startswith('0') else '0' + digits
            if not PHONE_PREFIX.match(digits):
                continue
        out.append((m.start(), m.end()))
    return out


def apply(text, cfg=PiiConfig()):
    """Return (masked_text, detections). detections: [{type,start,end}] in ORIGINAL offsets."""
    for name, allowed in (('phone', ('baseline', 'intl', 'intl_prefix')),
                          ('rrn', ('baseline', 'range_date', 'range_date_sep')),
                          ('email', ('baseline', 'no_git'))):
        if getattr(cfg, name) not in allowed:
            raise ValueError(f'{name} must be one of {allowed}')
    cur = text
    origin = list(range(len(text)))          # current index -> original index, -1 inside a mask
    detections = []
    for kind, finder, mode in (('email', _email, cfg.email), ('rrn', _rrn, cfg.rrn), ('phone', _phone, cfg.phone)):
        spans = finder(cur, mode)
        if not spans:
            continue
        pieces, new_origin, pos = [], [], 0
        for s, e in spans:
            real = [o for o in origin[s:e] if o >= 0]
            if len(real) != e - s:
                raise RuntimeError('match overlaps an earlier mask token')
            detections.append(dict(type=kind, start=real[0], end=real[-1] + 1))
            pieces.append(cur[pos:s]); new_origin += origin[pos:s]
            pieces.append(MASKS[kind]); new_origin += [-1] * len(MASKS[kind])
            pos = e
        pieces.append(cur[pos:]); new_origin += origin[pos:]
        cur, origin = ''.join(pieces), new_origin
    detections.sort(key=lambda d: (d['start'], d['end']))
    return cur, detections


def parity_mismatches(texts):
    """Indices where baseline apply() differs from production mask_pii (must be empty)."""
    return [i for i, t in enumerate(texts) if apply(t, PRESETS['baseline'])[0] != F.mask_pii(t)]
