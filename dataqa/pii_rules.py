"""PII detection as measurement instrument: production modes plus harness-only candidates.

Production `data.filters.mask_pii` applies EMAIL, then RRN, then PHONE, each on the output of the
previous step. `apply` reproduces that order and maps every match back to offsets in the ORIGINAL text,
so spans from different steps never double count. Matching comes from `data.filters.find_pii`, so
production modes (phone=intl, rrn=date_checked, email=no_git) are measured on the real code path.
`parity_mismatches` asserts that `apply(...)` masks exactly like `mask_pii(text, policy)`.

Harness-only candidates (NOT in production) encode factual claims that are not verified; they are
marked UNCERTAIN (불확실) and must be checked against an official source before adoption:
  * `intl_prefix`: area-code / mobile-prefix allowlist from the author's general knowledge of the
    Korean numbering plan, not checked against an official plan.
  * `range_date`, `range_date_sep`: accept 7th digit 1-8 on the assumption that 5-8 are foreign-resident
    registration numbers. UNCERTAIN; not verified. Birth date is only month 01-12 / day 01-31.
  * An RRN check digit is deliberately NOT implemented: whether it still holds for numbers issued after
    the 2020 numbering change is UNCERTAIN, and a wrong validator would silently drop real PII.
"""
import re
from dataclasses import dataclass, asdict

from data import filters as F

MASKS = {'email': '<email>', 'rrn': '<id>', 'phone': '<phone>'}
IMPLEMENTED_TYPES = ('phone', 'rrn', 'alien_reg', 'email')
FAMILY = {'phone': ('phone',), 'rrn': ('rrn', 'alien_reg'), 'email': ('email',)}

PHONE_PREFIX = re.compile(r'^(?:01[016789]|02|0[3-6][1-5]|070|080|050)')
RRN_WIDE = re.compile(r'(?<!\d)(\d{2})(\d{2})(\d{2})([- ]?)([1-8])\d{6}(?!\d)')
PRODUCTION_MODES = {'phone': ('baseline', 'intl'), 'rrn': ('baseline', 'date_checked'), 'email': ('baseline', 'no_git')}
LOCAL_MODES = {'phone': ('intl_prefix',), 'rrn': ('range_date', 'range_date_sep'), 'email': ()}


@dataclass(frozen=True)
class PiiConfig:
    phone: str = 'baseline'   # baseline | intl | intl_prefix(harness-only)
    rrn: str = 'baseline'     # baseline | date_checked | range_date | range_date_sep (last two harness-only)
    email: str = 'baseline'   # baseline | no_git

    def as_dict(self):
        return asdict(self)

    def production_policy(self):
        """The equivalent data.filters.PiiPolicy, or None if a harness-only mode is selected."""
        if any(getattr(self, k) in LOCAL_MODES[k] for k in LOCAL_MODES):
            return None
        return F.PiiPolicy(phone=self.phone, rrn=self.rrn, email=self.email)


PRESETS = {
    'baseline': PiiConfig(),
    'phone_intl': PiiConfig(phone='intl'),
    'phone_intl_prefix': PiiConfig(phone='intl_prefix'),
    'rrn_date_checked': PiiConfig(rrn='date_checked'),
    'rrn_range_date': PiiConfig(rrn='range_date'),
    'rrn_range_date_sep': PiiConfig(rrn='range_date_sep'),
    'email_no_git': PiiConfig(email='no_git'),
    'all_candidates_loose': PiiConfig('intl_prefix', 'range_date', 'no_git'),
    'all_candidates_strict': PiiConfig('intl_prefix', 'range_date_sep', 'no_git'),
    'production_safe': PiiConfig('intl', 'date_checked', 'no_git'),
}


def _email(text, mode):
    return F.find_pii(text, 'email', mode)


def _rrn(text, mode):
    if mode in PRODUCTION_MODES['rrn']:
        return F.find_pii(text, 'rrn', mode)
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
    if mode in PRODUCTION_MODES['phone']:
        return F.find_pii(text, 'phone', mode)
    out = []
    for s, e in F.find_pii(text, 'phone', 'intl'):
        raw = text[s:e]
        digits = re.sub(r'\D', '', raw)
        if raw.lstrip().startswith(('+82', '82')):
            digits = digits[2:]
            digits = digits if digits.startswith('0') else '0' + digits
        if PHONE_PREFIX.match(digits):
            out.append((s, e))
    return out


def apply(text, cfg=PiiConfig()):
    """Return (masked_text, detections). detections: [{type,start,end}] in ORIGINAL offsets."""
    for name in ('phone', 'rrn', 'email'):
        allowed = PRODUCTION_MODES[name] + LOCAL_MODES[name]
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
    """Indices where apply() masks differently from production mask_pii (must be empty).

    Checked for the baseline and for every preset whose modes all exist in production."""
    bad = []
    for i, t in enumerate(texts):
        if apply(t, PRESETS['baseline'])[0] != F.mask_pii(t):
            bad.append(i)
            continue
        for cfg in PRESETS.values():
            policy = cfg.production_policy()
            if policy is not None and apply(t, cfg)[0] != F.mask_pii(t, policy):
                bad.append(i)
                break
    return bad
