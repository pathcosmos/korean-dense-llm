"""Deterministic synthetic seed gold set (seed_v1).

IMPORTANT: the author writes both each case and its expected label, so results on this set
demonstrate that the harness works and that rules behave as designed. They are NOT evidence
about real-corpus precision/recall and must not be used to pick thresholds. Cases tagged
'boundary_by_construction' sit on a current threshold on purpose.

All identifiers are fictional: example.* mail domains, RFC 5737 documentation IPs, obviously
dummy phone/registration digits. Labels that encode a policy choice carry `policy_assumption`
so they can be flipped per run (dataqa.gold.apply_policy).

Usage: python -m dataqa.seed_gold --out dataqa/gold/seed_v1.jsonl
"""
import argparse
import random
import unicodedata

from data import filters as F
from dataqa import gold

SEED = 20261109
PLACES = ['서울', '부산', '대전', '광주', '인천', '울산', '수원', '창원', '청주', '전주',
          '포항', '제주', '춘천', '강릉', '원주', '안동', '여수', '순천', '목포', '군산']
FOODS = ['김치찌개', '비빔밥', '불고기', '칼국수', '떡볶이', '순두부찌개', '냉면', '삼겹살', '된장국', '잡채']
SEASONS = ['봄', '여름', '가을', '겨울']
TEMPLATES = [
    '{e}에는 {a}에서 {b}까지 기차를 타고 이동하며 창밖 풍경을 오래 바라보았습니다.',
    '{a}의 시장에서 맛본 {c} 한 그릇은 재료가 신선해서 오래도록 기억에 남았습니다.',
    '이번 여행은 {d}일 동안 이어졌고, 매일 저녁마다 그날의 일정을 꼼꼼히 정리했습니다.',
    '{b}에 사는 친구가 근처 식당을 추천해 주어서 {c} 한 상을 함께 나누어 먹었습니다.',
    '돌아오는 길에는 {a} 근처를 지나며 다음에는 더 여유롭게 둘러보자고 다짐했습니다.',
    '{e} 아침의 공기는 맑고 차가워서 산책하기에 알맞았고 발걸음도 가벼웠습니다.',
    '여행 경비는 교통비와 식비를 합쳐 대략 {d}0만 원이 들었으며 예상보다 적게 쓴 편이었습니다.',
    '숙소 주인은 {a}의 역사에 대해 설명해 주었고 우리는 메모를 하며 귀를 기울였습니다.',
    '{b}의 박물관에서는 오래된 도자기와 옛 지도를 구경하며 시간 가는 줄 몰랐습니다.',
    '사진을 정리하다 보니 {e}의 {a}에서 찍은 한 장이 유난히 마음에 들었습니다.',
    '다음 계획으로는 {b}에서 하룻밤을 묵고 이튿날 {a} 방향으로 돌아오는 일정을 세워 두었습니다.',
    '이야기를 나누다 보니 어느새 해가 저물었고 거리의 가로등이 하나둘 켜지기 시작했습니다.',
]
TECH = ['cache', 'thread', 'pipeline', 'batch', 'tokenizer', 'parser', 'scheduler', 'buffer']
TECH_TEMPLATES = [
    '이번 주에는 {w1} 모듈과 {w2} 모듈의 동작을 비교하고 결과를 문서로 남겼습니다.',
    '{w1} 설정을 바꾸었더니 처리 속도가 눈에 띄게 달라져서 팀원들과 원인을 함께 살펴보았습니다.',
    '{w2} 쪽에서 발생한 오류는 입력 형식이 맞지 않아서였고 검증 단계를 앞에 추가해 해결했습니다.',
    '다음 회의에서는 {w1} 구성의 장단점을 정리해 공유하고 개선 방향을 정하기로 했습니다.',
    '문서에는 {w2} 사용 예시와 주의할 점을 적어 두어 처음 보는 사람도 따라 할 수 있게 했습니다.',
]
EN_SENTENCES = [
    'The meeting was moved to Thursday because the main speaker was traveling abroad.',
    'Please review the attached report and send your comments before the end of the week.',
    'Our team shipped a new release that improves startup time and reduces memory usage.',
    'The library opens at nine in the morning and closes at six in the evening on weekdays.',
    'Several customers asked whether the warranty covers accidental damage to the screen.',
    'He spent the afternoon repairing the old bicycle he found in the garage last summer.',
    'The committee will announce the final schedule once all the departments have replied.',
    'A detailed changelog is available on the project page along with upgrade instructions.',
]
POEMS = [
    (['바람이 불어오는 언덕 위에서', '나는 오래된 편지를 다시 읽었네', '구름은 천천히 서쪽으로 흘러가고',
      '강물은 말없이 제 길을 찾아가네', '저녁 빛이 들판을 붉게 물들이면', '멀리서 들려오는 종소리 하나',
      '어린 날의 약속이 문득 떠올라', '나는 가만히 눈을 감고 서 있었네'], '그래도 우리는 다시 만나리라', 5),
    (['새벽 시장의 골목은 아직 어두운데', '따뜻한 국물 냄새가 길 끝에서 번지고', '상인들은 말없이 손을 놀리며',
      '하루의 첫 장사를 준비하고 있었네', '비 갠 하늘 사이로 별이 하나 남아', '지나는 이의 어깨를 조용히 비추면',
      '낡은 수레바퀴 소리가 멀어져 가고', '나는 오래도록 그 자리에 머물렀네'], '내일도 이 길 위에서 만나자', 6),
]


def slots(rng):
    a, b = rng.sample(PLACES, 2)
    return dict(a=a, b=b, c=rng.choice(FOODS), d=rng.randint(2, 9), e=rng.choice(SEASONS))


def prose(rng, min_chars, sep=' '):
    sents, order = [], list(range(len(TEMPLATES)))
    while True:
        rng.shuffle(order)
        for i in order:
            sents.append(TEMPLATES[i].format(**slots(rng)))
            if len(sep.join(sents)) >= min_chars:
                return sep.join(sents)


def exact_len(rng, n):
    s = prose(rng, n + 60)[:n]
    return s[:-1] + '다' if s[-1].isspace() else s


def tech_prose(rng, min_chars):
    """Each tech template at most once (few templates repeat n-grams); top up with plain prose."""
    out, order = [], list(range(len(TECH_TEMPLATES)))
    rng.shuffle(order)
    for i in order:
        w1, w2 = rng.sample(TECH, 2)
        out.append(TECH_TEMPLATES[i].format(w1=w1, w2=w2))
        if len(' '.join(out)) >= min_chars:
            return ' '.join(out)
    return ' '.join(out) + ' ' + prose(rng, min_chars - len(' '.join(out)))


def hangul_ratio(t):
    nonspace = sum(1 for c in t if not c.isspace())
    return len(F.HANGUL.findall(t)) / nonspace


def rec(rid, text, label, reason=None, tags=(), policy=None, policy_reason=None, spans=(), ambiguous=False):
    r = dict(id=rid, text=text, label=label, reject_reason=reason, source_kind='synthetic',
             label_source='author_synthetic', tags=list(tags), pii_spans=list(spans))
    if policy:
        r['policy_assumption'] = policy
        r['policy_reject_reason'] = policy_reason
    if ambiguous:
        r['ambiguous'] = True
    return r


def with_pii(rng, rid, items, extra_tags=(), ambiguous_types=()):
    """Prose plus one line per (label_text, value, type); spans point at `value`."""
    text = prose(rng, 220)
    spans, lines = [], []
    for lead, value, typ in items:
        line = f'{lead} {value} 입니다.'
        start = len(text) + 1 + sum(len(l) + 1 for l in lines) + len(lead) + 1
        lines.append(line)
        sp = dict(start=start, end=start + len(value), type=typ)
        if typ in ambiguous_types:
            sp['ambiguous'] = True
        spans.append(sp)
    text = text + '\n' + '\n'.join(lines)
    for sp in spans:
        assert text[sp['start']:sp['end']] in {v for _, v, _ in items}
    return rec(rid, text, 'keep', tags=['pii', *extra_tags], spans=spans)


def build():
    rng = random.Random(SEED)
    out = []

    # ---- keep: ordinary prose, mixed language, lists
    for i in range(12):
        out.append(rec(f'syn-keep-prose-{i:02d}', prose(rng, rng.randint(230, 900)), 'keep', tags=['prose']))
    for i in range(3):
        out.append(rec(f'syn-keep-mixed-en-{i:02d}', tech_prose(rng, rng.randint(230, 500)), 'keep', tags=['mixed_language']))
    for i in range(2):
        pairs = [(a, c) for a in PLACES for c in FOODS]
        rng.shuffle(pairs)
        text = '\n'.join(f'- {a}에서 {c} 먹기' for a, c in pairs[:22])
        out.append(rec(f'syn-keep-list-{i:02d}', text, 'keep', tags=['list']))

    # ---- boundary cases that sit on a threshold by construction
    out.append(rec('syn-bound-len-199', exact_len(rng, 199), 'reject', 'short', tags=['boundary_by_construction']))
    out.append(rec('syn-bound-len-200', exact_len(rng, 200), 'keep', tags=['boundary_by_construction']))
    for target, label, reason, rid in ((0.33, 'keep', None, 'syn-bound-hangul-033'), (0.27, 'reject', 'low_hangul_ratio', 'syn-bound-hangul-027')):
        base = prose(rng, 300)
        en = []
        while True:
            cand = base + ' ' + ' '.join(en)
            if hangul_ratio(cand) <= target:
                break
            en.append(rng.choice(EN_SENTENCES))
        out.append(rec(rid, cand, label, reason, tags=['boundary_by_construction']))

    # ---- policy-dependent keeps (label is a policy choice, flippable)
    for i in range(3):
        out.append(rec(f'syn-policy-nfd-{i:02d}', unicodedata.normalize('NFD', prose(rng, 300)), 'keep',
                       tags=['nfd'], policy='nfd_text_is_valid_korean', policy_reason='few_hangul'))
    for i in range(3):
        text = prose(rng, 500)
        n = 0
        while hangul_ratio(text) >= 0.285:
            text += f'\nvalue_{n} = compute_{n}(items, limit={n * 3 + 1}, mode="batch")'
            n += 1
        out.append(rec(f'syn-policy-code-tutorial-{i:02d}', text, 'keep', tags=['mixed_code'],
                       policy='korean_tutorial_with_code_is_keep', policy_reason='low_hangul_ratio'))
    for i, (lines, refrain, times) in enumerate(POEMS):
        body = list(lines) + [refrain] * times
        out.append(rec(f'syn-policy-poem-refrain-{i:02d}', '\n'.join(body), 'keep', tags=['poem', 'refrain'],
                       policy='refrain_poetry_is_keep', policy_reason='dup_lines'))

    # ---- rejects by design
    for i, n in enumerate((40, 90, 150)):
        out.append(rec(f'syn-rej-short-{i:02d}', exact_len(rng, n), 'reject', 'short', tags=['length']))
    for i in range(4):
        sents = rng.sample(EN_SENTENCES, 4 + i)
        out.append(rec(f'syn-rej-english-{i:02d}', ' '.join(sents), 'reject', 'few_hangul', tags=['language']))
    for i, target in enumerate((0.10, 0.15, 0.20)):
        base = prose(rng, 90)
        en = []
        while True:
            cand = base + ' ' + ' '.join(en)
            if hangul_ratio(cand) <= target:
                break
            en.append(rng.choice(EN_SENTENCES))
        out.append(rec(f'syn-rej-low-hangul-{i:02d}', cand, 'reject', 'low_hangul_ratio', tags=['language']))
    promos = ['오늘만 특가 할인 이벤트에 참여하세요 지금 바로 확인하세요 지금 바로 신청하세요',
              '무료 상담 예약 접수 중입니다 전화 한 통으로 간편하게 문의하세요',
              '최저가 보장 한정 수량 판매 중입니다 서두르시면 혜택을 받으실 수 있습니다']
    for i, line in enumerate(promos):
        out.append(rec(f'syn-rej-dup-lines-{i:02d}', '\n'.join([line] * (7 + i)), 'reject', 'dup_lines', tags=['spam']))
    for i in range(3):
        sentence = TEMPLATES[i].format(**slots(rng))
        out.append(rec(f'syn-rej-ngram-rep-{i:02d}', ' '.join([sentence] * 6), 'reject', 'ngram_rep', tags=['repetition']))
    for i, deco in enumerate(('★☆★', '♥♡♥', '▶▷▶')):
        words = prose(rng, 300).split()
        out.append(rec(f'syn-rej-low-alnum-{i:02d}', f' {deco} '.join(words).strip(), 'reject', 'low_alnum', tags=['decoration']))
    for i in range(2):
        text = prose(rng, 400)
        cut = [len(text) // 4, len(text) // 2, 3 * len(text) // 4]
        for pos in reversed(cut):
            text = text[:pos] + '�' + text[pos:]
        out.append(rec(f'syn-rej-bad-encoding-{i:02d}', text, 'reject', 'bad_encoding', tags=['encoding']))

    # ---- PII positives (fictional values only)
    phones = ['010-0000-1234', '01000001234', '010 0000 1234', '+82-10-0000-1234', '+82 10 0000 1234',
              '02-000-1234', '031-000-1234', '055-000-1234', '(02) 000-1234', '070-0000-1234']
    for i, v in enumerate(phones):
        out.append(with_pii(rng, f'syn-pii-phone-{i:02d}', [('연락처는', v, 'phone')], ['pii_phone']))
    for i, v in enumerate(['900101-1000001', '850315-2000002', '000101-3000003', '050505-4000004',
                           '9001011000005', '900101 1000006']):
        out.append(with_pii(rng, f'syn-pii-rrn-{i:02d}', [('번호는', v, 'rrn')], ['pii_rrn']))
    for i, v in enumerate(['900101-5000007', '850315-6000008', '000101-7000009', '050505-8000010']):
        out.append(with_pii(rng, f'syn-pii-alien-{i:02d}', [('번호는', v, 'alien_reg')], ['pii_alien_reg']))
    for i, v in enumerate(['user@example.com', 'first.last+tag@mail.example.org', 'USER_NAME@Sub.Example.com',
                           'a-b@example.co.kr', 'support@example.net']):
        out.append(with_pii(rng, f'syn-pii-email-{i:02d}', [('메일은', v, 'email')], ['pii_email']))
    for i, v in enumerate(['9999-8888-7777-6666', '9999 8888 7777 6666']):
        out.append(with_pii(rng, f'syn-pii-card-{i:02d}', [('카드는', v, 'card')], ['pii_card']))
    for i, v in enumerate(['000-000000-00-000', '999-123456-78-901']):
        out.append(with_pii(rng, f'syn-pii-account-{i:02d}', [('계좌는', v, 'account')], ['pii_account']))
    for i, v in enumerate(['192.0.2.10', '198.51.100.7']):
        out.append(with_pii(rng, f'syn-pii-ip-{i:02d}', [('접속 주소는', v, 'ip')], ['pii_ip']))
    out.append(with_pii(rng, 'syn-pii-multi-00',
                        [('메일은', 'user@example.com', 'email'), ('번호는', '900101-1000001', 'rrn'),
                         ('연락처는', '010-0000-1234', 'phone')], ['pii_multi']))

    # ---- PII hard negatives and ambiguous look-alikes
    negatives = ['2024-11-22', '2024.11.22 14:30', 'ISBN 978-89-000-0000-0', 'SKU 0123456789', '1,234,567원',
                 'v10.0.19045', '12:30-14:30', 'git@github.com:example/repo.git', 'Ver 1.2.3 build 20241122',
                 '37.5665, 126.9780']
    for i, v in enumerate(negatives):
        text = prose(rng, 220) + '\n참고 값은 ' + v + ' 입니다.'
        out.append(rec(f'syn-pii-negative-{i:02d}', text, 'keep', tags=['pii', 'pii_hard_negative']))
    out.append(with_pii(rng, 'syn-pii-ambiguous-00', [('문서번호', '02-2024-0001', 'phone')],
                        ['pii_hard_negative', 'pii_ambiguous'], ambiguous_types=('phone',)))
    out.append(with_pii(rng, 'syn-pii-ambiguous-01', [('주문번호', '2411221234567', 'rrn')],
                        ['pii_hard_negative', 'pii_ambiguous'], ambiguous_types=('rrn',)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    records = build()
    gold.dump(records, a.out)
    print(f'{len(records)} records -> {a.out}')


if __name__ == '__main__':
    main()
