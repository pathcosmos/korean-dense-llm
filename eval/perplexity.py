"""Full-coverage next-token NLL. Overlapping context never duplicates scored targets.
This is a single-process native-model evaluator, not FSDP orchestration.
"""
from dataclasses import dataclass
import math
import sys
import torch
import torch.nn.functional as F


@dataclass
class ScoringWindow:
    input_ids: torch.Tensor
    targets: torch.Tensor
    target_positions: list
    scored_positions: list


def scoring_windows(tokens, seq_len, stride):
    if type(seq_len) is not int or type(stride) is not int or not 1 <= stride <= seq_len:
        raise ValueError('Require 1 <= stride <= seq_len')
    count = len(tokens);next_target = 1
    while next_target < count:
        end = min(count, next_target + (seq_len if next_target == 1 else stride))
        start = max(0, end - seq_len - 1)
        # Only copy one bounded context, including for memmap inputs.
        raw = torch.tensor(tokens[start:end].tolist() if hasattr(tokens[start:end], 'tolist') else tokens[start:end])
        if raw.dtype.is_floating_point or raw.dtype == torch.bool:
            raise ValueError('Token IDs must be integers')
        ids = raw.long()
        if ids.ndim != 1 or (ids < 0).any():
            raise ValueError('Token sequence must contain nonnegative integer IDs')
        positions = list(range(start + 1, end))
        targets = ids[1:].clone()
        targets[:max(0, next_target - start - 1)] = -100
        yield ScoringWindow(ids[:-1], targets, positions, list(range(next_target, end)))
        next_target = end


class NLLAccumulator:
    def __init__(self):
        self.nll_sum = 0.0;self.valid_tokens = 0

    def add(self, nll_sum, valid_tokens):
        if not math.isfinite(nll_sum) or nll_sum < 0 or type(valid_tokens) is not int or valid_tokens < 0:
            raise ValueError('Invalid NLL/count')
        if valid_tokens == 0 and nll_sum != 0:
            raise ValueError('Nonzero NLL without valid targets')
        self.nll_sum += nll_sum;self.valid_tokens += valid_tokens

    def result(self):
        if self.valid_tokens == 0:
            raise ValueError('Invalid evaluation: no next-token targets')
        if not math.isfinite(self.nll_sum):
            raise ValueError('Nonfinite accumulated NLL')
        mean = self.nll_sum / self.valid_tokens
        overflow = mean > math.log(sys.float_info.max)
        return dict(nll_sum=self.nll_sum, valid_tokens=self.valid_tokens, mean_nll=mean,
                    perplexity=None if overflow else math.exp(mean), perplexity_overflow=overflow)


@torch.no_grad()
def evaluate_sequences(model, sequences, seq_len, stride, device='cpu'):
    """Current model API: model(input_ids) -> [B,T,V] logits; no target shift inside model."""
    training = model.training;model.eval()
    accumulator = NLLAccumulator();windows = 0
    try:
        for tokens in sequences:
            for window in scoring_windows(tokens, seq_len, stride):
                ids = window.input_ids[None].to(device)
                targets = window.targets[None].to(device)
                logits = model(ids)
                if logits.ndim != 3 or logits.shape[:2] != targets.shape:
                    raise ValueError('Unexpected native model logits shape')
                valid = targets[targets != -100]
                if valid.numel() and int(valid.max().item()) >= logits.shape[-1]:
                    raise ValueError('Target token exceeds model vocabulary')
                nll = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1),
                                      ignore_index=-100, reduction='sum')
                accumulator.add(float(nll.item()), len(window.scored_positions));windows += 1
    finally:
        model.train(training)
    return dict(**accumulator.result(), windows=windows)
