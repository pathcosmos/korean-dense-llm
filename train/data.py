"""Stateless uint16 window sampling with explicit resume-compatible versions.
Version 1 preserves legacy length-weighted sampling and exclusive last offset.
Version 2 weights by valid windows and includes every valid offset (default).
"""
import json
import math
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT
import numpy as np
import torch

ROOT = PROJECT_ROOT
SAMPLER_VERSION = 2
_abs = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)


def validate_sampler_resume(metadata, requested):
    stored = metadata.get('sampler_version', metadata.get('config', {}).get('sampler_version', 1))
    if stored != requested:
        raise ValueError(f'Sampler version mismatch: checkpoint={stored}, requested={requested}. '
                         'Use the original sampler_version explicitly or start a new run directory.')


def open_tokens(path):
    path = _abs(path)
    size = os.path.getsize(path)
    if size == 0 or size % 2:
        raise ValueError(f'Expected nonempty, uint16-aligned token shard: {path}')
    return np.memmap(path, dtype=np.uint16, mode='r')


class MixSampler:
    def __init__(self, mix_path, seq_len, seed=1234, version=SAMPLER_VERSION):
        if type(seq_len) is not int or seq_len < 1:
            raise ValueError('seq_len must be a positive integer')
        if type(version) is not int or version not in (1, 2):
            raise ValueError('Unknown sampler version')
        if type(seed) is not int or seed < 0:
            raise ValueError('seed must be a nonnegative integer')
        with open(_abs(mix_path)) as f:
            mix = json.load(f)
        if not isinstance(mix.get('sources'), dict) or not mix['sources']:
            raise ValueError('Mix requires nonempty sources')
        if mix.get('sampler_version', version) != version:
            raise ValueError('Mix sampler version differs from requested version')
        self.seq_len, self.seed, self.version = seq_len, seed, version
        self.names, weights, self.shards = [], [], []
        T = seq_len + 1
        seen_paths = set()
        for name, source in mix['sources'].items():
            weight = source['weight']
            if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
                raise ValueError(f'Invalid positive finite weight for {name}')
            files = source.get('files')
            if not isinstance(files, list) or not files:
                raise ValueError(f'No shards for {name}')
            shards = []
            for file in files:
                path = str(Path(_abs(file)).resolve())
                if path in seen_paths:
                    raise ValueError(f'Repeated shard path: {file}')
                seen_paths.add(path)
                mm = open_tokens(path)
                windows = len(mm) - T + (1 if version == 2 else 0)
                if windows < 1:
                    raise ValueError(f'Shard too short for sampler v{version}: {file}')
                shards.append((mm, len(mm)))
            self.names.append(name);weights.append(weight);self.shards.append(shards)
        total = sum(weights)
        if not math.isfinite(total):
            raise ValueError('Weight sum must be finite')
        self.w = np.array(weights, dtype=np.float64) / total
        self.window_counts = [np.array([n - T + 1 for _, n in shards], dtype=np.int64) for shards in self.shards]
        self.shard_probabilities = [self._sizes(i) / self._sizes(i).sum() for i in range(len(self.shards))]

    def _sizes(self, i):
        if self.version == 1:
            return np.array([n for _, n in self.shards[i]], dtype=np.float64)
        return self.window_counts[i].astype(np.float64)

    def batch(self, step, rank, micro, bsz):
        if any(type(x) is not int or x < 0 for x in (step, rank, micro)) or type(bsz) is not int or bsz < 1:
            raise ValueError('Expected nonnegative step/rank/micro and positive integer batch size')
        rng = np.random.default_rng([self.seed, step, rank, micro])
        T = self.seq_len + 1
        x = np.empty((bsz, T), dtype=np.int64)
        for b in range(bsz):
            i = rng.choice(len(self.w), p=self.w)
            j = rng.choice(len(self.shards[i]), p=self.shard_probabilities[i])
            mm, n = self.shards[i][j]
            high = n - T + (1 if self.version == 2 else 0)
            o = int(rng.integers(0, high))
            x[b] = mm[o:o + T]
        x = torch.from_numpy(x)
        return x[:, :-1].contiguous(), x[:, 1:].contiguous()


class HoldoutSet:
    """Fixed next-token windows; incomplete tails are ignored for train-time probes.
    Publication PPL uses the separate full-coverage evaluator instead.
    """
    def __init__(self, files, seq_len, n_windows=64):
        if type(seq_len) is not int or seq_len < 1 or type(n_windows) is not int or n_windows < 1:
            raise ValueError('seq_len and n_windows must be positive integers')
        self.seq_len = seq_len; T = seq_len + 1
        wins = []
        for file in files:
            mm = open_tokens(file)
            for w in range(min(n_windows, len(mm) // T)):
                wins.append(np.asarray(mm[w * T:(w + 1) * T], dtype=np.int64))
        self.wins = torch.from_numpy(np.stack(wins)) if wins else None

    def __len__(self):
        return 0 if self.wins is None else len(self.wins)
