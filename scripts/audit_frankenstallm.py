"""Offline reference audit and small CPU probes; never starts training.

Inputs are pinned reference snapshots described in docs/FRANKENSTALLM_REVIEW.md.
The reference definitions used by probes are loaded from inspected AST nodes;
their module-level launch, download, and notification code is not executed.
"""
from __future__ import annotations

import argparse
import ast
import collections
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import PROJECT_ROOT

GIT_REV = "50c6143629203815dfe7775e3b7c0dd0a3260324"
HF_REV = "b9f7777f01479b18e5a3a55f9cad4582df122f7d"
ROOT = Path(PROJECT_ROOT)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def definitions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    if {n.name for n in nodes} != set(names):
        raise ValueError(f"Missing definitions in {path}: {names}")
    future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *nodes], type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def validate_header(config, tensors):
    """Check required Llama tensors and shapes without downloading weights."""
    d, ff, vocab = config["hidden_size"], config["intermediate_size"], config["vocab_size"]
    head = config.get("head_dim", d // config["num_attention_heads"])
    kv = config["num_key_value_heads"] * head
    expected = {"model.embed_tokens.weight": [vocab, d], "model.norm.weight": [d]}
    for i in range(config["num_hidden_layers"]):
        p = f"model.layers.{i}."
        expected.update({p + "input_layernorm.weight": [d], p + "post_attention_layernorm.weight": [d],
            p + "self_attn.q_proj.weight": [config["num_attention_heads"] * head, d],
            p + "self_attn.k_proj.weight": [kv, d], p + "self_attn.v_proj.weight": [kv, d],
            p + "self_attn.o_proj.weight": [d, d], p + "mlp.gate_proj.weight": [ff, d],
            p + "mlp.up_proj.weight": [ff, d], p + "mlp.down_proj.weight": [d, ff]})
    if not config.get("tie_word_embeddings", False) or "lm_head.weight" in tensors:
        expected["lm_head.weight"] = [vocab, d]
    return {"missing_keys": sorted(set(expected) - set(tensors)),
            "unexpected_keys": sorted(set(tensors) - set(expected) - {"__metadata__"}),
            "shape_mismatches": {k: {"expected": shape, "actual": tensors[k]["shape"]}
                                 for k, shape in expected.items() if k in tensors and tensors[k]["shape"] != shape}}


def probes(ref, hf):
    import numpy as np
    import torch
    from tokenizers import Tokenizer, models

    torch.set_num_threads(2)
    torch.manual_seed(1234)
    result = {}

    ns = definitions(ref / "eval/tasks/ppl_task.py", ["SlidingWindowDataset"],
                     {"torch": torch, "np": np, "Dataset": torch.utils.data.Dataset})
    dataset = ns["SlidingWindowDataset"](np.arange(13, dtype=np.uint16), 8, 2)
    scored = []
    for i in range(len(dataset)):
        _, targets, mask = dataset[i]
        scored.extend(targets[mask].tolist())
    counts = collections.Counter(scored)
    short = len(ns["SlidingWindowDataset"](np.arange(8, dtype=np.uint16), 8, 2))
    result["legacy_ppl_coverage"] = dict(
        input_tokens=13, seq_len=8, stride=2, windows=len(dataset),
        scored_tokens=len(scored), unique_targets=len(counts),
        duplicate_evaluations=sum(v - 1 for v in counts.values()),
        missing_targets=sorted(set(range(1, 13)) - set(counts)),
        windows_when_length_equals_seq_len=short,
        interpretation="Overlapping targets repeat, final target is omitted; short input scores no tokens.")
    assert len(scored) == 17 and len(counts) == 11 and short == 0

    ns = definitions(hf / "data/sft_dataset.py", ["SFTDataset", "_build_conversation_turns"],
                     {"torch": torch, "Dataset": torch.utils.data.Dataset,
                      "_USER_TAG": "<|user|>\n", "_ASSISTANT_TAG": "<|assistant|>\n", "_EOS_STRING": "</s>"})
    cls = ns["SFTDataset"]
    tokenizer_a = Tokenizer(models.WordLevel({"<unk>": 0, "a": 1, "b": 2}, unk_token="<unk>"))
    tokenizer_b = Tokenizer(models.WordLevel({"<unk>": 0, "b": 1, "a": 2}, unk_token="<unk>"))
    with tempfile.TemporaryDirectory() as td:
        data_path = Path(td) / "sample.jsonl"
        data_path.write_text('{}\n')
        same_key = cls._make_cache_key(data_path, 16, tokenizer_a) == cls._make_cache_key(data_path, 16, tokenizer_b)
    assert tokenizer_a.encode("a").ids != tokenizer_b.encode("a").ids and same_key
    result["legacy_sft_cache"] = dict(different_token_ids=True, cache_keys_equal=same_key,
                                      interpretation="Vocabulary size does not identify tokenizer contents.")
    turns = ns["_build_conversation_turns"]([
        {"role": "system", "content": "SYSTEM_SENTINEL"},
        {"role": "user", "content": "question"},
        {"role": "assistant", "content": "answer"}])
    result["legacy_sft_system_role"] = {"system_retained": "SYSTEM_SENTINEL" in turns[0][0]}
    assert not result["legacy_sft_system_role"]["system_retained"]

    fake_tokenizer = SimpleNamespace(encode=lambda text: SimpleNamespace(ids=[3, 4] if text == "prompt" else [5] * 20 + [2]))
    instance = cls.__new__(cls)
    instance.eos_token_id, instance.pad_token_id = 2, 0
    with contextlib.redirect_stdout(io.StringIO()):
        samples = instance._tokenize_sequential([("prompt", "response")], fake_tokenizer, 16)
    inputs, labels = samples[0]
    assert inputs[-1].item() == 2 and labels[-2].item() == 2 and labels[-1].item() == -1
    assert labels[0].item() == -1 and labels[1].item() == 5
    result["legacy_sft_shift_and_eos"] = dict(sequence_length=len(inputs),
        eos_preserved_after_truncation=True, first_response_predicted_at_last_prompt_position=True,
        last_position_ignored=True, scope="Sequential tokenization on one synthetic example.")

    pref = definitions(hf / "data/prepare_preference_combined.py",
        ["_extract_text", "_build_prompt", "normalize_record", "passes_quality_filter"], {"MIN_LEN": 20})
    pair = pref["normalize_record"]({"prompt": "question", "response_a": "A" * 30, "response_b": "B" * 30}, "synthetic")
    accepted = pref["passes_quality_filter"](pair)
    assert pair["chosen"] == "A" * 30 and accepted
    result["legacy_preference_missing_winner"] = dict(defaulted_to_a=True, accepted_by_filter=accepted,
        interpretation="Unknown preference label is converted to an A preference, not rejected.")

    current = load_module("audit_current_model", ROOT / "model/model.py")
    rope = definitions(ref / "model/attention.py", ["apply_rotary_emb"], {"torch": torch})
    x = torch.randn(2, 7, 4, 8)
    cos, sin = current.precompute_rope(8, 7, 500000.0)
    delta = (current.apply_rope(x, cos, sin) - rope["apply_rotary_emb"](x, cos, sin)).abs().max().item()
    assert delta == 0
    result["rope_fp32"] = dict(max_abs_diff=delta, scope="One CPU tensor; not whole-model or FP8 equivalence.")
    cfg = current.ModelConfig(vocab_size=32, n_layers=1, dim=16, n_heads=4, n_kv_heads=2,
                              ffn_dim=32, max_seq_len=8)
    model = current.Transformer(cfg)
    try:
        model(torch.tensor([[1, 2, 3]]), torch.tensor([[-1, 3, 4]]))
    except IndexError as error:
        result["current_rejects_legacy_ignore_index"] = {"rejected": True, "error": str(error)}
    else:
        raise AssertionError("Expected current CE to reject legacy -1 labels")
    with torch.device("meta"):
        result["current_3b_parameter_count"] = current.Transformer(current.get_config("3b")).num_params()

    sampler = load_module("audit_current_data", ROOT / "train/data.py")
    with tempfile.TemporaryDirectory() as td:
        path, mix = Path(td) / "tokens.bin", Path(td) / "mix.json"
        np.arange(5, dtype=np.uint16).tofile(path)
        mix.write_text(json.dumps({"sources": {"test": {"weight": 1, "files": [str(path)]}}}))
        exact = sampler.MixSampler(str(mix), seq_len=4)
        try:
            exact.batch(0, 0, 0, 1)
        except ValueError as error:
            exact_error = str(error)
        else:
            raise AssertionError("Expected one-window edge failure")
        del exact
        np.arange(6, dtype=np.uint16).tofile(path)
        more = sampler.MixSampler(str(mix), seq_len=4)
        starts = sorted({int(more.batch(i, 0, 0, 1)[0][0, 0]) for i in range(64)})
        assert starts == [0]
        result["current_sampler_boundary"] = dict(exact_window_error=exact_error,
            sampled_starts=starts, valid_starts=[0, 1], trials=64)

    filters = load_module("audit_current_filters", ROOT / "data/filters.py")
    code_a = "if ready:\n    run()\n    stop()"
    code_b = "if ready:\n    run()\nstop()"
    same_hash = filters.exact_hash(code_a) == filters.exact_hash(code_b)
    assert same_hash and filters.exact_hash("X = 1") == filters.exact_hash("x = 1")
    result["current_web_hash_for_code"] = dict(different_indentation_same_hash=True,
        different_case_same_hash=True, interpretation="Web normalization cannot define exact duplicates for code.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--git-dir", default=".cache/references/FRANKENSTALLM")
    parser.add_argument("--hf-dir", default=f".cache/references/frankenstallm_hf/{HF_REV}")
    parser.add_argument("--out-dir", default="docs/reviews/frankenstallm")
    args = parser.parse_args()
    resolve = lambda p: Path(p) if Path(p).is_absolute() else ROOT / p
    ref, hf, out = map(resolve, (args.git_dir, args.hf_dir, args.out_dir))
    if git(ref, "rev-parse", "HEAD") != GIT_REV or git(ref, "status", "--porcelain"):
        raise ValueError("Reference Git checkout must be clean and at the pinned revision")
    manifest = json.loads((hf.parent / "download_manifest.json").read_text())
    if manifest["revision"] != HF_REV:
        raise ValueError("Unexpected HF revision")
    for item in manifest["files"]:
        if digest(hf / item["path"]) != item["sha256"]:
            raise ValueError("Reference snapshot changed: " + item["path"])
    local_paths = ["model/model.py", "train/train.py", "train/data.py", "data/filters.py",
                   "data/clean.py", "data/tokenize_shards.py", "data/build_mix.py",
                   "tokenizer/train_tokenizer.py", "tokenizer/check_tokenizer.py", "train/configs/3b.json",
                   "scripts/audit_frankenstallm.py"]
    local_hashes = {p: digest(ROOT / p) for p in local_paths}
    metadata_hashes = {name: digest(hf.parent / name) for name in
                       ["model_info_full.json", "all_safetensors_headers.json"]}
    fingerprint = hashlib.sha256(json.dumps([GIT_REV, manifest, local_hashes, metadata_hashes], sort_keys=True).encode()).hexdigest()
    output, done = out / "evidence.json", out / ".done"
    if done.exists():
        marker = json.loads(done.read_text())
        if marker["fingerprint"] != fingerprint or marker["output_sha256"] != digest(output):
            raise ValueError("Audit inputs/output changed; choose a new --out-dir")
        print(f"Verified completed audit: {out.relative_to(ROOT)}")
        return
    if output.exists():
        raise ValueError("Unfinished existing evidence; choose a new --out-dir")
    info = json.loads((hf.parent / "model_info_full.json").read_text())
    headers = json.loads((hf.parent / "all_safetensors_headers.json").read_text())
    header = headers["model.safetensors"]
    weight_profiles = []
    for path, tensors in headers.items():
        config_path = Path(path).parent / "config.json"
        config = json.loads((hf / config_path).read_text())
        weight_profiles.append({"path": path, "config": config,
            "stored_parameters": sum(math.prod(v["shape"]) for k, v in tensors.items() if k != "__metadata__"),
            "header_validation": validate_header(config, tensors), "header": tensors})
    matching, differing = [], []
    for item in manifest["files"]:
        path = item["path"]
        if path.startswith("source/") and (ref / path[7:]).is_file():
            (matching if digest(ref / path[7:]) == item["sha256"] else differing).append(path[7:])
    evaluations = []
    for path in sorted((hf / "source/eval/outputs").rglob("results_*.json")):
        data = json.loads(path.read_text())
        evaluations.append({"path": str(path.relative_to(hf)), "sha256": digest(path),
                            "config": data.get("config"), "date": data.get("date"),
                            "git_hash": data.get("git_hash"), "tasks": list(data.get("results", {}))})
    load_warnings = []
    for path in sorted((hf / "source/eval/outputs").rglob("*.log")):
        lines = path.read_text(errors="replace").splitlines()
        hits = [{"line": i + 1, "text": line[:500]} for i, line in enumerate(lines)
                if "| MISSING" in line and "self_attn." in line]
        if hits:
            load_warnings.append({"path": str(path.relative_to(hf)), "sha256": digest(path), "warnings": hits})
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "scope": "Static audit and synthetic CPU probes",
        "github": {"repo": "pathcosmos/FRANKENSTALLM", "revision": GIT_REV,
            "refs": git(ref, "for-each-ref", "--format=%(refname) %(objectname)").splitlines(),
            "reachable_commits": int(git(ref, "rev-list", "--all", "--count")),
            "data_or_trl_patch_history": git(ref, "log", "--all", "--format=%H", "--", "data", "scripts/trl_patch.py").splitlines(),
            "tracked_files": git(ref, "ls-tree", "-r", "--name-only", "HEAD").splitlines()},
        "huggingface": {"repo": "pathcosmos/frankenstallm", "revision": HF_REV,
            "remote_files": info["siblings"], "download_manifest": manifest,
            "root_config": json.loads((hf / "config.json").read_text()),
            "weight_profiles": weight_profiles, "metadata_sha256": metadata_hashes,
            "stored_parameters": sum(math.prod(v["shape"]) for k, v in header.items() if k != "__metadata__"),
            "weight_payload_downloaded": False},
        "source_comparison": {"identical": matching, "different": differing},
        "evaluation_provenance": evaluations, "evaluation_missing_weight_warnings": load_warnings,
        "current": {"git_head": git(ROOT, "rev-parse", "HEAD"), "working_file_sha256": local_hashes},
        "probes": probes(ref, hf),
        "limitations": ["No model quality evaluation or training run", "No full weight/data checksum verification",
                        "HF source snapshot is not proven identical to the historical training environment",
                        "Git history scope is the currently fetched branches, not deleted/unreachable commits"]}
    out.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temp.replace(output)
    done.write_text(json.dumps({"fingerprint": fingerprint, "output_sha256": digest(output)}, indent=2) + "\n")
    print(json.dumps({"evidence": str(output.relative_to(ROOT)), "probes": result["probes"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
