"""Create fake structured token data in /tmp/smoke_tokens for train/configs/smoke.json (pipeline smoke test only)."""
import numpy as np, json, os
d = "/tmp/smoke_tokens"; os.makedirs(d, exist_ok=True)
rng = np.random.default_rng(0); base = rng.integers(0, 65000, size=(2000, 16))
seq = base[rng.integers(0, 2000, size=400000)].reshape(-1).astype(np.uint16)
seq.tofile(f"{d}/a.bin"); seq[:200000].tofile(f"{d}/hold.bin")
json.dump({"sources": {"a": {"weight": 1, "files": [f"{d}/a.bin"]}}}, open(f"{d}/mix.json", "w")); print("ok", d)
