"""Arm A inputs: the same 2,500 pilot pairs as arm B, with the omicMSAs removed
so AlphaFast runs its own MMseqs2-GPU homology search.

B - A isolates the ALIGNMENT at fixed engine, which only works if nothing else
differs between the two arms. Two things are therefore held constant on purpose:

  * the pair set, taken from arm B's own input files rather than rebuilt from
    the benchmark, so a pair cannot silently differ between arms; and
  * `templates: []`, kept explicitly. AF3 skips template search for a chain
    whose `templates` key is present, so dropping the key would let arm A search
    templates while arm B had none, and the "MSA effect" would then be an MSA +
    template effect. Arm B ran with no templates, so arm A runs with none.

Only `pairedMsa` and `unpairedMsa` are removed. Their absence is what tells AF3
to run the data pipeline.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "work" / "armB_inputs"
DST = ROOT / "work" / "armA_inputs"
DST.mkdir(parents=True, exist_ok=True)

n = kept = 0
chains = set()
for f in sorted(SRC.glob("*.json")):
    d = json.loads(f.read_text())
    for s in d["sequences"]:
        p = s["protein"]
        p.pop("pairedMsa", None)
        p.pop("unpairedMsa", None)
        p["templates"] = []          # held identical to arm B
        chains.add(p["sequence"])
    (DST / f.name).write_text(json.dumps(d))
    n += 1
    kept += len(d["sequences"])

print(f"{n} pairs written to {DST}")
print(f"{kept} chain entries, {len(chains)} unique sequences to align")
print(f"input size: {sum(p.stat().st_size for p in DST.glob('*.json'))/1e6:.1f} MB "
      f"(arm B: {sum(p.stat().st_size for p in SRC.glob('*.json'))/1e9:.1f} GB)")
