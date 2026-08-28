"""Fetch UniProt sequences for one or more proposed proteins and fold them
together as a single complex using AlphaFast (local AlphaFold 3). Folding is
entirely optional: if you don't have a local AlphaFast install, just don't
pass GPU device IDs (`--fold-gpus ""` / `--no-fold` on the `chaperone`
command) and this module is never invoked.

If you do have AlphaFast, point this module at it via env vars (defaults
below match a specific dev box and are almost certainly wrong for you):
    CHAPERONE_ALPHAFAST_DIR  (default /data/yzy21/yy/af/alphafast)
    CHAPERONE_ALPHAFAST_DB_DIR      (default /data/yzy21/yy/af/alphafast_db)
    CHAPERONE_ALPHAFAST_WEIGHTS_DIR (default /data/yzy21/yy/af)

This is the "close the loop" step for a triage follow_up like "re-run AF3
multimer with subunit X" — it actually runs it, rather than just suggesting it.

Usage:
    python -m chaperone.fold_candidate RUNX1 CBFB --name runx1_cbfb
    python -m chaperone.fold_candidate Q01196 Q13951 --name runx1_cbfb_by_accession
"""
import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import httpx

from .sources.http_retry import get_with_retry  # noqa: E402
from .paths import FOLD_RUNS_DIR, GPU_LOCKS_DIR  # noqa: E402

ALPHAFAST_DIR = Path(os.environ.get("CHAPERONE_ALPHAFAST_DIR", "/data/yzy21/yy/af/alphafast"))
DB_DIR = Path(os.environ.get("CHAPERONE_ALPHAFAST_DB_DIR", "/data/yzy21/yy/af/alphafast_db"))
WEIGHTS_DIR = Path(os.environ.get("CHAPERONE_ALPHAFAST_WEIGHTS_DIR", "/data/yzy21/yy/af"))  # contains af3.bin.zst directly
SIF = ALPHAFAST_DIR / "alphafast.sif"
RUN_SCRIPT = ALPHAFAST_DIR / "scripts" / "run_alphafast.sh"
GPU_LOCK_DIR = GPU_LOCKS_DIR

CHAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# UniProt accession shape, e.g. P00533, Q9Y6K9, A0A024RBG1
UNIPROT_ACCESSION_RE = re.compile(
    r"^[OPQ][0-9][A-Z0-9]{3}[0-9](\.\d+)?$|^[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}(\.\d+)?$"
)


def fetch_sequence(identifier: str) -> dict:
    """Resolve a gene symbol or UniProt accession to its reviewed human sequence."""
    ident = identifier.strip()
    with httpx.Client() as client:
        if UNIPROT_ACCESSION_RE.match(ident.upper()):
            resp = get_with_retry(client, f"https://rest.uniprot.org/uniprotkb/{ident.upper()}.fasta", {}, 30.0)
            fasta = resp.text
        else:
            resp = get_with_retry(
                client,
                "https://rest.uniprot.org/uniprotkb/search",
                {"query": f"gene:{ident} AND organism_id:9606 AND reviewed:true", "format": "fasta"},
                30.0,
            )
            fasta = resp.text

    if not fasta.strip():
        raise ValueError(f"No reviewed human UniProt entry found for '{identifier}'")

    # The gene-symbol search can match more than one reviewed entry (e.g. a
    # broader hit than an exact gene match), returning multi-record FASTA.
    # Blindly treating every non-header line as sequence would silently
    # splice a second record's header text into the sequence string — this
    # happened for real (GRIK5) and corrupted the fold input. Split into
    # records and require exactly one exact GN= match.
    records = []
    current_header, current_seq = None, []
    for line in fasta.strip().splitlines():
        if line.startswith(">"):
            if current_header is not None:
                records.append((current_header, "".join(current_seq)))
            current_header, current_seq = line, []
        else:
            current_seq.append(line)
    if current_header is not None:
        records.append((current_header, "".join(current_seq)))

    if len(records) > 1:
        exact = [r for r in records if f"GN={ident.upper()} " in r[0].upper() + " " or r[0].upper().rstrip().endswith(f"GN={ident.upper()}")]
        if len(exact) != 1:
            raise ValueError(
                f"UniProt gene search for '{identifier}' returned {len(records)} entries, "
                f"none/multiple with an exact GN={ident.upper()} match — resolve manually "
                f"(try the exact UniProt accession instead)"
            )
        records = exact

    header, seq = records[0]
    accession = header.split("|")[1] if header.count("|") >= 2 else None
    return {
        "identifier": identifier,
        "accession": accession,
        "header": header.lstrip(">"),
        "sequence": seq,
    }


def build_af3_input(name: str, chains: list[dict], model_seeds: list[int]) -> dict:
    if len(chains) > len(CHAIN_LETTERS):
        raise ValueError(f"Too many chains ({len(chains)}) for single-letter chain IDs")
    return {
        "name": name,
        "sequences": [
            {"protein": {"id": [CHAIN_LETTERS[i]], "sequence": c["sequence"]}}
            for i, c in enumerate(chains)
        ],
        "modelSeeds": model_seeds,
        "dialect": "alphafold3",
        "version": 1,
    }


@contextmanager
def gpu_lock(gpu_device: str):
    """Real cross-process mutex on a physical GPU. AF3 inference is memory-
    hungry enough that two concurrent jobs on the same device OOM rather than
    time-slice — a per-process asyncio.Semaphore only coordinates callers
    *within* one Python process, so a separate script (or a separate agent
    run) targeting the same GPU can still collide with it. This flock is the
    actual safety net: any fold job, from any process, blocks here until the
    GPU it asked for is really free."""
    GPU_LOCK_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = GPU_LOCK_DIR / f"gpu_{gpu_device}.lock"
    with open(lock_path, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def run_alphafast(input_dir: Path, output_dir: Path, gpu_device: str) -> None:
    cmd = [
        "bash", str(RUN_SCRIPT),
        "--input_dir", str(input_dir),
        "--output_dir", str(output_dir),
        "--db_dir", str(DB_DIR),
        "--weights_dir", str(WEIGHTS_DIR),
        "--container", str(SIF),
        "--gpu_devices", str(gpu_device),
    ]
    # subprocess.run(check=True) alone gives an uninformative "exit status 1"
    # on failure — the container's real stdout/stderr just goes to the
    # parent's terminal and is lost from the exception. Tee it: still stream
    # live (so a running job stays observable), but also keep the tail so a
    # failure is self-diagnosing instead of requiring a grep through whatever
    # shared log happened to be capturing stdout at the time.
    tail = []
    with gpu_lock(gpu_device):
        proc = subprocess.Popen(
            cmd, cwd=ALPHAFAST_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        for line in proc.stdout:
            print(line, end="")
            tail.append(line)
            if len(tail) > 80:
                tail.pop(0)
        returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(
            f"run_alphafast.sh exited {returncode} on GPU {gpu_device}. Last output:\n" + "".join(tail)
        )


def parse_confidences(output_dir: Path, name: str) -> dict:
    candidates = [
        output_dir / name / f"{name}_summary_confidences.json",
        output_dir / f"{name}_summary_confidences.json",
    ]
    summary_path = next((p for p in candidates if p.exists()), None)
    if summary_path is None:
        raise FileNotFoundError(
            f"No summary_confidences.json found for '{name}' under {output_dir}"
        )
    data = json.loads(summary_path.read_text())
    model_cif = output_dir / name / f"{name}_model.cif"
    return {
        "iptm": data.get("iptm"),
        "ptm": data.get("ptm"),
        "ranking_score": data.get("ranking_score"),
        "has_clash": data.get("has_clash"),
        "fraction_disordered": data.get("fraction_disordered"),
        "chain_pair_iptm": data.get("chain_pair_iptm"),
        "model_cif": str(model_cif) if model_cif.exists() else None,
        "summary_confidences_path": str(summary_path),
    }


def fold(proteins: list[str], name: str, output_root: Path = None, gpu_device: str = "0",
         model_seeds: list[int] = (1,)) -> dict:
    chains = [fetch_sequence(p) for p in proteins]
    for c in chains:
        print(f"  {c['identifier']} -> {c['accession']} ({len(c['sequence'])} aa)", file=sys.stderr)

    af3_input = build_af3_input(name, chains, list(model_seeds))

    work_dir = Path(output_root) if output_root else FOLD_RUNS_DIR / name
    input_dir, output_dir = work_dir / "input", work_dir / "output"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / f"{name}.json").write_text(json.dumps(af3_input, indent=2))

    t0 = time.time()
    run_alphafast(input_dir, output_dir, gpu_device)
    elapsed = time.time() - t0

    result = parse_confidences(output_dir, name)
    result["elapsed_seconds"] = round(elapsed, 1)
    result["job_name"] = name
    result["chains"] = [
        {"identifier": c["identifier"], "accession": c["accession"], "length": len(c["sequence"])}
        for c in chains
    ]
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("proteins", nargs="+", help="Gene symbols or UniProt accessions to fold as one complex")
    ap.add_argument("--name", required=True, help="Job name (used for output dir/file naming)")
    ap.add_argument("--output_dir", default=None, help="Override the default fold_runs/<name>/ location")
    ap.add_argument("--gpu_device", default=os.environ.get("GPU_DEVICE", "0"))
    ap.add_argument("--model_seeds", default="1", help="Comma-separated seed list, e.g. 1,2,3")
    args = ap.parse_args()

    result = fold(
        args.proteins,
        args.name,
        output_root=args.output_dir,
        gpu_device=args.gpu_device,
        model_seeds=[int(s) for s in args.model_seeds.split(",")],
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
