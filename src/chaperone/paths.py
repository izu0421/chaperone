"""Where chaperone reads/writes its runtime data (real HPA/PubMed/STRING/
CellPhoneDB/UniProt cache, per-candidate transcripts, AF3 fold outputs,
generated reports) — always relative to the directory chaperone is run
FROM, never the installed package's own location (which may be a read-only
site-packages directory once pip-installed). Override with the
CHAPERONE_HOME env var to pin a fixed project directory regardless of the
current working directory."""
import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("CHAPERONE_HOME", ".")).resolve()
DATA_DIR = PROJECT_ROOT / "data"
LOG_DIR = PROJECT_ROOT / "log"
FOLD_RUNS_DIR = PROJECT_ROOT / "fold_runs"
GPU_LOCKS_DIR = PROJECT_ROOT / ".gpu_locks"
