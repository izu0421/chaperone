"""How closely CAN a negative set be degree-matched to the positives?

01_build_benchmark reports mean positive-set degree 50.5 for positives, 5.1 for random
negatives and 20.8 for the degree-matched arm. The residual gap is not a bug in the matching
until it is shown to be one: among well-studied proteins, most pairs may already HAVE evidence,
in which case a negative set at positive-level degree does not exist to be drawn.
This measures that ceiling instead of assuming either way.
"""
import gzip, logging
from pathlib import Path
import numpy as np, pandas as pd
logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger()
HERE = Path(__file__).resolve().parent.parent
d = pd.read_parquet(HERE / "bench" / "pairs.parquet")
pos = d[d.label == 1]
deg = pd.concat([pos.prot_a, pos.prot_b]).value_counts()
log.info("proteins appearing in >=1 positive: %d", len(deg))
log.info("degree quantiles: %s", deg.quantile([.5,.9,.99,1.0]).to_dict())

posset = set(map(tuple, np.sort(pos[["prot_a","prot_b"]].to_numpy(), axis=1)))
for top in (200, 500, 1000, 2000):
    hubs = list(deg.head(top).index)
    n_possible = top * (top - 1) // 2
    hit = 0
    for i in range(len(hubs)):
        for j in range(i + 1, len(hubs)):
            a, b = sorted((hubs[i], hubs[j]))
            if (a, b) in posset:
                hit += 1
    log.info("top %4d hubs: %7d possible pairs, %6d are POSITIVES (%.1f%%) -> "
             "%.1f%% available as negatives",
             top, n_possible, hit, 100*hit/n_possible, 100*(1-hit/n_possible))
