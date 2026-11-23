"""Describe the PrePPI table before anything is built on it.

Run as:  python -I scripts/00_describe_preppi.py data/preppi_raw/preppi_final600.txt
The path is an argument and the interpreter is isolated because the table is downloaded data.
"""
import sys
import pandas as pd

path = sys.argv[1]
d = pd.read_csv(path, sep="\t", low_memory=False)
print(f"rows {len(d):,}   cols {list(d.columns)}\n")
print(f"distinct proteins: {len(set(d.prot1) | set(d.prot2)):,}")
print(f"self pairs       : {int((d.prot1 == d.prot2).sum()):,}")
dup = d.assign(k=[tuple(sorted(x)) for x in zip(d.prot1, d.prot2)]).k.duplicated().sum()
print(f"duplicate pairs (unordered): {int(dup):,}\n")

# The experimental term is what makes final_score unusable as a label for a structure predictor.
has_exp = d.exp_score.notna()
print(f"pairs WITH an experimental term (exp_score): {int(has_exp.sum()):,} "
      f"({100 * has_exp.mean():.1f}%)")
print(f"  of those, final_score != total_score      : "
      f"{int((has_exp & (d.final_score != d.total_score)).sum()):,}")
print(f"  pairs with a literature reference (pubs)  : {int(d.pubs.notna().sum()):,}\n")

scores = ["str_score", "protpep_score", "str_max_score", "red_score", "ort_score",
          "phy_score", "coexp_score", "go_score", "total_score", "exp_score", "final_score"]
q = d[scores].describe(percentiles=[0.5, 0.9, 0.99]).T
q["n_distinct"] = [d[c].nunique() for c in scores]
# A score with few distinct values cannot rank: ties, not discrimination.
q["frac_distinct"] = q.n_distinct / len(d)
print(q[["count", "min", "50%", "90%", "99%", "max", "n_distinct", "frac_distinct"]]
      .to_string(float_format=lambda v: f"{v:,.4g}"))
