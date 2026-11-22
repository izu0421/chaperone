#!/bin/bash
# Rebuild small_bfd_padded, which is broken for MMseqs2-GPU.
#
# The shipped padded database kills the ungapped prefilter with a CUDA illegal
# memory access (libmarv dbbatching.cuh:69). It is the database, not the harness
# and not the machine:
#
#   * reproduces in a bare `mmseqs search`, with AF3 out of the picture;
#   * reproduces on 2/2 AF3 smoke attempts, same line both times;
#   * uniref90 (49 GB) and mgnify (108 GB) search fine on the SAME card, so it is
#     not GPU memory -- the failure is on the smallest of the four databases;
#   * --db-load-mode 2, --db-load-mode 3, --split 2 and --max-seqs 500 all die
#     identically, so no tuning flag routes around it.
#
# It also cannot be skipped: pipeline.py:271 hardcodes
# msas=[uniref90_msa, small_bfd_msa, mgnify_msa] with no flag to drop one, and
# dropping it anyway would make arm A's MSAs shallower for a reason unrelated to
# AlphaFast -- turning B-A from an alignment-method effect into an
# alignment-method-plus-missing-database effect.
#
# Two steps, matching src/scripts/convert_databases_to_mmseqs.sh exactly:
#   mmseqs createdb        FASTA -> standard db
#   mmseqs makepaddedseqdb standard -> GPU padded db
#
# The rebuilt database is written beside the old one under a _new suffix and is
# only swapped in after it passes a GPU search. A rebuild that silently produced
# the same broken database would otherwise look like a fix.
set -euo pipefail
SIF=/data/yzy21/yy/af/alphafast/alphafast.sif
DBDIR=/data/yzy21/yy/af/alphafast_db/mmseqs
WORK=/data/yzy21/yy/af/alphafast_db/_rebuild_small_bfd
FASTA="$WORK/bfd-first_non_consensus_sequences.fasta"

echo "[1/4] $(date +%T) decompressing"
if [[ ! -f "$FASTA" ]]; then
  singularity exec --bind "$WORK":/w "$SIF" \
    bash -c "zstd -d -f /w/bfd.fasta.zst -o /w/bfd-first_non_consensus_sequences.fasta"
fi
ls -la "$FASTA"
grep -c '^>' "$FASTA" | xargs echo "sequences:"

echo "[2/4] $(date +%T) mmseqs createdb"
singularity exec --bind "$WORK":/w "$SIF" \
  mmseqs createdb /w/bfd-first_non_consensus_sequences.fasta /w/small_bfd

echo "[3/4] $(date +%T) mmseqs makepaddedseqdb"
singularity exec --bind "$WORK":/w "$SIF" \
  mmseqs makepaddedseqdb /w/small_bfd /w/small_bfd_padded_new

echo "[4/4] $(date +%T) GPU verification -- the rebuild is not trusted until this passes"
printf '>t\nMSDKPDMAEIEKFDKSKLKKTETQEKNPLPSKETIEQEKQAGES\n' > "$WORK/verify.fasta"
mkdir -p "$WORK/verify"
CUDA_VISIBLE_DEVICES=${GPU:-2} singularity exec --nv --bind "$WORK":/w "$SIF" bash -c \
  "mmseqs createdb /w/verify.fasta /w/verify/q >/dev/null 2>&1; \
   mmseqs search /w/verify/q /w/small_bfd_padded_new /w/verify/res /w/verify/tmp \
     -s 7.5 -e 0.0001 --threads 16 --max-seqs 500 --gpu 1" 2>&1 | tail -5

if [[ -f "$WORK/verify/res.dbtype" ]]; then
  echo "VERIFIED: GPU search succeeded on the rebuilt database."
  echo "Swap it in with:"
  echo "  for f in $WORK/small_bfd_padded_new*; do"
  echo "    mv \"\$f\" \"$DBDIR/\$(basename \$f | sed s/_padded_new/_padded/)\""
  echo "  done"
else
  echo "FAILED: the rebuilt database ALSO dies on GPU."
  echo "That would mean the problem is MMseqs2-GPU on this card, not the database."
  exit 1
fi
