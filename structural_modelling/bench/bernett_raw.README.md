# Bernett gold-standard PPI dataset

Lives at `data/bernett_raw/`, which is ignored wholesale by
`structural_modelling/.gitignore`; this note sits in `bench/` so it is tracked.

Not committed: 14 MB of third-party files that figshare versions independently, and a
stale copy in the repo would silently become the labels for a pilot built later.

    mkdir -p data/bernett_raw && cd data/bernett_raw
    curl -sS -L -o meta.json https://api.figshare.com/v2/articles/21591618
    # then fetch each file['download_url'] from meta.json and check its md5 against
    # file['computed_md5'] -- scripts/30_build_bernett_pilot.py expects the names
    # Intra{0,1,2}_{pos,neg}_rr.txt and human_swissprot_oneliner.fasta

Bernett J, Blumenthal DB, List M. "Cracking the black box of deep sequence-based
protein-protein interaction prediction." Brief Bioinform 25(2):bbae076, 2024.
figshare DOI 10.6084/m9.figshare.21591618 — version 4 was current on 2026-11-24, and
the paper cites v3, so record which version a result came from.

Positives from HIPPIE v2.3. Splits built with KaHIP over a SIMAP2 all-against-all
sequence-similarity matrix, which controls train/test leakage for sequence models; we
train nothing, so what we use it for is its degree-balanced negatives (1.09x on our
independent reference, against 9.57x for the RF2-PPI benchmark) and its curated
positives.
