# Guide to Pharmacology interactions table

Not committed: it is a third-party file of several megabytes that is republished
on its own schedule, and a stale copy in the repo would silently become the
labels for a panel built later.

    mkdir -p bench/gtopdb_raw
    curl -L -o bench/gtopdb_raw/interactions.csv \
      https://www.guidetopharmacology.org/DATA/interactions.csv

`25_build_protein_counterscreen.py` reads it and prints the version line, so the
build log records which release the labels came from.
