# Runbook

Everything here is safe to re-run. The scorers are incremental, the fold runners skip
work already done, and the guards below refuse rather than guess.

## Where the pipeline is

    micromamba run -n structmod python -I scripts/00_status.py

One row per stage. Read `pct_of_foldable`, not `pct`: work that cannot be done is not work
that is pending. The ETA is cost-weighted because the queues are size-ordered, so folds/h
falls while GPU throughput is constant — a count-based estimate reads that as slowing down.

## Before launching a stage

    micromamba run -n structmod python -I scripts/check_af3_inputs.py

Every protein chain must arrive with its MSA embedded: these folds run with
`--norun_data_pipeline`, so AlphaFold3 cannot build one, and a chain that omits the key is
not folded single-sequence — it is rejected, per job, silently. 85 orphan jobs sat in that
state for four days. If this fails on the orphan set:

    micromamba run -n structmod python -I scripts/13_repair_unfoldable_inputs.py --dry-run
    micromamba run -n structmod python -I scripts/13_repair_unfoldable_inputs.py

It repairs only the failing jobs. Do not re-run `12_build_orphan_inputs_omic.py` to fix a
few: it rewrites all 5,000, which makes every input newer than its fold and turns every
clean fold into a provenance suspect.

## Folding

    nohup setsid bash scripts/run_orphan_fold.v2.sh 0 1 2 >>logs/orphan_shard0.out 2>&1 &
    nohup setsid bash scripts/run_orphan_fold.v2.sh 1 2 2 >>logs/orphan_shard1.out 2>&1 &

Arguments are shard, GPU, number of shards. Use the `.v2` runner: v1 measures progress by
counting output directories, which AlphaFold3 creates up front, so it is saturated once
every job has a directory and exits "NO PROGRESS" even while folds succeed.

The whole queue, orphan → specificity → HuRI, each scored as it finishes:

    ORPHAN_PIDS="<pids>" nohup setsid bash scripts/run_queue.v2.sh >/dev/null 2>&1 &

`ORPHAN_PIDS` is optional and only needed when orphan shards are already running from
before the marker convention. The queue stops rather than advancing past a stage that did
not reach its completion bar.

## Scoring

    micromamba run -n structmod python -I scripts/14_score_orphans.py --workers 3
    micromamba run -n structmod python -I scripts/18_score_specificity.py --workers 3

`--workers 3`, not more: this box has 4 cores and the GPU folds need the rest. 20 workers
starved the folds that were the actual critical path.

Both are incremental and write a `.coverage.json` beside the result. Useful flags:

- `--out NAME.csv` — score without touching the file the figures read
- `--seed-from NAME.csv` — inherit another run's still-valid rows
- `--full` — ignore the cache and re-score everything
- `--checkpoint-every N` — persist partials so an interrupted run resumes
- `--allow-shrink` — permit writing fewer folds than the existing result; needed only
  when a shrink is genuinely intended, and recorded in the sidecar
- `18_...--allow-partial` — report ranks on an unfinished matrix, knowing a ligand ranked
  against 6 receptors instead of 16 looks far better than it is

## Guards, and what each one is for

    micromamba run -n structmod python -I scripts/check_scorer_regression.py
    micromamba run -n structmod python -I scripts/check_fold_provenance.py
    /data/yzy21/micromamba/envs/saga/bin/python -m pytest tests/ -q

- **scorer regression** — `one()` is shared by every arm and every screen on purpose, so a
  benchmark pair and a candidate pair pass through identical code. Run this after any edit
  to the scoring code: 24 folds spanning ipTM 0.03–0.95 are pinned, and it exits non-zero
  if an existing column moves. Adding a column is fine.
- **fold provenance** — is each fold on disk the fold its input now describes? Checks by
  content, using the sequences AlphaFold3 records in `<job>_data.json`. `--deep` compares
  every fold instead of only those whose input is newer; slow, since each record embeds
  its MSAs.
- **tests** — 30, covering the scoring store's guarantees, the provenance fingerprint and
  the shard assignment. `structmod` has no pytest and does not need one; any environment
  with pandas and pytest runs them.

## Figures

    cd ../manuscript && jupyter nbconvert --to notebook --execute --inplace fig4_deorphanisation.ipynb

Figure 4 refuses to redraw from a scores file older than its own scoring code.
`ALLOW_STALE_SCORES=1` to inspect the old figure, never to publish it. An unfinished screen
is expected and only has to be stated; the notebook writes its coverage sentence to
`results/fig4_coverage.txt` from the data instead of having it typed into the legend.

`figstyle.assert_in_view()` refuses to draw a panel that leaves data outside its own axes.
If it raises, widen the limits or use a log scale — do not pass `allow=` without also
saying in the legend what is cropped.

## Still to do

- Fold the orphan jobs that remain. After the repair they are foldable; run the `.v2`
  runner once the queue's later stages are done.
- The orphan screen's own ceiling is worth stating in the Results: the largest fold that
  succeeded is 3,952 tokens, and 6 jobs failed with CUDA out-of-memory and never
  recovered, median 2,440 tokens. That is a size truncation, not a random dropout, and
  nothing currently records it per job.
