# Tests

    /data/yzy21/micromamba/envs/saga/bin/python -m pytest tests/ -q

The `structmod` environment has no pytest and does not need one: these tests
exercise `scripts/score_store.py`, whose only dependency is pandas. Any
environment with pandas and pytest will run them (`saga`, `microbetcr`,
`nerveint` all do).

`test_score_store.py` covers the four failures the store exists to prevent,
each of which has bitten this project in some form:

- a queued fold read as a stalled one (directory mtime is not a liveness
  signal, because the runners pre-create every output directory)
- a results file holding rows computed by two different versions of the
  scoring code
- a re-folded job reused from cache because the cache keyed on the job name
- a partial scoring pass silently overwriting a more complete result
