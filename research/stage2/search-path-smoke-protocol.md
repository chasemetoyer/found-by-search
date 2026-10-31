# Exact search-path pre-lock rehearsal

Authorized by user approval 24; not a scientific search and not a Stage 2 lock.
Run the committed fixed controller command with mode search/profile smoke.
The controller factory, search-plan orchestration, worker search validator,
training wrapper, candidate encoding, CMA ask/tell replay, random backfill,
finalist ranking and winner export are the same code used by registered search.
Only plan data differ: 8192 training steps; population6,2generations,12 random
candidates; two training tasks; one seed per task; three finalists per method
and two finalist seeds per task; all environment and optimizer seeds >=5000.
The production plan remains 2M steps,pop10,12generations,120 random candidates,
3training seeds,3finalists and10finalist seeds.

Rehearsal result is PASS or FAIL only. Do not inspect/report candidate fitness,
solve counts, winner parameters or tamper predictions from this test. Raw
training output and selection records are retained for provenance and replay.
Confirm full plan completion,72unique logical runs,zero uncertain calls,worker
search validation of each saved spec,source identities,allbounds,CMA replay,
finalist/winner structure and CPU metadata for every attempt. Save a PASS
receipt tied to the tracked run and implementation hashes. This is provenance
of an exercised path, not the scientific preregistration lock.

Full search cannot start without a public hash-locked registration and GitHub
receipt, whose mandatory code hashes equal the passing rehearsal hashes. The
lock step may switch mode data to profile registered; it adds no new runtime
implementation. Local tests exercise the full authorization branch using
temporary fixtures outside the project; they create no project lock.
