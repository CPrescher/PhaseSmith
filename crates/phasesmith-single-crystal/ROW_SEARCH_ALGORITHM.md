# Published direction methods and sparse-input selection

Algorithm credit: Aslandukov, Aslandukov, Dubrovinskaia and Dubrovinsky (2022),
J. Appl. Cryst. 55, 1383–1391, doi:10.1107/S1600576722008081, sections 2.3.1,
2.3.2 and 2.4. No original DAFi implementation code is used.

The existing independent implementation and all its numerical conventions are
inherited from the quality-panel baseline. Canonically sort and deduplicate
coordinates, centre by the unique-point mean, then normalize each axis by its
maximum centred absolute coordinate. Zero-spread axes have scale 1. Duplicate
occurrences remain conserved, with extra copies unassigned.

Two published first-stage distributions use the same unit-direction bins:

* Naive: every unordered pair contributes one vote to its antipodal direction
  bin. Rank descending integer votes and ascending bin key on ties.
* Smart: at each centre, bin outgoing directions, require a row of at least
  four points, and sweep the intervals D/(k+epsilon)..D/(k-epsilon), k=1..5,
  for the largest equidistant support. Only qualifying bins receive votes.

The previous selected variant always used smart nomination. With sparse/noisy
rows, this can discard directions before the projection stage can recognize
their useful rows. Increasing the 50,000-bin cap cannot recover a bin with zero
smart votes. Existing naive-mode controls recover supported groups with all
other arithmetic and thresholds unchanged. Global naive nomination, however,
regresses some denser training populations.

The candidate adds an execution policy: when `naive_max_points>0` and the number
of unique coordinates is at most this inclusive bound, use the published naive
distribution. Above it, use the requested smart setting. Explicit `smart=false`
always uses naive selection. The library default is 0 (automatic selection off)
for compatibility; candidate selected settings use 2048. This power-of-two small
input bound is an engineering crossover, not a published physical constant or
a claim about private original DAFi behavior. No dataset ID, fitted cell,
chemistry, reference label or observed test score enters this choice.

`effective_smart` in new results records the method actually used; it is absent
or null for legacy results. The requested options remain recorded separately.
Unique rather than raw observation count ensures coincident duplicate padding
cannot change the policy. Existing deterministic work, cancellation and wall
limits apply to the single selected histogram; the two methods are not run and
combined during production grouping.

The second stage is unchanged: decode bin-centre directions, project onto a
coordinate plane, choose the best of four fixed global projection-grid offsets,
find the modal relative-gap window, and choose circular phase subsets with
distinct integer sites. Seed/gap rows require 4 points; accepted phase rows may
have 3 sites. Epsilons remain direction .0033, projection .002, integer-site .02,
relative gap .02. This four-grid clustering and relative-gap interpretation are
disclosed independent choices, not exact original executable equivalence.

Normalization stays fixed; up to 6 residual candidates are extracted per batch.
The baseline's fixed histogram (no published subtraction/renormalization) is
unchanged. Thus this targeted improvement does not claim to resolve every
known difference from the paper or every quality-panel shortfall.

All cache representations, worker scheduling, phase predicates and precision
remain unchanged. 512 MiB is the total retained-payload default; 0 disables cache,
`cache_unlimited=true` removes the cap, and no arbitrary 1 GiB ceiling is added.
Payload is not total RSS. Completed results remain worker/cache/permutation
deterministic; partial deadline results are not quality-equivalent comparisons.

Conditional diagnostic probes can inspect already saved blind-cell directions
and their histogram ranks to isolate losses. Those vectors never enter the
production CLI, which accepts only a reciprocal-coordinate table and options.
Independent cell inference is evaluation only, not a replacement grouping
algorithm. All studied populations are development/regression evidence, not
known grain truth or a new blinded accuracy validation.
