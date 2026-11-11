# Filtered Vector Search Under Load — Milestone 1 (Exact Filtered Baseline)

Single-node filtered vector search in Go. Given a query embedding, a category equality filter, and `k`, the engine returns the `k` closest vectors among documents in that category.

## Run Tests

```bash
go test -v ./...
```

## Milestone 1 Specifications & Design Decisions

### 1. Distance Metric
- **Squared Euclidean Distance ($L_2^2$)** in `float32`:
  $$\mathrm{dist}(\mathbf{q}, \mathbf{x}) = \sum_{i=0}^{d-1} (q_i - x_i)^2$$
- Because $\sqrt{\cdot}$ is strictly monotonic on $[0, \infty)$, $L_2^2$ preserves exact nearest-neighbor rankings while avoiding per-candidate `sqrt` overhead.
- All vector dimensions are validated at index construction and query time (`len(v) == dim`, no `NaN` or `Inf`).

### 2. Stable Tie-Breaking
- Candidates are ordered by total order `(Distance ASC, ID ASC)`:
  1. Smaller `Distance` ranks first.
  2. Equal `Distance` breaks ties deterministically by lexicographically smaller `ID` (`a.ID < b.ID`).
- Tie-breaking is enforced inside the bounded size-`k` max-heap (`resultMaxHeap`) so that when a candidate ties with the current `k`-th best result at the heap root, the lexicographically larger `ID` is evicted.

### 3. Filter Semantics & Corpus Model
- **Read-only corpus**: `NewExactIndex(dim, docs)` validates all documents up front (rejecting non-positive dimensions, mismatched vector lengths, `NaN`/`Inf` values, empty IDs, and duplicate IDs across the entire corpus), copies vectors into a contiguous `[]float32` backing slice (`vectors[i*dim : (i+1)*dim]`), and builds an immutable `map[string][]int` category posting list.
- **Single equality filter**: Only documents where `doc.Category == category` are eligible.
- **Boundary behavior**:
  - Unknown or empty category returns `[]Result{}` (0 matches, `nil` error).
  - Underfilled category with $m < k$ eligible documents returns all $m$ documents ordered by `(Distance ASC, ID ASC)`.

### 4. Planned Benchmark Matrix (Next Milestones)
Before introducing the inverted-file (IVF) approximate index and query planners, the benchmark matrix is fixed as follows:

| Axis | Values | Purpose |
| --- | --- | --- |
| Corpus size ($N$) | `10,000` (dev/CI), `100,000` (full benchmark) | Measure crossover between category posting scan and IVF probing |
| Dimension ($d$) | `64`, `128` | Contiguous dot/L2 scan cost per candidate |
| Category selectivity ($s = N_{\text{cat}} / N$) | `0.1%`, `1%`, `5%`, `20%`, `50%` | Test rare vs. common filters |
| Cluster concentration | **Concentrated** (category resides in 1–3 centroids) vs. **Dispersed** (uniformly spread across centroids) | Isolate when per-centroid category counts beat a size-only planner |
| Search strategies | 1. Exact category scan (`ExactIndex`)<br>2. Global IVF top-$M$ + post-filter<br>3. IVF with inline filter + fixed `nprobe`<br>4. Size-only planner<br>5. Cluster-aware planner (per-centroid category histograms) | Compare `recall@k` (using exact filtered top-$k$ as ground truth), candidate distance evaluations, and p50/p95 latency |
