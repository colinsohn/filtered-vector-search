# Filtered Vector Search Under Load

Single-node filtered vector search engine in Go. Given a query embedding, a category equality filter (`category`), and `k`, the engine returns the `k` closest vectors among documents in that category.

## Quickstart

```bash
go test -race -v ./...
```

## Architecture & Core Decisions

1. **Squared Euclidean Distance ($L_2^2$) in `float64` Precision**
   - Vectors are stored compactly in a contiguous `[]float32` slice (`vectors[i*dim : (i+1)*dim]`), but coordinates are converted to `float64` **before** subtraction (`float64(a[i]) - float64(b[i])`) and accumulated into a `float64` `Result.Distance`.
   - This prevents `float32` subtraction overflow and squaring saturation (`+Inf`) on extreme coordinates (e.g., `MaxFloat32` vs. `MaxFloat32 / 2`).
2. **Deterministic Tie-Breaking (`Distance ASC, ID ASC`)**
   - Enforced both inside the bounded size-`k` max-heap (`resultMaxHeap`) and in final result ordering so ties across the $k$-th boundary always evict the lexicographically larger `ID`.
3. **Search Implementations**
   - `ExactIndex`: Scans the contiguous vectors referenced by `byCategory[category]` postings using a bounded size-`k` max-heap.
   - `IVFIndex.SearchPreFilter`: Partitions vectors into `numClusters` clusters via deterministic k-means++ and Lloyd's iterations, probes the `nprobe` nearest centroids, and scores only documents matching `category`.
   - `IVFIndex.SearchPostFilter`: Baseline that scores all documents in the `nprobe` nearest clusters for global top-$M$ before filtering by `category`.

## Measured Results (Seeded Workload: `N=1000`, `dim=16`, `clusters=16`, `queries=60`, `k=10`)

### 1. IVF Pre-Filter Recall vs. Probe Budget (`nprobe`)

| `nprobe` (of 16) | Mean `Recall@10` | Mean Candidates Scored | Underfilled Queries (of 300) |
| --- | --- | --- | --- |
| `1` | `0.6800` | `14.0` | `136` |
| `2` | `0.7960` | `23.5` | `91` |
| `4` | `0.8563` | `46.1` | `60` |
| `8` | `0.9060` | `85.5` | `60` |
| `16` (exhaustive) | `1.0000` | `200.0` | `0` |

### 2. Rare Category (`rare_vintage`, 2% of corpus) at `nprobe=4`, `k=10`
- **IVF Pre-Filter**: `Recall@10 = 0.2883`
- **IVF Post-Filter (global top-10 + filter)**: `Recall@10 = 0.0133` (global top-10 is dominated by common categories prior to filtering).

### 3. Why Searching Too Few Clusters Misses a Match (`TestIVFTooFewClustersMissesMatch`)
- Left cluster `[-3.0, -0.1, 0.0, 0.1]` has centroid `(-0.75, 0)` and contains `rare-far-in-left` at `(-3.0, 0)`.
- Right cluster `[6.5, 10.0, 10.0, 10.0]` has centroid `(9.125, 0)` and contains `rare-true-nearest` at `(6.5, 0)`.
- Query `(4.0, 0)` is closer to the left centroid (`distSq = 22.56`) than the right centroid (`distSq = 26.27`):
  - `nprobe = 1` scans only the left cluster and returns `rare-far-in-left` (`distSq = 49.00`, `Recall@1 = 0.00`).
  - `nprobe = 2` scans both clusters and recovers `rare-true-nearest` (`distSq = 6.25`, `Recall@1 = 1.00`).
