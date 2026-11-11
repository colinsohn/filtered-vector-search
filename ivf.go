package vecsearch

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"slices"
)

var (
	ErrInvalidClusterCount = errors.New("numClusters must be positive")
	ErrInvalidNProbe       = errors.New("nprobe must be positive")
)

const maxKMeansIterations = 15

// IVFIndex groups corpus vectors around k-means centroids in a contiguous float32
// arena so queries can scan a selected number of nearby clusters (nprobe).
type IVFIndex struct {
	dim               int
	ids               []string
	categories        []string
	vectors           []float32
	centroids         []float32        // len = numClusters * dim
	clusters          [][]int          // clusters[c] holds document indices assigned to centroid c
	categoryHistograms map[string][]int // categoryHistograms[cat][c] holds eligible doc count in cluster c
}

// NewIVFIndex validates docs, trains deterministic k-means centroids using seed,
// partitions documents into inverted lists by nearest centroid, and precomputes
// per-centroid category histograms for cluster-aware query planning.
func NewIVFIndex(dim int, docs []Document, numClusters int, seed uint64) (*IVFIndex, error) {
	if numClusters <= 0 {
		return nil, fmt.Errorf("%w: numClusters=%d", ErrInvalidClusterCount, numClusters)
	}
	ids, categories, vectors, err := validateAndCopyCorpus(dim, docs)
	if err != nil {
		return nil, err
	}
	if len(docs) == 0 {
		return &IVFIndex{
			dim:                dim,
			ids:                ids,
			categories:         categories,
			vectors:            vectors,
			categoryHistograms: make(map[string][]int),
		}, nil
	}

	actualClusters := min(numClusters, len(docs))
	centroids := trainKMeans(dim, vectors, len(docs), actualClusters, seed)
	clusters := make([][]int, actualClusters)
	categoryHistograms := make(map[string][]int)

	for docIdx := range len(docs) {
		vec := vectors[docIdx*dim : (docIdx+1)*dim]
		c := nearestCentroid(dim, centroids, actualClusters, vec)
		clusters[c] = append(clusters[c], docIdx)

		cat := categories[docIdx]
		hist, ok := categoryHistograms[cat]
		if !ok {
			hist = make([]int, actualClusters)
			categoryHistograms[cat] = hist
		}
		hist[c]++
	}

	return &IVFIndex{
		dim:                dim,
		ids:                ids,
		categories:         categories,
		vectors:            vectors,
		centroids:          centroids,
		clusters:           clusters,
		categoryHistograms: categoryHistograms,
	}, nil
}

// NumClusters returns the number of trained clusters in the index.
func (idx *IVFIndex) NumClusters() int {
	return len(idx.clusters)
}

// ClusterSize returns the total number of documents in cluster c.
func (idx *IVFIndex) ClusterSize(c int) int {
	if c < 0 || c >= len(idx.clusters) {
		return 0
	}
	return len(idx.clusters[c])
}

// CategoryClusterCounts returns the per-centroid document count slice for category,
// or nil if category does not appear in the corpus.
func (idx *IVFIndex) CategoryClusterCounts(category string) []int {
	return idx.categoryHistograms[category]
}

// SearchPreFilter probes the nprobe nearest clusters to query and scores only
// documents matching category within those clusters.
func (idx *IVFIndex) SearchPreFilter(query []float32, category string, k, nprobe int) ([]Result, int, error) {
	return idx.SearchPreFilterContext(context.Background(), query, category, k, nprobe)
}

// SearchPreFilterContext probes the nprobe nearest clusters to query with context cancellation.
func (idx *IVFIndex) SearchPreFilterContext(ctx context.Context, query []float32, category string, k, nprobe int) ([]Result, int, error) {
	if k <= 0 {
		return nil, 0, fmt.Errorf("%w: k=%d", ErrInvalidK, k)
	}
	if nprobe <= 0 {
		return nil, 0, fmt.Errorf("%w: nprobe=%d", ErrInvalidNProbe, nprobe)
	}
	if err := validateVector(idx.dim, query); err != nil {
		return nil, 0, fmt.Errorf("query: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	if len(idx.clusters) == 0 {
		return []Result{}, 0, nil
	}

	var stackOrder [128]int
	probeOrder := idx.rankClustersInto(query, nprobe, stackOrder[:0])
	return idx.searchSelectedClustersContext(ctx, query, category, k, probeOrder)
}

func (idx *IVFIndex) searchSelectedClustersContext(ctx context.Context, query []float32, category string, k int, clustersToProbe []int) ([]Result, int, error) {
	results := make([]Result, 0, k)
	evaluated := 0
	inspected := 0

	for _, c := range clustersToProbe {
		for _, docIdx := range idx.clusters[c] {
			inspected++
			if inspected&255 == 0 {
				if err := ctx.Err(); err != nil {
					return nil, evaluated, err
				}
			}
			if idx.categories[docIdx] != category {
				continue
			}
			vec := idx.vectors[docIdx*idx.dim : (docIdx+1)*idx.dim]
			evaluated++
			cand := Result{
				ID:       idx.ids[docIdx],
				Distance: squaredL2(query, vec),
			}
			if len(results) < k {
				results = pushResultHeap(results, cand)
			} else if betterResult(cand, results[0]) {
				results[0] = cand
				siftDownResultHeap(results, 0)
			}
		}
	}

	slices.SortFunc(results, compareResult)
	return results, evaluated, nil
}

// SearchPostFilter probes the nprobe nearest clusters without checking the category
// filter during scoring, retains the global top globalCandidates matches across all
// categories, and filters by category afterward.
func (idx *IVFIndex) SearchPostFilter(query []float32, category string, k, nprobe, globalCandidates int) ([]Result, int, error) {
	return idx.SearchPostFilterContext(context.Background(), query, category, k, nprobe, globalCandidates)
}

// SearchPostFilterContext executes post-filtered IVF search with context cancellation.
func (idx *IVFIndex) SearchPostFilterContext(ctx context.Context, query []float32, category string, k, nprobe, globalCandidates int) ([]Result, int, error) {
	if k <= 0 {
		return nil, 0, fmt.Errorf("%w: k=%d", ErrInvalidK, k)
	}
	if nprobe <= 0 {
		return nil, 0, fmt.Errorf("%w: nprobe=%d", ErrInvalidNProbe, nprobe)
	}
	if err := validateVector(idx.dim, query); err != nil {
		return nil, 0, fmt.Errorf("query: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, 0, err
	}
	if len(idx.clusters) == 0 {
		return []Result{}, 0, nil
	}
	if globalCandidates < k {
		globalCandidates = k
	}

	type docCandidate struct {
		docIdx int
		result Result
	}

	var stackOrder [128]int
	probeOrder := idx.rankClustersInto(query, nprobe, stackOrder[:0])
	var scored []docCandidate
	for _, c := range probeOrder {
		for _, docIdx := range idx.clusters[c] {
			if len(scored)&255 == 0 && len(scored) > 0 {
				if err := ctx.Err(); err != nil {
					return nil, len(scored), err
				}
			}
			vec := idx.vectors[docIdx*idx.dim : (docIdx+1)*idx.dim]
			scored = append(scored, docCandidate{
				docIdx: docIdx,
				result: Result{
					ID:       idx.ids[docIdx],
					Distance: squaredL2(query, vec),
				},
			})
		}
	}
	evaluated := len(scored)

	// Sort the probed candidate slice directly; post-filtering already materializes the
	// unscored cluster subset, so a full-slice sort avoids maintaining a second heap type.
	slices.SortFunc(scored, func(a, b docCandidate) int {
		return compareResult(a.result, b.result)
	})
	if len(scored) > globalCandidates {
		scored = scored[:globalCandidates]
	}

	results := make([]Result, 0, min(k, len(scored)))
	for _, cand := range scored {
		if idx.categories[cand.docIdx] == category {
			results = append(results, cand.result)
			if len(results) == k {
				break
			}
		}
	}
	return results, evaluated, nil
}

type clusterDistance struct {
	clusterIdx int
	distance   float64
}

func (idx *IVFIndex) rankClustersInto(query []float32, nprobe int, dst []int) []int {
	numClusters := len(idx.clusters)
	var stackRanked [128]clusterDistance
	var ranked []clusterDistance
	if numClusters <= len(stackRanked) {
		ranked = stackRanked[:numClusters]
	} else {
		ranked = make([]clusterDistance, numClusters)
	}

	for c := range numClusters {
		centroid := idx.centroids[c*idx.dim : (c+1)*idx.dim]
		ranked[c] = clusterDistance{
			clusterIdx: c,
			distance:   squaredL2(query, centroid),
		}
	}
	slices.SortFunc(ranked, func(a, b clusterDistance) int {
		if cmpDist := cmp.Compare(a.distance, b.distance); cmpDist != 0 {
			return cmpDist
		}
		return cmp.Compare(a.clusterIdx, b.clusterIdx)
	})

	limit := min(nprobe, numClusters)
	for i := range limit {
		dst = append(dst, ranked[i].clusterIdx)
	}
	return dst
}

// RecallAtK computes the fraction of exact eligible top-k results recovered by approx.
// When exact is empty, empty is true and recall is 0.
func RecallAtK(exact, approx []Result) (recall float64, empty bool) {
	if len(exact) == 0 {
		return 0, true
	}
	exactIDs := make(map[string]struct{}, len(exact))
	for _, r := range exact {
		exactIDs[r.ID] = struct{}{}
	}
	hits := 0
	limit := min(len(approx), len(exact))
	for i := range limit {
		if _, ok := exactIDs[approx[i].ID]; ok {
			hits++
		}
	}
	return float64(hits) / float64(len(exact)), false
}

// trainKMeans initializes centroids with k-means++ and refines them with Lloyd's algorithm.
func trainKMeans(dim int, vectors []float32, numDocs, numClusters int, seed uint64) []float32 {
	rng := rand.New(rand.NewPCG(seed, seed^0xda1ce5eed1234567))
	centroids := make([]float32, numClusters*dim)

	// k-means++ initialization spreads initial centroids across separated modes so
	// rare clusters are not merged by unlucky uniform picks.
	firstDoc := rng.IntN(numDocs)
	copy(centroids[0:dim], vectors[firstDoc*dim:(firstDoc+1)*dim])

	minDistSq := make([]float64, numDocs)
	for i := range numDocs {
		minDistSq[i] = math.Inf(1)
	}

	for c := 1; c < numClusters; c++ {
		prevCentroid := centroids[(c-1)*dim : c*dim]
		var totalWeight float64
		for i := range numDocs {
			vec := vectors[i*dim : (i+1)*dim]
			d := squaredL2(vec, prevCentroid)
			if d < minDistSq[i] {
				minDistSq[i] = d
			}
			totalWeight += minDistSq[i]
		}

		chosen := 0
		if totalWeight > 0 && !math.IsInf(totalWeight, 0) {
			target := rng.Float64() * totalWeight
			var cum float64
			chosen = numDocs - 1
			for i, w := range minDistSq {
				cum += w
				if cum >= target {
					chosen = i
					break
				}
			}
		} else {
			chosen = c % numDocs
		}
		copy(centroids[c*dim:(c+1)*dim], vectors[chosen*dim:(chosen+1)*dim])
	}

	assignments := make([]int, numDocs)
	for i := range assignments {
		assignments[i] = -1
	}
	sums := make([]float64, numClusters*dim)
	counts := make([]int, numClusters)

	for range maxKMeansIterations {
		changed := false
		clear(sums)
		clear(counts)

		for docIdx := range numDocs {
			vec := vectors[docIdx*dim : (docIdx+1)*dim]
			bestCluster := nearestCentroid(dim, centroids, numClusters, vec)
			if assignments[docIdx] != bestCluster {
				assignments[docIdx] = bestCluster
				changed = true
			}
			counts[bestCluster]++
			base := bestCluster * dim
			for j := range dim {
				sums[base+j] += float64(vec[j])
			}
		}

		for c := range numClusters {
			if counts[c] == 0 {
				continue
			}
			inv := 1.0 / float64(counts[c])
			base := c * dim
			for j := range dim {
				centroids[base+j] = float32(sums[base+j] * inv)
			}
		}

		if !changed {
			break
		}
	}

	return centroids
}

func nearestCentroid(dim int, centroids []float32, numClusters int, vec []float32) int {
	bestCluster := 0
	bestDist := squaredL2(vec, centroids[0:dim])
	for c := 1; c < numClusters; c++ {
		d := squaredL2(vec, centroids[c*dim:(c+1)*dim])
		if d < bestDist {
			bestDist = d
			bestCluster = c
		}
	}
	return bestCluster
}
