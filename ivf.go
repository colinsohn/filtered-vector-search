package vecsearch

import (
	"cmp"
	"container/heap"
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
// arena so queries can scan a fixed number of nearby clusters (nprobe).
type IVFIndex struct {
	dim        int
	ids        []string
	categories []string
	vectors    []float32
	centroids  []float32 // len = numClusters * dim
	clusters   [][]int   // clusters[c] holds document indices assigned to centroid c
}

// NewIVFIndex validates docs, trains deterministic k-means centroids using seed,
// and partitions documents into inverted lists by nearest centroid.
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
			dim:        dim,
			ids:        ids,
			categories: categories,
			vectors:    vectors,
		}, nil
	}

	actualClusters := min(numClusters, len(docs))
	centroids := trainKMeans(dim, vectors, len(docs), actualClusters, seed)
	clusters := make([][]int, actualClusters)
	for docIdx := range len(docs) {
		vec := vectors[docIdx*dim : (docIdx+1)*dim]
		c := nearestCentroid(dim, centroids, actualClusters, vec)
		clusters[c] = append(clusters[c], docIdx)
	}

	return &IVFIndex{
		dim:        dim,
		ids:        ids,
		categories: categories,
		vectors:    vectors,
		centroids:  centroids,
		clusters:   clusters,
	}, nil
}

// NumClusters returns the number of trained clusters in the index.
func (idx *IVFIndex) NumClusters() int {
	return len(idx.clusters)
}

// SearchPreFilter probes the nprobe nearest clusters to query and scores only
// documents matching category within those clusters.
func (idx *IVFIndex) SearchPreFilter(query []float32, category string, k, nprobe int) ([]Result, int, error) {
	if k <= 0 {
		return nil, 0, fmt.Errorf("%w: k=%d", ErrInvalidK, k)
	}
	if nprobe <= 0 {
		return nil, 0, fmt.Errorf("%w: nprobe=%d", ErrInvalidNProbe, nprobe)
	}
	if err := validateVector(idx.dim, query); err != nil {
		return nil, 0, fmt.Errorf("query: %w", err)
	}
	if len(idx.clusters) == 0 {
		return []Result{}, 0, nil
	}

	probeOrder := idx.rankClusters(query, nprobe)
	h := make(resultMaxHeap, 0, k)
	evaluated := 0

	for _, c := range probeOrder {
		for _, docIdx := range idx.clusters[c] {
			if idx.categories[docIdx] != category {
				continue
			}
			vec := idx.vectors[docIdx*idx.dim : (docIdx+1)*idx.dim]
			evaluated++
			cand := Result{
				ID:       idx.ids[docIdx],
				Distance: squaredL2(query, vec),
			}
			if len(h) < k {
				heap.Push(&h, cand)
			} else if betterResult(cand, h[0]) {
				h[0] = cand
				heap.Fix(&h, 0)
			}
		}
	}

	results := []Result(h)
	slices.SortFunc(results, compareResult)
	return results, evaluated, nil
}

// SearchPostFilter probes the nprobe nearest clusters without checking the category
// filter during scoring, retains the global top globalCandidates matches across all
// categories, and filters by category afterward.
func (idx *IVFIndex) SearchPostFilter(query []float32, category string, k, nprobe, globalCandidates int) ([]Result, int, error) {
	if k <= 0 {
		return nil, 0, fmt.Errorf("%w: k=%d", ErrInvalidK, k)
	}
	if nprobe <= 0 {
		return nil, 0, fmt.Errorf("%w: nprobe=%d", ErrInvalidNProbe, nprobe)
	}
	if err := validateVector(idx.dim, query); err != nil {
		return nil, 0, fmt.Errorf("query: %w", err)
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

	probeOrder := idx.rankClusters(query, nprobe)
	var scored []docCandidate
	for _, c := range probeOrder {
		for _, docIdx := range idx.clusters[c] {
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

	// ponytail: sort probed slice directly for the post-filter baseline; use a bounded heap only if post-filter is run on large nprobe.
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

func (idx *IVFIndex) rankClusters(query []float32, nprobe int) []int {
	numClusters := len(idx.clusters)
	ranked := make([]clusterDistance, numClusters)
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
	out := make([]int, limit)
	for i := range limit {
		out[i] = ranked[i].clusterIdx
	}
	return out
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
