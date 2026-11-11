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
	ErrInvalidDimension = errors.New("invalid vector dimension")
	ErrInvalidVector    = errors.New("vector contains NaN or Inf")
	ErrEmptyID          = errors.New("document ID must not be empty")
	ErrDuplicateID      = errors.New("duplicate document ID")
	ErrInvalidK         = errors.New("k must be positive")
)

// Document represents an immutable record in the corpus.
type Document struct {
	ID       string
	Category string
	Vector   []float32
}

// Result represents a scored search match.
type Result struct {
	ID       string
	Distance float64
}

// ExactIndex stores a read-only corpus in a contiguous float32 arena with a
// per-category inverted index for exact filtered nearest-neighbor search.
type ExactIndex struct {
	dim        int
	ids        []string
	vectors    []float32
	byCategory map[string][]int
}

// NewExactIndex validates the corpus and builds an immutable exact filtered search index.
func NewExactIndex(dim int, docs []Document) (*ExactIndex, error) {
	ids, categories, vectors, err := validateAndCopyCorpus(dim, docs)
	if err != nil {
		return nil, err
	}

	byCategory := make(map[string][]int)
	for i, cat := range categories {
		byCategory[cat] = append(byCategory[cat], i)
	}

	return &ExactIndex{
		dim:        dim,
		ids:        ids,
		vectors:    vectors,
		byCategory: byCategory,
	}, nil
}

func validateAndCopyCorpus(dim int, docs []Document) ([]string, []string, []float32, error) {
	if dim <= 0 {
		return nil, nil, nil, fmt.Errorf("%w: dim=%d", ErrInvalidDimension, dim)
	}

	seen := make(map[string]struct{}, len(docs))
	ids := make([]string, len(docs))
	categories := make([]string, len(docs))
	vectors := make([]float32, len(docs)*dim)

	for i, doc := range docs {
		if doc.ID == "" {
			return nil, nil, nil, fmt.Errorf("%w at index %d", ErrEmptyID, i)
		}
		if _, exists := seen[doc.ID]; exists {
			return nil, nil, nil, fmt.Errorf("%w: %q", ErrDuplicateID, doc.ID)
		}
		if err := validateVector(dim, doc.Vector); err != nil {
			return nil, nil, nil, fmt.Errorf("doc %q: %w", doc.ID, err)
		}

		seen[doc.ID] = struct{}{}
		ids[i] = doc.ID
		categories[i] = doc.Category
		copy(vectors[i*dim:(i+1)*dim], doc.Vector)
	}

	return ids, categories, vectors, nil
}

// Search returns up to k closest documents in category ordered by (Distance ASC, ID ASC).
func (idx *ExactIndex) Search(query []float32, category string, k int) ([]Result, error) {
	results, _, err := idx.SearchWithStats(query, category, k)
	return results, err
}

// SearchWithStats returns up to k closest documents in category along with the
// number of candidate document vectors scored.
func (idx *ExactIndex) SearchWithStats(query []float32, category string, k int) ([]Result, int, error) {
	if k <= 0 {
		return nil, 0, fmt.Errorf("%w: k=%d", ErrInvalidK, k)
	}
	if err := validateVector(idx.dim, query); err != nil {
		return nil, 0, fmt.Errorf("query: %w", err)
	}

	postings := idx.byCategory[category]
	if len(postings) == 0 {
		return []Result{}, 0, nil
	}

	target := min(k, len(postings))
	h := make(resultMaxHeap, 0, target)

	for _, docIdx := range postings {
		vec := idx.vectors[docIdx*idx.dim : (docIdx+1)*idx.dim]
		cand := Result{
			ID:       idx.ids[docIdx],
			Distance: squaredL2(query, vec),
		}
		if len(h) < target {
			heap.Push(&h, cand)
		} else if betterResult(cand, h[0]) {
			h[0] = cand
			heap.Fix(&h, 0)
		}
	}

	results := []Result(h)
	slices.SortFunc(results, compareResult)
	return results, len(postings), nil
}

func validateVector(dim int, vec []float32) error {
	if len(vec) != dim {
		return fmt.Errorf("%w: got %d, want %d", ErrInvalidDimension, len(vec), dim)
	}
	for _, v := range vec {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return ErrInvalidVector
		}
	}
	return nil
}

// squaredL2 converts coordinates to float64 before subtraction so large float32
// magnitudes neither overflow during subtraction nor saturate to +Inf when squared.
func squaredL2(a, b []float32) float64 {
	var sum float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		sum += d * d
	}
	return sum
}

func compareResult(a, b Result) int {
	if c := cmp.Compare(a.Distance, b.Distance); c != 0 {
		return c
	}
	return cmp.Compare(a.ID, b.ID)
}

func betterResult(a, b Result) bool {
	return compareResult(a, b) < 0
}

// resultMaxHeap keeps the worst (largest Distance, or largest ID on tie) at index 0.
type resultMaxHeap []Result

func (h resultMaxHeap) Len() int           { return len(h) }
func (h resultMaxHeap) Less(i, j int) bool { return betterResult(h[j], h[i]) }
func (h resultMaxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *resultMaxHeap) Push(x any)        { *h = append(*h, x.(Result)) }
func (h *resultMaxHeap) Pop() any {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}

// SeededWorkload holds a deterministic synthetic corpus and query set.
type SeededWorkload struct {
	Dim        int
	Docs       []Document
	Categories []string
	Queries    [][]float32
}

// NewSeededWorkload generates a deterministic synthetic corpus around latent
// cluster centers with skewed category sizes and queries for baseline and recall verification.
func NewSeededWorkload(seed uint64, numDocs, numQueries, dim int) SeededWorkload {
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	categories := []string{"electronics", "books", "home", "apparel", "rare_vintage"}
	// Skewed weights: electronics (45%), books (25%), home (18%), apparel (10%), rare_vintage (2%)
	cumWeights := []float64{0.45, 0.70, 0.88, 0.98, 1.00}

	const numCenters = 12
	centers := make([][]float32, numCenters)
	for c := range numCenters {
		center := make([]float32, dim)
		for j := range dim {
			center[j] = float32(rng.NormFloat64() * 4.0)
		}
		centers[c] = center
	}

	docs := make([]Document, numDocs)
	for i := range numDocs {
		p := rng.Float64()
		cat := categories[len(categories)-1]
		for cIdx, w := range cumWeights {
			if p < w {
				cat = categories[cIdx]
				break
			}
		}
		center := centers[rng.IntN(numCenters)]
		vec := make([]float32, dim)
		for j := range dim {
			vec[j] = center[j] + float32(rng.NormFloat64()*0.75)
		}
		docs[i] = Document{
			ID:       fmt.Sprintf("doc-%05d", i),
			Category: cat,
			Vector:   vec,
		}
	}

	queries := make([][]float32, numQueries)
	for i := range numQueries {
		center := centers[rng.IntN(numCenters)]
		q := make([]float32, dim)
		for j := range dim {
			q[j] = center[j] + float32(rng.NormFloat64()*0.75)
		}
		queries[i] = q
	}

	return SeededWorkload{
		Dim:        dim,
		Docs:       docs,
		Categories: categories,
		Queries:    queries,
	}
}
