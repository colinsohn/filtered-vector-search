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
	if dim <= 0 {
		return nil, fmt.Errorf("%w: dim=%d", ErrInvalidDimension, dim)
	}

	seen := make(map[string]struct{}, len(docs))
	ids := make([]string, len(docs))
	vectors := make([]float32, len(docs)*dim)
	byCategory := make(map[string][]int)

	for i, doc := range docs {
		if doc.ID == "" {
			return nil, fmt.Errorf("%w at index %d", ErrEmptyID, i)
		}
		if _, exists := seen[doc.ID]; exists {
			return nil, fmt.Errorf("%w: %q", ErrDuplicateID, doc.ID)
		}
		if err := validateVector(dim, doc.Vector); err != nil {
			return nil, fmt.Errorf("doc %q: %w", doc.ID, err)
		}

		seen[doc.ID] = struct{}{}
		ids[i] = doc.ID
		copy(vectors[i*dim:(i+1)*dim], doc.Vector)
		byCategory[doc.Category] = append(byCategory[doc.Category], i)
	}

	return &ExactIndex{
		dim:        dim,
		ids:        ids,
		vectors:    vectors,
		byCategory: byCategory,
	}, nil
}

// Search returns up to k closest documents in category ordered by (Distance ASC, ID ASC).
func (idx *ExactIndex) Search(query []float32, category string, k int) ([]Result, error) {
	if k <= 0 {
		return nil, fmt.Errorf("%w: k=%d", ErrInvalidK, k)
	}
	if err := validateVector(idx.dim, query); err != nil {
		return nil, fmt.Errorf("query: %w", err)
	}

	postings := idx.byCategory[category]
	if len(postings) == 0 {
		return []Result{}, nil
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
	return results, nil
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

// NewSeededWorkload generates a deterministic synthetic corpus with skewed category
// sizes and queries for testing and baseline verification.
func NewSeededWorkload(seed uint64, numDocs, numQueries, dim int) SeededWorkload {
	rng := rand.New(rand.NewPCG(seed, seed^0x9e3779b97f4a7c15))
	categories := []string{"electronics", "books", "home", "apparel", "rare_vintage"}
	// Skewed weights: electronics (45%), books (25%), home (18%), apparel (10%), rare_vintage (2%)
	cumWeights := []float64{0.45, 0.70, 0.88, 0.98, 1.00}

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
		vec := make([]float32, dim)
		for j := range dim {
			vec[j] = float32(rng.NormFloat64())
		}
		docs[i] = Document{
			ID:       fmt.Sprintf("doc-%05d", i),
			Category: cat,
			Vector:   vec,
		}
	}

	queries := make([][]float32, numQueries)
	for i := range numQueries {
		q := make([]float32, dim)
		for j := range dim {
			q[j] = float32(rng.NormFloat64())
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
