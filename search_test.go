package vecsearch

import (
	"cmp"
	"errors"
	"math"
	"slices"
	"testing"
)

// bruteForceSearch is an independent reference implementation that scans the raw
// Document slice directly, filters inline without an inverted index, and sorts
// all matching candidates without a heap.
func bruteForceSearch(dim int, docs []Document, query []float32, category string, k int) ([]Result, error) {
	if dim <= 0 {
		return nil, ErrInvalidDimension
	}
	if k <= 0 {
		return nil, ErrInvalidK
	}
	seen := make(map[string]struct{}, len(docs))
	for _, d := range docs {
		if d.ID == "" {
			return nil, ErrEmptyID
		}
		if _, ok := seen[d.ID]; ok {
			return nil, ErrDuplicateID
		}
		seen[d.ID] = struct{}{}
		if len(d.Vector) != dim {
			return nil, ErrInvalidDimension
		}
		for _, v := range d.Vector {
			if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
				return nil, ErrInvalidVector
			}
		}
	}
	if len(query) != dim {
		return nil, ErrInvalidDimension
	}
	for _, v := range query {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return nil, ErrInvalidVector
		}
	}

	var matches []Result
	for _, d := range docs {
		if d.Category != category {
			continue
		}
		var dist float64
		for j := 0; j < dim; j++ {
			diff := float64(query[j]) - float64(d.Vector[j])
			dist += diff * diff
		}
		matches = append(matches, Result{ID: d.ID, Distance: dist})
	}

	slices.SortFunc(matches, func(a, b Result) int {
		if c := cmp.Compare(a.Distance, b.Distance); c != 0 {
			return c
		}
		return cmp.Compare(a.ID, b.ID)
	})

	if len(matches) > k {
		matches = matches[:k]
	}
	if matches == nil {
		return []Result{}, nil
	}
	return matches, nil
}

func assertSameResults(t *testing.T, got, want []Result) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("result count mismatch: got %d (%v), want %d (%v)", len(got), got, len(want), want)
	}
	for i := range want {
		if got[i].ID != want[i].ID || got[i].Distance != want[i].Distance {
			t.Fatalf("mismatch at rank %d: got %+v, want %+v", i, got[i], want[i])
		}
	}
}

func TestCategoryExclusion(t *testing.T) {
	docs := []Document{
		{ID: "book-exact-match", Category: "books", Vector: []float32{1, 0, 0}},
		{ID: "elec-close", Category: "electronics", Vector: []float32{1, 1, 0}},
		{ID: "book-near", Category: "books", Vector: []float32{1, 0.1, 0}},
		{ID: "elec-far", Category: "electronics", Vector: []float32{0, 2, 2}},
	}
	query := []float32{1, 0, 0}

	idx, err := NewExactIndex(3, docs)
	if err != nil {
		t.Fatalf("NewExactIndex failed: %v", err)
	}

	got, err := idx.Search(query, "electronics", 2)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	want, err := bruteForceSearch(3, docs, query, "electronics", 2)
	if err != nil {
		t.Fatalf("bruteForceSearch failed: %v", err)
	}

	assertSameResults(t, got, want)
	if got[0].ID != "elec-close" || got[1].ID != "elec-far" {
		t.Fatalf("unexpected IDs: %+v", got)
	}
}

func TestEmptyCategoryAndEmptyCorpus(t *testing.T) {
	docs := []Document{
		{ID: "d1", Category: "books", Vector: []float32{1, 2}},
		{ID: "d2", Category: "books", Vector: []float32{3, 4}},
	}
	query := []float32{1, 2}

	idx, err := NewExactIndex(2, docs)
	if err != nil {
		t.Fatalf("NewExactIndex failed: %v", err)
	}

	got, err := idx.Search(query, "missing_category", 5)
	if err != nil {
		t.Fatalf("Search failed on empty category: %v", err)
	}
	want, err := bruteForceSearch(2, docs, query, "missing_category", 5)
	if err != nil {
		t.Fatalf("bruteForceSearch failed: %v", err)
	}
	assertSameResults(t, got, want)
	if len(got) != 0 {
		t.Fatalf("expected 0 results, got %d", len(got))
	}

	emptyIdx, err := NewExactIndex(2, nil)
	if err != nil {
		t.Fatalf("NewExactIndex on empty corpus failed: %v", err)
	}
	gotEmpty, err := emptyIdx.Search(query, "books", 5)
	if err != nil {
		t.Fatalf("Search on empty corpus failed: %v", err)
	}
	if len(gotEmpty) != 0 {
		t.Fatalf("expected 0 results from empty corpus, got %d", len(gotEmpty))
	}
}

func TestCategoryFewerThanKMatches(t *testing.T) {
	docs := []Document{
		{ID: "rare-2", Category: "rare", Vector: []float32{0, 3}},
		{ID: "common-1", Category: "common", Vector: []float32{0, 0}},
		{ID: "rare-1", Category: "rare", Vector: []float32{0, 1}},
	}
	query := []float32{0, 0}

	idx, err := NewExactIndex(2, docs)
	if err != nil {
		t.Fatalf("NewExactIndex failed: %v", err)
	}

	got, err := idx.Search(query, "rare", 10)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	want, err := bruteForceSearch(2, docs, query, "rare", 10)
	if err != nil {
		t.Fatalf("bruteForceSearch failed: %v", err)
	}

	assertSameResults(t, got, want)
	if len(got) != 2 || got[0].ID != "rare-1" || got[1].ID != "rare-2" {
		t.Fatalf("expected [rare-1, rare-2], got %+v", got)
	}
}

func TestDeterministicTieBreaking(t *testing.T) {
	// All four "tie" documents sit at squared L2 distance 1.0 from query (0, 0).
	// Input order is intentionally scrambled so heap eviction at k=2 must break ties by ID ASC.
	docs := []Document{
		{ID: "doc-d", Category: "tie", Vector: []float32{1, 0}},
		{ID: "doc-b", Category: "tie", Vector: []float32{0, 1}},
		{ID: "doc-a", Category: "tie", Vector: []float32{-1, 0}},
		{ID: "doc-c", Category: "tie", Vector: []float32{0, -1}},
		{ID: "doc-z-far", Category: "tie", Vector: []float32{2, 0}},
	}
	query := []float32{0, 0}

	idx, err := NewExactIndex(2, docs)
	if err != nil {
		t.Fatalf("NewExactIndex failed: %v", err)
	}

	for _, k := range []int{1, 2, 3, 4, 5} {
		got, err := idx.Search(query, "tie", k)
		if err != nil {
			t.Fatalf("Search(k=%d) failed: %v", k, err)
		}
		want, err := bruteForceSearch(2, docs, query, "tie", k)
		if err != nil {
			t.Fatalf("bruteForceSearch(k=%d) failed: %v", k, err)
		}
		assertSameResults(t, got, want)
	}
}

func TestInvalidDimensionsAndVectors(t *testing.T) {
	if _, err := NewExactIndex(0, nil); !errors.Is(err, ErrInvalidDimension) {
		t.Fatalf("expected ErrInvalidDimension for dim=0, got %v", err)
	}
	if _, err := NewExactIndex(-3, nil); !errors.Is(err, ErrInvalidDimension) {
		t.Fatalf("expected ErrInvalidDimension for dim=-3, got %v", err)
	}

	badDocs := []Document{
		{ID: "ok", Category: "a", Vector: []float32{1, 2, 3}},
		{ID: "short", Category: "a", Vector: []float32{1, 2}},
	}
	if _, err := NewExactIndex(3, badDocs); !errors.Is(err, ErrInvalidDimension) {
		t.Fatalf("expected ErrInvalidDimension for short doc vector, got %v", err)
	}

	nanDocs := []Document{
		{ID: "nan", Category: "a", Vector: []float32{1, float32(math.NaN()), 3}},
	}
	if _, err := NewExactIndex(3, nanDocs); !errors.Is(err, ErrInvalidVector) {
		t.Fatalf("expected ErrInvalidVector for NaN doc vector, got %v", err)
	}

	idx, err := NewExactIndex(3, []Document{{ID: "ok", Category: "a", Vector: []float32{1, 2, 3}}})
	if err != nil {
		t.Fatalf("NewExactIndex failed: %v", err)
	}

	if _, err := idx.Search([]float32{1, 2}, "a", 1); !errors.Is(err, ErrInvalidDimension) {
		t.Fatalf("expected ErrInvalidDimension for short query vector, got %v", err)
	}
	if _, err := idx.Search([]float32{1, 2, 3, 4}, "a", 1); !errors.Is(err, ErrInvalidDimension) {
		t.Fatalf("expected ErrInvalidDimension for long query vector, got %v", err)
	}
	if _, err := idx.Search([]float32{1, float32(math.Inf(1)), 3}, "a", 1); !errors.Is(err, ErrInvalidVector) {
		t.Fatalf("expected ErrInvalidVector for Inf query vector, got %v", err)
	}
	if _, err := idx.Search([]float32{1, 2, 3}, "a", 0); !errors.Is(err, ErrInvalidK) {
		t.Fatalf("expected ErrInvalidK for k=0, got %v", err)
	}
}

func TestDuplicateAndEmptyIDs(t *testing.T) {
	sameCatDup := []Document{
		{ID: "dup-1", Category: "books", Vector: []float32{1, 0}},
		{ID: "dup-1", Category: "books", Vector: []float32{0, 1}},
	}
	if _, err := NewExactIndex(2, sameCatDup); !errors.Is(err, ErrDuplicateID) {
		t.Fatalf("expected ErrDuplicateID within same category, got %v", err)
	}

	crossCatDup := []Document{
		{ID: "dup-1", Category: "books", Vector: []float32{1, 0}},
		{ID: "dup-1", Category: "electronics", Vector: []float32{0, 1}},
	}
	if _, err := NewExactIndex(2, crossCatDup); !errors.Is(err, ErrDuplicateID) {
		t.Fatalf("expected ErrDuplicateID across categories, got %v", err)
	}

	emptyID := []Document{
		{ID: "", Category: "books", Vector: []float32{1, 0}},
	}
	if _, err := NewExactIndex(2, emptyID); !errors.Is(err, ErrEmptyID) {
		t.Fatalf("expected ErrEmptyID, got %v", err)
	}
}

func TestFloat32OverflowPrecision(t *testing.T) {
	docs := []Document{
		{ID: "a-far", Category: "overflow", Vector: []float32{math.MaxFloat32}},
		{ID: "z-near", Category: "overflow", Vector: []float32{math.MaxFloat32 / 2}},
	}
	query := []float32{0}

	idx, err := NewExactIndex(1, docs)
	if err != nil {
		t.Fatalf("NewExactIndex failed: %v", err)
	}

	got, err := idx.Search(query, "overflow", 2)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	want, err := bruteForceSearch(1, docs, query, "overflow", 2)
	if err != nil {
		t.Fatalf("bruteForceSearch failed: %v", err)
	}
	assertSameResults(t, got, want)
	if got[0].ID != "z-near" || got[1].ID != "a-far" {
		t.Fatalf("overflow collapsed distances: got order [%s (%v), %s (%v)], want [z-near, a-far]",
			got[0].ID, got[0].Distance, got[1].ID, got[1].Distance)
	}
	if math.IsInf(got[0].Distance, 0) || math.IsInf(got[1].Distance, 0) {
		t.Fatalf("distances overflowed to Inf: %+v", got)
	}
}


func TestSeededWorkloadMatchesBruteForce(t *testing.T) {
	wl := NewSeededWorkload(42, 400, 50, 16)
	idx, err := NewExactIndex(wl.Dim, wl.Docs)
	if err != nil {
		t.Fatalf("NewExactIndex failed: %v", err)
	}

	categories := append(slices.Clone(wl.Categories), "empty_category")
	ks := []int{1, 5, 10, 25}

	for qIdx, q := range wl.Queries {
		for _, cat := range categories {
			for _, k := range ks {
				got, err := idx.Search(q, cat, k)
				if err != nil {
					t.Fatalf("query %d (cat=%q, k=%d) Search failed: %v", qIdx, cat, k, err)
				}
				want, err := bruteForceSearch(wl.Dim, wl.Docs, q, cat, k)
				if err != nil {
					t.Fatalf("query %d (cat=%q, k=%d) bruteForceSearch failed: %v", qIdx, cat, k, err)
				}
				assertSameResults(t, got, want)
			}
		}
	}
}
