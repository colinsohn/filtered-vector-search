package vecsearch

import (
	"cmp"
	"context"
	"errors"
	"fmt"
	"math"
	"net/http/httptest"
	"slices"
	"testing"
	"time"
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

func TestIVFTooFewClustersMissesMatch(t *testing.T) {
	// Left cluster [-3.0, -0.1, 0.0, 0.1] has centroid (-0.75, 0).
	// Right cluster [6.5, 10.0, 10.0, 10.0] has centroid (9.125, 0).
	// Query (4.0, 0) is closer to the left centroid (distSq=22.56) than the right centroid (distSq=26.27),
	// so nprobe=1 only scans the left cluster and returns "rare-far-in-left" at (-3.0, 0) (distSq=49.0),
	// missing "rare-true-nearest" at (6.5, 0) in the right cluster (distSq=6.25).
	docs := []Document{
		{ID: "common-L1", Category: "books", Vector: []float32{0.0, 0.0}},
		{ID: "common-L2", Category: "books", Vector: []float32{0.1, 0.0}},
		{ID: "common-L3", Category: "books", Vector: []float32{-0.1, 0.0}},
		{ID: "rare-far-in-left", Category: "rare", Vector: []float32{-3.0, 0.0}},
		{ID: "common-R1", Category: "books", Vector: []float32{10.0, 0.0}},
		{ID: "common-R2", Category: "books", Vector: []float32{10.0, 0.0}},
		{ID: "common-R3", Category: "books", Vector: []float32{10.0, 0.0}},
		{ID: "rare-true-nearest", Category: "rare", Vector: []float32{6.5, 0.0}},
	}
	query := []float32{4.0, 0.0}

	exactIdx, err := NewExactIndex(2, docs)
	if err != nil {
		t.Fatalf("NewExactIndex failed: %v", err)
	}
	ivfIdx, err := NewIVFIndex(2, docs, 2, 7)
	if err != nil {
		t.Fatalf("NewIVFIndex failed: %v", err)
	}

	exactRes, err := exactIdx.Search(query, "rare", 1)
	if err != nil {
		t.Fatalf("ExactIndex.Search failed: %v", err)
	}
	if len(exactRes) != 1 || exactRes[0].ID != "rare-true-nearest" {
		t.Fatalf("expected exact winner rare-true-nearest, got %+v", exactRes)
	}

	probe1Res, eval1, err := ivfIdx.SearchPreFilter(query, "rare", 1, 1)
	if err != nil {
		t.Fatalf("SearchPreFilter(nprobe=1) failed: %v", err)
	}
	recall1, _ := RecallAtK(exactRes, probe1Res)
	if recall1 != 0.0 || len(probe1Res) != 1 || probe1Res[0].ID != "rare-far-in-left" {
		t.Fatalf("expected nprobe=1 to miss rare-true-nearest and return rare-far-in-left, got %+v (recall=%v)", probe1Res, recall1)
	}

	probe2Res, eval2, err := ivfIdx.SearchPreFilter(query, "rare", 1, 2)
	if err != nil {
		t.Fatalf("SearchPreFilter(nprobe=2) failed: %v", err)
	}
	recall2, _ := RecallAtK(exactRes, probe2Res)
	assertSameResults(t, probe2Res, exactRes)
	if recall2 != 1.0 {
		t.Fatalf("expected nprobe=2 recall=1.0, got %v", recall2)
	}

	t.Logf("Boundary miss verified: query=%v exact=%s(dist=%.2f) | nprobe=1 -> %s(dist=%.2f, eval=%d, recall=%.2f) | nprobe=2 -> %s(dist=%.2f, eval=%d, recall=%.2f)",
		query,
		exactRes[0].ID, exactRes[0].Distance,
		probe1Res[0].ID, probe1Res[0].Distance, eval1, recall1,
		probe2Res[0].ID, probe2Res[0].Distance, eval2, recall2)
}

func TestIVFSeededWorkloadRecallAndExhaustiveParity(t *testing.T) {
	const (
		numDocs     = 1000
		numQueries  = 60
		dim         = 16
		numClusters = 16
		k           = 10
	)
	wl := NewSeededWorkload(42, numDocs, numQueries, dim)

	exactIdx, err := NewExactIndex(wl.Dim, wl.Docs)
	if err != nil {
		t.Fatalf("NewExactIndex failed: %v", err)
	}
	ivfIdx, err := NewIVFIndex(wl.Dim, wl.Docs, numClusters, 42)
	if err != nil {
		t.Fatalf("NewIVFIndex failed: %v", err)
	}

	probes := []int{1, 2, 4, 8, numClusters}
	prevMeanRecall := -1.0
	loggedSeededMiss := false

	for _, nprobe := range probes {
		var totalRecall float64
		var nonEmptyCount int
		var totalEval int
		var underfilled int

		for qIdx, q := range wl.Queries {
			for _, cat := range wl.Categories {
				exactRes, _, err := exactIdx.SearchWithStats(q, cat, k)
				if err != nil {
					t.Fatalf("exact search failed: %v", err)
				}
				approxRes, eval, err := ivfIdx.SearchPreFilter(q, cat, k, nprobe)
				if err != nil {
					t.Fatalf("ivf pre-filter search failed: %v", err)
				}

				if nprobe == numClusters {
					assertSameResults(t, approxRes, exactRes)
				}

				rec, empty := RecallAtK(exactRes, approxRes)
				if !empty {
					totalRecall += rec
					nonEmptyCount++
				}
				if len(exactRes) == k && len(approxRes) < k {
					underfilled++
				}
				totalEval += eval

				if nprobe == 1 && !loggedSeededMiss && !empty && rec < 1.0 && len(approxRes) > 0 {
					loggedSeededMiss = true
					t.Logf("Seeded workload miss at nprobe=1 (query=%d, category=%q, k=%d): recall=%.2f, exact[0]=%s(dist=%.2f) vs approx[0]=%s(dist=%.2f)",
						qIdx, cat, k, rec, exactRes[0].ID, exactRes[0].Distance, approxRes[0].ID, approxRes[0].Distance)
				}
			}
		}

		meanRecall := totalRecall / float64(nonEmptyCount)
		meanEval := float64(totalEval) / float64(nonEmptyCount)
		t.Logf("IVF Pre-Filter nprobe=%2d/%d: mean recall@%d = %.4f, mean candidates scored = %5.1f, underfilled = %d/%d",
			nprobe, numClusters, k, meanRecall, meanEval, underfilled, nonEmptyCount)

		if meanRecall+1e-9 < prevMeanRecall {
			t.Fatalf("expected non-decreasing recall with nprobe: nprobe=%d got %.4f < %.4f", nprobe, meanRecall, prevMeanRecall)
		}
		prevMeanRecall = meanRecall

		if nprobe == numClusters && math.Abs(meanRecall-1.0) > 1e-9 {
			t.Fatalf("expected 1.0 recall when probing all clusters, got %.6f", meanRecall)
		}
	}

	// Compare global post-filtering at nprobe=4 on rare category ("rare_vintage").
	var preRareRecall, postRareRecall float64
	var preRareUnderfilled, postRareUnderfilled int
	for _, q := range wl.Queries {
		exactRes, _ := exactIdx.Search(q, "rare_vintage", k)
		preRes, _, _ := ivfIdx.SearchPreFilter(q, "rare_vintage", k, 4)
		postRes, _, _ := ivfIdx.SearchPostFilter(q, "rare_vintage", k, 4, k)
		rPre, _ := RecallAtK(exactRes, preRes)
		rPost, _ := RecallAtK(exactRes, postRes)
		preRareRecall += rPre
		postRareRecall += rPost
		if len(exactRes) == k && len(preRes) < k {
			preRareUnderfilled++
		}
		if len(exactRes) == k && len(postRes) < k {
			postRareUnderfilled++
		}
	}
	preRareRecall /= float64(len(wl.Queries))
	postRareRecall /= float64(len(wl.Queries))
	t.Logf("Rare category (%q, nprobe=4, k=%d): pre-filter recall=%.4f (underfilled=%d/%d) vs post-filter(top-%d) recall=%.4f (underfilled=%d/%d)",
		"rare_vintage", k, preRareRecall, preRareUnderfilled, len(wl.Queries), k, postRareRecall, postRareUnderfilled, len(wl.Queries))
	if preRareRecall <= postRareRecall {
		t.Fatalf("expected pre-filter recall (%.4f) to beat global top-k post-filter recall (%.4f) on rare category", preRareRecall, postRareRecall)
	}
}

func TestIVFValidation(t *testing.T) {
	docs := []Document{{ID: "d1", Category: "a", Vector: []float32{1, 2}}}
	if _, err := NewIVFIndex(0, docs, 2, 1); !errors.Is(err, ErrInvalidDimension) {
		t.Fatalf("expected ErrInvalidDimension, got %v", err)
	}
	if _, err := NewIVFIndex(2, docs, 0, 1); !errors.Is(err, ErrInvalidClusterCount) {
		t.Fatalf("expected ErrInvalidClusterCount, got %v", err)
	}
	idx, err := NewIVFIndex(2, docs, 4, 1)
	if err != nil {
		t.Fatalf("NewIVFIndex failed: %v", err)
	}
	if idx.NumClusters() != 1 {
		t.Fatalf("expected NumClusters clamped to 1, got %d", idx.NumClusters())
	}
	if _, _, err := idx.SearchPreFilter([]float32{1, 2}, "a", 1, 0); !errors.Is(err, ErrInvalidNProbe) {
		t.Fatalf("expected ErrInvalidNProbe, got %v", err)
	}
	if _, _, err := idx.SearchPostFilter([]float32{1, 2}, "a", 1, 0, 5); !errors.Is(err, ErrInvalidNProbe) {
		t.Fatalf("expected ErrInvalidNProbe, got %v", err)
	}
}

func TestPlannersConcentratedVsDispersed(t *testing.T) {
	wl := NewBenchmarkWorkload(42, 6000, 32, 32, 80, 200)
	engine, err := NewEngine(wl.Dim, wl.Docs, wl.NumClusters, 42)
	if err != nil {
		t.Fatalf("NewEngine failed: %v", err)
	}

	const k = 10
	exactTruth, err := ComputeExactGroundTruth(engine.Exact, wl.HeldOutQueries, k)
	if err != nil {
		t.Fatalf("ComputeExactGroundTruth failed: %v", err)
	}

	ctx := context.Background()
	sizePlanner := DefaultSizeOnlyPlanner()
	clusterPlanner := DefaultClusterAwarePlanner()

	sizeMetrics, err := EvaluateMethod("size_only", wl.HeldOutQueries, exactTruth, k, 0, 1,
		func(q WorkloadQuery) ([]Result, Route, int, int, error) {
			res, plan, eval, err := engine.Execute(ctx, sizePlanner, q.Vector, q.Category, k)
			return res, plan.Route, plan.NProbe, eval, err
		})
	if err != nil {
		t.Fatalf("size_only eval failed: %v", err)
	}

	clusterMetrics, err := EvaluateMethod("cluster_aware", wl.HeldOutQueries, exactTruth, k, 0, 1,
		func(q WorkloadQuery) ([]Result, Route, int, int, error) {
			res, plan, eval, err := engine.Execute(ctx, clusterPlanner, q.Vector, q.Category, k)
			return res, plan.Route, plan.NProbe, eval, err
		})
	if err != nil {
		t.Fatalf("cluster_aware eval failed: %v", err)
	}

	// Verify cat_concentrated and cat_dispersed have identical corpus counts.
	concDocs := engine.Exact.CategoryCount("cat_concentrated")
	dispDocs := engine.Exact.CategoryCount("cat_dispersed")
	if concDocs != dispDocs {
		t.Fatalf("expected equal category sizes, got concentrated=%d vs dispersed=%d", concDocs, dispDocs)
	}

	sizeDisp := sizeMetrics.ByCategory["cat_dispersed"]
	clusterDisp := clusterMetrics.ByCategory["cat_dispersed"]
	sizeConc := sizeMetrics.ByCategory["cat_concentrated"]
	clusterConc := clusterMetrics.ByCategory["cat_concentrated"]

	t.Logf("cat_concentrated (docs=%d): size_only recall=%.4f (underfill=%.4f) vs cluster_aware recall=%.4f (underfill=%.4f)",
		concDocs, sizeConc.RecallAtK, sizeConc.UnderfillRate, clusterConc.RecallAtK, clusterConc.UnderfillRate)
	t.Logf("cat_dispersed    (docs=%d): size_only recall=%.4f (underfill=%.4f) vs cluster_aware recall=%.4f (underfill=%.4f)",
		dispDocs, sizeDisp.RecallAtK, sizeDisp.UnderfillRate, clusterDisp.RecallAtK, clusterDisp.UnderfillRate)

	if clusterDisp.RecallAtK <= sizeDisp.RecallAtK {
		t.Fatalf("expected cluster_aware recall (%.4f) > size_only recall (%.4f) on cat_dispersed",
			clusterDisp.RecallAtK, sizeDisp.RecallAtK)
	}
	if clusterConc.RecallAtK < sizeConc.RecallAtK {
		t.Fatalf("expected cluster_aware recall (%.4f) >= size_only recall (%.4f) on cat_concentrated",
			clusterConc.RecallAtK, sizeConc.RecallAtK)
	}
}

func TestClusterAwarePlannerFailureCase(t *testing.T) {
	// Failure case: Cluster 0 (around x=-1) contains 50 "target_cat" docs at (-1, 0),
	// satisfying the planner's candidate count threshold in a single cluster.
	// Cluster 1 (around x=10) is dominated by "other" docs at (10.5, 0) plus one
	// "target_cat" boundary outlier ("target-boundary-winner") at (5.5, 0).
	// Query (4.0, 0) is closer to Cluster 0's centroid (-1, 0) than Cluster 1's centroid (~10.4, 0),
	// and because Cluster 0 already holds 50 eligible docs, ClusterAwarePlanner (with BaseNProbe=1)
	// stops after Cluster 0 and misses "target-boundary-winner" in Cluster 1.
	var docs []Document
	for i := range 50 {
		docs = append(docs, Document{
			ID:       fmt.Sprintf("target-c0-%02d", i),
			Category: "target_cat",
			Vector:   []float32{-1.0, 0.0},
		})
	}
	for i := range 49 {
		docs = append(docs, Document{
			ID:       fmt.Sprintf("other-c1-%02d", i),
			Category: "other",
			Vector:   []float32{10.5, 0.0},
		})
	}
	docs = append(docs, Document{
		ID:       "target-boundary-winner",
		Category: "target_cat",
		Vector:   []float32{5.5, 0.0},
	})

	engine, err := NewEngine(2, docs, 2, 7)
	if err != nil {
		t.Fatalf("NewEngine failed: %v", err)
	}

	planner := ClusterAwarePlanner{
		MaxExactSelectivity: 0.01,
		MaxExactDocs:        5,
		BaseNProbe:          1,
		MaxNProbe:           2,
		MaxScanClusters:     2,
		TargetMultipleOfK:   10,
	}

	query := []float32{4.0, 0.0}
	exactRes, err := engine.Exact.Search(query, "target_cat", 1)
	if err != nil {
		t.Fatalf("Exact.Search failed: %v", err)
	}
	if exactRes[0].ID != "target-boundary-winner" {
		t.Fatalf("expected exact winner target-boundary-winner, got %+v", exactRes)
	}

	plannedRes, plan, eval, err := engine.Execute(context.Background(), planner, query, "target_cat", 1)
	if err != nil {
		t.Fatalf("Engine.Execute failed: %v", err)
	}
	rec, _ := RecallAtK(exactRes, plannedRes)
	if plan.NProbe != 1 || rec != 0.0 {
		t.Fatalf("expected planner to stop at nprobe=1 (50 eligible docs in nearest cluster) and miss boundary winner, got plan=%+v res=%+v recall=%v",
			plan, plannedRes, rec)
	}
	t.Logf("Planner failure case verified: nearest cluster has %d eligible docs >= target, so planner chose nprobe=%d (eval=%d) and returned %s(dist=%.2f) instead of %s(dist=%.2f)",
		plan.ExpectedEligible, plan.NProbe, eval, plannedRes[0].ID, plannedRes[0].Distance, exactRes[0].ID, exactRes[0].Distance)
}

func TestSearchCancellation(t *testing.T) {
	wl := NewSeededWorkload(42, 1000, 5, 16)
	engine, err := NewEngine(wl.Dim, wl.Docs, 8, 42)
	if err != nil {
		t.Fatalf("NewEngine failed: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	if _, _, err := engine.Exact.SearchContext(ctx, wl.Queries[0], "electronics", 10); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled from ExactIndex.SearchContext, got %v", err)
	}
	if _, _, err := engine.IVF.SearchPreFilterContext(ctx, wl.Queries[0], "electronics", 10, 4); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled from IVFIndex.SearchPreFilterContext, got %v", err)
	}
	if _, _, _, err := engine.Execute(ctx, DefaultClusterAwarePlanner(), wl.Queries[0], "electronics", 10); !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled from Engine.Execute, got %v", err)
	}
}

func TestServerBoundedConcurrencyAndLoad(t *testing.T) {
	wl := NewBenchmarkWorkload(42, 2000, 16, 16, 20, 50)
	engine, err := NewEngine(wl.Dim, wl.Docs, wl.NumClusters, 42)
	if err != nil {
		t.Fatalf("NewEngine failed: %v", err)
	}

	srv := httptest.NewServer(NewSearchServer(engine, DefaultClusterAwarePlanner(), 8, 200*time.Millisecond))
	defer srv.Close()

	report, err := RunHTTPLoadTest(srv.URL, wl.HeldOutQueries, 10, 4, 8, 200, 200*time.Millisecond)
	if err != nil {
		t.Fatalf("RunHTTPLoadTest failed: %v", err)
	}
	if report.Succeeded == 0 || report.OtherErrors > 0 {
		t.Fatalf("unexpected load test report: %+v", report)
	}
	if report.EngineP50Us <= 0 || report.HTTPClientP50Us <= report.EngineP50Us {
		t.Fatalf("expected separate positive engine (%v us) < http (%v us) latencies", report.EngineP50Us, report.HTTPClientP50Us)
	}
}

func BenchmarkExactSearch(b *testing.B) {
	wl := NewSeededWorkload(42, 10000, 100, 64)
	idx, err := NewExactIndex(wl.Dim, wl.Docs)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q := wl.Queries[i%len(wl.Queries)]
		if _, err := idx.Search(q, "electronics", 10); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkIVFPreFilterSearch(b *testing.B) {
	wl := NewSeededWorkload(42, 10000, 100, 64)
	idx, err := NewIVFIndex(wl.Dim, wl.Docs, 64, 42)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q := wl.Queries[i%len(wl.Queries)]
		if _, _, err := idx.SearchPreFilter(q, "electronics", 10, 4); err != nil {
			b.Fatal(err)
		}
	}
}


