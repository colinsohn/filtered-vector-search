package vecsearch

import (
	"context"
	"fmt"
)

// Route identifies the execution path chosen for a filtered vector query.
type Route string

const (
	RouteExact        Route = "exact"
	RouteIVFPreFilter Route = "ivf_prefilter"
)

// PlanDecision records the search path and cluster probe set chosen by a planner.
type PlanDecision struct {
	Route            Route
	NProbe           int
	ClustersToProbe  []int
	ExpectedEligible int
}

// SearchReport contains the ranked results and execution telemetry for a query.
type SearchReport struct {
	Results          []Result `json:"results"`
	Route            Route    `json:"route"`
	NProbe           int      `json:"nprobe"`
	Evaluated        int      `json:"evaluated"`
	EngineDurationUs int64    `json:"engine_duration_us"`
}

// Planner chooses an execution plan for a filtered vector query.
type Planner interface {
	Name() string
	Plan(engine *Engine, query []float32, category string, k int) PlanDecision
}

// Engine holds an immutable ExactIndex and IVFIndex over the same corpus.
type Engine struct {
	Exact *ExactIndex
	IVF   *IVFIndex
}

// NewEngine builds both the exact category index and the IVF cluster index over docs.
func NewEngine(dim int, docs []Document, numClusters int, seed uint64) (*Engine, error) {
	exact, err := NewExactIndex(dim, docs)
	if err != nil {
		return nil, err
	}
	ivf, err := NewIVFIndex(dim, docs, numClusters, seed)
	if err != nil {
		return nil, err
	}
	return &Engine{
		Exact: exact,
		IVF:   ivf,
	}, nil
}

// Execute runs query using the decision produced by planner and respects ctx cancellation.
func (e *Engine) Execute(ctx context.Context, planner Planner, query []float32, category string, k int) ([]Result, PlanDecision, int, error) {
	if k <= 0 {
		return nil, PlanDecision{}, 0, fmt.Errorf("%w: k=%d", ErrInvalidK, k)
	}
	if err := validateVector(e.Exact.Dim(), query); err != nil {
		return nil, PlanDecision{}, 0, fmt.Errorf("query: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, PlanDecision{}, 0, err
	}

	plan := planner.Plan(e, query, category, k)
	if plan.Route == RouteExact {
		res, eval, err := e.Exact.SearchContext(ctx, query, category, k)
		return res, plan, eval, err
	}

	if len(plan.ClustersToProbe) > 0 {
		res, eval, err := e.IVF.searchSelectedClustersContext(ctx, query, category, k, plan.ClustersToProbe)
		return res, plan, eval, err
	}
	res, eval, err := e.IVF.SearchPreFilterContext(ctx, query, category, k, plan.NProbe)
	return res, plan, eval, err
}

// SizeOnlyPlanner switches between exact category scan and a fixed-nprobe IVF search
// using only the total document count in the target category.
type SizeOnlyPlanner struct {
	// MaxExactSelectivity routes categories with (catDocs / totalDocs) <= threshold to exact scan.
	MaxExactSelectivity float64
	// MaxExactDocs is a minimum floor on exact scan threshold for small corpora.
	MaxExactDocs int
	// BaseNProbe is the fixed cluster probe budget used when routing to IVF.
	BaseNProbe int
}

// DefaultSizeOnlyPlanner returns the frozen size-only planner tuned on development queries.
func DefaultSizeOnlyPlanner() SizeOnlyPlanner {
	return SizeOnlyPlanner{
		MaxExactSelectivity: 0.015, // <= 1.5% of corpus uses exact category posting scan
		MaxExactDocs:        50,
		BaseNProbe:          4,
	}
}

func (p SizeOnlyPlanner) Name() string { return "size_only" }

func (p SizeOnlyPlanner) exactThreshold(totalDocs int) int {
	bySel := int(float64(totalDocs) * p.MaxExactSelectivity)
	return max(p.MaxExactDocs, bySel)
}

func (p SizeOnlyPlanner) Plan(engine *Engine, query []float32, category string, k int) PlanDecision {
	catDocs := engine.Exact.CategoryCount(category)
	if catDocs <= p.exactThreshold(engine.Exact.NumDocs()) {
		return PlanDecision{
			Route:            RouteExact,
			NProbe:           0,
			ExpectedEligible: catDocs,
		}
	}
	nprobe := max(1, min(p.BaseNProbe, engine.IVF.NumClusters()))
	return PlanDecision{
		Route:            RouteIVFPreFilter,
		NProbe:           nprobe,
		ExpectedEligible: catDocs * nprobe / max(1, engine.IVF.NumClusters()),
	}
}

// ClusterAwarePlanner uses per-centroid category histograms along the query's
// ranked centroids to skip empty clusters, expand nprobe when nearby clusters have
// few eligible candidates, or fall back to exact search when nearby clusters underfill k.
type ClusterAwarePlanner struct {
	MaxExactSelectivity float64
	MaxExactDocs        int
	BaseNProbe          int
	MaxNProbe           int
	MaxScanClusters     int
	TargetMultipleOfK   int
}

// DefaultClusterAwarePlanner returns the frozen cluster-aware planner policy tuned on development queries.
func DefaultClusterAwarePlanner() ClusterAwarePlanner {
	return ClusterAwarePlanner{
		MaxExactSelectivity: 0.015,
		MaxExactDocs:        50,
		BaseNProbe:          3,
		MaxNProbe:           14,
		MaxScanClusters:     28,
		TargetMultipleOfK:   8,
	}
}

func (p ClusterAwarePlanner) Name() string { return "cluster_aware" }

func (p ClusterAwarePlanner) exactThreshold(totalDocs int) int {
	bySel := int(float64(totalDocs) * p.MaxExactSelectivity)
	return max(p.MaxExactDocs, bySel)
}

func (p ClusterAwarePlanner) Plan(engine *Engine, query []float32, category string, k int) PlanDecision {
	catDocs := engine.Exact.CategoryCount(category)
	if catDocs <= p.exactThreshold(engine.Exact.NumDocs()) {
		return PlanDecision{
			Route:            RouteExact,
			NProbe:           0,
			ExpectedEligible: catDocs,
		}
	}

	hist := engine.IVF.CategoryClusterCounts(category)
	numClusters := engine.IVF.NumClusters()
	if len(hist) == 0 || numClusters == 0 {
		return PlanDecision{Route: RouteExact}
	}

	var stackOrder [128]int
	scanLimit := min(p.MaxScanClusters, numClusters)
	ranked := engine.IVF.rankClustersInto(query, scanLimit, stackOrder[:0])

	// Aim to score enough eligible candidates to recover top-k accurately, capped at
	// 50% of the category so concentrated categories stop once their populated clusters are covered.
	targetEligible := min(max(k*p.TargetMultipleOfK, 30), max(k, catDocs/2))

	selected := make([]int, 0, p.MaxNProbe)
	expectedEligible := 0

	for _, c := range ranked {
		eligibleInCluster := hist[c]
		if eligibleInCluster == 0 {
			// Skip clusters that contain zero documents of the target category.
			continue
		}
		selected = append(selected, c)
		expectedEligible += eligibleInCluster

		if len(selected) >= p.BaseNProbe && expectedEligible >= targetEligible {
			break
		}
		if len(selected) >= p.MaxNProbe {
			break
		}
	}

	// Fall back to exact scan if even scanning MaxScanClusters nearest centroids
	// cannot gather at least min(k, catDocs) eligible candidates.
	if expectedEligible < min(k, catDocs) {
		return PlanDecision{
			Route:            RouteExact,
			NProbe:           0,
			ExpectedEligible: catDocs,
		}
	}

	return PlanDecision{
		Route:            RouteIVFPreFilter,
		NProbe:           len(selected),
		ClustersToProbe:  selected,
		ExpectedEligible: expectedEligible,
	}
}
