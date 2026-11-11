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
	MaxExactSelectivity float64 `json:"max_exact_selectivity"`
	// MaxExactDocs is a minimum floor on exact scan threshold for small corpora.
	MaxExactDocs int `json:"max_exact_docs"`
	// BaseNProbe is the fixed cluster probe budget used when routing to IVF.
	BaseNProbe int `json:"base_nprobe"`
}

// DefaultSizeOnlyPlanner returns the default size-only planner baseline.
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
// few eligible candidates, or fall back to exact search when nearby clusters underfill k
// or when scanning dispersed clusters inspects too many non-matching postings.
type ClusterAwarePlanner struct {
	MaxExactSelectivity float64 `json:"max_exact_selectivity"`
	MaxExactDocs        int     `json:"max_exact_docs"`
	BaseNProbe          int     `json:"base_nprobe"`
	MaxNProbe           int     `json:"max_nprobe"`
	MaxScanClusters     int     `json:"max_scan_clusters"`
	TargetMultipleOfK   int     `json:"target_multiple_of_k"`
	MaxClusterScanRatio int     `json:"max_cluster_scan_ratio"`
}

// DefaultClusterAwarePlanner returns the default cluster-aware planner policy.
func DefaultClusterAwarePlanner() ClusterAwarePlanner {
	return ClusterAwarePlanner{
		MaxExactSelectivity: 0.015,
		MaxExactDocs:        50,
		BaseNProbe:          3,
		MaxNProbe:           14,
		MaxScanClusters:     28,
		TargetMultipleOfK:   8,
		MaxClusterScanRatio: 12,
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
	totalClusterDocs := 0

	for _, c := range ranked {
		eligibleInCluster := hist[c]
		if eligibleInCluster == 0 {
			// Skip clusters that contain zero documents of the target category.
			continue
		}
		selected = append(selected, c)
		expectedEligible += eligibleInCluster
		totalClusterDocs += engine.IVF.ClusterSize(c)

		if len(selected) >= p.BaseNProbe && expectedEligible >= targetEligible {
			break
		}
		if len(selected) >= p.MaxNProbe {
			break
		}
	}

	ratioLimit := p.MaxClusterScanRatio
	if ratioLimit <= 0 {
		ratioLimit = 12
	}

	// Fall back to exact scan if scanning MaxScanClusters nearest centroids still
	// cannot gather min(k, catDocs) eligible candidates, or if the probed IVF clusters
	// require inspecting more than ratioLimit * catDocs cluster postings.
	if expectedEligible < min(k, catDocs) || totalClusterDocs > catDocs*ratioLimit {
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

// PlannerTuningReport records how planner hyperparameters were selected on development queries
// and frozen prior to held-out evaluation.
type PlannerTuningReport struct {
	DevQueries           int                 `json:"dev_queries"`
	TargetRecall         float64             `json:"target_recall"`
	SizeCandidatesTested int                 `json:"size_candidates_tested"`
	ClusterConfigsTested int                 `json:"cluster_configs_tested"`
	FrozenSizeOnly       SizeOnlyPlanner     `json:"frozen_size_only"`
	SizeDevRecall        float64             `json:"size_dev_recall"`
	SizeDevEvaluated     float64             `json:"size_dev_evaluated"`
	FrozenClusterAware   ClusterAwarePlanner `json:"frozen_cluster_aware"`
	ClusterDevRecall     float64             `json:"cluster_dev_recall"`
	ClusterDevEvaluated  float64             `json:"cluster_dev_evaluated"`
}

// TunePlannersOnDevQueries sweeps planner hyperparameter grids exclusively on devQueries,
// selects the lowest-work configurations meeting the dev recall targets, and returns frozen
// planners ready for held-out evaluation.
func TunePlannersOnDevQueries(engine *Engine, devQueries []WorkloadQuery, k int, targetRecall float64) (SizeOnlyPlanner, ClusterAwarePlanner, PlannerTuningReport, error) {
	devTruth, err := ComputeExactGroundTruth(engine.Exact, devQueries, k)
	if err != nil {
		return SizeOnlyPlanner{}, ClusterAwarePlanner{}, PlannerTuningReport{}, err
	}
	ctx := context.Background()

	// 1. Tune SizeOnlyPlanner on devQueries (fixing BaseNProbe <= 6 so size-only remains an IVF planner on medium/large categories).
	sizeTarget := max(0.85, targetRecall-0.06)
	var (
		bestSize       = DefaultSizeOnlyPlanner()
		bestSizeRecall = -1.0
		bestSizeEval   = 1e18
		sizeTested     = 0
	)
	for _, maxSel := range []float64{0.010, 0.018} {
		for _, nprobe := range []int{3, 4, 6} {
			cand := SizeOnlyPlanner{
				MaxExactSelectivity: maxSel,
				MaxExactDocs:        50,
				BaseNProbe:          nprobe,
			}
			sizeTested++
			m, err := EvaluateMethod("tune_size", devQueries, devTruth, k, 0, 1,
				func(q WorkloadQuery) ([]Result, Route, int, int, error) {
					res, plan, eval, err := engine.Execute(ctx, cand, q.Vector, q.Category, k)
					return res, plan.Route, plan.NProbe, eval, err
				})
			if err != nil {
				return SizeOnlyPlanner{}, ClusterAwarePlanner{}, PlannerTuningReport{}, err
			}
			meetsOld := bestSizeRecall >= sizeTarget
			meetsNew := m.RecallAtK >= sizeTarget
			if (meetsNew && !meetsOld) ||
				(meetsNew && meetsOld && m.MeanEvaluated < bestSizeEval) ||
				(!meetsNew && !meetsOld && m.RecallAtK > bestSizeRecall) {
				bestSize = cand
				bestSizeRecall = m.RecallAtK
				bestSizeEval = m.MeanEvaluated
			}
		}
	}

	// 2. Tune ClusterAwarePlanner on devQueries to reach targetRecall with zero underfill and minimum candidate evaluations.
	var (
		bestCluster       = DefaultClusterAwarePlanner()
		bestClusterRecall = -1.0
		bestClusterEval   = 1e18
		clusterTested     = 0
	)
	for _, maxSel := range []float64{0.010, 0.018} {
		for _, baseNP := range []int{3, 4} {
			for _, maxNP := range []int{10, 14, 18} {
				for _, multK := range []int{10, 14, 18} {
					for _, ratio := range []int{8, 12} {
						cand := ClusterAwarePlanner{
							MaxExactSelectivity: maxSel,
							MaxExactDocs:        50,
							BaseNProbe:          baseNP,
							MaxNProbe:           maxNP,
							MaxScanClusters:     min(32, maxNP*2),
							TargetMultipleOfK:   multK,
							MaxClusterScanRatio: ratio,
						}
						clusterTested++
						m, err := EvaluateMethod("tune_cluster", devQueries, devTruth, k, 0, 1,
							func(q WorkloadQuery) ([]Result, Route, int, int, error) {
								res, plan, eval, err := engine.Execute(ctx, cand, q.Vector, q.Category, k)
								return res, plan.Route, plan.NProbe, eval, err
							})
						if err != nil {
							return SizeOnlyPlanner{}, ClusterAwarePlanner{}, PlannerTuningReport{}, err
						}
						meetsOld := bestClusterRecall >= targetRecall
						meetsNew := m.RecallAtK >= targetRecall && m.UnderfillRate == 0
						if (meetsNew && !meetsOld) ||
							(meetsNew && meetsOld && (m.MeanEvaluated < bestClusterEval-1.0 || (m.MeanEvaluated <= bestClusterEval+1.0 && m.RecallAtK > bestClusterRecall))) ||
							(!meetsNew && !meetsOld && m.RecallAtK > bestClusterRecall) {
							bestCluster = cand
							bestClusterRecall = m.RecallAtK
							bestClusterEval = m.MeanEvaluated
						}
					}
				}
			}
		}
	}

	report := PlannerTuningReport{
		DevQueries:           len(devQueries),
		TargetRecall:         targetRecall,
		SizeCandidatesTested: sizeTested,
		ClusterConfigsTested: clusterTested,
		FrozenSizeOnly:       bestSize,
		SizeDevRecall:        bestSizeRecall,
		SizeDevEvaluated:     bestSizeEval,
		FrozenClusterAware:   bestCluster,
		ClusterDevRecall:     bestClusterRecall,
		ClusterDevEvaluated:  bestClusterEval,
	}
	return bestSize, bestCluster, report, nil
}
