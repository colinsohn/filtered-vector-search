package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"runtime"
	"time"

	vecsearch "filtered-vector-search"
)

type BenchmarkArtifact struct {
	Scale              string                    `json:"scale"`
	Timestamp          string                    `json:"timestamp"`
	GoVersion          string                    `json:"go_version"`
	GOOS               string                    `json:"goos"`
	GOArch             string                    `json:"goarch"`
	NumCPU             int                       `json:"num_cpu"`
	CorpusDocs         int                       `json:"corpus_docs"`
	Dim                int                       `json:"dim"`
	NumClusters        int                       `json:"num_clusters"`
	K                  int                       `json:"k"`
	DevQueries         int                       `json:"dev_queries"`
	HeldOutQueries     int                       `json:"held_out_queries"`
	IndexBuildMs       float64                   `json:"index_build_ms"`
	CorpusMemoryBytes  int64                     `json:"corpus_memory_bytes"`
	CategorySpecs      []vecsearch.CategorySpec  `json:"category_specs"`
	Exact              vecsearch.MethodMetrics   `json:"exact"`
	SizeOnly           vecsearch.MethodMetrics   `json:"size_only_planner"`
	ClusterAware       vecsearch.MethodMetrics   `json:"cluster_aware_planner"`
	PreFilterSweep     []vecsearch.MethodMetrics `json:"ivf_prefilter_sweep"`
	PostFilterSweep    []vecsearch.MethodMetrics `json:"ivf_postfilter_sweep"`
	HTTPLoadNormal     vecsearch.LoadTestReport  `json:"http_load_normal"`
	HTTPLoadSaturated  vecsearch.LoadTestReport  `json:"http_load_saturated"`
}

func main() {
	scale := flag.String("scale", "small", "benchmark scale: 'small' (20k vectors) or 'full' (100k vectors)")
	outDir := flag.String("out", "results", "output directory for JSON and SVG artifacts")
	flag.Parse()

	var (
		numDocs           = 20000
		dim               = 64
		numClusters       = 64
		k                 = 10
		numDevQueries     = 200
		numHeldOutQueries = 500
		warmupRuns        = 1
		timingRepeats     = 3
	)
	if *scale == "full" {
		numDocs = 100000
	}

	fmt.Printf("=== Filtered Vector Search Benchmark (scale=%s, N=%d, dim=%d, clusters=%d, k=%d) ===\n",
		*scale, numDocs, dim, numClusters, k)

	wl := vecsearch.NewBenchmarkWorkload(42, numDocs, dim, numClusters, numDevQueries, numHeldOutQueries)

	var memBefore runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&memBefore)

	buildStart := time.Now()
	engine, err := vecsearch.NewEngine(wl.Dim, wl.Docs, wl.NumClusters, 42)
	if err != nil {
		fatalf("NewEngine failed: %v", err)
	}
	buildMs := float64(time.Since(buildStart).Nanoseconds()) / 1e6

	var memAfter runtime.MemStats
	runtime.GC()
	runtime.ReadMemStats(&memAfter)
	var memBytes int64
	if memAfter.HeapAlloc > memBefore.HeapAlloc {
		memBytes = int64(memAfter.HeapAlloc - memBefore.HeapAlloc)
	} else {
		memBytes = int64(numDocs * dim * 4 * 2)
	}

	// Ground truth on held-out queries
	exactTruth, err := vecsearch.ComputeExactGroundTruth(engine.Exact, wl.HeldOutQueries, k)
	if err != nil {
		fatalf("ComputeExactGroundTruth failed: %v", err)
	}

	ctx := context.Background()

	// 1. Exact baseline
	exactMetrics, err := vecsearch.EvaluateMethod("exact", wl.HeldOutQueries, exactTruth, k, warmupRuns, timingRepeats,
		func(q vecsearch.WorkloadQuery) ([]vecsearch.Result, vecsearch.Route, int, int, error) {
			res, eval, err := engine.Exact.SearchContext(ctx, q.Vector, q.Category, k)
			return res, vecsearch.RouteExact, 0, eval, err
		})
	if err != nil {
		fatalf("exact eval failed: %v", err)
	}

	// 2. IVF Pre-Filter nprobe sweep
	probes := []int{1, 2, 4, 8, 12, 16, 24, 32, numClusters}
	preSweep := make([]vecsearch.MethodMetrics, 0, len(probes))
	for _, nprobe := range probes {
		name := fmt.Sprintf("ivf_prefilter_nprobe_%d", nprobe)
		m, err := vecsearch.EvaluateMethod(name, wl.HeldOutQueries, exactTruth, k, warmupRuns, timingRepeats,
			func(q vecsearch.WorkloadQuery) ([]vecsearch.Result, vecsearch.Route, int, int, error) {
				res, eval, err := engine.IVF.SearchPreFilterContext(ctx, q.Vector, q.Category, k, nprobe)
				return res, vecsearch.RouteIVFPreFilter, nprobe, eval, err
			})
		if err != nil {
			fatalf("%s failed: %v", name, err)
		}
		preSweep = append(preSweep, m)
	}

	// 3. IVF Post-Filter candidate budget sweep beyond top-10 (at nprobe=8)
	postBudgets := []int{10, 25, 50, 100, 250, 500}
	postSweep := make([]vecsearch.MethodMetrics, 0, len(postBudgets))
	for _, candBudget := range postBudgets {
		name := fmt.Sprintf("ivf_postfilter_nprobe_8_top_%d", candBudget)
		m, err := vecsearch.EvaluateMethod(name, wl.HeldOutQueries, exactTruth, k, warmupRuns, timingRepeats,
			func(q vecsearch.WorkloadQuery) ([]vecsearch.Result, vecsearch.Route, int, int, error) {
				res, eval, err := engine.IVF.SearchPostFilterContext(ctx, q.Vector, q.Category, k, 8, candBudget)
				return res, vecsearch.RouteIVFPreFilter, 8, eval, err
			})
		if err != nil {
			fatalf("%s failed: %v", name, err)
		}
		postSweep = append(postSweep, m)
	}

	// 4. Size-Only Planner (frozen policy tuned on dev queries)
	sizePlanner := vecsearch.DefaultSizeOnlyPlanner()
	sizeMetrics, err := vecsearch.EvaluateMethod("size_only_planner", wl.HeldOutQueries, exactTruth, k, warmupRuns, timingRepeats,
		func(q vecsearch.WorkloadQuery) ([]vecsearch.Result, vecsearch.Route, int, int, error) {
			res, plan, eval, err := engine.Execute(ctx, sizePlanner, q.Vector, q.Category, k)
			return res, plan.Route, plan.NProbe, eval, err
		})
	if err != nil {
		fatalf("size_only_planner failed: %v", err)
	}

	// 5. Cluster-Aware Planner (frozen policy tuned on dev queries)
	clusterPlanner := vecsearch.DefaultClusterAwarePlanner()
	clusterMetrics, err := vecsearch.EvaluateMethod("cluster_aware_planner", wl.HeldOutQueries, exactTruth, k, warmupRuns, timingRepeats,
		func(q vecsearch.WorkloadQuery) ([]vecsearch.Result, vecsearch.Route, int, int, error) {
			res, plan, eval, err := engine.Execute(ctx, clusterPlanner, q.Vector, q.Category, k)
			return res, plan.Route, plan.NProbe, eval, err
		})
	if err != nil {
		fatalf("cluster_aware_planner failed: %v", err)
	}

	// 6. Concurrent HTTP load tests (normal vs. saturated in-flight cap showing 429 rejections)
	srvNormal := httptest.NewServer(vecsearch.NewSearchServer(engine, clusterPlanner, 16, 250*time.Millisecond))
	defer srvNormal.Close()
	loadNormal, err := vecsearch.RunHTTPLoadTest(srvNormal.URL, wl.HeldOutQueries, k, 8, 16, 1000, 250*time.Millisecond)
	if err != nil {
		fatalf("RunHTTPLoadTest normal failed: %v", err)
	}

	srvSaturated := httptest.NewServer(vecsearch.NewSearchServer(engine, clusterPlanner, 2, 250*time.Millisecond))
	defer srvSaturated.Close()
	loadSaturated, err := vecsearch.RunHTTPLoadTest(srvSaturated.URL, wl.HeldOutQueries, k, 16, 2, 1000, 250*time.Millisecond)
	if err != nil {
		fatalf("RunHTTPLoadTest saturated failed: %v", err)
	}

	printMarkdownSummary(wl, buildMs, memBytes, exactMetrics, sizeMetrics, clusterMetrics, preSweep, postSweep, loadNormal, loadSaturated)

	if err := os.MkdirAll(*outDir, 0o755); err != nil {
		fatalf("mkdir %s: %v", *outDir, err)
	}

	artifact := BenchmarkArtifact{
		Scale:             *scale,
		Timestamp:         time.Now().UTC().Format(time.RFC3339),
		GoVersion:         runtime.Version(),
		GOOS:              runtime.GOOS,
		GOArch:            runtime.GOARCH,
		NumCPU:            runtime.NumCPU(),
		CorpusDocs:        numDocs,
		Dim:               dim,
		NumClusters:       numClusters,
		K:                 k,
		DevQueries:        numDevQueries,
		HeldOutQueries:    numHeldOutQueries,
		IndexBuildMs:      buildMs,
		CorpusMemoryBytes: memBytes,
		CategorySpecs:     wl.Specs,
		Exact:             exactMetrics,
		SizeOnly:          sizeMetrics,
		ClusterAware:      clusterMetrics,
		PreFilterSweep:    preSweep,
		PostFilterSweep:   postSweep,
		HTTPLoadNormal:    loadNormal,
		HTTPLoadSaturated: loadSaturated,
	}

	jsonPath := filepath.Join(*outDir, fmt.Sprintf("benchmark_%s.json", *scale))
	jsonBytes, err := json.MarshalIndent(artifact, "", "  ")
	if err != nil {
		fatalf("json marshal: %v", err)
	}
	if err := os.WriteFile(jsonPath, jsonBytes, 0o644); err != nil {
		fatalf("write %s: %v", jsonPath, err)
	}

	svgPath := filepath.Join(*outDir, fmt.Sprintf("recall_latency_%s.svg", *scale))
	svgTitle := fmt.Sprintf("Filtered Vector Search (N=%d, d=%d, C=%d): p95 Latency vs. Recall@10", numDocs, dim, numClusters)
	svgContent := vecsearch.RenderRecallLatencySVG(svgTitle, preSweep, postSweep, sizeMetrics, clusterMetrics, exactMetrics)
	if err := os.WriteFile(svgPath, []byte(svgContent), 0o644); err != nil {
		fatalf("write %s: %v", svgPath, err)
	}

	fmt.Printf("\nWrote %s and %s\n", jsonPath, svgPath)
}

func printMarkdownSummary(
	wl vecsearch.BenchmarkWorkload,
	buildMs float64,
	memBytes int64,
	exact, sizeOnly, clusterAware vecsearch.MethodMetrics,
	preSweep, postSweep []vecsearch.MethodMetrics,
	loadNormal, loadSaturated vecsearch.LoadTestReport,
) {
	fmt.Printf("Index Build Time: %.1f ms | Estimated Index Memory: %.2f MB | Non-Empty Queries: %d | Empty-Filter Queries: %d\n\n",
		buildMs, float64(memBytes)/(1024*1024), exact.NonEmptyQueries, exact.EmptyQueries)

	fmt.Println("### Overall Held-Out Comparison")
	fmt.Println("| Method | Recall@10 | Underfill Rate | Mean Evaluated | p50 (µs) | p95 (µs) | QPS |")
	fmt.Println("| --- | --- | --- | --- | --- | --- | --- |")
	rows := []vecsearch.MethodMetrics{exact, sizeOnly, clusterAware}
	rows = append(rows, preSweep...)
	rows = append(rows, postSweep...)
	for _, r := range rows {
		fmt.Printf("| `%s` | `%.4f` | `%.4f` | `%6.1f` | `%6.1f` | `%6.1f` | `%8.0f` |\n",
			r.Method, r.RecallAtK, r.UnderfillRate, r.MeanEvaluated, r.P50LatencyUs, r.P95LatencyUs, r.QPS)
	}

	fmt.Println("\n### Equal-Sized Concentrated vs. Dispersed Breakdown (Size-Only vs. Cluster-Aware)")
	fmt.Println("| Planner | Category | Distribution | Docs | Recall@10 | Underfill Rate | Mean Evaluated | p95 (µs) | Exact / IVF Routes | Mean nprobe |")
	fmt.Println("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
	for _, spec := range wl.Specs {
		for _, m := range []vecsearch.MethodMetrics{sizeOnly, clusterAware} {
			cb := m.ByCategory[spec.Name]
			fmt.Printf("| `%s` | `%s` | %s | %d | `%.4f` | `%.4f` | `%6.1f` | `%6.1f` | `%d / %d` | `%.1f` |\n",
				m.Method, spec.Name, spec.Distribution, spec.DocCount,
				cb.RecallAtK, cb.UnderfillRate, cb.MeanEvaluated, cb.P95LatencyUs,
				cb.ExactRoutes, cb.IVFRoutes, cb.MeanNProbe)
		}
	}

	fmt.Println("\n### Concurrent HTTP Load Test (Engine Timing vs. End-to-End HTTP Timing)")
	fmt.Println("| Mode | Workers | Max In-Flight | Succeeded | Rejected (429) | Timed Out | QPS | Engine p50 / p95 (µs) | HTTP p50 / p95 (µs) |")
	fmt.Println("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
	for _, rep := range []struct {
		label string
		r     vecsearch.LoadTestReport
	}{
		{"Normal (c=8, cap=16)", loadNormal},
		{"Saturated (c=16, cap=2)", loadSaturated},
	} {
		fmt.Printf("| %s | %d | %d | %d | %d | %d | `%.0f` | `%.1f / %.1f` | `%.1f / %.1f` |\n",
			rep.label, rep.r.Concurrency, rep.r.MaxInFlight, rep.r.Succeeded, rep.r.Rejected429, rep.r.TimedOut,
			rep.r.ThroughputQPS, rep.r.EngineP50Us, rep.r.EngineP95Us, rep.r.HTTPClientP50Us, rep.r.HTTPClientP95Us)
	}
}

func fatalf(format string, args ...any) {
	fmt.Fprintf(os.Stderr, format+"\n", args...)
	os.Exit(1)
}
