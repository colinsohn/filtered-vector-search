package vecsearch

import (
	"context"
	"fmt"
	"strings"
	"time"
)

// MethodMetrics summarizes recall, latency, candidate work, and underfill rate for one strategy.
type MethodMetrics struct {
	Method          string                       `json:"method"`
	Queries         int                          `json:"queries"`
	NonEmptyQueries int                          `json:"non_empty_queries"`
	EmptyQueries    int                          `json:"empty_queries"`
	RecallAtK       float64                      `json:"recall_at_k"`
	Underfilled     int                          `json:"underfilled"`
	UnderfillRate   float64                      `json:"underfill_rate"`
	MeanEvaluated   float64                      `json:"mean_evaluated"`
	P50LatencyUs    float64                      `json:"p50_latency_us"`
	P95LatencyUs    float64                      `json:"p95_latency_us"`
	QPS             float64                      `json:"qps"`
	ByCategory      map[string]CategoryBreakdown `json:"by_category,omitempty"`
}

// CategoryBreakdown reports per-category recall, latency, underfill, and candidate evaluations.
type CategoryBreakdown struct {
	Category      string  `json:"category"`
	Queries       int     `json:"queries"`
	RecallAtK     float64 `json:"recall_at_k"`
	UnderfillRate float64 `json:"underfill_rate"`
	MeanEvaluated float64 `json:"mean_evaluated"`
	P95LatencyUs  float64 `json:"p95_latency_us"`
	ExactRoutes   int     `json:"exact_routes"`
	IVFRoutes     int     `json:"ivf_routes"`
	MeanNProbe    float64 `json:"mean_nprobe"`
}

type queryExecFunc func(q WorkloadQuery) (results []Result, route Route, nprobe int, evaluated int, err error)

// EvaluateMethod runs warmup and repeat timing passes of fn over queries and compares against exactGroundTruth.
func EvaluateMethod(name string, queries []WorkloadQuery, exactGroundTruth [][]Result, k, warmupRuns, timingRepeats int, fn queryExecFunc) (MethodMetrics, error) {
	if warmupRuns < 0 {
		warmupRuns = 0
	}
	if timingRepeats <= 0 {
		timingRepeats = 1
	}

	// Warmup pass
	for range warmupRuns {
		for _, q := range queries {
			if _, _, _, _, err := fn(q); err != nil {
				return MethodMetrics{}, err
			}
		}
	}

	type catAccum struct {
		queries     int
		recallSum   float64
		underfilled int
		eligibleK   int
		evalSum     int
		latenciesNs []int64
		exactRoutes int
		ivfRoutes   int
		nprobeSum   int
	}
	byCat := make(map[string]*catAccum)

	var (
		totalRecall     float64
		nonEmptyQueries int
		emptyQueries    int
		underfilled     int
		eligibleAtLeastK int
		totalEval       int
		allLatenciesNs  = make([]int64, 0, len(queries)*timingRepeats)
	)

	wallStart := time.Now()
	for rep := range timingRepeats {
		for i, q := range queries {
			start := time.Now()
			approx, route, nprobe, eval, err := fn(q)
			durNs := time.Since(start).Nanoseconds()
			if err != nil {
				return MethodMetrics{}, err
			}
			allLatenciesNs = append(allLatenciesNs, durNs)

			if rep > 0 {
				// Record latency across all repeats; accumulate accuracy/work once on rep 0.
				if acc, ok := byCat[q.Category]; ok {
					acc.latenciesNs = append(acc.latenciesNs, durNs)
				}
				continue
			}

			exact := exactGroundTruth[i]
			rec, empty := RecallAtK(exact, approx)
			if empty {
				emptyQueries++
				continue
			}

			nonEmptyQueries++
			totalRecall += rec
			totalEval += eval

			acc, ok := byCat[q.Category]
			if !ok {
				acc = &catAccum{}
				byCat[q.Category] = acc
			}
			acc.queries++
			acc.recallSum += rec
			acc.evalSum += eval
			acc.latenciesNs = append(acc.latenciesNs, durNs)
			if route == RouteExact {
				acc.exactRoutes++
			} else {
				acc.ivfRoutes++
				acc.nprobeSum += nprobe
			}

			if len(exact) == k {
				eligibleAtLeastK++
				acc.eligibleK++
				if len(approx) < k {
					underfilled++
					acc.underfilled++
				}
			}
		}
	}
	wallElapsed := time.Since(wallStart)

	denom := max(1, nonEmptyQueries)
	underfillDenom := max(1, eligibleAtLeastK)

	catMap := make(map[string]CategoryBreakdown, len(byCat))
	for cat, acc := range byCat {
		qDenom := max(1, acc.queries)
		uDenom := max(1, acc.eligibleK)
		catMap[cat] = CategoryBreakdown{
			Category:      cat,
			Queries:       acc.queries,
			RecallAtK:     acc.recallSum / float64(qDenom),
			UnderfillRate: float64(acc.underfilled) / float64(uDenom),
			MeanEvaluated: float64(acc.evalSum) / float64(qDenom),
			P95LatencyUs:  percentileUs(acc.latenciesNs, 0.95),
			ExactRoutes:   acc.exactRoutes,
			IVFRoutes:     acc.ivfRoutes,
			MeanNProbe:    float64(acc.nprobeSum) / float64(max(1, acc.ivfRoutes)),
		}
	}

	totalExecs := len(queries) * timingRepeats
	return MethodMetrics{
		Method:          name,
		Queries:         len(queries),
		NonEmptyQueries: nonEmptyQueries,
		EmptyQueries:    emptyQueries,
		RecallAtK:       totalRecall / float64(denom),
		Underfilled:     underfilled,
		UnderfillRate:   float64(underfilled) / float64(underfillDenom),
		MeanEvaluated:   float64(totalEval) / float64(denom),
		P50LatencyUs:    percentileUs(allLatenciesNs, 0.50),
		P95LatencyUs:    percentileUs(allLatenciesNs, 0.95),
		QPS:             float64(totalExecs) / wallElapsed.Seconds(),
		ByCategory:      catMap,
	}, nil
}

// ComputeExactGroundTruth computes exact filtered top-k results for every query in queries.
func ComputeExactGroundTruth(exact *ExactIndex, queries []WorkloadQuery, k int) ([][]Result, error) {
	out := make([][]Result, len(queries))
	for i, q := range queries {
		res, _, err := exact.SearchContext(context.Background(), q.Vector, q.Category, k)
		if err != nil {
			return nil, err
		}
		out[i] = res
	}
	return out, nil
}

// RenderRecallLatencySVG renders a standalone SVG chart plotting Recall@10 (x-axis) vs p95 Latency us (y-axis).
func RenderRecallLatencySVG(title string, preFilterSweep []MethodMetrics, postFilterSweep []MethodMetrics, sizeOnly, clusterAware, exact MethodMetrics) string {
	const (
		width  = 820
		height = 480
		padL   = 75
		padR   = 40
		padT   = 55
		padB   = 65
	)
	plotW := float64(width - padL - padR)
	plotH := float64(height - padT - padB)

	maxY := exact.P95LatencyUs * 1.15
	for _, m := range preFilterSweep {
		if m.P95LatencyUs*1.15 > maxY {
			maxY = m.P95LatencyUs * 1.15
		}
	}
	for _, m := range postFilterSweep {
		if m.P95LatencyUs*1.15 > maxY {
			maxY = m.P95LatencyUs * 1.15
		}
	}
	if maxY <= 0 {
		maxY = 100
	}

	xCoord := func(recall float64) float64 {
		r := max(0.0, min(1.0, recall))
		return float64(padL) + r*plotW
	}
	yCoord := func(latUs float64) float64 {
		v := max(0.0, min(maxY, latUs))
		return float64(padT) + plotH - (v/maxY)*plotH
	}

	var sb strings.Builder
	fmt.Fprintf(&sb, `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" font-family="system-ui, -apple-system, sans-serif">`+"\n", width, height, width, height)
	sb.WriteString(`<rect width="100%" height="100%" fill="#ffffff"/>` + "\n")
	fmt.Fprintf(&sb, `<text x="%d" y="30" text-anchor="middle" font-size="16" font-weight="600" fill="#111827">%s</text>`+"\n", width/2, title)

	// Gridlines
	for i := 0; i <= 5; i++ {
		r := float64(i) * 0.2
		x := xCoord(r)
		fmt.Fprintf(&sb, `<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#e5e7eb" stroke-width="1"/>`+"\n", x, padT, x, height-padB)
		fmt.Fprintf(&sb, `<text x="%.1f" y="%d" text-anchor="middle" font-size="11" fill="#4b5563">%.1f</text>`+"\n", x, height-padB+20, r)
	}
	for i := 0; i <= 5; i++ {
		val := maxY * float64(i) / 5.0
		y := yCoord(val)
		fmt.Fprintf(&sb, `<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#e5e7eb" stroke-width="1"/>`+"\n", padL, y, width-padR, y)
		fmt.Fprintf(&sb, `<text x="%d" y="%.1f" text-anchor="end" font-size="11" fill="#4b5563">%.0f</text>`+"\n", padL-8, y+4, val)
	}

	// Axes
	fmt.Fprintf(&sb, `<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#374151" stroke-width="1.5"/>`+"\n", padL, height-padB, width-padR, height-padB)
	fmt.Fprintf(&sb, `<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#374151" stroke-width="1.5"/>`+"\n", padL, padT, padL, height-padB)
	fmt.Fprintf(&sb, `<text x="%d" y="%d" text-anchor="middle" font-size="13" font-weight="500" fill="#1f2937">Recall@10 (Held-Out Queries)</text>`+"\n", width/2, height-18)
	fmt.Fprintf(&sb, `<text x="20" y="%d" text-anchor="middle" font-size="13" font-weight="500" fill="#1f2937" transform="rotate(-90 20 %d)">p95 Query Latency (µs)</text>`+"\n", height/2, height/2)

	// Pre-filter polyline
	if len(preFilterSweep) > 0 {
		var pts []string
		for _, m := range preFilterSweep {
			pts = append(pts, fmt.Sprintf("%.1f,%.1f", xCoord(m.RecallAtK), yCoord(m.P95LatencyUs)))
		}
		fmt.Fprintf(&sb, `<polyline fill="none" stroke="#2563eb" stroke-width="2.2" points="%s"/>`+"\n", strings.Join(pts, " "))
		for _, m := range preFilterSweep {
			cx, cy := xCoord(m.RecallAtK), yCoord(m.P95LatencyUs)
			fmt.Fprintf(&sb, `<circle cx="%.1f" cy="%.1f" r="4" fill="#2563eb"/>`+"\n", cx, cy)
		}
	}

	// Post-filter polyline
	if len(postFilterSweep) > 0 {
		var pts []string
		for _, m := range postFilterSweep {
			pts = append(pts, fmt.Sprintf("%.1f,%.1f", xCoord(m.RecallAtK), yCoord(m.P95LatencyUs)))
		}
		fmt.Fprintf(&sb, `<polyline fill="none" stroke="#9ca3af" stroke-width="2" stroke-dasharray="5,4" points="%s"/>`+"\n", strings.Join(pts, " "))
		for _, m := range postFilterSweep {
			cx, cy := xCoord(m.RecallAtK), yCoord(m.P95LatencyUs)
			fmt.Fprintf(&sb, `<circle cx="%.1f" cy="%.1f" r="3.5" fill="#6b7280"/>`+"\n", cx, cy)
		}
	}

	// Highlighted points: SizeOnly, ClusterAware, Exact
	drawStarPoint := func(m MethodMetrics, color, label string, dy int) {
		cx, cy := xCoord(m.RecallAtK), yCoord(m.P95LatencyUs)
		fmt.Fprintf(&sb, `<circle cx="%.1f" cy="%.1f" r="6.5" fill="%s" stroke="#ffffff" stroke-width="1.5"/>`+"\n", cx, cy, color)
		fmt.Fprintf(&sb, `<text x="%.1f" y="%.1f" font-size="11" font-weight="600" fill="%s">%s (%.2f, %.0fµs)</text>`+"\n", cx-45, cy+float64(dy), color, label, m.RecallAtK, m.P95LatencyUs)
	}

	drawStarPoint(sizeOnly, "#d97706", "SizeOnly", -12)
	drawStarPoint(clusterAware, "#16a34a", "ClusterAware", 18)
	drawStarPoint(exact, "#dc2626", "Exact", -12)

	// Legend
	fmt.Fprintf(&sb, `<rect x="%d" y="%d" width="235" height="92" rx="6" fill="#f9fafb" stroke="#d1d5db"/>`+"\n", padL+15, padT+10)
	fmt.Fprintf(&sb, `<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#2563eb" stroke-width="2.2"/><text x="%d" y="%d" font-size="11" fill="#1f2937">IVF Pre-Filter (nprobe sweep)</text>`+"\n", padL+25, padT+28, padL+48, padT+28, padL+55, padT+32)
	fmt.Fprintf(&sb, `<line x1="%d" y1="%d" x2="%d" y2="%d" stroke="#9ca3af" stroke-width="2" stroke-dasharray="4,3"/><text x="%d" y="%d" font-size="11" fill="#1f2937">IVF Post-Filter (top-M sweep)</text>`+"\n", padL+25, padT+48, padL+48, padT+48, padL+55, padT+52)
	fmt.Fprintf(&sb, `<circle cx="%d" cy="%d" r="5" fill="#d97706"/><text x="%d" y="%d" font-size="11" fill="#1f2937">Size-Only Planner</text>`+"\n", padL+36, padT+68, padL+55, padT+72)
	fmt.Fprintf(&sb, `<circle cx="%d" cy="%d" r="5" fill="#16a34a"/><text x="%d" y="%d" font-size="11" fill="#1f2937">Cluster-Aware Planner</text>`+"\n", padL+36, padT+88, padL+55, padT+92)

	sb.WriteString(`</svg>` + "\n")
	return sb.String()
}
