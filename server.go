package vecsearch

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"slices"
	"sync"
	"sync/atomic"
	"time"
)

// SearchRequest is the JSON request payload for POST /search.
type SearchRequest struct {
	Query    []float32 `json:"query"`
	Category string    `json:"category"`
	K        int       `json:"k"`
}

// SearchResponse is the JSON response payload for POST /search.
// EngineDurationNs measures only in-process index execution, excluding HTTP and JSON overhead.
type SearchResponse struct {
	Results          []Result `json:"results"`
	Route            Route    `json:"route"`
	NProbe           int      `json:"nprobe"`
	Evaluated        int      `json:"evaluated"`
	EngineDurationNs int64    `json:"engine_duration_ns"`
	EngineDurationUs float64  `json:"engine_duration_us"`
}

// SearchServer serves POST /search over an immutable Engine with bounded in-flight concurrency.
type SearchServer struct {
	engine         *Engine
	planner        Planner
	sem            chan struct{}
	requestTimeout time.Duration
	mux            *http.ServeMux
}

// NewSearchServer constructs a bounded-concurrency HTTP handler for engine and planner.
func NewSearchServer(engine *Engine, planner Planner, maxInFlight int, requestTimeout time.Duration) *SearchServer {
	if maxInFlight <= 0 {
		maxInFlight = 1
	}
	s := &SearchServer{
		engine:         engine,
		planner:        planner,
		sem:            make(chan struct{}, maxInFlight),
		requestTimeout: requestTimeout,
		mux:            http.NewServeMux(),
	}
	s.mux.HandleFunc("POST /search", s.handleSearch)
	return s
}

func (s *SearchServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	s.mux.ServeHTTP(w, r)
}

func (s *SearchServer) handleSearch(w http.ResponseWriter, r *http.Request) {
	// Bound in-flight search executions; reject excess load immediately with 429.
	select {
	case s.sem <- struct{}{}:
		defer func() { <-s.sem }()
	default:
		http.Error(w, "in-flight search limit exceeded", http.StatusTooManyRequests)
		return
	}

	var req SearchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid JSON: %v", err), http.StatusBadRequest)
		return
	}

	ctx := r.Context()
	if s.requestTimeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, s.requestTimeout)
		defer cancel()
	}

	engineStart := time.Now()
	results, plan, evaluated, err := s.engine.Execute(ctx, s.planner, req.Query, req.Category, req.K)
	engineDuration := time.Since(engineStart)

	if err != nil {
		if errors.Is(err, context.DeadlineExceeded) || errors.Is(err, context.Canceled) {
			http.Error(w, err.Error(), http.StatusGatewayTimeout)
			return
		}
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}

	resp := SearchResponse{
		Results:          results,
		Route:            plan.Route,
		NProbe:           plan.NProbe,
		Evaluated:        evaluated,
		EngineDurationNs: engineDuration.Nanoseconds(),
		EngineDurationUs: float64(engineDuration.Nanoseconds()) / 1000.0,
	}
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

// LoadTestReport summarizes concurrent HTTP load test metrics with separate engine and HTTP timing.
type LoadTestReport struct {
	Concurrency     int     `json:"concurrency"`
	MaxInFlight     int     `json:"max_in_flight"`
	TotalRequests   int     `json:"total_requests"`
	Succeeded       int     `json:"succeeded"`
	Rejected429     int     `json:"rejected_429"`
	TimedOut        int     `json:"timed_out"`
	OtherErrors     int     `json:"other_errors"`
	ThroughputQPS   float64 `json:"throughput_qps"`
	EngineP50Us     float64 `json:"engine_p50_us"`
	EngineP95Us     float64 `json:"engine_p95_us"`
	HTTPClientP50Us float64 `json:"http_client_p50_us"`
	HTTPClientP95Us float64 `json:"http_client_p95_us"`
}

// RunHTTPLoadTest drives concurrent HTTP requests against serverURL/search and separates
// in-process engine execution timing from end-to-end HTTP round-trip latency.
func RunHTTPLoadTest(serverURL string, queries []WorkloadQuery, k, concurrency, maxInFlight, totalRequests int, clientTimeout time.Duration) (LoadTestReport, error) {
	if len(queries) == 0 {
		return LoadTestReport{}, errors.New("queries must not be empty")
	}

	// Pre-encode request payloads so client timing measures HTTP transport + server execution.
	payloads := make([][]byte, len(queries))
	for i, q := range queries {
		b, err := json.Marshal(SearchRequest{
			Query:    q.Vector,
			Category: q.Category,
			K:        k,
		})
		if err != nil {
			return LoadTestReport{}, err
		}
		payloads[i] = b
	}

	client := &http.Client{
		Timeout: clientTimeout,
		Transport: &http.Transport{
			MaxIdleConns:        concurrency * 2,
			MaxIdleConnsPerHost: concurrency * 2,
		},
	}
	defer client.CloseIdleConnections()

	var (
		nextIdx     atomic.Int64
		succeeded   atomic.Int64
		rejected429 atomic.Int64
		timedOut    atomic.Int64
		otherErrors atomic.Int64
		mu          sync.Mutex
		engineNs    = make([]int64, 0, totalRequests)
		httpNs      = make([]int64, 0, totalRequests)
	)

	wallStart := time.Now()
	var wg sync.WaitGroup
	for range concurrency {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				idx := int(nextIdx.Add(1) - 1)
				if idx >= totalRequests {
					return
				}
				body := payloads[idx%len(payloads)]
				req, err := http.NewRequest(http.MethodPost, serverURL+"/search", bytes.NewReader(body))
				if err != nil {
					otherErrors.Add(1)
					continue
				}
				req.Header.Set("Content-Type", "application/json")

				reqStart := time.Now()
				resp, err := client.Do(req)
				reqDur := time.Since(reqStart)
				if err != nil {
					if errors.Is(err, context.DeadlineExceeded) || osTimeout(err) {
						timedOut.Add(1)
					} else {
						otherErrors.Add(1)
					}
					continue
				}

				if resp.StatusCode == http.StatusTooManyRequests {
					_ = resp.Body.Close()
					rejected429.Add(1)
					continue
				}
				if resp.StatusCode == http.StatusGatewayTimeout {
					_ = resp.Body.Close()
					timedOut.Add(1)
					continue
				}
				if resp.StatusCode != http.StatusOK {
					_ = resp.Body.Close()
					otherErrors.Add(1)
					continue
				}

				var sr SearchResponse
				decErr := json.NewDecoder(resp.Body).Decode(&sr)
				_ = resp.Body.Close()
				if decErr != nil {
					otherErrors.Add(1)
					continue
				}

				succeeded.Add(1)
				mu.Lock()
				engineNs = append(engineNs, sr.EngineDurationNs)
				httpNs = append(httpNs, reqDur.Nanoseconds())
				mu.Unlock()
			}
		}()
	}
	wg.Wait()
	wallElapsed := time.Since(wallStart)

	return LoadTestReport{
		Concurrency:     concurrency,
		MaxInFlight:     maxInFlight,
		TotalRequests:   totalRequests,
		Succeeded:       int(succeeded.Load()),
		Rejected429:     int(rejected429.Load()),
		TimedOut:        int(timedOut.Load()),
		OtherErrors:     int(otherErrors.Load()),
		ThroughputQPS:   float64(succeeded.Load()) / wallElapsed.Seconds(),
		EngineP50Us:     percentileUs(engineNs, 0.50),
		EngineP95Us:     percentileUs(engineNs, 0.95),
		HTTPClientP50Us: percentileUs(httpNs, 0.50),
		HTTPClientP95Us: percentileUs(httpNs, 0.95),
	}, nil
}

func osTimeout(err error) bool {
	var netErr interface{ Timeout() bool }
	return errors.As(err, &netErr) && netErr.Timeout()
}

// percentileUs computes the q-th quantile (0..1) of durations in nanoseconds and returns microseconds.
func percentileUs(samplesNs []int64, q float64) float64 {
	if len(samplesNs) == 0 {
		return 0
	}
	sorted := slices.Clone(samplesNs)
	slices.Sort(sorted)
	pos := int(float64(len(sorted)-1) * q)
	pos = max(0, min(len(sorted)-1, pos))
	return float64(sorted[pos]) / 1000.0
}
