package vecsearch

import (
	"bufio"
	"compress/gzip"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand/v2"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	// PublicCovertypeURL is the official UCI Machine Learning Repository Covertype dataset (CC BY 4.0).
	PublicCovertypeURL = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
	// PublicCovertypeSHA256 pins the exact compressed archive for reproducible experiments.
	PublicCovertypeSHA256 = "614360d0257557dd1792834a85a1cdebfadc3c4f30b011d56afee7ffb5b15771"
	// PublicCovertypeDim is the 54-feature cartographic vector dimensionality.
	PublicCovertypeDim = 54
)

var covertypeCategoryNames = map[string]string{
	"1": "spruce_fir",
	"2": "lodgepole_pine",
	"3": "ponderosa_pine",
	"4": "cottonwood_willow",
	"5": "aspen",
	"6": "douglas_fir",
	"7": "krummholz",
}

var covertypeCategoryOrder = []string{
	"cottonwood_willow",
	"aspen",
	"douglas_fir",
	"krummholz",
	"ponderosa_pine",
	"spruce_fir",
	"lodgepole_pine",
}

// EnsurePublicCovertypeArchive verifies or downloads covtype.data.gz into cacheDir and checks its SHA-256.
func EnsurePublicCovertypeArchive(cacheDir string) (string, error) {
	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return "", err
	}
	target := filepath.Join(cacheDir, "covtype.data.gz")
	if ok, _ := verifySHA256(target, PublicCovertypeSHA256); ok {
		return target, nil
	}

	client := &http.Client{Timeout: 90 * time.Second}
	resp, err := client.Get(PublicCovertypeURL)
	if err != nil {
		return "", fmt.Errorf("download covtype.data.gz: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("download covtype.data.gz: unexpected HTTP %d", resp.StatusCode)
	}

	tmpPath := target + ".tmp"
	f, err := os.Create(tmpPath)
	if err != nil {
		return "", err
	}
	hasher := sha256.New()
	if _, err := io.Copy(io.MultiWriter(f, hasher), resp.Body); err != nil {
		_ = f.Close()
		_ = os.Remove(tmpPath)
		return "", err
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(tmpPath)
		return "", err
	}

	gotHash := hex.EncodeToString(hasher.Sum(nil))
	if gotHash != PublicCovertypeSHA256 {
		_ = os.Remove(tmpPath)
		return "", fmt.Errorf("covtype.data.gz SHA-256 mismatch: got %s, want %s", gotHash, PublicCovertypeSHA256)
	}
	if err := os.Rename(tmpPath, target); err != nil {
		return "", err
	}
	return target, nil
}

func verifySHA256(path, wantHex string) (bool, error) {
	f, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return false, err
	}
	return hex.EncodeToString(h.Sum(nil)) == wantHex, nil
}

// LoadPublicCovertypeWorkload loads a reproducible subset of the UCI Covertype corpus
// with real ecological category labels, z-score standardized and L2-normalized 54D vectors,
// and separate stratified development and held-out query sets.
func LoadPublicCovertypeWorkload(cacheDir string, seed uint64, numDocs, numClusters, numDevQueries, numHeldOutQueries int) (BenchmarkWorkload, error) {
	gzPath, err := EnsurePublicCovertypeArchive(cacheDir)
	if err != nil {
		return BenchmarkWorkload{}, err
	}

	f, err := os.Open(gzPath)
	if err != nil {
		return BenchmarkWorkload{}, err
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return BenchmarkWorkload{}, err
	}
	defer gz.Close()

	type rawRecord struct {
		cat      string
		features [PublicCovertypeDim]float32
	}

	// Read the full 581,012-row archive so the seeded shuffle reflects the true dataset distribution.
	records := make([]rawRecord, 0, 581012)
	scanner := bufio.NewScanner(gz)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		parts := strings.Split(line, ",")
		if len(parts) != PublicCovertypeDim+1 {
			return BenchmarkWorkload{}, fmt.Errorf("unexpected column count %d in covtype", len(parts))
		}
		catLabel, ok := covertypeCategoryNames[parts[PublicCovertypeDim]]
		if !ok {
			return BenchmarkWorkload{}, fmt.Errorf("unknown cover_type %q", parts[PublicCovertypeDim])
		}
		var rec rawRecord
		rec.cat = catLabel
		for j := range PublicCovertypeDim {
			val, err := strconv.ParseFloat(parts[j], 32)
			if err != nil {
				return BenchmarkWorkload{}, err
			}
			rec.features[j] = float32(val)
		}
		records = append(records, rec)
	}
	if err := scanner.Err(); err != nil {
		return BenchmarkWorkload{}, err
	}
	if len(records) < numDocs+numDevQueries+numHeldOutQueries {
		return BenchmarkWorkload{}, errors.New("insufficient rows in covtype archive")
	}

	// Deterministic shuffle using seed so corpus, dev queries, and held-out queries are disjoint.
	rng := rand.New(rand.NewPCG(seed, seed^0x517cc1b727220a95))
	rng.Shuffle(len(records), func(i, j int) {
		records[i], records[j] = records[j], records[i]
	})

	// Compute per-feature mean and std on the corpus slice (records[:numDocs]) for z-score standardization.
	var means [PublicCovertypeDim]float64
	var m2s [PublicCovertypeDim]float64
	for i := range numDocs {
		for j := range PublicCovertypeDim {
			v := float64(records[i].features[j])
			means[j] += v
		}
	}
	invN := 1.0 / float64(numDocs)
	for j := range PublicCovertypeDim {
		means[j] *= invN
	}
	for i := range numDocs {
		for j := range PublicCovertypeDim {
			d := float64(records[i].features[j]) - means[j]
			m2s[j] += d * d
		}
	}
	var invStds [PublicCovertypeDim]float64
	for j := range PublicCovertypeDim {
		variance := m2s[j] * invN
		if variance > 1e-12 {
			invStds[j] = 1.0 / math.Sqrt(variance)
		} else {
			invStds[j] = 1.0
		}
	}

	normalizeVec := func(feat *[PublicCovertypeDim]float32) []float32 {
		out := make([]float32, PublicCovertypeDim)
		var normSq float64
		for j := range PublicCovertypeDim {
			z := (float64(feat[j]) - means[j]) * invStds[j]
			out[j] = float32(z)
			normSq += z * z
		}
		if normSq > 1e-12 {
			scale := float32(1.0 / math.Sqrt(normSq))
			for j := range PublicCovertypeDim {
				out[j] *= scale
			}
		}
		return out
	}

	docs := make([]Document, numDocs)
	catCounts := make(map[string]int, len(covertypeCategoryOrder))
	for i := range numDocs {
		cat := records[i].cat
		catCounts[cat]++
		docs[i] = Document{
			ID:       fmt.Sprintf("cov-%06d", i),
			Category: cat,
			Vector:   normalizeVec(&records[i].features),
		}
	}

	// Group remaining disjoint records by category to form stratified dev and held-out query pools.
	poolByCat := make(map[string][][]float32, len(covertypeCategoryOrder))
	for i := numDocs; i < len(records); i++ {
		cat := records[i].cat
		poolByCat[cat] = append(poolByCat[cat], normalizeVec(&records[i].features))
	}
	poolCursor := make(map[string]int, len(covertypeCategoryOrder))

	takeQueries := func(total int) []WorkloadQuery {
		out := make([]WorkloadQuery, 0, total)
		for len(out) < total {
			for _, cat := range covertypeCategoryOrder {
				if len(out) >= total {
					break
				}
				pool := poolByCat[cat]
				if len(pool) == 0 {
					continue
				}
				cur := poolCursor[cat] % len(pool)
				poolCursor[cat]++
				out = append(out, WorkloadQuery{
					Vector:   pool[cur],
					Category: cat,
				})
			}
		}
		return out
	}

	devQueries := takeQueries(numDevQueries)
	heldOutQueries := takeQueries(numHeldOutQueries)

	specs := make([]CategorySpec, 0, len(covertypeCategoryOrder))
	for _, cat := range covertypeCategoryOrder {
		cnt := catCounts[cat]
		distLabel := "real_ecological"
		specs = append(specs, CategorySpec{
			Name:         cat,
			Distribution: distLabel,
			Selectivity:  float64(cnt) / float64(numDocs),
			NumClusters:  numClusters,
			DocCount:     cnt,
		})
	}

	return BenchmarkWorkload{
		Dim:            PublicCovertypeDim,
		NumClusters:    numClusters,
		Docs:           docs,
		Specs:          specs,
		DevQueries:     devQueries,
		HeldOutQueries: heldOutQueries,
	}, nil
}

// AnnotateActiveClusters updates each CategorySpec with the number of IVF clusters
// containing at least one document of that category and labels concentrated vs dispersed.
func AnnotateActiveClusters(specs []CategorySpec, ivf *IVFIndex) {
	totalClusters := max(1, ivf.NumClusters())
	for i := range specs {
		hist := ivf.CategoryClusterCounts(specs[i].Name)
		active := 0
		for _, c := range hist {
			if c > 0 {
				active++
			}
		}
		specs[i].NumClusters = active
		if specs[i].Distribution == "real_ecological" {
			if active <= totalClusters/3 {
				specs[i].Distribution = fmt.Sprintf("concentrated (%d/%d clusters)", active, totalClusters)
			} else {
				specs[i].Distribution = fmt.Sprintf("dispersed (%d/%d clusters)", active, totalClusters)
			}
		}
	}
}
