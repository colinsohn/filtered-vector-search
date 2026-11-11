package vecsearch

import (
	"fmt"
	"math/rand/v2"
)

// CategorySpec describes a category's corpus count and cluster concentration in a benchmark workload.
type CategorySpec struct {
	Name         string  `json:"name"`
	Distribution string  `json:"distribution"` // "concentrated" or "dispersed"
	Selectivity  float64 `json:"selectivity"`
	NumClusters  int     `json:"num_clusters"`
	DocCount     int     `json:"doc_count"`
}

// WorkloadQuery pairs a query embedding with its target equality filter category.
type WorkloadQuery struct {
	Vector   []float32
	Category string
}

// BenchmarkWorkload contains a deterministic corpus with controlled category-cluster
// distributions plus separate development and held-out query sets.
type BenchmarkWorkload struct {
	Dim            int
	NumClusters    int
	Docs           []Document
	Specs          []CategorySpec
	DevQueries     []WorkloadQuery
	HeldOutQueries []WorkloadQuery
}

// NewBenchmarkWorkload generates a corpus where cat_concentrated and cat_dispersed
// have identical document counts (3.5% of corpus each) but opposite cluster spreads.
func NewBenchmarkWorkload(seed uint64, numDocs, dim, numClusters, numDevQueries, numHeldOutQueries int) BenchmarkWorkload {
	rng := rand.New(rand.NewPCG(seed, seed^0x6a09e667f3bcc908))

	centers := make([][]float32, numClusters)
	for c := range numClusters {
		center := make([]float32, dim)
		for j := range dim {
			center[j] = float32(rng.NormFloat64() * 1.8)
		}
		centers[c] = center
	}

	rareCount := max(12, numDocs*5/1000)         // 0.5%
	concCount := max(40, numDocs*25/1000)        // 2.5%
	dispCount := concCount                       // 2.5% (exact same size as cat_concentrated)
	medCount := max(100, numDocs*150/1000)       // 15.0%
	commonCount := numDocs - rareCount - concCount - dispCount - medCount

	specs := []CategorySpec{
		{Name: "cat_rare", Distribution: "concentrated", Selectivity: float64(rareCount) / float64(numDocs), NumClusters: min(2, numClusters), DocCount: rareCount},
		{Name: "cat_concentrated", Distribution: "concentrated", Selectivity: float64(concCount) / float64(numDocs), NumClusters: min(3, numClusters), DocCount: concCount},
		{Name: "cat_dispersed", Distribution: "dispersed", Selectivity: float64(dispCount) / float64(numDocs), NumClusters: numClusters, DocCount: dispCount},
		{Name: "cat_medium", Distribution: "concentrated", Selectivity: float64(medCount) / float64(numDocs), NumClusters: min(10, numClusters), DocCount: medCount},
		{Name: "cat_common", Distribution: "dispersed", Selectivity: float64(commonCount) / float64(numDocs), NumClusters: numClusters, DocCount: commonCount},
	}

	docs := make([]Document, 0, numDocs)
	docsByCat := make(map[string][]int, len(specs))

	// Assign dedicated cluster subsets for concentrated categories.
	rareClusters := []int{0, 1 % numClusters}
	concClusters := []int{2 % numClusters, 3 % numClusters, 4 % numClusters}
	medClusters := make([]int, min(10, numClusters))
	for i := range medClusters {
		medClusters[i] = (i + 5) % numClusters
	}

	appendCatDocs := func(cat string, count int, allowedClusters []int) {
		for range count {
			idx := len(docs)
			c := allowedClusters[rng.IntN(len(allowedClusters))]
			center := centers[c]
			vec := make([]float32, dim)
			for j := range dim {
				vec[j] = center[j] + float32(rng.NormFloat64()*1.2)
			}
			docs = append(docs, Document{
				ID:       fmt.Sprintf("doc-%06d", idx),
				Category: cat,
				Vector:   vec,
			})
			docsByCat[cat] = append(docsByCat[cat], idx)
		}
	}

	allClusters := make([]int, numClusters)
	for i := range numClusters {
		allClusters[i] = i
	}

	appendCatDocs("cat_rare", rareCount, rareClusters)
	appendCatDocs("cat_concentrated", concCount, concClusters)
	appendCatDocs("cat_dispersed", dispCount, allClusters)
	appendCatDocs("cat_medium", medCount, medClusters)
	appendCatDocs("cat_common", commonCount, allClusters)

	// Shuffle document order deterministically so document IDs/positions are not grouped by category.
	rng.Shuffle(len(docs), func(i, j int) {
		docs[i], docs[j] = docs[j], docs[i]
	})
	clear(docsByCat)
	for i, d := range docs {
		docsByCat[d.Category] = append(docsByCat[d.Category], i)
	}

	buildQueries := func(qSeed uint64, count int) []WorkloadQuery {
		qRng := rand.New(rand.NewPCG(qSeed, qSeed^0xbb67ae8584caa73b))
		queries := make([]WorkloadQuery, count)
		queryCats := []string{"cat_rare", "cat_concentrated", "cat_dispersed", "cat_medium", "cat_common"}

		for i := range count {
			// Reserve every 25th query for an empty/unmatched category check.
			if i%25 == 24 {
				center := centers[qRng.IntN(numClusters)]
				vec := make([]float32, dim)
				for j := range dim {
					vec[j] = center[j] + float32(qRng.NormFloat64()*0.85)
				}
				queries[i] = WorkloadQuery{Vector: vec, Category: "cat_empty"}
				continue
			}

			cat := queryCats[i%len(queryCats)]
			catDocIndices := docsByCat[cat]
			vec := make([]float32, dim)

			// 75% of queries are perturbed around a document in that category;
			// 25% are centered near a random global cluster (testing off-cluster filtering).
			if qRng.Float64() < 0.75 && len(catDocIndices) > 0 {
				anchor := docs[catDocIndices[qRng.IntN(len(catDocIndices))]].Vector
				for j := range dim {
					vec[j] = anchor[j] + float32(qRng.NormFloat64()*1.1)
				}
			} else {
				center := centers[qRng.IntN(numClusters)]
				for j := range dim {
					vec[j] = center[j] + float32(qRng.NormFloat64()*1.1)
				}
			}
			queries[i] = WorkloadQuery{
				Vector:   vec,
				Category: cat,
			}
		}
		return queries
	}

	return BenchmarkWorkload{
		Dim:            dim,
		NumClusters:    numClusters,
		Docs:           docs,
		Specs:          specs,
		DevQueries:     buildQueries(seed^0x10101010, numDevQueries),
		HeldOutQueries: buildQueries(seed^0x20202020, numHeldOutQueries),
	}
}
