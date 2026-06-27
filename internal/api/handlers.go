package api

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sort"
	"strconv"
	"time"

	"tf-concurrente/internal/cluster"
	"tf-concurrente/internal/loader"
	"tf-concurrente/internal/model"
	"tf-concurrente/internal/store"
)

// handlers holds shared dependencies for all HTTP handlers.
type handlers struct {
	store *store.Store
	cache *store.Cache
	coord *cluster.Coordinator
	nodes int
}

// handleTrain runs the full distributed training pipeline.
// POST /train
func (h *handlers) handleTrain(w http.ResponseWriter, r *http.Request) {
	var req TrainRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "invalid JSON: " + err.Error()})
		return
	}
	if req.File == "" {
		req.File = "data/yellow_tripdata_2015-01.csv"
	}
	if req.Epochs <= 0 {
		req.Epochs = 100
	}
	if req.LR <= 0 {
		req.LR = 0.05
	}
	if req.Batch <= 0 {
		req.Batch = 1024
	}

	log.Printf("[train] file=%s limit=%d epochs=%d lr=%.4f", req.File, req.Limit, req.Epochs, req.LR)
	started := time.Now()

	result, err := loader.Load(loader.LoadOptions{
		FilePath: req.File,
		Limit:    req.Limit,
		Workers:  4,
	})
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: "load: " + err.Error()})
		return
	}
	log.Printf("[train] loaded %d trips", len(result.Trips))

	opts := model.TrainOptions{
		Epochs:       req.Epochs,
		LearningRate: req.LR,
		BatchSize:    req.Batch,
		Seed:         req.Seed,
	}
	m, err := h.coord.TrainDistributed(result.Trips, opts)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: "train: " + err.Error()})
		return
	}
	durationMS := time.Since(started).Milliseconds()

	version, err := h.store.NextVersion()
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: "version: " + err.Error()})
		return
	}

	if err := h.store.SaveModelVersion(m, version, h.nodes); err != nil {
		log.Printf("[train] save model: %v", err)
	}
	if err := h.store.SaveTrainingRun(store.TrainingRunDoc{
		Version:     version,
		StartedAt:   started,
		FinishedAt:  time.Now().UTC(),
		DurationMS:  durationMS,
		Nodes:       h.nodes,
		LossHistory: m.LossHistory,
		DatasetFile: req.File,
		RowsUsed:    len(result.Trips),
	}); err != nil {
		log.Printf("[train] save run: %v", err)
	}
	if err := h.cache.SetActiveModel(m, version); err != nil {
		log.Printf("[train] cache model: %v", err)
	}
	h.cache.PublishModelUpdate(version)

	writeJSON(w, http.StatusOK, TrainResponse{
		Version:    version,
		Nodes:      h.nodes,
		DurationMS: durationMS,
		TrainMAE:   m.TrainMAE,
		TrainRMSE:  m.TrainRMSE,
		TrainR2:    m.TrainR2,
		TestMAE:    m.TestMAE,
		TestRMSE:   m.TestRMSE,
		TestR2:     m.TestR2,
	})
}

// handlePredict serves a prediction from the cached model (< 100 ms target).
// GET /predict?trip_distance=2.5&hour=18&day_of_week=3&pickup_lat=40.75&pickup_lon=-73.98&rate_code=1&passenger_count=1
func (h *handlers) handlePredict(w http.ResponseWriter, r *http.Request) {
	q := r.URL.Query()

	tripDistance, err := parseFloat(q.Get("trip_distance"))
	if err != nil {
		writeJSON(w, http.StatusBadRequest, ErrorResponse{Error: "trip_distance: " + err.Error()})
		return
	}
	hour, _ := parseInt(q.Get("hour"))
	dayOfWeek, _ := parseInt(q.Get("day_of_week"))
	pickupLat, _ := parseFloat(q.Get("pickup_lat"))
	pickupLon, _ := parseFloat(q.Get("pickup_lon"))
	rateCode, _ := parseInt(q.Get("rate_code"))
	passengerCount, _ := parseInt(q.Get("passenger_count"))
	if passengerCount <= 0 {
		passengerCount = 1
	}

	// Build cache key (input hash)
	cacheKey := fmt.Sprintf("%.3f:%d:%d:%.4f:%.4f:%d:%d",
		tripDistance, hour, dayOfWeek, pickupLat, pickupLon, rateCode, passengerCount)

	if pred, ok := h.cache.GetCachedPred(cacheKey); ok {
		writeJSON(w, http.StatusOK, PredictResponse{
			DurationMin:  pred,
			ModelVersion: -1, // cached, version info not re-fetched
			Cached:       true,
		})
		return
	}

	m, version, err := h.cache.GetActiveModel()
	if err != nil {
		writeJSON(w, http.StatusServiceUnavailable, ErrorResponse{Error: "no active model: run POST /train first"})
		return
	}

	trip := loader.Trip{
		TripDistance:   tripDistance,
		HourOfDay:      hour,
		DayOfWeek:      dayOfWeek,
		PickupLat:      pickupLat,
		PickupLon:      pickupLon,
		RateCodeID:     rateCode,
		PassengerCount: passengerCount,
	}

	start := time.Now()
	pred := m.PredictTrip(trip)
	latencyMS := time.Since(start).Milliseconds()

	h.cache.SetCachedPred(cacheKey, pred)

	// Async: log to MongoDB (best-effort, does not block response)
	go func() {
		_ = h.store.SavePrediction(store.PredictionDoc{
			Input: map[string]interface{}{
				"trip_distance":   tripDistance,
				"hour":            hour,
				"day_of_week":     dayOfWeek,
				"pickup_lat":      pickupLat,
				"pickup_lon":      pickupLon,
				"rate_code":       rateCode,
				"passenger_count": passengerCount,
			},
			DurationMin:  pred,
			ModelVersion: version,
			LatencyMS:    latencyMS,
			CreatedAt:    time.Now().UTC(),
		})
	}()

	writeJSON(w, http.StatusOK, PredictResponse{
		DurationMin:  pred,
		ModelVersion: version,
		Cached:       false,
	})
}

// handleModel returns the latest model metadata from MongoDB.
// GET /model
func (h *handlers) handleModel(w http.ResponseWriter, r *http.Request) {
	doc, err := h.store.GetLatestModel()
	if err != nil {
		writeJSON(w, http.StatusNotFound, ErrorResponse{Error: "no model found: " + err.Error()})
		return
	}
	writeJSON(w, http.StatusOK, doc)
}

// handleMetrics returns cluster runtime metrics from Redis.
// GET /metrics
func (h *handlers) handleMetrics(w http.ResponseWriter, r *http.Request) {
	metrics, err := h.cache.GetClusterMetrics()
	if err != nil {
		// Return empty metrics rather than error if none exist yet
		writeJSON(w, http.StatusOK, store.ClusterMetrics{Nodes: h.nodes})
		return
	}
	// Enrich with live p50/p95 from latency samples
	samples := h.cache.Latencies()
	if len(samples) > 0 {
		sort.Slice(samples, func(i, j int) bool { return samples[i] < samples[j] })
		metrics.P50MS = float64(samples[len(samples)/2])
		metrics.P95MS = float64(samples[int(float64(len(samples))*0.95)])
	}
	writeJSON(w, http.StatusOK, metrics)
}

// handleInsights serves one of the 6 EDA datasets from MongoDB.
// GET /insights/{name}
func (h *handlers) handleInsights(w http.ResponseWriter, r *http.Request) {
	name := r.PathValue("name")
	doc, err := h.store.GetInsight(name)
	if err != nil {
		writeJSON(w, http.StatusNotFound, ErrorResponse{Error: "insight not found: " + name})
		return
	}
	writeJSON(w, http.StatusOK, doc)
}

// handleHealth is a liveness probe.
// GET /healthz
func (h *handlers) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, HealthResponse{Status: "ok"})
}

// --- helpers ---

func parseFloat(s string) (float64, error) {
	if s == "" {
		return 0, fmt.Errorf("missing value")
	}
	return strconv.ParseFloat(s, 64)
}

func parseInt(s string) (int, error) {
	if s == "" {
		return 0, nil
	}
	n, err := strconv.Atoi(s)
	return n, err
}
