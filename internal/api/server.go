package api

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"
	"time"

	"tf-concurrente/internal/cluster"
	"tf-concurrente/internal/store"
)

// Config holds runtime configuration for the API server.
type Config struct {
	Addr      string   // e.g. ":8080"
	MongoURI  string   // e.g. "mongodb://mongo:27017"
	RedisAddr string   // e.g. "redis:6379"
	Nodes     []string // host:port of ML worker nodes
}

// StartServer connects to Mongo + Redis, wires up routes, and starts serving.
// Blocks until the server exits.
func StartServer(cfg Config) error {
	mongo, err := store.Connect(cfg.MongoURI)
	if err != nil {
		return err
	}
	defer mongo.Close()

	cache, err := store.ConnectRedis(cfg.RedisAddr)
	if err != nil {
		return err
	}
	defer cache.Close()

	coord := cluster.NewCoordinator(cfg.Nodes)

	h := &handlers{
		store: mongo,
		cache: cache,
		coord: coord,
		nodes: len(cfg.Nodes),
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /train", h.handleTrain)
	mux.HandleFunc("GET /predict", h.handlePredict)
	mux.HandleFunc("GET /model", h.handleModel)
	mux.HandleFunc("GET /metrics", h.handleMetrics)
	mux.HandleFunc("GET /insights/{name}", h.handleInsights)
	mux.HandleFunc("GET /healthz", h.handleHealth)

	srv := &http.Server{
		Addr:         cfg.Addr,
		Handler:      withMiddleware(mux, cache),
		ReadTimeout:  10 * time.Minute, // /train can take a while
		WriteTimeout: 10 * time.Minute,
		IdleTimeout:  60 * time.Second,
	}

	log.Printf("[api] listening on %s | nodes=%v", cfg.Addr, cfg.Nodes)
	return srv.ListenAndServe()
}

// withMiddleware wraps the mux with logging, CORS, recover, and latency tracking.
func withMiddleware(next http.Handler, cache *store.Cache) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer func() {
			if rec := recover(); rec != nil {
				log.Printf("[api] panic: %v", rec)
				writeJSON(w, http.StatusInternalServerError, ErrorResponse{Error: "internal server error"})
			}
		}()

		// CORS
		w.Header().Set("Access-Control-Allow-Origin", "*")
		w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
		w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}

		start := time.Now()
		next.ServeHTTP(w, r)
		ms := time.Since(start).Milliseconds()

		log.Printf("[api] %s %s %dms", r.Method, r.URL.Path, ms)

		// Track latency for /predict only
		if strings.HasPrefix(r.URL.Path, "/predict") {
			cache.RecordLatency(ms)
		}
	})
}

// writeJSON serializes v to JSON and writes it with the given status code.
func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("[api] encode response: %v", err)
	}
}
