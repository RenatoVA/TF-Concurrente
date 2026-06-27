package store

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/redis/go-redis/v9"

	"tf-concurrente/internal/model"
)

// Cache wraps a Redis client for the taxi ML service.
// go-redis clients are safe for concurrent use.
type Cache struct {
	client *redis.Client
}

// CachedModel bundles a LinearModel with its version for Redis storage.
type CachedModel struct {
	Version int              `json:"version"`
	Model   *model.LinearModel `json:"model"`
}

// ClusterMetrics holds runtime metrics stored in Redis.
type ClusterMetrics struct {
	Nodes        int     `json:"nodes"`
	LastRunMS    int64   `json:"last_run_ms"`
	LastVersion  int     `json:"last_version"`
	P50MS        float64 `json:"p50_ms"`
	P95MS        float64 `json:"p95_ms"`
	TotalRequests int64  `json:"total_requests"`
}

// ConnectRedis creates a Redis client and pings the server.
func ConnectRedis(addr string) (*Cache, error) {
	rdb := redis.NewClient(&redis.Options{Addr: addr})
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	if err := rdb.Ping(ctx).Err(); err != nil {
		return nil, fmt.Errorf("redis ping %s: %w", addr, err)
	}
	return &Cache{client: rdb}, nil
}

// Close closes the Redis connection.
func (c *Cache) Close() error {
	return c.client.Close()
}

// SetActiveModel stores the full model + version in Redis with no expiry.
func (c *Cache) SetActiveModel(m *model.LinearModel, version int) error {
	data, err := json.Marshal(CachedModel{Version: version, Model: m})
	if err != nil {
		return fmt.Errorf("marshal active model: %w", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	return c.client.Set(ctx, "model:active", data, 0).Err()
}

// GetActiveModel retrieves the current model and its version from Redis.
func (c *Cache) GetActiveModel() (*model.LinearModel, int, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	data, err := c.client.Get(ctx, "model:active").Bytes()
	if err != nil {
		return nil, 0, err
	}
	var cm CachedModel
	if err := json.Unmarshal(data, &cm); err != nil {
		return nil, 0, fmt.Errorf("unmarshal active model: %w", err)
	}
	return cm.Model, cm.Version, nil
}

// GetCachedPred returns a cached prediction for the given input hash, if present.
func (c *Cache) GetCachedPred(hashKey string) (float64, bool) {
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	val, err := c.client.Get(ctx, "pred:cache:"+hashKey).Float64()
	if err != nil {
		return 0, false
	}
	return val, true
}

// SetCachedPred stores a prediction result with a 5-minute TTL.
func (c *Cache) SetCachedPred(hashKey string, durationMin float64) {
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	_ = c.client.Set(ctx, "pred:cache:"+hashKey, durationMin, 5*time.Minute).Err()
}

// SetClusterMetrics stores cluster metrics in Redis.
func (c *Cache) SetClusterMetrics(m ClusterMetrics) error {
	data, err := json.Marshal(m)
	if err != nil {
		return err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	return c.client.Set(ctx, "cluster:metrics", data, 0).Err()
}

// GetClusterMetrics retrieves cluster metrics from Redis.
func (c *Cache) GetClusterMetrics() (*ClusterMetrics, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	data, err := c.client.Get(ctx, "cluster:metrics").Bytes()
	if err != nil {
		return nil, err
	}
	var m ClusterMetrics
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return &m, nil
}

// PublishModelUpdate publishes a model version update to the model:updates channel.
func (c *Cache) PublishModelUpdate(version int) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	msg, _ := json.Marshal(map[string]int{"version": version})
	_ = c.client.Publish(ctx, "model:updates", msg).Err()
}

// IncrRequests atomically increments the total request counter and returns p50/p95
// approximations based on a running list of latency samples (last 200).
func (c *Cache) RecordLatency(ms int64) {
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	// Push latency to a capped list for percentile estimation
	pipe := c.client.Pipeline()
	pipe.RPush(ctx, "latencies", ms)
	pipe.LTrim(ctx, "latencies", -200, -1)
	_, _ = pipe.Exec(ctx)
}

// Latencies returns recent latency samples for p50/p95 calculation.
func (c *Cache) Latencies() []int64 {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	vals, err := c.client.LRange(ctx, "latencies", 0, -1).Result()
	if err != nil {
		return nil
	}
	out := make([]int64, 0, len(vals))
	for _, v := range vals {
		var n int64
		if _, err := fmt.Sscan(v, &n); err == nil {
			out = append(out, n)
		}
	}
	return out
}
