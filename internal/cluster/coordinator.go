package cluster

import (
	"context"
	"encoding/gob"
	"fmt"
	"log"
	"math"
	"net"
	"sync"
	"time"

	"tf-concurrente/internal/loader"
	"tf-concurrente/internal/model"
)

// Coordinator manages TCP connections to ML worker nodes and orchestrates
// distributed SGD training using a parameter-server topology.
// The coordinator is the sole writer of theta (the aggregator role).
type Coordinator struct {
	Nodes []string // host:port addresses of worker nodes
}

// NewCoordinator creates a Coordinator targeting the given node addresses.
func NewCoordinator(nodes []string) *Coordinator {
	return &Coordinator{Nodes: nodes}
}

// nodeConn holds the persistent gob connection to one worker node.
type nodeConn struct {
	enc  *gob.Encoder
	dec  *gob.Decoder
	conn net.Conn
}

// TrainDistributed runs distributed mini-batch gradient descent across all nodes.
// Split is deterministic (first 80% / last 20%), no global shuffle, so the
// cluster_test can reproduce the exact theta from AccumulateGradient.
func (c *Coordinator) TrainDistributed(trips []loader.Trip, opts model.TrainOptions) (*model.LinearModel, error) {
	if len(trips) == 0 {
		return nil, fmt.Errorf("no data to train")
	}
	if opts.Epochs <= 0 {
		opts.Epochs = 100
	}
	if opts.LearningRate <= 0 {
		opts.LearningRate = 0.05
	}
	if opts.TrainFraction <= 0 || opts.TrainFraction >= 1 {
		opts.TrainFraction = 0.8
	}

	// Deterministic 80/20 split (no shuffle — test relies on this)
	splitAt := int(float64(len(trips)) * opts.TrainFraction)
	trainTrips := trips[:splitAt]
	testTrips := trips[splitAt:]
	log.Printf("[coordinator] train=%d test=%d nodes=%d", len(trainTrips), len(testTrips), len(c.Nodes))

	scaler := model.FitScaler(trainTrips)
	X, y := scaler.BuildMatrix(trainTrips)

	parts := model.MakePartitions(len(X), len(c.Nodes))

	// Dial all nodes and send their data partitions
	conns := make([]nodeConn, len(c.Nodes))
	for i, addr := range c.Nodes {
		conn, err := net.DialTimeout("tcp", addr, 15*time.Second)
		if err != nil {
			return nil, fmt.Errorf("dial node %s: %w", addr, err)
		}
		nc := nodeConn{
			enc:  NewEncoder(conn),
			dec:  NewDecoder(conn),
			conn: conn,
		}
		conns[i] = nc

		p := parts[i]
		if err := Send(nc.enc, Envelope{
			Kind: KindDataInit,
			Payload: DataInit{
				X: X[p.Start:p.End],
				Y: y[p.Start:p.End],
			},
		}); err != nil {
			return nil, fmt.Errorf("send DataInit to node %d: %w", i, err)
		}
		env, err := Recv(nc.dec)
		if err != nil || env.Kind != KindAck {
			return nil, fmt.Errorf("ack from node %d: %w", i, err)
		}
		log.Printf("[coordinator] node %d (%s) ready, rows %d-%d", i, addr, p.Start, p.End)
	}

	// Training loop
	nWeights := model.NumFeatures + 1
	theta := make([]float64, nWeights)
	lossHistory := make([]float64, 0, opts.Epochs)
	prevMSE := math.MaxFloat64

	for epoch := 1; epoch <= opts.Epochs; epoch++ {
		ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
		thetaCopy := make([]float64, len(theta))
		copy(thetaCopy, theta)

		type result struct {
			gr  GradientResult
			err error
		}
		resultCh := make(chan result, len(conns))

		var wg sync.WaitGroup
		for _, nc := range conns {
			wg.Add(1)
			go func(nc nodeConn) {
				defer wg.Done()
				paramCopy := make([]float64, len(thetaCopy))
				copy(paramCopy, thetaCopy)
				if err := Send(nc.enc, Envelope{
					Kind:    KindParam,
					Payload: ParamUpdate{Epoch: epoch, Theta: paramCopy},
				}); err != nil {
					resultCh <- result{err: fmt.Errorf("send param: %w", err)}
					return
				}
				env, err := Recv(nc.dec)
				if err != nil {
					resultCh <- result{err: fmt.Errorf("recv gradient: %w", err)}
					return
				}
				gr, ok := env.Payload.(GradientResult)
				if !ok {
					resultCh <- result{err: fmt.Errorf("unexpected payload kind %d", env.Kind)}
					return
				}
				resultCh <- result{gr: gr}
			}(nc)
		}

		// Wait for all goroutines, then close channel
		go func() {
			wg.Wait()
			close(resultCh)
		}()

		// Aggregate
		gradTotal := make([]float64, nWeights)
		totalN := 0
		var firstErr error
		for res := range resultCh {
			select {
			case <-ctx.Done():
				cancel()
				return nil, fmt.Errorf("epoch %d timeout", epoch)
			default:
			}
			if res.err != nil {
				if firstErr == nil {
					firstErr = res.err
				}
				continue
			}
			for j := range gradTotal {
				gradTotal[j] += res.gr.Grad[j]
			}
			totalN += res.gr.N
		}
		cancel()

		if firstErr != nil {
			return nil, fmt.Errorf("epoch %d: %w", epoch, firstErr)
		}

		// Apply update: θ ← θ - lr · ∇J_avg  (single writer = coordinator)
		if totalN > 0 {
			for j := range theta {
				theta[j] -= opts.LearningRate * gradTotal[j] / float64(totalN)
			}
		}

		mse := computeMSECoord(X, y, theta)
		lossHistory = append(lossHistory, mse)
		log.Printf("[coordinator] epoch %3d/%d | MSE=%.4f", epoch, opts.Epochs, mse)
		if mse > prevMSE*10 {
			log.Printf("[coordinator] WARNING: MSE diverging (%.4f → %.4f)", prevMSE, mse)
		}
		prevMSE = mse
	}

	// Send Stop to all nodes
	for i, nc := range conns {
		_ = Send(nc.enc, Envelope{Kind: KindStop, Payload: Stop{}})
		nc.conn.Close()
		log.Printf("[coordinator] node %d stopped", i)
	}

	// Build model
	m := &model.LinearModel{
		Weights:      theta,
		Means:        scaler.Means,
		Stds:         scaler.Stds,
		FeatureNames: append(append([]string{}, model.FeatureNames...), "bias"),
		Epochs:       opts.Epochs,
		LearningRate: opts.LearningRate,
		BatchSize:    opts.BatchSize,
		Workers:      len(c.Nodes),
		LossHistory:  lossHistory,
	}

	trainMetrics := m.Evaluate(trainTrips)
	testMetrics := m.Evaluate(testTrips)
	m.TrainMAE, m.TrainRMSE, m.TrainR2 = trainMetrics.MAE, trainMetrics.RMSE, trainMetrics.R2
	m.TestMAE, m.TestRMSE, m.TestR2 = testMetrics.MAE, testMetrics.RMSE, testMetrics.R2

	log.Printf("[coordinator] done | train MAE=%.3f R2=%.4f | test MAE=%.3f R2=%.4f",
		m.TrainMAE, m.TrainR2, m.TestMAE, m.TestR2)

	return m, nil
}

// computeMSECoord calculates MSE locally in the coordinator for loss tracking.
func computeMSECoord(X [][]float64, y, theta []float64) float64 {
	var sum float64
	for i := range X {
		pred := model.DotProduct(X[i], theta)
		d := pred - y[i]
		sum += d * d
	}
	return sum / float64(len(X))
}
