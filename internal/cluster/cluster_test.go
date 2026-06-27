package cluster_test

import (
	"math"
	"net"
	"testing"
	"time"

	"tf-concurrente/internal/cluster"
	"tf-concurrente/internal/loader"
	"tf-concurrente/internal/model"
)

// makeSyntheticTrips builds deterministic trips for testing.
func makeSyntheticTrips(n int) []loader.Trip {
	trips := make([]loader.Trip, n)
	for i := range trips {
		f := float64(i)
		trips[i] = loader.Trip{
			TripDistance:   1.0 + f*0.05,
			DurationMin:    3.0 + f*0.15,
			PickupLat:      40.70 + float64(i%10)*0.01,
			PickupLon:      -73.90 + float64(i%10)*0.01,
			RateCodeID:     1 + i%3,
			PassengerCount: 1 + i%4,
			HourOfDay:      i % 24,
			DayOfWeek:      i % 7,
		}
	}
	return trips
}

// TestGobRoundTrip verifies that every message type survives encode→decode.
func TestGobRoundTrip(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()

	msgs := []cluster.Envelope{
		{Kind: cluster.KindDataInit, Payload: cluster.DataInit{X: [][]float64{{1, 2}}, Y: []float64{3}}},
		{Kind: cluster.KindParam, Payload: cluster.ParamUpdate{Epoch: 1, Theta: []float64{0.1, 0.2}}},
		{Kind: cluster.KindGradient, Payload: cluster.GradientResult{Epoch: 1, Grad: []float64{0.5}, N: 10}},
		{Kind: cluster.KindStop, Payload: cluster.Stop{}},
		{Kind: cluster.KindAck, Payload: cluster.Ack{}},
	}

	// Server side: receive all messages
	done := make(chan []cluster.Envelope, 1)
	go func() {
		conn, _ := ln.Accept()
		defer conn.Close()
		dec := cluster.NewDecoder(conn)
		var received []cluster.Envelope
		for i := 0; i < len(msgs); i++ {
			env, err := cluster.Recv(dec)
			if err != nil {
				t.Errorf("recv %d: %v", i, err)
				return
			}
			received = append(received, env)
		}
		done <- received
	}()

	// Client side: send all messages
	conn, err := net.Dial("tcp", ln.Addr().String())
	if err != nil {
		t.Fatal(err)
	}
	defer conn.Close()
	enc := cluster.NewEncoder(conn)
	for _, msg := range msgs {
		if err := cluster.Send(enc, msg); err != nil {
			t.Fatalf("send: %v", err)
		}
	}

	select {
	case received := <-done:
		for i, want := range msgs {
			if i >= len(received) {
				t.Fatalf("only got %d messages", len(received))
			}
			if received[i].Kind != want.Kind {
				t.Errorf("msg[%d].Kind = %d, want %d", i, received[i].Kind, want.Kind)
			}
		}
	case <-time.After(5 * time.Second):
		t.Fatal("timeout waiting for messages")
	}
}

// TestEquivalence verifies that distributed training produces theta identical
// to the manual reference computation (tolerance 1e-9). Runs with -race.
func TestEquivalence(t *testing.T) {
	const nTrips = 60
	const numNodes = 2
	const epochs = 5
	const lr = 0.01

	trips := makeSyntheticTrips(nTrips)
	opts := model.TrainOptions{
		Epochs:        epochs,
		LearningRate:  lr,
		BatchSize:     1024,
		TrainFraction: 0.8,
	}

	// Start 2 node servers with OS-assigned ports
	ln1, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln1.Close()
	ln2, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln2.Close()

	go cluster.ServeListener(ln1) //nolint:errcheck
	go cluster.ServeListener(ln2) //nolint:errcheck

	coord := cluster.NewCoordinator([]string{
		ln1.Addr().String(),
		ln2.Addr().String(),
	})

	m, err := coord.TrainDistributed(trips, opts)
	if err != nil {
		t.Fatalf("TrainDistributed: %v", err)
	}

	// Reference: replicate coordinator's exact computation locally
	splitAt := int(float64(len(trips)) * opts.TrainFraction)
	trainTrips := trips[:splitAt]
	scaler := model.FitScaler(trainTrips)
	X, y := scaler.BuildMatrix(trainTrips)

	theta := make([]float64, model.NumFeatures+1)
	for e := 0; e < epochs; e++ {
		parts := model.MakePartitions(len(X), numNodes)
		gradTotal := make([]float64, len(theta))
		totalN := 0
		for _, p := range parts {
			g, n := model.AccumulateGradient(X[p.Start:p.End], y[p.Start:p.End], theta)
			for j := range gradTotal {
				gradTotal[j] += g[j]
			}
			totalN += n
		}
		if totalN > 0 {
			for j := range theta {
				theta[j] -= lr * gradTotal[j] / float64(totalN)
			}
		}
	}

	// Assert theta equality within 1e-9
	if len(m.Weights) != len(theta) {
		t.Fatalf("weight len: got %d, want %d", len(m.Weights), len(theta))
	}
	for j := range theta {
		diff := math.Abs(m.Weights[j] - theta[j])
		if diff > 1e-9 {
			t.Errorf("theta[%d]: distributed=%.12f reference=%.12f diff=%.2e",
				j, m.Weights[j], theta[j], diff)
		}
	}
}
