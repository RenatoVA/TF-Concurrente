package cluster

import (
	"fmt"
	"log"
	"net"
	"time"

	"tf-concurrente/internal/model"
)

// Serve listens on listenAddr and handles one coordinator connection per job.
// It blocks until the listener is closed.
func Serve(listenAddr string) error {
	ln, err := net.Listen("tcp", listenAddr)
	if err != nil {
		return fmt.Errorf("node listen %s: %w", listenAddr, err)
	}
	log.Printf("[node] listening on %s", listenAddr)
	return ServeListener(ln)
}

// ServeListener accepts connections on ln and handles each job in the caller's
// goroutine. Useful for tests that need OS-assigned ports.
func ServeListener(ln net.Listener) error {
	for {
		conn, err := ln.Accept()
		if err != nil {
			return err // listener closed → caller exits
		}
		handleConn(conn)
	}
}

// handleConn processes a single coordinator connection (one training job).
func handleConn(conn net.Conn) {
	defer conn.Close()
	enc := NewEncoder(conn)
	dec := NewDecoder(conn)

	var X [][]float64
	var Y []float64

	for {
		env, err := Recv(dec)
		if err != nil {
			log.Printf("[node] connection closed: %v", err)
			return
		}

		switch env.Kind {
		case KindDataInit:
			init, ok := env.Payload.(DataInit)
			if !ok {
				log.Printf("[node] bad DataInit payload")
				return
			}
			X = init.X
			Y = init.Y
			log.Printf("[node] received %d rows", len(X))
			if err := Send(enc, Envelope{Kind: KindAck, Payload: Ack{}}); err != nil {
				log.Printf("[node] send ack: %v", err)
				return
			}

		case KindParam:
			update, ok := env.Payload.(ParamUpdate)
			if !ok {
				log.Printf("[node] bad ParamUpdate payload")
				return
			}
			t0 := time.Now()
			gradSum, n := model.AccumulateGradient(X, Y, update.Theta)
			elapsed := time.Since(t0)
			log.Printf("[node] epoch %d | rows=%d | %.1fms", update.Epoch, n, float64(elapsed.Milliseconds()))

			if err := Send(enc, Envelope{
				Kind: KindGradient,
				Payload: GradientResult{
					Epoch: update.Epoch,
					Grad:  gradSum,
					N:     n,
				},
			}); err != nil {
				log.Printf("[node] send gradient: %v", err)
				return
			}

		case KindStop:
			log.Printf("[node] received Stop, resetting")
			return

		default:
			log.Printf("[node] unknown message kind %d", env.Kind)
			return
		}
	}
}
