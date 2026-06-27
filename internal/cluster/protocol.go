// Package cluster implements the distributed TCP parameter-server topology.
// The API coordinator dials nodes; nodes are TCP servers.
package cluster

import (
	"encoding/gob"
	"io"
)

// MsgKind discriminates the type of message in an Envelope.
type MsgKind int

const (
	KindDataInit  MsgKind = iota // coordinator → node: send partition data (once per job)
	KindParam                    // coordinator → node: send theta each epoch
	KindGradient                 // node → coordinator: accumulated gradient result
	KindStop                     // coordinator → node: end of job
	KindAck                      // node → coordinator: acknowledge DataInit
)

// DataInit carries the node's data partition (already scaled).
type DataInit struct {
	X [][]float64
	Y []float64
}

// ParamUpdate carries theta for a given epoch.
type ParamUpdate struct {
	Epoch int
	Theta []float64
}

// Stop signals the end of a training job.
type Stop struct{}

// Ack acknowledges receipt of DataInit.
type Ack struct{}

// GradientResult carries the accumulated gradient sum and count for an epoch.
// Grad is the SUM (not averaged) of per-example gradients over the partition.
type GradientResult struct {
	Epoch int
	Grad  []float64
	N     int
}

// Envelope wraps any message with a kind discriminator for gob decoding.
type Envelope struct {
	Kind    MsgKind
	Payload any
}

func init() {
	gob.Register(DataInit{})
	gob.Register(ParamUpdate{})
	gob.Register(Stop{})
	gob.Register(Ack{})
	gob.Register(GradientResult{})
}

// NewEncoder wraps gob.NewEncoder for use by tests and node/coordinator code.
func NewEncoder(w io.Writer) *gob.Encoder { return gob.NewEncoder(w) }

// NewDecoder wraps gob.NewDecoder for use by tests and node/coordinator code.
func NewDecoder(r io.Reader) *gob.Decoder { return gob.NewDecoder(r) }

// Send encodes and sends an Envelope over the gob stream.
func Send(enc *gob.Encoder, e Envelope) error {
	return enc.Encode(e)
}

// Recv decodes the next Envelope from the gob stream.
func Recv(dec *gob.Decoder) (Envelope, error) {
	var e Envelope
	if err := dec.Decode(&e); err != nil {
		return Envelope{}, err
	}
	return e, nil
}
