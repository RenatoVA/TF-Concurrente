package model

import (
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"time"

	"tf-concurrente/internal/loader"
)

// TrainOptions configura el entrenamiento.
type TrainOptions struct {
	Workers       int
	Epochs        int
	LearningRate  float64
	BatchSize     int
	Seed          int64
	TrainFraction float64
}

func (o *TrainOptions) setDefaults() {
	if o.Workers <= 0 {
		o.Workers = 1
	}
	if o.Epochs <= 0 {
		o.Epochs = 10
	}
	if o.LearningRate <= 0 {
		o.LearningRate = 0.01
	}
	if o.BatchSize <= 0 {
		o.BatchSize = 1024
	}
	if o.TrainFraction <= 0 || o.TrainFraction >= 1 {
		o.TrainFraction = 0.8
	}
}

// msgKind distingue los dos tipos de mensaje que fluyen por gradCh.
type msgKind int

const (
	kindGrad msgKind = iota // gradiente parcial de un mini-batch
	kindDone                // señal de fin de época del worker
)

// gradMsg es el mensaje que cada worker envía al agregador.
// Usar un canal único con mensajes tipados garantiza que el kindDone de un
// worker SIEMPRE llega después de todos sus kindGrad de esa época,
// porque Go preserva el orden de envío en un canal.
// Esto elimina la necesidad de un WaitGroup y los races asociados.
type gradMsg struct {
	kind msgKind
	grad []float64 // sólo válido si kind == kindGrad
	n    int       // tamaño del batch; sólo válido si kind == kindGrad
}

// Train entrena el modelo con mini-batch SGD paralelo y devuelve el modelo final.
//
// Arquitectura de concurrencia (race-free):
//
//	Coordinator ──paramCh[i]──▶ Worker_i ──┐
//	                                        ├── gradCh ──▶ Aggregator
//	Coordinator ◀──epochResultCh ───────── ─┘
//
// El agregador es la ÚNICA goroutine que ESCRIBE θ.
// Los workers reciben una COPIA de θ por su canal paramCh y nunca
// acceden al θ compartido.
//
// La barrera de época se implementa con mensajes kindDone en gradCh:
// cada worker envía un kindDone como último mensaje de la época.
// El agregador cuenta W dones antes de aplicar la actualización.
// Esto garantiza ordering correcto porque Go preserva el orden en un canal:
// los gradientes de un worker siempre llegan antes de su propio done.
func Train(trips []loader.Trip, opts TrainOptions) (*LinearModel, error) {
	opts.setDefaults()

	if len(trips) == 0 {
		return nil, fmt.Errorf("sin datos para entrenar")
	}

	// Shuffle determinista para reproducibilidad
	rng := rand.New(rand.NewPCG(uint64(opts.Seed), 0))
	indices := make([]int, len(trips))
	for i := range indices {
		indices[i] = i
	}
	rng.Shuffle(len(indices), func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })

	// Split train/test
	splitAt := int(float64(len(trips)) * opts.TrainFraction)
	trainIdx := indices[:splitAt]
	testIdx := indices[splitAt:]

	trainTrips := make([]loader.Trip, len(trainIdx))
	testTrips := make([]loader.Trip, len(testIdx))
	for i, idx := range trainIdx {
		trainTrips[i] = trips[idx]
	}
	for i, idx := range testIdx {
		testTrips[i] = trips[idx]
	}

	log.Printf("Train/Test split: %d / %d trips", len(trainTrips), len(testTrips))

	// Ajustar scaler SOLO sobre el train set
	scaler := FitScaler(trainTrips)

	// Construir matrices de features una sola vez (evita recalcular en cada época)
	log.Printf("Construyendo matrices de features...")
	X, y := scaler.BuildMatrix(trainTrips)

	// θ inicializado en cero
	nWeights := NumFeatures + 1
	theta := make([]float64, nWeights)

	// Dividir el train set en W particiones contiguas
	W := opts.Workers
	parts := makePartitions(len(X), W)

	// paramCh[i]: coordinator → worker i (buffered 1 evita bloqueo al broadcast)
	// gradCh: workers → aggregator (buffered W*4 para absorber ráfagas)
	// epochResultCh: aggregator → coordinator con θ actualizado
	paramCh := make([]chan []float64, W)
	for i := range paramCh {
		paramCh[i] = make(chan []float64, 1)
	}
	gradCh := make(chan gradMsg, W*4)
	epochResultCh := make(chan []float64, 1)

	// Lanzar workers permanentes (uno por partición)
	for i := 0; i < W; i++ {
		go runWorker(i, X, y, parts[i], paramCh[i], gradCh, opts.BatchSize, opts.Seed)
	}

	// Lanzar agregador permanente
	go runAggregator(gradCh, epochResultCh, theta, opts.LearningRate, W, opts.Epochs)

	// Loop de coordinación: por cada época, broadcast θ y esperar resultado
	lossHistory := make([]float64, 0, opts.Epochs)
	prevMSE := math.MaxFloat64
	for epoch := 1; epoch <= opts.Epochs; epoch++ {
		epochStart := time.Now()

		// Broadcast: cada worker recibe su propia copia de θ
		// La copia es fundamental — los workers NO deben compartir el mismo slice
		for i := 0; i < W; i++ {
			thetaCopy := make([]float64, len(theta))
			copy(thetaCopy, theta)
			paramCh[i] <- thetaCopy
		}

		// Esperar θ actualizado del agregador
		theta = <-epochResultCh

		mse := computeMSE(X, y, theta)
		lossHistory = append(lossHistory, mse)
		elapsed := time.Since(epochStart).Round(time.Millisecond)
		log.Printf("Época %3d/%d | MSE train: %8.4f | tiempo: %v", epoch, opts.Epochs, mse, elapsed)

		if mse > prevMSE*10 {
			log.Printf("ADVERTENCIA: MSE divergiendo (%.4f → %.4f), considerar reducir --lr", prevMSE, mse)
		}
		prevMSE = mse
	}

	// Señalar fin a los workers cerrando sus canales de parámetros
	for i := 0; i < W; i++ {
		close(paramCh[i])
	}

	// Construir modelo final con estadísticas
	m := &LinearModel{
		Weights:      theta,
		Means:        scaler.Means,
		Stds:         scaler.Stds,
		FeatureNames: append(FeatureNames, "bias"),
		Epochs:       opts.Epochs,
		LearningRate: opts.LearningRate,
		BatchSize:    opts.BatchSize,
		Workers:      W,
		LossHistory:  lossHistory,
	}

	trainMetrics := m.Evaluate(trainTrips)
	testMetrics := m.Evaluate(testTrips)
	m.TrainMAE, m.TrainRMSE, m.TrainR2 = trainMetrics.MAE, trainMetrics.RMSE, trainMetrics.R2
	m.TestMAE, m.TestRMSE, m.TestR2 = testMetrics.MAE, testMetrics.RMSE, testMetrics.R2

	meanBaseline, speedBaseline := BaselineMetrics(trainTrips, testTrips)
	PrintComparativeTable(testMetrics, meanBaseline, speedBaseline)

	return m, nil
}

// runWorker procesa una partición de la matriz X en mini-batches.
// Recibe θ al inicio de cada época y NUNCA escribe θ compartido.
// Al terminar todos sus mini-batches, envía kindDone como último mensaje,
// garantizando que el agregador reciba todos los gradientes antes del done.
func runWorker(id int, X [][]float64, y []float64, part partition,
	paramCh <-chan []float64, gradCh chan<- gradMsg, batchSize int, seed int64) {

	// RNG propio por worker (semillas distintas evitan correlación entre workers)
	workerRng := rand.New(rand.NewPCG(uint64(seed)+uint64(id)*1000, 0))

	localIdx := make([]int, part.end-part.start)
	for i := range localIdx {
		localIdx[i] = part.start + i
	}

	for theta := range paramCh {
		// Shuffle local de índices en cada época para mejor convergencia
		workerRng.Shuffle(len(localIdx), func(i, j int) {
			localIdx[i], localIdx[j] = localIdx[j], localIdx[i]
		})

		// Iterar mini-batches: calcular y enviar gradiente de cada uno
		for start := 0; start < len(localIdx); start += batchSize {
			end := start + batchSize
			if end > len(localIdx) {
				end = len(localIdx)
			}
			batch := localIdx[start:end]
			grad := computeGrad(X, y, theta, batch)
			// Cada grad es un slice recién asignado — sin aliasing entre mensajes
			gradCh <- gradMsg{kind: kindGrad, grad: grad, n: len(batch)}
		}

		// kindDone SIEMPRE se envía DESPUÉS de todos los gradientes de esta época,
		// porque Go preserva el orden de envío en un canal dado.
		gradCh <- gradMsg{kind: kindDone}
	}
}

// runAggregator es la ÚNICA goroutine que escribe θ.
// Acumula gradientes de todos los workers, cuenta W señales kindDone
// para saber que la época terminó, aplica la actualización SGD y envía
// el nuevo θ al coordinator.
func runAggregator(gradCh <-chan gradMsg, epochResultCh chan<- []float64,
	theta []float64, lr float64, workers, epochs int) {

	nWeights := len(theta)

	for range epochs {
		accumGrad := make([]float64, nWeights)
		totalN := 0
		doneCount := 0

		// Leer mensajes hasta recibir W señales kindDone (una por worker)
		for doneCount < workers {
			msg := <-gradCh
			switch msg.kind {
			case kindGrad:
				// Acumular gradiente ponderado por tamaño del batch
				for j := 0; j < nWeights; j++ {
					accumGrad[j] += msg.grad[j] * float64(msg.n)
				}
				totalN += msg.n
			case kindDone:
				doneCount++
			}
		}

		// Aplicar actualización SGD: θ ← θ - lr·∇J_avg
		if totalN > 0 {
			for j := 0; j < nWeights; j++ {
				theta[j] -= lr * accumGrad[j] / float64(totalN)
			}
		}

		// Enviar copia del θ actualizado al coordinator
		thetaCopy := make([]float64, nWeights)
		copy(thetaCopy, theta)
		epochResultCh <- thetaCopy
	}
}

// computeGrad calcula el gradiente del MSE para un mini-batch.
// grad_j = (1/B) Σ_i (pred_i - y_i) · x_{ij}
// Función pura — segura para llamar concurrentemente desde cualquier goroutine.
func computeGrad(X [][]float64, y []float64, theta []float64, batchIdx []int) []float64 {
	nWeights := len(theta)
	grad := make([]float64, nWeights)
	for _, i := range batchIdx {
		pred := DotProduct(X[i], theta)
		err := pred - y[i]
		for j := 0; j < nWeights; j++ {
			grad[j] += err * X[i][j]
		}
	}
	n := float64(len(batchIdx))
	for j := range grad {
		grad[j] /= n
	}
	return grad
}

// computeMSE calcula el error cuadrático medio en toda la matriz X.
func computeMSE(X [][]float64, y []float64, theta []float64) float64 {
	var sum float64
	for i := range X {
		pred := DotProduct(X[i], theta)
		d := pred - y[i]
		sum += d * d
	}
	return sum / float64(len(X))
}

// partition define el rango [start, end) de un worker en la matriz X.
type partition struct {
	start, end int
}

// makePartitions divide n elementos en W particiones contiguas.
func makePartitions(n, W int) []partition {
	exported := MakePartitions(n, W)
	parts := make([]partition, len(exported))
	for i, p := range exported {
		parts[i] = partition{start: p.Start, end: p.End}
	}
	return parts
}

// Partition defines the range [Start, End) for a cluster worker in matrix X.
// Exported for use by the distributed Coordinator in internal/cluster.
type Partition struct {
	Start, End int
}

// MakePartitions divides n elements into W contiguous partitions.
// The last partition absorbs the integer remainder of n/W.
func MakePartitions(n, W int) []Partition {
	parts := make([]Partition, W)
	size := n / W
	for i := 0; i < W; i++ {
		parts[i].Start = i * size
		parts[i].End = (i + 1) * size
	}
	parts[W-1].End = n
	return parts
}

// AccumulateGradient sums gradients over all rows in (X, y) given theta.
// Processes sequentially in mini-batches of 1024 for memory efficiency,
// reusing computeGrad to guarantee per-example math identical to local SGD.
// Returns gradSum (not averaged) and n (number of examples).
func AccumulateGradient(X [][]float64, y, theta []float64) (gradSum []float64, n int) {
	n = len(X)
	gradSum = make([]float64, len(theta))
	if n == 0 {
		return
	}
	const batchSz = 1024
	for start := 0; start < n; start += batchSz {
		end := start + batchSz
		if end > n {
			end = n
		}
		batchIdx := make([]int, end-start)
		for k := range batchIdx {
			batchIdx[k] = start + k
		}
		batchGrad := computeGrad(X, y, theta, batchIdx)
		sz := float64(end - start)
		for j := range gradSum {
			gradSum[j] += batchGrad[j] * sz
		}
	}
	return
}
