package model

import (
	"bufio"
	"encoding/json"
	"fmt"
	"math"
	"os"

	"tf-concurrente/internal/loader"
)

// LinearModel representa el modelo entrenado.
// La interfaz está diseñada para ser consumida directamente por la API REST
// del entregable 2 sin modificaciones.
type LinearModel struct {
	Weights      []float64            `json:"weights"`
	Means        [NumFeatures]float64 `json:"means"`
	Stds         [NumFeatures]float64 `json:"stds"`
	FeatureNames []string             `json:"feature_names"`
	// Metadatos de entrenamiento
	Epochs       int       `json:"epochs"`
	LearningRate float64   `json:"learning_rate"`
	BatchSize    int       `json:"batch_size"`
	Workers      int       `json:"workers"`
	LossHistory  []float64 `json:"loss_history"` // MSE por época
	TrainMAE     float64   `json:"train_mae"`
	TrainRMSE    float64   `json:"train_rmse"`
	TrainR2      float64   `json:"train_r2"`
	TestMAE      float64   `json:"test_mae"`
	TestRMSE     float64   `json:"test_rmse"`
	TestR2       float64   `json:"test_r2"`
}

// Predict predice la duración en minutos para un vector de features ya estandarizado.
// x debe tener largo NumFeatures+1 (incluye bias en posición final).
func (m *LinearModel) Predict(x []float64) float64 {
	return DotProduct(m.Weights, x)
}

// PredictTrip construye el vector de features para un Trip y predice la duración.
func (m *LinearModel) PredictTrip(t loader.Trip) float64 {
	s := &Scaler{Means: m.Means, Stds: m.Stds}
	return m.Predict(s.BuildFeatureVector(t))
}

// Save serializa el modelo a JSON en la ruta indicada.
func (m *LinearModel) Save(path string) error {
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return fmt.Errorf("serializar modelo: %w", err)
	}
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("escribir modelo: %w", err)
	}
	return nil
}

// LoadModel deserializa un modelo guardado con Save.
func LoadModel(path string) (*LinearModel, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("leer modelo: %w", err)
	}
	var m LinearModel
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, fmt.Errorf("deserializar modelo: %w", err)
	}
	return &m, nil
}

// Metrics agrupa las métricas de evaluación.
type Metrics struct {
	MAE  float64
	RMSE float64
	R2   float64
}

// Evaluate calcula MAE, RMSE y R² para un conjunto de trips.
func (m *LinearModel) Evaluate(trips []loader.Trip) Metrics {
	if len(trips) == 0 {
		return Metrics{}
	}
	s := &Scaler{Means: m.Means, Stds: m.Stds}

	var sumAbs, sumSq, sumY float64
	for _, t := range trips {
		pred := m.Predict(s.BuildFeatureVector(t))
		actual := t.DurationMin
		err := pred - actual
		sumAbs += math.Abs(err)
		sumSq += err * err
		sumY += actual
	}

	n := float64(len(trips))
	meanY := sumY / n

	var ssTot float64
	for _, t := range trips {
		d := t.DurationMin - meanY
		ssTot += d * d
	}

	r2 := 0.0
	if ssTot > 0 {
		r2 = 1.0 - sumSq/ssTot
	}

	return Metrics{
		MAE:  sumAbs / n,
		RMSE: math.Sqrt(sumSq / n),
		R2:   r2,
	}
}

// BaselineMetrics calcula las métricas de las dos líneas base:
// (a) predecir la media global, (b) predecir con velocidad media global.
func BaselineMetrics(train, test []loader.Trip) (meanBaseline, speedBaseline Metrics) {
	if len(train) == 0 || len(test) == 0 {
		return
	}

	// Calcular media y velocidad media en train
	var sumDur, sumSpeed float64
	var speedCount float64
	for _, t := range train {
		sumDur += t.DurationMin
		if t.DurationMin > 0 {
			speedMph := t.TripDistance / (t.DurationMin / 60.0)
			sumSpeed += speedMph
			speedCount++
		}
	}
	globalMean := sumDur / float64(len(train))
	globalSpeed := sumSpeed / speedCount

	// Evaluar en test
	var sumAbsMean, sumSqMean float64
	var sumAbsSpeed, sumSqSpeed float64
	var sumY, ssTot float64
	meanY := func() float64 {
		var s float64
		for _, t := range test {
			s += t.DurationMin
		}
		return s / float64(len(test))
	}()

	for _, t := range test {
		actual := t.DurationMin
		d := actual - meanY
		ssTot += d * d

		// baseline (a): predecir media
		errMean := globalMean - actual
		sumAbsMean += math.Abs(errMean)
		sumSqMean += errMean * errMean

		// baseline (b): distancia / velocidad media
		predSpeed := 0.0
		if globalSpeed > 0 {
			predSpeed = t.TripDistance / globalSpeed * 60.0 // convertir horas a minutos
		}
		errSpeed := predSpeed - actual
		sumAbsSpeed += math.Abs(errSpeed)
		sumSqSpeed += errSpeed * errSpeed

		sumY += actual
	}

	n := float64(len(test))
	_ = sumY

	r2 := func(ssSq float64) float64 {
		if ssTot > 0 {
			return 1.0 - ssSq/ssTot
		}
		return 0
	}

	meanBaseline = Metrics{
		MAE:  sumAbsMean / n,
		RMSE: math.Sqrt(sumSqMean / n),
		R2:   r2(sumSqMean),
	}
	speedBaseline = Metrics{
		MAE:  sumAbsSpeed / n,
		RMSE: math.Sqrt(sumSqSpeed / n),
		R2:   r2(sumSqSpeed),
	}
	return
}

// SaveLossCSV exporta la curva de loss (MSE por época) a un archivo CSV.
func (m *LinearModel) SaveLossCSV(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("crear loss CSV: %w", err)
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	fmt.Fprintln(w, "epoca,mse_train")
	for i, mse := range m.LossHistory {
		fmt.Fprintf(w, "%d,%.6f\n", i+1, mse)
	}
	return w.Flush()
}

// PrintComparativeTable imprime la tabla comparativa de métricas.
func PrintComparativeTable(model Metrics, mean, speed Metrics) {
	fmt.Printf("\n=== Evaluación en Test ===\n")
	fmt.Printf("%-25s %8s %8s %8s\n", "Modelo", "MAE", "RMSE", "R²")
	fmt.Printf("%-25s %8.2f %8.2f %8.4f\n", "Regresión Lineal", model.MAE, model.RMSE, model.R2)
	fmt.Printf("%-25s %8.2f %8.2f %8.4f\n", "Baseline: media", mean.MAE, mean.RMSE, mean.R2)
	fmt.Printf("%-25s %8.2f %8.2f %8.4f\n", "Baseline: vel. media", speed.MAE, speed.RMSE, speed.R2)
	fmt.Println()
}
