package model

import (
	"math"
	"os"
	"testing"
	"time"

	"tf-concurrente/internal/loader"
)

// TestGradient verifica el cálculo del gradiente contra un resultado manual.
//
// Datos:
//
//	X = [[2.0, 1.0], [4.0, 1.0]]   (2 features + bias)
//	y = [3.0, 7.0]
//	θ = [1.0, 0.0]
//
// Predicciones: [2.0, 4.0]
// Errores:      [-1.0, -3.0]
// grad[0] = ((-1)×2 + (-3)×4) / 2 = -7.0
// grad[1] = ((-1)×1 + (-3)×1) / 2 = -2.0
func TestGradient(t *testing.T) {
	X := [][]float64{
		{2.0, 1.0},
		{4.0, 1.0},
	}
	y := []float64{3.0, 7.0}
	theta := []float64{1.0, 0.0}
	batchIdx := []int{0, 1}

	grad := computeGrad(X, y, theta, batchIdx)

	if len(grad) != 2 {
		t.Fatalf("len(grad): got %d, want 2", len(grad))
	}
	if math.Abs(grad[0]-(-7.0)) > 1e-9 {
		t.Errorf("grad[0]: got %f, want -7.0", grad[0])
	}
	if math.Abs(grad[1]-(-2.0)) > 1e-9 {
		t.Errorf("grad[1]: got %f, want -2.0", grad[1])
	}
}

// TestGradientSinglePoint verifica el gradiente con un solo punto.
func TestGradientSinglePoint(t *testing.T) {
	X := [][]float64{{1.0, 0.0, 1.0}}
	y := []float64{5.0}
	theta := []float64{2.0, 0.0, 0.0}
	// pred = 2.0, err = 2.0 - 5.0 = -3.0
	// grad[0] = -3.0 * 1.0 = -3.0
	// grad[1] = -3.0 * 0.0 = 0.0
	// grad[2] = -3.0 * 1.0 = -3.0
	grad := computeGrad(X, y, theta, []int{0})

	want := []float64{-3.0, 0.0, -3.0}
	for i, w := range want {
		if math.Abs(grad[i]-w) > 1e-9 {
			t.Errorf("grad[%d]: got %f, want %f", i, grad[i], w)
		}
	}
}

// makeTrip crea un Trip sintético con los valores dados.
func makeTrip(distance, durationMin float64, hour, weekday int) loader.Trip {
	base := time.Date(2015, 1, 5, hour, 0, 0, 0, time.UTC) // lunes por defecto
	// Ajustar weekday: time.Weekday(weekday)
	base = base.AddDate(0, 0, weekday-int(base.Weekday()))
	return loader.Trip{
		PickupTime:     base,
		DurationMin:    durationMin,
		TripDistance:   distance,
		PickupLat:      40.75,
		PickupLon:      -73.98,
		RateCodeID:     1,
		PassengerCount: 1,
		HourOfDay:      hour,
		DayOfWeek:      weekday,
	}
}

// TestTrainSmall verifica que el MSE decrece durante el entrenamiento.
// Los datos sintéticos tienen una relación lineal exacta: DurationMin = 5*distance + 5.
// Con lr=0.05 y suficientes épocas, el modelo debe converger hacia esa relación.
func TestTrainSmall(t *testing.T) {
	// Variar distancias y horas para que el scaler tenga varianza > 0
	trips := make([]loader.Trip, 500)
	for i := range trips {
		dist := 0.5 + float64(i%20)*0.5 // 0.5 a 10 mi
		hour := (i % 24)
		dur := 5.0*dist + 5.0
		trips[i] = makeTrip(dist, dur, hour, 1+i%5)
	}

	opts := TrainOptions{
		Workers:       2,
		Epochs:        100,
		LearningRate:  0.05,
		BatchSize:     32,
		Seed:          42,
		TrainFraction: 0.8,
	}

	m, err := Train(trips, opts)
	if err != nil {
		t.Fatalf("Train: %v", err)
	}

	// El MSE debe descender durante el entrenamiento (convergencia básica)
	if m.TrainRMSE >= 50.0 {
		t.Errorf("TrainRMSE demasiado alto: %.4f (esperado < 50.0)", m.TrainRMSE)
	}
	// R² en train debe ser positivo (mejor que predecir la media)
	if m.TrainR2 <= 0 {
		t.Errorf("TrainR2 no positivo: %.4f (modelo peor que baseline)", m.TrainR2)
	}
}

// TestModelJSON verifica que el modelo se puede serializar y deserializar
// sin pérdida de información.
func TestModelJSON(t *testing.T) {
	trips := make([]loader.Trip, 200)
	for i := range trips {
		dist := 1.0 + float64(i%10)
		trips[i] = makeTrip(dist, dist*5+5, 9, 1)
	}

	opts := TrainOptions{
		Workers:       1,
		Epochs:        5,
		LearningRate:  0.01,
		BatchSize:     32,
		Seed:          7,
		TrainFraction: 0.8,
	}

	m, err := Train(trips, opts)
	if err != nil {
		t.Fatalf("Train: %v", err)
	}

	tmpFile := t.TempDir() + "/model.json"
	if err := m.Save(tmpFile); err != nil {
		t.Fatalf("Save: %v", err)
	}

	m2, err := LoadModel(tmpFile)
	if err != nil {
		t.Fatalf("LoadModel: %v", err)
	}

	if len(m2.Weights) != len(m.Weights) {
		t.Fatalf("Weights length: got %d, want %d", len(m2.Weights), len(m.Weights))
	}
	for i, w := range m.Weights {
		if math.Abs(m2.Weights[i]-w) > 1e-12 {
			t.Errorf("Weights[%d]: got %f, want %f", i, m2.Weights[i], w)
		}
	}

	// Verificar que el archivo existe
	if _, err := os.Stat(tmpFile); err != nil {
		t.Errorf("archivo del modelo no existe: %v", err)
	}
}

// TestPredictZero verifica que Predict con θ=0 devuelve 0.
func TestPredictZero(t *testing.T) {
	m := &LinearModel{Weights: make([]float64, NumFeatures+1)}
	x := make([]float64, NumFeatures+1)
	x[0] = 5.0 // algún valor
	x[NumFeatures] = 1.0
	if got := m.Predict(x); got != 0.0 {
		t.Errorf("Predict con θ=0: got %f, want 0.0", got)
	}
}

// TestFitScalerBasic verifica que FitScaler produce medias y stds razonables.
func TestFitScalerBasic(t *testing.T) {
	trips := []loader.Trip{
		makeTrip(1.0, 10.0, 8, 1),
		makeTrip(3.0, 20.0, 12, 3),
		makeTrip(5.0, 30.0, 16, 5),
	}
	s := FitScaler(trips)

	// La media de trip_distance debe ser (1+3+5)/3 = 3.0
	if math.Abs(s.Means[0]-3.0) > 1e-9 {
		t.Errorf("Means[0] (distance): got %f, want 3.0", s.Means[0])
	}
	// Todos los stds deben ser > 0
	for i, std := range s.Stds {
		if std <= 0 {
			t.Errorf("Stds[%d]: got %f, debe ser positivo", i, std)
		}
	}
}

// TestMakePartitions verifica que las particiones cubren todos los índices.
func TestMakePartitions(t *testing.T) {
	parts := makePartitions(10, 3)
	if len(parts) != 3 {
		t.Fatalf("len: got %d, want 3", len(parts))
	}
	// La primera empieza en 0
	if parts[0].start != 0 {
		t.Errorf("parts[0].start: got %d, want 0", parts[0].start)
	}
	// La última termina en n
	if parts[2].end != 10 {
		t.Errorf("parts[2].end: got %d, want 10", parts[2].end)
	}
	// Sin huecos ni solapamientos
	for i := 1; i < len(parts); i++ {
		if parts[i].start != parts[i-1].end {
			t.Errorf("hueco entre particiones %d y %d", i-1, i)
		}
	}
}
