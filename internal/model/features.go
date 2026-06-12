// Package model implementa la regresión lineal con entrenamiento SGD paralelo.
package model

import (
	"math"

	"github.com/remii/tf-concurrente/internal/loader"
)

const NumFeatures = 8 // sin incluir el bias; el vector de pesos tiene largo 9

// FeatureNames describe cada posición del vector de features (sin bias).
var FeatureNames = []string{
	"trip_distance",
	"sin_hour",
	"cos_hour",
	"is_weekend",
	"pickup_lat",
	"pickup_lon",
	"is_airport",
	"passenger_count",
}

// ExtractRaw devuelve el vector de 8 features sin estandarizar para un Trip.
// Orden: distance, sin(hour), cos(hour), isWeekend, lat, lon, isAirport, passengers.
func ExtractRaw(t loader.Trip) [NumFeatures]float64 {
	hour := float64(t.HourOfDay)
	isWeekend := 0.0
	if t.DayOfWeek == 0 || t.DayOfWeek == 6 { // Sunday=0, Saturday=6
		isWeekend = 1.0
	}
	isAirport := 0.0
	if t.RateCodeID == 2 || t.RateCodeID == 3 {
		isAirport = 1.0
	}
	return [NumFeatures]float64{
		t.TripDistance,
		math.Sin(2 * math.Pi * hour / 24),
		math.Cos(2 * math.Pi * hour / 24),
		isWeekend,
		t.PickupLat,
		t.PickupLon,
		isAirport,
		float64(t.PassengerCount),
	}
}

// Scaler contiene las estadísticas para estandarización z-score.
// Se ajusta (Fit) sobre el conjunto de entrenamiento y se aplica (Transform)
// tanto al train como al test set.
type Scaler struct {
	Means [NumFeatures]float64
	Stds  [NumFeatures]float64
}

// Fit calcula media y desviación estándar de cada feature sobre el conjunto dado.
// Usa algoritmo de dos pasadas para mayor estabilidad numérica.
func FitScaler(trips []loader.Trip) Scaler {
	n := float64(len(trips))
	if n == 0 {
		var s Scaler
		for i := range s.Stds {
			s.Stds[i] = 1.0
		}
		return s
	}

	var sums [NumFeatures]float64
	for _, t := range trips {
		raw := ExtractRaw(t)
		for j := 0; j < NumFeatures; j++ {
			sums[j] += raw[j]
		}
	}

	var s Scaler
	for j := 0; j < NumFeatures; j++ {
		s.Means[j] = sums[j] / n
	}

	var varSums [NumFeatures]float64
	for _, t := range trips {
		raw := ExtractRaw(t)
		for j := 0; j < NumFeatures; j++ {
			d := raw[j] - s.Means[j]
			varSums[j] += d * d
		}
	}

	for j := 0; j < NumFeatures; j++ {
		std := math.Sqrt(varSums[j] / n)
		if std < 1e-10 {
			std = 1.0 // evitar división por cero en features constantes
		}
		s.Stds[j] = std
	}
	return s
}

// BuildFeatureVector construye el vector de 9 elementos (8 features + bias)
// para un Trip, aplicando la estandarización z-score.
// El bias (1.0) se agrega al final como elemento 8.
func (s *Scaler) BuildFeatureVector(t loader.Trip) []float64 {
	raw := ExtractRaw(t)
	x := make([]float64, NumFeatures+1)
	for j := 0; j < NumFeatures; j++ {
		x[j] = (raw[j] - s.Means[j]) / s.Stds[j]
	}
	x[NumFeatures] = 1.0 // bias
	return x
}

// BuildMatrix construye la matriz de features X y el vector de targets y
// para un slice de trips. Reutiliza el scaler ya ajustado.
func (s *Scaler) BuildMatrix(trips []loader.Trip) (X [][]float64, y []float64) {
	X = make([][]float64, len(trips))
	y = make([]float64, len(trips))
	for i, t := range trips {
		X[i] = s.BuildFeatureVector(t)
		y[i] = t.DurationMin
	}
	return
}

// DotProduct calcula el producto punto entre dos slices.
func DotProduct(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
