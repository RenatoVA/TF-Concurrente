// Package loader implementa la carga concurrente del dataset NYC Yellow Taxi.
package loader

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"
)

// colIndex almacena la posición de cada columna relevante en el CSV.
// Se construye una sola vez a partir del header, mapeando por nombre
// (case-insensitive) para ser robusto ante cambios de orden.
type colIndex struct {
	pickup     int
	dropoff    int
	passenger  int
	distance   int
	lon        int // pickup_longitude
	lat        int // pickup_latitude
	rateCode   int
	fare       int
	total      int
}

// buildIndex parsea el header del CSV y devuelve los índices de columna.
// Retorna error si alguna columna requerida no se encuentra.
func buildIndex(header []string) (colIndex, error) {
	m := make(map[string]int, len(header))
	for i, name := range header {
		m[strings.ToLower(strings.TrimSpace(name))] = i
	}

	required := map[string]*int{}
	var idx colIndex
	required["tpep_pickup_datetime"] = &idx.pickup
	required["tpep_dropoff_datetime"] = &idx.dropoff
	required["passenger_count"] = &idx.passenger
	required["trip_distance"] = &idx.distance
	required["pickup_longitude"] = &idx.lon
	required["pickup_latitude"] = &idx.lat
	required["ratecodeid"] = &idx.rateCode
	required["fare_amount"] = &idx.fare
	required["total_amount"] = &idx.total

	for name, ptr := range required {
		pos, ok := m[name]
		if !ok {
			return colIndex{}, fmt.Errorf("columna requerida no encontrada: %s", name)
		}
		*ptr = pos
	}
	return idx, nil
}

var (
	pickupMin = time.Date(2015, 1, 1, 0, 0, 0, 0, time.UTC)
	pickupMax = time.Date(2015, 2, 1, 0, 0, 0, 0, time.UTC)
	dtFormat  = "2006-01-02 15:04:05"
)

// parseLine parsea un slice de campos ya separados y devuelve un Trip válido.
// El segundo valor de retorno es el número de regla violada (0 = OK, 1-9 = R1..R9).
// Esta función es stateless y puede llamarse de forma concurrente sin sincronización.
func parseLine(fields []string, idx colIndex) (Trip, int) {
	// R1: número incorrecto de campos o campos clave no parseables
	if len(fields) != 19 {
		return Trip{}, 1
	}

	pickup, err := time.Parse(dtFormat, strings.TrimSpace(fields[idx.pickup]))
	if err != nil {
		return Trip{}, 1
	}
	dropoff, err := time.Parse(dtFormat, strings.TrimSpace(fields[idx.dropoff]))
	if err != nil {
		return Trip{}, 1
	}
	passenger, err := strconv.Atoi(strings.TrimSpace(fields[idx.passenger]))
	if err != nil {
		return Trip{}, 1
	}
	distance, err := strconv.ParseFloat(strings.TrimSpace(fields[idx.distance]), 64)
	if err != nil {
		return Trip{}, 1
	}
	lon, err := strconv.ParseFloat(strings.TrimSpace(fields[idx.lon]), 64)
	if err != nil {
		return Trip{}, 1
	}
	lat, err := strconv.ParseFloat(strings.TrimSpace(fields[idx.lat]), 64)
	if err != nil {
		return Trip{}, 1
	}
	rateCode, err := strconv.Atoi(strings.TrimSpace(fields[idx.rateCode]))
	if err != nil {
		return Trip{}, 1
	}
	fare, err := strconv.ParseFloat(strings.TrimSpace(fields[idx.fare]), 64)
	if err != nil {
		return Trip{}, 1
	}
	total, err := strconv.ParseFloat(strings.TrimSpace(fields[idx.total]), 64)
	if err != nil {
		return Trip{}, 1
	}

	// R2: pickup fuera del rango [2015-01-01, 2015-02-01)
	if pickup.Before(pickupMin) || !pickup.Before(pickupMax) {
		return Trip{}, 2
	}

	// R3: dropoff <= pickup
	if !dropoff.After(pickup) {
		return Trip{}, 3
	}

	// R4: duración fuera de [1, 180] minutos
	durationMin := dropoff.Sub(pickup).Minutes()
	if durationMin < 1.0 || durationMin > 180.0 {
		return Trip{}, 4
	}

	// R5: coordenadas de pickup fuera del bounding box de NYC
	if lat < 40.50 || lat > 41.00 || lon < -74.30 || lon > -73.70 {
		return Trip{}, 5
	}

	// R6: distancia no positiva o excesiva
	if distance <= 0 || distance > 100 {
		return Trip{}, 6
	}

	// R7: tarifa o total no positivos
	if fare <= 0 || total <= 0 {
		return Trip{}, 7
	}

	// R8: número de pasajeros fuera de [1, 6]
	if passenger < 1 || passenger > 6 {
		return Trip{}, 8
	}

	// R9: velocidad media fuera de [0.5, 80] mph
	// durationMin >= 1 por R4, así que no hay división por cero
	speedMph := distance / (durationMin / 60.0)
	if math.IsNaN(speedMph) || speedMph < 0.5 || speedMph > 80.0 {
		return Trip{}, 9
	}

	t := Trip{
		PickupTime:     pickup,
		DurationMin:    durationMin,
		TripDistance:   distance,
		PickupLat:      lat,
		PickupLon:      lon,
		RateCodeID:     rateCode,
		PassengerCount: passenger,
		HourOfDay:      pickup.Hour(),
		DayOfWeek:      int(pickup.Weekday()),
	}
	return t, 0
}
