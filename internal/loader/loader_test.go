package loader

import (
	"fmt"
	"strings"
	"testing"
	"time"
)

// validLine devuelve una fila CSV correcta en forma de slice de campos.
// Usada como base para los tests de cada regla: se muta un campo a la vez.
func validFields() []string {
	return []string{
		"1",                    // VendorID
		"2015-01-10 08:00:00", // tpep_pickup_datetime
		"2015-01-10 08:20:00", // tpep_dropoff_datetime  (20 min)
		"2",                    // passenger_count
		"3.5",                  // trip_distance          (3.5 mi en 20 min = 10.5 mph ✓)
		"-73.985",              // pickup_longitude
		"40.750",               // pickup_latitude
		"1",                    // RateCodeID
		"N",                    // store_and_fwd_flag
		"-73.960",              // dropoff_longitude
		"40.760",               // dropoff_latitude
		"1",                    // payment_type
		"12.00",                // fare_amount
		"0.50",                 // extra
		"0.50",                 // mta_tax
		"2.00",                 // tip_amount
		"0.00",                 // tolls_amount
		"0.30",                 // improvement_surcharge
		"15.30",                // total_amount
	}
}

func validHeader() []string {
	return []string{
		"VendorID", "tpep_pickup_datetime", "tpep_dropoff_datetime",
		"passenger_count", "trip_distance", "pickup_longitude",
		"pickup_latitude", "RateCodeID", "store_and_fwd_flag",
		"dropoff_longitude", "dropoff_latitude", "payment_type",
		"fare_amount", "extra", "mta_tax", "tip_amount",
		"tolls_amount", "improvement_surcharge", "total_amount",
	}
}

func mustBuildIndex(t *testing.T) colIndex {
	t.Helper()
	idx, err := buildIndex(validHeader())
	if err != nil {
		t.Fatalf("buildIndex: %v", err)
	}
	return idx
}

// TestParseLineValid verifica que una fila correcta se parsee sin errores
// y que los campos calculados sean correctos.
func TestParseLineValid(t *testing.T) {
	idx := mustBuildIndex(t)
	fields := validFields()

	trip, rule := parseLine(fields, idx)
	if rule != 0 {
		t.Fatalf("se esperaba rule=0, se obtuvo rule=%d", rule)
	}
	if trip.PassengerCount != 2 {
		t.Errorf("PassengerCount: got %d, want 2", trip.PassengerCount)
	}
	if trip.HourOfDay != 8 {
		t.Errorf("HourOfDay: got %d, want 8", trip.HourOfDay)
	}
	if trip.DayOfWeek != int(time.Saturday) {
		t.Errorf("DayOfWeek: got %d, want Saturday(%d)", trip.DayOfWeek, time.Saturday)
	}
	if trip.RateCodeID != 1 {
		t.Errorf("RateCodeID: got %d, want 1", trip.RateCodeID)
	}
	if trip.TripDistance != 3.5 {
		t.Errorf("TripDistance: got %f, want 3.5", trip.TripDistance)
	}
	if trip.DurationMin != 20.0 {
		t.Errorf("DurationMin: got %f, want 20.0", trip.DurationMin)
	}
}

// TestParseLineR1 verifica el descarte por número incorrecto de campos.
func TestParseLineR1(t *testing.T) {
	idx := mustBuildIndex(t)

	// demasiados campos
	fields := append(validFields(), "extra")
	_, rule := parseLine(fields, idx)
	if rule != 1 {
		t.Errorf("R1 (campos extra): got rule=%d, want 1", rule)
	}

	// muy pocos campos
	_, rule = parseLine(validFields()[:10], idx)
	if rule != 1 {
		t.Errorf("R1 (campos faltantes): got rule=%d, want 1", rule)
	}

	// datetime inválido
	f := validFields()
	f[1] = "no-es-fecha"
	_, rule = parseLine(f, idx)
	if rule != 1 {
		t.Errorf("R1 (datetime inválido): got rule=%d, want 1", rule)
	}
}

// TestParseLineR2 verifica el descarte por pickup fuera de enero 2015.
func TestParseLineR2(t *testing.T) {
	idx := mustBuildIndex(t)
	f := validFields()
	f[1] = "2015-02-15 10:00:00" // después de enero
	f[2] = "2015-02-15 10:30:00"
	_, rule := parseLine(f, idx)
	if rule != 2 {
		t.Errorf("R2: got rule=%d, want 2", rule)
	}

	f2 := validFields()
	f2[1] = "2014-12-31 23:00:00"
	f2[2] = "2014-12-31 23:30:00"
	_, rule = parseLine(f2, idx)
	if rule != 2 {
		t.Errorf("R2 (antes de enero): got rule=%d, want 2", rule)
	}
}

// TestParseLineR3 verifica el descarte por dropoff <= pickup.
func TestParseLineR3(t *testing.T) {
	idx := mustBuildIndex(t)
	f := validFields()
	f[2] = f[1] // dropoff == pickup
	_, rule := parseLine(f, idx)
	if rule != 3 {
		t.Errorf("R3 (igual): got rule=%d, want 3", rule)
	}

	f2 := validFields()
	f2[2] = "2015-01-10 07:00:00" // dropoff antes de pickup
	_, rule = parseLine(f2, idx)
	if rule != 3 {
		t.Errorf("R3 (antes): got rule=%d, want 3", rule)
	}
}

// TestParseLineR4 verifica el descarte por duración fuera de [1, 180] minutos.
func TestParseLineR4(t *testing.T) {
	idx := mustBuildIndex(t)

	// duración < 1 min (30 segundos)
	f := validFields()
	f[2] = "2015-01-10 08:00:30"
	_, rule := parseLine(f, idx)
	if rule != 4 {
		t.Errorf("R4 (<1min): got rule=%d, want 4", rule)
	}

	// duración > 180 min (181 min)
	f2 := validFields()
	f2[2] = "2015-01-10 11:01:00"
	_, rule = parseLine(f2, idx)
	if rule != 4 {
		t.Errorf("R4 (>180min): got rule=%d, want 4", rule)
	}
}

// TestParseLineR5 verifica el descarte por coordenadas de pickup fuera de NYC.
func TestParseLineR5(t *testing.T) {
	idx := mustBuildIndex(t)

	// lat fuera de rango (GPS en 0,0 es el caso más común en el dataset)
	f := validFields()
	f[6] = "0.0" // pickup_latitude = 0
	f[5] = "0.0" // pickup_longitude = 0
	_, rule := parseLine(f, idx)
	if rule != 5 {
		t.Errorf("R5 (GPS en 0,0): got rule=%d, want 5", rule)
	}

	// lat demasiado alta
	f2 := validFields()
	f2[6] = "42.0"
	_, rule = parseLine(f2, idx)
	if rule != 5 {
		t.Errorf("R5 (lat alta): got rule=%d, want 5", rule)
	}
}

// TestParseLineR6 verifica el descarte por distancia inválida.
func TestParseLineR6(t *testing.T) {
	idx := mustBuildIndex(t)

	f := validFields()
	f[4] = "0.0" // distancia = 0
	_, rule := parseLine(f, idx)
	if rule != 6 {
		t.Errorf("R6 (distancia=0): got rule=%d, want 6", rule)
	}

	f2 := validFields()
	f2[4] = "150.0" // distancia > 100
	_, rule = parseLine(f2, idx)
	if rule != 6 {
		t.Errorf("R6 (distancia>100): got rule=%d, want 6", rule)
	}
}

// TestParseLineR7 verifica el descarte por tarifa o total no positivos.
func TestParseLineR7(t *testing.T) {
	idx := mustBuildIndex(t)

	f := validFields()
	f[12] = "-5.00" // fare negativo
	_, rule := parseLine(f, idx)
	if rule != 7 {
		t.Errorf("R7 (fare negativo): got rule=%d, want 7", rule)
	}

	f2 := validFields()
	f2[18] = "0.00" // total = 0
	_, rule = parseLine(f2, idx)
	if rule != 7 {
		t.Errorf("R7 (total=0): got rule=%d, want 7", rule)
	}
}

// TestParseLineR8 verifica el descarte por número de pasajeros inválido.
func TestParseLineR8(t *testing.T) {
	idx := mustBuildIndex(t)

	f := validFields()
	f[3] = "0" // 0 pasajeros
	_, rule := parseLine(f, idx)
	if rule != 8 {
		t.Errorf("R8 (0 pasajeros): got rule=%d, want 8", rule)
	}

	f2 := validFields()
	f2[3] = "7" // 7 pasajeros (> 6)
	_, rule = parseLine(f2, idx)
	if rule != 8 {
		t.Errorf("R8 (7 pasajeros): got rule=%d, want 8", rule)
	}
}

// TestParseLineR9 verifica el descarte por velocidad media fuera de [0.5, 80] mph.
func TestParseLineR9(t *testing.T) {
	idx := mustBuildIndex(t)

	// velocidad > 80 mph: 100 millas en 20 minutos = 300 mph
	f := validFields()
	f[4] = "100.0" // distancia = 100 mi, pero R6 descarta > 100; usar 99.9
	f[4] = "99.9"
	// 99.9 mi en 20 min = 299.7 mph → R9
	_, rule := parseLine(f, idx)
	if rule != 9 {
		t.Errorf("R9 (velocidad alta): got rule=%d, want 9", rule)
	}

	// velocidad < 0.5 mph: 0.01 mi en 20 min = 0.03 mph
	f2 := validFields()
	f2[4] = "0.001"
	_, rule = parseLine(f2, idx)
	if rule != 9 {
		t.Errorf("R9 (velocidad baja): got rule=%d, want 9", rule)
	}
}

// buildSyntheticCSV genera un CSV en memoria con filas válidas e inválidas
// controladas. Parámetros: validCount filas válidas, luego invalidCount filas
// por cada regla (R1..R9).
func buildSyntheticCSV(validCount, invalidPerRule int) string {
	var b strings.Builder
	b.WriteString("VendorID,tpep_pickup_datetime,tpep_dropoff_datetime,passenger_count,trip_distance,pickup_longitude,pickup_latitude,RateCodeID,store_and_fwd_flag,dropoff_longitude,dropoff_latitude,payment_type,fare_amount,extra,mta_tax,tip_amount,tolls_amount,improvement_surcharge,total_amount\n")

	writeRow := func(fields []string) {
		b.WriteString(strings.Join(fields, ","))
		b.WriteByte('\n')
	}

	// Filas válidas
	for i := 0; i < validCount; i++ {
		writeRow(validFields())
	}

	// R1: campo extra
	for i := 0; i < invalidPerRule; i++ {
		f := append(validFields(), "extra")
		writeRow(f)
	}

	// R2: pickup fuera de enero 2015
	for i := 0; i < invalidPerRule; i++ {
		f := validFields()
		f[1] = "2015-03-01 10:00:00"
		f[2] = "2015-03-01 10:20:00"
		writeRow(f)
	}

	// R3: dropoff == pickup (deben pasar R2, fallan R3)
	for i := 0; i < invalidPerRule; i++ {
		f := validFields()
		f[2] = f[1]
		writeRow(f)
	}

	// R4: duración < 1 min (pasan R2, R3, fallan R4)
	for i := 0; i < invalidPerRule; i++ {
		f := validFields()
		f[2] = "2015-01-10 08:00:30"
		writeRow(f)
	}

	// R5: lat=0 (pasan R2-R4, fallan R5)
	for i := 0; i < invalidPerRule; i++ {
		f := validFields()
		f[6] = "0.0"
		f[5] = "0.0"
		writeRow(f)
	}

	// R6: distancia=0 (pasan R2-R5, fallan R6)
	for i := 0; i < invalidPerRule; i++ {
		f := validFields()
		f[4] = "0.0"
		writeRow(f)
	}

	// R7: fare negativo (pasan R2-R6, fallan R7)
	for i := 0; i < invalidPerRule; i++ {
		f := validFields()
		f[12] = "-1.00"
		writeRow(f)
	}

	// R8: 0 pasajeros (pasan R2-R7, fallan R8)
	for i := 0; i < invalidPerRule; i++ {
		f := validFields()
		f[3] = "0"
		writeRow(f)
	}

	// R9: velocidad excesiva (pasan R2-R8, fallan R9)
	// 99.9 mi en 20 min = ~300 mph
	for i := 0; i < invalidPerRule; i++ {
		f := validFields()
		f[4] = "99.9"
		writeRow(f)
	}

	return b.String()
}

// TestLoadPipeline verifica el pipeline completo con un CSV sintético en memoria.
// Ejercita todas las goroutines concurrentes bajo el race detector.
func TestLoadPipeline(t *testing.T) {
	const validCount = 900
	const invalidPerRule = 10

	csv := buildSyntheticCSV(validCount, invalidPerRule)

	opts := LoadOptions{
		Reader:    strings.NewReader(csv),
		Workers:   4, // forzar paralelismo para ejercitar el race detector
		BatchSize: 100,
	}

	result, err := Load(opts)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}

	if got := len(result.Trips); got != validCount {
		t.Errorf("Trips: got %d, want %d", got, validCount)
	}

	totalExpected := int64(validCount + 9*invalidPerRule)
	if result.TotalRead != totalExpected {
		t.Errorf("TotalRead: got %d, want %d", result.TotalRead, totalExpected)
	}

	for rule := 1; rule <= 9; rule++ {
		got := result.Discards[rule]
		if got != int64(invalidPerRule) {
			t.Errorf("Discards[R%d]: got %d, want %d", rule, got, invalidPerRule)
		}
	}

	if result.Workers != 4 {
		t.Errorf("Workers: got %d, want 4", result.Workers)
	}
}

// TestBuildIndex verifica que buildIndex mapee correctamente por nombre.
func TestBuildIndex(t *testing.T) {
	// header con nombres en mayúsculas y espacios (como podría venir del archivo)
	header := []string{
		"VendorID", " tpep_pickup_datetime ", "tpep_dropoff_datetime",
		"passenger_count", "trip_distance", "pickup_longitude",
		"pickup_latitude", "RateCodeID", "store_and_fwd_flag",
		"dropoff_longitude", "dropoff_latitude", "payment_type",
		"fare_amount", "extra", "mta_tax", "tip_amount",
		"tolls_amount", "improvement_surcharge", "total_amount",
	}
	idx, err := buildIndex(header)
	if err != nil {
		t.Fatalf("buildIndex: %v", err)
	}
	if idx.pickup != 1 {
		t.Errorf("pickup idx: got %d, want 1", idx.pickup)
	}
	if idx.lat != 6 {
		t.Errorf("lat idx: got %d, want 6", idx.lat)
	}
}

// TestLoadPipelineSingleWorker verifica que el pipeline funcione con un solo worker.
func TestLoadPipelineSingleWorker(t *testing.T) {
	csv := fmt.Sprintf("VendorID,tpep_pickup_datetime,tpep_dropoff_datetime,passenger_count,trip_distance,pickup_longitude,pickup_latitude,RateCodeID,store_and_fwd_flag,dropoff_longitude,dropoff_latitude,payment_type,fare_amount,extra,mta_tax,tip_amount,tolls_amount,improvement_surcharge,total_amount\n%s",
		strings.Repeat(strings.Join(validFields(), ",")+"\n", 50))

	result, err := Load(LoadOptions{
		Reader:  strings.NewReader(csv),
		Workers: 1,
	})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if len(result.Trips) != 50 {
		t.Errorf("got %d trips, want 50", len(result.Trips))
	}
}
