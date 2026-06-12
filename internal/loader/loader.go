package loader

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"os"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Trip contiene los campos de un viaje ya validado y parseado.
// El struct es deliberadamente compacto (~80 bytes) para que 12M instancias
// quepan en ~1 GB de RAM.
type Trip struct {
	PickupTime     time.Time
	DurationMin    float64
	TripDistance   float64
	PickupLat      float64
	PickupLon      float64
	RateCodeID     int
	PassengerCount int
	HourOfDay      int
	DayOfWeek      int
}

// discardCounters agrupa los contadores atómicos de descarte por regla.
// Usamos atomic.Int64 en lugar de sync.Mutex porque los workers solo escriben
// y el collector solo lee después de que todos los workers terminaron —
// un mutex sería un punto de contención innecesario en el camino crítico.
type discardCounters struct {
	// índice 1-9 = reglas R1-R9; índice 0 no se usa
	counts [10]atomic.Int64
}

func (d *discardCounters) inc(rule int) {
	if rule >= 1 && rule <= 9 {
		d.counts[rule].Add(1)
	}
}

func (d *discardCounters) snapshot() [10]int64 {
	var out [10]int64
	for i := 1; i <= 9; i++ {
		out[i] = d.counts[i].Load()
	}
	return out
}

// LoadOptions configura el comportamiento del cargador.
type LoadOptions struct {
	FilePath  string    // ruta al CSV; ignorado si Reader != nil
	Reader    io.Reader // opcional: inyectar un reader (útil en tests)
	Workers   int       // número de goroutines worker; 0 = runtime.NumCPU()
	BatchSize int       // líneas por lote; 0 = 10_000
	Limit     int64     // máximo de filas a leer (0 = sin límite)
}

// LoadResult contiene el resultado de la carga.
type LoadResult struct {
	Trips     []Trip
	TotalRead int64
	Discards  [10]int64 // índices 1-9 = R1..R9
	Duration  time.Duration
	Workers   int
}

const defaultBatchSize = 10_000

// Load ejecuta el pipeline productor–consumidor para cargar el CSV.
//
// Arquitectura:
//   Reader goroutine  ──batchCh──▶  W Workers  ──outCh──▶  Collector goroutine
//
// cap(batchCh) = 2×W: doble buffer clásico; el reader se adelanta 2 lotes
// sin bloquearse, evitando que los workers queden ociosos por latencia de IO.
// cap(outCh)   = W×1024: margen para que los workers puedan emitir trips
// sin bloquearse mientras el collector procesa el lote anterior.
func Load(opts LoadOptions) (LoadResult, error) {
	start := time.Now()

	workers := opts.Workers
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	batchSize := opts.BatchSize
	if batchSize <= 0 {
		batchSize = defaultBatchSize
	}

	var r io.Reader
	if opts.Reader != nil {
		r = opts.Reader
	} else {
		f, err := os.Open(opts.FilePath)
		if err != nil {
			return LoadResult{}, fmt.Errorf("abrir archivo: %w", err)
		}
		defer f.Close()
		r = f
	}

	scanner := bufio.NewScanner(r)
	// Buffer de 1MB para manejar líneas anchas (19 campos × ~20 chars cada uno)
	scanner.Buffer(make([]byte, 1<<20), 1<<20)

	// Leer y parsear el header
	if !scanner.Scan() {
		return LoadResult{}, fmt.Errorf("CSV vacío o sin header")
	}
	header := strings.Split(scanner.Text(), ",")
	idx, err := buildIndex(header)
	if err != nil {
		return LoadResult{}, err
	}

	// Canal de lotes: cap = 2×workers (doble buffer entre reader y workers)
	batchCh := make(chan []string, 2*workers)
	// Canal de trips válidos: cap = workers×1024 (headroom para el collector)
	outCh := make(chan Trip, workers*1024)

	var counters discardCounters
	var totalRead atomic.Int64

	// --- Workers ---
	// Cada worker consume lotes del batchCh, parsea cada línea y emite
	// los trips válidos a outCh. Los contadores de descarte son atómicos
	// y compartidos por puntero — sin mutex, sin contención.
	var wg sync.WaitGroup
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for batch := range batchCh {
				for _, line := range batch {
					trip, rule := parseLine(strings.Split(line, ","), idx)
					if rule == 0 {
						outCh <- trip
					} else {
						counters.inc(rule)
					}
				}
			}
		}()
	}

	// Goroutine que cierra outCh cuando TODOS los workers terminan.
	// Es la ÚNICA goroutine responsable de cerrar outCh, evitando el race
	// de cierre múltiple que ocurriría si cada worker intentara cerrarlo.
	go func() {
		wg.Wait()
		close(outCh)
	}()

	// --- Reader (goroutine principal actúa como productor) ---
	// Lee líneas, las agrupa en lotes y los envía a batchCh.
	// Cada lote es un slice recién asignado para evitar aliasing.
	go func() {
		defer close(batchCh)
		batch := make([]string, 0, batchSize)
		for scanner.Scan() {
			if opts.Limit > 0 && totalRead.Load() >= opts.Limit {
				break
			}
			totalRead.Add(1)
			batch = append(batch, scanner.Text())
			if len(batch) == batchSize {
				batchCh <- batch
				// Nueva asignación para evitar que el worker lea un slice
				// cuyo backing array el reader reutilizaría
				batch = make([]string, 0, batchSize)
			}
		}
		if len(batch) > 0 {
			batchCh <- batch
		}
		if err := scanner.Err(); err != nil {
			log.Printf("error leyendo CSV: %v", err)
		}
	}()

	// --- Collector ---
	// Única goroutine que escribe en el slice de resultados.
	// Preasignamos capacidad estimada para evitar re-allocations costosas.
	trips := make([]Trip, 0, 12_000_000)
	for trip := range outCh {
		trips = append(trips, trip)
	}

	result := LoadResult{
		Trips:     trips,
		TotalRead: totalRead.Load(),
		Discards:  counters.snapshot(),
		Duration:  time.Since(start),
		Workers:   workers,
	}
	return result, nil
}

// PrintReport imprime el resumen de carga al stdout.
func PrintReport(r LoadResult) {
	valid := int64(len(r.Trips))
	total := r.TotalRead
	var retention float64
	if total > 0 {
		retention = float64(valid) / float64(total) * 100
	}
	throughput := float64(total) / r.Duration.Seconds()

	fmt.Printf("\n=== Reporte de Carga ===\n")
	fmt.Printf("Archivo leído en:   %v\n", r.Duration.Round(time.Millisecond))
	fmt.Printf("Workers usados:     %d\n", r.Workers)
	fmt.Printf("Total leídas:       %d\n", total)
	fmt.Printf("Válidas:            %d (%.1f%% retención)\n", valid, retention)
	fmt.Printf("Throughput:         %.0f filas/seg\n", throughput)
	fmt.Printf("\nDescartes por regla:\n")
	for i := 1; i <= 9; i++ {
		fmt.Printf("  R%d: %d\n", i, r.Discards[i])
	}
	fmt.Println()
}
