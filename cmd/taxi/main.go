// taxi es el CLI del trabajo final CC65 — cargador concurrente y entrenamiento paralelo.
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"runtime"
	"strings"
	"time"

	"tf-concurrente/internal/api"
	"tf-concurrente/internal/cluster"
	"tf-concurrente/internal/loader"
	"tf-concurrente/internal/model"
	"tf-concurrente/internal/stats"
	"tf-concurrente/internal/store"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	switch os.Args[1] {
	case "load":
		runLoad(os.Args[2:])
	case "train":
		runTrain(os.Args[2:])
	case "benchmark":
		runBenchmark(os.Args[2:])
	case "stats":
		runStats(os.Args[2:])
	case "node":
		runNode(os.Args[2:])
	case "api":
		runAPI(os.Args[2:])
	case "seed-insights":
		runSeedInsights(os.Args[2:])
	default:
		fmt.Fprintf(os.Stderr, "subcomando desconocido: %s\n\n", os.Args[1])
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println(`Uso: taxi <subcomando> [opciones]

Subcomandos (PC3 — locales):
  load           Carga y valida el CSV (Parte A)
  train          Carga + entrena modelo SGD paralelo + evalúa (Partes A+B)
  benchmark      Mide speedup con workers=1,2,4,8
  stats          Exporta CSVs de análisis exploratorio

Subcomandos (PC4 — distribuidos):
  node           Arranca un worker TCP que calcula gradientes
  api            Arranca el coordinador REST + parameter-server
  seed-insights  Carga los 6 CSVs de EDA en MongoDB

Ejemplos:
  taxi load      --file data/yellow_tripdata_2015-01.csv --workers 4
  taxi train     --file data/yellow_tripdata_2015-01.csv --workers 4 --epochs 10 --lr 0.01
  taxi benchmark --file data/yellow_tripdata_2015-01.csv --limit 500000
  taxi stats     --file data/yellow_tripdata_2015-01.csv
  taxi node      --listen :9100
  taxi api       --addr :8080 --mongo mongodb://localhost:27017 --redis localhost:6379 --nodes node1:9100,node2:9100
  taxi seed-insights --file data/yellow_tripdata_2015-01.csv --mongo mongodb://localhost:27017`)
}

// runLoad ejecuta solo la Parte A e imprime el reporte de limpieza.
func runLoad(args []string) {
	fs := flag.NewFlagSet("load", flag.ExitOnError)
	file := fs.String("file", "data/yellow_tripdata_2015-01.csv", "ruta al CSV")
	workers := fs.Int("workers", runtime.NumCPU(), "número de goroutines worker")
	limit := fs.Int64("limit", 0, "límite de filas (0=todas)")
	if err := fs.Parse(args); err != nil {
		os.Exit(1)
	}

	log.Printf("Iniciando carga: file=%s workers=%d", *file, *workers)
	result, err := loader.Load(loader.LoadOptions{
		FilePath: *file,
		Workers:  *workers,
		Limit:    *limit,
	})
	if err != nil {
		log.Fatalf("Error de carga: %v", err)
	}
	loader.PrintReport(result)
}

// runTrain ejecuta carga + entrenamiento + evaluación y guarda el modelo.
func runTrain(args []string) {
	fs := flag.NewFlagSet("train", flag.ExitOnError)
	file := fs.String("file", "data/yellow_tripdata_2015-01.csv", "ruta al CSV")
	workers := fs.Int("workers", runtime.NumCPU(), "número de goroutines")
	epochs := fs.Int("epochs", 10, "número de épocas SGD")
	lr := fs.Float64("lr", 0.01, "learning rate")
	batch := fs.Int("batch", 1024, "tamaño de mini-batch")
	seed := fs.Int64("seed", 42, "semilla para reproducibilidad")
	output := fs.String("output", "data/model.json", "ruta del modelo guardado")
	lossCSV := fs.String("loss-csv", "data/loss_curve.csv", "ruta del CSV con la curva de loss")
	limit := fs.Int64("limit", 0, "límite de filas (0=todas)")
	if err := fs.Parse(args); err != nil {
		os.Exit(1)
	}

	log.Printf("Cargando datos: %s", *file)
	loadResult, err := loader.Load(loader.LoadOptions{
		FilePath: *file,
		Workers:  *workers,
		Limit:    *limit,
	})
	if err != nil {
		log.Fatalf("Error de carga: %v", err)
	}
	loader.PrintReport(loadResult)

	log.Printf("Iniciando entrenamiento: epochs=%d lr=%.4f batch=%d workers=%d seed=%d",
		*epochs, *lr, *batch, *workers, *seed)

	m, err := model.Train(loadResult.Trips, model.TrainOptions{
		Workers:      *workers,
		Epochs:       *epochs,
		LearningRate: *lr,
		BatchSize:    *batch,
		Seed:         *seed,
	})
	if err != nil {
		log.Fatalf("Error de entrenamiento: %v", err)
	}

	if err := m.Save(*output); err != nil {
		log.Fatalf("Error guardando modelo: %v", err)
	}
	log.Printf("Modelo guardado en: %s", *output)

	if err := m.SaveLossCSV(*lossCSV); err != nil {
		log.Fatalf("Error guardando loss CSV: %v", err)
	}
	log.Printf("Curva de loss guardada en: %s", *lossCSV)
}

// runBenchmark mide tiempos de carga y entrenamiento con distintos números de workers.
func runBenchmark(args []string) {
	fs := flag.NewFlagSet("benchmark", flag.ExitOnError)
	file := fs.String("file", "data/yellow_tripdata_2015-01.csv", "ruta al CSV")
	limit := fs.Int64("limit", 0, "límite de filas (0=todas)")
	epochs := fs.Int("epochs", 3, "épocas para el benchmark de entrenamiento")
	if err := fs.Parse(args); err != nil {
		os.Exit(1)
	}

	workerCounts := []int{1, 2, 4, 8}

	type result struct {
		workers   int
		loadTime  time.Duration
		trainTime time.Duration
		total     time.Duration
	}
	results := make([]result, 0, len(workerCounts))

	for _, w := range workerCounts {
		log.Printf("Benchmark workers=%d...", w)

		t0 := time.Now()
		loadRes, err := loader.Load(loader.LoadOptions{
			FilePath: *file,
			Workers:  w,
			Limit:    *limit,
		})
		if err != nil {
			log.Fatalf("Error carga workers=%d: %v", w, err)
		}
		loadTime := time.Since(t0)

		t1 := time.Now()
		_, err = model.Train(loadRes.Trips, model.TrainOptions{
			Workers:  w,
			Epochs:   *epochs,
			BatchSize: 1024,
			LearningRate: 0.01,
			Seed:     42,
		})
		if err != nil {
			log.Fatalf("Error entrenamiento workers=%d: %v", w, err)
		}
		trainTime := time.Since(t1)

		results = append(results, result{
			workers:   w,
			loadTime:  loadTime,
			trainTime: trainTime,
			total:     loadTime + trainTime,
		})
	}

	baseTotal := results[0].total
	fmt.Printf("\n=== Benchmark de Speedup ===\n")
	fmt.Printf("Filas: ")
	if *limit > 0 {
		fmt.Printf("%d (limitadas)\n", *limit)
	} else {
		fmt.Println("todas")
	}
	fmt.Printf("Épocas: %d\n\n", *epochs)
	fmt.Printf("%-10s %-14s %-14s %-14s %-10s\n", "Workers", "Carga", "Entrenamiento", "Total", "Speedup")
	fmt.Printf("%-10s %-14s %-14s %-14s %-10s\n", "-------", "-----", "-------------", "-----", "-------")
	for _, r := range results {
		speedup := float64(baseTotal) / float64(r.total)
		fmt.Printf("%-10d %-14s %-14s %-14s %.2fx\n",
			r.workers,
			r.loadTime.Round(time.Millisecond),
			r.trainTime.Round(time.Millisecond),
			r.total.Round(time.Millisecond),
			speedup)
	}
	fmt.Println()
}

// runNode arranca un servidor TCP worker que calcula gradientes para el coordinador.
func runNode(args []string) {
	fs := flag.NewFlagSet("node", flag.ExitOnError)
	listen := fs.String("listen", envOr("NODE_LISTEN", ":9100"), "dirección TCP de escucha")
	if err := fs.Parse(args); err != nil {
		os.Exit(1)
	}
	log.Printf("Iniciando nodo ML en %s", *listen)
	if err := cluster.Serve(*listen); err != nil {
		log.Fatalf("Error en nodo: %v", err)
	}
}

// runAPI arranca el servidor REST coordinador del cluster.
func runAPI(args []string) {
	fs := flag.NewFlagSet("api", flag.ExitOnError)
	addr := fs.String("addr", envOr("API_ADDR", ":8080"), "dirección HTTP")
	mongoURI := fs.String("mongo", envOr("MONGO_URI", "mongodb://localhost:27017"), "URI de MongoDB")
	redisAddr := fs.String("redis", envOr("REDIS_ADDR", "localhost:6379"), "dirección de Redis")
	nodesFlag := fs.String("nodes", envOr("NODES", ""), "nodos separados por coma, ej: node1:9100,node2:9100")
	if err := fs.Parse(args); err != nil {
		os.Exit(1)
	}

	var nodes []string
	if *nodesFlag != "" {
		for _, n := range strings.Split(*nodesFlag, ",") {
			n = strings.TrimSpace(n)
			if n != "" {
				nodes = append(nodes, n)
			}
		}
	}
	if len(nodes) == 0 {
		log.Fatal("Se requiere --nodes o la variable NODES con al menos un nodo")
	}

	cfg := api.Config{
		Addr:      *addr,
		MongoURI:  *mongoURI,
		RedisAddr: *redisAddr,
		Nodes:     nodes,
	}
	log.Printf("Iniciando API en %s con %d nodos", *addr, len(nodes))
	if err := api.StartServer(cfg); err != nil {
		log.Fatalf("Error en API: %v", err)
	}
}

// runSeedInsights carga los 6 CSVs de EDA en MongoDB.
func runSeedInsights(args []string) {
	fs := flag.NewFlagSet("seed-insights", flag.ExitOnError)
	file := fs.String("file", "data/yellow_tripdata_2015-01.csv", "ruta al CSV")
	workers := fs.Int("workers", runtime.NumCPU(), "número de goroutines")
	mongoURI := fs.String("mongo", envOr("MONGO_URI", "mongodb://localhost:27017"), "URI de MongoDB")
	if err := fs.Parse(args); err != nil {
		os.Exit(1)
	}

	log.Printf("Cargando datos para EDA...")
	loadRes, err := loader.Load(loader.LoadOptions{FilePath: *file, Workers: *workers})
	if err != nil {
		log.Fatalf("Error de carga: %v", err)
	}
	log.Printf("Calculando estadísticas sobre %d viajes...", len(loadRes.Trips))
	statsResult := stats.Compute(loadRes.Trips)

	db, err := store.Connect(*mongoURI)
	if err != nil {
		log.Fatalf("Error conectando a MongoDB: %v", err)
	}
	defer db.Close()

	insights := map[string][][]interface{}{
		"trips_por_hora":           int64SliceToRows(statsResult.TripsByHour[:]),
		"trips_por_dia_semana":     int64SliceToRows(statsResult.TripsByWeekday[:]),
		"histograma_duracion":      binCountToRows(statsResult.DurationHist),
		"histograma_distancia":     binCountToRows(statsResult.DistanceHist),
		"velocidad_media_por_hora": avgAccumToRows(statsResult.SpeedByHour[:]),
		"top_celdas_pickup":        cellCountToRows(statsResult.TopCells),
	}

	for name, rows := range insights {
		if err := db.UpsertInsight(name, rows); err != nil {
			log.Printf("Error upsert %s: %v", name, err)
		} else {
			log.Printf("Insight '%s' cargado (%d filas)", name, len(rows))
		}
	}
	fmt.Println("Insights cargados en MongoDB.")
}

// envOr returns the env variable value or a default.
func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func int64SliceToRows(counts []int64) [][]interface{} {
	rows := make([][]interface{}, len(counts))
	for i, v := range counts {
		rows[i] = []interface{}{i, v}
	}
	return rows
}

func binCountToRows(hist []stats.BinCount) [][]interface{} {
	rows := make([][]interface{}, len(hist))
	for i, b := range hist {
		rows[i] = []interface{}{b.BinStart, b.Count}
	}
	return rows
}

func avgAccumToRows(accums []stats.AvgAccum) [][]interface{} {
	rows := make([][]interface{}, len(accums))
	for i, a := range accums {
		rows[i] = []interface{}{i, a.Avg()}
	}
	return rows
}

func cellCountToRows(cells []stats.CellCount) [][]interface{} {
	rows := make([][]interface{}, len(cells))
	for i, c := range cells {
		rows[i] = []interface{}{c.LatCell, c.LonCell, c.Count}
	}
	return rows
}

// runStats carga los datos y exporta CSVs de análisis exploratorio.
func runStats(args []string) {
	fs := flag.NewFlagSet("stats", flag.ExitOnError)
	file := fs.String("file", "data/yellow_tripdata_2015-01.csv", "ruta al CSV")
	workers := fs.Int("workers", runtime.NumCPU(), "número de goroutines worker")
	outDir := fs.String("outdir", "data", "directorio de salida para los CSVs")
	if err := fs.Parse(args); err != nil {
		os.Exit(1)
	}

	log.Printf("Cargando datos para análisis...")
	loadRes, err := loader.Load(loader.LoadOptions{
		FilePath: *file,
		Workers:  *workers,
	})
	if err != nil {
		log.Fatalf("Error de carga: %v", err)
	}
	loader.PrintReport(loadRes)

	log.Printf("Calculando estadísticas...")
	statsResult := stats.Compute(loadRes.Trips)

	log.Printf("Exportando CSVs a %s/...", *outDir)
	if err := stats.ExportCSVs(statsResult, *outDir); err != nil {
		log.Fatalf("Error exportando CSVs: %v", err)
	}

	fmt.Printf("\n=== Estadísticas Exportadas ===\n")
	fmt.Printf("Directorio: %s/\n", *outDir)
	fmt.Println("  trips_por_hora.csv")
	fmt.Println("  trips_por_dia_semana.csv")
	fmt.Println("  histograma_duracion.csv")
	fmt.Println("  histograma_distancia.csv")
	fmt.Println("  velocidad_media_por_hora.csv")
	fmt.Printf("  top_celdas_pickup.csv (%d celdas)\n", len(statsResult.TopCells))
}
