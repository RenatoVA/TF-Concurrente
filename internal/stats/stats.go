// Package stats implementa agregados estadísticos y exportación de CSVs
// para el análisis exploratorio del dataset de taxis.
package stats

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"sort"

	"github.com/remii/tf-concurrente/internal/loader"
)

// StatsResult agrupa todos los agregados calculados sobre el dataset.
type StatsResult struct {
	TripsByHour    [24]int64
	TripsByWeekday [7]int64
	DurationHist   []BinCount  // bins de 2 min, 0-180
	DistanceHist   []BinCount  // bins de 1 mi, 0-100
	SpeedByHour    [24]AvgAccum
	TopCells       []CellCount // grilla 0.01° lat/lon
}

// BinCount representa un bin de histograma.
type BinCount struct {
	BinStart float64
	Count    int64
}

// AvgAccum acumula valores para calcular el promedio.
type AvgAccum struct {
	Sum   float64
	Count int64
}

func (a AvgAccum) Avg() float64 {
	if a.Count == 0 {
		return 0
	}
	return a.Sum / float64(a.Count)
}

// CellCount representa una celda de la grilla de pickup con su conteo.
type CellCount struct {
	LatCell int // lat * 100 (truncado)
	LonCell int // lon * 100 (truncado)
	Count   int64
}

// Compute calcula todos los agregados en un único pase sobre los trips.
func Compute(trips []loader.Trip) StatsResult {
	const (
		durBinSize  = 2.0 // minutos
		distBinSize = 1.0 // millas
		maxDur      = 180.0
		maxDist     = 100.0
	)

	nDurBins := int(maxDur/durBinSize) + 1
	nDistBins := int(maxDist/distBinSize) + 1

	var r StatsResult
	r.DurationHist = make([]BinCount, nDurBins)
	r.DistanceHist = make([]BinCount, nDistBins)
	for i := range r.DurationHist {
		r.DurationHist[i].BinStart = float64(i) * durBinSize
	}
	for i := range r.DistanceHist {
		r.DistanceHist[i].BinStart = float64(i) * distBinSize
	}

	// Mapa de celdas de pickup (lat*100, lon*100) → conteo
	cellMap := make(map[[2]int]int64, 10000)

	for _, t := range trips {
		r.TripsByHour[t.HourOfDay]++
		r.TripsByWeekday[t.DayOfWeek]++

		durBin := int(t.DurationMin / durBinSize)
		if durBin >= 0 && durBin < nDurBins {
			r.DurationHist[durBin].Count++
		}

		distBin := int(t.TripDistance / distBinSize)
		if distBin >= 0 && distBin < nDistBins {
			r.DistanceHist[distBin].Count++
		}

		if t.DurationMin > 0 {
			speedMph := t.TripDistance / (t.DurationMin / 60.0)
			r.SpeedByHour[t.HourOfDay].Sum += speedMph
			r.SpeedByHour[t.HourOfDay].Count++
		}

		cell := [2]int{int(t.PickupLat * 100), int(t.PickupLon * 100)}
		cellMap[cell]++
	}

	// Convertir mapa de celdas a slice y ordenar por conteo descendente
	cells := make([]CellCount, 0, len(cellMap))
	for k, v := range cellMap {
		cells = append(cells, CellCount{LatCell: k[0], LonCell: k[1], Count: v})
	}
	sort.Slice(cells, func(i, j int) bool {
		return cells[i].Count > cells[j].Count
	})
	const topN = 50
	if len(cells) > topN {
		cells = cells[:topN]
	}
	r.TopCells = cells

	return r
}

// ExportCSVs escribe todos los CSVs de resumen en el directorio dado.
func ExportCSVs(r StatsResult, dir string) error {
	if err := writeHourCSV(filepath.Join(dir, "trips_por_hora.csv"), r); err != nil {
		return err
	}
	if err := writeWeekdayCSV(filepath.Join(dir, "trips_por_dia_semana.csv"), r); err != nil {
		return err
	}
	if err := writeHistCSV(filepath.Join(dir, "histograma_duracion.csv"), "bin_inicio_min", r.DurationHist); err != nil {
		return err
	}
	if err := writeHistCSV(filepath.Join(dir, "histograma_distancia.csv"), "bin_inicio_mi", r.DistanceHist); err != nil {
		return err
	}
	if err := writeSpeedCSV(filepath.Join(dir, "velocidad_media_por_hora.csv"), r); err != nil {
		return err
	}
	if err := writeCellsCSV(filepath.Join(dir, "top_celdas_pickup.csv"), r); err != nil {
		return err
	}
	return nil
}

func writeCSV(path string, fn func(*bufio.Writer) error) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("crear %s: %w", path, err)
	}
	defer f.Close()
	w := bufio.NewWriter(f)
	if err := fn(w); err != nil {
		return err
	}
	return w.Flush()
}

func writeHourCSV(path string, r StatsResult) error {
	return writeCSV(path, func(w *bufio.Writer) error {
		fmt.Fprintln(w, "hora,viajes")
		for h, n := range r.TripsByHour {
			fmt.Fprintf(w, "%d,%d\n", h, n)
		}
		return nil
	})
}

var weekdayNames = [7]string{"Domingo", "Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado"}

func writeWeekdayCSV(path string, r StatsResult) error {
	return writeCSV(path, func(w *bufio.Writer) error {
		fmt.Fprintln(w, "dia,nombre,viajes")
		for d, n := range r.TripsByWeekday {
			fmt.Fprintf(w, "%d,%s,%d\n", d, weekdayNames[d], n)
		}
		return nil
	})
}

func writeHistCSV(path, colName string, bins []BinCount) error {
	return writeCSV(path, func(w *bufio.Writer) error {
		fmt.Fprintf(w, "%s,conteo\n", colName)
		for _, b := range bins {
			if b.Count > 0 {
				fmt.Fprintf(w, "%.1f,%d\n", b.BinStart, b.Count)
			}
		}
		return nil
	})
}

func writeSpeedCSV(path string, r StatsResult) error {
	return writeCSV(path, func(w *bufio.Writer) error {
		fmt.Fprintln(w, "hora,velocidad_media_mph")
		for h, acc := range r.SpeedByHour {
			fmt.Fprintf(w, "%d,%.4f\n", h, acc.Avg())
		}
		return nil
	})
}

func writeCellsCSV(path string, r StatsResult) error {
	return writeCSV(path, func(w *bufio.Writer) error {
		fmt.Fprintln(w, "lat_celda,lon_celda,lat_centro,lon_centro,conteo")
		for _, c := range r.TopCells {
			latCentro := float64(c.LatCell)/100.0 + 0.005
			lonCentro := float64(c.LonCell)/100.0 + 0.005
			fmt.Fprintf(w, "%d,%d,%.5f,%.5f,%d\n",
				c.LatCell, c.LonCell, latCentro, lonCentro, c.Count)
		}
		return nil
	})
}
