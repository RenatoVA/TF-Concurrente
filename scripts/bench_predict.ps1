# bench_predict.ps1 — mide latencia de GET /predict (objetivo: p95 < 100 ms)

param(
    [string]$Host_   = "localhost",
    [int]   $Port    = 8080,
    [int]   $N       = 100
)

$BaseUrl = "http://${Host_}:${Port}/predict"

$Params = @(
    "trip_distance=2.5&hour=18&day_of_week=3&pickup_lat=40.75&pickup_lon=-73.98&rate_code=1&passenger_count=1",
    "trip_distance=1.0&hour=8&day_of_week=1&pickup_lat=40.70&pickup_lon=-74.00&rate_code=1&passenger_count=2",
    "trip_distance=5.3&hour=22&day_of_week=5&pickup_lat=40.76&pickup_lon=-73.97&rate_code=2&passenger_count=1",
    "trip_distance=0.8&hour=12&day_of_week=0&pickup_lat=40.72&pickup_lon=-73.99&rate_code=1&passenger_count=3"
)

Write-Host "[bench] Enviando $N requests a $BaseUrl..."

$latencies = @()

for ($i = 1; $i -le $N; $i++) {
    $p = $Params[$i % $Params.Count]
    $url = "${BaseUrl}?${p}"

    $ms = (Measure-Command {
        try {
            Invoke-WebRequest -Uri $url -UseBasicParsing -ErrorAction Stop | Out-Null
        } catch {
            # ignora errores de red individuales
        }
    }).TotalMilliseconds

    $latencies += [math]::Round($ms)

    if ($i % 10 -eq 0) {
        Write-Host "  $i/$N completados..."
    }
}

$sorted = $latencies | Sort-Object
$count  = $sorted.Count
$min    = $sorted[0]
$max    = $sorted[$count - 1]
$p50    = $sorted[[int]($count * 0.50)]
$p95    = $sorted[[int]($count * 0.95)]
$avg    = [math]::Round(($latencies | Measure-Object -Average).Average)

Write-Host ""
Write-Host "=== Benchmark /predict (N=$count) ==="
Write-Host ("  min:  {0} ms" -f $min)
Write-Host ("  avg:  {0} ms" -f $avg)
Write-Host ("  p50:  {0} ms" -f $p50)
Write-Host ("  p95:  {0} ms" -f $p95)
Write-Host ("  max:  {0} ms" -f $max)
Write-Host ""

if ($p95 -lt 100) {
    Write-Host "  [OK] p95 < 100 ms" -ForegroundColor Green
} else {
    Write-Host "  [FAIL] p95 >= 100 ms" -ForegroundColor Red
    exit 1
}
