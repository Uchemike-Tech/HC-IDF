# HC-IDF Full System Starter
# Starts: Grafana (Docker), Metrics API, Streamlit Dashboard

Write-Host "=== HC-IDF System Starter ===" -ForegroundColor Cyan

# 1. Start Grafana via Docker
Write-Host "[1/3] Starting Grafana (Docker)..." -ForegroundColor Yellow
Set-Location -LiteralPath "$PSScriptRoot\..\grafana"
docker compose up -d 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "  Grafana at http://localhost:3000 (admin:admin)" -ForegroundColor Green
} else {
    Write-Host "  Grafana failed to start. Is Docker running?" -ForegroundColor Red
}

# 2. Start Metrics API
Write-Host "[2/3] Starting Metrics API..." -ForegroundColor Yellow
$apiJob = Start-Job -ScriptBlock {
    Set-Location -LiteralPath "$using:PSScriptRoot\.."
    python grafana\metrics_server.py
}
Write-Host "  Metrics API at http://localhost:5050" -ForegroundColor Green

# 3. Start Streamlit Dashboard
Set-Location -LiteralPath "$PSScriptRoot\.."
Write-Host "[3/3] Starting Streamlit Dashboard..." -ForegroundColor Yellow
Write-Host "  Dashboard at http://localhost:8501" -ForegroundColor Green
Write-Host ""
Write-Host "=== Press Ctrl+C to stop all services ===" -ForegroundColor Cyan

python -m streamlit run dashboard.py --server.port 8501

# Cleanup on exit
docker compose down 2>$null
Stop-Job $apiJob 2>$null
Remove-Job $apiJob 2>$null
