# Pulls the Ollama models used by this project.
# Run once after installing Ollama.

Write-Host "Pulling chat model: llama3.1:8b (~4.9 GB)..." -ForegroundColor Cyan
ollama pull llama3.1:8b

Write-Host "Pulling embedding model: nomic-embed-text (~274 MB)..." -ForegroundColor Cyan
ollama pull nomic-embed-text

Write-Host "Done. Installed models:" -ForegroundColor Green
ollama list
