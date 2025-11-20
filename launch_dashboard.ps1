# Launch Dashboard
# =================
# Quick script to start the Streamlit dashboard

Write-Host "=" * 80
Write-Host " " * 25 "LAUNCHING DASHBOARD"
Write-Host "=" * 80
Write-Host ""
Write-Host "Starting Streamlit server..."
Write-Host "The dashboard will open in your browser automatically."
Write-Host ""
Write-Host "Press Ctrl+C to stop the server when done."
Write-Host ""
Write-Host "=" * 80

streamlit run dashboard.py
