#!/bin/bash
# Start Flask backend in background
FLASK_PORT=8000 python3.11 artifacts/ecg-app/app.py &
FLASK_PID=$!
echo "Flask backend started (PID: $FLASK_PID) on port 8000"

# Start Streamlit frontend
streamlit run artifacts/ecg-app/frontend.py --server.port 5000 --server.address 0.0.0.0 --server.headless true
