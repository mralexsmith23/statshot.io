#!/bin/sh
echo "=== StatShot starting on port ${PORT:-8501} ==="
exec streamlit run app.py \
    --server.port="${PORT:-8501}" \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false
