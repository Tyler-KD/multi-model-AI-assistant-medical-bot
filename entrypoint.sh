#!/bin/bash

# First argument determines which service to run
if [ "$1" = "api" ]; then
    echo "Starting FastAPI backend..."
    exec uvicorn api.fastapi_backend:app --host 0.0.0.0 --port 8000
elif [ "$1" = "app" ]; then
    echo "Starting main application..."
    exec python app.py
elif [ "$1" = "all" ]; then
    echo "Starting both services..."
    uvicorn api.fastapi_backend:app --host 0.0.0.0 --port 8000 &
    python app.py
    wait
elif [ "$1" = "ingest" ]; then
    echo "Ingesting data..."
    if [ "$2" = "--file" ]; then
        exec python ingest_rag_data.py --file "$3"
    elif [ "$2" = "--dir" ]; then
        exec python ingest_rag_data.py --dir "$3"
    else
        echo "Please provide valid arguments: --file or --dir"
        exit 1
    fi
else
    echo "Please specify which service to run: api, app, all, or ingest"
    exit 1
fi