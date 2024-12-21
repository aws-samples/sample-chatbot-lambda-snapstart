#!/bin/bash
export PATH=$PATH:$LAMBDA_TASK_ROOT/bin
export PYTHONPATH=/opt/python:$LAMBDA_RUNTIME_DIR
LOG_LEVEL=${AWS_LAMBDA_LOG_LEVEL:-info}
PORT=${PORT:-8080}
exec python -m uvicorn --workers=1 --log-level=$LOG_LEVEL --port=$PORT main:app