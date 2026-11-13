#!/bin/sh
# Ephemeral env pinned to experiments/E05-probes/requirements-lock.txt versions (no torch needed).
cd /Users/leo/precrime-agents && exec uv run --no-project --with scikit-learn==1.8.0 --with numpy==2.4.6 --with scipy==1.17.1 --with threadpoolctl==3.7.0 --with python-dotenv python "$@"
