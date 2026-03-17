#!/bin/bash
# Deploy Suyven — builds and starts Docker containers.
#
# Usage:
#   ./scripts/deploy.sh              # CPU mode
#   ./scripts/deploy.sh --gpu        # GPU mode (requires nvidia-docker)
#   ./scripts/deploy.sh --down       # Stop all containers
#   ./scripts/deploy.sh --rebuild    # Force rebuild + restart

set -euo pipefail

cd "$(dirname "$0")/.."

GPU=false
DOWN=false
REBUILD=""

for arg in "$@"; do
    case $arg in
        --gpu) GPU=true ;;
        --down) DOWN=true ;;
        --rebuild) REBUILD="--build" ;;
    esac
done

if [ "$DOWN" = true ]; then
    echo "Stopping containers..."
    docker compose down
    exit 0
fi

# Check .env exists
if [ ! -f .env ]; then
    echo "ERROR: .env file not found. Copy .env.example to .env and configure it."
    exit 1
fi

if [ "$GPU" = true ]; then
    echo "Starting Suyven (GPU mode)..."
    docker compose -f docker-compose.gpu.yml up -d $REBUILD
else
    echo "Starting Suyven (CPU mode)..."
    docker compose up -d $REBUILD
fi

echo ""
echo "Suyven is starting..."
echo "  API:      http://localhost:8000/api/health"
echo "  Frontend: http://localhost:8000"
echo ""
echo "Logs: docker compose logs -f"
