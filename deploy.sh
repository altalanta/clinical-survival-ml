#!/bin/bash
# Deployment script for clinical-survival-ml

set -e

echo "🚀 Clinical Survival ML Deployment Script"
echo "========================================"

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if docker-compose exists (fallback to docker compose)
get_docker_compose_cmd() {
    if command_exists "docker-compose"; then
        echo "docker-compose"
    elif command_exists "docker" && docker compose version >/dev/null 2>&1; then
        echo "docker compose"
    else
        echo "❌ Neither docker-compose nor 'docker compose' is available"
        exit 1
    fi
}

DOCKER_COMPOSE_CMD=$(get_docker_compose_cmd)

# Parse arguments
ACTION=${1:-help}
MODEL_DIR=${2:-results/artifacts/models}

case $ACTION in
    "build")
        echo "🏗️  Building Docker image..."
        docker build -t clinical-survival-ml .
        echo "✅ Docker image built successfully"
        ;;

    "serve")
        echo "🌐 Starting API server..."
        echo "📁 Models directory: $MODEL_DIR"

        # Check if models exist
        if [ ! -d "$MODEL_DIR" ]; then
            echo "❌ Models directory not found: $MODEL_DIR"
            echo "💡 Run training first: clinical-ml run --config configs/params.yaml --grid configs/model_grid.yaml"
            exit 1
        fi

        # Check if models are present
        if [ -z "$(ls -A "$MODEL_DIR" 2>/dev/null)" ]; then
            echo "❌ No models found in $MODEL_DIR"
            echo "💡 Run training first to generate models"
            exit 1
        fi

        echo "🚀 Starting server with $DOCKER_COMPOSE_CMD..."
        $DOCKER_COMPOSE_CMD up clinical-survival-api
        ;;

    "train")
        echo "🎯 Running training pipeline..."
        echo "Using docker-compose training profile..."
        $DOCKER_COMPOSE_CMD --profile training up clinical-survival-training
        ;;

    "stop")
        echo "🛑 Stopping all services..."
        $DOCKER_COMPOSE_CMD down
        ;;

    "logs")
        echo "📋 Showing logs..."
        $DOCKER_COMPOSE_CMD logs -f clinical-survival-api
        ;;

    "status")
        echo "📊 Service status..."
        $DOCKER_COMPOSE_CMD ps
        ;;

    "clean")
        echo "🧹 Cleaning up..."
        $DOCKER_COMPOSE_CMD down -v --remove-orphans
        docker system prune -f
        ;;

    "help"|*)
        echo "Usage: $0 <action> [model_dir]"
        echo ""
        echo "Actions:"
        echo "  build     Build the Docker image"
        echo "  serve     Start the API server (requires trained models)"
        echo "  train     Run the training pipeline"
        echo "  stop      Stop all services"
        echo "  logs      Show API server logs"
        echo "  status    Show service status"
        echo "  clean     Clean up containers and images"
        echo "  help      Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0 build"
        echo "  $0 train"
        echo "  $0 serve"
        echo ""
        echo "API will be available at: http://localhost:8000"
        echo "Interactive docs at: http://localhost:8000/docs"
        ;;
esac


