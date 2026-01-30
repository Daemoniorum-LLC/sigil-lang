#!/bin/bash
# Sigil Protocol Test Infrastructure Manager
#
# Usage:
#   ./infra.sh up [--debug]    Start infrastructure
#   ./infra.sh down            Stop infrastructure
#   ./infra.sh status          Check service health
#   ./infra.sh logs [service]  View logs
#   ./infra.sh wait            Wait for services to be ready

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
}

cmd_up() {
    check_docker
    log_info "Starting Sigil protocol test infrastructure..."

    local profile_arg=""
    if [[ "$1" == "--debug" ]]; then
        profile_arg="--profile debug"
        log_info "Debug mode: Kafka UI will be available at http://localhost:8080"
    fi

    docker compose $profile_arg up -d

    log_info "Waiting for services to be healthy..."
    cmd_wait

    log_success "Infrastructure ready!"
    echo ""
    echo "Services:"
    echo "  Kafka:       localhost:9092"
    echo "  RabbitMQ:    localhost:5672 (AMQP)"
    echo "  RabbitMQ UI: http://localhost:15672 (sigil/sigil)"
    if [[ "$1" == "--debug" ]]; then
        echo "  Kafka UI:    http://localhost:8080"
    fi
}

cmd_down() {
    check_docker
    log_info "Stopping infrastructure..."
    docker compose --profile debug down
    log_success "Infrastructure stopped"
}

cmd_status() {
    check_docker
    echo "Service Status:"
    echo "==============="

    # Kafka
    if docker ps --filter "name=sigil-kafka" --filter "status=running" -q | grep -q .; then
        if nc -zv localhost 9092 2>/dev/null; then
            log_success "Kafka:    Running (localhost:9092)"
        else
            log_warn "Kafka:    Container up, port not responding"
        fi
    else
        log_error "Kafka:    Not running"
    fi

    # RabbitMQ
    if docker ps --filter "name=sigil-rabbitmq" --filter "status=running" -q | grep -q .; then
        if nc -zv localhost 5672 2>/dev/null; then
            log_success "RabbitMQ: Running (localhost:5672)"
        else
            log_warn "RabbitMQ: Container up, port not responding"
        fi
    else
        log_error "RabbitMQ: Not running"
    fi
}

cmd_logs() {
    check_docker
    local service="${1:-}"
    if [[ -n "$service" ]]; then
        docker compose logs -f "$service"
    else
        docker compose logs -f
    fi
}

cmd_wait() {
    check_docker
    local max_attempts=30
    local attempt=0

    # Wait for Kafka
    log_info "Waiting for Kafka..."
    while ! nc -zv localhost 9092 2>/dev/null; do
        attempt=$((attempt + 1))
        if [[ $attempt -ge $max_attempts ]]; then
            log_error "Kafka failed to start within timeout"
            exit 1
        fi
        sleep 1
    done
    log_success "Kafka is ready"

    # Wait for RabbitMQ
    attempt=0
    log_info "Waiting for RabbitMQ..."
    while ! nc -zv localhost 5672 2>/dev/null; do
        attempt=$((attempt + 1))
        if [[ $attempt -ge $max_attempts ]]; then
            log_error "RabbitMQ failed to start within timeout"
            exit 1
        fi
        sleep 1
    done
    log_success "RabbitMQ is ready"
}

cmd_help() {
    echo "Sigil Protocol Test Infrastructure"
    echo ""
    echo "Usage: $0 <command> [options]"
    echo ""
    echo "Commands:"
    echo "  up [--debug]    Start infrastructure (--debug enables Kafka UI)"
    echo "  down            Stop infrastructure"
    echo "  status          Check service health"
    echo "  logs [service]  View logs (optionally for specific service)"
    echo "  wait            Wait for services to be ready"
    echo "  help            Show this help"
}

# Main
case "${1:-help}" in
    up)     cmd_up "$2" ;;
    down)   cmd_down ;;
    status) cmd_status ;;
    logs)   cmd_logs "$2" ;;
    wait)   cmd_wait ;;
    help)   cmd_help ;;
    *)      log_error "Unknown command: $1"; cmd_help; exit 1 ;;
esac
