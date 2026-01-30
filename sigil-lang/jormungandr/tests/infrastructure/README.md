# Sigil Protocol Test Infrastructure

This directory contains Docker infrastructure for testing Sigil's protocol implementations against real message brokers.

## Services

| Service | Port | Purpose |
|---------|------|---------|
| Kafka | 9092 | Apache Kafka broker (KRaft mode) |
| RabbitMQ | 5672 | AMQP broker |
| RabbitMQ UI | 15672 | Management interface (sigil/sigil) |
| Kafka UI | 8080 | Optional debug UI (use `--profile debug`) |

## Usage

```bash
# Start infrastructure
./infra.sh up

# Stop infrastructure
./infra.sh down

# Check status
./infra.sh status

# View logs
./infra.sh logs

# Start with debug UIs
./infra.sh up --debug
```

## Protocol Implementations

The Kafka and AMQP implementations in Sigil are **pure Sigil** using TCP sockets. They implement the actual wire protocols:

- **Kafka**: Binary protocol over TCP (API versions, produce, fetch, etc.)
- **AMQP 0-9-1**: Binary protocol over TCP (connection, channel, basic, etc.)

## Test Requirements

Tests that require this infrastructure will:
1. Check if services are available
2. Skip gracefully with a clear message if infrastructure is down
3. Run real protocol operations when infrastructure is up

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Sigil Test Suite                     │
├─────────────────────────────────────────────────────────┤
│  P0_005_kafka_producer.sg    P0_007_amqp.sg            │
│         │                           │                   │
│         ▼                           ▼                   │
│  ┌─────────────────┐    ┌─────────────────┐            │
│  │ protocol::kafka │    │ protocol::amqp  │            │
│  │ (Pure Sigil)    │    │ (Pure Sigil)    │            │
│  └────────┬────────┘    └────────┬────────┘            │
│           │                      │                      │
│           ▼                      ▼                      │
│  ┌─────────────────────────────────────────┐           │
│  │     TcpStream (Sigil stdlib)            │           │
│  └─────────────────────────────────────────┘           │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   Docker Network                         │
├─────────────────────────────────────────────────────────┤
│   ┌─────────────┐              ┌─────────────┐          │
│   │    Kafka    │              │  RabbitMQ   │          │
│   │  :9092      │              │  :5672      │          │
│   └─────────────┘              └─────────────┘          │
└─────────────────────────────────────────────────────────┘
```

## Troubleshooting

**Kafka not starting?**
- Ensure Docker has enough memory (Kafka needs ~1GB)
- Check logs: `docker logs sigil-kafka`

**RabbitMQ connection refused?**
- Wait for healthcheck to pass
- Check logs: `docker logs sigil-rabbitmq`

**Tests timing out?**
- Verify infrastructure is running: `./infra.sh status`
- Check network connectivity: `nc -zv localhost 9092`
