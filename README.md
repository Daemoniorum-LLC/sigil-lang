# Persona Framework

Autonomous agent platform built with Kotlin/Spring Boot, React, and IntelliJ tooling. The repo hosts the core runtime, REST API, admin UI, IDE plugin, and a collection of verticalized services (art generation and CNC ERP microservices).

## What’s in the repo
- **Leviathan** – primary Spring Boot backend with agent orchestration, REST API, tool registry, and integrations (Ollama/AWS/Anthropic/OpenAI). See `leviathan/README.md`.
- **Bael** – React + TypeScript admin UI that talks to Leviathan. See `bael/README.md`.
- **Hydra** – Spring Boot gateway for legacy integrations. See `hydra/README.md`.
- **Paimon** – IntelliJ IDEA plugin for in-editor agent workflows. See `paimon/README.md`.
- **Art Service** – Spring Boot app for digital painting workflows; also embedded in Leviathan. See `art-service/README.md`.
- **CNC ERP services** – standalone Spring Boot services for CNC domains (customer, quote, job, schedule, shop-floor, inventory, quality, accounting, analytics, portal, integration-hub, machine-monitor). Each has its own README under its module directory.
- **Nexus Standalone** – slim Spring Boot packaging that reuses Leviathan while trimming optional modules. See `nexus-standalone/README.md`.
- **Libraries** – shared modules: `persona-core`, `persona-api`, provider adapters (`persona-aws`, `persona-anthropic`, `persona-openai`, `persona-ollama`), `persona-agent`, `persona-rest`, `persona-autoconfigure`, `persona-mcp`, `persona-sandbox`, `mammon`, `cnc-common`, and `grimoire`.

## Prerequisites
- Java 21 (Gradle wrapper is provided)
- Node.js 18+ (for Bael)
- PostgreSQL 13+ (used by Leviathan and the Spring Boot services)
- Optional: Ollama running at `http://localhost:11434` for local ML, Kafka for CNC event streaming, Redis/Elasticsearch if you enable those integrations

## Quick start (Leviathan + Bael)
1. **Configure environment** (example):
   ```bash
   export SPRING_DATASOURCE_URL=jdbc:postgresql://localhost:5432/persona
   export SPRING_DATASOURCE_USERNAME=postgres
   export SPRING_DATASOURCE_PASSWORD=postgres
   export PERSONA_AI_OLLAMA_ENABLED=true
   export PERSONA_AI_OLLAMA_BASE_URL=http://localhost:11434
   # Optional workspace roots
   export WORKSPACE_BASE_PATH=/home/lilith/development/projects/persona-framework/workspace
   export WORKSPACE_PROJECTS_PATH=/home/lilith/development/projects
   ```
   Additional knobs live in `leviathan/src/main/resources/application.yaml`.
2. **Run Leviathan** (default port 8080):
   ```bash
   ./gradlew :leviathan:bootRun
   ```
3. **Run Bael** (default Vite dev server port 5173):
   ```bash
   cd bael
   npm install
   VITE_API_BASE_URL=http://localhost:8080 npm run dev
   ```

## Running other applications
- Hydra: `./gradlew :hydra:bootRun`
- Paimon (IntelliJ plugin dev): `./gradlew :paimon:runIde`
- Art Service: `./gradlew :art-service:bootRun`
- Nexus Standalone: `./gradlew :nexus-standalone:bootRun`
- CNC ERP microservices (one at a time, choose a free port): `./gradlew :cnc-customer-service:bootRun` (repeat for other `cnc-*` services)

## Testing
- Backend/agents/services: `./gradlew test`
- Bael unit tests: `cd bael && npm test`
- Bael e2e: `cd bael && npm run e2e`
- Paimon plugin tests: `./gradlew :paimon:test`

## Docker/Jib
- Build Leviathan image: `./gradlew :leviathan:jibDockerBuild`
- Build Hydra image: `./gradlew :hydra:jibDockerBuild`
  (uses Amazon Corretto 21 base image and respects gradle properties `containerName`, `imageTag`, etc.)

## Documentation
- API usage: `docs/API_USAGE_GUIDE.md`
- Developer index: `docs/developer/INDEX.md`
- Additional roadmaps and reports: `docs/` (numerous session summaries and audits)
- Application-specific guides: see the per-application READMEs referenced above.

## Support
For questions or issues, contact **Lilith Crook** (`lilith@daemoniorum.com`) or open a ticket in this repository.
