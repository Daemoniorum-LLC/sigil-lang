# Persona Framework

Autonomous AI agent platform for Spring Boot applications with domain‑specific personas, tool‑based execution, and optional IntelliJ IDEA integration. This repository hosts a multi‑module Kotlin/TypeScript codebase, a production app (Hydra), and comprehensive docs.


## Quick start

- Prerequisites
  - Java 21+, Gradle 8+
  - Node 18+ (for Admin UI development)
  - Docker (optional; for container builds/runs)

- Run the backend (Hydra)
  - Copy `.env.example` to `.env` and adjust values as needed
  - Build: `./gradlew build`
  - Run: `./gradlew :hydra:bootRun`

- Admin UI (optional, runs standalone in dev)
  - Path: `persona-admin-ui`
  - Install deps: `npm i`
  - Start: `npm run dev`

For broader setup details and workflows, see the Documentation section below.


## Modules (high level)

- persona-core — Domain model and ports (pure Kotlin)
- persona-persistence — JPA repositories and Flyway migrations
- persona-api — Agent/AI orchestration ports and DTOs
- persona-aws — AWS adapters (Bedrock, S3, SageMaker, Titan embeddings)
- persona-agent — Autonomous tool execution (file ops, web fetch, codegen)
- persona-rest — Spring Boot REST API and OpenAPI
- hydra — Main application packaging the platform
- persona-intellij-plugin — IntelliJ IDEA plugin (optional)
- persona-admin-ui — React Admin UI (optional; can be built/served separately)

Note: Some modules may be disabled in Gradle settings by default for faster builds.


## Documentation

Start here:
- docs/DOCUMENTATION-INDEX.md — Unified docs index for developers and LLM agents

Helpful entries:
- docs/developer/INDEX.md — Developer guides, roadmaps, testing
- docs/llm/INDEX.md — Context for LLM agents
- docs/hydra/llm/INDEX.md — Hydra task/automation docs


## Scripts and utilities

- submit-task-to-hydra.sh — Example submission flow for tasks
- scripts/ — Assorted automation and build helpers


## Contributing

- Please read docs/developer/INDEX.md for architecture, testing strategy, and active roadmaps.
- Open PRs with focused changes and include tests where applicable.


## License

Copyright (c) LightSpeed DMS. All rights reserved. Licensing terms may be provided separately.

---

## Comprehensive Documentation (merged from docs/README.md)

Below is the comprehensive, previously separate documentation merged into the root README for a single source of truth.

## What's New

### Latest Features (Phase 16+)
* 🆕 **OverseerAgent** - Planning and coordination agent for complex multi-step tasks (like Claude Code in hybrid development)
* 🆕 **Autonomous Agent System** (`persona-agent`) - Complete tool-based agent framework with web fetch, file operations, and code generation
* 🆕 **IntelliJ IDEA Plugin** (`persona-intellij-plugin`) - Native IDE integration with tool windows, context panels, and agent task submission
* 🆕 **Hydra Application** - Production-ready Spring Boot app integrating all modules with Docker/Jib support
* 🆕 **Agent REST API** - Submit tasks, stream execution events (SSE), monitor progress
* 🆕 **Hybrid Workflows** - Combine Claude Code and Persona agents for complex multi-step development
* 🆕 **Tool Registry** - Extensible tool system (WebFetch, FileRead, FileWrite, CodeGen, etc.)

### Core Features
* **PostgreSQL-first schema & seed**: `V001__persona_schema.sql` and `V002__persona_seed.sql` (timestamptz/jsonb, `BIGSERIAL`, `updated_at` triggers)
* **AI Catalog & Routing**: `ai_provider`, `ai_credential`, `ai_integration`, `ai_routing_policy`, and persona bindings in `persona_integration_binding` (with `fallback_integration_ids` CSV)
* **Safety pipeline**: **Supervisor Agent + Guardrails + Deterministic Verifier** enforcing "no action without evidence" for mission-critical personas
* **AWS adapters**: Bedrock (Converse), Titan embeddings, SageMaker predict, S3 blob ops (GET/PUT)
* **Knowledge base management**: `research_source`, `research_citation`, `research_official_domain` tables with URL validation, S3 integration hooks
* **RAG source catalog**: `persona_rag_source` table linking personas to S3/URL/DB/VECTOR retrieval sources
* **Usage analytics**: `ai_usage_log` with persona/provider/integration tracking, cost attribution, and append-only audit trail

---

## Module Layout

```
persona-framework/
├─ settings.gradle.kts
├─ build.gradle.kts                      # root (conventions + BOMs)
│
├─ persona-core/                         # domain model + ports (no Spring, pure Kotlin)
├─ persona-persistence/                  # Spring Data JPA adapters (PostgreSQL)
├─ persona-api/                          # AI orchestration ports + DTOs
├─ persona-aws/                          # AWS adapters (Bedrock, SageMaker, S3, Titan embeddings)
├─ persona-autoconfigure/                # Spring Boot autoconfiguration & dependency injection
├─ persona-rest/                         # REST controllers + OpenAPI docs
│
├─ persona-agent/                        # 🆕 Autonomous agent framework (tools, web fetch, code gen)
├─ persona-intellij-plugin/              # 🆕 IntelliJ IDEA plugin for IDE integration
│
├─ hydra/                                # 🎯 Main Spring Boot application (demo & production)
│
├─ persona-nl2sql/                       # [DISABLED] Natural language → SQL (optional)
├─ persona-admin-ui/                     # [DISABLED] React app (source)
├─ persona-admin-ui-resources/           # [DISABLED] Built static assets
└─ persona-admin-ui-starter/             # [DISABLED] Tiny starter to mount UI routes
```

### Active Modules (Currently Enabled)

| Module | Purpose | Dependencies |
|--------|---------|--------------|
| **persona-core** | Domain model, ports, DTOs (framework-agnostic) | None (pure Kotlin) |
| **persona-persistence** | JPA entities, repositories, Flyway migrations | `core` |
| **persona-api** | AI ports, configuration properties | `core` |
| **persona-aws** | AWS Bedrock, SageMaker, S3, Titan implementations | `core`, `api` |
| **persona-autoconfigure** | Spring Boot auto-configuration | `core`, `api` |
| **persona-rest** | REST API endpoints, OpenAPI specs | `core`, `api`, `persistence`, `agent` |
| **persona-agent** | Autonomous tool execution, web fetch, file ops | `core`, `api`, `aws` |
| **persona-intellij-plugin** | IntelliJ IDEA integration (tool windows, actions) | `core`, `api`, `persistence` |
| **hydra** | Main application (all integrations) | All above |

### Module Dependency Graph

```
┌─────────────┐
│ persona-core│◄─────┐
└──────┬──────┘      │
       │             │
       ▼             │
┌──────────────────┐ │
│persona-persistence│ │
└──────────────────┘ │
                    │
┌──────────────┐     │
│ persona-api  │◄────┤
└──────┬───────┘     │
       │             │
       ▼             │
┌──────────────┐     │
│ persona-aws  │     │
└──────────────┘     │
                    │
┌──────────────┐     │
│persona-agent │◄────┘
└──────────────┘
       │
       ▼
┌──────────────────┐
│ persona-rest     │
└──────────────────┘
       │
       ▼
┌──────────────────────┐
│persona-autoconfigure │
└──────────────────────┘
       │
       ▼
┌──────────────┐
│    hydra     │ (main application)
└──────────────┘

┌────────────────────────┐
│persona-intellij-plugin │ (standalone IDE plugin)
└────────────────────────┘
```

**Dependency Rules:**
- No module may introduce `core → persistence` (avoid build cycles)
- `persona-intellij-plugin` is standalone (uses core, api, persistence but doesn't affect main app)
- `persona-agent` can use web tools, file operations, and code generation
- All Spring dependencies isolated in `autoconfigure`, `rest`, `persistence`

---

## Requirements

* **Java 21+**
* **Gradle 8.13+**
* **PostgreSQL 13+** (no H2)
* AWS credentials (if you use AWS adapters)

---

## Getting Started
### 0) Local Development Setup

**Environment Configuration:**

Copy the environment template and configure for your setup:

```bash
cp .env.example .env
# OR for Hydra specifically:
cp .env.example hydra/.env
```

Edit `.env` with your configuration:

```bash
# Required: Database connection
SPRING_DATASOURCE_URL=jdbc:postgresql://localhost:5432/persona
SPRING_DATASOURCE_PASSWORD=your_actual_password

# Required: AWS Bedrock access
AWS_PROFILE=your_aws_profile
AWS_REGION=us-west-2
```

**Prerequisites:**

- **PostgreSQL 14+**: Running locally or via Docker
- **AWS Credentials**: Configured with Bedrock access
- **Java 17+** and **Gradle** installed

See `.env.example` for all available configuration options.

### 1) Add dependencies to your Spring Boot app

```kotlin
dependencies {
    implementation(platform("org.springframework.boot:spring-boot-dependencies:3.4.9"))

    implementation("com.lightspeeddms.persona:persona-core:0.1.0")
    implementation("com.lightspeeddms.persona:persona-persistence:0.1.0")
    implementation("com.lightspeeddms.persona:persona-ai-api:0.1.0")
    implementation("com.lightspeeddms.persona:persona-ai-aws:0.1.0")
    implementation("com.lightspeeddms.persona:persona-autoconfigure:0.1.0")

    // Optional
    implementation("com.lightspeeddms.persona:persona-nl2sql:0.1.0")
    implementation("com.lightspeeddms.persona:admin-ui-starter:0.1.0")
}
```

### 2) Database & Migrations (PostgreSQL only)

Place these two files on your classpath (e.g., `src/main/resources/db/migration`):

* `V001__persona_schema.sql` — creates **personas & config** tables and **AI Catalog & Routing** tables with `updated_at` triggers.
* `V002__persona_seed.sql` — idempotent seed: providers, credentials (AWS default), integrations (Bedrock/Titan/S3/SageMaker), prompts & guardrails, bindings, routing policy.

Spring config:

```yaml
spring:
  datasource:
    url: jdbc:postgresql://172.17.0.1:5432/persona
    username: postgres
    password:
  jpa:
    hibernate:
      ddl-auto: validate
    properties:
      hibernate:
        format_sql: true
        jdbc.time_zone: UTC
  flyway:
    enabled: true
    locations: classpath:db/migration
```

> **Testing**: use **Testcontainers** (`jdbc:tc:postgresql:16:///persona`) for integration tests.

### 3) Configure AWS adapters (example)

```yaml
persona:
  ai:
    default-provider: aws
    aws:
      region: us-west-2
      chat:
        model-id: anthropic.claude-3-haiku-20240307-v1:0
        max-tokens: 2000
        temperature: 0.1
      embed:
        model-id: amazon.titan-embed-text-v2:0
      sagemaker:
        endpoint: estimate-predictor
      s3:
        bucket: persona-assets-dev
```

Env overrides:

| Property                            | Env Var                             |
| ----------------------------------- | ----------------------------------- |
| `persona.ai.aws.region`             | `PERSONA_AI_AWS_REGION`             |
| `persona.ai.aws.chat.model-id`      | `PERSONA_AI_AWS_CHAT_MODEL_ID`      |
| `persona.ai.aws.embed.model-id`     | `PERSONA_AI_AWS_EMBED_MODEL_ID`     |
| `persona.ai.aws.sagemaker.endpoint` | `PERSONA_AI_AWS_SAGEMAKER_ENDPOINT` |
| `persona.ai.aws.s3.bucket`          | `PERSONA_AI_AWS_S3_BUCKET`          |

Standard AWS envs (`AWS_REGION`, `AWS_PROFILE`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`) are respected.

---

## Module Deep Dive

### persona-core
**Purpose:** Framework-agnostic domain model, ports, and DTOs.

**Key Components:**
- Domain entities (`Persona`, `PersonaPrompt`, `ContextSession`, etc.)
- Port interfaces (no implementations)
- DTOs for API contracts
- Validation annotations

**No Dependencies:** Pure Kotlin, no Spring, no external frameworks.

### persona-persistence
**Purpose:** PostgreSQL persistence layer using Spring Data JPA.

**Key Components:**
- JPA entities (`PersonaEntity`, `PersonaPromptEntity`, etc.)
- Spring Data repositories
- Flyway migrations (`V001__persona_schema.sql`, `V002__persona_seed.sql`)
- Database triggers for `updated_at` timestamps

**Schema Highlights:**
- `persona` - Domain personas (Technician, Service Writer, etc.)
- `persona_prompt` - Versioned prompt templates
- `ai_provider` - Provider catalog (AWS, OpenAI, etc.)
- `ai_integration` - Model/endpoint configurations
- `persona_integration_binding` - Persona→Integration routing with fallbacks
- `ai_usage_log` - Append-only usage tracking for cost attribution
- `research_source` - Knowledge base sources
- `persona_rag_source` - RAG retrieval bindings

### persona-api
**Purpose:** AI orchestration contracts and configuration.

**Key Components:**
- `ChatClient` interface - AI chat operations
- `EmbeddingClient` interface - Text embedding generation
- `ContextManager` interface - Conversation history
- Configuration properties (`PersonaAiProperties`)
- DTOs for chat, embeddings, context sessions

### persona-aws
**Purpose:** AWS service integrations (Bedrock, SageMaker, S3).

**Implementations:**
- **Bedrock Converse API** - `BedrockChatClient` (Claude, Titan, Llama models)
- **Titan Embeddings** - `TitanEmbeddingClient` (amazon.titan-embed-text-v2:0)
- **SageMaker** - `SageMakerPredictClient` (custom model endpoints)
- **S3** - `S3BlobClient` (GET/PUT operations, presigned URLs)

**Stub Implementations:**
- `StubChatClient` - Local development without AWS
- `StubS3ContentFetcher` - Filesystem-based S3 simulation (`build/stub-s3/`)

### persona-autoconfigure
**Purpose:** Spring Boot auto-configuration and dependency wiring.

**What It Does:**
- Auto-detects AWS credentials (profile, environment, IAM role)
- Configures chat, embedding, SageMaker, S3 clients
- Enables RAG retrieval services
- Registers repositories and services
- Stub/production mode switching

### persona-rest
**Purpose:** REST API endpoints with OpenAPI documentation.

**Endpoints:**
- `POST /api/public/chat/{personaCode}` - Chat with persona (streaming/non-streaming)
- `POST /api/public/embed` - Generate embeddings
- `GET /api/personas` - List all personas
- `GET /api/usage/logs` - Usage analytics
- `GET /api/usage/export` - CSV export
- `GET /api/knowledge/personas/{code}/rag-sources` - RAG source management
- `POST /api/agent/tasks` - 🆕 Submit autonomous agent tasks
- `GET /api/agent/tasks/{id}` - 🆕 Get agent task status
- `GET /api/agent/tasks/{id}/events` - 🆕 Stream agent execution events (SSE)

**OpenAPI Docs:** Available at `/swagger-ui.html`

### persona-agent 🆕
**Purpose:** Autonomous agent framework with tool execution capabilities.

**Architecture:**
```
                   ┌─────────────────────┐
                   │   OverseerAgent     │ (Planning & Coordination)
                   │  - Break down tasks │
                   │  - Create subtasks  │
                   │  - Validate results │
                   └──────────┬──────────┘
                             ↓
User Request → AgentOrchestrator → LLM Reasoning → Tool Execution → Result
                    ↓                                    ↑
               Task Planner ←──────── Memory ←──────────┘
```

**Tool System:**
- **Web Tools** - `WebFetchTool` (HTTP GET, HTML→Markdown conversion)
- **File Tools** - Read, Write, Edit, Glob, Grep operations
- **Code Tools** - Generate classes, interfaces, tests
- **Database Tools** - SQL execution, migrations
- **Build Tools** - Gradle, Maven, npm execution (future)

**Execution Loop:**
1. **Observe** - Gather context (project files, previous results)
2. **Think** - LLM reasons about next action
3. **Act** - Execute tool(s) based on LLM decision
4. **Update** - Store results in memory
5. **Repeat** - Continue until task complete or max iterations

**Example Tools:**
```kotlin
class WebFetchTool : AgentTool {
    override val name = "web_fetch"
    override val description = "Fetch and convert web page to markdown"

    override suspend fun execute(params: Map<String, Any>): ToolResult {
        val url = params["url"] as String
        val html = httpClient.get(url)
        val markdown = htmlToMarkdown(html)
        return ToolResult.success(markdown)
    }
}
```

**Agent Types:**

1. **OverseerAgent** (Planning & Coordination)
   - Analyzes complex tasks and creates execution plans
   - Breaks work into 3-7 subtasks with dependencies
   - Delegates to execution agents
   - Validates results against criteria
   - Handles retries and error recovery

2. **AgentOrchestrator** (Execution)
   - Executes individual subtasks
   - Multi-turn reasoning loop
   - Tool selection and invocation
   - Memory management

**Use Cases:**
- Autonomous code generation
- Multi-step refactoring
- Research and documentation
- Test creation and debugging
- Dependency updates
- Complex features spanning multiple layers

**API Integration:**

**Simple tasks** (single agent execution):
```bash
curl -X POST http://localhost:8989/api/agent/tasks \
  -H 'Content-Type: application/json' \
  -d '{
    "description": "Add logging to UserService",
    "goal": "Add SLF4J logging to all public methods",
    "projectPath": "/home/user/project",
    "personaCode": "SR_SOFTWARE_ENGINEER",
    "maxIterations": 25
  }'
```

**Complex tasks** (OverseerAgent with planning):
```bash
curl -X POST http://localhost:8989/api/overseer/tasks \
  -H 'Content-Type: application/json' \
  -d '{
    "goal": "Add user authentication feature",
    "description": "Create entity, repository, service, and tests",
    "projectPath": "/home/user/project",
    "constraints": ["Use PostgreSQL", "Follow existing patterns"],
    "maxIterations": 25
  }'

# Check task status
curl http://localhost:8989/api/overseer/tasks/{taskId}
```

**Safety Features:**
- Max iteration limits
- Tool execution sandboxing
- Human approval gates (optional)
- Audit logging
- Token budget tracking

### persona-intellij-plugin 🆕
**Purpose:** IntelliJ IDEA integration for Persona agent workflows.

**Features:**
- **Tool Window** - Chat interface with persona selection
- **Context Panel** - Automatic project context gathering
- **Actions** - Right-click context menus for file/code operations
- **Agent Integration** - Submit autonomous tasks from IDE
- **Real-time Updates** - SSE event streaming to UI

**UI Components:**
- `PersonaToolWindowFactory` - Main tool window
- `ChatPanel` - Message display and input
- `ContextPanel` - Project context visualization
- `PersonaApiService` - Backend API client
- `ProjectContextService` - IDE project introspection

**Usage:**
1. Open IntelliJ IDEA
2. View → Tool Windows → Persona
3. Select persona (e.g., "Software Engineer")
4. Enable project context
5. Chat or submit autonomous task

**Testing:**
```bash
# Run plugin in test IDE
./gradlew :persona-intellij-plugin:runIde

# Run tests
./gradlew :persona-intellij-plugin:test

# Build plugin distribution
./gradlew :persona-intellij-plugin:buildPlugin
```

**Installation:**
1. Build plugin: `./gradlew :persona-intellij-plugin:buildPlugin`
2. Distribution: `persona-intellij-plugin/build/distributions/persona-intellij-plugin-0.1.0.zip`
3. Install in IDE: Settings → Plugins → ⚙️ → Install Plugin from Disk

### hydra 🎯
**Purpose:** Main Spring Boot application integrating all modules.

**What It Includes:**
- All persona modules (core, persistence, api, aws, rest, agent)
- Flyway database migrations
- Security configuration
- Docker support (Jib)
- Health checks and actuators
- Usage analytics endpoints
- RAG retrieval
- Agent task management

**Configuration:**
- Database: PostgreSQL (Docker or remote)
- AI Provider: AWS Bedrock (Haiku, Sonnet, Opus)
- Embeddings: Titan v2
- Storage: S3 or local stub
- Profiles: `local`, `dev`, `prod`

**Deployment:**
```bash
# Build Docker image
./gradlew :hydra:jibDockerBuild

# Run container
cd hydra
docker run -d \
  --name hydra \
  --network host \
  --env-file .env \
  -v ~/.aws:/root/.aws:ro \
  hydra:0.1.0
```

---

## Running the Application

### Local Development (bootRun)

```bash
# Run Hydra (main application)
./gradlew :hydra:bootRun

# Or use convenience task
./gradlew :hydra:devUp
```

### Docker Deployment (Jib)

The project uses [Jib](https://github.com/GoogleContainerTools/jib) for containerization without requiring a Dockerfile.

**1. Build Docker image:**

```bash
./gradlew :hydra:jibDockerBuild
```

This creates `hydra:0.1.0` in your local Docker daemon.

**2. Create environment file** (`hydra/.env`):

```env
SPRING_DATASOURCE_URL=jdbc:postgresql://172.17.0.1:5432/persona
SPRING_DATASOURCE_USERNAME=postgres
SPRING_DATASOURCE_PASSWORD=yourpassword
AWS_REGION=us-west-2
AWS_PROFILE=dev
```

**3. Run container:**

```bash
cd hydra
docker run -d \
  --name hydra \
  --network host \
  --env-file .env \
  -v ~/.aws:/root/.aws:ro \
  hydra:0.1.0
```

**Or use convenience Gradle tasks:**

```bash
# Build image + create/start container in one step
./gradlew :hydra:devUp

# Stop and remove container
./gradlew :hydra:devDown
```

**4. Check logs:**

```bash
docker logs -f hydra
```

**5. Test endpoints:**

```bash
# Health check
curl http://localhost:8080/actuator/health

# List personas
curl http://localhost:8080/api/personas | jq

# Chat with Software Engineer persona
curl -X POST http://localhost:8080/api/public/chat/SR_SOFTWARE_ENGINEER \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is 2+2?"}],"stream":false}' | jq

# View usage logs
curl http://localhost:8080/api/usage/logs | jq

# Export usage to CSV
curl http://localhost:8080/api/usage/export?format=csv

# Chat with RAG context retrieval enabled
curl -X POST 'http://localhost:8080/api/public/chat/SR_SOFTWARE_ENGINEER?useRag=true' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What are Spring Boot configuration best practices?"}],"stream":false}' | jq

# List RAG sources for a persona
curl http://localhost:8080/api/knowledge/personas/SR_SOFTWARE_ENGINEER/rag-sources | jq
```

**Notes:**
- `--network host` allows container to access PostgreSQL on host machine
- `-v ~/.aws:/root/.aws:ro` mounts AWS credentials (read-only)
- Ensure AWS SSO session is active: `aws sso login --profile dev`
- Container uses Java 21 (Amazon Corretto Alpine)
- Mount stub-s3 directory for local RAG testing: `-v /path/to/build/stub-s3:/app/build/stub-s3:ro`

---

## RAG (Retrieval Augmented Generation)

The framework supports RAG context retrieval from multiple source types to enhance chat responses with relevant information.

### RAG Source Types

- **S3**: Fetch documents from S3 buckets (or local filesystem in stub mode)
- **URL**: Retrieve content from HTTP/HTTPS URLs
- **DB**: Execute database queries for structured data
- **VECTOR**: Perform vector similarity search (requires vector database)

### Configuration

Enable RAG retrieval on a per-request basis using the `useRag` query parameter:

```bash
# Without RAG (default)
POST /api/public/chat/{personaCode}

# With RAG enabled
POST /api/public/chat/{personaCode}?useRag=true
```

### Managing RAG Sources

RAG sources are stored in the `persona_rag_source` table and bound to specific personas:

```bash
# List RAG sources for a persona
GET /api/knowledge/personas/{personaCode}/rag-sources

# Response:
# [
#   {
#     "id": "uuid",
#     "personaCode": "SR_SOFTWARE_ENGINEER",
#     "sourceType": "S3",
#     "uri": "s3://bucket/key",
#     "metadata": {"description": "..."},
#     "createdAt": "2025-10-18T..."
#   }
# ]
```

### Development Mode (Stub Implementations)

For local development without AWS dependencies, the framework includes stub implementations:

```yaml
persona:
  ai:
    aws:
      s3:
        stub:
          enabled: true
          path: ./build/stub-s3  # Local filesystem path
    rag:
      stub:
        enabled: true  # Use stub fetchers for URL/DB/VECTOR
```

**Stub fetchers:**
- `StubS3ContentFetcher`: Reads from local `build/stub-s3/{bucket}/{key}` directory
- `StubUrlContentFetcher`: Returns placeholder HTTP content
- `StubDatabaseQueryExecutor`: Returns placeholder SQL results
- `StubVectorSearchExecutor`: Returns placeholder search results

### Example RAG Workflow

1. **Create RAG sources** (via migration or API):
```sql
INSERT INTO persona_rag_source (id, persona_id, source_type, uri, metadata_json)
VALUES (
  gen_random_uuid(),
  (SELECT id FROM persona WHERE code = 'SR_SOFTWARE_ENGINEER'),
  'S3',
  's3://docs/spring-boot-guide.md',
  '{"description": "Spring Boot best practices"}'::jsonb
);
```

2. **Place content in stub-s3** (development):
```bash
mkdir -p build/stub-s3/docs
echo "# Spring Boot Best Practices..." > build/stub-s3/docs/spring-boot-guide.md
```

3. **Chat with RAG enabled**:
```bash
curl -X POST 'http://localhost:8080/api/public/chat/SR_SOFTWARE_ENGINEER?useRag=true' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What are Spring Boot best practices?"}],"stream":false}'
```

The system will:
- Retrieve RAG sources for the persona from the database
- Fetch content from each source (S3, URL, DB, VECTOR)
- Append retrieved context to the system message
- Send enriched prompt to the AI model
- Return contextually enhanced response

### Token Budget Management

RAG retrieval respects token limits to avoid exceeding model context windows:

- Default: 2000 tokens for RAG context
- Estimation: ~0.75 tokens per word
- Retrieval stops when budget is reached
- Sources are processed in order until limit

---

## Safety Pipeline (New)

**Supervisor Agent** sits between persona outputs and users, enforcing:

* **Deterministic verification** against manuals/spec DBs (no "trust the LLM").
* **Guardrails** (e.g., forbid disabling safety systems; enforce spec ranges).
* **Evidence requirement** (every actionable claim must include provenance).
* **Confidence thresholds** (e.g., ≥0.95 for technician instructions).
* **Auditable outcomes** (append-only logs + usage metrics).

Minimal flow:

1. Persona builds prompt → model returns **structured claims** (JSON).
2. Verifier confirms each claim's evidence against authoritative sources.
3. Guardrails evaluate text for safety/compliance.
4. Supervisor approves or escalates; all decisions are logged.

---

## AI Catalog & Routing (New)

Centralized, DB-driven configuration:

* **Providers** (`ai_provider`): AWS / OpenAI / Azure / GCP / …
* **Credentials** (`ai_credential`): references to secret managers.
* **Integrations** (`ai_integration`): operation + model/endpoint/region.
* **Bindings** (`persona_integration_binding`): persona→primary integration (+ CSV fallbacks).
* **Policies** (`ai_routing_policy`): weighted/priority routing by `selector_json`.

Example binding (seeded):

* All personas' `CHAT` → **Bedrock Haiku** (@ `us-west-2`).
* Technician `PREDICT` → **SageMaker** endpoint (`estimate-predictor`) with optional fallbacks.

---

## Admin UI (Optional)

Include `admin-ui-starter` to serve the built React bundle:

```kotlin
implementation("com.lightspeeddms.persona:admin-ui-starter:0.1.0")
```

Expose it (dev example):

```kotlin
@Bean
@Order(100)
fun personaAdminUiChain(http: HttpSecurity): SecurityFilterChain =
    http.securityMatcher("/persona-admin/**")
        .authorizeHttpRequests { it.anyRequest().permitAll() }
        .csrf { it.disable() }
        .httpBasic { it.disable() }
        .formLogin { it.disable() }
        .build()
```

Default path: `/persona-admin`.

---

## REST API Endpoints

### Chat
- `POST /api/public/chat/{personaCode}` – Chat with a persona
  - Query params: `useRag=true` (enable RAG retrieval)
  - Request body: `{"messages": [{"role": "user", "content": "..."}], "stream": false}`
  - Response: `{"text": "...", "usage": {...}}`

### Embeddings
- `POST /api/public/embed` – Generate embeddings for texts
  - Request body: `{"texts": ["text1", "text2"]}`
  - Response: `{"embeddings": [[0.1, 0.2, ...], ...]}`

### Natural Language to SQL (DISABLED)
- `POST /api/public/nl2sql/{personaCode}` – Generate SQL from natural language (module currently disabled)

### Agent Tasks 🆕

**Simple Agent Execution** (`/api/agent/tasks`):
- `POST /api/agent/tasks` – Submit single-agent task
  - Request body:
    ```json
    {
      "description": "Add logging to UserService",
      "goal": "Add SLF4J logging to all public methods",
      "projectPath": "/path/to/project",
      "personaCode": "SR_SOFTWARE_ENGINEER",
      "maxIterations": 25
    }
    ```
  - Response: `{"taskId": "uuid", "status": "PENDING"}`

- `GET /api/agent/tasks/{taskId}` – Get task status and result
- `GET /api/agent/tasks/{taskId}/events` – Stream execution events (SSE)
- `GET /api/agent/tasks` – List all tasks (with filters)

**Delegation API** (`/api/delegation`) - Multi-Persona Task Routing:
- `POST /api/delegation/analyze` – Analyze task and create delegation plan
  - Request body:
    ```json
    {
      "goal": "Optimize database queries",
      "description": "Review and optimize slow queries in UserService",
      "projectPath": "/path/to/project",
      "fileContexts": [
        {
          "relativePath": "src/main/kotlin/UserService.kt",
          "content": "...",
          "language": "kotlin",
          "lineCount": 150,
          "sizeBytes": 4500
        }
      ]
    }
    ```
  - Response: Delegation plan with persona assignment, confidence, reasoning, subtasks

- `POST /api/delegation/execute` – Execute task with delegation
  - Analyzes task, routes to appropriate persona(s), coordinates execution
  - Supports file context transmission (filesystem-agnostic)
  - Returns consolidated results with execution metrics

**Overseer Agent** (`/api/overseer/tasks`) - Complex Task Planning:
- `POST /api/overseer/tasks` – Submit complex task requiring planning
  - Request body:
    ```json
    {
      "goal": "Add user authentication feature",
      "description": "Create entity, repository, service, and tests",
      "projectPath": "/path/to/project",
      "constraints": ["Use PostgreSQL", "Follow existing patterns"],
      "maxIterations": 25
    }
    ```
  - Response: `{"taskId": "uuid", "status": "PLANNING"}`

- `GET /api/overseer/tasks/{taskId}` – Get task status, execution plan, and results
- `GET /api/overseer/tasks` – List overseer tasks (with filters)

**When to use which:**
- **Simple Agent** (`/api/agent/tasks`): Single-file changes, focused refactoring, quick fixes
- **Overseer Agent** (`/api/overseer/tasks`): Multi-file features, layered changes, complex refactoring

See [OVERSEER-API-GUIDE.md](docs/OVERSEER-API-GUIDE.md) for detailed documentation.

### Usage Analytics
- `GET /api/usage/logs` – List usage logs with filters (persona, provider, operation, date range)
- `GET /api/usage/summary` – Aggregated cost/token statistics
- `GET /api/usage/export` – Export usage logs as CSV

### Knowledge Base
- `GET /api/knowledge/sources` – List research sources
- `POST /api/knowledge/sources` – Store a new research source
- `GET /api/knowledge/citations/{sourceId}` – Get citations for a source
- `POST /api/knowledge/sources/{id}/attach-s3` – Attach S3 object to source
- `GET /api/knowledge/personas/{code}/rag-sources` – List RAG sources bound to persona

### Admin
- `GET /api/personas` – List all personas
- `GET /actuator/health` – Health check
- OpenAPI docs available at `/swagger-ui.html`

---

## Using Personas in Code

### Chat with a Persona

```kotlin
@Service
class SupportAgent(
  private val personas: PersonaRepositoryPort,
  private val prompts: PersonaPromptRepositoryPort,
  private val chat: ChatClient
) {
  fun answer(code: String, userText: String): String {
    val persona = personas.findByCode(code) ?: error("persona not found: $code")
    val systemStack = prompts.findByPersonaId(persona.id)
      .filter { it.kind == "SYSTEM" }
      .sortedBy { it.version }
      .joinToString("\n") { it.content }
    val messages = listOf(ChatMessage.system(systemStack), ChatMessage.user(userText))
    return chat.complete(messages).text
  }
}
```

### NL→SQL (Optional)

```kotlin
class ReportsService(private val nl2sql: Nl2SqlClient) {
  fun ask(reportIntent: String): String =
    nl2sql.generate(
      personaCode = "ACCOUNTING",
      instruction = reportIntent,
      constraints = mapOf("schemas" to listOf("public","acct"), "max_rows" to 1000),
      dryRun = true
    )
}
```

---

## Build & Test

```bash
./gradlew clean build
```

Testcontainers example (`src/test/resources/application-test.yml`):

```yaml
spring:
  datasource:
    url: jdbc:tc:postgresql:16:///persona
    driver-class-name: org.testcontainers.jdbc.ContainerDatabaseDriver
  jpa:
    hibernate.ddl-auto: validate
  flyway.enabled: true
```

---

## Troubleshooting

* **Build cycles**: ensure `core` does not depend on `persistence`.
* **Spring Data types not found**: add Boot BOM and `spring-boot-starter-data-jpa` to `persona-persistence`.
* **Kotlin JPA plugin**: apply `kotlin("plugin.jpa")` with a version in module `plugins`.
* **RepositoriesMode error**: define repos in `settings.gradle.kts`.
* **JSONB errors**: you're not on Postgres—switch to Postgres (no H2).

---

## Roadmap

### Completed ✅
* ✅ **Autonomous Agent System** - Full tool-based execution framework
* ✅ **IntelliJ IDEA Plugin** - IDE integration with UI and API client
* ✅ **Usage Analytics Endpoints** - Full REST API for ai_usage_log with CSV export
* ✅ **RAG Retrieval Service** - S3/URL/DB/VECTOR retrieval with stub implementations
* ✅ **Agent Task Management** - Submit, monitor, stream events via REST
* ✅ **Hybrid Workflow Scripts** - `scripts/hybrid-test.sh` for autonomous execution
* ✅ **Docker/Jib Deployment** - Production-ready containerization

### High Priority (Q1 2025)
* **IntelliJ Plugin Marketplace** – Publish to JetBrains marketplace
* **Agent Tool Expansion** – Add Gradle, Maven, Git, Test execution tools
* **Vector Database Integration** – Implement real vector search (Pinecone/Weaviate/pgvector)
* **Knowledge Base UI** – Admin panel for research_source, research_citation, persona_rag_source management
* **Multi-persona Coordination** – Multiple agents working together on complex tasks
* **Cost Optimization** – Smart model routing (Haiku → Sonnet → Opus based on complexity)

### Medium Priority (Q2 2025)
* **Streaming Chat** – Server-sent events for real-time token streaming
* **SageMaker Adapter** – Complete endpoint invocation + batch transform
* **Tool Sandboxing** – Enhanced security for agent file/shell operations
* **Prompt Versioning** – S3-based prompt bundles with A/B testing
* **Agent Memory** – Persistent working memory across sessions
* **Human-in-the-Loop** – Approval gates for critical operations

### Future (Q3-Q4 2025)
* **Multi-tenant Partitioning** – Workspace/organization isolation
* **OpenTelemetry Tracing** – Distributed cost attribution and performance monitoring
* **Expanded Guardrails** – Regex + semantic + policy graph validation
* **WrenchML Integration** – Supervisor + evidence verification for automotive workflows
* **Function Calling DSL** – Declarative tool/function schemas with validation
* **Agent Swarms** – Coordinated multi-agent problem solving

---

## Hybrid Workflow: Claude Code + Persona Agent

The Persona Framework enables a powerful hybrid workflow combining human-guided AI (Claude Code) with autonomous execution (Persona Agent and OverseerAgent).

### Three-Tier Development Model

```
┌────────────────────────────────────────────────────────┐
│  Claude Code (Human-guided AI)                         │
│  - Research and understand codebases                   |
│  - Make architectural decisions                        |
│  - Review and validate work                            |
│  - Handle ambiguity and complex reasoning              |
└───────────────────┬────────────────────────────────────┘
                   │ Delegates complex tasks
                   ↓
┌────────────────────────────────────────────────────────┐
│  OverseerAgent (AI Planning & Coordination)            │
│  - Break down complex tasks into subtasks              │
│  - Create execution plans with dependencies            │
│  - Validate results against criteria                   │
│  - Coordinate multi-step execution                     │
└───────────────────┬────────────────────────────────────┘
                   │ Delegates subtasks
                   ↓
┌────────────────────────────────────────────────────────┐
│  AgentOrchestrator (AI Execution)                      │
│  - Execute focused subtasks                            │
│  - Use tools (read, write, edit files)                 │
│  - Multi-turn reasoning                                │
│  - Report results back to overseer                     │
└────────────────────────────────────────────────────────┘
```

### Workflow Patterns

#### Pattern 1: Research → Planning → Autonomous Implementation
```
1. Claude Code: Research codebase, design solution
2. OverseerAgent: Break down into execution plan
3. AgentOrchestrator: Execute each subtask
4. OverseerAgent: Validate results
5. Claude Code: Review and merge
```

**Example:**
```bash
# Submit complex task to OverseerAgent
curl -X POST http://localhost:8989/api/overseer/tasks \
  -H 'Content-Type: application/json' \
  -d '{
    "goal": "Add user authentication feature",
    "description": "Create entity, repository, service layer, and tests",
    "projectPath": "/home/user/project",
    "constraints": [
      "Use PostgreSQL for persistence",
      "Follow existing project patterns",
      "Include comprehensive unit tests"
    ],
    "maxIterations": 25
  }'

# Monitor execution plan and results
curl http://localhost:8989/api/overseer/tasks/{taskId}
```

#### Pattern 2: Test-Driven Development
```
1. Claude Code: Write failing tests
2. Persona Agent: Implement code to make tests pass
3. Claude Code: Review and refactor
```

**Example:**
```bash
# Automated via script
./scripts/hybrid-test.sh
```

#### Pattern 3: Iterative Refinement
```
1. Persona Agent: Generate initial implementation
2. Claude Code: Review, provide feedback
3. Persona Agent: Apply feedback autonomously
4. Repeat until satisfied
```

### Scripts

#### `scripts/hybrid-test.sh`
Automated test execution with agent-driven fixes:
```bash
#!/bin/bash
# 1. Run tests
# 2. If failures, submit agent task to fix
# 3. Stream progress
# 4. Re-run tests
# 5. Report results
```

**Usage:**
```bash
./scripts/hybrid-test.sh
```

**Environment Variables:**
- `PERSONA_SERVER_URL` - Backend URL (default: http://localhost:8989)
- `PERSONA_API_TOKEN` - Optional bearer token
- `PERSONA_CODE` - Persona to use (default: SR_SOFTWARE_ENGINEER)

### Best Practices

**When to use Claude Code:**
- Research and understanding codebases
- High-level planning and architecture
- Complex decision-making requiring judgment
- Code review and quality assessment
- Writing documentation and specifications

**When to use OverseerAgent:**
- Complex features spanning multiple layers
- Multi-file coordinated changes
- Tasks requiring planning and validation
- Features with clear validation criteria
- Work that benefits from subtask breakdown

**When to use Simple Agent:**
- Single-file modifications
- Focused refactoring tasks
- Quick fixes and improvements
- Test generation from specifications
- Simple documentation updates

**When to use Hybrid (All Three):**
- Large features requiring architecture + planning + implementation
- Test-driven development workflows
- Legacy code modernization projects
- Performance optimization (analyze → plan → fix → validate)
- Security enhancements (audit → plan → remediate → verify)

**Success Metrics**

Track your hybrid workflow effectiveness:
```bash
# Agent task success rate
curl http://localhost:8989/api/agent/tasks?status=COMPLETED

# Usage analytics
curl http://localhost:8989/api/usage/summary

# Export for analysis
curl http://localhost:8989/api/usage/export?format=csv > usage.csv
```

---

## Contributing

1. Fork and create a feature branch.
2. Keep modules acyclic (`./gradlew :persona-core:dependencies`).
3. Add tests (prefer Testcontainers).
4. Open a PR with a clear description.

---

## License

© 2025 LightspeedDMS. All rights reserved.

---

## Appendix: Architecture Diagrams

### System Architecture

```
┌────────────────────────────────────────────────────────────┐
│                      Client Layer                          │
│  ┌─────────────┐  ┌────────────┐  ┌──────────────────┐   │
│  │ IntelliJ    │  │  HTTP      │  │  CLI Scripts     │   │
│  │  Plugin     │  │  Client    │  │  (curl/hybrid)   │   │
│  └─────────────┘  └────────────┘  └──────────────────┘   │
└────────────────────────────────────────────────────────────┘
                         ↓ REST API
┌────────────────────────────────────────────────────────────┐
│                   Hydra (Spring Boot)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ REST         │  │ Agent        │  │ Chat           │  │
│  │ Controllers  │  │ Orchestrator │  │ Service        │  │
│  └──────────────┘  └──────────────┘  └────────────────┘  │
└────────────────────────────────────────────────────────────┘
         ↓                    ↓                   ↓
┌────────────────────────────────────────────────────────────┐
│                    Service Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ Persistence  │  │ AI Services  │  │ RAG Retrieval  │  │
│  │ (JPA/JDBC)   │  │ (AWS/Stub)   │  │ (S3/URL/DB)    │  │
│  └──────────────┘  └──────────────┘  └────────────────┘  │
└────────────────────────────────────────────────────────────┘
         ↓                    ↓                   ↓
┌────────────────────────────────────────────────────────────┐
│                  External Systems                          │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────┐  │
│  │ PostgreSQL   │  │ AWS Bedrock  │  │ AWS S3         │  │
│  │ Database     │  │ (Claude,etc) │  │ (RAG docs)     │  │
│  └──────────────┘  └──────────────┘  └────────────────┘  │
└────────────────────────────────────────────────────────────┘
```

### Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                        Core Layer                           │
│                     (No Dependencies)                       │
│                    ┌─────────────┐                          │
│                    │persona-core │                          │
│                    └──────┬──────┘                          │
└───────────────────────────┼──────────────────────────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
       ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│persona-      │    │persona-      │    │persona-      │
│persistence   │    │api           │    │agent         │
│              │    │              │    │              │
│(JPA/Flyway)  │    │(Interfaces)  │    │(Tools)       │
└──────────────┘    └──────┬───────┘    └──────────────┘
                           │
                           ▼
                   ┌──────────────┐
                   │persona-      │
                   │aws           │
                   │              │
                   │(Bedrock/S3)  │
                   └──────────────┘
                           │
       ┌───────────────────┼───────────────────┐
       │                   │                   │
       ▼                   ▼                   ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│persona-      │    │persona-      │    │persona-      │
│autoconfigure │    │rest          │    │intellij-     │
│              │    │              │    │plugin        │
│(Spring Boot) │    │(Controllers) │    │(IDE UI)      │
└──────┬───────┘    └──────┬───────┘    └──────────────┘
      │                   │
      └───────────┬───────┘
                  │
                  ▼
          ┌──────────────┐
          │   hydra      │
          │              │
          │(Main App)    │
          └──────────────┘
```

### Agent Execution Flow

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Task Submission                                          │
│    POST /api/agent/tasks                                    │
│    { "description": "...", "goal": "...", ... }             │
└────────────────────────┬────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Agent Orchestrator                                       │
│    - Parse task                                             │
│    - Create execution plan                                  │
│    - Initialize context                                     │
└────────────────────────┬────────────────────────────────────┘
                        ↓
        ┌───────────────┴───────────────┐
        ↓                               ↓
┌──────────────────┐            ┌──────────────────┐
│ 3a. Observe      │            │ 3b. LLM Reasoning│
│ - Read files     │            │ - Analyze context│
│ - Check status   │───────────▶│ - Plan next step │
│ - Review history │            │ - Select tool(s) │
└──────────────────┘            └────────┬─────────┘
                                        ↓
                             ┌──────────────────┐
                             │ 4. Tool Execution│
                             │ - WebFetch       │
                             │ - FileRead/Write │
                             │ - CodeGen        │
                             │ - SQL            │
                             └────────┬─────────┘
                                      ↓
                             ┌──────────────────┐
                             │ 5. Update Memory │
                             │ - Store results  │
                             │ - Update context │
                             └────────┬─────────┘
                                      ↓
                             ┌──────────────────┐
                             │ 6. Check Status  │
                             │ - Goal achieved? │
                             │ - Max iterations?│
                             │ - Error?         │
                             └────────┬─────────┘
                                      ↓
                        ┌─────────────┴─────────────┐
                        ↓                           ↓
             ┌──────────────────┐       ┌──────────────────┐
             │ 7a. Continue     │       │ 7b. Complete     │
             │ Loop to step 3   │       │ Return result    │
             └──────────────────┘       └──────────────────┘
```
