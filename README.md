# Persona Framework

**Autonomous AI agent platform for multi-step task execution with domain-specific personas, tool-based orchestration, and multi-provider AI integration.**

A production-ready Spring Boot + React system for building AI-powered development workflows with automatic tool usage, conversation persistence, and enterprise-grade safety controls.

---

## Quick Start

### Prerequisites
- **Java 21+** and **Gradle 8.13+**
- **Node 18+** and **npm/pnpm**
- **PostgreSQL 13+** (running locally or via Docker)
- **Ollama** (for local AI models) OR AWS/Anthropic/OpenAI credentials

### 1. Database Setup

```bash
# Start PostgreSQL (if using Docker)
docker run -d --name postgres \
  -e POSTGRES_PASSWORD=password \
  -p 5432:5432 postgres:16

# Create database
psql -h localhost -U postgres -c "CREATE DATABASE persona;"
```

### 2. Environment Configuration

```bash
# Copy example config
cp .env.example .env

# Edit .env with your settings:
# - Database connection (PostgreSQL)
# - AI provider (Ollama local / AWS / Anthropic / OpenAI)
# - Grimoire path (persona definitions)
```

**Minimal `.env` for local development:**
```bash
SPRING_DATASOURCE_URL=jdbc:postgresql://localhost:5432/persona
SPRING_DATASOURCE_USERNAME=postgres
SPRING_DATASOURCE_PASSWORD=password

# Use Ollama (local, no API keys needed)
PERSONA_AI_OLLAMA_ENABLED=true
PERSONA_AI_OLLAMA_BASE_URL=http://localhost:11434
```

### 3. Run Backend (Leviathan)

```bash
# Build and run
./gradlew :leviathan:bootRun

# Access at http://localhost:8989
# Flyway will auto-migrate database (119 migrations)
```

### 4. Run Frontend (Bael)

```bash
# Install dependencies
cd bael
npm install

# Start dev server
npm run dev

# Access at http://localhost:5173
```

### 5. Test the API

```bash
# Conversation-first API (recommended)
curl -X POST http://localhost:8989/api/conversation \
  -H 'Content-Type: application/json' \
  -d '{
    "userMessage": "List files in the current directory",
    "personaCode": "general-coder"
  }'

# The system will automatically:
# 1. Detect intent (needs file listing tool)
# 2. Execute bash tool
# 3. Return results with conversation history
```

---

## Architecture

### Three-Tier System

```
┌─────────────────────────────────────────────────────────┐
│  Bael (React + TypeScript Frontend)                     │
│  - Chat interface, Dashboard, Admin panels              │
│  - Fluent UI components, Terminal, Metrics              │
│  Port: 5173 (dev)                                        │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP/SSE
┌────────────────────┴────────────────────────────────────┐
│  Leviathan (Spring Boot Backend)                        │
│  - REST API (40+ endpoints)                             │
│  - Agent orchestration & tool execution                 │
│  - Multi-provider AI integration                        │
│  - Conversation management                              │
│  Port: 8989                                              │
└────────────────────┬────────────────────────────────────┘
                     │ JDBC
┌────────────────────┴────────────────────────────────────┐
│  PostgreSQL Database                                     │
│  - 119 Flyway migrations                                │
│  - Conversations, tasks, personas, prompts              │
│  - AI catalog, usage logs, workspaces                   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Paimon (IntelliJ IDEA Plugin)                          │
│  - IDE integration for in-editor agent workflows        │
└─────────────────────────────────────────────────────────┘
```

---

## Core Concepts

### 1. Conversation-First API

The **recommended way** to interact with the system. Automatically detects intent and uses tools.

```bash
POST /api/conversation
{
  "userMessage": "Create a new Kotlin class called UserService",
  "personaCode": "backend-engineer",
  "conversationId": "uuid" (optional, for follow-ups)
}
```

**What happens:**
1. LLM analyzes message and detects intent
2. Determines if tools are needed (WriteFile, Bash, etc.)
3. Executes tools autonomously
4. Returns results + conversation history
5. Stores conversation in database for follow-ups

### 2. Personas

Domain-specific AI agents loaded from **Grimoire** (`/home/lilith/development/projects/grimoire/personas/`):

- `general-coder` - General programming tasks
- `backend-engineer` - Spring Boot, Kotlin, Java
- `frontend-engineer` - React, TypeScript, UI
- `devops-engineer` - Docker, CI/CD, infra
- `dba-sql` - Database design, SQL
- `security-reviewer` - Security audits
- `architect` - System design
- `qa-tester` - Testing strategies
- And 15+ more specialized roles

Each persona has:
- **System prompts** - Role definition and behavior
- **Tool awareness** - Which tools they can use
- **Safety constraints** - Approval requirements
- **Provider bindings** - Which AI model to use

### 3. Agent Orchestration (4 Layers)

```
┌─────────────────────────────────────────────────────┐
│  1. Planning Layer (OverseerAgent)                  │
│     - Analyzes complex tasks                        │
│     - Creates execution plans (3-7 subtasks)        │
│     - Validates results against criteria            │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│  2. Coordination Layer (AgentOrchestrator)          │
│     - Multi-turn reasoning loop                     │
│     - Tool selection and sequencing                 │
│     - Context management                            │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│  3. Execution Layer (Tool Registry - 15+ tools)     │
│     - File ops: Read, Write, Edit, Glob, Grep       │
│     - System: Bash, Docker, PackageManager          │
│     - Web: WebFetch, WebSearch                      │
│     - Code: CodeGeneration, Testing                 │
│     - VCS: Git operations                           │
└──────────────────────┬──────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────┐
│  4. Safety Layer                                    │
│     - Path validation (no /etc, /sys writes)        │
│     - Approval gates (destructive operations)       │
│     - Audit logging (all actions tracked)           │
│     - Token budgets and iteration limits            │
└─────────────────────────────────────────────────────┘
```

### 4. Multi-Provider AI Support

**Active Providers:**
- **Ollama** (default, local) - `llama3.1`, `codellama`, `mistral`, etc.
- **AWS Bedrock** - Claude 3 (Haiku, Sonnet, Opus)
- **Anthropic API** - Direct Claude API access
- **OpenAI API** - GPT-4, GPT-3.5

**Provider Switching:**
```yaml
persona:
  ai:
    ollama:
      enabled: true
      base-url: http://localhost:11434
      model: llama3.1:latest
    aws:
      enabled: false  # Enable for AWS Bedrock
    anthropic:
      enabled: false  # Enable for direct Anthropic API
```

---

## API Endpoints

### Conversation API (Recommended)

**POST** `/api/conversation` - Conversation-first interface with automatic tool usage
```json
{
  "userMessage": "Add a REST controller for User CRUD",
  "personaCode": "backend-engineer",
  "conversationId": "optional-uuid-for-follow-ups",
  "workspaceId": "optional-project-workspace-id"
}
```

**Response:**
```json
{
  "conversationId": "uuid",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "toolsUsed": ["WriteFile", "Bash"]}
  ],
  "detectedIntent": "CODE_GENERATION",
  "requiresFollowUp": false
}
```

### Agent Execution

**POST** `/api/agent/tasks` - Submit autonomous agent task
```json
{
  "description": "Create a Spring Boot REST controller for User entity",
  "goal": "Implement CRUD operations with validation",
  "projectPath": "/path/to/project",
  "personaCode": "backend-engineer",
  "maxIterations": 25
}
```

**GET** `/api/agent/tasks/{id}` - Get task status

**GET** `/api/agent/tasks/{id}/events` - Stream execution events (SSE)

**POST** `/api/agent/analyze` - Get persona recommendations for a task

### Overseer (Complex Multi-Step Tasks)

**POST** `/api/overseer/tasks` - Submit complex task requiring planning
```json
{
  "goal": "Add user authentication with JWT",
  "description": "Create entity, repository, service, security config, and tests",
  "projectPath": "/path/to/project",
  "constraints": ["Use Spring Security", "PostgreSQL database"],
  "maxIterations": 50
}
```

**GET** `/api/overseer/tasks/{id}` - Get execution plan and results

### Chat (Simple Q&A, No Tools)

**POST** `/api/public/chat/{personaCode}` - Simple question/answer without tool execution

### Workspaces (Multi-Project Orchestration)

**GET** `/api/workspaces` - List all workspaces

**POST** `/api/workspaces` - Create workspace with multiple projects

**GET** `/api/workspaces/{id}/pipeline` - Get project build/test pipeline

### Administration (40+ Endpoints)

- **Personas**: `/api/personas`, `/api/admin/personas`
- **Prompts**: `/api/admin/prompts`, `/api/admin/prompts/sync`
- **Providers**: `/api/providers`, `/api/credentials`, `/api/integrations`
- **Usage Analytics**: `/api/usage/logs`, `/api/usage/summary`, `/api/usage/export`
- **Metrics**: `/api/metrics/historical`, `/api/metrics/tools`
- **Terminal**: `/api/terminal/sessions`, `/api/terminal/execute`
- **VCS**: `/api/vcs/status`, `/api/vcs/commit`
- **Docker**: `/api/docker/containers`, `/api/docker/images`
- **Dashboard**: `/api/dashboard/summary`
- **Settings**: `/api/settings`
- **Guardrails**: `/api/guardrails`
- **Knowledge Base**: `/api/knowledge/sources`
- **Logs**: `/api/logs/stream`

**Full API documentation:** `docs/API_USAGE_GUIDE.md` (880 lines)

---

## Active Modules

### Core Framework (14 Modules)

| Module | Purpose |
|--------|---------|
| **persona-core** | Domain model (framework-agnostic Kotlin) |
| **persona-api** | AI client interfaces (ChatClient, EmbeddingClient) |
| **persona-aws** | AWS Bedrock, S3, SageMaker adapters |
| **persona-anthropic** | Anthropic Claude API client |
| **persona-openai** | OpenAI GPT API client |
| **persona-ollama** | Ollama local model integration |
| **persona-agent** | Agent orchestration + 15+ tools |
| **persona-rest** | REST API controllers (40+ endpoints) |
| **persona-autoconfigure** | Spring Boot auto-configuration |
| **persona-mcp** | Model Context Protocol support |
| **persona-sandbox** | Sandboxed execution environment |
| **mammon** | PostgreSQL persistence (JPA + Flyway) |
| **grimoire** | Persona definitions and prompts |
| **eidolon** | Shared utilities and common code |

### Applications

| Application | Purpose | Port |
|-------------|---------|------|
| **leviathan** | Main Spring Boot backend | 8989 |
| **hydra** | Production Spring Boot app (similar to leviathan) | 8080 |
| **bael** | React + TypeScript admin UI | 5173 |
| **paimon** | IntelliJ IDEA plugin | N/A |

---

## Technology Stack

### Backend
- **Language**: Kotlin 2.2.0
- **Framework**: Spring Boot 3.4.9 (Java 21)
- **Database**: PostgreSQL 13+ (119 Flyway migrations)
- **Build**: Gradle 8.13+
- **Async**: Kotlin Coroutines
- **Containerization**: Jib (no Dockerfile needed)

### Frontend (Bael)
- **Framework**: React 18.3
- **Language**: TypeScript 5.9
- **Build**: Vite 7.1
- **UI**: Fluent UI 9 (Microsoft)
- **Testing**: Vitest, Playwright
- **Storybook**: Component development

### AI Integration
- **Ollama** - Local models (llama3.1, codellama, mistral)
- **AWS Bedrock** - Claude 3 via Converse API
- **Anthropic API** - Direct Claude access
- **OpenAI API** - GPT-4 and GPT-3.5

---

## Database Schema (119 Migrations)

### Core Tables
- **Conversations**: `conversation_session`, `conversation_message` (V119)
- **Personas**: `persona`, `persona_prompt`, `persona_integration_binding`
- **AI Catalog**: `ai_provider`, `ai_credential`, `ai_integration`, `ai_routing_policy`
- **Tasks**: `agent_task`, `agent_task_execution`, `agent_task_approval`
- **Workspaces**: `workspace`, `workspace_project` (V118)
- **Configuration**: `datasource` (V115), `agent_configuration` (V114)
- **Usage Analytics**: `ai_usage_log` (append-only audit trail)
- **Knowledge Base**: `research_source`, `persona_rag_source`
- **Metrics**: Tool execution stats, historical metrics

### Migration Strategy
- **Flyway** auto-migrates on startup
- **V001-V119** - Comprehensive schema evolution
- **Idempotent** - Safe to re-run
- **Testcontainers** - Automated testing with PostgreSQL containers

---

## Agent Tools (15+)

### File Operations
- **ReadFile** - Read file contents with line ranges
- **WriteFile** - Create new files with content
- **EditFile** - Replace specific text in files
- **Glob** - Pattern-based file search
- **Grep** - Content search with regex

### System Execution
- **Bash** - Execute shell commands (sandboxed)
- **Docker** - Container management
- **PackageManager** - npm, gradle, maven operations
- **Testing** - Run tests and parse results

### Web & Research
- **WebFetch** - HTTP GET with HTML→Markdown
- **WebSearch** - Search engine integration

### Code Generation
- **CodeGeneration** - Generate classes, interfaces, tests

### Version Control
- **Git** - Status, commit, push, branch operations

### Database
- **SQL** - Query execution and migrations

**Tool Success Rate Target**: 99% (recent improvements focused on error handling)

---

## Features

### ✅ Production Features

- **Conversation Management** - Persistent multi-turn conversations with intent detection
- **Multi-Provider AI** - Switch between Ollama, AWS, Anthropic, OpenAI
- **Autonomous Agents** - Planning, execution, and validation layers
- **Tool Ecosystem** - 15+ tools with safety controls
- **Multi-Project Workspaces** - Orchestrate tasks across multiple codebases
- **Terminal Integration** - Web-based terminal access
- **Approval Workflows** - Human-in-the-loop for destructive operations
- **Usage Analytics** - Track costs, tokens, tool usage across providers
- **RAG Retrieval** - S3/URL/DB/Vector knowledge base integration
- **Docker Deployment** - Jib containerization without Dockerfiles
- **IntelliJ Plugin** - IDE-native agent workflows

### 🚧 In Development

- **Vector Database** - pgvector integration for semantic search
- **Multi-Agent Coordination** - Multiple personas working together
- **Enhanced Metrics** - Real-time performance dashboards
- **Marketplace** - Extension system for custom tools

---

## Running in Production

### Docker (Jib)

```bash
# Build container image
./gradlew :leviathan:jibDockerBuild

# Run with environment file
docker run -d \
  --name persona-leviathan \
  --network host \
  --env-file .env \
  -v ~/.aws:/root/.aws:ro \
  leviathan:0.1.0

# Check logs
docker logs -f persona-leviathan
```

### Gradle Tasks

```bash
# Build and start container
./gradlew :leviathan:devUp

# Stop and remove container
./gradlew :leviathan:devDown

# Run tests
./gradlew test

# Build all modules
./gradlew build
```

### Frontend Deployment

```bash
# Build production bundle
cd bael
npm run build

# Output: bael/dist/
# Serve with nginx, Caddy, or any static host
```

---

## Configuration

### Environment Variables

**Database:**
```bash
SPRING_DATASOURCE_URL=jdbc:postgresql://localhost:5432/persona
SPRING_DATASOURCE_USERNAME=postgres
SPRING_DATASOURCE_PASSWORD=password
```

**AI Providers (Ollama - Local):**
```bash
PERSONA_AI_OLLAMA_ENABLED=true
PERSONA_AI_OLLAMA_BASE_URL=http://localhost:11434
PERSONA_AI_OLLAMA_MODEL=llama3.1:latest
PERSONA_AI_OLLAMA_TIMEOUT_SECONDS=600
```

**AI Providers (AWS Bedrock):**
```bash
PERSONA_AI_AWS_ENABLED=true
AWS_REGION=us-west-2
AWS_PROFILE=your-profile
# Or direct credentials:
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

**AI Providers (Anthropic):**
```bash
PERSONA_AI_ANTHROPIC_ENABLED=true
ANTHROPIC_API_KEY=sk-ant-...
```

**AI Providers (OpenAI):**
```bash
PERSONA_AI_OPENAI_ENABLED=true
OPENAI_API_KEY=sk-...
```

**Grimoire (Persona Definitions):**
```bash
GRIMOIRE_PERSONAS_PATH=/home/lilith/development/projects/grimoire/personas/
```

### Application YAML

See `leviathan/src/main/resources/application.yaml` for full configuration options.

---

## Development Workflow

### Local Development

```bash
# Terminal 1: Backend
./gradlew :leviathan:bootRun

# Terminal 2: Frontend
cd bael && npm run dev

# Terminal 3: Watch tests
./gradlew test --continuous

# Terminal 4: Ollama (if using local models)
ollama serve
```

### Hot Reload

- **Backend**: Spring DevTools (automatic restart on file changes)
- **Frontend**: Vite HMR (instant module replacement)

### Testing

```bash
# Backend unit tests
./gradlew test

# Backend integration tests (Testcontainers)
./gradlew integrationTest

# Frontend tests
cd bael && npm test

# E2E tests (Playwright)
cd bael && npm run test:e2e
```

---

## Documentation

### Main Docs

- **API_USAGE_GUIDE.md** - Comprehensive API documentation (880 lines)
- **docs/developer/INDEX.md** - Architecture, roadmaps, testing
- **docs/llm/INDEX.md** - LLM-optimized context
- **docs/DOCUMENTATION-INDEX.md** - Full documentation index

### Specialized Docs

- **Architecture**: `docs/developer/ARCHITECTURE.md`
- **Features**: `docs/developer/FEATURES.md`
- **Testing Strategy**: `docs/developer/TESTING-STRATEGY.md`
- **Roadmaps**: `docs/developer/roadmaps/`

---

## Troubleshooting

### Database Connection Issues

```bash
# Check PostgreSQL is running
psql -h localhost -U postgres -c "SELECT version();"

# Verify database exists
psql -h localhost -U postgres -l | grep persona

# Check Flyway migrations
./gradlew :leviathan:flywayInfo
```

### Ollama Not Working

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# Pull model if needed
ollama pull llama3.1

# Check logs
./gradlew :leviathan:bootRun --info | grep -i ollama
```

### Build Failures

```bash
# Clean build
./gradlew clean build --refresh-dependencies

# Check Java version
java -version  # Must be 21+

# Check Gradle version
./gradlew --version  # Must be 8.13+
```

### Port Already in Use

```bash
# Find process using port 8989
lsof -i :8989

# Kill process
kill -9 <PID>

# Or change port in application.yaml
server.port: 8990
```

---

## Roadmap

### ✅ Completed (Nov 2025)

- Conversation-first API with automatic tool usage
- LLM intent detection and follow-up handling
- Database persistence for conversations
- Multi-project workspace orchestration
- Endpoint consolidation (/api/delegation → /api/agent)
- Agent orchestrator performance improvements (50-70% boost)
- Comprehensive tool awareness system
- Terminal web access
- Enhanced error handling (99% tool success target)

### 🎯 Current Focus (Q4 2025)

- Vector database integration (pgvector)
- Multi-agent coordination (multiple personas collaborating)
- Real-time metrics dashboards
- Extension marketplace
- Enhanced approval workflows
- Streaming chat responses (SSE)

### 📋 Planned (Q1 2026)

- Agent memory persistence across sessions
- Cost optimization and smart model routing
- Advanced RAG with hybrid search
- Plugin ecosystem for custom tools
- Multi-tenant workspace isolation
- Enhanced IDE integrations (VS Code, Cursor)

---

## Contributing

### Guidelines

1. **Fork and branch**: Create feature branches from `main`
2. **Tests required**: Add tests for all new features
3. **Code style**: Follow Kotlin conventions, ktlint
4. **Commit messages**: Use conventional commits (`feat:`, `fix:`, `docs:`)
5. **Documentation**: Update relevant docs in `docs/`

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/persona-framework.git

# Setup environment
cp .env.example .env
# Edit .env with your settings

# Start PostgreSQL
docker-compose up -d postgres

# Run migrations
./gradlew :leviathan:flywayMigrate

# Start development
./gradlew :leviathan:bootRun
cd bael && npm run dev
```

---

## License

© 2025 LightSpeed DMS. All rights reserved.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/your-org/persona-framework/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/persona-framework/discussions)
- **Documentation**: `docs/` directory
- **API Guide**: `API_USAGE_GUIDE.md`

---

## Related Projects

- **Grimoire** - Persona definitions and prompt library
- **Wraith IDE** - Web-based IDE integration
- **Paimon** - IntelliJ IDEA plugin

---

**Built with** ❤️ **by the Persona Framework team**
