# Persona Framework v2 v2.0 - Test

Production-ready autonomous AI agent platform for Spring Boot applications.

## Quick Start (Docker Compose)

The fastest way to get started is with Docker Compose:

```bash
# 1. Clone and setup
git clone <repository-url>
cd persona-framework
make setup-env

# 2. Edit .env with your configuration
nano .env
# Set: SPRING_DATASOURCE_PASSWORD, AWS_REGION, AWS_PROFILE

# 3. Build and run
make build
make up

# 4. Access the application
# Hydra API:  http://localhost:8989
# Admin UI:   http://localhost:8989/persona-admin
# Swagger:    http://localhost:8989/swagger-ui.html
# Health:     http://localhost:8989/actuator/health
```

**See [DOCKER-COMPOSE-GUIDE.md](DOCKER-COMPOSE-GUIDE.md) for complete Docker Compose documentation.**

## Alternative: Manual Setup

### Prerequisites

- Java 21+
- PostgreSQL 13+
- AWS account with Bedrock access
- Node.js 20+ (for frontend development)

### Build and Run

```bash
# 1. Create database
createdb persona

# 2. Configure environment
cp .env.example .env
# Edit .env with your settings

# 3. Build
./gradlew clean build

# 4. Run
./gradlew :hydra:bootRun
```

## Project Structure

This project uses a **multi-repository architecture** where core framework modules are in this repo, while major applications live in separate repositories:

**Main Repository** (this repo):
```
persona-framework/
├── persona-core/          # Domain model (framework-agnostic)
├── persona-api/           # AI client interfaces
├── persona-aws/           # AWS Bedrock implementation
├── persona-anthropic/     # Anthropic API implementation
├── persona-openai/        # OpenAI API implementation
├── persona-agent/         # Autonomous agent framework
├── persona-mcp/           # Model Context Protocol integration
├── persona-sandbox/       # Sandboxed execution environment
├── persona-autoconfigure/ # Spring Boot auto-config
└── persona-rest/          # REST API controllers
```

**External Repositories** (clone locally for development):
```
├── mammon/     → github.com/LightspeedDMS/mammon (JPA + Flyway migrations)
├── hydra/      → github.com/LightspeedDMS/hydra (main Spring Boot application)
├── paimon/     → github.com/LightspeedDMS/paimon (IntelliJ IDEA plugin)
└── bael/       → (separate repo) (React admin UI)
```

**Build Setup:** For full project builds, clone external repos into their respective directories. See `settings.gradle.kts` for configuration details.

## Key Features

- **Autonomous Agents:** Multi-step task planning and execution with 15+ tools
- **Multi-Provider AI:** AWS Bedrock, Anthropic, OpenAI support
- **Multi-Project Workspaces:** Orchestrated build/deploy/test for multiple projects
- **RAG Integration:** S3, URL, database, and vector retrieval
- **IntelliJ Plugin:** IDE integration for code assistance
- **Database-Driven Routing:** Dynamic AI provider and model selection
- **Production-Ready:** Syntax validation, task verification, audit logging

## Documentation

- **[Complete Documentation](docs/README.md)** - Comprehensive project docs
- **[Docker Compose Guide](DOCKER-COMPOSE-GUIDE.md)** - Container deployment
- **[Architecture](docs/developer/ARCHITECTURE.md)** - System design
- **[Deployment](docs/developer/DEPLOYMENT.md)** - Production deployment
- **[API Guide](docs/developer/OVERSEER-API-GUIDE.md)** - Agent API usage

## Make Commands

```bash
make help          # Show all commands
make build         # Build Hydra Docker image
make up            # Start services (postgres + hydra)
make up-dev        # Start with admin-ui dev mode
make down          # Stop services
make logs          # View logs
make shell         # Open shell in Hydra container
make db-shell      # PostgreSQL shell
make test          # Run tests
make clean         # Clean up
```

See `make help` for full command list.

## Configuration

All configuration is via `.env` file:

**Required:**
```bash
SPRING_DATASOURCE_URL=jdbc:postgresql://localhost:5432/persona
SPRING_DATASOURCE_PASSWORD=your_password
AWS_REGION=us-west-2
AWS_PROFILE=dev
```

**Optional:**
```bash
HOST_PORT=8989
PERSONA_ADMIN_UI_ENABLED=true
PERSONA_AGENT_ENABLED=true
LOGGING_LEVEL_PERSONA=DEBUG
```

See [.env.example](.env.example) for all options.

## API Examples

### Chat with Persona

```bash
curl -X POST http://localhost:8989/api/public/chat/SR_SOFTWARE_ENGINEER \
  -H 'Content-Type: application/json' \
  -d '{"message": "Explain dependency injection"}'
```

### Submit Agent Task

```bash
curl -X POST http://localhost:8989/api/agent/tasks \
  -H 'Content-Type: application/json' \
  -d '{
    "description": "Add logging to UserService",
    "goal": "Add SLF4J logging to all public methods",
    "projectPath": "/workspace",
    "personaCode": "SR_SOFTWARE_ENGINEER"
  }'
```

### Submit Overseer Task (Multi-Step)

```bash
curl -X POST http://localhost:8989/api/overseer/tasks \
  -H 'Content-Type: application/json' \
  -d '{
    "goal": "Add user authentication feature",
    "description": "Create entity, repository, service, tests",
    "projectPath": "/workspace",
    "constraints": ["Use PostgreSQL", "Follow existing patterns"]
  }'
```

## Tech Stack

- **Backend:** Kotlin 2.2, Spring Boot 3.4, Java 21
- **Database:** PostgreSQL 13+, Flyway
- **AI:** AWS Bedrock (Claude), Anthropic API, OpenAI API
- **Frontend:** React 18, TypeScript 5, Vite 7, Fluent UI 9
- **Build:** Gradle 8.13+, Jib (Docker)
- **IDE:** IntelliJ IDEA plugin

## Development

### Prerequisites

```bash
# Install dependencies
brew install postgresql@16
brew install openjdk@21
brew install node@20
brew install docker

# Setup AWS
aws configure sso
```

### Running Locally

```bash
# Start PostgreSQL
brew services start postgresql@16

# Run application
./gradlew :hydra:bootRun

# Or with Docker
make build && make up
```

### Frontend Development

```bash
cd persona-admin-ui
npm install
npm run dev
# Open http://localhost:5173
```

### Running Tests

```bash
# Backend
make test
./gradlew test

# Frontend
cd persona-admin-ui
npm test
```

## IntelliJ Plugin (Paimon)

Paimon is now maintained in a separate repository. See: https://github.com/LightspeedDMS/paimon

```bash
# Clone paimon locally (if building full project)
git clone git@github.com:LightspeedDMS/paimon.git

# Build plugin
./gradlew :paimon:buildPlugin

# Run in test IDE
./gradlew :paimon:runIde
```

## Troubleshooting

### Database Connection Refused

```bash
# Check PostgreSQL is running
pg_isready

# For Docker Compose
make logs-postgres
make health
```

### AWS Credentials Not Found

```bash
# Login to AWS SSO
make aws-login
# OR: aws sso login --profile dev

# Verify credentials
aws sts get-caller-identity --profile dev
```

### Port Already in Use

```bash
# Change port in .env
echo "HOST_PORT=9090" >> .env
make down && make up
```

See [DOCKER-COMPOSE-GUIDE.md](DOCKER-COMPOSE-GUIDE.md#troubleshooting) for more troubleshooting.

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/description`)
3. Commit changes (`git commit -m 'feat(module): description'`)
4. Push to branch (`git push origin feature/description`)
5. Open Pull Request

## License

[Your License Here]

## Support

- **Documentation:** [docs/README.md](docs/README.md)
- **Issues:** [GitHub Issues]
- **API Docs:** http://localhost:8989/swagger-ui.html (when running)
