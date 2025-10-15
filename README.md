# Persona Framework — Domain-specific AI chat agents for WrenchML (and friends)

*A lightweight, domain-oriented chat-agent framework for Spring Boot apps.*

Persona lets you define **domain personas** (Technician, Service Writer, Inventory, Sales, Accounting, Marketing, Management) with their own prompt stacks and policies, then plug in AI providers (starting with **AWS Bedrock**, **SageMaker**, **S3**) via clean ports/adapters. An optional **NL→SQL** module adds governed database querying to any persona.



## What’s New

* **PostgreSQL-first schema & seed**: `V001__persona_schema.sql` and `V002__persona_seed.sql` (timestamptz/jsonb, `BIGSERIAL`, `updated_at` triggers).
* **AI Catalog & Routing**: `ai_provider`, `ai_credential`, `ai_integration`, `ai_routing_policy`, and persona bindings in `persona_integration_binding` (with `fallback_integration_ids` CSV).
* **Safety pipeline**: **Supervisor Agent + Guardrails + Deterministic Verifier** enforcing “no action without evidence” for mission-critical personas.
* **Admin UI starter**: optional React UI bundle served by Spring Boot to manage personas, prompts, and integrations.
* **Seeded prompts**: WrenchML diagnostic JSON template, data normalization template, and an assistant system voice for all personas.
* **AWS adapters**: Bedrock (Converse), Titan embeddings, SageMaker predict, S3 blob ops (GET/PUT).

---

## Module Layout

```
persona-framework/
├─ settings.gradle.kts
├─ build.gradle.kts                 # root (conventions + BOMs)
├─ persona-core/                    # domain model + ports (no Spring)
├─ persona-persistence/             # Spring Data JPA adapters (PostgreSQL)
├─ persona-ai-api/                  # AI orchestration ports + DTOs
├─ persona-ai-aws/                  # AWS adapters (Bedrock, SageMaker, S3)
├─ persona-nl2sql/                  # optional natural language → SQL
├─ persona-autoconfigure/           # Spring Boot autoconfiguration
├─ admin-ui/                        # React app (source)
├─ admin-ui-resources/              # built static assets
└─ admin-ui-starter/                # tiny starter to mount the UI routes
```

**Dependency rule**

```
core  <-- persistence
core  <-- ai-api
ai-api <-- ai-aws
core  <-- nl2sql   (optional)
autoconfigure depends on: core, persistence, ai-api (+ ai-aws / nl2sql if present)
admin-ui(-resources/-starter) are optional at runtime
```

No module may introduce `core → persistence` (avoid build cycles).

---

## Requirements

* **Java 21+**
* **Gradle 8.13+**
* **PostgreSQL 13+** (no H2)
* AWS credentials (if you use AWS adapters)

---

## Getting Started

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
    username: persona
    password: persona
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

## Safety Pipeline (New)

**Supervisor Agent** sits between persona outputs and users, enforcing:

* **Deterministic verification** against manuals/spec DBs (no “trust the LLM”).
* **Guardrails** (e.g., forbid disabling safety systems; enforce spec ranges).
* **Evidence requirement** (every actionable claim must include provenance).
* **Confidence thresholds** (e.g., ≥0.95 for technician instructions).
* **Auditable outcomes** (append-only logs + usage metrics).

Minimal flow:

1. Persona builds prompt → model returns **structured claims** (JSON).
2. Verifier confirms each claim’s evidence against authoritative sources.
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

* All personas’ `CHAT` → **Bedrock Haiku** (@ `us-west-2`).
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
* **JSONB errors**: you’re not on Postgres—switch to Postgres (no H2).

---

## Roadmap

* Vector search helpers & S3 ingestion workers
* Streaming interfaces (server-sent tokens) across adapters
* Tool/Function calling DSL + validation schema
* Prompt bundle versioning via S3 manifests & A/B testing
* Multi-tenant partitioning (workspace/org)
* OpenTelemetry tracing + cost attribution
* Expanded guardrails (regex + semantic + policy graph)
* **High priority**: deeper WrenchML integration using Supervisor + evidence verification for technician workflows

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

## Appendix: ASCII Package/Dependency Diagram

```
[ persona-core ]  <-- domain, ports
      ^
      |
[ persona-persistence ] -- Spring Data JPA (PostgreSQL)
      ^
      |
[ persona-autoconfigure ] -- Boot configs/wiring
      ^
      |
[ persona-ai-api ] <---[ persona-ai-aws ] (Bedrock, SageMaker, S3)

[ persona-nl2sql ] (optional) --> plugs into ai-api + core ports

[ admin-ui ] (React) -> [ admin-ui-resources ] -> [ admin-ui-starter ] -> Spring Boot static route
```
