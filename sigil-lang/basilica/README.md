# Basilica

**Evidentiality-native web framework for Sigil**

Basilica is not a port of Spring Boot or Express. It's a web framework designed from first principles to embrace Sigil's unique features:

- **Evidentiality flows through the request lifecycle** - Data enters as `~` (external), gets validated to `!` (known)
- **Daemons as services** - Long-lived processes with memory, learning, and heartbeat cycles
- **Covenant integration** - Human-in-the-loop patterns built into the request pipeline
- **Oracle by default** - Every decision is explainable, every response carries confidence
- **Morpheme query language** - τ, φ, σ, ρ compose into database queries

## Philosophy

> "Umuntu ngumuntu ngabantu" — I am because we are

Web applications are not isolated request handlers. They are participants in relationships:
- Between agents and humans (Covenant)
- Between the service and its memory (Engram)
- Between certainty and uncertainty (Evidentiality)
- Between action and explanation (Oracle)

Basilica makes these relationships explicit and type-safe.

## Core Concepts

### Evidence Lifecycle

Every HTTP request flows through an evidence transformation:

```
External World (untrusted)
         ↓
    Request body: ~T           // Reported - from outside
         ↓
    Validation gate
         ↓
    Validated data: !T         // Known - we verified it
         ↓
    Business logic
         ↓
    Response: T!{confidence}   // Known with confidence metadata
```

### The Nave

The `Nave` is the application container (named for the central hall of a basilica where the congregation gathers). It orchestrates:

- Route registration
- Middleware chains
- Service daemons
- Memory systems
- Covenant boundaries

```sigil
let app = Nave::new()
    .with_memory(EngramConfig::default())
    .with_covenant(Covenant::collaborative())
    .route(routes![
        brain_dumps::list,
        brain_dumps::create,
        brain_dumps::triage,
    ])
    .daemon(WellnessService::new())
    .listen("0.0.0.0:8080");
```

### Routes with Evidentiality

Routes explicitly declare evidence transformations:

```sigil
#[route(POST, "/api/brain-dumps")]
pub fn create_brain_dump(
    body: Json<CreateBrainDumpDto>~,  // External input
    ctx: &Context!,                    // Known context
) -> Response<BrainDump!> {
    // Validation gate: ~ → !
    let validated! = body.validate()?;

    // Business logic with known data
    let brain_dump! = ctx.service::<BrainDumpService>()
        .create(validated!)?;

    // Response carries evidence metadata
    Response::created(brain_dump!)
        .with_evidence(Evidence::Computed)
}
```

### Daemon Services

Services are not stateless beans. They are daemons with memory and lifecycle:

```sigil
daemon BrainDumpService {
    memory: Engram,
    covenant: Covenant,

    fn on_init(&mut self) {
        // Load semantic knowledge
        self.memory.load_semantic("wellness_patterns");
    }

    fn on_heartbeat(&mut self) {
        // Proactive: consolidate memories, cleanup, health checks
        self.memory.consolidate();
    }

    pub fn create(&mut self, dto: CreateBrainDumpDto!) -> Result<BrainDump!, Error> {
        // Check covenant boundaries
        self.covenant.check_action("brain_dump.create")?;

        // Create with full context
        let brain_dump! = BrainDump::from(dto);

        // Remember this experience
        self.memory.experience(Event::brain_dump_created(brain_dump.id));

        Ok(brain_dump!)
    }

    pub fn suggest_triage(&self, id: Uuid!) -> TriageSuggestion~ {
        // AI inference returns ~ (reported/inferred)
        let context! = self.memory.recall(
            Query::similar_to(&self.get(id)?.content)
        );

        // Inference produces ~ result with confidence
        TriageSuggestion::infer(context)~
    }
}
```

### Repository with Morphemes

Queries compose from morpheme operators, not SQL strings:

```sigil
repository BrainDumpRepository {
    table: "brain_dumps",

    pub fn find_by_instance(id: Uuid!) -> Vec<BrainDump>~ {
        brain_dumps
            |> φ{ .instance_id == id }     // WHERE
            |> σ{ .created_at.desc() }     // ORDER BY
    }

    pub fn find_urgent() -> Vec<BrainDump>~ {
        brain_dumps
            |> φ{ .triage_status == Crisis }
            |> φ{ .resolved_at.is_none() }
            |> σ{ .urgency.desc(), .created_at.desc() }
            |> ν{ 0..20 }                   // LIMIT
    }

    pub fn analytics(period: TimeRange!) -> Analytics~ {
        brain_dumps
            |> φ{ .created_at.within(period) }
            |> group_by{ .triage_status }
            |> τ{ (status, items) => StatusCount { status, count: items.len() } }
            |> ρ{ Analytics::from }
    }
}
```

The compiler generates parameterized SQL. Evidentiality prevents injection by construction.

### Covenant Middleware

Human-in-the-loop patterns are middleware, not afterthoughts:

```sigil
middleware CovenantMiddleware {
    fn before(req: Request~, ctx: &Context!) -> MiddlewareResult {
        let action = req.route_action();

        match ctx.covenant.check_action(action) {
            BoundaryCheck::Allowed => Continue,
            BoundaryCheck::AllowedWithReport => {
                ctx.covenant.inform(&format!("Executing: {}", action));
                Continue
            }
            BoundaryCheck::NeedsApproval => {
                let handoff = ctx.covenant.request_approval(action, req.explain());
                Suspend(handoff)  // Pause until human responds
            }
            BoundaryCheck::Forbidden { reason } => {
                Reject(Error::forbidden(reason))
            }
        }
    }
}
```

### Oracle Integration

Every response can explain itself:

```sigil
#[route(GET, "/api/brain-dumps/{id}/explain")]
pub fn explain_triage(
    id: Path<Uuid>!,
    level: Query<ExplanationLevel>?,
    ctx: &Context!,
) -> Response<Explanation!> {
    let brain_dump! = ctx.repo::<BrainDumpRepo>().find(id)?;
    let triage~ = ctx.service::<BrainDumpService>().suggest_triage(id)?;

    // Oracle explains the reasoning
    let explanation! = ctx.oracle.explain(
        &triage,
        level.unwrap_or(ExplanationLevel::Standard)
    );

    Response::ok(explanation!)
}
```

## Architecture

```
basilica/
├── src/
│   ├── lib.sg              # Main exports
│   ├── nave.sg             # Application container
│   ├── route.sg            # Routing with evidentiality
│   ├── request.sg          # Request types
│   ├── response.sg         # Response types
│   ├── middleware.sg       # Middleware chain
│   ├── validation.sg       # Validation gates (~ → !)
│   ├── service.sg          # Daemon service base
│   ├── repository.sg       # Data access layer
│   ├── query.sg            # Morpheme query builder
│   ├── evidence.sg         # Evidence tracking types
│   ├── context.sg          # Request context
│   └── error.sg            # Error handling
├── examples/
│   └── sanctum/            # Sanctum backend example
├── tests/
│   └── integration.sg
└── docs/
    ├── architecture.md
    ├── evidentiality.md
    └── daemon-services.md
```

## Evidence Types

Basilica extends Sigil's evidentiality with web-specific markers:

```sigil
/// Evidence level for HTTP data
pub enum HttpEvidence {
    /// Computed locally, verified
    Computed,

    /// From authenticated user (trust level applies)
    Authenticated { user_id: Uuid!, trust: TrustLevel },

    /// From external API call
    External { source: String, fetched_at: Timestamp },

    /// AI inference with confidence
    Inferred { confidence: Confidence, model: String },

    /// User input, validated
    UserInput { validated: bool },

    /// Cached value with staleness
    Cached { stored_at: Timestamp, ttl: Duration },
}
```

## Quick Start

```sigil
use basilica::prelude::*;

#[route(GET, "/")]
fn index() -> Response<String!> {
    Response::ok("Hello, Basilica!"!)
}

fn main() {
    Nave::new()
        .route(routes![index])
        .listen("0.0.0.0:8080")
        .run();
}
```

## See Also

- [Sigil Language](../README.md)
- [Daemon Runtime](../daemon/README.md)
- [Engram Memory](../engram/README.md)
- [Covenant Collaboration](../covenant/README.md)
- [Oracle Explainability](../oracle/README.md)
