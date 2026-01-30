# Engram API Reference

*Complete API documentation for Engram*

---

## Module Structure

```
engram
├── Engram              # Main entry point
├── instant             # Instant memory module
├── episodic            # Episodic memory module
├── semantic            # Semantic memory module
├── procedural          # Procedural memory module
├── query               # Anamnesis query engine
├── types               # Core type definitions
├── config              # Configuration types
└── sync                # Distribution and sync
```

---

## Engram (Main Interface)

The primary interface for interacting with the memory system.

### Constructor

```sigil
fn Engram::new(config: EngramConfig) -> Result<Engram, EngramError>!
```

Creates a new Engram instance with the specified configuration.

**Parameters:**
- `config`: Configuration options for all memory subsystems

**Returns:**
- `Result<Engram, EngramError>!`: The engram instance or an error

**Example:**
```sigil
use engram::{Engram, EngramConfig}

let memory = Engram::new(EngramConfig::default())!
```

### Core Methods

#### recall

```sigil
fn recall(query: impl Into<Query>) -> RecallResult~
```

Primary recall interface - searches across all memory systems.

**Parameters:**
- `query`: A query string, Query struct, or Anamnesis expression

**Returns:**
- `RecallResult~`: Results with uncertainty metadata

**Example:**
```sigil
let result = memory.recall("project requirements")
let filtered = memory.recall(Query::new("errors")
    .where_(confidence > 0.8)
    .during(last_hour))
```

#### learn

```sigil
fn learn(fact: Fact) -> Result<EngramId, LearnError>!
```

Add knowledge to semantic memory.

**Parameters:**
- `fact`: The fact to learn, including epistemic metadata

**Returns:**
- `Result<EngramId, LearnError>!`: ID of created engram or error

**Example:**
```sigil
memory.learn(Fact {
    subject: "Sigil",
    claims: vec![
        (:is_a, "programming language"),
        (:designed_for, "AI agents"),
    ],
    epistemic: Observed!,
    confidence: 1.0,
    sources: vec![Source::Document("spec.md")],
})!
```

#### experience

```sigil
fn experience(episode: Episode) -> Result<EpisodeId, ExperienceError>!
```

Record an experience to episodic memory.

**Parameters:**
- `episode`: The episode to record

**Returns:**
- `Result<EpisodeId, ExperienceError>!`: ID of created episode or error

**Example:**
```sigil
memory.experience(Episode {
    context: current_context(),
    events: vec![
        Event::user_message("Help me debug this"),
        Event::agent_response("I'll analyze the error..."),
        Event::tool_call("read_file", args),
    ],
    outcome: Outcome::Success { result: "Bug fixed" },
    significance: 0.7,
})!
```

#### attend

```sigil
fn attend(content: impl Into<Content>, relevance: f64) -> Result<ItemId, AttendError>!
```

Bring something into instant (working) memory.

**Parameters:**
- `content`: Content to add to working memory
- `relevance`: Initial relevance score (0.0 - 1.0)

**Returns:**
- `Result<ItemId, AttendError>!`: ID of the item or error

**Example:**
```sigil
memory.attend("Important context for current task", relevance: 0.9)!
memory.attend(EngramRef::from(fact_id), relevance: 0.7)!
```

#### match_skills

```sigil
fn match_skills(context: &Context) -> Vec<SkillMatch>
```

Find applicable skills for the current situation.

**Parameters:**
- `context`: Current context to match against

**Returns:**
- `Vec<SkillMatch>`: Matching skills with scores

**Example:**
```sigil
let skills = memory.match_skills(&current_context())
for skill in skills.take(3) {
    print("Skill: {} ({}% success)", skill.name, skill.success_rate * 100)
}
```

#### build_context

```sigil
fn build_context(config: ContextConfig) -> ExportedContext
```

Build a context export suitable for LLM consumption.

**Parameters:**
- `config`: Configuration for context building

**Returns:**
- `ExportedContext`: Formatted context within token budget

**Example:**
```sigil
let ctx = memory.build_context(ContextConfig {
    budget: 8000 tokens,
    include: [Instant, SemanticRelevant, EpisodicRecent],
    format: ContextFormat::XML,
    prioritize: .relevance,
})
```

### Lifecycle Methods

#### consolidate

```sigil
fn consolidate() -> ConsolidationReport
```

Trigger memory consolidation (normally runs automatically).

**Returns:**
- `ConsolidationReport`: Summary of consolidation actions

#### archive

```sigil
fn archive(predicate: fn(&Engram) -> bool) -> ArchiveReport
```

Archive memories matching predicate to cold storage.

**Parameters:**
- `predicate`: Function to select memories for archival

**Returns:**
- `ArchiveReport`: Summary of archived items

#### sync

```sigil
fn sync(config: SyncConfig) -> Result<SyncReport, SyncError>
```

Synchronize with remote memory stores.

**Parameters:**
- `config`: Sync configuration (scope, strategy, etc.)

**Returns:**
- `Result<SyncReport, SyncError>`: Sync outcome

---

## Instant Memory API

### InstantMemory

```sigil
struct InstantMemory {
    capacity: TokenCount,
    decay_rate: f64,
    // ...
}
```

#### Methods

```sigil
// Add item to working memory
fn attend(content: Content, relevance: f64) -> Result<ItemId, AttendError>

// Refresh item priority
fn refresh(id: ItemId, boost: f64) -> Result<(), RefreshError>

// Remove item
fn dismiss(id: ItemId) -> Result<(), DismissError>

// Get all active items
fn active() -> Vec<&InstantItem>

// Get by ID
fn get(id: ItemId) -> Option<&InstantItem>

// Export for LLM
fn export(format: ExportFormat, budget: Option<TokenCount>) -> ExportedContext

// Apply decay tick
fn tick()

// Clear all
fn clear()

// Current token usage
fn token_usage() -> TokenUsage
```

### InstantItem

```sigil
struct InstantItem {
    id: ItemId,
    content: Content,
    tokens: TokenCount,
    base_relevance: f64,
    current_priority: f64,
    added_at: Instant,
    last_refreshed: Instant,
    source: MemorySource,
    source_id: Option<EngramId>,
}
```

### Content

```sigil
enum Content {
    Text(String),
    Structured(Value),
    Reference(EngramRef),
    Summary { original_id: EngramId, text: String },
}
```

---

## Episodic Memory API

### EpisodicMemory

```sigil
struct EpisodicMemory {
    config: EpisodicConfig,
    // ...
}
```

#### Methods

```sigil
// Record a new episode
fn record(episode: Episode) -> Result<EpisodeId, RecordError>

// Start an ongoing episode
fn begin_episode(context: Context) -> EpisodeBuilder

// Recall by similarity
fn recall_similar(query: &Context, limit: usize) -> Vec<Episode>

// Recall by time
fn recall_at(time: Instant, window: Duration) -> Vec<Episode>
fn recall_before(time: Instant, limit: usize) -> Vec<Episode>
fn recall_after(time: Instant, limit: usize) -> Vec<Episode>
fn recall_between(start: Instant, end: Instant) -> Vec<Episode>

// Get specific episode
fn get(id: EpisodeId) -> Option<&Episode>

// Update episode (for ongoing)
fn update(id: EpisodeId, update: EpisodeUpdate) -> Result<(), UpdateError>

// Complete ongoing episode
fn complete(id: EpisodeId, outcome: Outcome) -> Result<(), CompleteError>

// Apply decay
fn decay()

// Trigger consolidation
fn consolidate() -> ConsolidationReport

// Statistics
fn stats() -> EpisodicStats
```

### Episode

```sigil
struct Episode {
    id: EpisodeId,
    started_at: Instant,
    ended_at: Option<Instant>,
    context: ContextSnapshot,
    participants: Vec<Participant>,
    events: Vec<Event>,
    outcome: Outcome,
    valence: f64,
    significance: f64,
    strength: f64,
    access_count: u32,
    last_accessed: Instant,
    consolidated: bool,
    caused_by: Vec<EpisodeId>,
    leads_to: Vec<EpisodeId>,
}

impl Episode {
    fn duration() -> Option<Duration>
    fn is_ongoing() -> bool
    fn is_success() -> bool
    fn is_failure() -> bool
    fn key_events() -> Vec<&Event>
    fn embedding() -> Vector<f32>
    fn summarize() -> String
}
```

### EpisodeBuilder

For building episodes incrementally:

```sigil
struct EpisodeBuilder {
    // ...
}

impl EpisodeBuilder {
    fn with_context(context: Context) -> Self
    fn add_event(event: Event) -> Self
    fn add_participant(participant: Participant) -> Self
    fn set_goal(goal: String) -> Self
    fn complete(outcome: Outcome) -> Episode
    fn abandon(reason: String) -> Episode
}
```

### Event

```sigil
struct Event {
    timestamp: Instant,
    sequence: u32,
    event_type: EventType,
    content: Value,
    actor: Option<ParticipantId>,
    target: Option<ParticipantId>,
    duration: Option<Duration>,
    success: Option<bool>,
}

// Convenience constructors
impl Event {
    fn user_message(content: &str) -> Self
    fn agent_response(content: &str) -> Self
    fn tool_call(tool: &str, args: Value) -> Self
    fn tool_result(tool: &str, result: Value, success: bool) -> Self
    fn decision(options: Vec<&str>, chosen: usize, reasoning: &str) -> Self
    fn error(error: &str, recoverable: bool) -> Self
    fn state_change(key: &str, old: Value, new: Value) -> Self
}
```

### Outcome

```sigil
enum Outcome {
    Success { result: Value, goal_alignment: f64 },
    Failure { error: String, recoverable: bool, learned: Option<String> },
    Partial { completed: Vec<String>, remaining: Vec<String>, blockers: Vec<String> },
    Abandoned { reason: String, progress: f64 },
    Ongoing,
}

impl Outcome {
    fn is_success() -> bool
    fn is_failure() -> bool
    fn is_partial() -> bool
    fn is_ongoing() -> bool
    fn confidence() -> f64
    fn summarize() -> String
}
```

---

## Semantic Memory API

### SemanticMemory

```sigil
struct SemanticMemory {
    config: SemanticConfig,
    // ...
}
```

#### Methods

```sigil
// Add knowledge
fn learn(fact: Fact) -> Result<Vec<EngramId>, LearnError>

// Query by meaning
fn query(q: &str, limit: usize) -> Vec<QueryResult>

// Get specific node
fn get_node(id: NodeId) -> Option<&Node>

// Get specific edge
fn get_edge(id: EdgeId) -> Option<&Edge>

// Find or create entity
fn entity(label: &str) -> &mut Node

// Graph traversal
fn traverse(start: NodeId, path: GraphPath) -> Vec<TraversalResult>

// Get neighbors
fn neighbors(node: NodeId, relation: Option<Relation>) -> Vec<NodeId>

// Check belief status
fn belief_status(subject: NodeId, relation: Relation, object: NodeId) -> BeliefState

// Retract belief
fn retract(edge_id: EdgeId, reason: &str) -> Result<(), RetractError>

// Statistics
fn stats() -> SemanticStats
```

### Fact

```sigil
struct Fact {
    subject: String,
    subject_type: Option<NodeType>,
    claims: Vec<Claim>,
    epistemic: Epistemic,
    confidence: f64,
    sources: Vec<SourceRef>,
    valid_from: Option<Instant>,
    valid_until: Option<Instant>,
}

struct Claim {
    relation: Relation,
    object: String,
    object_type: Option<NodeType>,
    properties: HashMap<String, Value>,
}
```

### Node

```sigil
struct Node {
    id: NodeId,
    node_type: NodeType,
    label: String,
    aliases: Vec<String>,
    properties: HashMap<String, Property>,
    embedding: Vector<f32>,
    epistemic: Epistemic,
    confidence: f64,
    sources: Vec<SourceRef>,
    valid_from: Option<Instant>,
    valid_until: Option<Instant>,
    created_at: Instant,
    updated_at: Instant,
    access_count: u32,
}

impl Node {
    fn get_property(key: &str) -> Option<&Value>
    fn set_property(key: &str, value: Value, epistemic: Epistemic)
    fn add_alias(alias: &str)
    fn is_valid_at(time: Instant) -> bool
}
```

### QueryResult

```sigil
struct QueryResult {
    node_id: NodeId,
    node: Node,
    relevance: f64,
    confidence: f64,
    epistemic: Epistemic,
    path: Option<GraphPath>,
    matched_terms: Vec<String>,
}
```

### GraphPath

```sigil
struct GraphPath {
    steps: Vec<PathStep>,
}

struct PathStep {
    relation: Relation,
    direction: Direction,
    filter: Option<fn(&Node) -> bool>,
}

enum Direction {
    Outgoing,
    Incoming,
    Both,
}

impl GraphPath {
    fn new() -> Self
    fn follow(relation: Relation) -> Self
    fn follow_incoming(relation: Relation) -> Self
    fn follow_both(relation: Relation) -> Self
    fn where_(predicate: fn(&Node) -> bool) -> Self
}
```

---

## Procedural Memory API

### ProceduralMemory

```sigil
struct ProceduralMemory {
    config: ProceduralConfig,
    // ...
}
```

#### Methods

```sigil
// Learn skill from episodes
fn learn_from_episodes(episodes: Vec<Episode>) -> Option<Skill>

// Register skill directly
fn register(skill: Skill) -> SkillId

// Match situation to skills
fn match_situation(context: &Context) -> Vec<SkillMatch>

// Get specific skill
fn get(id: SkillId) -> Option<&Skill>

// Record execution feedback
fn feedback(skill_id: SkillId, episode: Episode)

// Add refinement
fn refine(skill_id: SkillId, refinement: Refinement) -> Result<(), RefineError>

// Disable skill
fn disable(skill_id: SkillId) -> Result<(), DisableError>

// Statistics
fn stats() -> ProceduralStats
```

### Skill

```sigil
struct Skill {
    id: SkillId,
    name: String,
    description: String,
    trigger: TriggerPattern,
    preconditions: Vec<Precondition>,
    procedure: Procedure,
    stats: SkillStats,
    refinements: Vec<Refinement>,
    failure_modes: Vec<FailureMode>,
    source_episodes: Vec<EpisodeId>,
    created_at: Instant,
    last_used: Option<Instant>,
    enabled: bool,
}

impl Skill {
    fn success_rate() -> f64
    fn is_applicable(context: &Context) -> bool
    fn execute(context: &Context) -> ExecutionResult
    fn with_refinements(context: &Context) -> Procedure
}
```

### SkillMatch

```sigil
struct SkillMatch {
    skill: Skill,
    match_score: f64,
    preconditions_met: bool,
    failed_preconditions: Vec<String>,
    suggested_adaptations: Vec<Adaptation>,
}
```

### Procedure

```sigil
struct Procedure {
    steps: Vec<ProcedureStep>,
    estimated_duration: Option<Duration>,
    resource_requirements: Vec<Resource>,
}

struct ProcedureStep {
    description: String,
    action: Action,
    expected_outcome: Option<Expectation>,
    on_failure: FailureHandler,
}

impl Procedure {
    fn step_count() -> usize
    fn estimated_duration() -> Duration
    fn requires_tools() -> Vec<ToolId>
    fn to_plan() -> Plan
}
```

---

## Query API

### Query

```sigil
struct Query {
    text: Option<String>,
    filters: Vec<Filter>,
    temporal: Option<TemporalConstraint>,
    limit: Option<usize>,
    offset: Option<usize>,
    include_gaps: bool,
    hints: Vec<QueryHint>,
}

impl Query {
    fn new(text: &str) -> Self
    fn semantic(text: &str) -> Self
    fn entity(label: &str) -> Self

    // Filters
    fn where_(filter: Filter) -> Self
    fn where_confidence(op: Comparator, value: f64) -> Self
    fn where_epistemic(check: EpistemicCheck) -> Self

    // Temporal
    fn during(range: TimeRange) -> Self
    fn before(time: TimePoint) -> Self
    fn after(time: TimePoint) -> Self
    fn at(time: TimePoint) -> Self

    // Traversal
    fn follow(relation: Relation) -> Self
    fn follow_incoming(relation: Relation) -> Self

    // Limits
    fn limit(n: usize) -> Self
    fn offset(n: usize) -> Self
    fn top(n: usize) -> Self

    // Options
    fn with_gap_analysis() -> Self
    fn hint(hint: QueryHint) -> Self

    // Execute
    fn execute(engram: &Engram) -> RecallResult
}
```

### RecallResult

```sigil
struct RecallResult {
    memories: Vec<Memory>,
    total_found: usize,
    returned: usize,
    confidence: ConfidenceDistribution,
    gaps: Vec<Gap>,
    query_metadata: QueryMetadata,
}

impl RecallResult {
    fn is_empty() -> bool
    fn has_gaps() -> bool
    fn high_confidence() -> Vec<&Memory>
    fn by_source(source: MemorySource) -> Vec<&Memory>
    fn summarize() -> String
}
```

### Memory

```sigil
struct Memory {
    id: EngramId,
    content: Value,
    relevance: f64,
    confidence: f64,
    epistemic: Epistemic,
    source: MemorySource,
    timestamp: Instant,
    access_path: Option<Vec<EngramId>>,
}
```

### Gap

```sigil
struct Gap {
    description: String,
    query_aspect: String,
    severity: f64,
    suggested_actions: Vec<SuggestedAction>,
}

enum SuggestedAction {
    AskUser { question: String },
    SearchExternal { query: String },
    InvokeTool { tool: ToolId, args: Value },
    RecallDifferently { alternative_query: Query },
}
```

---

## Types

### Epistemic

```sigil
enum Epistemic {
    Axiomatic,
    Observed { observer: AgentId, timestamp: Instant },
    Computed { derivation: DerivationChain },
    Reported { source: Source, trust_level: f64 },
    Inferred { premises: Vec<EngramId>, inference_type: InferenceType, confidence: f64 },
    Consensus { sources: Vec<Source>, agreement_level: f64 },
    Hypothetical { status: HypothesisStatus },
    Hearsay { chain: Vec<Source> },
    Contested { positions: Vec<(EngramId, f64)> },
    Unknown,
    Retracted { original: Box<Epistemic>, reason: String },
}

impl Epistemic {
    fn is_observed() -> bool
    fn is_reported() -> bool
    fn is_inferred() -> bool
    fn is_contested() -> bool
    fn is_certain() -> bool      // High confidence threshold
    fn is_uncertain() -> bool    // Low confidence threshold
    fn strength() -> f64         // Epistemic strength score
    fn source() -> Option<Source>
}
```

### Source

```sigil
enum Source {
    Agent(AgentId),
    User(UserId),
    Api { endpoint: String, timestamp: Instant },
    Document { path: String, section: Option<String> },
    Sensor(SensorId),
    Episode(EpisodeId),
    Inference { from: Vec<EngramId> },
    Unknown,
}
```

### Relation

```sigil
enum Relation {
    // Taxonomic
    IsA,
    InstanceOf,
    SubclassOf,

    // Compositional
    HasPart,
    PartOf,
    Contains,

    // Associative
    RelatedTo,
    SimilarTo,
    OppositeOf,

    // Causal
    Causes,
    Enables,
    Prevents,

    // Temporal
    Before,
    After,
    During,

    // Spatial
    LocatedIn,
    NearTo,

    // Agentive
    CreatedBy,
    OwnedBy,
    UsedBy,
    WorksFor,
    Knows,

    // Attributive
    HasProperty,
    HasState,

    // Custom
    Custom(String),
}
```

### NodeType

```sigil
enum NodeType {
    // Concrete
    Person,
    Organization,
    Place,
    Event,
    Document,
    Tool,

    // Abstract
    Concept,
    Category,
    Skill,
    Goal,

    // Agent-specific
    User,
    Agent,
    Task,
    Project,

    // Custom
    Custom(String),
}
```

### TimeRange / TimePoint

```sigil
enum TimeRange {
    LastHour,
    LastDay,
    LastWeek,
    LastMonth,
    Today,
    Yesterday,
    ThisWeek,
    ThisMonth,
    Between { start: Instant, end: Instant },
    Since(Instant),
    Until(Instant),
}

enum TimePoint {
    Now,
    Instant(Instant),
    Relative(Duration, Direction),
}
```

---

## Configuration

### EngramConfig

```sigil
struct EngramConfig {
    // Memory configurations
    instant: InstantConfig,
    episodic: EpisodicConfig,
    semantic: SemanticConfig,
    procedural: ProceduralConfig,

    // Index configuration
    vector: VectorConfig,

    // Storage configuration
    storage: StorageConfig,

    // Embedding configuration
    embedding: EmbeddingConfig,

    // Distribution configuration
    distribution: DistributionConfig,

    // Background processes
    consolidation: ConsolidationConfig,
}

impl EngramConfig {
    fn default() -> Self
    fn high_memory() -> Self      // Large context window agents
    fn low_memory() -> Self       // Constrained environments
    fn distributed() -> Self      // Multi-agent setup
    fn ephemeral() -> Self        // No persistence
}
```

### InstantConfig

```sigil
struct InstantConfig {
    capacity: TokenCount,           // Default: 8000
    decay_rate: f64,                // Default: 0.05
    eviction_threshold: f64,        // Default: 0.1
    compression_strategy: CompressionStrategy,
    token_counter: TokenCounterType,
}
```

### EpisodicConfig

```sigil
struct EpisodicConfig {
    decay_function: DecayFunction,  // Default: Exponential(λ=0.1)
    consolidation_threshold: f64,   // Default: 0.3
    consolidation_interval: Duration, // Default: 6h
    max_active_episodes: usize,     // Default: 10000
    min_significance_to_keep: f64,  // Default: 0.1
}
```

### SemanticConfig

```sigil
struct SemanticConfig {
    max_nodes: Option<usize>,
    max_edges_per_node: usize,      // Default: 100
    belief_conflict_strategy: ConflictStrategy,
    auto_link_threshold: f64,       // Default: 0.8
}
```

### ProceduralConfig

```sigil
struct ProceduralConfig {
    min_episodes_for_skill: usize,  // Default: 3
    skill_success_threshold: f64,   // Default: 0.5
    pattern_match_threshold: f64,   // Default: 0.6
    max_skills: Option<usize>,
}
```

---

## Errors

### EngramError

```sigil
enum EngramError {
    // Initialization
    ConfigInvalid { field: String, reason: String },
    StorageInitFailed { path: Path, cause: String },

    // Capacity
    CapacityExceeded { current: usize, max: usize },
    TokenBudgetExceeded { needed: TokenCount, available: TokenCount },

    // Not found
    EngramNotFound { id: EngramId },
    EpisodeNotFound { id: EpisodeId },
    NodeNotFound { id: NodeId },
    SkillNotFound { id: SkillId },

    // Conflicts
    BeliefConflict { existing: EngramId, new_claim: String },

    // Sync
    SyncFailed { reason: String },
    ConflictResolutionFailed { conflicts: Vec<Conflict> },

    // Query
    QueryParseError { query: String, error: String },
    QueryTimeout { query: String, elapsed: Duration },

    // Storage
    StorageCorrupted { details: String },
    StorageFull,

    // Internal
    Internal { message: String },
}
```

---

## Events

Engram emits events for observability:

```sigil
enum EngramEvent {
    // Memory changes
    EngramCreated { id: EngramId, source: MemorySource },
    EngramUpdated { id: EngramId, changes: Vec<Change> },
    EngramArchived { id: EngramId },
    EngramRetracted { id: EngramId, reason: String },

    // Episodes
    EpisodeStarted { id: EpisodeId },
    EpisodeCompleted { id: EpisodeId, outcome: Outcome },
    EpisodeConsolidated { id: EpisodeId },

    // Skills
    SkillLearned { id: SkillId, from_episodes: Vec<EpisodeId> },
    SkillRefined { id: SkillId, refinement: Refinement },
    SkillExecuted { id: SkillId, success: bool },

    // Beliefs
    BeliefConflictDetected { subject: NodeId, relation: Relation },
    BeliefResolved { subject: NodeId, resolution: Resolution },

    // Lifecycle
    ConsolidationStarted,
    ConsolidationCompleted { report: ConsolidationReport },
    SyncStarted { scope: Scope },
    SyncCompleted { report: SyncReport },

    // Capacity
    CapacityWarning { memory_type: MemoryType, usage: f64 },
    EvictionOccurred { count: usize, memory_type: MemoryType },
}

// Subscribe to events
memory.on_event(|event| {
    match event {
        EngramEvent::SkillLearned { id, .. } => {
            log("New skill learned: {}", id)
        }
        _ => {}
    }
})
```

---

*This API reference covers the primary interfaces. For advanced usage and internal APIs, see the source documentation.*
