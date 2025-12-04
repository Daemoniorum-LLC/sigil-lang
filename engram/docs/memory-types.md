# Memory Types Specification

*Detailed specification of Engram's four memory systems*

---

## Overview

Engram implements four distinct memory systems, each serving a specific cognitive function:

| Memory | Cognitive Analog | Primary Purpose | Persistence |
|--------|------------------|-----------------|-------------|
| **Instant** | Working Memory | Current task context | Ephemeral |
| **Episodic** | Autobiographical | Experiences over time | Decaying |
| **Semantic** | Factual Knowledge | Facts and relationships | Persistent |
| **Procedural** | Skills & Habits | How to do things | Persistent |

These systems interact continuously:
- Instant memory draws from other systems based on current needs
- Episodic experiences consolidate into semantic facts and procedural skills
- Semantic knowledge informs episodic interpretation
- Procedural skills guide episodic action selection

---

## Instant Memory

### Purpose

Instant Memory manages what the agent is currently "thinking about"—the active context window. It is bounded by token capacity and subject to continuous decay and refresh.

Unlike other memory systems, Instant Memory is not about storage—it's about attention. What deserves to be in the limited context window right now?

### Characteristics

- **Bounded**: Hard limit based on token capacity
- **Ephemeral**: Does not persist across sessions
- **Prioritized**: Contents ranked by relevance
- **Decaying**: Unrefreshed items lose priority
- **Compressible**: Low-priority items can be summarized

### Data Model

```sigil
struct InstantMemory {
    // Configuration
    capacity: TokenCount,
    decay_rate: f64,
    eviction_threshold: f64,

    // Contents
    items: PriorityQueue<InstantItem>,
    token_count: TokenCount,

    // Statistics
    stats: InstantStats,
}

struct InstantItem {
    id: ItemId,
    content: Content,
    tokens: TokenCount,

    // Priority factors
    base_relevance: f64,        // Initial relevance when added
    current_priority: f64,       // After decay applied
    added_at: Instant,
    last_refreshed: Instant,

    // Source tracking
    source: MemorySource,
    source_id: Option<EngramId>,
}

enum Content {
    Text(String),
    Structured(Value),
    Reference(EngramRef),        // Pointer to full content elsewhere
    Summary(SummaryOf),          // Compressed representation
}

enum MemorySource {
    User,                        // From user input
    Episodic(EpisodeId),        // From episodic recall
    Semantic(NodeId),           // From semantic query
    Procedural(SkillId),        // From skill matching
    System,                      // System-generated
}
```

### Operations

#### Attend

Bring something into working memory.

```sigil
fn attend(content: Content, relevance: f64) -> Result<ItemId, AttendError> {
    let tokens = tokenize(content).count()

    // Check if we can fit it
    if tokens > self.capacity {
        return Err(AttendError::ContentTooLarge)
    }

    // Make room if needed
    while self.token_count + tokens > self.capacity {
        self.evict_or_compress_lowest()
    }

    let item = InstantItem {
        id: generate_id(),
        content,
        tokens,
        base_relevance: relevance,
        current_priority: relevance,
        added_at: now(),
        last_refreshed: now(),
        source: infer_source(content),
        source_id: extract_source_id(content),
    }

    self.items.push(item)
    self.token_count += tokens

    Ok(item.id)
}
```

#### Refresh

Boost priority of an item (it's still relevant).

```sigil
fn refresh(id: ItemId, boost: f64) -> Result<(), RefreshError> {
    let item = self.items.get_mut(id)?

    item.current_priority = f64::min(1.0, item.current_priority + boost)
    item.last_refreshed = now()

    self.items.reheapify()
    Ok(())
}
```

#### Tick

Apply decay, called periodically.

```sigil
fn tick() {
    for item in self.items.iter_mut() {
        // Exponential decay based on time since last refresh
        let elapsed = now() - item.last_refreshed
        let decay = (-self.decay_rate * elapsed.as_secs_f64()).exp()
        item.current_priority *= decay
    }

    // Evict items below threshold
    self.items.retain(|item| {
        if item.current_priority < self.eviction_threshold {
            self.token_count -= item.tokens
            false
        } else {
            true
        }
    })
}
```

#### Export

Generate context for LLM consumption.

```sigil
fn export(format: ExportFormat, budget: Option<TokenCount>) -> ExportedContext {
    let budget = budget.unwrap_or(self.capacity)

    let selected = self.items
        |σ↓{_.current_priority}
        |> take_while_budget(budget)

    match format {
        ExportFormat::Plain => {
            selected |τ{|item| item.content.to_string()} |> join("\n\n")
        }
        ExportFormat::Structured => {
            ExportedContext {
                items: selected |τ{|item| ExportedItem {
                    content: item.content.to_string(),
                    relevance: item.current_priority,
                    source: item.source,
                }},
                total_tokens: selected |τ{_.tokens} |Σ,
            }
        }
        ExportFormat::XML => {
            selected |τ{|item| format!(
                "<context relevance=\"{:.2}\" source=\"{}\">\n{}\n</context>",
                item.current_priority,
                item.source,
                item.content.to_string()
            )} |> join("\n")
        }
    }
}
```

### Compression Strategies

When capacity is exceeded, Instant Memory can compress rather than evict.

```sigil
enum CompressionStrategy {
    // Simply remove lowest priority
    EvictLowest,

    // Compress lowest priority item
    CompressLowest {
        target_ratio: f64,       // e.g., 0.5 = compress to 50%
    },

    // Summarize a batch of related low-priority items
    SummarizeBatch {
        batch_size: usize,
        target_tokens: TokenCount,
    },

    // Hierarchical: keep high-level, drop details
    Hierarchical {
        keep_depth: usize,
    },
}

fn compress_lowest(strategy: CompressionStrategy) {
    match strategy {
        CompressionStrategy::CompressLowest { target_ratio } => {
            let lowest = self.items.peek_min()?
            let summary = summarize(lowest.content, target_tokens: lowest.tokens * target_ratio)

            self.items.update_min(|item| {
                self.token_count -= item.tokens
                item.content = Content::Summary(summary)
                item.tokens = tokenize(summary).count()
                self.token_count += item.tokens
            })
        }

        CompressionStrategy::SummarizeBatch { batch_size, target_tokens } => {
            let batch = self.items.pop_n_min(batch_size)
            let combined_tokens = batch |τ{_.tokens} |Σ

            if combined_tokens > target_tokens {
                let summary = summarize_batch(batch, target_tokens)
                self.attend(Content::Summary(summary), relevance: batch.avg_priority())
            }
        }

        // ... other strategies
    }
}
```

---

## Episodic Memory

### Purpose

Episodic Memory stores experiences—specific events in time, with their context, outcomes, and causal relationships. It is the autobiographical record of what the agent has done and encountered.

Episodic memories decay unless reinforced, eventually consolidating into more abstract semantic knowledge or procedural skills.

### Characteristics

- **Time-indexed**: Every episode has temporal bounds
- **Contextual**: Episodes capture surrounding context
- **Outcome-tracked**: Success, failure, and partial results recorded
- **Decaying**: Unreinforced episodes fade over time
- **Consolidatable**: Important patterns extract to other memory systems

### Data Model

```sigil
struct EpisodicMemory {
    // Storage
    episodes: TemporalIndex<Episode>,

    // Indices
    by_context: VectorIndex,         // For similarity search
    by_outcome: HashMap<OutcomeType, Vec<EpisodeId>>,
    by_participant: HashMap<ParticipantId, Vec<EpisodeId>>,

    // Configuration
    config: EpisodicConfig,

    // Background processes
    consolidator: ConsolidationProcess,
}

struct Episode {
    id: EpisodeId,

    // Temporal bounds
    started_at: Instant,
    ended_at: Option<Instant>,       // None if ongoing

    // Context
    context: ContextSnapshot,
    embedding: Vector<f32>,          // For similarity search

    // Participants
    participants: Vec<Participant>,

    // Events
    events: Vec<Event>,

    // Outcome
    outcome: Outcome,
    outcome_embedding: Option<Vector<f32>>,

    // Evaluation
    valence: f64,                    // -1.0 (bad) to 1.0 (good)
    significance: f64,               // 0.0 to 1.0
    surprise: f64,                   // How unexpected was this?

    // Lifecycle
    strength: f64,                   // Current memory strength
    access_count: u32,
    last_accessed: Instant,
    consolidated: bool,

    // Links
    caused_by: Vec<EpisodeId>,
    leads_to: Vec<EpisodeId>,
    similar_to: Vec<(EpisodeId, f64)>,
}

struct ContextSnapshot {
    // What was the agent trying to do?
    goal: Option<String>,
    task: Option<TaskId>,

    // What was known at the time?
    active_beliefs: Vec<EngramId>,

    // What tools/capabilities were available?
    available_tools: Vec<ToolId>,

    // Environmental factors
    environment: HashMap<String, Value>,
}

struct Event {
    timestamp: Instant,
    sequence: u32,                   // Order within episode

    event_type: EventType,
    content: Value,

    actor: Option<ParticipantId>,
    target: Option<ParticipantId>,

    // Metadata
    duration: Option<Duration>,
    success: Option<bool>,
}

enum EventType {
    // Communication
    UserMessage { content: String },
    AgentResponse { content: String },
    SystemMessage { content: String },

    // Actions
    ToolCall { tool: ToolId, args: Value },
    ToolResult { tool: ToolId, result: Value, success: bool },

    // State changes
    StateChange { key: String, old: Value, new: Value },
    BeliefUpdate { engram: EngramId, change: BeliefChange },

    // Decisions
    Decision { options: Vec<String>, chosen: usize, reasoning: String },
    PlanCreated { steps: Vec<String> },
    PlanModified { modification: String },

    // Outcomes
    GoalAchieved { goal: String },
    GoalFailed { goal: String, reason: String },
    ErrorOccurred { error: String, recoverable: bool },

    // Custom
    Custom { type_name: String, data: Value },
}

enum Outcome {
    Success {
        result: Value,
        goal_alignment: f64,         // How well did result match goal?
    },
    Failure {
        error: String,
        recoverable: bool,
        learned: Option<String>,     // What can we learn from this?
    },
    Partial {
        completed: Vec<String>,
        remaining: Vec<String>,
        blockers: Vec<String>,
    },
    Abandoned {
        reason: String,
        progress: f64,
    },
    Ongoing,                          // Episode not yet complete
}

struct Participant {
    id: ParticipantId,
    role: ParticipantRole,
    identifier: String,              // Name, user ID, etc.
}

enum ParticipantRole {
    Self_,                           // This agent
    User,
    OtherAgent(AgentId),
    System,
    Tool(ToolId),
}
```

### Operations

#### Record Episode

```sigil
fn record(episode: Episode) -> EpisodeId {
    // Enrich episode
    let enriched = episode
        |> self.compute_embedding
        |> self.extract_significance
        |> self.link_to_similar
        |> self.infer_causality

    // Store
    self.episodes.insert(enriched.started_at, enriched)

    // Update indices
    self.by_context.insert(enriched.id, enriched.embedding)

    if let Some(outcome_type) = enriched.outcome.type_() {
        self.by_outcome.entry(outcome_type).or_default().push(enriched.id)
    }

    for participant in &enriched.participants {
        self.by_participant.entry(participant.id).or_default().push(enriched.id)
    }

    enriched.id
}

fn compute_embedding(episode: &mut Episode) {
    // Embed the concatenation of context + key events + outcome
    let text = format!(
        "Context: {}\nEvents: {}\nOutcome: {}",
        episode.context.summarize(),
        episode.events.key_events().summarize(),
        episode.outcome.summarize()
    )
    episode.embedding = embed(text)
}

fn extract_significance(episode: &mut Episode) {
    // Significance based on:
    // - Outcome valence (extreme outcomes are significant)
    // - Surprise (unexpected outcomes are significant)
    // - Goal relevance (achieving/failing goals is significant)
    // - Novelty (new situations are significant)

    let outcome_significance = episode.outcome.significance()
    let surprise = self.compute_surprise(episode)
    let goal_relevance = episode.context.goal.map(|_| 0.3).unwrap_or(0.0)
    let novelty = 1.0 - self.max_similarity(episode)

    episode.significance = weighted_average([
        (outcome_significance, 0.4),
        (surprise, 0.3),
        (goal_relevance, 0.2),
        (novelty, 0.1),
    ])

    episode.surprise = surprise
}
```

#### Recall by Similarity

```sigil
fn recall_similar(query: &Context, limit: usize) -> Vec<Episode> {
    let query_embedding = embed(query.summarize())

    self.by_context.search(query_embedding, k: limit * 2)
        |τ{|id| self.episodes.get(id)}
        |φ{_.strength > 0.1}         // Filter faded memories
        |ω{limit}
}
```

#### Recall by Time

```sigil
fn recall_at(time: Instant, window: Duration) -> Vec<Episode> {
    self.episodes.range(time - window, time + window)
        |φ{_.strength > 0.1}
        |σ↓{_.significance}
}

fn recall_before(time: Instant, limit: usize) -> Vec<Episode> {
    self.episodes.before(time)
        |φ{_.strength > 0.1}
        |σ↓{_.significance}
        |ω{limit}
}

fn recall_sequence(start: EpisodeId, direction: Direction) -> Vec<Episode> {
    let mut current = self.episodes.get(start)
    let mut sequence = vec![current]

    loop {
        let next = match direction {
            Direction::Forward => current.leads_to.first(),
            Direction::Backward => current.caused_by.first(),
        }

        match next {
            Some(id) => {
                current = self.episodes.get(id)
                sequence.push(current)
            }
            None => break
        }
    }

    sequence
}
```

#### Decay and Consolidation

```sigil
fn apply_decay() {
    for episode in self.episodes.iter_mut() {
        let elapsed = now() - episode.last_accessed

        // Decay function: strength decreases over time
        // Modified by significance (important memories decay slower)
        let decay_rate = self.config.base_decay_rate * (1.0 - episode.significance * 0.5)
        let decay = (-decay_rate * elapsed.as_hours()).exp()

        episode.strength *= decay
    }
}

fn consolidate() -> ConsolidationReport {
    let mut report = ConsolidationReport::new()

    // Find episodes ready for consolidation
    let candidates = self.episodes
        |φ{!_.consolidated}
        |φ{_.strength < self.config.consolidation_threshold}
        |φ{_.significance > 0.3}     // Only consolidate meaningful episodes

    // Cluster similar episodes
    let clusters = cluster_by_similarity(candidates, threshold: 0.75)

    for cluster in clusters {
        if cluster.len() >= 3 {
            // Extract pattern for procedural memory
            if let Some(skill) = extract_skill_pattern(cluster) {
                procedural_memory.learn(skill)
                report.skills_learned.push(skill.id)
            }

            // Extract facts for semantic memory
            let facts = extract_semantic_facts(cluster)
            for fact in facts {
                semantic_memory.learn(fact)
                report.facts_learned.push(fact.id)
            }
        }

        // Mark as consolidated
        for episode in cluster {
            episode.consolidated = true
        }
    }

    report
}
```

---

## Semantic Memory

### Purpose

Semantic Memory holds factual knowledge—what the agent knows about the world. It is structured as a knowledge graph with vector embeddings, supporting both relational queries and semantic similarity search.

Unlike episodic memory, semantic memory is not about specific experiences but about general knowledge extracted from those experiences or learned directly.

### Characteristics

- **Graph-structured**: Entities and relationships
- **Vector-indexed**: Semantic similarity search
- **Belief-tracked**: Conflicts and uncertainty managed
- **Persistent**: Does not decay (but can be revised)
- **Inferential**: Supports reasoning over relationships

### Data Model

```sigil
struct SemanticMemory {
    // Knowledge graph
    graph: KnowledgeGraph,

    // Vector index for semantic search
    vectors: VectorIndex,

    // Belief management
    beliefs: BeliefTracker,

    // Configuration
    config: SemanticConfig,
}

struct KnowledgeGraph {
    nodes: HashMap<NodeId, Node>,
    edges: HashMap<EdgeId, Edge>,

    // Indices
    by_type: MultiMap<NodeType, NodeId>,
    by_label: TrieIndex<NodeId>,
    adjacency: AdjacencyIndex,
}

struct Node {
    id: NodeId,

    // Identity
    node_type: NodeType,
    label: String,
    aliases: Vec<String>,

    // Content
    properties: HashMap<String, Property>,
    embedding: Vector<f32>,

    // Epistemic
    epistemic: Epistemic,
    confidence: f64,
    sources: Vec<SourceRef>,

    // Temporal validity
    valid_from: Option<Instant>,
    valid_until: Option<Instant>,

    // Metadata
    created_at: Instant,
    updated_at: Instant,
    access_count: u32,
}

struct Property {
    value: Value,
    epistemic: Epistemic,
    confidence: f64,
    source: Option<SourceRef>,
}

enum NodeType {
    // Concrete entities
    Person,
    Organization,
    Place,
    Event,
    Document,
    Tool,

    // Abstract concepts
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

struct Edge {
    id: EdgeId,
    source: NodeId,
    target: NodeId,

    // Relationship
    relation: Relation,
    properties: HashMap<String, Property>,
    weight: f64,                     // Relationship strength

    // Epistemic
    epistemic: Epistemic,
    confidence: f64,
    sources: Vec<SourceRef>,

    // Temporal validity
    valid_from: Option<Instant>,
    valid_until: Option<Instant>,
}

enum Relation {
    // Taxonomic
    IsA,                             // Cat IsA Animal
    InstanceOf,                      // Fluffy InstanceOf Cat
    SubclassOf,                      // Mammal SubclassOf Animal

    // Compositional
    HasPart,                         // Car HasPart Engine
    PartOf,                          // Engine PartOf Car
    Contains,                        // Box Contains Items

    // Associative
    RelatedTo,                       // Coffee RelatedTo Morning
    SimilarTo,                       // Cat SimilarTo Dog
    OppositeOf,                      // Hot OppositeOf Cold

    // Causal
    Causes,                          // Fire Causes Heat
    Enables,                         // Key Enables Door
    Prevents,                        // Lock Prevents Entry

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

### Operations

#### Learn

Add knowledge to semantic memory.

```sigil
fn learn(fact: Fact) {
    // Get or create subject node
    let subject = self.get_or_create_node(
        type_: fact.subject_type,
        label: fact.subject,
    )

    for claim in fact.claims {
        // Get or create object node
        let object = self.get_or_create_node(
            type_: claim.object_type,
            label: claim.object,
        )

        // Check for existing beliefs
        let existing = self.graph.edges_between(subject.id, object.id)
            |φ{_.relation == claim.relation}

        if existing.is_empty() {
            // New knowledge - add directly
            self.graph.add_edge(Edge {
                source: subject.id,
                target: object.id,
                relation: claim.relation,
                epistemic: fact.epistemic,
                confidence: fact.confidence,
                sources: fact.sources.clone(),
                ..Default::default()
            })
        } else {
            // Existing knowledge - handle potential conflict
            self.beliefs.reconcile(existing, claim, fact.epistemic)
        }
    }

    // Update vector index
    self.vectors.upsert(subject.id, subject.embedding)
    self.vectors.upsert(object.id, object.embedding)
}

fn get_or_create_node(type_: NodeType, label: &str) -> &mut Node {
    // Try to find existing node
    if let Some(id) = self.graph.by_label.get(label) {
        return self.graph.nodes.get_mut(id)
    }

    // Check aliases
    for (id, node) in &self.graph.nodes {
        if node.aliases.contains(&label.to_string()) {
            return self.graph.nodes.get_mut(id)
        }
    }

    // Create new node
    let node = Node {
        id: generate_id(),
        node_type: type_,
        label: label.to_string(),
        embedding: embed(label),
        epistemic: Epistemic::Unknown,
        confidence: 0.0,
        ..Default::default()
    }

    let id = node.id
    self.graph.nodes.insert(id, node)
    self.graph.by_label.insert(label, id)
    self.graph.by_type.insert(type_, id)

    self.graph.nodes.get_mut(id)
}
```

#### Query by Meaning

```sigil
fn query(q: &str, limit: usize) -> Vec<QueryResult> {
    let embedding = embed(q)

    // Vector search for initial candidates
    let candidates = self.vectors.search(embedding, k: limit * 3)

    // Expand through graph to find related knowledge
    let expanded = candidates
        |> flat_map(|id| self.expand_neighborhood(id, hops: 1))
        |> dedupe

    // Score results
    expanded
        |τ{|id| {
            let node = self.graph.nodes.get(id)
            QueryResult {
                node_id: id,
                node: node.clone(),
                relevance: cosine_similarity(node.embedding, embedding),
                confidence: node.confidence,
                epistemic: node.epistemic,
                path: None,
            }
        }}
        |φ{_.confidence > 0.1 || _.epistemic.is_uncertain()}
        |σ↓{_.relevance * _.confidence.max(0.5)}
        |ω{limit}
}
```

#### Graph Traversal

```sigil
fn traverse(start: NodeId, path: GraphPath) -> Vec<TraversalResult> {
    let mut current = vec![(start, vec![])]

    for step in path.steps {
        current = current
            |> flat_map(|(node, path)| {
                self.graph.adjacency.follow(node, step.relation)
                    |φ{step.filter}
                    |τ{|next| (next, path.clone().push(step))}
            })
    }

    current |τ{|(node, path)| TraversalResult {
        node: self.graph.nodes.get(node),
        path,
    }}
}

// Example: Find who created tools used by the current user
let creators = semantic.traverse(
    start: user_node,
    path: GraphPath::new()
        .follow(Relation::UsedBy, direction: Incoming)  // Tools used by user
        .follow(Relation::CreatedBy, direction: Outgoing) // Who created them
)
```

#### Belief Management

```sigil
struct BeliefTracker {
    beliefs: HashMap<BeliefKey, BeliefState>,
    revision_history: Vec<BeliefRevision>,
}

// Key uniquely identifies a belief
struct BeliefKey {
    subject: NodeId,
    relation: Relation,
    object: NodeId,
}

enum BeliefState {
    // Single belief held with confidence
    Held {
        edge_id: EdgeId,
        confidence: f64,
        since: Instant,
    },

    // Multiple conflicting beliefs
    Contested {
        positions: Vec<ContestedPosition>,
        noted_at: Instant,
    },

    // Previously held, now withdrawn
    Retracted {
        original_edge: EdgeId,
        reason: String,
        retracted_at: Instant,
    },
}

struct ContestedPosition {
    edge_id: EdgeId,
    confidence: f64,
    supporting_sources: Vec<SourceRef>,
}

impl BeliefTracker {
    fn reconcile(
        existing: Vec<Edge>,
        new_claim: Claim,
        new_epistemic: Epistemic
    ) {
        let key = BeliefKey {
            subject: new_claim.subject,
            relation: new_claim.relation,
            object: new_claim.object,
        }

        // Check if new claim contradicts existing
        for edge in &existing {
            if self.contradicts(edge, new_claim) {
                self.handle_contradiction(edge, new_claim, new_epistemic)
                return
            }
        }

        // No contradiction - strengthen or add
        if let Some(edge) = existing.first() {
            self.strengthen(edge, new_claim, new_epistemic)
        } else {
            self.add_new(new_claim, new_epistemic)
        }
    }

    fn handle_contradiction(
        existing: &Edge,
        new_claim: Claim,
        new_epistemic: Epistemic
    ) {
        let existing_strength = self.epistemic_strength(existing.epistemic)
        let new_strength = self.epistemic_strength(new_epistemic)

        match (existing_strength, new_strength) {
            // New is significantly stronger - replace
            (old, new) if new > old + 0.3 => {
                self.retract(existing, reason: "superseded by stronger evidence")
                self.add_new(new_claim, new_epistemic)
            }

            // Old is significantly stronger - keep, note alternative
            (old, new) if old > new + 0.3 => {
                // Log but don't change
                self.note_rejected(new_claim, reason: "weaker than existing")
            }

            // Similar strength - mark as contested
            _ => {
                self.mark_contested(existing, new_claim, new_epistemic)
            }
        }
    }

    fn epistemic_strength(e: Epistemic) -> f64 {
        match e {
            Epistemic::Axiomatic => 1.0,
            Epistemic::Observed { .. } => 0.95,
            Epistemic::Computed { .. } => 0.9,
            Epistemic::Consensus { agreement_level, .. } => 0.7 + agreement_level * 0.2,
            Epistemic::Reported { trust_level, .. } => 0.4 + trust_level * 0.3,
            Epistemic::Inferred { confidence, .. } => 0.3 + confidence * 0.4,
            Epistemic::Hypothetical { .. } => 0.2,
            Epistemic::Hearsay { .. } => 0.1,
            Epistemic::Unknown => 0.0,
            _ => 0.5,
        }
    }
}
```

---

## Procedural Memory

### Purpose

Procedural Memory stores skills—patterns of action that have worked before. When the agent encounters a situation matching a known pattern, procedural memory suggests relevant approaches.

Skills are learned from successful (and failed) episodic experiences and refined through ongoing feedback.

### Characteristics

- **Pattern-triggered**: Skills activate on situation match
- **Success-weighted**: More successful skills rank higher
- **Refineable**: Skills improve with feedback
- **Failure-aware**: Failure modes tracked for avoidance
- **Transferable**: Skills can apply to novel but similar situations

### Data Model

```sigil
struct ProceduralMemory {
    skills: HashMap<SkillId, Skill>,
    patterns: PatternIndex,

    config: ProceduralConfig,
}

struct Skill {
    id: SkillId,
    name: String,
    description: String,

    // When to apply
    trigger: TriggerPattern,
    preconditions: Vec<Precondition>,

    // What to do
    procedure: Procedure,

    // Performance tracking
    stats: SkillStats,

    // Learning
    refinements: Vec<Refinement>,
    failure_modes: Vec<FailureMode>,

    // Provenance
    source_episodes: Vec<EpisodeId>,
    created_at: Instant,
    last_used: Option<Instant>,
}

struct TriggerPattern {
    // Feature vector for matching
    features: Vec<Feature>,
    weights: Vec<f64>,

    // Minimum similarity to trigger
    threshold: f64,

    // Optional semantic description for embedding match
    description: Option<String>,
    embedding: Option<Vector<f32>>,
}

struct Feature {
    name: String,
    extractor: FeatureExtractor,
    importance: f64,
}

enum FeatureExtractor {
    // Check for keyword presence
    KeywordPresent(Vec<String>),

    // Check entity type
    EntityType(NodeType),

    // Check context property
    ContextProperty { key: String, expected: Value },

    // Check goal pattern
    GoalMatches(String),

    // Semantic similarity to description
    SemanticSimilarity { description: String, threshold: f64 },

    // Custom extractor
    Custom(fn(&Context) -> f64),
}

struct Precondition {
    description: String,
    check: fn(&Context) -> bool,
    error_if_failed: String,
}

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

enum Action {
    // Tool invocation
    CallTool { tool: ToolId, args_template: Value },

    // Communication
    AskUser { question: String },
    Respond { template: String },

    // Memory operations
    Recall { query: String },
    Learn { fact_template: String },

    // Control flow
    If { condition: Condition, then: Box<Action>, else_: Option<Box<Action>> },
    Loop { condition: Condition, body: Box<Action>, max_iterations: usize },
    Sequence(Vec<Action>),

    // Sub-skill
    InvokeSkill { skill_id: SkillId, context_transform: Option<fn(Context) -> Context> },

    // Custom
    Custom { name: String, handler: fn(&Context) -> ActionResult },
}

enum FailureHandler {
    Retry { max_attempts: usize, backoff: Duration },
    Fallback(Box<Action>),
    Skip,
    Abort { error: String },
    AskUser { question: String },
}

struct SkillStats {
    execution_count: u32,
    success_count: u32,
    failure_count: u32,
    partial_count: u32,

    success_rate: f64,              // Computed: success / total
    avg_duration: Duration,
    last_outcome: Option<Outcome>,
}

struct Refinement {
    // When to apply this refinement
    condition: Condition,

    // How to modify the procedure
    modification: Modification,

    // Why this refinement exists
    reason: String,
    learned_from: EpisodeId,
    added_at: Instant,
}

enum Modification {
    // Add a step
    InsertStep { after: usize, step: ProcedureStep },

    // Remove a step
    RemoveStep { index: usize },

    // Modify a step
    ModifyStep { index: usize, new_step: ProcedureStep },

    // Add precondition
    AddPrecondition(Precondition),

    // Modify trigger
    AdjustTrigger { feature: String, adjustment: f64 },
}

struct FailureMode {
    // Pattern that triggers this failure
    trigger: TriggerPattern,

    // What goes wrong
    failure_type: String,
    description: String,

    // How to recover (if known)
    recovery: Option<RecoveryStrategy>,

    // Statistics
    occurrences: u32,
    last_occurrence: Instant,
}

enum RecoveryStrategy {
    Retry { with_modifications: Vec<Modification> },
    Alternative { skill_id: SkillId },
    AskUser { context: String },
    Abort { gracefully: bool },
}
```

### Operations

#### Learn from Episodes

```sigil
fn learn_from_episodes(episodes: Vec<Episode>) -> Option<Skill> {
    // Need multiple similar successful episodes to extract a skill
    let successful = episodes |φ{_.outcome.is_success()}

    if successful.len() < 3 {
        return None
    }

    // Extract common trigger pattern
    let trigger = self.extract_trigger_pattern(successful)?

    // Extract common procedure
    let procedure = self.extract_procedure(successful)?

    // Compute initial stats
    let success_rate = successful.len() as f64 / episodes.len() as f64

    let skill = Skill {
        id: generate_id(),
        name: self.generate_skill_name(trigger, procedure),
        description: self.generate_description(episodes),
        trigger,
        preconditions: self.extract_preconditions(episodes),
        procedure,
        stats: SkillStats {
            execution_count: episodes.len() as u32,
            success_count: successful.len() as u32,
            failure_count: (episodes.len() - successful.len()) as u32,
            success_rate,
            ..Default::default()
        },
        refinements: vec![],
        failure_modes: self.extract_failure_modes(episodes),
        source_episodes: episodes |τ{_.id},
        created_at: now(),
        last_used: None,
    }

    // Register in pattern index
    self.patterns.register(skill.trigger.clone(), skill.id)
    self.skills.insert(skill.id, skill.clone())

    Some(skill)
}

fn extract_trigger_pattern(episodes: Vec<Episode>) -> Option<TriggerPattern> {
    // Find common features across episode contexts
    let all_features = episodes
        |τ{|e| extract_features(e.context)}

    let common_features = find_common_features(all_features, threshold: 0.7)

    if common_features.is_empty() {
        return None
    }

    // Weight features by discriminative power
    let weighted = common_features
        |τ{|f| {
            let importance = compute_importance(f, episodes)
            (f, importance)
        }}
        |φ{_.1 > 0.1}

    Some(TriggerPattern {
        features: weighted |τ{_.0},
        weights: weighted |τ{_.1},
        threshold: 0.6,
        description: Some(summarize_pattern(weighted)),
        embedding: Some(embed(summarize_pattern(weighted))),
    })
}

fn extract_procedure(episodes: Vec<Episode>) -> Option<Procedure> {
    // Align event sequences
    let sequences = episodes |τ{|e| e.events |φ{is_action_event}}

    // Find common subsequence
    let common = longest_common_subsequence(sequences)

    if common.len() < 2 {
        return None
    }

    // Abstract events into procedure steps
    let steps = common
        |τ{|event| abstract_to_step(event)}

    Some(Procedure {
        steps,
        estimated_duration: episodes |τ{_.duration()} |> average,
        resource_requirements: extract_resources(common),
    })
}
```

#### Match Situation

```sigil
fn match_situation(context: &Context) -> Vec<SkillMatch> {
    // Extract features from current context
    let features = extract_features(context)

    // Find matching skills via pattern index
    let candidates = self.patterns.match(features)

    // Score and filter
    candidates
        |τ{|skill_id| {
            let skill = self.skills.get(skill_id)
            let match_score = self.compute_match_score(skill, features)
            let applicability = self.check_preconditions(skill, context)

            SkillMatch {
                skill: skill.clone(),
                match_score,
                preconditions_met: applicability.all_met,
                failed_preconditions: applicability.failed,
            }
        }}
        |φ{_.match_score > _.skill.trigger.threshold}
        |φ{_.preconditions_met || _.skill.stats.success_rate > 0.8}
        |σ↓{_.match_score * _.skill.stats.success_rate}
}
```

#### Record Feedback

```sigil
fn record_feedback(skill_id: SkillId, episode: Episode) {
    let skill = self.skills.get_mut(skill_id)

    skill.stats.execution_count += 1
    skill.last_used = Some(now())

    match &episode.outcome {
        Outcome::Success { .. } => {
            skill.stats.success_count += 1
            skill.stats.success_rate = ema(skill.stats.success_rate, 1.0, α: 0.1)
        }

        Outcome::Failure { error, .. } => {
            skill.stats.failure_count += 1
            skill.stats.success_rate = ema(skill.stats.success_rate, 0.0, α: 0.1)

            // Record failure mode
            let failure_pattern = extract_failure_pattern(episode)
            if let Some(existing) = skill.failure_modes
                |> find(|fm| fm.trigger.matches(failure_pattern))
            {
                existing.occurrences += 1
                existing.last_occurrence = now()
            } else {
                skill.failure_modes.push(FailureMode {
                    trigger: failure_pattern,
                    failure_type: categorize_failure(error),
                    description: error.clone(),
                    recovery: infer_recovery(episode),
                    occurrences: 1,
                    last_occurrence: now(),
                })
            }

            // Consider refinement
            if let Some(refinement) = self.infer_refinement(skill, episode) {
                skill.refinements.push(refinement)
            }
        }

        Outcome::Partial { completed, remaining, .. } => {
            skill.stats.partial_count += 1
            let partial_score = completed.len() as f64 /
                (completed.len() + remaining.len()) as f64
            skill.stats.success_rate = ema(skill.stats.success_rate, partial_score, α: 0.1)
        }

        _ => {}
    }

    skill.stats.avg_duration = ema_duration(
        skill.stats.avg_duration,
        episode.duration(),
        α: 0.1
    )
}
```

---

## Memory Interaction

The four memory systems interact continuously:

### Episodic → Semantic

When episodes consolidate, facts are extracted:

```sigil
fn consolidate_to_semantic(episodes: Vec<Episode>) {
    for episode in episodes {
        // Extract entities mentioned
        let entities = extract_entities(episode)

        // Extract relationships observed
        let relationships = extract_relationships(episode)

        // Learn as semantic facts
        for (entity, properties) in entities {
            semantic_memory.learn(Fact {
                subject: entity,
                claims: properties |τ{|p| Claim::HasProperty(p)},
                epistemic: Epistemic::Observed {
                    observer: self.agent_id,
                    timestamp: episode.started_at,
                },
                confidence: episode.significance,
            })
        }

        for (subject, relation, object) in relationships {
            semantic_memory.learn(Fact {
                subject,
                claims: vec![Claim::new(relation, object)],
                epistemic: Epistemic::Observed {
                    observer: self.agent_id,
                    timestamp: episode.started_at,
                },
                confidence: episode.significance,
            })
        }
    }
}
```

### Episodic → Procedural

When patterns emerge, skills are extracted:

```sigil
fn consolidate_to_procedural(episodes: Vec<Episode>) {
    // Group by similar context
    let clusters = cluster_by_context_similarity(episodes)

    for cluster in clusters {
        if cluster.len() >= 3 {
            if let Some(skill) = procedural_memory.learn_from_episodes(cluster) {
                log("Learned skill: {}", skill.name)
            }
        }
    }
}
```

### Semantic → Instant

When recalling, semantic knowledge enters working memory:

```sigil
fn recall_to_instant(query: &str) {
    let results = semantic_memory.query(query, limit: 10)

    for result in results {
        instant_memory.attend(
            Content::Reference(result.node.to_ref()),
            relevance: result.relevance * result.confidence
        )
    }
}
```

### Procedural → Instant

When skills match, they enter working memory:

```sigil
fn suggest_skills(context: &Context) {
    let matches = procedural_memory.match_situation(context)

    for match_ in matches.take(3) {
        instant_memory.attend(
            Content::Structured(json!({
                "type": "skill_suggestion",
                "skill": match_.skill.name,
                "description": match_.skill.description,
                "confidence": match_.match_score * match_.skill.stats.success_rate,
            })),
            relevance: match_.match_score
        )
    }
}
```

---

*This specification describes the intended behavior of each memory system. Implementation details may vary.*
