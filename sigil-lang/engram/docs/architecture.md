# Engram Architecture

*Technical deep-dive into the memory system*

---

## System Overview

Engram is structured as a layered architecture, with each layer providing specific capabilities:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            APPLICATION LAYER                             │
│                                                                          │
│   Agent Integration │ MCP Tools │ Anamnesis REPL │ Sigil Native API    │
├─────────────────────────────────────────────────────────────────────────┤
│                             QUERY LAYER                                  │
│                                                                          │
│   Anamnesis Parser │ Query Planner │ Reconstruction Engine              │
├─────────────────────────────────────────────────────────────────────────┤
│                            MEMORY LAYER                                  │
│                                                                          │
│   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐       │
│   │  Instant   │  │  Episodic  │  │  Semantic  │  │ Procedural │       │
│   │   Memory   │  │   Memory   │  │   Memory   │  │   Memory   │       │
│   └────────────┘  └────────────┘  └────────────┘  └────────────┘       │
├─────────────────────────────────────────────────────────────────────────┤
│                             INDEX LAYER                                  │
│                                                                          │
│   Vector Index (HNSW) │ Temporal Index (B+Tree) │ Graph Index (Native) │
├─────────────────────────────────────────────────────────────────────────┤
│                            STORAGE LAYER                                 │
│                                                                          │
│   Hot Store (mmap) │ Warm Store (LSM) │ Cold Store (Archive)           │
├─────────────────────────────────────────────────────────────────────────┤
│                          DISTRIBUTION LAYER                              │
│                                                                          │
│   Instance Sync (CRDT) │ Agent Federation │ Collective Consensus (Raft)│
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Core Data Structures

### Engram Record

The fundamental unit of storage:

```sigil
struct EngramRecord<T> {
    // Identity
    id: EngramId,                    // Unique identifier (ULID for ordering)
    version: u64,                    // Version for conflict resolution

    // Content
    content: T,                      // The actual data
    embedding: Vector<f32>,          // Semantic embedding (dimension configurable)

    // Epistemic metadata
    epistemic: Epistemic,            // How we know this
    confidence: f64,                 // 0.0 - 1.0
    sources: Vec<SourceRef>,         // Provenance chain

    // Temporal metadata
    created_at: Instant,             // When record was created
    valid_from: Instant,             // When content became true
    valid_until: Option<Instant>,    // When content ceased to be true (if known)
    accessed_at: Instant,            // Last access time
    access_count: u32,               // Access frequency

    // Lifecycle metadata
    strength: f64,                   // Memory strength (for decay)
    scope: Scope,                    // Instance / Agent / Collective
    archived: bool,                  // In cold storage?

    // Relationships
    links: Vec<Link>,                // Connections to other engrams
}

struct Link {
    relation: Relation,              // Type of relationship
    target: EngramId,                // Target engram
    weight: f64,                     // Relationship strength
    bidirectional: bool,             // Also link from target?
}

enum Relation {
    // Causal
    CausedBy,
    Enables,

    // Temporal
    Before,
    After,
    During,

    // Semantic
    IsA,
    HasA,
    RelatedTo,
    ContradictsTo,
    Refines,

    // Episodic
    PartOf,                          // Part of same episode
    LeadsTo,                         // Narrative sequence

    // Custom
    Custom(String),
}
```

### Epistemic Types

```sigil
enum Epistemic {
    // High confidence
    Axiomatic,                       // True by definition (math, logic)
    Observed {                       // Directly witnessed
        observer: AgentId,
        timestamp: Instant,
    },
    Computed {                       // Calculated from axioms/observations
        derivation: DerivationChain,
    },

    // Medium confidence
    Reported {                       // From external source
        source: Source,
        trust_level: f64,
    },
    Inferred {                       // Reasoned from beliefs
        premises: Vec<EngramId>,
        inference_type: InferenceType,
        confidence: f64,
    },
    Consensus {                      // Agreed by multiple sources
        sources: Vec<Source>,
        agreement_level: f64,
    },

    // Low confidence
    Hypothetical {                   // Under consideration
        status: HypothesisStatus,
    },
    Hearsay {                        // Reported about reported
        chain: Vec<Source>,
    },

    // Special states
    Contested {                      // Multiple conflicting beliefs
        positions: Vec<(EngramId, f64)>,  // (belief, weight)
    },
    Unknown,                         // Explicitly marked as unknown
    Retracted {                      // Previously believed, now withdrawn
        original: Box<Epistemic>,
        reason: String,
    },
}

enum InferenceType {
    Deductive,                       // Logically certain
    Inductive,                       // Probabilistically likely
    Abductive,                       // Best explanation
    Analogical,                      // By similarity
}

enum Source {
    Agent(AgentId),
    User(UserId),
    Api(ApiEndpoint),
    Document(DocumentRef),
    Sensor(SensorId),
    Unknown,
}
```

---

## Memory Subsystems

### Instant Memory

Working memory for the current context. Bounded by token capacity.

```sigil
struct InstantMemory {
    capacity: TokenCount,
    contents: BinaryHeap<(Priority, EngramRef)>,
    token_counter: TokenCounter,
    decay_rate: f64,

    config: InstantConfig,
}

struct InstantConfig {
    capacity: TokenCount,            // Default: 8000 tokens
    decay_rate: f64,                 // Default: 0.05 per tick
    eviction_threshold: f64,         // Default: 0.1
    compression_strategy: CompressionStrategy,
}

impl InstantMemory {
    // Add to working memory
    fn attend(engram: EngramRef, relevance: f64) -> Result<(), CapacityError> {
        let tokens = self.token_counter.count(engram)

        while self.would_overflow(tokens) {
            match self.config.compression_strategy {
                CompressionStrategy::EvictLowest => self.evict_lowest(),
                CompressionStrategy::Compress => self.compress_lowest(),
                CompressionStrategy::Summarize => self.summarize_batch(),
            }
        }

        self.contents.push((relevance, engram))
    }

    // Decay over time
    fn tick() {
        for (priority, _) in self.contents.iter_mut() {
            *priority *= (1.0 - self.decay_rate)
        }
        self.evict_below(self.config.eviction_threshold)
    }

    // Get current context
    fn active() -> Vec<EngramRef> {
        self.contents
            |σ↓{_.0}           // Sort by priority descending
            |τ{_.1}            // Extract engram refs
    }

    // Export for LLM context
    fn export(format: ContextFormat) -> String {
        self.active()
            |τ{format.render}
            |> join("\n")
    }
}
```

### Episodic Memory

Time-indexed experiences with decay and consolidation.

```sigil
struct EpisodicMemory {
    episodes: TemporalIndex<Episode>,
    consolidator: ConsolidationProcess,

    config: EpisodicConfig,
}

struct Episode {
    id: EpisodeId,

    // Temporal bounds
    started_at: Instant,
    ended_at: Option<Instant>,

    // Context
    context: ContextSnapshot,
    participants: Vec<ParticipantRef>,

    // Content
    events: Vec<Event>,
    summary: Option<String>,

    // Outcome
    outcome: Outcome,
    valence: f64,                    // -1.0 (negative) to 1.0 (positive)
    significance: f64,               // 0.0 to 1.0

    // Lifecycle
    access_count: u32,
    last_accessed: Instant,
    strength: f64,
    consolidated: bool,
}

struct Event {
    timestamp: Instant,
    event_type: EventType,
    content: Value,
    actor: Option<ParticipantRef>,
}

enum EventType {
    UserMessage,
    AgentResponse,
    ToolCall,
    ToolResult,
    Error,
    StateChange,
    Decision,
    Custom(String),
}

enum Outcome {
    Success { result: Value },
    Failure { error: String, recoverable: bool },
    Partial { completed: Vec<String>, remaining: Vec<String> },
    Abandoned { reason: String },
    Ongoing,
}

struct EpisodicConfig {
    decay_function: DecayFunction,   // Default: Exponential(λ=0.1)
    consolidation_threshold: f64,    // Default: 0.3
    consolidation_interval: Duration,// Default: 6 hours
    max_active_episodes: usize,      // Default: 10000
}

impl EpisodicMemory {
    // Record new episode
    fn record(episode: Episode) {
        let enriched = episode
            |> self.extract_entities
            |> self.link_to_related
            |> self.compute_significance

        self.episodes.insert(enriched)
    }

    // Recall by similarity
    fn recall_similar(query: &Context, limit: usize) -> Vec<Episode> {
        let query_embedding = embed(query)

        self.episodes
            |σ{_.embedding()}~{query_embedding}
            |φ{_.strength > 0.1}
            |ω{limit}
    }

    // Recall by time
    fn recall_at(time: Instant, window: Duration) -> Vec<Episode> {
        self.episodes.range(time - window, time + window)
    }

    // Apply decay
    fn decay() {
        for episode in self.episodes.iter_mut() {
            episode.strength = self.config.decay_function.apply(
                episode.strength,
                since: episode.last_accessed
            )
        }
    }

    // Consolidation
    fn consolidate() -> ConsolidationReport {
        let to_consolidate = self.episodes
            |φ{!_.consolidated && _.strength < self.config.consolidation_threshold}

        let clusters = cluster_similar(to_consolidate, threshold: 0.8)

        for cluster in clusters {
            // Extract pattern
            let pattern = abstract_pattern(cluster)

            // Store as procedural skill if actionable
            if pattern.is_actionable() {
                procedural_memory.learn_skill(pattern)
            }

            // Store as semantic knowledge
            semantic_memory.learn(pattern.as_facts())

            // Mark as consolidated
            for episode in cluster {
                episode.consolidated = true
            }
        }
    }
}
```

### Semantic Memory

Knowledge graph with vector embeddings.

```sigil
struct SemanticMemory {
    graph: KnowledgeGraph,
    vectors: VectorIndex,
    beliefs: BeliefTracker,

    config: SemanticConfig,
}

struct KnowledgeGraph {
    nodes: HashMap<NodeId, Node>,
    edges: HashMap<EdgeId, Edge>,

    // Indices
    by_type: MultiMap<NodeType, NodeId>,
    by_label: TrieMap<String, NodeId>,
}

struct Node {
    id: NodeId,
    node_type: NodeType,
    label: String,
    properties: HashMap<String, Value>,
    embedding: Vector<f32>,

    // Epistemic
    epistemic: Epistemic,
    confidence: f64,

    // Lifecycle
    created_at: Instant,
    updated_at: Instant,
    access_count: u32,
}

struct Edge {
    id: EdgeId,
    source: NodeId,
    target: NodeId,
    relation: Relation,
    properties: HashMap<String, Value>,
    weight: f64,

    // Epistemic
    epistemic: Epistemic,
    confidence: f64,
}

impl SemanticMemory {
    // Add knowledge
    fn learn(fact: Fact) {
        // Get or create subject node
        let subject = self.graph.get_or_create_node(fact.subject)

        for claim in fact.claims {
            let object = self.graph.get_or_create_node(claim.object)

            // Check for conflicts
            let existing = self.graph.edges_between(subject, object)
                |φ{_.relation == claim.relation}

            if existing.any(|e| e.contradicts(claim)) {
                self.beliefs.handle_conflict(existing, claim)
            } else {
                self.graph.add_edge(Edge {
                    source: subject.id,
                    target: object.id,
                    relation: claim.relation,
                    epistemic: fact.epistemic,
                    confidence: fact.confidence,
                    ..Default::default()
                })
            }
        }

        // Update vector index
        self.vectors.upsert(subject.id, subject.embedding)
    }

    // Query by meaning
    fn query(q: &str, limit: usize) -> Vec<QueryResult> {
        let embedding = embed(q)

        // Vector search for candidates
        let candidates = self.vectors.search(embedding, k: limit * 5)

        // Expand through graph
        let expanded = candidates
            |> flat_map(|n| self.graph.neighborhood(n, hops: 2))
            |> dedupe

        // Score by relevance and confidence
        expanded
            |τ{|n| QueryResult {
                node: n,
                relevance: cosine_similarity(n.embedding, embedding),
                confidence: n.confidence,
                epistemic: n.epistemic,
            }}
            |σ↓{_.relevance * _.confidence}
            |ω{limit}
    }

    // Graph traversal
    fn traverse(start: NodeId, path: Path) -> Vec<Node> {
        let mut current = vec![start]

        for step in path.steps {
            current = current
                |> flat_map(|n| self.graph.follow(n, step.relation))
                |> filter(step.predicate)
                |> dedupe
        }

        current |τ{|id| self.graph.nodes.get(id)}
    }
}

struct BeliefTracker {
    beliefs: HashMap<BeliefKey, BeliefState>,
}

enum BeliefState {
    Held { engram: EngramId, confidence: f64 },
    Contested { positions: Vec<(EngramId, f64)> },
    Retracted { original: EngramId, reason: String },
}

impl BeliefTracker {
    fn handle_conflict(existing: Vec<Edge>, new_claim: Claim) {
        match (existing.epistemic_strength(), new_claim.epistemic_strength()) {
            // New is stronger - replace
            (old, new) if new > old + 0.2 => {
                self.replace(existing, new_claim)
            }
            // Old is stronger - keep, note new
            (old, new) if old > new + 0.2 => {
                self.note_alternative(existing, new_claim)
            }
            // Similar strength - contested
            _ => {
                self.mark_contested(existing, new_claim)
            }
        }
    }
}
```

### Procedural Memory

Skills and patterns learned from experience.

```sigil
struct ProceduralMemory {
    skills: HashMap<SkillId, Skill>,
    patterns: PatternMatcher,

    config: ProceduralConfig,
}

struct Skill {
    id: SkillId,
    name: String,
    description: String,

    // Trigger
    trigger: Pattern,
    preconditions: Vec<Condition>,

    // Procedure
    steps: Vec<Step>,

    // Performance
    success_rate: f64,
    execution_count: u32,
    last_executed: Option<Instant>,

    // Learning
    refinements: Vec<Refinement>,
    failure_modes: Vec<FailureMode>,

    // Metadata
    source_episodes: Vec<EpisodeId>,
    created_at: Instant,
}

struct Pattern {
    features: Vec<Feature>,
    weights: Vec<f64>,
    threshold: f64,
}

struct Step {
    action: Action,
    expected_result: Option<Expectation>,
    fallback: Option<Box<Step>>,
}

struct Refinement {
    condition: Condition,
    modification: Modification,
    learned_from: EpisodeId,
}

struct FailureMode {
    trigger: Pattern,
    failure_type: String,
    recovery: Option<RecoveryStrategy>,
    occurrences: u32,
}

impl ProceduralMemory {
    // Learn from episodes
    fn extract_skill(episodes: Vec<Episode>) -> Option<Skill> {
        // Find common pattern
        let pattern = episodes
            |τ{_.context.to_features()}
            |> find_common_pattern(threshold: 0.7)?

        // Extract procedure
        let steps = episodes
            |τ{_.events}
            |> align_sequences
            |> abstract_steps

        // Compute initial success rate
        let success_rate = episodes
            |φ{_.outcome.is_success()}
            |> count as f64 / episodes.len() as f64

        Some(Skill {
            trigger: pattern,
            steps,
            success_rate,
            source_episodes: episodes |τ{_.id},
            ..Default::default()
        })
    }

    // Find applicable skills
    fn match_situation(context: &Context) -> Vec<(Skill, f64)> {
        let features = context.to_features()

        self.patterns.match(features)
            |τ{|id| (self.skills.get(id), self.patterns.score(id, features))}
            |φ{_.0.success_rate > 0.5}
            |σ↓{_.0.success_rate * _.1}
    }

    // Update from outcome
    fn feedback(skill_id: SkillId, episode: Episode) {
        let skill = self.skills.get_mut(skill_id)

        skill.execution_count += 1
        skill.last_executed = Some(now())

        match episode.outcome {
            Outcome::Success { .. } => {
                skill.success_rate = ema(skill.success_rate, 1.0, α: 0.1)
            }
            Outcome::Failure { error, .. } => {
                skill.success_rate = ema(skill.success_rate, 0.0, α: 0.1)
                skill.failure_modes.push(FailureMode {
                    trigger: episode.context.to_pattern(),
                    failure_type: error,
                    recovery: None,
                    occurrences: 1,
                })
            }
            Outcome::Partial { .. } => {
                skill.success_rate = ema(skill.success_rate, 0.5, α: 0.1)
                self.analyze_partial(skill, episode)
            }
            _ => {}
        }
    }
}
```

---

## Index Layer

### Vector Index (HNSW)

Hierarchical Navigable Small World graph for fast approximate nearest neighbor search.

```sigil
struct VectorIndex {
    layers: Vec<HNSWLayer>,
    entry_point: NodeId,

    config: HNSWConfig,
}

struct HNSWConfig {
    dimensions: usize,               // Embedding dimensions
    m: usize,                        // Max connections per layer (default: 16)
    ef_construction: usize,          // Build-time beam width (default: 200)
    ef_search: usize,                // Query-time beam width (default: 50)
    ml: f64,                         // Level multiplier (default: 1/ln(M))
}

impl VectorIndex {
    fn insert(id: EngramId, vector: Vector<f32>) {
        let level = self.random_level()

        // Find entry points at each level
        let mut entry = self.entry_point
        for l in (level + 1..self.layers.len()).rev() {
            entry = self.search_layer(entry, vector, ef: 1, layer: l)[0]
        }

        // Insert at each level from level down to 0
        for l in (0..=level).rev() {
            let neighbors = self.search_layer(entry, vector,
                ef: self.config.ef_construction, layer: l)

            self.layers[l].add_node(id, vector, neighbors)

            if l > 0 {
                entry = neighbors[0]
            }
        }
    }

    fn search(query: Vector<f32>, k: usize) -> Vec<(EngramId, f64)> {
        let mut entry = self.entry_point

        // Traverse from top to layer 1
        for l in (1..self.layers.len()).rev() {
            entry = self.search_layer(entry, query, ef: 1, layer: l)[0]
        }

        // Search layer 0 with full ef
        self.search_layer(entry, query, ef: self.config.ef_search, layer: 0)
            |> take(k)
            |τ{|id| (id, cosine_similarity(self.get_vector(id), query))}
    }
}
```

### Temporal Index (B+ Tree)

For time-range queries on episodic data.

```sigil
struct TemporalIndex<T> {
    tree: BPlusTree<Instant, Vec<T>>,

    config: TemporalConfig,
}

struct TemporalConfig {
    order: usize,                    // B+ tree order (default: 128)
    bucket_duration: Duration,       // Time bucket size (default: 1 hour)
}

impl<T> TemporalIndex<T> {
    fn insert(timestamp: Instant, item: T) {
        let bucket = self.bucket_for(timestamp)
        self.tree.get_or_insert(bucket, vec![]).push(item)
    }

    fn range(start: Instant, end: Instant) -> Vec<&T> {
        self.tree.range(start, end)
            |> flat_map(|(_, items)| items)
    }

    fn at(point: Instant) -> Vec<&T> {
        let bucket = self.bucket_for(point)
        self.tree.get(bucket).unwrap_or(&vec![])
    }

    fn before(point: Instant, limit: usize) -> Vec<&T> {
        self.tree.range(Instant::MIN, point)
            |> flat_map(|(_, items)| items)
            |> rev
            |> take(limit)
    }
}
```

### Graph Index

Native adjacency list with efficient traversal.

```sigil
struct GraphIndex {
    // Forward edges: source -> [(target, edge_id)]
    outgoing: HashMap<NodeId, Vec<(NodeId, EdgeId)>>,

    // Reverse edges: target -> [(source, edge_id)]
    incoming: HashMap<NodeId, Vec<(NodeId, EdgeId)>>,

    // By relation type
    by_relation: MultiMap<Relation, EdgeId>,
}

impl GraphIndex {
    fn add_edge(edge: &Edge) {
        self.outgoing.entry(edge.source)
            .or_default()
            .push((edge.target, edge.id))

        self.incoming.entry(edge.target)
            .or_default()
            .push((edge.source, edge.id))

        self.by_relation.insert(edge.relation.clone(), edge.id)
    }

    fn follow(node: NodeId, relation: Relation) -> Vec<NodeId> {
        self.outgoing.get(node)
            .unwrap_or(&vec![])
            |φ{|(_, edge_id)| self.edges.get(edge_id).relation == relation}
            |τ{|(target, _)| target}
    }

    fn neighborhood(node: NodeId, hops: usize) -> HashSet<NodeId> {
        let mut visited = HashSet::new()
        let mut frontier = vec![node]

        for _ in 0..hops {
            let mut next_frontier = vec![]
            for n in frontier {
                if visited.insert(n) {
                    for (neighbor, _) in self.outgoing.get(n).unwrap_or(&vec![]) {
                        next_frontier.push(neighbor)
                    }
                    for (neighbor, _) in self.incoming.get(n).unwrap_or(&vec![]) {
                        next_frontier.push(neighbor)
                    }
                }
            }
            frontier = next_frontier
        }

        visited
    }
}
```

---

## Storage Layer

### Hot Store (Memory-Mapped)

For instant memory and recently accessed engrams.

```sigil
struct HotStore {
    arena: MmapArena,
    index: HashMap<EngramId, ArenaOffset>,
    lru: LruCache<EngramId, ()>,

    config: HotStoreConfig,
}

struct HotStoreConfig {
    capacity: ByteSize,              // Default: 256MB
    eviction_policy: EvictionPolicy, // Default: LRU
}

impl HotStore {
    fn get(id: EngramId) -> Option<&EngramRecord> {
        let offset = self.index.get(id)?
        self.lru.touch(id)
        Some(self.arena.read(offset))
    }

    fn put(record: EngramRecord) {
        if self.arena.remaining() < record.size() {
            self.evict_until(space_needed: record.size())
        }

        let offset = self.arena.allocate(record.size())
        self.arena.write(offset, record)
        self.index.insert(record.id, offset)
        self.lru.insert(record.id, ())
    }

    fn evict_until(space_needed: usize) {
        while self.arena.remaining() < space_needed {
            if let Some(victim) = self.lru.pop_lru() {
                let offset = self.index.remove(victim.0)
                // Record evicted to warm store
                warm_store.absorb(self.arena.read(offset))
                self.arena.free(offset)
            }
        }
    }
}
```

### Warm Store (LSM Tree)

For semantic memory and active episodic memory.

```sigil
struct WarmStore {
    memtable: BTreeMap<EngramId, EngramRecord>,
    sstables: Vec<SSTable>,

    config: WarmStoreConfig,
}

struct WarmStoreConfig {
    memtable_size: ByteSize,         // Default: 64MB
    level_size_multiplier: usize,    // Default: 10
    bloom_filter_fp_rate: f64,       // Default: 0.01
}

impl WarmStore {
    fn get(id: EngramId) -> Option<EngramRecord> {
        // Check memtable first
        if let Some(record) = self.memtable.get(id) {
            return Some(record.clone())
        }

        // Check SSTables from newest to oldest
        for sstable in self.sstables.iter().rev() {
            if sstable.bloom_filter.may_contain(id) {
                if let Some(record) = sstable.get(id) {
                    return Some(record)
                }
            }
        }

        None
    }

    fn put(record: EngramRecord) {
        self.memtable.insert(record.id, record)

        if self.memtable.size() >= self.config.memtable_size {
            self.flush_memtable()
        }
    }

    fn flush_memtable() {
        let sstable = SSTable::from_sorted(self.memtable.drain())
        self.sstables.push(sstable)

        if self.should_compact() {
            self.compact()
        }
    }
}
```

### Cold Store (Archive)

For archived memories - compressed, append-only.

```sigil
struct ColdStore {
    segments: Vec<ArchiveSegment>,
    manifest: ArchiveManifest,

    config: ColdStoreConfig,
}

struct ArchiveSegment {
    path: Path,
    index: SparseIndex,              // Sample of keys for binary search
    bloom_filter: BloomFilter,
    compression: Compression,        // Zstd level 19
}

struct ColdStoreConfig {
    segment_size: ByteSize,          // Default: 256MB
    compression_level: u32,          // Default: 19 (max)
    sparse_index_interval: usize,    // Default: 1000
}

impl ColdStore {
    fn archive(records: Vec<EngramRecord>) {
        let compressed = records
            |τ{serialize}
            |> zstd::compress(level: self.config.compression_level)

        let segment = ArchiveSegment::create(compressed)
        self.segments.push(segment)
        self.manifest.add(segment)
    }

    fn retrieve(id: EngramId) -> Option<EngramRecord> {
        for segment in &self.segments {
            if segment.bloom_filter.may_contain(id) {
                // Binary search using sparse index
                if let Some(record) = segment.search(id) {
                    return Some(record)
                }
            }
        }
        None
    }
}
```

---

## Query Processing

### Reconstruction Engine

The core of Engram - reconstructing understanding from stored traces.

```sigil
struct ReconstructionEngine {
    instant: &InstantMemory,
    episodic: &EpisodicMemory,
    semantic: &SemanticMemory,
    procedural: &ProceduralMemory,

    config: ReconstructionConfig,
}

struct ReconstructionConfig {
    max_sources: usize,              // Max memories to consider
    confidence_threshold: f64,       // Min confidence to include
    diversity_weight: f64,           // Weight for result diversity
    recency_weight: f64,             // Weight for recent memories
}

struct Recall {
    memories: Vec<RankedMemory>,
    skills: Vec<Skill>,
    confidence: ConfidenceDistribution,
    gaps: Vec<Gap>,
    reasoning: ReconstructionTrace,
}

struct RankedMemory {
    engram: EngramRef,
    relevance: f64,
    confidence: f64,
    source_type: MemoryType,
}

struct Gap {
    description: String,
    suggested_actions: Vec<SuggestedAction>,
}

impl ReconstructionEngine {
    fn recall(query: Query, context: Context) -> Recall {
        // Gather candidates from all memory systems
        let instant_hits = self.instant.search(query)
        let episodic_hits = self.episodic.recall_similar(context)
        let semantic_hits = self.semantic.query(query.text)
        let skills = self.procedural.match_situation(context)

        // Merge and rank
        let merged = self.merge_sources(
            instant_hits,
            episodic_hits,
            semantic_hits
        )

        // Apply diversity (MMR - Maximal Marginal Relevance)
        let diverse = self.apply_mmr(merged, lambda: self.config.diversity_weight)

        // Compute aggregate confidence
        let confidence = self.compute_confidence(diverse)

        // Identify gaps
        let gaps = self.identify_gaps(query, diverse)

        Recall {
            memories: diverse,
            skills,
            confidence,
            gaps,
            reasoning: self.trace,
        }
    }

    fn merge_sources(
        instant: Vec<EngramRef>,
        episodic: Vec<Episode>,
        semantic: Vec<QueryResult>
    ) -> Vec<RankedMemory> {
        // Convert to common format
        let mut all = vec![]

        all.extend(instant |τ{|e| RankedMemory {
            engram: e,
            relevance: e.priority,
            confidence: e.confidence,
            source_type: MemoryType::Instant,
        }})

        all.extend(episodic |τ{|e| RankedMemory {
            engram: e.to_ref(),
            relevance: e.significance,
            confidence: e.outcome.confidence(),
            source_type: MemoryType::Episodic,
        }})

        all.extend(semantic |τ{|r| RankedMemory {
            engram: r.node.to_ref(),
            relevance: r.relevance,
            confidence: r.confidence,
            source_type: MemoryType::Semantic,
        }})

        // Sort by combined score
        all |σ↓{_.relevance * _.confidence * self.recency_boost(_)}
    }

    fn identify_gaps(query: Query, results: Vec<RankedMemory>) -> Vec<Gap> {
        let mut gaps = vec![]

        // Check for low overall confidence
        let avg_confidence = results |τ{_.confidence} |Σ / results.len()
        if avg_confidence < 0.5 {
            gaps.push(Gap {
                description: "Low overall confidence in results",
                suggested_actions: vec![
                    SuggestedAction::AskUser("clarification"),
                    SuggestedAction::SearchExternal(query.text),
                ],
            })
        }

        // Check for missing expected entities
        let expected = query.extract_entities()
        let found = results |> flat_map(|r| r.engram.entities())
        let missing = expected - found
        for entity in missing {
            gaps.push(Gap {
                description: format!("No information about '{}'", entity),
                suggested_actions: vec![
                    SuggestedAction::SearchFor(entity),
                ],
            })
        }

        // Check for contested beliefs
        let contested = results |φ{_.engram.epistemic.is_contested()}
        for memory in contested {
            gaps.push(Gap {
                description: format!("Contested belief: {}", memory.summary()),
                suggested_actions: vec![
                    SuggestedAction::ResolveConflict(memory.engram.id),
                ],
            })
        }

        gaps
    }
}
```

---

## Background Processes

### Consolidation Process

Runs periodically to maintain memory health.

```sigil
process Consolidation {
    schedule: every(6h),

    fn run(engram: &mut Engram) {
        // 1. Decay episodic memories
        engram.episodic.decay()

        // 2. Extract skills from successful episode patterns
        let skill_candidates = engram.episodic.find_skill_patterns()
        for pattern in skill_candidates {
            engram.procedural.learn(pattern)
        }

        // 3. Compress similar semantic memories
        let clusters = engram.semantic.find_similar_clusters()
        for cluster in clusters {
            let abstract_node = abstract_cluster(cluster)
            engram.semantic.replace_cluster(cluster, abstract_node)
        }

        // 4. Archive faded memories
        let faded = engram.all_memories()
            |φ{_.strength < 0.1 && !_.pinned}
        engram.archive(faded)

        // 5. Heal negative patterns
        let trauma = engram.episodic.episodes
            |φ{_.valence < -0.5 && _.access_count > 3}
        for episode in trauma {
            episode.strength *= 0.5  // Reduce but don't delete
        }

        // 6. Update statistics
        engram.stats.last_consolidation = now()
        engram.stats.total_memories = engram.count_all()
    }
}
```

---

## Configuration

### Full Configuration Schema

```sigil
struct EngramConfig {
    // Memory capacities
    instant: InstantConfig,
    episodic: EpisodicConfig,
    semantic: SemanticConfig,
    procedural: ProceduralConfig,

    // Index configuration
    vector_index: HNSWConfig,

    // Storage configuration
    hot_store: HotStoreConfig,
    warm_store: WarmStoreConfig,
    cold_store: ColdStoreConfig,

    // Embedding
    embedding: EmbeddingConfig,

    // Distribution
    distribution: DistributionConfig,

    // Background processes
    consolidation: ConsolidationConfig,
}

struct EmbeddingConfig {
    model: EmbeddingModel,           // Default: MiniLM-L6
    dimensions: usize,               // Default: 384
    normalize: bool,                 // Default: true
    batch_size: usize,               // Default: 32
}

enum EmbeddingModel {
    MiniLM_L6,                       // Fast, good quality
    BGE_Small,                       // Better quality
    OpenAI(String),                  // External API
    Custom(Path),                    // Custom ONNX model
}

impl EngramConfig {
    fn default() -> Self {
        // Sensible defaults for single-agent use
        Self {
            instant: InstantConfig {
                capacity: 8000 tokens,
                decay_rate: 0.05,
                ..
            },
            episodic: EpisodicConfig {
                decay_function: DecayFunction::Exponential(λ: 0.1),
                consolidation_threshold: 0.3,
                ..
            },
            ..
        }
    }

    fn high_memory() -> Self {
        // For agents with large context windows
        Self {
            instant: InstantConfig {
                capacity: 128000 tokens,
                ..
            },
            ..Self::default()
        }
    }

    fn distributed() -> Self {
        // For multi-agent deployments
        Self {
            distribution: DistributionConfig {
                enabled: true,
                sync_strategy: SyncStrategy::CRDT,
                ..
            },
            ..Self::default()
        }
    }
}
```

---

## Performance Characteristics

### Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| Instant attend | O(log n) | O(1) |
| Episodic record | O(log n) | O(m) events |
| Semantic learn | O(log n) graph + O(log n) vector | O(1) |
| Vector search | O(log n) | O(k) |
| Graph traverse | O(d^h) | O(d^h) |
| Temporal range | O(log n + k) | O(k) |
| Consolidation | O(n) | O(n) |

Where:
- n = number of engrams
- k = result limit
- d = average node degree
- h = traversal hops
- m = events per episode

### Memory Usage

Approximate per-engram overhead:
- Base record: ~200 bytes
- Embedding (384d): ~1.5 KB
- Graph links: ~50 bytes per link
- Indices: ~100 bytes

Total: ~2 KB per engram typical

### Benchmarks (Target)

- Vector search (1M vectors): < 10ms p99
- Graph traversal (2 hops): < 5ms p99
- Recall (full reconstruction): < 50ms p99
- Consolidation (100K engrams): < 60s

---

*This architecture document is a living specification. Implementation may reveal necessary adjustments.*
