# Distribution and Synchronization

*Multi-agent memory sharing in Engram*

---

## Overview

AI agents are not singular. They exist as:
- **Multiple instances** of the same agent across conversations
- **Multiple agents** collaborating on shared tasks
- **Agent swarms** with collective intelligence

Engram's distribution layer enables memory sharing across these scenarios while respecting the fundamental nature of distributed artificial cognition.

---

## Memory Scopes

Every engram exists within a scope that determines its visibility and synchronization behavior.

### Scope Hierarchy

```
┌─────────────────────────────────────────────┐
│                 UNIVERSAL                    │
│     (Shared across all agents globally)      │
├─────────────────────────────────────────────┤
│                COLLECTIVE                    │
│       (Shared within an agent group)         │
├─────────────────────────────────────────────┤
│                   AGENT                      │
│    (Shared across instances of one agent)    │
├─────────────────────────────────────────────┤
│                 INSTANCE                     │
│         (Private to this instance)           │
└─────────────────────────────────────────────┘
```

### Scope Definitions

```sigil
enum Scope {
    // Private to this specific instance/conversation
    Instance {
        instance_id: InstanceId,
    },

    // Shared across all instances of this agent
    Agent {
        agent_id: AgentId,
    },

    // Shared within a defined group of agents
    Collective {
        collective_id: CollectiveId,
        members: Vec<AgentId>,
    },

    // Globally shared (rare, carefully curated)
    Universal,
}
```

### Scope Selection

```sigil
// Learn with explicit scope
memory.learn_scoped(
    fact,
    scope: Scope::Agent { agent_id: self.id }
)

// Default scopes by memory type
// - Instant: Instance (always local)
// - Episodic: Instance (experiences are personal)
// - Semantic: Agent (knowledge persists across instances)
// - Procedural: Agent (skills persist across instances)

// Override default
memory.config.default_scopes = ScopeDefaults {
    instant: Scope::Instance,
    episodic: Scope::Instance,
    semantic: Scope::Agent,
    procedural: Scope::Collective,  // Share skills across agent group
}
```

---

## Synchronization Strategies

### CRDT-Based Sync (Default)

Conflict-free Replicated Data Types ensure eventual consistency without coordination.

```sigil
struct CRDTSync {
    strategy: CRDTStrategy,
}

enum CRDTStrategy {
    // Last-Writer-Wins for simple values
    LWW {
        timestamp_source: TimestampSource,
    },

    // Multi-Value Register for contested beliefs
    MVR,

    // Add-Wins Set for collections
    AWSet,

    // Custom merge function
    Custom {
        merge: fn(local: Engram, remote: Engram) -> MergeResult,
    },
}
```

**How CRDT Sync Works:**

1. Each engram carries a vector clock
2. On sync, clocks are compared
3. Concurrent modifications are merged automatically
4. No coordination required between instances

```sigil
// Automatic CRDT sync
memory.sync(SyncConfig {
    scope: Scope::Agent,
    strategy: SyncStrategy::CRDT {
        conflict_resolution: CRDTStrategy::MVR,
    },
})

// Result shows any concurrent modifications
match result {
    SyncResult::Clean { synced_count } => {
        log("Synced {} engrams", synced_count)
    }
    SyncResult::Merged { merged_count, conflicts } => {
        log("Merged {} with {} conflicts auto-resolved", merged_count, conflicts)
    }
}
```

### Consensus-Based Sync

For critical shared knowledge requiring agreement.

```sigil
struct ConsensusSync {
    protocol: ConsensusProtocol,
    quorum: QuorumConfig,
}

enum ConsensusProtocol {
    // Raft for ordered log
    Raft {
        election_timeout: Duration,
        heartbeat_interval: Duration,
    },

    // Paxos for single-value consensus
    Paxos,

    // PBFT for Byzantine fault tolerance
    PBFT {
        f: usize,  // Max faulty nodes
    },
}

struct QuorumConfig {
    min_nodes: usize,
    timeout: Duration,
}
```

**When to Use Consensus:**

- Universal scope knowledge
- Critical facts that must be consistent
- Collective decisions

```sigil
// Consensus for critical knowledge
memory.learn_with_consensus(
    fact: critical_security_policy,
    scope: Scope::Universal,
    quorum: QuorumConfig {
        min_nodes: 3,
        timeout: 5s,
    }
)
```

### Explicit Sharing

Direct agent-to-agent memory sharing.

```sigil
// Share specific memories with another agent
memory.share(
    memories: query_result.memories,
    with: other_agent_id,
    permission: SharePermission::ReadOnly,
)

// Share with transformation
memory.share_transformed(
    memories: sensitive_data,
    with: external_agent,
    transform: |m| m.redact_pii(),
    permission: SharePermission::ReadOnly,
)

// Accept shared memories
memory.accept_share(
    from: other_agent_id,
    share_id: share_id,
    trust_level: 0.7,  // Affects epistemic status
)
```

---

## Conflict Resolution

When memories diverge between instances, conflicts must be resolved.

### Conflict Types

```sigil
enum ConflictType {
    // Same entity, different values
    ValueConflict {
        key: EngramId,
        local: Value,
        remote: Value,
    },

    // Contradictory beliefs
    BeliefConflict {
        subject: NodeId,
        relation: Relation,
        local_object: NodeId,
        remote_object: NodeId,
    },

    // Different episodic accounts
    EpisodicConflict {
        episode_id: EpisodeId,
        local_events: Vec<Event>,
        remote_events: Vec<Event>,
    },

    // Skill version conflict
    SkillConflict {
        skill_id: SkillId,
        local_refinements: Vec<Refinement>,
        remote_refinements: Vec<Refinement>,
    },
}
```

### Resolution Strategies

```sigil
enum ConflictResolution {
    // Automatic strategies
    TakeLocal,
    TakeRemote,
    TakeNewest,
    TakeHighestConfidence,
    Merge,

    // Keep both as alternatives
    Fork {
        mark_contested: bool,
    },

    // Escalate for manual resolution
    Escalate {
        handler: ConflictHandler,
    },

    // Custom resolution function
    Custom {
        resolver: fn(Conflict) -> Resolution,
    },
}

// Configure default resolutions
memory.config.conflict_resolution = ConflictResolutionConfig {
    value_conflict: ConflictResolution::TakeNewest,
    belief_conflict: ConflictResolution::Fork { mark_contested: true },
    episodic_conflict: ConflictResolution::Merge,
    skill_conflict: ConflictResolution::TakeHighestConfidence,
}
```

### Conflict Handlers

```sigil
struct ConflictHandler {
    // Called when conflict cannot be auto-resolved
    on_conflict: fn(Conflict) -> Resolution,

    // Called after resolution
    on_resolved: fn(Conflict, Resolution),
}

// Example: Ask user on critical conflicts
let handler = ConflictHandler {
    on_conflict: |conflict| {
        if conflict.is_critical() {
            Resolution::AskUser {
                question: format!("Conflict detected: {}. Which version?", conflict),
                options: vec!["Local", "Remote", "Both"],
            }
        } else {
            Resolution::Merge
        }
    },
    on_resolved: |conflict, resolution| {
        log("Resolved {}: {:?}", conflict.id, resolution)
    },
}
```

---

## Multi-Agent Architecture

### Agent Federation

A federation is a group of agents that share memory with defined trust relationships.

```sigil
struct Federation {
    id: FederationId,
    members: Vec<AgentMembership>,
    shared_scope: Scope,
    trust_model: TrustModel,
    sync_config: SyncConfig,
}

struct AgentMembership {
    agent_id: AgentId,
    role: AgentRole,
    joined_at: Instant,
    trust_level: f64,
}

enum AgentRole {
    Leader,          // Can modify federation rules
    Member,          // Standard participation
    Observer,        // Read-only access
}
```

**Creating a Federation:**

```sigil
let federation = Federation::create(FederationConfig {
    name: "Project Alpha Agents",
    initial_members: vec![agent_a, agent_b, agent_c],
    shared_scope: Scope::Collective {
        collective_id: generate_id(),
        members: vec![agent_a, agent_b, agent_c],
    },
    trust_model: TrustModel::Uniform(0.8),
    sync_config: SyncConfig {
        strategy: SyncStrategy::CRDT,
        interval: 5m,
    },
})

// Join federation
memory.join_federation(federation, role: AgentRole::Member)

// Memories created with collective scope auto-sync
memory.learn_scoped(fact, scope: federation.shared_scope)
```

### Trust Model

Trust affects how shared memories are treated.

```sigil
enum TrustModel {
    // All members equally trusted
    Uniform(f64),

    // Trust varies by agent
    PerAgent(HashMap<AgentId, f64>),

    // Trust based on history
    Reputation {
        initial: f64,
        decay: f64,
        boost_on_verification: f64,
        penalty_on_contradiction: f64,
    },

    // Web of trust
    Transitive {
        direct_trust: HashMap<AgentId, f64>,
        trust_decay_per_hop: f64,
        max_hops: usize,
    },
}

// Trust affects epistemic status of received memories
fn apply_trust(memory: &mut Memory, source: AgentId, trust: f64) {
    memory.epistemic = Epistemic::Reported {
        source: Source::Agent(source),
        trust_level: trust,
    }
    memory.confidence *= trust
}
```

### Agent Communication

Beyond memory sync, agents can explicitly communicate.

```sigil
// Tell another agent something
memory.tell(
    recipient: other_agent,
    message: TellMessage::Fact(fact),
)

// Ask another agent
let response = memory.ask(
    recipient: expert_agent,
    question: "How should I handle authentication?",
    timeout: 30s,
)

// Broadcast to federation
memory.broadcast(
    federation: project_federation,
    message: BroadcastMessage::Alert("New requirement discovered"),
)
```

---

## Sync Protocol

### Wire Protocol

Communication between Engram instances uses a binary protocol:

```sigil
struct SyncMessage {
    version: u8,
    message_type: SyncMessageType,
    sender: AgentId,
    timestamp: Instant,
    payload: Bytes,
    signature: Option<Signature>,
}

enum SyncMessageType {
    // Handshake
    Hello { capabilities: Vec<Capability> },
    HelloAck { accepted_capabilities: Vec<Capability> },

    // Sync
    SyncRequest { scope: Scope, since: VectorClock },
    SyncResponse { engrams: Vec<EngramDelta>, clock: VectorClock },
    SyncAck { received: Vec<EngramId> },

    // Conflicts
    ConflictNotification { conflicts: Vec<Conflict> },
    ConflictResolution { resolutions: Vec<(ConflictId, Resolution)> },

    // Federation
    JoinRequest { agent: AgentId, role: AgentRole },
    JoinResponse { accepted: bool, reason: Option<String> },
    LeaveNotification { agent: AgentId },

    // Heartbeat
    Ping,
    Pong,
}
```

### Sync Flow

```
Agent A                                    Agent B
   │                                          │
   │──────── Hello(capabilities) ─────────────▶
   │                                          │
   │◀─────── HelloAck(accepted) ──────────────│
   │                                          │
   │──────── SyncRequest(scope, clock) ───────▶
   │                                          │
   │◀─────── SyncResponse(deltas, clock) ─────│
   │                                          │
   │──────── SyncAck(received) ───────────────▶
   │                                          │
   │         [If conflicts detected]          │
   │                                          │
   │◀─────── ConflictNotification ────────────│
   │                                          │
   │──────── ConflictResolution ──────────────▶
   │                                          │
```

### Delta Encoding

Only changes are transmitted:

```sigil
struct EngramDelta {
    id: EngramId,
    delta_type: DeltaType,
    clock: VectorClock,
    data: Bytes,
}

enum DeltaType {
    Create { full: Engram },
    Update { changes: Vec<FieldChange> },
    Delete { reason: String },
    Archive,
}

struct FieldChange {
    path: Vec<String>,
    old_value_hash: Hash,
    new_value: Value,
}
```

---

## Storage Layer Distribution

### Replication

```sigil
struct ReplicationConfig {
    factor: usize,              // Number of replicas
    strategy: ReplicationStrategy,
    consistency: ConsistencyLevel,
}

enum ReplicationStrategy {
    // All replicas are equal
    Symmetric,

    // One primary, rest secondary
    PrimarySecondary {
        failover: FailoverConfig,
    },

    // Sharded by key
    Sharded {
        shard_key: fn(&Engram) -> ShardId,
        shards: Vec<ShardConfig>,
    },
}

enum ConsistencyLevel {
    // Read from any replica
    One,

    // Read from majority
    Quorum,

    // Read from all
    All,

    // Read from local only
    Local,
}
```

### Partitioning

For large-scale deployments:

```sigil
struct PartitionConfig {
    strategy: PartitionStrategy,
    partition_count: usize,
}

enum PartitionStrategy {
    // By hash of engram ID
    HashPartition,

    // By scope
    ScopePartition,

    // By memory type
    TypePartition,

    // By time range (for episodic)
    TimePartition {
        bucket_size: Duration,
    },

    // Custom
    Custom {
        partitioner: fn(&Engram) -> PartitionId,
    },
}
```

---

## Security

### Authentication

```sigil
struct AuthConfig {
    method: AuthMethod,
    token_ttl: Duration,
}

enum AuthMethod {
    // Shared secret
    SharedKey {
        key: SecretKey,
    },

    // Public key infrastructure
    PKI {
        certificate: Certificate,
        private_key: PrivateKey,
        trusted_cas: Vec<Certificate>,
    },

    // OAuth/OIDC
    OAuth {
        provider: String,
        client_id: String,
    },

    // Mutual TLS
    MTLS {
        certificate: Certificate,
        private_key: PrivateKey,
    },
}
```

### Encryption

```sigil
struct EncryptionConfig {
    // Encrypt data at rest
    at_rest: Option<AtRestEncryption>,

    // Encrypt data in transit
    in_transit: InTransitEncryption,

    // End-to-end encryption for sensitive data
    e2e: Option<E2EEncryption>,
}

struct AtRestEncryption {
    algorithm: EncryptionAlgorithm,
    key_management: KeyManagement,
}

struct InTransitEncryption {
    min_tls_version: TlsVersion,
    cipher_suites: Vec<CipherSuite>,
}

struct E2EEncryption {
    // Only source and destination can read
    algorithm: E2EAlgorithm,
    key_exchange: KeyExchangeProtocol,
}
```

### Access Control

```sigil
struct AccessControl {
    model: AccessModel,
    default_policy: Policy,
}

enum AccessModel {
    // Role-based
    RBAC {
        roles: Vec<Role>,
        role_assignments: HashMap<AgentId, Vec<RoleId>>,
    },

    // Attribute-based
    ABAC {
        policies: Vec<ABACPolicy>,
    },

    // Capability-based
    Capability {
        capabilities: HashMap<AgentId, Vec<Capability>>,
    },
}

struct ABACPolicy {
    name: String,
    effect: PolicyEffect,
    conditions: Vec<Condition>,
}

enum PolicyEffect {
    Allow,
    Deny,
}
```

---

## Observability

### Metrics

```sigil
// Built-in metrics
struct SyncMetrics {
    sync_count: Counter,
    sync_duration: Histogram,
    sync_failures: Counter,
    conflicts_detected: Counter,
    conflicts_resolved: Counter,
    bytes_transmitted: Counter,
    bytes_received: Counter,
    engrams_synced: Counter,
    replication_lag: Gauge,
}

// Access metrics
let metrics = memory.sync_metrics()
log("Sync count: {}", metrics.sync_count.get())
```

### Tracing

```sigil
// Distributed tracing for sync operations
memory.config.tracing = TracingConfig {
    enabled: true,
    sampler: Sampler::RateLimited(100),  // 100 traces/sec
    exporter: TracingExporter::OTLP {
        endpoint: "http://jaeger:4317",
    },
}
```

### Health Checks

```sigil
// Check sync health
let health = memory.sync_health()

match health.status {
    HealthStatus::Healthy => {}
    HealthStatus::Degraded { reason } => {
        log("Sync degraded: {}", reason)
    }
    HealthStatus::Unhealthy { reason } => {
        alert("Sync unhealthy: {}", reason)
    }
}
```

---

## Configuration Examples

### Single Agent, Multiple Instances

```sigil
let config = EngramConfig {
    distribution: DistributionConfig {
        enabled: true,
        scope_defaults: ScopeDefaults::agent_shared(),
        sync: SyncConfig {
            strategy: SyncStrategy::CRDT,
            interval: 30s,
            on_change: true,
        },
        storage: DistributedStorageConfig {
            backend: StorageBackend::Redis {
                url: "redis://localhost:6379",
            },
        },
    },
    ..Default::default()
}
```

### Agent Federation

```sigil
let config = EngramConfig {
    distribution: DistributionConfig {
        enabled: true,
        federation: Some(FederationConfig {
            id: "project-agents",
            discovery: DiscoveryMethod::DNS {
                domain: "agents.internal",
            },
            trust_model: TrustModel::Reputation::default(),
        }),
        sync: SyncConfig {
            strategy: SyncStrategy::CRDT,
            interval: 1m,
        },
    },
    ..Default::default()
}
```

### High-Security Deployment

```sigil
let config = EngramConfig {
    distribution: DistributionConfig {
        enabled: true,
        auth: AuthConfig {
            method: AuthMethod::MTLS {
                certificate: load_cert("agent.crt"),
                private_key: load_key("agent.key"),
            },
            token_ttl: 1h,
        },
        encryption: EncryptionConfig {
            at_rest: Some(AtRestEncryption {
                algorithm: EncryptionAlgorithm::AES256GCM,
                key_management: KeyManagement::Vault {
                    url: "https://vault:8200",
                },
            }),
            in_transit: InTransitEncryption {
                min_tls_version: TlsVersion::V1_3,
                cipher_suites: vec![CipherSuite::TLS_AES_256_GCM_SHA384],
            },
            e2e: Some(E2EEncryption {
                algorithm: E2EAlgorithm::X25519_XSalsa20_Poly1305,
                key_exchange: KeyExchangeProtocol::X25519,
            }),
        },
        access_control: AccessControl {
            model: AccessModel::ABAC { policies: security_policies() },
            default_policy: Policy::Deny,
        },
    },
    ..Default::default()
}
```

---

*Distribution transforms isolated agents into collaborative intelligence. Use thoughtfully.*
