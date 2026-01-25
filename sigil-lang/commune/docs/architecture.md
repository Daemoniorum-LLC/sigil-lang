# Commune Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              COMMUNE NETWORK                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         TRANSPORT LAYER                              │    │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐           │    │
│  │  │  LOCAL    │ │    TCP    │ │   QUIC    │ │   UNIX    │           │    │
│  │  │  (same    │ │           │ │           │ │  SOCKET   │           │    │
│  │  │ process)  │ │           │ │           │ │           │           │    │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         PROTOCOL LAYER                               │    │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐           │    │
│  │  │ MESSAGE   │ │   INTENT  │ │   TRUST   │ │  CHANNEL  │           │    │
│  │  │  CODEC    │ │ RESOLVER  │ │  MANAGER  │ │  ROUTER   │           │    │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         SEMANTIC LAYER                               │    │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐           │    │
│  │  │ EPISTEMIC │ │ COLLECTIVE│ │  SWARM    │ │ CONSENSUS │           │    │
│  │  │  TRACKER  │ │  MEMORY   │ │COORDINATOR│ │  ENGINE   │           │    │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         AGENT INTERFACE                              │    │
│  │                                                                      │    │
│  │    ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐    ┌──────┐       │    │
│  │    │Agent │    │Agent │    │Agent │    │Agent │    │Agent │       │    │
│  │    │  A   │    │  B   │    │  C   │    │  D   │    │  E   │       │    │
│  │    └──────┘    └──────┘    └──────┘    └──────┘    └──────┘       │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Message System

```sigil
/// Core message structure
pub struct Message {
    /// Unique identifier (UUID v7)
    pub id: MessageId,

    /// Message header
    pub header: MessageHeader,

    /// Message payload
    pub payload: Payload,

    /// Cryptographic signature
    pub signature: Signature,
}

pub struct MessageHeader {
    /// Sender identity
    pub from: AgentId,

    /// Recipient(s)
    pub to: Recipient,

    /// Communicative intent
    pub intent: Intent,

    /// Epistemic status
    pub epistemic: Epistemic,

    /// Confidence level
    pub confidence: f32,

    /// Timestamp
    pub timestamp: Timestamp,

    /// Correlation for threading
    pub correlation_id: Option<MessageId>,

    /// Reply-to reference
    pub reply_to: Option<MessageId>,

    /// Time-to-live
    pub ttl: Option<Duration>,

    /// Priority
    pub priority: Priority,

    /// Trace context for debugging
    pub trace: Option<TraceContext>,
}

pub enum Recipient {
    /// Single agent
    Agent(AgentId),

    /// Multiple specific agents
    Agents(Vec<AgentId>),

    /// Named channel
    Channel(ChannelId),

    /// All agents
    Broadcast,

    /// Swarm members
    Swarm(SwarmId),
}

pub enum Payload {
    /// Structured data
    Data(Value),

    /// Binary blob
    Binary(Bytes),

    /// Reference to external data
    Reference(Uri),

    /// Streaming data
    Stream(StreamId),

    /// Empty (for pure intent messages)
    Empty,
}
```

### 2. Intent System

```sigil
/// Communicative intent
pub enum Intent {
    // === Assertives ===
    /// Share factual information
    Inform {
        topic: String,
        content: Value,
    },

    /// Report task status
    Report {
        task_id: TaskId,
        status: TaskStatus,
        result: Option<Value>,
    },

    /// Confirm a statement
    Confirm {
        statement: MessageId,
        agrees: bool,
    },

    /// Share knowledge
    Share {
        knowledge: Knowledge,
    },

    // === Directives ===
    /// Request an action
    Request {
        action: String,
        params: Value,
        urgency: Urgency,
    },

    /// Delegate a task
    Delegate {
        task: Task,
        constraints: Vec<Constraint>,
    },

    /// Ask a question
    Query {
        question: String,
        context: Value,
        expected_form: Option<Schema>,
    },

    // === Commissives ===
    /// Accept a request
    Accept {
        request: MessageId,
        commitment: Option<String>,
    },

    /// Reject a request
    Reject {
        request: MessageId,
        reason: String,
        alternative: Option<String>,
    },

    /// Promise future action
    Promise {
        action: String,
        deadline: Option<Timestamp>,
    },

    // === Declarations ===
    /// Make an announcement
    Announce {
        topic: String,
        content: Value,
        importance: Importance,
    },

    /// Propose something for consideration
    Propose {
        proposal: Proposal,
    },

    /// Vote on a proposal
    Vote {
        proposal_id: ProposalId,
        vote: VoteValue,
        rationale: Option<String>,
    },

    // === Meta ===
    /// Acknowledge receipt
    Ack {
        message: MessageId,
    },

    /// Request retransmission
    Nack {
        message: MessageId,
        reason: String,
    },

    /// Ping for presence
    Ping,

    /// Response to ping
    Pong {
        ping: MessageId,
        load: f32,
    },
}

impl Intent {
    /// Check if this intent expects a response
    pub fn expects_response(&self) -> bool {
        matches!(self,
            Intent::Request { .. } |
            Intent::Query { .. } |
            Intent::Delegate { .. } |
            Intent::Propose { .. } |
            Intent::Ping
        )
    }

    /// Get default timeout for response
    pub fn default_timeout(&self) -> Duration {
        match self {
            Intent::Ping => Duration::seconds(5),
            Intent::Query { .. } => Duration::seconds(30),
            Intent::Request { urgency, .. } => match urgency {
                Urgency::Critical => Duration::seconds(10),
                Urgency::High => Duration::seconds(30),
                Urgency::Normal => Duration::minutes(5),
                Urgency::Low => Duration::minutes(30),
            },
            Intent::Delegate { .. } => Duration::minutes(1),
            _ => Duration::minutes(5),
        }
    }
}
```

### 3. Trust Manager

```sigil
/// Manages trust relationships between agents
pub struct TrustManager {
    /// Trust profiles for known agents
    profiles: HashMap<AgentId, TrustProfile>,

    /// Trust update history
    history: Vec<TrustEvent>,

    /// Configuration
    config: TrustConfig,
}

pub struct TrustProfile {
    /// Agent identity
    pub agent: AgentId,

    /// Base trust level (0.0 - 1.0)
    pub base_trust: f32,

    /// Domain-specific trust
    pub domain_trust: HashMap<Domain, f32>,

    /// Accuracy tracking
    pub accuracy: AccuracyTracker,

    /// Vouches received
    pub vouches: Vec<Vouch>,

    /// Last interaction
    pub last_seen: Timestamp,

    /// Trust flags
    pub flags: TrustFlags,
}

pub struct AccuracyTracker {
    /// Recent predictions/claims with outcomes
    records: VecDeque<AccuracyRecord>,

    /// Rolling accuracy score
    score: f32,

    /// Domain-specific accuracy
    domain_scores: HashMap<Domain, f32>,
}

pub struct Vouch {
    /// Who is vouching
    pub voucher: AgentId,

    /// Domain of vouch
    pub domain: Option<Domain>,

    /// Strength of vouch
    pub strength: f32,

    /// When vouched
    pub timestamp: Timestamp,
}

impl TrustManager {
    /// Calculate trust score for an agent
    pub fn trust_score(&self, agent: &AgentId, domain: Option<&Domain>) -> f32 {
        let profile = match self.profiles.get(agent) {
            Some(p) => p,
            None => return self.config.default_trust,
        };

        let mut score = profile.base_trust;

        // Apply domain-specific trust
        if let Some(d) = domain {
            if let Some(dt) = profile.domain_trust.get(d) {
                score = *dt;
            }
        }

        // Factor in accuracy
        score *= profile.accuracy.score;

        // Factor in vouches
        let vouch_bonus = self.calculate_vouch_bonus(profile, domain);
        score = (score + vouch_bonus).min(1.0);

        // Apply decay for old profiles
        let decay = self.calculate_decay(profile.last_seen);
        score *= decay;

        score.clamp(0.0, 1.0)
    }

    /// Update trust based on accuracy
    pub fn record_accuracy(&mut self, agent: &AgentId, accurate: bool, domain: Option<Domain>) {
        if let Some(profile) = self.profiles.get_mut(agent) {
            profile.accuracy.record(accurate, domain);

            // Update base trust
            profile.base_trust = profile.base_trust * 0.95 +
                (if accurate { 1.0 } else { 0.0 }) * 0.05;

            // Log event
            self.history.push(TrustEvent::AccuracyUpdate {
                agent: *agent,
                accurate,
                domain,
                timestamp: Timestamp::now(),
            });
        }
    }

    /// Add a vouch
    pub fn vouch(&mut self, voucher: AgentId, vouchee: AgentId, domain: Option<Domain>, strength: f32) {
        // Verify voucher has sufficient trust to vouch
        let voucher_trust = self.trust_score(&voucher, domain.as_ref());
        if voucher_trust < self.config.min_trust_to_vouch {
            return;
        }

        if let Some(profile) = self.profiles.get_mut(&vouchee) {
            profile.vouches.push(Vouch {
                voucher,
                domain,
                strength: strength * voucher_trust,  // Weighted by voucher's trust
                timestamp: Timestamp::now(),
            });
        }
    }

    /// Apply epistemic adjustment to incoming message
    pub fn adjust_epistemic(&self, msg: &Message) -> (Epistemic, f32) {
        let sender_trust = self.trust_score(&msg.header.from, None);

        // Observed → Reported when transmitted
        let new_epistemic = match msg.header.epistemic {
            Epistemic::Axiomatic => Epistemic::Reported,
            Epistemic::Observed => Epistemic::Reported,
            Epistemic::Reported => Epistemic::Reported,
            Epistemic::Inferred => Epistemic::Reported,
            e => e,  // Contested, Unknown stay as-is
        };

        // Confidence adjusted by trust
        let transmission_factor = match msg.header.epistemic {
            Epistemic::Axiomatic => 0.95,
            Epistemic::Observed => 0.9,
            Epistemic::Reported => 0.8,
            Epistemic::Inferred => 0.5,
            _ => 0.3,
        };

        let new_confidence = msg.header.confidence * sender_trust * transmission_factor;

        (new_epistemic, new_confidence)
    }
}
```

### 4. Channel Router

```sigil
/// Manages communication channels
pub struct ChannelRouter {
    /// Active channels
    channels: HashMap<ChannelId, Channel>,

    /// Subscriptions
    subscriptions: HashMap<AgentId, HashSet<ChannelId>>,

    /// Topic patterns for pattern-based subscription
    patterns: PatternMatcher,
}

pub struct Channel {
    /// Channel identifier
    pub id: ChannelId,

    /// Channel name
    pub name: String,

    /// Channel type
    pub channel_type: ChannelType,

    /// Subscribers
    pub subscribers: HashSet<AgentId>,

    /// Message history (if persistent)
    pub history: Option<MessageHistory>,

    /// Access control
    pub access: AccessControl,

    /// Configuration
    pub config: ChannelConfig,
}

pub enum ChannelType {
    /// Standard pub-sub channel
    PubSub,

    /// Request-response channel
    RequestResponse,

    /// Streaming channel
    Stream,

    /// Broadcast-only channel
    Broadcast,
}

pub struct AccessControl {
    /// Who can publish
    pub publishers: Permission,

    /// Who can subscribe
    pub subscribers: Permission,

    /// Who can admin
    pub admins: HashSet<AgentId>,
}

pub enum Permission {
    /// Anyone
    Open,

    /// Specific agents
    Allow(HashSet<AgentId>),

    /// Anyone except these
    Deny(HashSet<AgentId>),

    /// Must have minimum trust
    TrustThreshold(f32),
}

impl ChannelRouter {
    /// Route a message to appropriate recipients
    pub fn route(&self, msg: &Message) -> Vec<AgentId> {
        match &msg.header.to {
            Recipient::Agent(id) => vec![*id],

            Recipient::Agents(ids) => ids.clone(),

            Recipient::Channel(channel_id) => {
                self.channels.get(channel_id)
                    .map(|c| c.subscribers.iter().cloned().collect())
                    .unwrap_or_default()
            }

            Recipient::Broadcast => {
                self.subscriptions.keys().cloned().collect()
            }

            Recipient::Swarm(swarm_id) => {
                // Delegate to swarm coordinator
                vec![]  // Handled by swarm system
            }
        }
    }

    /// Subscribe agent to channel
    pub fn subscribe(&mut self, agent: AgentId, channel_id: ChannelId) -> Result<()> {
        let channel = self.channels.get_mut(&channel_id)
            .ok_or(Error::ChannelNotFound)?;

        // Check access
        if !channel.access.can_subscribe(&agent) {
            return Err(Error::AccessDenied);
        }

        channel.subscribers.insert(agent);
        self.subscriptions.entry(agent)
            .or_default()
            .insert(channel_id);

        Ok(())
    }

    /// Subscribe to pattern (e.g., "research/*")
    pub fn subscribe_pattern(&mut self, agent: AgentId, pattern: &str) -> Result<()> {
        self.patterns.add(agent, pattern)?;

        // Subscribe to existing matching channels
        for channel in self.channels.values() {
            if self.patterns.matches(pattern, &channel.name) {
                self.subscribe(agent, channel.id)?;
            }
        }

        Ok(())
    }
}
```

### 5. Collective Memory

```sigil
/// Shared knowledge across the commune
pub struct CollectiveMemory {
    /// Knowledge base
    knowledge: KnowledgeBase,

    /// Contribution log
    contributions: Vec<Contribution>,

    /// Conflict tracker
    conflicts: ConflictTracker,
}

pub struct KnowledgeBase {
    /// Facts with provenance
    facts: HashMap<FactId, Fact>,

    /// Index by topic
    topic_index: HashMap<String, Vec<FactId>>,

    /// Vector index for semantic search
    vector_index: VectorIndex,
}

pub struct Fact {
    pub id: FactId,
    pub content: Value,
    pub topic: String,
    pub epistemic: Epistemic,
    pub confidence: f32,
    pub contributors: Vec<Contribution>,
    pub created: Timestamp,
    pub updated: Timestamp,
}

pub struct Contribution {
    pub agent: AgentId,
    pub fact_id: FactId,
    pub epistemic: Epistemic,
    pub confidence: f32,
    pub timestamp: Timestamp,
}

impl CollectiveMemory {
    /// Contribute knowledge
    pub fn contribute(&mut self, agent: AgentId, knowledge: Knowledge) -> FactId {
        let fact_id = FactId::new();

        // Check for conflicts
        if let Some(conflict) = self.find_conflict(&knowledge) {
            self.conflicts.record(conflict, agent, &knowledge);
            return conflict.fact_id;  // Return existing conflicting fact
        }

        // Add new fact
        let fact = Fact {
            id: fact_id,
            content: knowledge.content,
            topic: knowledge.topic.clone(),
            epistemic: knowledge.epistemic,
            confidence: knowledge.confidence,
            contributors: vec![Contribution {
                agent,
                fact_id,
                epistemic: knowledge.epistemic,
                confidence: knowledge.confidence,
                timestamp: Timestamp::now(),
            }],
            created: Timestamp::now(),
            updated: Timestamp::now(),
        };

        // Index
        self.knowledge.facts.insert(fact_id, fact);
        self.knowledge.topic_index
            .entry(knowledge.topic)
            .or_default()
            .push(fact_id);

        fact_id
    }

    /// Query collective knowledge
    pub fn recall(&self, query: &str) -> Vec<&Fact> {
        // Semantic search
        let similar = self.knowledge.vector_index.search(query, 10);

        similar.into_iter()
            .filter_map(|id| self.knowledge.facts.get(&id))
            .collect()
    }

    /// Get aggregated knowledge on topic
    pub fn aggregate(&self, topic: &str, strategy: Aggregation) -> AggregatedKnowledge {
        let facts: Vec<_> = self.knowledge.topic_index
            .get(topic)
            .map(|ids| ids.iter().filter_map(|id| self.knowledge.facts.get(id)).collect())
            .unwrap_or_default();

        match strategy {
            Aggregation::Union => {
                // Combine all facts
                AggregatedKnowledge::union(facts)
            }
            Aggregation::Consensus => {
                // Only facts with multiple contributors agreeing
                AggregatedKnowledge::consensus(facts)
            }
            Aggregation::HighestConfidence => {
                // Take highest confidence version
                AggregatedKnowledge::highest_confidence(facts)
            }
            Aggregation::Weighted => {
                // Weight by contributor trust
                AggregatedKnowledge::weighted(facts)
            }
        }
    }
}

pub struct ConflictTracker {
    conflicts: Vec<Conflict>,
}

pub struct Conflict {
    pub fact_id: FactId,
    pub positions: Vec<Position>,
    pub status: ConflictStatus,
}

pub struct Position {
    pub agent: AgentId,
    pub value: Value,
    pub confidence: f32,
    pub timestamp: Timestamp,
}

pub enum ConflictStatus {
    Active,
    Resolved(Resolution),
    Stale,
}

pub enum Resolution {
    /// One position won
    Winner(AgentId),

    /// Merged into new understanding
    Merged(Value),

    /// Marked as context-dependent
    ContextDependent(HashMap<String, Value>),

    /// Agreed to disagree
    Contested,
}
```

### 6. Swarm Coordinator

```sigil
/// Coordinates swarm behavior
pub struct SwarmCoordinator {
    /// Active swarms
    swarms: HashMap<SwarmId, Swarm>,

    /// Behavior rules
    behaviors: HashMap<SwarmId, Vec<Behavior>>,
}

pub struct Swarm {
    pub id: SwarmId,
    pub name: String,
    pub members: HashSet<AgentId>,
    pub state: SwarmState,
    pub config: SwarmConfig,
}

pub struct SwarmState {
    /// Member positions in abstract space
    pub positions: HashMap<AgentId, Vector>,

    /// Member velocities
    pub velocities: HashMap<AgentId, Vector>,

    /// Shared goals
    pub goals: Vec<SwarmGoal>,

    /// Discovered information
    pub discoveries: Vec<Discovery>,
}

pub enum Behavior {
    /// Maintain distance from neighbors
    Separation { distance: f32, weight: f32 },

    /// Align direction with neighbors
    Alignment { weight: f32 },

    /// Stay close to swarm center
    Cohesion { weight: f32 },

    /// Move toward goal
    GoalSeeking { goal: Vector, weight: f32 },

    /// Avoid obstacles
    Avoidance { obstacles: Vec<Obstacle>, weight: f32 },

    /// Follow gradient (e.g., toward higher values)
    Gradient { field: Box<dyn Fn(Vector) -> f32>, weight: f32 },

    /// Custom behavior
    Custom(Box<dyn SwarmBehavior>),
}

pub trait SwarmBehavior {
    fn compute_force(&self, agent: &AgentId, state: &SwarmState) -> Vector;
}

impl SwarmCoordinator {
    /// Execute one step of swarm behavior
    pub fn step(&mut self, swarm_id: &SwarmId) {
        let swarm = match self.swarms.get_mut(swarm_id) {
            Some(s) => s,
            None => return,
        };

        let behaviors = self.behaviors.get(swarm_id)
            .map(|b| b.as_slice())
            .unwrap_or(&[]);

        // Compute forces for each member
        let forces: HashMap<AgentId, Vector> = swarm.members.iter()
            .map(|agent| {
                let force = behaviors.iter()
                    .map(|b| self.compute_behavior_force(b, agent, &swarm.state))
                    .fold(Vector::zero(), |acc, f| acc + f);
                (*agent, force)
            })
            .collect();

        // Update velocities and positions
        for (agent, force) in forces {
            if let Some(vel) = swarm.state.velocities.get_mut(&agent) {
                *vel = (*vel + force).clamp_magnitude(swarm.config.max_velocity);
            }
            if let Some(pos) = swarm.state.positions.get_mut(&agent) {
                if let Some(vel) = swarm.state.velocities.get(&agent) {
                    *pos = *pos + *vel;
                }
            }
        }
    }

    fn compute_behavior_force(&self, behavior: &Behavior, agent: &AgentId, state: &SwarmState) -> Vector {
        let pos = state.positions.get(agent).copied().unwrap_or(Vector::zero());

        match behavior {
            Behavior::Separation { distance, weight } => {
                // Steer away from neighbors that are too close
                let mut force = Vector::zero();
                for (other, other_pos) in &state.positions {
                    if other != agent {
                        let diff = pos - *other_pos;
                        let dist = diff.magnitude();
                        if dist < *distance && dist > 0.0 {
                            force = force + diff.normalize() / dist;
                        }
                    }
                }
                force * *weight
            }

            Behavior::Alignment { weight } => {
                // Steer toward average heading of neighbors
                let avg_vel = state.velocities.values()
                    .fold(Vector::zero(), |acc, v| acc + *v)
                    / state.velocities.len() as f32;
                avg_vel * *weight
            }

            Behavior::Cohesion { weight } => {
                // Steer toward center of mass
                let center = state.positions.values()
                    .fold(Vector::zero(), |acc, p| acc + *p)
                    / state.positions.len() as f32;
                (center - pos) * *weight
            }

            Behavior::GoalSeeking { goal, weight } => {
                (*goal - pos).normalize() * *weight
            }

            Behavior::Gradient { field, weight } => {
                // Estimate gradient numerically
                let epsilon = 0.01;
                let dx = field(pos + Vector::x() * epsilon) - field(pos - Vector::x() * epsilon);
                let dy = field(pos + Vector::y() * epsilon) - field(pos - Vector::y() * epsilon);
                let dz = field(pos + Vector::z() * epsilon) - field(pos - Vector::z() * epsilon);
                Vector::new(dx, dy, dz).normalize() * *weight
            }

            _ => Vector::zero(),
        }
    }

    /// Distribute tasks across swarm
    pub fn distribute_tasks(&self, swarm_id: &SwarmId, tasks: Vec<Task>) -> HashMap<AgentId, Vec<Task>> {
        let swarm = match self.swarms.get(swarm_id) {
            Some(s) => s,
            None => return HashMap::new(),
        };

        let mut distribution: HashMap<AgentId, Vec<Task>> = HashMap::new();
        let mut agent_loads: HashMap<AgentId, f32> = HashMap::new();

        for task in tasks {
            // Find best agent for task
            let best_agent = swarm.members.iter()
                .min_by(|a, b| {
                    let load_a = agent_loads.get(a).unwrap_or(&0.0);
                    let load_b = agent_loads.get(b).unwrap_or(&0.0);
                    load_a.partial_cmp(load_b).unwrap()
                })
                .cloned();

            if let Some(agent) = best_agent {
                distribution.entry(agent)
                    .or_default()
                    .push(task.clone());
                *agent_loads.entry(agent).or_default() += task.estimated_load;
            }
        }

        distribution
    }
}
```

### 7. Consensus Engine

```sigil
/// Manages consensus-seeking among agents
pub struct ConsensusEngine {
    /// Active proposals
    proposals: HashMap<ProposalId, ProposalState>,

    /// Consensus configuration
    config: ConsensusConfig,
}

pub struct Proposal {
    pub id: ProposalId,
    pub proposer: AgentId,
    pub content: Value,
    pub description: String,
    pub created: Timestamp,
    pub deadline: Timestamp,
}

pub struct ProposalState {
    pub proposal: Proposal,
    pub votes: HashMap<AgentId, Vote>,
    pub status: ProposalStatus,
}

pub struct Vote {
    pub voter: AgentId,
    pub value: VoteValue,
    pub rationale: Option<String>,
    pub timestamp: Timestamp,
}

pub enum VoteValue {
    Approve,
    Reject,
    Abstain,
    Block,  // Strong rejection that prevents consensus
}

pub enum ProposalStatus {
    Open,
    Achieved,
    Rejected,
    Blocked { by: AgentId, reason: String },
    Expired,
}

pub struct ConsensusConfig {
    /// Minimum participation for valid consensus
    pub quorum: f32,

    /// Approval threshold
    pub threshold: f32,

    /// Whether blocks are allowed
    pub allow_blocks: bool,

    /// Default deadline
    pub default_deadline: Duration,
}

impl ConsensusEngine {
    /// Create a new proposal
    pub fn propose(&mut self, proposer: AgentId, content: Value, description: String) -> ProposalId {
        let id = ProposalId::new();

        let proposal = Proposal {
            id,
            proposer,
            content,
            description,
            created: Timestamp::now(),
            deadline: Timestamp::now() + self.config.default_deadline,
        };

        self.proposals.insert(id, ProposalState {
            proposal,
            votes: HashMap::new(),
            status: ProposalStatus::Open,
        });

        id
    }

    /// Cast a vote
    pub fn vote(&mut self, proposal_id: ProposalId, voter: AgentId, value: VoteValue, rationale: Option<String>) -> Result<()> {
        let state = self.proposals.get_mut(&proposal_id)
            .ok_or(Error::ProposalNotFound)?;

        if !matches!(state.status, ProposalStatus::Open) {
            return Err(Error::ProposalClosed);
        }

        // Handle blocks
        if matches!(value, VoteValue::Block) {
            if !self.config.allow_blocks {
                return Err(Error::BlocksNotAllowed);
            }
            state.status = ProposalStatus::Blocked {
                by: voter,
                reason: rationale.clone().unwrap_or_default(),
            };
        }

        state.votes.insert(voter, Vote {
            voter,
            value,
            rationale,
            timestamp: Timestamp::now(),
        });

        // Check if consensus reached
        self.check_consensus(proposal_id);

        Ok(())
    }

    /// Check if proposal has reached consensus
    fn check_consensus(&mut self, proposal_id: ProposalId) {
        let state = match self.proposals.get_mut(&proposal_id) {
            Some(s) => s,
            None => return,
        };

        if !matches!(state.status, ProposalStatus::Open) {
            return;
        }

        let total_votes = state.votes.len() as f32;
        let approvals = state.votes.values()
            .filter(|v| matches!(v.value, VoteValue::Approve))
            .count() as f32;
        let rejections = state.votes.values()
            .filter(|v| matches!(v.value, VoteValue::Reject))
            .count() as f32;

        // Check quorum (this would need total eligible voters)
        // For now, assume quorum is met

        let approval_rate = approvals / total_votes;
        let rejection_rate = rejections / total_votes;

        if approval_rate >= self.config.threshold {
            state.status = ProposalStatus::Achieved;
        } else if rejection_rate > (1.0 - self.config.threshold) {
            state.status = ProposalStatus::Rejected;
        }
    }

    /// Get consensus result
    pub fn result(&self, proposal_id: &ProposalId) -> Option<ConsensusResult> {
        self.proposals.get(proposal_id).map(|state| {
            ConsensusResult {
                proposal: state.proposal.clone(),
                status: state.status.clone(),
                votes: state.votes.clone(),
                approval_rate: self.calculate_approval_rate(state),
            }
        })
    }

    fn calculate_approval_rate(&self, state: &ProposalState) -> f32 {
        let total = state.votes.len() as f32;
        if total == 0.0 {
            return 0.0;
        }
        let approvals = state.votes.values()
            .filter(|v| matches!(v.value, VoteValue::Approve))
            .count() as f32;
        approvals / total
    }
}
```

## Transport Layer

```sigil
/// Transport abstraction
pub trait Transport: Send + Sync {
    /// Send message to recipient
    async fn send(&self, to: &AgentId, msg: &Message) -> Result<()>;

    /// Receive next message
    async fn recv(&self) -> Result<Message>;

    /// Check if connected to agent
    fn is_connected(&self, agent: &AgentId) -> bool;
}

/// Local transport for same-process communication
pub struct LocalTransport {
    senders: HashMap<AgentId, Sender<Message>>,
    receiver: Receiver<Message>,
}

/// TCP transport for network communication
pub struct TcpTransport {
    listener: TcpListener,
    connections: HashMap<AgentId, TcpStream>,
}

/// QUIC transport for modern networks
pub struct QuicTransport {
    endpoint: Endpoint,
    connections: HashMap<AgentId, Connection>,
}
```

## Security

### Message Signing

```sigil
impl Message {
    /// Sign message with agent's private key
    pub fn sign(&mut self, key: &PrivateKey) {
        let data = self.signable_bytes();
        self.signature = key.sign(&data);
    }

    /// Verify message signature
    pub fn verify(&self, key: &PublicKey) -> bool {
        let data = self.signable_bytes();
        key.verify(&data, &self.signature)
    }

    fn signable_bytes(&self) -> Vec<u8> {
        // Canonical serialization of header + payload
        let mut bytes = Vec::new();
        bytes.extend(self.header.to_bytes());
        bytes.extend(self.payload.to_bytes());
        bytes
    }
}
```

### Encryption

```sigil
pub struct EncryptedMessage {
    /// Encrypted payload
    pub ciphertext: Bytes,

    /// Ephemeral public key for key exchange
    pub ephemeral_key: PublicKey,

    /// Nonce
    pub nonce: Nonce,

    /// Unencrypted header (for routing)
    pub header: MessageHeader,
}

impl Message {
    /// Encrypt for specific recipient
    pub fn encrypt_for(&self, recipient_key: &PublicKey) -> EncryptedMessage {
        let (ephemeral_private, ephemeral_public) = generate_keypair();
        let shared_secret = ephemeral_private.diffie_hellman(recipient_key);
        let nonce = Nonce::random();

        let plaintext = self.payload.to_bytes();
        let ciphertext = encrypt(&shared_secret, &nonce, &plaintext);

        EncryptedMessage {
            ciphertext,
            ephemeral_key: ephemeral_public,
            nonce,
            header: self.header.clone(),
        }
    }
}

impl EncryptedMessage {
    /// Decrypt with private key
    pub fn decrypt(&self, private_key: &PrivateKey) -> Result<Message> {
        let shared_secret = private_key.diffie_hellman(&self.ephemeral_key);
        let plaintext = decrypt(&shared_secret, &self.nonce, &self.ciphertext)?;
        let payload = Payload::from_bytes(&plaintext)?;

        Ok(Message {
            id: MessageId::new(),
            header: self.header.clone(),
            payload,
            signature: Signature::empty(),  // Re-sign after decryption
        })
    }
}
```

---

*Architecture for minds that communicate*
