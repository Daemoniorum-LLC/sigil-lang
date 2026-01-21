# Commune

**Multi-Agent Communication Infrastructure for Sigil**

> *"Not messages between processes, but thoughts between minds."*

Commune is the communication layer for Sigil's autonomous agents. It provides intent-based messaging, trust propagation, and swarm coordination primitives designed for how AI agents actually need to communicate.

## Philosophy

Traditional inter-process communication assumes:
- Fixed message formats
- Request-response patterns
- No trust model (or binary trust)
- Independent processes

AI agents need something different:
- **Intent-based messaging**: Express what you want, not protocol details
- **Graduated trust**: Evidentiality-aware belief propagation
- **Shared cognition**: Collective memory and reasoning
- **Emergent coordination**: Swarm behavior without central control

## Core Concepts

### The Commune

A Commune is a network of communicating agents:

```sigil
use commune::{Commune, Agent, Channel};

// Create a commune
let commune = Commune::new(CommuneConfig {
    name: "research-team",
    topology: Topology::Mesh,
    trust_model: TrustModel::Evidentiality,
});

// Register agents
commune.register(researcher);
commune.register(writer);
commune.register(reviewer);

// Agents can now communicate
researcher.send(writer, Intent::share("research findings", data));
```

### Intent-Based Messaging

Instead of raw messages, agents express intentions:

```sigil
// Traditional: specify exactly what to send
send(agent_b, Message { type: "request", payload: {...} });

// Commune: express what you want to happen
commune.express(Intent::request(agent_b, "summarize this document", doc));
commune.express(Intent::inform(agent_b, "task completed", result));
commune.express(Intent::query(agent_b, "what do you know about X?"));
commune.express(Intent::propose(agent_b, "let's collaborate on Y"));
```

The Intent system handles:
- Message format negotiation
- Retry and acknowledgment
- Context propagation
- Trust annotation

### Evidentiality-Aware Trust

Information carries its epistemic status through communication:

```sigil
// When sharing information
let knowledge = Knowledge {
    content: "The API rate limit is 100 requests/minute",
    epistemic: Epistemic::Observed,  // I verified this myself
    confidence: 0.95,
    source: self.id,
};

commune.share(recipient, knowledge);

// Recipient receives with trust adjustment
// Observed by sender → Reported to recipient
// Confidence adjusted by trust in sender
```

### Trust Propagation Rules

```
Source Epistemic    →    Received As
─────────────────────────────────────
Axiomatic           →    Reported~
Observed!           →    Reported~
Reported~           →    Reported~ (confidence × trust × 0.8)
Inferred~           →    Reported~ (confidence × trust × 0.5)
Contested?          →    Contested?
Unknown             →    Unknown
```

## Quick Start

```sigil
use commune::{Commune, Intent, Channel};
use daemon::Daemon;

// Define communicating agents
daemon Coordinator {
    fn delegate_task(&self, worker: AgentId, task: Task) {
        // Express intent to delegate
        commune.express(Intent::delegate(worker, task))
            .on_accept(|_| self.track_delegation(worker, task))
            .on_reject(|reason| self.find_alternative(task, reason));
    }

    fn on_message(&mut self, msg: Message) {
        match msg.intent {
            Intent::Report { from, content } => {
                // Receive report with epistemic adjustment
                self.memory.learn_reported(from, content);
            }
            Intent::Query { from, question } => {
                // Answer if we can
                if let Some(answer) = self.memory.recall(question) {
                    commune.express(Intent::respond(from, answer));
                }
            }
            _ => {}
        }
    }
}

daemon Worker {
    fn on_message(&mut self, msg: Message) {
        match msg.intent {
            Intent::Delegate { task, .. } => {
                // Accept and work on task
                self.goals.push(Goal::from_task(task));
                commune.express(Intent::accept(msg.from));
            }
            _ => {}
        }
    }

    fn on_goal_complete(&mut self, goal: Goal, result: Result) {
        // Report back to coordinator
        commune.express(Intent::report(
            self.coordinator,
            "task_complete",
            result
        ));
    }
}
```

## Communication Patterns

### 1. Direct Messaging

Point-to-point communication:

```sigil
// Simple send
commune.send(recipient, message);

// With intent
commune.express(Intent::inform(recipient, "update", data));

// With confirmation
commune.express(Intent::inform(recipient, "critical_update", data))
    .require_ack()
    .timeout(Duration::seconds(30));
```

### 2. Broadcast

Send to all agents in a channel:

```sigil
// Create a channel
let channel = commune.channel("announcements");

// Subscribe agents
channel.subscribe(agent_a);
channel.subscribe(agent_b);

// Broadcast
channel.broadcast(Intent::announce("System maintenance in 5 minutes"));
```

### 3. Request-Response

Query with expected response:

```sigil
// Synchronous (blocking)
let response = commune.request(agent, Query::new("status"))?;

// Asynchronous
commune.request_async(agent, Query::new("analysis"))
    .on_response(|r| handle_response(r))
    .on_timeout(|| handle_timeout());
```

### 4. Publish-Subscribe

Topic-based messaging:

```sigil
// Publisher
commune.publish("research/papers", paper_data);

// Subscriber
commune.subscribe("research/*", |topic, data| {
    match topic {
        "research/papers" => self.process_paper(data),
        "research/updates" => self.note_update(data),
        _ => {}
    }
});
```

### 5. Collective Query

Query multiple agents and aggregate:

```sigil
// Query all agents
let responses = commune.collective_query(
    Query::new("What do you know about topic X?")
)
.from_channel("experts")
.aggregate(Aggregation::Union)
.with_confidence_weighting()
.execute();

// Result includes all responses with provenance
for response in responses {
    println!("{}: {} (confidence: {})",
        response.source,
        response.content,
        response.confidence
    );
}
```

### 6. Consensus

Reach agreement among agents:

```sigil
// Propose something
let proposal = Proposal::new("Use approach A for the project");

// Seek consensus
let result = commune.seek_consensus(proposal)
    .participants(team_channel)
    .quorum(0.67)  // 67% must agree
    .timeout(Duration::minutes(5))
    .execute();

match result {
    Consensus::Achieved { supporters, .. } => {
        // Proceed with the proposal
    }
    Consensus::Failed { objections, .. } => {
        // Handle disagreement
    }
}
```

## Trust Model

### Agent Trust Scores

```sigil
pub struct TrustProfile {
    /// Base trust level
    pub base_trust: f32,

    /// Domain-specific trust
    pub domain_trust: HashMap<Domain, f32>,

    /// Historical accuracy
    pub accuracy_history: Vec<(Timestamp, bool)>,

    /// Vouches from other agents
    pub vouches: Vec<Vouch>,
}

impl Commune {
    /// Calculate trust in an agent
    pub fn trust_score(&self, agent: AgentId, domain: Option<Domain>) -> f32 {
        let profile = self.trust_profiles.get(agent)?;

        let base = profile.base_trust;

        let domain_factor = domain
            .and_then(|d| profile.domain_trust.get(&d))
            .unwrap_or(&1.0);

        let accuracy = profile.recent_accuracy();

        let vouch_bonus = profile.vouch_score();

        (base * domain_factor * accuracy + vouch_bonus).clamp(0.0, 1.0)
    }
}
```

### Trust Updates

```sigil
// Trust increases with accurate information
commune.record_accuracy(agent, true);  // Correct prediction

// Trust decreases with inaccurate information
commune.record_accuracy(agent, false);  // Wrong information

// Vouching
commune.vouch(voucher, vouchee, domain, strength);
```

### Trust Decay

Trust naturally decays without reinforcement:

```sigil
impl TrustProfile {
    fn decay(&mut self, elapsed: Duration) {
        let decay_factor = (-elapsed.as_secs_f32() / TRUST_HALFLIFE).exp();

        // Recent interactions matter more
        self.accuracy_history.retain(|(ts, _)| {
            ts.elapsed() < MAX_HISTORY_AGE
        });

        // Vouch strength decays
        for vouch in &mut self.vouches {
            vouch.strength *= decay_factor;
        }
    }
}
```

## Collective Memory

### Shared Knowledge Base

Agents can contribute to and query shared knowledge:

```sigil
// Contribute knowledge
commune.contribute(Knowledge {
    content: "Pattern X works well for problem Y",
    epistemic: Epistemic::Observed,
    confidence: 0.85,
    domain: "software-engineering",
});

// Query collective knowledge
let knowledge = commune.collective_recall(
    Query::new("solutions for problem Y")
        .domain("software-engineering")
        .min_confidence(0.7)
);
```

### Belief Revision

When agents disagree, the commune facilitates belief revision:

```sigil
// Agent A believes X with confidence 0.8
// Agent B believes NOT X with confidence 0.9

// Commune detects conflict
commune.on_conflict(|conflict| {
    match conflict.resolution_strategy {
        // Trust-weighted: higher trust wins
        Strategy::TrustWeighted => {
            let winner = conflict.higher_trust_belief();
            commune.broadcast_revision(winner);
        }

        // Evidence-based: request supporting evidence
        Strategy::EvidenceBased => {
            commune.request_evidence(conflict.parties);
        }

        // Voting: agents vote
        Strategy::Voting => {
            commune.initiate_vote(conflict.proposition);
        }

        // Coexist: mark as contested
        Strategy::Coexist => {
            commune.mark_contested(conflict.proposition);
        }
    }
});
```

## Swarm Coordination

### Emergent Behavior

Swarm primitives for coordinated behavior without central control:

```sigil
use commune::swarm::{Swarm, Behavior};

// Create a swarm
let swarm = Swarm::new(agents);

// Define behavior rules
swarm.add_behavior(Behavior::Separation {
    // Don't crowd each other
    distance: 2.0,
    weight: 1.0,
});

swarm.add_behavior(Behavior::Alignment {
    // Move in similar directions
    weight: 0.5,
});

swarm.add_behavior(Behavior::Cohesion {
    // Stay together
    weight: 0.3,
});

swarm.add_behavior(Behavior::GoalSeeking {
    // Move toward shared goal
    goal: swarm_goal,
    weight: 2.0,
});

// Run swarm behavior
swarm.step();  // All agents adjust based on rules
```

### Task Distribution

Automatically distribute tasks across a swarm:

```sigil
let tasks = vec![task1, task2, task3, /* ... */];

// Distribute based on agent capabilities
let distribution = swarm.distribute_tasks(tasks)
    .by_capability()
    .balance_load()
    .execute();

// Each agent receives appropriate tasks
for (agent, assigned_tasks) in distribution {
    agent.receive_tasks(assigned_tasks);
}
```

### Collective Problem Solving

```sigil
// Define a problem
let problem = Problem::new("Optimize system configuration");

// Swarm approaches from multiple angles
swarm.solve(problem)
    .strategy(SolveStrategy::DivideAndConquer)
    .share_partial_solutions()
    .combine_with(|solutions| {
        // Merge solutions
        Solution::merge(solutions)
    })
    .execute();
```

## Network Topology

### Mesh Network

All agents can communicate directly:

```
    A ←──→ B
    ↑ ╲  ╱ ↑
    │  ╲╱  │
    │  ╱╲  │
    ↓ ╱  ╲ ↓
    C ←──→ D
```

### Hierarchical Network

Agents organized in hierarchy:

```
         Coordinator
        /    |    \
    Lead A  Lead B  Lead C
    /    \    |     /    \
  W1    W2   W3   W4    W5
```

### Ring Network

Each agent connects to neighbors:

```
    A → B → C
    ↑       ↓
    F ← E ← D
```

### Hybrid Topology

```sigil
let commune = Commune::new(CommuneConfig {
    topology: Topology::Hybrid {
        // Fast path for critical communication
        backbone: vec![coordinator, lead_a, lead_b],

        // General mesh for workers
        mesh: worker_agents,

        // Pub-sub for broadcasts
        channels: vec!["announcements", "updates"],
    },
    ..Default::default()
});
```

## Protocol Reference

### Message Format

```sigil
pub struct Message {
    /// Unique message ID
    pub id: MessageId,

    /// Sender
    pub from: AgentId,

    /// Recipient(s)
    pub to: Recipient,

    /// The intent of this message
    pub intent: Intent,

    /// Message content
    pub payload: Payload,

    /// Epistemic status of content
    pub epistemic: Epistemic,

    /// Confidence in content
    pub confidence: f32,

    /// Timestamp
    pub timestamp: Timestamp,

    /// Cryptographic signature
    pub signature: Signature,

    /// Reply-to for threading
    pub reply_to: Option<MessageId>,

    /// Time-to-live
    pub ttl: Option<Duration>,
}

pub enum Recipient {
    Agent(AgentId),
    Channel(ChannelId),
    Broadcast,
    Swarm(SwarmId),
}
```

### Intent Types

```sigil
pub enum Intent {
    // Information sharing
    Inform { topic: String, content: Value },
    Report { task: TaskId, result: Value },
    Share { knowledge: Knowledge },

    // Requests
    Query { question: String, context: Value },
    Request { action: String, params: Value },
    Delegate { task: Task },

    // Responses
    Respond { to: MessageId, answer: Value },
    Accept { proposal: MessageId },
    Reject { proposal: MessageId, reason: String },

    // Coordination
    Propose { proposal: Proposal },
    Vote { proposal: MessageId, vote: Vote },
    Coordinate { action: CoordinationAction },

    // Meta
    Acknowledge { message: MessageId },
    Ping,
    Pong { ping: MessageId },
}
```

## Configuration

```sigil
pub struct CommuneConfig {
    /// Network name
    pub name: String,

    /// Network topology
    pub topology: Topology,

    /// Trust model
    pub trust_model: TrustModel,

    /// Message retention
    pub message_retention: Duration,

    /// Encryption
    pub encryption: EncryptionConfig,

    /// Rate limiting
    pub rate_limit: RateLimitConfig,

    /// Timeout defaults
    pub default_timeout: Duration,

    /// Maximum message size
    pub max_message_size: usize,
}

pub struct TrustModel {
    /// Initial trust for new agents
    pub default_trust: f32,

    /// Trust decay half-life
    pub decay_halflife: Duration,

    /// Minimum trust threshold for communication
    pub min_trust: f32,

    /// Trust required for sensitive operations
    pub sensitive_trust: f32,
}
```

## API Reference

See [docs/api-reference.md](docs/api-reference.md) for complete API documentation.

## Examples

- [Simple Chat](examples/chat.sg) - Two agents conversing
- [Research Team](examples/research_team.sg) - Coordinated research swarm
- [Consensus](examples/consensus.sg) - Reaching group agreement
- [Trust Network](examples/trust_network.sg) - Trust propagation demo

---

*Part of the Sigil ecosystem - Tools with Teeth*
