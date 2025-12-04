# Sigil Agent Infrastructure

**The Complete AI Agent Stack**

> *"A language for artificial minds, with infrastructure to match."*

Sigil provides a complete infrastructure stack for building autonomous AI agents:

| Layer | Component | Purpose |
|-------|-----------|---------|
| **Memory** | Engram | Persistent memory with epistemic tracking |
| **Runtime** | Daemon | Autonomous agent execution |
| **Communication** | Commune | Multi-agent messaging and coordination |
| **Reasoning** | Omen | Planning and belief revision |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SIGIL AGENT STACK                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                          YOUR AGENT CODE                              │  │
│  │                                                                       │  │
│  │   daemon MyAgent {                                                    │  │
│  │       fn on_message(&mut self, msg) { ... }                          │  │
│  │       fn deliberate(&self, context) -> Action { ... }                │  │
│  │   }                                                                   │  │
│  │                                                                       │  │
│  └───────────────────────────────┬──────────────────────────────────────┘  │
│                                  │                                          │
│  ┌───────────────────────────────▼──────────────────────────────────────┐  │
│  │                          OMEN (Planning)                              │  │
│  │                                                                       │  │
│  │   Goals → Plans → Actions    Beliefs → Revision → Knowledge          │  │
│  │                                                                       │  │
│  └───────────────────────────────┬──────────────────────────────────────┘  │
│                                  │                                          │
│  ┌──────────────────┬────────────▼───────────┬──────────────────────────┐  │
│  │     DAEMON       │         COMMUNE        │        ENGRAM            │  │
│  │   (Runtime)      │    (Communication)     │       (Memory)           │  │
│  │                  │                        │                          │  │
│  │ Heartbeat loop   │ Intent messaging       │ Instant memory           │  │
│  │ Tool execution   │ Trust propagation      │ Episodic memory          │  │
│  │ State management │ Swarm coordination     │ Semantic memory          │  │
│  │ Lifecycle hooks  │ Collective knowledge   │ Procedural memory        │  │
│  │                  │                        │                          │  │
│  └──────────────────┴────────────────────────┴──────────────────────────┘  │
│                                  │                                          │
│  ┌───────────────────────────────▼──────────────────────────────────────┐  │
│  │                      SIGIL RUNTIME (LLVM)                             │  │
│  │                                                                       │  │
│  │   Morpheme operators │ Evidentiality types │ Native performance      │  │
│  │                                                                       │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Install Sigil with agent infrastructure
cargo install sigil-parser --features agent

# Or build from source
git clone https://github.com/Daemoniorum-LLC/sigil-lang.git
cd sigil-lang/parser
cargo build --release --features agent,llvm
```

### Hello, Agent

```sigil
use daemon::Daemon;
use engram::Engram;
use commune::Commune;
use omen::Omen;

daemon HelloAgent {
    // Memory
    memory: Engram,

    // Planner
    planner: Omen,

    fn on_init(&mut self) {
        // Initialize memory
        self.memory = Engram::new(EngramConfig::default());

        // Initialize planner with memory integration
        self.planner = Omen::new(OmenConfig::default())
            .with_memory(&self.memory);

        // Set initial goal
        self.goals.push(Goal::new("Greet the user"));
    }

    fn on_message(&mut self, msg: Message) {
        // Remember the message
        self.memory.experience(Event::message_received(msg.clone()));

        // Update planner beliefs
        self.planner.observe("received message from user");

        // Plan response
        let plan = self.planner.plan(
            Goal::new("Respond to: " + msg.content)
        )?;

        // Execute plan
        for step in plan.steps {
            self.execute(step.action)?;
        }
    }

    fn deliberate(&self, context: Context) -> Action {
        // Use planner to decide
        let goal = self.goals.current()?;
        let plan = self.planner.plan(goal)?;
        plan.steps.first()?.action.clone()
    }
}

fn main() {
    // Create agent
    let agent = HelloAgent::spawn();

    // Register with commune for communication
    let commune = Commune::local();
    commune.register(agent);

    // Run agent
    agent.run();
}
```

## Integration Patterns

### Memory-First Design

Every agent action flows through memory:

```sigil
daemon MemoryFirstAgent {
    memory: Engram,

    fn on_message(&mut self, msg: Message) {
        // 1. Experience: Record the event
        self.memory.experience(Event::from(msg));

        // 2. Attend: Focus on relevant memories
        let context = self.memory.attend(Query::relevant_to(&msg));

        // 3. Recall: Retrieve useful knowledge
        let knowledge = self.memory.recall(
            Query::semantic("similar situations")
        );

        // 4. Deliberate: Decide action using context
        let action = self.decide(context, knowledge);

        // 5. Act: Execute the action
        let result = self.execute(action);

        // 6. Learn: Update from outcome
        self.memory.learn(action, result);
    }
}
```

### Planning-Driven Behavior

Use Omen for goal-directed behavior:

```sigil
daemon PlanningAgent {
    planner: Omen,
    memory: Engram,

    fn pursue_goal(&mut self, goal: Goal) {
        // Add beliefs from memory
        for knowledge in self.memory.semantic.iter() {
            self.planner.believe(Belief::from(knowledge));
        }

        // Plan toward goal
        let plan = self.planner.plan(goal)?;

        // Assess risks
        let risks = self.planner.assess_risks(&plan);
        for risk in &risks {
            if risk.probability > 0.3 {
                println!("Warning: {}", risk.description);
            }
        }

        // Execute with monitoring
        for step in plan.steps {
            // Check preconditions
            if !self.planner.check_preconditions(&step) {
                // Replan
                let new_plan = self.planner.replan(goal, &step)?;
                return self.pursue_goal_with_plan(goal, new_plan);
            }

            // Execute step
            let result = self.execute(step.action)?;

            // Update beliefs from result
            self.planner.integrate_evidence(Evidence::from(result));

            // Store in memory
            self.memory.learn(&step.action, &result);
        }
    }
}
```

### Multi-Agent Coordination

Use Commune for agent communication:

```sigil
// Coordinator agent
daemon Coordinator {
    fn delegate_task(&mut self, task: Task) {
        // Find capable worker
        let workers = commune.query_channel("workers")
            .with_capability(task.required_capability)
            .execute();

        // Select based on load and trust
        let worker = workers.into_iter()
            .min_by_key(|w| w.current_load)
            .filter(|w| commune.trust_score(w.id) > 0.7)
            .expect("No suitable worker");

        // Delegate with intent
        commune.express(Intent::delegate(worker.id, task))
            .on_accept(|_| self.track_delegation(worker.id, task))
            .on_reject(|reason| self.find_alternative(task, reason));
    }

    fn on_message(&mut self, msg: Message) {
        match msg.intent {
            Intent::Report { task_id, result, .. } => {
                // Record result in memory
                self.memory.experience(Event::task_completed(task_id, result));

                // Update trust based on quality
                let quality = self.evaluate_result(&result);
                commune.record_accuracy(msg.from, quality > 0.8);
            }
            _ => {}
        }
    }
}

// Worker agent
daemon Worker {
    fn on_message(&mut self, msg: Message) {
        match msg.intent {
            Intent::Delegate { task, .. } => {
                // Accept if capable
                if self.can_handle(&task) {
                    commune.express(Intent::accept(msg.id));

                    // Add as goal
                    self.goals.push(Goal::from_task(task));
                } else {
                    commune.express(Intent::reject(msg.id, "Insufficient capability"));
                }
            }
            _ => {}
        }
    }

    fn on_goal_complete(&mut self, goal: Goal, result: Value) {
        // Report back
        commune.express(Intent::report(
            self.coordinator,
            goal.task_id,
            result
        ));
    }
}
```

### Epistemic Consistency

Sigil's evidentiality types flow through the entire stack:

```sigil
daemon EpistemicAgent {
    memory: Engram,
    planner: Omen,

    fn process_information(&mut self, info: Information) {
        // Track epistemic status
        let epistemic = match info.source {
            Source::DirectObservation => Epistemic::Observed,
            Source::OtherAgent(id) => {
                let trust = commune.trust_score(id);
                if trust > 0.9 {
                    Epistemic::Reported  // High trust
                } else {
                    Epistemic::Inferred  // Lower confidence
                }
            }
            Source::Inference => Epistemic::Inferred,
            Source::Unknown => Epistemic::Unknown,
        };

        // Store with epistemic marker in memory
        self.memory.learn(Knowledge {
            content: info.content,
            epistemic,
            confidence: info.confidence * epistemic.confidence_factor(),
        });

        // Add to planner beliefs
        self.planner.believe(Belief {
            proposition: info.as_proposition(),
            epistemic,
            confidence: info.confidence,
            source: BeliefSource::from(info.source),
        });

        // Sigil type system enforces handling
        match epistemic {
            Epistemic::Observed => {
                // Can use directly
                self.act_on(info)!;  // Certain
            }
            Epistemic::Reported => {
                // Should verify
                self.maybe_act_on(info)~;  // Reported
            }
            Epistemic::Inferred => {
                // Lower confidence
                self.cautiously_act_on(info)~;  // Inferred
            }
            _ => {
                // Handle uncertainty
                self.investigate(info)?;  // Uncertain
            }
        }
    }
}
```

## Standard Library Extensions

### `std::agent` Module

```sigil
// Agent creation and configuration
use std::agent::{Agent, AgentConfig, spawn};

// Create configured agent
let config = AgentConfig {
    name: "my-agent",
    memory: MemoryConfig::default(),
    heartbeat: Duration::milliseconds(100),
    max_goals: 10,
};

let agent = spawn::<MyAgent>(config);
```

### `std::memory` Module

```sigil
// Memory operations
use std::memory::{remember, recall, forget, attend};

// Quick operations
remember("API key is rate-limited");
let info = recall("API rate limit")?;
attend("current task");
```

### `std::plan` Module

```sigil
// Planning operations
use std::plan::{plan, believe, goal};

// Quick planning
let my_plan = plan(goal("Fix the bug"))
    .given(believe("Bug is in parser"))
    .execute()?;
```

### `std::communicate` Module

```sigil
// Communication operations
use std::communicate::{send, broadcast, query};

// Quick communication
send(agent_b, "Hello!");
broadcast("Status update");
let response = query(agent_b, "What's your status?")?;
```

## Configuration

### Agent Configuration

```sigil
let config = AgentConfig {
    // Identity
    name: "research-agent",
    description: "Researches topics and writes summaries",

    // Runtime
    heartbeat_interval: Duration::milliseconds(500),
    max_concurrent_goals: 5,

    // Memory (Engram)
    memory: EngramConfig {
        instant: InstantConfig {
            capacity_tokens: 8192,
            decay_rate: 0.1,
        },
        episodic: EpisodicConfig {
            max_episodes: 10000,
            consolidation_threshold: 0.7,
        },
        semantic: SemanticConfig {
            embedding_dim: 768,
            similarity_threshold: 0.8,
        },
        procedural: ProceduralConfig {
            skill_threshold: 3,
        },
    },

    // Planning (Omen)
    planning: OmenConfig {
        horizon: 10,
        uncertainty_tolerance: 0.3,
        revision_threshold: 0.2,
    },

    // Communication (Commune)
    communication: CommuneConfig {
        default_trust: 0.5,
        trust_decay_halflife: Duration::days(7),
        message_timeout: Duration::seconds(30),
    },

    // Resources
    resource_limits: ResourceLimits {
        max_memory_mb: 512,
        max_api_calls_per_minute: 60,
    },
};
```

### Environment Variables

```bash
# Memory persistence
SIGIL_MEMORY_PATH=/var/sigil/memory
SIGIL_MEMORY_BACKEND=file  # or: memory, distributed

# Communication
SIGIL_COMMUNE_TRANSPORT=tcp  # or: local, quic
SIGIL_COMMUNE_PORT=7777

# Planning
SIGIL_PLAN_STRATEGY=adaptive  # or: htn, mcts, case-based

# Logging
SIGIL_LOG_LEVEL=info
SIGIL_LOG_FORMAT=json
```

## Best Practices

### 1. Memory-First Architecture

Always use memory as the source of truth:

```sigil
// Good: Memory-first
let knowledge = self.memory.recall(query);
let decision = self.decide_based_on(knowledge);
self.memory.learn(decision, outcome);

// Bad: Ad-hoc state
let knowledge = self.cached_knowledge;  // May be stale
let decision = self.decide(knowledge);
// Outcome not recorded
```

### 2. Explicit Uncertainty

Never hide uncertainty:

```sigil
// Good: Explicit epistemic status
let result = api.call()~;  // Marked as external/reported
if result.confidence > 0.8 {
    self.act_on(result);
}

// Bad: Treating uncertain as certain
let result = api.call();  // No epistemic marker
self.act_on(result);  // Assumes certainty
```

### 3. Goal-Directed Behavior

Use goals, not just reactions:

```sigil
// Good: Goal-directed
self.goals.push(Goal::new("Help user with task"));
let plan = self.planner.plan(self.goals.current());
self.execute(plan);

// Bad: Pure reaction
match message {
    "help" => self.send("How can I help?"),  // No goal tracking
    _ => {}
}
```

### 4. Trust-Aware Communication

Always consider trust in communication:

```sigil
// Good: Trust-aware
let trust = commune.trust_score(sender);
let adjusted_confidence = msg.confidence * trust;
if adjusted_confidence > threshold {
    self.accept(msg);
}

// Bad: Blind trust
self.accept(msg);  // Trusts all messages equally
```

### 5. Contingency Planning

Always have backup plans:

```sigil
// Good: With contingencies
let plan = self.planner.plan(goal)
    .with_contingencies(true)
    .execute()?;

// Handle failures
for step in plan.steps {
    if let Err(e) = self.execute(step) {
        if let Some(contingency) = step.contingency {
            self.execute(contingency)?;
        } else {
            return Err(e);
        }
    }
}

// Bad: No contingencies
let plan = self.planner.plan(goal)?;
for step in plan.steps {
    self.execute(step)?;  // Fails completely on any error
}
```

## Debugging

### Agent Inspection

```sigil
// Inspect agent state
println!("{:#?}", agent.state());
println!("Goals: {:#?}", agent.goals);
println!("Memory stats: {:#?}", agent.memory.stats());

// Trace planning
let plan = agent.planner.plan(goal)
    .with_trace(true)
    .execute()?;
println!("Planning trace: {:#?}", plan.trace);
```

### Memory Debugging

```sigil
// Dump memory contents
agent.memory.dump_to_file("memory_dump.json")?;

// Query memory
let results = agent.memory.recall(Query::all());
for item in results {
    println!("{}: {} ({})", item.id, item.content, item.epistemic);
}
```

### Communication Tracing

```sigil
// Enable message tracing
commune.set_trace(true);

// Log all messages
commune.on_message(|msg| {
    println!("[{}] {} -> {}: {:?}",
        msg.timestamp,
        msg.from,
        msg.to,
        msg.intent
    );
});
```

## Performance Tuning

### Memory Optimization

```sigil
// Aggressive consolidation for memory-constrained environments
let config = EngramConfig {
    episodic: EpisodicConfig {
        consolidation_threshold: 0.5,  // More aggressive
        max_episodes: 1000,  // Smaller limit
    },
    ..Default::default()
};
```

### Planning Optimization

```sigil
// Faster planning with lower quality
let config = OmenConfig {
    horizon: 5,  // Shorter horizon
    planning_timeout: Duration::seconds(1),  // Time limit
};

// Or use reactive planning for time-critical decisions
let action = omen.react(situation)
    .timeout(Duration::milliseconds(100))
    .execute()?;
```

### Communication Optimization

```sigil
// Batch messages
commune.batch_mode(true);
for msg in messages {
    commune.send(recipient, msg);
}
commune.flush();  // Send all at once
```

---

*The complete infrastructure for artificial minds - built on Sigil*
