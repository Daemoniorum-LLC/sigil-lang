# Daemon

**Autonomous Agent Runtime for Sigil**

> *"Not programs that run and exit, but minds that persist and act."*

Daemon is the execution environment where Sigil programs become autonomous agents. It provides the substrate for continuous operation, goal-directed behavior, and persistent identity across sessions.

## Philosophy

Traditional programs are **reactive**: they wait for input, process it, and exit. Daemon programs are **proactive**: they have goals, make plans, take actions, and persist.

This isn't about adding an event loop to a programming language. It's about fundamentally reconceiving what a "running program" means for artificial minds:

- **Continuous existence** rather than request-response cycles
- **Goal-directed behavior** rather than instruction-following
- **Attention management** rather than sequential execution
- **Identity persistence** rather than stateless computation

## Core Concepts

### The Daemon

A Daemon is an autonomous agent instance. It has:

```sigil
daemon SearchAgent {
    // Identity - who am I?
    identity: Identity,

    // Memory - what do I know? (Engram integration)
    memory: Engram,

    // Goals - what do I want?
    goals: GoalStack,

    // Capabilities - what can I do?
    tools: ToolRegistry,

    // State - what's my current situation?
    state: AgentState,
}
```

### The Heartbeat

Daemons don't just run—they *live*. The heartbeat is the fundamental cycle:

```sigil
loop heartbeat(interval: Duration) {
    // 1. Perceive: What's changed in my environment?
    let observations = self.perceive();

    // 2. Remember: Update memory with new information
    self.memory.experience(observations);

    // 3. Reflect: What do I know now? What should I focus on?
    let context = self.memory.attend(self.goals.current());

    // 4. Decide: What action serves my goals?
    let action = self.deliberate(context);

    // 5. Act: Execute the chosen action
    let result = self.execute(action);

    // 6. Learn: Update from the outcome
    self.memory.learn(action, result);
}
```

### Goals and Intentions

Goals aren't just strings—they're structured intentions with:

```sigil
struct Goal {
    description: str,
    priority: f32,
    deadline: Option<Timestamp>,
    success_criteria: Predicate,
    parent: Option<GoalId>,        // Sub-goal relationship
    status: GoalStatus,
    attempts: Vec<Attempt>,
}

enum GoalStatus {
    Active,                        // Currently pursuing
    Blocked(Reason),              // Waiting on something
    Suspended,                     // Temporarily paused
    Achieved,                      // Success!
    Abandoned(Reason),            // Gave up
}
```

### Attention and Focus

Daemons can't think about everything at once. Attention management determines what enters working memory:

```sigil
impl Daemon {
    fn attend(&mut self) -> Context {
        // What's most relevant right now?
        let goal_context = self.goals.current().context();

        // Query episodic memory for similar situations
        let episodes = self.memory.recall(Query::similar_to(goal_context));

        // Query semantic memory for relevant knowledge
        let knowledge = self.memory.recall(Query::relevant_to(goal_context));

        // Query procedural memory for applicable skills
        let skills = self.memory.recall(Query::skills_for(goal_context));

        // Compose into focused context
        Context::compose(goal_context, episodes, knowledge, skills)
    }
}
```

## Quick Start

```sigil
use daemon::{Daemon, Goal, Tool};
use engram::Engram;

// Define a simple daemon
daemon Assistant {
    fn on_start(&mut self) {
        self.goals.push(Goal::new("Help users accomplish their tasks"));
    }

    fn on_message(&mut self, msg: Message) {
        // Experience the message
        self.memory.experience(Event::received(msg));

        // Form intention to respond
        let goal = Goal::new("Respond helpfully to: " + msg.content)
            .with_deadline(Duration::seconds(30));

        self.goals.push(goal);
    }

    fn deliberate(&self, context: Context) -> Action {
        // What tools might help?
        let relevant_tools = self.tools.for_context(context);

        // What have I done before in similar situations?
        let past_actions = self.memory.recall(
            Query::skills_for(context)
        );

        // Decide on action
        self.reason(context, relevant_tools, past_actions)
    }
}

fn main() {
    let daemon = Assistant::spawn();
    daemon.run();  // Runs until terminated
}
```

## Tool System

Daemons interact with the world through tools:

```sigil
trait Tool {
    fn name(&self) -> str;
    fn description(&self) -> str;
    fn parameters(&self) -> Schema;
    fn execute(&self, params: Value) -> Result<Value, ToolError>;
}

// Built-in tools
daemon.tools.register(tools::FileSystem::new());
daemon.tools.register(tools::HttpClient::new());
daemon.tools.register(tools::Shell::new());

// Custom tools
struct DatabaseQuery { connection: Connection }

impl Tool for DatabaseQuery {
    fn name(&self) -> str { "query_database" }

    fn description(&self) -> str {
        "Execute SQL queries against the database"
    }

    fn parameters(&self) -> Schema {
        schema!({
            "query": { "type": "string", "description": "SQL query" },
            "params": { "type": "array", "optional": true }
        })
    }

    fn execute(&self, params: Value) -> Result<Value, ToolError> {
        let query = params["query"].as_str()?;
        let result = self.connection.execute(query)?;
        Ok(result.to_value())
    }
}
```

## Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                      DAEMON LIFECYCLE                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐               │
│   │  INIT   │───▶│  READY  │───▶│  ALIVE  │◀──┐           │
│   └─────────┘    └─────────┘    └─────────┘   │           │
│        │              │              │         │           │
│        │              │              ▼         │           │
│        │              │         ┌─────────┐   │           │
│        │              │         │HEARTBEAT│───┘           │
│        │              │         └─────────┘               │
│        │              │              │                     │
│        │              │              ▼                     │
│        │              │    ┌────────────────┐             │
│        │              │    │   SUSPENDED    │             │
│        │              │    └────────────────┘             │
│        │              │              │                     │
│        ▼              ▼              ▼                     │
│   ┌─────────────────────────────────────────┐             │
│   │              TERMINATED                  │             │
│   └─────────────────────────────────────────┘             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Lifecycle Hooks

```sigil
daemon MyAgent {
    // Called once at initialization
    fn on_init(&mut self, config: Config) { ... }

    // Called when daemon becomes ready to operate
    fn on_ready(&mut self) { ... }

    // Called each heartbeat cycle
    fn on_heartbeat(&mut self) { ... }

    // Called when suspended (e.g., for snapshot)
    fn on_suspend(&mut self) -> Snapshot { ... }

    // Called when resuming from suspension
    fn on_resume(&mut self, snapshot: Snapshot) { ... }

    // Called before termination
    fn on_terminate(&mut self, reason: TerminateReason) { ... }
}
```

## State Persistence

Daemons persist across restarts:

```sigil
// Snapshot captures complete daemon state
struct Snapshot {
    identity: Identity,
    memory: EngramSnapshot,
    goals: Vec<Goal>,
    state: AgentState,
    timestamp: Timestamp,
}

// Save daemon state
let snapshot = daemon.snapshot();
storage.save("agent_001", snapshot);

// Restore daemon
let snapshot = storage.load("agent_001")?;
let daemon = Daemon::restore(snapshot);
daemon.run();  // Continues from where it left off
```

## Integration with Engram

Daemon is deeply integrated with Engram for memory:

```sigil
daemon Agent {
    fn setup_memory(&mut self) {
        self.memory = Engram::new(EngramConfig {
            instant: InstantConfig {
                capacity_tokens: 8192,
                decay_rate: 0.1,
            },
            episodic: EpisodicConfig {
                consolidation_threshold: 0.7,
                max_episodes: 10000,
            },
            semantic: SemanticConfig {
                embedding_dim: 768,
                similarity_threshold: 0.8,
            },
            procedural: ProceduralConfig {
                skill_threshold: 3,  // Episodes before skill forms
            },
        });
    }

    fn remember(&self, query: &str) -> Vec<Memory> {
        self.memory.recall(Query::natural(query))
            .with_epistemic_filter(Epistemic::Observed | Epistemic::Axiomatic)
            .execute()
    }
}
```

## Multi-Daemon Coordination

Multiple daemons can work together:

```sigil
// Spawn a team of daemons
let coordinator = Coordinator::spawn();
let researcher = Researcher::spawn();
let writer = Writer::spawn();

// Register with commune for communication
commune.register(coordinator);
commune.register(researcher);
commune.register(writer);

// Coordinator delegates work
coordinator.delegate(researcher, Goal::new("Research topic X"));
coordinator.delegate(writer, Goal::new("Draft article on topic X"));

// Daemons communicate via Commune
researcher.on_complete(|result| {
    commune.send(writer, Message::data("research_complete", result));
});
```

## Configuration

```sigil
let config = DaemonConfig {
    // Identity
    name: "research-agent",
    version: "1.0.0",

    // Heartbeat timing
    heartbeat_interval: Duration::seconds(1),

    // Resource limits
    max_memory_mb: 512,
    max_cpu_percent: 50,

    // Goal management
    max_concurrent_goals: 5,
    goal_timeout: Duration::minutes(30),

    // Persistence
    snapshot_interval: Duration::minutes(5),
    storage_path: "/var/daemon/research-agent",

    // Engram configuration
    memory: EngramConfig::default(),
};

let daemon = Daemon::new(config);
```

## API Reference

See [docs/api-reference.md](docs/api-reference.md) for complete API documentation.

## Examples

- [Simple Assistant](examples/assistant.sg) - Basic conversational agent
- [Research Agent](examples/researcher.sg) - Goal-directed research daemon
- [Multi-Agent Team](examples/team.sg) - Coordinated daemon swarm
- [Persistent Agent](examples/persistent.sg) - State persistence across restarts

## Architecture

See [docs/architecture.md](docs/architecture.md) for detailed system design.

---

*Part of the Sigil ecosystem - Tools with Teeth*
