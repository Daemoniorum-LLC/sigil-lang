# Daemon Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              DAEMON RUNTIME                               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                         DAEMON INSTANCE                          │    │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐       │    │
│  │  │ IDENTITY  │ │   GOALS   │ │   STATE   │ │   TOOLS   │       │    │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘       │    │
│  │        │             │             │             │              │    │
│  │        └─────────────┴──────┬──────┴─────────────┘              │    │
│  │                             │                                    │    │
│  │                    ┌────────▼────────┐                          │    │
│  │                    │   HEARTBEAT     │                          │    │
│  │                    │    ENGINE       │                          │    │
│  │                    └────────┬────────┘                          │    │
│  │                             │                                    │    │
│  │        ┌────────────────────┼────────────────────┐              │    │
│  │        ▼                    ▼                    ▼              │    │
│  │  ┌──────────┐        ┌──────────┐        ┌──────────┐          │    │
│  │  │ PERCEIVE │        │ DELIBER- │        │   ACT    │          │    │
│  │  │          │        │   ATE    │        │          │          │    │
│  │  └────┬─────┘        └────┬─────┘        └────┬─────┘          │    │
│  │       │                   │                   │                 │    │
│  │       └───────────────────┼───────────────────┘                 │    │
│  │                           │                                     │    │
│  │                    ┌──────▼──────┐                              │    │
│  │                    │   ENGRAM    │                              │    │
│  │                    │   MEMORY    │                              │    │
│  │                    └─────────────┘                              │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │    SCHEDULER    │  │   PERSISTENCE   │  │    COMMUNE      │         │
│  │                 │  │                 │  │   INTERFACE     │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Identity System

```sigil
/// Unique daemon identity
pub struct Identity {
    /// Globally unique identifier (UUID v7 for time-ordering)
    pub id: DaemonId,

    /// Human-readable name
    pub name: String,

    /// Self-description
    pub description: String,

    /// Core directives that persist across updates
    pub constitution: Constitution,

    /// Creation metadata
    pub created_at: Timestamp,
    pub created_by: Option<DaemonId>,

    /// Version for schema evolution
    pub schema_version: u32,

    /// Cryptographic identity
    pub public_key: PublicKey,
    private_key: PrivateKey,
}

/// Core values and constraints
pub struct Constitution {
    /// Inviolable directives
    pub directives: Vec<Directive>,

    /// Capability boundaries
    pub allowed_actions: ActionSet,

    /// Resource limits
    pub resource_bounds: ResourceBounds,

    /// Ethical constraints
    pub constraints: Vec<Constraint>,
}

pub struct Directive {
    pub priority: u32,
    pub content: String,
    pub immutable: bool,
}
```

### 2. Goal Management

```sigil
/// The goal stack manages daemon intentions
pub struct GoalStack {
    /// Active goals sorted by priority
    goals: BTreeMap<Priority, Goal>,

    /// Goal relationships (parent-child)
    hierarchy: HashMap<GoalId, Vec<GoalId>>,

    /// Completed goals for learning
    history: Vec<CompletedGoal>,

    /// Configuration
    config: GoalConfig,
}

pub struct Goal {
    pub id: GoalId,
    pub description: String,
    pub priority: Priority,
    pub status: GoalStatus,

    /// Success criteria as predicate
    pub success: Predicate,

    /// Optional deadline
    pub deadline: Option<Timestamp>,

    /// Parent goal (if sub-goal)
    pub parent: Option<GoalId>,

    /// Tracking
    pub created_at: Timestamp,
    pub attempts: Vec<Attempt>,
    pub context: GoalContext,
}

pub enum GoalStatus {
    Pending,
    Active,
    Blocked { reason: String, since: Timestamp },
    Suspended { reason: String },
    Completed { at: Timestamp, outcome: Outcome },
    Abandoned { at: Timestamp, reason: String },
}

impl GoalStack {
    /// Get the highest-priority active goal
    pub fn current(&self) -> Option<&Goal> {
        self.goals.values()
            .filter(|g| matches!(g.status, GoalStatus::Active))
            .next()
    }

    /// Add a new goal
    pub fn push(&mut self, goal: Goal) {
        let id = goal.id;
        let priority = goal.priority;

        if let Some(parent_id) = goal.parent {
            self.hierarchy.entry(parent_id)
                .or_default()
                .push(id);
        }

        self.goals.insert(priority, goal);
    }

    /// Mark goal as completed
    pub fn complete(&mut self, id: GoalId, outcome: Outcome) {
        if let Some(goal) = self.goals.values_mut().find(|g| g.id == id) {
            goal.status = GoalStatus::Completed {
                at: Timestamp::now(),
                outcome,
            };

            // Move to history
            self.history.push(CompletedGoal::from(goal.clone()));
        }
    }

    /// Decompose goal into sub-goals
    pub fn decompose(&mut self, parent_id: GoalId, sub_goals: Vec<Goal>) {
        for mut sub in sub_goals {
            sub.parent = Some(parent_id);
            self.push(sub);
        }
    }
}
```

### 3. Heartbeat Engine

```sigil
/// The core execution loop
pub struct HeartbeatEngine {
    /// Timing configuration
    interval: Duration,
    last_beat: Timestamp,

    /// Phase executors
    perceiver: Perceiver,
    deliberator: Deliberator,
    executor: Executor,

    /// Metrics
    metrics: HeartbeatMetrics,
}

impl HeartbeatEngine {
    /// Execute one heartbeat cycle
    pub async fn beat(&mut self, daemon: &mut Daemon) -> HeartbeatResult {
        let start = Timestamp::now();

        // Phase 1: Perceive
        let observations = self.perceiver.perceive(daemon).await?;

        // Phase 2: Remember
        for obs in &observations {
            daemon.memory.experience(obs.to_event());
        }

        // Phase 3: Reflect (build context)
        let context = daemon.attend()?;

        // Phase 4: Deliberate
        let action = self.deliberator.deliberate(daemon, &context).await?;

        // Phase 5: Act
        let result = self.executor.execute(daemon, action).await?;

        // Phase 6: Learn
        daemon.memory.learn(&action, &result);

        // Update metrics
        self.metrics.record(start.elapsed());

        Ok(HeartbeatResult { observations, action, result })
    }

    /// Run the heartbeat loop
    pub async fn run(&mut self, daemon: &mut Daemon) -> ! {
        loop {
            // Wait for next heartbeat
            let elapsed = self.last_beat.elapsed();
            if elapsed < self.interval {
                sleep(self.interval - elapsed).await;
            }

            // Execute heartbeat
            match self.beat(daemon).await {
                Ok(result) => {
                    self.last_beat = Timestamp::now();
                    daemon.on_heartbeat_complete(result);
                }
                Err(e) => {
                    daemon.on_heartbeat_error(e);
                }
            }

            // Check for termination
            if daemon.should_terminate() {
                daemon.terminate();
                break;
            }
        }
    }
}
```

### 4. Perception System

```sigil
/// Perception aggregates observations from multiple sources
pub struct Perceiver {
    /// Registered perception sources
    sources: Vec<Box<dyn PerceptionSource>>,

    /// Attention filter
    attention: AttentionFilter,

    /// Observation buffer
    buffer: VecDeque<Observation>,
}

pub trait PerceptionSource {
    /// Source identifier
    fn id(&self) -> &str;

    /// Poll for new observations
    async fn poll(&mut self) -> Vec<Observation>;

    /// Priority for attention allocation
    fn priority(&self) -> f32;
}

pub struct Observation {
    pub source: String,
    pub timestamp: Timestamp,
    pub content: ObservationContent,
    pub salience: f32,
}

pub enum ObservationContent {
    Message(Message),
    Event(SystemEvent),
    TimeElapsed(Duration),
    EnvironmentChange(EnvironmentDelta),
    ToolResult(ToolResult),
    Custom(Value),
}

/// Built-in perception sources
pub struct MessageSource {
    receiver: Receiver<Message>,
}

pub struct TimerSource {
    interval: Duration,
    last_tick: Timestamp,
}

pub struct FileWatchSource {
    watcher: FileWatcher,
    paths: Vec<PathBuf>,
}

pub struct EnvironmentSource {
    sensors: Vec<Box<dyn Sensor>>,
}
```

### 5. Deliberation System

```sigil
/// Deliberation decides what action to take
pub struct Deliberator {
    /// Reasoning strategy
    strategy: Box<dyn ReasoningStrategy>,

    /// Action generator
    generator: ActionGenerator,

    /// Action evaluator
    evaluator: ActionEvaluator,
}

pub trait ReasoningStrategy {
    /// Generate possible actions given context
    fn generate_actions(&self, context: &Context) -> Vec<ProposedAction>;

    /// Evaluate and rank actions
    fn evaluate(&self, actions: &[ProposedAction], context: &Context) -> Vec<(ProposedAction, f32)>;

    /// Select best action
    fn select(&self, ranked: &[(ProposedAction, f32)]) -> ProposedAction;
}

/// Simple utility-based reasoning
pub struct UtilityReasoning {
    utility_fn: Box<dyn Fn(&ProposedAction, &Context) -> f32>,
}

/// Goal-directed reasoning with planning
pub struct GoalDirectedReasoning {
    planner: Box<dyn Planner>,
    horizon: usize,
}

/// Hybrid reasoning with multiple strategies
pub struct HybridReasoning {
    strategies: Vec<(Box<dyn ReasoningStrategy>, f32)>,  // strategy, weight
}

impl Deliberator {
    pub async fn deliberate(&self, daemon: &Daemon, context: &Context) -> Result<Action> {
        // Get current goal
        let goal = daemon.goals.current()
            .ok_or(Error::NoActiveGoal)?;

        // Generate possible actions
        let proposals = self.generator.generate(daemon, context, goal);

        // Filter by feasibility
        let feasible: Vec<_> = proposals.into_iter()
            .filter(|p| self.is_feasible(daemon, p))
            .collect();

        // Evaluate
        let ranked = self.evaluator.evaluate(&feasible, context, goal);

        // Select
        let selected = self.strategy.select(&ranked);

        Ok(selected.into_action())
    }
}
```

### 6. Action Execution

```sigil
/// Execute actions and handle results
pub struct Executor {
    /// Tool registry
    tools: ToolRegistry,

    /// Execution sandbox
    sandbox: ExecutionSandbox,

    /// Result processor
    processor: ResultProcessor,
}

pub struct ToolRegistry {
    tools: HashMap<String, Box<dyn Tool>>,
    schemas: HashMap<String, Schema>,
}

pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> &str;
    fn schema(&self) -> &Schema;

    async fn execute(&self, params: Value) -> Result<ToolResult>;

    /// Optional: cleanup on daemon shutdown
    fn cleanup(&self) {}
}

pub struct ToolResult {
    pub success: bool,
    pub output: Value,
    pub error: Option<String>,
    pub duration: Duration,
    pub side_effects: Vec<SideEffect>,
}

pub enum Action {
    /// Internal action (thinking, planning)
    Internal(InternalAction),

    /// Tool invocation
    Tool { name: String, params: Value },

    /// Communication
    Communicate(CommunicateAction),

    /// Goal management
    Goal(GoalAction),

    /// No action (wait)
    Wait(Duration),
}

impl Executor {
    pub async fn execute(&self, daemon: &mut Daemon, action: Action) -> Result<ActionResult> {
        match action {
            Action::Internal(internal) => {
                self.execute_internal(daemon, internal).await
            }
            Action::Tool { name, params } => {
                let tool = self.tools.get(&name)?;

                // Validate parameters
                tool.schema().validate(&params)?;

                // Execute in sandbox
                let result = self.sandbox.execute(|| {
                    tool.execute(params)
                }).await?;

                // Process result
                self.processor.process(daemon, &result);

                Ok(ActionResult::Tool(result))
            }
            Action::Communicate(comm) => {
                self.execute_communication(daemon, comm).await
            }
            Action::Goal(goal_action) => {
                self.execute_goal_action(daemon, goal_action).await
            }
            Action::Wait(duration) => {
                sleep(duration).await;
                Ok(ActionResult::Waited(duration))
            }
        }
    }
}
```

### 7. State Management

```sigil
/// Daemon operational state
pub struct AgentState {
    /// Lifecycle state
    pub lifecycle: LifecycleState,

    /// Operational metrics
    pub metrics: OperationalMetrics,

    /// Resource usage
    pub resources: ResourceUsage,

    /// Custom state (daemon-specific)
    pub custom: HashMap<String, Value>,
}

pub enum LifecycleState {
    Initializing,
    Ready,
    Running,
    Suspended { reason: String, since: Timestamp },
    Terminating { reason: TerminateReason },
    Terminated,
}

pub struct OperationalMetrics {
    pub uptime: Duration,
    pub heartbeats: u64,
    pub actions_taken: u64,
    pub goals_completed: u64,
    pub errors: u64,
}

pub struct ResourceUsage {
    pub memory_bytes: usize,
    pub cpu_time: Duration,
    pub tokens_used: u64,
    pub api_calls: u64,
}
```

### 8. Persistence Layer

```sigil
/// Snapshot for persistence
pub struct Snapshot {
    /// Version for schema evolution
    pub version: u32,

    /// Identity (always preserved)
    pub identity: Identity,

    /// Memory snapshot
    pub memory: EngramSnapshot,

    /// Goal state
    pub goals: GoalStackSnapshot,

    /// Agent state
    pub state: AgentState,

    /// Timestamp
    pub timestamp: Timestamp,

    /// Checksum for integrity
    pub checksum: [u8; 32],
}

pub trait PersistenceBackend {
    /// Save snapshot
    async fn save(&self, id: &DaemonId, snapshot: &Snapshot) -> Result<()>;

    /// Load snapshot
    async fn load(&self, id: &DaemonId) -> Result<Option<Snapshot>>;

    /// List all snapshots for daemon
    async fn list(&self, id: &DaemonId) -> Result<Vec<SnapshotMetadata>>;

    /// Delete snapshot
    async fn delete(&self, id: &DaemonId, timestamp: Timestamp) -> Result<()>;
}

/// File-based persistence
pub struct FilePersistence {
    base_path: PathBuf,
    compression: CompressionLevel,
}

/// Distributed persistence
pub struct DistributedPersistence {
    nodes: Vec<NodeAddress>,
    replication_factor: usize,
}
```

## Data Flow

### Heartbeat Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          HEARTBEAT CYCLE                                 │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐
  │  ENVIRONMENT │
  └──────┬───────┘
         │ stimuli
         ▼
  ┌──────────────┐     ┌──────────────┐
  │   PERCEIVE   │────▶│   INSTANT    │
  │              │     │   MEMORY     │
  └──────┬───────┘     └──────────────┘
         │ observations
         ▼
  ┌──────────────┐     ┌──────────────┐
  │   REMEMBER   │────▶│   EPISODIC   │
  │              │     │   MEMORY     │
  └──────┬───────┘     └──────────────┘
         │ context query
         ▼
  ┌──────────────┐     ┌──────────────────────────────┐
  │   REFLECT    │◀───▶│  SEMANTIC + PROCEDURAL MEM  │
  │  (attend)    │     └──────────────────────────────┘
  └──────┬───────┘
         │ context
         ▼
  ┌──────────────┐     ┌──────────────┐
  │  DELIBERATE  │◀───▶│    GOALS     │
  │              │     └──────────────┘
  └──────┬───────┘
         │ action
         ▼
  ┌──────────────┐     ┌──────────────┐
  │     ACT      │────▶│    TOOLS     │
  │              │     └──────────────┘
  └──────┬───────┘
         │ result
         ▼
  ┌──────────────┐     ┌──────────────┐
  │    LEARN     │────▶│  PROCEDURAL  │
  │              │     │    MEMORY    │
  └──────┬───────┘     └──────────────┘
         │ effects
         ▼
  ┌──────────────┐
  │  ENVIRONMENT │
  └──────────────┘
```

### Goal Lifecycle

```
                    ┌─────────┐
                    │ CREATED │
                    └────┬────┘
                         │
                         ▼
              ┌──────────────────┐
              │     PENDING      │
              └────────┬─────────┘
                       │
         ┌─────────────┼─────────────┐
         │             │             │
         ▼             ▼             ▼
   ┌──────────┐  ┌──────────┐  ┌──────────┐
   │  ACTIVE  │  │ BLOCKED  │  │SUSPENDED │
   └────┬─────┘  └────┬─────┘  └────┬─────┘
        │             │             │
        │             └──────┬──────┘
        │                    │
        ▼                    ▼
   ┌──────────┐        ┌──────────┐
   │COMPLETED │        │ABANDONED │
   └──────────┘        └──────────┘
```

## Concurrency Model

### Single-Threaded Core

The daemon core is single-threaded for simplicity and determinism:

```sigil
pub struct DaemonCore {
    // All core state is single-threaded
    daemon: Daemon,
    engine: HeartbeatEngine,
}

impl DaemonCore {
    pub async fn run(&mut self) {
        // Single async runtime
        self.engine.run(&mut self.daemon).await
    }
}
```

### Async I/O

I/O operations are async but don't require thread synchronization:

```sigil
impl Perceiver {
    pub async fn perceive(&mut self, daemon: &Daemon) -> Vec<Observation> {
        // Concurrent polling of sources
        let futures: Vec<_> = self.sources.iter_mut()
            .map(|s| s.poll())
            .collect();

        // Gather results
        let results = join_all(futures).await;

        // Flatten and filter
        results.into_iter()
            .flatten()
            .filter(|o| self.attention.passes(o))
            .collect()
    }
}
```

### Tool Execution

Tools may run in separate processes for isolation:

```sigil
pub struct ExecutionSandbox {
    /// Maximum execution time
    timeout: Duration,

    /// Resource limits
    limits: ResourceLimits,

    /// Isolation level
    isolation: IsolationLevel,
}

pub enum IsolationLevel {
    /// Run in same process (fast, less safe)
    InProcess,

    /// Run in separate thread with limits
    Threaded,

    /// Run in separate process (slow, safe)
    Process,

    /// Run in container (slowest, safest)
    Container,
}
```

## Error Handling

### Error Hierarchy

```sigil
pub enum DaemonError {
    // Initialization errors
    Init(InitError),

    // Perception errors
    Perception(PerceptionError),

    // Deliberation errors
    Deliberation(DeliberationError),

    // Execution errors
    Execution(ExecutionError),

    // Memory errors
    Memory(EngramError),

    // Communication errors
    Communication(CommuneError),

    // Resource errors
    Resource(ResourceError),

    // Fatal errors
    Fatal(FatalError),
}

impl Daemon {
    fn handle_error(&mut self, error: DaemonError) {
        match error {
            DaemonError::Fatal(e) => {
                self.terminate(TerminateReason::Fatal(e));
            }
            DaemonError::Resource(e) if e.is_exhausted() => {
                self.suspend(SuspendReason::ResourceExhaustion);
            }
            _ => {
                // Log and continue
                self.memory.experience(Event::error(error.clone()));
                self.state.metrics.errors += 1;
            }
        }
    }
}
```

### Recovery Strategies

```sigil
pub struct RecoveryConfig {
    /// Maximum retries for recoverable errors
    pub max_retries: u32,

    /// Backoff strategy
    pub backoff: BackoffStrategy,

    /// Error handlers
    pub handlers: HashMap<ErrorType, RecoveryHandler>,
}

pub enum RecoveryHandler {
    /// Retry the operation
    Retry { max: u32, backoff: Duration },

    /// Skip and continue
    Skip,

    /// Suspend the daemon
    Suspend,

    /// Terminate the daemon
    Terminate,

    /// Custom handler
    Custom(Box<dyn Fn(DaemonError) -> RecoveryAction>),
}
```

## Security Model

### Capability-Based Security

```sigil
pub struct Capabilities {
    /// Allowed tools
    pub tools: HashSet<String>,

    /// Allowed file paths
    pub file_access: Vec<PathPattern>,

    /// Allowed network access
    pub network: NetworkCapabilities,

    /// Allowed system calls
    pub syscalls: SyscallFilter,
}

impl Daemon {
    pub fn can_execute(&self, action: &Action) -> bool {
        match action {
            Action::Tool { name, .. } => {
                self.capabilities.tools.contains(name)
            }
            Action::Internal(_) => true,
            Action::Communicate(comm) => {
                self.capabilities.can_communicate(&comm.target)
            }
            _ => true,
        }
    }
}
```

### Audit Logging

```sigil
pub struct AuditLog {
    entries: Vec<AuditEntry>,
    sink: Box<dyn AuditSink>,
}

pub struct AuditEntry {
    pub timestamp: Timestamp,
    pub daemon_id: DaemonId,
    pub action: AuditAction,
    pub outcome: AuditOutcome,
    pub context: HashMap<String, Value>,
}

pub enum AuditAction {
    ToolInvocation { tool: String, params: Value },
    GoalCreation { goal: Goal },
    Communication { target: DaemonId, message: String },
    StateChange { from: LifecycleState, to: LifecycleState },
    ResourceAccess { resource: String, operation: String },
}
```

## Performance Considerations

### Memory Efficiency

```sigil
/// Daemon memory budget
pub struct MemoryBudget {
    /// Maximum total memory
    pub max_bytes: usize,

    /// Allocation per component
    pub instant_memory: usize,
    pub episodic_memory: usize,
    pub semantic_memory: usize,
    pub procedural_memory: usize,
    pub working_state: usize,
}

impl Daemon {
    fn enforce_memory_budget(&mut self) {
        let usage = self.memory_usage();

        if usage > self.budget.max_bytes {
            // Trigger memory consolidation
            self.memory.consolidate();

            // If still over budget, archive old episodes
            if self.memory_usage() > self.budget.max_bytes {
                self.memory.archive_oldest();
            }
        }
    }
}
```

### Heartbeat Optimization

```sigil
/// Adaptive heartbeat timing
pub struct AdaptiveHeartbeat {
    /// Base interval
    base_interval: Duration,

    /// Current interval
    current_interval: Duration,

    /// Load metrics
    load: LoadMetrics,
}

impl AdaptiveHeartbeat {
    fn adjust(&mut self) {
        // If idle, slow down
        if self.load.recent_actions == 0 {
            self.current_interval = min(
                self.current_interval * 2,
                Duration::seconds(60)
            );
        }
        // If busy, speed up
        else if self.load.pending_observations > 10 {
            self.current_interval = max(
                self.current_interval / 2,
                self.base_interval
            );
        }
    }
}
```

---

*Architecture designed for autonomous artificial agents*
