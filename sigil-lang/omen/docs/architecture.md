# Omen Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              OMEN SYSTEM                                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         REASONING ENGINE                             │    │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐           │    │
│  │  │  BELIEF   │ │  CAUSAL   │ │ INFERENCE │ │ ABDUCTION │           │    │
│  │  │  MANAGER  │ │   MODEL   │ │  ENGINE   │ │  ENGINE   │           │    │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         PLANNING ENGINE                              │    │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐           │    │
│  │  │    HTN    │ │   MCTS    │ │  MEANS-   │ │   CASE-   │           │    │
│  │  │  PLANNER  │ │  PLANNER  │ │   ENDS    │ │   BASED   │           │    │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         EXECUTION LAYER                              │    │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐           │    │
│  │  │  MONITOR  │ │ CONTINGENCY│ │ RESOURCE  │ │   RISK    │           │    │
│  │  │           │ │  MANAGER  │ │  TRACKER  │ │ ASSESSOR  │           │    │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘           │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                     │                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         LEARNING LAYER                               │    │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐                          │    │
│  │  │  ACTION   │ │  PATTERN  │ │ STRATEGY  │                          │    │
│  │  │  MODELS   │ │  LIBRARY  │ │  SELECTOR │                          │    │
│  │  └───────────┘ └───────────┘ └───────────┘                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                         │
│  │   ENGRAM    │  │   DAEMON    │  │   COMMUNE   │                         │
│  │ INTEGRATION │  │ INTEGRATION │  │ INTEGRATION │                         │
│  └─────────────┘  └─────────────┘  └─────────────┘                         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Belief Manager

```sigil
/// Manages agent beliefs with uncertainty tracking
pub struct BeliefManager {
    /// Current beliefs
    beliefs: HashMap<BeliefId, Belief>,

    /// Belief dependencies
    dependencies: DependencyGraph,

    /// Contradiction tracker
    contradictions: Vec<Contradiction>,

    /// Revision history
    history: Vec<RevisionEvent>,

    /// Configuration
    config: BeliefConfig,
}

pub struct Belief {
    /// Unique identifier
    pub id: BeliefId,

    /// The proposition believed
    pub proposition: Proposition,

    /// Epistemic status
    pub epistemic: Epistemic,

    /// Confidence level (0.0 - 1.0)
    pub confidence: f32,

    /// Evidence supporting this belief
    pub evidence: Vec<Evidence>,

    /// How we acquired this belief
    pub source: BeliefSource,

    /// When acquired
    pub acquired: Timestamp,

    /// When last updated
    pub updated: Timestamp,
}

pub struct Proposition {
    /// Subject of the proposition
    pub subject: String,

    /// Predicate
    pub predicate: String,

    /// Optional value
    pub value: Option<Value>,

    /// Temporal scope
    pub temporal: TemporalScope,
}

pub enum BeliefSource {
    /// Direct observation
    Observation(ObservationId),

    /// Inference from other beliefs
    Inference {
        premises: Vec<BeliefId>,
        rule: InferenceRule,
    },

    /// Reported by another agent
    Testimony {
        source: AgentId,
        original_epistemic: Epistemic,
        original_confidence: f32,
    },

    /// Assumed without evidence
    Assumption {
        reason: String,
    },

    /// Defined as true
    Axiom,
}

impl BeliefManager {
    /// Add a new belief
    pub fn believe(&mut self, belief: Belief) -> BeliefId {
        let id = belief.id;

        // Check for contradictions
        let contradictions = self.find_contradictions(&belief);
        if !contradictions.is_empty() {
            self.handle_contradictions(&belief, contradictions);
        }

        // Add to belief set
        self.beliefs.insert(id, belief.clone());

        // Update dependencies
        if let BeliefSource::Inference { premises, .. } = &belief.source {
            for premise in premises {
                self.dependencies.add_edge(*premise, id);
            }
        }

        // Log
        self.history.push(RevisionEvent::Added(id));

        id
    }

    /// Revise beliefs given new evidence
    pub fn revise(&mut self, evidence: Evidence) {
        // Find affected beliefs
        let affected = self.find_affected_by(&evidence);

        for belief_id in affected {
            if let Some(belief) = self.beliefs.get_mut(&belief_id) {
                // Update confidence based on evidence
                let support = evidence.support_for(&belief.proposition);

                match support {
                    Support::Confirms(strength) => {
                        belief.confidence = (belief.confidence + strength * (1.0 - belief.confidence))
                            .min(1.0);
                        belief.evidence.push(evidence.clone());
                    }
                    Support::Contradicts(strength) => {
                        belief.confidence = (belief.confidence * (1.0 - strength))
                            .max(0.0);

                        if belief.confidence < self.config.removal_threshold {
                            self.remove_belief(belief_id);
                        }
                    }
                    Support::Neutral => {}
                }
            }
        }

        // Propagate changes through dependency graph
        self.propagate_revisions();
    }

    /// Remove a belief and all dependent beliefs
    pub fn contract(&mut self, belief_id: BeliefId) {
        // Find all beliefs that depend on this one
        let dependents = self.dependencies.descendants(belief_id);

        // Remove in reverse topological order
        for dependent in dependents.into_iter().rev() {
            self.beliefs.remove(&dependent);
            self.history.push(RevisionEvent::Removed(dependent));
        }

        // Remove the belief itself
        self.beliefs.remove(&belief_id);
        self.history.push(RevisionEvent::Removed(belief_id));
    }

    /// Find contradictions
    fn find_contradictions(&self, new_belief: &Belief) -> Vec<BeliefId> {
        self.beliefs.values()
            .filter(|existing| existing.contradicts(new_belief))
            .map(|b| b.id)
            .collect()
    }

    /// Handle contradictions
    fn handle_contradictions(&mut self, new_belief: &Belief, contradictions: Vec<BeliefId>) {
        for contra_id in contradictions {
            if let Some(existing) = self.beliefs.get(&contra_id) {
                // Compare epistemic status and confidence
                let prefer_new = self.should_prefer(new_belief, existing);

                if prefer_new {
                    self.contract(contra_id);
                } else {
                    // Record contradiction but don't add new belief
                    self.contradictions.push(Contradiction {
                        new_belief: new_belief.clone(),
                        existing: contra_id,
                        resolved: false,
                    });
                }
            }
        }
    }

    fn should_prefer(&self, new: &Belief, existing: &Belief) -> bool {
        // Prefer higher epistemic status
        let new_epistemic_rank = new.epistemic.rank();
        let existing_epistemic_rank = existing.epistemic.rank();

        if new_epistemic_rank > existing_epistemic_rank {
            return true;
        }
        if new_epistemic_rank < existing_epistemic_rank {
            return false;
        }

        // Same epistemic status: prefer higher confidence
        new.confidence > existing.confidence
    }
}
```

### 2. Goal Manager

```sigil
/// Manages goals and their relationships
pub struct GoalManager {
    /// Active goals
    goals: HashMap<GoalId, Goal>,

    /// Goal hierarchy
    hierarchy: GoalHierarchy,

    /// Goal conflicts
    conflicts: Vec<GoalConflict>,

    /// Completed goals
    completed: Vec<CompletedGoal>,
}

pub struct Goal {
    /// Unique identifier
    pub id: GoalId,

    /// Description
    pub description: String,

    /// Success condition
    pub success: Predicate,

    /// Priority (0.0 - 1.0)
    pub priority: f32,

    /// Constraints
    pub constraints: Vec<Constraint>,

    /// Deadline
    pub deadline: Option<Timestamp>,

    /// Status
    pub status: GoalStatus,

    /// Parent goal (if sub-goal)
    pub parent: Option<GoalId>,

    /// Creation time
    pub created: Timestamp,
}

pub struct GoalHierarchy {
    /// Parent -> children mapping
    children: HashMap<GoalId, Vec<GoalId>>,

    /// Child -> parent mapping
    parents: HashMap<GoalId, GoalId>,
}

pub enum GoalStatus {
    /// Not yet started
    Pending,

    /// Currently being pursued
    Active,

    /// Waiting on something
    Blocked { reason: String },

    /// Temporarily suspended
    Suspended { reason: String },

    /// Successfully achieved
    Achieved { at: Timestamp },

    /// Could not be achieved
    Failed { at: Timestamp, reason: String },

    /// Intentionally abandoned
    Abandoned { at: Timestamp, reason: String },
}

impl GoalManager {
    /// Add a new goal
    pub fn add_goal(&mut self, goal: Goal) -> GoalId {
        let id = goal.id;

        // Check for conflicts with existing goals
        let conflicts = self.find_conflicts(&goal);
        for conflict in conflicts {
            self.conflicts.push(conflict);
        }

        // Add to hierarchy if sub-goal
        if let Some(parent) = goal.parent {
            self.hierarchy.add_child(parent, id);
        }

        self.goals.insert(id, goal);
        id
    }

    /// Decompose a goal into sub-goals
    pub fn decompose(&mut self, goal_id: GoalId, sub_goals: Vec<Goal>) {
        for mut sub_goal in sub_goals {
            sub_goal.parent = Some(goal_id);
            self.add_goal(sub_goal);
        }
    }

    /// Get highest priority active goal
    pub fn current_goal(&self) -> Option<&Goal> {
        self.goals.values()
            .filter(|g| matches!(g.status, GoalStatus::Active))
            .max_by(|a, b| a.priority.partial_cmp(&b.priority).unwrap())
    }

    /// Mark goal as achieved
    pub fn achieve(&mut self, goal_id: GoalId) {
        if let Some(goal) = self.goals.get_mut(&goal_id) {
            goal.status = GoalStatus::Achieved { at: Timestamp::now() };

            // Check if parent goal is now achievable
            if let Some(parent_id) = goal.parent {
                self.check_parent_completion(parent_id);
            }
        }
    }

    /// Check if all sub-goals are complete
    fn check_parent_completion(&mut self, parent_id: GoalId) {
        let children = self.hierarchy.children.get(&parent_id)
            .map(|c| c.clone())
            .unwrap_or_default();

        let all_achieved = children.iter().all(|child_id| {
            self.goals.get(child_id)
                .map(|g| matches!(g.status, GoalStatus::Achieved { .. }))
                .unwrap_or(false)
        });

        if all_achieved {
            self.achieve(parent_id);
        }
    }
}
```

### 3. Planning Engine

```sigil
/// Core planning engine with multiple strategies
pub struct PlanningEngine {
    /// Available planning strategies
    strategies: Vec<Box<dyn PlanningStrategy>>,

    /// Strategy selector
    selector: StrategySelector,

    /// Action library
    actions: ActionLibrary,

    /// Plan optimizer
    optimizer: PlanOptimizer,
}

pub trait PlanningStrategy {
    /// Name of the strategy
    fn name(&self) -> &str;

    /// Check if this strategy is applicable
    fn is_applicable(&self, goal: &Goal, beliefs: &BeliefManager) -> bool;

    /// Generate a plan
    fn plan(
        &self,
        goal: &Goal,
        beliefs: &BeliefManager,
        actions: &ActionLibrary,
        config: &PlanConfig,
    ) -> Result<Plan, PlanningError>;

    /// Estimated quality of plans from this strategy
    fn expected_quality(&self, goal: &Goal) -> f32;
}

pub struct Plan {
    /// Unique identifier
    pub id: PlanId,

    /// Goal this plan achieves
    pub goal: GoalId,

    /// Plan steps
    pub steps: Vec<PlanStep>,

    /// Expected success probability
    pub success_probability: f32,

    /// Expected cost
    pub expected_cost: ResourceCost,

    /// Assumptions the plan relies on
    pub assumptions: Vec<Assumption>,

    /// Identified risks
    pub risks: Vec<Risk>,

    /// Planning strategy used
    pub strategy: String,

    /// Generation time
    pub generated: Timestamp,
}

pub struct PlanStep {
    /// Step identifier
    pub id: StepId,

    /// Action to take
    pub action: Action,

    /// Preconditions
    pub preconditions: Vec<Predicate>,

    /// Expected effects
    pub effects: Vec<Effect>,

    /// Confidence this step will succeed
    pub confidence: f32,

    /// Expected duration
    pub duration: Duration,

    /// Expected resource cost
    pub cost: ResourceCost,

    /// Contingency plan if this step fails
    pub contingency: Option<PlanId>,
}

impl PlanningEngine {
    /// Generate a plan for a goal
    pub fn plan(&self, goal: &Goal, beliefs: &BeliefManager, config: &PlanConfig) -> Result<Plan, PlanningError> {
        // Select best strategy
        let strategy = self.selector.select(&self.strategies, goal, beliefs);

        // Generate initial plan
        let mut plan = strategy.plan(goal, beliefs, &self.actions, config)?;

        // Optimize
        plan = self.optimizer.optimize(plan, config);

        // Add contingencies
        plan = self.add_contingencies(plan, beliefs);

        Ok(plan)
    }

    /// Add contingency plans for risky steps
    fn add_contingencies(&self, mut plan: Plan, beliefs: &BeliefManager) -> Plan {
        for step in &mut plan.steps {
            if step.confidence < 0.8 {
                // Generate contingency
                let contingency_goal = Goal::recovery_from(step);
                if let Ok(contingency) = self.plan(&contingency_goal, beliefs, &PlanConfig::quick()) {
                    step.contingency = Some(contingency.id);
                }
            }
        }
        plan
    }
}
```

### 4. HTN Planner

```sigil
/// Hierarchical Task Network planner
pub struct HTNPlanner {
    /// Decomposition methods
    methods: HashMap<String, Vec<Method>>,

    /// Primitive actions
    primitives: HashMap<String, Action>,
}

pub struct Method {
    /// Method name
    pub name: String,

    /// Task this method decomposes
    pub task: String,

    /// Preconditions for using this method
    pub preconditions: Vec<Predicate>,

    /// Sub-tasks
    pub subtasks: Vec<Task>,

    /// Ordering constraints
    pub ordering: OrderingConstraints,
}

pub struct Task {
    /// Task name
    pub name: String,

    /// Task parameters
    pub params: HashMap<String, Value>,

    /// Is this primitive (directly executable)?
    pub primitive: bool,
}

impl PlanningStrategy for HTNPlanner {
    fn name(&self) -> &str { "HTN" }

    fn is_applicable(&self, goal: &Goal, _beliefs: &BeliefManager) -> bool {
        // HTN is applicable when we have methods for the goal
        self.methods.contains_key(&goal.description)
    }

    fn plan(
        &self,
        goal: &Goal,
        beliefs: &BeliefManager,
        actions: &ActionLibrary,
        config: &PlanConfig,
    ) -> Result<Plan, PlanningError> {
        let root_task = Task {
            name: goal.description.clone(),
            params: HashMap::new(),
            primitive: false,
        };

        let steps = self.decompose(root_task, beliefs, 0, config.max_depth)?;

        Ok(Plan {
            id: PlanId::new(),
            goal: goal.id,
            steps,
            success_probability: self.estimate_success(&steps),
            expected_cost: self.estimate_cost(&steps),
            assumptions: vec![],
            risks: vec![],
            strategy: "HTN".to_string(),
            generated: Timestamp::now(),
        })
    }

    fn expected_quality(&self, _goal: &Goal) -> f32 { 0.8 }
}

impl HTNPlanner {
    /// Recursively decompose task
    fn decompose(
        &self,
        task: Task,
        beliefs: &BeliefManager,
        depth: usize,
        max_depth: usize,
    ) -> Result<Vec<PlanStep>, PlanningError> {
        if depth > max_depth {
            return Err(PlanningError::MaxDepthExceeded);
        }

        if task.primitive {
            // Primitive task: convert to plan step
            let action = self.primitives.get(&task.name)
                .ok_or(PlanningError::UnknownAction(task.name.clone()))?;

            return Ok(vec![PlanStep {
                id: StepId::new(),
                action: action.clone(),
                preconditions: action.preconditions.clone(),
                effects: action.effects.clone(),
                confidence: action.success_rate,
                duration: action.expected_duration,
                cost: action.expected_cost.clone(),
                contingency: None,
            }]);
        }

        // Non-primitive: find applicable method
        let methods = self.methods.get(&task.name)
            .ok_or(PlanningError::NoMethodForTask(task.name.clone()))?;

        for method in methods {
            if self.preconditions_met(&method.preconditions, beliefs) {
                // Decompose using this method
                let mut all_steps = Vec::new();

                for subtask in &method.subtasks {
                    let substeps = self.decompose(subtask.clone(), beliefs, depth + 1, max_depth)?;
                    all_steps.extend(substeps);
                }

                return Ok(all_steps);
            }
        }

        Err(PlanningError::NoApplicableMethod(task.name))
    }
}
```

### 5. MCTS Planner

```sigil
/// Monte Carlo Tree Search planner
pub struct MCTSPlanner {
    /// Number of simulations
    simulations: usize,

    /// Exploration constant (UCB1)
    exploration: f32,

    /// Maximum simulation depth
    max_depth: usize,

    /// State simulator
    simulator: Box<dyn Simulator>,
}

pub struct MCTSNode {
    /// State at this node
    state: State,

    /// Action that led here
    action: Option<Action>,

    /// Visit count
    visits: u32,

    /// Total reward
    total_reward: f32,

    /// Children
    children: Vec<MCTSNode>,

    /// Parent
    parent: Option<*mut MCTSNode>,
}

impl PlanningStrategy for MCTSPlanner {
    fn name(&self) -> &str { "MCTS" }

    fn is_applicable(&self, _goal: &Goal, _beliefs: &BeliefManager) -> bool {
        true  // MCTS is generally applicable
    }

    fn plan(
        &self,
        goal: &Goal,
        beliefs: &BeliefManager,
        actions: &ActionLibrary,
        _config: &PlanConfig,
    ) -> Result<Plan, PlanningError> {
        let initial_state = State::from_beliefs(beliefs);
        let mut root = MCTSNode::new(initial_state);

        for _ in 0..self.simulations {
            // Selection
            let mut node = self.select(&mut root);

            // Expansion
            if !node.is_terminal() && node.visits > 0 {
                node = self.expand(node, actions);
            }

            // Simulation
            let reward = self.simulate(node, goal, actions);

            // Backpropagation
            self.backpropagate(node, reward);
        }

        // Extract best plan
        let steps = self.extract_plan(&root);

        Ok(Plan {
            id: PlanId::new(),
            goal: goal.id,
            steps,
            success_probability: root.total_reward / root.visits as f32,
            expected_cost: ResourceCost::default(),
            assumptions: vec![],
            risks: vec![],
            strategy: "MCTS".to_string(),
            generated: Timestamp::now(),
        })
    }

    fn expected_quality(&self, _goal: &Goal) -> f32 { 0.7 }
}

impl MCTSPlanner {
    /// Select node using UCB1
    fn select<'a>(&self, node: &'a mut MCTSNode) -> &'a mut MCTSNode {
        if node.children.is_empty() || node.is_terminal() {
            return node;
        }

        let total_visits = node.visits as f32;

        let best_child = node.children.iter_mut()
            .max_by(|a, b| {
                let ucb_a = self.ucb1(a, total_visits);
                let ucb_b = self.ucb1(b, total_visits);
                ucb_a.partial_cmp(&ucb_b).unwrap()
            })
            .unwrap();

        self.select(best_child)
    }

    /// UCB1 formula
    fn ucb1(&self, node: &MCTSNode, parent_visits: f32) -> f32 {
        if node.visits == 0 {
            return f32::INFINITY;
        }

        let exploitation = node.total_reward / node.visits as f32;
        let exploration = self.exploration * (parent_visits.ln() / node.visits as f32).sqrt();

        exploitation + exploration
    }

    /// Expand node with new children
    fn expand<'a>(&self, node: &'a mut MCTSNode, actions: &ActionLibrary) -> &'a mut MCTSNode {
        let available_actions = actions.applicable_in(&node.state);

        for action in available_actions {
            let new_state = self.simulator.apply(&node.state, &action);
            let child = MCTSNode {
                state: new_state,
                action: Some(action),
                visits: 0,
                total_reward: 0.0,
                children: vec![],
                parent: Some(node as *mut _),
            };
            node.children.push(child);
        }

        // Return random child
        let idx = rand::random::<usize>() % node.children.len();
        &mut node.children[idx]
    }

    /// Simulate random playout
    fn simulate(&self, node: &MCTSNode, goal: &Goal, actions: &ActionLibrary) -> f32 {
        let mut state = node.state.clone();
        let mut depth = 0;

        while depth < self.max_depth && !goal.is_satisfied_by(&state) {
            let available = actions.applicable_in(&state);
            if available.is_empty() {
                break;
            }

            let action = &available[rand::random::<usize>() % available.len()];
            state = self.simulator.apply(&state, action);
            depth += 1;
        }

        if goal.is_satisfied_by(&state) {
            1.0 - (depth as f32 / self.max_depth as f32) * 0.5  // Prefer shorter plans
        } else {
            0.0
        }
    }

    /// Backpropagate reward
    fn backpropagate(&self, node: &mut MCTSNode, reward: f32) {
        node.visits += 1;
        node.total_reward += reward;

        if let Some(parent_ptr) = node.parent {
            unsafe {
                self.backpropagate(&mut *parent_ptr, reward);
            }
        }
    }

    /// Extract best plan from tree
    fn extract_plan(&self, root: &MCTSNode) -> Vec<PlanStep> {
        let mut steps = Vec::new();
        let mut current = root;

        while !current.children.is_empty() {
            // Select most visited child
            let best = current.children.iter()
                .max_by_key(|c| c.visits)
                .unwrap();

            if let Some(action) = &best.action {
                steps.push(PlanStep {
                    id: StepId::new(),
                    action: action.clone(),
                    preconditions: vec![],
                    effects: vec![],
                    confidence: best.total_reward / best.visits as f32,
                    duration: Duration::default(),
                    cost: ResourceCost::default(),
                    contingency: None,
                });
            }

            current = best;
        }

        steps
    }
}
```

### 6. Causal Model

```sigil
/// Causal model for reasoning about cause and effect
pub struct CausalModel {
    /// Variables in the model
    variables: HashMap<String, Variable>,

    /// Causal edges
    edges: Vec<CausalEdge>,

    /// Structural equations
    equations: HashMap<String, StructuralEquation>,
}

pub struct Variable {
    pub name: String,
    pub var_type: VariableType,
    pub domain: Domain,
}

pub enum VariableType {
    Boolean,
    Discrete(Vec<Value>),
    Continuous { min: f64, max: f64 },
}

pub struct CausalEdge {
    pub from: String,
    pub to: String,
    pub mechanism: Box<dyn Fn(&Value) -> Distribution>,
}

pub struct StructuralEquation {
    pub variable: String,
    pub parents: Vec<String>,
    pub function: Box<dyn Fn(&HashMap<String, Value>) -> Value>,
}

impl CausalModel {
    /// Predict outcome given observations
    pub fn predict(&self, observations: &HashMap<String, Value>) -> HashMap<String, Distribution> {
        let mut predictions = HashMap::new();

        // Topological order
        let order = self.topological_sort();

        for var_name in order {
            if observations.contains_key(&var_name) {
                // Observed: use observed value
                predictions.insert(var_name.clone(), Distribution::point(observations[&var_name].clone()));
            } else if let Some(eq) = self.equations.get(&var_name) {
                // Compute from parents
                let parent_values: HashMap<String, Value> = eq.parents.iter()
                    .map(|p| {
                        let dist = predictions.get(p).unwrap();
                        (p.clone(), dist.expected_value())
                    })
                    .collect();

                let value = (eq.function)(&parent_values);
                predictions.insert(var_name, Distribution::point(value));
            }
        }

        predictions
    }

    /// Predict outcome given intervention (do-calculus)
    pub fn intervene(&self, intervention: &HashMap<String, Value>) -> HashMap<String, Distribution> {
        // Create mutilated graph: remove edges into intervened variables
        let mut mutilated = self.clone();
        for var in intervention.keys() {
            mutilated.edges.retain(|e| &e.to != var);
        }

        // Predict with intervention values as observations
        mutilated.predict(intervention)
    }

    /// Evaluate counterfactual: what would Y be if X had been x, given evidence E?
    pub fn counterfactual(
        &self,
        target: &str,
        intervention: &HashMap<String, Value>,
        evidence: &HashMap<String, Value>,
    ) -> Distribution {
        // Step 1: Abduction - infer exogenous variables given evidence
        let exogenous = self.abduct_exogenous(evidence);

        // Step 2: Action - apply intervention
        let mut mutilated = self.clone();
        for var in intervention.keys() {
            mutilated.edges.retain(|e| &e.to != var);
        }

        // Step 3: Prediction - compute target given exogenous and intervention
        let mut context = exogenous;
        context.extend(intervention.clone());

        let predictions = mutilated.predict(&context);
        predictions.get(target).cloned().unwrap_or(Distribution::unknown())
    }

    /// Abductive inference: infer likely causes from effects
    pub fn abduce(&self, effects: &HashMap<String, Value>) -> Vec<Explanation> {
        let mut explanations = Vec::new();

        // Find variables that are ancestors of observed effects
        let effect_vars: Vec<_> = effects.keys().collect();
        let potential_causes = self.find_ancestors(&effect_vars);

        // For each potential cause configuration, compute likelihood
        for cause_config in self.enumerate_configurations(&potential_causes) {
            let predicted = self.predict(&cause_config);

            // Check consistency with observed effects
            let likelihood = self.compute_likelihood(&predicted, effects);

            if likelihood > 0.0 {
                explanations.push(Explanation {
                    causes: cause_config,
                    likelihood,
                });
            }
        }

        // Sort by likelihood
        explanations.sort_by(|a, b| b.likelihood.partial_cmp(&a.likelihood).unwrap());
        explanations
    }
}
```

### 7. Risk Assessor

```sigil
/// Assesses risks in plans
pub struct RiskAssessor {
    /// Risk patterns
    patterns: Vec<RiskPattern>,

    /// Historical risk data
    history: Vec<RiskOutcome>,
}

pub struct Risk {
    /// Risk identifier
    pub id: RiskId,

    /// Description
    pub description: String,

    /// Probability of occurrence
    pub probability: f32,

    /// Impact if it occurs
    pub impact: Impact,

    /// Which plan steps are affected
    pub affected_steps: Vec<StepId>,

    /// Possible mitigations
    pub mitigations: Vec<Mitigation>,
}

pub enum Impact {
    /// Plan fails completely
    PlanFailure,

    /// Goal partially achieved
    PartialSuccess { degree: f32 },

    /// Additional cost
    IncreasedCost { factor: f32 },

    /// Delay
    Delay { duration: Duration },

    /// Side effects
    SideEffects { effects: Vec<Effect> },
}

pub struct Mitigation {
    pub description: String,
    pub actions: Vec<Action>,
    pub effectiveness: f32,
    pub cost: ResourceCost,
}

impl RiskAssessor {
    /// Assess risks in a plan
    pub fn assess(&self, plan: &Plan, beliefs: &BeliefManager) -> Vec<Risk> {
        let mut risks = Vec::new();

        // Pattern-based risk detection
        for pattern in &self.patterns {
            if let Some(risk) = pattern.detect(plan, beliefs) {
                risks.push(risk);
            }
        }

        // Step-level risk analysis
        for step in &plan.steps {
            // Low confidence = risk
            if step.confidence < 0.7 {
                risks.push(Risk {
                    id: RiskId::new(),
                    description: format!("Step '{}' has low confidence", step.action.name()),
                    probability: 1.0 - step.confidence,
                    impact: Impact::PlanFailure,
                    affected_steps: vec![step.id],
                    mitigations: self.generate_mitigations(step),
                });
            }

            // Assumption-dependent steps
            for assumption in &plan.assumptions {
                if step.depends_on(assumption) && assumption.confidence < 0.8 {
                    risks.push(Risk {
                        id: RiskId::new(),
                        description: format!("Step depends on uncertain assumption: {}", assumption.description),
                        probability: 1.0 - assumption.confidence,
                        impact: Impact::PlanFailure,
                        affected_steps: vec![step.id],
                        mitigations: vec![],
                    });
                }
            }
        }

        // Historical analysis
        for similar_plan in self.find_similar_historical(plan) {
            if let Some(failure) = &similar_plan.failure {
                risks.push(Risk {
                    id: RiskId::new(),
                    description: format!("Similar plan failed previously: {}", failure.reason),
                    probability: similar_plan.failure_rate,
                    impact: Impact::PlanFailure,
                    affected_steps: vec![],
                    mitigations: vec![],
                });
            }
        }

        risks
    }

    /// Generate mitigations for a step
    fn generate_mitigations(&self, step: &PlanStep) -> Vec<Mitigation> {
        let mut mitigations = Vec::new();

        // Retry mitigation
        mitigations.push(Mitigation {
            description: "Retry on failure".to_string(),
            actions: vec![step.action.clone()],
            effectiveness: 0.5,
            cost: step.cost.clone(),
        });

        // Alternative action mitigation
        if let Some(alternative) = self.find_alternative(&step.action) {
            mitigations.push(Mitigation {
                description: format!("Use alternative: {}", alternative.name()),
                actions: vec![alternative],
                effectiveness: 0.7,
                cost: ResourceCost::default(),
            });
        }

        mitigations
    }
}
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           OMEN DATA FLOW                                     │
└─────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐
  │    GOAL      │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐     ┌──────────────┐
  │   STRATEGY   │◀───▶│   BELIEFS    │
  │   SELECTOR   │     │              │
  └──────┬───────┘     └──────────────┘
         │
         ▼
  ┌──────────────┐     ┌──────────────┐
  │   PLANNING   │◀───▶│   ACTIONS    │
  │   ENGINE     │     │   LIBRARY    │
  └──────┬───────┘     └──────────────┘
         │
         ▼
  ┌──────────────┐
  │    PLAN      │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐     ┌──────────────┐
  │     RISK     │◀───▶│ CONTINGENCY  │
  │   ASSESSOR   │     │   PLANNER    │
  └──────┬───────┘     └──────────────┘
         │
         ▼
  ┌──────────────┐
  │   OPTIMIZED  │
  │     PLAN     │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐     ┌──────────────┐
  │   EXECUTOR   │◀───▶│   MONITOR    │
  │              │     │              │
  └──────┬───────┘     └──────────────┘
         │
         ▼
  ┌──────────────┐
  │   OUTCOME    │
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │   LEARNER    │
  └──────────────┘
```

---

*Architecture for minds that plan*
