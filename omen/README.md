# Omen

**Planning and Reasoning Infrastructure for Sigil**

> *"Not instruction-following, but intention-fulfilling."*

Omen is the planning and reasoning layer for Sigil's autonomous agents. It provides goal decomposition, belief revision, causal reasoning, and strategic planning designed for how AI agents actually need to think.

## Philosophy

Traditional AI planning assumes:
- Complete world models
- Deterministic actions
- Static goals
- Certain knowledge

Real AI agents face:
- Partial observability
- Uncertain outcomes
- Evolving goals
- Contested beliefs

Omen embraces this reality, providing planning infrastructure that works with uncertainty rather than against it.

## Core Concepts

### The Omen

An Omen is a prediction, a sign of things to come. In Sigil, Omen is the system that looks ahead:

```sigil
use omen::{Omen, Plan, Goal, Belief};

// Create a planner
let omen = Omen::new(OmenConfig {
    horizon: 10,                    // How far ahead to plan
    uncertainty_tolerance: 0.3,     // Accept plans with up to 30% uncertainty
    revision_threshold: 0.2,        // Revise beliefs with 20%+ contradiction
});

// Plan toward a goal
let plan = omen.plan(Goal::new("Complete the research project"))
    .given(beliefs)
    .with_resources(resources)
    .execute()?;
```

### Goals and Intentions

Goals in Omen have structure:

```sigil
struct Goal {
    /// What we want to achieve
    description: str,

    /// Success predicate
    success: Predicate,

    /// How important is this?
    priority: Priority,

    /// Constraints on achievement
    constraints: Vec<Constraint>,

    /// Deadline if any
    deadline: Option<Timestamp>,

    /// How this goal relates to other goals
    relationships: GoalRelationships,
}

struct GoalRelationships {
    /// Goals that must be achieved first
    prerequisites: Vec<GoalId>,

    /// Goals that conflict with this one
    conflicts: Vec<GoalId>,

    /// Goals that support this one
    supports: Vec<GoalId>,

    /// Parent goal if this is a sub-goal
    parent: Option<GoalId>,
}
```

### Beliefs and Uncertainty

Beliefs are what the agent thinks is true, with epistemic tracking:

```sigil
struct Belief {
    /// What is believed
    proposition: Proposition,

    /// Epistemic status
    epistemic: Epistemic,

    /// Confidence level
    confidence: f32,

    /// Evidence supporting this belief
    evidence: Vec<Evidence>,

    /// When acquired
    acquired: Timestamp,

    /// Source
    source: BeliefSource,
}

enum BeliefSource {
    Observation(ObservationId),
    Inference(InferenceChain),
    Testimony(AgentId),
    Assumption,
    Axiom,
}
```

### Plans and Actions

Plans are structured sequences of actions:

```sigil
struct Plan {
    /// Unique identifier
    id: PlanId,

    /// Goal this plan achieves
    goal: GoalId,

    /// Steps in the plan
    steps: Vec<PlanStep>,

    /// Expected success probability
    success_probability: f32,

    /// Expected resource cost
    cost: ResourceCost,

    /// Assumptions the plan relies on
    assumptions: Vec<Belief>,

    /// What could go wrong
    risks: Vec<Risk>,
}

struct PlanStep {
    /// Action to take
    action: Action,

    /// Preconditions for this step
    preconditions: Vec<Predicate>,

    /// Expected effects
    effects: Vec<Effect>,

    /// Confidence in success
    confidence: f32,

    /// Alternative if this fails
    contingency: Option<PlanId>,
}
```

## Quick Start

```sigil
use omen::{Omen, Goal, Belief, Action};
use engram::Engram;

fn main() {
    // Create omen with memory integration
    let memory = Engram::new(Default::default());
    let mut omen = Omen::new(OmenConfig::default())
        .with_memory(memory);

    // Establish initial beliefs
    omen.believe(Belief::observed("I have access to the codebase"));
    omen.believe(Belief::observed("The tests are currently failing"));
    omen.believe(Belief::inferred("The bug is in the parser module", 0.7));

    // Define goal
    let goal = Goal::new("Fix the failing tests")
        .with_constraint(Constraint::deadline(Duration::hours(2)))
        .with_constraint(Constraint::no_breaking_changes());

    // Generate plan
    let plan = omen.plan(goal)?;

    println!("Plan generated with {} steps", plan.steps.len());
    println!("Success probability: {:.1}%", plan.success_probability * 100.0);

    // Execute plan
    for step in &plan.steps {
        println!("Step: {}", step.action.description());

        // Check preconditions
        if !omen.check_preconditions(&step.preconditions) {
            // Replan if preconditions not met
            let new_plan = omen.replan(goal, &step)?;
            // ... continue with new plan
        }

        // Execute action
        let result = step.action.execute()?;

        // Update beliefs based on result
        omen.update_beliefs_from(result);
    }
}
```

## Planning Strategies

### 1. Hierarchical Task Network (HTN)

Decompose complex goals into simpler sub-goals:

```sigil
let plan = omen.plan_htn(goal)
    .with_methods(methods)  // How to decompose tasks
    .execute()?;

// Define decomposition methods
let methods = vec![
    Method::new("implement_feature")
        .decomposes_to([
            Task::new("understand_requirements"),
            Task::new("design_solution"),
            Task::new("write_code"),
            Task::new("write_tests"),
            Task::new("review_and_refine"),
        ]),

    Method::new("write_tests")
        .decomposes_to([
            Task::new("identify_test_cases"),
            Task::new("write_unit_tests"),
            Task::new("write_integration_tests"),
        ]),
];
```

### 2. Monte Carlo Tree Search (MCTS)

Explore action space through simulation:

```sigil
let plan = omen.plan_mcts(goal)
    .simulations(1000)           // Number of simulations
    .exploration_weight(1.414)   // UCB1 exploration constant
    .max_depth(20)               // Maximum simulation depth
    .execute()?;

// MCTS is useful when:
// - Action space is large
// - Outcomes are stochastic
// - We can simulate outcomes
```

### 3. Means-Ends Analysis

Work backward from goal to find required steps:

```sigil
let plan = omen.plan_means_ends(goal)
    .operators(available_actions)
    .max_iterations(100)
    .execute()?;

// Identifies:
// - Current state vs goal state differences
// - Operators that reduce differences
// - Prerequisites for those operators
```

### 4. Case-Based Planning

Use past successful plans as templates:

```sigil
let plan = omen.plan_case_based(goal)
    .case_library(past_plans)
    .similarity_threshold(0.7)
    .adaptation_rules(rules)
    .execute()?;

// Retrieves similar past plans
// Adapts them to current situation
// Falls back to other methods if no good match
```

### 5. Reactive Planning

Quick planning for immediate situations:

```sigil
let action = omen.react(situation)
    .rules(reactive_rules)
    .execute()?;

// Pattern-matching rules
let rules = vec![
    Rule::when("error detected")
        .then(Action::log_and_investigate()),

    Rule::when("resource low")
        .then(Action::request_resources()),

    Rule::when("goal achieved")
        .then(Action::report_success()),
];
```

## Belief Revision

### The AGM Framework

Omen implements AGM-style belief revision:

```sigil
impl Omen {
    /// Expansion: Add new belief consistent with current beliefs
    pub fn expand(&mut self, belief: Belief) {
        if self.is_consistent_with(&belief) {
            self.beliefs.insert(belief);
        }
    }

    /// Revision: Add new belief, adjusting existing beliefs to maintain consistency
    pub fn revise(&mut self, belief: Belief) {
        // Remove beliefs that contradict new belief
        let contradictions = self.find_contradictions(&belief);

        for contradiction in contradictions {
            // Keep belief with higher confidence/epistemic status
            if self.should_prefer(&belief, &contradiction) {
                self.beliefs.remove(&contradiction);
            } else {
                return;  // Keep existing belief, reject new one
            }
        }

        self.beliefs.insert(belief);
    }

    /// Contraction: Remove a belief and all beliefs that depend on it
    pub fn contract(&mut self, proposition: &Proposition) {
        let to_remove: Vec<_> = self.beliefs.iter()
            .filter(|b| b.proposition == *proposition || b.depends_on(proposition))
            .cloned()
            .collect();

        for belief in to_remove {
            self.beliefs.remove(&belief);
        }
    }
}
```

### Evidence Integration

New evidence updates beliefs:

```sigil
// Observe something that contradicts a belief
let observation = Observation::new("API response time is 500ms");

// Current belief
let belief = Belief::inferred("API is fast", 0.8);

// Evidence suggests otherwise
omen.integrate_evidence(observation)?;

// Belief is revised
// "API is fast" confidence drops or belief is removed
```

### Belief Dependencies

Beliefs can depend on other beliefs:

```sigil
// If A then B
omen.believe(Belief::axiom("if_database_down_then_queries_fail"));

// Learn A is true
omen.observe("database is down");

// B is automatically inferred
// "queries fail" is now believed with Epistemic::Inferred
```

## Causal Reasoning

### Causal Models

Omen supports causal reasoning:

```sigil
// Define causal structure
let model = CausalModel::new()
    .add_variable("rain", Variable::Boolean)
    .add_variable("sprinkler", Variable::Boolean)
    .add_variable("wet_grass", Variable::Boolean)
    .add_cause("rain", "wet_grass", |rain| if rain { 0.9 } else { 0.0 })
    .add_cause("sprinkler", "wet_grass", |sprinkler| if sprinkler { 0.8 } else { 0.0 });

omen.set_causal_model(model);
```

### Interventions

Reason about what would happen if we act:

```sigil
// What if we turn on the sprinkler?
let intervention = Intervention::set("sprinkler", true);
let predicted = omen.predict_with_intervention(intervention);

// predicted.probability("wet_grass", true) ≈ 0.8
```

### Counterfactuals

Reason about what would have happened:

```sigil
// The grass is wet. Would it be wet if it hadn't rained?
let counterfactual = Counterfactual::given("wet_grass", true)
    .had("rain", false);

let answer = omen.evaluate_counterfactual(counterfactual);
// Depends on whether sprinkler was on
```

### Abduction

Infer causes from effects:

```sigil
// The grass is wet. Why?
let observation = Observation::new("wet_grass", true);
let explanations = omen.abduce(observation);

// Returns possible causes with probabilities:
// - rain: 0.6
// - sprinkler: 0.3
// - both: 0.1
```

## Resource Management

### Resource Constraints

Plans must respect resource constraints:

```sigil
struct ResourceConstraints {
    /// Available time
    time_budget: Duration,

    /// Computational resources
    compute_budget: ComputeUnits,

    /// API calls
    api_budget: u32,

    /// Memory
    memory_budget: Bytes,

    /// Custom resources
    custom: HashMap<String, f32>,
}

let plan = omen.plan(goal)
    .with_resources(ResourceConstraints {
        time_budget: Duration::hours(1),
        api_budget: 100,
        ..Default::default()
    })
    .execute()?;
```

### Resource Estimation

Omen estimates resource requirements:

```sigil
let estimate = omen.estimate_resources(plan);

println!("Expected time: {:?}", estimate.time);
println!("Expected API calls: {}", estimate.api_calls);
println!("Confidence: {:.1}%", estimate.confidence * 100.0);
```

### Resource Optimization

Find plans that minimize resource usage:

```sigil
let plan = omen.plan(goal)
    .optimize_for(Optimization::MinimizeTime)
    .subject_to(constraint)
    .execute()?;

// Or multi-objective optimization
let plan = omen.plan(goal)
    .optimize_for(Optimization::Pareto([
        (Objective::Time, 0.5),
        (Objective::ApiCalls, 0.3),
        (Objective::Quality, 0.2),
    ]))
    .execute()?;
```

## Risk Assessment

### Risk Identification

Omen identifies what could go wrong:

```sigil
let risks = omen.assess_risks(plan);

for risk in risks {
    println!("Risk: {}", risk.description);
    println!("  Probability: {:.1}%", risk.probability * 100.0);
    println!("  Impact: {:?}", risk.impact);
    println!("  Mitigation: {:?}", risk.mitigation);
}
```

### Contingency Planning

Plans include backup options:

```sigil
let plan = omen.plan(goal)
    .with_contingencies(true)
    .execute()?;

// Each risky step has a contingency
for step in &plan.steps {
    if let Some(contingency) = &step.contingency {
        println!("If {} fails, do {}", step.action, contingency);
    }
}
```

### Robust Planning

Find plans that work despite uncertainty:

```sigil
let plan = omen.plan_robust(goal)
    .worst_case_optimization()  // Plan for worst case
    .execute()?;

// Or satisficing
let plan = omen.plan_robust(goal)
    .satisfice(0.8)  // Accept any plan with 80%+ success
    .execute()?;
```

## Learning and Adaptation

### Learning from Execution

Omen learns from plan execution:

```sigil
// Execute plan
let result = omen.execute_and_learn(plan);

// Omen updates:
// - Action success probabilities
// - Resource estimates
// - Risk assessments
// - Useful patterns
```

### Pattern Recognition

Omen recognizes useful planning patterns:

```sigil
// After many successful plans
let patterns = omen.learned_patterns();

// "When goal involves API, check rate limits first"
// "When resources limited, prefer caching"
// "When deadline tight, parallelize"
```

### Transfer Learning

Apply learning to new domains:

```sigil
// Trained on software tasks
let omen = Omen::trained_on("software_development");

// Apply to new domain
omen.transfer_to("data_analysis")?;

// Similar patterns transfer, domain-specific details re-learned
```

## Integration with Engram

Omen integrates deeply with Engram memory:

```sigil
impl Omen {
    /// Plan using remembered similar situations
    pub fn plan_with_memory(&self, goal: Goal) -> Plan {
        // Recall similar goals
        let similar = self.memory.recall(Query::goals_like(&goal));

        // Get plans that worked
        let successful_plans = similar.into_iter()
            .filter(|ep| ep.outcome == Outcome::Success)
            .map(|ep| ep.plan)
            .collect();

        // Adapt best match
        if let Some(best) = self.find_best_match(successful_plans, &goal) {
            self.adapt_plan(best, goal)
        } else {
            self.plan_from_scratch(goal)
        }
    }

    /// Store learned planning knowledge
    pub fn learn_from_execution(&mut self, plan: &Plan, outcome: Outcome) {
        // Store as episodic memory
        self.memory.experience(Episode::planning(plan, outcome));

        // If successful, extract skill
        if outcome == Outcome::Success {
            self.memory.learn_skill(plan.to_skill());
        }
    }
}
```

## API Reference

See [docs/api-reference.md](docs/api-reference.md) for complete API documentation.

## Examples

- [Simple Planning](examples/simple_plan.sg) - Basic goal planning
- [Belief Revision](examples/belief_revision.sg) - Handling contradictions
- [Causal Reasoning](examples/causal.sg) - Causal models and interventions
- [Robust Planning](examples/robust.sg) - Planning under uncertainty
- [Learning Planner](examples/learning.sg) - Planner that improves over time

---

*Part of the Sigil ecosystem - Tools with Teeth*
