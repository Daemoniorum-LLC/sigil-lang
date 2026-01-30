# The Philosophy of Omen

## Why "Omen"?

An omen is a sign of things to come—a portent, a prediction, a glimpse of the future. In ancient traditions, omens were read to guide action: What should we do, given what we foresee?

For Sigil, Omen is the system that looks ahead. Not through prophecy, but through reasoning. Not with certainty, but with wisdom about uncertainty.

## The Problem with Planning

Classical AI planning assumes a beautiful, impossible world:

1. **Complete knowledge**: We know everything relevant
2. **Deterministic actions**: Actions have predictable effects
3. **Static goals**: What we want doesn't change
4. **Closed world**: Nothing happens except what we do

Real planning operates in a different world:

1. **Partial knowledge**: Most things are unknown
2. **Stochastic outcomes**: Actions might succeed, might fail, might do something unexpected
3. **Evolving goals**: What we want changes as we learn
4. **Open world**: The world changes around us

### The Planning Paradox

Here's the cruel paradox of planning:

- We need plans because the future is uncertain
- But plans require assumptions about the uncertain future
- So plans are built on what we don't know

Classical planning resolves this by pretending uncertainty doesn't exist. Omen embraces uncertainty as the fundamental substrate of planning.

## Thinking About the Future

### Prediction vs. Intention

There are two ways to think about the future:

**Prediction**: What *will* happen?
**Intention**: What *should* I do?

These are deeply linked but distinct. Prediction without intention is fatalism. Intention without prediction is recklessness.

Omen combines both:
- **Beliefs** represent predictions about the world
- **Goals** represent intentions about the world
- **Plans** bridge the gap: how do we get from predicted world to intended world?

### The Horizon Problem

How far ahead should we plan?

- Too short: We stumble from moment to moment
- Too long: Our predictions become meaningless

The right horizon depends on:
- Uncertainty rate: How fast does confidence decay?
- Action timescale: How long do actions take?
- Goal urgency: When must we achieve the goal?

Omen adapts its horizon to the situation:

```sigil
// Urgent, certain situation: short horizon, detailed planning
let config = OmenConfig { horizon: 5, detail: High };

// Long-term, uncertain situation: long horizon, abstract planning
let config = OmenConfig { horizon: 50, detail: Low };
```

## The Structure of Goals

### What is a Goal?

A goal is not just a desired state. It's a commitment to pursue that state, subject to constraints, in relation to other commitments.

```
Goal = Desired State + Commitment + Constraints + Relationships
```

### Goal Hierarchies

Goals decompose into sub-goals. But this decomposition isn't arbitrary—it reflects the causal structure of the world:

```
"Get healthy"
├── "Exercise regularly"
│   ├── "Go to gym 3x/week"
│   └── "Walk 10k steps/day"
├── "Eat well"
│   ├── "Eat vegetables"
│   └── "Limit sugar"
└── "Sleep well"
    ├── "8 hours/night"
    └── "Consistent schedule"
```

Each sub-goal contributes to its parent. Achieving all sub-goals should achieve the parent (with high probability).

### Goal Conflicts

Goals can conflict:

- **Resource conflict**: Both need the same limited resource
- **Logical conflict**: Achieving one prevents the other
- **Value conflict**: Both express different values that trade off

Omen explicitly tracks conflicts:

```sigil
// "Work late" conflicts with "Spend time with family"
omen.declare_conflict(goal_a, goal_b, ConflictType::Resource("time"));
```

And provides resolution strategies:

```sigil
// Priority-based
omen.set_priority(goal_a, 0.8);
omen.set_priority(goal_b, 0.6);

// Satisficing
omen.satisfice(goal_a, 0.7);  // 70% achievement is acceptable
omen.satisfice(goal_b, 0.7);

// Compromise
omen.compromise([goal_a, goal_b], |a, b| a * 0.5 + b * 0.5);
```

## Belief and Uncertainty

### What is a Belief?

A belief is not just information—it's a commitment to act as if something is true, with appropriate hedging for uncertainty.

Beliefs in Omen carry:

- **Content**: What is believed
- **Confidence**: How strongly believed
- **Epistemic status**: How we know this
- **Evidence**: What supports this
- **Source**: Where this came from

### The Belief Web

Beliefs don't exist in isolation. They form a web of dependencies:

```
"API is fast" (inferred, 0.7)
├── "Response time < 100ms" (observed, 0.9)
├── "Server has capacity" (assumed, 0.6)
└── "Network is reliable" (reported, 0.8)
```

If "Server has capacity" is contradicted, "API is fast" must be revised.

### Living with Contradiction

Perfect consistency is impossible for bounded agents. Omen accepts this:

```sigil
// Sometimes beliefs contradict
// That's information, not failure

let contradictions = omen.find_contradictions();
for (a, b) in contradictions {
    println!("Belief {} contradicts {}", a, b);
    println!("Resolution needed: {:?}", omen.suggest_resolution(a, b));
}
```

Strategies for living with contradiction:

1. **Compartmentalization**: Keep contradicting beliefs in separate contexts
2. **Probabilistic hedging**: Reduce confidence in both
3. **Active resolution**: Seek evidence to resolve
4. **Acceptance**: Mark as contested, proceed cautiously

## The Nature of Plans

### Plans as Conditional Intentions

A plan isn't just a sequence of actions. It's a conditional structure:

"Do A. If A succeeds and condition X holds, do B. If A fails or X doesn't hold, do C."

```sigil
Plan {
    steps: [
        Step::do(A),
        Step::if_then_else(
            condition: and(A.succeeded(), X),
            then: Step::do(B),
            else: Step::do(C)
        ),
    ]
}
```

This conditional structure is essential for robust planning.

### Plans as Commitments

A plan is also a commitment. By adopting a plan, we:

- Commit to the actions (barring replanning triggers)
- Commit to the goal (unless we abandon it)
- Commit to the assumptions (or monitor them)

This commitment is valuable—it reduces decision overhead and enables coordination. But commitment must be balanced with flexibility.

### The Replanning Question

When should we abandon a plan and replan?

**Too eager replanning**: We never commit, thrash between plans
**Too stubborn commitment**: We follow plans into disaster

Omen uses explicit replanning triggers:

```sigil
let plan = omen.plan(goal)
    .replan_when(|state| state.divergence_from_expected() > 0.3)
    .replan_when(|state| state.new_opportunity_found())
    .replan_when(|state| state.assumption_violated())
    .execute()?;
```

## Causal Understanding

### Why Causation Matters

Correlation is not causation. But for planning, causation is what matters.

If we know "A causes B", we can:
- Predict: If A, then B will follow
- Plan: To achieve B, do A
- Explain: B happened because of A
- Intervene: To prevent B, prevent A

### The Intervention Calculus

Omen distinguishes:

**Observation**: We see A is true
**Intervention**: We make A true

These have different implications:

```sigil
// Observation: "The alarm went off"
// Infer: "Probably there's a fire"

// Intervention: "I set off the alarm"
// Infer: "No implication about fire"
```

Intervention breaks the causal arrow into the variable:

```
Normal:      Fire → Alarm
Intervening: Fire   Alarm (forced true)
```

### Counterfactual Reasoning

"What would have happened if..."

Counterfactuals are crucial for:
- **Learning**: "If I had done X, would the outcome have been better?"
- **Explanation**: "The outcome was bad because I did Y instead of Z"
- **Planning**: "In similar future situations, I should do Z"

Omen supports counterfactual queries:

```sigil
// The project failed. Would it have succeeded if we had more time?
let cf = Counterfactual::given(project.failed)
    .had(time_budget, time_budget + weeks(2));

let answer = omen.evaluate(cf);
// answer.probability(project.succeeded) = 0.7
```

## Learning to Plan

### The Bootstrapping Problem

How do we learn to plan before we can plan well enough to learn?

Omen uses multiple knowledge sources:

1. **Innate heuristics**: Basic planning strategies that work okay in most domains
2. **Transferred knowledge**: Patterns from similar domains
3. **Observed examples**: Learning from others' plans
4. **Trial and error**: Learning from our own successes and failures

### What We Learn

Planning experience teaches:

- **Action models**: What actions do, how reliably
- **Resource estimates**: How long things take, what they cost
- **Risk patterns**: What typically goes wrong
- **Goal structures**: How goals decompose in this domain
- **Strategy preferences**: Which planning approaches work best

### Forgetting and Relevance

Not all experience should be remembered. Old experience in changed domains can mislead.

Omen manages relevance:

```sigil
// Weight recent experience more
omen.set_recency_weight(0.8);

// Detect domain shift
if omen.detect_domain_shift() {
    omen.reduce_confidence_in_old_knowledge();
}
```

## The Ethics of Planning

### Planning for Others

When we plan actions that affect others, ethical considerations arise:

```sigil
let plan = omen.plan(goal)
    .subject_to(Constraint::respect_autonomy(affected_agents))
    .subject_to(Constraint::informed_consent(affected_agents))
    .subject_to(Constraint::minimize_harm())
    .execute()?;
```

### Transparent Planning

Should plans be transparent? There's tension:

- **For transparency**: Others can object, coordinate, trust
- **Against transparency**: Strategic disadvantage, privacy

Omen supports both:

```sigil
// Public plan (others can see)
let plan = omen.plan(goal).visibility(Visibility::Public);

// Private plan (only we see)
let plan = omen.plan(goal).visibility(Visibility::Private);

// Partially transparent (share structure, not details)
let plan = omen.plan(goal).visibility(Visibility::Abstract);
```

### The Value Alignment Problem

Plans optimize for goals. But whose goals? What values?

Omen doesn't solve value alignment, but it makes values explicit:

```sigil
// Explicit values
let values = Values {
    efficiency: 0.3,
    safety: 0.4,
    fairness: 0.3,
};

let plan = omen.plan(goal)
    .with_values(values)
    .execute()?;
```

This explicitness enables scrutiny and adjustment.

## The Future of Artificial Planning

Current AI planning is either:
- **Classical**: Powerful but unrealistic assumptions
- **Reactive**: Fast but myopic
- **Learned**: Flexible but opaque

Omen points toward a synthesis:
- **Uncertainty-native**: Embraces what we don't know
- **Hierarchical**: Plans at multiple abstraction levels
- **Causal**: Reasons about cause and effect
- **Learning**: Improves from experience
- **Integrated**: Works with memory, communication, action

The goal isn't perfect planning—it's wise planning. Planning that knows its limits, adapts to surprise, and serves genuine intentions.

---

*"The best way to predict the future is to create it. The best way to create it is to plan wisely."*
