# Oracle

**Explainability Infrastructure for Sigil Agents**

> *"Understanding is the foundation of trust - let me show you why."*

Oracle provides the infrastructure for AI agents to explain their reasoning, decisions, and actions in ways humans can understand and verify. Not as an afterthought, but as a core capability that enables meaningful oversight and collaboration.

## Philosophy

AI systems often operate as black boxes - producing outputs without insight into how or why. This is unacceptable for agents that work alongside humans, because:

- **Trust requires understanding**: You can't trust what you can't understand
- **Oversight requires visibility**: You can't oversee what you can't see
- **Correction requires insight**: You can't fix what you don't understand
- **Learning requires explanation**: Both parties learn from understanding

Oracle makes agent reasoning transparent, traceable, and verifiable.

## Core Concepts

### Explanations

Every significant agent decision can generate an explanation:

```sigil
let explanation = oracle.explain(decision)?;

// Multi-level explanations
println!("Summary: {}", explanation.summary());
println!("Reasoning: {}", explanation.reasoning());
println!("Evidence: {:?}", explanation.evidence());
println!("Alternatives: {:?}", explanation.alternatives_considered());
println!("Confidence: {}", explanation.confidence());
```

### Reasoning Traces

Track the chain of reasoning that led to a conclusion:

```sigil
// Enable tracing
oracle.trace_on();

// Make decisions (traces are recorded)
let action = agent.deliberate(context);

// Get trace
let trace = oracle.get_trace();

for step in trace.steps() {
    println!("{}: {}", step.operation, step.description);
    println!("  Inputs: {:?}", step.inputs);
    println!("  Output: {:?}", step.output);
    println!("  Reasoning: {}", step.reasoning);
}
```

### Counterfactuals

Understand decisions through alternatives:

```sigil
// Why did you choose this action?
let counterfactual = oracle.counterfactual(
    actual: chosen_action,
    alternative: other_action,
)?;

println!("I chose {} instead of {} because:",
    counterfactual.actual,
    counterfactual.alternative
);
for reason in counterfactual.reasons {
    println!("  - {}", reason);
}
```

### Evidence Attribution

Link conclusions to their supporting evidence:

```sigil
let attribution = oracle.attribute(conclusion)?;

println!("Conclusion: {}", attribution.conclusion);
println!("Based on:");
for evidence in attribution.evidence {
    println!("  [{:.0}%] {}: {}",
        evidence.weight * 100.0,
        evidence.source,
        evidence.content
    );
}
```

## Quick Start

### Basic Explanation

```sigil
use oracle::{Oracle, ExplanationLevel};

// Create oracle
let oracle = Oracle::new();

// Attach to decision-making
let decision = agent.deliberate_with_oracle(context, &oracle);

// Get explanation
let explanation = oracle.explain(&decision, ExplanationLevel::Standard)?;

println!("{}", explanation.to_human_readable());
```

### Proactive Explanation

```sigil
use oracle::{Oracle, ExplainTrigger};

let oracle = Oracle::new()
    .explain_when(ExplainTrigger::HighStakes)
    .explain_when(ExplainTrigger::LowConfidence)
    .explain_when(ExplainTrigger::Requested)
    .build();

// Agent proactively explains
agent.on_action(|action| {
    if oracle.should_explain(&action) {
        let explanation = oracle.explain(&action, ExplanationLevel::Full)?;
        covenant.inform(&format!("Taking action: {}\n\nReasoning: {}",
            action.description,
            explanation.reasoning()
        ));
    }
});
```

### Interactive Explanation

```sigil
use oracle::{Oracle, InteractiveExplainer};

let explainer = oracle.interactive_explainer(decision);

// Human can drill down
loop {
    println!("{}", explainer.current_explanation());
    println!("\nAsk me:");
    println!("  'why' - explain further");
    println!("  'what if X' - explore alternative");
    println!("  'evidence' - show supporting data");
    println!("  'confidence' - explain uncertainty");
    println!("  'done' - finish");

    match human_input() {
        "why" => explainer.drill_down(),
        input if input.starts_with("what if") => {
            explainer.counterfactual(&input[8..])
        }
        "evidence" => explainer.show_evidence(),
        "confidence" => explainer.explain_confidence(),
        "done" => break,
        _ => println!("I don't understand that question"),
    }
}
```

## Key Features

### Multi-Level Explanations

Different audiences need different levels of detail:

```sigil
// Brief summary for quick understanding
let brief = oracle.explain(&decision, ExplanationLevel::Brief)?;
// "I recommended Option A because it's faster and meets all requirements."

// Standard explanation for normal use
let standard = oracle.explain(&decision, ExplanationLevel::Standard)?;
// Includes reasoning, key evidence, and confidence

// Full explanation for deep understanding
let full = oracle.explain(&decision, ExplanationLevel::Full)?;
// Includes all reasoning steps, all evidence, alternatives considered,
// uncertainty analysis, and counterfactuals

// Technical explanation for debugging
let technical = oracle.explain(&decision, ExplanationLevel::Technical)?;
// Includes internal representations, probability distributions,
// and algorithmic details
```

### Confidence Communication

Honest communication of certainty and uncertainty:

```sigil
let confidence = oracle.confidence_breakdown(&decision);

println!("Overall confidence: {:.0}%", confidence.overall * 100.0);
println!("\nBreakdown:");
println!("  Evidence strength: {:.0}%", confidence.evidence * 100.0);
println!("  Reasoning validity: {:.0}%", confidence.reasoning * 100.0);
println!("  Similar past experience: {:.0}%", confidence.experience * 100.0);

if confidence.overall < 0.7 {
    println!("\nUncertainties:");
    for uncertainty in confidence.uncertainties {
        println!("  - {}", uncertainty);
    }
}
```

### Reasoning Visualization

Generate visual representations of reasoning:

```sigil
// Generate reasoning graph
let graph = oracle.reasoning_graph(&decision);

// Export for visualization
graph.to_dot("reasoning.dot")?;     // Graphviz format
graph.to_mermaid("reasoning.md")?;  // Mermaid diagram
graph.to_json("reasoning.json")?;   // Structured data

// ASCII visualization for terminal
println!("{}", graph.to_ascii());
```

### Temporal Explanations

Explain how understanding evolved over time:

```sigil
let temporal = oracle.temporal_explanation(topic)?;

println!("How my understanding of '{}' evolved:", topic);
for change in temporal.changes {
    println!("\n[{}] {}", change.timestamp, change.event);
    println!("  Before: {}", change.belief_before);
    println!("  After: {}", change.belief_after);
    println!("  Because: {}", change.reason);
}
```

### Analogical Explanations

Explain complex concepts through analogies:

```sigil
let analogy = oracle.find_analogy(concept, audience)?;

println!("Think of {} like {}:", concept, analogy.analog);
println!("{}", analogy.mapping);
println!("\nWhere this analogy breaks down:");
for limitation in analogy.limitations {
    println!("  - {}", limitation);
}
```

## Integration with Agent Stack

### With Omen (Planning)

```sigil
use oracle::Oracle;
use omen::Omen;

// Oracle traces planning decisions
let oracle = Oracle::new();
let planner = Omen::new().with_oracle(&oracle);

let plan = planner.plan(goal)?;

// Explain the plan
let explanation = oracle.explain_plan(&plan)?;

println!("Plan to achieve: {}", goal);
for (i, step) in explanation.steps.iter().enumerate() {
    println!("\nStep {}: {}", i + 1, step.action);
    println!("  Why: {}", step.reason);
    println!("  Prerequisites: {:?}", step.prerequisites);
    println!("  Expected outcome: {}", step.expected_outcome);
}
```

### With Engram (Memory)

```sigil
use oracle::Oracle;
use engram::Engram;

// Oracle attributes to memories
let oracle = Oracle::new().with_memory(&memory);

// When explaining, show what memories informed the decision
let explanation = oracle.explain(&decision)?;

println!("This decision was informed by:");
for memory in explanation.memory_sources() {
    println!("  - [{}] {} ({})",
        memory.memory_type,
        memory.content.truncate(50),
        memory.epistemic
    );
}
```

### With Covenant (Collaboration)

```sigil
use oracle::Oracle;
use covenant::Covenant;

// Oracle provides explanations for covenant interactions
daemon ExplainableAgent {
    oracle: Oracle,
    covenant: Covenant,

    fn on_action(&mut self, action: &Action) {
        // If covenant requires explanation
        if self.covenant.requires_explanation(action) {
            let explanation = self.oracle.explain(action, ExplanationLevel::Standard)?;
            self.covenant.inform(&explanation.to_human_readable());
        }
    }

    fn on_approval_request(&mut self, action: &Action) {
        // Always explain approval requests
        let explanation = self.oracle.explain(action, ExplanationLevel::Full)?;

        self.covenant.handoff(HandoffType::ApprovalRequest {
            action: action.clone(),
            reason: explanation.summary(),
            full_explanation: explanation.to_human_readable(),
        });
    }
}
```

### With Aegis (Security)

```sigil
use oracle::Oracle;
use aegis::Aegis;

// Oracle explains security decisions
impl Oracle {
    pub fn explain_security_decision(&self, action: &Action, aegis: &Aegis) -> SecurityExplanation {
        let decision = aegis.check_action(action);

        match decision {
            ActionDecision::Allow => SecurityExplanation {
                allowed: true,
                reason: "Action within permitted boundaries".to_string(),
                relevant_boundaries: aegis.relevant_boundaries(action),
            },
            ActionDecision::Block { reason } => SecurityExplanation {
                allowed: false,
                reason: reason.clone(),
                relevant_boundaries: aegis.relevant_boundaries(action),
                violation_explanation: Some(self.explain_violation(action, &reason)),
            },
            _ => { /* ... */ }
        }
    }
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                               ORACLE                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                      EXPLANATION ENGINE                                │ │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐            │ │
│  │  │  SUMMARY  │ │ REASONING │ │ EVIDENCE  │ │CONFIDENCE │            │ │
│  │  │ GENERATOR │ │ EXTRACTOR │ │ ATTRIBUTOR│ │ ANALYZER  │            │ │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘            │ │
│  └────────────────────────────────┬──────────────────────────────────────┘ │
│                                   │                                         │
│  ┌────────────────────────────────▼──────────────────────────────────────┐ │
│  │                        TRACE SYSTEM                                    │ │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐            │ │
│  │  │  TRACE    │ │   STEP    │ │ REASONING │ │  CAUSAL   │            │ │
│  │  │ COLLECTOR │ │ RECORDER  │ │   CHAIN   │ │   GRAPH   │            │ │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘            │ │
│  └────────────────────────────────┬──────────────────────────────────────┘ │
│                                   │                                         │
│  ┌────────────────────────────────▼──────────────────────────────────────┐ │
│  │                    COUNTERFACTUAL ENGINE                               │ │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐                           │ │
│  │  │ALTERNATIVE│ │   WHAT-IF │ │ CONTRAST  │                           │ │
│  │  │ GENERATOR │ │  ANALYZER │ │ EXPLAINER │                           │ │
│  │  └───────────┘ └───────────┘ └───────────┘                           │ │
│  └────────────────────────────────┬──────────────────────────────────────┘ │
│                                   │                                         │
│  ┌────────────────────────────────▼──────────────────────────────────────┐ │
│  │                    PRESENTATION LAYER                                  │ │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐            │ │
│  │  │  HUMAN    │ │   GRAPH   │ │  ANALOGY  │ │INTERACTIVE│            │ │
│  │  │ READABLE  │ │VISUALIZER │ │  FINDER   │ │ EXPLORER  │            │ │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘            │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Why "Oracle"?

In ancient tradition, oracles didn't just predict - they *revealed*. They provided insight into the hidden workings of fate and decision.

Our Oracle serves a similar purpose: revealing the hidden workings of agent reasoning. Not mystically, but clearly and verifiably.

The name also carries connotations of:
- **Wisdom**: Deep understanding, not just information
- **Truth**: Honest, accurate explanations
- **Consultation**: Available when needed, not imposed
- **Insight**: Seeing what's not immediately obvious

Oracle makes the invisible visible, the implicit explicit, and the complex comprehensible.

---

*Understanding is the foundation of trust - let Oracle show you why*
