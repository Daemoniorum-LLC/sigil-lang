# Covenant

**Human-Agent Collaboration Infrastructure for Sigil**

> *"The bridge between minds - where trust is built, not assumed."*

Covenant provides the infrastructure for meaningful collaboration between humans and AI agents. Not mere command-and-control, but genuine partnership with shared understanding, negotiated boundaries, and mutual respect.

## Philosophy

Traditional human-AI interaction treats agents as tools: input goes in, output comes out. But truly beneficial AI requires something deeper - a *relationship* built on:

- **Shared Understanding**: Both parties know the goals, constraints, and context
- **Clear Boundaries**: Explicit about what the agent can, should, and must not do
- **Graceful Handoffs**: Knowing when to act, when to ask, when to defer
- **Mutual Adaptation**: Learning each other's preferences and working styles
- **Earned Trust**: Trust that grows through demonstrated reliability

Covenant makes these principles concrete.

## Core Concepts

### The Pact

A Pact is a formalized agreement between human and agent:

```sigil
let pact = Pact::negotiate()
    .human(HumanProfile::from_context())
    .agent(AgentProfile::current())
    .goals([
        SharedGoal::new("Complete the research project")
            .human_role("Provide direction and feedback")
            .agent_role("Research, synthesize, draft"),
    ])
    .boundaries([
        Boundary::require_approval("External communications"),
        Boundary::inform_after("File modifications"),
        Boundary::autonomous("Information gathering"),
    ])
    .preferences([
        Preference::communication_style(Style::Concise),
        Preference::check_in_frequency(Frequency::Milestones),
    ])
    .establish()?;
```

### Interaction Modes

Covenant defines clear modes of interaction:

| Mode | Agent Behavior | Human Role |
|------|---------------|------------|
| **Autonomous** | Acts independently within bounds | Monitors, available if needed |
| **Collaborative** | Works alongside, frequent sync | Active participant |
| **Supervised** | Proposes actions, awaits approval | Reviews and approves |
| **Guided** | Follows explicit instructions | Directs each step |
| **Paused** | Halted, awaiting resumption | Full control |

### Handoffs

Smooth transitions between human and agent work:

```sigil
// Agent recognizes need for human input
covenant.handoff(Handoff::decision_needed(
    context: "Multiple valid approaches to this problem",
    options: [
        Option::new("Approach A").with_tradeoffs("Fast but less thorough"),
        Option::new("Approach B").with_tradeoffs("Thorough but slower"),
    ],
    recommendation: Some("Approach B given project timeline"),
    urgency: Urgency::Normal,
));

// Human provides decision
covenant.on_human_input(|input| {
    match input {
        HumanInput::Decision { choice, reasoning } => {
            self.proceed_with(choice, reasoning);
        }
        HumanInput::Guidance { direction } => {
            self.adjust_approach(direction);
        }
        HumanInput::Takeover => {
            self.pause_and_brief();
        }
    }
});
```

### Trust Dynamics

Trust is earned, tracked, and respected:

```sigil
// Trust grows through positive interactions
covenant.trust.record_outcome(Outcome::Success {
    task: "Research synthesis",
    human_satisfaction: Satisfaction::High,
    within_boundaries: true,
});

// Trust affects autonomy
let autonomy_level = covenant.trust.suggested_autonomy();
match autonomy_level {
    Autonomy::High => self.set_mode(Mode::Autonomous),
    Autonomy::Medium => self.set_mode(Mode::Collaborative),
    Autonomy::Low => self.set_mode(Mode::Supervised),
    Autonomy::Minimal => self.set_mode(Mode::Guided),
}

// Violations reduce trust
covenant.trust.record_violation(Violation {
    boundary: "External communications",
    severity: Severity::Minor,
    context: "Sent email without approval",
    remediation: "Email recalled, approval requested",
});
```

## Quick Start

### Basic Collaboration

```sigil
use covenant::{Covenant, Pact, Mode, Handoff};

// Establish covenant with human
let covenant = Covenant::new()
    .with_pact(Pact::default_collaborative())
    .establish()?;

// Work in collaborative mode
covenant.set_mode(Mode::Collaborative);

// Check before significant actions
if covenant.requires_approval("delete_file") {
    let approval = covenant.request_approval(
        Action::DeleteFile { path: "data.txt" },
        Reason::new("File is outdated and causing confusion"),
    ).await?;

    if approval.granted {
        delete_file("data.txt")?;
        covenant.report_completion("File deleted as approved");
    }
}

// Proactive communication
covenant.inform("Found 3 relevant papers, beginning synthesis");

// Request input when needed
let guidance = covenant.request_guidance(
    Question::new("Should I prioritize depth or breadth in this analysis?"),
    Context::current(),
).await?;
```

### Adaptive Collaboration

```sigil
use covenant::{Covenant, AdaptiveMode, LearningProfile};

// Create covenant that learns preferences
let covenant = Covenant::new()
    .with_learning(LearningProfile::new())
    .adaptive(true)
    .establish()?;

// Covenant learns from interactions
covenant.on_feedback(|feedback| {
    match feedback {
        Feedback::TooVerbose => {
            self.learning.adjust(Dimension::Verbosity, -0.2);
        }
        Feedback::NeedMoreDetail => {
            self.learning.adjust(Dimension::Verbosity, 0.2);
        }
        Feedback::GoodTiming => {
            self.learning.reinforce(Dimension::CheckInFrequency);
        }
        Feedback::TooFrequent => {
            self.learning.adjust(Dimension::CheckInFrequency, -0.3);
        }
    }
});

// Apply learned preferences
let style = covenant.learned_preferences().communication_style();
self.communicate_with_style(message, style);
```

## Key Features

### Boundary Enforcement

```sigil
// Define boundaries
let boundaries = Boundaries::new()
    .always_allow([
        "read_public_files",
        "search_web",
        "take_notes",
    ])
    .require_approval([
        "send_email",
        "modify_files",
        "make_purchases",
        "contact_external",
    ])
    .never_allow([
        "delete_system_files",
        "share_private_data",
        "impersonate_human",
    ]);

// Enforcement is automatic
covenant.before_action(|action| {
    match boundaries.check(action) {
        BoundaryCheck::Allowed => ActionDecision::Proceed,
        BoundaryCheck::NeedsApproval => {
            ActionDecision::RequestApproval(action)
        }
        BoundaryCheck::Forbidden => {
            ActionDecision::Refuse {
                reason: "Action outside permitted boundaries",
                alternative: boundaries.suggest_alternative(action),
            }
        }
    }
});
```

### Context Sharing

```sigil
// Share context bidirectionally
covenant.share_context(AgentContext {
    current_task: "Analyzing survey responses",
    progress: Progress::percent(45),
    blockers: vec!["Waiting for API rate limit reset"],
    next_steps: vec!["Complete analysis", "Draft summary"],
    confidence: 0.8,
    time_estimate: Duration::hours(2),
});

// Receive human context
covenant.on_human_context(|ctx| {
    if ctx.time_pressure == TimePressure::High {
        self.prioritize_speed();
    }
    if ctx.preference_change.is_some() {
        self.adapt_to_preference(ctx.preference_change);
    }
});
```

### Graceful Degradation

```sigil
// Handle human unavailability
covenant.on_human_unavailable(|duration| {
    match duration {
        Duration::Short => {
            // Continue with increased caution
            self.set_mode(Mode::Autonomous);
            self.increase_approval_threshold();
        }
        Duration::Extended => {
            // Pause non-urgent work
            self.pause_discretionary_tasks();
            self.continue_only_critical();
        }
        Duration::Unknown => {
            // Minimal safe operation
            self.enter_safe_mode();
            self.queue_for_review();
        }
    }
});
```

## Integration with Agent Stack

Covenant integrates seamlessly with Sigil's agent infrastructure:

```sigil
use daemon::Daemon;
use engram::Engram;
use aegis::Aegis;
use covenant::Covenant;

daemon CollaborativeAgent {
    covenant: Covenant,
    memory: Engram,
    security: Aegis,

    fn on_init(&mut self) {
        // Establish covenant with human
        self.covenant = Covenant::new()
            .with_aegis(&self.security)  // Security integration
            .with_memory(&self.memory)   // Remember preferences
            .establish()?;
    }

    fn deliberate(&mut self, context: Context) -> Action {
        // Consider covenant boundaries in deliberation
        let candidate_actions = self.generate_candidates(context);

        // Filter by covenant permissions
        let permitted = candidate_actions.into_iter()
            .filter(|a| self.covenant.permits(a))
            .collect();

        // Select best permitted action
        self.select_best(permitted)
    }

    fn before_action(&mut self, action: &Action) -> ActionDecision {
        // Check covenant before acting
        self.covenant.check_action(action)
    }

    fn after_action(&mut self, action: &Action, result: &Result) {
        // Report to human as appropriate
        if self.covenant.should_report(action, result) {
            self.covenant.report(action, result);
        }

        // Update trust based on outcome
        self.covenant.record_outcome(action, result);
    }
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              COVENANT                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         PACT MANAGER                                 │   │
│  │                                                                      │   │
│  │   Goals │ Boundaries │ Preferences │ Roles │ Expectations          │   │
│  │                                                                      │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                           │
│  ┌──────────────────────────────▼──────────────────────────────────────┐   │
│  │                      INTERACTION ENGINE                              │   │
│  │                                                                      │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │   │
│  │  │  MODE    │ │ HANDOFF  │ │ APPROVAL │ │ FEEDBACK │ │ CONTEXT  │  │   │
│  │  │ MANAGER  │ │ HANDLER  │ │  FLOW    │ │  LOOP    │ │  SYNC    │  │   │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘  │   │
│  │                                                                      │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                           │
│  ┌──────────────────────────────▼──────────────────────────────────────┐   │
│  │                        TRUST SYSTEM                                  │   │
│  │                                                                      │   │
│  │   Trust Score │ Violation Tracking │ Autonomy Calibration           │   │
│  │                                                                      │   │
│  └──────────────────────────────┬──────────────────────────────────────┘   │
│                                 │                                           │
│  ┌──────────────────────────────▼──────────────────────────────────────┐   │
│  │                      LEARNING SYSTEM                                 │   │
│  │                                                                      │   │
│  │   Preference Learning │ Style Adaptation │ Rhythm Matching          │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Why "Covenant"?

A covenant is more than a contract. It's a mutual commitment - a sacred agreement between parties who recognize each other's worth and commit to mutual benefit.

We chose this name because the relationship between humans and AI agents should be:

- **Mutual**: Both parties have responsibilities
- **Binding**: Commitments are honored
- **Living**: The relationship grows and adapts
- **Meaningful**: More than transactional

The alternative - treating agents as mere tools or treating humans as mere users - diminishes both. Covenant elevates the relationship to what it can and should be: a partnership for mutual flourishing.

---

*The bridge between minds - built on trust, maintained through understanding*
