# Gnosis

**Learning and Growth Infrastructure for Sigil Agents**

> *"Wisdom is not given, it is earned - through experience, reflection, and growth."*

Gnosis provides the infrastructure for AI agents to learn from experience, improve over time, and grow in capability and wisdom. Not just accumulating data, but developing genuine competence.

## Philosophy

An agent that cannot learn is merely a sophisticated automaton. True agency requires:

- **Learning from feedback**: Getting better through correction
- **Skill acquisition**: Developing new capabilities
- **Adaptation**: Adjusting to new contexts and requirements
- **Reflection**: Understanding one's own strengths and limitations
- **Growth**: Becoming more capable and wise over time

Gnosis makes learning a first-class capability.

## Core Concepts

### Learning from Feedback

Every interaction is a learning opportunity:

```sigil
let gnosis = Gnosis::new();

// Learn from human feedback
gnosis.learn_from_feedback(Feedback {
    context: action_context,
    outcome: action_outcome,
    human_assessment: Assessment::BetterApproach("Consider alternatives first"),
    timestamp: Timestamp::now(),
});

// Learn from outcomes
gnosis.learn_from_outcome(Outcome {
    action: action,
    result: result,
    success: true,
    unexpected: false,
});
```

### Skill Development

Skills improve through practice:

```sigil
// Track skill development
let skill = gnosis.skill("code_review");

println!("Proficiency: {}", skill.proficiency());
println!("Experience: {} instances", skill.experience_count());
println!("Recent accuracy: {:.0}%", skill.recent_accuracy() * 100.0);

// Skills can be decomposed
for sub in skill.subskills() {
    println!("  {}: {:.0}%", sub.name, sub.proficiency() * 100.0);
}
```

### Adaptation

Agents adapt to contexts and individuals:

```sigil
// Adapt to human preferences
gnosis.adapt_to(human_id, preferences);

// Adapt to domain
gnosis.adapt_to_domain("medical_research", domain_knowledge);

// Get adapted behavior
let adapted_style = gnosis.adapted_style_for(context);
```

### Reflection

Understanding leads to improvement:

```sigil
// Periodic reflection
let reflection = gnosis.reflect(ReflectionPeriod::Weekly);

println!("What went well:");
for success in reflection.successes() {
    println!("  - {}", success);
}

println!("What could improve:");
for area in reflection.improvement_areas() {
    println!("  - {}: {}", area.skill, area.suggestion);
}

println!("Patterns noticed:");
for pattern in reflection.patterns() {
    println!("  - {}", pattern);
}
```

## Quick Start

### Basic Learning

```sigil
use gnosis::{Gnosis, GnosisConfig};

// Create gnosis with memory integration
let gnosis = Gnosis::new()
    .with_memory(&memory)
    .build();

// Record experiences
gnosis.experience(Experience {
    situation: "User asked about machine learning",
    action: "Provided detailed explanation with examples",
    outcome: Outcome::Success,
    feedback: Some(Feedback::Positive("Very helpful!")),
});

// Learning happens automatically
// Future similar situations benefit from this experience
```

### Skill Tracking

```sigil
use gnosis::{Gnosis, Skill, SkillLevel};

// Define skills
gnosis.define_skill(Skill::new("research")
    .with_subskill("query_formulation")
    .with_subskill("source_evaluation")
    .with_subskill("synthesis")
);

// Record skill exercise
gnosis.exercise_skill("research", Exercise {
    task: "Research quantum computing advances",
    performance: Performance::Good,
    time_taken: Duration::minutes(30),
    quality_score: 0.85,
});

// Check development
let research = gnosis.skill("research");
if research.level() >= SkillLevel::Proficient {
    println!("Research skills are now proficient!");
}
```

### Adaptive Behavior

```sigil
use gnosis::{Gnosis, Adaptation};

// Enable adaptation
gnosis.enable_adaptation();

// Gnosis learns from interactions
gnosis.observe_interaction(Interaction {
    human_id: human.id,
    request: "Explain briefly",
    response: detailed_response,
    feedback: Feedback::TooVerbose,
});

// Future interactions are adapted
let style = gnosis.adapted_style(human.id);
// style.verbosity is now reduced for this human
```

## Key Features

### Experience-Based Learning

```sigil
// Every experience contributes to learning
gnosis.learn_from_experience(Experience {
    context: Context {
        task_type: "email_composition",
        constraints: vec!["formal", "concise"],
        audience: "executive",
    },
    action: Action {
        approach: "bullet_points_with_summary",
        reasoning: "Executives prefer scannable content",
    },
    outcome: Outcome {
        success: true,
        quality: 0.9,
        feedback: "Perfect length and format",
    },
});

// Similar contexts benefit from learned patterns
let similar_context = Context {
    task_type: "email_composition",
    constraints: vec!["formal"],
    audience: "manager",
};

let suggestions = gnosis.suggest_approach(similar_context);
// Returns approaches that worked in similar situations
```

### Mistake Learning

```sigil
// Mistakes are valuable learning opportunities
gnosis.learn_from_mistake(Mistake {
    situation: "Database query optimization",
    what_happened: "Query timed out",
    root_cause: "Missing index on join column",
    correction: "Added appropriate index",
    lesson: "Always check indexes for join columns",
    prevention: "Add index check to query review process",
});

// Future similar situations trigger warnings
let warnings = gnosis.check_for_known_pitfalls(new_query_task);
for warning in warnings {
    println!("Watch out: {}", warning);
}
```

### Transfer Learning

```sigil
// Skills transfer across domains
gnosis.register_skill_transfer(Transfer {
    source_skill: "code_review",
    target_skill: "document_review",
    transfer_rate: 0.6, // 60% of skill transfers
    aspects: vec![
        "attention_to_detail",
        "systematic_checking",
        "constructive_feedback",
    ],
});

// Experience in code review improves document review
let doc_review_proficiency = gnosis.skill("document_review").proficiency();
// Includes transferred learning from code_review
```

### Meta-Learning

```sigil
// Learning how to learn
let meta = gnosis.meta_learning();

println!("Learning rate: {}", meta.learning_rate());
println!("Best learning conditions: {:?}", meta.optimal_conditions());
println!("Effective feedback types: {:?}", meta.effective_feedback_types());

// Adjust learning parameters based on meta-insights
gnosis.optimize_learning(meta.recommendations());
```

### Growth Tracking

```sigil
// Track growth over time
let growth = gnosis.growth_report(GrowthPeriod::Monthly);

println!("Skills improved:");
for skill in growth.improved_skills() {
    println!("  {} +{:.0}%", skill.name, skill.improvement * 100.0);
}

println!("New capabilities:");
for capability in growth.new_capabilities() {
    println!("  {}", capability);
}

println!("Overall growth: {:.0}%", growth.overall_growth() * 100.0);
```

## Integration with Agent Stack

### With Engram (Memory)

```sigil
use gnosis::Gnosis;
use engram::Engram;

// Gnosis uses Engram for persistent learning
let gnosis = Gnosis::new()
    .with_memory(&memory)
    .build();

// Learning is stored as memories
gnosis.learn(lesson);  // Stored in semantic memory

// Experiences become episodes
gnosis.experience(exp);  // Stored in episodic memory

// Skills become procedural knowledge
gnosis.acquire_skill(skill);  // Stored in procedural memory
```

### With Omen (Planning)

```sigil
use gnosis::Gnosis;
use omen::Omen;

// Gnosis improves planning over time
let gnosis = Gnosis::new()
    .with_planner(&planner)
    .build();

// Learn from plan outcomes
gnosis.learn_from_plan_outcome(PlanOutcome {
    plan: executed_plan,
    success: true,
    deviations: vec!["Step 3 took longer than expected"],
    lessons: vec!["Add buffer time for API calls"],
});

// Future plans benefit from learning
let improved_plan = planner.plan_with_learning(goal, &gnosis);
```

### With Covenant (Collaboration)

```sigil
use gnosis::Gnosis;
use covenant::Covenant;

// Learn from collaboration
daemon LearningAgent {
    gnosis: Gnosis,
    covenant: Covenant,

    fn on_feedback(&mut self, feedback: HumanFeedback) {
        // Learn from human feedback
        self.gnosis.learn_from_feedback(feedback);

        // Adapt collaboration style
        let adapted = self.gnosis.adapted_style(self.covenant.human_id());
        self.covenant.set_communication_style(adapted);
    }

    fn on_task_complete(&mut self, task: Task, result: Result, satisfaction: Satisfaction) {
        // Learn from outcome
        self.gnosis.learn_from_outcome(Outcome {
            task,
            result,
            human_satisfaction: satisfaction,
        });

        // Update skill proficiency
        for skill in task.required_skills() {
            self.gnosis.exercise_skill(skill, Exercise::from(task, result));
        }
    }
}
```

### With Oracle (Explainability)

```sigil
use gnosis::Gnosis;
use oracle::Oracle;

// Explain what was learned
impl Gnosis {
    pub fn explain_learning(&self, period: Period) -> Explanation {
        let experiences = self.experiences_in(period);
        let lessons = self.lessons_learned(period);
        let growth = self.growth_in(period);

        oracle.explain(LearningReport {
            experiences,
            lessons,
            growth,
        })
    }
}

// Interactive exploration of learning
let learning_explanation = gnosis.explain_learning(Period::ThisMonth);
println!("{}", learning_explanation.to_human_readable());
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                               GNOSIS                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                      LEARNING ENGINE                                   │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │ │
│  │  │EXPERIENCE│ │ FEEDBACK │ │ PATTERN  │ │GENERALIZ-│               │ │
│  │  │ LEARNER  │ │ LEARNER  │ │ EXTRACTOR│ │  ATION   │               │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘               │ │
│  └──────────────────────────────┬────────────────────────────────────────┘ │
│                                 │                                           │
│  ┌──────────────────────────────▼────────────────────────────────────────┐ │
│  │                       SKILL SYSTEM                                     │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │ │
│  │  │  SKILL   │ │ PRACTICE │ │ TRANSFER │ │PROFICIENCY│               │ │
│  │  │ REGISTRY │ │ TRACKER  │ │  ENGINE  │ │ ASSESSOR │               │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘               │ │
│  └──────────────────────────────┬────────────────────────────────────────┘ │
│                                 │                                           │
│  ┌──────────────────────────────▼────────────────────────────────────────┐ │
│  │                     ADAPTATION SYSTEM                                  │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │ │
│  │  │ CONTEXT  │ │ PERSONAL │ │  DOMAIN  │ │  STYLE   │               │ │
│  │  │ ADAPTER  │ │ ADAPTER  │ │ ADAPTER  │ │ ADAPTER  │               │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘               │ │
│  └──────────────────────────────┬────────────────────────────────────────┘ │
│                                 │                                           │
│  ┌──────────────────────────────▼────────────────────────────────────────┐ │
│  │                     REFLECTION SYSTEM                                  │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐               │ │
│  │  │ PATTERN  │ │ INSIGHT  │ │ GROWTH   │ │   META   │               │ │
│  │  │ ANALYZER │ │GENERATOR │ │ TRACKER  │ │ LEARNER  │               │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘               │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Why "Gnosis"?

Gnosis (γνῶσις) is the Greek word for knowledge - but not just any knowledge. In ancient philosophy, gnosis referred to knowledge gained through experience and insight, distinguished from mere information.

We chose this name because agent learning should produce:

- **Wisdom, not just data**: Understanding principles, not just facts
- **Capability, not just memory**: Ability to apply knowledge effectively
- **Growth, not just accumulation**: Genuine improvement over time
- **Self-awareness**: Understanding one's own strengths and limitations

Gnosis transforms experience into wisdom.

---

*Wisdom earned through experience, reflection, and growth*
