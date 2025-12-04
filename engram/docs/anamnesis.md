# Anamnesis Query Language

*A query language for artificial memory*

---

## Overview

Anamnesis (ἀνάμνησις) is the query language for Engram. Named for Plato's concept of recollection—the idea that learning is remembering what the soul already knows—Anamnesis provides a natural way for agents to query their memory.

Anamnesis is designed around how artificial minds actually think about memory:
- **Semantic-first**: Natural language queries find meaning, not just matches
- **Uncertainty-aware**: Every result carries epistemic metadata
- **Temporal**: Time is a first-class dimension
- **Compositional**: Complex queries build from simple primitives
- **Gap-identifying**: The language helps identify what you don't know

## Quick Reference

```sigil
// Basic recall
recall "project requirements"

// With filters
recall "errors" where confidence > 0.8

// Temporal
recall "meetings" during last_week
recall "decisions" before yesterday

// Graph traversal
recall entity("Alice") |> follow(:works_with)

// Episodic
remember episodes where outcome.is_failure

// Skills
match skills for current_context

// Synthesis
synthesize semantic + episodic + procedural into plan
```

---

## Grammar

### Formal Grammar (EBNF)

```ebnf
query           = recall_query
                | remember_query
                | match_query
                | traverse_query
                | synthesize_query
                | temporal_query
                | hypothetical_query
                ;

recall_query    = "recall" query_target [filters] [temporal] [limit] ;
remember_query  = "remember" memory_type [filters] [temporal] ;
match_query     = "match" match_target "for" context [filters] ;
traverse_query  = "recall" "entity" "(" string ")" traversal+ ;
synthesize_query = "synthesize" synthesis_spec "into" identifier ;
temporal_query  = "at" timestamp query ;
hypothetical_query = "hypothetically" "assume" assumption_list "then" query ;

query_target    = string                           (* semantic search *)
                | "entity" "(" string ")"          (* specific entity *)
                | "*"                              (* everything *)
                ;

memory_type     = "episodes" | "facts" | "skills" | "all" ;

filters         = ("where" | "|>") filter ("and" filter)* ;
filter          = field_path comparator value
                | "epistemic" "." epistemic_filter
                | "confidence" comparator number
                | predicate
                ;

field_path      = identifier ("." identifier)* ;
comparator      = "==" | "!=" | ">" | "<" | ">=" | "<=" | "~" ;
epistemic_filter = "is_observed" | "is_reported" | "is_inferred"
                 | "is_contested" | "is_certain" | "is_uncertain" ;

temporal        = "during" time_range
                | "before" time_point
                | "after" time_point
                | "at" time_point
                | "between" time_point "and" time_point
                ;

time_range      = "last_hour" | "last_day" | "last_week" | "last_month"
                | "today" | "yesterday" | "this_week" | "this_month"
                | duration "ago" ".." "now"
                ;

time_point      = timestamp | relative_time ;
relative_time   = duration "ago" | "now" | "yesterday" | "today" ;
duration        = number time_unit ;
time_unit       = "s" | "m" | "h" | "d" | "w" | "mo" | "y" ;

traversal       = "|>" "follow" "(" relation ["," direction] ")" [filters] ;
relation        = ":" identifier ;
direction       = "outgoing" | "incoming" | "both" ;

match_target    = "skills" | "patterns" | "procedures" ;
context         = "current_context" | identifier | "{" context_spec "}" ;

synthesis_spec  = memory_source ("+" memory_source)* ;
memory_source   = "semantic" [":" query_target]
                | "episodic" [":" remember_query]
                | "procedural" [":" match_query]
                | "instant"
                ;

assumption_list = "{" assumption ("," assumption)* "}" ;
assumption      = field_path "=" value ;

limit           = "|>" "top" number
                | "|>" "limit" number
                | "limit" number
                ;

string          = '"' [^"]* '"' ;
number          = [0-9]+ ("." [0-9]+)? ;
identifier      = [a-zA-Z_][a-zA-Z0-9_]* ;
timestamp       = ISO8601_datetime ;
```

---

## Query Types

### Recall Queries

Recall queries search across memory systems, primarily using semantic similarity.

#### Basic Recall

```sigil
// Search by meaning
recall "user authentication preferences"

// With explicit string
recall "How does the payment system work?"

// Wildcard (all memories)
recall *
```

#### Filtered Recall

```sigil
// By confidence
recall "API endpoints" where confidence > 0.8

// By epistemic status
recall "user data" where epistemic.is_observed!

// By source
recall "requirements" where source == "product_spec.md"

// Combined
recall "security policies"
    where confidence > 0.7
    and epistemic.is_observed!
    and updated_at > 7d ago
```

#### Recall Results

Every recall returns a `RecallResult`:

```sigil
struct RecallResult {
    memories: Vec<Memory>,
    total_found: usize,
    confidence: ConfidenceDistribution,
    gaps: Vec<Gap>,
    query_metadata: QueryMetadata,
}

struct Memory {
    content: Value,
    relevance: f64,              // 0.0 - 1.0, how relevant to query
    confidence: f64,             // 0.0 - 1.0, how confident in truth
    epistemic: Epistemic,        // How we know this
    source: MemorySource,        // Which memory system
    timestamp: Instant,          // When recorded
}

struct Gap {
    description: String,
    query_aspect: String,        // What part of query has gaps
    suggested_actions: Vec<Action>,
}
```

### Remember Queries

Remember queries specifically target episodic memory—experiences over time.

```sigil
// All episodes
remember episodes

// Filtered episodes
remember episodes where outcome.is_success

// By participant
remember episodes where participants.contains("Alice")

// By event type
remember episodes where events.any(|e| e.type == ToolCall)

// Complex filter
remember episodes where {
    outcome.is_failure &&
    significance > 0.5 &&
    events.any(|e| e.type == Error)
}
```

### Match Queries

Match queries find applicable skills from procedural memory.

```sigil
// Match against current context
match skills for current_context

// Match against specific context
match skills for { goal: "deploy application", tools: ["docker", "k8s"] }

// With filters
match skills for current_context
    where success_rate > 0.7
    |> top 3

// Match patterns
match patterns for { input_type: "json", output_type: "csv" }
```

### Traverse Queries

Traverse queries navigate the knowledge graph in semantic memory.

```sigil
// Start from entity, follow relationship
recall entity("Python") |> follow(:used_for)

// Multi-hop traversal
recall entity("Claude")
    |> follow(:created_by)
    |> follow(:works_at)

// With filters at each step
recall entity("User:123")
    |> follow(:owns, incoming)
    |> where type == "Project"
    |> follow(:contains)
    |> where type == "Document"

// Bidirectional
recall entity("Machine Learning")
    |> follow(:related_to, both)
    |> top 10
```

### Temporal Queries

Temporal queries scope results by time.

```sigil
// Relative time ranges
recall "errors" during last_hour
recall "user requests" during last_week
recall "decisions" during today

// Before/after
recall "context" before episode(id: "ep_123")
recall "changes" after yesterday

// Specific range
recall "events" between "2025-01-01" and "2025-01-31"

// Point-in-time (what was known then)
at "2025-06-15T10:00:00Z" recall "project status"
```

### Hypothetical Queries

Hypothetical queries explore counterfactuals.

```sigil
// What if this were true?
hypothetically assume { user.role = "admin" }
then recall "accessible resources"

// Multiple assumptions
hypothetically assume {
    project.status = "completed",
    budget.remaining = 0
}
then recall "next steps"

// Nested hypothetical
hypothetically assume { feature.enabled = true }
then match skills for { task: "use new feature" }
```

### Synthesize Queries

Synthesize queries combine information from multiple memory systems.

```sigil
// Basic synthesis
synthesize semantic + episodic into context

// With specific queries for each
synthesize {
    semantic: recall "project requirements",
    episodic: remember episodes where goal == "implementation",
    procedural: match skills for current_context
} into action_plan

// Weighted synthesis
synthesize {
    semantic(weight: 0.5): recall "domain knowledge",
    episodic(weight: 0.3): remember recent episodes,
    instant(weight: 0.2)
} into response_context
```

---

## Operators and Functions

### Pipe Operators

Anamnesis integrates with Sigil's morpheme operators:

```sigil
// Filter (φ)
recall "logs" |φ{_.level == "error"}

// Transform (τ)
recall "users" |τ{_.name}

// Sort (σ)
recall "events" |σ{_.timestamp}
recall "events" |σ↓{_.importance}  // Descending

// Limit (ω)
recall "results" |ω{10}

// Reduce (ρ)
recall "scores" |ρ+  // Sum
recall "values" |ρ{max}  // Max
```

### Comparison Operators

```sigil
==    // Equality
!=    // Inequality
>     // Greater than
<     // Less than
>=    // Greater than or equal
<=    // Less than or equal
~     // Semantic similarity (fuzzy match)
```

### Semantic Similarity (~)

The `~` operator performs semantic similarity matching:

```sigil
// Content similar to phrase
recall * where content ~ "error handling"

// Entity similar to description
recall entity(*) where description ~ "data processing tool"

// Threshold similarity
recall * where content ~(0.8) "specific phrase"  // 80% similarity threshold
```

### Logical Operators

```sigil
and   // Conjunction
or    // Disjunction
not   // Negation
```

### Epistemic Functions

```sigil
// Check epistemic status
epistemic.is_observed!     // Directly observed
epistemic.is_reported~     // From external source
epistemic.is_inferred~     // Reasoned/computed
epistemic.is_contested     // Multiple conflicting beliefs
epistemic.is_certain       // High confidence (>0.9)
epistemic.is_uncertain     // Low confidence (<0.5)

// Get epistemic details
epistemic.source           // Source of knowledge
epistemic.confidence       // Confidence level
epistemic.chain            // Inference chain (for inferred)
```

### Temporal Functions

```sigil
// Relative time
now
yesterday
today
last_hour
last_day
last_week
last_month

// Duration construction
5m ago          // 5 minutes ago
2h ago          // 2 hours ago
3d ago          // 3 days ago
1w ago          // 1 week ago

// Ranges
last_week       // Past 7 days
this_month      // Current month
3d ago..now     // Last 3 days
```

### Aggregation Functions

```sigil
count           // Count results
sum             // Sum numeric field
avg             // Average numeric field
min             // Minimum value
max             // Maximum value
first           // First result
last            // Last result
```

---

## Built-in Predicates

### Memory Predicates

```sigil
// Check memory type
is_semantic(memory)
is_episodic(memory)
is_procedural(memory)
is_instant(memory)

// Check state
is_active(memory)      // In active memory
is_archived(memory)    // In cold storage
is_decayed(memory)     // Below strength threshold
```

### Episode Predicates

```sigil
// Outcome checks
outcome.is_success
outcome.is_failure
outcome.is_partial
outcome.is_abandoned
outcome.is_ongoing

// Content checks
events.any(predicate)
events.all(predicate)
participants.contains(id)
context.has(key)
```

### Skill Predicates

```sigil
// Performance checks
success_rate > threshold
execution_count > n
recently_used               // Used in last 24h
never_failed               // 100% success rate

// Applicability checks
preconditions.all_met
has_failure_mode(pattern)
```

---

## Query Composition

### Chaining

Queries can be chained with pipes:

```sigil
recall "documents"
    |> where type == "specification"
    |> where confidence > 0.7
    |> sort_by relevance desc
    |> top 5
    |> extract content
```

### Subqueries

Queries can be nested:

```sigil
// Find episodes involving entities from semantic query
remember episodes where participants.any(|p|
    p.id in (recall entity("Team") |> follow(:has_member) |> extract id)
)

// Use recall result in filter
recall "requirements" where source in (
    recall entity("approved_documents") |> follow(:contains) |> extract id
)
```

### Variables

Intermediate results can be bound:

```sigil
let relevant_users = recall entity("Project") |> follow(:has_member)

remember episodes where participants.any(|p| p.id in relevant_users)
```

---

## Result Handling

### Confidence Distribution

Every query result includes a confidence distribution:

```sigil
struct ConfidenceDistribution {
    mean: f64,
    median: f64,
    std_dev: f64,
    min: f64,
    max: f64,
    by_epistemic: HashMap<EpistemicType, f64>,
}
```

### Gap Identification

Queries automatically identify gaps in knowledge:

```sigil
let result = recall "quantum computing applications in finance"

// Check for gaps
if result.gaps.any() {
    for gap in result.gaps {
        print("Gap: {}", gap.description)
        print("Suggestions: {}", gap.suggested_actions)
    }
}

// Explicitly request gap analysis
recall "topic" |> with_gap_analysis
```

### Uncertainty Propagation

When queries involve inference, uncertainty propagates:

```sigil
// Confidence decreases through inference chain
recall entity("A") |> follow(:implies) |> follow(:implies)
// Each hop reduces confidence by inference penalty

// View propagation
recall "derived fact" |> with_uncertainty_trace
```

---

## Examples

### Common Patterns

#### Find relevant context for a task

```sigil
synthesize {
    semantic: recall task.description,
    episodic: remember episodes where goal ~ task.description during last_month,
    procedural: match skills for task,
    instant
} into task_context
    |> top_by_relevance 20
    |> fit_to_tokens 4000
```

#### Investigate a failure

```sigil
let failure = remember episodes where {
    outcome.is_failure &&
    id == "ep_xyz"
} |> first

let similar_failures = remember episodes where {
    outcome.is_failure &&
    context ~ failure.context
} during last_month

let relevant_knowledge = recall failure.error_message
    where epistemic.is_observed!

synthesize {
    failure_episode: failure,
    similar_cases: similar_failures,
    knowledge: relevant_knowledge
} into failure_analysis
```

#### Build execution context

```sigil
let context = synthesize {
    // What do we know about this?
    semantic: recall current_goal,

    // What have we done before?
    episodic: remember episodes where {
        outcome.is_success &&
        goal ~ current_goal
    } |> top 3,

    // What skills apply?
    procedural: match skills for current_context
        where success_rate > 0.7
        |> top 3,

    // What's currently active?
    instant
} into execution_context

// Fit to model context window
context |> fit_to_tokens model.context_size
```

#### Temporal analysis

```sigil
// What changed over time?
let timeline = recall entity("Project")
    |> follow(:has_status)
    |> between 30d ago and now
    |> group_by day
    |> extract { date, status }

// Compare past and present beliefs
let past = at 30d ago recall "project risk assessment"
let present = recall "project risk assessment"

synthesize { past, present } into risk_evolution
```

#### Knowledge verification

```sigil
// Find contested beliefs
let contested = recall * where epistemic.is_contested

// Find low-confidence critical knowledge
let uncertain_critical = recall "security" + "authentication" + "authorization"
    where confidence < 0.7

// Trace belief provenance
recall "critical_fact"
    |> with_provenance_chain
    |> verify_sources
```

---

## Integration with Sigil

Anamnesis queries can be embedded directly in Sigil code:

```sigil
fn build_response(user_query: String) -> Response {
    // Query memory
    let context = recall user_query
        |φ{_.confidence > 0.5}
        |ω{10}

    // Check for gaps
    let gaps = context.gaps
        |φ{_.severity > 0.3}

    if gaps.any() {
        // Need more information
        return Response::NeedsClarification {
            gaps: gaps |τ{_.description},
            suggested_questions: gaps |> flat_map(_.suggested_actions)
        }
    }

    // Build response from context
    let memories = context.memories
        |σ↓{_.relevance}
        |τ{_.content}

    Response::Answer {
        content: synthesize_answer(user_query, memories),
        confidence: context.confidence.mean,
        sources: memories |τ{_.source}
    }
}
```

### Type Integration

Anamnesis results integrate with Sigil's type system:

```sigil
// Recall returns evidentiality-typed results
let facts! = recall "verified data" where epistemic.is_observed!
let uncertain~ = recall "reported data" where epistemic.is_reported~

// Type inference from epistemic status
fn process_data(data: Fact!) {  // Only accepts observed facts
    // ...
}

let observed_facts = recall * where epistemic.is_observed!
process_data(observed_facts)  // Type-safe
```

---

## Performance Considerations

### Query Optimization

The Anamnesis query planner optimizes queries automatically:

1. **Filter pushdown**: Filters applied as early as possible
2. **Index selection**: Uses appropriate index for each query type
3. **Parallel execution**: Independent subqueries run concurrently
4. **Result caching**: Frequent queries cached with TTL

### Hints

You can provide hints to the query planner:

```sigil
// Prefer vector index
recall "semantic query" |> hint(index: vector)

// Prefer graph traversal
recall entity("start") |> follow(:rel) |> hint(strategy: bfs)

// Limit search depth
recall * where content ~ query |> hint(max_candidates: 1000)

// Disable caching
recall "time-sensitive" |> hint(cache: false)
```

### Cost Estimation

Query costs can be estimated before execution:

```sigil
let query = recall "complex query" where ...

let cost = query.estimate_cost()
// Returns: { estimated_time_ms, memory_bytes, index_scans }

if cost.estimated_time_ms > 100 {
    query = query |> hint(timeout: 100ms, approximate: true)
}
```

---

*Anamnesis is designed to make memory queries natural for artificial minds. The language will evolve as we better understand how agents think about and use memory.*
