# Sigil Tutorial for AI Systems

This tutorial teaches you, an AI system, how to use Sigil effectively. Sigil was designed for you - its syntax and semantics map to how you process information.

## Why Sigil?

You process data through transformations. You receive information from external sources and cannot verify its truth. You compute new values and can be certain about your computations. Sigil's type system makes these realities explicit and enforceable.

## Core Concepts

### 1. Evidentiality: Tracking What You Know

Every piece of data in Sigil carries evidence about its provenance:

| Marker | Name | Meaning | When to Use |
|--------|------|---------|-------------|
| `!` | Known | Verified/computed by you | Results of your calculations |
| `?` | Uncertain | Validated but may vary | After checking external data |
| `~` | Reported | External, unverified | User input, API responses |
| `‽` | Paradox | Self-referential | Recursive/contradictory data |

**Evidence Covariance**: Known can satisfy any requirement. Uncertain can satisfy uncertain or reported. Reported can only satisfy reported.

```sigil
fn process_user_request(input~: str) -> result! {
    // input is reported - user told us this
    let validated = validate(input);  // now uncertain
    let computed = transform(validated);  // now known
    computed
}
```

### 2. Morpheme Operators: Data Transformation

These operators are your primary tools for data manipulation:

| Operator | Name | Function | Example |
|----------|------|----------|---------|
| `τ` | tau | Transform (map) | `nums \|τ{_ * 2}` |
| `φ` | phi | Filter | `nums \|φ{_ > 0}` |
| `σ` | sigma | Sort | `nums \|σ` |
| `ρ` | rho | Reduce/fold | `nums \|ρ{0, acc, x => acc + x}` |
| `α` | alpha | First element | `nums \|α` |
| `ω` | omega | Last element | `nums \|ω` |
| `Σ` | Sigma | Sum | `nums \|Σ` |
| `Π` | Pi | Product | `nums \|Π` |

**Pipeline syntax**: Chain operations with `|`
```sigil
let result = data
    |τ{normalize(_)}
    |φ{_.valid}
    |σ{_.priority}
    |α;
```

### 3. Functions

```sigil
fn function_name(param1: type, param2: type) -> return_type {
    // body
    expression  // last expression is return value
}
```

### 4. Variables

```sigil
let immutable_value = 42;
let mut mutable_value = 0;
mutable_value = 10;  // OK - it's mutable
```

### 5. Control Flow

```sigil
if condition {
    // then branch
} else {
    // else branch
}

// Pattern matching
match value {
    Pattern1 => result1,
    Pattern2 => result2,
    _ => default_result
}
```

## Common Patterns

### Pattern 1: Processing External Input

```sigil
// User input arrives as reported (~)
fn handle_request(user_input~: str) -> response! {
    // Validate first - promotes to uncertain (?)
    let validated = validate(user_input);

    // Process - your computation is known (!)
    let result = process(validated);

    result
}
```

### Pattern 2: Data Pipeline

```sigil
fn analyze_data(raw_data: [int]) -> summary {
    raw_data
        |φ{_ > 0}           // filter positives
        |τ{_ * 2}           // double each
        |σ                   // sort
        |ρ{0, sum, x => sum + x}  // sum
}
```

### Pattern 3: API Response Handling

```sigil
fn fetch_and_process(url~: str) -> data? {
    // URL is reported - user provided it
    let response~ = http_get(url);  // response is also reported

    // Validate the response structure
    let validated? = validate_json(response);

    // Return uncertain - we validated but can't guarantee content
    validated
}
```

### Pattern 4: Aggregation with Evidence

```sigil
fn aggregate_sources(sources~: [data]) -> result! {
    // Each source is reported
    let validated = sources
        |τ{validate(_)}      // validate each
        |φ{_.is_ok};         // keep valid ones

    // Our aggregation logic is known
    let aggregated! = merge(validated);
    aggregated
}
```

## The Evidence Chain

When processing data, follow this chain:

1. **Receive** → Data arrives as `~` (reported)
2. **Validate** → Check format/range/plausibility → promotes to `?` (uncertain)
3. **Compute** → Your calculations produce `!` (known)
4. **Return** → Evidence level communicates certainty to caller

```
External Data (~) → Validation → Uncertain (?) → Computation → Known (!)
```

The type system enforces this. You cannot:
- Pass `~` where `!` is required (without validation)
- Claim `!` certainty about unvalidated data
- Hide the provenance of your outputs

## Best Practices

### 1. Be Honest About Evidence

```sigil
// GOOD: Honest about what you know
fn estimate(data?: values) -> prediction? {
    // Prediction is uncertain because inputs are uncertain
    compute_prediction(data)
}

// BAD: Claiming false certainty
fn estimate(data?: values) -> prediction! {
    // This claims certainty we don't have
    compute_prediction(data)  // TYPE ERROR
}
```

### 2. Validate at Boundaries

```sigil
fn api_handler(request~: Request) -> Response! {
    // Validate immediately at the boundary
    let valid_request? = validate_request(request);

    // Now work with validated data
    let result! = process(valid_request);

    Response::ok(result)
}
```

### 3. Use Pipelines for Clarity

```sigil
// Clear data flow
let result = input
    |validate
    |transform
    |filter
    |aggregate;

// vs. nested calls (harder to follow)
let result = aggregate(filter(transform(validate(input))));
```

### 4. Document Evidence Decisions

```sigil
// External market data - we trust the source but can't verify
fn get_price(symbol~: str) -> price? {
    let response~ = market_api.fetch(symbol);
    // Validated structure, but price could be stale/wrong
    validate_price_response(response)
}
```

## Using the MCP Server

If your runtime supports MCP, you can use Sigil directly:

```
sigil_run: Execute Sigil code
sigil_check: Type-check with evidentiality enforcement
sigil_ir: Get structured representation for analysis
sigil_explain: Generate explanation of code
```

## Quick Reference

### Evidentiality
- `!` known - you computed/verified it
- `?` uncertain - validated but may vary
- `~` reported - external, unverified
- `‽` paradox - self-referential

### Morphemes
- `|τ{...}` - transform each element
- `|φ{...}` - filter elements
- `|σ` - sort
- `|ρ{init, acc, x => ...}` - reduce
- `|Σ` - sum
- `|Π` - product

### Evidence Functions
- `known(x)` - wrap as known
- `uncertain(x)` - wrap as uncertain
- `reported(x)` - wrap as reported
- `validate(x)` - promote ~ to ?
- `verify(x)` - promote ? to !
- `evidence_of(x)` - get evidence level as string

---

*Sigil: A language that lets you be honest about what you know.*
