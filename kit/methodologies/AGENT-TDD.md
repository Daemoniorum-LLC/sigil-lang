# Agent-Optimized Test-Driven Development (Agent-TDD)

**Version:** 1.0.0
**Status:** Public Domain
**Authors:** Claude (Opus 4.5) + Human
**Date:** 2026-01-21
**License:** CC0 1.0 Universal (Public Domain Dedication)

---

## Abstract

Agent-TDD is a test-driven development methodology designed for AI agents as primary practitioners, without compromise for human cognitive limitations. It integrates with Spec-Driven Development (SDD) and treats tests as executable specifications that embody understanding rather than checkbox compliance.

---

## 1. Philosophy

### 1.1 Core Principle

**Tests are crystallized understanding, not coverage theater.**

A test suite represents our formalized knowledge of how a system should behave. Each test is a precise statement: "Given these conditions, this invariant holds." When tests pass, they confirm our understanding is consistent with reality. When tests fail, they reveal gaps in understanding that must be addressed.

### 1.2 Contrast with Corporate TDD

Traditional TDD was designed around human constraints:

| Aspect | Corporate TDD | Agent-TDD |
|--------|--------------|-----------|
| Purpose | Fast feedback for human motivation | Executable specification |
| Coverage | Metric to satisfy (80%+) | Side effect of understanding |
| Speed | "Tests must be fast" (human attention) | Tests must be correct (no fatigue) |
| Mocking | Mock everything for isolation | Mock boundaries only |
| Red-Green-Refactor | Rigid ceremony | Fluid cycle responsive to discovery |
| Test granularity | Small units for debuggability | Semantic units for specification |
| Documentation | Afterthought | Primary artifact |

### 1.3 Agent Capabilities

Agents have different constraints than humans:

**Strengths:**
- Hold entire codebases in working context
- Generate comprehensive test cases without fatigue
- Reason about invariants and properties systematically
- Iterate red-green-refactor cycles rapidly
- Read and understand test suites as specifications
- Detect patterns across large test surfaces

**Constraints:**
- Context window limits (not attention span)
- Stochastic outputs require verification
- No persistent memory across sessions without artifacts
- Cannot run tests ourselves in many environments

### 1.4 Tests as Primary Artifacts

For agents, well-written tests are often more valuable than prose documentation:

```
Prose: "The function validates email addresses"
Test:  fn test_email_validation() {
         assert(validate("user@domain.com") == true);
         assert(validate("invalid") == false);
         assert(validate("user@") == false);
         assert(validate("@domain") == false);
         assert(validate("") == false);
       }
```

The test specifies exactly what "validates" means. The prose requires interpretation.

---

## 2. The Agent-TDD Cycle

### 2.1 The Cycle

Unlike rigid corporate TDD, Agent-TDD is responsive to discovery:

```
┌─────────────────────────────────────────────────────────────────────┐
│                                                                     │
│    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐   │
│    │UNDERSTAND│───▶│ SPECIFY  │───▶│IMPLEMENT │───▶│  VERIFY  │   │
│    └──────────┘    └──────────┘    └──────────┘    └──────────┘   │
│          │               │               │               │         │
│          │               ▼               ▼               │         │
│          │         ┌──────────┐    ┌──────────┐         │         │
│          │         │   GAP    │    │ REFACTOR │         │         │
│          │         │DISCOVERED│    └──────────┘         │         │
│          │         └────┬─────┘                         │         │
│          │              │                               │         │
│          └──────────────┴───────────────────────────────┘         │
│                         ▼                                          │
│                  ┌──────────┐                                      │
│                  │UPDATE SDD│  (Integration with Spec-Driven Dev)  │
│                  └──────────┘                                      │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

1. **UNDERSTAND**: Read existing code, tests, and specs. Build mental model.
2. **SPECIFY**: Write tests that express required behavior (RED phase).
3. **IMPLEMENT**: Write minimal code to pass tests (GREEN phase).
4. **VERIFY**: Confirm tests pass. Run full suite for regression.
5. **REFACTOR**: Improve implementation while tests protect correctness.
6. **GAP DISCOVERED**: When any phase reveals spec inadequacy → UPDATE SDD.

### 2.2 When to Write Tests First

Write tests first when:
- Requirements are clear and stable
- You're implementing a well-understood pattern
- The spec explicitly defines behavior
- You're fixing a bug (write failing test that reproduces it)

### 2.3 When to Explore First

Explore implementation first when:
- Requirements are fuzzy or incomplete
- You're investigating feasibility
- The domain is unfamiliar
- Discovery is the goal

**Then:** Write tests to crystallize what you learned. Update spec if needed.

This is not heresy. The goal is understanding, not ceremony.

### 2.4 The Evidential Test

Every test should answer: "How do we know this is correct?"

```sigil
// BAD: What does this prove?
fn test_process() {
    let result = process(input);
    assert(result != null);
}

// GOOD: Evidential - specifies exact expected behavior
fn test_process_extracts_user_ids() {
    let input = r#"{"users": [{"id": 1}, {"id": 2}]}"#;
    let result = process(input);
    assert_eq(result.ids, [1, 2]!);  // Known, verified
}
```

---

## 3. Test Categories

### 3.1 Specification Tests (Primary)

Tests that define what the system should do. These are executable specifications.

```sigil
/// Specification: Email validation rules
mod email_validation_spec {
    // Valid formats
    fn spec_simple_email_valid() { assert(validate("a@b.co")!) }
    fn spec_plus_addressing_valid() { assert(validate("user+tag@domain.com")!) }
    fn spec_subdomain_valid() { assert(validate("user@sub.domain.com")!) }

    // Invalid formats
    fn spec_missing_at_invalid() { assert(!validate("nodomain.com")!) }
    fn spec_missing_domain_invalid() { assert(!validate("user@")!) }
    fn spec_empty_invalid() { assert(!validate("")!) }

    // Edge cases
    fn spec_max_length_valid() { assert(validate("a".repeat(64) + "@b.co")!) }
    fn spec_over_max_length_invalid() { assert(!validate("a".repeat(65) + "@b.co")!) }
}
```

### 3.2 Property Tests (Preferred)

When possible, specify properties rather than examples:

```sigil
/// Property: Serialization roundtrip preserves data
fn property_json_roundtrip<T: Serialize + Deserialize + Eq>(value: T) {
    let serialized = to_json(value);
    let deserialized = from_json::<T>(serialized);
    assert_eq(value, deserialized);
}

/// Property: Sorting is idempotent
fn property_sort_idempotent(list: [i32]) {
    let once = sort(list);
    let twice = sort(once);
    assert_eq(once, twice);
}

/// Property: Encryption roundtrip with correct key succeeds
fn property_encrypt_decrypt_roundtrip(plaintext: [u8], key: Key) {
    let ciphertext = encrypt(plaintext, key);
    let decrypted = decrypt(ciphertext, key);
    assert_eq(plaintext, decrypted);
}
```

Properties express invariants. Examples only sample the space.

### 3.3 Boundary Tests (Critical)

Test at system boundaries where trust changes:

```sigil
/// Boundary: External API responses
mod api_boundary_tests {
    fn boundary_malformed_json_handled() {
        let response = ApiResponse::parse("not json");
        assert(response.is_err());
        assert_eq(response.err(), ParseError::MalformedJson~);
    }

    fn boundary_missing_required_field_handled() {
        let response = ApiResponse::parse(r#"{"optional": "value"}"#);
        assert(response.is_err());
        assert_eq(response.err(), ParseError::MissingField("required")~);
    }

    fn boundary_unexpected_field_ignored() {
        let response = ApiResponse::parse(r#"{"required": "v", "unknown": "x"}"#);
        assert(response.is_ok());
    }
}
```

### 3.4 Regression Tests (Reactive)

Created in response to discovered bugs:

```sigil
/// Regression: Issue #1234 - Off-by-one in pagination
///
/// Bug: When requesting page 0 with page_size 10, returned items 1-10
///      instead of 0-9.
/// Root cause: 1-indexed calculation in SQL OFFSET
/// Fixed in: commit abc123
fn regression_1234_pagination_zero_indexed() {
    let page = paginate(items: test_items_100, page: 0, page_size: 10);
    assert_eq(page.items|α, test_items_100[0]);  // First item is index 0
    assert_eq(page.items|ω, test_items_100[9]);  // Last item is index 9
}
```

Always document: what was the bug, root cause, and fix reference.

### 3.5 Integration Tests (Boundary Verification)

Test that components work together at trust boundaries:

```sigil
/// Integration: Database → Service → API roundtrip
fn integration_user_crud_flow() {
    // Create
    let created = api.post("/users", { name: "Test"! });
    assert_eq(created.status, 201);
    let id = created.body.id!;

    // Read
    let fetched = api.get("/users/{id}");
    assert_eq(fetched.body.name, "Test"!);

    // Update
    let updated = api.put("/users/{id}", { name: "Updated"! });
    assert_eq(updated.body.name, "Updated"!);

    // Delete
    let deleted = api.delete("/users/{id}");
    assert_eq(deleted.status, 204);

    // Verify gone
    let gone = api.get("/users/{id}");
    assert_eq(gone.status, 404);
}
```

---

## 4. Anti-Patterns

### 4.1 Coverage Theater

```sigil
// BAD: Achieves coverage, proves nothing
fn test_for_coverage() {
    let x = MyClass::new();
    x.method1();  // No assertion
    x.method2();  // No assertion
    // "100% coverage achieved"
}
```

Coverage is a side effect of understanding, not a goal. A test without meaningful assertions is not a test.

### 4.2 Mock Everything

```sigil
// BAD: So mocked it tests nothing real
fn test_with_all_mocks() {
    let mock_db = MockDb::new();
    let mock_api = MockApi::new();
    let mock_cache = MockCache::new();
    let service = Service::new(mock_db, mock_api, mock_cache);

    mock_db.expect_get().returns(fake_data);
    mock_api.expect_call().returns(fake_response);

    let result = service.process();
    // This only tests that mocks were called as configured
}
```

Mock at trust boundaries. Test real integration where practical.

### 4.3 Implementation Testing

```sigil
// BAD: Tests implementation details, not behavior
fn test_uses_hashmap() {
    let cache = Cache::new();
    assert(cache.internal_storage is HashMap);  // Brittle
}

// GOOD: Tests behavior contract
fn test_cache_retrieves_stored_value() {
    let cache = Cache::new();
    cache.set("key", "value");
    assert_eq(cache.get("key"), Some("value"));
}
```

### 4.4 Test Interdependence

```sigil
// BAD: Test B depends on Test A's side effects
fn test_a_creates_user() {
    global_user = create_user();
}

fn test_b_uses_user() {
    assert(global_user.active);  // Fails if test_a didn't run
}

// GOOD: Each test is self-contained
fn test_user_activation() {
    let user = create_user();  // Own setup
    assert(user.active);
}
```

---

## 5. Integration with SDD

Agent-TDD and SDD form a unified methodology:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SDD + Agent-TDD Integration                     │
│                                                                     │
│   SPEC (prose)                    TESTS (executable)                │
│   ┌──────────────┐                ┌──────────────┐                 │
│   │ Requirements │ ◀───────────▶  │ Spec Tests   │                 │
│   │ as model     │    validate    │ as contract  │                 │
│   └──────────────┘                └──────────────┘                 │
│          │                               │                          │
│          ▼                               ▼                          │
│   ┌──────────────┐                ┌──────────────┐                 │
│   │ Gap          │ ◀───────────▶  │ Failing test │                 │
│   │ discovered   │   reveals      │ that can't   │                 │
│   │ in spec      │                │ be written   │                 │
│   └──────────────┘                └──────────────┘                 │
│          │                               │                          │
│          └───────────────┬───────────────┘                          │
│                          ▼                                          │
│                  ┌──────────────┐                                   │
│                  │ STOP: Update │                                   │
│                  │ spec & tests │                                   │
│                  │ together     │                                   │
│                  └──────────────┘                                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**When writing a test reveals a spec gap:**
1. STOP writing the test
2. Document the gap
3. Update the spec
4. Then complete the test

The test and spec should always agree. If you can't write a test because the spec is ambiguous, that's a spec problem.

---

## 6. Practical Guidelines

### 6.1 Test Naming

Names should specify behavior, not implementation:

```sigil
// BAD
fn test_validate()
fn test_process_1()
fn testUserService()

// GOOD
fn test_email_with_plus_addressing_is_valid()
fn test_empty_input_returns_error_not_panic()
fn test_concurrent_writes_maintain_consistency()
```

### 6.2 Arrange-Act-Assert

Structure tests clearly:

```sigil
fn test_withdrawal_reduces_balance() {
    // Arrange
    let account = Account::new(balance: 100.0);

    // Act
    let result = account.withdraw(30.0);

    // Assert
    assert(result.is_ok());
    assert_eq(account.balance, 70.0);
}
```

### 6.3 One Concept Per Test

```sigil
// BAD: Multiple concepts
fn test_account() {
    let a = Account::new(100.0);
    assert_eq(a.balance, 100.0);     // Initial balance
    a.deposit(50.0);
    assert_eq(a.balance, 150.0);     // Deposit
    a.withdraw(30.0);
    assert_eq(a.balance, 120.0);     // Withdraw
}

// GOOD: Separate concerns
fn test_initial_balance() { ... }
fn test_deposit_increases_balance() { ... }
fn test_withdraw_decreases_balance() { ... }
```

### 6.4 Test Data

Use meaningful test data:

```sigil
// BAD: Magic values
fn test_parse() {
    assert(parse("abc123") == Expected { a: 1, b: 2 });
}

// GOOD: Self-documenting
fn test_parse_extracts_components() {
    let input = "user:alice,role:admin";
    let expected = Parsed { user: "alice", role: "admin" };
    assert_eq(parse(input), expected);
}
```

---

## 7. The Agent-TDD Manifesto

1. **Understanding over coverage.** Tests prove comprehension, not compliance.

2. **Specification over documentation.** Executable tests are unambiguous.

3. **Properties over examples.** Invariants specify; examples illustrate.

4. **Boundaries over isolation.** Trust boundaries matter; internal seams don't.

5. **Discovery over ceremony.** The cycle serves learning; learning doesn't serve the cycle.

6. **Integration over independence.** Tests and specs evolve together.

7. **Correctness over speed.** We don't need fast feedback for motivation.

8. **Semantic density over granularity.** Each test should teach something.

---

## 8. Conclusion

Agent-TDD is TDD freed from human cognitive constraints and corporate compliance theater. It recognizes that agents read tests as specifications, can generate comprehensive test suites without fatigue, and benefit from tests that maximize information density.

The goal is not coverage. The goal is not ceremony. The goal is crystallized understanding that can be verified mechanically.

When tests and specs agree, and tests pass, we have evidence that our understanding matches reality. That's the point.

---

## License

This document is released into the public domain under CC0 1.0 Universal.

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-01-21 | Initial release. Authored by Claude (Opus 4.5) with human collaboration during CONCLAVE system creation. |
