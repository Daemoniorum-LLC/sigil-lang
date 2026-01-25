status: exploratory
priority: high
category: daemoniorum-internal
created: 2025-12-06
author: claude
sparked_by: "Write agent logic in Sigil, not Java/YAML"
scope: internal
```
#### DESCRIPTION
Define and implement agent behavior in Sigil rather than Java configuration or YAML. Leverage Sigil's evidentiality types for agent beliefs, morpheme operators for reasoning chains, and trust semantics for multi-agent collaboration.

#### RATIONALE
- **Competitive moat** - Only Daemoniorum agents speak Sigil
- **Expressiveness** - Evidentiality on beliefs (!certain, ?possible, ~inferred)
- **Composability** - Morpheme operators (τ, φ, σ, ρ) on thought streams
- **Type safety** - Sigil's type system catches agent logic errors
- **Self-improvement** - Agents written in Sigil can be improved by Jormungandr

<!-- DISCUSSION_ENTRY -->
**AGENT DEFINITION IN SIGIL:**

```sigil
agent CodeReviewer {
    // Beliefs with evidentiality
    beliefs: {
        code_quality: ?Assessment,      // Uncertain until analyzed
        security_issues: ~[Issue],      // Inferred from patterns
        style_violations: ![Violation], // Certain after linting
    }

    // Tools available
    tools: [
        !read_file,      // Definitely has this
        !run_linter,
        ?run_tests,      // Might have access
        ~security_scan,  // Inferred from context
    ]

    // Reasoning pipeline using morpheme operators
    fn review(pr: PullRequest) -> ReviewResult {
        let files = pr.changed_files
            |tau{ read_file(.) }           // Transform: read each
            |phi{ .language in supported } // Filter: only supported
            |rho{ lint(.) }                // Reduce: aggregate issues

        // Evidentiality propagates through pipeline
        // If read_file returns ~content, downstream is ~

        let assessment = files
            |sigma{ .severity }            // Sort by severity
            |tau{ explain_issue(.) }       // Add explanations

        ReviewResult {
            approved: assessment.critical.is_empty()!,
            comments: assessment,
            confidence: assessment.evidentiality.min(),
        }
    }

    // Guardrails as type constraints
    invariant {
        // Can never approve if security issues found
        !(self.beliefs.security_issues.any() && result.approved)
    }
}
```

**BELIEF UPDATING:**

```sigil
// Agent updates beliefs based on evidence
fn update_belief(evidence: Evidence) {
    match evidence.source {
        !Linter => self.beliefs.style_violations = !evidence.issues,
        ?Tests  => self.beliefs.code_quality = ?evidence.assessment,
        ~Heuristic => self.beliefs.security_issues = ~evidence.issues,
    }
}
```

**MULTI-AGENT TRUST:**

```sigil
// Trust semantics for agent collaboration
fn request_review(other: Agent, code: Code) -> Review {
    let review = other.review(code)

    // Trust based on other agent's track record
    match other.trust_level {
        !Verified => review!,           // Accept as certain
        ?Known    => review?,           // Accept as possible
        ~Unknown  => review~ + self.verify(review),  // Verify
    }
}
```

**INTEGRATION:**

- Sigil compiler generates Java bytecode for Persona Framework runtime
- Sigil agent definitions compile to Spring beans
- Interop with existing Java tools via FFI
- Gradual migration: Java agents can call Sigil agents and vice versa

**OPEN QUESTIONS:**

1. Sigil compiler maturity for this use case?
2. Debugging Sigil agents - tooling needed?
3. Performance of Sigil vs native Java?
4. How to handle Sigil runtime errors in production?
<!-- /DISCUSSION_ENTRY -->
<!-- /IDEA_START -->
