# Sigil Native Symbols TDD Roadmap

**Branch:** `feature/sigil-native-symbols`
**Date:** 2026-01-17
**Goal:** Transform Sigil from Rust-with-different-words to a truly symbolic AI-native language

---

## Philosophy

Sigil is **AI-agent native**. Humans are the afterthought. We optimize for:
- Semantic density over typing comfort
- Symbolic clarity over ASCII compatibility
- Mathematical/logical notation where it belongs
- Polycultural symbols (Greek, Runic, Mathematical, Alchemical)

---

## Complete Symbol Vocabulary

### Type Definitions
| Concept | Symbol | Origin | Mnemonic |
|---------|--------|--------|----------|
| struct | `Σ` | Greek | **S**tructure, **S**igil |
| enum | `ᛈ` | Runic (Perthro) | Lot cup - choices/fate |
| trait | `Θ` | Greek | **Th**eory, aspect |
| impl | `⊢` | Logic | Turnstile - proves/provides |

### Functions
| Concept | Symbol | Origin | Mnemonic |
|---------|--------|--------|----------|
| fn | `λ` | Lambda calculus | Function |
| async | `⌛` | Unicode | Time/waiting |
| return | `⤺` | Arrow | Go back |

### Variables
| Concept | Symbol | Origin | Mnemonic |
|---------|--------|--------|----------|
| let | `≔` | Math | Definition |
| mut | `Δ` | Greek | Delta - change |
| const | `◆` | Geometric | Solid, fixed |

### Control Flow
| Concept | Symbol | Origin | Mnemonic |
|---------|--------|--------|----------|
| if | `⎇` | ISO | Branch |
| else | `⎉` | ISO | Alternative |
| match | `⌥` | Computing | Options |
| for | `∀` | Logic | For all |
| in | `∈` | Math | Element of |
| while | `⟳` | Arrow | Cycle |
| loop | `∞` | Math | Infinite |
| break | `⊗` | Math | Stop |
| continue | `↻` | Arrow | Continue |

### Visibility
| Concept | Symbol | Origin | Mnemonic |
|---------|--------|--------|----------|
| pub | `☉` | Alchemy | Sun - visible |

### References & Self
| Concept | Symbol | Origin | Mnemonic |
|---------|--------|--------|----------|
| self | `⊙` | Math | Self-reference |

### Logic
| Concept | Symbol | Origin | Mnemonic |
|---------|--------|--------|----------|
| true | `⊤` | Logic | Top |
| false | `⊥` | Logic | Bottom |
| && / and | `∧` | Logic | Conjunction |
| \|\| / or | `∨` | Logic | Disjunction |
| ! / not | `¬` | Logic | Negation |

### Paths & Other
| Concept | Symbol | Origin | Mnemonic |
|---------|--------|--------|----------|
| :: | `·` | Sigil | Middle-dot paths |
| where | `∋` | Math | Such that |
| as | `→` | Arrow | Conversion |

### Already Implemented (Aliases)
| Rust | Sigil | Status |
|------|-------|--------|
| use | invoke | ✅ Working |
| mod | scroll | ✅ Working |
| crate | tome | ✅ Working |

---

## TDD Implementation Phases

### Phase 1: Lexer - Token Aliases (Foundation)
Add all symbols as token aliases in `parser/src/lexer.rs`

**Tests to write first:**
```
tests/symbols/P0_001_sigma_struct.sg        - Σ parses as struct
tests/symbols/P0_002_perthro_enum.sg        - ᛈ parses as enum
tests/symbols/P0_003_theta_trait.sg         - Θ parses as trait
tests/symbols/P0_004_turnstile_impl.sg      - ⊢ parses as impl
tests/symbols/P0_005_lambda_fn.sg           - λ parses as fn
tests/symbols/P0_006_assign_let.sg          - ≔ parses as let
tests/symbols/P0_007_delta_mut.sg           - Δ parses as mut
tests/symbols/P0_008_diamond_const.sg       - ◆ parses as const
tests/symbols/P0_009_branch_if.sg           - ⎇ parses as if
tests/symbols/P0_010_alt_else.sg            - ⎉ parses as else
tests/symbols/P0_011_option_match.sg        - ⌥ parses as match
tests/symbols/P0_012_forall_for.sg          - ∀ parses as for
tests/symbols/P0_013_element_in.sg          - ∈ parses as in
tests/symbols/P0_014_cycle_while.sg         - ⟳ parses as while
tests/symbols/P0_015_infinity_loop.sg       - ∞ parses as loop
tests/symbols/P0_016_stop_break.sg          - ⊗ parses as break
tests/symbols/P0_017_continue_arrow.sg      - ↻ parses as continue
tests/symbols/P0_018_sun_pub.sg             - ☉ parses as pub
tests/symbols/P0_019_self_circle.sg         - ⊙ parses as self
tests/symbols/P0_020_top_true.sg            - ⊤ parses as true
tests/symbols/P0_021_bottom_false.sg        - ⊥ parses as false
tests/symbols/P0_022_and_logic.sg           - ∧ parses as &&
tests/symbols/P0_023_or_logic.sg            - ∨ parses as ||
tests/symbols/P0_024_not_logic.sg           - ¬ parses as !
tests/symbols/P0_025_middot_path.sg         - · parses as ::
tests/symbols/P0_026_return_arrow.sg        - ⤺ parses as return
tests/symbols/P0_027_such_that_where.sg     - ∋ parses as where
tests/symbols/P0_028_arrow_as.sg            - → parses as as
```

### Phase 2: Rune Annotations
Wire up `//@ rune:` processing in parser

**Tests:**
```
tests/runes/P0_001_rune_test.sg             - //@ rune: test works
tests/runes/P0_002_rune_derive.sg           - //@ rune: derive(...) works
tests/runes/P0_003_rune_cfg.sg              - //@ rune: cfg(...) works
```

### Phase 3: Integration Tests
Full programs using all native symbols

**Tests:**
```
tests/integration/P0_001_full_sigil_struct.sg
tests/integration/P0_002_full_sigil_enum.sg
tests/integration/P0_003_full_sigil_trait_impl.sg
tests/integration/P0_004_full_sigil_control_flow.sg
tests/integration/P0_005_full_sigil_logic.sg
```

### Phase 4: Library Rewrites (Post-Foundation)
After symbols work, rewrite libraries with native syntax:
- [ ] shared/
- [ ] aegis/
- [ ] engram/
- [ ] omen/
- [ ] gnosis/
- [ ] oracle/
- [ ] (others...)

---

## Implementation Order

1. **Create test directory**: `tests/symbols/`
2. **Write first failing test**: `P0_001_sigma_struct.sg`
3. **Add `Σ` to lexer**: Map to `Token::Struct`
4. **Verify test passes**
5. **Repeat for each symbol**
6. **Wire up `//@ rune:`**
7. **Integration tests**
8. **Rewrite Aegis tests with native syntax**
9. **Continue with other libraries**

---

## Files to Modify

| File | Changes |
|------|---------|
| `parser/src/lexer.rs` | Add Unicode symbol tokens |
| `parser/src/parser.rs` | Handle `//@ rune:` annotations |
| `parser/src/interpreter.rs` | Ensure symbols evaluate correctly |
| `parser/src/main.rs` | Test runner recognizes `//@ rune: test` |

---

## Success Criteria

- [ ] All 28 symbol tests pass
- [ ] All 3 rune tests pass
- [ ] All 5 integration tests pass
- [ ] Aegis TDD complete with native syntax
- [ ] No regressions in existing 233 P0 tests

---

## Notes

- Symbols are **aliases**, not replacements - both syntaxes work
- This enables gradual migration
- AI agents can use native syntax immediately
- Humans can still read/write Rust-style if needed

---

**Last Updated:** 2026-01-17
