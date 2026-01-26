# Lessons Learned: Sigil + Qliphoth Dogfooding

Living document tracking friction points, wins, and improvement opportunities discovered while building sigil-lang.com with the Qliphoth framework.

## Session Log

### 2026-01-21: Initial Assessment

**Context:** Attempting to build full website using Qliphoth framework instead of raw WASM vdom calls.

#### Discovery: Syntax Confusion

**Issue:** The `lib.sigil`, `components.sigil`, and `pages.sigil` files were written in Rust-like syntax, not actual Sigil.

**Wrong (Rust-style):**
```rust
pub fn Component() -> Element! {
    let count! = use_state(0);
}
```

**Correct (Sigil):**
```sigil
☉ rite Component() → Element! {
    ≔ count! = use_state(0)
}
```

**Root Cause:**
- No syntax highlighting for `.sigil` files in common editors
- Documentation examples mix Sigil and Rust syntax
- Easy to slip into Rust habits when the languages are similar

**Friction Level:** 🔴 High - Code won't compile at all

**Suggested Fixes:**
1. [ ] Create VS Code extension for Sigil syntax highlighting
2. [ ] Add `sigil fmt` command that auto-fixes common Rust→Sigil mistakes
3. [ ] Add linter warnings for Rust-isms (`fn` instead of `rite`, `let` instead of `≔`)

---

## Friction Points

### 🔴 Critical (Blocks Progress)

| Issue | Description | Suggested Fix | Status |
|-------|-------------|---------------|--------|
| No syntax highlighting | Hard to spot syntax errors | VS Code extension | Open |
| Rust/Sigil confusion | Easy to write `fn` instead of `rite` | Linter + auto-fix | Open |
| Symbol entry | Typing ≔, →, ☉, ⊢ is awkward | Editor snippets/IME | Open |

### 🟡 Moderate (Slows Progress)

| Issue | Description | Suggested Fix | Status |
|-------|-------------|---------------|--------|
| Middot ambiguity | When to use `·` vs `.`? | Document clearly | Open |
| Evidentiality placement | `!i32` vs `i32!`? Both appear | Pick one, deprecate other | Open |
| Component syntax | `component X {}` vs `#[component] rite X()` | Document preferred style | Open |

### 🟢 Minor (Paper Cuts)

| Issue | Description | Suggested Fix | Status |
|-------|-------------|---------------|--------|
| Semicolon rules | When required vs optional | Consistent rule | Open |
| Import syntax | `use` vs `☉ use` | Document | Open |

---

## Easy Wins

Quick improvements that would significantly help adoption:

### Editor Support
- [ ] **Sigil VS Code extension** with syntax highlighting
- [ ] **Snippets** for common patterns (`rite`, `sigil`, `⊢`)
- [ ] **Symbol input** via keyboard shortcuts (e.g., `\rite` → `rite`, `\assign` → `≔`)

### Tooling
- [ ] **`sigil fmt`** - Auto-format Sigil code
- [ ] **`sigil check`** - Type check without compiling
- [ ] **`sigil fix`** - Auto-fix common mistakes (Rust→Sigil)

### Documentation
- [ ] **Examples repo** with working Qliphoth apps
- [ ] **Migration guide** from React/Rust to Sigil/Qliphoth
- [ ] **Component gallery** with copy-paste examples

---

## Wins

Things that work well and should be preserved:

| Feature | Why It's Good |
|---------|---------------|
| Evidentiality markers | Instantly visible data provenance |
| Morpheme operators | Dense, readable pipelines |
| WASM compilation | Fast, small output |
| `main.sigil` approach | Low-level control when needed |

---

## Questions to Resolve

1. **Which component syntax is canonical?**
   - `component Name { state x: T; rite render() {} }`
   - `#[component] rite Name() → Element! {}`
   - Both valid? When to use which?
   - **Partial Answer:** Qliphoth uses `☉ sigil` + `⊢` with `Component` trait

2. **Middot usage rules?**
   - `Http·get()` - yes
   - `array.len()` - uses dot
   - What's the rule?
   - **Partial Answer:** Middot used for most method chains in Qliphoth source

3. **Evidentiality marker position?**
   - `!i32` (prefix) appears in test files
   - `i32!` (suffix) appears in some docs
   - **Answer:** Prefix `!i32` is correct (see test files)

4. **Export marker (☉) rules?**
   - When exactly is ☉ required?
   - Module-level only? Or also inside modules?
   - **Answer:** Used for public items - structs, functions, traits, modules

5. **Boolean literals?**
   - **Answer:** `yea` and `nay` instead of `true`/`false`

6. **Mutable references?**
   - **Answer:** `&vary` instead of `&mut`, `vary this` for moving mutable self

7. **If syntax?**
   - **Answer:** Both `⎇` symbol and `if` keyword work

8. **Trait keyword?**
   - **Answer:** `aspect` instead of `trait`

---

## Resources Consulted

- `/home/crook/dev2/workspace/sigil/jormungandr/tests/` - Working test examples
- `/home/crook/dev2/workspace/qliphoth/src/lib.sigil` - Qliphoth source
- `/home/crook/dev2/workspace/sigil/website-qliphoth/src/main.sigil` - Working WASM site
- `/home/crook/dev2/workspace/sigil/CLAUDE.md` - Language overview

---

## Session Notes

### 2026-01-21

Starting fresh attempt at proper Sigil/Qliphoth implementation.

**Plan:**
1. Create cheatsheets from working examples ✅
2. Rewrite `lib.sigil` with correct syntax ✅
3. Rewrite `components.sigil` with correct syntax ✅
4. Build out `pages.sigil` with Learn page
5. Test compilation
6. Iterate based on errors

**Mindset:** This is dogfooding. Every frustration is a data point for improving the language and tooling.

**Observations during rewrite:**

1. **Builder pattern works well** - The `element()·class()·child()·to_vnode()` chain is clean
2. **`⊢ Trait for Type` syntax** - Implementing Component trait feels natural
3. **`vary this` for chained mutations** - Good pattern for builder methods
4. **`yea`/`nay` instead of `true`/`false`** - Initially confusing, but distinctive
5. **No semicolons needed** at end of blocks (expression-oriented)

**New friction points discovered:**
- Need to implement `props_to_map` and `from_props` for every component (boilerplate)
- `Option<!String>` vs `Option<String>?` syntax is unclear
- When exactly do you need `HashMap::new()` vs an import?

**Patterns that work well:**
- Helper functions like `footer_column()` reduce repetition
- Separating style definitions as local variables keeps render() readable
- Component structs can be empty `☉ sigil Nav { }` for stateless components

### 2026-01-21 (continued): Two-Architecture Reality

**Observation:** We now have two parallel implementations:

1. **main.sigil** (Working) - Raw VDOM calls, compiles to WASM, deployed in production
2. **lib.sigil + components.sigil + pages.sigil** (Qliphoth-style) - Proper Sigil syntax, NOT compiled yet

**E2E Test Status:**
- 11/11 active tests pass (Home, Navigation, Responsive, Error Handling)
- 23 tests skipped (Learn, Docs, Examples, Playground) - require Qliphoth compilation

**Path Forward:**
- Current site (main.sigil) is production-ready
- Qliphoth-style files are ready for when framework compilation is available
- Cheatsheets and lessons learned will accelerate future development

**Key Insight:** Dogfooding revealed the gap between having Qliphoth source code and having a working compilation pipeline. The framework exists but the Sigil→WASM compiler doesn't yet support all Qliphoth patterns.

---

*Format: Add new entries at top of relevant section. Date all entries.*
