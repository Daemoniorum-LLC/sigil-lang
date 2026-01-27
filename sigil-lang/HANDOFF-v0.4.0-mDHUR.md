# Handoff Document: Sigil v0.4.0 Prep (Session mDHUR)

**Date:** 2026-01-26
**Branch:** `feature/v0.4.0-structural-cleanup`
**Status:** In Progress - Qliphoth Component Work Ongoing

## Summary

This session focused on v0.4.0 parser improvements and structural cleanup:
1. Macro/rune enhancements (assert_eq!/assert_ne!, named args)
2. Structural cleanup (moved duplicates into sigil-lang/)
3. Qliphoth VNode rendering improvements

## Test Status

**Current:** 681/714 tests passing (95%)

### Completed Work

#### 1. Structural Cleanup
- Removed 35 duplicate directories from sigil/ root
- Synced v0.4.0 features (assert_eq/ne, WebSocket) from prep branch to sigil-lang/
- Moved benchmarks, tree-sitter-sigil, sigil-web-interface into sigil-lang/
- Hoisted website-qliphoth to sigil/ ecosystem level
- Added qliphoth as submodule under sigil/

#### 2. Macro Improvements (from original session)
- `assert_eq!` and `assert_ne!` macros
- Named argument support in macros
- Re-enabled `&&` and `||` token lexing
- New `detokenize()` function for macro body parsing

#### 3. Qliphoth VNode Improvements (this continuation)
- **Button component**: Fully working with all tests passing
  - Proper tag mapping (button -> "button")
  - Disabled/type/aria-* attributes
  - Loading state with spinner child
  - Icon support with btn-icon child
- **Generic component render**: Improved tag mapping
  - Dialog -> dialog, Header -> header, Footer -> footer, Nav -> nav
  - Textarea -> textarea, Icon -> svg, Hero -> section
- **Attribute population**: Component props now map to VNode attrs
  - Dialog: role, aria-modal, aria-labelledby, data-initial-focus
  - Link: href, target
  - Input/Textarea: placeholder, value, name, type
  - Avatar: src, alt
  - Common: id, aria-label, disabled

## Remaining Work

### Qliphoth Component Tests (31 failing)

These are "TDD: RED" tests - they define expected behavior for a full component library.
Each component needs:
1. Specific render handler (like Button has)
2. Child VNode generation for internal structure
3. Proper class/attribute propagation

**Priority components to fix:**
- Card (with CardHeader, CardBody, CardFooter)
- Dialog (with DialogHeader, DialogBody, DialogFooter)
- Input, Select, Checkbox, Radio, Switch
- Header, Footer, Nav (with child structure)

### i18n Tests (3 failing)
- P0_001_daemoniorum_locale
- P0_002_common_translations
- P0_003_landing_translations

These require i18n module implementation in stdlib.

## Files Changed (This Session)

- `parser/src/interpreter.rs` - Button component render, generic VNode improvements

## Verification

```bash
# Build compiler
cd sigil-lang/parser
CARGO_INCREMENTAL=0 cargo build --release

# Run tests
cd ../jormungandr/tests
./run_tests_rust.sh

# Test specific component
../../parser/target/release/sigil run spec/24_qliphoth_components/P0_001_button.sg
```

## Next Steps

1. **Fix Card component** - Similar pattern to Button
2. **Fix Dialog component** - Add child generation for DialogHeader/Body/Footer
3. **Consider component framework** - Current approach is repetitive; could abstract
4. **i18n module** - Implement translation loading in stdlib

## Context

Part of Sigil v0.4.0 preparation. The Qliphoth component tests define the target API
for a production component library. The Button implementation serves as a template
for other components.
