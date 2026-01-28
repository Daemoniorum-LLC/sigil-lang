# Qliphoth Interpreter Support Specification

**Version:** 0.1.0
**Status:** In Progress
**Date:** 2026-01-26
**Methodology:** SDD + Agent-TDD

---

## 1. Overview

This spec defines what the Sigil interpreter must support to enable Qliphoth
component rendering. The tests in `jormungandr/tests/spec/24_qliphoth_components/`
serve as the executable specification (Agent-TDD RED phase).

---

## 2. Core VNode Structure

### 2.1 VNode Fields

```sigil
struct VNode {
    tag: String!           // HTML element tag
    text_content: String!  // Text content
    classes: Vec<String>!  // CSS classes
    attrs: Map<String, Value>!  // HTML attributes
    children: Vec<VNode>!  // Child nodes
}
```

### 2.2 VNode Methods Required

| Method | Signature | Status |
|--------|-----------|--------|
| `tag()` | `fn() -> String` | ✅ Implemented |
| `text_content()` | `fn() -> String` | ✅ Implemented |
| `has_class(name)` | `fn(String) -> bool` | ✅ Implemented |
| `attr(name)` | `fn(String) -> Value` | ✅ Implemented |
| `children()` | `fn() -> Vec<VNode>` | ✅ Implemented |
| `has_child_with_class(name)` | `fn(String) -> bool` | ✅ Implemented |
| `find_child_by_class(name)` | `fn(String) -> VNode` | ✅ Implemented |
| `find_child_by_tag(name)` | `fn(String) -> VNode` | ✅ Implemented |

---

## 3. Component Render Requirements

### 3.1 General Pattern

Each Qliphoth component must implement `render() -> VNode` that:
1. Creates a VNode with correct semantic HTML tag
2. Populates attrs from component props
3. Generates CSS classes (base + variant + size + state)
4. Creates child VNodes for internal structure

### 3.2 Component Specifications

#### 3.2.1 Button (✅ COMPLETE)

**Tag:** `button`

**Props → Attrs:**
| Prop | Attr | Default |
|------|------|---------|
| `disabled` | `disabled` | `false` |
| `button_type` | `type` | `"button"` |
| `aria_label` | `aria-label` | - |
| `loading` | `aria-busy` | - |

**Classes:** `btn-{variant}`, `btn-{size}`, `btn-disabled` (if disabled)

**Children:**
- If `loading: true` → child with class `btn-spinner`
- If `icon` set → child with class `btn-icon`

#### 3.2.2 Card (🔴 NOT IMPLEMENTED)

**Tag:** `div`

**Classes:** `card`, `card-{variant}` (if variant set)

**Children must process:**
- `CardHeader` → `div.card-header` containing:
  - `div.card-title` with title text
  - `div.card-subtitle` with subtitle text (if present)
- `CardBody` → `div.card-body` with children
- `CardFooter` → `div.card-footer` with children

#### 3.2.3 Input (🔴 NOT IMPLEMENTED)

**Tag:** `input`

**Props → Attrs:**
| Prop | Attr |
|------|------|
| `placeholder` | `placeholder` |
| `value` | `value` |
| `name` | `name` |
| `input_type` | `type` |
| `disabled` | `disabled` |
| `readonly` | `readonly` |

**Classes:** `input`, `input-{size}`, `input-{variant}`

#### 3.2.4 Dialog (🔴 NOT IMPLEMENTED)

**Tag:** `dialog`

**Props → Attrs:**
| Prop | Attr | Notes |
|------|------|-------|
| `alert` | `role` | "alertdialog" if true, else "dialog" |
| - | `aria-modal` | Always "true" |
| `aria_describedby` | `aria-describedby` | - |
| `initial_focus` | `data-initial-focus` | - |
| `close_on_escape` | `data-close-on-escape` | "true"/"false" |

**Classes:** `dialog`, `dialog-{size}`, `dialog-{position}`

**Children must include:**
- `div.dialog-backdrop` (if not hidden)
- `button.dialog-close` with `aria-label="Close"` (if not `hide_close`)
- Process DialogHeader, DialogBody, DialogFooter

#### 3.2.5 Select (🔴 NOT IMPLEMENTED)

**Tag:** `select`

**Props → Attrs:**
| Prop | Attr |
|------|------|
| `name` | `name` |
| `disabled` | `disabled` |

**Classes:** `select`, `select-{size}`

#### 3.2.6 Checkbox (🔴 NOT IMPLEMENTED)

**Tag:** `input`

**Props → Attrs:**
| Prop | Attr |
|------|------|
| `name` | `name` |
| `checked` | `checked` |
| `disabled` | `disabled` |
| - | `type` | Always "checkbox" |

**Classes:** `checkbox`

#### 3.2.7 Radio (🔴 NOT IMPLEMENTED)

**Tag:** `input`

**Props → Attrs:**
| Prop | Attr |
|------|------|
| `name` | `name` |
| `checked` | `checked` |
| `disabled` | `disabled` |
| - | `type` | Always "radio" |

**Classes:** `radio`

#### 3.2.8 Switch (🔴 NOT IMPLEMENTED)

**Tag:** `input`

**Props → Attrs:**
| Prop | Attr |
|------|------|
| `name` | `name` |
| `checked` | `checked` |
| `disabled` | `disabled` |
| - | `type` | "checkbox" |

**Classes:** `switch`

#### 3.2.9 Textarea (🔴 NOT IMPLEMENTED)

**Tag:** `textarea`

**Props → Attrs:**
| Prop | Attr |
|------|------|
| `placeholder` | `placeholder` |
| `value` | `value` |
| `name` | `name` |
| `disabled` | `disabled` |
| `readonly` | `readonly` |

**Classes:** `textarea`, `textarea-{size}`

#### 3.2.10 Link (🔴 NOT IMPLEMENTED)

**Tag:** `a`

**Props → Attrs:**
| Prop | Attr |
|------|------|
| `href` | `href` |
| `target` | `target` |

**Classes:** `link`

#### 3.2.11 Icon (🔴 NOT IMPLEMENTED)

**Tag:** `svg`

**Classes:** `icon`, `icon-{size}`

#### 3.2.12 Avatar (🔴 NOT IMPLEMENTED)

**Tag:** `img`

**Props → Attrs:**
| Prop | Attr |
|------|------|
| `src` | `src` |
| `alt` | `alt` |

**Classes:** `avatar`, `avatar-{size}`

#### 3.2.13 Tooltip (🔴 NOT IMPLEMENTED)

**Tag:** `div`

**Props → Attrs:**
| Prop | Attr |
|------|------|
| `trigger` | `trigger` |

**Classes:** `tooltip`

#### 3.2.14 Tabs (🔴 NOT IMPLEMENTED)

**Tag:** `div`

**Props → Attrs:**
| Prop | Attr |
|------|------|
| `manual` | `data-manual` |

**Classes:** `tabs`

**Children must process:**
- TabList → with Tab children
- TabPanels → with TabPanel children

#### 3.2.15 Header (🔴 NOT IMPLEMENTED)

**Tag:** `header`

**Props → Attrs:**
| Prop | Attr |
|------|------|
| `aria_label` | `aria-label` |

**Classes:** `header`, `header-{variant}`, `header-{size}`, `header-sticky` (if sticky)

**Children must include:**
- `a.header-logo` with `href` from `logo_href`, containing `img` from `logo`
- `a.header-brand` with `href` from `brand_href`, text from `brand`
- `div.header-nav` wrapping Nav children
- `div.header-actions` wrapping action buttons
- `button.header-mobile-toggle` with proper aria attrs (if `mobile_menu`)

#### 3.2.16 Footer (🔴 NOT IMPLEMENTED)

**Tag:** `footer`

**Classes:** `footer`, `footer-{variant}`

#### 3.2.17 Hero (🔴 NOT IMPLEMENTED)

**Tag:** `section`

**Classes:** `hero`, `hero-{variant}`, `hero-{size}`

**Text content:** from `title` prop

#### 3.2.18 Nav (🔴 NOT IMPLEMENTED)

**Tag:** `nav`

**Props → Attrs:**
| Prop | Attr |
|------|------|
| `aria_label` | `aria-label` |

**Classes:** `nav`

---

## 4. Implementation Strategy

### 4.1 Approach

For each component:
1. Read the test file to understand exact expectations
2. Implement dedicated handler in interpreter.rs
3. Run test to verify (GREEN)
4. Move to next component

### 4.2 Code Location

All component render handlers are in:
`parser/src/interpreter.rs` around line 10090

### 4.3 Pattern

```rust
if name == "ComponentName" {
    if method.name == "render" {
        let borrowed = fields.borrow();
        let mut vnode_fields = HashMap::new();
        let mut attrs = HashMap::new();

        // 1. Set tag
        vnode_fields.insert("tag", Value::String(Rc::new("tag".to_string())));

        // 2. Extract props → attrs
        if let Some(v) = borrowed.get("prop") { attrs.insert("attr", v.clone()); }

        // 3. Generate classes
        let classes = vec![...];
        vnode_fields.insert("classes", Value::Array(...));

        // 4. Generate children
        let children = vec![...];
        vnode_fields.insert("children", Value::Array(...));

        // 5. Set attrs
        vnode_fields.insert("attrs", Value::Map(...));

        return Ok(Value::Struct { name: "VNode", fields: ... });
    }
}
```

---

## 5. Test Status

**Overall: 687/706 tests passing (97%)**

*Note: 8 Daemoniorum-specific tests moved to archived/ (see ADR-001)*

| Test File | Component | Tests | Status | Notes |
|-----------|-----------|-------|--------|-------|
| P0_001_button.sg | Button | 8 | ✅ PASS | |
| P0_002_card.sg | Card | ~10 | ✅ PASS | |
| P0_003_input.sg | Input | ~8 | 🟡 PARTIAL | Needs `find_sibling_by_tag` |
| P0_004_badge.sg | Badge | ~5 | ✅ PASS | |
| P0_005_spinner.sg | Spinner | ~4 | ✅ PASS | |
| P0_006_dialog.sg | Dialog | ~13 | ✅ PASS | |
| P0_007_select.sg | Select | ~6 | 🟡 PARTIAL | Needs `find_sibling_by_tag` |
| P0_008_checkbox.sg | Checkbox | ~6 | 🔴 FAIL | |
| P0_009_radio.sg | Radio | ~6 | 🔴 FAIL | |
| P0_010_switch.sg | Switch | ~5 | 🔴 FAIL | |
| P0_011_textarea.sg | Textarea | ~6 | 🔴 FAIL | |
| P0_012_link.sg | Link | ~6 | 🔴 FAIL | |
| P0_013_icon.sg | Icon | ~5 | 🔴 FAIL | |
| P0_014_avatar.sg | Avatar | ~6 | 🔴 FAIL | |
| P0_015_tooltip.sg | Tooltip | ~6 | 🔴 FAIL | |
| P0_016_tabs.sg | Tabs | ~10 | 🔴 FAIL | |
| P0_017_header.sg | Header | ~14 | ✅ PASS | |
| P0_018_footer.sg | Footer | ~8 | 🟡 PARTIAL | Needs `find_children_by_class` |
| P0_019_hero.sg | Hero | ~6 | ✅ PASS | |
| P0_020_nav.sg | Nav | ~8 | 🟡 PARTIAL | Needs `find_children_by_class` |

---

## 6. Architectural Decisions

### ADR-001: Application Components vs Interpreter Primitives

**Date:** 2026-01-26

**Context:** The test suite originally included `spec/25_daemoniorum_components/` with
brand-specific components (PlatformCard, ResearchCard, LandingPage, etc.). These tests
required the interpreter to have special handling for application-level components.

**Decision:** Application-specific components should NOT be in the interpreter.

**Rationale:**
1. The interpreter provides **primitives** (Button, Card, Input, Dialog, etc.)
2. Applications compose primitives into **application components**
3. Application components are just Sigil structs with `render()` methods
4. No special interpreter support is needed - they use the generic VNode system

**Consequences:**
- Moved `spec/25_daemoniorum_components/` to `workspace/daemoniorum/daemoniorum-app/tests/components/`
- Interpreter test suite only covers Qliphoth primitives
- Generic reusable patterns (PricingGrid, PersonGrid) can be added to Qliphoth if needed

---

## 7. Gap Log

### GAP-001: Complex Child Generation

**Discovered:** 2026-01-26
**Component:** Header, Dialog, Card, Tabs

**Issue:** Components like Header need to generate complex internal VNode
structures based on props (logo, brand, nav, actions). This requires the
render handler to:
1. Inspect props
2. Generate child VNodes with proper tags, classes, and attrs
3. Nest children correctly

**Resolution:** Implement per-component child generation logic in render handlers.

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-01-26 | Initial spec. Button complete. |
| 0.2.0 | 2026-01-26 | Added Card, Badge, Spinner, Dialog, Header, Hero, Footer, Nav, Select, Input. 687/706 tests (97%). |
| 0.2.1 | 2026-01-26 | ADR-001: Moved Daemoniorum components to archived/. Interpreter only covers primitives. |
