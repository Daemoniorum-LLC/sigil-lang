# Website v0.4.0 Update Specification

**Version:** 0.1.0
**Status:** Draft
**Author:** Claude (Conclave Agent)
**Date:** 2026-01-25
**Methodology:** Spec-Driven Development (SDD)

---

## 1. Overview

### 1.1 Purpose

Update the Sigil website (website-qliphoth) to reflect the v0.4.0 release, establish Qliphoth as the canonical website, deprecate the static HTML version, and ensure all deployment artifacts are current.

### 1.2 Scope

| Component | Action |
|-----------|--------|
| Version references | Update 0.3.0 → 0.4.0 |
| JSON-LD schema | Fix incorrect version (1.0.0 → 0.4.0) |
| Changelog page | Add new page to navigation |
| CLAUDE.md | Document Qliphoth canonical status |
| WASM runtime | Verify latest runtime deployed |
| HTML website | Mark as deprecated |

### 1.3 Out of Scope

- WASM compilation pipeline changes
- New feature additions to website
- Design/styling updates

---

## 2. Current State Analysis

### 2.1 Version Inconsistencies

| Location | File | Current | Target |
|----------|------|---------|--------|
| Version badge | `main.sigil:108` | `0.3` | `0.4` |
| Announcement banner | `main.sigil:890` | `Sigil 0.3.0` | `Sigil 0.4.0` |
| JSON-LD schema | `index.html:34` | `1.0.0` | `0.4.0` |
| Announcement | `index.sigil` | `Sigil 0.3.0 Released!` | `Sigil 0.4.0` |
| Release link | `components.sigil` | `v1.0.0` | `v0.4.0` |

### 2.2 Website Architecture

```
website-qliphoth/           # Canonical (WASM-powered)
├── src/                    # Sigil source files
│   ├── main.sigil          # Homepage + all pages (1956 lines)
│   ├── index.sigil         # Index page variant
│   ├── components.sigil    # Reusable components
│   └── helpers.sigil       # Helper functions
├── index.html              # HTML shell with loader
├── deploy/                 # Production build output
└── build.sh                # WASM compilation script

website/                    # Deprecated (static HTML)
├── index.html              # Static HTML fallback
└── ...
```

### 2.3 WASM Runtime Status

| Component | Location | Status |
|-----------|----------|--------|
| Runtime JS | `website-qliphoth/sigil_runtime.js` | Symlink → `/home/crook/dev2/workspace/qliphoth/runtime/sigil_runtime.js` |
| Deploy runtime | `website-qliphoth/deploy/sigil_runtime.js` | 48KB - needs verification |
| WASM binaries | `website-qliphoth/public/wasm/` | Not present - needs build |

### 2.4 Navigation Structure

Current pages in `main.sigil`:
- Home (main)
- Learn
- Docs
- Examples
- Playground
- Agents
- Pattern

**Missing:** Changelog/Releases page

---

## 3. Prerequisites

### 3.1 Compiler Requirements

| Requirement | Status |
|-------------|--------|
| Sigil compiler with WASM support | ✅ Available at `parser/target/release/sigil` |
| WASM feature flag | ⚠️ Needs verification (`--features wasm`) |

### 3.2 Runtime Requirements

| Requirement | Status |
|-------------|--------|
| Qliphoth runtime | ✅ Available via symlink |
| Deploy copy | ⚠️ May need update |

---

## 4. Implementation Plan

### Phase 1: Source Updates (No Build Required)

#### 4.1.1 Update `main.sigil`

**File:** `website-qliphoth/src/main.sigil`

| Line | Change |
|------|--------|
| 108 | `create_text("0.3")` → `create_text("0.4")` |
| 890 | `"Sigil 0.3.0 — ..."` → `"Sigil 0.4.0 — Native syntax, SGDOC, native runtime"` |
| 895 | Release link → `v0.4.0` |

#### 4.1.2 Update `index.html`

**File:** `website-qliphoth/index.html`

| Line | Change |
|------|--------|
| 34 | `"version": "1.0.0"` → `"version": "0.4.0"` |

#### 4.1.3 Update `index.sigil`

**File:** `website-qliphoth/src/index.sigil`

| Change | Description |
|--------|-------------|
| Announcement text | Update to v0.4.0 |
| Release link | Point to v0.4.0 |

#### 4.1.4 Update `components.sigil`

**File:** `website-qliphoth/src/components.sigil`

| Change | Description |
|--------|-------------|
| Release link | `v1.0.0` → `v0.4.0` |

### Phase 2: Add Changelog Page

#### 4.2.1 Navigation Update

Add to header navigation in `main.sigil`:

```sigil
≔ changelog! = create_element("a");
set_class(changelog, "nav-link");
set_attr(changelog, "href", "/pages/changelog.html");
append_child(changelog, create_text("Changelog"));
append_child(links, changelog);
```

#### 4.2.2 Changelog Page Implementation

Create `changelog_app()` function with:
- Version history (0.4.0, 0.3.0, 0.2.0)
- Release dates
- Feature highlights per version
- Links to GitHub releases

#### 4.2.3 Footer Update

Add "Changelog" to footer navigation under "Learn" or "Agent Infrastructure" column.

### Phase 3: Documentation Updates

#### 4.3.1 Update `CLAUDE.md`

**File:** `/home/crook/dev2/workspace/sigil/sigil-lang/CLAUDE.md`

Add section:

```markdown
## Website

### Canonical Website: Qliphoth (WASM)

The **website-qliphoth/** directory contains the canonical Sigil website, written entirely in Sigil and compiled to WebAssembly.

**URL:** https://sigil-lang.com

**Architecture:**
- Source: `website-qliphoth/src/*.sigil`
- Build: `./build.sh` (compiles to WASM)
- Runtime: `sigil_runtime.js` (Qliphoth runtime)

### Deprecated: Static HTML

The **website/** directory contains a deprecated static HTML fallback.

**Status:** DEPRECATED - Do not update. Exists only as fallback for browsers without WASM support.
```

### Phase 4: Runtime Verification

#### 4.4.1 Verify Runtime Symlink

```bash
ls -la website-qliphoth/sigil_runtime.js
# Should point to: /home/crook/dev2/workspace/qliphoth/runtime/sigil_runtime.js
```

#### 4.4.2 Update Deploy Runtime

```bash
cp /home/crook/dev2/workspace/qliphoth/runtime/sigil_runtime.js website-qliphoth/deploy/
```

#### 4.4.3 WASM Build (Optional)

If WASM files need regeneration:

```bash
cd website-qliphoth
./build.sh --clean
```

**Note:** This requires `cargo build --release --features wasm` to have been run.

### Phase 5: Validation

#### 4.5.1 Version Consistency Check

```bash
grep -r "0\.3\." website-qliphoth/src/*.sigil
grep -r "1\.0\.0" website-qliphoth/*.html
# Should return empty if all updated
```

#### 4.5.2 Link Verification

All release links should point to:
- `https://github.com/Daemoniorum-LLC/sigil-lang/releases/tag/v0.4.0`

---

## 5. Test Plan

### 5.1 Manual Verification

| Test | Expected Result |
|------|-----------------|
| Version badge | Shows "0.4" |
| Announcement | Shows "Sigil 0.4.0" |
| JSON-LD version | `0.4.0` in page source |
| Changelog link | Navigates to changelog page |
| Release links | Open v0.4.0 GitHub release |

### 5.2 Build Verification

```bash
cd website-qliphoth
./build.sh --clean
# All files should compile successfully
```

---

## 6. Rollback Plan

If issues discovered:
1. Revert source files via git
2. Static HTML fallback remains functional at `website/`

---

## 7. Gap Documentation

### 7.1 Discovered Gaps

| Gap | Impact | Resolution |
|-----|--------|------------|
| JSON-LD shows 1.0.0 not 0.3.0 | Incorrect structured data | Fix in this spec |
| No changelog page | Missing feature | Add in Phase 2 |

### 7.2 Deferred Items

| Item | Reason |
|------|--------|
| WASM recompilation | May not be needed if sources don't change behavior |
| CD publishing (crates.io/npm) | Separate spec required |

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-01-25 | Initial specification |
