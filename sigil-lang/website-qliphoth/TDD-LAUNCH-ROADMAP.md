# sigil-lang.com Launch Roadmap (TDD)

## Overview
Prepare the Qliphoth-powered Sigil website for production deployment at sigil-lang.com.

**Target:** MVP Launch Ready
**Approach:** Test-Driven Development with Playwright E2E tests

---

## Phase 1: Infrastructure & Build System

### 1.1 Create Build Script
- [ ] Create `build.sh` that compiles all Sigil source to WASM
- [ ] Document build dependencies (Sigil compiler version, etc.)
- [ ] Add `package.json` scripts for build/dev/test

### 1.2 E2E Test Setup
- [ ] Create Playwright config for website-qliphoth
- [ ] Write smoke tests for all existing pages
- [ ] Verify WASM loads and renders correctly

---

## Phase 2: Navigation & UX Fixes

### 2.1 Mobile Navigation
- [ ] **TEST:** Mobile menu button visible at viewport < 768px
- [ ] **TEST:** Clicking menu button opens navigation drawer
- [ ] **TEST:** Menu items navigate correctly
- [ ] **IMPL:** Add hamburger menu component
- [ ] **IMPL:** Add slide-out navigation drawer

### 2.2 Fix Navigation Links
- [ ] **TEST:** All nav links resolve correctly
- [ ] **TEST:** Active page highlighted in nav
- [ ] **IMPL:** Audit and fix all relative paths
- [ ] **IMPL:** Add active state styling

### 2.3 Consistent Header/Footer
- [ ] **TEST:** Header present on all pages
- [ ] **TEST:** Footer present on all pages
- [ ] **IMPL:** Extract shared header/footer components

---

## Phase 3: Critical Content Pages

### 3.1 Installation Page (`install.html`)
- [ ] **TEST:** Page loads and renders
- [ ] **TEST:** Contains system requirements section
- [ ] **TEST:** Contains installation commands for Linux/macOS/Windows
- [ ] **TEST:** Contains cargo install option
- [ ] **TEST:** Contains verification command
- [ ] **IMPL:** Write install.sigil source
- [ ] **IMPL:** Compile to install.wasm

**Content Requirements:**
- System requirements (OS, architecture)
- One-liner curl install script
- Cargo install option
- Manual build from source
- Verify installation command
- Troubleshooting section

### 3.2 Getting Started Page (`getting-started.html`)
- [ ] **TEST:** Page loads and renders
- [ ] **TEST:** Contains "Hello World" example
- [ ] **TEST:** Contains project setup instructions
- [ ] **TEST:** Contains "What's Next" links
- [ ] **IMPL:** Write getting-started.sigil source
- [ ] **IMPL:** Compile to getting-started.wasm

**Content Requirements:**
- Hello World program
- Creating a new project (`sigil new`)
- Project structure explanation
- Running your first program
- Understanding output
- Next steps links

### 3.3 API Reference Page (`api.html`)
- [ ] **TEST:** Page loads and renders
- [ ] **TEST:** Contains searchable/filterable list
- [ ] **TEST:** Contains type definitions
- [ ] **TEST:** Contains function signatures
- [ ] **IMPL:** Write api.sigil source
- [ ] **IMPL:** Compile to api.wasm

**Content Requirements:**
- Core types (i32, i64, f32, f64, bool, String, etc.)
- Evidentiality markers (!, ~, ^, *)
- Control flow (⎇/⎉, ⌥, ⟳, ∀)
- Standard library functions
- Module system (invoke, ☉)

### 3.4 Examples Gallery (`examples.html`)
- [ ] **TEST:** Page loads and renders
- [ ] **TEST:** Contains at least 8 examples
- [ ] **TEST:** Each example has copy button
- [ ] **TEST:** Each example has "Run in Playground" link
- [ ] **IMPL:** Write examples.sigil source
- [ ] **IMPL:** Compile to examples.wasm

**Examples to Include:**
1. Hello World
2. FizzBuzz
3. Fibonacci
4. Factorial
5. String manipulation
6. Pattern matching
7. Error handling (Result type)
8. Async/await example
9. Struct/sigil definition
10. Evidentiality tracking

---

## Phase 4: Downloads & Releases

### 4.1 Downloads Page (`downloads.html`)
- [ ] **TEST:** Page loads and renders
- [ ] **TEST:** Contains download links for all platforms
- [ ] **TEST:** Shows current version
- [ ] **TEST:** Links to GitHub releases
- [ ] **IMPL:** Write downloads.sigil source
- [ ] **IMPL:** Compile to downloads.wasm

**Content Requirements:**
- Latest stable version badge
- Download buttons: Linux, macOS, Windows
- Architecture options (x64, ARM64)
- Checksums/signatures
- Link to GitHub releases
- Previous versions link

---

## Phase 5: Polish & Launch Prep

### 5.1 SEO & Meta Tags
- [ ] **TEST:** All pages have unique titles
- [ ] **TEST:** All pages have meta descriptions
- [ ] **TEST:** Open Graph tags present
- [ ] **IMPL:** Add OG image
- [ ] **IMPL:** Add favicon
- [ ] **IMPL:** Add sitemap.xml
- [ ] **IMPL:** Add robots.txt

### 5.2 Performance
- [ ] **TEST:** Lighthouse score > 90
- [ ] **TEST:** WASM files < 50KB each
- [ ] **TEST:** First contentful paint < 1.5s
- [ ] **IMPL:** Optimize WASM compilation flags
- [ ] **IMPL:** Add resource preloading

### 5.3 Accessibility
- [ ] **TEST:** Skip-to-content link present
- [ ] **TEST:** All images have alt text
- [ ] **TEST:** Color contrast passes WCAG AA
- [ ] **TEST:** Keyboard navigation works
- [ ] **IMPL:** Add ARIA labels where needed

---

## Phase 6: Deployment

### 6.1 CI/CD Setup
- [ ] GitHub Actions workflow for build
- [ ] Automated deployment to hosting
- [ ] Preview deployments for PRs

### 6.2 Domain & Hosting
- [ ] Configure sigil-lang.com DNS
- [ ] Set up Vercel/Netlify/CloudFlare Pages
- [ ] Enable HTTPS
- [ ] Configure CDN caching

### 6.3 Monitoring
- [ ] Set up error tracking (Sentry)
- [ ] Set up analytics (Plausible)
- [ ] Set up uptime monitoring

---

## Test Commands

```bash
# Run all E2E tests
npm run test:e2e

# Run specific test file
npm run test:e2e -- install.e2e.spec.ts

# Run with UI
npm run test:e2e:ui

# Build all WASM
./build.sh

# Start dev server
npm run dev
```

---

## Success Criteria

MVP launch is ready when:
- [ ] All Phase 1-4 tests pass (100%)
- [ ] All critical pages exist and render
- [ ] Navigation works on desktop and mobile
- [ ] Lighthouse score > 90
- [ ] No broken links
- [ ] Build is reproducible from source

---

## Timeline Estimate

| Phase | Effort | Parallel? |
|-------|--------|-----------|
| Phase 1: Infrastructure | 3-4 hrs | No |
| Phase 2: Navigation | 2-3 hrs | Yes |
| Phase 3: Content Pages | 8-10 hrs | Yes |
| Phase 4: Downloads | 2-3 hrs | Yes |
| Phase 5: Polish | 3-4 hrs | Yes |
| Phase 6: Deployment | 2-3 hrs | No |
| **TOTAL** | **20-27 hrs** | |

---

## File Structure (Target)

```
website-qliphoth/
├── index.html          ✅ EXISTS
├── docs.html           ✅ EXISTS
├── learn.html          ✅ EXISTS
├── agents.html         ✅ EXISTS
├── pattern.html        ✅ EXISTS
├── install.html        🔴 CREATE
├── getting-started.html 🔴 CREATE
├── api.html            🔴 CREATE
├── examples.html       🔴 CREATE
├── downloads.html      🔴 CREATE
├── *.wasm              (compiled from src/*.sigil)
├── src/
│   ├── docs.sigil      ✅ EXISTS
│   ├── learn.sigil     ✅ EXISTS
│   ├── agents.sigil    ✅ EXISTS
│   ├── pattern.sigil   ✅ EXISTS
│   ├── install.sigil   🔴 CREATE
│   ├── getting-started.sigil 🔴 CREATE
│   ├── api.sigil       🔴 CREATE
│   ├── examples.sigil  🔴 CREATE
│   └── downloads.sigil 🔴 CREATE
├── sigil_runtime.js    ✅ EXISTS (symlink)
├── build.sh            🔴 CREATE
├── package.json        🔴 CREATE
├── playwright.config.ts 🔴 CREATE
└── e2e/
    ├── smoke.e2e.spec.ts      🔴 CREATE
    ├── navigation.e2e.spec.ts 🔴 CREATE
    ├── install.e2e.spec.ts    🔴 CREATE
    ├── examples.e2e.spec.ts   🔴 CREATE
    └── mobile.e2e.spec.ts     🔴 CREATE
```
