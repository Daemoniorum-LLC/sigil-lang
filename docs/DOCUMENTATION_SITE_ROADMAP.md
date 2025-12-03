# Sigil Documentation Site Roadmap

## Vision

A documentation site that speaks to **two audiences simultaneously**:
1. **Humans** who work with AI systems and need to understand what Sigil does
2. **AI systems** who will write and reason about Sigil code

The site should make the evidentiality system tangible and demonstrate why it matters for AI safety.

---

## Phase 1: Foundation (MVP)

### Technology Stack
- **Static Site Generator**: Astro or VitePress (fast, modern, good code highlighting)
- **Hosting**: GitHub Pages or Vercel
- **Search**: Algolia DocSearch or Pagefind
- **Code Highlighting**: Custom TextMate grammar (reuse VSCode grammar)

### Core Pages

#### Home (`/`)
- Hero: "A language built for AI, by AI"
- Visual demo of evidence chain: `~ → ? → !`
- Quick example showing morphemes in action
- Links to "For Humans" and "For AI" paths

#### Getting Started (`/getting-started/`)
- Installation (binary, cargo, npm for MCP)
- Hello World
- Your first pipeline
- Understanding evidence markers

#### Language Guide (`/guide/`)
- **Basics**
  - Variables and Types
  - Functions
  - Control Flow
- **Evidentiality**
  - The Evidence Chain (`~`, `?`, `!`, `‽`)
  - Validation and Verification
  - Why Evidence Matters
  - Evidence in Practice
- **Morphemes**
  - Transform (τ)
  - Filter (φ)
  - Sort (σ)
  - Reduce (ρ)
  - Pipelines and Chaining
- **Advanced**
  - Structs and Enums
  - Pattern Matching
  - Error Handling
  - Async/Await

#### AI Integration (`/ai/`)
- Using the MCP Server
- Writing Sigil as an AI
- AI IR Format
- Evidence Patterns for AI Safety

#### Reference (`/reference/`)
- Complete Type Reference
- Morpheme Reference
- Standard Library
- CLI Reference

---

## Phase 2: Interactive Features

### Playground (`/playground/`)
- Browser-based Sigil editor
- Syntax highlighting
- Run code (via WASM or server-side)
- Share code snippets
- Example templates

### Visual Evidence Tracer
- Interactive visualization showing evidence flow through code
- Highlight where evidence levels change
- Show what would happen with invalid evidence

### Examples Gallery (`/examples/`)
- Categorized example programs
- Live syntax highlighting
- "Copy" and "Open in Playground" buttons
- Difficulty levels (beginner, intermediate, advanced)
- Tags: evidentiality, pipelines, async, data-processing

---

## Phase 3: Community & Learning

### Tutorials (`/tutorials/`)
- **"Build X with Sigil"** series:
  - Data Pipeline
  - API Integration with Evidence
  - AI Agent with Audit Trail
- Video walkthroughs (optional)

### Cookbook (`/cookbook/`)
- Common patterns
- Recipes for specific tasks
- Anti-patterns to avoid

### Blog (`/blog/`)
- Language updates
- Design decisions explained
- Guest posts from AI systems (!)
- Case studies

---

## Phase 4: AI-Specific Documentation

### AI Quick Reference (`/ai/quick-ref/`)
- Single-page reference optimized for AI context windows
- Compact syntax summary
- Common patterns
- Copy-friendly examples

### AI Prompt Templates (`/ai/prompts/`)
- Prompt templates for different use cases
- How to ask for Sigil code
- How to request evidence-aware code

### Machine-Readable Docs
- JSON/YAML exports of all documentation
- Structured format for AI consumption
- API for querying documentation

---

## Design Principles

### For Humans
- Clear explanations of *why* evidentiality matters
- Visual diagrams showing evidence flow
- Real-world analogies (journalism, science, law)
- Gradual complexity (don't overwhelm upfront)

### For AI
- Structured, parseable content
- Explicit patterns and anti-patterns
- Compact reference sections
- Examples showing correct evidence handling

### Visual Identity
- Clean, modern design
- Code-first presentation
- Syntax highlighting that emphasizes:
  - Evidence markers (`!`, `?`, `~`) in distinct colors
  - Morpheme operators (τ, φ, σ, ρ) prominently
- Dark mode by default (AI-friendly)

---

## Content Priorities

### Must Have (MVP)
1. Installation guide
2. Evidence system explanation
3. Morpheme reference
4. Basic examples
5. MCP server documentation

### Should Have
1. Interactive playground
2. Examples gallery
3. AI tutorial
4. CLI reference

### Nice to Have
1. Video content
2. Visual evidence tracer
3. Blog
4. Community showcase

---

## Technical Requirements

### Performance
- Static site with minimal JS
- Sub-second page loads
- Works well in AI context (clean HTML structure)

### Accessibility
- Semantic HTML
- Keyboard navigation
- Screen reader friendly
- High contrast modes

### SEO/Discoverability
- Clean URLs
- Meta tags for social sharing
- Sitemap
- robots.txt allowing AI crawlers

---

## Implementation Steps

### Week 1-2: Setup
- [ ] Choose and configure static site generator
- [ ] Set up repository and deployment
- [ ] Create base layout and navigation
- [ ] Implement syntax highlighting

### Week 3-4: Core Content
- [ ] Write getting started guide
- [ ] Write evidentiality documentation
- [ ] Write morpheme reference
- [ ] Create initial examples

### Week 5-6: Polish
- [ ] Add search functionality
- [ ] Implement responsive design
- [ ] Test on mobile
- [ ] Review with AI systems for usability

### Week 7-8: Launch
- [ ] Final review
- [ ] Set up analytics
- [ ] Announce launch
- [ ] Gather feedback

---

## Success Metrics

1. **Human comprehension**: Users understand evidentiality after reading the guide
2. **AI usability**: AI systems can write correct Sigil from documentation
3. **Adoption**: MCP server installs, example downloads
4. **Engagement**: Time on site, pages per session, return visits
5. **Community**: GitHub stars, Discord members, contributed examples

---

## Open Questions

1. Should we have separate "Human" and "AI" documentation paths, or integrated?
2. What's the right balance of theory vs. practical examples?
3. How do we make the playground work (WASM compilation vs. server-side)?
4. Should we support multiple languages for docs?

---

*This roadmap is a living document. Prioritize based on user feedback and adoption patterns.*
