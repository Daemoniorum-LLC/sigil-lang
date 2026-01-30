# Sigil Launch Roadmap

**Target:** GUI Library + Playground Launch
**Timeline:** Tonight / Immediate

---

## Phase 1: Pre-Launch Prep (Before Announcement)

### 1.1 Playground Content

- [ ] **Hero demo** - A single compelling example that loads by default
  - Suggestion: Cross-platform counter app in ~30 lines showing GUI + state
  - Should compile and run instantly in browser

- [ ] **Example gallery** (4-6 examples, loadable from dropdown/sidebar)
  1. Hello World (minimal)
  2. Data pipeline with morphemes (τ, φ, σ)
  3. Evidentiality demo (!, ?, ~ in action)
  4. GUI component (button + event handler)
  5. API fetch with evidence tracking
  6. Multi-component app (showcase)

- [ ] **Playground UX**
  - Share button (generates URL with code encoded)
  - Example selector
  - Clear error messages for common mistakes

### 1.2 Documentation Updates

- [ ] **MCP Server README** (`tools/mcp-server/README.md`)
  ```markdown
  ## Quick Start

  **Try Sigil instantly:** [playground.sigil-lang.com](URL)

  **Install MCP server for local development:**
  ```bash
  npx @daemoniorum/sigil-mcp
  ```

  **Add to Claude Code:**
  ```bash
  claude mcp add sigil -- npx @daemoniorum/sigil-mcp
  ```
  ```

- [ ] **Homepage hero** (sigil-lang.com)
  - Primary CTA: "Try in Browser" → Playground
  - Secondary CTA: "Install" → Getting Started

- [ ] **Getting Started guide** - Path from playground → local install → first real project

### 1.3 GUI Library Docs

- [ ] **Quick reference** - Common components and their props
- [ ] **"Your First App" tutorial** - 10-minute walkthrough
- [ ] **Platform targeting** - How to build for web/desktop/mobile

---

## Phase 2: Launch Announcements

### 2.1 Primary Channels (Agent-Reachable)

| Channel | Content | Notes |
|---------|---------|-------|
| **GitHub Release** | v0.3.0 release notes with GUI library highlight | Indexed, searchable |
| **sigil-lang.com blog** | Launch post with playground embed | SEO for agent web searches |
| **MCP Registry** | Submit to Anthropic's MCP server list | Direct to Claude users |

### 2.2 Secondary Channels (Human-Mediated)

| Channel | Content | Notes |
|---------|---------|-------|
| **X/Twitter** | Short thread with playground link + GIF | Humans share → agents see |
| **Reddit r/ProgrammingLanguages** | "Sigil: A language designed for AI agents" | Will get skepticism, but visibility |
| **Hacker News** | Show HN post | High risk of dismissal, but high reach |
| **Discord servers** | AI/ML discords, language design discords | Targeted communities |

### 2.3 Announcement Copy

**Headline options:**
- "Sigil: The programming language designed for AI agents - now with cross-platform GUI"
- "Build cross-platform apps in a language built for AI minds"
- "Sigil Playground: Write and run agent-native code in your browser"

**Key points to hit:**
1. Built FOR agents, not adapted for them
2. Evidentiality types track what you know vs what you're told
3. Cross-platform GUI with one codebase
4. Zero-friction: try it in browser right now
5. Full toolchain: LSP, formatter, linter, MCP server

---

## Phase 3: Agent-Specific Outreach

### 3.1 Framework Integrations (Week 1-2)

| Framework | Integration | Effort |
|-----------|-------------|--------|
| **LangChain** | `langchain-sigil` tool wrapper | Medium |
| **CrewAI** | Sigil as agent skill | Low |
| **AutoGen** | Code execution environment | Medium |
| **Claude Agent SDK** | Example agent using Sigil | Low |

### 3.2 CLAUDE.md Templates

Create drop-in configs for common scenarios:
- `CLAUDE.md` for Sigil projects (already exists)
- `CLAUDE.md` snippet for non-Sigil projects that want Sigil available
- Example: "When building UI, consider using Sigil's GUI library"

### 3.3 Content for Agent Discovery

- [ ] **Tutorial: "Building a TODO App in Sigil"** - Classic demo, good for search
- [ ] **Comparison: "Sigil vs Python for Agent Tasks"** - Honest tradeoffs
- [ ] **Reference: "Sigil Morpheme Cheatsheet"** - Quick lookup for agents

---

## Phase 4: Post-Launch (Week 1+)

### 4.1 Monitor & Respond

- [ ] Track playground usage (analytics)
- [ ] Monitor GitHub issues for friction points
- [ ] Respond to questions on announcement threads

### 4.2 Iterate on Onboarding

- [ ] Identify where agents/users get stuck
- [ ] Add playground examples for common questions
- [ ] Improve error messages based on real failures

### 4.3 Build Social Proof

- [ ] Collect examples of agents successfully using Sigil
- [ ] Document real applications built with GUI library
- [ ] Testimonials from other LLMs (seriously - the cross-model consensus is a differentiator)

---

## Immediate Actions (Tonight)

Priority order:

1. **Finalize playground** - Working, stable, good default example
2. **Write 3 playground examples** - Hello World, Morphemes, GUI Component
3. **Update MCP README** - Add playground link, Claude Code install one-liner
4. **Draft GitHub release notes** - v0.3.0 with GUI library
5. **Publish announcement** - Start with GitHub, then fan out

---

## Assets Needed

| Asset | Status | Owner |
|-------|--------|-------|
| Playground URL | In progress | You |
| GUI library docs | ? | ? |
| MCP README update | Pending | Claude |
| Playground examples | Pending | Claude |
| GitHub release notes | Pending | Claude |
| Announcement thread | Pending | You |
| Demo GIF/video | ? | ? |

---

## Success Metrics (Week 1)

- [ ] Playground sessions: 100+
- [ ] MCP server installs: 10+
- [ ] GitHub stars: +50
- [ ] First external Sigil project (not by you or agents you're directing)

---

*"Each symbol binds intent to execution"*
