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

## Phase 5: AI Agent Development Guide (NEW)

Since Sigil is designed specifically for AI agents, this section is **critical**.

### Agent Development Tutorial Series

#### Part 1: Agent Fundamentals
```
/docs/agents/fundamentals/
├── why-sigil-for-agents.md     # Evidentiality as trust tracking
├── agent-architecture.md        # How agents work in Sigil
├── evidence-for-safety.md       # Trust boundaries, injection prevention
└── getting-started.md           # First agent tutorial
```

#### Part 2: Tool Calling
```
/docs/agents/tools/
├── defining-tools.md           # tool_define, parameters, evidence
├── generating-schemas.md       # tool_schema for OpenAI/Claude
├── executing-tools.md          # tool_call with evidence wrapping
├── tool-patterns.md            # Common patterns and best practices
└── reference.md                # Complete API reference
```

**Example: Defining an Agent Tool**
```sigil
// Define a tool for LLM introspection
tool_define(
    "get_weather",
    "Get current weather for a location",
    [
        {name: "location", type: "string", description: "City name", required: true, evidence: "~"},
        {name: "units", type: "string", description: "celsius or fahrenheit", required: false}
    ],
    "WeatherData",
    fn(location~, units) {
        // Implementation - result is automatically reported~
        let data~ = http_get("api.weather.com/" ++ location~)
        return parse_weather(data~)
    }
)

// Generate schema for LLM tool_choice
let schema = tool_schema("get_weather")
// Returns OpenAI/Claude compatible JSON schema
```

#### Part 3: LLM Integration
```
/docs/agents/llm/
├── providers.md                # OpenAI, Claude, Gemini, Mistral, Ollama
├── requests.md                 # Building and configuring requests
├── responses.md                # Parsing responses, handling tool calls
├── structured-output.md        # Type-safe extraction
├── prompts.md                  # Templates and rendering
└── reference.md                # Complete API reference
```

**Example: Complete Agent Interaction**
```sigil
fn agent_loop(session) {
    // Create LLM request with tools
    let request = llm_request("claude")
        |llm_with_system("You are a helpful assistant with tools.")
        |llm_with_messages(memory_history_get(session))
        |llm_with_tools(tool_schemas_all())

    // Send request (returns reported~ data)
    let response~ = llm_send(request)

    // Parse tool call
    let parsed = llm_parse_tool_call(response~)

    if parsed.is_tool_call {
        // Execute tool (returns reported~ result)
        let result~ = tool_call(parsed.tool_name, parsed.tool_input)

        // Add to history
        memory_history_add(session, "tool_result", to_string(result~))

        // Continue loop
        return agent_loop(session)
    } else {
        // Text response
        memory_history_add(session, "assistant", parsed.content)
        return parsed.content
    }
}
```

#### Part 4: Agent Memory
```
/docs/agents/memory/
├── sessions.md                 # Creating and managing sessions
├── context.md                  # Storing and retrieving context
├── history.md                  # Conversation history management
├── windowing.md                # Context window strategies
└── reference.md                # Complete API reference
```

#### Part 5: Planning & State Machines
```
/docs/agents/planning/
├── state-machines.md           # Defining agent workflows
├── transitions.md              # State transitions and guards
├── goals.md                    # Goal tracking and decomposition
├── progress.md                 # Progress monitoring
└── reference.md                # Complete API reference
```

**Example: Agent State Machine**
```sigil
// Define agent workflow
let agent = plan_state_machine("research_agent", [
    "idle",
    "gathering",
    "analyzing",
    "synthesizing",
    "complete"
])

// Add transitions
plan_add_transition(agent, "idle", "gathering")
plan_add_transition(agent, "gathering", "analyzing")
plan_add_transition(agent, "analyzing", "synthesizing")
plan_add_transition(agent, "synthesizing", "complete")
plan_add_transition(agent, "analyzing", "gathering")  // Can loop back

// Check current state
plan_current_state(agent)  // "idle"

// Execute transition
plan_transition(agent, "gathering")  // {success: true, from: "idle", to: "gathering"}

// Get available next states
plan_available_transitions(agent)  // ["analyzing"]
```

#### Part 6: Vector Search & RAG
```
/docs/agents/vectors/
├── embeddings.md               # Creating embeddings
├── similarity.md               # Cosine, Euclidean, dot product
├── vector-stores.md            # In-memory vector stores
├── search.md                   # k-NN search
├── rag-patterns.md             # RAG implementation patterns
└── reference.md                # Complete API reference
```

**Example: RAG Pipeline**
```sigil
// Create vector store
let store = vec_store()

// Add documents with embeddings
let docs = ["Sigil is a language for AI.", "Evidentiality tracks trust.", "Morphemes are operators."]
for doc in docs {
    let embedding? = vec_embedding(doc)  // uncertain? evidence
    vec_store_add(store, embedding?, {text: doc})
}

// Search for similar
let query? = vec_embedding("How does Sigil track trust?")
let results = vec_store_search(store, query?, 3)

// Results are sorted by similarity
for result in results {
    println(result.similarity, result.metadata.text)
}
```

#### Part 7: Multi-Agent Systems
```
/docs/agents/multi-agent/
├── actors.md                   # Actor model overview
├── communication.md            # Message passing patterns
├── coordination.md             # Consensus and coordination
├── swarms.md                   # Swarm behaviors
└── examples.md                 # Multi-agent examples
```

#### Part 8: Safety & Trust
```
/docs/agents/safety/
├── trust-boundaries.md         # Evidence-based boundaries
├── validation.md               # Input validation patterns
├── injection.md                # Preventing prompt injection
├── audit-trails.md             # Logging and accountability
└── best-practices.md           # Security best practices
```

---

## Phase 6: Standard Library Reference (Expanded)

### New AI Agent Modules

#### agent_tools
| Function | Description | Evidence |
|----------|-------------|----------|
| `tool_define(name, desc, params, returns, handler?)` | Register a tool | - |
| `tool_list()` | List all registered tools | - |
| `tool_get(name)` | Get tool definition | - |
| `tool_schema(name)` | Generate OpenAI/Claude schema | - |
| `tool_schemas_all()` | Get all schemas | - |
| `tool_call(name, ...args)` | Execute tool | reported~ |
| `tool_remove(name)` | Remove tool | - |
| `tool_clear()` | Clear all tools | - |

#### agent_llm
| Function | Description | Evidence |
|----------|-------------|----------|
| `llm_message(role, content)` | Create chat message | - |
| `llm_messages(...)` | Create messages array | - |
| `llm_request(provider, model?)` | Build request | - |
| `llm_with_tools(request, tools)` | Add tools | - |
| `llm_with_system(request, prompt)` | Add system prompt | - |
| `llm_with_messages(request, msgs)` | Add messages | - |
| `llm_send(request)` | Send request | reported~ |
| `llm_parse_tool_call(response)` | Parse tool call | - |
| `llm_extract(response, schema)` | Extract structured data | uncertain? |
| `prompt_template(template)` | Create template | - |
| `prompt_render(template, values)` | Render template | - |

#### agent_memory
| Function | Description | Evidence |
|----------|-------------|----------|
| `memory_session(id)` | Create/get session | - |
| `memory_set(session, key, value)` | Store context | - |
| `memory_get(session, key)` | Retrieve context | - |
| `memory_history_add(session, role, content)` | Add to history | - |
| `memory_history_get(session, limit?)` | Get history | - |
| `memory_context_all(session)` | Get all context | - |
| `memory_clear(session)` | Clear session | - |
| `memory_sessions_list()` | List sessions | - |

#### agent_planning
| Function | Description | Evidence |
|----------|-------------|----------|
| `plan_state_machine(name, states)` | Create state machine | - |
| `plan_add_transition(machine, from, to)` | Add transition | - |
| `plan_current_state(machine)` | Get current state | - |
| `plan_transition(machine, to)` | Execute transition | - |
| `plan_can_transition(machine, to)` | Check if valid | - |
| `plan_available_transitions(machine)` | Get available | - |
| `plan_history(machine)` | Get history | - |
| `plan_goal(name, criteria)` | Create goal | - |
| `plan_subgoals(goal, subgoals)` | Add subgoals | - |
| `plan_update_progress(goal, progress)` | Update progress | - |
| `plan_check_goal(goal, context)` | Check criteria | - |

#### agent_vectors
| Function | Description | Evidence |
|----------|-------------|----------|
| `vec_embedding(text)` | Create embedding | uncertain? |
| `vec_cosine_similarity(a, b)` | Cosine similarity | - |
| `vec_euclidean_distance(a, b)` | Euclidean distance | - |
| `vec_dot_product(a, b)` | Dot product | - |
| `vec_normalize(vec)` | Normalize to unit | - |
| `vec_search(query, corpus, k)` | k-NN search | - |
| `vec_store()` | Create vector store | - |
| `vec_store_add(store, vec, meta)` | Add to store | - |
| `vec_store_search(store, query, k)` | Search store | - |

---

## Phase 7: Technical Implementation

### Astro Starlight Configuration

```javascript
// astro.config.mjs
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';

export default defineConfig({
  integrations: [
    starlight({
      title: 'Sigil',
      description: 'A polysynthetic language for AI agents',
      social: {
        github: 'https://github.com/Daemoniorum-LLC/sigil-lang',
        discord: 'https://discord.gg/sigil',
      },
      sidebar: [
        { label: 'Getting Started', link: '/getting-started/' },
        { label: 'Language Guide', items: [
          { label: 'Basics', link: '/guide/basics/' },
          { label: 'Evidentiality', link: '/guide/evidentiality/' },
          { label: 'Morphemes', link: '/guide/morphemes/' },
        ]},
        { label: 'AI Agents', items: [
          { label: 'Why Sigil?', link: '/agents/why/' },
          { label: 'Tool Calling', link: '/agents/tools/' },
          { label: 'LLM Integration', link: '/agents/llm/' },
          { label: 'Memory', link: '/agents/memory/' },
          { label: 'Planning', link: '/agents/planning/' },
          { label: 'Vectors', link: '/agents/vectors/' },
        ]},
        { label: 'Reference', items: [
          { label: 'Standard Library', link: '/reference/stdlib/' },
          { label: 'CLI', link: '/reference/cli/' },
        ]},
      ],
      customCss: ['./src/styles/sigil.css'],
    }),
  ],
});
```

### Sigil Syntax Highlighting Grammar

```json
{
  "name": "Sigil",
  "scopeName": "source.sigil",
  "patterns": [
    {
      "name": "keyword.evidence.sigil",
      "match": "[!?~‽]"
    },
    {
      "name": "keyword.morpheme.sigil",
      "match": "[τφσρΣΠαωμχνξ]"
    },
    {
      "name": "keyword.control.sigil",
      "match": "\\b(if|else|match|while|for|loop|return|break|continue)\\b"
    },
    {
      "name": "keyword.declaration.sigil",
      "match": "\\b(fn|let|mut|struct|enum|trait|impl|mod|use|pub)\\b"
    }
  ]
}
```

---

## Revised Timeline

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| Foundation | 2 weeks | Site setup, landing, getting started |
| Language Reference | 2 weeks | Types, evidentiality, morphemes |
| **Agent Development** | **3 weeks** | **Complete agent tutorial series** |
| Stdlib Reference | 2 weeks | All 47 modules documented |
| Playground | 2 weeks | Interactive editor + WASM |
| Examples | 2 weeks | Cookbook + agent examples |
| Internals | 1 week | Compiler docs |
| Polish | 1 week | Search, SEO, performance |

**Total: 15 weeks to comprehensive documentation**

---

*This roadmap is a living document. Prioritize based on user feedback and adoption patterns.*
