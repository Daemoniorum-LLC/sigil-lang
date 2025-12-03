# Sigil Playground Design

## Overview

A browser-based environment for writing, running, and sharing Sigil code. The playground makes Sigil accessible without installation and serves as an interactive learning tool.

---

## User Experience

### Interface Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  Sigil Playground               [Examples ▼] [Share] [Settings] │
├────────────────────────────────────┬────────────────────────────┤
│                                    │                            │
│   ┌────────────────────────────┐   │   ┌────────────────────┐   │
│   │ // Your code here          │   │   │ Output             │   │
│   │                            │   │   │                    │   │
│   │ fn main() {                │   │   │ > 55               │   │
│   │     let nums = [1,2,3,4,5];│   │   │                    │   │
│   │     let sum = nums|Σ;      │   │   │                    │   │
│   │     print(sum);            │   │   │                    │   │
│   │ }                          │   │   │                    │   │
│   │                            │   │   │                    │   │
│   └────────────────────────────┘   │   └────────────────────┘   │
│                                    │                            │
│   [▶ Run] [Check] [Format] [IR]    │   [Clear] [Copy]          │
│                                    │                            │
├────────────────────────────────────┴────────────────────────────┤
│  Evidence Flow: ~ reported → ? uncertain → ! known              │
│  ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
└─────────────────────────────────────────────────────────────────┘
```

### Key Features

#### 1. Code Editor
- Syntax highlighting (reuse VSCode TextMate grammar)
- Auto-completion for morphemes
- Bracket matching
- Line numbers
- Error highlighting inline

#### 2. Actions
- **Run**: Execute code and show output
- **Check**: Type-check only, show errors
- **Format**: Auto-format code
- **IR**: Show AI IR representation

#### 3. Output Panel
- Standard output from print statements
- Error messages with helpful context
- Execution time

#### 4. Evidence Visualizer
- Visual representation of evidence flow
- Shows where evidence levels change
- Highlights validation/verification points

#### 5. Examples Dropdown
- Categorized example programs
- One-click load into editor
- Searchable

#### 6. Sharing
- Generate shareable URL with code embedded
- Copy link button
- Optional: GitHub Gist export

---

## Technical Architecture

### Option A: WASM Compilation (Recommended)

Compile the Sigil interpreter to WebAssembly for client-side execution.

**Pros:**
- No server needed
- Fast execution
- Works offline
- Privacy (code stays local)

**Cons:**
- Initial WASM download (~2-5MB)
- Limited to interpreter (no JIT/LLVM)

**Implementation:**
1. Add WASM target to Rust parser
2. Create JS bindings with wasm-bindgen
3. Load WASM module in browser
4. Call execution functions from JS

```rust
// In parser/src/lib.rs
#[wasm_bindgen]
pub fn run_code(source: &str) -> String {
    // Parse and interpret, return output
}

#[wasm_bindgen]
pub fn check_code(source: &str) -> String {
    // Type check, return JSON errors
}

#[wasm_bindgen]
pub fn get_ir(source: &str) -> String {
    // Return AI IR as JSON
}
```

### Option B: Server-Side Execution

Run code on a backend server.

**Pros:**
- Full feature support (JIT, LLVM)
- Smaller client payload
- Easier to add new features

**Cons:**
- Server infrastructure needed
- Network latency
- Security concerns (sandboxing)
- Costs

**Implementation:**
1. Create simple HTTP API
2. Use container sandboxing (gVisor, Firecracker)
3. Rate limiting and timeouts
4. Queue for heavy workloads

---

## Frontend Implementation

### Technology Stack
- **Framework**: Preact or Solid.js (small bundle)
- **Editor**: CodeMirror 6 (customizable, modern)
- **Styling**: Tailwind CSS or vanilla CSS
- **Build**: Vite

### Editor Configuration

```javascript
import { EditorState } from '@codemirror/state';
import { EditorView, keymap } from '@codemirror/view';
import { sigilLanguage } from './sigil-lang';

const editor = new EditorView({
  state: EditorState.create({
    doc: initialCode,
    extensions: [
      sigilLanguage(),
      sigilTheme,
      morphemeCompletion,
      evidentialityHighlight,
    ],
  }),
  parent: document.getElementById('editor'),
});
```

### Evidence Visualizer

```javascript
function EvidenceFlow({ code, analysis }) {
  return (
    <div className="evidence-flow">
      {analysis.evidencePoints.map(point => (
        <EvidenceMarker
          type={point.level}  // 'reported' | 'uncertain' | 'known'
          line={point.line}
          description={point.description}
        />
      ))}
    </div>
  );
}
```

---

## URL Schema

Shareable URLs encode the code:

```
https://sigil-lang.org/playground#code=BASE64_ENCODED_CODE
```

Or with compression:

```
https://sigil-lang.org/playground#v=1&c=COMPRESSED_CODE
```

### Short URL Option

For frequently shared examples, use short codes:

```
https://sigil-lang.org/p/abc123
```

Stored in a simple KV store (Redis, Cloudflare KV, etc.)

---

## Security Considerations

### Client-Side (WASM)
- Runs in browser sandbox
- No file system access
- Memory limits enforced by browser
- Infinite loop protection via fuel/gas mechanism

### Server-Side
- Container sandboxing (gVisor recommended)
- CPU time limits (5 seconds max)
- Memory limits (256MB max)
- No network access from sandbox
- Rate limiting per IP

---

## Mobile Support

- Responsive layout (stacked on mobile)
- Touch-friendly buttons
- Keyboard popup consideration
- Minimal scrolling needed

---

## Analytics & Monitoring

Track (privacy-respecting):
- Popular examples
- Common errors (to improve error messages)
- Execution success rate
- Feature usage (Run vs Check vs IR)

---

## Development Phases

### Phase 1: MVP
- [ ] Basic editor with syntax highlighting
- [ ] WASM compilation of interpreter
- [ ] Run button with output
- [ ] 5 starter examples

### Phase 2: Enhanced
- [ ] Type checking with inline errors
- [ ] IR viewer
- [ ] More examples
- [ ] Share functionality

### Phase 3: Polish
- [ ] Evidence visualizer
- [ ] Auto-completion
- [ ] Format on save
- [ ] Mobile optimization

### Phase 4: Community
- [ ] Short URL sharing
- [ ] Embed widget for docs
- [ ] Example submissions
- [ ] Social sharing previews

---

## Estimated Effort

| Component | Estimate |
|-----------|----------|
| WASM build setup | 1-2 days |
| Basic editor UI | 2-3 days |
| Syntax highlighting | 1 day |
| Run/output integration | 1-2 days |
| Examples system | 1 day |
| Sharing | 1-2 days |
| Evidence visualizer | 3-5 days |
| Mobile optimization | 1-2 days |
| **Total MVP** | **~2 weeks** |

---

## Open Questions

1. Do we want login/accounts for saving code?
2. Should we support multiple files/projects?
3. Integrate with MCP server for AI assistance?
4. Embed capability for external sites?

---

## Success Metrics

- Time to first successful run < 10 seconds
- Error rate < 5%
- Sharing rate > 10%
- Return visitors > 20%
- Example completion rate > 50%
