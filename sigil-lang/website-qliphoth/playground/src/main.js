/**
 * Sigil Playground
 * Browser-based environment for writing and running Sigil code
 */

import { EditorView, basicSetup } from 'codemirror';
import { EditorState } from '@codemirror/state';
import { keymap } from '@codemirror/view';
import { indentWithTab } from '@codemirror/commands';
import { StreamLanguage } from '@codemirror/language';

// Sigil syntax highlighting (simplified mode)
const sigilLanguage = StreamLanguage.define({
  token(stream, state) {
    // Comments
    if (stream.match('//')) {
      stream.skipToEnd();
      return 'comment';
    }

    // Strings
    if (stream.match('"')) {
      while (!stream.eol()) {
        if (stream.next() === '"' && stream.peek() !== '\\') break;
      }
      return 'string';
    }

    // Numbers
    if (stream.match(/^-?\d+\.?\d*/)) {
      return 'number';
    }

    // Morpheme operators
    if (stream.match(/[τφσρΤΦΣΡαωΑΩΠλΛ⌛]/)) {
      return 'morpheme';
    }

    // Evidence markers
    if (stream.match('!')) return 'evidence-known';
    if (stream.match('?')) return 'evidence-uncertain';
    if (stream.match('~')) return 'evidence-reported';
    if (stream.match('‽')) return 'evidence-paradox';

    // Keywords (includes both Jormungandr and canonical syntax)
    if (stream.match(/\b(fn|rite|let|mut|if|else|match|return|for|while|in|struct|sigil|enum|trait|impl|use|pub|async|await)\b/)) {
      return 'keyword';
    }

    // Unicode keywords
    if (stream.match(/[≔⊢ᛈ→]/)) {
      return 'keyword';
    }

    // Types
    if (stream.match(/\b(i8|i16|i32|i64|u8|u16|u32|u64|f32|f64|bool|str|char|void)\b/)) {
      return 'type';
    }

    // Booleans
    if (stream.match(/\b(true|false)\b/)) {
      return 'atom';
    }

    // Functions (followed by paren)
    if (stream.match(/[a-z_][a-z0-9_]*(?=\s*\()/)) {
      return 'function';
    }

    // Identifiers
    if (stream.match(/[a-zA-Z_][a-zA-Z0-9_]*/)) {
      return 'variable';
    }

    // Operators
    if (stream.match(/[+\-*/%=<>!&|^]+/)) {
      return 'operator';
    }

    stream.next();
    return null;
  }
});

// Custom theme for Sigil - Teal design system
const sigilTheme = EditorView.theme({
  '&': {
    backgroundColor: '#050507',
    color: '#e8e8ec',
  },
  '.cm-content': {
    caretColor: '#14A088',
    padding: '1rem',
  },
  '.cm-cursor': {
    borderLeftColor: '#14A088',
  },
  '.cm-activeLine': {
    backgroundColor: 'rgba(20, 160, 136, 0.08)',
  },
  '.cm-gutters': {
    backgroundColor: '#0a0a0d',
    color: '#707078',
    border: 'none',
    borderRight: '1px solid rgba(255, 255, 255, 0.1)',
  },
  '.cm-activeLineGutter': {
    backgroundColor: 'rgba(20, 160, 136, 0.08)',
  },
  '.cm-selectionBackground': {
    backgroundColor: 'rgba(20, 160, 136, 0.25) !important',
  },
}, { dark: true });

// Syntax highlighting styles - aligned with docs page colors
const sigilHighlight = EditorView.baseTheme({
  '.cm-keyword': { color: '#C792EA' },
  '.cm-string': { color: '#C3E88D' },
  '.cm-number': { color: '#F78C6C' },
  '.cm-comment': { color: '#707078', fontStyle: 'italic' },
  '.cm-function': { color: '#82AAFF' },
  '.cm-variable': { color: '#e8e8ec' },
  '.cm-type': { color: '#FFCB6B' },
  '.cm-atom': { color: '#F78C6C' },
  '.cm-operator': { color: '#89DDFF' },
  '.cm-morpheme': { color: '#14A088', fontWeight: 'bold' },
  '.cm-evidence-known': { color: '#4CAF50', fontWeight: 'bold' },
  '.cm-evidence-uncertain': { color: '#FFC107', fontWeight: 'bold' },
  '.cm-evidence-reported': { color: '#2196F3', fontWeight: 'bold' },
  '.cm-evidence-paradox': { color: '#CE93D8', fontWeight: 'bold' },
});

// Example programs - comprehensive collection
const EXAMPLES = {
  // Getting Started (Canonical Sigil syntax)
  hello: `// Hello World in Sigil (Canonical Syntax)
// Uses: rite (function), → (arrow), ≔ (let)
rite main() → i64 {
    ≔ message = "Hello, Sigil!";
    ≔ x = 42;
    ≔ y = 10;
    x + y
}`,

  variables: `// Variables and Types
fn main() {
    // Immutable by default
    let x = 42;
    let name = "Sigil";
    let pi = 3.14159;
    let flag = true;

    // Mutable variables
    let mut counter = 0;
    counter = counter + 1;

    // Explicit types
    let age: i64 = 25;
    let price: f64 = 19.99;

    print("x = " + str(x));
    print("name = " + name);
    print("counter = " + str(counter));

    return 0;
}`,

  functions: `// Functions in Sigil
fn add(a: i64, b: i64) -> i64 {
    a + b  // Implicit return
}

fn greet(name: Str) {
    print("Hello, " + name + "!");
}

fn factorial(n: i64) -> i64 {
    if n <= 1 {
        return 1;
    }
    n * factorial(n - 1)
}

fn main() {
    let sum = add(10, 20);
    print("10 + 20 = " + str(sum));

    greet("World");

    let fact5 = factorial(5);
    print("5! = " + str(fact5));

    return 0;
}`,

  // Morphemes
  pipes: `// Morpheme Pipelines
fn main() {
    let nums = [1, 2, 3, 4, 5];

    // τ (tau) - Transform each element
    let doubled = nums|τ{_ * 2};
    print("Doubled: " + str(doubled));

    // φ (phi) - Filter elements
    let evens = nums|φ{_ % 2 == 0};
    print("Evens: " + str(evens));

    // Σ (sigma) - Sum all elements
    let total = nums|Σ;
    print("Sum: " + str(total));

    // Chain them together!
    let result = nums|τ{_ * 2}|φ{_ > 5}|Σ;
    print("Double, keep >5, sum: " + str(result));

    return 0;
}`,

  transform: `// Transform (τ) - Map operations
fn main() {
    let numbers = [1, 2, 3, 4, 5];

    // Double each number
    let doubled = numbers|τ{_ * 2};
    print("Doubled: " + str(doubled));

    // Square each number
    let squared = numbers|τ{_ * _};
    print("Squared: " + str(squared));

    // Add 10 to each
    let added = numbers|τ{_ + 10};
    print("Added 10: " + str(added));

    // Chain transforms
    let chained = numbers|τ{_ * 2}|τ{_ + 1};
    print("*2 then +1: " + str(chained));

    return 0;
}`,

  filter: `// Filter (φ) - Select elements
fn main() {
    let numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    // Keep even numbers
    let evens = numbers|φ{_ % 2 == 0};
    print("Evens: " + str(evens));

    // Keep numbers > 5
    let big = numbers|φ{_ > 5};
    print("Greater than 5: " + str(big));

    // Keep odd numbers < 8
    let filtered = numbers|φ{_ % 2 != 0}|φ{_ < 8};
    print("Odd and < 8: " + str(filtered));

    return 0;
}`,

  aggregate: `// Aggregate Morphemes: Σ, Π, μ, α, ω
fn main() {
    let nums = [1, 2, 3, 4, 5];

    // Σ (sigma) - Sum
    let sum = nums|Σ;
    print("Σ Sum: " + str(sum));

    // Π (pi) - Product
    let product = nums|Π;
    print("Π Product: " + str(product));

    // μ (mu) - Mean/Average
    let avg = nums|μ;
    print("μ Mean: " + str(avg));

    // α (alpha) - First element
    let first = nums|α;
    print("α First: " + str(first));

    // ω (omega) - Last element
    let last = nums|ω;
    print("ω Last: " + str(last));

    // λ (lambda) - Length
    let len = nums|λ;
    print("λ Length: " + str(len));

    return 0;
}`,

  // Evidentiality
  evidence: `// Evidentiality Markers Demonstration
//
// Evidence tracks data provenance at the type level:
//   ! (known)     - Computed locally, verified
//   ? (uncertain) - Might be absent (like Option)
//   ~ (reported)  - External data, untrusted
//   ‽ (paradox)   - Trust boundary crossing

fn main() {
    print("=== Evidence Chain: ~ -> ? -> ! ===");
    print("");

    // Stage 1: External data arrives as reported (~)
    print("Stage 1: External data (reported ~)");
    let raw_data~ = 42;  // Marked as external
    print("  raw_data~ received from API");

    // Stage 2: Validation promotes to uncertain (?)
    print("");
    print("Stage 2: After validation (uncertain ?)");
    let validated? = raw_data~;  // Still not fully trusted
    print("  validated? passed basic checks");

    // Stage 3: Computation produces known (!)
    print("");
    print("Stage 3: After verification (known !)");
    let result! = validated? * 2;  // Now trusted
    print("  result! = " + str(result!));

    print("");
    print("Evidence promotes pessimistically:");
    print("  ! + ~ = ~ (known polluted by reported)");

    return 0;
}`,

  validation: `// Data Validation with Evidence
fn main() {
    // Simulate external API data
    let api_response~ = 100;

    // Validation pipeline
    print("Raw data from API: " + str(api_response~));

    // Check if positive (promotes ~ to ?)
    let checked? = api_response~;
    if checked? > 0 {
        print("✓ Passed: value is positive");
    }

    // Verify bounds (promotes ? to !)
    let safe! = checked?;
    if safe! >= 0 && safe! <= 1000 {
        print("✓ Verified: value in safe range");
    }

    // Now we can use it with confidence
    let result! = safe! * 2;
    print("Final result!: " + str(result!));

    return 0;
}`,

  // Patterns
  structs: `// Structs and Implementations
struct Point {
    x: i64,
    y: i64,
}

impl Point {
    fn new(x: i64, y: i64) -> Point {
        Point { x: x, y: y }
    }

    fn origin() -> Point {
        Point::new(0, 0)
    }

    fn distance_from_origin(self) -> f64 {
        let x2 = (self.x * self.x) as f64;
        let y2 = (self.y * self.y) as f64;
        (x2 + y2).sqrt()
    }

    fn translate(self, dx: i64, dy: i64) -> Point {
        Point::new(self.x + dx, self.y + dy)
    }
}

fn main() {
    let p1 = Point::new(3, 4);
    print("Point: (" + str(p1.x) + ", " + str(p1.y) + ")");

    let dist = p1.distance_from_origin();
    print("Distance from origin: " + str(dist));

    let p2 = p1.translate(2, 3);
    print("Translated: (" + str(p2.x) + ", " + str(p2.y) + ")");

    return 0;
}`,

  matching: `// Pattern Matching
enum Status {
    Active,
    Pending(Str),
    Completed(i64),
    Failed(Str, i64),
}

fn describe_status(s: Status) -> Str {
    match s {
        Status::Active => "Currently active",
        Status::Pending(reason) => "Pending: " + reason,
        Status::Completed(code) => "Done with code " + str(code),
        Status::Failed(msg, code) => msg + " (error " + str(code) + ")",
    }
}

fn main() {
    let s1 = Status::Active;
    let s2 = Status::Pending("Awaiting approval");
    let s3 = Status::Completed(0);
    let s4 = Status::Failed("Connection lost", 503);

    print(describe_status(s1));
    print(describe_status(s2));
    print(describe_status(s3));
    print(describe_status(s4));

    // Pattern matching with guards
    let value = 42;
    let description = match value {
        0 => "zero",
        n if n < 0 => "negative",
        n if n > 100 => "large",
        _ => "normal positive",
    };
    print("42 is: " + description);

    return 0;
}`,

  pipeline: `// Full Data Processing Pipeline
fn main() {
    print("=== Sigil Data Pipeline Demo ===");
    print("");

    // Source data
    let data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    print("Source: " + str(data));
    print("");

    // Step 1: Transform (double each value)
    let step1 = data|τ{_ * 2};
    print("After τ{_ * 2}: " + str(step1));

    // Step 2: Filter (keep values > 10)
    let step2 = step1|φ{_ > 10};
    print("After φ{_ > 10}: " + str(step2));

    // Step 3: Sort (σ)
    let step3 = step2|σ;
    print("After σ (sort): " + str(step3));

    // Step 4: Aggregate
    let sum = step3|Σ;
    let count = step3|λ;
    let avg = step3|μ;
    print("");
    print("Σ Sum: " + str(sum));
    print("λ Count: " + str(count));
    print("μ Average: " + str(avg));

    // Or all in one pipeline!
    print("");
    print("=== Same as single pipeline ===");
    let result = data|τ{_ * 2}|φ{_ > 10}|σ|Σ;
    print("data|τ{_ * 2}|φ{_ > 10}|σ|Σ = " + str(result));

    return 0;
}`
};

// Configuration - can be overridden via environment or URL params
const API_BASE = new URLSearchParams(window.location.search).get('api') ||
                 (window.location.hostname === 'localhost' ? 'http://localhost:8080' : '');

// Sigil runtime interface
class SigilRuntime {
  constructor() {
    this.ready = false;
    this.useBackend = false;
    this.useWasm = false;
    this.wasmModule = null;
    this.apiBase = API_BASE;
  }

  async init() {
    // Try to connect to backend API first
    if (this.apiBase) {
      try {
        const response = await fetch(`${this.apiBase}/health`, {
          method: 'GET',
          timeout: 2000,
        });
        if (response.ok) {
          const health = await response.json();
          console.log('Connected to Sigil backend:', health);
          this.useBackend = true;
          this.ready = true;
          return true;
        }
      } catch (e) {
        console.log('Backend not available, trying WASM...', e.message);
      }
    }

    // Try to load WASM module
    try {
      await this.initWasm();
      console.log('WASM compiler loaded successfully');
      this.useWasm = true;
      this.ready = true;
      return true;
    } catch (e) {
      console.log('WASM not available, using mock mode:', e.message);
    }

    // Fall back to mock mode
    this.useBackend = false;
    this.useWasm = false;
    this.ready = true;
    return true;
  }

  async initWasm() {
    // Load the Rust WASM interpreter module
    const module = await import('./wasm/sigil_wasm_playground.js');
    // Initialize the WASM module
    await module.default();
    this.wasmModule = module;
  }

  wasmCheck(code) {
    try {
      const resultJson = this.wasmModule.check(code);
      const result = JSON.parse(resultJson);

      if (result.ok) {
        return {
          success: true,
          output: result.output || 'Syntax check passed.',
          errors: [],
          warnings: []
        };
      } else {
        return {
          success: false,
          errors: [result.error || 'Unknown error'],
          warnings: []
        };
      }
    } catch (e) {
      return {
        success: false,
        errors: [`WASM error: ${e.message}`],
        warnings: []
      };
    }
  }

  wasmRun(code) {
    try {
      const startTime = performance.now();
      const resultJson = this.wasmModule.execute(code);
      const endTime = performance.now();
      const result = JSON.parse(resultJson);

      if (result.ok) {
        // Build output from both stdout capture and return value
        let output = '';
        if (result.output) {
          output = result.output;
        }
        if (result.value && result.value !== '()' && result.value !== 'null') {
          if (output) output += '\n';
          output += `Result: ${result.value}`;
        }

        return {
          success: true,
          output: output || '(executed successfully)',
          errors: [],
          time: endTime - startTime
        };
      } else {
        return {
          success: false,
          output: '',
          errors: [result.error || 'Unknown error'],
          time: 0
        };
      }
    } catch (e) {
      return {
        success: false,
        output: '',
        errors: [`WASM error: ${e.message}`],
        time: 0
      };
    }
  }

  async run(code, backend = 'interpreter') {
    if (!this.ready) await this.init();

    if (this.useBackend) {
      return this.backendRun(code, backend);
    }
    if (this.useWasm) {
      return this.wasmRun(code);
    }
    return this.mockRun(code);
  }

  async check(code) {
    if (!this.ready) await this.init();

    if (this.useBackend) {
      return this.backendCheck(code);
    }
    if (this.useWasm) {
      return this.wasmCheck(code);
    }
    return this.mockCheck(code);
  }

  async getIR(code) {
    if (!this.ready) await this.init();

    if (this.useBackend) {
      return this.backendIR(code);
    }
    return this.mockIR(code);
  }

  async compile(code, filename = 'program') {
    if (!this.ready) await this.init();

    if (this.useBackend) {
      return this.backendCompile(code, filename);
    }
    return {
      success: false,
      error: 'WASM compilation requires the backend server. Start it with: cd playground/server && npm start',
    };
  }

  // Backend WASM compilation
  async backendCompile(code, filename = 'program') {
    try {
      const response = await fetch(`${this.apiBase}/compile`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, filename }),
      });

      if (response.ok) {
        const blob = await response.blob();
        return {
          success: true,
          blob: blob,
          size: blob.size,
          filename: filename + '.wasm',
          executionTime: parseFloat(response.headers.get('X-Execution-Time-Ms') || '0'),
        };
      }

      // Error response is JSON
      const error = await response.json();
      return {
        success: false,
        error: error.error?.message || 'Compilation failed',
      };
    } catch (e) {
      return {
        success: false,
        error: `Backend error: ${e.message}`,
      };
    }
  }

  // Backend API implementations
  async backendRun(code, backend = 'interpreter') {
    try {
      const response = await fetch(`${this.apiBase}/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code, backend }),
      });
      const result = await response.json();
      return {
        success: result.success,
        output: result.output || '',
        errors: result.error ? [result.error.message] : [],
        time: result.execution_time_ms || 0,
      };
    } catch (e) {
      return {
        success: false,
        output: '',
        errors: [`Backend error: ${e.message}`],
        time: 0,
      };
    }
  }

  async backendCheck(code) {
    try {
      const response = await fetch(`${this.apiBase}/check`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code }),
      });
      const result = await response.json();
      return {
        success: result.success,
        errors: result.errors?.map(e => e.message) || [],
        warnings: result.warnings?.map(w => w.message) || [],
      };
    } catch (e) {
      return {
        success: false,
        errors: [`Backend error: ${e.message}`],
        warnings: [],
      };
    }
  }

  async backendIR(code) {
    try {
      const response = await fetch(`${this.apiBase}/ir`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code }),
      });
      const result = await response.json();
      if (result.success) {
        return result.ir;
      }
      throw new Error(result.error?.message || 'Failed to generate IR');
    } catch (e) {
      throw e;
    }
  }

  // Mock implementations for demo
  mockRun(code) {
    // Simple mock that extracts print statements
    const lines = [];
    const printRegex = /print\(["']([^"']+)["']\)/g;
    const printExprRegex = /print\(([^)]+)\)/g;

    let match;
    while ((match = printRegex.exec(code)) !== null) {
      lines.push(match[1]);
    }

    // Handle print with expressions
    code.split('\n').forEach(line => {
      if (line.includes('print(') && !line.includes('print("')) {
        // Try to evaluate simple expressions
        const m = line.match(/print\((.+)\);?/);
        if (m) {
          lines.push(`[expression: ${m[1].trim()}]`);
        }
      }
    });

    if (lines.length === 0) {
      lines.push('(program executed successfully)');
    }

    // Check for obvious errors
    const errors = [];
    if (!code.includes('fn main')) {
      errors.push('Warning: No main function found');
    }

    return {
      success: errors.length === 0,
      output: lines.join('\n'),
      errors: errors,
      time: Math.random() * 10 + 1
    };
  }

  mockCheck(code) {
    const errors = [];
    const warnings = [];

    // Simple validation
    if (!code.includes('fn main')) {
      warnings.push('No main function found');
    }

    // Check for unclosed braces
    const openBraces = (code.match(/\{/g) || []).length;
    const closeBraces = (code.match(/\}/g) || []).length;
    if (openBraces !== closeBraces) {
      errors.push(`Mismatched braces: ${openBraces} open, ${closeBraces} close`);
    }

    // Check for unclosed parens
    const openParens = (code.match(/\(/g) || []).length;
    const closeParens = (code.match(/\)/g) || []).length;
    if (openParens !== closeParens) {
      errors.push(`Mismatched parentheses: ${openParens} open, ${closeParens} close`);
    }

    return {
      success: errors.length === 0,
      errors: errors,
      warnings: warnings
    };
  }

  mockIR(code) {
    // Generate a mock IR structure
    const functions = [];
    const fnRegex = /fn\s+(\w+)\s*\([^)]*\)/g;
    let match;
    while ((match = fnRegex.exec(code)) !== null) {
      functions.push({
        name: match[1],
        params: [],
        return_type: { kind: 'unit' }
      });
    }

    return {
      version: '1.0.0',
      source: 'playground',
      functions: functions
    };
  }
}

// Initialize the playground
async function init() {
  const runtime = new SigilRuntime();
  const output = document.getElementById('output');
  const status = document.getElementById('status');

  // Initialize the editor
  const editor = new EditorView({
    state: EditorState.create({
      doc: EXAMPLES.hello,
      extensions: [
        basicSetup,
        keymap.of([indentWithTab]),
        sigilLanguage,
        sigilTheme,
        sigilHighlight,
        EditorView.lineWrapping,
      ],
    }),
    parent: document.getElementById('editor'),
  });

  // Initialize runtime
  await runtime.init();
  const mode = runtime.useBackend ? 'Backend' : runtime.useWasm ? 'WASM' : 'Mock';
  const modeColor = runtime.useBackend ? '#22c55e' : runtime.useWasm ? '#14A088' : '#f59e0b';
  output.innerHTML = `<span style="color: ${modeColor};">✓ Sigil runtime ready (${mode} mode)</span>\n\nClick "Run" to execute your code.`;

  // Run button
  document.getElementById('run').addEventListener('click', async () => {
    const code = editor.state.doc.toString();
    output.innerHTML = '<span style="color: #a1a1aa;">Running...</span>';

    try {
      const result = await runtime.run(code);
      if (result.success) {
        output.innerHTML = result.output +
          `\n\n<span style="color: #22c55e;">✓ Completed in ${result.time.toFixed(2)}ms</span>`;
      } else {
        output.innerHTML = `<span style="color: #ef4444;">Error:</span>\n${result.errors.join('\n')}\n\n${result.output}`;
      }
    } catch (e) {
      output.innerHTML = `<span style="color: #ef4444;">Runtime error:</span>\n${e.message}`;
    }
  });

  // Check button
  document.getElementById('check').addEventListener('click', async () => {
    const code = editor.state.doc.toString();
    output.innerHTML = '<span style="color: #a1a1aa;">Type checking...</span>';

    try {
      const result = await runtime.check(code);
      if (result.success) {
        output.innerHTML = '<span style="color: #22c55e;">✓ Type check passed - no errors</span>';
        if (result.warnings.length > 0) {
          output.innerHTML += `\n\n<span style="color: #f59e0b;">Warnings:</span>\n${result.warnings.join('\n')}`;
        }
      } else {
        output.innerHTML = `<span style="color: #ef4444;">Type errors:</span>\n${result.errors.join('\n')}`;
      }
    } catch (e) {
      output.innerHTML = `<span style="color: #ef4444;">Check failed:</span>\n${e.message}`;
    }
  });

  // IR button
  document.getElementById('ir').addEventListener('click', async () => {
    const code = editor.state.doc.toString();

    try {
      const ir = await runtime.getIR(code);
      output.innerHTML = `<span style="color: #a1a1aa;">AI IR (JSON):</span>\n\n${JSON.stringify(ir, null, 2)}`;
    } catch (e) {
      output.innerHTML = `<span style="color: #ef4444;">IR generation failed:</span>\n${e.message}`;
    }
  });

  // Download source button
  document.getElementById('download-source').addEventListener('click', () => {
    const code = editor.state.doc.toString();
    const blob = new Blob([code], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'program.sigil';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    output.innerHTML = '<span style="color: #4CAF50;">✓ Downloaded program.sigil</span>';
  });

  // Download WASM button
  document.getElementById('download-wasm').addEventListener('click', async () => {
    const code = editor.state.doc.toString();
    output.innerHTML = '<span style="color: #a0a0aa;">Compiling to WebAssembly...</span>';

    try {
      const result = await runtime.compile(code, 'program');

      if (result.success) {
        // Download the WASM blob
        const url = URL.createObjectURL(result.blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = result.filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        output.innerHTML = `<span style="color: #4CAF50;">✓ Downloaded ${result.filename}</span>\n` +
          `<span style="color: #707078;">Size: ${(result.size / 1024).toFixed(2)} KB</span>\n` +
          `<span style="color: #707078;">Compiled in ${result.executionTime.toFixed(2)}ms</span>`;
      } else {
        output.innerHTML = `<span style="color: #F44336;">Compilation failed:</span>\n${result.error}`;
      }
    } catch (e) {
      output.innerHTML = `<span style="color: #F44336;">Error:</span>\n${e.message}`;
    }
  });

  // Clear output
  document.getElementById('clear-output').addEventListener('click', () => {
    output.innerHTML = '';
  });

  // Examples dropdown
  document.getElementById('examples').addEventListener('change', (e) => {
    const example = EXAMPLES[e.target.value];
    if (example) {
      editor.dispatch({
        changes: { from: 0, to: editor.state.doc.length, insert: example }
      });
      output.innerHTML = `<span style="color: #a1a1aa;">Loaded example: ${e.target.value}</span>`;
    }
    e.target.value = '';
  });

  // Share button
  document.getElementById('share').addEventListener('click', () => {
    const code = editor.state.doc.toString();
    const encoded = btoa(encodeURIComponent(code));
    const url = `${window.location.origin}${window.location.pathname}#code=${encoded}`;

    navigator.clipboard.writeText(url).then(() => {
      output.innerHTML = '<span style="color: #22c55e;">✓ Share URL copied to clipboard!</span>';
    }).catch(() => {
      output.innerHTML = `Share URL:\n${url}`;
    });
  });

  // Load code from URL hash
  const hash = window.location.hash;
  if (hash.startsWith('#code=')) {
    try {
      const encoded = hash.slice(6);
      const code = decodeURIComponent(atob(encoded));
      editor.dispatch({
        changes: { from: 0, to: editor.state.doc.length, insert: code }
      });
      output.innerHTML = '<span style="color: #a1a1aa;">Code loaded from shared URL</span>';
    } catch (e) {
      console.error('Failed to decode shared code:', e);
    }
  }

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
      e.preventDefault();
      document.getElementById('run').click();
    }
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'Enter') {
      e.preventDefault();
      document.getElementById('check').click();
    }
  });

  // Update status
  status.textContent = 'Ready';
}

// Start the playground
init().catch(console.error);
