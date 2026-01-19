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
rite main() → i64 {
    // Immutable by default
    ≔ x = 42;
    ≔ name = "Sigil";
    ≔ pi = 3.14159;
    ≔ flag = yea;

    // Mutable variables
    vary counter = 0;
    counter = counter + 1;

    // Explicit types
    ≔ age: i64 = 25;
    ≔ price: f64 = 19.99;

    println("x = " ++ x.to_string());
    println("name = " ++ name);
    println("counter = " ++ counter.to_string());

    0
}`,

  functions: `// Functions in Sigil
rite add(a: i64, b: i64) → i64 {
    a + b  // Implicit return
}

rite greet(name: String) {
    println("Hello, " ++ name ++ "!");
}

rite factorial(n: i64) → i64 {
    ⎇ n <= 1 { 1 }
    ⎉ { n * factorial(n - 1) }
}

rite main() → i64 {
    ≔ sum = add(10, 20);
    println("10 + 20 = " ++ sum.to_string());

    greet("World");

    ≔ fact5 = factorial(5);
    println("5! = " ++ fact5.to_string());

    0
}`,

  // Morphemes
  pipes: `// Morpheme Pipelines
rite main() → i64 {
    ≔ nums = [1, 2, 3, 4, 5];

    // τ (tau) - Transform each element
    ≔ doubled = nums |τ{. * 2};
    println("Doubled: " ++ doubled.to_string());

    // φ (phi) - Filter elements
    ≔ evens = nums |φ{. % 2 == 0};
    println("Evens: " ++ evens.to_string());

    // Σ (sigma) - Sum all elements
    ≔ total = nums |Σ;
    println("Sum: " ++ total.to_string());

    // Chain them together!
    ≔ result = nums |τ{. * 2} |φ{. > 5} |Σ;
    println("Double, keep >5, sum: " ++ result.to_string());

    0
}`,

  transform: `// Transform (τ) - Map operations
rite main() → i64 {
    ≔ numbers = [1, 2, 3, 4, 5];

    // Double each number
    ≔ doubled = numbers |τ{. * 2};
    println("Doubled: " ++ doubled.to_string());

    // Square each number
    ≔ squared = numbers |τ{. * .};
    println("Squared: " ++ squared.to_string());

    // Add 10 to each
    ≔ added = numbers |τ{. + 10};
    println("Added 10: " ++ added.to_string());

    // Chain transforms
    ≔ chained = numbers |τ{. * 2} |τ{. + 1};
    println("*2 then +1: " ++ chained.to_string());

    0
}`,

  filter: `// Filter (φ) - Select elements
rite main() → i64 {
    ≔ numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    // Keep even numbers
    ≔ evens = numbers |φ{. % 2 == 0};
    println("Evens: " ++ evens.to_string());

    // Keep numbers > 5
    ≔ big = numbers |φ{. > 5};
    println("Greater than 5: " ++ big.to_string());

    // Keep odd numbers < 8
    ≔ filtered = numbers |φ{. % 2 != 0} |φ{. < 8};
    println("Odd and < 8: " ++ filtered.to_string());

    0
}`,

  aggregate: `// Aggregate Morphemes: Σ, Π, μ, α, ω
rite main() → i64 {
    ≔ nums = [1, 2, 3, 4, 5];

    // Σ (sigma) - Sum
    ≔ sum = nums |Σ;
    println("Σ Sum: " ++ sum.to_string());

    // Π (pi) - Product
    ≔ product = nums |Π;
    println("Π Product: " ++ product.to_string());

    // μ (mu) - Mean/Average
    ≔ avg = nums |μ;
    println("μ Mean: " ++ avg.to_string());

    // α (alpha) - First element
    ≔ first = nums |α;
    println("α First: " ++ first.to_string());

    // ω (omega) - Last element
    ≔ last = nums |ω;
    println("ω Last: " ++ last.to_string());

    // λ (lambda) - Length
    ≔ len = nums |λ;
    println("λ Length: " ++ len.to_string());

    0
}`,

  // Evidentiality
  evidence: `// Evidentiality Markers Demonstration
//
// Evidence tracks data provenance at the type level:
//   ! (known)     - Computed locally, verified
//   ? (uncertain) - Might be absent (like Option)
//   ~ (reported)  - External data, untrusted
//   ‽ (paradox)   - Trust boundary crossing

rite main() → i64 {
    println("=== Evidence Chain: ~ -> ? -> ! ===");
    println("");

    // Stage 1: External data arrives as reported (~)
    println("Stage 1: External data (reported ~)");
    ≔ raw_data~ = 42;  // Marked as external
    println("  raw_data~ received from API");

    // Stage 2: Validation promotes to uncertain (?)
    println("");
    println("Stage 2: After validation (uncertain ?)");
    ≔ validated? = raw_data~;  // Still not fully trusted
    println("  validated? passed basic checks");

    // Stage 3: Computation produces known (!)
    println("");
    println("Stage 3: After verification (known !)");
    ≔ result! = validated? * 2;  // Now trusted
    println("  result! = " ++ result!.to_string());

    println("");
    println("Evidence promotes pessimistically:");
    println("  ! + ~ = ~ (known polluted by reported)");

    0
}`,

  validation: `// Data Validation with Evidence
rite main() → i64 {
    // Simulate external API data
    ≔ api_response~ = 100;

    // Validation pipeline
    println("Raw data from API: " ++ api_response~.to_string());

    // Check if positive (promotes ~ to ?)
    ≔ checked? = api_response~;
    ⎇ checked? > 0 {
        println("✓ Passed: value is positive");
    }

    // Verify bounds (promotes ? to !)
    ≔ safe! = checked?;
    ⎇ safe! >= 0 ∧ safe! <= 1000 {
        println("✓ Verified: value in safe range");
    }

    // Now we can use it with confidence
    ≔ result! = safe! * 2;
    println("Final result!: " ++ result!.to_string());

    0
}`,

  // Patterns
  structs: `// Sigils (Structs) and Implementations
sigil Point {
    x: i64,
    y: i64,
}

impl Point {
    rite new(x: i64, y: i64) → Point {
        Point { x: x, y: y }
    }

    rite origin() → Point {
        Point·new(0, 0)
    }

    rite distance_from_origin(self) → f64 {
        ≔ x2 = (self.x * self.x) as f64;
        ≔ y2 = (self.y * self.y) as f64;
        (x2 + y2).sqrt()
    }

    rite translate(self, dx: i64, dy: i64) → Point {
        Point·new(self.x + dx, self.y + dy)
    }
}

rite main() → i64 {
    ≔ p1 = Point·new(3, 4);
    println("Point: (" ++ p1.x.to_string() ++ ", " ++ p1.y.to_string() ++ ")");

    ≔ dist = p1.distance_from_origin();
    println("Distance from origin: " ++ dist.to_string());

    ≔ p2 = p1.translate(2, 3);
    println("Translated: (" ++ p2.x.to_string() ++ ", " ++ p2.y.to_string() ++ ")");

    0
}`,

  matching: `// Pattern Matching
enum Status {
    Active,
    Pending(String),
    Completed(i64),
    Failed(String, i64),
}

rite describe_status(s: Status) → String {
    ⌥ s {
        Status·Active => "Currently active",
        Status·Pending(reason) => "Pending: " ++ reason,
        Status·Completed(code) => "Done with code " ++ code.to_string(),
        Status·Failed(msg, code) => msg ++ " (error " ++ code.to_string() ++ ")",
    }
}

rite main() → i64 {
    ≔ s1 = Status·Active;
    ≔ s2 = Status·Pending("Awaiting approval");
    ≔ s3 = Status·Completed(0);
    ≔ s4 = Status·Failed("Connection lost", 503);

    println(describe_status(s1));
    println(describe_status(s2));
    println(describe_status(s3));
    println(describe_status(s4));

    // Pattern matching with guards
    ≔ value = 42;
    ≔ description = ⌥ value {
        0 => "zero",
        n ⎇ n < 0 => "negative",
        n ⎇ n > 100 => "large",
        _ => "normal positive",
    };
    println("42 is: " ++ description);

    0
}`,

  pipeline: `// Full Data Processing Pipeline
rite main() → i64 {
    println("=== Sigil Data Pipeline Demo ===");
    println("");

    // Source data
    ≔ data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    println("Source: " ++ data.to_string());
    println("");

    // Step 1: Transform (double each value)
    ≔ step1 = data |τ{. * 2};
    println("After τ{. * 2}: " ++ step1.to_string());

    // Step 2: Filter (keep values > 10)
    ≔ step2 = step1 |φ{. > 10};
    println("After φ{. > 10}: " ++ step2.to_string());

    // Step 3: Sort (σ)
    ≔ step3 = step2 |σ;
    println("After σ (sort): " ++ step3.to_string());

    // Step 4: Aggregate
    ≔ sum = step3 |Σ;
    ≔ count = step3 |λ;
    ≔ avg = step3 |μ;
    println("");
    println("Σ Sum: " ++ sum.to_string());
    println("λ Count: " ++ count.to_string());
    println("μ Average: " ++ avg.to_string());

    // Or all in one pipeline!
    println("");
    println("=== Same as single pipeline ===");
    ≔ result = data |τ{. * 2} |φ{. > 10} |σ |Σ;
    println("data|τ{. * 2}|φ{. > 10}|σ|Σ = " ++ result.to_string());

    0
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
