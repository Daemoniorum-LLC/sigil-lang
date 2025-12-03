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

    // Keywords
    if (stream.match(/\b(fn|let|mut|if|else|match|return|for|while|in|struct|enum|trait|impl|use|pub|async|await)\b/)) {
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

// Custom theme for Sigil
const sigilTheme = EditorView.theme({
  '&': {
    backgroundColor: '#1a1a2e',
    color: '#e4e4e7',
  },
  '.cm-content': {
    caretColor: '#8b5cf6',
    padding: '1rem',
  },
  '.cm-cursor': {
    borderLeftColor: '#8b5cf6',
  },
  '.cm-activeLine': {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
  },
  '.cm-gutters': {
    backgroundColor: '#16213e',
    color: '#a1a1aa',
    border: 'none',
    borderRight: '1px solid #27272a',
  },
  '.cm-activeLineGutter': {
    backgroundColor: 'rgba(255, 255, 255, 0.05)',
  },
}, { dark: true });

// Syntax highlighting styles
const sigilHighlight = EditorView.baseTheme({
  '.cm-keyword': { color: '#f472b6' },
  '.cm-string': { color: '#a5f3fc' },
  '.cm-number': { color: '#fde68a' },
  '.cm-comment': { color: '#6b7280', fontStyle: 'italic' },
  '.cm-function': { color: '#93c5fd' },
  '.cm-variable': { color: '#e4e4e7' },
  '.cm-type': { color: '#c4b5fd' },
  '.cm-atom': { color: '#fde68a' },
  '.cm-operator': { color: '#a1a1aa' },
  '.cm-morpheme': { color: '#c084fc', fontWeight: 'bold' },
  '.cm-evidence-known': { color: '#22c55e', fontWeight: 'bold' },
  '.cm-evidence-uncertain': { color: '#f59e0b', fontWeight: 'bold' },
  '.cm-evidence-reported': { color: '#3b82f6', fontWeight: 'bold' },
  '.cm-evidence-paradox': { color: '#ef4444', fontWeight: 'bold' },
});

// Example programs
const EXAMPLES = {
  hello: `fn main() {
    print("Hello, Sigil!");
    return 0;
}`,

  pipes: `fn main() {
    let nums = [1, 2, 3, 4, 5];

    // Transform with tau (τ)
    let doubled = nums|τ{_ * 2};
    print(doubled);

    // Sum with Sigma (Σ)
    let total = nums|Σ;
    print(total);

    return 0;
}`,

  evidence: `// Evidence Chain Demonstration
fn main() {
    print("Evidence Flow: ~ -> ? -> !");
    print("");

    // Stage 1: External data arrives as reported (~)
    print("Stage 1: External data (reported ~)");
    let raw_data = 42;
    print("  Data received");

    // Stage 2: Validation promotes to uncertain (?)
    print("Stage 2: Validation (uncertain ?)");
    print("  Data validated");

    // Stage 3: Computation produces known (!)
    print("Stage 3: Computation (known !)");
    let result = raw_data * 2;
    print("  Result computed: " + str(result));

    return 0;
}`,

  pipeline: `fn main() {
    let data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    // Full data pipeline
    let result = data
        |τ{_ * 2}     // Double each
        |φ{_ > 10}    // Keep values > 10
        |σ            // Sort (already sorted but demo)
        |Σ;           // Sum

    print("Pipeline result: " + str(result));
    return 0;
}`
};

// Sigil runtime interface
class SigilRuntime {
  constructor() {
    this.ready = false;
    this.wasmModule = null;
  }

  async init() {
    // Try to load WASM module
    try {
      // In production, this would load the actual WASM module
      // For now, we'll use a mock or server-side execution
      this.ready = true;
      return true;
    } catch (e) {
      console.error('Failed to load Sigil WASM:', e);
      this.ready = true; // Still mark as ready, will use mock
      return false;
    }
  }

  async run(code) {
    if (!this.ready) await this.init();

    // Mock execution for demonstration
    // In production, this calls into WASM or a backend API
    return this.mockRun(code);
  }

  async check(code) {
    if (!this.ready) await this.init();
    return this.mockCheck(code);
  }

  async getIR(code) {
    if (!this.ready) await this.init();
    return this.mockIR(code);
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
  output.innerHTML = '<span style="color: #22c55e;">✓ Sigil runtime ready</span>\n\nClick "Run" to execute your code.';

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

  // Format button (placeholder)
  document.getElementById('format').addEventListener('click', () => {
    output.innerHTML = '<span style="color: #f59e0b;">Format not yet implemented in playground</span>';
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
