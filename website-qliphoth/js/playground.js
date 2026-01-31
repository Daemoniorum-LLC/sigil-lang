/**
 * Sigil Playground Controller
 *
 * Provides interactive functionality for the Athame-powered Sigil playground:
 * - Syntax highlighting via JS port of Athame tokenizer.sigil
 * - Sandbox communication (postMessage to iframe + Web Worker)
 * - Editor features (line numbers, scroll sync, tab handling)
 * - UI controls (resize, theme toggle, example selector)
 */

'use strict';

// =============================================================================
// Sigil Tokenizer (faithful port of Athame tokenizer.sigil)
// =============================================================================

const KEYWORDS = new Set([
    // Legacy keywords (still supported as aliases)
    'fn', 'let', 'mut', 'if', 'else', 'match', 'return', 'for',
    'while', 'in', 'struct', 'enum', 'trait', 'impl', 'use', 'pub',
    'async', 'await', 'true', 'false', 'self', 'Self', 'super', 'loop',
    'break', 'continue', 'const', 'where',
    // Native Sigil keywords
    'rite', 'sigil', 'aspect', 'vary', 'yea', 'nay', 'each', 'of',
    'forever', 'this', 'This', 'above', 'invoke', 'scroll', 'tome',
]);

const MORPHEME_CHARS = new Set([
    '\u03C4', '\u03C6', '\u03C3', '\u03C1',  // τ φ σ ρ
    '\u03A3', '\u03A0', '\u03B1', '\u03C9',  // Σ Π α ω
    '\u03BC', '\u03BB',                        // μ λ
    '\u03A4', '\u03A6', '\u03A1', '\u0391',  // Τ Φ Ρ Α
    '\u03A9', '\u039C', '\u039B', '\u0398',  // Ω Μ Λ Θ
]);

const NATIVE_SYMBOLS = new Set([
    '\u2254',  // ≔
    '\u25C6',  // ◆
    '\u0394',  // Δ
    '\u2387',  // ⎇
    '\u2389',  // ⎉
    '\u2325',  // ⌥
    '\u27F3',  // ⟳
    '\u221E',  // ∞
    '\u2297',  // ⊗
    '\u21BB',  // ↻
    '\u293A',  // ⤺
    '\u2200',  // ∀
    '\u2208',  // ∈
    '\u22A4',  // ⊤
    '\u22A5',  // ⊥
    '\u2227',  // ∧
    '\u2228',  // ∨
    '\u00AC',  // ¬
    '\u22A2',  // ⊢
    '\u16C8',  // ᛈ
    '\u2609',  // ☉
    '\u2299',  // ⊙
    '\u220B',  // ∋
    '\u00B7',  // ·
    '\u2192',  // →
]);

const TWO_CHAR_OPS = new Set([
    '==', '!=', '<=', '>=', '&&', '||', '->', '=>',
    '+=', '-=', '*=', '/=', '::',
]);

function isWhitespace(c) { return c === ' ' || c === '\t' || c === '\r'; }
function isNewline(c) { return c === '\n'; }
function isDigit(c) { return c >= '0' && c <= '9'; }
function isHexDigit(c) { return isDigit(c) || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); }
function isAlpha(c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'); }
function isIdentStart(c) { return isAlpha(c) || c === '_'; }
function isIdentContinue(c) { return isIdentStart(c) || isDigit(c); }
function isUppercase(c) { return c >= 'A' && c <= 'Z'; }

function tokenize(source) {
    const tokens = [];
    const len = source.length;
    let pos = 0;
    let prevKind = 'unknown';

    while (pos < len) {
        const start = pos;
        const c = source[pos];

        if (isWhitespace(c)) {
            while (pos < len && isWhitespace(source[pos])) pos++;
            tokens.push({ kind: 'whitespace', start, end: pos });
            prevKind = 'whitespace';
        } else if (isNewline(c)) {
            pos++;
            tokens.push({ kind: 'newline', start, end: pos });
            prevKind = 'newline';
        } else if (isIdentStart(c)) {
            while (pos < len && isIdentContinue(source[pos])) pos++;
            const text = source.slice(start, pos);
            let kind;
            if (KEYWORDS.has(text)) {
                kind = 'keyword';
            } else if (text.length > 0 && isUppercase(text[0])) {
                kind = 'type';
            } else {
                kind = 'identifier';
            }
            tokens.push({ kind, start, end: pos });
            prevKind = kind;
        } else if (isDigit(c)) {
            if (c === '0' && pos + 1 < len && source[pos + 1] === 'x') {
                pos += 2;
                while (pos < len && isHexDigit(source[pos])) pos++;
            } else {
                while (pos < len && isDigit(source[pos])) pos++;
                if (pos < len && source[pos] === '.' && pos + 1 < len && isDigit(source[pos + 1])) {
                    pos++;
                    while (pos < len && isDigit(source[pos])) pos++;
                }
            }
            tokens.push({ kind: 'number', start, end: pos });
            prevKind = 'number';
        } else if (c === '"') {
            pos++;
            while (pos < len && source[pos] !== '"' && !isNewline(source[pos])) {
                if (source[pos] === '\\' && pos + 1 < len) {
                    pos += 2;
                } else {
                    pos++;
                }
            }
            if (pos < len && source[pos] === '"') pos++;
            tokens.push({ kind: 'string', start, end: pos });
            prevKind = 'string';
        } else if (c === '/' && pos + 1 < len && source[pos + 1] === '/') {
            // Line comment
            pos += 2;
            while (pos < len && !isNewline(source[pos])) pos++;
            tokens.push({ kind: 'comment', start, end: pos });
            prevKind = 'comment';
        } else if (c === '/' && pos + 1 < len && source[pos + 1] === '*') {
            // Block comment (with nesting)
            pos += 2;
            let depth = 1;
            while (pos < len && depth > 0) {
                if (pos + 1 < len && source[pos] === '*' && source[pos + 1] === '/') {
                    pos += 2;
                    depth--;
                } else if (pos + 1 < len && source[pos] === '/' && source[pos + 1] === '*') {
                    pos += 2;
                    depth++;
                } else {
                    pos++;
                }
            }
            tokens.push({ kind: 'comment', start, end: pos });
            prevKind = 'comment';
        } else if (MORPHEME_CHARS.has(c)) {
            pos++;
            tokens.push({ kind: 'morpheme', start, end: pos });
            prevKind = 'morpheme';
        } else if (NATIVE_SYMBOLS.has(c)) {
            pos++;
            tokens.push({ kind: 'native', start, end: pos });
            prevKind = 'native';
        } else if (c === '!' || c === '?' || c === '~') {
            // Evidentiality markers (context-sensitive)
            let kind;
            if (prevKind === 'identifier' || prevKind === 'type') {
                if (c === '!') kind = 'evidence-known';
                else if (c === '?') kind = 'evidence-uncertain';
                else kind = 'evidence-reported';
            } else {
                kind = 'operator';
            }
            pos++;
            tokens.push({ kind, start, end: pos });
            prevKind = kind;
        } else if (c === '\u203D') {
            // ‽ interrobang = evidence paradox
            pos++;
            tokens.push({ kind: 'evidence-paradox', start, end: pos });
            prevKind = 'evidence-paradox';
        } else {
            // Two-char operators
            const next = pos + 1 < len ? source[pos + 1] : '';
            const two = c + next;
            if (TWO_CHAR_OPS.has(two)) {
                pos += 2;
                tokens.push({ kind: 'operator', start, end: pos });
                prevKind = 'operator';
            } else if ('{})([],;:.'.includes(c)) {
                pos++;
                tokens.push({ kind: 'punctuation', start, end: pos });
                prevKind = 'punctuation';
            } else if ('+-*/%=<>&|^'.includes(c)) {
                pos++;
                tokens.push({ kind: 'operator', start, end: pos });
                prevKind = 'operator';
            } else {
                pos++;
                tokens.push({ kind: 'unknown', start, end: pos });
                prevKind = 'unknown';
            }
        }
    }

    return tokens;
}

// =============================================================================
// Syntax Highlighting
// =============================================================================

const TOKEN_CSS = {
    'keyword': 'ath-keyword',
    'type': 'ath-type',
    'string': 'ath-string',
    'number': 'ath-number',
    'comment': 'ath-comment',
    'operator': 'ath-operator',
    'morpheme': 'ath-morpheme',
    'native': 'ath-native',
    'evidence-known': 'ath-evidence-known',
    'evidence-uncertain': 'ath-evidence-uncertain',
    'evidence-reported': 'ath-evidence-reported',
    'evidence-paradox': 'ath-evidence-paradox',
};

function escapeHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function highlightCode(source) {
    const tokens = tokenize(source);
    let html = '';
    for (const token of tokens) {
        const text = source.slice(token.start, token.end);
        const cls = TOKEN_CSS[token.kind];
        const escaped = escapeHtml(text);
        if (cls) {
            html += `<span class="${cls}">${escaped}</span>`;
        } else {
            html += escaped;
        }
    }
    return html;
}

// =============================================================================
// Example Code Snippets
// =============================================================================

const EXAMPLES = {
    hello: `// Hello World in Sigil
\u2609 rite main() {
    print("Hello, Sigil!");
    print("A polysynthetic language for AI minds");

    \u2254 name = "World";
    print("Greetings, " + name + "!");
}`,

    counter: `// Counter with Loops
\u2609 rite main() {
    \u2254 vary count = 0;

    \u27F3 count < 5 {
        print("Count: " + to_string(count));
        count = count + 1;
    }

    print("Done! Final count: " + to_string(count));
}`,

    morphemes: `// Morpheme Operators (Pipeline Processing)
\u2609 rite main() {
    \u2254 numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

    // Filter evens, double them, sum
    \u2254 result = numbers
        |> \u03C6(|x| x % 2 == 0)    // filter evens
        |> \u03C4(|x| x * 2)          // transform: double
        |> \u03A3();                    // sum: 60

    print("Pipeline result: " + to_string(result));
}`,

    evidentiality: `// Evidentiality Types
// Track data provenance at the type level

sigil Sensor {
    temperature: f64,
    confidence: f64,
}

rite read_sensor() \u2192 Sensor! {
    // ! = Known evidence (direct measurement)
    Sensor { temperature: 23.5, confidence: 0.99 }!
}

rite estimate_weather() \u2192 String? {
    // ? = Uncertain evidence (inference)
    \u2254 sensor = read_sensor();
    \u2387 sensor.temperature > 20.0 {
        "Warm"?
    } \u2389 {
        "Cold"?
    }
}

\u2609 rite main() {
    \u2254 reading = read_sensor();
    \u2254 weather = estimate_weather();
    print("Temperature: " + to_string(reading.temperature));
    print("Weather: " + weather);
}`,

    async: `// Async/Await with Protocols
// Sigil has native HTTP and WebSocket support

async rite fetch_data(url: String) \u2192 String {
    \u2254 response = await http_get(url);
    response.body
}

\u2609 rite main() {
    print("Sigil supports async/await");
    print("with protocol-native HTTP and WebSocket");
    print("Built on Tokio for production performance");
}`,

    todo: `// Todo App Structure

sigil Todo {
    id: i64,
    text: String,
    done: bool,
}

rite create_todo(id: i64, text: String) \u2192 Todo {
    Todo { id: id, text: text, done: nay }
}

rite toggle_todo(todo: Todo) \u2192 Todo {
    Todo { id: todo.id, text: todo.text, done: \u00ACtodo.done }
}

\u2609 rite main() {
    \u2254 vary todos = Vec\u00B7new();
    todos.push(create_todo(1, "Learn Sigil"));
    todos.push(create_todo(2, "Build an agent"));
    todos.push(create_todo(3, "Deploy to production"));

    // Toggle first todo
    \u2254 first = todos[0];
    todos[0] = toggle_todo(first);

    \u2254 vary i = 0;
    \u27F3 i < todos.len() {
        \u2254 todo = todos[i];
        \u2254 status = \u2387 todo.done { "\u2713" } \u2389 { "\u25CB" };
        print(status + " " + todo.text);
        i = i + 1;
    }
}`,
};

// =============================================================================
// Editor Controller
// =============================================================================

class EditorController {
    constructor() {
        this.codeInput = null;
        this.highlightCode = null;
        this.highlightLayer = null;
        this.lineGutter = null;
        this._highlightScheduled = false;
    }

    init() {
        this.codeInput = document.getElementById('code-input');
        this.highlightCode = document.getElementById('highlight-code');
        this.highlightLayer = document.getElementById('highlight-layer');
        this.lineGutter = document.getElementById('line-gutter');

        if (!this.codeInput || !this.highlightCode) return;

        // Input handler: update highlighting on every change
        this.codeInput.addEventListener('input', () => this.scheduleHighlight());

        // Scroll sync: keep highlight layer aligned with textarea
        this.codeInput.addEventListener('scroll', () => this.syncScroll());

        // Tab key: insert spaces instead of moving focus
        this.codeInput.addEventListener('keydown', (e) => {
            if (e.key === 'Tab') {
                e.preventDefault();
                const start = this.codeInput.selectionStart;
                const end = this.codeInput.selectionEnd;
                const value = this.codeInput.value;
                this.codeInput.value = value.substring(0, start) + '    ' + value.substring(end);
                this.codeInput.selectionStart = this.codeInput.selectionEnd = start + 4;
                this.scheduleHighlight();
            }
        });

        // Set initial code
        this.setCode(EXAMPLES.hello);
    }

    setCode(code) {
        if (!this.codeInput) return;
        this.codeInput.value = code;
        this.updateHighlight();
    }

    getCode() {
        return this.codeInput ? this.codeInput.value : '';
    }

    scheduleHighlight() {
        if (this._highlightScheduled) return;
        this._highlightScheduled = true;
        requestAnimationFrame(() => {
            this._highlightScheduled = false;
            this.updateHighlight();
        });
    }

    updateHighlight() {
        if (!this.highlightCode || !this.codeInput) return;
        const source = this.codeInput.value;
        this.highlightCode.innerHTML = highlightCode(source) + '\n';
        this.updateLineNumbers(source);
        this.syncScroll();
    }

    updateLineNumbers(source) {
        if (!this.lineGutter) return;
        const lineCount = (source.match(/\n/g) || []).length + 1;
        let html = '';
        for (let i = 1; i <= lineCount; i++) {
            html += `<div class="line-number">${i}</div>`;
        }
        this.lineGutter.innerHTML = html;
    }

    syncScroll() {
        if (!this.highlightLayer || !this.codeInput) return;
        this.highlightLayer.scrollTop = this.codeInput.scrollTop;
        this.highlightLayer.scrollLeft = this.codeInput.scrollLeft;
        // Sync gutter scroll
        if (this.lineGutter) {
            this.lineGutter.scrollTop = this.codeInput.scrollTop;
        }
    }
}

// =============================================================================
// Sandbox Controller
// =============================================================================

class SandboxController {
    constructor() {
        this.iframe = null;
        this.ready = false;
        this.consoleOutput = null;
        this.statusDot = null;
        this.statusText = null;
        this._onResult = null;
    }

    init() {
        this.consoleOutput = document.getElementById('console-output');
        this.statusDot = document.getElementById('status-dot');
        this.statusText = document.getElementById('status-text');

        // Create sandbox iframe
        this.iframe = document.createElement('iframe');
        this.iframe.id = 'sandbox-frame';
        this.iframe.src = 'sandbox/sandbox.html';
        this.iframe.sandbox = 'allow-scripts';
        this.iframe.style.display = 'none';
        document.body.appendChild(this.iframe);

        // Listen for messages from sandbox
        window.addEventListener('message', (e) => this.handleMessage(e));
    }

    handleMessage(e) {
        const msg = e.data;
        if (!msg || !msg.type) return;

        switch (msg.type) {
            case 'sandbox-ready':
                this.ready = true;
                this.setStatus('idle', 'Ready');
                break;

            case 'sandbox-result':
                this.setStatus('success', `Done (${msg.data.elapsed_ms}ms)`);
                this.displayResult(msg.data);
                break;

            case 'sandbox-error':
                this.setStatus('error', 'Error');
                this.appendConsole(msg.message, 'error');
                break;
        }
    }

    run(code) {
        if (!this.iframe) return;

        this.clearConsole();

        if (!this.ready) {
            this.appendConsole('Compiler loading... please wait.', 'warning');
            return;
        }

        this.setStatus('compiling', 'Running...');
        this.iframe.contentWindow.postMessage({ type: 'run', code }, '*');
    }

    check(code) {
        if (!this.iframe || !this.ready) return;

        this.setStatus('compiling', 'Checking...');
        this.iframe.contentWindow.postMessage({ type: 'check', code }, '*');
    }

    displayResult(data) {
        if (data.output && data.output.length > 0) {
            for (const line of data.output) {
                this.appendConsole(line, 'info');
            }
        }
        if (data.errors && data.errors.length > 0) {
            this.setStatus('error', 'Error');
            for (const err of data.errors) {
                this.appendConsole(err, 'error');
            }
        }
        if (data.diagnostics) {
            for (const diag of data.diagnostics) {
                const cls = diag.severity === 'error' ? 'error' : 'warning';
                this.appendConsole(diag.message, cls);
            }
        }
    }

    setStatus(state, text) {
        if (this.statusDot) {
            this.statusDot.className = `status-dot status-${state}`;
        }
        if (this.statusText) {
            this.statusText.textContent = text;
        }
    }

    clearConsole() {
        if (this.consoleOutput) {
            this.consoleOutput.innerHTML = '';
        }
    }

    appendConsole(text, type) {
        if (!this.consoleOutput) return;
        const line = document.createElement('div');
        line.className = `console-line console-${type}`;
        line.textContent = text;
        this.consoleOutput.appendChild(line);
        this.consoleOutput.scrollTop = this.consoleOutput.scrollHeight;
    }
}

// =============================================================================
// Resize Handle
// =============================================================================

function initResize() {
    const handle = document.getElementById('resize-handle');
    const main = document.getElementById('playground-main');
    const editorPanel = document.getElementById('editor-panel');

    if (!handle || !main || !editorPanel) return;

    let isDragging = false;
    let startX = 0;
    let startWidth = 0;

    handle.addEventListener('mousedown', (e) => {
        isDragging = true;
        startX = e.clientX;
        startWidth = editorPanel.offsetWidth;
        main.classList.add('resizing');
        handle.classList.add('dragging');
    });

    document.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        const dx = e.clientX - startX;
        const newWidth = Math.max(300, Math.min(startWidth + dx, main.offsetWidth - 300));
        editorPanel.style.width = newWidth + 'px';
    });

    document.addEventListener('mouseup', () => {
        if (!isDragging) return;
        isDragging = false;
        main.classList.remove('resizing');
        handle.classList.remove('dragging');
    });
}

// =============================================================================
// Theme Toggle
// =============================================================================

function initThemeToggle() {
    const btn = document.getElementById('theme-toggle');
    if (!btn) return;

    // Restore saved theme
    const saved = localStorage.getItem('sigil-playground-theme');
    if (saved === 'light') {
        document.documentElement.setAttribute('data-theme', 'light');
    }

    btn.addEventListener('click', () => {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'light' ? 'dark' : 'light';
        if (next === 'light') {
            document.documentElement.setAttribute('data-theme', 'light');
        } else {
            document.documentElement.removeAttribute('data-theme');
        }
        localStorage.setItem('sigil-playground-theme', next);

        // Toggle sun/moon icons
        const sun = btn.querySelector('#sun-icon');
        const moon = btn.querySelector('#moon-icon');
        if (sun && moon) {
            sun.style.display = next === 'light' ? 'none' : '';
            moon.style.display = next === 'light' ? '' : 'none';
        }
    });
}

// =============================================================================
// Example Selector
// =============================================================================

function initExampleSelector(editor) {
    const select = document.getElementById('example-select');
    if (!select) return;

    select.addEventListener('change', () => {
        const example = EXAMPLES[select.value];
        if (example) {
            editor.setCode(example);
        }
    });
}

// =============================================================================
// Main Initialization
// =============================================================================

export function initPlayground() {
    const editor = new EditorController();
    const sandbox = new SandboxController();

    editor.init();
    sandbox.init();

    initResize();
    initThemeToggle();
    initExampleSelector(editor);

    // Wire up Run button
    const runBtn = document.getElementById('run-btn');
    if (runBtn) {
        runBtn.addEventListener('click', () => {
            sandbox.run(editor.getCode());
        });
    }

    // Wire up Clear Console button
    const clearBtn = document.getElementById('clear-console-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            sandbox.clearConsole();
        });
    }

    // Keyboard shortcut: Ctrl+Enter to run
    document.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            sandbox.run(editor.getCode());
        }
    });
}
