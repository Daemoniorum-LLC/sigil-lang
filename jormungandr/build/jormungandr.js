/**
 * Jormungandr - Sigil Compiler for the Browser
 *
 * This module provides a JavaScript interface to the Jormungandr WASM compiler,
 * enabling Sigil code compilation and execution in browsers and Node.js.
 *
 * @example
 * // Browser
 * const compiler = await Jormungandr.load('./jormungandr.wasm');
 * const result = await compiler.run('rite main() -> i64 { println("Hello"); 0 }');
 * console.log(result.output); // "Hello\n"
 * compiler.dispose();
 *
 * @example
 * // Node.js
 * const { Jormungandr } = require('./jormungandr.js');
 * const compiler = await Jormungandr.load('./jormungandr.wasm');
 * const result = await compiler.run('rite main() -> i64 { 42 }');
 * console.log(result.exitCode); // 42
 */

class Jormungandr {
    #instance = null;
    #memory = null;
    #encoder = new TextEncoder();
    #decoder = new TextDecoder();
    #outputBuffer = '';

    /**
     * Private constructor - use Jormungandr.load() instead
     * @param {WebAssembly.Instance} instance
     */
    constructor(instance) {
        this.#instance = instance;
        this.#refreshMemory();
    }

    /**
     * Load the Jormungandr WASM module
     *
     * @param {string} wasmPath - Path or URL to jormungandr.wasm
     * @returns {Promise<Jormungandr>} Loaded compiler instance
     *
     * @example
     * const compiler = await Jormungandr.load('./jormungandr.wasm');
     */
    static async load(wasmPath) {
        const self = { outputBuffer: '' };

        const imports = {
            env: {
                // Performance timer
                performance_now: () => {
                    if (typeof performance !== 'undefined') {
                        return performance.now();
                    }
                    // Node.js fallback
                    const [sec, nsec] = process.hrtime();
                    return sec * 1000 + nsec / 1000000;
                },
            },
            console: {
                // Log string (ptr, len) - captures output from Sigil println
                log_str: (ptr, len) => {
                    // This is handled internally by the WASM module
                    // Output is returned in the result JSON
                },
                // Log i64
                log_i64: (value) => {
                    self.outputBuffer += value.toString();
                },
                // Log f64
                log_f64: (value) => {
                    self.outputBuffer += value.toString();
                },
                // Print (general)
                print: (ptr, len) => {
                    // Handled internally
                },
                // Println
                println_str: (ptr, len) => {
                    // Handled internally
                },
                println_i64: (value) => {
                    self.outputBuffer += value.toString() + '\n';
                },
                println_f64: (value) => {
                    self.outputBuffer += value.toString() + '\n';
                },
            },
            // String operations (stubs - real impl in WASM)
            string: {
                concat: (ptr1, len1, ptr2, len2) => 0,
                length: (ptr) => 0,
                eq: (ptr1, len1, ptr2, len2) => 0,
                from_int: (value) => 0,
                from_float: (value) => 0,
            },
            // Math operations
            math: {
                sqrt: Math.sqrt,
                sin: Math.sin,
                cos: Math.cos,
                tan: Math.tan,
                pow: Math.pow,
                exp: Math.exp,
                log: Math.log,
                floor: Math.floor,
                ceil: Math.ceil,
                round: Math.round,
                abs: Math.abs,
                random: Math.random,
            },
            // Memory operations (stubs)
            memory: {
                alloc: (size) => 0,
                free: (ptr) => {},
                realloc: (ptr, oldSize, newSize) => 0,
            },
        };

        let bytes;
        if (typeof fetch !== 'undefined') {
            // Browser environment
            const response = await fetch(wasmPath);
            if (!response.ok) {
                throw new Error(`Failed to fetch WASM: ${response.status} ${response.statusText}`);
            }
            bytes = await response.arrayBuffer();
        } else {
            // Node.js environment
            const fs = require('fs');
            const path = require('path');
            const resolvedPath = path.resolve(wasmPath);
            bytes = fs.readFileSync(resolvedPath);
        }

        const { instance } = await WebAssembly.instantiate(bytes, imports);

        // Initialize the WASM heap if init function exists
        if (instance.exports.wasm_init) {
            // Initialize with 1MB heap starting at 64KB offset
            instance.exports.wasm_init(0x10000, 0x100000);
        }

        const compiler = new Jormungandr(instance);
        compiler.#outputBuffer = self.outputBuffer;
        return compiler;
    }

    /**
     * Compile and run Sigil source code
     *
     * @param {string} source - Sigil source code to compile and run
     * @returns {Promise<CompileResult>} Compilation result
     *
     * @typedef {Object} CompileResult
     * @property {boolean} ok - Whether compilation and execution succeeded
     * @property {string} [output] - Standard output from the program (if ok)
     * @property {number} [exitCode] - Exit code from main() (if ok)
     * @property {number} [durationMs] - Execution time in milliseconds (if ok)
     * @property {string} [error] - Error message (if not ok)
     *
     * @example
     * const result = await compiler.run('rite main() -> i64 { println("Hi"); 0 }');
     * if (result.ok) {
     *     console.log(result.output); // "Hi\n"
     * } else {
     *     console.error(result.error);
     * }
     */
    async run(source) {
        if (!this.#instance) {
            throw new Error('Jormungandr instance has been disposed');
        }

        const sourceBytes = this.#encoder.encode(source);
        const sourcePtr = this.#alloc(sourceBytes.length);

        if (sourcePtr === 0) {
            return {
                ok: false,
                error: 'Failed to allocate memory for source code',
            };
        }

        try {
            // Write source to WASM memory
            this.#refreshMemory();
            this.#memory.set(sourceBytes, sourcePtr);

            // Call WASM compile_and_run
            const resultPtr = this.#instance.exports.wasm_compile_and_run(
                sourcePtr,
                sourceBytes.length
            );

            // Read and parse result
            return this.#readResult(resultPtr);
        } finally {
            this.#free(sourcePtr, sourceBytes.length);
        }
    }

    /**
     * Check syntax without executing
     *
     * Performs parsing and type checking but does not run the code.
     * Useful for real-time syntax highlighting and error reporting in editors.
     *
     * @param {string} source - Sigil source code to check
     * @returns {Promise<Diagnostic[]>} Array of diagnostics (empty if valid)
     *
     * @typedef {Object} Diagnostic
     * @property {'error'|'warning'|'info'} severity - Diagnostic severity
     * @property {string} message - Human-readable message
     * @property {number} [line] - Line number (1-indexed)
     * @property {number} [column] - Column number (1-indexed)
     *
     * @example
     * const diagnostics = await compiler.check('rite main() -> i64 { "oops" }');
     * if (diagnostics.length > 0) {
     *     console.log(diagnostics[0].message); // Type error message
     * }
     */
    async check(source) {
        if (!this.#instance) {
            throw new Error('Jormungandr instance has been disposed');
        }

        // If wasm_check_syntax doesn't exist, fall back to run() and extract errors
        if (!this.#instance.exports.wasm_check_syntax) {
            const result = await this.run(source);
            if (result.ok) {
                return [];
            } else {
                return [{
                    severity: 'error',
                    message: result.error,
                    line: null,
                    column: null,
                }];
            }
        }

        const sourceBytes = this.#encoder.encode(source);
        const sourcePtr = this.#alloc(sourceBytes.length);

        if (sourcePtr === 0) {
            return [{
                severity: 'error',
                message: 'Failed to allocate memory',
                line: null,
                column: null,
            }];
        }

        try {
            this.#refreshMemory();
            this.#memory.set(sourceBytes, sourcePtr);

            const resultPtr = this.#instance.exports.wasm_check_syntax(
                sourcePtr,
                sourceBytes.length
            );

            const result = this.#readResult(resultPtr);

            // Parse diagnostics from result
            if (result.ok) {
                return [];
            } else {
                return [{
                    severity: 'error',
                    message: result.error,
                    line: null,
                    column: null,
                }];
            }
        } finally {
            this.#free(sourcePtr, sourceBytes.length);
        }
    }

    /**
     * Release WASM resources
     *
     * Call this when you're done with the compiler to free memory.
     * After calling dispose(), the compiler instance cannot be used.
     *
     * @example
     * const compiler = await Jormungandr.load('./jormungandr.wasm');
     * // ... use compiler ...
     * compiler.dispose();
     */
    dispose() {
        if (this.#instance && this.#instance.exports.wasm_reset) {
            this.#instance.exports.wasm_reset();
        }
        this.#instance = null;
        this.#memory = null;
    }

    /**
     * Check if the compiler instance is still valid
     * @returns {boolean}
     */
    get isDisposed() {
        return this.#instance === null;
    }

    // ========================================================================
    // Private Helpers
    // ========================================================================

    #alloc(size) {
        if (!this.#instance.exports.wasm_alloc) {
            throw new Error('WASM module missing wasm_alloc export');
        }
        return this.#instance.exports.wasm_alloc(size);
    }

    #free(ptr, size) {
        if (this.#instance.exports.wasm_free) {
            this.#instance.exports.wasm_free(ptr, size);
        }
    }

    #refreshMemory() {
        // Memory buffer may have grown due to WASM memory.grow()
        this.#memory = new Uint8Array(this.#instance.exports.memory.buffer);
    }

    #readString(ptr) {
        this.#refreshMemory();
        const view = new DataView(this.#instance.exports.memory.buffer);

        // Read length prefix (4 bytes, little-endian)
        const len = view.getInt32(ptr, true);

        if (len < 0 || len > 10_000_000) {
            throw new Error(`Invalid string length: ${len}`);
        }

        // Read string bytes
        const bytes = this.#memory.slice(ptr + 4, ptr + 4 + len);
        return this.#decoder.decode(bytes);
    }

    #readResult(ptr) {
        try {
            const jsonStr = this.#readString(ptr);
            const parsed = JSON.parse(jsonStr);

            if (parsed.ok) {
                return {
                    ok: true,
                    output: parsed.output || '',
                    exitCode: parsed.exit_code || 0,
                    durationMs: parsed.duration_ms || 0,
                };
            } else {
                return {
                    ok: false,
                    error: parsed.error || 'Unknown error',
                };
            }
        } catch (e) {
            return {
                ok: false,
                error: `Failed to parse result: ${e.message}`,
            };
        }
    }
}

// ============================================================================
// Exports
// ============================================================================

// CommonJS (Node.js)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Jormungandr };
}

// Browser global
if (typeof window !== 'undefined') {
    window.Jormungandr = Jormungandr;
}

// ES Modules
export { Jormungandr };
