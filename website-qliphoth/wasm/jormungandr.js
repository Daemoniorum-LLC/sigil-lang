/**
 * Jormungandr - Sigil Compiler for the Browser
 *
 * This module provides a JavaScript interface to the Jormungandr WASM compiler,
 * enabling Sigil code compilation and execution in browsers and Node.js.
 *
 * @example
 * const compiler = await Jormungandr.load('./jormungandr.wasm');
 * const result = await compiler.run('rite main() -> i64 { println("Hello"); 0 }');
 * console.log(result.output); // "Hello\n"
 * compiler.dispose();
 */

// Emscripten-generated factory function (inlined for single-file distribution)
var createJormungandrModule = (() => {
    var _scriptName = typeof document !== 'undefined' ? document.currentScript?.src : undefined;

    return async function(moduleArg = {}) {
        var Module = moduleArg;
        var ENVIRONMENT_IS_WEB = typeof window !== 'undefined';
        var ENVIRONMENT_IS_NODE = typeof process !== 'undefined' && process.versions?.node;

        var readAsync, readBinary;
        var scriptDirectory = "";

        if (ENVIRONMENT_IS_NODE) {
            var fs = require("fs");
            var path = require("path");
            scriptDirectory = __dirname + "/";
            readBinary = (filename) => fs.readFileSync(filename);
            readAsync = async (filename) => fs.readFileSync(filename);
        } else if (ENVIRONMENT_IS_WEB) {
            if (_scriptName) {
                scriptDirectory = _scriptName.substring(0, _scriptName.lastIndexOf('/') + 1);
            }
            readAsync = async (url) => {
                var response = await fetch(url);
                if (!response.ok) throw new Error(`${response.status}: ${response.url}`);
                return response.arrayBuffer();
            };
        }

        function locateFile(path) {
            if (Module["locateFile"]) return Module["locateFile"](path, scriptDirectory);
            return scriptDirectory + path;
        }

        // Memory views
        var HEAP8, HEAPU8, HEAP32;
        var wasmMemory;

        function updateMemoryViews() {
            var b = wasmMemory.buffer;
            HEAP8 = new Int8Array(b);
            HEAPU8 = new Uint8Array(b);
            HEAP32 = new Int32Array(b);
            Module.HEAP8 = HEAP8;
            Module.HEAPU8 = HEAPU8;
            Module.HEAP32 = HEAP32;
        }

        // Memory growth
        function growMemory(size) {
            var oldSize = wasmMemory.buffer.byteLength;
            var pages = Math.ceil((size - oldSize) / 65536);
            try {
                wasmMemory.grow(pages);
                updateMemoryViews();
                return 1;
            } catch (e) {
                return 0;
            }
        }

        var _emscripten_resize_heap = (requestedSize) => {
            var oldSize = HEAPU8.length;
            requestedSize >>>= 0;
            var maxSize = 2147483648;
            if (requestedSize > maxSize) return false;
            var newSize = Math.min(maxSize, Math.max(requestedSize, oldSize * 1.2));
            newSize = Math.ceil(newSize / 65536) * 65536;
            return growMemory(newSize);
        };

        // Instantiate WASM
        var wasmExports;
        var wasmBinaryFile = Module.wasmBinaryFile || locateFile("jormungandr.wasm");

        async function createWasm() {
            var imports = {
                env: { emscripten_resize_heap: _emscripten_resize_heap },
                wasi_snapshot_preview1: { emscripten_resize_heap: _emscripten_resize_heap }
            };

            var binary;
            if (ENVIRONMENT_IS_NODE) {
                binary = readBinary(wasmBinaryFile);
            } else {
                binary = await readAsync(wasmBinaryFile);
            }

            var result = await WebAssembly.instantiate(binary, imports);
            wasmExports = result.instance.exports;
            wasmMemory = wasmExports.memory;
            updateMemoryViews();

            // Export functions
            Module._wasm_init = wasmExports.wasm_init;
            Module._wasm_alloc = wasmExports.wasm_alloc;
            Module._wasm_free = wasmExports.wasm_free;
            Module._wasm_reset = wasmExports.wasm_reset;
            Module._wasm_compile_and_run = wasmExports.wasm_compile_and_run;
            Module._wasm_check_syntax = wasmExports.wasm_check_syntax;

            return wasmExports;
        }

        // UTF8 helpers
        function lengthBytesUTF8(str) {
            var len = 0;
            for (var i = 0; i < str.length; i++) {
                var c = str.charCodeAt(i);
                if (c <= 0x7F) len++;
                else if (c <= 0x7FF) len += 2;
                else if (c >= 0xD800 && c <= 0xDFFF) { len += 4; i++; }
                else len += 3;
            }
            return len;
        }

        function stringToUTF8(str, outPtr, maxBytes) {
            var i = 0, outIdx = outPtr;
            var endIdx = outPtr + maxBytes - 1;
            for (; i < str.length; i++) {
                var u = str.codePointAt(i);
                if (u <= 0x7F) {
                    if (outIdx >= endIdx) break;
                    HEAPU8[outIdx++] = u;
                } else if (u <= 0x7FF) {
                    if (outIdx + 1 >= endIdx) break;
                    HEAPU8[outIdx++] = 0xC0 | (u >> 6);
                    HEAPU8[outIdx++] = 0x80 | (u & 0x3F);
                } else if (u <= 0xFFFF) {
                    if (outIdx + 2 >= endIdx) break;
                    HEAPU8[outIdx++] = 0xE0 | (u >> 12);
                    HEAPU8[outIdx++] = 0x80 | ((u >> 6) & 0x3F);
                    HEAPU8[outIdx++] = 0x80 | (u & 0x3F);
                } else {
                    if (outIdx + 3 >= endIdx) break;
                    HEAPU8[outIdx++] = 0xF0 | (u >> 18);
                    HEAPU8[outIdx++] = 0x80 | ((u >> 12) & 0x3F);
                    HEAPU8[outIdx++] = 0x80 | ((u >> 6) & 0x3F);
                    HEAPU8[outIdx++] = 0x80 | (u & 0x3F);
                    i++;
                }
            }
            HEAPU8[outIdx] = 0;
            return outIdx - outPtr;
        }

        function UTF8ToString(ptr, maxBytes) {
            if (!ptr) return "";
            var endPtr = ptr;
            if (maxBytes !== undefined) {
                endPtr = ptr + maxBytes;
            } else {
                while (HEAPU8[endPtr]) endPtr++;
            }
            var decoder = new TextDecoder();
            return decoder.decode(HEAPU8.subarray(ptr, endPtr));
        }

        function getValue(ptr, type) {
            switch (type) {
                case 'i32': return HEAP32[ptr >> 2];
                default: return HEAP8[ptr];
            }
        }

        Module.stringToUTF8 = stringToUTF8;
        Module.UTF8ToString = UTF8ToString;
        Module.lengthBytesUTF8 = lengthBytesUTF8;
        Module.getValue = getValue;

        await createWasm();
        return Module;
    };
})();

/**
 * High-level Jormungandr compiler interface
 */
class Jormungandr {
    #module = null;

    constructor(module) {
        this.#module = module;
    }

    /**
     * Load the Jormungandr WASM module
     * @param {string} wasmPath - Path or URL to jormungandr.wasm
     * @returns {Promise<Jormungandr>} Loaded compiler instance
     */
    static async load(wasmPath) {
        const module = await createJormungandrModule({
            wasmBinaryFile: wasmPath
        });

        // Initialize heap (1MB starting at 64KB)
        if (module._wasm_init) {
            module._wasm_init(0x10000, 0x100000);
        }

        return new Jormungandr(module);
    }

    /**
     * Compile and run Sigil source code
     * @param {string} source - Sigil source code
     * @returns {Promise<{ok: boolean, output?: string, exitCode?: number, error?: string}>}
     */
    async run(source) {
        if (!this.#module) {
            throw new Error('Jormungandr instance has been disposed');
        }

        const m = this.#module;
        const sourceLen = m.lengthBytesUTF8(source);
        const sourcePtr = m._wasm_alloc(sourceLen + 1);

        if (sourcePtr === 0) {
            return { ok: false, error: 'Failed to allocate memory' };
        }

        try {
            m.stringToUTF8(source, sourcePtr, sourceLen + 1);
            const resultPtr = m._wasm_compile_and_run(sourcePtr, sourceLen);
            return this.#readResult(resultPtr);
        } finally {
            m._wasm_free(sourcePtr, sourceLen + 1);
        }
    }

    /**
     * Check syntax without executing
     * @param {string} source - Sigil source code
     * @returns {Promise<Array<{severity: string, message: string}>>}
     */
    async check(source) {
        if (!this.#module) {
            throw new Error('Jormungandr instance has been disposed');
        }

        if (!this.#module._wasm_check_syntax) {
            const result = await this.run(source);
            if (result.ok) return [];
            return [{ severity: 'error', message: result.error }];
        }

        const m = this.#module;
        const sourceLen = m.lengthBytesUTF8(source);
        const sourcePtr = m._wasm_alloc(sourceLen + 1);

        if (sourcePtr === 0) {
            return [{ severity: 'error', message: 'Failed to allocate memory' }];
        }

        try {
            m.stringToUTF8(source, sourcePtr, sourceLen + 1);
            const resultPtr = m._wasm_check_syntax(sourcePtr, sourceLen);
            const result = this.#readResult(resultPtr);
            if (result.ok) return [];
            return [{ severity: 'error', message: result.error }];
        } finally {
            m._wasm_free(sourcePtr, sourceLen + 1);
        }
    }

    /**
     * Release WASM resources
     */
    dispose() {
        if (this.#module && this.#module._wasm_reset) {
            this.#module._wasm_reset();
        }
        this.#module = null;
    }

    get isDisposed() {
        return this.#module === null;
    }

    #readResult(ptr) {
        const m = this.#module;
        const len = m.getValue(ptr, 'i32');

        if (len < 0 || len > 10_000_000) {
            return { ok: false, error: `Invalid result length: ${len}` };
        }

        const jsonStr = m.UTF8ToString(ptr + 4, len);

        try {
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
            return { ok: false, error: `Failed to parse result: ${e.message}` };
        }
    }
}

// Exports
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { Jormungandr };
}
if (typeof window !== 'undefined') {
    window.Jormungandr = Jormungandr;
}
