// Sigil Playground Worker
// Runs in a Web Worker inside a sandboxed iframe.
// Loads the Sigil compiler WASM module and executes user code.
//
// Messages:
//   { type: 'init' }                     -> Load WASM module
//   { type: 'run', code: '...' }         -> Execute code, return output
//   { type: 'check', code: '...' }       -> Check code for errors
//
// Responses:
//   { type: 'ready' }                    -> WASM loaded successfully
//   { type: 'error', message: '...' }    -> Initialization error
//   { type: 'result', data: {...} }      -> Execution/check result

let wasmModule = null;

self.onmessage = async function(e) {
    const { type, code, wasmUrl } = e.data;

    switch (type) {
        case 'init':
            try {
                // Import the wasm-bindgen generated JS glue
                const glueUrl = (wasmUrl || '/wasm/compiler') + '/sigil_parser.js';
                importScripts(glueUrl);

                // Initialize the WASM module
                const wasmBinaryUrl = (wasmUrl || '/wasm/compiler') + '/sigil_parser_bg.wasm';
                const response = await fetch(wasmBinaryUrl);
                const bytes = await response.arrayBuffer();

                // Initialize wasm-bindgen module
                await wasm_bindgen(bytes);
                wasmModule = wasm_bindgen;

                self.postMessage({ type: 'ready' });
            } catch (err) {
                self.postMessage({
                    type: 'error',
                    message: 'Failed to load Sigil compiler: ' + err.message
                });
            }
            break;

        case 'run':
            if (!wasmModule) {
                self.postMessage({
                    type: 'error',
                    message: 'Compiler not loaded. Send init first.'
                });
                return;
            }
            try {
                const resultJson = wasmModule.playground_run(code);
                const result = JSON.parse(resultJson);
                self.postMessage({ type: 'result', action: 'run', data: result });
            } catch (err) {
                self.postMessage({
                    type: 'result',
                    action: 'run',
                    data: {
                        output: [],
                        errors: ['Compiler crashed: ' + err.message],
                        elapsed_ms: 0
                    }
                });
            }
            break;

        case 'check':
            if (!wasmModule) {
                self.postMessage({
                    type: 'error',
                    message: 'Compiler not loaded. Send init first.'
                });
                return;
            }
            try {
                const resultJson = wasmModule.playground_check(code);
                const result = JSON.parse(resultJson);
                self.postMessage({ type: 'result', action: 'check', data: result });
            } catch (err) {
                self.postMessage({
                    type: 'result',
                    action: 'check',
                    data: {
                        diagnostics: [{ severity: 'error', message: 'Compiler crashed: ' + err.message }],
                        elapsed_ms: 0
                    }
                });
            }
            break;

        default:
            self.postMessage({
                type: 'error',
                message: 'Unknown message type: ' + type
            });
    }
};
