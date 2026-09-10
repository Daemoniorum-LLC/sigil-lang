// Instantiate a Qliphoth component with a minimal host that records VDOM calls,
// so we can see whether the compiled module actually drives the runtime.
import { readFile } from 'node:fs/promises';
const bytes = await readFile('hello.wasm');
const mod = await WebAssembly.compile(bytes);
const log = [];
const imports = {};
for (const { module: m, name, kind } of WebAssembly.Module.imports(mod)) {
  imports[m] ??= {};
  imports[m][name] = kind === 'function'
    ? ((...a) => { log.push(`${m}.${name}(${a.join(', ')})`); return 1; })
    : 0;
}
const inst = await WebAssembly.instantiate(mod, imports);
inst.exports.main();
console.log('host calls made by main():');
for (const l of log) console.log('  ' + l);
