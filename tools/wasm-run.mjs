import { readFile } from 'node:fs/promises';
const bytes = await readFile(process.argv[2]);
const mod = await WebAssembly.compile(bytes);
// Stub every import so the module can instantiate outside a browser.
const imports = {};
for (const { module: m, name, kind } of WebAssembly.Module.imports(mod)) {
  imports[m] ??= {};
  imports[m][name] = kind === 'function' ? (() => 0) : 0;
}
const inst = await WebAssembly.instantiate(mod, imports);
const fn = process.argv[3] ?? 'main';
console.log(`${fn}() =`, inst.exports[fn]());
