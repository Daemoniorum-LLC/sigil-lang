// The host import contract, checked by actually crossing the boundary.
//
// `--list-imports` states each host function's signature. A runtime can supply
// every name and still be wrong: WebAssembly converts the RESULT at the call,
// so a function declared `-> i32` that returns a JavaScript BigInt traps with
// "Cannot convert a BigInt value to a number" — and only when that exact path
// runs. Three separate defects of this shape reached a browser before anyone
// saw them.
//
// This calls every declared import through a module with that exact signature
// and reports the ones whose return value the boundary rejects. Argument
// values are zeros, so a function that dereferences them may fail for its own
// reasons; only conversion errors are reported.
import { execFileSync } from 'child_process';
import path from 'path';

const SIGIL = process.env.SIGIL || '../../parser/target/release/sigil';
const RUNTIME = process.env.RUNTIME || '../../../qliphoth/runtime/sigil_runtime.js';

const listed = execFileSync(SIGIL, ['wasm', '--list-imports'], { encoding: 'utf8' })
    .split('\n')
    .map((l) => l.trim())
    .filter(Boolean);

const sigs = [];
for (const line of listed) {
    const m = line.match(/^(\w+)\.(\w+)\(([^)]*)\)(?:\s*->\s*(\w+))?$/);
    if (!m) continue;
    const [, mod, name, argstr, ret] = m;
    const params = argstr.trim() ? argstr.split(',').map((s) => s.trim()) : [];
    sigs.push({ mod, name, params, ret: ret || null });
}

// A one-function module that imports `mod.name` with the declared signature and
// calls it. Hand-assembled: the point is that the signature is EXACTLY what the
// compiler declares, not what a helper library infers.
const TYPE = { i32: 0x7f, i64: 0x7e, f32: 0x7d, f64: 0x7c };
const CONST = { i32: 0x41, i64: 0x42, f32: 0x43, f64: 0x44 };

function leb(n) {
    const out = [];
    do {
        let b = n & 0x7f;
        n >>>= 7;
        if (n) b |= 0x80;
        out.push(b);
    } while (n);
    return out;
}
function section(id, bytes) {
    return [id, ...leb(bytes.length), ...bytes];
}
function vec(items) {
    return [...leb(items.length), ...items.flat()];
}
function str(s) {
    const b = [...Buffer.from(s, 'utf8')];
    return [...leb(b.length), ...b];
}

function moduleFor({ mod, name, params, ret }) {
    const pt = params.map((p) => TYPE[p]);
    const rt = ret ? [TYPE[ret]] : [];
    const types = vec([
        [0x60, ...leb(pt.length), ...pt, ...leb(rt.length), ...rt], // the import's type
        [0x60, 0x00, 0x00], // () -> () for the wrapper
    ]);
    const imports = vec([[...str(mod), ...str(name), 0x00, ...leb(0)]]);
    const funcs = vec([[0x01]]); // wrapper uses type 1
    // A memory, exported: the host reads its arguments out of linear memory, so
    // without one every string or collection function throws before it can
    // return anything for the boundary to reject.
    const mems = vec([[0x00, ...leb(2)]]);
    const exports = vec([
        [...str('probe'), 0x00, ...leb(1)],
        [...str('memory'), 0x02, ...leb(0)],
    ]);
    const body = [
        ...params.flatMap((p) =>
            p === 'f32' || p === 'f64' ? [CONST[p], 0, 0, 0, 0] : [CONST[p], 0x00],
        ),
        0x10, ...leb(0), // call the import
        ...(ret ? [0x1a] : []), // drop its result — the CONVERSION is what we test
        0x0b,
    ];
    const code = vec([[...leb(body.length + 1), 0x00, ...body]]);
    return Uint8Array.from([
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
        ...section(1, types),
        ...section(2, imports),
        ...section(3, funcs),
        ...section(5, mems),
        ...section(7, exports),
        ...section(10, code),
    ]);
}

const rt = await import(path.resolve(RUNTIME));
const hostImports = rt.createImports();

let checked = 0;
const bad = [];
const missing = [];
for (const sig of sigs) {
    const fn = hostImports[sig.mod] && hostImports[sig.mod][sig.name];
    if (typeof fn !== 'function') {
        missing.push(`${sig.mod}.${sig.name}`);
        continue;
    }
    if (sig.params.some((p) => !(p in TYPE)) || (sig.ret && !(sig.ret in TYPE))) continue;
    checked++;
    try {
        const instance = rt.instantiateWasm(moduleFor(sig));
        instance.exports.probe();
    } catch (e) {
        const msg = String(e.message || e);
        // Only the boundary's own type conversion. Anything else is the host
        // reacting to the zero arguments, which says nothing about the contract.
        if (/BigInt|[Cc]onvert/.test(msg)) {
            bad.push(`${sig.mod}.${sig.name}(${sig.params.join(', ')})${sig.ret ? ' -> ' + sig.ret : ''}  ${msg}`);
        }
    }
}

console.log(`import contract: ${checked} checked, ${bad.length} returning the wrong type`);
for (const b of bad) console.log(`  WRONG TYPE  ${b}`);
if (missing.length) {
    console.log(`\n${missing.length} declared import(s) the runtime does not supply:`);
    for (const m of missing) console.log(`  MISSING  ${m}`);
}
process.exit(bad.length || missing.length ? 1 : 0);
