// Run one case file under both backends and diff the results.
//
// The interpreter is the oracle. `main` prints one line per probe, in
// declaration order; the WASM module exports the same probes as functions. A
// probe that returns a string is compared as text, read back out of the
// module's own memory.
import { execFileSync } from 'child_process';
import fs from 'fs';
import path from 'path';

const caseFile = process.argv[2];
const SIGIL = process.env.SIGIL || '../../parser/target/release/sigil';
const RUNTIME = process.env.RUNTIME || '../../../qliphoth/runtime/sigil_runtime.js';
const name = path.basename(caseFile, '.sigil');

// Probe names, in the order `main` prints them.
const src = fs.readFileSync(caseFile, 'utf8');
const probes = [...src.matchAll(/^☉ rite (t_\w+)\(\)/gm)].map((m) => m[1]);
// A probe whose declared return type is a String is compared as text.
const isText = new Map(
    [...src.matchAll(/^☉ rite (t_\w+)\(\)\s*->\s*(\w+)/gm)].map((m) => [m[1], m[2] === 'String']),
);

function fail(msg) {
    console.log(`${name}: ${msg}`);
    console.log(`${name}: total=${probes.length} agree=0`);
    process.exit(1);
}

let expected;
try {
    expected = execFileSync(SIGIL, ['run', caseFile], { encoding: 'utf8' })
        .split('\n')
        .map((l) => l.trim())
        .filter((l) => l.length > 0);
} catch (e) {
    fail(`interpreter failed: ${(e.stdout || '') + (e.stderr || e.message)}`.trim());
}

if (expected.length !== probes.length) {
    fail(`interpreter printed ${expected.length} line(s) for ${probes.length} probe(s) — main and the probe list disagree`);
}

const wasmPath = caseFile.replace(/\.sigil$/, '.wasm');
try {
    execFileSync(SIGIL, ['wasm', caseFile], { encoding: 'utf8', stdio: 'pipe' });
} catch (e) {
    fail(`wasm compile failed: ${(e.stdout || '') + (e.stderr || e.message)}`.trim());
}

const rt = await import(path.resolve(RUNTIME));
const inst = rt.instantiateWasm(fs.readFileSync(wasmPath));
fs.unlinkSync(wasmPath);

let agree = 0;
const rows = [];
for (let i = 0; i < probes.length; i++) {
    const probe = probes[i];
    const want = expected[i];
    let got;
    try {
        const raw = inst.exports[probe]();
        got = isText.get(probe) ? rt.readSigilString(Number(raw)) : String(Number(raw));
    } catch (e) {
        got = `trap: ${e.message}`;
    }
    const ok = got === want;
    if (ok) agree++;
    rows.push({ ok, probe, got, want });
}

for (const r of rows) {
    if (r.ok) continue;
    console.log(`  FAIL ${name}/${r.probe.padEnd(24)} wasm=${JSON.stringify(r.got)}  interpreter=${JSON.stringify(r.want)}`);
}
console.log(`${name}: total=${probes.length} agree=${agree}`);
process.exit(agree === probes.length ? 0 : 1);
