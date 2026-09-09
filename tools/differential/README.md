# Differential sweep: interpreter vs WASM

Two backends, one language. The interpreter is the oracle: whatever it computes
is what the WASM module has to compute too.

This exists because the failures in this area do not look like failures. A
backend stub that hands the receiver back compiles, validates, and reports
success — `xs·map(f)` returned `xs` for the entire life of the WASM backend,
and the only symptom was a list with the wrong things in it. Nothing in the
test suite compared the two backends, so nothing said so.

## Running

    ./run.sh              # every case
    ./run.sh arrays       # one case file

Each case is a `.sigil` file in `cases/` exporting `☉ rite t_<name>() -> …`
probes and a `main` that prints each one, in the same order. The runner runs
`main` under the interpreter to get the expected values, compiles the same file
to WASM, calls each export, and diffs.

A probe must return something printable as a single line — an integer, or a
string. Print through `·to_string()` so the interpreter's output and the WASM
value are comparable.

## The import contract

    node imports.mjs

`--list-imports` states each host function's signature. A runtime can supply
every name and still be wrong: WebAssembly converts the RESULT at the call, so
a function declared `-> i32` that returns a JavaScript BigInt throws "Cannot
convert a BigInt value to a number" — and only when that exact path runs. This
calls every declared import through a module with that exact signature and
reports the ones the boundary rejects. Twelve were wrong the first time it ran,
`string.contains`, `starts_with` and `ends_with` among them: every call to any
of the three had always trapped.
