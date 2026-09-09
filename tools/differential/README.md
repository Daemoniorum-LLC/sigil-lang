# Differential sweep: interpreter vs WASM

Two backends, one language. The interpreter is the oracle: whatever it computes
is what the WASM module has to compute too.

This exists because the failures in this area do not look like failures. A
backend stub that hands the receiver back compiles, validates, and reports
success — `xs·map(f)` returned `xs` for the entire life of the WASM backend,
and the only symptom was a list with the wrong things in it. Nothing in the
test suite compared the two backends, so nothing said so.

## Known disagreements

Two probes fail on purpose. They are findings, recorded rather than fixed, and
the sweep's job is to keep them visible:

- **`division/t_float_divide`** — `10.0 / 4.0` is `2.5` in the interpreter and
  `1` in WASM. The WASM value model is uniformly i64 and has no float division.
- **`option/t_none_to_string`** — `(None)·to_string()` is `"None"` in the
  interpreter and `""` in WASM. React renders `{null}` as nothing, and
  generated code emits `(x)·to_string()` wherever React had one, so the WASM
  answer is what the migration needs and the interpreter's is what a Sigil
  programmer would expect. Reconciling them is a language decision, not a bug
  fix.

`cases/option.sigil` also lists, in its header, what the interpreter REFUSES to
evaluate at all: `==` is strictly typed there and untyped in WASM, so
`∅ == None`, `false == None` and `0 == None` are runtime type errors under one
backend and comparisons under the other. A migrated component is made of
`x == None` where `x` is `Any`.

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
