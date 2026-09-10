# How `invoke tome·…` finds a module file

**Recorded:** 2026-09-10, closing #109.

A tome used to have to be one flat directory. `invoke tome·tui·grid·{Region}`
took the module name from the **second** segment alone, looked for `tui.sg`,
and failed with `undefined variable: Region` — which points nowhere near the
nesting.

Subdirectories were never meant to be unreachable: the workspace loader
(`main.rs`) has always read `router/types.sigil` as the module `router·types`
and `router/mod.sigil` as `router`. `run` and `run-dir` simply did not.

## The rule

A path's segments do not say where the module stops and the imported item
begins:

```sigil
invoke tome·analyze;              // analyze.sg
invoke tome·output·Output;        // output.sg,      item Output
invoke tome·tui·grid·{Region};    // tui/grid.sg,    item Region
invoke tome·tui·{layer_name};     // tui/mod.sg,     item layer_name
```

So `resolve_tome_module` drops the `tome` root, appends the final name, and
tries the **longest** module path first, giving up one segment at a time. For
each candidate length it looks for, in order:

```
<path>.sigil    <path>.sg    <path>/mod.sigil    <path>/mod.sg
```

The first that names a file wins, and the module is registered under the
`·`-joined form — `tui/grid.sg` is the module `tui·grid`, matching the
workspace loader.

Longest-first means `a/b.sg` beats `a.sg` for `tome·a·b·{…}` when both exist.
A flat `output.sg` is still found for `tome·output·{…}`, which is the spelling
every existing tome uses.

`scroll foo;` goes through the same resolver, so a module may be a file or a
directory with a `mod.sg` in it.

## What is still flat

`run-dir` lists and eagerly loads only the top level of the directory it is
given. Modules in subdirectories load on demand, when an `invoke` or `scroll`
reaches them — which is the Rust rule, and means an unreferenced file is not
executed. Making `run-dir`'s eager listing recurse belongs with the load-order
rework in #107.
