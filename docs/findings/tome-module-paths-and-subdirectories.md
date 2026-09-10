# How `invoke tome·…` finds a module file

**Recorded:** 2026-09-10, closing #109; extended the same day for the
`scroll` half.

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

## `scroll` is relative; `tome·` is not

The two roots differ, and that is the whole reason they are tracked separately:

| | rooted at |
|---|---|
| `scroll grid;` | the **declaring file's own directory**, then the tome root |
| `invoke tome·…` / `crate·` / `above·` | the tome root, wherever it is written |

`scroll grid;` inside `tui/vterm.sg` names `tui/grid.sg`, the way `mod grid;`
in `tui/vterm.rs` names `tui/grid.rs`. Resolving it against the tome root
wherever it appeared meant a module in a subdirectory could not name the module
beside it, and the failure — `undefined variable` — said nothing about why.

The root stays in the search list *after* the declaring file's directory, so a
nested module may still name one of the tome's top-level modules
(`scroll constants;` from `tui/grid.sg` reaches `constants.sg`), and every flat
tome is unaffected. When both exist, the sibling wins: searching the root first
would stop a subdirectory from having a module of its own whenever the tome
already had one by that name, which is exactly when grouping is wanted.

`current_module_dir` carries the declaring file's directory and is saved and
restored around each module load, alongside `current_module`.

`above·` is *not* currently relative — it resolves from the tome root like
`tome·`. That happens to give the right answer for a top-level module and is
left alone deliberately; making it mean "the parent module" is a semantic
change, not a fix.

## What is still flat

`run-dir` lists and eagerly loads only the top level of the directory it is
given. Modules in subdirectories load on demand, when an `invoke` or `scroll`
reaches them — which is the Rust rule, and means an unreferenced file is not
executed. Making `run-dir`'s eager listing recurse belongs with the load-order
rework in #107.
