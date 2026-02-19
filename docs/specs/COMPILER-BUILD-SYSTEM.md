# Sigil Compiler Build System Enhancement Specification

**Version:** 0.1.0
**Status:** Draft
**Date:** 2025-02-10
**Methodology:** Spec-Driven Development

---

## 1. Problem Statement

The Sigil compiler currently only supports building single-file executables. Real-world projects like Styx require:

1. **Library compilation** - Producing `.a` (static) or `.so` (dynamic) libraries
2. **Workspace builds** - Building multiple tomes in dependency order
3. **Dependency resolution** - Resolving and compiling dependencies before dependents
4. **Multi-file tomes** - Combining multiple source files into a single compilation unit

### Current Limitations

| Feature | Current State | Required State |
|---------|---------------|----------------|
| Binary tomes | ✅ Works | ✅ Works |
| Library tomes | ❌ Requires `main` | ✅ Compile to `.a`/`.rlib` |
| Workspaces | ❌ Not supported | ✅ Build all members |
| Dependencies | ❌ Ignored | ✅ Resolve and link |
| Multi-file | ❌ Single file only | ✅ Merge `mod` tree |

---

## 2. Conceptual Model

### Tome Types

```
Tome
├── Binary (has src/main.sigil)
│   └── Produces: executable
├── Library (has src/lib.sigil)
│   └── Produces: .a (static lib) or .rlib (Sigil archive)
└── Both (has both)
    └── Produces: executable + library
```

### Build Graph

```
Workspace
├── Sigil.toml (workspace manifest)
└── members/
    ├── tome-a/
    │   ├── sigil.toml
    │   └── src/lib.sigil
    ├── tome-b/
    │   ├── sigil.toml (depends on tome-a)
    │   └── src/lib.sigil
    └── tome-c/
        ├── sigil.toml (depends on tome-a, tome-b)
        └── src/main.sigil

Build order: tome-a → tome-b → tome-c
```

---

## 3. Manifest Format

### Project Manifest (sigil.toml)

```toml
[project]
name = "my-tome"
version = "0.1.0"
edition = "2025"

# Tome type: "lib", "bin", or "both" (default: inferred from src/)
tome-type = "lib"

# For libraries: static (.a), dynamic (.so), or rlib (.rlib)
lib-type = "rlib"  # default

# Binary configuration (if tome-type includes bin)
[[bin]]
name = "my-app"
path = "src/main.sigil"

# Library configuration (if tome-type includes lib)
[lib]
name = "my_tome"
path = "src/lib.sigil"

[dependencies]
other-tome = { path = "../other-tome" }
```

### Workspace Manifest

```toml
[workspace]
members = [
    "tomes/core",
    "tomes/utils",
    "tomes/app",
]

# Shared dependencies (inherited by members)
[workspace.dependencies]
common-dep = { path = "../common" }
```

---

## 4. Compilation Pipeline

### Phase 1: Discovery

```
discover_tomes(manifest_path) → Vec<TomeInfo>
  1. Parse sigil.toml
  2. If workspace: collect all member paths
  3. For each tome:
     a. Determine tome type (lib/bin/both)
     b. Collect source files (lib.sigil, main.sigil, mod files)
     c. Parse dependencies
  4. Return tome metadata
```

### Phase 2: Dependency Resolution

```
resolve_dependencies(tomes) → BuildGraph
  1. Build dependency graph (DAG)
  2. Detect cycles → error
  3. Topological sort → build order
  4. Return ordered list with dependency info
```

### Phase 3: Compilation

```
compile_tome(tome_info, compiled_deps) → CompileResult
  1. Merge source files into single AST
  2. Import symbols from compiled dependencies
  3. Type-check with dependency types available
  4. Generate LLVM IR
  5. Based on tome type:
     - Binary: link with runtime → executable
     - Library: emit object file → archive (.a/.rlib)
  6. Return compiled artifact path
```

### Phase 4: Linking (Binary Only)

```
link_binary(main_obj, dep_libs, runtime) → Executable
  1. Collect all dependency .a/.rlib files
  2. Link with: clang main.o dep1.a dep2.a ... runtime.a -o binary
  3. Return executable path
```

---

## 5. Library Compilation Details

### Object File Generation

Libraries compile to object files without linking:

```rust
// Current: compile_file() always links with runtime
// New: compile_to_object() stops after object generation

fn compile_to_object(source: &str, output: &Path) -> Result<()> {
    let context = Context::create();
    let mut compiler = LlvmCompiler::with_mode(&context, opt, CompileMode::Aot)?;
    compiler.compile(source)?;
    compiler.write_object_file(output)?;
    Ok(())
}
```

### Archive Creation

Convert object files to static library:

```rust
fn create_archive(objects: &[Path], output: &Path) -> Result<()> {
    // Use `ar` or `llvm-ar` to create .a file
    Command::new("ar")
        .args(&["rcs", output.to_str().unwrap()])
        .args(objects.iter().map(|p| p.to_str().unwrap()))
        .status()?;
    Ok(())
}
```

### Sigil Archive (.rlib) Format

For richer metadata, use a custom archive format:

```
.rlib structure:
├── metadata.json       # Tome metadata, public types, exports
├── lib.o               # Compiled object code
└── source_hash.txt     # Hash of source for incremental builds
```

---

## 6. Multi-File Tome Support

### Module Tree Resolution

```
src/
├── lib.sigil           # Tome root
├── utils.sigil         # invoke utils; in lib.sigil
├── types/
│   ├── mod.sigil       # invoke types; in lib.sigil
│   └── common.sigil    # invoke common; in mod.sigil
```

### AST Merging

```rust
fn merge_modules(root: &Path) -> Result<MergedAst> {
    let mut merged = Ast::new();
    let root_ast = parse_file(root)?;

    for invoke in root_ast.invokes() {
        if invoke.is_local_module() {
            let module_path = resolve_module_path(root, &invoke.path);
            let module_ast = merge_modules(&module_path)?; // Recursive
            merged.add_module(invoke.name, module_ast);
        }
    }

    merged.add_items(root_ast.items());
    Ok(merged)
}
```

---

## 7. Dependency Type Resolution

### Type Export/Import

When compiling a tome that depends on another:

```rust
fn load_dependency_types(dep_path: &Path) -> Result<TypeContext> {
    // Option 1: Parse .rlib metadata
    let rlib = RlibArchive::open(dep_path)?;
    let metadata = rlib.read_metadata()?;
    Ok(metadata.exported_types)

    // Option 2: Re-parse source (slower but simpler for bootstrap)
    let source = fs::read_to_string(dep_path.join("src/lib.sigil"))?;
    let ast = parse(&source)?;
    let mut tc = TypeChecker::new();
    tc.check_file(&ast)?;
    Ok(tc.export_public_types())
}
```

### Symbol Resolution

```
invoke other_tome·SomeType;

Resolution:
1. Find 'other_tome' in dependencies
2. Load its type context
3. Look up 'SomeType' in its exports
4. Add to current scope
```

---

## 8. Build Commands

### Updated CLI

```
sigil build                    # Build current tome
sigil build --release          # Release mode (optimized)
sigil build --lib              # Build library only
sigil build --bin <name>       # Build specific binary
sigil build --workspace        # Build all workspace members
sigil build -p <tome>         # Build specific tome in workspace
```

### Build Output Structure

```
target/
├── debug/
│   ├── deps/                  # Dependency artifacts
│   │   ├── styx_core.rlib
│   │   └── styx_db.rlib
│   ├── styx-server            # Binary
│   └── libstyx_core.a         # Library (if requested)
└── release/
    └── ...
```

---

## 9. Implementation Plan

### Phase 1: Library Compilation (Minimal)

**Goal:** `sigil build` works for library tomes

1. Detect lib.sigil vs main.sigil
2. For libraries: compile to .o, skip linking
3. Create .a archive from object file
4. Output to target/debug/<name>.a

**Files to modify:**
- `main.rs`: `build_project()` function
- New: `compile_library()` function

### Phase 2: Dependency Resolution

**Goal:** Tomes can depend on other local tomes

1. Parse `[dependencies]` from sigil.toml
2. Resolve path dependencies
3. Compile dependencies first (topological order)
4. Pass dependency types to type checker
5. Link binaries with dependency libraries

**Files to modify:**
- `main.rs`: Add manifest parsing
- `typeck.rs`: Add dependency type loading
- New: `manifest.rs` for TOML parsing
- New: `resolver.rs` for dependency resolution

### Phase 3: Workspace Support

**Goal:** `sigil build --workspace` builds all members

1. Detect workspace in sigil.toml
2. Collect all member tomes
3. Build dependency graph across workspace
4. Compile in topological order

**Files to modify:**
- `main.rs`: Add workspace detection
- `resolver.rs`: Handle workspace graph

### Phase 4: Multi-File Tomes

**Goal:** Tomes can have multiple source files

1. Follow `invoke` declarations to find modules
2. Parse all modules
3. Merge ASTs before type checking
4. Compile merged AST

**Files to modify:**
- `parser.rs`: Add module resolution
- `main.rs`: Add AST merging

---

## 10. Success Criteria

### Phase 1 Complete When:
- [ ] `sigil build` in styx-core produces `target/debug/styx_core.a`
- [ ] `sigil build` in styx-server still produces executable
- [ ] Library type auto-detected from src/lib.sigil

### Phase 2 Complete When:
- [ ] styx-server compiles with styx-core as dependency
- [ ] Type errors in styx-server reference styx-core types correctly
- [ ] Binary links against dependency .a files

### Phase 3 Complete When:
- [ ] `sigil build --workspace` in styx/ builds all 26 tomes
- [ ] Correct build order (core → db → http → ...)
- [ ] All artifacts in target/debug/

### Phase 4 Complete When:
- [ ] Tomes with multiple .sigil files compile correctly
- [ ] `invoke` statements resolve to local modules
- [ ] Module-private items not visible outside module

---

## 11. Open Questions

1. **Incremental compilation?** - Track source hashes, skip unchanged
2. **Parallel builds?** - Tomes at same dependency level can build in parallel
3. **Feature flags?** - Support `[features]` in sigil.toml
4. **External dependencies?** - Tomes from registry (future)

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2025-02-10 | Initial draft based on Styx compilation requirements |
