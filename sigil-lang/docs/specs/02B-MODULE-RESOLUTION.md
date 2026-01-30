# Sigil Module Resolution Specification

> *"A module system is a theory of program composition. For agents composing programs,
> that theory must be as precise as the programs themselves."*

## 1. Overview

This specification defines the algorithmic semantics for Sigil's module system — how names
are resolved, how files map to modules, and how visibility is computed. Unlike simpler
languages where module resolution is implicit, Sigil's module system is a **first-class
reasoning target** for agent-driven code generation.

### 1.1 Design Philosophy

For an agent composing large programs:

1. **Deterministic Resolution** — Every path resolves to exactly one item (or error)
2. **No Implicit Dependencies** — All dependencies are explicitly declared
3. **Cyclic Awareness** — Cycles are detected and reported, not silently broken
4. **Visibility as Capability** — Pub/priv is a capability-based access control system

### 1.2 Document Structure

| Section | Content |
|---------|---------|
| §2 | Module tree structure |
| §3 | Path resolution algorithm |
| §4 | Use declaration processing |
| §5 | Visibility and access control |
| §6 | Re-export resolution |
| §7 | Cycle detection |
| §8 | Error semantics |

---

## 2. Module Tree Structure

### 2.1 Crate as Root

Every Sigil program has a **crate root** which forms the top of its module tree:

```
CrateRoot
├── lib.sg or main.sg
├── submodule/
│   ├── mod.sg
│   └── child.sg
└── sibling.sg
```

**Crate Root Determination:**

```
CRATE_ROOT(path) :=
    1. If path contains Sigil.toml, that directory is root
    2. Else, walk up until Sigil.toml found
    3. Else, the file's directory is an implicit single-file crate
```

### 2.2 Module Item

A module is defined by the `mod` keyword:

```sigil
// External module (loads from file)
mod parser;              // Loads parser.sg or parser/mod.sg

// Inline module
mod helpers {
    pub fn utility() { }
}

// Public external module
pub mod ast;
```

**File Resolution for External Modules:**

```
RESOLVE_FILE(mod_name, parent_dir) :=
    1. Try: parent_dir/mod_name.sg
    2. Else try: parent_dir/mod_name/mod.sg
    3. Else: ERROR(ModuleNotFound, mod_name)
```

### 2.3 Module Path Representation

Internally, modules are identified by paths:

```
ModulePath := CrateName ('·' Segment)*
Segment := Identifier

Examples:
  mylib                     // Crate root
  mylib·parser              // mylib/parser.sg
  mylib·parser·ast          // mylib/parser/ast.sg
  std·io·File               // std library io::File
```

### 2.4 Module Tree Data Structure

```
ModuleTree := Map<ModulePath, Module>

Module := {
    path: ModulePath,
    items: Map<Name, Item>,
    children: Map<Name, ModulePath>,  // Child modules
    imports: Vec<Import>,              // use declarations
    exports: Map<Name, ExportedItem>,  // pub use re-exports
    visibility: Visibility,
}

Item := Function | Struct | Enum | Trait | Impl | Const | Type | Module

Visibility := Private | Pub | PubCrate | PubSuper | PubIn(ModulePath)
```

---

## 3. Path Resolution Algorithm

### 3.1 Path Syntax

```sigil
// Absolute paths (from crate root)
crate·parser·Token

// Relative paths (from current module)
super·sibling·Item        // Parent's sibling
self·submodule·Item       // Current module's child

// External crate paths
std·io·Read               // std library
serde·Serialize           // external crate
```

### 3.2 Path Grammar

```ebnf
path
    : path_prefix? segment ('·' segment)*
    ;

path_prefix
    : 'crate'    // From crate root
    | 'self'     // Current module
    | 'super'    // Parent module
    ;

segment
    : IDENT
    | '<' type 'as' trait '>'  // Qualified path
    ;
```

### 3.3 Resolution Algorithm

```
Algorithm: RESOLVE_PATH(path, context)

Input:
  - path: The path to resolve
  - context: Current module context {module: ModulePath, scope: LocalScope}

Output:
  - ResolvedItem | Error

Steps:

1. DETERMINE START POINT:
   match path.prefix:
     | 'crate' => start = CRATE_ROOT
     | 'self'  => start = context.module
     | 'super' => start = PARENT(context.module)
     | None    =>
         // Check local scope first
         if path.first_segment in context.scope:
             return context.scope[path.first_segment]
         // Check current module items
         if path.first_segment in MODULE(context.module).items:
             start = context.module
         // Check imports
         if path.first_segment in IMPORTS(context.module):
             return RESOLVE_IMPORT(path.first_segment, path.rest)
         // Check prelude
         if path.first_segment in PRELUDE:
             return PRELUDE[path.first_segment]
         // Check external crates
         if path.first_segment in EXTERN_CRATES:
             start = CRATE_ROOT(path.first_segment)
         else:
             return Error(UnresolvedPath, path.first_segment)

2. WALK PATH:
   current = start
   for segment in path.segments:
       module = MODULE(current)

       // Try direct item
       if segment in module.items:
           item = module.items[segment]
           if not VISIBLE(item, context.module):
               return Error(PrivateItem, segment, current)
           if MORE_SEGMENTS(path, segment):
               if item is Module:
                   current = item.path
               else:
                   return Error(NotAModule, segment)
           else:
               return item

       // Try re-export
       if segment in module.exports:
           return RESOLVE_REEXPORT(module.exports[segment], context)

       // Try child module
       if segment in module.children:
           current = module.children[segment]
       else:
           return Error(NotFound, segment, current)

3. RETURN:
   return MODULE(current)  // If path ends at module
```

### 3.4 Local Scope Priority

Names are resolved in priority order:

```
1. Local bindings (let, fn parameters)
2. Type parameters
3. Current module items
4. Imports (use declarations)
5. Prelude (implicit imports)
6. External crates
```

**Example:**

```sigil
use std·io·Read;

struct Read { }  // Shadows imported Read

fn example() {
    let Read = 42;  // Shadows struct Read

    let x = Read;           // Local binding (42)
    let y = self·Read { };  // Struct from current module
    let z: std·io·Read = .. // Absolute path to import
}
```

---

## 4. Use Declaration Processing

### 4.1 Use Syntax

```sigil
use path;                        // Simple import
use path·{a, b, c};              // Multiple imports
use path·{a as alias, b, c};     // Renamed import
use path·*;                      // Glob import
use path·{nested·{a, b}, c};     // Nested paths

pub use path·Item;               // Re-export
```

### 4.2 Import Processing Algorithm

```
Algorithm: PROCESS_IMPORTS(module)

Input:
  - module: Module being processed

Output:
  - ImportTable: Map<Name, ResolvedItem>

Steps:

1. COLLECT USE DECLARATIONS:
   uses = module.use_declarations

2. EXPAND COMPOUND USES:
   expanded = []
   for use in uses:
       expanded.append_all(EXPAND_USE(use))

   // EXPAND_USE example:
   // use a·{b, c·{d, e}}
   // => [a·b, a·c·d, a·c·e]

3. RESOLVE EACH IMPORT:
   imports = {}
   for use in expanded:
       target = RESOLVE_PATH(use.path, ModuleContext(module))
       if target is Error:
           emit Error(ImportFailed, use.path, target)
           continue

       name = use.alias or use.path.last_segment
       if name in imports:
           emit Error(DuplicateImport, name)
           continue

       imports[name] = target

4. PROCESS GLOB IMPORTS:
   for use in expanded where use.is_glob:
       target_module = RESOLVE_PATH(use.path.without_glob, ...)
       for (name, item) in target_module.public_items:
           if name not in imports:  // Don't shadow explicit imports
               imports[name] = item

5. RETURN imports
```

### 4.3 Glob Import Semantics

Glob imports (`use module·*`) bring all public items into scope:

```sigil
mod prelude {
    pub struct Vec<T> { }
    pub struct String { }
    pub fn print(s: &str) { }

    struct Internal { }  // Not exported by glob
}

use prelude·*;

fn main() {
    let v = Vec::new();     // OK: Vec is public
    let s = String::new();  // OK: String is public
    print("hello");         // OK: print is public

    let i = Internal { };   // ERROR: Internal not visible
}
```

**Glob Priority:**

Glob imports have **lowest priority** — they never shadow explicit imports or local definitions:

```sigil
use std·io·*;      // Includes Read trait
use custom·Read;   // Explicit import shadows glob

// Read here refers to custom·Read, not std·io·Read
```

---

## 5. Visibility and Access Control

### 5.1 Visibility Modifiers

```sigil
pub            // Public to all
pub(crate)     // Public within crate
pub(super)     // Public to parent module
pub(in path)   // Public to specific ancestor
               // (no modifier) Private to containing module
```

### 5.2 Visibility Hierarchy

```
        pub
         │
    pub(crate)
         │
   pub(in ancestor)
         │
    pub(super)
         │
      private
```

### 5.3 Visibility Check Algorithm

```
Algorithm: VISIBLE(item, from_module)

Input:
  - item: Item with visibility modifier
  - from_module: Module attempting access

Output:
  - bool

Steps:

1. EXTRACT VISIBILITY:
   vis = item.visibility
   item_module = item.containing_module

2. CHECK BY VISIBILITY KIND:
   match vis:
     | Pub =>
         return true

     | PubCrate =>
         return SAME_CRATE(item_module, from_module)

     | PubSuper =>
         return PARENT(item_module) == from_module
             or ANCESTOR(from_module, PARENT(item_module))

     | PubIn(path) =>
         target = RESOLVE_PATH(path, item_module)
         return from_module == target
             or ANCESTOR(from_module, target)

     | Private =>
         return item_module == from_module
             or ANCESTOR(from_module, item_module)
             // Private items visible to child modules
```

### 5.4 Struct Field Visibility

Struct fields have independent visibility:

```sigil
pub struct Config {
    pub name: String,           // Public field
    pub(crate) secret: Key,     // Crate-visible
    internal: State,            // Private
}

impl Config {
    pub fn new() -> Config {
        Config {
            name: "default".into(),
            secret: Key::generate(),  // OK: same module
            internal: State::new(),   // OK: same module
        }
    }
}

// In another module:
fn use_config(c: Config) {
    println(c.name);      // OK: public
    println(c.secret);    // OK if same crate
    println(c.internal);  // ERROR: private field
}
```

### 5.5 Visibility Leaking Prevention

Items cannot be more visible than their dependencies:

```sigil
struct PrivateType { }

pub fn leak() -> PrivateType {  // ERROR: PrivateType is private
    PrivateType { }
}

pub struct Container {
    inner: PrivateType,  // OK: field is private by default
}

pub struct BadContainer {
    pub inner: PrivateType,  // ERROR: pub field of private type
}
```

**Leak Check Algorithm:**

```
Algorithm: CHECK_VISIBILITY_LEAK(item)

For each type T mentioned in item's public signature:
    if VISIBILITY(T) < VISIBILITY(item):
        emit Error(VisibilityLeak, item, T)
```

---

## 6. Re-export Resolution

### 6.1 Re-export Syntax

```sigil
pub use internal·Type;           // Re-export single item
pub use internal·{A, B, C};      // Re-export multiple
pub use internal·Type as Alias;  // Re-export with rename
pub use internal·*;              // Re-export all (rare)
```

### 6.2 Re-export Semantics

Re-exports create a new path to an existing item:

```sigil
// In lib.sg
mod internal {
    pub struct Widget { }
}

pub use internal·Widget;  // Widget now accessible as crate·Widget

// External users can do:
use mycrate·Widget;       // Instead of mycrate·internal·Widget
```

### 6.3 Re-export Chain Resolution

```
Algorithm: RESOLVE_REEXPORT(export, context)

Input:
  - export: The re-export declaration
  - context: Current resolution context

Output:
  - ResolvedItem (the ultimate target)

Steps:

1. visited = {export.source_module}
2. current = export.target
3. while current is ReExport:
     if current.source_module in visited:
         return Error(CircularReexport, visited)
     visited.add(current.source_module)
     current = current.target
4. return current
```

### 6.4 Facade Pattern

Re-exports enable the facade pattern for API design:

```sigil
// src/lib.sg
mod parser;
mod lexer;
mod ast;
mod codegen;

// Clean public API
pub use parser·Parser;
pub use parser·ParseError;
pub use ast·{Expr, Stmt, Type};
pub use codegen·generate;

// Internal modules remain private
```

---

## 7. Cycle Detection

### 7.1 Module Cycle Types

**Import Cycles:**
```sigil
// a.sg
mod b;
use b·Item;

// b.sg
mod a;
use a·Item;  // Cycle: a → b → a
```

**Type Definition Cycles:**
```sigil
struct A { b: B }
struct B { a: A }  // Infinite size cycle
```

### 7.2 Cycle Detection Algorithm

```
Algorithm: DETECT_MODULE_CYCLES(crate)

Input:
  - crate: The crate to analyze

Output:
  - List<Cycle> where Cycle = [ModulePath]

Steps:

1. BUILD DEPENDENCY GRAPH:
   graph = {}
   for module in crate.modules:
       deps = []
       for use in module.use_declarations:
           target_module = MODULE_OF(RESOLVE_PATH(use.path))
           deps.append(target_module)
       graph[module.path] = deps

2. DETECT CYCLES (Tarjan's SCC Algorithm):
   index = 0
   stack = []
   indices = {}
   lowlinks = {}
   on_stack = {}
   sccs = []

   fn strongconnect(v):
       indices[v] = index
       lowlinks[v] = index
       index += 1
       stack.push(v)
       on_stack[v] = true

       for w in graph[v]:
           if w not in indices:
               strongconnect(w)
               lowlinks[v] = min(lowlinks[v], lowlinks[w])
           else if on_stack[w]:
               lowlinks[v] = min(lowlinks[v], indices[w])

       if lowlinks[v] == indices[v]:
           scc = []
           repeat:
               w = stack.pop()
               on_stack[w] = false
               scc.append(w)
           until w == v

           if len(scc) > 1:
               sccs.append(scc)  // Cycle found

   for v in graph:
       if v not in indices:
           strongconnect(v)

   return sccs
```

### 7.3 Cycle Handling Strategies

**Hard Error (Default):**
```
error[E0391]: cyclic dependency detected
  --> src/a.sg:1:1
   |
1  | mod b;
   | ^^^^^^ module `a` depends on `b`
   |
  --> src/b.sg:1:1
   |
1  | mod a;
   | ^^^^^^ module `b` depends on `a`
   |
   = note: cycle: a → b → a
   = help: consider splitting into a third module both can depend on
```

**Lazy Resolution (For Types):**

Some cycles are allowed if they can be lazily resolved:

```sigil
// OK: pointer indirection breaks the cycle
struct A { b: Box<B> }
struct B { a: Box<A> }

// OK: trait objects break concrete dependency
trait Node {
    fn children(&self) -> Vec<&dyn Node>;
}
```

---

## 8. Error Semantics

### 8.1 Resolution Errors

| Error | Cause | Recovery |
|-------|-------|----------|
| `UnresolvedPath` | Path segment not found | Suggest similar names |
| `PrivateItem` | Accessing non-visible item | Show visibility modifier needed |
| `NotAModule` | Using non-module as path segment | Show item kind |
| `AmbiguousGlob` | Glob imports conflict | Require explicit import |
| `CircularImport` | Import cycle detected | Show cycle path |
| `VisibilityLeak` | Public item exposes private type | Suggest visibility fixes |

### 8.2 Error Message Format

```
error[E0433]: failed to resolve `parser·Token`
  --> src/main.sg:5:10
   |
5  | use parser·Token;
   |     ^^^^^^^^^^^^
   |     │      │
   |     │      └── not found in `parser`
   |     └── module found here
   |
   = note: `parser` contains: Lexer, Parser, Error
   = help: did you mean `parser·Lexer`?

error[E0603]: struct `Internal` is private
  --> src/main.sg:10:15
   |
10 |     let x = lib·Internal { };
   |               ^^^^^^^^^^ private struct
   |
  --> src/lib.sg:5:1
   |
5  | struct Internal { }
   | ------------------- not public
   |
   = help: add `pub` to make it accessible
   = help: or use `pub(crate)` for crate-internal access
```

### 8.3 Diagnostic Suggestions

The resolver should provide intelligent suggestions:

```
Algorithm: SUGGEST_SIMILAR(name, module)

1. candidates = module.items.keys() + module.imports.keys()
2. suggestions = []
3. for candidate in candidates:
     distance = LEVENSHTEIN(name, candidate)
     if distance <= 3:
         suggestions.append((candidate, distance))
4. return suggestions.sort_by(distance).take(3)
```

---

## 9. Prelude and Implicit Imports

### 9.1 Standard Prelude

The standard prelude is implicitly imported:

```sigil
// Implicit at every module:
use std·prelude·*;
```

**Prelude Contents:**

```sigil
// std/prelude.sg
pub use crate·option·Option·{self, Some, None};
pub use crate·result·Result·{self, Ok, Err};
pub use crate·vec·Vec;
pub use crate·string·String;
pub use crate·clone·Clone;
pub use crate·copy·Copy;
pub use crate·drop·Drop;
pub use crate·cmp·{Eq, Ord, PartialEq, PartialOrd};
pub use crate·iter·Iterator;
pub use crate·fmt·{Debug, Display};
```

### 9.2 No-Prelude Mode

```sigil
//! No prelude for low-level code
#![no_std]
#![no_prelude]

// Must import everything explicitly
use core·option·Option;
```

---

## 10. Implementation Data Structures

### 10.1 Name Resolution Context

```
struct ResolutionContext {
    crate_root: ModulePath,
    current_module: ModulePath,
    local_scope: Vec<Scope>,      // Stack of local scopes
    extern_crates: Map<Name, CrateId>,
    prelude: Map<Name, ResolvedItem>,
}

struct Scope {
    bindings: Map<Name, LocalBinding>,
    type_params: Map<Name, TypeParam>,
}
```

### 10.2 Resolution Cache

```
struct ResolutionCache {
    resolved_paths: Map<(ModulePath, Path), ResolvedItem>,
    module_imports: Map<ModulePath, ImportTable>,
    visibility_checks: Map<(ItemId, ModulePath), bool>,
}
```

---

## 11. Formal Semantics

### 11.1 Module Context Judgment

```
Γ ⊢ mod : ModuleContext

──────────────────────── (MOD-ROOT)
Γ ⊢ crate : {path: [], items: ITEMS(crate)}

Γ ⊢ M : {path: p, items: I}    'mod m' ∈ I
FILE(p, m) = source
─────────────────────────────────────────── (MOD-EXTERNAL)
Γ ⊢ M·m : {path: p·m, items: PARSE(source)}

Γ ⊢ M : {path: p, items: I}    'mod m { body }' ∈ I
──────────────────────────────────────────────────── (MOD-INLINE)
Γ ⊢ M·m : {path: p·m, items: body}
```

### 11.2 Path Resolution Judgment

```
Γ; M ⊢ path ⇒ item

'crate' · segments = path
Γ ⊢ CRATE : root
root ⊢ segments ⇒ item
────────────────────────── (PATH-ABSOLUTE)
Γ; M ⊢ path ⇒ item

'self' · segments = path
M ⊢ segments ⇒ item
────────────────────────── (PATH-SELF)
Γ; M ⊢ path ⇒ item

name · segments = path
name ∈ IMPORTS(M)
IMPORTS(M)[name] ⊢ segments ⇒ item
────────────────────────────────── (PATH-IMPORT)
Γ; M ⊢ path ⇒ item
```

### 11.3 Visibility Judgment

```
vis ⊢ item VISIBLE_FROM from_module

──────────────────────────────── (VIS-PUB)
pub ⊢ item VISIBLE_FROM _

SAME_CRATE(item.module, from_module)
────────────────────────────────────── (VIS-CRATE)
pub(crate) ⊢ item VISIBLE_FROM from_module

from_module = PARENT(item.module) ∨ DESCENDANT(from_module, PARENT(item.module))
───────────────────────────────────────────────────────────────────────────────── (VIS-SUPER)
pub(super) ⊢ item VISIBLE_FROM from_module

from_module = item.module ∨ DESCENDANT(from_module, item.module)
──────────────────────────────────────────────────────────────── (VIS-PRIVATE)
private ⊢ item VISIBLE_FROM from_module
```

---

*This specification provides the algorithmic foundation for Sigil's module system —
enabling agents to reason precisely about code organization, visibility, and composition.*
