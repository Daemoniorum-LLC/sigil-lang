//! The set of files one check actually covers.
//!
//! `sigil check lib.sigil` used to read `lib.sigil` and nothing else. Every
//! module reached through `☉ scroll registry;` went unopened, so a clean check
//! on a crate entry point said nothing about the crate — a deliberate syntax
//! error one file below the entry still reported `✓ no errors` (LARES-453).
//!
//! This module answers the prior question: given an entry file, which files is
//! the check responsible for? It walks `scroll` declarations and the
//! `invoke tome·…` module loads the same way the interpreter does, so the set
//! of files checked matches the set of files run.

use crate::ast::{Block, Expr, ImplItem, Item, Module, SourceFile, Stmt, UseTree};
use crate::parser::Parser;
use crate::span::Span;
use std::collections::HashSet;
use std::path::{Path, PathBuf};

/// One file in the graph, parsed or not.
pub struct CrateFile {
    pub path: PathBuf,
    pub source: String,
    /// `Ok` on a clean parse; `Err` carries the parse error's rendered message.
    pub parsed: Result<SourceFile, String>,
    /// The module declaration that pulled this file in — `None` for the entry.
    pub declared_by: Option<ModuleRef>,
}

/// Where a module declaration appeared.
#[derive(Clone, Debug)]
pub struct ModuleRef {
    /// File carrying the declaration.
    pub from: PathBuf,
    /// Module name as declared.
    pub name: String,
    /// Span of the name within `from`, so a diagnostic can point at it.
    pub span: Span,
}

/// A `scroll foo;` whose file could not be found.
pub struct MissingModule {
    pub module: ModuleRef,
    /// Paths that were tried, in order.
    pub candidates: Vec<PathBuf>,
}

/// The entry file plus every file reachable from it through module declarations.
pub struct CrateGraph {
    /// Entry file first, then modules in the order they were reached.
    pub files: Vec<CrateFile>,
    pub missing: Vec<MissingModule>,
}

impl CrateGraph {
    /// Walk the module graph rooted at `entry`.
    ///
    /// Reading and parsing failures are recorded, never fatal: a crate with one
    /// broken module still yields the other modules, so one bad file does not
    /// blind the check to the rest.
    pub fn load(entry: &Path, entry_source: String) -> Self {
        let mut graph = CrateGraph {
            files: Vec::new(),
            missing: Vec::new(),
        };
        let mut seen: HashSet<PathBuf> = HashSet::new();
        seen.insert(canonical(entry));

        // Breadth-first, so a module's own modules are reported after its
        // siblings rather than interleaved with them.
        let mut queue: std::collections::VecDeque<(PathBuf, String, Option<ModuleRef>)> =
            std::collections::VecDeque::new();
        queue.push_back((entry.to_path_buf(), entry_source, None));

        while let Some((path, source, declared_by)) = queue.pop_front() {
            let parsed = Parser::new(&source)
                .parse_file()
                .map_err(|e| format!("{}", e));

            if let Ok(ref ast) = parsed {
                let dir = path.parent().unwrap_or_else(|| Path::new(".")).to_path_buf();
                for (name, span) in declared_modules(ast) {
                    let module = ModuleRef {
                        from: path.clone(),
                        name: name.clone(),
                        span,
                    };
                    let candidates = module_candidates(&dir, &name);
                    match candidates.iter().find(|c| c.is_file()) {
                        Some(found) => {
                            // `seen` is keyed on the canonical path, so a module
                            // two files reach is walked once and a cycle
                            // terminates.
                            if !seen.insert(canonical(found)) {
                                continue;
                            }
                            match std::fs::read_to_string(found) {
                                Ok(src) => queue.push_back((found.clone(), src, Some(module))),
                                Err(e) => graph.files.push(CrateFile {
                                    path: found.clone(),
                                    source: String::new(),
                                    parsed: Err(format!("failed to read module file: {}", e)),
                                    declared_by: Some(module),
                                }),
                            }
                        }
                        None => graph.missing.push(MissingModule { module, candidates }),
                    }
                }
            }

            graph.files.push(CrateFile {
                path,
                source,
                parsed,
                declared_by,
            });
        }

        graph
    }

    /// The entry file alone, module declarations left unfollowed.
    ///
    /// What `sigil check` did before LARES-453, kept behind `--no-traverse` for
    /// the case where following a module makes things worse rather than better.
    pub fn single(entry: &Path, source: String) -> Self {
        let parsed = Parser::new(&source)
            .parse_file()
            .map_err(|e| format!("{}", e));
        CrateGraph {
            files: vec![CrateFile {
                path: entry.to_path_buf(),
                source,
                parsed,
                declared_by: None,
            }],
            missing: Vec::new(),
        }
    }

    /// Every file that parsed, with its AST.
    pub fn parsed_files(&self) -> impl Iterator<Item = (&PathBuf, &SourceFile)> {
        self.files
            .iter()
            .filter_map(|f| f.parsed.as_ref().ok().map(|ast| (&f.path, ast)))
    }
}

/// Where a module named `name` may live, in the order the interpreter tries.
fn module_candidates(dir: &Path, name: &str) -> Vec<PathBuf> {
    vec![
        dir.join(format!("{}.sigil", name)),
        dir.join(format!("{}.sg", name)),
        dir.join(name).join("mod.sigil"),
        dir.join(name).join("mod.sg"),
    ]
}

fn canonical(p: &Path) -> PathBuf {
    p.canonicalize().unwrap_or_else(|_| p.to_path_buf())
}

/// Names of the out-of-line modules a file declares.
///
/// Two spellings reach a sibling file, and the interpreter honours both:
/// a bodiless `☉ scroll registry;`, and `invoke tome·registry·Thing;`, where
/// `tome` (or `crate`/`above`) names this crate rather than a dependency.
pub fn declared_modules(file: &SourceFile) -> Vec<(String, Span)> {
    let mut out = Vec::new();
    let mut seen = HashSet::new();
    for item in &file.items {
        collect_from_item(&item.node, &mut out, &mut seen);
    }
    out
}

fn collect_from_item(item: &Item, out: &mut Vec<(String, Span)>, seen: &mut HashSet<String>) {
    match item {
        Item::Module(m) => collect_from_module(m, out, seen),
        Item::Use(u) => {
            if let Some((name, span)) = tome_module_of(&u.tree) {
                if seen.insert(name.clone()) {
                    out.push((name, span));
                }
            }
        }
        // An `invoke` is not confined to the top of a file — Sigil code puts one
        // in the body of the rite that needs it, right above the call:
        //
        //     rite init_alters(system: &Δ PluralSystem) {
        //         invoke tome·alters·council·register_council_alters;
        //         register_council_alters(system);
        //     }
        //
        // Collecting only top-level items missed those, which made a function
        // that is plainly defined look undefined.
        _ => walk_nested_items(item, &mut |nested| collect_from_item(nested, out, seen)),
    }
}

/// Call `f` on every item nested inside `item`'s bodies.
fn walk_nested_items(item: &Item, f: &mut impl FnMut(&Item)) {
    match item {
        Item::Function(func) => {
            if let Some(body) = &func.body {
                walk_block(body, f);
            }
        }
        Item::Impl(block) => {
            for it in &block.items {
                if let ImplItem::Function(func) = it {
                    if let Some(body) = &func.body {
                        walk_block(body, f);
                    }
                }
            }
        }
        Item::Module(m) => {
            if let Some(items) = &m.items {
                for it in items {
                    walk_nested_items(&it.node, f);
                }
            }
        }
        _ => {}
    }
}

fn walk_block(block: &Block, f: &mut impl FnMut(&Item)) {
    for stmt in &block.stmts {
        match stmt {
            Stmt::Item(item) => {
                f(item);
                walk_nested_items(item, f);
            }
            Stmt::Let { init: Some(e), .. } => walk_expr(e, f),
            Stmt::LetElse {
                init, else_branch, ..
            } => {
                walk_expr(init, f);
                walk_expr(else_branch, f);
            }
            Stmt::Expr(e) | Stmt::Semi(e) => walk_expr(e, f),
            Stmt::Let { init: None, .. } => {}
        }
    }
    if let Some(e) = &block.expr {
        walk_expr(e, f);
    }
}

/// Only the expressions that can hold a block, which are the only ones that can
/// hold an item.
fn walk_expr(expr: &Expr, f: &mut impl FnMut(&Item)) {
    match expr {
        Expr::Block(b) => walk_block(b, f),
        Expr::If {
            then_branch,
            else_branch,
            ..
        } => {
            walk_block(then_branch, f);
            if let Some(e) = else_branch {
                walk_expr(e, f);
            }
        }
        Expr::Match { arms, .. } => {
            for arm in arms {
                walk_expr(&arm.body, f);
            }
        }
        Expr::Loop { body, .. } | Expr::While { body, .. } | Expr::For { body, .. } => {
            walk_block(body, f)
        }
        Expr::Closure { body, .. } => walk_expr(body, f),
        _ => {}
    }
}

fn collect_from_module(m: &Module, out: &mut Vec<(String, Span)>, seen: &mut HashSet<String>) {
    match &m.items {
        // `scroll foo { … }` is already in this file; its children may still
        // declare out-of-line modules.
        Some(items) => {
            for item in items {
                collect_from_item(&item.node, out, seen);
            }
        }
        None => {
            if seen.insert(m.name.name.clone()) {
                out.push((m.name.name.clone(), m.name.span));
            }
        }
    }
}

/// What a file brings into scope for the files around it.
#[derive(Default)]
pub struct DeclaredNames {
    /// Every name a call in this crate could legitimately target.
    pub names: HashSet<String>,
    /// The file glob-imports from somewhere — `invoke qliphoth·prelude·*;`.
    ///
    /// A glob from a dependency we do not compile brings in names we cannot
    /// enumerate, so call-target resolution has to hold its tongue for that
    /// file rather than blame code that is fine.
    pub has_unresolved_glob: bool,
}

/// Names a file declares or imports, as a call could spell them.
///
/// This is deliberately generous. It is the denominator of a "does this call
/// target exist" question, and the answer to that question becomes an error, so
/// a name that is arguably out of scope belongs here anyway — reporting a real
/// call as unresolvable is far worse than missing one.
pub fn declared_names(file: &SourceFile) -> DeclaredNames {
    let mut out = DeclaredNames::default();
    for item in &file.items {
        collect_names_from_item(&item.node, &mut out);
    }
    out
}

fn collect_names_from_item(item: &Item, out: &mut DeclaredNames) {
    match item {
        Item::Function(f) => {
            out.names.insert(f.name.name.clone());
        }
        Item::Struct(s) => {
            out.names.insert(s.name.name.clone());
        }
        Item::Enum(e) => {
            out.names.insert(e.name.name.clone());
            // A variant is callable unqualified: `Some(x)`, and the same for a
            // user enum's tuple variants.
            for v in &e.variants {
                out.names.insert(v.name.name.clone());
            }
        }
        Item::Trait(t) => {
            out.names.insert(t.name.name.clone());
        }
        Item::TypeAlias(t) => {
            out.names.insert(t.name.name.clone());
        }
        Item::Const(c) => {
            out.names.insert(c.name.name.clone());
        }
        Item::Static(s) => {
            out.names.insert(s.name.name.clone());
        }
        Item::Actor(a) => {
            out.names.insert(a.name.name.clone());
        }
        Item::Macro(m) => {
            out.names.insert(m.name.name.clone());
        }
        Item::ExternBlock(b) => {
            for it in &b.items {
                match it {
                    crate::ast::ExternItem::Function(f) => {
                        out.names.insert(f.name.name.clone());
                    }
                    crate::ast::ExternItem::Static(s) => {
                        out.names.insert(s.name.name.clone());
                    }
                    crate::ast::ExternItem::Type(_) => {}
                }
            }
        }
        Item::Module(m) => {
            out.names.insert(m.name.name.clone());
            if let Some(items) = &m.items {
                // An inline module's items are reachable unqualified from the
                // rest of the crate in practice, so count them.
                for it in items {
                    collect_names_from_item(&it.node, out);
                }
            }
        }
        Item::Use(u) => collect_names_from_use(&u.tree, None, out),
        _ => {}
    }
    // An `invoke` inside a rite's body imports a name just as a top-level one does.
    walk_nested_items(item, &mut |nested| {
        if let Item::Use(u) = nested {
            collect_names_from_use(&u.tree, None, out);
        } else if let Item::Function(func) = nested {
            out.names.insert(func.name.name.clone());
        }
    });
}

/// Does a path starting with `root` stay inside this crate?
///
/// It matters for globs only. `invoke tome·hooks·*` pulls in names from a file
/// the traversal already walked, so they are in hand; `invoke qliphoth·prelude·*`
/// pulls in names from a dependency this check never compiles, and those are not.
fn is_intra_crate_root(root: &str) -> bool {
    matches!(root, "tome" | "crate" | "above" | "super" | "self" | "scroll")
}

fn collect_names_from_use(tree: &UseTree, root: Option<&str>, out: &mut DeclaredNames) {
    match tree {
        UseTree::Path { prefix, suffix } => {
            out.names.insert(prefix.name.clone());
            collect_names_from_use(suffix, root.or(Some(&prefix.name)), out);
        }
        UseTree::Name(n) => {
            out.names.insert(n.name.clone());
        }
        UseTree::Rename { name, alias } => {
            out.names.insert(name.name.clone());
            out.names.insert(alias.name.clone());
        }
        UseTree::Glob => {
            if !root.map_or(false, is_intra_crate_root) {
                out.has_unresolved_glob = true;
            }
        }
        UseTree::Group(trees) => {
            for t in trees {
                collect_names_from_use(t, root, out);
            }
        }
    }
}

/// For `invoke tome·registry·Thing`, the module name `registry`.
///
/// `invoke tome·registry;` — one segment after the crate marker — names the
/// module itself, which is why the single-segment case still yields a name.
fn tome_module_of(tree: &UseTree) -> Option<(String, Span)> {
    let (prefix, suffix) = match tree {
        UseTree::Path { prefix, suffix } => (prefix, suffix),
        _ => return None,
    };
    if !matches!(prefix.name.as_str(), "tome" | "crate" | "above") {
        return None;
    }
    match suffix.as_ref() {
        UseTree::Path { prefix, .. } => Some((prefix.name.clone(), prefix.span)),
        UseTree::Name(n) => Some((n.name.clone(), n.span)),
        UseTree::Rename { name, .. } => Some((name.name.clone(), name.span)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &str) -> SourceFile {
        Parser::new(src).parse_file().expect("parses")
    }

    fn names(f: &SourceFile) -> Vec<String> {
        declared_modules(f).into_iter().map(|(n, _)| n).collect()
    }

    #[test]
    fn bodiless_scroll_is_a_module_declaration() {
        let f = parse("☉ scroll registry;\nrite main() { }\n");
        assert_eq!(names(&f), vec!["registry".to_string()]);
    }

    #[test]
    fn inline_scroll_declares_nothing_out_of_line() {
        let f = parse("scroll inner { rite helper() { } }\n");
        assert!(declared_modules(&f).is_empty());
    }

    #[test]
    fn invoke_tome_names_a_sibling_module() {
        let f = parse("invoke tome·registry·Thing;\nrite main() { }\n");
        assert_eq!(names(&f), vec!["registry".to_string()]);
    }

    #[test]
    fn invoke_of_a_dependency_is_not_a_sibling_module() {
        let f = parse("invoke qliphoth·prelude·*;\nrite main() { }\n");
        assert!(declared_modules(&f).is_empty());
    }

    #[test]
    fn a_glob_from_a_dependency_cannot_be_enumerated() {
        let f = parse("invoke qliphoth·prelude·*;\nrite main() { }\n");
        assert!(declared_names(&f).has_unresolved_glob);
    }

    #[test]
    fn a_glob_from_this_crate_is_already_in_hand() {
        // `tome·hooks` is a file the traversal walks, so its names are known
        // and the glob is no reason to stop resolving call targets.
        let f = parse("☉ invoke tome·hooks·*;\nrite main() { }\n");
        assert!(!declared_names(&f).has_unresolved_glob);
    }

    #[test]
    fn a_module_declared_twice_is_walked_once() {
        let f = parse("☉ scroll registry;\ninvoke tome·registry·Thing;\nrite main() { }\n");
        assert_eq!(names(&f), vec!["registry".to_string()]);
    }
}
