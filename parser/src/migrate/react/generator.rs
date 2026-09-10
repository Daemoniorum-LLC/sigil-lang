//! Qliphoth code generation from MigrationSpec.
//!
//! Generates idiomatic Sigil/Qliphoth code from migration specifications:
//! - Actor structures with state fields
//! - Message enums and handlers
//! - VNode builder chains from JSX
//! - Qliphoth framework imports
//!
//! See docs/specs/REACT-MIGRATION.md Section 7 for Qliphoth mapping.

use super::ast_transform::{self, TransformConfig};
use super::stmt_transform;
use super::extraction::*;
use super::spec::*;

// =============================================================================
// Generated Code Output
// =============================================================================

/// Complete generated Sigil file.
#[derive(Debug, Clone)]
pub struct GeneratedSigil {
    pub path: String,
    pub code: String,
    pub component_name: String,
}

/// Code generator for Qliphoth components.
pub struct QliphothGenerator<'a> {
    spec: &'a ComponentMigrationSpec,
    indent: usize,
    /// Whether we're generating for an actor (true) or pure function (false)
    is_actor: bool,
    /// Parameter names for pure functions (used for expression interpolation)
    param_names: Vec<String>,
    /// Every other component's props, by component name.
    ///
    /// A child component in JSX is a CALL to that component's function, and its
    /// attributes are that function's parameters. Without this the generator
    /// emitted `InventoryList·view()` and hung the attributes off it as if it
    /// were a VNode — a method on a type that does not exist, on a value that
    /// is not one.
    component_props: std::collections::HashMap<String, ComponentCallShape>,
}

/// What a call site needs to know about a sibling component: whether it is a
/// function it can call, and what that function's parameters are.
#[derive(Clone, Debug)]
pub struct ComponentCallShape {
    pub is_actor: bool,
    pub props: Vec<String>,
}

/// Scope for tracking local variables (like map iterators) that shouldn't get self. prefix
#[derive(Default, Clone)]
struct VNodeScope {
    /// Local variable names (iterator vars, etc.)
    locals: Vec<String>,
}

impl<'a> QliphothGenerator<'a> {
    pub fn new(spec: &'a ComponentMigrationSpec) -> Self {
        let is_actor = spec.target.pattern == TargetPattern::Actor;
        let param_names: Vec<String> = spec.recommendations.props_handling.fields
            .iter()
            .map(|f| f.name.clone())
            .collect();

        Self {
            spec,
            indent: 0,
            is_actor,
            param_names,
            component_props: std::collections::HashMap::new(),
        }
    }

    /// As `new`, but able to call sibling components.
    pub fn with_components(
        spec: &'a ComponentMigrationSpec,
        component_props: std::collections::HashMap<String, ComponentCallShape>,
    ) -> Self {
        let mut g = Self::new(spec);
        g.component_props = component_props;
        g
    }

    /// Generate complete Sigil file for the component.
    pub fn generate(&self) -> GeneratedSigil {
        self.generate_with(true)
    }

    /// The component alone — no module constants, no helper functions.
    ///
    /// For the project layout, where those are emitted once into a shared
    /// module instead of copied into every file that references them. Copying
    /// is what makes the per-file output self-contained and the project output
    /// impossible: concatenated, the 93 files redefine `get_json` 84 times.
    pub fn generate_component_only(&self) -> GeneratedSigil {
        self.generate_with(false)
    }

    fn generate_with(&self, with_module_scope: bool) -> GeneratedSigil {
        let mut code = String::new();

        // Generate imports
        code.push_str(&self.generate_imports());
        code.push('\n');

        // Module-scope constants from the same file
        let (constants, emitted_constants) = self.generate_module_constants();
        if with_module_scope && !constants.is_empty() {
            code.push_str(&constants);
            code.push('\n');
        }

        // Helper functions from this file and the modules it imports
        let helpers = self.generate_helper_functions(&emitted_constants);
        if with_module_scope && !helpers.is_empty() {
            code.push_str(&helpers);
            code.push('\n');
        }

        // Generate message enum if needed
        if !self.spec.recommendations.messages.is_empty() {
            code.push_str(&self.generate_message_enum());
            code.push('\n');
        }

        // Generate actor or function based on target pattern
        match self.spec.target.pattern {
            TargetPattern::Actor => {
                code.push_str(&self.generate_actor());
            }
            TargetPattern::Function => {
                code.push_str(&self.generate_function());
            }
        }

        GeneratedSigil {
            path: self.spec.target.suggested_path.clone(),
            code,
            component_name: self.spec.name.clone(),
        }
    }

    // =========================================================================
    // Import Generation
    // =========================================================================

    fn generate_imports(&self) -> String {
        let mut imports = Vec::new();

        // Always include qliphoth prelude
        imports.push("invoke qliphoth·prelude·*;".to_string());

        // Check for DOM/browser API usage (simplified detection)
        let needs_sys = self.spec.recommendations.effects.iter()
            .any(|e| e.reasoning.contains("document") || e.reasoning.contains("window"));

        if needs_sys {
            imports.push("invoke qliphoth_sys·*;".to_string());
        }

        // Note: Router detection can be added when router hooks are extracted

        imports.join("\n")
    }

    /// Module constants that will be emitted as zero-argument functions.
    ///
    /// Must agree exactly with `generate_module_constants`, or a reference
    /// becomes a call to something that was never emitted.
    /// Declared parameter counts for the helpers this component can reach.
    fn helper_arity(&self) -> std::collections::HashMap<String, usize> {
        self.spec
            .source
            .helpers
            .iter()
            .map(|h| (to_snake_case(&h.name), h.parameters.len()))
            .collect()
    }

    fn value_constant_names(&self) -> Vec<String> {
        // Deliberately NOT through `transform_expression_scoped`: that builds a
        // TransformConfig, which calls this, which calls it — a stack overflow
        // the first time the migrator ran. The bucket a constant falls into does
        // not depend on which other constants became functions, so an empty
        // `value_constants` here is not just safe, it is correct.
        let config = TransformConfig {
            prefix_self: self.is_actor,
            state_fields: self
                .spec
                .recommendations
                .state_fields
                .iter()
                .map(|f| f.to_field.clone())
                .collect(),
            locals: self.spec.source.module_scope.clone(),
            props: self.param_names.clone(),
            lookup_constants: Vec::new(),
            value_constants: Vec::new(),
            fn_arity: std::collections::HashMap::new(),
        };
        self.spec
            .source
            .module_constants
            .iter()
            .filter(|c| {
                if !c.entries.is_empty() {
                    return false;
                }
                let value = ast_transform::transform_expression(&c.init, &config).code;
                !is_constant_expression(&value) && is_literal_only(&value)
            })
            .map(|c| c.name.clone())
            .collect()
    }

    /// Module-scope `const`s from the component's own file.
    ///
    /// `DEFAULT_SORT` and its siblings are read by the view but declared
    /// nowhere in the generated file — the unknown-identifier rule used to turn
    /// them into `self.default_sort`, a field no actor has, and putting them in
    /// scope only moved the problem from a wrong field to an undefined name.
    fn generate_module_constants(&self) -> (String, std::collections::HashSet<String>) {
        let mut lines = Vec::new();
        let mut scope = VNodeScope::default();
        scope.locals.extend(self.spec.source.module_scope.iter().cloned());

        // Names this file has already declared, in order. A constant body that
        // names only these is self-contained *given what precedes it*, which is
        // the property the emission guard actually wants — `is_literal_only`
        // alone rejected `[DEFAULT_NEW_STATUS, DEFAULT_DONE_STATUS, …]` even
        // once both of those were emitted two lines above it.
        let mut emitted: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut known: std::collections::HashSet<String> =
            HOST_FUNCTIONS.iter().map(|s| s.to_string()).collect();

        for c in &self.spec.source.module_constants {
            let name = to_snake_case(&c.name);
            let value = self.transform_expression_scoped(&c.init, &scope);
            let one_line: String = c.init.split_whitespace().collect::<Vec<_>>().join(" ");
            let one_line = if one_line.chars().count() > 100 {
                one_line.chars().take(97).collect::<String>() + "..."
            } else {
                one_line
            };
            // An object literal is a LOOKUP TABLE — `DEFAULT_SORT[view]`,
            // `refusalText[reason]`. It cannot be a module binding, but it can be
            // a function, and the reference site becomes a call (see
            // `TransformConfig::lookup_constants`). Eight of these across the
            // client, each taking its file with it.
            if !c.entries.is_empty() {
                let arms: Vec<String> = c
                    .entries
                    .iter()
                    .map(|(key, val)| {
                        let v = self.transform_expression_scoped(val, &scope);
                        format!("        {:?} => {},", key, v)
                    })
                    .collect();
                lines.push(format!(
                    "// React: `const {}` — an object literal, which a module-scope\n\
                     // binding cannot hold, so a lookup function.\n\
                     \u{2609} rite {}(key: Any) -> Any! {{\n    \u{2325} key {{\n{}\n        _ => \u{2205},\n    }}\n}}",
                    c.name, name, arms.join("\n")
                ));
                known.insert(name.clone());
                emitted.insert(name);
                continue;
            }

            if value.trim() == "None" || value.trim() == "∅" {
                // Not emitted: nothing may depend on it.
                lines.push(format!(
                    "// {} — initialiser did not survive transformation. React: {}",
                    name, one_line
                ));
                continue;
            }

            // A module-scope binding has to be a CONSTANT expression, and most of
            // these are not — an object literal, a call, a lookup. Emitting them
            // anyway cost seven files ("expression is not constant") and ten
            // strict-clean ones, so only literal-shaped values are emitted.
            // `≔ x = None;` at module scope passes `check` and fails to
            // compile (§8.2.9) — and `None` is what the transform yields for
            // anything it could not read, so this arm was turning an
            // unreadable initialiser into a file that would not build. A
            // constant nobody can express is a comment.
            if is_constant_expression(&value) {
                lines.push(format!("≔ {} = {};", name, value));
                known.insert(name.clone());
                emitted.insert(name);
                continue;
            }

            // Not a constant and not a lookup table: an array literal, or a
            // call. Still a perfectly good FUNCTION BODY, and a bare reference
            // becomes a call (see `TransformConfig::value_constants`).
            //
            // Only when the body is SELF-CONTAINED, though. Emitting one that
            // names anything else trades a missing constant for a missing
            // dependency: `const WATCHTOWER_ROWS = [{ storageKey: keys.x }]`
            // brought `keys` with it, and cost two compiling files and seven
            // strict-clean ones the first time this was tried.
            if is_literal_only(&value) || free_identifiers_known(&value, &known) {
                lines.push(format!(
                    "// React: `const {}` — not a constant expression, so a function.\n\
                     \u{2609} rite {}() -> Any! {{\n    {}\n}}",
                    c.name, name, value
                ));
                known.insert(name.clone());
                emitted.insert(name);
            } else {
                // The body names something this module does not have, so it
                // cannot be emitted — but the NAME still has to resolve, or
                // every reference is an undefined variable and nothing that
                // mentions it compiles. The value is visibly missing instead.
                lines.push(format!(
                    "// {} — not a constant expression, and its dependencies are \
                     not available here. React: {}\n\u{2609} rite {}() -> Any! {{\n    \u{2205}\n}}",
                    name, one_line, name
                ));
                known.insert(name.clone());
                emitted.insert(name);
            }
        }

        let code = if lines.is_empty() {
            String::new()
        } else {
            format!("{}\n", lines.join("\n"))
        };
        (code, emitted)
    }

    /// Emit the helper functions this component reaches.
    ///
    /// A helper whose whole body is `return <expr>` translates in full. Anything
    /// else gets a declared signature and an untranslated body: the name
    /// resolves, the shape is right, and the React source sits above it in a
    /// comment so what is missing is visible rather than inferred from a
    /// compile error three files away.
    fn generate_helper_functions(
        &self,
        emitted_constants: &std::collections::HashSet<String>,
    ) -> String {
        // Only the constants that were actually EMITTED. Seeding this from
        // every extracted constant let a helper body name one that had been
        // dropped for a comment — `terminal·contains_key(s)` where `terminal`
        // is `new Set([…])`, which no module binding can hold.
        let mut known: std::collections::HashSet<String> =
            HOST_FUNCTIONS.iter().map(|s| s.to_string()).collect();
        known.extend(emitted_constants.iter().cloned());
        known.insert(to_snake_case(&self.spec.name));

        let taken: std::collections::HashSet<String> = std::iter::once(to_snake_case(&self.spec.name))
            .chain(
                self.spec
                    .source
                    .module_constants
                    .iter()
                    .map(|c| to_snake_case(&c.name)),
            )
            .collect();

        let mut scope = VNodeScope::default();
        scope.locals.extend(self.spec.source.module_scope.iter().cloned());

        // Translate what can be translated, in dependency order: a helper whose
        // body calls another helper can only be emitted once that one is.
        // Repeat until a pass adds nothing.
        //
        // The gate is the same one the module constants needed, and for the
        // same reason: emitting a body that names something undeclared trades a
        // missing function for a missing dependency, and the first attempt at
        // this cost 27 compiling files. A helper that does not pass it is
        // declared with an untranslated body — that resolves the name, which is
        // the whole point, and adds no dependency of its own.
        let mut pending: Vec<&HelperFunctionExtraction> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        for h in &self.spec.source.helpers {
            let name = to_snake_case(&h.name);
            if taken.contains(&name) || !seen.insert(name) {
                continue;
            }
            pending.push(h);
        }

        // Every pending helper is emitted — with a translated body if it passes
        // the gates below, and otherwise as a declared signature over `∅`. Its
        // NAME therefore resolves either way, which is what the known-names
        // gate is actually asking about.
        //
        // The gate did not know that: the stub loop runs after this one, so a
        // helper whose body named `fuzzyMatch` was rejected because
        // `fuzzy_match` had not been emitted yet, and it never would be —
        // `fuzzy_match` was itself waiting on something else. 165 of the 166
        // untranslated bodies in the Lares client failed this gate, and the
        // cascade is most of why.
        for h in &pending {
            known.insert(to_snake_case(&h.name));
        }

        let mut emitted: Vec<String> = Vec::new();
        let mut done: std::collections::HashSet<String> = std::collections::HashSet::new();
        loop {
            let mut progress = false;
            for h in &pending {
                let name = to_snake_case(&h.name);
                if done.contains(&name) {
                    continue;
                }
                let mut inner = scope.clone();
                let mut params: Vec<String> = Vec::new();
                let mut local_known = known.clone();
                let mut defaults: Vec<String> = Vec::new();
                for p in &h.parameters {
                    let pn = to_snake_case(&p.name);
                    // Every parameter, including the defaulted ones: Sigil has
                    // no defaults, and the call sites are padded to match (see
                    // `TransformConfig::fn_arity`).
                    params.push(format!("{}: Any", pn));
                    inner.locals.push(p.name.clone());
                    local_known.insert(pn.clone());
                    // Padded with `∅`, a defaulted parameter arrived as nothing
                    // and the default was lost: `formatRelativeTime(ms, now =
                    // Date.now())` compared every timestamp against 0, so
                    // everything was "just now". Sigil has no defaults, so the
                    // function applies its own on the way in.
                    if let Some(d) = &p.default_value {
                        let value = self.transform_expression_free(d, &inner);
                        // Truthiness, not `== None`: the padded argument is
                        // `∅`, which is the integer 0, and `None` is a fresh
                        // heap allocation — so `∅ == None` is false and the
                        // default never applied. Wider than JavaScript's rule,
                        // which fires only on `undefined`; for a defaulted
                        // parameter the difference is 0 and "", and taking the
                        // default for those beats never taking it at all.
                        defaults.push(format!(
                            "    \u{2254} {pn} = \u{2387} {pn}\u{00B7}to_bool() {{ {pn} }} \u{2389} {{ {value} }};"
                        ));
                    }
                }

                // A body that is one `return` is an expression; anything else
                // is statements. Both are translatable now — the statement
                // transform is what 206 of 346 helpers were waiting for.
                let body = match h.returns_expression.as_ref() {
                    Some(expr) => self.transform_expression_free(expr, &inner),
                    None => {
                        let Some(src) = stmt_transform::body_of_function(&h.source) else {
                            helper_skip(&name, "no function body found");
                            continue;
                        };
                        let config = self.statement_config(&inner);
                        let r = stmt_transform::transform_statements(&src, &config, "    ");
                        if r.code.trim().is_empty() {
                            helper_skip(&name, "statement transform produced nothing");
                            continue;
                        }
                        if r.complete {
                            r.code
                        } else {
                            format!(
                                "    // NOTE: parts of the React body are marked below, not translated.\n{}",
                                r.code
                            )
                        }
                    }
                };
                if !free_identifiers_known(&body, &local_known) {
                    let unknown = first_unknown(&body, &local_known);
                    helper_skip(&name, &format!("unknown name: {:?}", unknown));
                    if let Some(u) = &unknown {
                        helper_skip_line(&name, u, &body);
                    }
                    continue;
                }
                let body = if body.starts_with("    ") {
                    body
                } else {
                    format!("    {}", body)
                };
                let body = if defaults.is_empty() {
                    body
                } else {
                    format!("{}\n{}", defaults.join("\n"), body)
                };
                // The function as it will be emitted has to type-check, not just
                // parse. A body whose tail is uncertain — `w?.tickets` — under a
                // `-> Any!` signature parses and then fails `sigil check` with an
                // evidence mismatch. Both paths go through the same probe now;
                // only the statement one had it.
                let signature = format!("rite __probe({}) -> Any!", params.join(", "));
                if !parses_as_statements(&signature, &body) {
                    helper_skip(&name, "the emitted body does not parse or check");
                    continue;
                }
                emitted.push(format!(
                    "☉ rite {}({}) -> Any! {{\n{}\n}}",
                    name,
                    params.join(", "),
                    body
                ));
                known.insert(name.clone());
                done.insert(name);
                progress = true;
            }
            if !progress {
                break;
            }
        }

        // Everything else: a declared signature over an untranslated body.
        for h in &pending {
            let name = to_snake_case(&h.name);
            if done.contains(&name) {
                continue;
            }
            let params: Vec<String> = h
                .parameters
                .iter()
                .map(|p| format!("{}: Any", to_snake_case(&p.name)))
                .collect();
            let src: String = h
                .source
                .lines()
                .take(6)
                .map(|l| format!("//   {}", sanitize_comment(l)))
                .collect::<Vec<_>>()
                .join("\n");
            emitted.push(format!(
                "// TODO: body not translated — needs a statement transform. React:\n{}\n☉ rite {}({}) -> Any! {{\n    ∅\n}}",
                src,
                name,
                params.join(", ")
            ));
            done.insert(name);
        }

        if emitted.is_empty() {
            String::new()
        } else {
            format!("{}\n", emitted.join("\n"))
        }
    }

    // =========================================================================
    // Message Enum Generation
    // =========================================================================

    fn generate_message_enum(&self) -> String {
        let name = format!("{}Msg", self.spec.name);
        let mut variants = Vec::new();

        for msg in &self.spec.recommendations.messages {
            if let Some(payload) = &msg.payload {
                // Tuple payloads bind tight: `Expand(Any)`, not `Expand (Any)`.
                // Struct payloads keep the separating space: `Expand { id: Any }`.
                let sep = if payload.starts_with('(') { "" } else { " " };
                variants.push(format!("    {}{}{},", msg.name, sep, payload));
            } else {
                variants.push(format!("    {},", msg.name));
            }
        }

        format!(
            "☉ ᛈ {} {{\n{}\n}}",
            name,
            variants.join("\n")
        )
    }

    // =========================================================================
    // Actor Generation
    // =========================================================================

    fn generate_actor(&self) -> String {
        let mut sections = Vec::new();

        // Bodies first: they decide what the state block has to declare.
        //
        // The unknown-identifier rule turns any name an actor's body does not
        // recognise into `self.<name>` — deliberately, on the grounds that it is
        // "likely a state field we didn't detect". Nothing then declared it, so
        // the guess produced an undefined field instead of an undefined
        // variable: `self.branch_picker = { detail: self.detail, … }` where the
        // React handler's `detail` was a local the mutation walker flattened
        // away. Whatever the bodies reference through `self`, the actor
        // declares.
        let handlers = self.generate_message_handlers();
        let lifecycle = self.generate_lifecycle_handlers();
        // Which locals the handlers call through `self` — those are methods.
        // Which locals are called through `self` — those are methods. To a
        // fixed point: a method's own body calls other component functions
        // (`reload` calls `onPollSuccess`), and a name first seen there has to
        // become a method too, or the call is an undefined function.
        let mut bodies_owned: Vec<String> =
            vec![handlers.clone(), lifecycle.clone()];
        let mut self_called = std::collections::HashSet::new();
        let mut callbacks = String::new();
        for _ in 0..4 {
            let refs: Vec<&str> = bodies_owned.iter().map(|s| s.as_str()).collect();
            let next = self_called_names(&refs);
            let grown = !next.is_subset(&self_called);
            self_called.extend(next);
            callbacks = self.generate_callback_methods(&self_called, &refs);
            if !grown {
                break;
            }
            bodies_owned = vec![handlers.clone(), lifecycle.clone(), callbacks.clone()];
        }
        let bodies_owned = vec![handlers.clone(), lifecycle.clone(), callbacks.clone()];
        let bodies: Vec<&str> = bodies_owned.iter().map(|s| s.as_str()).collect();
        let view = self.generate_view_method(&self_called, &bodies);
        let constructor = if self.spec.recommendations.props_handling.fields.is_empty() {
            String::new()
        } else {
            self.generate_constructor()
        };

        // State fields
        let mut state_fields = self.generate_state_fields();
        let inferred = self.infer_missing_state_fields(&[
            handlers.as_str(),
            lifecycle.as_str(),
            callbacks.as_str(),
            view.as_str(),
        ]);
        if !inferred.is_empty() {
            if !state_fields.is_empty() {
                state_fields.push('\n');
            }
            state_fields.push_str(&inferred);
        }
        if !state_fields.is_empty() {
            sections.push(state_fields);
        }

        // Constructor if props
        if !constructor.is_empty() {
            sections.push(constructor);
        }

        if !handlers.is_empty() {
            sections.push(handlers);
        }

        if !lifecycle.is_empty() {
            sections.push(lifecycle);
        }

        if !callbacks.is_empty() {
            sections.push(callbacks);
        }

        sections.push(view);

        format!(
            "☉ actor {} {{\n{}\n}}",
            self.spec.name,
            sections.join("\n\n")
        )
    }

    /// Declare every `self.<name>` the generated bodies read that the state
    /// block does not already have.
    ///
    /// Method calls are excluded — `self.set_tab("todo")` names a method, not a
    /// field — and so is anything the constructor or state block declares.
    /// Everything left is a name the unknown-identifier rule guessed at, and a
    /// guess that compiles is worth more than one that does not.
    fn infer_missing_state_fields(&self, bodies: &[&str]) -> String {
        let declared: std::collections::HashSet<String> =
            self.field_inits().into_iter().map(|(n, _)| n).collect();

        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut out: Vec<String> = Vec::new();
        for body in bodies {
            let chars: Vec<char> = body.chars().collect();
            let is_word = |c: char| c.is_alphanumeric() || c == '_' || c == '$';
            let mut i = 0;
            while i + 5 <= chars.len() {
                if chars[i..].starts_with(&['s', 'e', 'l', 'f', '.'])
                    && (i == 0 || !is_word(chars[i - 1]))
                {
                    let start = i + 5;
                    let mut j = start;
                    while j < chars.len() && is_word(chars[j]) {
                        j += 1;
                    }
                    let name: String = chars[start..j].iter().collect();
                    // `self.foo(` is a method; `self.foo` is a field.
                    let is_call = chars.get(j) == Some(&'(');
                    if !name.is_empty()
                        && !is_call
                        && !declared.contains(&name)
                        && seen.insert(name.clone())
                    {
                        out.push(format!(
                            "    state {}: Any~ = ∅,  // inferred: referenced but never declared",
                            name
                        ));
                    }
                    i = j;
                    continue;
                }
                i += 1;
            }
        }
        out.join("\n")
    }

    /// A `useState` initialiser, as something Sigil can evaluate.
    ///
    /// The extractor hands this back as raw JS source. Most of it is a literal
    /// and passes through; the cases that do not are a placeholder comment, a
    /// lazy initialiser (`useState(storedView)` — a function reference React
    /// calls for you), and a bare name from module scope. Anything left that
    /// this cannot account for becomes `None` rather than an identifier no
    /// binding in the file defines.
    fn resolve_initial(&self, raw: &str) -> String {
        let trimmed = raw.trim();
        match trimmed {
            "null" | "undefined" | "/* expr */" | "/* expression */" => return "None".to_string(),
            other if other.contains("/*") => return "None".to_string(),
            _ => {}
        }

        let is_bare_ident = !trimmed.is_empty()
            && trimmed
                .chars()
                .all(|c| c.is_alphanumeric() || c == '_' || c == '$')
            && !trimmed.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(true);
        if !is_bare_ident || matches!(trimmed, "true" | "false") {
            return trimmed.to_string();
        }

        let snake = to_snake_case(trimmed);
        if self
            .spec
            .source
            .module_functions
            .iter()
            .any(|f| to_snake_case(f) == snake)
        {
            // Lazy initialiser: React calls it once, so Sigil calls it here.
            return format!("{}()", snake);
        }
        if self.value_constant_names().iter().any(|c| *c == snake) {
            return format!("{}()", snake);
        }
        if self
            .spec
            .source
            .module_constants
            .iter()
            .any(|c| to_snake_case(&c.name) == snake)
        {
            return snake;
        }
        "None".to_string()
    }

    /// Every field the actor declares, paired with its initial value, in the
    /// order they are emitted: `useState` fields first, then props.
    ///
    /// `generate_state_fields` and `generate_constructor` have to agree on this
    /// list exactly — the constructor builds a struct literal, and a literal
    /// that names a field the actor does not declare (or omits one it does) is
    /// a different kind of broken than the `self.<x> = x` it replaced.
    fn field_inits(&self) -> Vec<(String, String)> {
        let mut out: Vec<(String, String)> = self
            .spec
            .recommendations
            .state_fields
            .iter()
            .map(|field| {
                (
                    to_snake_case(&field.to_field),
                    self.resolve_initial(&field.initial_value),
                )
            })
            .collect();
        // A duplicated key is a malformed struct literal, and two `state`
        // declarations of the same name are already wrong upstream.
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        out.retain(|(n, _)| seen.insert(n.clone()));
        let declared = seen;
        for f in &self.spec.recommendations.props_handling.fields {
            let name = to_snake_case(&f.name);
            if !declared.contains(&name) {
                out.push((name, "∅".to_string()));
            }
        }
        out
    }

    fn generate_state_fields(&self) -> String {
        let fields: Vec<String> = self.spec.recommendations.state_fields.iter()
            .map(|field| {
                let initial = self.resolve_initial(&field.initial_value);
                // The field name has to match every reference to it. The view is
                // generated through the expression transform, which snake-cases
                // and keyword-escapes, so a declaration left in React's camelCase
                // named something no `self.<x>` in the file ever reached.
                format!(
                    "    state {}: {}{} = {},",
                    to_snake_case(&field.to_field),
                    field.field_type,
                    field.evidentiality,
                    initial
                )
            })
            .collect();

        // Props are fields too. `rite new(…)` assigns every prop to `self.<name>`
        // and the view reads them back, but only `useState` fields were ever
        // declared — so 204 references across the client resolved to nothing, and
        // the constructor's own assignments were the first of them. Invisible to
        // `sigil check`, which does not resolve names (S23); the WASM backend
        // reports them as "undefined variable: self".
        let declared: std::collections::HashSet<String> = self
            .spec
            .recommendations
            .state_fields
            .iter()
            .map(|f| to_snake_case(&f.to_field))
            .collect();
        let prop_fields: Vec<String> = self
            .spec
            .recommendations
            .props_handling
            .fields
            .iter()
            .filter(|f| !declared.contains(&to_snake_case(&f.name)))
            .map(|f| {
                format!(
                    "    state {}: {}~ = ∅,",
                    to_snake_case(&f.name),
                    normalize_prop_type(&f.field_type)
                )
            })
            .collect();

        let fields: Vec<String> = fields.into_iter().chain(prop_fields).collect();
        fields.join("\n")
    }

    fn generate_constructor(&self) -> String {
        let props = &self.spec.recommendations.props_handling;

        let params: Vec<String> = props.fields.iter()
            .map(|f| format!("{}: {}", f.name, normalize_prop_type(&f.field_type)))
            .collect();

        // A constructor is an associated function: it has no receiver, so its
        // body cannot assign through `self`. Emitting `self.<prop> = <prop>;`
        // here made `new` the single largest source of "undefined variable:
        // self" in the generated client — seven files, none of which were
        // wrong anywhere else. Build the value instead, the way every
        // hand-written Sigil constructor does.
        let prop_params: std::collections::HashSet<String> = props
            .fields
            .iter()
            .map(|f| to_snake_case(&f.name))
            .collect();
        let inits: Vec<String> = self
            .field_inits()
            .into_iter()
            .map(|(name, init)| {
                let value = if prop_params.contains(&name) {
                    // The parameter keeps React's spelling; the field is
                    // snake_cased. Find the parameter this field came from.
                    props
                        .fields
                        .iter()
                        .find(|f| to_snake_case(&f.name) == name)
                        .map(|f| f.name.clone())
                        .unwrap_or_else(|| init.clone())
                } else {
                    init
                };
                format!("            {}: {},", name, value)
            })
            .collect();

        format!(
            "    rite new({}) -> This! {{\n        This {{\n{}\n        }}\n    }}",
            params.join(", "),
            inits.join("\n")
        )
    }

    /// A handler's body, translated whole, or `None` when any statement in it
    /// could not be — in which case the caller falls back to the flat list and
    /// marks it. Half a body is not better here: the assignments the statement
    /// transform did emit would sit alongside the flat list's copies of the
    /// same ones.
    fn translate_handler_body(&self, msg: &MessageRecommendation) -> Option<String> {
        let source = msg.body_source.as_ref()?;
        if source.trim().is_empty() {
            return None;
        }
        // A handler that calls into a service actor is not this transform's to
        // write: `addMessage(…)` from `useChat()` is `ChatService ! AddMessage`,
        // a message send, and the statement walk has no way to know that — it
        // would emit `self.add_message(…)`, a method the actor does not have.
        // The flat path below still emits the sends.
        if !msg.service_calls.is_empty() {
            return None;
        }
        let mut scope = VNodeScope::default();
        scope.locals.extend(self.spec.source.module_scope.iter().cloned());
        for p in &msg.param_bindings {
            scope.locals.push(p.clone());
        }
        let mut config = self.build_transform_config(&scope, self.is_actor);
        // The payload names the handler's parameters bind to.
        config.locals.extend(msg.param_bindings.iter().map(|p| to_snake_case(p)));

        // `body_summary` is the handler as React wrote it — `async () => { … }`
        // for a `useCallback`, not its block. Handing the arrow to the statement
        // transform made the whole handler one closure literal that nothing
        // called.
        let body = stmt_transform::body_of_function(source)
            .unwrap_or_else(|| source.to_string());
        let r = stmt_transform::transform_statements(&body, &config, "        ");
        if r.code.trim().is_empty()
            || !parses_as_statements("rite __probe()", &r.code)
        {
            return None;
        }

        // The payload bindings still have to be introduced, exactly as the
        // flat path does.
        let mut lines: Vec<String> = Vec::new();
        for (i, p) in msg.param_bindings.iter().enumerate() {
            let name = to_snake_case(p);
            let value = if msg.payload.is_some() && i == 0 {
                "msg.0".to_string()
            } else {
                "∅".to_string()
            };
            lines.push(format!("        ≔ {} = {};", name, value));
        }
        if !r.complete {
            // What the transform could not translate is a `// React:` line in
            // the body; say up front that there is one.
            lines.insert(
                0,
                "        // NOTE: parts of the React handler are marked below, not translated."
                    .to_string(),
            );
        }
        lines.push(r.code);
        Some(lines.join("\n"))
    }

    fn generate_message_handlers(&self) -> String {
        let handlers: Vec<String> = self.spec.recommendations.messages.iter()
            .map(|msg| {
                let mut body_parts: Vec<String> = Vec::new();

                // S30: the handler's own body, statement by statement, when the
                // whole of it translates. The alternative below reduces it to a
                // flat list of state assignments, which loses the branches — a
                // handler that wrote one field on success and another on failure
                // emitted both, unconditionally, in source order.
                if let Some(body) = self.translate_handler_body(msg) {
                    return format!(
                        "    on {} {{\n{}\n    }}",
                        msg.name,
                        body
                    );
                }

                // Add service calls (hook-returned function calls -> actor messages)
                for call in &msg.service_calls {
                    let args = if call.args.is_empty() {
                        String::new()
                    } else {
                        format!("({})", call.args.join(", "))
                    };
                    body_parts.push(format!("        {} ! {}{};", call.service, call.method, args));
                }

                // Add state changes. The structured form carries the value as
                // React wrote it, so it goes through the same expression transform
                // as the view — `setCollapsed(c => !c)` becomes `self.collapsed =
                // ¬self.collapsed` instead of a JavaScript arrow pasted into an
                // actor. `state_changes` is the pre-transform fallback, used for
                // the setter messages whose value is the payload itself.
                if !msg.state_assignments.is_empty() {
                    for a in &msg.state_assignments {
                        // The handler's own parameters are in scope for its body;
                        // without them every `setName(tmuxName)` transformed to
                        // `self.tmux_name`, a field that does not exist, instead of
                        // the value just bound from the payload.
                        let mut scope = VNodeScope::default();
                        for p in &msg.param_bindings {
                            scope.locals.push(to_snake_case(p));
                        }
                        // A trailing comma from a multi-line call survives the span.
                        let src = a.value.trim().trim_end_matches(',').trim();
                        let value = match &a.updater_param {
                            Some(param) => {
                                // The updater's parameter IS the current value.
                                let p = to_snake_case(param);
                                scope.locals.push(p.clone());
                                let body = self.transform_expression_scoped(src, &scope);
                                replace_ident(&body, &p, &format!("self.{}", a.field))
                            }
                            None => self.transform_expression_scoped(src, &scope),
                        };
                        // Some values carry JavaScript the transform has no Sigil
                        // spelling for — optional chaining, nullish coalescing,
                        // spread, a nested arrow, `instanceof`. `generate_locals`
                        // binds those to ∅; a state field already holds a value, so
                        // overwriting it with ∅ would be worse than not writing it.
                        // Record what React did and leave the field alone.
                        // A value that is still an arrow is a functional updater
                        // `split_updater` declined — a block body, or more than
                        // one parameter. The transform renders it as a perfectly
                        // valid Sigil closure, so the parse probe below waves it
                        // through, and the field ends up holding a function.
                        let unread_updater = a.updater_param.is_none() && src.contains("=>");
                        // JS spread. The parse probe cannot catch this one: `...`
                        // is valid Sigil, it just means something else there, so
                        // `[...cur, tag]` compiled and did the wrong thing.
                        let spread = value.contains("...");
                        if unread_updater || spread || !parses_as_expression(&value) {
                            let one_line: String =
                                src.split_whitespace().collect::<Vec<_>>().join(" ");
                            let one_line = if one_line.chars().count() > 100 {
                                one_line.chars().take(97).collect::<String>() + "..."
                            } else {
                                one_line
                            };
                            body_parts.push(format!(
                                "        // TODO: self.{} = {}",
                                a.field, one_line
                            ));
                        } else {
                            body_parts.push(format!("        self.{} = {};", a.field, value));
                        }
                    }
                }

                // Add state changes. When the structured form covered them, only
                // the `// React:` notes for mutations it could not structure are
                // still wanted — the assignments themselves would be duplicates.
                let structured = !msg.state_assignments.is_empty();
                for change in &msg.state_changes {
                    // Transform placeholder to valid expression
                    let change = change.replace("/* new value */", "msg.0");
                    // An untranslatable mutation comes through as a `// React:`
                    // note; a statement terminator after it is noise.
                    if change.trim_start().starts_with("//") {
                        body_parts.push(format!("        {}", change));
                    } else if !structured {
                        body_parts.push(format!("        {};", change));
                    }
                }

                // Include inlined effects if any
                for effect in &self.spec.recommendations.effects {
                    if effect.strategy == EffectStrategy::Inline {
                        if effect.inline_in.as_ref().map(|i| i.contains(&msg.name.to_lowercase())).unwrap_or(false) {
                            body_parts.push(format!("        // Effect: {}", effect.reasoning));
                        }
                    }
                }

                // If no body parts, say what the React code actually did rather than
                // leaving an anonymous TODO. Most of these are a child telling its
                // parent something — `onClick={() => onOpenSession(id)}` has no local
                // state to change, and a bare "TODO: implement" hides that.
                // The handler's parameters, bound to the payload it was declared
                // with. Its state changes are written in React's terms — a
                // handler `(next) => setView(next)` becomes `self.view = next`,
                // and `next` is nothing here until it is bound. Prepended only
                // when there is a body to bind them for; alone they would suppress
                // the `// React:` signpost below without saying anything.
                if !body_parts.is_empty() && msg.flattened_control_flow {
                    body_parts.insert(
                        0,
                        "        // NOTE: the React handler branched; these assignments \
                         are flattened.".to_string(),
                    );
                }

                if !body_parts.is_empty() {
                    // The payload fills the first N parameters; the rest of the
                    // handler's signature has nowhere to come from — the event
                    // site carries one value at most — so they bind ∅. Leaving
                    // them out entirely is what made them `self.` fields.
                    let arity = msg
                        .payload
                        .as_deref()
                        .filter(|p| p.starts_with('('))
                        .map(|p| p.trim_start_matches('(').trim_end_matches(')').split(',').count())
                        .unwrap_or(0);
                    let bindings: Vec<String> = msg
                        .param_bindings
                        .iter()
                        .enumerate()
                        .map(|(i, p)| {
                            if i < arity {
                                format!("        ≔ {} = msg.{};", to_snake_case(p), i)
                            } else {
                                format!(
                                    "        ≔ {} = ∅;  // no payload slot for this parameter",
                                    to_snake_case(p)
                                )
                            }
                        })
                        .collect();
                    body_parts.splice(0..0, bindings);
                }

                let body = if body_parts.is_empty() {
                    let mut lines = Vec::new();
                    if !msg.from_handler.is_empty() {
                        // A Sigil comment ends at the newline, and an inline handler
                        // is frequently a multi-line arrow body — pasting it raw
                        // spilled JS into the actor and broke nine files. Collapse to
                        // one line and cap it; this is a signpost, not the source.
                        let mut src: String =
                            msg.from_handler.split_whitespace().collect::<Vec<_>>().join(" ");
                        if src.chars().count() > 100 {
                            src = src.chars().take(97).collect::<String>() + "...";
                        }
                        lines.push(format!("        // React: {}", src));
                    }
                    let callback = crate::migrate::react::spec::derive_invoked_callback(
                        &msg.from_handler,
                    );
                    let is_prop_callback = callback.as_ref().is_some_and(|c| {
                        c.strip_prefix("on")
                            .and_then(|r| r.chars().next())
                            .is_some_and(|c| c.is_uppercase())
                    });
                    if is_prop_callback {
                        lines.push(format!(
                            "        // `{}` was a prop callback — forward this to the parent actor.",
                            callback.unwrap_or_default()
                        ));
                    } else {
                        lines.push("        // TODO: implement".to_string());
                    }
                    lines.join("\n")
                } else {
                    body_parts.join("\n")
                };

                format!("    on {} {{\n{}\n    }}", msg.name, body)
            })
            .collect();

        handlers.join("\n\n")
    }

    fn generate_lifecycle_handlers(&self) -> String {
        // One handler per lifecycle event. A component with several mount-time
        // effects — `useEffect(…, [])` more than once, which is ordinary React —
        // produced `on Mount` twice, and two handlers of the same name compile to
        // a function with a second body appended to it: "operators remaining
        // after end of function". Their reasons are merged instead.
        let mut order: Vec<String> = Vec::new();
        let mut reasons: std::collections::HashMap<String, Vec<String>> =
            std::collections::HashMap::new();

        for effect in &self.spec.recommendations.effects {
            if effect.strategy == EffectStrategy::Lifecycle {
                if let Some(event) = &effect.lifecycle_event {
                    if !reasons.contains_key(event) {
                        order.push(event.clone());
                    }
                    reasons
                        .entry(event.clone())
                        .or_default()
                        .push(effect.reasoning.clone());
                }
            }
        }

        let handlers: Vec<String> = order
            .into_iter()
            .map(|event| {
                let body: Vec<String> = reasons
                    .remove(&event)
                    .unwrap_or_default()
                    .into_iter()
                    .map(|r| format!("        // {}", r))
                    .collect();
                format!("    on {} {{\n{}\n    }}", event, body.join("\n"))
            })
            .collect();

        handlers.join("\n\n")
    }

    /// The component body's `const` bindings, rendered as Sigil `≔` statements.
    ///
    /// React destructures props into locals and derives values from them; those
    /// statements used to be dropped, so the view referred to names nothing bound.
    /// `repo` alone appeared 80 times across the generated Lares client, resolving
    /// to nothing — invisible to `sigil check`, which does not resolve names, and
    /// caught by `--strict`.
    /// A `useCallback` / `useMemo` local as a real Sigil binding.
    ///
    /// `useCallback` becomes a closure — the callers spell it `name(args)`, so
    /// a closure is exactly right. `useMemo` becomes its value, and only when
    /// the callback is a concise arrow: a block body would need a block
    /// expression whose tail is the `return`, and the statement transform emits
    /// `⤺`, which is not a tail. Anything that does not translate whole returns
    /// `None` so the caller's ∅-with-the-React-beside-it arm still applies —
    /// half a body here would be worse than none.
    fn translate_hook_local(
        &self,
        local: &crate::migrate::react::extraction::LocalBinding,
        scope: &VNodeScope,
        indent: &str,
        name: &str,
    ) -> Option<String> {
        let is_memo = stmt_transform::hook_callback(&local.init, &["useMemo"]).is_some();
        let hook = stmt_transform::hook_callback(&local.init, &["useCallback", "useMemo"])?;
        if is_memo && hook.is_block {
            return None;
        }

        let mut inner = self.build_transform_config(scope, self.is_actor);
        inner.locals.extend(hook.params.iter().cloned());
        let body_indent = format!("{}    ", indent);
        let r = stmt_transform::transform_statements(&hook.body, &inner, &body_indent);
        if r.code.trim().is_empty() {
            return None;
        }

        if is_memo {
            // A concise arrow: the body is the value.
            let probe = format!("rite __probe() -> Any! {{\n{}\n}}", r.code);
            if !parses_as_statements("rite __probe() -> Any!", &r.code) {
                let _ = probe;
                return None;
            }
            return Some(format!("{}≔ {} = {};", indent, name, r.code.trim()));
        }

        let params = if hook.params.is_empty() {
            "||".to_string()
        } else {
            format!("|{}|", hook.params.join(", "))
        };
        let body = if hook.is_block {
            format!("{{\n{}\n{}}}", r.code, indent)
        } else {
            format!("{{\n{}\n{}}}", r.code, indent)
        };
        let candidate = format!("{}≔ {} = {} {};", indent, name, params, body);
        // The same parse-and-typecheck probe the helper paths use: a body whose
        // tail is uncertain parses and then fails `sigil check`.
        if !parses_as_statements("rite __probe()", candidate.trim()) {
            return None;
        }
        Some(candidate)
    }

    fn generate_locals(
        &self,
        indent: &str,
        scope: &mut VNodeScope,
        self_called: &std::collections::HashSet<String>,
        bodies: &[&str],
    ) -> String {
        let locals = self.spec.source.extraction.locals.clone();
        if locals.is_empty() {
            return String::new();
        }
        let mut seen = std::collections::HashSet::new();
        let mut lines = Vec::new();
        for local in &locals {
            let name = to_snake_case(&local.name);
            // A prop of the same name already binds it; re-binding would shadow the
            // parameter with itself.
            if self
                .spec
                .recommendations
                .props_handling
                .fields
                .iter()
                .any(|f| f.name == name)
                || !seen.insert(name.clone())
            {
                continue;
            }
            // Transform against the scope BEFORE adding this name, so a local may
            // refer to earlier locals but not to itself. Without accumulating them,
            // every reference to a local from a later local or from the view body
            // was rewritten to `self.<name>` by the unknown-identifier rule, which
            // is how `self.current_sprint` and `self.summarize(…)` appeared for
            // things that are not fields at all.
            let init = self.transform_expression_scoped(&local.init, scope);
            scope.locals.push(name.clone());

            // Some initialisers survive the transform carrying JS that Sigil has no
            // spelling for — optional chaining (`a?.b`) and spread (`...x`). Emitting
            // those verbatim breaks the file. Bind the name to ∅ instead and keep the
            // original beside it: the name still resolves, which is the point, and
            // the value is visibly missing rather than silently wrong.
            // A custom hook's return has nothing to translate — only a name to
            // bind. Treated the same way as the JavaScript this transform has no
            // spelling for: bound to ∅, with the React beside it.
            // A component-level `useCallback` in an ACTOR is a method, not a
            // view local: the handlers call it as `self.refresh()`, and they are
            // not in the view's scope. Emitted by `generate_callback_methods`;
            // skipped here, and deliberately NOT added to `scope.locals`, so a
            // reference from the view becomes `self.refresh()` too.
            if self.is_actor
                && (self.callback_method(local, self_called).is_some()
                    || self.callback_method_stub(local, self_called, bodies).is_some())
            {
                scope.locals.retain(|l| l != &name);
                continue;
            }

            // `useCallback(fn, deps)` and `useMemo(() => expr, deps)` are the
            // two React hooks whose value the migration can actually produce: a
            // closure and an expression. They were reaching the ∅ arm below, so
            // `refresh()` in a handler called a name bound to ∅ — an undefined
            // function in the WASM build, and silently nothing before that.
            if let Some(hook) = self.translate_hook_local(local, scope, indent, &name) {
                lines.push(hook);
                continue;
            }

            let untranslatable = local.opaque || init.contains("?.") || init.contains("...");
            if untranslatable {
                let one_line: String = local.init.split_whitespace().collect::<Vec<_>>().join(" ");
                let capped = if one_line.chars().count() > 90 {
                    one_line.chars().take(87).collect::<String>() + "..."
                } else {
                    one_line
                };
                lines.push(format!("{}≔ {} = ∅;  // React: {}", indent, name, capped));
            } else {
                lines.push(format!("{}≔ {} = {};", indent, name, init));
            }
        }
        if lines.is_empty() {
            String::new()
        } else {
            format!("{}\n", lines.join("\n"))
        }
    }

    /// A component-level `useCallback`, as an actor method.
    ///
    /// React's `useCallback` at component scope is a method of the component:
    /// handlers call it, the view calls it, and both spell it `self.<name>` in
    /// an actor. Emitted as a view local instead, `self.refresh()` in a handler
    /// named a method no actor had — an undefined function in the WASM build.
    /// `useMemo` is a value, not a function, so it stays a local.
    fn callback_method(
        &self,
        local: &crate::migrate::react::extraction::LocalBinding,
        self_called: &std::collections::HashSet<String>,
    ) -> Option<String> {
        let name = to_snake_case(&local.name);
        // Only a name the handlers actually call through `self` becomes a
        // method. A local bound to a function is not always a method — a sort
        // comparator is a value, and turning it into one would break the call
        // that passes it — so the handlers decide.
        if !self_called.contains(&name) {
            return None;
        }
        let hook = stmt_transform::hook_callback(&local.init, &["useCallback"])
            .or_else(|| stmt_transform::function_local(&local.init))?;

        let mut scope = VNodeScope::default();
        scope.locals.extend(self.spec.source.module_scope.iter().cloned());
        let mut config = self.build_transform_config(&scope, true);
        config.locals.extend(hook.params.iter().cloned());

        let r = stmt_transform::transform_statements(&hook.body, &config, "        ");
        if r.code.trim().is_empty() {
            return None;
        }

        let mut params = vec!["self".to_string()];
        params.extend(hook.params.iter().map(|p| format!("{}: Any", p)));
        let signature = format!("    rite {}({}) -> Any!", name, params.join(", "));
        let body = if hook.is_block {
            r.code.clone()
        } else {
            format!("        ⤺ {};", r.code.trim())
        };
        // The same parse-and-typecheck probe the helper paths use, in the
        // impl block a method actually lives in.
        if !parses_as_method(signature.trim(), &body) {
            return None;
        }

        let note = if r.complete {
            String::new()
        } else {
            "        // NOTE: parts of the React callback are marked below, not translated.\n"
                .to_string()
        };
        Some(format!("{} {{\n{}{}\n    }}", signature, note, body))
    }

    /// The method a handler calls, when its body would not compile.
    ///
    /// The handlers call component-level functions as `self.<name>(…)`. If the
    /// body cannot be translated the method still has to EXIST — otherwise the
    /// call is an undefined function and nothing in the actor compiles at all.
    /// The React is kept beside it, so the missing body is visible.
    fn callback_method_stub(
        &self,
        local: &crate::migrate::react::extraction::LocalBinding,
        self_called: &std::collections::HashSet<String>,
        bodies: &[&str],
    ) -> Option<String> {
        let name = to_snake_case(&local.name);
        if !self_called.contains(&name) {
            return None;
        }
        // The parameters, from the binding when it is a function, and from the
        // call sites otherwise: `const { onPollSuccess } = usePollError()` is a
        // callback out of a custom hook, opaque to the migration, and the
        // handlers still call it.
        let param_names: Vec<String> = match stmt_transform::hook_callback(&local.init, &["useCallback"])
            .or_else(|| stmt_transform::function_local(&local.init))
        {
            Some(hook) => hook.params,
            None => {
                let argc = call_arity(bodies, &name)?;
                (0..argc).map(|i| format!("__a{}", i)).collect()
            }
        };
        let mut params = vec!["self".to_string()];
        params.extend(param_names.iter().map(|p| format!("{}: Any", p)));
        let one_line: String = local.init.split_whitespace().collect::<Vec<_>>().join(" ");
        let capped = if one_line.chars().count() > 90 {
            one_line.chars().take(87).collect::<String>() + "..."
        } else {
            one_line
        };
        Some(format!(
            "    rite {}({}) -> Any! {{\n        // React: {}\n        \u{2205}\n    }}",
            name,
            params.join(", "),
            capped.replace('\n', " ")
        ))
    }

    fn generate_callback_methods(
        &self,
        self_called: &std::collections::HashSet<String>,
        bodies: &[&str],
    ) -> String {
        if !self.is_actor {
            return String::new();
        }
        let mut seen = std::collections::HashSet::new();
        let methods: Vec<String> = self
            .spec
            .source
            .extraction
            .locals
            .iter()
            .filter(|l| seen.insert(to_snake_case(&l.name)))
            .filter_map(|l| {
                self.callback_method(l, self_called)
                    .or_else(|| self.callback_method_stub(l, self_called, bodies))
            })
            .collect();
        methods.join("\n\n")
    }

    fn generate_view_method(
        &self,
        self_called: &std::collections::HashSet<String>,
        bodies: &[&str],
    ) -> String {
        let jsx = &self.spec.source.extraction.jsx;
        let mut scope = VNodeScope::default();

        // Locals first: they extend the scope the body is generated against.
        let locals = self.generate_locals("        ", &mut scope, self_called, bodies);

        let body = if let Some(root) = &jsx.root {
            self.generate_vnode(root, 2, &scope)
        } else {
            "        VNode·div()".to_string()
        };

        // `☉`: a component's view is the one thing outside the module has to
        // be able to call. Without it the actor compiled, exported its
        // handlers and its dispatcher, and offered no way to render anything.
        format!("    ☉ rite view(self) -> VNode! {{\n{}{}\n    }}", locals, body)
    }

    // =========================================================================
    // Function Generation (for pure components)
    // =========================================================================

    fn generate_function(&self) -> String {
        let props = &self.spec.recommendations.props_handling;

        let params = if props.fields.is_empty() {
            String::new()
        } else {
            props.fields.iter()
                .map(|f| {
                    // Handle rest parameters: ...props -> props: Vec<Any>
                    if f.name.starts_with("...") {
                        let name = &f.name[3..]; // Remove "..." prefix
                        format!("{}: Vec<Any>", name)
                    } else {
                        format!("{}: {}", f.name, normalize_prop_type(&f.field_type))
                    }
                })
                .collect::<Vec<_>>()
                .join(", ")
        };

        let jsx = &self.spec.source.extraction.jsx;
        let mut scope = VNodeScope::default();
        let locals = self.generate_locals(
            "    ",
            &mut scope,
            &std::collections::HashSet::new(),
            &[],
        );
        let body = if let Some(root) = &jsx.root {
            self.generate_vnode(root, 1, &scope)
        } else {
            "    VNode·div()".to_string()
        };

        format!(
            "☉ rite {}({}) -> VNode! {{\n{}{}\n}}",
            to_snake_case(&self.spec.name),
            params,
            locals,
            body
        )
    }

    // =========================================================================
    // VNode Builder Generation
    // =========================================================================

    fn generate_vnode(&self, node: &JsxNode, indent: usize, scope: &VNodeScope) -> String {
        let pad = "    ".repeat(indent);

        match &node.node_type {
            JsxNodeType::Element { tag, is_component, attributes, children } => {
                self.generate_element_vnode(tag, *is_component, attributes, children, indent, scope)
            }
            JsxNodeType::Fragment { children } => {
                self.generate_fragment_vnode(children, indent, scope)
            }
            JsxNodeType::Text { value } => {
                format!("{}·text_child(\"{}\")", pad, escape_string(value))
            }
            JsxNodeType::Expression { code } => {
                let expr = self.transform_expression_scoped(code, scope);
                // A call to a SIBLING COMPONENT in a child position is a node,
                // not text. `{worktreeChips(status, branch)}` became
                // `·text_child(worktree_chips(…)·to_string())`, so a VNode
                // handle rendered into the document as the decimal of its own
                // address — a bare five-digit number in the middle of a card.
                // `component_props` is every component this migration knows.
                if self.renders_a_node(&expr) {
                    return format!("{}·child({})", pad, expr);
                }
                // Everything else is interpolated text.
                format!("{}·text_child({}·to_string())", pad, expr)
            }
            JsxNodeType::Conditional { condition, consequent, alternate } => {
                let cond_expr = self.transform_expression_scoped(condition, scope);
                let cons = self.generate_vnode(consequent, indent, scope);
                if let Some(alt) = alternate {
                    let alt_code = self.generate_vnode(alt, indent, scope);
                    format!("{}·when_else({}, {}, {})", pad, cond_expr, cons.trim(), alt_code.trim())
                } else {
                    format!("{}·when({}, {})", pad, cond_expr, cons.trim())
                }
            }
            JsxNodeType::Map { iterable, item_name, key_expr: _, body } => {
                let iter_expr = self.transform_expression_scoped(iterable, scope);
                // Create new scope with iterator variable
                let mut inner_scope = scope.clone();
                inner_scope.locals.push(item_name.clone());
                let body_code = self.generate_vnode(body, indent + 1, &inner_scope);
                // The comment carries the raw JS iterable, and a Sigil comment ends at
                // the newline — so a multi-line one (`epicOptions\n.filter(e => …)`)
                // spilled its tail out as code. Collapse and cap it.
                let mut iter_src: String =
                    iterable.split_whitespace().collect::<Vec<_>>().join(" ");
                if iter_src.chars().count() > 80 {
                    iter_src = iter_src.chars().take(77).collect::<String>() + "...";
                }
                format!(
                    // `·map(…)` directly. `.iter()` and `.collect()` are Rust's
                    // shape, not Sigil's: the backend dispatches `map` on the
                    // array itself (morpheme.array_map), and `iter`/`collect`
                    // are two more lowercase names for it to fail to resolve —
                    // silently, under S34.
                    "{pad}// Map: ∀ {item} ∈ {iter}\n{pad}·children({iter_expr}·map(|{item}| {body}))",
                    pad = pad,
                    item = item_name,
                    iter = iter_src,
                    iter_expr = iter_expr,
                    body = body_code.trim()
                )
            }
        }
    }

    /// A JSX attribute value as the string a DOM attribute actually holds.
    ///
    /// `Qliphoth::attr` takes `&str`. React's `step={0.5}` and `rows={3}` are
    /// numbers, and passing one through unchanged handed the host the f64 bit
    /// pattern where a string handle belonged. A literal is emitted as its own
    /// text; anything else goes through `·to_string()`.
    fn attr_value_as_string(&self, code: &str, scope: &VNodeScope) -> String {
        let transformed = self.transform_expression_scoped(code, scope);
        let trimmed = transformed.trim();
        if trimmed.starts_with('"') && trimmed.ends_with('"') && trimmed.len() >= 2 {
            return transformed;
        }
        // A bare numeric or boolean literal is known here; quote it rather than
        // emitting a conversion the runtime has to do on every render.
        if trimmed.parse::<f64>().is_ok() {
            return format!("\"{}\"", trimmed);
        }
        if trimmed == "true" || trimmed == "false" {
            return format!("\"{}\"", trimmed);
        }
        format!("({})·to_string()", transformed)
    }

    /// The VALUE of a JSX attribute, with no `·attr(…)` wrapper — for passing
    /// a child component's props as arguments.
    fn generate_attribute_value(&self, attr: &JsxAttribute, scope: &VNodeScope) -> String {
        match &attr.value {
            JsxAttributeValue::String { value } => format!("{:?}", value),
            JsxAttributeValue::Expression { code } => {
                self.transform_expression_scoped(code, scope)
            }
            _ => "∅".to_string(),
        }
    }

    fn generate_element_vnode(
        &self,
        tag: &str,
        is_component: bool,
        attributes: &[JsxAttribute],
        children: &[JsxNode],
        indent: usize,
        scope: &VNodeScope,
    ) -> String {
        let pad = "    ".repeat(indent);
        if is_component {
            if self.component_props.is_empty() {
                // No sibling components known — the per-file layout, where a
                // cross-file call could not resolve anyway.
            } else if !self.component_props.contains_key(tag) {
                // A component with no spec of its own: a React context
                // provider, or something defined outside the walked tree. There
                // is nothing to call.
                return format!(
                    "{}// `{}` has no generated component to call\n{}VNode·fragment()",
                    pad, tag, pad
                );
            }
            if let Some(shape) = self.component_props.get(tag) {
                if shape.is_actor {
                    // A child ACTOR is not a function call. Mounting one is a
                    // Qliphoth runtime concern — the parent would send it
                    // messages — and there is no expression for it here.
                    return format!(
                        "{}// child actor `{}` — mounting one is a runtime concern, not a call\n{}VNode·fragment()",
                        pad, tag, pad
                    );
                }
                let params = &shape.props;
                // A call, with the JSX attributes matched onto the component's
                // own parameters by name. Anything the call site does not
                // supply is ∅ — Sigil has no default arguments.
                let args: Vec<String> = params
                    .iter()
                    .map(|p| {
                        attributes
                            .iter()
                            .find(|a| to_snake_case(&a.name) == to_snake_case(p))
                            .map(|a| self.generate_attribute_value(a, scope))
                            .unwrap_or_else(|| "∅".to_string())
                    })
                    .collect();
                return format!("{}{}({})", pad, to_snake_case(tag), args.join(", "));
            }
        }

        let mut builder = if is_component {
            // Component reference
            format!("{}{}·view()", pad, tag)
        } else {
            // HTML element
            format!("{}VNode·{}()", pad, tag)
        };

        // Add attributes
        for attr in attributes {
            let attr_code = self.generate_attribute_scoped(attr, scope);
            if !attr_code.is_empty() {
                builder.push_str(&format!("\n{}    {}", pad, attr_code));
            }
        }

        // Add children
        for child in children {
            let child_code = self.generate_vnode(child, indent + 1, scope);
            match &child.node_type {
                JsxNodeType::Text { value } => {
                    // Text: append text_child directly
                    builder.push_str(&format!("\n{}    ·text_child(\"{}\")", pad, escape_string(value)));
                }
                JsxNodeType::Expression { .. } => {
                    // Expression: append the generated text_child directly (no wrapping)
                    builder.push_str(&format!("\n{}    {}", pad, child_code.trim()));
                }
                JsxNodeType::Map { .. } => {
                    // Map: append the generated ·children() directly (no wrapping)
                    builder.push_str(&format!("\n{}    {}", pad, child_code.trim_start()));
                }
                JsxNodeType::Conditional { .. } => {
                    // Conditional: append ·when() or ·when_else() directly
                    builder.push_str(&format!("\n{}    {}", pad, child_code.trim()));
                }
                _ => {
                    // Element, Fragment: wrap in ·child()
                    builder.push_str(&format!("\n{}    ·child(\n{}\n{}    )", pad, child_code, pad));
                }
            }
        }

        builder
    }

    fn generate_fragment_vnode(&self, children: &[JsxNode], indent: usize, scope: &VNodeScope) -> String {
        let pad = "    ".repeat(indent);
        let mut builder = format!("{}VNode·fragment()", pad);

        for child in children {
            let child_code = self.generate_vnode(child, indent + 1, scope);
            match &child.node_type {
                JsxNodeType::Text { value } => {
                    builder.push_str(&format!("\n{}    ·text_child(\"{}\")", pad, escape_string(value)));
                }
                JsxNodeType::Expression { .. } | JsxNodeType::Map { .. } | JsxNodeType::Conditional { .. } => {
                    builder.push_str(&format!("\n{}    {}", pad, child_code.trim()));
                }
                _ => {
                    builder.push_str(&format!("\n{}    ·child(\n{}\n{}    )", pad, child_code, pad));
                }
            }
        }

        builder
    }
}

// =============================================================================
// Public API
// =============================================================================

/// Generate Sigil code from a component migration spec.
/// The component alone, for the project layout — see `generate_component_only`.
pub fn generate_component_only(
    spec: &ComponentMigrationSpec,
    component_props: std::collections::HashMap<String, ComponentCallShape>,
) -> GeneratedSigil {
    QliphothGenerator::with_components(spec, component_props).generate_component_only()
}

pub fn generate_component(spec: &ComponentMigrationSpec) -> GeneratedSigil {
    let generator = QliphothGenerator::new(spec);
    generator.generate()
}

/// As `generate_component`, but able to see the other components.
///
/// A child component in JSX is a call to that component's function — in the
/// per-file layout the call cannot resolve, because the sibling is a different
/// file, but a call to a name that is not there is an honest unresolved call.
/// `X·view()` was a method on a type that does not exist.
pub fn generate_component_with(
    spec: &ComponentMigrationSpec,
    component_props: std::collections::HashMap<String, ComponentCallShape>,
) -> GeneratedSigil {
    QliphothGenerator::with_components(spec, component_props).generate()
}

/// Every component's props, by component name — what a call site needs to pass
/// a sibling component its arguments.
pub fn component_prop_map(
    specs: &[&ComponentMigrationSpec],
) -> std::collections::HashMap<String, ComponentCallShape> {
    specs
        .iter()
        .map(|s| {
            (
                s.name.clone(),
                ComponentCallShape {
                    is_actor: s.target.pattern == TargetPattern::Actor,
                    props: s
                        .recommendations
                        .props_handling
                        .fields
                        .iter()
                        .map(|f| f.name.clone())
                        .collect(),
                },
            )
        })
        .collect()
}

/// Generate Sigil code for all components in a migration spec.
pub fn generate_all(spec: &MigrationSpec) -> Vec<GeneratedSigil> {
    spec.components.iter()
        .map(|comp| generate_component(comp))
        .collect()
}

/// Generate Sigil code for a service actor.
pub fn generate_service_actor(actor: &ServiceActorSpec) -> GeneratedSigil {
    let generator = ServiceActorGenerator::new(actor);
    generator.generate()
}

/// Generate Sigil code for all service actors in a migration spec.
pub fn generate_all_service_actors(spec: &MigrationSpec) -> Vec<GeneratedSigil> {
    spec.service_actors.iter()
        .map(|actor| generate_service_actor(actor))
        .collect()
}

// =============================================================================
// Service Actor Generator (Phase 7)
// =============================================================================

/// Generator for service actor Sigil code.
struct ServiceActorGenerator<'a> {
    actor: &'a ServiceActorSpec,
}

impl<'a> ServiceActorGenerator<'a> {
    fn new(actor: &'a ServiceActorSpec) -> Self {
        Self { actor }
    }

    fn generate(&self) -> GeneratedSigil {
        let mut code = String::new();

        // Imports
        code.push_str("invoke qliphoth·prelude·*;\n\n");

        // Message enum
        code.push_str(&self.generate_message_enum());
        code.push_str("\n\n");

        // Actor definition
        code.push_str(&self.generate_actor());

        GeneratedSigil {
            path: format!("src/services/{}.sigil", to_snake_case(&self.actor.name)),
            code,
            component_name: self.actor.name.clone(),
        }
    }

    fn generate_message_enum(&self) -> String {
        if self.actor.messages.is_empty() {
            return String::new();
        }

        let name = format!("{}Msg", self.actor.name);
        let variants: Vec<String> = self.actor.messages.iter()
            .map(|msg| {
                if msg.parameters.is_empty() {
                    format!("    {},", msg.name)
                } else {
                    // Generate payload type from parameters
                    let payload = format!("{{ {} }}", msg.parameters.join(", "));
                    format!("    {} {},", msg.name, payload)
                }
            })
            .collect();

        format!("☉ ᛈ {} {{\n{}\n}}", name, variants.join("\n"))
    }

    fn generate_actor(&self) -> String {
        let mut sections = Vec::new();

        // Comment showing derivation
        sections.push(format!("    // Derived from hook: {}", self.actor.derived_from));

        // State fields
        if !self.actor.state_fields.is_empty() {
            let fields: Vec<String> = self.actor.state_fields.iter()
                .map(|f| {
                    let evidentiality = if f.is_observable { "!" } else { "~" };
                    format!("    state {}: {}{} = /* initial */,", f.name, f.field_type, evidentiality)
                })
                .collect();
            sections.push(fields.join("\n"));
        }

        // Message handlers
        if !self.actor.messages.is_empty() {
            let handlers: Vec<String> = self.actor.messages.iter()
                .map(|msg| {
                    format!(
                        "    on {} {{\n        // TODO: implement {}\n    }}",
                        msg.name,
                        msg.original_name
                    )
                })
                .collect();
            sections.push(handlers.join("\n\n"));
        }

        // Query methods for state (observable fields become queryable)
        let queries: Vec<String> = self.actor.state_fields.iter()
            .filter(|f| f.is_observable)
            .map(|f| {
                format!(
                    "    rite {}(self) -> {}! {{\n        self.{}\n    }}",
                    f.name,
                    f.field_type,
                    f.name
                )
            })
            .collect();
        if !queries.is_empty() {
            sections.push(queries.join("\n\n"));
        }

        format!(
            "☉ actor {} {{\n{}\n}}",
            self.actor.name,
            sections.join("\n\n")
        )
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

impl<'a> QliphothGenerator<'a> {
    /// Transform a React expression into Sigil syntax.
    /// For actors: prefix state with `self.`
    /// For pure functions: use parameters directly
    fn transform_expression(&self, code: &str) -> String {
        self.transform_expression_scoped(code, &VNodeScope::default())
    }

    /// Transform with scope awareness (don't prefix iterator variables with self.)
    /// As `transform_expression_scoped`, but never prefixing `self.`.
    ///
    /// A helper function is a free `rite`: it has no receiver, so the
    /// unknown-identifier rule's `self.<name>` guess is not merely unhelpful
    /// there, it cannot compile.
    fn transform_expression_free(&self, code: &str, scope: &VNodeScope) -> String {
        self.transform_expression_with(code, scope, false)
    }

    fn transform_expression_scoped(&self, code: &str, scope: &VNodeScope) -> String {
        self.transform_expression_with(code, scope, self.is_actor)
    }

    /// The same `TransformConfig` the expression path builds, for the statement
    /// transform to reuse — one description of what is in scope, not two.
    fn statement_config(&self, scope: &VNodeScope) -> TransformConfig {
        self.build_transform_config(scope, false)
    }

    fn build_transform_config(&self, scope: &VNodeScope, prefix_self: bool) -> TransformConfig {
        let state_fields: Vec<String> = self
            .spec
            .recommendations
            .state_fields
            .iter()
            .map(|f| f.to_field.clone())
            .collect();
        let mut locals = scope.locals.clone();
        locals.extend(self.spec.source.module_scope.iter().cloned());
        TransformConfig {
            prefix_self,
            state_fields,
            locals,
            props: self.param_names.clone(),
            lookup_constants: self
                .spec
                .source
                .module_constants
                .iter()
                .filter(|c| !c.entries.is_empty())
                .map(|c| c.name.clone())
                .collect(),
            value_constants: self.value_constant_names(),
            fn_arity: {
                // Component-level functions too, not only module helpers. Those
                // become actor methods, and a JavaScript call site is free to
                // omit trailing arguments — Sigil is not, so the call has to be
                // padded or the module fails to build on the arity.
                let mut a = self.helper_arity();
                for local in &self.spec.source.extraction.locals {
                    if let Some(hook) =
                        stmt_transform::hook_callback(&local.init, &["useCallback"])
                            .or_else(|| stmt_transform::function_local(&local.init))
                    {
                        a.insert(to_snake_case(&local.name), hook.params.len());
                    }
                }
                a
            },
        }
    }

    fn transform_expression_with(
        &self,
        code: &str,
        scope: &VNodeScope,
        prefix_self: bool,
    ) -> String {
        // Handle placeholder/invalid expressions
        // The placeholder markers the extractor emits when it could not read
        // something — not any comment. Rejecting every `/*` threw away genuine
        // source: `export const identity = { /** Display name */ name: "Lares",
        // … }` is an ordinary documented object literal, and it came back
        // unreadable because of its own doc comments.
        if code.contains("/* expr */")
            || code.contains("/* expression */")
            || code.contains("/* block */")
            || code.contains("/* pattern */")
            || code.contains("/* method */")
            || code.is_empty()
        {
            return "None".to_string();
        }

        // Clean up the expression
        let code = code.trim();

        // Handle JSX in expressions - these need special handling
        // For now, replace with None as a placeholder
        if code.contains('<') && code.contains('>') {
            return "None".to_string();
        }

        // Module-scope names are in scope for the whole file and are emitted as
        // top-level items, so a reference to one must be left alone rather than
        // swept into `self.`.
        let config = self.build_transform_config(scope, prefix_self);

        // Use AST-based transformation
        let result = ast_transform::transform_expression(code, &config);

        // Note: warnings are available in result.warnings if needed for debugging
        result.code
    }

    /// For actors, prefix state field references with self.
    fn prefix_state_variables(&self, code: &str) -> String {
        self.prefix_state_variables_scoped(code, &VNodeScope::default())
    }

    /// Prefix state variables, but not scoped locals
    fn prefix_state_variables_scoped(&self, code: &str, scope: &VNodeScope) -> String {
        let state_fields: Vec<String> = self.spec.recommendations.state_fields
            .iter()
            .map(|f| f.to_field.clone())
            .collect();

        let mut result = code.to_string();

        // Simple approach: prefix known state fields with self.
        // This is a basic implementation - a real one would parse the expression
        for field in &state_fields {
            // Skip if this is a scoped local variable
            if scope.locals.contains(field) {
                continue;
            }

            // Replace field name when it appears as a word boundary
            let patterns = [
                (format!("{}", field), format!("self.{}", field)),
            ];
            for (from, to) in &patterns {
                // Only replace if it's a standalone identifier (not part of a larger word)
                if result == *from {
                    result = to.clone();
                } else if result.starts_with(&format!("{}.", from)) {
                    result = format!("self.{}", result);
                    break;
                }
            }
        }

        // If no state field was found and it's a simple identifier that's not a local, prefix with self
        if !result.starts_with("self.") && is_simple_identifier(&result) && !scope.locals.iter().any(|l| result == *l) {
            result = format!("self.{}", result);
        }

        result
    }

    /// Generate attribute with scope awareness
    /// Does this transformed expression evaluate to a VNode?
    ///
    /// True when it is a call to a component this migration generated — those
    /// are emitted as `rite <name>(…) -> VNode!`. Nothing else is assumed: a
    /// helper that formats a string is still text, and guessing wrong either
    /// way puts the wrong thing in the document.
    fn renders_a_node(&self, expr: &str) -> bool {
        let head = expr.trim();
        let Some(open) = head.find('(') else { return false };
        let name = &head[..open];
        if !name.chars().all(|c| c.is_alphanumeric() || c == '_') || name.is_empty() {
            return false;
        }
        // The map is keyed by the React name; the emitted call is snake_case.
        self.component_props
            .keys()
            .any(|k| to_snake_case(k) == name)
    }

    fn generate_attribute_scoped(&self, attr: &JsxAttribute, scope: &VNodeScope) -> String {
        // `ref` and `key` are React's, not the DOM's. React never renders them;
        // emitted as attributes they showed up in the document as `ref="0"`,
        // because the value is a ref object the migration has no equivalent for.
        if matches!(attr.name.as_str(), "ref" | "key" | "dangerouslySetInnerHTML") {
            return String::new();
        }

        // Handle special attributes
        match attr.name.as_str() {
            "className" | "class" => {
                match &attr.value {
                    JsxAttributeValue::String { value } => format!("·class(\"{}\")", value),
                    JsxAttributeValue::Expression { code } => {
                        let transformed = self.transform_expression_scoped(code, scope);
                        // Ensure class strings are properly quoted.
                        //
                        // The test is whether the quotes are BALANCED, not whether the
                        // text ends in one. A concatenation such as
                        //     "base " + ⎇ cond { "a" } ⎉ { "b" }·to_string()
                        // starts with a quote and ends with ')', but is already
                        // well-formed; appending a quote there produced an unterminated
                        // string literal that swallowed the following lines. That single
                        // mistake accounted for 30 of 44 failures when generating the
                        // Lares UI.
                        let unbalanced_quotes = transformed.matches('"').count() % 2 == 1;
                        if transformed.starts_with('"') && unbalanced_quotes {
                            format!("·class({}\")", transformed)
                        } else {
                            // Never quoted. `className={x}` in React is the VALUE
                            // of `x`, never the text "x" — and an arm here wrapped
                            // anything without a quote in one, so
                            // `className={chip.cls}` rendered as the literal class
                            // name `chip.cls`, in the DOM, visible.
                            format!("·class({})", transformed)
                        }
                    }
                    _ => String::new(),
                }
            }
            "id" => {
                match &attr.value {
                    JsxAttributeValue::String { value } => format!("·id(\"{}\")", value),
                    JsxAttributeValue::Expression { code } => {
                        let transformed = self.transform_expression_scoped(code, scope);
                        format!("·id({})", transformed)
                    }
                    _ => String::new(),
                }
            }
            "style" => {
                // React's `style={{ width: "8px", opacity: 0.5 }}` is one object;
                // Qliphoth's `style` takes a property and a value, one call per
                // declaration. Emitting `·style(∅)` was a call with one argument
                // to a method that takes two, which `sigil check` passed and the
                // WASM backend turned into an unloadable module.
                match &attr.value {
                    JsxAttributeValue::Expression { code } => {
                        match style_entries(code) {
                            Some(entries) if !entries.is_empty() => entries
                                .into_iter()
                                .map(|(prop, val)| {
                                    format!(
                                        "·style({:?}, {}·to_string())",
                                        css_property(&prop),
                                        self.transform_expression_scoped(&val, scope)
                                    )
                                })
                                .collect::<Vec<_>>()
                                .join("\n"),
                            // A computed style object has no fixed set of
                            // properties, and there is no call that means "these,
                            // whatever they turn out to be".
                            _ => format!(
                                "// style: not a literal object, so no property/value pairs. React: {}",
                                code.split_whitespace().collect::<Vec<_>>().join(" ")
                            ),
                        }
                    }
                    _ => String::new(),
                }
            }
            name if attr.is_event_handler => {
                // Event handler -> message dispatch.
                //
                // The message name comes from the callback the handler invokes, not
                // from the event type. Deriving it from the event meant every onClick
                // in a component dispatched the same `Click`, so distinct buttons were
                // indistinguishable — and `Click` was never declared in the actor's
                // enum either. derive_event_message_name is shared with
                // recommend_messages so dispatch and declaration cannot drift.
                let event_name = name.strip_prefix("on").unwrap_or(name);
                let code = match &attr.value {
                    JsxAttributeValue::Expression { code } => code.as_str(),
                    _ => "",
                };
                let msg_name = crate::migrate::react::spec::derive_event_message_name(code, name);

                // Arity is dictated by the DECLARATION, never by this call site.
                // The same message name can be reached from several handlers, and
                // the enum only carries the first one's shape; deriving the
                // dispatch arity locally would emit `Expand(x)` against a bare
                // `Expand` variant. Missing arguments become ∅, extras are dropped.
                let declared_arity = self
                    .spec
                    .recommendations
                    .messages
                    .iter()
                    .find(|m| m.name == msg_name)
                    .and_then(|m| m.payload.as_ref())
                    .filter(|p| p.starts_with('('))
                    .map(|p| p.trim_start_matches('(').trim_end_matches(')').split(',').count())
                    .unwrap_or(0);

                // Qliphoth's event methods take a message ID, not a message
                // value: `VNode·on_click(Δ self, message_id: u64)`. A qualified
                // unit variant IS that id — the compiler evaluates
                // `<Enum>·<Variant>` to the variant's tag, which is the same
                // number `<Actor>_dispatch` switches on. Constructing the message
                // instead (`Click(id)`) type-checked, because `sigil check` does
                // not resolve names, and failed the moment the file was compiled
                // against the real prelude with "undefined function: Click".
                //
                // The call site's argument is dropped: `on_click(u64)` has no
                // second parameter, so a message that needs to say WHICH row was
                // clicked cannot. The variant still declares its payload in the
                // enum above, and the expression is in the migration spec, so
                // the gap is visible without a trailing comment here — one of
                // those took out `app.sigil`, because a dispatch is frequently
                // mid-expression and a Sigil comment runs to end of line.
                let enum_name = format!("{}Msg", self.spec.name);
                let _ = declared_arity;
                // Only a real DOM event gets a named method. Everything else is
                // a CHILD COMPONENT'S callback prop — `onOpenSession`,
                // `onLaunchClaude`, `onTicketsLoaded` — which is not an event on
                // a DOM node at all; lowercasing those produced
                // `·on_opensession(…)`, a method that could never exist, once per
                // distinct prop name across the client.
                match dom_event_method(event_name) {
                    Some(method) => format!("·{}({}·{})", method, enum_name, msg_name),
                    None => format!(
                        "·on_event(\"{}\", {}·{})",
                        name, enum_name, msg_name
                    ),
                }
            }
            "disabled" | "checked" | "selected" | "readonly" => {
                // Boolean attributes
                match &attr.value {
                    JsxAttributeValue::True => format!("·attr(\"{}\", \"true\")", dom_attribute_name(&attr.name)),
                    JsxAttributeValue::Expression { code } => {
                        // `·when(cond, |n| n·attr(…))` — a closure — is not a form
                        // Qliphoth's builder has: "No DSL, no function types, no
                        // callbacks" is the first thing `core/vdom.sigil` says.
                        // `when` takes a VNode child, so 66 of these compiled
                        // against a signature that does not exist. `attr_when` is
                        // the same idea without the closure.
                        let transformed = self.transform_expression_scoped(code, scope);
                        format!("·attr_when({}, \"{}\", \"true\")", transformed, dom_attribute_name(&attr.name))
                    }
                    _ => String::new(),
                }
            }
            "href" | "src" | "alt" | "type" | "name" | "value" | "placeholder" => {
                // Common attributes
                match &attr.value {
                    JsxAttributeValue::String { value } => {
                        format!("·attr(\"{}\", \"{}\")", dom_attribute_name(&attr.name), escape_string(value))
                    }
                    JsxAttributeValue::Expression { code } => {
                        format!("·attr(\"{}\", {})", dom_attribute_name(&attr.name), self.attr_value_as_string(code, scope))
                    }
                    _ => String::new(),
                }
            }
            _ => {
                // Generic attribute
                match &attr.value {
                    JsxAttributeValue::String { value } => {
                        format!("·attr(\"{}\", \"{}\")", dom_attribute_name(&attr.name), escape_string(value))
                    }
                    JsxAttributeValue::Expression { code } => {
                        format!("·attr(\"{}\", {})", dom_attribute_name(&attr.name), self.attr_value_as_string(code, scope))
                    }
                    JsxAttributeValue::Spread { name } => {
                        // Spread attributes can't be directly represented, skip
                        String::new()
                    }
                    JsxAttributeValue::True => {
                        format!("·attr(\"{}\", \"true\")", dom_attribute_name(&attr.name))
                    }
                }
            }
        }
    }
}

/// Check if a string is a simple identifier (no operators, dots, etc.)
fn is_simple_identifier(s: &str) -> bool {
    !s.is_empty() &&
    s.chars().next().map_or(false, |c| c.is_alphabetic() || c == '_') &&
    s.chars().all(|c| c.is_alphanumeric() || c == '_')
}

/// Does this actually parse as a Sigil expression?
///
/// The expression transform is a source-to-source heuristic over JavaScript it
/// only partly understands, so some of what it emits is not Sigil at all. The
/// alternative is a hand-maintained list of forbidden substrings, and that list
/// grew by one entry every time the generator learned to emit more — `?.`, then
/// `??`, then `...`, then `=>`, then `instanceof`. Ask the parser instead: it is
/// the same one that will read the generated file.
/// Whether a translated body parses as a sequence of Sigil statements.
///
/// The gate on a translated body used to be "no warnings", which threw away
/// every body containing a `catch` or a `throw` — constructs Sigil does not
/// have, which the transform marks with a `// React:` comment. A body with a
/// visible gap still keeps its branches, and the alternative is the flat list
/// of assignments that loses them. What actually has to hold is that the output
/// parses.
/// The same probe for a method, which needs a `self` to be valid at all.
///
/// A bare top-level `rite flush(self, …)` does not parse `self`, so every
/// generated actor method failed the probe and stayed a view local — which is
/// the shape that produced `undefined function: flush`.
/// The names generated handler bodies call as `self.<name>(…)`.
///
/// A component-level function the handlers call is a method of the actor. One
/// they merely pass around is a value, and making it a method would break the
/// call that passes it — so the handlers, not the shape of the binding, decide.
fn self_called_names(bodies: &[&str]) -> std::collections::HashSet<String> {
    let mut out = std::collections::HashSet::new();
    for body in bodies {
        let mut rest = *body;
        while let Some(at) = rest.find("self.") {
            rest = &rest[at + "self.".len()..];
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() && rest[name.len()..].starts_with('(') {
                out.insert(name);
            }
        }
    }
    out
}

/// How many arguments the bodies pass to `self.<name>(…)`.
///
/// `None` when the name is not called, or when two call sites disagree — a stub
/// with the wrong arity is a module no runtime will load, which is worse than
/// the undefined function it replaces.
fn call_arity(bodies: &[&str], name: &str) -> Option<usize> {
    let needle = format!("self.{}(", name);
    let mut found: Option<usize> = None;
    for body in bodies {
        let mut rest: &str = body;
        while let Some(at) = rest.find(&needle) {
            let after = &rest[at + needle.len()..];
            rest = after;
            let mut depth = 0usize;
            let mut commas = 0usize;
            let mut any = false;
            let mut closed = false;
            for c in after.chars() {
                match c {
                    '(' | '[' | '{' => depth += 1,
                    ')' if depth == 0 => {
                        closed = true;
                        break;
                    }
                    ')' | ']' | '}' => depth = depth.saturating_sub(1),
                    ',' if depth == 0 => commas += 1,
                    c if !c.is_whitespace() => any = true,
                    _ => {}
                }
            }
            if !closed {
                return None;
            }
            let argc = if any { commas + 1 } else { 0 };
            match found {
                None => found = Some(argc),
                Some(prev) if prev == argc => {}
                Some(_) => return None,
            }
        }
    }
    found
}

fn parses_as_method(signature: &str, body: &str) -> bool {
    let probe = format!(
        "\u{3a3} __Probe {{ __x: Any }}\n\u{22a2} __Probe {{\n{} {{\n{}\n}}\n}}\n",
        signature, body
    );
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        match crate::parser::Parser::new(&probe).parse_file() {
            Ok(ast) => crate::typeck::TypeChecker::new().check_file(&ast).is_ok(),
            Err(_) => false,
        }
    }))
    .unwrap_or(false);
    std::panic::set_hook(prev);
    ok
}

fn parses_as_statements(signature: &str, body: &str) -> bool {
    // The function as it will be emitted, parsed AND type-checked. Parsing
    // alone is not enough: a body whose tail expression is uncertain under a
    // `-> Any!` signature parses perfectly and fails `sigil check` with an
    // evidence mismatch, which is how two files stopped compiling.
    let probe = format!("{} {{\n{}\n}}\n", signature, body);
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        match crate::parser::Parser::new(&probe).parse_file() {
            Ok(ast) => crate::typeck::TypeChecker::new().check_file(&ast).is_ok(),
            Err(_) => false,
        }
    }))
    .unwrap_or(false);
    std::panic::set_hook(prev);
    ok
}

fn parses_as_expression(expr: &str) -> bool {
    let probe = format!("rite __probe() {{\n    \u{2254} __v = {};\n}}\n", expr);
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let ok = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::parser::Parser::new(&probe).parse_file().is_ok()
    }))
    .unwrap_or(false);
    std::panic::set_hook(prev);
    ok
}

/// The Qliphoth builder method for a React event name, if it is a DOM event.
///
/// `onClick` -> `on_click`. A name that is not in this list is a component's
/// callback prop and goes through `on_event("<name>", id)` with its React
/// spelling intact.
fn dom_event_method(event_name: &str) -> Option<&'static str> {
    Some(match event_name.to_lowercase().as_str() {
        "click" => "on_click",
        "input" => "on_input",
        "submit" => "on_submit",
        "change" => "on_change",
        "keydown" => "on_keydown",
        "keyup" => "on_keyup",
        "mousedown" => "on_mousedown",
        "mouseup" => "on_mouseup",
        "focus" => "on_focus",
        "blur" => "on_blur",
        "doubleclick" | "dblclick" => "on_dblclick",
        _ => return None,
    })
}

/// Does this expression name nothing outside itself?
///
/// String contents are ignored; what is left must be literals, brackets and
/// punctuation. A body that passes can be emitted as a function without
/// dragging an unresolved name in with it.
fn is_literal_only(value: &str) -> bool {
    let mut bare = String::with_capacity(value.len());
    let mut in_str = false;
    let mut escaped = false;
    for c in value.chars() {
        if in_str {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_str = false;
            }
            continue;
        }
        if c == '"' {
            in_str = true;
        } else {
            bare.push(c);
        }
    }
    if in_str {
        return false;
    }
    // An object literal's KEYS are field names, not references to anything, so
    // they do not make the value depend on the surrounding scope. Without this,
    // `const VIEWS = [{ key: "all", label: "All" }, …]` — a plain table of
    // strings — was rejected for containing the letters in `key` and `label`,
    // and the constant was dropped for a comment.
    let bare = strip_object_keys(&bare);

    let stripped = bare
        .replace("true", " ")
        .replace("false", " ")
        .replace("None", " ")
        .replace('\u{2205}', " ");
    stripped
        .chars()
        .all(|c| !(c.is_alphabetic() || c == '_' || c == '\u{00B7}'))
}

/// `{ width: "8px", opacity: 0.5 }` -> the property/value pairs, when the style
/// prop is a literal object. `None` for anything computed.
fn style_entries(code: &str) -> Option<Vec<(String, String)>> {
    let body = code.trim();
    let body = body.strip_prefix('{')?.strip_suffix('}')?.trim();
    if body.is_empty() {
        return Some(Vec::new());
    }
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut in_str: Option<char> = None;
    let mut cur = String::new();
    let mut parts: Vec<String> = Vec::new();
    for c in body.chars() {
        if let Some(q) = in_str {
            cur.push(c);
            if c == q {
                in_str = None;
            }
            continue;
        }
        match c {
            '"' | '\'' | '`' => {
                in_str = Some(c);
                cur.push(c);
            }
            '{' | '[' | '(' => {
                depth += 1;
                cur.push(c);
            }
            '}' | ']' | ')' => {
                depth -= 1;
                cur.push(c);
            }
            ',' if depth == 0 => {
                parts.push(std::mem::take(&mut cur));
            }
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        parts.push(cur);
    }
    for part in parts {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        // A spread carries properties this cannot name.
        if part.starts_with("...") {
            return None;
        }
        let (k, v) = part.split_once(':')?;
        let key = k.trim().trim_matches('"').trim_matches('\'').to_string();
        if key.is_empty() || !key.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '$') {
            return None;
        }
        out.push((key, v.trim().to_string()));
    }
    Some(out)
}

/// `marginTop` -> `margin-top`. CSS property names are what the DOM wants;
/// React's camelCase is a JS-object accommodation.
fn css_property(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    for c in name.chars() {
        if c.is_uppercase() {
            out.push('-');
            out.extend(c.to_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}


/// The module scope every component shares, emitted once.
///
/// Each generated component carries copies of the constants and helpers it
/// references, which is what makes a single file compile on its own and what
/// makes the 93 of them impossible to compile together. This collects the union
/// — deduplicated by name, and ordered so a constant follows what it references
/// — and hands it back as one module for the project layout.
pub fn generate_shared_module(specs: &[&ComponentMigrationSpec]) -> String {
    let Some(first) = specs.first() else {
        return String::new();
    };

    let mut constants: Vec<ModuleConstantExtraction> = Vec::new();
    let mut seen_c: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut helpers: Vec<HelperFunctionExtraction> = Vec::new();
    let mut seen_h: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut scope: Vec<String> = Vec::new();
    let mut seen_s: std::collections::HashSet<String> = std::collections::HashSet::new();

    for spec in specs {
        for c in &spec.source.module_constants {
            if seen_c.insert(c.name.clone()) {
                constants.push(c.clone());
            }
        }
        for h in &spec.source.helpers {
            if seen_h.insert(h.name.clone()) {
                helpers.push(h.clone());
            }
        }
        for n in &spec.source.module_scope {
            if seen_s.insert(n.clone()) {
                scope.push(n.clone());
            }
        }
    }

    constants = super::mcp::topo_sort_constants(constants);

    // A synthetic component whose module scope is the union and whose body is
    // empty: the generator's own emission paths do the rest, so the shared
    // module cannot drift from what the per-file output would have produced.
    let mut synthetic = (*first).clone();
    synthetic.source.module_constants = constants;
    synthetic.source.helpers = helpers;
    synthetic.source.module_scope = scope;
    synthetic.target.pattern = TargetPattern::Function;

    let gen = QliphothGenerator::new(&synthetic);
    let (constants_code, emitted) = gen.generate_module_constants();
    let helpers_code = gen.generate_helper_functions(&emitted);

    let mut out = String::from(
        "// Module scope shared by every generated component: the constants and\n\
         // helper functions their React modules declared, emitted once.\n\n",
    );
    out.push_str(JS_DIVIDE_HELPER);
    out.push_str(&constants_code);
    if !constants_code.is_empty() {
        out.push('\n');
    }
    out.push_str(&helpers_code);
    out
}

/// JavaScript's `/` and Sigil's are different operators.
///
/// `a / 0` is `Infinity` or `NaN` in JavaScript — ugly output, never a failure.
/// In Sigil it is an error, in both backends: `averageTrendTotal` divides by
/// `trend.length` with no guard, exactly as its React did, and the whole
/// dashboard trapped on an empty list. Migrated divisions go through this so a
/// program that was merely untidy in JavaScript does not become one that stops.
///
/// Zero is not NaN. The value is wrong in the same place JavaScript's was, and
/// visibly so.
const JS_DIVIDE_HELPER: &str = "\
// JavaScript's `/` never fails; Sigil's is an error on a zero divisor. Migrated\n\
// divisions come here so a program that printed NaN does not stop instead.\n\
\u{2609} rite js_divide(a: Any, b: Any) -> Any! {\n\
\u{20}   \u{2387} b \u{2260} 0 { a / b } \u{2389} { 0 }\n\
}\n\n";

/// Free-standing functions the WASM backend resolves, which a helper body may
/// therefore name without the file declaring anything.
const HOST_FUNCTIONS: &[&str] = &[
    "to_bool", "to_fixed", "to_string", "is_array", "is_finite", "is_nan", "is_integer",
    "object_values", "object_keys", "object_entries", "json_parse", "json_stringify",
    "json_pretty", "json_get", "json_set", "timing_now", "timing_parse", "math_random",
    "date_now", "date_format",
    "None", "true", "false", "Some", "Ok", "Err", "This", "Self", "VNode", "VElement",
    "HashMap", "HashSet", "Vec", "String",
    // Emitted into the shared module by `JS_DIVIDE_HELPER`, so a helper body
    // may name it without declaring anything. Left out, every body containing
    // a division failed the known-names check and went back to untranslated.
    "js_divide",
    // Import aliases the WASM backend already resolves. Left out, a body that
    // named any of them failed the known-names gate and went back to
    // untranslated — `type_of` and `fetch_request` between them blocked 17
    // helpers in the Lares client, each of which blocked more.
    "type_of", "fetch_request", "fetch_status", "fetch_ok", "fetch_get_body",
    "encode_uri_component", "decode_uri_component", "string_from_value",
    "window", "document", "confirm", "alert", "prompt", "console_log", "print",
    // `Math.*` over the uniform value domain. The F64-typed `math.*` imports
    // cannot be called from a migrated program, where every number is an i64.
    "math_floor", "math_ceil", "math_round", "math_trunc", "math_abs",
    "math_min", "math_max", "math_sqrt", "math_pow",
    // `string.parse_int` / `parse_float`, aliased bare since the start and
    // never listed here — so `hexToRgb` and every other body that parses a
    // number failed the gate on a name the backend resolves.
    "parse_int", "parse_float",
];

/// React's attribute spelling, as the DOM's.
///
/// JSX writes `strokeWidth`, `htmlFor` and `tabIndex`; the DOM has
/// `stroke-width`, `for` and `tabindex`. Emitted verbatim, a browser does not
/// recognise them and silently ignores them — the notification bell drew at the
/// default hairline weight instead of 1.8, which no test can see and a
/// screenshot can.
///
/// SVG presentation attributes are the bulk of it: every hyphenated SVG
/// attribute is camelCase in JSX. The general rule below covers them, and the
/// exceptions above it are the ones the rule would get wrong — `viewBox` and
/// friends really are camelCase in the DOM too.
fn dom_attribute_name(name: &str) -> String {
    match name {
        // React-only spellings with a different DOM name.
        "htmlFor" => return "for".to_string(),
        "className" => return "class".to_string(),
        // Genuinely camelCase in SVG: the rule below must not touch these.
        "viewBox" | "preserveAspectRatio" | "baseProfile" | "gradientUnits"
        | "gradientTransform" | "patternUnits" | "patternContentUnits"
        | "clipPathUnits" | "maskUnits" | "maskContentUnits" | "markerUnits"
        | "markerWidth" | "markerHeight" | "refX" | "refY" | "spreadMethod"
        | "startOffset" | "textLength" | "lengthAdjust" | "primitiveUnits"
        | "filterUnits" | "surfaceScale" | "specularConstant"
        | "specularExponent" | "diffuseConstant" | "kernelMatrix"
        | "stdDeviation" | "attributeName" | "repeatCount" | "keyPoints"
        | "keyTimes" | "keySplines" | "calcMode" | "pathLength" => {
            return name.to_string();
        }
        _ => {}
    }
    // `data-*` and `aria-*` are already hyphenated, and anything with no
    // capital is already the DOM's spelling.
    if name.contains('-') || !name.chars().any(|c| c.is_ascii_uppercase()) {
        return name.to_string();
    }
    let mut out = String::with_capacity(name.len() + 2);
    for c in name.chars() {
        if c.is_ascii_uppercase() {
            out.push('-');
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

/// Strip characters a Sigil comment cannot carry.
///
/// React source goes into a comment verbatim, and one of the Lares helpers
/// separates map keys with a literal NUL inside a template string — which made
/// the generated file binary, and `grep` skip it.
fn sanitize_comment(line: &str) -> String {
    line.trim_end()
        .chars()
        .map(|c| if (c as u32) < 0x20 || c == '\u{007f}' { '\u{fffd}' } else { c })
        .collect()
}

/// Does this transformed expression name only things that are in scope?
///
/// Bare identifiers only: a word after `·` or `.` is a member or a method, and
/// a word before `:` is an object-literal key. Neither reads a binding.
/// Names a translated body binds: `≔ x = …`, `≔ Δ x = …`, `∀ x ∈ …`, and
/// closure parameters `|a, b|`.
fn names_bound_in(body: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in body.lines() {
        let t = line.trim();
        if let Some(rest) = t.strip_prefix('\u{2254}') {
            let rest = rest.trim_start();
            let rest = rest.strip_prefix('\u{0394}').unwrap_or(rest).trim_start();
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                out.push(name);
            }
        }
        if let Some(rest) = t.strip_prefix('\u{2200}') {
            let name: String = rest
                .trim_start()
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                out.push(name);
            }
        }
        // Closure parameters anywhere on the line.
        let chars: Vec<char> = t.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            if chars[i] == '|' {
                let start = i + 1;
                let mut j = start;
                while j < chars.len() && chars[j] != '|' && chars[j] != '\n' {
                    j += 1;
                }
                if j < chars.len() && chars[j] == '|' {
                    let inner: String = chars[start..j].iter().collect();
                    if inner
                        .chars()
                        .all(|c| c.is_alphanumeric() || c == '_' || c == ',' || c.is_whitespace())
                    {
                        for p in inner.split(',') {
                            let p = p.trim();
                            if !p.is_empty() {
                                out.push(p.to_string());
                            }
                        }
                    }
                    i = j + 1;
                    continue;
                }
            }
            i += 1;
        }
    }
    out
}

/// Sigil markers that a word-scanner reads as identifiers but which bind and
/// resolve nothing: the match wildcard and the mutable-binding marker.
fn is_not_a_name(word: &str) -> bool {
    matches!(word, "_" | "\u{0394}")
}

/// Report why a helper's body was not translated, under `SIGIL_MIGRATE_SKIPS`.
///
/// 157 helpers in the Lares client came out as `∅` under a "needs a statement
/// transform" comment when the statement transform had been written months
/// earlier. Which gate each one actually failed is not guessable from the
/// output — the emitted stub is identical whichever it was.
fn helper_skip(name: &str, why: &str) {
    if std::env::var_os("SIGIL_MIGRATE_SKIPS").is_some() {
        eprintln!("[skip] {name}: {why}");
    }
}

/// The line the rejected name appears on, under `SIGIL_MIGRATE_SKIPS=2`.
fn helper_skip_line(name: &str, unknown: &str, body: &str) {
    if std::env::var("SIGIL_MIGRATE_SKIPS").ok().as_deref() != Some("2") {
        return;
    }
    for line in body.lines() {
        if line.contains(unknown) {
            eprintln!("[skip-line] {name}: {}", line.trim());
            return;
        }
    }
}

/// The first name in `value` that `known` does not have — the one
/// `free_identifiers_known` rejected on. Diagnostic only.
fn first_unknown(value: &str, known: &std::collections::HashSet<String>) -> Option<String> {
    let mut known = known.clone();
    known.extend(names_bound_in(value));
    for word in words_of(value) {
        if !known.contains(&word) {
            return Some(word);
        }
    }
    None
}

/// The free words of a Sigil body, by the same reading `free_identifiers_known`
/// uses: outside strings and comments, not after `·` or `.`, not an object key.
fn words_of(value: &str) -> Vec<String> {
    let chars: Vec<char> = value.chars().collect();
    let is_word = |c: char| c.is_alphanumeric() || c == '_' || c == '$';
    let mut out = Vec::new();
    let mut i = 0;
    let mut in_str = false;
    let mut escaped = false;
    while i < chars.len() {
        let c = chars[i];
        if !in_str && c == '/' && chars.get(i + 1) == Some(&'/') {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        if !in_str && c == '/' && chars.get(i + 1) == Some(&'*') {
            i += 2;
            while i < chars.len() && !(chars[i] == '*' && chars.get(i + 1) == Some(&'/')) {
                i += 1;
            }
            i = (i + 2).min(chars.len());
            continue;
        }
        if in_str {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        if c == '"' {
            in_str = true;
            i += 1;
            continue;
        }
        if is_word(c) && !c.is_ascii_digit() {
            let start = i;
            while i < chars.len() && is_word(chars[i]) {
                i += 1;
            }
            let after_sep = start > 0 && matches!(chars[start - 1], '\u{00B7}' | '.');
            let mut j = i;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            let is_key = chars.get(j) == Some(&':') && chars.get(j + 1) != Some(&':');
            let word: String = chars[start..i].iter().collect();
            if !after_sep && !is_key && !is_not_a_name(&word) {
                out.push(word);
            }
            continue;
        }
        i += 1;
    }
    out
}

/// `free_identifiers_known`, reachable from the test module.
#[cfg(test)]
pub(crate) fn free_identifiers_known_for_test(
    value: &str,
    known: &std::collections::HashSet<String>,
) -> bool {
    free_identifiers_known(value, known)
}

fn free_identifiers_known(
    value: &str,
    known: &std::collections::HashSet<String>,
) -> bool {
    // Names the body binds itself. Written for expressions, this gate saw every
    // word as free — so a translated statement body failed on the first `≔` it
    // declared, and on every word inside its own `// React:` comments.
    let mut known = known.clone();
    known.extend(names_bound_in(value));

    let chars: Vec<char> = value.chars().collect();
    let is_word = |c: char| c.is_alphanumeric() || c == '_' || c == '$';
    let mut i = 0;
    let mut in_str = false;
    let mut escaped = false;
    while i < chars.len() {
        let c = chars[i];
        // A `//` comment carries React source, not Sigil bindings.
        if !in_str && c == '/' && chars.get(i + 1) == Some(&'/') {
            while i < chars.len() && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        // So does a `/* … */` one, and those were being read as code: the
        // transform marks what it cannot translate with them, and
        // `/* regex: /\.service$/ — unsupported */` then failed the gate on the
        // word `service`. A quote inside such a comment was worse — it opened a
        // string state that swallowed the rest of the body.
        if !in_str && c == '/' && chars.get(i + 1) == Some(&'*') {
            i += 2;
            while i < chars.len() && !(chars[i] == '*' && chars.get(i + 1) == Some(&'/')) {
                i += 1;
            }
            i = (i + 2).min(chars.len());
            continue;
        }
        if in_str {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_str = false;
            }
            i += 1;
            continue;
        }
        if c == '"' {
            in_str = true;
            i += 1;
            continue;
        }
        if is_word(c) && !c.is_ascii_digit() {
            let start = i;
            while i < chars.len() && is_word(chars[i]) {
                i += 1;
            }
            let word: String = chars[start..i].iter().collect();
            let after_sep = start > 0 && matches!(chars[start - 1], '\u{00B7}' | '.');
            let mut j = i;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            let is_key = chars.get(j) == Some(&':') && chars.get(j + 1) != Some(&':');
            // `_` is a match wildcard and `Δ` is the mutable-binding marker.
            // Neither is a name, but both are alphanumeric to Rust, so the
            // scanner read them as free identifiers: `≔ Δ i = 0;` failed the
            // gate on the `Δ`, and `_ => { }` on the wildcard.
            if after_sep || is_key || is_not_a_name(&word) {
                continue;
            }
            if !known.contains(&word) {
                return false;
            }
            continue;
        }
        i += 1;
    }
    !in_str
}

/// Blank out `ident:` sequences — object-literal keys, which name a field
/// rather than read a binding.
fn strip_object_keys(src: &str) -> String {
    let chars: Vec<char> = src.chars().collect();
    let is_word = |c: char| c.is_alphanumeric() || c == '_' || c == '$';
    let mut out = String::with_capacity(src.len());
    let mut i = 0;
    while i < chars.len() {
        if is_word(chars[i]) && !chars[i].is_ascii_digit() {
            let start = i;
            while i < chars.len() && is_word(chars[i]) {
                i += 1;
            }
            let mut j = i;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            // A key, not a reference — and not `::`, which is not a key either.
            if j < chars.len() && chars[j] == ':' && chars.get(j + 1) != Some(&':') {
                out.push(' ');
                continue;
            }
            out.extend(&chars[start..i]);
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// Is this exactly one double-quoted string literal, quotes at both ends and
/// nothing outside them?
fn is_single_string_literal(v: &str) -> bool {
    let mut chars = v.chars();
    if chars.next() != Some('"') {
        return false;
    }
    let mut escaped = false;
    let mut closed = false;
    for c in chars {
        if closed {
            // Anything after the closing quote means this is an expression.
            return false;
        }
        if escaped {
            escaped = false;
        } else if c == '\\' {
            escaped = true;
        } else if c == '"' {
            closed = true;
        }
    }
    closed
}

/// Is this a value a module-scope binding can hold?
///
/// Sigil evaluates a top-level `≔` at compile time, so anything with a call, a
/// field access or an object literal in it is rejected outright — and a rejected
/// one takes the whole file with it, which is worse than the undefined name it
/// was meant to fix. Literals and arrays of literals only.
fn is_constant_expression(value: &str) -> bool {
    fn atom(v: &str) -> bool {
        let v = v.trim();
        if v.is_empty() {
            return false;
        }
        matches!(v, "true" | "false" | "None" | "∅")
            // A STRING LITERAL, not merely something that begins and ends with a
            // quote: `"[" + identity("name") + "]"` does both, and was emitted as
            // a module binding — "expression is not constant", from the backend,
            // for a concatenation with a call in the middle.
            || is_single_string_literal(v)
            || v.trim_start_matches('-')
                .chars()
                .all(|c| c.is_ascii_digit() || c == '.' || c == '_')
    }

    // An array literal is not a constant expression to this backend either,
    // measured rather than assumed: `≔ k = ["a","b"];` compiles under `check` and
    // fails under `wasm`.
    atom(value)
}

/// Replace whole-word occurrences of `from` with `to`. Used to substitute a
/// functional updater's parameter for the field it stands for; a plain
/// `str::replace` would rewrite `prevented` and `x.prev` too.
fn replace_ident(src: &str, from: &str, to: &str) -> String {
    let mut out = String::with_capacity(src.len());
    let chars: Vec<char> = src.chars().collect();
    let target: Vec<char> = from.chars().collect();
    let is_word = |c: char| c.is_alphanumeric() || c == '_' || c == '$';
    let mut i = 0;
    while i < chars.len() {
        let matches = i + target.len() <= chars.len()
            && chars[i..i + target.len()] == target[..]
            && (i == 0 || !(is_word(chars[i - 1]) || chars[i - 1] == '.'))
            && (i + target.len() == chars.len() || !is_word(chars[i + target.len()]));
        if matches {
            out.push_str(to);
            i += target.len();
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

fn to_snake_case(s: &str) -> String {
    crate::migrate::react::spec::to_snake_case(s)
}

fn to_pascal_case(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = true;
    for c in s.chars() {
        if c == '_' || c == '-' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(c.to_uppercase().next().unwrap());
            capitalize_next = false;
        } else {
            result.push(c);
        }
    }
    result
}

fn escape_string(s: &str) -> String {
    s.replace('\\', "\\\\")
     .replace('"', "\\\"")
     .replace('\n', "\\n")
     .replace('\t', "\\t")
}

// =============================================================================
// JavaScript to Sigil Expression Transformation
// =============================================================================

/// Transform JavaScript expression syntax to Sigil syntax.
/// Converts operators, method calls, and identifiers.
fn transform_js_to_sigil(code: &str) -> String {
    // FIRST: Handle template literals to prevent ${expr} from being misinterpreted as object literals
    // JS template literals: `foo` → "foo", ${expr} → stripped
    let mut result = transform_template_literals(code);

    // Handle JS object literals { key: value } or { foo, bar } → ∅
    // These can't be directly represented, so simplify
    result = transform_object_literals(&result);

    // Handle TypeScript type casts: (expr as Type) → expr
    result = transform_type_casts(&result);

    // Handle array indexing: arr[idx] → arr.get(idx)
    result = transform_array_indexing(&result);

    // First transform arrow functions to Sigil closures
    result = transform_arrow_functions(&result);

    // Transform ternary operator: a ? b : c → (if a { b } else { c })
    result = transform_ternary(&result);

    // Transform JS operators to Sigil operators
    // Order matters: longer patterns first to avoid partial matches

    // Logical operators
    result = result.replace(" && ", " ∧ ");
    result = result.replace(" || ", " ∨ ");
    result = result.replace("&&", " ∧ ");
    result = result.replace("||", " ∨ ");

    // Comparison operators (handle !== and === before != and ==)
    // Sigil uses == for equality (not =, which is assignment)
    result = result.replace("!==", "≠");
    result = result.replace("===", "==");
    result = result.replace("!=", "≠");
    // == stays as == in Sigil

    // Unary not: !foo → ¬foo (be careful not to replace != which is already handled)
    // Use a simple approach: replace ! at word boundaries
    result = transform_unary_not(&result);

    // JS method calls to Sigil
    result = result.replace(".length", ".len()");
    result = result.replace(".toString()", ".to_string()");
    result = result.replace(".trim()", ".trim()");
    result = result.replace(".toLowerCase()", ".to_lowercase()");
    result = result.replace(".toUpperCase()", ".to_uppercase()");
    result = result.replace(".includes(", ".contains(");
    result = result.replace(".indexOf(", ".find(");
    result = result.replace(".startsWith(", ".starts_with(");
    result = result.replace(".endsWith(", ".ends_with(");
    result = result.replace(".push(", ".append(");
    result = result.replace(".pop()", ".pop()");
    result = result.replace(".shift()", ".remove(0)");
    result = result.replace(".slice(", ".slice(");
    result = result.replace(".join(", ".join(");
    result = result.replace(".split(", ".split(");
    result = result.replace(".map(", ".map(");
    result = result.replace(".filter(", ".filter(");
    result = result.replace(".find(", ".find(");
    result = result.replace(".some(", ".any(");
    result = result.replace(".every(", ".all(");
    result = result.replace(".reduce(", ".fold(");

    // JS string quotes: 'foo' → "foo"
    result = transform_single_quotes(&result);

    // Note: Template literals are now handled at the start of this function
    // to prevent ${expr} from being misinterpreted as object literals

    // JS boolean literals: true → True, false → False
    // Must be done BEFORE snake_case transform to avoid true → True being converted to true
    result = transform_boolean_literals(&result);

    // Transform camelCase identifiers to snake_case in the expression
    result = transform_identifiers_to_snake_case(&result);

    result
}

/// Transform JavaScript arrow functions to Sigil closures.
/// Examples:
///   p => expr       →  |p| expr
///   (p) => expr     →  |p| expr
///   (a, b) => expr  →  |a, b| expr
fn transform_arrow_functions(code: &str) -> String {
    let mut result = code.to_string();

    // Pattern: identifier => expr (simple arrow function without parens)
    // e.g., "p => { id: p }" → "|p| { id: p }"
    let simple_arrow_re = regex::Regex::new(r"(\b)([a-zA-Z_][a-zA-Z0-9_]*)\s*=>\s*").unwrap();
    result = simple_arrow_re.replace_all(&result, "$1|$2| ").to_string();

    // Pattern: (params) => expr (arrow function with parens)
    // e.g., "(a, b) => a + b" → "|a, b| a + b"
    let paren_arrow_re = regex::Regex::new(r"\(([^)]*)\)\s*=>\s*").unwrap();
    result = paren_arrow_re.replace_all(&result, "|$1| ").to_string();

    result
}

/// Transform JavaScript ternary operators to Sigil if-else expressions.
/// For simple cases like standalone expressions: a ? b : c → if a { b } else { c }
/// For complex cases inside method calls: we simplify to just the first value (true branch)
fn transform_ternary(code: &str) -> String {
    // For ternaries inside method calls/arguments, use simpler transform
    // This avoids parser issues with (if...) inside method arguments
    // Just use the consequent (true branch) as a simplified fallback
    //
    // Pattern matches: condition ? true_branch : false_branch
    // The condition can be a complex expression (e.g., a == b, a && b, etc.)
    // We capture everything up to the first ? as condition, then true branch, then false branch
    //
    // Use non-greedy matching for the condition part, and greedy for branches
    let re = regex::Regex::new(r"(.+?)\s*\?\s*(.+?)\s*:\s*(.+)$").unwrap();
    re.replace_all(code, "$2").to_string()
}

/// Transform JavaScript template literals to Sigil strings.
/// Converts backticks to double quotes. For complex interpolations,
/// we simplify by just taking the static parts.
fn transform_template_literals(code: &str) -> String {
    let mut result = String::new();
    let chars: Vec<char> = code.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '`' {
            // Convert backtick to double quote
            result.push('"');
            i += 1;
        } else if chars[i] == '$' && i + 1 < chars.len() && chars[i + 1] == '{' {
            // Skip ${...} interpolation entirely (just remove it)
            i += 2; // Skip ${
            let mut brace_depth = 1;
            while i < chars.len() && brace_depth > 0 {
                if chars[i] == '{' {
                    brace_depth += 1;
                } else if chars[i] == '}' {
                    brace_depth -= 1;
                }
                i += 1;
            }
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    result
}

/// Transform unary not operator: !foo → ¬foo
fn transform_unary_not(code: &str) -> String {
    let mut result = String::new();
    let chars: Vec<char> = code.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '!' {
            // Check if this is part of != or !== (already transformed to ≠)
            if i + 1 < chars.len() && chars[i + 1] == '=' {
                // This shouldn't happen since we already replaced != with ≠
                result.push(chars[i]);
            } else {
                // This is unary not
                result.push('¬');
            }
        } else {
            result.push(chars[i]);
        }
        i += 1;
    }

    result
}

/// Transform JavaScript boolean literals to Sigil format.
/// true → True, false → False
fn transform_boolean_literals(code: &str) -> String {
    // Use word boundary matching to avoid changing "trueValue" to "TrueValue"
    let true_re = regex::Regex::new(r"\btrue\b").unwrap();
    let result = true_re.replace_all(code, "True").to_string();

    let false_re = regex::Regex::new(r"\bfalse\b").unwrap();
    false_re.replace_all(&result, "False").to_string()
}

/// Transform JavaScript single quotes to double quotes.
fn transform_single_quotes(code: &str) -> String {
    let mut result = String::new();
    let mut in_single_quote = false;
    let mut in_double_quote = false;

    for c in code.chars() {
        match c {
            '\'' if !in_double_quote => {
                in_single_quote = !in_single_quote;
                result.push('"');
            }
            '"' if !in_single_quote => {
                in_double_quote = !in_double_quote;
                result.push('"');
            }
            _ => result.push(c),
        }
    }

    result
}

/// Transform camelCase identifiers to snake_case.
/// Only transforms identifiers, not strings or operators.
fn transform_identifiers_to_snake_case(code: &str) -> String {
    let mut result = String::new();
    let mut current_ident = String::new();
    let mut in_string = false;
    let mut string_char = '"';

    for c in code.chars() {
        // Track string state
        if (c == '"' || c == '\'') && !in_string {
            in_string = true;
            string_char = c;
            // Flush current identifier
            if !current_ident.is_empty() {
                result.push_str(&camel_to_snake(&current_ident));
                current_ident.clear();
            }
            result.push(c);
            continue;
        }
        if c == string_char && in_string {
            in_string = false;
            result.push(c);
            continue;
        }
        if in_string {
            result.push(c);
            continue;
        }

        // Build identifiers
        if c.is_alphanumeric() || c == '_' {
            current_ident.push(c);
        } else {
            // Flush current identifier
            if !current_ident.is_empty() {
                result.push_str(&camel_to_snake(&current_ident));
                current_ident.clear();
            }
            result.push(c);
        }
    }

    // Flush remaining identifier
    if !current_ident.is_empty() {
        result.push_str(&camel_to_snake(&current_ident));
    }

    result
}

/// Transform JavaScript object literals to Sigil.
/// { key: value, ... } or { foo, bar } → ∅ (simplified for now)
fn transform_object_literals(code: &str) -> String {
    let mut result = String::new();
    let chars: Vec<char> = code.chars().collect();
    let mut i = 0;
    let mut depth = 0;
    let mut brace_start = None;
    let mut in_string = false;
    let mut string_char = '"';

    while i < chars.len() {
        let c = chars[i];

        // Track string state
        if (c == '"' || c == '\'') && !in_string {
            in_string = true;
            string_char = c;
            result.push(c);
            i += 1;
            continue;
        }
        if c == string_char && in_string {
            in_string = false;
            result.push(c);
            i += 1;
            continue;
        }
        if in_string {
            result.push(c);
            i += 1;
            continue;
        }

        if c == '{' {
            if depth == 0 {
                brace_start = Some(result.len());
            }
            depth += 1;
            result.push(c);
        } else if c == '}' {
            depth -= 1;
            if depth == 0 {
                // Check if this looks like an object literal (not a block)
                if let Some(start) = brace_start {
                    let content = &result[start + 1..];
                    // Object literal indicators: contains ':', or is shorthand { foo, bar }
                    if content.contains(':') || (content.contains(',') && !content.contains(';')) {
                        // Replace the whole object literal with ∅
                        result.truncate(start);
                        result.push_str("∅");
                        brace_start = None;
                        i += 1;
                        continue;
                    }
                }
                brace_start = None;
            }
            result.push(c);
        } else {
            result.push(c);
        }
        i += 1;
    }

    result
}

/// Transform TypeScript type casts: (expr as Type) → expr, expr as Type → expr
fn transform_type_casts(code: &str) -> String {
    // Pattern: (expr as Type) → expr
    let paren_cast_re = regex::Regex::new(r"\(([^)]+)\s+as\s+[A-Za-z_][A-Za-z0-9_\[\]<>]*\)").unwrap();
    let result = paren_cast_re.replace_all(code, "$1").to_string();

    // Pattern: expr as Type → expr (without parens)
    let cast_re = regex::Regex::new(r"(\b[A-Za-z_][A-Za-z0-9_.()]*)\s+as\s+[A-Za-z_][A-Za-z0-9_\[\]<>]*").unwrap();
    cast_re.replace_all(&result, "$1").to_string()
}

/// Transform JavaScript array indexing: arr[idx] → arr.get(idx)
fn transform_array_indexing(code: &str) -> String {
    // Pattern: identifier[expr] → identifier.get(expr)
    // Be careful not to transform things like:
    // - Type[] or generic syntax
    // - CSS custom properties: text-[var(--accent-primary)]
    // - Tailwind CSS classes: bg-[#fff], w-[100px]
    //
    // Only transform if the bracket content looks like a simple identifier
    // (not CSS values which typically have dashes, hashes, or parens)
    let bracket_re = regex::Regex::new(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\[([a-zA-Z_][a-zA-Z0-9_]*)\]").unwrap();

    // First check if we're inside CSS (contains patterns like text-[, bg-[, etc.)
    // If so, don't transform anything
    if code.contains("-[") {
        return code.to_string();
    }

    bracket_re.replace_all(code, "$1.get($2)").to_string()
}

/// Convert a single camelCase identifier to snake_case.
/// Preserves all-caps abbreviations and numbers.
fn camel_to_snake(s: &str) -> String {
    // Don't transform Sigil boolean literals
    if s == "True" || s == "False" {
        return s.to_string();
    }

    // Don't transform keywords, operators, or already snake_case
    if s.contains('_') || s.chars().all(|c| c.is_lowercase() || c.is_numeric()) {
        return s.to_string();
    }

    // Don't transform if it's all uppercase (likely a constant)
    if s.chars().all(|c| c.is_uppercase() || c.is_numeric()) {
        return s.to_string();
    }

    // Don't transform short identifiers (likely intentional)
    if s.len() <= 2 {
        return s.to_string();
    }

    let mut result = String::new();
    let chars: Vec<char> = s.chars().collect();

    for (i, c) in chars.iter().enumerate() {
        if c.is_uppercase() && i > 0 {
            // Check if this is part of an acronym (multiple caps in a row)
            let prev_is_upper = chars.get(i.saturating_sub(1)).map(|c| c.is_uppercase()).unwrap_or(false);
            let next_is_lower = chars.get(i + 1).map(|c| c.is_lowercase()).unwrap_or(false);

            // Add underscore if transitioning from lowercase or at acronym boundary
            if !prev_is_upper || next_is_lower {
                result.push('_');
            }
        }
        result.push(c.to_lowercase().next().unwrap());
    }

    result
}

/// Normalise an extracted prop type into something that can appear in a Sigil
/// signature.
///
/// Extraction usually yields `Any`, but an inline TypeScript object type comes
/// through raw with its leading colon still attached, e.g.
/// `": { label: string, up: boolean | null }"`. Emitting that produced
/// `props: : { ... }` — a double colon followed by TypeScript — which failed to
/// parse. Anything that is not a plain Sigil type name degrades to `Any`, which
/// is already what the generator uses for the overwhelming majority of props
/// (324 of 328 across the Lares UI).
fn normalize_prop_type(raw: &str) -> String {
    let t = raw.trim().trim_start_matches(':').trim();
    if t.is_empty() {
        return "Any".to_string();
    }
    let simple = t
        .chars()
        .all(|c| c.is_alphanumeric() || matches!(c, '_' | '<' | '>' | ',' | ' ' | '[' | ']'));
    if simple {
        t.to_string()
    } else {
        "Any".to_string()
    }
}
