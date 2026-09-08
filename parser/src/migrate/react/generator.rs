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
        }
    }

    /// Generate complete Sigil file for the component.
    pub fn generate(&self) -> GeneratedSigil {
        let mut code = String::new();

        // Generate imports
        code.push_str(&self.generate_imports());
        code.push('\n');

        // Module-scope constants from the same file
        let (constants, emitted_constants) = self.generate_module_constants();
        if !constants.is_empty() {
            code.push_str(&constants);
            code.push('\n');
        }

        // Helper functions from this file and the modules it imports
        let helpers = self.generate_helper_functions(&emitted_constants);
        if !helpers.is_empty() {
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
            // A module-scope binding has to be a CONSTANT expression, and most of
            // these are not — an object literal, a call, a lookup. Emitting them
            // anyway cost seven files ("expression is not constant") and ten
            // strict-clean ones, so only literal-shaped values are emitted.
            // `≔ x = None;` at module scope passes `check` and fails to
            // compile (§8.2.9) — and `None` is what the transform yields for
            // anything it could not read, so this arm was turning an
            // unreadable initialiser into a file that would not build. A
            // constant nobody can express is a comment.
            if value.trim() == "None" || value.trim() == "∅" {
                // Not emitted: nothing may depend on it.
                lines.push(format!(
                    "// {} — initialiser did not survive transformation. React: {}",
                    name, one_line
                ));
                continue;
            }

            if is_constant_expression(&value) {
                lines.push(format!("≔ {} = {};", name, value));
                known.insert(name.clone());
                emitted.insert(name);
                continue;
            }

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
                lines.push(format!(
                    "// {} — not a constant expression. React: {}",
                    name, one_line
                ));
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

        let mut emitted: Vec<String> = Vec::new();
        let mut done: std::collections::HashSet<String> = std::collections::HashSet::new();
        loop {
            let mut progress = false;
            for h in &pending {
                let name = to_snake_case(&h.name);
                if done.contains(&name) {
                    continue;
                }
                let Some(expr) = h.returns_expression.as_ref() else {
                    continue;
                };
                let mut inner = scope.clone();
                let mut params: Vec<String> = Vec::new();
                let mut local_known = known.clone();
                for p in &h.parameters {
                    let pn = to_snake_case(&p.name);
                    // A parameter with a default is optional at the call site,
                    // and Sigil has no defaults — declaring it made every call
                    // "expected 2 arguments, found 1". It stays in scope for the
                    // body; it just is not in the signature.
                    if p.default_value.is_none() && !p.optional {
                        params.push(format!("{}: Any", pn));
                    }
                    inner.locals.push(p.name.clone());
                    local_known.insert(pn);
                }
                let body = self.transform_expression_free(expr, &inner);
                if !free_identifiers_known(&body, &local_known) {
                    continue;
                }
                emitted.push(format!(
                    "☉ rite {}({}) -> Any! {{\n    {}\n}}",
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
                .filter(|p| p.default_value.is_none() && !p.optional)
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
        let view = self.generate_view_method();
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

    fn generate_message_handlers(&self) -> String {
        let handlers: Vec<String> = self.spec.recommendations.messages.iter()
            .map(|msg| {
                let mut body_parts: Vec<String> = Vec::new();

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
    fn generate_locals(&self, indent: &str, scope: &mut VNodeScope) -> String {
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

    fn generate_view_method(&self) -> String {
        let jsx = &self.spec.source.extraction.jsx;
        let mut scope = VNodeScope::default();

        // Locals first: they extend the scope the body is generated against.
        let locals = self.generate_locals("        ", &mut scope);

        let body = if let Some(root) = &jsx.root {
            self.generate_vnode(root, 2, &scope)
        } else {
            "        VNode·div()".to_string()
        };

        format!("    rite view(self) -> VNode! {{\n{}{}\n    }}", locals, body)
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
        let locals = self.generate_locals("    ", &mut scope);
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
                // Expression interpolation - convert to text_child with to_string
                let expr = self.transform_expression_scoped(code, scope);
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
pub fn generate_component(spec: &ComponentMigrationSpec) -> GeneratedSigil {
    let generator = QliphothGenerator::new(spec);
    generator.generate()
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

    fn transform_expression_with(
        &self,
        code: &str,
        scope: &VNodeScope,
        prefix_self: bool,
    ) -> String {
        // Handle placeholder/invalid expressions
        if code.contains("/*") || code.is_empty() {
            return "None".to_string();
        }

        // Clean up the expression
        let code = code.trim();

        // Handle JSX in expressions - these need special handling
        // For now, replace with None as a placeholder
        if code.contains('<') && code.contains('>') {
            return "None".to_string();
        }

        // Build transformation config
        let state_fields: Vec<String> = self.spec.recommendations.state_fields
            .iter()
            .map(|f| f.to_field.clone())
            .collect();

        // Module-scope names are in scope for the whole file and are emitted as
        // top-level items, so a reference to one must be left alone rather than
        // swept into `self.`.
        let mut locals = scope.locals.clone();
        locals.extend(self.spec.source.module_scope.iter().cloned());

        let config = TransformConfig {
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
        };

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
    fn generate_attribute_scoped(&self, attr: &JsxAttribute, scope: &VNodeScope) -> String {
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
                        } else if !transformed.starts_with('"') && !transformed.contains('"') {
                            // Simple identifier, wrap in quotes
                            format!("·class(\"{}\")", transformed)
                        } else {
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
                    JsxAttributeValue::True => format!("·attr(\"{}\", \"true\")", attr.name),
                    JsxAttributeValue::Expression { code } => {
                        // `·when(cond, |n| n·attr(…))` — a closure — is not a form
                        // Qliphoth's builder has: "No DSL, no function types, no
                        // callbacks" is the first thing `core/vdom.sigil` says.
                        // `when` takes a VNode child, so 66 of these compiled
                        // against a signature that does not exist. `attr_when` is
                        // the same idea without the closure.
                        let transformed = self.transform_expression_scoped(code, scope);
                        format!("·attr_when({}, \"{}\", \"true\")", transformed, attr.name)
                    }
                    _ => String::new(),
                }
            }
            "href" | "src" | "alt" | "type" | "name" | "value" | "placeholder" => {
                // Common attributes
                match &attr.value {
                    JsxAttributeValue::String { value } => {
                        format!("·attr(\"{}\", \"{}\")", attr.name, escape_string(value))
                    }
                    JsxAttributeValue::Expression { code } => {
                        let transformed = self.transform_expression_scoped(code, scope);
                        format!("·attr(\"{}\", {})", attr.name, transformed)
                    }
                    _ => String::new(),
                }
            }
            _ => {
                // Generic attribute
                match &attr.value {
                    JsxAttributeValue::String { value } => {
                        format!("·attr(\"{}\", \"{}\")", attr.name, escape_string(value))
                    }
                    JsxAttributeValue::Expression { code } => {
                        let transformed = self.transform_expression_scoped(code, scope);
                        format!("·attr(\"{}\", {})", attr.name, transformed)
                    }
                    JsxAttributeValue::Spread { name } => {
                        // Spread attributes can't be directly represented, skip
                        String::new()
                    }
                    JsxAttributeValue::True => {
                        format!("·attr(\"{}\", \"true\")", attr.name)
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

/// Free-standing functions the WASM backend resolves, which a helper body may
/// therefore name without the file declaring anything.
const HOST_FUNCTIONS: &[&str] = &[
    "to_bool", "to_fixed", "to_string", "is_array", "is_finite", "is_nan", "is_integer",
    "object_values", "object_keys", "object_entries", "json_parse", "json_stringify",
    "json_pretty", "json_get", "json_set", "timing_now", "timing_parse", "math_random",
    "None", "true", "false", "Some", "Ok", "Err", "This", "Self", "VNode", "VElement",
    "HashMap", "HashSet", "Vec", "String",
];

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
fn free_identifiers_known(
    value: &str,
    known: &std::collections::HashSet<String>,
) -> bool {
    let chars: Vec<char> = value.chars().collect();
    let is_word = |c: char| c.is_alphanumeric() || c == '_' || c == '$';
    let mut i = 0;
    let mut in_str = false;
    let mut escaped = false;
    while i < chars.len() {
        let c = chars[i];
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
            if after_sep || is_key {
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
            || (v.starts_with('"') && v.ends_with('"') && v.len() >= 2)
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
