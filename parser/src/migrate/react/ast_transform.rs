//! AST-based JavaScript to Sigil expression transformation.
//!
//! This module provides reliable transformation of JavaScript/TypeScript expressions
//! to Sigil syntax by parsing the source into an AST and walking it, rather than
//! using fragile regex-based string manipulation.
//!
//! ## Transformation Rules
//!
//! | JavaScript | Sigil |
//! |------------|-------|
//! | `a && b` | `a ∧ b` |
//! | `a \|\| b` | `a ∨ b` |
//! | `!a` | `¬a` |
//! | `a === b` | `a == b` |
//! | `a !== b` | `a ≠ b` |
//! | `a ? b : c` | `⎇ a { b } ⎉ { c }` |
//! | `() => expr` | `\|\| expr` |
//! | `(x) => expr` | `\|x\| expr` |
//! | `arr.length` | `arr.len()` |
//! | `str.toString()` | `str.to_string()` |
//! | Template literal | Regular string (interpolations marked) |

use swc_common::{SourceMap, FilePathMapping, FileName, Spanned, DUMMY_SP, sync::Lrc};
use swc_ecma_parser::{Parser, StringInput, Syntax, TsSyntax};
use swc_ecma_ast::*;

/// Configuration for expression transformation
#[derive(Debug, Clone, Default)]
pub struct TransformConfig {
    /// If true, prefix simple identifiers with `self.` (for actor state)
    pub prefix_self: bool,
    /// State field names that should be prefixed with `self.`
    pub state_fields: Vec<String>,
    /// Local variables that should NOT be prefixed (e.g., map iterator vars)
    pub locals: Vec<String>,
    /// Prop parameter names (for pure function components)
    pub props: Vec<String>,
    /// Module constants the generator emitted as LOOKUP FUNCTIONS because their
    /// value is an object literal, which a module-scope binding cannot hold.
    /// `DEFAULT_SORT[view]` has to become `default_sort(view)` to match.
    pub lookup_constants: Vec<String>,
    /// Module constants the generator emitted as zero-argument functions, for
    /// the same reason: an array or a computed value is not a constant
    /// expression, but it is a perfectly good function body. A bare reference to
    /// one has to become a call.
    pub value_constants: Vec<String>,
    /// Declared parameter count of the helper functions in scope, by snake-cased
    /// name.
    ///
    /// Sigil has no default parameters, so a helper written
    /// `evidenceIdFromThread(texts, questionId = null)` declares two and every
    /// call site has to pass two. Dropping the defaulted parameter from the
    /// signature instead breaks the call sites that DO pass it — both shapes
    /// occur in the same client. Declaring all of them and padding the short
    /// calls with ∅ is the one rule that fits both.
    pub fn_arity: std::collections::HashMap<String, usize>,
}

/// Result of transforming a JS expression
#[derive(Debug, Clone)]
pub struct TransformResult {
    /// The transformed Sigil code
    pub code: String,
    /// Whether the transformation was complete (vs. falling back to placeholder)
    pub complete: bool,
    /// Any warnings or notes about the transformation
    pub warnings: Vec<String>,
}

impl TransformResult {
    fn complete(code: String) -> Self {
        Self { code, complete: true, warnings: vec![] }
    }

    fn incomplete(code: String, reason: &str) -> Self {
        Self {
            code,
            complete: false,
            warnings: vec![reason.to_string()],
        }
    }

    fn placeholder(reason: &str) -> Self {
        Self {
            code: "None".to_string(),
            complete: false,
            warnings: vec![reason.to_string()],
        }
    }
}

/// Transform a JavaScript expression string to Sigil syntax.
///
/// This parses the expression using swc and transforms the AST.
/// Falls back gracefully if parsing fails.
pub fn transform_expression(code: &str, config: &TransformConfig) -> TransformResult {
    let code = code.trim();

    // Handle empty or trivial cases
    if code.is_empty() {
        return TransformResult::complete("None".to_string());
    }

    // Try to parse as expression
    match parse_expression(code) {
        Ok(expr) => {
            let mut transformer = ExprTransformer::new(config, code);
            let result = transformer.transform_expr(&expr);
            TransformResult {
                code: result,
                complete: transformer.warnings.is_empty(),
                warnings: transformer.warnings,
            }
        }
        Err(e) => {
            // Parsing failed - use conservative fallback
            let fallback = conservative_transform(code, config);
            TransformResult::incomplete(
                fallback,
                &format!("Parse error, using fallback: {}", e)
            )
        }
    }
}

/// Parse a JavaScript expression string into an AST.
fn parse_expression(code: &str) -> Result<Box<Expr>, String> {
    let cm: Lrc<SourceMap> = Lrc::new(SourceMap::new(FilePathMapping::empty()));
    let fm = cm.new_source_file(FileName::Anon.into(), code.to_string());

    let mut parser = Parser::new(
        Syntax::Typescript(TsSyntax {
            tsx: true,
            ..Default::default()
        }),
        StringInput::from(&*fm),
        None,
    );

    parser.parse_expr().map_err(|e| format!("{:?}", e))
}

/// Conservative string-based fallback for when AST parsing fails.
/// Only does safe, simple replacements.
fn conservative_transform(code: &str, config: &TransformConfig) -> String {
    let mut result = code.to_string();

    // Only do the safest replacements
    result = result.replace("===", "==");
    result = result.replace("!==", "≠");
    result = result.replace(" && ", " ∧ ");
    result = result.replace(" || ", " ∨ ");
    result = result.replace(".length", ".len()");

    // If it's a simple identifier and we're in actor context, prefix with self.
    if config.prefix_self && is_simple_identifier(&result) && !config.locals.contains(&result) {
        result = format!("self.{}", to_snake_case(&result));
    }

    result
}

/// Check if a string is a simple identifier (no operators, dots, etc.)
fn is_simple_identifier(s: &str) -> bool {
    !s.is_empty()
        && s.chars().next().map(|c| c.is_alphabetic() || c == '_').unwrap_or(false)
        && s.chars().all(|c| c.is_alphanumeric() || c == '_')
}

/// Methods an optional call may keep verbatim: Sigil spells them the same, so
/// `a?.b()` coming out as `a\u{b7}b()` means it WAS recognised.
const KNOWN_OPTIONAL_METHODS: &[&str] = &[
    "trim", "to_string", "to_lowercase", "to_uppercase", "len", "push", "pop",
    "join", "split", "map", "filter", "find", "contains", "starts_with",
    "ends_with", "replace", "slice", "reverse", "sort", "insert", "remove",
    "keys", "values", "entries", "clone", "get", "set", "for_each", "all",
    "any", "fold", "flat_map", "char_at", "contains_key", "to_bool", "await",
];

/// A JavaScript global with a Sigil host function behind it — the timers and
/// the three dialogs, which `prefix_self` otherwise turned into actor fields.
fn js_timer_global(name: &str) -> Option<&'static str> {
    Some(match name {
        "parseFloat" => "parse_float",
        "confirm" => "browser_confirm",
        "alert" => "browser_alert",
        "prompt" => "browser_prompt",
        "setTimeout" => "timing_set_timeout",
        "clearTimeout" => "timing_clear_timeout",
        "setInterval" => "timing_set_interval",
        "clearInterval" => "timing_clear_interval",
        "requestAnimationFrame" => "timing_request_animation_frame",
        "cancelAnimationFrame" => "timing_clear_interval",
        _ => return None,
    })
}

/// A JavaScript static call with a Sigil host function behind it.
///
/// `Date.now()` is the one that matters — four of the generated components call
/// it. The receiver arrives already transformed, so it is lower-cased here.
fn js_static_call(obj: &str, method: &str) -> Option<&'static str> {
    Some(match (obj, method) {
        // `window.setTimeout(…)` — the host `timing` group has had these since
        // the start, but nothing routed the browser spelling to them, so
        // `window·clear_timeout(id)` reached the backend as an undefined
        // function. The bare-global spelling is handled in `transform_call`.
        ("window", "confirm") => "browser_confirm",
        ("window", "alert") => "browser_alert",
        ("window", "prompt") => "browser_prompt",
        ("window", "setTimeout") => "timing_set_timeout",
        ("window", "clearTimeout") => "timing_clear_timeout",
        ("window", "setInterval") => "timing_set_interval",
        ("window", "clearInterval") => "timing_clear_interval",
        ("window", "requestAnimationFrame") => "timing_request_animation_frame",
        ("date", "now") => "timing_now",
        ("date", "parse") => "timing_parse",
        ("json", "stringify") => "json_stringify",
        ("json", "parse") => "json_parse",
        ("math", "random") => "math_random",
        // `Math.round(x)` was `math·round(x)`, a lowercase path root that
        // resolves to nothing; the imports have been there since the start.
        ("math", "round") => "math_round",
        ("math", "floor") => "math_floor",
        ("math", "ceil") => "math_ceil",
        ("math", "abs") => "math_abs",
        ("math", "min") => "math_min",
        ("math", "max") => "math_max",
        ("math", "sqrt") => "math_sqrt",
        ("math", "pow") => "math_pow",
        // `Object.values(x)` was becoming `object·values(x)`, which resolves to
        // a `values` free function that does not exist. Same for the other two.
        ("object", "values") => "object_values",
        ("object", "keys") => "object_keys",
        ("object", "entries") => "object_entries",
        ("array", "isArray") => "is_array",
        ("number", "isFinite") => "is_finite",
        ("number", "isNaN") => "is_nan",
        ("number", "isInteger") => "is_integer",
        _ => return None,
    })
}

/// Does a module-scope or local name refer to this identifier?
///
/// Comparing both sides snake-cased is right for the camelCase bindings this
/// was written for — `currentSprint` is declared as `current_sprint` and read
/// as `currentSprint`. It is wrong for anything capitalised: `Error` in the
/// JS globals list snake-cases to `error`, so every component with a
/// `const [error, setError] = useState(null)` had its own state field shadowed
/// by the global and emitted a bare `error` no actor declares. `Set`/`set`,
/// `Map`/`map` and `Date`/`date` are the same collision. A capitalised name
/// matches only itself.
fn scope_name_matches(candidate: &str, ident: &str) -> bool {
    if candidate == ident {
        return true;
    }
    let capitalised = |s: &str| s.chars().next().map(|c| c.is_uppercase()).unwrap_or(false);
    if capitalised(candidate) || capitalised(ident) {
        return false;
    }
    to_snake_case(candidate) == to_snake_case(ident)
}

/// Convert camelCase to snake_case
pub(crate) fn to_snake_case(s: &str) -> String {
    crate::migrate::react::spec::to_snake_case(s)
}

/// snake_case for a member or method name — no keyword escaping. See
/// `spec::to_snake_case_member`.
fn to_snake_case_member(s: &str) -> String {
    crate::migrate::react::spec::to_snake_case_member(s)
}

/// Convert snake_case to PascalCase
fn to_pascal_case(s: &str) -> String {
    s.split('_')
        .map(|part| {
            let mut chars = part.chars();
            match chars.next() {
                Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect()
}

// =============================================================================
// AST Transformer
// =============================================================================

struct ExprTransformer<'a> {
    config: &'a TransformConfig,
    /// The expression's own source, so an arrow with a block body can be sliced
    /// out and handed to the statement transform.
    src: &'a str,
    /// Names bound by enclosing closures. Separate from `config.locals`, which is
    /// the caller's fixed scope; this one is pushed and popped as arrows nest.
    scope_locals: Vec<String>,
    warnings: Vec<String>,
}

impl<'a> ExprTransformer<'a> {
    fn new(config: &'a TransformConfig, src: &'a str) -> Self {
        Self {
            config,
            src,
            scope_locals: vec![],
            warnings: vec![],
        }
    }

    fn warn(&mut self, msg: &str) {
        self.warnings.push(msg.to_string());
    }

    /// Transform an expression AST node to Sigil code
    fn transform_expr(&mut self, expr: &Expr) -> String {
        match expr {
            // Literals
            Expr::Lit(lit) => self.transform_lit(lit),

            // Identifiers
            Expr::Ident(ident) => self.transform_ident(ident),

            // Binary operations: a + b, a && b, etc.
            Expr::Bin(bin) => self.transform_bin(bin),

            // Unary operations: !a, -a, etc.
            Expr::Unary(unary) => self.transform_unary(unary),

            // Conditional/ternary: a ? b : c
            Expr::Cond(cond) => self.transform_cond(cond),

            // Member access: obj.prop, arr[idx]
            Expr::Member(member) => self.transform_member(member),

            // Function calls: foo(), obj.method()
            Expr::Call(call) => self.transform_call(call),

            // Arrow functions: () => x, (a, b) => a + b
            Expr::Arrow(arrow) => self.transform_arrow(arrow),

            // Parenthesized: (expr)
            Expr::Paren(paren) => {
                let inner = self.transform_expr(&paren.expr);
                format!("({})", inner)
            }

            // Template literals: `hello ${name}`
            Expr::Tpl(tpl) => self.transform_template(tpl),

            // Array literals: [1, 2, 3]
            Expr::Array(arr) => self.transform_array(arr),

            // Object literals: { a: 1, b: 2 }
            Expr::Object(obj) => self.transform_object(obj),

            // This expression
            Expr::This(_) => "self".to_string(),

            // Assignment: a = b
            Expr::Assign(assign) => self.transform_assign(assign),

            // Sequence: a, b, c
            Expr::Seq(seq) => {
                // Take the last expression
                if let Some(last) = seq.exprs.last() {
                    self.transform_expr(last)
                } else {
                    "None".to_string()
                }
            }

            // Optional chaining: a?.b
            Expr::OptChain(opt) => self.transform_opt_chain(opt),

            // Await expression
            Expr::Await(await_expr) => {
                let inner = self.transform_expr(&await_expr.arg);
                // `await p.then(cb)` — `then` already became a block that awaits
                // inside it, so a second `.await` would await the block's value.
                if inner.trim_start().starts_with('{') && inner.trim_end().ends_with('}') {
                    return inner;
                }
                format!("{}.await", inner)
            }

            // New expression: new Foo()
            Expr::New(new_expr) => {
                let callee = self.transform_expr(&new_expr.callee);
                let args = new_expr.args.as_ref()
                    .map(|args| self.transform_args(args))
                    .unwrap_or_default();
                // `new Set(…)` and `new Map(…)` are Sigil's `HashSet` and
                // `HashMap`. Lowercasing the constructor produced `set·new(…)`
                // and `map·new(…)`, which resolve to nothing — and under S34 a
                // lowercase unresolved call compiles to a constant 0, so the
                // collection silently became a number.
                // A Sigil program's only representation of a date is epoch
                // millis, so `new Date(s)` IS `timing_parse(s)` — `date·new(s)`
                // named nothing, and `date·new(s)·to_string()` named nothing
                // twice.
                if callee == "date" {
                    return if args.trim().is_empty() {
                        "timing_now()".to_string()
                    } else {
                        format!("timing_parse({})", args)
                    };
                }
                // `new Promise((resolve) => …)` has no Sigil analogue: Qliphoth
                // actors communicate by message, and there is no value to hand a
                // resolver to. It reached the backend as `promise·new(…)`, whose
                // callback then called an undefined `resolve`.
                if callee == "promise" {
                    self.warn("new Promise — actors have no promises");
                    let flat: String =
                        args.split_whitespace().collect::<Vec<_>>().join(" ");
                    let capped = if flat.chars().count() > 80 {
                        flat.chars().take(77).collect::<String>() + "..."
                    } else {
                        flat
                    };
                    return format!(
                        "/* new Promise({}) — Sigil has no promises */ \u{2205}",
                        capped.replace("*/", "* /")
                    );
                }
                let collection = match callee.as_str() {
                    "set" => Some("HashSet"),
                    "map" => Some("HashMap"),
                    _ => None,
                };
                if let Some(ty) = collection {
                    return if args.trim().is_empty() {
                        format!("{}·new()", ty)
                    } else {
                        format!("{}·from({})", ty, args)
                    };
                }
                format!("{}·new({})", callee, args)
            }

            // TypeScript cast: expr as Type
            Expr::TsAs(ts_as) => {
                // Just return the expression, ignoring the type
                self.transform_expr(&ts_as.expr)
            }

            // TypeScript non-null assertion: expr!
            Expr::TsNonNull(non_null) => {
                self.transform_expr(&non_null.expr)
            }

            // JSX element (shouldn't appear in expressions normally)
            Expr::JSXElement(_) | Expr::JSXFragment(_) => {
                self.warn("JSX in expression context");
                "/* JSX */".to_string()
            }

            // Fallback for unhandled cases
            _ => {
                self.warn(&format!("Unhandled expression type: {:?}", std::mem::discriminant(expr)));
                "None".to_string()
            }
        }
    }

    fn transform_lit(&mut self, lit: &Lit) -> String {
        match lit {
            Lit::Str(s) => {
                let val = s.value.as_str().unwrap_or("");
                format!("\"{}\"", val.replace('"', "\\\""))
            }
            Lit::Num(n) => n.value.to_string(),
            Lit::Bool(b) => if b.value { "true" } else { "false" }.to_string(),
            Lit::Null(_) => "None".to_string(),
            Lit::BigInt(bi) => bi.value.to_string(),
            Lit::Regex(r) => {
                self.warn("Regex literal");
                // Regex exp and flags are Atom, need to convert to string
                format!("∅ /* regex: /{}/{} — unsupported */", &r.exp, &r.flags)
            }
            Lit::JSXText(t) => {
                // JSXText value is Atom, convert to string
                let val = t.value.to_string();
                format!("\"{}\"", val.trim())
            }
        }
    }

    fn transform_ident(&mut self, ident: &Ident) -> String {
        let name = ident.sym.to_string();

        // Check for JS boolean literals (swc might parse them as identifiers in some contexts)
        if name == "true" || name == "false" {
            return name;
        }
        if name == "null" || name == "undefined" {
            return "None".to_string();
        }

        // `xs.filter(Boolean)` — the constructor used as a truthiness
        // predicate. Passed through, it was an undefined variable.
        if name == "Boolean" {
            return "|__b| __b\u{b7}to_bool()".to_string();
        }

        // Check if it's a local variable (shouldn't be prefixed).
        //
        // Compared snake-cased on both sides. Locals are registered under their
        // emitted Sigil name (`current_sprint`) while the AST still carries the
        // JS one (`currentSprint`), so an exact match never fired and every local
        // was rewritten to `self.current_sprint` — a field that does not exist.
        let snake = to_snake_case(&name);

        // A module constant the generator emitted as a function — checked before
        // anything else, because the name is also in `locals` (it is module
        // scope) and would otherwise come back bare and unresolved.
        if self
            .config
            .value_constants
            .iter()
            .any(|c| to_snake_case(c) == snake)
        {
            return format!("{}()", snake);
        }

        if self.config.locals.iter().any(|l| scope_name_matches(l, &name))
            || self.scope_locals.iter().any(|l| scope_name_matches(l, &name))
        {
            return snake;
        }

        // Check if it's a prop.
        //
        // In a pure function component a prop is a parameter, so the bare name is
        // right. In an ACTOR it is a field: `rite new(…)` assigns every prop to
        // `self.<name>` and the state block declares it, so referring to it bare
        // in the view resolves to nothing. This arm returned bare either way, so
        // `hunk.header` in an actor whose state includes `hunk` came out as an
        // undefined name — and `sigil check` does not resolve names, so nothing
        // said so until the file was compiled.
        if self.config.props.iter().any(|p| to_snake_case(p) == snake) {
            return if self.config.prefix_self {
                format!("self.{}", snake)
            } else {
                snake
            };
        }

        // Check if we should prefix with self (actor state)
        if self.config.prefix_self {
            // Check if it's a known state field
            let snake_name = to_snake_case(&name);
            if self.config.state_fields.iter().any(|f| to_snake_case(f) == snake_name) {
                return format!("self.{}", snake_name);
            }
            // For unknown identifiers in actor context, still prefix with self
            // (it's likely a state field we didn't detect)
            return format!("self.{}", snake_name);
        }

        to_snake_case(&name)
    }

    fn transform_bin(&mut self, bin: &BinExpr) -> String {
        // Special case: `condition && "string"` pattern (common in React className)
        // This should become `⎇ condition { "string" } ⎉ { "" }`
        if bin.op == BinaryOp::LogicalAnd {
            if self.is_non_boolean_value(&bin.right) {
                let cond_src = self.transform_expr(&bin.left);
                let cond = self.boolify(&bin.left, cond_src);
                let value = self.transform_expr(&bin.right);
                return format!("⎇ {} {{ {} }} ⎉ {{ \"\" }}", cond, value);
            }

            // Otherwise it is a genuine boolean conjunction — but Sigil's ∧ takes
            // bools on BOTH sides where JS takes truthiness on both. Boolifying the
            // conjunction as a whole reached only the last operand, so
            // `error && !refreshing` left `error` as a raw Option next to ∧.
            let left = self.transform_expr(&bin.left);
            let right = self.transform_expr(&bin.right);
            let left = self.boolify(&bin.left, left);
            let right = self.boolify(&bin.right, right);
            return format!("{} ∧ {}", left, right);
        }

        // Special case: `condition || defaultValue` (nullish coalescing pattern)
        // Keep as-is if both sides are same type, otherwise use ⎇/⎉
        // JS `a || b` and `a ?? b` are fallbacks, not boolean ors, and `∨` requires
        // bools in Sigil. Emit the conditional they actually mean.
        //
        // This is deliberately unconditional rather than gated on
        // is_non_boolean_value: that heuristic returns false for identifiers, member
        // accesses and index expressions — the very things `||` falls back on in JSX —
        // so gating on it let most real cases through as a bare `∨`.
        //
        // `a` is evaluated twice. For the property accesses that dominate JSX that is
        // fine; a side-effecting left operand would run twice.
        if bin.op == BinaryOp::LogicalOr || bin.op == BinaryOp::NullishCoalescing {
            let left = self.transform_expr(&bin.left);
            let right = self.transform_expr(&bin.right);
            let guard = self.boolify(&bin.left, left.clone());
            return format!("⎇ {} {{ {} }} ⎉ {{ {} }}", guard, left, right);
        }

        let left = self.transform_expr(&bin.left);
        let right = self.transform_expr(&bin.right);

        // `k in obj` is a key test, and `x instanceof T` a type test — neither
        // is an infix operator in Sigil, and both were emitted verbatim, which
        // is a parse error in the generated file.
        if matches!(bin.op, BinaryOp::In) {
            return format!("{}\u{b7}contains_key({})", right, left);
        }
        if matches!(bin.op, BinaryOp::InstanceOf) {
            self.warn("instanceof — Sigil has no runtime class test");
            return format!(
                "/* {} instanceof {} — Sigil has no runtime class test */ false",
                left.replace("*/", "* /"),
                right.replace("*/", "* /")
            );
        }

        let op = match bin.op {
            // Logical (pure boolean operations)
            BinaryOp::LogicalAnd => "∧",
            BinaryOp::LogicalOr => "∨",
            BinaryOp::NullishCoalescing => "∨", // ?? → ∨ (close enough)

            // Comparison
            BinaryOp::EqEq | BinaryOp::EqEqEq => "==",
            BinaryOp::NotEq | BinaryOp::NotEqEq => "≠",
            BinaryOp::Lt => "<",
            BinaryOp::LtEq => "<=",
            BinaryOp::Gt => ">",
            BinaryOp::GtEq => ">=",

            // Arithmetic
            BinaryOp::Add => "+",
            BinaryOp::Sub => "-",
            BinaryOp::Mul => "*",
            BinaryOp::Div => "/",
            BinaryOp::Mod => "%",
            BinaryOp::Exp => "**",

            // Bitwise
            BinaryOp::BitAnd => "&",
            BinaryOp::BitOr => "|",
            BinaryOp::BitXor => "^",
            BinaryOp::LShift => "<<",
            BinaryOp::RShift => ">>",
            BinaryOp::ZeroFillRShift => ">>>",

            // Other
            BinaryOp::In => "in",
            BinaryOp::InstanceOf => "instanceof",
        };

        format!("{} {} {}", left, op, right)
    }

    /// Check if an expression is a non-boolean value (string, number, object, etc.)
    /// Used to detect `cond && "value"` patterns
    fn is_non_boolean_value(&self, expr: &Expr) -> bool {
        match expr {
            // Literals that are clearly non-boolean
            Expr::Lit(Lit::Str(_)) => true,
            Expr::Lit(Lit::Num(_)) => true,
            Expr::Lit(Lit::Null(_)) => true,
            Expr::Lit(Lit::Regex(_)) => true,
            Expr::Lit(Lit::BigInt(_)) => true,

            // Boolean literals are boolean
            Expr::Lit(Lit::Bool(_)) => false,

            // Template literals produce strings
            Expr::Tpl(_) => true,

            // Array and object literals are non-boolean
            Expr::Array(_) => true,
            Expr::Object(_) => true,

            // Function expressions produce functions, not booleans
            Expr::Arrow(_) | Expr::Fn(_) => true,

            // JSX produces elements
            Expr::JSXElement(_) | Expr::JSXFragment(_) => true,

            // Parenthesized: check inner
            Expr::Paren(p) => self.is_non_boolean_value(&p.expr),

            // For other expressions (identifiers, calls, etc.), we can't know
            // Default to false (treat as potentially boolean)
            _ => false,
        }
    }

    fn transform_unary(&mut self, unary: &UnaryExpr) -> String {
        let arg = self.transform_expr(&unary.arg);

        match unary.op {
            // JS `!x` is truthiness-negation over any value; Sigil's ¬ wants a bool.
            UnaryOp::Bang => format!("¬{}", self.boolify(&unary.arg, arg)),
            UnaryOp::Minus => format!("-{}", arg),
            UnaryOp::Plus => arg, // Unary + is a no-op
            UnaryOp::Tilde => format!("~{}", arg),
            // `typeof x` — a host call, because the answer depends on what the
            // host knows about the value. `typeof(…)` named nothing at all.
            UnaryOp::TypeOf => format!("type_of({})", arg),
            UnaryOp::Void => "None".to_string(),
            // `delete obj[k]` is a map removal, and Sigil has one. This used
            // to emit a bare `/* delete … */` — a comment with no value, which
            // is a parse error wherever a statement needs an expression.
            UnaryOp::Delete => match unary.arg.as_ref() {
                Expr::Member(member) => {
                    let obj = self.transform_expr(&member.obj);
                    match &member.prop {
                        MemberProp::Computed(c) => {
                            let key = self.transform_expr(&c.expr);
                            format!("{}\u{b7}remove({})", obj, key)
                        }
                        MemberProp::Ident(id) => format!(
                            "{}\u{b7}remove(\"{}\")",
                            obj,
                            to_snake_case_member(&id.sym.to_string())
                        ),
                        _ => {
                            self.warn("delete operator");
                            format!("/* delete {} */ \u{2205}", arg)
                        }
                    }
                }
                _ => {
                    self.warn("delete operator");
                    format!("/* delete {} */ \u{2205}", arg)
                }
            },
        }
    }

    fn transform_cond(&mut self, cond: &CondExpr) -> String {
        let test = self.transform_expr(&cond.test);
        let cons = self.transform_expr(&cond.cons);
        let alt = self.transform_expr(&cond.alt);

        // A JS ternary test is truthy-evaluated; Sigil's ⎇ wants a real bool. A
        // bare prop (`disabled ? … : …`) would otherwise emit `⎇ disabled` and
        // fail with "if condition must be bool".
        let test = self.boolify(&cond.test, test);

        // Sigil uses ⎇ (U+2387) for if and ⎉ (U+2389) for else
        format!("⎇ {} {{ {} }} ⎉ {{ {} }}", test, cons, alt)
    }

    /// Coerce a JS truthiness test into a Sigil bool, unless it already is one.
    ///
    /// Comparisons, logical operators and `!x` are already boolean; everything
    /// else — identifiers, member accesses, calls — needs `·to_bool()`.
    fn boolify(&self, test_ast: &Expr, rendered: String) -> String {
        let already_bool = match test_ast {
            Expr::Lit(Lit::Bool(_)) => true,
            Expr::Unary(u) => matches!(u.op, UnaryOp::Bang),
            Expr::Paren(p) => return self.boolify(&p.expr, rendered),
            Expr::Bin(b) => match b.op {
                BinaryOp::EqEq
                | BinaryOp::EqEqEq
                | BinaryOp::NotEq
                | BinaryOp::NotEqEq
                | BinaryOp::Lt
                | BinaryOp::LtEq
                | BinaryOp::Gt
                | BinaryOp::GtEq
                | BinaryOp::In
                | BinaryOp::InstanceOf => true,
                // `a && b` renders as `a ∧ b` with both sides boolified, which is a
                // bool — but only on that path. When the right side is a plain value
                // it renders as ⎇/⎉ yielding that value instead.
                BinaryOp::LogicalAnd => !self.is_non_boolean_value(&b.right),
                // `a || b` and `a ?? b` are fallbacks: they render as ⎇/⎉ and yield
                // one of the operands. Calling them bool appended ·to_bool() to
                // nothing, or to the wrong operand.
                BinaryOp::LogicalOr | BinaryOp::NullishCoalescing => false,
                _ => false,
            },
            _ => false,
        };
        if already_bool {
            return rendered;
        }
        // `·to_bool()` binds to the last operand, not the expression. Appending it to
        // `error ∧ ¬refreshing` produced `error ∧ ¬(refreshing·to_bool())`, leaving
        // `error` — an Option — as a bare logical operand. Parenthesise anything that
        // is not a single atom.
        let atomic = !rendered.contains(' ') && !rendered.starts_with('⎇');
        if atomic {
            format!("{}·to_bool()", rendered)
        } else {
            format!("({})·to_bool()", rendered)
        }
    }

    fn transform_member(&mut self, member: &MemberExpr) -> String {
        let obj = self.transform_expr(&member.obj);

        match &member.prop {
            MemberProp::Ident(ident) => {
                let prop = ident.sym.to_string();

                // A lookup table the generator emitted as a function: a named
                // member is a call too, not just an index. `identity.name` on
                // `export const identity = { name: "Lares", … }` was a field
                // access on a function.
                if self
                    .config
                    .lookup_constants
                    .iter()
                    .any(|c| to_snake_case(c) == obj)
                {
                    return format!("{}({:?})", obj, prop);
                }

                // Transform common JS properties to Sigil equivalents
                match prop.as_str() {
                    "length" => format!("{}.len()", obj),
                    _ => format!("{}.{}", obj, to_snake_case_member(&prop)),
                }
            }
            MemberProp::Computed(computed) => {
                let prop = self.transform_expr(&computed.expr);
                // A lookup table the generator emitted as a function: index
                // becomes call. Matched on the emitted (snake-cased) name, which
                // is what `obj` already holds.
                if self
                    .config
                    .lookup_constants
                    .iter()
                    .any(|c| to_snake_case(c) == obj)
                {
                    return format!("{}({})", obj, prop);
                }
                format!("{}[{}]", obj, prop)
            }
            MemberProp::PrivateName(private) => {
                format!("{}._{}", obj, private.name)
            }
        }
    }

    /// `p.then(|v| …)` as a Sigil block expression: bind the awaited value,
    /// then the callback's own body.
    ///
    /// `None` when the callback is not an arrow — there is nothing to inline,
    /// and the caller falls back to marking the call.
    fn then_block(&mut self, receiver: &str, cb: &Expr) -> Option<String> {
        let Expr::Arrow(arrow) = cb else { return None };
        let param = match arrow.params.first() {
            None => None,
            Some(Pat::Ident(id)) => Some(to_snake_case(&id.id.sym)),
            // A destructured parameter has no single name to bind.
            Some(_) => return None,
        };

        let mut inner = self.config.clone();
        inner.locals.extend(self.scope_locals.iter().cloned());
        if let Some(p) = &param {
            inner.locals.push(p.clone());
        }

        let bind = match &param {
            Some(p) => format!("    \u{2254} {} = {}.await;\n", p, receiver),
            None => format!("    {}.await;\n", receiver),
        };

        let body = match &*arrow.body {
            BlockStmtOrExpr::Expr(e) => {
                let src = super::stmt_transform::slice(self.src, e.span());
                let r = super::stmt_transform::transform_statements(&src, &inner, "    ");
                if !r.complete {
                    self.warnings.extend(r.warnings);
                }
                r.code
            }
            BlockStmtOrExpr::BlockStmt(block) => {
                let src = super::stmt_transform::slice(self.src, block.span);
                let r = super::stmt_transform::transform_statements(&src, &inner, "");
                if !r.complete {
                    self.warnings.extend(r.warnings);
                }
                r.code
                    .lines()
                    .map(|l| format!("    {}", l))
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        };
        if body.trim().is_empty() {
            return None;
        }
        Some(format!("{{\n{}{}\n}}", bind, body))
    }

    fn transform_call(&mut self, call: &CallExpr) -> String {
        let args = self.transform_args(&call.args);

        match &call.callee {
            Callee::Expr(callee_expr) => {
                // `a.b?.c(x)` puts the optional member INSIDE the callee, so the
                // member-call arms below never saw it and every such call took
                // the generic path: `terminalRef.current?.sendInput(…)` came out
                // as a plain call to an undefined `send_input`. Unwrapped here,
                // once, so one set of arms handles both spellings.
                let unwrapped: Option<Expr> = match callee_expr.as_ref() {
                    Expr::OptChain(opt) => match &*opt.base {
                        OptChainBase::Member(m) => Some(Expr::Member(m.clone())),
                        _ => None,
                    },
                    _ => None,
                };
                let expr: &Expr = unwrapped.as_ref().unwrap_or(callee_expr);

                // Check for method calls: obj.method()
                if let Expr::Member(member) = expr {
                    let obj = self.transform_expr(&member.obj);

                    if let MemberProp::Ident(ident) = &member.prop {
                        let method = ident.sym.to_string();

                        // A few JS statics have a Sigil host function rather than
                        // a method. `Date.now()` was becoming `date·now()`, which
                        // resolves to the non-existent `date_now` import.
                        if let Some(replacement) = js_static_call(&obj, &method) {
                            return format!("{}({})", replacement, args);
                        }

                        // `re.test(s)` / `re.exec(s)`. `test` and `exec` are
                        // RegExp-only method names, and Sigil has no regex — the
                        // literal is already marked, so the call on it named an
                        // undefined function on an undefined receiver.
                        if matches!(method.as_str(), "test" | "exec") {
                            self.warn(&format!("regex .{}() — Sigil has no regex", method));
                            return format!(
                                "/* {}·{}({}) — Sigil has no regex */ \u{2205}",
                                obj.replace("*/", "* /"),
                                method,
                                args.replace("*/", "* /")
                            );
                        }

                        // `Number.parseInt(s, 10)` — the host `parse_int`
                        // takes the string alone, so the radix made the call an
                        // arity mismatch and it resolved to nothing. Base 10 is
                        // the only one it implements; any other is reported.
                        if (obj == "number" || obj == "string")
                            && matches!(method.as_str(), "parseInt" | "parseFloat")
                        {
                            let first = call
                                .args
                                .first()
                                .map(|a| self.transform_expr(&a.expr))
                                .unwrap_or_else(|| "\u{2205}".to_string());
                            if method == "parseInt" {
                                if let Some(radix) = call.args.get(1) {
                                    let r = self.transform_expr(&radix.expr);
                                    if r.trim() != "10" {
                                        self.warn(&format!(
                                            "parseInt radix {} — only base 10 is implemented",
                                            r.trim()
                                        ));
                                    }
                                }
                                return format!("parse_int({})", first);
                            }
                            return format!("parse_float({})", first);
                        }

                        // `Promise.resolve(x)` is `x` where nothing is
                        // asynchronous; `reject` and `all` have no synchronous
                        // meaning at all. Both reached the backend as calls to
                        // an undefined `resolve` / `reject` / `all`.
                        if obj == "promise" {
                            match method.as_str() {
                                "resolve" => return args,
                                "reject" | "all" | "allSettled" | "race" => {
                                    self.warn(&format!(
                                        "Promise.{} — actors have no promises",
                                        method
                                    ));
                                    return format!(
                                        "/* Promise.{}({}) — Sigil has no promises */ \u{2205}",
                                        method,
                                        args.replace("*/", "* /")
                                    );
                                }
                                _ => {}
                            }
                        }

                        // `onCommitRef.current(text)` — a callback prop parked
                        // in a ref. `current` is the ref's FIELD, not a method,
                        // and Sigil cannot call a value held in a field; the
                        // member call reached the backend as an undefined
                        // `current`. Same gap as a directly-called callback prop
                        // (§8.2.9): actors communicate by message.
                        // A method reached THROUGH a ref — `terminalRef.current
                        // .sendInput(x)` — is the same imperative-handle gap:
                        // the value in the ref is a JavaScript object with
                        // function fields, and Sigil has neither.
                        if obj.ends_with(".current") {
                            self.warn(&format!("imperative ref method: .{}", method));
                            return format!(
                                "/* {}\u{b7}{}({}) — actors have no imperative handles */ \u{2205}",
                                obj.replace("*/", "* /"),
                                method,
                                args.replace("*/", "* /")
                            );
                        }

                        if method == "current" {
                            self.warn(&format!("callback ref called: {}·current", obj));
                            return format!(
                                "/* callback ref `{}.current({})` — actors have no callbacks */ ∅",
                                obj.replace("*/", "* /"),
                                args.replace("*/", "* /")
                            );
                        }

                        // `p.then(cb)` in expression position. Sigil awaits
                        // rather than chaining, so this is `cb(p.await)` — as a
                        // block expression, which binds the parameter without
                        // needing an immediately-invoked closure. Emitting the
                        // call left an undefined `then`; a chain in STATEMENT
                        // position is unrolled in `stmt_transform`.
                        if method == "then" && call.args.len() == 1 {
                            if let Some(block) = self.then_block(&obj, &call.args[0].expr) {
                                return block;
                            }
                        }

                        // A promise `.catch(cb)` / `.finally(cb)` in
                        // expression position — `await res.json().catch(() =>
                        // ({}))`. Sigil has no exceptions, so the handler cannot
                        // run and the receiver IS the value; emitting the call
                        // produced an undefined `catch` in the WASM build. A
                        // chain in STATEMENT position keeps its callback bodies
                        // — see `promise_chain` in stmt_transform.
                        if (method == "catch" || method == "finally") && call.args.len() == 1 {
                            self.warnings.push(format!(
                                "promise .{}() handler dropped — Sigil has no exceptions",
                                method
                            ));
                            return obj;
                        }

                        // `new Date(s)` is already epoch millis (see the `New`
                        // arm), so `.getTime()` on it is the identity.
                        //
                        // `getTime` exists on nothing but a Date, and the only
                        // Dates this migration can produce are `timing_parse(..)`
                        // and `timing_now()` — both already millis — so it is the
                        // identity whatever the receiver's spelling. Matching the
                        // call text alone missed `let d = new Date(iso)` followed
                        // by `d.getTime()`, which reached the backend as an
                        // undefined `get_time`. `valueOf` DOES exist on other
                        // types, so it stays pinned to the two date forms.
                        if method == "getTime"
                            || (method == "valueOf"
                                && (obj.starts_with("timing_parse(")
                                    || obj == "timing_now()"))
                        {
                            return obj;
                        }

                        // Transform common JS methods to Sigil equivalents
                        let sigil_method = match method.as_str() {
                            // `toLocaleString` is a formatted `toString`; Sigil
                            // has no locale layer, so the plain one is honest.
                            "toLocaleString" | "toLocaleDateString"
                            | "toLocaleTimeString" | "toISOString" => "to_string",
                            "charAt" => "char_at",
                            "has" => "contains_key",
                            "hasOwnProperty" => "contains_key",
                            "delete" => "remove",
                            "toString" => "to_string",
                            "trim" => "trim",
                            "toLowerCase" => "to_lowercase",
                            "toUpperCase" => "to_uppercase",
                            "includes" => "contains",
                            "indexOf" => "find",
                            "startsWith" => "starts_with",
                            "endsWith" => "ends_with",
                            // Sigil's Vec method is `push`; there is no `append`.
                            // Nothing exercised this until statement bodies started
                            // emitting the calls that were `{ /* block */ }` before.
                            "push" => "push",
                            "pop" => "pop",
                            "shift" => "remove_first",
                            "join" => "join",
                            "split" => "split",
                            "map" => "map",
                            "filter" => "filter",
                            "find" => "find",
                            "some" => "any",
                            "every" => "all",
                            "reduce" => "fold",
                            "forEach" => "for_each",
                            "slice" => "slice",
                            "concat" => "concat",
                            "reverse" => "reverse",
                            "sort" => "sort",
                            "keys" => "keys",
                            "values" => "values",
                            "entries" => "entries",
                            _ => &method,
                        };

                        return format!("{}·{}({})", obj, to_snake_case_member(sigil_method), args);
                    }
                }

                // `Number(x)` and `String(x)` are conversions, not calls to
                // anything a Sigil program declares — they came out as
                // `number(x)` and `string(x)`, two more lowercase names the
                // backend stubbed to a constant 0.
                if let Expr::Ident(id) = expr {
                    match id.sym.to_string().as_str() {
                        "Number" => return format!("parse_float({})", args),
                        "String" => return format!("({})·to_string()", args),
                        "Boolean" => return format!("({})·to_bool()", args),
                        _ => {}
                    }
                }

                // A prop that holds a function, called.
                //
                // `onPatch({ links: … })` is a child telling its parent
                // something. In an actor that name is a FIELD, and Sigil has no
                // way to call a value — Qliphoth actors communicate by message,
                // not by callback, which is the same gap as the dropped payload
                // in §8.2.9. Emitting `self.on_patch(…)` produced a call that
                // resolved to whatever free function shared the name. The value
                // is ∅ and the call is recorded beside it.
                if self.config.prefix_self {
                    if let Expr::Ident(id) = expr {
                        let name = id.sym.to_string();
                        let snake = to_snake_case(&name);
                        let is_prop = self
                            .config
                            .props
                            .iter()
                            .any(|p| to_snake_case(p) == snake);
                        if is_prop {
                            self.warn(&format!("callback prop called: {}", name));
                            return format!(
                                "/* callback prop `{}({})` — actors have no callbacks */ ∅",
                                name,
                                args.replace("*/", "* /")
                            );
                        }
                    }
                }

                // `setTimeout(fn, ms)` and friends are globals, not fields.
                // Under `prefix_self` they became `self.set_timeout(…)`, a
                // method no actor has. The host `timing` group is the target.
                //
                // JavaScript's trailing arguments are passed on to the callback,
                // which the host import has no room for; a wrapper closure says
                // the same thing with the arity the import declares.
                if let Expr::Ident(id) = expr {
                    if id.sym.as_ref() == "parseInt" {
                        let first = call
                            .args
                            .first()
                            .map(|a| self.transform_expr(&a.expr))
                            .unwrap_or_else(|| "\u{2205}".to_string());
                        if let Some(radix) = call.args.get(1) {
                            let r = self.transform_expr(&radix.expr);
                            if r.trim() != "10" {
                                self.warn(&format!(
                                    "parseInt radix {} — only base 10 is implemented",
                                    r.trim()
                                ));
                            }
                        }
                        return format!("parse_int({})", first);
                    }
                    if let Some(target) = js_timer_global(&id.sym) {
                        let mut parts: Vec<String> = call
                            .args
                            .iter()
                            .map(|a| self.transform_expr(&a.expr))
                            .collect();
                        if target == "timing_set_timeout" || target == "timing_set_interval" {
                            if parts.len() > 2 {
                                let extra = parts.split_off(2).join(", ");
                                parts[0] = format!("|| {}({})", parts[0], extra);
                            }
                        }
                        return format!("{}({})", target, parts.join(", "));
                    }
                }

                // A `useState` setter is an assignment, not a call.
                //
                // `setTab("todo")` on a component whose state includes `tab`
                // means `self.tab = "todo"`. The mutation walker rewrites these
                // in handler bodies, but a setter reached from the view — a
                // closure bound to a local, `≔ go_todo = || setTab("todo")` —
                // went through here and came out `self.set_tab("todo")`, a
                // method no actor has.
                if let Expr::Ident(id) = expr {
                    let name = id.sym.to_string();
                    if let Some(field) = self.setter_target(&name) {
                        if call.args.len() == 1 {
                            return format!("{{ self.{} = {}; }}", field, args);
                        }
                    }
                }

                // Regular function call
                let callee = self.transform_expr(expr);
                if let Expr::Ident(id) = expr {
                    let snake = to_snake_case(&id.sym.to_string());
                    if let Some(&declared) = self.config.fn_arity.get(&snake) {
                        if call.args.len() < declared {
                            let mut parts: Vec<String> = if args.trim().is_empty() {
                                Vec::new()
                            } else {
                                vec![args.clone()]
                            };
                            for _ in call.args.len()..declared {
                                parts.push("∅".to_string());
                            }
                            return format!("{}({})", callee, parts.join(", "));
                        }
                    }
                }
                format!("{}({})", callee, args)
            }
            Callee::Super(_) => format!("super({})", args),
            Callee::Import(_) => {
                self.warn("dynamic import");
                format!("/* import({}) */", args)
            }
        }
    }

    /// `setFoo` → the state field `foo`, when the actor declares one.
    ///
    /// Only for actors: in a plain function component the setter is a prop and
    /// there is no field to assign.
    fn setter_target(&self, name: &str) -> Option<String> {
        if !self.config.prefix_self {
            return None;
        }
        let rest = name.strip_prefix("set")?;
        if !rest.chars().next()?.is_uppercase() {
            return None;
        }
        let field = to_snake_case(rest);
        self.config
            .state_fields
            .iter()
            .find(|f| to_snake_case(f) == field)
            .map(|_| field)
    }

    fn transform_args(&mut self, args: &[ExprOrSpread]) -> String {
        args.iter()
            .map(|arg| {
                let expr = self.transform_expr(&arg.expr);
                if arg.spread.is_some() {
                    format!("...{}", expr)
                } else {
                    expr
                }
            })
            .collect::<Vec<_>>()
            .join(", ")
    }

    fn transform_arrow(&mut self, arrow: &ArrowExpr) -> String {
        // Extract parameter names
        let params: Vec<String> = arrow.params.iter().filter_map(|p| {
            match p {
                Pat::Ident(ident) => Some(to_snake_case(&ident.sym.to_string())),
                Pat::Rest(rest) => {
                    if let Pat::Ident(ident) = rest.arg.as_ref() {
                        Some(format!("...{}", to_snake_case(&ident.sym.to_string())))
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }).collect();

        let params_str = params.join(", ");

        // The parameters are locals inside the body. Without registering them, the
        // actor's unknown-identifier rule rewrote every closure parameter to a state
        // field: `|a| a.fault` became `|a| self.a.fault`, referring to a field named
        // `a` that no actor has.
        let saved_locals = self.scope_locals.len();
        for p in &params {
            self.scope_locals.push(p.trim_start_matches("...").to_string());
        }

        // Transform body
        let body = match &*arrow.body {
            BlockStmtOrExpr::Expr(expr) => self.transform_expr(expr),
            BlockStmtOrExpr::BlockStmt(block) => {
                // An arrow with a block body is statements, and there is a
                // statement transform now. This used to emit `{ /* block */ }`
                // — 441 of them across the Lares client, every one a callback
                // that did nothing.
                let mut inner = self.config.clone();
                inner.locals.extend(self.scope_locals.iter().cloned());
                let src = super::stmt_transform::slice(self.src, block.span);
                let r = super::stmt_transform::transform_statements(&src, &inner, "");
                if !r.complete {
                    for w in r.warnings {
                        self.warn(&w);
                    }
                }
                if r.code.trim().is_empty() {
                    "{ }".to_string()
                } else {
                    // Multi-line, not joined: a `// React:` comment runs to the
                    // end of its line, so flattening a body onto one line makes
                    // the comment swallow every statement after it. The caller
                    // re-indents the whole expression, so indent relative to it.
                    let indented = r
                        .code
                        .lines()
                        .map(|l| format!("    {}", l))
                        .collect::<Vec<_>>()
                        .join("\n");
                    format!("{{\n{}\n}}", indented)
                }
            }
        };

        self.scope_locals.truncate(saved_locals);

        format!("|{}| {}", params_str, body)
    }

    fn transform_template(&mut self, tpl: &Tpl) -> String {
        // Template literals: `hello ${name} world`
        // We convert to a concatenation or format string

        let mut parts = Vec::new();

        for (i, quasi) in tpl.quasis.iter().enumerate() {
            // Add the static part
            // quasi.cooked is Option<Atom>, quasi.raw is Atom
            let text = quasi.cooked.as_ref()
                .and_then(|s| s.as_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| quasi.raw.to_string());

            if !text.is_empty() {
                parts.push(format!("\"{}\"", text.replace('"', "\\\"")));
            }

            // Add the expression if there's one after this quasi
            if i < tpl.exprs.len() {
                let expr = self.transform_expr(&tpl.exprs[i]);
                // `·`, not `.`. A dot is field access, so `x.to_string()` parses
                // as a field named `to_string` and then a call to nothing —
                // which the backend stubs to a constant 0. The last Rustism in
                // the emitted output, and it came from template literals.
                parts.push(format!("({})·to_string()", expr));
            }
        }

        if parts.is_empty() {
            "\"\"".to_string()
        } else if parts.len() == 1 {
            parts[0].clone()
        } else {
            // Join with concatenation
            parts.join(" + ")
        }
    }

    fn transform_array(&mut self, arr: &ArrayLit) -> String {
        // `...` is a valid Sigil token — a rest pattern — so a spread emitted
        // verbatim compiles and means something else. `[...xs]` is a copy;
        // anything else with a spread in it has no Sigil spelling.
        let spreads = arr
            .elems
            .iter()
            .filter(|e| matches!(e, Some(ExprOrSpread { spread: Some(_), .. })))
            .count();
        if spreads == 1 && arr.elems.len() == 1 {
            if let Some(Some(ExprOrSpread { expr, .. })) = arr.elems.first() {
                return format!("{}·clone()", self.transform_expr(expr));
            }
        }
        if spreads > 0 {
            self.warn("array spread has no Sigil form");
            return "∅".to_string();
        }

        let elements: Vec<String> = arr.elems.iter().map(|elem| {
            match elem {
                Some(ExprOrSpread { spread: Some(_), expr }) => {
                    format!("...{}", self.transform_expr(expr))
                }
                Some(ExprOrSpread { spread: None, expr }) => {
                    self.transform_expr(expr)
                }
                None => "None".to_string(),
            }
        }).collect();

        format!("[{}]", elements.join(", "))
    }

    fn transform_object(&mut self, obj: &ObjectLit) -> String {
        // Object literals are tricky in Sigil
        // For now, transform to struct-like syntax or ∅
        if obj.props.is_empty() {
            return "∅".to_string();
        }

        let props: Vec<String> = obj.props.iter().filter_map(|prop| {
            match prop {
                PropOrSpread::Prop(prop) => {
                    match prop.as_ref() {
                        Prop::KeyValue(kv) => {
                            let key = match &kv.key {
                                PropName::Ident(id) => id.sym.to_string(),
                                PropName::Str(s) => s.value.as_str().unwrap_or("").to_string(),
                                PropName::Num(n) => n.value.to_string(),
                                PropName::Computed(c) => self.transform_expr(&c.expr),
                                PropName::BigInt(bi) => bi.value.to_string(),
                            };
                            let value = self.transform_expr(&kv.value);
                            Some(format!("{}: {}", to_snake_case(&key), value))
                        }
                        Prop::Shorthand(id) => {
                            // `{ aliveNames }` is `{ aliveNames: aliveNames }` —
                            // the value half is an ordinary identifier reference
                            // and needs the same treatment as one. Copying the
                            // key into the value position skipped it, so a
                            // shorthand over state or a prop emitted a bare
                            // `alive_names` the actor has no binding for.
                            let key = to_snake_case(&id.sym.to_string());
                            let value = self.transform_ident(id);
                            Some(format!("{}: {}", key, value))
                        }
                        Prop::Method(m) => {
                            let key = match &m.key {
                                PropName::Ident(id) => id.sym.to_string(),
                                _ => return None,
                            };
                            self.warn(&format!("Method property: {}", key));
                            Some(format!("{}: /* method */", to_snake_case(&key)))
                        }
                        _ => None,
                    }
                }
                PropOrSpread::Spread(spread) => {
                    let expr = self.transform_expr(&spread.expr);
                    Some(format!("...{}", expr))
                }
            }
        }).collect();

        // A Sigil struct literal's keys are identifiers. A JS key that is not
        // one — `{ 'content-type': … }` — emitted unquoted parses as
        // subtraction, and quoting it is not accepted either. Renaming the key
        // would change what the object means, so say so instead.
        let bad_key = props.iter().any(|p| {
            let key = p.split(':').next().unwrap_or("").trim();
            key.is_empty()
                || !key.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
                || !key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        });
        if bad_key {
            self.warn("object key is not a Sigil identifier");
            return "∅".to_string();
        }

        format!("{{ {} }}", props.join(", "))
    }

    fn transform_assign(&mut self, assign: &AssignExpr) -> String {
        let right = self.transform_expr(&assign.right);

        let left = match &assign.left {
            AssignTarget::Simple(simple) => {
                match simple {
                    SimpleAssignTarget::Ident(ident) => {
                        self.transform_ident(&ident.id)
                    }
                    SimpleAssignTarget::Member(member) => {
                        self.transform_member(member)
                    }
                    _ => "/* target */".to_string(),
                }
            }
            AssignTarget::Pat(_) => {
                self.warn("Pattern assignment");
                "/* pattern */".to_string()
            }
        };

        // `x ??= v`, `x ||= v`, `x &&= v` — Sigil has none of the three, and
        // they were emitted verbatim (`port ??= …`), which the type checker
        // rejects as an invalid assignment target. Written out as the
        // conditional assignment each one abbreviates.
        match assign.op {
            AssignOp::NullishAssign => {
                return format!(
                    "\u{2387} \u{ac}{}\u{b7}to_bool() {{ {} = {}; }}",
                    left, left, right
                );
            }
            AssignOp::OrAssign => {
                return format!(
                    "\u{2387} \u{ac}{}\u{b7}to_bool() {{ {} = {}; }}",
                    left, left, right
                );
            }
            AssignOp::AndAssign => {
                return format!(
                    "\u{2387} {}\u{b7}to_bool() {{ {} = {}; }}",
                    left, left, right
                );
            }
            _ => {}
        }

        let op = match assign.op {
            AssignOp::Assign => "=",
            AssignOp::AddAssign => "+=",
            AssignOp::SubAssign => "-=",
            AssignOp::MulAssign => "*=",
            AssignOp::DivAssign => "/=",
            AssignOp::ModAssign => "%=",
            AssignOp::BitAndAssign => "&=",
            AssignOp::BitOrAssign => "|=",
            AssignOp::BitXorAssign => "^=",
            AssignOp::LShiftAssign => "<<=",
            AssignOp::RShiftAssign => ">>=",
            AssignOp::ZeroFillRShiftAssign => ">>>=",
            AssignOp::ExpAssign => "**=",
            AssignOp::AndAssign => "∧=",
            AssignOp::OrAssign => "∨=",
            AssignOp::NullishAssign => "??=",
        };

        format!("{} {} {}", left, op, right)
    }

    /// `a?.b` — plain access, because Sigil has no null-propagating one.
    ///
    /// This used to emit `?.` and `?[…]` verbatim, which is not Sigil syntax at
    /// all: `cwd?.replace?(…)` reached the backend and produced a module with
    /// an out-of-bounds table index. `a.b` is what the migration can actually
    /// say. It is not the same thing — a missing `a` propagates in JavaScript
    /// and does not here — so every one is reported.
    fn transform_opt_chain(&mut self, opt: &OptChainExpr) -> String {
        self.warn("optional chaining — Sigil has no null-propagating access");
        match &*opt.base {
            OptChainBase::Member(member) => {
                let obj = self.transform_expr(&member.obj);
                match &member.prop {
                    MemberProp::Ident(ident) => {
                        format!("{}.{}", obj, to_snake_case_member(&ident.sym.to_string()))
                    }
                    MemberProp::Computed(computed) => {
                        let prop = self.transform_expr(&computed.expr);
                        format!("{}[{}]", obj, prop)
                    }
                    _ => obj,
                }
            }
            OptChainBase::Call(call) => {
                // Rebuilt as a plain call and sent through the ordinary path,
                // so a method that has a Sigil spelling still gets it:
                // `res?.json()` has to become the fetch-body read, not a member
                // call on an undefined `json`.
                let plain = CallExpr {
                    span: call.span,
                    ctxt: Default::default(),
                    callee: Callee::Expr(call.callee.clone()),
                    args: call.args.clone(),
                    type_args: call.type_args.clone(),
                };
                let out = self.transform_call(&plain);

                // `a?.b()` means "call it if it is there". Sigil has neither
                // that nor a callable field, so when nothing above recognised
                // the method — the output still spells the JavaScript name —
                // the call cannot be expressed and is marked instead of emitted
                // as a function that does not exist. `handle?.stop()` on a
                // `{ stop }` object is the shape this catches.
                let callee_member: Option<MemberExpr> = match call.callee.as_ref() {
                    Expr::Member(m) => Some(m.clone()),
                    Expr::OptChain(o) => match &*o.base {
                        OptChainBase::Member(m) => Some(m.clone()),
                        _ => None,
                    },
                    _ => None,
                };
                if let Some(member) = callee_member {
                    if let MemberProp::Ident(ident) = &member.prop {
                        let snake = to_snake_case(&ident.sym.to_string());
                        let untouched = out.contains(&format!("\u{b7}{}(", snake));
                        if untouched && !KNOWN_OPTIONAL_METHODS.contains(&snake.as_str()) {
                            self.warn(&format!(
                                "optional call .{}() — Sigil has no callable fields",
                                ident.sym
                            ));
                            return format!(
                                "/* {} — Sigil has no callable fields */ \u{2205}",
                                out.replace("*/", "* /")
                            );
                        }
                    }
                }
                out
            }
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn transform(code: &str) -> String {
        transform_expression(code, &TransformConfig::default()).code
    }

    fn transform_actor(code: &str, state_fields: Vec<&str>) -> String {
        let config = TransformConfig {
            prefix_self: true,
            state_fields: state_fields.into_iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        };
        transform_expression(code, &config).code
    }

    #[test]
    fn test_literals() {
        assert_eq!(transform("42"), "42");
        assert_eq!(transform("3.14"), "3.14");
        assert_eq!(transform("\"hello\""), "\"hello\"");
        assert_eq!(transform("'hello'"), "\"hello\"");
        assert_eq!(transform("true"), "true");
        assert_eq!(transform("false"), "false");
        assert_eq!(transform("null"), "None");
    }

    #[test]
    fn test_logical_operators() {
        // Sigil's ∧/∨ take booleans; JavaScript's &&/|| take anything and lean on
        // truthiness. `·to_bool()` is that coercion made explicit, so an operand
        // that is not already a comparison carries it.
        assert_eq!(transform("a && b"), "a·to_bool() ∧ b·to_bool()");
        // `||` is not ∨: JavaScript's yields the left operand when it is truthy,
        // not `true`, and it is overwhelmingly used as a default-value operator.
        assert_eq!(transform("a || b"), "⎇ a·to_bool() { a } ⎉ { b }");
        assert_eq!(transform("!a"), "¬a·to_bool()");
    }

    #[test]
    fn test_comparison_operators() {
        assert_eq!(transform("a === b"), "a == b");
        assert_eq!(transform("a == b"), "a == b");
        assert_eq!(transform("a !== b"), "a ≠ b");
        assert_eq!(transform("a != b"), "a ≠ b");
        assert_eq!(transform("a < b"), "a < b");
        assert_eq!(transform("a <= b"), "a <= b");
    }

    #[test]
    fn test_ternary() {
        // The test — `a` alone — needs `·to_bool()`; `x > 0` below is already a
        // boolean and does not.
        assert_eq!(
            transform("a ? b : c"),
            "⎇ a·to_bool() { b } ⎉ { c }"
        );
        assert_eq!(
            transform("x > 0 ? \"positive\" : \"non-positive\""),
            "⎇ x > 0 { \"positive\" } ⎉ { \"non-positive\" }"
        );
    }

    #[test]
    fn test_conditional_class_pattern() {
        // React pattern: condition && "string" should become conditional
        assert_eq!(
            transform("isActive && \"active\""),
            "⎇ is_active·to_bool() { \"active\" } ⎉ { \"\" }"
        );
        assert_eq!(
            transform("x > 0 && \"positive\""),
            "⎇ x > 0 { \"positive\" } ⎉ { \"\" }"
        );
        // Pure boolean && stays as ∧ (over coerced operands — see
        // test_logical_operators) rather than becoming a conditional.
        assert_eq!(
            transform("a && b"),
            "a·to_bool() ∧ b·to_bool()"
        );
    }

    #[test]
    fn test_arrow_functions() {
        assert_eq!(transform("() => x"), "|| x");
        assert_eq!(transform("x => x * 2"), "|x| x * 2");
        assert_eq!(transform("(a, b) => a + b"), "|a, b| a + b");
    }

    #[test]
    fn test_method_calls() {
        assert_eq!(transform("arr.length"), "arr.len()");
        assert_eq!(transform("str.toString()"), "str·to_string()");
        assert_eq!(transform("arr.map(x => x * 2)"), "arr·map(|x| x * 2)");
        assert_eq!(transform("arr.filter(x => x > 0)"), "arr·filter(|x| x > 0)");
    }

    #[test]
    fn test_template_literals() {
        assert_eq!(transform("`hello`"), "\"hello\"");
        assert_eq!(
            transform("`hello ${name}`"),
            "\"hello \" + (name)·to_string()"
        );
    }

    #[test]
    fn test_actor_state_prefix() {
        assert_eq!(
            transform_actor("count", vec!["count"]),
            "self.count"
        );
        assert_eq!(
            transform_actor("count + 1", vec!["count"]),
            "self.count + 1"
        );
    }

    #[test]
    fn test_complex_expression() {
        let result = transform("items.filter(item => item.active).map(item => item.name)");
        assert!(result.contains("filter"));
        assert!(result.contains("map"));
    }
}
