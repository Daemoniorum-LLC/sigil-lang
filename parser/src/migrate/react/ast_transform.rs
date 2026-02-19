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
            let mut transformer = ExprTransformer::new(config);
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

/// Convert camelCase to snake_case
fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                result.push('_');
            }
            result.push(c.to_lowercase().next().unwrap());
        } else {
            result.push(c);
        }
    }
    result
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
    warnings: Vec<String>,
}

impl<'a> ExprTransformer<'a> {
    fn new(config: &'a TransformConfig) -> Self {
        Self {
            config,
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
                format!("{}.await", inner)
            }

            // New expression: new Foo()
            Expr::New(new_expr) => {
                let callee = self.transform_expr(&new_expr.callee);
                let args = new_expr.args.as_ref()
                    .map(|args| self.transform_args(args))
                    .unwrap_or_default();
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
                format!("/* regex: /{}/{} */", &r.exp, &r.flags)
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

        // Check if it's a local variable (shouldn't be prefixed)
        if self.config.locals.contains(&name) {
            return to_snake_case(&name);
        }

        // Check if it's a prop (for pure function components)
        if self.config.props.contains(&name) {
            return to_snake_case(&name);
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
                let cond = self.transform_expr(&bin.left);
                let value = self.transform_expr(&bin.right);
                return format!("⎇ {} {{ {} }} ⎉ {{ \"\" }}", cond, value);
            }
        }

        // Special case: `condition || defaultValue` (nullish coalescing pattern)
        // Keep as-is if both sides are same type, otherwise use ⎇/⎉
        if bin.op == BinaryOp::LogicalOr || bin.op == BinaryOp::NullishCoalescing {
            if self.is_non_boolean_value(&bin.right) && self.is_non_boolean_value(&bin.left) {
                // Both are values, use ∨ for "or" / fallback
                let left = self.transform_expr(&bin.left);
                let right = self.transform_expr(&bin.right);
                return format!("{} ∨ {}", left, right);
            }
        }

        let left = self.transform_expr(&bin.left);
        let right = self.transform_expr(&bin.right);

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
            UnaryOp::Bang => format!("¬{}", arg),
            UnaryOp::Minus => format!("-{}", arg),
            UnaryOp::Plus => arg, // Unary + is a no-op
            UnaryOp::Tilde => format!("~{}", arg),
            UnaryOp::TypeOf => format!("typeof({})", arg),
            UnaryOp::Void => "None".to_string(),
            UnaryOp::Delete => {
                self.warn("delete operator");
                format!("/* delete {} */", arg)
            }
        }
    }

    fn transform_cond(&mut self, cond: &CondExpr) -> String {
        let test = self.transform_expr(&cond.test);
        let cons = self.transform_expr(&cond.cons);
        let alt = self.transform_expr(&cond.alt);

        // Sigil uses ⎇ (U+2387) for if and ⎉ (U+2389) for else
        format!("⎇ {} {{ {} }} ⎉ {{ {} }}", test, cons, alt)
    }

    fn transform_member(&mut self, member: &MemberExpr) -> String {
        let obj = self.transform_expr(&member.obj);

        match &member.prop {
            MemberProp::Ident(ident) => {
                let prop = ident.sym.to_string();

                // Transform common JS properties to Sigil equivalents
                match prop.as_str() {
                    "length" => format!("{}.len()", obj),
                    _ => format!("{}.{}", obj, to_snake_case(&prop)),
                }
            }
            MemberProp::Computed(computed) => {
                let prop = self.transform_expr(&computed.expr);
                format!("{}[{}]", obj, prop)
            }
            MemberProp::PrivateName(private) => {
                format!("{}._{}", obj, private.name)
            }
        }
    }

    fn transform_call(&mut self, call: &CallExpr) -> String {
        let args = self.transform_args(&call.args);

        match &call.callee {
            Callee::Expr(expr) => {
                // Check for method calls: obj.method()
                if let Expr::Member(member) = expr.as_ref() {
                    let obj = self.transform_expr(&member.obj);

                    if let MemberProp::Ident(ident) = &member.prop {
                        let method = ident.sym.to_string();

                        // Transform common JS methods to Sigil equivalents
                        let sigil_method = match method.as_str() {
                            "toString" => "to_string",
                            "trim" => "trim",
                            "toLowerCase" => "to_lowercase",
                            "toUpperCase" => "to_uppercase",
                            "includes" => "contains",
                            "indexOf" => "find",
                            "startsWith" => "starts_with",
                            "endsWith" => "ends_with",
                            "push" => "append",
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

                        return format!("{}·{}({})", obj, to_snake_case(sigil_method), args);
                    }
                }

                // Regular function call
                let callee = self.transform_expr(expr);
                format!("{}({})", callee, args)
            }
            Callee::Super(_) => format!("super({})", args),
            Callee::Import(_) => {
                self.warn("dynamic import");
                format!("/* import({}) */", args)
            }
        }
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

        // Transform body
        let body = match &*arrow.body {
            BlockStmtOrExpr::Expr(expr) => self.transform_expr(expr),
            BlockStmtOrExpr::BlockStmt(block) => {
                // For block bodies, we'd need statement transformation
                // For now, just indicate it's a block
                self.warn("Arrow function with block body");
                "{ /* block */ }".to_string()
            }
        };

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
                parts.push(format!("{}.to_string()", expr));
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
                            let name = to_snake_case(&id.sym.to_string());
                            Some(format!("{}: {}", name, name))
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

    fn transform_opt_chain(&mut self, opt: &OptChainExpr) -> String {
        match &*opt.base {
            OptChainBase::Member(member) => {
                let obj = self.transform_expr(&member.obj);
                match &member.prop {
                    MemberProp::Ident(ident) => {
                        format!("{}?.{}", obj, to_snake_case(&ident.sym.to_string()))
                    }
                    MemberProp::Computed(computed) => {
                        let prop = self.transform_expr(&computed.expr);
                        format!("{}?[{}]", obj, prop)
                    }
                    _ => format!("{}?", obj),
                }
            }
            OptChainBase::Call(call) => {
                let args = self.transform_args(&call.args);
                // call.callee is Box<Expr>, not OptChainBase
                if let Expr::Member(member) = call.callee.as_ref() {
                    let obj = self.transform_expr(&member.obj);
                    if let MemberProp::Ident(ident) = &member.prop {
                        format!("{}?.{}({})", obj, to_snake_case(&ident.sym.to_string()), args)
                    } else {
                        let callee = self.transform_expr(&call.callee);
                        format!("{}?({})", callee, args)
                    }
                } else {
                    let callee = self.transform_expr(&call.callee);
                    format!("{}?({})", callee, args)
                }
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
        assert_eq!(transform("a && b"), "a ∧ b");
        assert_eq!(transform("a || b"), "a ∨ b");
        assert_eq!(transform("!a"), "¬a");
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
        assert_eq!(
            transform("a ? b : c"),
            "⎇ a { b } ⎉ { c }"
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
            "⎇ is_active { \"active\" } ⎉ { \"\" }"
        );
        assert_eq!(
            transform("x > 0 && \"positive\""),
            "⎇ x > 0 { \"positive\" } ⎉ { \"\" }"
        );
        // Pure boolean && should stay as ∧
        assert_eq!(
            transform("a && b"),
            "a ∧ b"
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
            "\"hello \" + name.to_string()"
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
