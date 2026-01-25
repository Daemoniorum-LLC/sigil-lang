//! Sigil Interpreter - Executes AST

use crate::parser::{*, MorphOp, BinOp};
use std::collections::HashMap;
use std::cell::RefCell;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Array(Rc<RefCell<Vec<Value>>>),
    Struct { name: String, fields: HashMap<String, Value> },
    Variant { enum_name: String, variant: String, fields: Vec<Value> },
    Function { name: String, params: Vec<(String, String)>, body: Vec<Stmt> },
    BuiltIn(String),
}

impl Value {
    fn to_string(&self) -> String {
        match self {
            Value::Null => "null".to_string(),
            Value::Bool(b) => b.to_string(),
            Value::Int(n) => n.to_string(),
            Value::Float(f) => f.to_string(),
            Value::Str(s) => s.clone(),
            Value::Array(arr) => {
                let items: Vec<String> = arr.borrow().iter().map(|v| v.to_string()).collect();
                format!("[{}]", items.join(", "))
            }
            Value::Struct { name, fields } => {
                let items: Vec<String> = fields.iter()
                    .map(|(k, v)| format!("{}: {}", k, v.to_string()))
                    .collect();
                format!("{} {{ {} }}", name, items.join(", "))
            }
            Value::Variant { enum_name, variant, fields } => {
                if fields.is_empty() {
                    format!("{}::{}", enum_name, variant)
                } else {
                    let items: Vec<String> = fields.iter().map(|v| v.to_string()).collect();
                    format!("{}::{}({})", enum_name, variant, items.join(", "))
                }
            }
            Value::Function { name, .. } => format!("<fn {}>", name),
            Value::BuiltIn(name) => format!("<builtin {}>", name),
        }
    }

    fn is_truthy(&self) -> bool {
        match self {
            Value::Null => false,
            Value::Bool(b) => *b,
            Value::Int(n) => *n != 0,
            Value::Float(f) => *f != 0.0,
            Value::Str(s) => !s.is_empty(),
            Value::Array(arr) => !arr.borrow().is_empty(),
            _ => true,
        }
    }
}

pub struct Interpreter {
    globals: HashMap<String, Value>,
    locals: Vec<HashMap<String, Value>>,
    structs: HashMap<String, Vec<(String, String)>>,
    enums: HashMap<String, Vec<(String, Option<Vec<String>>)>>,
    impls: HashMap<String, HashMap<String, Value>>,
    output: String,
}

impl Interpreter {
    pub fn new() -> Self {
        let mut globals = HashMap::new();
        globals.insert("print".to_string(), Value::BuiltIn("print".to_string()));
        globals.insert("println".to_string(), Value::BuiltIn("println".to_string()));
        globals.insert("len".to_string(), Value::BuiltIn("len".to_string()));
        globals.insert("push".to_string(), Value::BuiltIn("push".to_string()));
        globals.insert("abs".to_string(), Value::BuiltIn("abs".to_string()));
        globals.insert("min".to_string(), Value::BuiltIn("min".to_string()));
        globals.insert("max".to_string(), Value::BuiltIn("max".to_string()));

        Self {
            globals,
            locals: vec![HashMap::new()],
            structs: HashMap::new(),
            enums: HashMap::new(),
            impls: HashMap::new(),
            output: String::new(),
        }
    }

    pub fn execute(&mut self, items: &[Item]) -> Result<(String, String), String> {
        // First pass: collect definitions
        for item in items {
            match item {
                Item::Function { name, params, ret_ty: _, body } => {
                    self.globals.insert(name.clone(), Value::Function {
                        name: name.clone(),
                        params: params.clone(),
                        body: body.clone(),
                    });
                }
                Item::Struct { name, fields } => {
                    self.structs.insert(name.clone(), fields.clone());
                }
                Item::Enum { name, variants } => {
                    self.enums.insert(name.clone(), variants.clone());
                    // Register variant constructors
                    for (variant, _) in variants {
                        self.globals.insert(
                            variant.clone(),
                            Value::Variant {
                                enum_name: name.clone(),
                                variant: variant.clone(),
                                fields: vec![],
                            }
                        );
                    }
                }
                Item::Impl { name, methods } => {
                    let mut method_map = self.impls.remove(name).unwrap_or_default();
                    for method in methods {
                        if let Item::Function { name: method_name, params, ret_ty: _, body } = method {
                            method_map.insert(method_name.clone(), Value::Function {
                                name: method_name.clone(),
                                params: params.clone(),
                                body: body.clone(),
                            });
                        }
                    }
                    self.impls.insert(name.clone(), method_map);
                }
            }
        }

        // Second pass: find and execute main
        if let Some(Value::Function { params, body, .. }) = self.globals.get("main").cloned() {
            if !params.is_empty() {
                return Err("main function should take no arguments".to_string());
            }
            let result = self.eval_block(&body)?;
            return Ok((result.to_string(), self.output.clone()));
        }

        // No main, just return last expression
        Ok(("()".to_string(), self.output.clone()))
    }

    fn push_scope(&mut self) {
        self.locals.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.locals.pop();
    }

    fn get_var(&self, name: &str) -> Option<Value> {
        // Check local scopes from innermost to outermost
        for scope in self.locals.iter().rev() {
            if let Some(v) = scope.get(name) {
                return Some(v.clone());
            }
        }
        // Check globals
        self.globals.get(name).cloned()
    }

    fn set_var(&mut self, name: String, value: Value) {
        if let Some(scope) = self.locals.last_mut() {
            scope.insert(name, value);
        }
    }

    fn update_var(&mut self, name: String, value: Value) -> Result<(), String> {
        // Search local scopes from innermost to outermost
        for scope in self.locals.iter_mut().rev() {
            if scope.contains_key(&name) {
                scope.insert(name, value);
                return Ok(());
            }
        }
        // Check globals
        if self.globals.contains_key(&name) {
            self.globals.insert(name, value);
            return Ok(());
        }
        Err(format!("Cannot assign to undefined variable: {}", name))
    }

    fn eval_block(&mut self, stmts: &[Stmt]) -> Result<Value, String> {
        let mut result = Value::Null;
        for stmt in stmts {
            result = self.eval_stmt(stmt)?;
            if matches!(stmt, Stmt::Return(_)) {
                break;
            }
        }
        Ok(result)
    }

    fn eval_stmt(&mut self, stmt: &Stmt) -> Result<Value, String> {
        match stmt {
            Stmt::Let { name, ty: _, value, mutable: _ } => {
                let v = self.eval_expr(value)?;
                self.set_var(name.clone(), v);
                Ok(Value::Null)
            }
            Stmt::Expr(expr) => self.eval_expr(expr),
            Stmt::Return(Some(expr)) => self.eval_expr(expr),
            Stmt::Return(None) => Ok(Value::Null),
            Stmt::While { cond, body } => {
                while self.eval_expr(cond)?.is_truthy() {
                    self.eval_expr(body)?;
                }
                Ok(Value::Null)
            }
        }
    }

    fn eval_expr(&mut self, expr: &Expr) -> Result<Value, String> {
        match expr {
            Expr::Int(n) => Ok(Value::Int(*n)),
            Expr::Float(f) => Ok(Value::Float(*f)),
            Expr::Bool(b) => Ok(Value::Bool(*b)),
            Expr::Str(s) => Ok(Value::Str(s.clone())),
            Expr::Ident(name) => {
                self.get_var(name)
                    .ok_or_else(|| format!("Undefined variable: {}", name))
            }

            Expr::Binary { op, left, right } => {
                // Handle assignment specially
                if matches!(op, BinOp::Assign) {
                    if let Expr::Ident(name) = left.as_ref() {
                        let value = self.eval_expr(right)?;
                        self.update_var(name.clone(), value.clone())?;
                        return Ok(value);
                    }
                }

                let l = self.eval_expr(left)?;
                let r = self.eval_expr(right)?;
                self.eval_binary(*op, l, r)
            }

            Expr::Unary { op, expr } => {
                let v = self.eval_expr(expr)?;
                match op {
                    UnaryOp::Neg => match v {
                        Value::Int(n) => Ok(Value::Int(-n)),
                        Value::Float(f) => Ok(Value::Float(-f)),
                        _ => Err("Cannot negate non-number".to_string()),
                    },
                    UnaryOp::Not => Ok(Value::Bool(!v.is_truthy())),
                }
            }

            Expr::Call { func, args } => {
                // Check if this is a method call (func is FieldAccess)
                if let Expr::FieldAccess { expr: receiver_expr, field: method_name } = func.as_ref() {
                    let receiver = self.eval_expr(receiver_expr)?;
                    let arg_vals: Result<Vec<Value>, String> = args.iter()
                        .map(|a| self.eval_expr(a))
                        .collect();
                    let arg_vals = arg_vals?;

                    // Check for impl methods first
                    if let Value::Struct { name: struct_name, .. } = &receiver {
                        if let Some(method_map) = self.impls.get(struct_name) {
                            if let Some(method) = method_map.get(method_name) {
                                // Call the method with self as first arg
                                let mut full_args = vec![receiver.clone()];
                                full_args.extend(arg_vals);
                                return self.call_function(method.clone(), full_args);
                            }
                        }
                    }

                    // Try builtin methods
                    return self.call_builtin_method(method_name, &receiver, arg_vals);
                }

                let callee = self.eval_expr(func)?;
                let arg_vals: Result<Vec<Value>, String> = args.iter()
                    .map(|a| self.eval_expr(a))
                    .collect();
                let arg_vals = arg_vals?;
                self.call_function(callee, arg_vals)
            }

            Expr::If { cond, then_, else_ } => {
                if self.eval_expr(cond)?.is_truthy() {
                    self.eval_expr(then_)
                } else if let Some(e) = else_ {
                    self.eval_expr(e)
                } else {
                    Ok(Value::Null)
                }
            }

            Expr::Block(stmts) => {
                self.push_scope();
                let result = self.eval_block(stmts);
                self.pop_scope();
                result
            }

            Expr::Match { expr, arms } => {
                let val = self.eval_expr(expr)?;
                for arm in arms {
                    if self.pattern_matches(&arm.pattern, &val) {
                        self.push_scope();
                        self.bind_pattern(&arm.pattern, &val);
                        let result = self.eval_expr(&arm.body);
                        self.pop_scope();
                        return result;
                    }
                }
                Err("No matching arm in match expression".to_string())
            }

            Expr::FieldAccess { expr, field } => {
                let val = self.eval_expr(expr)?;
                match val {
                    Value::Struct { name: _, ref fields } => {
                        fields.get(field)
                            .cloned()
                            .ok_or_else(|| format!("Unknown field: {}", field))
                    }
                    _ => {
                        // Check for builtin methods (non-call form like x.sqrt without parens)
                        self.call_builtin_method(field, &val, vec![])
                    }
                }
            }

            Expr::StructLit { name, fields } => {
                let mut field_vals = HashMap::new();
                for (fname, fexpr) in fields {
                    field_vals.insert(fname.clone(), self.eval_expr(fexpr)?);
                }
                Ok(Value::Struct {
                    name: name.clone(),
                    fields: field_vals,
                })
            }

            Expr::Array(elements) => {
                let vals: Result<Vec<Value>, String> = elements.iter()
                    .map(|e| self.eval_expr(e))
                    .collect();
                Ok(Value::Array(Rc::new(RefCell::new(vals?))))
            }

            Expr::Index { expr, index } => {
                let arr = self.eval_expr(expr)?;
                let idx = self.eval_expr(index)?;
                match (arr, idx) {
                    (Value::Array(arr), Value::Int(i)) => {
                        let borrowed = arr.borrow();
                        let idx = if i < 0 {
                            (borrowed.len() as i64 + i) as usize
                        } else {
                            i as usize
                        };
                        borrowed.get(idx)
                            .cloned()
                            .ok_or_else(|| format!("Index {} out of bounds", i))
                    }
                    (Value::Str(s), Value::Int(i)) => {
                        let idx = if i < 0 {
                            (s.len() as i64 + i) as usize
                        } else {
                            i as usize
                        };
                        s.chars().nth(idx)
                            .map(|c| Value::Str(c.to_string()))
                            .ok_or_else(|| format!("Index {} out of bounds", i))
                    }
                    _ => Err("Cannot index non-array".to_string()),
                }
            }

            Expr::Morpheme { op, expr, closure } => {
                let val = self.eval_expr(expr)?;
                self.eval_morpheme(*op, val, closure.as_deref())
            }
        }
    }

    fn eval_morpheme(&mut self, op: MorphOp, value: Value, closure: Option<&Expr>) -> Result<Value, String> {
        let arr = match value {
            Value::Array(arr) => arr,
            _ => return Err("Morpheme operations require an array".to_string()),
        };

        match op {
            MorphOp::Tau => {
                // τ - transform/map: apply closure to each element
                let closure = closure.ok_or("τ (map) requires a closure")?;
                let mut results = Vec::new();
                for item in arr.borrow().iter() {
                    self.push_scope();
                    self.set_var("_".to_string(), item.clone());
                    let result = self.eval_expr(closure)?;
                    self.pop_scope();
                    results.push(result);
                }
                Ok(Value::Array(Rc::new(RefCell::new(results))))
            }

            MorphOp::Phi => {
                // φ - filter: keep elements where closure returns true
                let closure = closure.ok_or("φ (filter) requires a closure")?;
                let mut results = Vec::new();
                for item in arr.borrow().iter() {
                    self.push_scope();
                    self.set_var("_".to_string(), item.clone());
                    let keep = self.eval_expr(closure)?;
                    self.pop_scope();
                    if keep.is_truthy() {
                        results.push(item.clone());
                    }
                }
                Ok(Value::Array(Rc::new(RefCell::new(results))))
            }

            MorphOp::Sigma => {
                // Σ - sum: add all elements
                let borrowed = arr.borrow();
                let mut sum = 0i64;
                let mut is_float = false;
                let mut float_sum = 0.0f64;

                for item in borrowed.iter() {
                    match item {
                        Value::Int(n) => {
                            if is_float {
                                float_sum += *n as f64;
                            } else {
                                sum += n;
                            }
                        }
                        Value::Float(f) => {
                            if !is_float {
                                is_float = true;
                                float_sum = sum as f64;
                            }
                            float_sum += f;
                        }
                        _ => return Err("Σ (sum) requires numeric elements".to_string()),
                    }
                }

                if is_float {
                    Ok(Value::Float(float_sum))
                } else {
                    Ok(Value::Int(sum))
                }
            }

            MorphOp::Pi => {
                // Π - product: multiply all elements
                let borrowed = arr.borrow();
                let mut product = 1i64;
                let mut is_float = false;
                let mut float_product = 1.0f64;

                for item in borrowed.iter() {
                    match item {
                        Value::Int(n) => {
                            if is_float {
                                float_product *= *n as f64;
                            } else {
                                product *= n;
                            }
                        }
                        Value::Float(f) => {
                            if !is_float {
                                is_float = true;
                                float_product = product as f64;
                            }
                            float_product *= f;
                        }
                        _ => return Err("Π (product) requires numeric elements".to_string()),
                    }
                }

                if is_float {
                    Ok(Value::Float(float_product))
                } else {
                    Ok(Value::Int(product))
                }
            }

            MorphOp::Mu => {
                // μ - mean: average of elements
                let borrowed = arr.borrow();
                if borrowed.is_empty() {
                    return Ok(Value::Float(0.0));
                }

                let mut sum = 0.0f64;
                for item in borrowed.iter() {
                    match item {
                        Value::Int(n) => sum += *n as f64,
                        Value::Float(f) => sum += f,
                        _ => return Err("μ (mean) requires numeric elements".to_string()),
                    }
                }

                Ok(Value::Float(sum / borrowed.len() as f64))
            }

            MorphOp::Alpha => {
                // α - first: first element
                arr.borrow().first().cloned()
                    .ok_or_else(|| "α (first) called on empty array".to_string())
            }

            MorphOp::Omega => {
                // ω - last: last element
                arr.borrow().last().cloned()
                    .ok_or_else(|| "ω (last) called on empty array".to_string())
            }

            MorphOp::Lambda => {
                // λ - length: count of elements
                Ok(Value::Int(arr.borrow().len() as i64))
            }

            MorphOp::Sort => {
                // σ - sort: sort elements (ascending)
                let mut items: Vec<Value> = arr.borrow().clone();
                items.sort_by(|a, b| {
                    match (a, b) {
                        (Value::Int(x), Value::Int(y)) => x.cmp(y),
                        (Value::Float(x), Value::Float(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
                        (Value::Str(x), Value::Str(y)) => x.cmp(y),
                        _ => std::cmp::Ordering::Equal,
                    }
                });
                Ok(Value::Array(Rc::new(RefCell::new(items))))
            }

            MorphOp::Rho => {
                // ρ - reduce: fold with initial value and closure
                // Usage: arr |ρ (init, |acc, _| acc + _)
                let closure = closure.ok_or("ρ (reduce) requires a closure")?;
                let borrowed = arr.borrow();
                if borrowed.is_empty() {
                    return Ok(Value::Null);
                }

                let mut acc = borrowed[0].clone();
                for item in borrowed.iter().skip(1) {
                    self.push_scope();
                    self.set_var("acc".to_string(), acc);
                    self.set_var("_".to_string(), item.clone());
                    acc = self.eval_expr(closure)?;
                    self.pop_scope();
                }
                Ok(acc)
            }
        }
    }

    fn eval_binary(&self, op: BinOp, left: Value, right: Value) -> Result<Value, String> {
        match (op, &left, &right) {
            // Integer arithmetic
            (BinOp::Add, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a + b)),
            (BinOp::Sub, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a - b)),
            (BinOp::Mul, Value::Int(a), Value::Int(b)) => Ok(Value::Int(a * b)),
            (BinOp::Div, Value::Int(a), Value::Int(b)) => {
                if *b == 0 {
                    Err("Division by zero".to_string())
                } else {
                    Ok(Value::Int(a / b))
                }
            }
            (BinOp::Mod, Value::Int(a), Value::Int(b)) => {
                if *b == 0 {
                    Err("Modulo by zero".to_string())
                } else {
                    Ok(Value::Int(a % b))
                }
            }

            // Float arithmetic
            (BinOp::Add, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a + b)),
            (BinOp::Sub, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a - b)),
            (BinOp::Mul, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a * b)),
            (BinOp::Div, Value::Float(a), Value::Float(b)) => Ok(Value::Float(a / b)),

            // Mixed int/float
            (BinOp::Add, Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 + b)),
            (BinOp::Add, Value::Float(a), Value::Int(b)) => Ok(Value::Float(a + *b as f64)),
            (BinOp::Sub, Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 - b)),
            (BinOp::Sub, Value::Float(a), Value::Int(b)) => Ok(Value::Float(a - *b as f64)),
            (BinOp::Mul, Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 * b)),
            (BinOp::Mul, Value::Float(a), Value::Int(b)) => Ok(Value::Float(a * *b as f64)),
            (BinOp::Div, Value::Int(a), Value::Float(b)) => Ok(Value::Float(*a as f64 / b)),
            (BinOp::Div, Value::Float(a), Value::Int(b)) => Ok(Value::Float(a / *b as f64)),

            // String concatenation (with + for legacy compat)
            (BinOp::Add, Value::Str(a), Value::Str(b)) => Ok(Value::Str(format!("{}{}", a, b))),

            // String concatenation with ++ operator
            (BinOp::Concat, Value::Str(a), Value::Str(b)) => Ok(Value::Str(format!("{}{}", a, b))),
            (BinOp::Concat, Value::Str(a), b) => Ok(Value::Str(format!("{}{}", a, b.to_string()))),
            (BinOp::Concat, a, Value::Str(b)) => Ok(Value::Str(format!("{}{}", a.to_string(), b))),
            (BinOp::Concat, a, b) => Ok(Value::Str(format!("{}{}", a.to_string(), b.to_string()))),

            // Comparisons
            (BinOp::Eq, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a == b)),
            (BinOp::NotEq, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a != b)),
            (BinOp::Lt, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a < b)),
            (BinOp::LtEq, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a <= b)),
            (BinOp::Gt, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a > b)),
            (BinOp::GtEq, Value::Int(a), Value::Int(b)) => Ok(Value::Bool(a >= b)),

            (BinOp::Eq, Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a == b)),
            (BinOp::Lt, Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a < b)),
            (BinOp::Gt, Value::Float(a), Value::Float(b)) => Ok(Value::Bool(a > b)),

            (BinOp::Eq, Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a == b)),
            (BinOp::NotEq, Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(a != b)),

            (BinOp::Eq, Value::Str(a), Value::Str(b)) => Ok(Value::Bool(a == b)),
            (BinOp::NotEq, Value::Str(a), Value::Str(b)) => Ok(Value::Bool(a != b)),

            // Logical
            (BinOp::And, _, _) => Ok(Value::Bool(left.is_truthy() && right.is_truthy())),
            (BinOp::Or, _, _) => Ok(Value::Bool(left.is_truthy() || right.is_truthy())),

            _ => Err(format!("Invalid operation {:?} on {:?} and {:?}", op, left, right)),
        }
    }

    fn call_function(&mut self, callee: Value, args: Vec<Value>) -> Result<Value, String> {
        match callee {
            Value::BuiltIn(name) => self.call_builtin(&name, args),
            Value::Function { params, body, .. } => {
                if args.len() != params.len() {
                    return Err(format!(
                        "Expected {} arguments, got {}",
                        params.len(),
                        args.len()
                    ));
                }

                self.push_scope();
                for ((name, _), value) in params.iter().zip(args) {
                    self.set_var(name.clone(), value);
                }
                let result = self.eval_block(&body);
                self.pop_scope();
                result
            }
            _ => Err("Cannot call non-function".to_string()),
        }
    }

    fn call_builtin(&mut self, name: &str, args: Vec<Value>) -> Result<Value, String> {
        match name {
            "print" => {
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        self.output.push(' ');
                    }
                    self.output.push_str(&arg.to_string());
                }
                Ok(Value::Null)
            }
            "println" => {
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        self.output.push(' ');
                    }
                    self.output.push_str(&arg.to_string());
                }
                self.output.push('\n');
                Ok(Value::Null)
            }
            "len" => {
                if args.len() != 1 {
                    return Err("len takes 1 argument".to_string());
                }
                match &args[0] {
                    Value::Str(s) => Ok(Value::Int(s.len() as i64)),
                    Value::Array(arr) => Ok(Value::Int(arr.borrow().len() as i64)),
                    _ => Err("len requires string or array".to_string()),
                }
            }
            "push" => {
                if args.len() != 2 {
                    return Err("push takes 2 arguments".to_string());
                }
                if let Value::Array(arr) = &args[0] {
                    arr.borrow_mut().push(args[1].clone());
                    Ok(Value::Null)
                } else {
                    Err("push requires array as first argument".to_string())
                }
            }
            "abs" => {
                if args.len() != 1 {
                    return Err("abs takes 1 argument".to_string());
                }
                match &args[0] {
                    Value::Int(n) => Ok(Value::Int(n.abs())),
                    Value::Float(f) => Ok(Value::Float(f.abs())),
                    _ => Err("abs requires number".to_string()),
                }
            }
            "min" => {
                if args.len() != 2 {
                    return Err("min takes 2 arguments".to_string());
                }
                match (&args[0], &args[1]) {
                    (Value::Int(a), Value::Int(b)) => Ok(Value::Int(*a.min(b))),
                    (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a.min(*b))),
                    _ => Err("min requires two numbers".to_string()),
                }
            }
            "max" => {
                if args.len() != 2 {
                    return Err("max takes 2 arguments".to_string());
                }
                match (&args[0], &args[1]) {
                    (Value::Int(a), Value::Int(b)) => Ok(Value::Int(*a.max(b))),
                    (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a.max(*b))),
                    _ => Err("max requires two numbers".to_string()),
                }
            }
            _ => Err(format!("Unknown builtin: {}", name)),
        }
    }

    fn call_builtin_method(&self, method: &str, receiver: &Value, _args: Vec<Value>) -> Result<Value, String> {
        match method {
            "to_string" => Ok(Value::Str(receiver.to_string())),
            "sqrt" => {
                match receiver {
                    Value::Float(f) => Ok(Value::Float(f.sqrt())),
                    Value::Int(n) => Ok(Value::Float((*n as f64).sqrt())),
                    _ => Err("sqrt requires a number".to_string()),
                }
            }
            "abs" => {
                match receiver {
                    Value::Float(f) => Ok(Value::Float(f.abs())),
                    Value::Int(n) => Ok(Value::Int(n.abs())),
                    _ => Err("abs requires a number".to_string()),
                }
            }
            "len" => {
                match receiver {
                    Value::Str(s) => Ok(Value::Int(s.len() as i64)),
                    Value::Array(arr) => Ok(Value::Int(arr.borrow().len() as i64)),
                    _ => Err("len requires string or array".to_string()),
                }
            }
            "is_empty" => {
                match receiver {
                    Value::Str(s) => Ok(Value::Bool(s.is_empty())),
                    Value::Array(arr) => Ok(Value::Bool(arr.borrow().is_empty())),
                    _ => Err("is_empty requires string or array".to_string()),
                }
            }
            _ => Err(format!("Unknown method: {}", method)),
        }
    }

    fn pattern_matches(&self, pattern: &Pattern, value: &Value) -> bool {
        match (pattern, value) {
            (Pattern::Wildcard, _) => true,
            (Pattern::Int(p), Value::Int(v)) => p == v,
            (Pattern::Ident(_), _) => true, // Binds to anything
            (Pattern::Variant { name, .. }, Value::Variant { variant, .. }) => name == variant,
            _ => false,
        }
    }

    fn bind_pattern(&mut self, pattern: &Pattern, value: &Value) {
        if let Pattern::Ident(name) = pattern {
            self.set_var(name.clone(), value.clone());
        }
    }
}
