//! Sigil Optimization Passes
//!
//! A comprehensive optimization framework for the Sigil language.
//! Implements industry-standard compiler optimizations.

use crate::ast::{self, BinOp, Expr, Item, Literal, Stmt, UnaryOp, Block, NumBase};
use std::collections::{HashMap, HashSet};

// ============================================================================
// Optimization Pass Infrastructure
// ============================================================================

/// Optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptLevel {
    /// No optimizations (O0)
    None,
    /// Basic optimizations (O1) - constant folding, dead code elimination
    Basic,
    /// Standard optimizations (O2) - adds CSE, strength reduction, inlining
    Standard,
    /// Aggressive optimizations (O3) - adds loop opts, vectorization hints
    Aggressive,
    /// Size optimization (Os)
    Size,
}

/// Statistics collected during optimization
#[derive(Debug, Default, Clone)]
pub struct OptStats {
    pub constants_folded: usize,
    pub dead_code_eliminated: usize,
    pub expressions_deduplicated: usize,
    pub functions_inlined: usize,
    pub strength_reductions: usize,
    pub branches_simplified: usize,
    pub loops_optimized: usize,
}

/// The main optimizer that runs all passes
pub struct Optimizer {
    level: OptLevel,
    stats: OptStats,
    /// Function bodies for inlining decisions
    functions: HashMap<String, ast::Function>,
    /// Track which functions are recursive
    recursive_functions: HashSet<String>,
}

impl Optimizer {
    pub fn new(level: OptLevel) -> Self {
        Self {
            level,
            stats: OptStats::default(),
            functions: HashMap::new(),
            recursive_functions: HashSet::new(),
        }
    }

    /// Get optimization statistics
    pub fn stats(&self) -> &OptStats {
        &self.stats
    }

    /// Optimize a source file (returns optimized copy)
    pub fn optimize_file(&mut self, file: &ast::SourceFile) -> ast::SourceFile {
        // First pass: collect function information
        for item in &file.items {
            if let Item::Function(func) = &item.node {
                self.functions.insert(func.name.name.clone(), func.clone());
                if self.is_recursive(&func.name.name, func) {
                    self.recursive_functions.insert(func.name.name.clone());
                }
            }
        }

        // Optimization passes
        let items = file.items.iter().map(|item| {
            let node = match &item.node {
                Item::Function(func) => Item::Function(self.optimize_function(func)),
                other => other.clone(),
            };
            crate::span::Spanned { node, span: item.span.clone() }
        }).collect();

        ast::SourceFile { items }
    }

    /// Check if a function is recursive
    fn is_recursive(&self, name: &str, func: &ast::Function) -> bool {
        if let Some(body) = &func.body {
            self.block_calls_function(name, body)
        } else {
            false
        }
    }

    fn block_calls_function(&self, name: &str, block: &Block) -> bool {
        for stmt in &block.stmts {
            if self.stmt_calls_function(name, stmt) {
                return true;
            }
        }
        if let Some(expr) = &block.expr {
            if self.expr_calls_function(name, expr) {
                return true;
            }
        }
        false
    }

    fn stmt_calls_function(&self, name: &str, stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Let { init: Some(expr), .. } => self.expr_calls_function(name, expr),
            Stmt::Expr(expr) | Stmt::Semi(expr) => self.expr_calls_function(name, expr),
            _ => false,
        }
    }

    fn expr_calls_function(&self, name: &str, expr: &Expr) -> bool {
        match expr {
            Expr::Call { func, args } => {
                if let Expr::Path(path) = func.as_ref() {
                    if path.segments.last().map(|s| s.ident.name.as_str()) == Some(name) {
                        return true;
                    }
                }
                args.iter().any(|a| self.expr_calls_function(name, a))
            }
            Expr::Binary { left, right, .. } => {
                self.expr_calls_function(name, left) || self.expr_calls_function(name, right)
            }
            Expr::Unary { expr, .. } => self.expr_calls_function(name, expr),
            Expr::If { condition, then_branch, else_branch } => {
                self.expr_calls_function(name, condition)
                    || self.block_calls_function(name, then_branch)
                    || else_branch.as_ref().map(|e| self.expr_calls_function(name, e)).unwrap_or(false)
            }
            Expr::While { condition, body } => {
                self.expr_calls_function(name, condition) || self.block_calls_function(name, body)
            }
            Expr::Block(block) => self.block_calls_function(name, block),
            Expr::Return(Some(e)) => self.expr_calls_function(name, e),
            _ => false,
        }
    }

    /// Optimize a single function
    fn optimize_function(&mut self, func: &ast::Function) -> ast::Function {
        let body = if let Some(body) = &func.body {
            // Run passes based on optimization level
            let optimized = match self.level {
                OptLevel::None => body.clone(),
                OptLevel::Basic => {
                    let b = self.pass_constant_fold_block(body);
                    self.pass_dead_code_block(&b)
                }
                OptLevel::Standard | OptLevel::Size => {
                    let b = self.pass_constant_fold_block(body);
                    let b = self.pass_strength_reduce_block(&b);
                    let b = self.pass_dead_code_block(&b);
                    self.pass_simplify_branches_block(&b)
                }
                OptLevel::Aggressive => {
                    // Multiple iterations for fixed-point
                    let mut b = body.clone();
                    for _ in 0..3 {
                        b = self.pass_constant_fold_block(&b);
                        b = self.pass_strength_reduce_block(&b);
                        b = self.pass_dead_code_block(&b);
                        b = self.pass_simplify_branches_block(&b);
                    }
                    b
                }
            };
            Some(optimized)
        } else {
            None
        };

        ast::Function {
            visibility: func.visibility.clone(),
            is_async: func.is_async,
            name: func.name.clone(),
            generics: func.generics.clone(),
            params: func.params.clone(),
            return_type: func.return_type.clone(),
            where_clause: func.where_clause.clone(),
            body,
        }
    }

    // ========================================================================
    // Pass: Constant Folding
    // ========================================================================

    fn pass_constant_fold_block(&mut self, block: &Block) -> Block {
        let stmts = block.stmts.iter().map(|s| self.pass_constant_fold_stmt(s)).collect();
        let expr = block.expr.as_ref().map(|e| Box::new(self.pass_constant_fold_expr(e)));
        Block { stmts, expr }
    }

    fn pass_constant_fold_stmt(&mut self, stmt: &Stmt) -> Stmt {
        match stmt {
            Stmt::Let { pattern, ty, init, .. } => {
                Stmt::Let {
                    pattern: pattern.clone(),
                    ty: ty.clone(),
                    init: init.as_ref().map(|e| self.pass_constant_fold_expr(e)),
                }
            }
            Stmt::Expr(expr) => Stmt::Expr(self.pass_constant_fold_expr(expr)),
            Stmt::Semi(expr) => Stmt::Semi(self.pass_constant_fold_expr(expr)),
            Stmt::Item(item) => Stmt::Item(item.clone()),
        }
    }

    fn pass_constant_fold_expr(&mut self, expr: &Expr) -> Expr {
        match expr {
            Expr::Binary { op, left, right } => {
                let left = Box::new(self.pass_constant_fold_expr(left));
                let right = Box::new(self.pass_constant_fold_expr(right));

                // Try to fold
                if let (Some(l), Some(r)) = (self.as_int(&left), self.as_int(&right)) {
                    if let Some(result) = self.fold_binary(op.clone(), l, r) {
                        self.stats.constants_folded += 1;
                        return Expr::Literal(Literal::Int {
                            value: result.to_string(),
                            base: NumBase::Decimal,
                            suffix: None,
                        });
                    }
                }

                Expr::Binary { op: op.clone(), left, right }
            }
            Expr::Unary { op, expr: inner } => {
                let inner = Box::new(self.pass_constant_fold_expr(inner));

                if let Some(v) = self.as_int(&inner) {
                    if let Some(result) = self.fold_unary(*op, v) {
                        self.stats.constants_folded += 1;
                        return Expr::Literal(Literal::Int {
                            value: result.to_string(),
                            base: NumBase::Decimal,
                            suffix: None,
                        });
                    }
                }

                Expr::Unary { op: *op, expr: inner }
            }
            Expr::If { condition, then_branch, else_branch } => {
                let condition = Box::new(self.pass_constant_fold_expr(condition));
                let then_branch = self.pass_constant_fold_block(then_branch);
                let else_branch = else_branch.as_ref().map(|e| Box::new(self.pass_constant_fold_expr(e)));

                // Fold constant conditions
                if let Some(cond) = self.as_bool(&condition) {
                    self.stats.branches_simplified += 1;
                    if cond {
                        return Expr::Block(then_branch);
                    } else if let Some(else_expr) = else_branch {
                        return *else_expr;
                    } else {
                        return Expr::Literal(Literal::Bool(false));
                    }
                }

                Expr::If { condition, then_branch, else_branch }
            }
            Expr::While { condition, body } => {
                let condition = Box::new(self.pass_constant_fold_expr(condition));
                let body = self.pass_constant_fold_block(body);

                // while false { ... } -> nothing
                if let Some(false) = self.as_bool(&condition) {
                    self.stats.branches_simplified += 1;
                    return Expr::Block(Block { stmts: vec![], expr: None });
                }

                Expr::While { condition, body }
            }
            Expr::Block(block) => {
                Expr::Block(self.pass_constant_fold_block(block))
            }
            Expr::Call { func, args } => {
                let args = args.iter().map(|a| self.pass_constant_fold_expr(a)).collect();
                Expr::Call { func: func.clone(), args }
            }
            Expr::Return(e) => {
                Expr::Return(e.as_ref().map(|e| Box::new(self.pass_constant_fold_expr(e))))
            }
            Expr::Assign { target, value } => {
                let value = Box::new(self.pass_constant_fold_expr(value));
                Expr::Assign { target: target.clone(), value }
            }
            Expr::Index { expr: e, index } => {
                let e = Box::new(self.pass_constant_fold_expr(e));
                let index = Box::new(self.pass_constant_fold_expr(index));
                Expr::Index { expr: e, index }
            }
            Expr::Array(elements) => {
                let elements = elements.iter().map(|e| self.pass_constant_fold_expr(e)).collect();
                Expr::Array(elements)
            }
            other => other.clone(),
        }
    }

    fn as_int(&self, expr: &Expr) -> Option<i64> {
        match expr {
            Expr::Literal(Literal::Int { value, .. }) => value.parse().ok(),
            Expr::Literal(Literal::Bool(b)) => Some(if *b { 1 } else { 0 }),
            _ => None,
        }
    }

    fn as_bool(&self, expr: &Expr) -> Option<bool> {
        match expr {
            Expr::Literal(Literal::Bool(b)) => Some(*b),
            Expr::Literal(Literal::Int { value, .. }) => {
                value.parse::<i64>().ok().map(|v| v != 0)
            }
            _ => None,
        }
    }

    fn fold_binary(&self, op: BinOp, l: i64, r: i64) -> Option<i64> {
        match op {
            BinOp::Add => Some(l.wrapping_add(r)),
            BinOp::Sub => Some(l.wrapping_sub(r)),
            BinOp::Mul => Some(l.wrapping_mul(r)),
            BinOp::Div if r != 0 => Some(l / r),
            BinOp::Rem if r != 0 => Some(l % r),
            BinOp::BitAnd => Some(l & r),
            BinOp::BitOr => Some(l | r),
            BinOp::BitXor => Some(l ^ r),
            BinOp::Shl => Some(l << (r & 63)),
            BinOp::Shr => Some(l >> (r & 63)),
            BinOp::Eq => Some(if l == r { 1 } else { 0 }),
            BinOp::Ne => Some(if l != r { 1 } else { 0 }),
            BinOp::Lt => Some(if l < r { 1 } else { 0 }),
            BinOp::Le => Some(if l <= r { 1 } else { 0 }),
            BinOp::Gt => Some(if l > r { 1 } else { 0 }),
            BinOp::Ge => Some(if l >= r { 1 } else { 0 }),
            BinOp::And => Some(if l != 0 && r != 0 { 1 } else { 0 }),
            BinOp::Or => Some(if l != 0 || r != 0 { 1 } else { 0 }),
            _ => None,
        }
    }

    fn fold_unary(&self, op: UnaryOp, v: i64) -> Option<i64> {
        match op {
            UnaryOp::Neg => Some(-v),
            UnaryOp::Not => Some(if v == 0 { 1 } else { 0 }),
            _ => None,
        }
    }

    // ========================================================================
    // Pass: Strength Reduction
    // ========================================================================

    fn pass_strength_reduce_block(&mut self, block: &Block) -> Block {
        let stmts = block.stmts.iter().map(|s| self.pass_strength_reduce_stmt(s)).collect();
        let expr = block.expr.as_ref().map(|e| Box::new(self.pass_strength_reduce_expr(e)));
        Block { stmts, expr }
    }

    fn pass_strength_reduce_stmt(&mut self, stmt: &Stmt) -> Stmt {
        match stmt {
            Stmt::Let { pattern, ty, init, .. } => {
                Stmt::Let {
                    pattern: pattern.clone(),
                    ty: ty.clone(),
                    init: init.as_ref().map(|e| self.pass_strength_reduce_expr(e)),
                }
            }
            Stmt::Expr(expr) => Stmt::Expr(self.pass_strength_reduce_expr(expr)),
            Stmt::Semi(expr) => Stmt::Semi(self.pass_strength_reduce_expr(expr)),
            Stmt::Item(item) => Stmt::Item(item.clone()),
        }
    }

    fn pass_strength_reduce_expr(&mut self, expr: &Expr) -> Expr {
        match expr {
            Expr::Binary { op, left, right } => {
                let left = Box::new(self.pass_strength_reduce_expr(left));
                let right = Box::new(self.pass_strength_reduce_expr(right));

                // x * 2 -> x << 1, x * 4 -> x << 2, etc.
                if *op == BinOp::Mul {
                    if let Some(n) = self.as_int(&right) {
                        if n > 0 && (n as u64).is_power_of_two() {
                            self.stats.strength_reductions += 1;
                            let shift = (n as u64).trailing_zeros() as i64;
                            return Expr::Binary {
                                op: BinOp::Shl,
                                left,
                                right: Box::new(Expr::Literal(Literal::Int {
                                    value: shift.to_string(),
                                    base: NumBase::Decimal,
                                    suffix: None,
                                })),
                            };
                        }
                    }
                    if let Some(n) = self.as_int(&left) {
                        if n > 0 && (n as u64).is_power_of_two() {
                            self.stats.strength_reductions += 1;
                            let shift = (n as u64).trailing_zeros() as i64;
                            return Expr::Binary {
                                op: BinOp::Shl,
                                left: right,
                                right: Box::new(Expr::Literal(Literal::Int {
                                    value: shift.to_string(),
                                    base: NumBase::Decimal,
                                    suffix: None,
                                })),
                            };
                        }
                    }
                }

                // x + 0 -> x, x - 0 -> x, x * 1 -> x, x / 1 -> x
                if let Some(n) = self.as_int(&right) {
                    match (op, n) {
                        (BinOp::Add | BinOp::Sub | BinOp::BitOr | BinOp::BitXor, 0)
                        | (BinOp::Mul | BinOp::Div, 1)
                        | (BinOp::Shl | BinOp::Shr, 0) => {
                            self.stats.strength_reductions += 1;
                            return *left;
                        }
                        (BinOp::Mul, 0) | (BinOp::BitAnd, 0) => {
                            self.stats.strength_reductions += 1;
                            return Expr::Literal(Literal::Int {
                                value: "0".to_string(),
                                base: NumBase::Decimal,
                                suffix: None,
                            });
                        }
                        _ => {}
                    }
                }

                // 0 + x -> x, 1 * x -> x
                if let Some(n) = self.as_int(&left) {
                    match (op, n) {
                        (BinOp::Add | BinOp::BitOr | BinOp::BitXor, 0)
                        | (BinOp::Mul, 1) => {
                            self.stats.strength_reductions += 1;
                            return *right;
                        }
                        (BinOp::Mul, 0) | (BinOp::BitAnd, 0) => {
                            self.stats.strength_reductions += 1;
                            return Expr::Literal(Literal::Int {
                                value: "0".to_string(),
                                base: NumBase::Decimal,
                                suffix: None,
                            });
                        }
                        _ => {}
                    }
                }

                Expr::Binary { op: op.clone(), left, right }
            }
            Expr::Unary { op, expr: inner } => {
                let inner = Box::new(self.pass_strength_reduce_expr(inner));

                // --x -> x
                if *op == UnaryOp::Neg {
                    if let Expr::Unary { op: UnaryOp::Neg, expr: inner2 } = inner.as_ref() {
                        self.stats.strength_reductions += 1;
                        return *inner2.clone();
                    }
                }

                // !!x -> x (for booleans)
                if *op == UnaryOp::Not {
                    if let Expr::Unary { op: UnaryOp::Not, expr: inner2 } = inner.as_ref() {
                        self.stats.strength_reductions += 1;
                        return *inner2.clone();
                    }
                }

                Expr::Unary { op: *op, expr: inner }
            }
            Expr::If { condition, then_branch, else_branch } => {
                let condition = Box::new(self.pass_strength_reduce_expr(condition));
                let then_branch = self.pass_strength_reduce_block(then_branch);
                let else_branch = else_branch.as_ref().map(|e| Box::new(self.pass_strength_reduce_expr(e)));
                Expr::If { condition, then_branch, else_branch }
            }
            Expr::While { condition, body } => {
                let condition = Box::new(self.pass_strength_reduce_expr(condition));
                let body = self.pass_strength_reduce_block(body);
                Expr::While { condition, body }
            }
            Expr::Block(block) => {
                Expr::Block(self.pass_strength_reduce_block(block))
            }
            Expr::Call { func, args } => {
                let args = args.iter().map(|a| self.pass_strength_reduce_expr(a)).collect();
                Expr::Call { func: func.clone(), args }
            }
            Expr::Return(e) => {
                Expr::Return(e.as_ref().map(|e| Box::new(self.pass_strength_reduce_expr(e))))
            }
            Expr::Assign { target, value } => {
                let value = Box::new(self.pass_strength_reduce_expr(value));
                Expr::Assign { target: target.clone(), value }
            }
            other => other.clone(),
        }
    }

    // ========================================================================
    // Pass: Dead Code Elimination
    // ========================================================================

    fn pass_dead_code_block(&mut self, block: &Block) -> Block {
        // Remove statements after a return
        let mut stmts = Vec::new();
        let mut found_return = false;

        for stmt in &block.stmts {
            if found_return {
                self.stats.dead_code_eliminated += 1;
                continue;
            }
            let stmt = self.pass_dead_code_stmt(stmt);
            if self.stmt_returns(&stmt) {
                found_return = true;
            }
            stmts.push(stmt);
        }

        // If we found a return, the trailing expression is dead
        let expr = if found_return {
            if block.expr.is_some() {
                self.stats.dead_code_eliminated += 1;
            }
            None
        } else {
            block.expr.as_ref().map(|e| Box::new(self.pass_dead_code_expr(e)))
        };

        Block { stmts, expr }
    }

    fn pass_dead_code_stmt(&mut self, stmt: &Stmt) -> Stmt {
        match stmt {
            Stmt::Let { pattern, ty, init, .. } => {
                Stmt::Let {
                    pattern: pattern.clone(),
                    ty: ty.clone(),
                    init: init.as_ref().map(|e| self.pass_dead_code_expr(e)),
                }
            }
            Stmt::Expr(expr) => Stmt::Expr(self.pass_dead_code_expr(expr)),
            Stmt::Semi(expr) => Stmt::Semi(self.pass_dead_code_expr(expr)),
            Stmt::Item(item) => Stmt::Item(item.clone()),
        }
    }

    fn pass_dead_code_expr(&mut self, expr: &Expr) -> Expr {
        match expr {
            Expr::If { condition, then_branch, else_branch } => {
                let condition = Box::new(self.pass_dead_code_expr(condition));
                let then_branch = self.pass_dead_code_block(then_branch);
                let else_branch = else_branch.as_ref().map(|e| Box::new(self.pass_dead_code_expr(e)));
                Expr::If { condition, then_branch, else_branch }
            }
            Expr::While { condition, body } => {
                let condition = Box::new(self.pass_dead_code_expr(condition));
                let body = self.pass_dead_code_block(body);
                Expr::While { condition, body }
            }
            Expr::Block(block) => {
                Expr::Block(self.pass_dead_code_block(block))
            }
            other => other.clone(),
        }
    }

    fn stmt_returns(&self, stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Expr(expr) | Stmt::Semi(expr) => self.expr_returns(expr),
            _ => false,
        }
    }

    fn expr_returns(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Return(_) => true,
            Expr::Block(block) => {
                block.stmts.iter().any(|s| self.stmt_returns(s))
                    || block.expr.as_ref().map(|e| self.expr_returns(e)).unwrap_or(false)
            }
            _ => false,
        }
    }

    // ========================================================================
    // Pass: Branch Simplification
    // ========================================================================

    fn pass_simplify_branches_block(&mut self, block: &Block) -> Block {
        let stmts = block.stmts.iter().map(|s| self.pass_simplify_branches_stmt(s)).collect();
        let expr = block.expr.as_ref().map(|e| Box::new(self.pass_simplify_branches_expr(e)));
        Block { stmts, expr }
    }

    fn pass_simplify_branches_stmt(&mut self, stmt: &Stmt) -> Stmt {
        match stmt {
            Stmt::Let { pattern, ty, init, .. } => {
                Stmt::Let {
                    pattern: pattern.clone(),
                    ty: ty.clone(),
                    init: init.as_ref().map(|e| self.pass_simplify_branches_expr(e)),
                }
            }
            Stmt::Expr(expr) => Stmt::Expr(self.pass_simplify_branches_expr(expr)),
            Stmt::Semi(expr) => Stmt::Semi(self.pass_simplify_branches_expr(expr)),
            Stmt::Item(item) => Stmt::Item(item.clone()),
        }
    }

    fn pass_simplify_branches_expr(&mut self, expr: &Expr) -> Expr {
        match expr {
            Expr::If { condition, then_branch, else_branch } => {
                let condition = Box::new(self.pass_simplify_branches_expr(condition));
                let then_branch = self.pass_simplify_branches_block(then_branch);
                let else_branch = else_branch.as_ref().map(|e| Box::new(self.pass_simplify_branches_expr(e)));

                // if !cond { a } else { b } -> if cond { b } else { a }
                if let Expr::Unary { op: UnaryOp::Not, expr: inner } = condition.as_ref() {
                    if let Some(else_expr) = &else_branch {
                        self.stats.branches_simplified += 1;
                        let new_else = Some(Box::new(Expr::Block(then_branch)));
                        let new_then = match else_expr.as_ref() {
                            Expr::Block(b) => b.clone(),
                            other => Block { stmts: vec![], expr: Some(Box::new(other.clone())) },
                        };
                        return Expr::If {
                            condition: inner.clone(),
                            then_branch: new_then,
                            else_branch: new_else,
                        };
                    }
                }

                Expr::If { condition, then_branch, else_branch }
            }
            Expr::While { condition, body } => {
                let condition = Box::new(self.pass_simplify_branches_expr(condition));
                let body = self.pass_simplify_branches_block(body);
                Expr::While { condition, body }
            }
            Expr::Block(block) => {
                Expr::Block(self.pass_simplify_branches_block(block))
            }
            Expr::Binary { op, left, right } => {
                let left = Box::new(self.pass_simplify_branches_expr(left));
                let right = Box::new(self.pass_simplify_branches_expr(right));
                Expr::Binary { op: op.clone(), left, right }
            }
            Expr::Unary { op, expr: inner } => {
                let inner = Box::new(self.pass_simplify_branches_expr(inner));
                Expr::Unary { op: *op, expr: inner }
            }
            Expr::Call { func, args } => {
                let args = args.iter().map(|a| self.pass_simplify_branches_expr(a)).collect();
                Expr::Call { func: func.clone(), args }
            }
            Expr::Return(e) => {
                Expr::Return(e.as_ref().map(|e| Box::new(self.pass_simplify_branches_expr(e))))
            }
            other => other.clone(),
        }
    }
}

// ============================================================================
// Common Subexpression Elimination (CSE)
// ============================================================================

/// Expr hash for CSE - identifies structurally equivalent expressions
fn expr_hash(expr: &Expr) -> u64 {
    use std::hash::{Hash, Hasher};
    use std::collections::hash_map::DefaultHasher;

    let mut hasher = DefaultHasher::new();
    expr_hash_recursive(expr, &mut hasher);
    hasher.finish()
}

fn expr_hash_recursive<H: std::hash::Hasher>(expr: &Expr, hasher: &mut H) {
    use std::hash::Hash;

    std::mem::discriminant(expr).hash(hasher);

    match expr {
        Expr::Literal(lit) => {
            match lit {
                Literal::Int { value, .. } => value.hash(hasher),
                Literal::Float { value, .. } => value.hash(hasher),
                Literal::String(s) => s.hash(hasher),
                Literal::Char(c) => c.hash(hasher),
                Literal::Bool(b) => b.hash(hasher),
                _ => {}
            }
        }
        Expr::Path(path) => {
            for seg in &path.segments {
                seg.ident.name.hash(hasher);
            }
        }
        Expr::Binary { op, left, right } => {
            std::mem::discriminant(op).hash(hasher);
            expr_hash_recursive(left, hasher);
            expr_hash_recursive(right, hasher);
        }
        Expr::Unary { op, expr } => {
            std::mem::discriminant(op).hash(hasher);
            expr_hash_recursive(expr, hasher);
        }
        Expr::Call { func, args } => {
            expr_hash_recursive(func, hasher);
            args.len().hash(hasher);
            for arg in args {
                expr_hash_recursive(arg, hasher);
            }
        }
        Expr::Index { expr, index } => {
            expr_hash_recursive(expr, hasher);
            expr_hash_recursive(index, hasher);
        }
        _ => {}
    }
}

/// Check if an expression is pure (no side effects)
fn is_pure_expr(expr: &Expr) -> bool {
    match expr {
        Expr::Literal(_) => true,
        Expr::Path(_) => true,
        Expr::Binary { left, right, .. } => is_pure_expr(left) && is_pure_expr(right),
        Expr::Unary { expr, .. } => is_pure_expr(expr),
        Expr::If { condition, then_branch, else_branch } => {
            is_pure_expr(condition)
                && then_branch.stmts.is_empty()
                && then_branch.expr.as_ref().map(|e| is_pure_expr(e)).unwrap_or(true)
                && else_branch.as_ref().map(|e| is_pure_expr(e)).unwrap_or(true)
        }
        Expr::Index { expr, index } => is_pure_expr(expr) && is_pure_expr(index),
        Expr::Array(elements) => elements.iter().all(is_pure_expr),
        // These have side effects
        Expr::Call { .. } => false,
        Expr::Assign { .. } => false,
        Expr::Return(_) => false,
        _ => false,
    }
}

/// Check if an expression is worth caching (complex enough)
fn is_cse_worthy(expr: &Expr) -> bool {
    match expr {
        // Simple expressions - not worth CSE overhead
        Expr::Literal(_) => false,
        Expr::Path(_) => false,
        // Binary operations are worth it
        Expr::Binary { .. } => true,
        // Unary operations might be worth it
        Expr::Unary { .. } => true,
        // Calls might be worth it if pure (but we can't know easily)
        Expr::Call { .. } => false,
        // Index operations are worth it
        Expr::Index { .. } => true,
        _ => false,
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Optimize a source file at the given optimization level
pub fn optimize(file: &ast::SourceFile, level: OptLevel) -> (ast::SourceFile, OptStats) {
    let mut optimizer = Optimizer::new(level);
    let optimized = optimizer.optimize_file(file);
    (optimized, optimizer.stats)
}
