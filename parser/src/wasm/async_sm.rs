//! Async State Machine Generation
//!
//! Transforms async functions with multiple await points into state machines.
//! Each await becomes a yield point where the function can suspend and resume.
//!
//! # Strategy
//!
//! An async function like:
//! ```sigil
//! async fn fetch_both(url1: str, url2: str) -> (Data, Data) {
//!     let a = fetch(url1)|await;
//!     let b = fetch(url2)|await;
//!     (a, b)
//! }
//! ```
//!
//! Is transformed into a state machine:
//! ```wasm
//! ;; State 0: Initial - start first fetch
//! ;; State 1: After first await - start second fetch
//! ;; State 2: After second await - return result
//! ```
//!
//! The state machine stores:
//! - Current state number
//! - Saved local variables
//! - Intermediate results
//!
//! When an await is encountered:
//! 1. Save current state and locals to memory
//! 2. Call create_continuation with resumption point
//! 3. Return the promise
//!
//! When resumed:
//! 1. Load state and locals from memory
//! 2. Continue execution from saved state

use super::error::WasmResult;
use super::WasmCompiler;
use crate::ast::{Block, Expr, Function, Pattern, Stmt};

/// Information about an await point in an async function
#[derive(Debug, Clone)]
pub struct AwaitPoint {
    /// Index of this await point (0-based)
    pub index: usize,
    /// Offset into the state frame for saved locals
    pub frame_offset: u32,
    /// Variables that need to be saved at this point
    pub saved_locals: Vec<String>,
}

/// State machine representation of an async function
#[derive(Debug)]
pub struct AsyncStateMachine {
    /// Original function name
    pub name: String,
    /// All await points in order
    pub await_points: Vec<AwaitPoint>,
    /// Size of the state frame in bytes
    pub frame_size: u32,
    /// Number of local variables to save
    pub num_saved_locals: usize,
}

impl WasmCompiler {
    /// Analyze an async function to find await points and build state machine info
    pub fn analyze_async_function(&self, func: &Function) -> Option<AsyncStateMachine> {
        if !func.is_async {
            return None;
        }

        let mut await_points = Vec::new();
        let mut frame_offset = 0u32;

        // Collect all locals that might need saving - extract names from patterns
        let saved_locals: Vec<String> = func
            .params
            .iter()
            .filter_map(|p| {
                if let Pattern::Ident { name, .. } = &p.pattern {
                    Some(name.name.clone())
                } else {
                    None
                }
            })
            .collect();

        // Find all await expressions
        if let Some(ref body) = func.body {
            self.find_await_points(body, &mut await_points, &saved_locals, &mut frame_offset);
        }

        if await_points.is_empty() {
            // No await points - can use simple async compilation
            return None;
        }

        // Each saved local takes 8 bytes (i64)
        let num_saved_locals = saved_locals.len();
        let frame_size = (num_saved_locals as u32 * 8) + 8; // +8 for state number

        Some(AsyncStateMachine {
            name: func.name.name.clone(),
            await_points,
            frame_size,
            num_saved_locals,
        })
    }

    /// Recursively find await expressions in a block
    fn find_await_points(
        &self,
        block: &Block,
        points: &mut Vec<AwaitPoint>,
        saved_locals: &[String],
        frame_offset: &mut u32,
    ) {
        for stmt in &block.stmts {
            self.find_await_in_stmt(stmt, points, saved_locals, frame_offset);
        }
        if let Some(expr) = &block.expr {
            self.find_await_in_expr(expr, points, saved_locals, frame_offset);
        }
    }

    fn find_await_in_stmt(
        &self,
        stmt: &Stmt,
        points: &mut Vec<AwaitPoint>,
        saved_locals: &[String],
        frame_offset: &mut u32,
    ) {
        match stmt {
            Stmt::Let {
                init: Some(expr), ..
            }
            | Stmt::Expr(expr)
            | Stmt::Semi(expr) => {
                self.find_await_in_expr(expr, points, saved_locals, frame_offset);
            }
            Stmt::LetElse {
                init, else_branch, ..
            } => {
                self.find_await_in_expr(init, points, saved_locals, frame_offset);
                self.find_await_in_expr(else_branch, points, saved_locals, frame_offset);
            }
            _ => {}
        }
    }

    fn find_await_in_expr(
        &self,
        expr: &Expr,
        points: &mut Vec<AwaitPoint>,
        saved_locals: &[String],
        frame_offset: &mut u32,
    ) {
        match expr {
            Expr::Await { .. } => {
                points.push(AwaitPoint {
                    index: points.len(),
                    frame_offset: *frame_offset,
                    saved_locals: saved_locals.to_vec(),
                });
                *frame_offset += (saved_locals.len() as u32 * 8) + 8;
            }
            Expr::Binary { left, right, .. } => {
                self.find_await_in_expr(left, points, saved_locals, frame_offset);
                self.find_await_in_expr(right, points, saved_locals, frame_offset);
            }
            Expr::Call { func, args } => {
                self.find_await_in_expr(func, points, saved_locals, frame_offset);
                for arg in args {
                    self.find_await_in_expr(arg, points, saved_locals, frame_offset);
                }
            }
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.find_await_in_expr(condition, points, saved_locals, frame_offset);
                self.find_await_points(then_branch, points, saved_locals, frame_offset);
                if let Some(else_expr) = else_branch {
                    self.find_await_in_expr(else_expr, points, saved_locals, frame_offset);
                }
            }
            Expr::Block(block) => {
                self.find_await_points(block, points, saved_locals, frame_offset);
            }
            Expr::Pipe { expr, operations } => {
                self.find_await_in_expr(expr, points, saved_locals, frame_offset);
                for op in operations {
                    if let crate::ast::PipeOp::Await { .. } = op {
                        points.push(AwaitPoint {
                            index: points.len(),
                            frame_offset: *frame_offset,
                            saved_locals: saved_locals.to_vec(),
                        });
                        *frame_offset += (saved_locals.len() as u32 * 8) + 8;
                    }
                }
            }
            _ => {}
        }
    }

    /// Compile an async function as a state machine
    pub fn compile_async_state_machine(
        &mut self,
        _func: &Function,
        sm: &AsyncStateMachine,
    ) -> WasmResult<()> {
        // For now, fall back to simple await for single await points
        // Full state machine for multiple awaits would require:
        // 1. Allocating a frame in memory
        // 2. Generating a dispatch table at function entry
        // 3. Saving/restoring locals at each await point

        if sm.await_points.len() <= 1 {
            // Simple case - just use regular await
            return Ok(());
        }

        // For multiple awaits, we need the state machine transformation
        // This is a complex transformation that requires:
        // - Splitting the function body at each await
        // - Generating state save/restore code
        // - Creating a dispatcher that jumps to the right state

        // For now, log that we're using the simplified approach
        // The full implementation would require significant refactoring
        // of how we compile function bodies

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::*;
    use crate::span::Span;

    fn make_ident(name: &str) -> Ident {
        Ident {
            name: name.to_string(),
            evidentiality: None,
            affect: None,
            span: Span::new(0, 0),
        }
    }

    fn make_type_path(name: &str) -> TypePath {
        TypePath {
            segments: vec![PathSegment {
                ident: make_ident(name),
                generics: None,
            }],
        }
    }

    fn make_param(name: &str) -> Param {
        Param {
            pattern: Pattern::Ident {
                mutable: false,
                name: make_ident(name),
                evidentiality: None,
            },
            ty: TypeExpr::Path(make_type_path("i64")),
        }
    }

    fn make_path_expr(name: &str) -> Expr {
        Expr::Path(make_type_path(name))
    }

    fn make_await_expr() -> Expr {
        Expr::Await {
            expr: Box::new(Expr::Call {
                func: Box::new(make_path_expr("fetch")),
                args: vec![],
            }),
            evidentiality: None,
        }
    }

    fn make_function(name: &str, params: Vec<Param>, body: Block, is_async: bool) -> Function {
        Function {
            visibility: Visibility::Private,
            is_async,
            attrs: FunctionAttrs::default(),
            name: make_ident(name),
            aspect: None,
            generics: None,
            params,
            return_type: None,
            where_clause: None,
            body: Some(body),
        }
    }

    #[test]
    fn test_analyze_async_no_awaits() {
        let compiler = WasmCompiler::new();
        let func = make_function(
            "simple",
            vec![],
            Block {
                stmts: vec![],
                expr: Some(Box::new(Expr::Literal(Literal::Int {
                    value: "42".to_string(),
                    base: NumBase::Decimal,
                    suffix: None,
                }))),
            },
            true,
        );

        let result = compiler.analyze_async_function(&func);
        assert!(result.is_none()); // No await points
    }

    #[test]
    fn test_analyze_async_single_await() {
        let compiler = WasmCompiler::new();
        let func = make_function(
            "fetch_one",
            vec![make_param("url")],
            Block {
                stmts: vec![],
                expr: Some(Box::new(make_await_expr())),
            },
            true,
        );

        let result = compiler.analyze_async_function(&func);
        assert!(result.is_some());
        let sm = result.unwrap();
        assert_eq!(sm.await_points.len(), 1);
        assert_eq!(sm.num_saved_locals, 1); // url parameter
    }

    #[test]
    fn test_analyze_async_multiple_awaits() {
        let compiler = WasmCompiler::new();
        let func = make_function(
            "fetch_two",
            vec![make_param("url1"), make_param("url2")],
            Block {
                stmts: vec![
                    Stmt::Let {
                        pattern: Pattern::Ident {
                            mutable: false,
                            name: make_ident("a"),
                            evidentiality: None,
                        },
                        ty: None,
                        init: Some(make_await_expr()),
                    },
                    Stmt::Let {
                        pattern: Pattern::Ident {
                            mutable: false,
                            name: make_ident("b"),
                            evidentiality: None,
                        },
                        ty: None,
                        init: Some(make_await_expr()),
                    },
                ],
                expr: Some(Box::new(Expr::Tuple(vec![
                    make_path_expr("a"),
                    make_path_expr("b"),
                ]))),
            },
            true,
        );

        let result = compiler.analyze_async_function(&func);
        assert!(result.is_some());
        let sm = result.unwrap();
        assert_eq!(sm.await_points.len(), 2);
        assert_eq!(sm.num_saved_locals, 2); // url1, url2 parameters
    }

    #[test]
    fn test_state_machine_frame_size() {
        let compiler = WasmCompiler::new();
        let func = make_function(
            "test",
            vec![make_param("a"), make_param("b"), make_param("c")],
            Block {
                stmts: vec![],
                expr: Some(Box::new(make_await_expr())),
            },
            true,
        );

        let result = compiler.analyze_async_function(&func);
        assert!(result.is_some());
        let sm = result.unwrap();
        // 3 locals * 8 bytes + 8 bytes for state = 32 bytes
        assert_eq!(sm.frame_size, 32);
    }
}
