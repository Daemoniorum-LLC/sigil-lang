//! Sigil LLVM Compiler Backend
//!
//! Native code generation using LLVM for maximum performance.
//! This backend targets near-native Rust/C++ performance.

#[cfg(feature = "llvm")]
pub mod llvm {
    use inkwell::builder::Builder;
    use inkwell::context::Context;
    use inkwell::execution_engine::{ExecutionEngine, JitFunction};
    use inkwell::module::Module;
    use inkwell::passes::PassBuilderOptions;
    use inkwell::targets::{
        CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple,
    };
    use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum};
    use inkwell::values::{BasicValue, BasicValueEnum, FunctionValue, IntValue, PointerValue};
    use inkwell::{AddressSpace, IntPredicate, OptimizationLevel};

    use std::collections::HashMap;
    use std::path::Path;

    use crate::ast::{self, BinOp, Expr, Item, Literal, UnaryOp};
    use crate::optimize::{OptLevel, Optimizer};
    use crate::parser::Parser;

    /// Type alias for JIT-compiled main function
    type MainFn = unsafe extern "C" fn() -> i64;

    /// Compilation mode
    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum CompileMode {
        /// JIT execution - main stays as "main"
        Jit,
        /// AOT compilation - main becomes "main_sigil" for linking with C runtime
        Aot,
    }

    /// LLVM-based compiler for Sigil
    pub struct LlvmCompiler<'ctx> {
        context: &'ctx Context,
        module: Module<'ctx>,
        builder: Builder<'ctx>,
        execution_engine: Option<ExecutionEngine<'ctx>>,
        /// Compiled functions
        functions: HashMap<String, FunctionValue<'ctx>>,
        /// Optimization level
        opt_level: OptLevel,
        /// Compilation mode (JIT vs AOT)
        compile_mode: CompileMode,
    }

    // Runtime helper: get current time in milliseconds
    extern "C" fn sigil_now() -> i64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0)
    }

    impl<'ctx> LlvmCompiler<'ctx> {
        /// Create a new LLVM compiler for JIT execution
        pub fn new(context: &'ctx Context, opt_level: OptLevel) -> Result<Self, String> {
            Self::with_mode(context, opt_level, CompileMode::Jit)
        }

        /// Create a new LLVM compiler with specific compile mode
        pub fn with_mode(context: &'ctx Context, opt_level: OptLevel, compile_mode: CompileMode) -> Result<Self, String> {
            let module = context.create_module("sigil_main");
            let builder = context.create_builder();

            Ok(Self {
                context,
                module,
                builder,
                execution_engine: None,
                functions: HashMap::new(),
                opt_level,
                compile_mode,
            })
        }

        /// Compile source code
        pub fn compile(&mut self, source: &str) -> Result<(), String> {
            let mut parser = Parser::new(source);
            let source_file = parser.parse_file().map_err(|e| format!("{:?}", e))?;

            // Run AST optimizations
            let mut optimizer = Optimizer::new(self.opt_level);
            let optimized = optimizer.optimize_file(&source_file);

            // Declare runtime functions
            self.declare_runtime_functions();

            // First pass: declare all functions
            for spanned_item in &optimized.items {
                if let Item::Function(func) = &spanned_item.node {
                    self.declare_function(func)?;
                }
            }

            // Second pass: compile function bodies
            for spanned_item in &optimized.items {
                if let Item::Function(func) = &spanned_item.node {
                    self.compile_function(func)?;
                }
            }

            // Run LLVM optimizations
            self.run_llvm_optimizations()?;

            Ok(())
        }

        /// Declare runtime helper functions
        fn declare_runtime_functions(&self) {
            let i64_type = self.context.i64_type();
            let void_type = self.context.void_type();

            // sigil_now() -> i64
            let now_type = i64_type.fn_type(&[], false);
            self.module.add_function("sigil_now", now_type, None);

            // For AOT mode, also declare print functions as external
            if self.compile_mode == CompileMode::Aot {
                // sigil_print_int(i64) -> void
                let print_int_type = void_type.fn_type(&[i64_type.into()], false);
                self.module.add_function("sigil_print_int", print_int_type, None);
            }
        }

        /// Declare a function (creates the signature)
        fn declare_function(&mut self, func: &ast::Function) -> Result<FunctionValue<'ctx>, String> {
            let name = &func.name.name;

            // In AOT mode, rename "main" to "main_sigil" so it doesn't conflict with C runtime
            let actual_name = if self.compile_mode == CompileMode::Aot && name == "main" {
                "main_sigil".to_string()
            } else {
                name.clone()
            };

            let i64_type = self.context.i64_type();

            // Build parameter types (all i64 for simplicity)
            let param_types: Vec<BasicMetadataTypeEnum> = func
                .params
                .iter()
                .map(|_| i64_type.into())
                .collect();

            // Create function type
            let fn_type = i64_type.fn_type(&param_types, false);

            // Declare the function
            let fn_value = self.module.add_function(&actual_name, fn_type, None);

            // Name parameters
            for (i, param) in func.params.iter().enumerate() {
                if let ast::Pattern::Ident { name: ref ident, .. } = param.pattern {
                    fn_value
                        .get_nth_param(i as u32)
                        .unwrap()
                        .set_name(&ident.name);
                }
            }

            // Store function with original name for lookups
            self.functions.insert(name.clone(), fn_value);
            Ok(fn_value)
        }

        /// Compile a function body
        fn compile_function(&mut self, func: &ast::Function) -> Result<(), String> {
            let name = &func.name.name;
            let fn_value = *self.functions.get(name).ok_or("Function not declared")?;

            // Create entry block
            let entry = self.context.append_basic_block(fn_value, "entry");
            self.builder.position_at_end(entry);

            // Set up variable scope
            let mut scope = CompileScope::new();

            // Add parameters to scope
            for (i, param) in func.params.iter().enumerate() {
                if let ast::Pattern::Ident { name: ref ident, .. } = param.pattern {
                    let param_value = fn_value.get_nth_param(i as u32).unwrap();
                    // Allocate on stack for potential mutation
                    let alloca = self.builder.build_alloca(
                        self.context.i64_type(),
                        &ident.name,
                    ).map_err(|e| e.to_string())?;
                    self.builder.build_store(alloca, param_value).map_err(|e| e.to_string())?;
                    scope.vars.insert(ident.name.clone(), alloca);
                }
            }

            // Compile function body
            if let Some(ref body) = func.body {
                let result = self.compile_block(fn_value, &mut scope, body)?;

                // Only add return if block isn't already terminated
                let current_block = self.builder.get_insert_block().unwrap();
                if current_block.get_terminator().is_none() {
                    if let Some(val) = result {
                        self.builder.build_return(Some(&val)).map_err(|e| e.to_string())?;
                    } else {
                        let zero = self.context.i64_type().const_int(0, false);
                        self.builder.build_return(Some(&zero)).map_err(|e| e.to_string())?;
                    }
                }
            } else {
                // No body, return 0
                let zero = self.context.i64_type().const_int(0, false);
                self.builder.build_return(Some(&zero)).map_err(|e| e.to_string())?;
            }

            Ok(())
        }

        /// Compile a block
        fn compile_block(
            &self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            block: &ast::Block,
        ) -> Result<Option<IntValue<'ctx>>, String> {
            let mut result = None;

            for stmt in &block.stmts {
                result = self.compile_stmt(fn_value, scope, stmt)?;
                // Check if we hit a return
                if self.builder.get_insert_block().unwrap().get_terminator().is_some() {
                    return Ok(result);
                }
            }

            // Trailing expression
            if let Some(ref expr) = block.expr {
                result = Some(self.compile_expr(fn_value, scope, expr)?);
            }

            Ok(result)
        }

        /// Compile a statement
        fn compile_stmt(
            &self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            stmt: &ast::Stmt,
        ) -> Result<Option<IntValue<'ctx>>, String> {
            match stmt {
                ast::Stmt::Let { pattern, init, .. } => {
                if let ast::Pattern::Ident { name: ref ident, .. } = pattern {
                        let init_val = if let Some(ref expr) = init {
                            self.compile_expr(fn_value, scope, expr)?
                        } else {
                            self.context.i64_type().const_int(0, false)
                        };

                        // Allocate on stack
                        let alloca = self.builder.build_alloca(
                            self.context.i64_type(),
                            &ident.name,
                        ).map_err(|e| e.to_string())?;
                        self.builder.build_store(alloca, init_val).map_err(|e| e.to_string())?;
                        scope.vars.insert(ident.name.clone(), alloca);
                    }
                    Ok(None)
                }
                ast::Stmt::Expr(expr) => {
                    let val = self.compile_expr(fn_value, scope, expr)?;
                    Ok(Some(val))
                }
                ast::Stmt::Semi(expr) => {
                    self.compile_expr(fn_value, scope, expr)?;
                    Ok(None)
                }
                ast::Stmt::Item(_) => Ok(None),
            }
        }

        /// Compile an expression
        fn compile_expr(
            &self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            expr: &Expr,
        ) -> Result<IntValue<'ctx>, String> {
            match expr {
                Expr::Literal(lit) => self.compile_literal(lit),
                Expr::Path(path) => {
                    // Variable lookup
                    let name = path.segments.last()
                        .map(|s| s.ident.name.as_str())
                        .ok_or("Empty path")?;

                    if let Some(&ptr) = scope.vars.get(name) {
                        let val = self.builder.build_load(
                            self.context.i64_type(),
                            ptr,
                            name,
                        ).map_err(|e| e.to_string())?;
                        Ok(val.into_int_value())
                    } else {
                        Err(format!("Unknown variable: {}", name))
                    }
                }
                Expr::Binary { op, left, right } => {
                    let lhs = self.compile_expr(fn_value, scope, left)?;
                    let rhs = self.compile_expr(fn_value, scope, right)?;
                    self.compile_binary_op(*op, lhs, rhs)
                }
                Expr::Unary { op, expr: inner } => {
                    let val = self.compile_expr(fn_value, scope, inner)?;
                    self.compile_unary_op(*op, val)
                }
                Expr::If { condition, then_branch, else_branch } => {
                    self.compile_if(fn_value, scope, condition, then_branch, else_branch.as_deref())
                }
                Expr::While { condition, body } => {
                    self.compile_while(fn_value, scope, condition, body)
                }
                Expr::Call { func, args } => {
                    self.compile_call(fn_value, scope, func, args)
                }
                Expr::Return(val) => {
                    let ret_val = if let Some(ref e) = val {
                        self.compile_expr(fn_value, scope, e)?
                    } else {
                        self.context.i64_type().const_int(0, false)
                    };
                    self.builder.build_return(Some(&ret_val)).map_err(|e| e.to_string())?;
                    // Return a dummy value (code after return is unreachable)
                    Ok(ret_val)
                }
                Expr::Assign { target, value } => {
                    let val = self.compile_expr(fn_value, scope, value)?;
                    if let Expr::Path(path) = target.as_ref() {
                        let name = path.segments.last()
                            .map(|s| s.ident.name.as_str())
                            .ok_or("Empty path")?;
                        if let Some(&ptr) = scope.vars.get(name) {
                            self.builder.build_store(ptr, val).map_err(|e| e.to_string())?;
                            Ok(val)
                        } else {
                            Err(format!("Unknown variable: {}", name))
                        }
                    } else {
                        Err("Invalid assignment target".to_string())
                    }
                }
                Expr::Block(block) => {
                    let result = self.compile_block(fn_value, scope, block)?;
                    Ok(result.unwrap_or_else(|| self.context.i64_type().const_int(0, false)))
                }
                _ => {
                    // Unsupported expression, return 0
                    Ok(self.context.i64_type().const_int(0, false))
                }
            }
        }

        /// Compile a literal
        fn compile_literal(&self, lit: &Literal) -> Result<IntValue<'ctx>, String> {
            match lit {
                Literal::Int { value, .. } => {
                    let v: i64 = value.parse().map_err(|_| "Invalid integer")?;
                    Ok(self.context.i64_type().const_int(v as u64, false))
                }
                Literal::Bool(b) => {
                    Ok(self.context.i64_type().const_int(if *b { 1 } else { 0 }, false))
                }
                Literal::Float { value, .. } => {
                    // Convert float to int bits for now
                    let v: f64 = value.parse().map_err(|_| "Invalid float")?;
                    Ok(self.context.i64_type().const_int(v.to_bits(), false))
                }
                _ => Ok(self.context.i64_type().const_int(0, false)),
            }
        }

        /// Compile a binary operation
        fn compile_binary_op(
            &self,
            op: BinOp,
            lhs: IntValue<'ctx>,
            rhs: IntValue<'ctx>,
        ) -> Result<IntValue<'ctx>, String> {
            match op {
                BinOp::Add => self.builder.build_int_add(lhs, rhs, "add").map_err(|e| e.to_string()),
                BinOp::Sub => self.builder.build_int_sub(lhs, rhs, "sub").map_err(|e| e.to_string()),
                BinOp::Mul => self.builder.build_int_mul(lhs, rhs, "mul").map_err(|e| e.to_string()),
                BinOp::Div => self.builder.build_int_signed_div(lhs, rhs, "div").map_err(|e| e.to_string()),
                BinOp::Rem => self.builder.build_int_signed_rem(lhs, rhs, "rem").map_err(|e| e.to_string()),
                BinOp::BitAnd => self.builder.build_and(lhs, rhs, "and").map_err(|e| e.to_string()),
                BinOp::BitOr => self.builder.build_or(lhs, rhs, "or").map_err(|e| e.to_string()),
                BinOp::BitXor => self.builder.build_xor(lhs, rhs, "xor").map_err(|e| e.to_string()),
                BinOp::Shl => self.builder.build_left_shift(lhs, rhs, "shl").map_err(|e| e.to_string()),
                BinOp::Shr => self.builder.build_right_shift(lhs, rhs, true, "shr").map_err(|e| e.to_string()),
                BinOp::Eq => {
                    let cmp = self.builder.build_int_compare(IntPredicate::EQ, lhs, rhs, "eq")
                        .map_err(|e| e.to_string())?;
                    self.builder.build_int_z_extend(cmp, self.context.i64_type(), "eq_ext")
                        .map_err(|e| e.to_string())
                }
                BinOp::Ne => {
                    let cmp = self.builder.build_int_compare(IntPredicate::NE, lhs, rhs, "ne")
                        .map_err(|e| e.to_string())?;
                    self.builder.build_int_z_extend(cmp, self.context.i64_type(), "ne_ext")
                        .map_err(|e| e.to_string())
                }
                BinOp::Lt => {
                    let cmp = self.builder.build_int_compare(IntPredicate::SLT, lhs, rhs, "lt")
                        .map_err(|e| e.to_string())?;
                    self.builder.build_int_z_extend(cmp, self.context.i64_type(), "lt_ext")
                        .map_err(|e| e.to_string())
                }
                BinOp::Le => {
                    let cmp = self.builder.build_int_compare(IntPredicate::SLE, lhs, rhs, "le")
                        .map_err(|e| e.to_string())?;
                    self.builder.build_int_z_extend(cmp, self.context.i64_type(), "le_ext")
                        .map_err(|e| e.to_string())
                }
                BinOp::Gt => {
                    let cmp = self.builder.build_int_compare(IntPredicate::SGT, lhs, rhs, "gt")
                        .map_err(|e| e.to_string())?;
                    self.builder.build_int_z_extend(cmp, self.context.i64_type(), "gt_ext")
                        .map_err(|e| e.to_string())
                }
                BinOp::Ge => {
                    let cmp = self.builder.build_int_compare(IntPredicate::SGE, lhs, rhs, "ge")
                        .map_err(|e| e.to_string())?;
                    self.builder.build_int_z_extend(cmp, self.context.i64_type(), "ge_ext")
                        .map_err(|e| e.to_string())
                }
                BinOp::And => {
                    // Logical AND: (lhs != 0) && (rhs != 0)
                    let zero = self.context.i64_type().const_int(0, false);
                    let lhs_bool = self.builder.build_int_compare(IntPredicate::NE, lhs, zero, "lhs_bool")
                        .map_err(|e| e.to_string())?;
                    let rhs_bool = self.builder.build_int_compare(IntPredicate::NE, rhs, zero, "rhs_bool")
                        .map_err(|e| e.to_string())?;
                    let and = self.builder.build_and(lhs_bool, rhs_bool, "and")
                        .map_err(|e| e.to_string())?;
                    self.builder.build_int_z_extend(and, self.context.i64_type(), "and_ext")
                        .map_err(|e| e.to_string())
                }
                BinOp::Or => {
                    // Logical OR: (lhs != 0) || (rhs != 0)
                    let zero = self.context.i64_type().const_int(0, false);
                    let lhs_bool = self.builder.build_int_compare(IntPredicate::NE, lhs, zero, "lhs_bool")
                        .map_err(|e| e.to_string())?;
                    let rhs_bool = self.builder.build_int_compare(IntPredicate::NE, rhs, zero, "rhs_bool")
                        .map_err(|e| e.to_string())?;
                    let or = self.builder.build_or(lhs_bool, rhs_bool, "or")
                        .map_err(|e| e.to_string())?;
                    self.builder.build_int_z_extend(or, self.context.i64_type(), "or_ext")
                        .map_err(|e| e.to_string())
                }
                _ => Ok(self.context.i64_type().const_int(0, false)),
            }
        }

        /// Compile a unary operation
        fn compile_unary_op(
            &self,
            op: UnaryOp,
            val: IntValue<'ctx>,
        ) -> Result<IntValue<'ctx>, String> {
            match op {
                UnaryOp::Neg => self.builder.build_int_neg(val, "neg").map_err(|e| e.to_string()),
                UnaryOp::Not => {
                    // Logical NOT: val == 0 ? 1 : 0
                    let zero = self.context.i64_type().const_int(0, false);
                    let is_zero = self.builder.build_int_compare(IntPredicate::EQ, val, zero, "is_zero")
                        .map_err(|e| e.to_string())?;
                    self.builder.build_int_z_extend(is_zero, self.context.i64_type(), "not")
                        .map_err(|e| e.to_string())
                }
                _ => Ok(val),
            }
        }

        /// Compile an if expression
        fn compile_if(
            &self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            condition: &Expr,
            then_branch: &ast::Block,
            else_branch: Option<&Expr>,
        ) -> Result<IntValue<'ctx>, String> {
            let cond_val = self.compile_expr(fn_value, scope, condition)?;

            // Convert to i1 (bool)
            let zero = self.context.i64_type().const_int(0, false);
            let cond_bool = self.builder.build_int_compare(IntPredicate::NE, cond_val, zero, "cond")
                .map_err(|e| e.to_string())?;

            // Create blocks
            let then_bb = self.context.append_basic_block(fn_value, "then");
            let else_bb = self.context.append_basic_block(fn_value, "else");
            let merge_bb = self.context.append_basic_block(fn_value, "merge");

            self.builder.build_conditional_branch(cond_bool, then_bb, else_bb)
                .map_err(|e| e.to_string())?;

            // Then block
            self.builder.position_at_end(then_bb);
            let then_val = self.compile_block(fn_value, scope, then_branch)?
                .unwrap_or_else(|| self.context.i64_type().const_int(0, false));
            let then_terminated = self.builder.get_insert_block().unwrap().get_terminator().is_some();
            if !then_terminated {
                self.builder.build_unconditional_branch(merge_bb).map_err(|e| e.to_string())?;
            }
            let then_bb_end = self.builder.get_insert_block().unwrap();

            // Else block
            self.builder.position_at_end(else_bb);
            let else_val = if let Some(else_expr) = else_branch {
                self.compile_expr(fn_value, scope, else_expr)?
            } else {
                self.context.i64_type().const_int(0, false)
            };
            let else_terminated = self.builder.get_insert_block().unwrap().get_terminator().is_some();
            if !else_terminated {
                self.builder.build_unconditional_branch(merge_bb).map_err(|e| e.to_string())?;
            }
            let else_bb_end = self.builder.get_insert_block().unwrap();

            // Merge block with phi
            self.builder.position_at_end(merge_bb);

            // If both branches terminated (e.g., both returned), we can't create a phi
            if then_terminated && else_terminated {
                // This block is unreachable, but we need to return something
                return Ok(self.context.i64_type().const_int(0, false));
            }

            let phi = self.builder.build_phi(self.context.i64_type(), "if_result")
                .map_err(|e| e.to_string())?;

            if !then_terminated {
                phi.add_incoming(&[(&then_val, then_bb_end)]);
            }
            if !else_terminated {
                phi.add_incoming(&[(&else_val, else_bb_end)]);
            }

            Ok(phi.as_basic_value().into_int_value())
        }

        /// Compile a while loop
        fn compile_while(
            &self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            condition: &Expr,
            body: &ast::Block,
        ) -> Result<IntValue<'ctx>, String> {
            // Create blocks
            let cond_bb = self.context.append_basic_block(fn_value, "while_cond");
            let body_bb = self.context.append_basic_block(fn_value, "while_body");
            let after_bb = self.context.append_basic_block(fn_value, "while_after");

            // Jump to condition
            self.builder.build_unconditional_branch(cond_bb).map_err(|e| e.to_string())?;

            // Condition block
            self.builder.position_at_end(cond_bb);
            let cond_val = self.compile_expr(fn_value, scope, condition)?;
            let zero = self.context.i64_type().const_int(0, false);
            let cond_bool = self.builder.build_int_compare(IntPredicate::NE, cond_val, zero, "cond")
                .map_err(|e| e.to_string())?;
            self.builder.build_conditional_branch(cond_bool, body_bb, after_bb)
                .map_err(|e| e.to_string())?;

            // Body block
            self.builder.position_at_end(body_bb);
            self.compile_block(fn_value, scope, body)?;
            // Check if body terminated (e.g., return)
            if self.builder.get_insert_block().unwrap().get_terminator().is_none() {
                self.builder.build_unconditional_branch(cond_bb).map_err(|e| e.to_string())?;
            }

            // After block
            self.builder.position_at_end(after_bb);
            Ok(self.context.i64_type().const_int(0, false))
        }

        /// Compile a function call
        fn compile_call(
            &self,
            fn_value: FunctionValue<'ctx>,
            scope: &mut CompileScope<'ctx>,
            func: &Expr,
            args: &[Expr],
        ) -> Result<IntValue<'ctx>, String> {
            // Get function name
            let fn_name = if let Expr::Path(path) = func {
                path.segments.last()
                    .map(|s| s.ident.name.as_str())
                    .ok_or("Empty path")?
            } else {
                return Err("Expected function name".to_string());
            };

            // Handle built-in functions
            match fn_name {
                "print" => {
                    if !args.is_empty() {
                        let arg_val = self.compile_expr(fn_value, scope, &args[0])?;
                        // In AOT mode, call the external print function
                        if self.compile_mode == CompileMode::Aot {
                            let print_fn = self.module.get_function("sigil_print_int")
                                .ok_or("sigil_print_int not declared")?;
                            self.builder.build_call(print_fn, &[arg_val.into()], "")
                                .map_err(|e| e.to_string())?;
                        }
                        return Ok(arg_val);
                    }
                    return Ok(self.context.i64_type().const_int(0, false));
                }
                "now" => {
                    // Call sigil_now runtime function
                    let now_fn = self.module.get_function("sigil_now")
                        .ok_or("sigil_now not declared")?;
                    let call = self.builder.build_call(now_fn, &[], "now")
                        .map_err(|e| e.to_string())?;
                    return Ok(call.try_as_basic_value().left()
                        .map(|v| v.into_int_value())
                        .unwrap_or_else(|| self.context.i64_type().const_int(0, false)));
                }
                _ => {}
            }

            // Get the function
            let callee = if let Some(f) = self.functions.get(fn_name) {
                *f
            } else if let Some(f) = self.module.get_function(fn_name) {
                f
            } else {
                return Err(format!("Unknown function: {}", fn_name));
            };

            // Compile arguments
            let compiled_args: Result<Vec<_>, _> = args
                .iter()
                .map(|arg| {
                    self.compile_expr(fn_value, scope, arg)
                        .map(|v| v.into())
                })
                .collect();
            let compiled_args = compiled_args?;

            // Build call
            let call = self.builder.build_call(callee, &compiled_args, "call")
                .map_err(|e| e.to_string())?;

            // Get return value
            Ok(call
                .try_as_basic_value()
                .left()
                .map(|v| v.into_int_value())
                .unwrap_or_else(|| self.context.i64_type().const_int(0, false)))
        }

        /// Run LLVM optimization passes
        fn run_llvm_optimizations(&self) -> Result<(), String> {
            Target::initialize_native(&InitializationConfig::default())
                .map_err(|e| e.to_string())?;

            let triple = TargetMachine::get_default_triple();
            let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;
            let target_machine = target
                .create_target_machine(
                    &triple,
                    "generic",
                    "",
                    OptimizationLevel::Aggressive,
                    RelocMode::Default,
                    CodeModel::Default,
                )
                .ok_or("Failed to create target machine")?;

            // Run the new pass manager
            let passes = match self.opt_level {
                OptLevel::None => "default<O0>",
                OptLevel::Basic => "default<O1>",
                OptLevel::Standard | OptLevel::Size => "default<O2>",
                OptLevel::Aggressive => "default<O3>",
            };

            self.module
                .run_passes(passes, &target_machine, PassBuilderOptions::create())
                .map_err(|e| e.to_string())?;

            Ok(())
        }

        /// Create JIT execution engine and run
        pub fn run(&mut self) -> Result<i64, String> {
            // Verify module before execution
            if let Err(msg) = self.module.verify() {
                return Err(format!("Module verification failed: {}", msg.to_string()));
            }

            // Create execution engine
            let ee = self.module
                .create_jit_execution_engine(OptimizationLevel::Aggressive)
                .map_err(|e| e.to_string())?;

            // Register runtime functions
            ee.add_global_mapping(
                &self.module.get_function("sigil_now").unwrap(),
                sigil_now as usize,
            );

            self.execution_engine = Some(ee);

            // Get main function
            unsafe {
                let main: JitFunction<MainFn> = self.execution_engine
                    .as_ref()
                    .unwrap()
                    .get_function("main")
                    .map_err(|e| e.to_string())?;

                Ok(main.call())
            }
        }

        /// Write object file
        pub fn write_object_file(&self, path: &Path) -> Result<(), String> {
            Target::initialize_native(&InitializationConfig::default())
                .map_err(|e| e.to_string())?;

            let triple = TargetMachine::get_default_triple();
            let target = Target::from_triple(&triple).map_err(|e| e.to_string())?;
            let target_machine = target
                .create_target_machine(
                    &triple,
                    "generic",
                    "",
                    OptimizationLevel::Aggressive,
                    RelocMode::Default,
                    CodeModel::Default,
                )
                .ok_or("Failed to create target machine")?;

            target_machine
                .write_to_file(&self.module, FileType::Object, path)
                .map_err(|e| e.to_string())
        }

        /// Get LLVM IR as string
        pub fn get_ir(&self) -> String {
            self.module.print_to_string().to_string()
        }
    }

    /// Variable scope for compilation
    struct CompileScope<'ctx> {
        vars: HashMap<String, PointerValue<'ctx>>,
    }

    impl<'ctx> CompileScope<'ctx> {
        fn new() -> Self {
            Self {
                vars: HashMap::new(),
            }
        }
    }
}
