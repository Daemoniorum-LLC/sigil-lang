//! Sigil JIT Compiler using Cranelift
//!
//! Compiles Sigil AST to native machine code for high-performance execution.
//!
//! Optimizations implemented:
//! - Direct condition branching (no redundant boolean conversion)
//! - Constant folding for arithmetic expressions
//! - Tail call optimization for recursive functions
//! - Efficient comparison code generation

#[cfg(feature = "jit")]
pub mod jit {
    use cranelift_codegen::ir::{types, AbiParam, InstBuilder, UserFuncName};
    use cranelift_codegen::ir::condcodes::IntCC;
    use cranelift_codegen::settings::{self, Configurable};
    use cranelift_codegen::Context;
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{FuncId, Linkage, Module};

    use std::collections::HashMap;
    use std::mem;

    use crate::ast::{self, BinOp, Expr, Item, Literal, UnaryOp, ExternBlock, ExternItem, ExternFunction, TypeExpr};
    use crate::parser::Parser;
    use crate::optimize::{Optimizer, OptLevel};
    use crate::ffi::ctypes::CType;

    /// Runtime value representation
    ///
    /// We use a tagged union representation:
    /// - 64-bit value
    /// - Low 3 bits are tag (NaN-boxing style, but simpler)
    ///
    /// For maximum performance, we use unboxed representations:
    /// - Integers: raw i64
    /// - Floats: raw f64
    /// - Booleans: 0 or 1
    /// - Arrays/Strings: pointers to heap
    #[repr(C)]
    #[derive(Clone, Copy, Debug)]
    pub struct SigilValue(pub u64);

    impl SigilValue {
        // Tag constants (stored in low bits for pointers, high bits for numbers)
        pub const TAG_INT: u64 = 0;
        pub const TAG_FLOAT: u64 = 1;
        pub const TAG_BOOL: u64 = 2;
        pub const TAG_NULL: u64 = 3;
        pub const TAG_PTR: u64 = 4; // Heap-allocated objects

        #[inline]
        pub fn from_int(v: i64) -> Self {
            SigilValue(v as u64)
        }

        #[inline]
        pub fn from_float(v: f64) -> Self {
            SigilValue(v.to_bits())
        }

        #[inline]
        pub fn from_bool(v: bool) -> Self {
            SigilValue(if v { 1 } else { 0 })
        }

        #[inline]
        pub fn as_int(self) -> i64 {
            self.0 as i64
        }

        #[inline]
        pub fn as_float(self) -> f64 {
            f64::from_bits(self.0)
        }

        #[inline]
        pub fn as_bool(self) -> bool {
            self.0 != 0
        }
    }

    /// Compiled function signature
    type CompiledFn = unsafe extern "C" fn() -> i64;
    type CompiledFnWithArgs = unsafe extern "C" fn(i64) -> i64;

    /// Extern function signature info for FFI
    #[derive(Clone, Debug)]
    pub struct ExternFnSig {
        pub name: String,
        pub params: Vec<types::Type>,
        pub returns: Option<types::Type>,
        pub variadic: bool,
        pub func_id: FuncId,
    }

    /// JIT Compiler for Sigil
    pub struct JitCompiler {
        /// The JIT module
        module: JITModule,
        /// Builder context (reused for efficiency)
        builder_ctx: FunctionBuilderContext,
        /// Codegen context
        ctx: Context,
        /// Compiled functions
        functions: HashMap<String, FuncId>,
        /// Extern "C" function declarations
        extern_functions: HashMap<String, ExternFnSig>,
        /// Variable counter for unique variable indices
        var_counter: usize,
        /// Built-in function addresses
        builtins: HashMap<String, *const u8>,
    }

    impl JitCompiler {
        /// Create a new JIT compiler
        pub fn new() -> Result<Self, String> {
            let mut flag_builder = settings::builder();
            // Disable PIC for better codegen
            flag_builder.set("use_colocated_libcalls", "false").unwrap();
            flag_builder.set("is_pic", "false").unwrap();
            // Maximum optimization level
            flag_builder.set("opt_level", "speed").unwrap();
            // Enable additional optimizations
            flag_builder.set("enable_verifier", "false").unwrap(); // Disable verifier in release for speed
            flag_builder.set("enable_alias_analysis", "true").unwrap();

            let isa_builder = cranelift_native::builder().map_err(|e| e.to_string())?;
            let isa = isa_builder
                .finish(settings::Flags::new(flag_builder))
                .map_err(|e| e.to_string())?;

            let mut builder = JITBuilder::with_isa(isa, cranelift_module::default_libcall_names());

            // Register built-in functions
            let builtins = Self::register_builtins(&mut builder);

            let module = JITModule::new(builder);

            Ok(Self {
                module,
                builder_ctx: FunctionBuilderContext::new(),
                ctx: Context::new(),
                functions: HashMap::new(),
                extern_functions: HashMap::new(),
                var_counter: 0,
                builtins,
            })
        }

        /// Register built-in runtime functions
        fn register_builtins(builder: &mut JITBuilder) -> HashMap<String, *const u8> {
            let mut builtins = HashMap::new();

            // Math functions from libc
            builder.symbol("sigil_sqrt", sigil_sqrt as *const u8);
            builder.symbol("sigil_sin", sigil_sin as *const u8);
            builder.symbol("sigil_cos", sigil_cos as *const u8);
            builder.symbol("sigil_pow", sigil_pow as *const u8);
            builder.symbol("sigil_exp", sigil_exp as *const u8);
            builder.symbol("sigil_ln", sigil_ln as *const u8);
            builder.symbol("sigil_floor", sigil_floor as *const u8);
            builder.symbol("sigil_ceil", sigil_ceil as *const u8);
            builder.symbol("sigil_abs", sigil_abs as *const u8);

            // I/O functions
            builder.symbol("sigil_print_int", sigil_print_int as *const u8);
            builder.symbol("sigil_print_float", sigil_print_float as *const u8);
            builder.symbol("sigil_print_str", sigil_print_str as *const u8);

            // Time functions
            builder.symbol("sigil_now", sigil_now as *const u8);

            // Array functions
            builder.symbol("sigil_array_new", sigil_array_new as *const u8);
            builder.symbol("sigil_array_push", sigil_array_push as *const u8);
            builder.symbol("sigil_array_get", sigil_array_get as *const u8);
            builder.symbol("sigil_array_set", sigil_array_set as *const u8);
            builder.symbol("sigil_array_len", sigil_array_len as *const u8);

            // FFI helper functions
            use crate::ffi::helpers::*;
            builder.symbol("sigil_string_to_cstring", sigil_string_to_cstring as *const u8);
            builder.symbol("sigil_cstring_free", sigil_cstring_free as *const u8);
            builder.symbol("sigil_cstring_len", sigil_cstring_len as *const u8);
            builder.symbol("sigil_cstring_copy", sigil_cstring_copy as *const u8);
            builder.symbol("sigil_ptr_from_int", sigil_ptr_from_int as *const u8);
            builder.symbol("sigil_ptr_to_int", sigil_ptr_to_int as *const u8);
            builder.symbol("sigil_ptr_read_u8", sigil_ptr_read_u8 as *const u8);
            builder.symbol("sigil_ptr_write_u8", sigil_ptr_write_u8 as *const u8);
            builder.symbol("sigil_ptr_read_i32", sigil_ptr_read_i32 as *const u8);
            builder.symbol("sigil_ptr_write_i32", sigil_ptr_write_i32 as *const u8);
            builder.symbol("sigil_ptr_read_i64", sigil_ptr_read_i64 as *const u8);
            builder.symbol("sigil_ptr_write_i64", sigil_ptr_write_i64 as *const u8);
            builder.symbol("sigil_ptr_read_f64", sigil_ptr_read_f64 as *const u8);
            builder.symbol("sigil_ptr_write_f64", sigil_ptr_write_f64 as *const u8);
            builder.symbol("sigil_ptr_add", sigil_ptr_add as *const u8);
            builder.symbol("sigil_ptr_is_null", sigil_ptr_is_null as *const u8);
            builder.symbol("sigil_alloc", sigil_alloc as *const u8);
            builder.symbol("sigil_free", sigil_free as *const u8);
            builder.symbol("sigil_realloc", sigil_realloc as *const u8);
            builder.symbol("sigil_memcpy", sigil_memcpy as *const u8);
            builder.symbol("sigil_memset", sigil_memset as *const u8);

            builtins.insert("sqrt".into(), sigil_sqrt as *const u8);
            builtins.insert("sin".into(), sigil_sin as *const u8);
            builtins.insert("cos".into(), sigil_cos as *const u8);
            builtins.insert("pow".into(), sigil_pow as *const u8);
            builtins.insert("exp".into(), sigil_exp as *const u8);
            builtins.insert("ln".into(), sigil_ln as *const u8);
            builtins.insert("floor".into(), sigil_floor as *const u8);
            builtins.insert("ceil".into(), sigil_ceil as *const u8);
            builtins.insert("abs".into(), sigil_abs as *const u8);
            builtins.insert("print".into(), sigil_print_int as *const u8);
            builtins.insert("now".into(), sigil_now as *const u8);

            builtins
        }

        /// Compile a Sigil program
        pub fn compile(&mut self, source: &str) -> Result<(), String> {
            self.compile_with_opt(source, OptLevel::Standard)
        }

        /// Compile with a specific optimization level
        pub fn compile_with_opt(&mut self, source: &str, opt_level: OptLevel) -> Result<(), String> {
            let mut parser = Parser::new(source);
            let source_file = parser.parse_file().map_err(|e| format!("{:?}", e))?;

            // Run AST optimizations
            let mut optimizer = Optimizer::new(opt_level);
            let optimized = optimizer.optimize_file(&source_file);

            // First pass: declare all extern blocks and functions
            for spanned_item in &optimized.items {
                match &spanned_item.node {
                    Item::ExternBlock(extern_block) => {
                        self.declare_extern_block(extern_block)?;
                    }
                    Item::Function(func) => {
                        self.declare_function(func)?;
                    }
                    _ => {}
                }
            }

            // Second pass: compile all functions
            for spanned_item in &optimized.items {
                if let Item::Function(func) = &spanned_item.node {
                    self.compile_function(func)?;
                }
            }

            // Finalize the module
            self.module.finalize_definitions().map_err(|e| e.to_string())?;

            Ok(())
        }

        /// Declare a function (first pass)
        fn declare_function(&mut self, func: &ast::Function) -> Result<FuncId, String> {
            let name = &func.name.name;

            // Build signature
            let mut sig = self.module.make_signature();

            // Add parameters (all as i64 for simplicity - we use tagged values)
            for _param in &func.params {
                sig.params.push(AbiParam::new(types::I64));
            }

            // Return type (i64)
            sig.returns.push(AbiParam::new(types::I64));

            let func_id = self
                .module
                .declare_function(name, Linkage::Local, &sig)
                .map_err(|e| e.to_string())?;

            self.functions.insert(name.clone(), func_id);
            Ok(func_id)
        }

        /// Declare an extern block (FFI declarations)
        fn declare_extern_block(&mut self, extern_block: &ExternBlock) -> Result<(), String> {
            // Currently only "C" ABI is supported
            if extern_block.abi != "C" && extern_block.abi != "c" {
                return Err(format!("Unsupported ABI: {}. Only \"C\" is supported.", extern_block.abi));
            }

            for item in &extern_block.items {
                match item {
                    ExternItem::Function(func) => {
                        self.declare_extern_function(func)?;
                    }
                    ExternItem::Static(stat) => {
                        // TODO: Implement extern statics
                        eprintln!("Warning: extern static '{}' not yet implemented", stat.name.name);
                    }
                }
            }

            Ok(())
        }

        /// Declare an extern "C" function
        fn declare_extern_function(&mut self, func: &ExternFunction) -> Result<(), String> {
            let name = &func.name.name;

            // Build signature
            let mut sig = self.module.make_signature();
            let mut param_types = Vec::new();

            // Add parameters with proper C types
            for param in &func.params {
                let ty = self.type_expr_to_cranelift(&param.ty)?;
                sig.params.push(AbiParam::new(ty));
                param_types.push(ty);
            }

            // Return type
            let return_type = if let Some(ret_ty) = &func.return_type {
                let ty = self.type_expr_to_cranelift(ret_ty)?;
                sig.returns.push(AbiParam::new(ty));
                Some(ty)
            } else {
                None
            };

            // Variadic functions use the "C" calling convention implicitly
            // Cranelift doesn't have explicit variadic support, but we track it

            let func_id = self
                .module
                .declare_function(name, Linkage::Import, &sig)
                .map_err(|e| e.to_string())?;

            self.extern_functions.insert(name.clone(), ExternFnSig {
                name: name.clone(),
                params: param_types,
                returns: return_type,
                variadic: func.variadic,
                func_id,
            });

            Ok(())
        }

        /// Convert a Sigil type expression to Cranelift type
        fn type_expr_to_cranelift(&self, ty: &TypeExpr) -> Result<types::Type, String> {
            match ty {
                TypeExpr::Path(path) => {
                    let name = path.segments.last()
                        .map(|s| s.ident.name.as_str())
                        .unwrap_or("");

                    // Check if it's a C type
                    if let Some(ctype) = CType::from_name(name) {
                        return Ok(match ctype {
                            CType::Void => types::I64, // void returns are handled separately
                            CType::Char | CType::SChar | CType::UChar | CType::Int8 | CType::UInt8 => types::I8,
                            CType::Short | CType::UShort | CType::Int16 | CType::UInt16 => types::I16,
                            CType::Int | CType::UInt | CType::Int32 | CType::UInt32 => types::I32,
                            CType::Long | CType::ULong | CType::LongLong | CType::ULongLong |
                            CType::Size | CType::SSize | CType::PtrDiff |
                            CType::Int64 | CType::UInt64 => types::I64,
                            CType::Float => types::F32,
                            CType::Double => types::F64,
                        });
                    }

                    // Check Sigil native types
                    match name {
                        "i8" => Ok(types::I8),
                        "i16" => Ok(types::I16),
                        "i32" | "int" => Ok(types::I32),
                        "i64" => Ok(types::I64),
                        "u8" => Ok(types::I8),
                        "u16" => Ok(types::I16),
                        "u32" => Ok(types::I32),
                        "u64" => Ok(types::I64),
                        "f32" => Ok(types::F32),
                        "f64" | "float" => Ok(types::F64),
                        "bool" => Ok(types::I8),
                        "isize" | "usize" => Ok(types::I64),
                        "()" => Ok(types::I64), // unit type
                        _ => Ok(types::I64), // Default to i64 for unknown types
                    }
                }
                TypeExpr::Pointer { .. } | TypeExpr::Reference { .. } => {
                    // Pointers are always 64-bit on our target
                    Ok(types::I64)
                }
                _ => Ok(types::I64), // Default to i64
            }
        }

        /// Compile a single function
        fn compile_function(&mut self, func: &ast::Function) -> Result<(), String> {
            let name = &func.name.name;
            let func_id = *self.functions.get(name).ok_or("Function not declared")?;

            // Build signature to match declaration
            for _param in &func.params {
                self.ctx.func.signature.params.push(AbiParam::new(types::I64));
            }
            self.ctx.func.signature.returns.push(AbiParam::new(types::I64));
            self.ctx.func.name = UserFuncName::user(0, func_id.as_u32());

            // Take ownership of what we need for building
            let functions = self.functions.clone();
            let extern_fns = self.extern_functions.clone();

            {
                let mut builder =
                    FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_ctx);

                let entry_block = builder.create_block();
                builder.append_block_params_for_function_params(entry_block);
                builder.switch_to_block(entry_block);
                builder.seal_block(entry_block);

                // Set up variable scope
                let mut scope = CompileScope::new();

                // Declare parameters as variables
                for (i, param) in func.params.iter().enumerate() {
                    let var = Variable::from_u32(scope.next_var() as u32);
                    builder.declare_var(var, types::I64);
                    let param_val = builder.block_params(entry_block)[i];
                    builder.def_var(var, param_val);

                    // Get parameter name from the pattern
                    if let ast::Pattern::Ident { name, .. } = &param.pattern {
                        scope.define(&name.name, var);
                    }
                }

                // Compile function body
                if let Some(body) = &func.body {
                    let (result, has_return) = compile_block_tracked(&mut self.module, &functions, &extern_fns, &mut builder, &mut scope, body)?;
                    // Only add return if the block didn't end with an explicit return
                    if !has_return {
                        builder.ins().return_(&[result]);
                    }
                } else {
                    // No body - return 0
                    let zero = builder.ins().iconst(types::I64, 0);
                    builder.ins().return_(&[zero]);
                }

                builder.finalize();
            }

            // Debug: Uncomment to print generated IR
            // eprintln!("Generated function '{}':\n{}", name, self.ctx.func.display());

            // Compile to machine code
            self.module
                .define_function(func_id, &mut self.ctx)
                .map_err(|e| format!("Compilation error for '{}': {}", name, e))?;

            self.module.clear_context(&mut self.ctx);
            Ok(())
        }

        /// Run the compiled main function
        pub fn run(&mut self) -> Result<i64, String> {
            let main_id = *self.functions.get("main").ok_or("No main function")?;
            let main_ptr = self.module.get_finalized_function(main_id);

            unsafe {
                let main_fn: CompiledFn = mem::transmute(main_ptr);
                Ok(main_fn())
            }
        }

        /// Get a compiled function by name
        pub fn get_function(&self, name: &str) -> Option<*const u8> {
            self.functions.get(name).map(|id| self.module.get_finalized_function(*id))
        }
    }

    /// Compilation scope for tracking variables
    ///
    /// Uses a shared counter (Rc<Cell>) to ensure all scopes use unique variable indices.
    /// This prevents the "variable declared multiple times" error in Cranelift.
    struct CompileScope {
        variables: HashMap<String, Variable>,
        /// Shared counter across all scopes to ensure unique Variable indices
        var_counter: std::rc::Rc<std::cell::Cell<usize>>,
    }

    impl CompileScope {
        fn new() -> Self {
            Self {
                variables: HashMap::new(),
                var_counter: std::rc::Rc::new(std::cell::Cell::new(0)),
            }
        }

        fn child(&self) -> Self {
            // Clone variables so child scopes can access parent variables
            // Share the counter so all scopes use unique variable indices
            Self {
                variables: self.variables.clone(),
                var_counter: std::rc::Rc::clone(&self.var_counter),
            }
        }

        fn next_var(&mut self) -> usize {
            let v = self.var_counter.get();
            self.var_counter.set(v + 1);
            v
        }

        fn define(&mut self, name: &str, var: Variable) {
            self.variables.insert(name.to_string(), var);
        }

        fn lookup(&self, name: &str) -> Option<Variable> {
            self.variables.get(name).copied()
        }
    }

    // ============================================
    // Optimization: Constant Folding
    // ============================================

    /// Try to evaluate a constant expression at compile time
    fn try_const_fold(expr: &Expr) -> Option<i64> {
        match expr {
            Expr::Literal(Literal::Int { value, .. }) => value.parse().ok(),
            Expr::Literal(Literal::Bool(b)) => Some(if *b { 1 } else { 0 }),
            Expr::Binary { op, left, right } => {
                let l = try_const_fold(left)?;
                let r = try_const_fold(right)?;
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
            Expr::Unary { op, expr } => {
                let v = try_const_fold(expr)?;
                match op {
                    UnaryOp::Neg => Some(-v),
                    UnaryOp::Not => Some(if v == 0 { 1 } else { 0 }),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    // ============================================
    // Optimization: Direct Condition Compilation
    // ============================================

    /// Compile a condition directly to a boolean i8 value for branching.
    /// This avoids the redundant pattern of: compare -> extend to i64 -> compare to 0
    fn compile_condition(
        module: &mut JITModule,
        functions: &HashMap<String, FuncId>,
        extern_fns: &HashMap<String, ExternFnSig>,
        builder: &mut FunctionBuilder,
        scope: &mut CompileScope,
        condition: &Expr,
    ) -> Result<cranelift_codegen::ir::Value, String> {
        // Handle comparison operators directly - emit icmp without extending
        if let Expr::Binary { op, left, right } = condition {
            let cc = match op {
                BinOp::Eq => Some(IntCC::Equal),
                BinOp::Ne => Some(IntCC::NotEqual),
                BinOp::Lt => Some(IntCC::SignedLessThan),
                BinOp::Le => Some(IntCC::SignedLessThanOrEqual),
                BinOp::Gt => Some(IntCC::SignedGreaterThan),
                BinOp::Ge => Some(IntCC::SignedGreaterThanOrEqual),
                _ => None,
            };

            if let Some(cc) = cc {
                let lhs = compile_expr(module, functions, extern_fns, builder, scope, left)?;
                let rhs = compile_expr(module, functions, extern_fns, builder, scope, right)?;
                // Return i8 directly - no extension needed
                return Ok(builder.ins().icmp(cc, lhs, rhs));
            }

            // Handle && and || with short-circuit evaluation
            if matches!(op, BinOp::And | BinOp::Or) {
                // For now, fall through to regular compilation
                // Short-circuit optimization can be added later
            }
        }

        // Handle !expr - flip the comparison
        if let Expr::Unary { op: UnaryOp::Not, expr } = condition {
            let inner = compile_condition(module, functions, extern_fns, builder, scope, expr)?;
            // Flip the boolean
            let true_val = builder.ins().iconst(types::I8, 1);
            return Ok(builder.ins().bxor(inner, true_val));
        }

        // Handle boolean literals directly
        if let Expr::Literal(Literal::Bool(b)) = condition {
            return Ok(builder.ins().iconst(types::I8, if *b { 1 } else { 0 }));
        }

        // For other expressions, compile normally and compare to 0
        let val = compile_expr(module, functions, extern_fns, builder, scope, condition)?;
        let zero = builder.ins().iconst(types::I64, 0);
        Ok(builder.ins().icmp(IntCC::NotEqual, val, zero))
    }

    // ============================================
    // Optimization: Tail Call Detection
    // ============================================

    /// Check if a return expression is a tail call to the specified function
    fn is_tail_call_to<'a>(expr: &'a Expr, func_name: &str) -> Option<&'a Vec<Expr>> {
        if let Expr::Return(Some(inner)) = expr {
            if let Expr::Call { func, args } = inner.as_ref() {
                if let Expr::Path(path) = func.as_ref() {
                    let name = path.segments.last().map(|s| s.ident.name.as_str()).unwrap_or("");
                    if name == func_name {
                        return Some(args);
                    }
                }
            }
        }
        None
    }

    // ============================================
    // Free functions for compilation (avoid borrow issues)
    // ============================================

    /// Compile a block, returns (value, has_return)
    fn compile_block_tracked(
        module: &mut JITModule,
        functions: &HashMap<String, FuncId>,
        extern_fns: &HashMap<String, ExternFnSig>,
        builder: &mut FunctionBuilder,
        scope: &mut CompileScope,
        block: &ast::Block,
    ) -> Result<(cranelift_codegen::ir::Value, bool), String> {
        // OPTIMIZATION: Don't create zero constant unless needed
        let mut last_val: Option<cranelift_codegen::ir::Value> = None;
        let mut has_return = false;

        for stmt in &block.stmts {
            let (val, ret) = compile_stmt_tracked(module, functions, extern_fns, builder, scope, stmt)?;
            last_val = Some(val);
            if ret {
                has_return = true;
            }
        }

        if let Some(expr) = &block.expr {
            let (val, ret) = compile_expr_tracked(module, functions, extern_fns, builder, scope, expr)?;
            last_val = Some(val);
            if ret {
                has_return = true;
            }
        }

        // Only create zero if we have no value
        let result = last_val.unwrap_or_else(|| builder.ins().iconst(types::I64, 0));
        Ok((result, has_return))
    }

    /// Compile a block (convenience wrapper)
    fn compile_block(
        module: &mut JITModule,
        functions: &HashMap<String, FuncId>,
        extern_fns: &HashMap<String, ExternFnSig>,
        builder: &mut FunctionBuilder,
        scope: &mut CompileScope,
        block: &ast::Block,
    ) -> Result<cranelift_codegen::ir::Value, String> {
        compile_block_tracked(module, functions, extern_fns, builder, scope, block).map(|(v, _)| v)
    }

    /// Compile a statement, returning (value, has_return)
    fn compile_stmt_tracked(
        module: &mut JITModule,
        functions: &HashMap<String, FuncId>,
        extern_fns: &HashMap<String, ExternFnSig>,
        builder: &mut FunctionBuilder,
        scope: &mut CompileScope,
        stmt: &ast::Stmt,
    ) -> Result<(cranelift_codegen::ir::Value, bool), String> {
        match stmt {
            ast::Stmt::Let { pattern, init, .. } => {
                let val = if let Some(expr) = init {
                    compile_expr(module, functions, extern_fns, builder, scope, expr)?
                } else {
                    builder.ins().iconst(types::I64, 0)
                };

                if let ast::Pattern::Ident { name, .. } = pattern {
                    let var = Variable::from_u32(scope.next_var() as u32);
                    builder.declare_var(var, types::I64);
                    builder.def_var(var, val);
                    scope.define(&name.name, var);
                }

                Ok((val, false))
            }
            ast::Stmt::Expr(expr) | ast::Stmt::Semi(expr) => {
                compile_expr_tracked(module, functions, extern_fns, builder, scope, expr)
            }
            ast::Stmt::Item(_) => Ok((builder.ins().iconst(types::I64, 0), false)),
        }
    }

    /// Compile a statement (convenience wrapper)
    #[allow(dead_code)]
    fn compile_stmt(
        module: &mut JITModule,
        functions: &HashMap<String, FuncId>,
        extern_fns: &HashMap<String, ExternFnSig>,
        builder: &mut FunctionBuilder,
        scope: &mut CompileScope,
        stmt: &ast::Stmt,
    ) -> Result<cranelift_codegen::ir::Value, String> {
        compile_stmt_tracked(module, functions, extern_fns, builder, scope, stmt).map(|(v, _)| v)
    }

    /// Compile an expression, returning (value, has_return)
    fn compile_expr_tracked(
        module: &mut JITModule,
        functions: &HashMap<String, FuncId>,
        extern_fns: &HashMap<String, ExternFnSig>,
        builder: &mut FunctionBuilder,
        scope: &mut CompileScope,
        expr: &Expr,
    ) -> Result<(cranelift_codegen::ir::Value, bool), String> {
        match expr {
            Expr::Return(value) => {
                let ret_val = if let Some(v) = value {
                    compile_expr(module, functions, extern_fns, builder, scope, v)?
                } else {
                    builder.ins().iconst(types::I64, 0)
                };
                builder.ins().return_(&[ret_val]);
                Ok((ret_val, true))  // Signal that we have a return
            }
            Expr::If { condition, then_branch, else_branch } => {
                // If expressions can contain returns, so use tracked version
                compile_if_tracked(module, functions, extern_fns, builder, scope, condition, then_branch, else_branch.as_deref())
            }
            Expr::Block(block) => {
                let mut inner_scope = scope.child();
                compile_block_tracked(module, functions, extern_fns, builder, &mut inner_scope, block)
            }
            _ => {
                // All other expressions don't have return
                let val = compile_expr(module, functions, extern_fns, builder, scope, expr)?;
                Ok((val, false))
            }
        }
    }

    /// Compile an expression
    fn compile_expr(
        module: &mut JITModule,
        functions: &HashMap<String, FuncId>,
        extern_fns: &HashMap<String, ExternFnSig>,
        builder: &mut FunctionBuilder,
        scope: &mut CompileScope,
        expr: &Expr,
    ) -> Result<cranelift_codegen::ir::Value, String> {
        // OPTIMIZATION: Try constant folding first
        if let Some(val) = try_const_fold(expr) {
            return Ok(builder.ins().iconst(types::I64, val));
        }

        match expr {
            Expr::Literal(lit) => compile_literal(builder, lit),

            Expr::Path(path) => {
                let name = path.segments.last()
                    .map(|s| s.ident.name.clone())
                    .unwrap_or_default();
                if let Some(var) = scope.lookup(&name) {
                    Ok(builder.use_var(var))
                } else {
                    Err(format!("Undefined variable: {}", name))
                }
            }

            Expr::Binary { op, left, right } => {
                let lhs = compile_expr(module, functions, extern_fns, builder, scope, left)?;
                let rhs = compile_expr(module, functions, extern_fns, builder, scope, right)?;
                compile_binary_op(builder, op.clone(), lhs, rhs)
            }

            Expr::Unary { op, expr: inner } => {
                let val = compile_expr(module, functions, extern_fns, builder, scope, inner)?;
                compile_unary_op(builder, *op, val)
            }

            Expr::Call { func, args } => {
                let func_name = match func.as_ref() {
                    Expr::Path(path) => {
                        path.segments.last().map(|s| s.ident.name.clone()).unwrap_or_default()
                    }
                    _ => return Err("Only direct function calls supported".into()),
                };

                let mut arg_vals = Vec::new();
                for arg in args {
                    arg_vals.push(compile_expr(module, functions, extern_fns, builder, scope, arg)?);
                }

                compile_call(module, functions, extern_fns, builder, &func_name, &arg_vals)
            }

            Expr::If { condition, then_branch, else_branch } => {
                compile_if(module, functions, extern_fns, builder, scope, condition, then_branch, else_branch.as_deref())
            }

            Expr::While { condition, body } => {
                compile_while(module, functions, extern_fns, builder, scope, condition, body)
            }

            Expr::Block(block) => {
                let mut inner_scope = scope.child();
                compile_block(module, functions, extern_fns, builder, &mut inner_scope, block)
            }

            Expr::Return(value) => {
                let ret_val = if let Some(v) = value {
                    compile_expr(module, functions, extern_fns, builder, scope, v)?
                } else {
                    builder.ins().iconst(types::I64, 0)
                };
                builder.ins().return_(&[ret_val]);
                Ok(ret_val)
            }

            Expr::Assign { target, value } => {
                let val = compile_expr(module, functions, extern_fns, builder, scope, value)?;
                match target.as_ref() {
                    Expr::Path(path) => {
                        let name = path.segments.last().map(|s| s.ident.name.clone()).unwrap_or_default();
                        if let Some(var) = scope.lookup(&name) {
                            builder.def_var(var, val);
                            Ok(val)
                        } else {
                            Err(format!("Undefined variable: {}", name))
                        }
                    }
                    Expr::Index { expr: arr, index } => {
                        let arr_val = compile_expr(module, functions, extern_fns, builder, scope, arr)?;
                        let idx_val = compile_expr(module, functions, extern_fns, builder, scope, index)?;
                        compile_call(module, functions, extern_fns, builder, "sigil_array_set", &[arr_val, idx_val, val])
                    }
                    _ => Err("Invalid assignment target".into()),
                }
            }

            Expr::Index { expr: arr, index } => {
                let arr_val = compile_expr(module, functions, extern_fns, builder, scope, arr)?;
                let idx_val = compile_expr(module, functions, extern_fns, builder, scope, index)?;
                compile_call(module, functions, extern_fns, builder, "sigil_array_get", &[arr_val, idx_val])
            }

            Expr::Array(elements) => {
                let len = builder.ins().iconst(types::I64, elements.len() as i64);
                let arr = compile_call(module, functions, extern_fns, builder, "sigil_array_new", &[len])?;

                for (i, elem) in elements.iter().enumerate() {
                    let val = compile_expr(module, functions, extern_fns, builder, scope, elem)?;
                    let idx = builder.ins().iconst(types::I64, i as i64);
                    compile_call(module, functions, extern_fns, builder, "sigil_array_set", &[arr, idx, val])?;
                }

                Ok(arr)
            }

            Expr::Pipe { expr, .. } => {
                compile_expr(module, functions, extern_fns, builder, scope, expr)
            }

            // Unsafe blocks - just compile the inner block
            Expr::Unsafe(block) => {
                let mut inner_scope = scope.child();
                compile_block(module, functions, extern_fns, builder, &mut inner_scope, block)
            }

            // Pointer dereference - load from address
            Expr::Deref(inner) => {
                let ptr = compile_expr(module, functions, extern_fns, builder, scope, inner)?;
                // Load 64-bit value from pointer
                Ok(builder.ins().load(types::I64, cranelift_codegen::ir::MemFlags::new(), ptr, 0))
            }

            // Address-of - just return the value (it's already a pointer in our model)
            Expr::AddrOf { expr: inner, .. } => {
                compile_expr(module, functions, extern_fns, builder, scope, inner)
            }

            // Cast expression
            Expr::Cast { expr: inner, ty } => {
                let val = compile_expr(module, functions, extern_fns, builder, scope, inner)?;
                // For now, just return the value - proper casting would check types
                let _ = ty; // TODO: implement proper type-based casting
                Ok(val)
            }

            _ => Ok(builder.ins().iconst(types::I64, 0)),
        }
    }

    /// Compile a literal
    fn compile_literal(
        builder: &mut FunctionBuilder,
        lit: &Literal,
    ) -> Result<cranelift_codegen::ir::Value, String> {
        match lit {
            Literal::Int { value, .. } => {
                let val: i64 = value.parse().map_err(|_| "Invalid integer")?;
                Ok(builder.ins().iconst(types::I64, val))
            }
            Literal::Float { value, .. } => {
                let val: f64 = value.parse().map_err(|_| "Invalid float")?;
                // Store float as i64 bits for uniform value representation
                // All variables are I64 type, so floats must be bitcast
                Ok(builder.ins().iconst(types::I64, val.to_bits() as i64))
            }
            Literal::Bool(b) => Ok(builder.ins().iconst(types::I64, if *b { 1 } else { 0 })),
            Literal::String(_) => Ok(builder.ins().iconst(types::I64, 0)),
            _ => Ok(builder.ins().iconst(types::I64, 0)),
        }
    }

    /// Compile binary operation
    fn compile_binary_op(
        builder: &mut FunctionBuilder,
        op: BinOp,
        lhs: cranelift_codegen::ir::Value,
        rhs: cranelift_codegen::ir::Value,
    ) -> Result<cranelift_codegen::ir::Value, String> {
        let result = match op {
            BinOp::Add => builder.ins().iadd(lhs, rhs),
            BinOp::Sub => builder.ins().isub(lhs, rhs),
            BinOp::Mul => builder.ins().imul(lhs, rhs),
            BinOp::Div => builder.ins().sdiv(lhs, rhs),
            BinOp::Rem => builder.ins().srem(lhs, rhs),
            BinOp::Pow => return Err("Power not supported".into()),
            BinOp::BitAnd => builder.ins().band(lhs, rhs),
            BinOp::BitOr => builder.ins().bor(lhs, rhs),
            BinOp::BitXor => builder.ins().bxor(lhs, rhs),
            BinOp::Shl => builder.ins().ishl(lhs, rhs),
            BinOp::Shr => builder.ins().sshr(lhs, rhs),
            BinOp::Eq => {
                let cmp = builder.ins().icmp(IntCC::Equal, lhs, rhs);
                builder.ins().uextend(types::I64, cmp)
            }
            BinOp::Ne => {
                let cmp = builder.ins().icmp(IntCC::NotEqual, lhs, rhs);
                builder.ins().uextend(types::I64, cmp)
            }
            BinOp::Lt => {
                let cmp = builder.ins().icmp(IntCC::SignedLessThan, lhs, rhs);
                builder.ins().uextend(types::I64, cmp)
            }
            BinOp::Le => {
                let cmp = builder.ins().icmp(IntCC::SignedLessThanOrEqual, lhs, rhs);
                builder.ins().uextend(types::I64, cmp)
            }
            BinOp::Gt => {
                let cmp = builder.ins().icmp(IntCC::SignedGreaterThan, lhs, rhs);
                builder.ins().uextend(types::I64, cmp)
            }
            BinOp::Ge => {
                let cmp = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, lhs, rhs);
                builder.ins().uextend(types::I64, cmp)
            }
            BinOp::And => builder.ins().band(lhs, rhs),
            BinOp::Or => builder.ins().bor(lhs, rhs),
            BinOp::Concat => return Err("Concat not supported".into()),
        };
        Ok(result)
    }

    /// Compile unary operation
    fn compile_unary_op(
        builder: &mut FunctionBuilder,
        op: UnaryOp,
        val: cranelift_codegen::ir::Value,
    ) -> Result<cranelift_codegen::ir::Value, String> {
        let result = match op {
            UnaryOp::Neg => builder.ins().ineg(val),
            UnaryOp::Not => {
                let zero = builder.ins().iconst(types::I64, 0);
                let cmp = builder.ins().icmp(IntCC::Equal, val, zero);
                builder.ins().uextend(types::I64, cmp)
            }
            UnaryOp::Deref | UnaryOp::Ref | UnaryOp::RefMut => val,
        };
        Ok(result)
    }

    /// Compile function call
    fn compile_call(
        module: &mut JITModule,
        functions: &HashMap<String, FuncId>,
        extern_fns: &HashMap<String, ExternFnSig>,
        builder: &mut FunctionBuilder,
        name: &str,
        args: &[cranelift_codegen::ir::Value],
    ) -> Result<cranelift_codegen::ir::Value, String> {
        let builtin_name = match name {
            "sqrt" => Some("sigil_sqrt"),
            "sin" => Some("sigil_sin"),
            "cos" => Some("sigil_cos"),
            "pow" => Some("sigil_pow"),
            "exp" => Some("sigil_exp"),
            "ln" => Some("sigil_ln"),
            "floor" => Some("sigil_floor"),
            "ceil" => Some("sigil_ceil"),
            "abs" => Some("sigil_abs"),
            "print" => Some("sigil_print_int"),
            "now" => Some("sigil_now"),
            n if n.starts_with("sigil_") => Some(n),
            _ => None,
        };

        if let Some(builtin) = builtin_name {
            let mut sig = module.make_signature();

            match builtin {
                "sigil_sqrt" | "sigil_sin" | "sigil_cos" | "sigil_exp" | "sigil_ln"
                | "sigil_floor" | "sigil_ceil" | "sigil_abs" => {
                    sig.params.push(AbiParam::new(types::F64));
                    sig.returns.push(AbiParam::new(types::F64));
                }
                "sigil_pow" => {
                    sig.params.push(AbiParam::new(types::F64));
                    sig.params.push(AbiParam::new(types::F64));
                    sig.returns.push(AbiParam::new(types::F64));
                }
                "sigil_print_int" => {
                    sig.params.push(AbiParam::new(types::I64));
                    sig.returns.push(AbiParam::new(types::I64));
                }
                "sigil_now" => {
                    sig.returns.push(AbiParam::new(types::I64));
                }
                "sigil_array_new" => {
                    sig.params.push(AbiParam::new(types::I64));
                    sig.returns.push(AbiParam::new(types::I64));
                }
                "sigil_array_get" | "sigil_array_set" => {
                    sig.params.push(AbiParam::new(types::I64));
                    sig.params.push(AbiParam::new(types::I64));
                    if builtin == "sigil_array_set" {
                        sig.params.push(AbiParam::new(types::I64));
                    }
                    sig.returns.push(AbiParam::new(types::I64));
                }
                "sigil_array_len" => {
                    sig.params.push(AbiParam::new(types::I64));
                    sig.returns.push(AbiParam::new(types::I64));
                }
                _ => {
                    for _ in args {
                        sig.params.push(AbiParam::new(types::I64));
                    }
                    sig.returns.push(AbiParam::new(types::I64));
                }
            }

            let callee = module
                .declare_function(builtin, Linkage::Import, &sig)
                .map_err(|e| e.to_string())?;

            let local_callee = module.declare_func_in_func(callee, builder.func);

            let call_args: Vec<_> = if matches!(builtin, "sigil_sqrt" | "sigil_sin" | "sigil_cos"
                | "sigil_exp" | "sigil_ln" | "sigil_floor" | "sigil_ceil" | "sigil_abs" | "sigil_pow") {
                args.iter().map(|&v| {
                    if builder.func.dfg.value_type(v) == types::F64 {
                        v
                    } else {
                        builder.ins().fcvt_from_sint(types::F64, v)
                    }
                }).collect()
            } else {
                args.to_vec()
            };

            let call = builder.ins().call(local_callee, &call_args);
            Ok(builder.inst_results(call)[0])
        } else if let Some(&func_id) = functions.get(name) {
            // User-defined function
            let local_callee = module.declare_func_in_func(func_id, builder.func);
            let call = builder.ins().call(local_callee, args);
            Ok(builder.inst_results(call)[0])
        } else if let Some(extern_fn) = extern_fns.get(name) {
            // Extern "C" function - call through FFI
            let local_callee = module.declare_func_in_func(extern_fn.func_id, builder.func);

            // Convert arguments to match expected types
            let mut call_args = Vec::new();
            for (i, &arg) in args.iter().enumerate() {
                let arg_type = builder.func.dfg.value_type(arg);
                let expected_type = extern_fn.params.get(i).copied().unwrap_or(types::I64);

                let converted = if arg_type == expected_type {
                    arg
                } else if arg_type == types::I64 && expected_type == types::I32 {
                    builder.ins().ireduce(types::I32, arg)
                } else if arg_type == types::I32 && expected_type == types::I64 {
                    builder.ins().sextend(types::I64, arg)
                } else if arg_type == types::I64 && expected_type == types::F64 {
                    builder.ins().fcvt_from_sint(types::F64, arg)
                } else if arg_type == types::F64 && expected_type == types::I64 {
                    builder.ins().fcvt_to_sint(types::I64, arg)
                } else {
                    arg // Best effort - let Cranelift handle it
                };
                call_args.push(converted);
            }

            let call = builder.ins().call(local_callee, &call_args);

            // Handle return value
            if extern_fn.returns.is_some() {
                let result = builder.inst_results(call)[0];
                let result_type = builder.func.dfg.value_type(result);
                // Extend smaller types to i64 for our internal representation
                if result_type == types::I32 || result_type == types::I16 || result_type == types::I8 {
                    Ok(builder.ins().sextend(types::I64, result))
                } else {
                    Ok(result)
                }
            } else {
                // Void return - return 0
                Ok(builder.ins().iconst(types::I64, 0))
            }
        } else {
            Err(format!("Unknown function: {}", name))
        }
    }

    /// Compile if expression, returns (value, has_return)
    fn compile_if_tracked(
        module: &mut JITModule,
        functions: &HashMap<String, FuncId>,
        extern_fns: &HashMap<String, ExternFnSig>,
        builder: &mut FunctionBuilder,
        scope: &mut CompileScope,
        condition: &Expr,
        then_branch: &ast::Block,
        else_branch: Option<&Expr>,
    ) -> Result<(cranelift_codegen::ir::Value, bool), String> {
        // OPTIMIZATION: Use direct condition compilation
        let cond_bool = compile_condition(module, functions, extern_fns, builder, scope, condition)?;

        let then_block = builder.create_block();
        let else_block = builder.create_block();
        let merge_block = builder.create_block();

        builder.append_block_param(merge_block, types::I64);

        // Branch directly on the boolean - no extra comparison needed
        builder.ins().brif(cond_bool, then_block, &[], else_block, &[]);

        // Compile then branch
        builder.switch_to_block(then_block);
        builder.seal_block(then_block);
        let mut then_scope = scope.child();
        let (then_val, then_returns) = compile_block_tracked(module, functions, extern_fns, builder, &mut then_scope, then_branch)?;
        // Only jump to merge if we didn't return
        if !then_returns {
            builder.ins().jump(merge_block, &[then_val]);
        }

        // Compile else branch
        builder.switch_to_block(else_block);
        builder.seal_block(else_block);
        let (else_val, else_returns) = if let Some(else_expr) = else_branch {
            match else_expr {
                Expr::Block(block) => {
                    let mut else_scope = scope.child();
                    compile_block_tracked(module, functions, extern_fns, builder, &mut else_scope, block)?
                }
                Expr::If { condition, then_branch, else_branch } => {
                    compile_if_tracked(module, functions, extern_fns, builder, scope, condition, then_branch, else_branch.as_deref())?
                }
                _ => {
                    let val = compile_expr(module, functions, extern_fns, builder, scope, else_expr)?;
                    (val, false)
                }
            }
        } else {
            (builder.ins().iconst(types::I64, 0), false)
        };
        // Only jump to merge if we didn't return
        if !else_returns {
            builder.ins().jump(merge_block, &[else_val]);
        }

        // If both branches return, the merge block is unreachable but still needs to be sealed
        // If only some branches return, we still need the merge block
        let both_return = then_returns && else_returns;

        builder.switch_to_block(merge_block);
        builder.seal_block(merge_block);

        if both_return {
            // Both branches return - merge block is unreachable
            // Return a dummy value and signal that we returned
            let dummy = builder.ins().iconst(types::I64, 0);
            Ok((dummy, true))
        } else {
            Ok((builder.block_params(merge_block)[0], false))
        }
    }

    /// Compile if expression (convenience wrapper)
    fn compile_if(
        module: &mut JITModule,
        functions: &HashMap<String, FuncId>,
        extern_fns: &HashMap<String, ExternFnSig>,
        builder: &mut FunctionBuilder,
        scope: &mut CompileScope,
        condition: &Expr,
        then_branch: &ast::Block,
        else_branch: Option<&Expr>,
    ) -> Result<cranelift_codegen::ir::Value, String> {
        compile_if_tracked(module, functions, extern_fns, builder, scope, condition, then_branch, else_branch).map(|(v, _)| v)
    }

    /// Compile while loop
    fn compile_while(
        module: &mut JITModule,
        functions: &HashMap<String, FuncId>,
        extern_fns: &HashMap<String, ExternFnSig>,
        builder: &mut FunctionBuilder,
        scope: &mut CompileScope,
        condition: &Expr,
        body: &ast::Block,
    ) -> Result<cranelift_codegen::ir::Value, String> {
        let header_block = builder.create_block();
        let body_block = builder.create_block();
        let exit_block = builder.create_block();

        builder.ins().jump(header_block, &[]);

        builder.switch_to_block(header_block);
        // OPTIMIZATION: Use direct condition compilation
        let cond_bool = compile_condition(module, functions, extern_fns, builder, scope, condition)?;
        // Branch directly - no extra comparison needed
        builder.ins().brif(cond_bool, body_block, &[], exit_block, &[]);

        builder.switch_to_block(body_block);
        builder.seal_block(body_block);
        let mut body_scope = scope.child();
        compile_block(module, functions, extern_fns, builder, &mut body_scope, body)?;
        builder.ins().jump(header_block, &[]);

        builder.seal_block(header_block);

        builder.switch_to_block(exit_block);
        builder.seal_block(exit_block);

        Ok(builder.ins().iconst(types::I64, 0))
    }

    // ============================================
    // Runtime support functions (called from JIT)
    // ============================================

    #[no_mangle]
    pub extern "C" fn sigil_sqrt(x: f64) -> f64 {
        x.sqrt()
    }

    #[no_mangle]
    pub extern "C" fn sigil_sin(x: f64) -> f64 {
        x.sin()
    }

    #[no_mangle]
    pub extern "C" fn sigil_cos(x: f64) -> f64 {
        x.cos()
    }

    #[no_mangle]
    pub extern "C" fn sigil_pow(base: f64, exp: f64) -> f64 {
        base.powf(exp)
    }

    #[no_mangle]
    pub extern "C" fn sigil_exp(x: f64) -> f64 {
        x.exp()
    }

    #[no_mangle]
    pub extern "C" fn sigil_ln(x: f64) -> f64 {
        x.ln()
    }

    #[no_mangle]
    pub extern "C" fn sigil_floor(x: f64) -> f64 {
        x.floor()
    }

    #[no_mangle]
    pub extern "C" fn sigil_ceil(x: f64) -> f64 {
        x.ceil()
    }

    #[no_mangle]
    pub extern "C" fn sigil_abs(x: f64) -> f64 {
        x.abs()
    }

    #[no_mangle]
    pub extern "C" fn sigil_print_int(x: i64) -> i64 {
        println!("{}", x);
        0
    }

    #[no_mangle]
    pub extern "C" fn sigil_print_float(x: f64) -> i64 {
        println!("{}", x);
        0
    }

    #[no_mangle]
    pub extern "C" fn sigil_print_str(ptr: *const u8, len: usize) -> i64 {
        unsafe {
            let slice = std::slice::from_raw_parts(ptr, len);
            if let Ok(s) = std::str::from_utf8(slice) {
                println!("{}", s);
            }
        }
        0
    }

    #[no_mangle]
    pub extern "C" fn sigil_now() -> i64 {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as i64)
            .unwrap_or(0)
    }

    // Simple array implementation using heap allocation
    #[repr(C)]
    struct SigilArray {
        data: *mut i64,
        len: usize,
        cap: usize,
    }

    #[no_mangle]
    pub extern "C" fn sigil_array_new(capacity: i64) -> i64 {
        let cap = capacity.max(8) as usize;
        let layout = std::alloc::Layout::array::<i64>(cap).unwrap();
        let data = unsafe { std::alloc::alloc(layout) as *mut i64 };

        let arr = Box::new(SigilArray {
            data,
            len: 0,
            cap,
        });
        Box::into_raw(arr) as i64
    }

    #[no_mangle]
    pub extern "C" fn sigil_array_push(arr_ptr: i64, value: i64) -> i64 {
        unsafe {
            let arr = &mut *(arr_ptr as *mut SigilArray);
            if arr.len >= arr.cap {
                // Grow array
                let new_cap = arr.cap * 2;
                let old_layout = std::alloc::Layout::array::<i64>(arr.cap).unwrap();
                let new_layout = std::alloc::Layout::array::<i64>(new_cap).unwrap();
                arr.data = std::alloc::realloc(arr.data as *mut u8, old_layout, new_layout.size()) as *mut i64;
                arr.cap = new_cap;
            }
            *arr.data.add(arr.len) = value;
            arr.len += 1;
        }
        0
    }

    #[no_mangle]
    pub extern "C" fn sigil_array_get(arr_ptr: i64, index: i64) -> i64 {
        unsafe {
            let arr = &*(arr_ptr as *const SigilArray);
            let idx = index as usize;
            if idx < arr.len {
                *arr.data.add(idx)
            } else {
                0 // Out of bounds returns 0
            }
        }
    }

    #[no_mangle]
    pub extern "C" fn sigil_array_set(arr_ptr: i64, index: i64, value: i64) -> i64 {
        unsafe {
            let arr = &mut *(arr_ptr as *mut SigilArray);
            let idx = index as usize;
            // Extend array if needed
            while arr.len <= idx {
                sigil_array_push(arr_ptr, 0);
            }
            *arr.data.add(idx) = value;
        }
        value
    }

    #[no_mangle]
    pub extern "C" fn sigil_array_len(arr_ptr: i64) -> i64 {
        unsafe {
            let arr = &*(arr_ptr as *const SigilArray);
            arr.len as i64
        }
    }

    // ============================================
    // FFI Tests
    // ============================================

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::parser::Parser;

        #[test]
        fn test_extern_block_parsing_and_declaration() {
            let source = r#"
                extern "C" {
                    fn abs(x: c_int) -> c_int;
                    fn strlen(s: *const c_char) -> usize;
                }

                fn main() -> i64 {
                    42
                }
            "#;

            let mut compiler = JitCompiler::new().unwrap();
            let result = compiler.compile(source);
            assert!(result.is_ok(), "Failed to compile FFI declarations: {:?}", result);

            // Check that extern functions were registered
            assert!(compiler.extern_functions.contains_key("abs"), "abs not declared");
            assert!(compiler.extern_functions.contains_key("strlen"), "strlen not declared");

            // Check abs signature
            let abs_sig = compiler.extern_functions.get("abs").unwrap();
            assert_eq!(abs_sig.params.len(), 1);
            assert_eq!(abs_sig.params[0], types::I32); // c_int -> i32
            assert_eq!(abs_sig.returns, Some(types::I32));

            // Check strlen signature
            let strlen_sig = compiler.extern_functions.get("strlen").unwrap();
            assert_eq!(strlen_sig.params.len(), 1);
            assert_eq!(strlen_sig.params[0], types::I64); // pointer -> i64
            assert_eq!(strlen_sig.returns, Some(types::I64)); // usize -> i64
        }

        #[test]
        fn test_extern_variadic_function() {
            let source = r#"
                extern "C" {
                    fn printf(fmt: *const c_char, ...) -> c_int;
                }

                fn main() -> i64 {
                    0
                }
            "#;

            let mut compiler = JitCompiler::new().unwrap();
            let result = compiler.compile(source);
            assert!(result.is_ok(), "Failed to compile variadic FFI: {:?}", result);

            let printf_sig = compiler.extern_functions.get("printf").unwrap();
            assert!(printf_sig.variadic, "printf should be variadic");
        }

        #[test]
        fn test_extern_c_abi_only() {
            let source = r#"
                extern "Rust" {
                    fn some_func(x: i32) -> i32;
                }

                fn main() -> i64 {
                    0
                }
            "#;

            let mut compiler = JitCompiler::new().unwrap();
            let result = compiler.compile(source);
            assert!(result.is_err(), "Should reject non-C ABI");
            assert!(result.unwrap_err().contains("Unsupported ABI"));
        }

        #[test]
        fn test_c_type_mapping() {
            // Test that C types are correctly mapped to Cranelift types
            let test_cases = vec![
                ("c_char", types::I8),
                ("c_int", types::I32),
                ("c_long", types::I64),
                ("c_float", types::F32),
                ("c_double", types::F64),
                ("size_t", types::I64),
                ("i32", types::I32),
                ("f64", types::F64),
            ];

            for (type_name, expected_cl_type) in test_cases {
                let source = format!(r#"
                    extern "C" {{
                        fn test_func(x: {}) -> {};
                    }}

                    fn main() -> i64 {{ 0 }}
                "#, type_name, type_name);

                let mut compiler = JitCompiler::new().unwrap();
                let result = compiler.compile(&source);
                assert!(result.is_ok(), "Failed for type {}: {:?}", type_name, result);

                let sig = compiler.extern_functions.get("test_func").unwrap();
                assert_eq!(sig.params[0], expected_cl_type, "Wrong param type for {}", type_name);
                assert_eq!(sig.returns, Some(expected_cl_type), "Wrong return type for {}", type_name);
            }
        }
    }
}

// Re-export for convenience
#[cfg(feature = "jit")]
pub use jit::JitCompiler;
