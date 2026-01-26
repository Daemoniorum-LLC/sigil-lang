//! Top-level item compilation.
//!
//! Compiles Sigil functions, structs, enums, and other top-level items to WASM.

use wasm_encoder::{Instruction, ValType};

use super::error::{WasmError, WasmResult};
use super::types::{CompiledFunction, EnumLayout, StructLayout};
use super::WasmCompiler;
use crate::ast::{
    ConstDef, EnumDef, Function, ImplItem, Item, Param, SourceFile, StaticDef, StructDef,
    StructFields, Visibility,
};

impl WasmCompiler {
    /// Compile a source file.
    pub fn compile_file(&mut self, file: &SourceFile) -> WasmResult<()> {
        // First pass: collect type definitions (structs, enums)
        for item in &file.items {
            self.collect_type_def(&item.node)?;
        }

        // Second pass: collect function signatures
        for item in &file.items {
            self.collect_function_sig(&item.node)?;
        }

        // Third pass: compile function bodies
        for item in &file.items {
            self.compile_item(&item.node)?;
        }

        Ok(())
    }

    /// Collect type definition from an item.
    fn collect_type_def(&mut self, item: &Item) -> WasmResult<()> {
        match item {
            Item::Struct(def) => self.register_struct(def),
            Item::Enum(def) => self.register_enum(def),
            _ => Ok(()),
        }
    }

    /// Collect function signature from an item.
    fn collect_function_sig(&mut self, item: &Item) -> WasmResult<()> {
        match item {
            Item::Function(func) => {
                self.register_function_sig(func)?;
            }
            Item::Impl(impl_block) => {
                // Extract type name from self_ty for method name mangling
                let type_name = self.extract_type_name(&impl_block.self_ty);

                // Register each method with mangled name: TypeName_methodName
                for impl_item in &impl_block.items {
                    if let ImplItem::Function(func) = impl_item {
                        self.register_impl_method_sig(func, &type_name)?;
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Extract the simple type name from a TypeExpr.
    fn extract_type_name(&self, ty: &crate::ast::TypeExpr) -> String {
        match ty {
            crate::ast::TypeExpr::Path(path) => {
                // Get the first segment's name (e.g., "Counter" from "Counter<T>")
                path.segments
                    .first()
                    .map(|s| s.ident.name.clone())
                    .unwrap_or_else(|| "Unknown".to_string())
            }
            _ => "Unknown".to_string(),
        }
    }

    /// Register an impl method signature with mangled name.
    fn register_impl_method_sig(&mut self, func: &Function, type_name: &str) -> WasmResult<()> {
        // Mangle name: TypeName_methodName
        let mangled_name = format!("{}_{}", type_name, func.name.name);

        // Skip if already registered
        if self.func_map.contains_key(&mangled_name) {
            return Ok(());
        }

        // Build parameter types - include self parameter if present
        let param_types: Vec<ValType> = func.params.iter().map(|_| ValType::I64).collect();

        // Result type
        let result_types = if func.return_type.is_some() {
            vec![ValType::I64]
        } else {
            vec![] // Unit return
        };

        let type_idx = self.get_or_create_type(param_types.clone(), result_types.clone());
        let func_idx = self.imports.import_count() + self.functions.len() as u32;

        // Record function index with mangled name
        self.func_map.insert(mangled_name.clone(), func_idx);

        // Also register with simple name for method calls (receiver.method())
        // This allows both Counter·new() and counter.increment() to work
        if !self.func_map.contains_key(&func.name.name) {
            self.func_map.insert(func.name.name.clone(), func_idx);
        }

        // Create function
        let params_with_names: Vec<(String, ValType)> = func
            .params
            .iter()
            .map(|p| (p.pattern_name().unwrap_or_default(), ValType::I64))
            .collect();

        let is_exported = matches!(func.visibility, Visibility::Public);

        let compiled_func = CompiledFunction::new(
            mangled_name,
            type_idx,
            func_idx,
            params_with_names,
            result_types,
            is_exported,
        );

        self.functions.push(compiled_func);

        Ok(())
    }

    /// Compile a top-level item.
    fn compile_item(&mut self, item: &Item) -> WasmResult<()> {
        match item {
            Item::Function(func) => self.compile_function(func),
            Item::Const(def) => self.compile_const(def),
            Item::Static(def) => self.compile_static(def),
            // Other items are handled in earlier passes or not supported
            Item::Struct(_) | Item::Enum(_) => Ok(()),
            Item::Trait(_) => Ok(()), // Traits are compile-time only
            Item::Impl(impl_block) => {
                let type_name = self.extract_type_name(&impl_block.self_ty);
                for item in &impl_block.items {
                    if let ImplItem::Function(func) = item {
                        self.compile_impl_method(func, &type_name)?;
                    }
                }
                Ok(())
            }
            Item::TypeAlias(_) => Ok(()), // Type aliases are compile-time only
            Item::Module(_) => Err(WasmError::unsupported("nested modules")),
            Item::Use(_) => Ok(()), // Use declarations are resolved during parsing
            Item::Actor(_) => Err(WasmError::unsupported("actors")),
            Item::ExternBlock(_) => Ok(()), // Extern functions are imports
            Item::Macro(_) => Ok(()),       // Macro definitions are compile-time only
            Item::MacroInvocation(_) => Err(WasmError::unsupported("macro invocations")),
            Item::Plurality(_) => Err(WasmError::unsupported("plurality items")),
            Item::Form(_) => Err(WasmError::unsupported("form definitions")),
            Item::Translations(_) => Err(WasmError::unsupported("translation definitions")),
            Item::LocaleEnum(_) => Err(WasmError::unsupported("locale enum definitions")),
        }
    }

    /// Register a struct definition.
    fn register_struct(&mut self, def: &StructDef) -> WasmResult<()> {
        let mut layout = StructLayout::new(&def.name.name);

        match &def.fields {
            StructFields::Named(fields) => {
                for field in fields {
                    layout.add_field(&field.name.name);
                }
            }
            StructFields::Tuple(types) => {
                for (i, _) in types.iter().enumerate() {
                    layout.add_field(&format!("_{}", i));
                }
            }
            StructFields::Unit => {}
        }

        self.struct_layouts.insert(def.name.name.clone(), layout);
        Ok(())
    }

    /// Register an enum definition.
    fn register_enum(&mut self, def: &EnumDef) -> WasmResult<()> {
        let mut layout = EnumLayout::new(&def.name.name);

        for variant in &def.variants {
            let is_unit = match &variant.fields {
                StructFields::Unit => true,
                StructFields::Named(f) => f.is_empty(),
                StructFields::Tuple(f) => f.is_empty(),
            };

            if is_unit {
                layout.add_unit_variant(&variant.name.name);
            } else {
                let mut payload = StructLayout::new(&variant.name.name);
                match &variant.fields {
                    StructFields::Named(fields) => {
                        for field in fields {
                            payload.add_field(&field.name.name);
                        }
                    }
                    StructFields::Tuple(types) => {
                        for (i, _) in types.iter().enumerate() {
                            payload.add_field(&format!("_{}", i));
                        }
                    }
                    StructFields::Unit => {}
                }
                layout.add_variant_with_payload(&variant.name.name, payload);
            }
        }

        self.enum_layouts.insert(def.name.name.clone(), layout);
        Ok(())
    }

    /// Register a function signature (without compiling body).
    fn register_function_sig(&mut self, func: &Function) -> WasmResult<()> {
        // Skip if already registered
        if self.func_map.contains_key(&func.name.name) {
            return Ok(());
        }

        // Build parameter types
        let param_types: Vec<ValType> = func.params.iter().map(|_| ValType::I64).collect();

        // Result type
        let result_types = if func.return_type.is_some() {
            vec![ValType::I64]
        } else {
            vec![] // Unit return
        };

        let type_idx = self.get_or_create_type(param_types.clone(), result_types.clone());

        let func_idx = self.imports.import_count() + self.functions.len() as u32;

        // Record function index
        self.func_map.insert(func.name.name.clone(), func_idx);

        // Create function (body will be compiled later)
        let params_with_names: Vec<(String, ValType)> = func
            .params
            .iter()
            .map(|p| (p.pattern_name().unwrap_or_default(), ValType::I64))
            .collect();

        let is_exported = matches!(func.visibility, Visibility::Public);

        let compiled_func = CompiledFunction::new(
            func.name.name.clone(),
            type_idx,
            func_idx,
            params_with_names,
            result_types,
            is_exported,
        );

        self.functions.push(compiled_func);

        Ok(())
    }

    /// Compile a function.
    fn compile_function(&mut self, func: &Function) -> WasmResult<()> {
        self.compile_function_with_name(func, &func.name.name)
    }

    /// Compile an impl method using its mangled name.
    fn compile_impl_method(&mut self, func: &Function, type_name: &str) -> WasmResult<()> {
        let mangled_name = format!("{}_{}", type_name, func.name.name);
        self.compile_function_with_name(func, &mangled_name)
    }

    /// Compile a function with a specific lookup name.
    fn compile_function_with_name(&mut self, func: &Function, lookup_name: &str) -> WasmResult<()> {
        // Find the function index
        let func_idx = self
            .func_map
            .get(lookup_name)
            .copied()
            .ok_or_else(|| WasmError::internal("function not registered"))?;

        // Find the function in our list
        let fn_list_idx = (func_idx - self.imports.import_count()) as usize;

        // Set as current function
        self.current_fn_idx = Some(fn_list_idx);

        // Mark if this is an async function
        let is_async = func.is_async;

        // For async functions, create a Promise and wrap the body
        if is_async {
            self.compile_async_function_body(func)?;
        } else {
            // Normal function compilation
            if let Some(body) = &func.body {
                // Check if function returns a value
                let returns_value = func.return_type.is_some();

                self.compile_block(body)?;

                // Handle block result based on function return type
                let compiled_func = self.current_function_mut().unwrap();
                if !returns_value {
                    // Void function - drop the block's value
                    compiled_func.push(Instruction::Drop);
                }
                compiled_func.push(Instruction::End);
            } else {
                // No body - just return unit
                let compiled_func = self.current_function_mut().unwrap();
                if !compiled_func.results.is_empty() {
                    compiled_func.push(Instruction::I64Const(0));
                }
                compiled_func.push(Instruction::End);
            }
        }

        // Clear current function
        self.current_fn_idx = None;

        Ok(())
    }

    /// Compile an async function body.
    ///
    /// Async functions wrap their body in a Promise and return immediately.
    /// The actual execution happens when the Promise is awaited.
    fn compile_async_function_body(&mut self, func: &Function) -> WasmResult<()> {
        // Create a new Promise
        let promise_new = self
            .get_func("async_promise_new")
            .ok_or_else(|| WasmError::internal("async_promise_new not found"))?;
        let promise_resolve = self
            .get_func("async_promise_resolve")
            .ok_or_else(|| WasmError::internal("async_promise_resolve not found"))?;

        let compiled_func = self.current_function_mut().unwrap();

        // Create promise: promise_ptr = promise_new()
        compiled_func.push(Instruction::Call(promise_new));
        let promise_ptr = compiled_func.alloc_local("__promise".to_string(), ValType::I32);
        compiled_func.push(Instruction::LocalSet(promise_ptr));

        drop(compiled_func);

        // Compile the body
        if let Some(body) = &func.body {
            self.compile_block(body)?;

            // Resolve the promise with the result
            let compiled_func = self.current_function_mut().unwrap();

            // Stack has result value, resolve promise
            let result_local = compiled_func.alloc_local("__result".to_string(), ValType::I64);
            compiled_func.push(Instruction::LocalSet(result_local));

            // promise_resolve(promise_ptr, result)
            compiled_func.push(Instruction::LocalGet(promise_ptr));
            compiled_func.push(Instruction::LocalGet(result_local));
            compiled_func.push(Instruction::Call(promise_resolve));

            // Return promise pointer (extended to i64)
            compiled_func.push(Instruction::LocalGet(promise_ptr));
            compiled_func.push(Instruction::I64ExtendI32U);
        } else {
            // No body - resolve with unit
            let compiled_func = self.current_function_mut().unwrap();
            compiled_func.push(Instruction::LocalGet(promise_ptr));
            compiled_func.push(Instruction::I64Const(0)); // unit value
            compiled_func.push(Instruction::Call(promise_resolve));

            // Return promise pointer
            compiled_func.push(Instruction::LocalGet(promise_ptr));
            compiled_func.push(Instruction::I64ExtendI32U);
        }

        let compiled_func = self.current_function_mut().unwrap();
        compiled_func.push(Instruction::End);

        Ok(())
    }

    /// Compile a const definition.
    fn compile_const(&mut self, def: &ConstDef) -> WasmResult<()> {
        // Consts are inlined at use sites, but we need to evaluate them
        // For now, add as a global

        // Evaluate constant expression at compile time
        let const_val = self.eval_const_expr(&def.value)?;

        // Add as global (immutable)
        let idx = self.globals.len() as u32;
        self.globals.push((ValType::I64, false, const_val));
        self.global_map.insert(def.name.name.clone(), idx);
        Ok(())
    }

    /// Compile a static definition.
    fn compile_static(&mut self, def: &StaticDef) -> WasmResult<()> {
        // Add as mutable global
        let init_val = self.eval_const_expr(&def.value)?;

        let idx = self.globals.len() as u32;
        self.globals.push((ValType::I64, def.mutable, init_val));
        self.global_map.insert(def.name.name.clone(), idx);

        Ok(())
    }

    /// Evaluate a constant expression at compile time.
    fn eval_const_expr(&self, expr: &crate::ast::Expr) -> WasmResult<i64> {
        use crate::ast::{BinOp, Expr, Literal, NumBase, UnaryOp};

        match expr {
            Expr::Literal(Literal::Int { value, base, .. }) => {
                let radix = match base {
                    NumBase::Decimal => 10,
                    NumBase::Binary => 2,
                    NumBase::Octal => 8,
                    NumBase::Hex => 16,
                    _ => 10,
                };
                let clean: String = value.chars().filter(|c| *c != '_').collect();
                i64::from_str_radix(&clean, radix)
                    .map_err(|_| WasmError::parse(format!("invalid integer: {}", value)))
            }

            Expr::Literal(Literal::Bool(b)) => Ok(if *b { 1 } else { 0 }),

            Expr::Literal(Literal::Null | Literal::Empty) => Ok(0),

            Expr::Unary { op, expr } => {
                let val = self.eval_const_expr(expr)?;
                match op {
                    UnaryOp::Neg => Ok(-val),
                    UnaryOp::Not => Ok(if val == 0 { 1 } else { 0 }),
                    _ => Err(WasmError::not_const()),
                }
            }

            Expr::Binary { left, op, right } => {
                let l = self.eval_const_expr(left)?;
                let r = self.eval_const_expr(right)?;

                match op {
                    BinOp::Add => Ok(l + r),
                    BinOp::Sub => Ok(l - r),
                    BinOp::Mul => Ok(l * r),
                    BinOp::Div => {
                        if r == 0 {
                            Err(WasmError::div_by_zero())
                        } else {
                            Ok(l / r)
                        }
                    }
                    BinOp::Rem => {
                        if r == 0 {
                            Err(WasmError::div_by_zero())
                        } else {
                            Ok(l % r)
                        }
                    }
                    BinOp::BitAnd => Ok(l & r),
                    BinOp::BitOr => Ok(l | r),
                    BinOp::BitXor => Ok(l ^ r),
                    BinOp::Shl => Ok(l << r),
                    BinOp::Shr => Ok(l >> r),
                    _ => Err(WasmError::not_const()),
                }
            }

            Expr::Path(path) => {
                // Look up const
                let name = path
                    .segments
                    .first()
                    .map(|s| s.ident.name.as_str())
                    .unwrap_or("");
                if let Some(&idx) = self.global_map.get(name) {
                    Ok(self.globals[idx as usize].2)
                } else {
                    Err(WasmError::not_const())
                }
            }

            _ => Err(WasmError::not_const()),
        }
    }
}

/// Helper trait for extracting pattern names.
trait ParamExt {
    fn pattern_name(&self) -> Option<String>;
}

impl ParamExt for Param {
    fn pattern_name(&self) -> Option<String> {
        extract_pattern_name(&self.pattern)
    }
}

/// Extract name from a pattern, handling nested patterns like &this or &vary this.
fn extract_pattern_name(pattern: &crate::ast::Pattern) -> Option<String> {
    use crate::ast::Pattern;
    match pattern {
        Pattern::Ident { name, .. } => Some(name.name.clone()),
        // Handle &this and &vary this - extract inner pattern name
        Pattern::Ref { pattern: inner, .. } => extract_pattern_name(inner),
        // Handle ref binding
        Pattern::RefBinding { name, .. } => Some(name.name.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{
        BinOp, Block, CrateConfig, EnumVariant, FieldDef, Ident, Literal, NumBase, StructAttrs,
        TypePath, UnaryOp,
    };
    use crate::span::{Span, Spanned};

    fn make_int(value: i64) -> crate::ast::Expr {
        crate::ast::Expr::Literal(Literal::Int {
            value: value.to_string(),
            base: NumBase::Decimal,
            suffix: None,
        })
    }

    fn make_ident(name: &str) -> Ident {
        Ident {
            name: name.to_string(),
            evidentiality: None,
            affect: None,
            span: Span::new(0, 0),
        }
    }

    fn make_function(name: &str, body: Option<Block>) -> Function {
        Function {
            visibility: Visibility::Public,
            is_async: false,
            is_const: false,
            is_unsafe: false,
            attrs: Default::default(),
            outer_attrs: Vec::new(),
            name: make_ident(name),
            aspect: None,
            generics: None,
            params: vec![],
            return_type: None,
            where_clause: None,
            body,
        }
    }

    #[test]
    fn test_compile_empty_file() {
        let mut compiler = WasmCompiler::new();

        let file = SourceFile {
            attrs: vec![],
            config: CrateConfig::default(),
            items: vec![],
        };

        compiler.compile_file(&file).unwrap();
        assert!(compiler.functions.is_empty());
    }

    #[test]
    fn test_compile_simple_function() {
        let mut compiler = WasmCompiler::new();

        let func = make_function(
            "main",
            Some(Block {
                stmts: vec![],
                expr: Some(Box::new(make_int(42))),
            }),
        );

        let file = SourceFile {
            attrs: vec![],
            config: CrateConfig::default(),
            items: vec![Spanned {
                node: Item::Function(func),
                span: Span::new(0, 0),
            }],
        };

        compiler.compile_file(&file).unwrap();

        assert_eq!(compiler.functions.len(), 1);
        assert!(compiler.functions[0].is_exported);
    }

    #[test]
    fn test_compile_const() {
        let mut compiler = WasmCompiler::new();

        let const_def = ConstDef {
            visibility: Visibility::Public,
            name: make_ident("MAX"),
            ty: crate::ast::TypeExpr::Path(TypePath { segments: vec![] }),
            value: make_int(100),
        };

        let file = SourceFile {
            attrs: vec![],
            config: CrateConfig::default(),
            items: vec![Spanned {
                node: Item::Const(const_def),
                span: Span::new(0, 0),
            }],
        };

        compiler.compile_file(&file).unwrap();

        assert!(compiler.global_map.contains_key("MAX"));
    }

    #[test]
    fn test_eval_const_expr_int() {
        let compiler = WasmCompiler::new();

        let result = compiler.eval_const_expr(&make_int(42)).unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    fn test_eval_const_expr_binary() {
        let compiler = WasmCompiler::new();

        let expr = crate::ast::Expr::Binary {
            left: Box::new(make_int(10)),
            op: crate::ast::BinOp::Add,
            right: Box::new(make_int(5)),
        };

        let result = compiler.eval_const_expr(&expr).unwrap();
        assert_eq!(result, 15);
    }

    #[test]
    fn test_eval_const_expr_unary_neg() {
        let compiler = WasmCompiler::new();

        let expr = crate::ast::Expr::Unary {
            op: crate::ast::UnaryOp::Neg,
            expr: Box::new(make_int(42)),
        };

        let result = compiler.eval_const_expr(&expr).unwrap();
        assert_eq!(result, -42);
    }

    #[test]
    fn test_register_struct() {
        let mut compiler = WasmCompiler::new();

        let def = StructDef {
            visibility: Visibility::Public,
            attrs: StructAttrs::default(),
            name: make_ident("Point"),
            generics: None,
            fields: StructFields::Named(vec![
                FieldDef {
                    visibility: Visibility::Public,
                    name: make_ident("x"),
                    ty: crate::ast::TypeExpr::Path(TypePath { segments: vec![] }),
                    default: None,
                },
                FieldDef {
                    visibility: Visibility::Public,
                    name: make_ident("y"),
                    ty: crate::ast::TypeExpr::Path(TypePath { segments: vec![] }),
                    default: None,
                },
            ]),
            is_translations: false,
        };

        compiler.register_struct(&def).unwrap();

        let layout = compiler.struct_layouts.get("Point").unwrap();
        assert_eq!(layout.size, 16);
        assert_eq!(layout.field_offset("x"), Some(0));
        assert_eq!(layout.field_offset("y"), Some(8));
    }

    #[test]
    fn test_compile_static() {
        let mut compiler = WasmCompiler::new();

        let static_def = StaticDef {
            visibility: Visibility::Public,
            mutable: true,
            name: make_ident("COUNTER"),
            ty: crate::ast::TypeExpr::Path(TypePath { segments: vec![] }),
            value: make_int(0),
        };

        let file = SourceFile {
            attrs: vec![],
            config: CrateConfig::default(),
            items: vec![Spanned {
                node: Item::Static(static_def),
                span: Span::new(0, 0),
            }],
        };

        compiler.compile_file(&file).unwrap();

        assert!(compiler.global_map.contains_key("COUNTER"));
        // Static should be mutable
        let idx = compiler.global_map.get("COUNTER").unwrap();
        assert!(compiler.globals[*idx as usize].1); // mutable flag
    }

    #[test]
    fn test_register_enum() {
        let mut compiler = WasmCompiler::new();

        let def = EnumDef {
            visibility: Visibility::Public,
            name: make_ident("Option"),
            generics: None,
            variants: vec![
                EnumVariant {
                    name: make_ident("None"),
                    fields: StructFields::Unit,
                    discriminant: None,
                },
                EnumVariant {
                    name: make_ident("Some"),
                    fields: StructFields::Tuple(vec![crate::ast::TypeExpr::Path(TypePath {
                        segments: vec![],
                    })]),
                    discriminant: None,
                },
            ],
        };

        compiler.register_enum(&def).unwrap();

        let layout = compiler.enum_layouts.get("Option").unwrap();
        assert_eq!(layout.variant_tag("None"), Some(0));
        assert_eq!(layout.variant_tag("Some"), Some(1));
        // Some has payload, None doesn't
        assert!(layout.variants[0].2.is_none());
        assert!(layout.variants[1].2.is_some());
    }

    #[test]
    fn test_eval_const_expr_bool() {
        let compiler = WasmCompiler::new();

        let true_result = compiler
            .eval_const_expr(&crate::ast::Expr::Literal(Literal::Bool(true)))
            .unwrap();
        assert_eq!(true_result, 1);

        let false_result = compiler
            .eval_const_expr(&crate::ast::Expr::Literal(Literal::Bool(false)))
            .unwrap();
        assert_eq!(false_result, 0);
    }

    #[test]
    fn test_eval_const_expr_null() {
        let compiler = WasmCompiler::new();

        let null_result = compiler
            .eval_const_expr(&crate::ast::Expr::Literal(Literal::Null))
            .unwrap();
        assert_eq!(null_result, 0);

        let empty_result = compiler
            .eval_const_expr(&crate::ast::Expr::Literal(Literal::Empty))
            .unwrap();
        assert_eq!(empty_result, 0);
    }

    #[test]
    fn test_eval_const_expr_bitwise() {
        let compiler = WasmCompiler::new();

        // BitAnd: 0b1100 & 0b1010 = 0b1000 = 8
        let and_expr = crate::ast::Expr::Binary {
            left: Box::new(make_int(0b1100)),
            op: BinOp::BitAnd,
            right: Box::new(make_int(0b1010)),
        };
        assert_eq!(compiler.eval_const_expr(&and_expr).unwrap(), 8);

        // BitOr: 0b1100 | 0b1010 = 0b1110 = 14
        let or_expr = crate::ast::Expr::Binary {
            left: Box::new(make_int(0b1100)),
            op: BinOp::BitOr,
            right: Box::new(make_int(0b1010)),
        };
        assert_eq!(compiler.eval_const_expr(&or_expr).unwrap(), 14);

        // BitXor: 0b1100 ^ 0b1010 = 0b0110 = 6
        let xor_expr = crate::ast::Expr::Binary {
            left: Box::new(make_int(0b1100)),
            op: BinOp::BitXor,
            right: Box::new(make_int(0b1010)),
        };
        assert_eq!(compiler.eval_const_expr(&xor_expr).unwrap(), 6);

        // Shl: 1 << 4 = 16
        let shl_expr = crate::ast::Expr::Binary {
            left: Box::new(make_int(1)),
            op: BinOp::Shl,
            right: Box::new(make_int(4)),
        };
        assert_eq!(compiler.eval_const_expr(&shl_expr).unwrap(), 16);

        // Shr: 32 >> 2 = 8
        let shr_expr = crate::ast::Expr::Binary {
            left: Box::new(make_int(32)),
            op: BinOp::Shr,
            right: Box::new(make_int(2)),
        };
        assert_eq!(compiler.eval_const_expr(&shr_expr).unwrap(), 8);
    }

    #[test]
    fn test_eval_const_expr_div_by_zero() {
        let compiler = WasmCompiler::new();

        let expr = crate::ast::Expr::Binary {
            left: Box::new(make_int(10)),
            op: BinOp::Div,
            right: Box::new(make_int(0)),
        };

        let result = compiler.eval_const_expr(&expr);
        assert!(result.is_err());
    }

    #[test]
    fn test_eval_const_expr_rem_by_zero() {
        let compiler = WasmCompiler::new();

        let expr = crate::ast::Expr::Binary {
            left: Box::new(make_int(10)),
            op: BinOp::Rem,
            right: Box::new(make_int(0)),
        };

        let result = compiler.eval_const_expr(&expr);
        assert!(result.is_err());
    }

    #[test]
    fn test_eval_const_expr_mul_sub() {
        let compiler = WasmCompiler::new();

        // Mul: 6 * 7 = 42
        let mul_expr = crate::ast::Expr::Binary {
            left: Box::new(make_int(6)),
            op: BinOp::Mul,
            right: Box::new(make_int(7)),
        };
        assert_eq!(compiler.eval_const_expr(&mul_expr).unwrap(), 42);

        // Sub: 50 - 8 = 42
        let sub_expr = crate::ast::Expr::Binary {
            left: Box::new(make_int(50)),
            op: BinOp::Sub,
            right: Box::new(make_int(8)),
        };
        assert_eq!(compiler.eval_const_expr(&sub_expr).unwrap(), 42);

        // Div: 84 / 2 = 42
        let div_expr = crate::ast::Expr::Binary {
            left: Box::new(make_int(84)),
            op: BinOp::Div,
            right: Box::new(make_int(2)),
        };
        assert_eq!(compiler.eval_const_expr(&div_expr).unwrap(), 42);

        // Rem: 42 % 5 = 2
        let rem_expr = crate::ast::Expr::Binary {
            left: Box::new(make_int(42)),
            op: BinOp::Rem,
            right: Box::new(make_int(5)),
        };
        assert_eq!(compiler.eval_const_expr(&rem_expr).unwrap(), 2);
    }

    #[test]
    fn test_eval_const_expr_not() {
        let compiler = WasmCompiler::new();

        // Not of 0 = 1
        let not_zero = crate::ast::Expr::Unary {
            op: UnaryOp::Not,
            expr: Box::new(make_int(0)),
        };
        assert_eq!(compiler.eval_const_expr(&not_zero).unwrap(), 1);

        // Not of non-zero = 0
        let not_nonzero = crate::ast::Expr::Unary {
            op: UnaryOp::Not,
            expr: Box::new(make_int(42)),
        };
        assert_eq!(compiler.eval_const_expr(&not_nonzero).unwrap(), 0);
    }

    #[test]
    fn test_eval_const_expr_non_const() {
        let compiler = WasmCompiler::new();

        // A function call is not a constant expression
        let call_expr = crate::ast::Expr::Call {
            func: Box::new(crate::ast::Expr::Path(TypePath {
                segments: vec![crate::ast::PathSegment {
                    ident: make_ident("foo"),
                    generics: None,
                }],
            })),
            args: vec![],
        };

        let result = compiler.eval_const_expr(&call_expr);
        assert!(result.is_err());
    }

    #[test]
    fn test_register_tuple_struct() {
        let mut compiler = WasmCompiler::new();

        let def = StructDef {
            visibility: Visibility::Public,
            attrs: StructAttrs::default(),
            name: make_ident("Color"),
            generics: None,
            fields: StructFields::Tuple(vec![
                crate::ast::TypeExpr::Path(TypePath { segments: vec![] }), // r
                crate::ast::TypeExpr::Path(TypePath { segments: vec![] }), // g
                crate::ast::TypeExpr::Path(TypePath { segments: vec![] }), // b
            ]),
            is_translations: false,
        };

        compiler.register_struct(&def).unwrap();

        let layout = compiler.struct_layouts.get("Color").unwrap();
        assert_eq!(layout.size, 24); // 3 fields * 8 bytes
        assert_eq!(layout.field_offset("_0"), Some(0));
        assert_eq!(layout.field_offset("_1"), Some(8));
        assert_eq!(layout.field_offset("_2"), Some(16));
    }

    #[test]
    fn test_register_unit_struct() {
        let mut compiler = WasmCompiler::new();

        let def = StructDef {
            visibility: Visibility::Public,
            attrs: StructAttrs::default(),
            name: make_ident("Unit"),
            generics: None,
            fields: StructFields::Unit,
            is_translations: false,
        };

        compiler.register_struct(&def).unwrap();

        let layout = compiler.struct_layouts.get("Unit").unwrap();
        assert_eq!(layout.size, 0); // No fields
    }

    #[test]
    fn test_compile_private_function() {
        let mut compiler = WasmCompiler::new();

        let func = Function {
            visibility: Visibility::Private,
            is_async: false,
            is_const: false,
            is_unsafe: false,
            attrs: Default::default(),
            outer_attrs: Vec::new(),
            name: make_ident("helper"),
            aspect: None,
            generics: None,
            params: vec![],
            return_type: None,
            where_clause: None,
            body: Some(Block {
                stmts: vec![],
                expr: Some(Box::new(make_int(0))),
            }),
        };

        let file = SourceFile {
            attrs: vec![],
            config: CrateConfig::default(),
            items: vec![Spanned {
                node: Item::Function(func),
                span: Span::new(0, 0),
            }],
        };

        compiler.compile_file(&file).unwrap();

        assert_eq!(compiler.functions.len(), 1);
        assert!(!compiler.functions[0].is_exported); // Private
    }

    #[test]
    fn test_register_enum_with_named_fields() {
        let mut compiler = WasmCompiler::new();

        let def = EnumDef {
            visibility: Visibility::Public,
            name: make_ident("Shape"),
            generics: None,
            variants: vec![
                EnumVariant {
                    name: make_ident("Circle"),
                    fields: StructFields::Named(vec![FieldDef {
                        visibility: Visibility::Public,
                        name: make_ident("radius"),
                        ty: crate::ast::TypeExpr::Path(TypePath { segments: vec![] }),
                        default: None,
                    }]),
                    discriminant: None,
                },
                EnumVariant {
                    name: make_ident("Rectangle"),
                    fields: StructFields::Named(vec![
                        FieldDef {
                            visibility: Visibility::Public,
                            name: make_ident("width"),
                            ty: crate::ast::TypeExpr::Path(TypePath { segments: vec![] }),
                            default: None,
                        },
                        FieldDef {
                            visibility: Visibility::Public,
                            name: make_ident("height"),
                            ty: crate::ast::TypeExpr::Path(TypePath { segments: vec![] }),
                            default: None,
                        },
                    ]),
                    discriminant: None,
                },
            ],
        };

        compiler.register_enum(&def).unwrap();

        let layout = compiler.enum_layouts.get("Shape").unwrap();
        assert_eq!(layout.variant_tag("Circle"), Some(0));
        assert_eq!(layout.variant_tag("Rectangle"), Some(1));

        // Both variants have payloads
        let circle_payload = layout.variants[0].2.as_ref().unwrap();
        assert_eq!(circle_payload.field_offset("radius"), Some(0));

        let rect_payload = layout.variants[1].2.as_ref().unwrap();
        assert_eq!(rect_payload.field_offset("width"), Some(0));
        assert_eq!(rect_payload.field_offset("height"), Some(8));
    }
}
