//! Top-level item compilation.
//!
//! Compiles Sigil functions, structs, enums, and other top-level items to WASM.

use wasm_encoder::{Instruction, ValType};

use std::path::PathBuf;

use super::error::{WasmError, WasmResult};
use super::types::{CompiledFunction, EnumLayout, StructLayout};
use super::WasmCompiler;
use crate::ast::{
    ConstDef, EnumDef, Function, ImplItem, Item, MacroInvocation, Module, Param, SourceFile,
    StaticDef, StructDef, StructFields, UseDecl, UseTree, Visibility,
};
use crate::parser::Parser;

impl WasmCompiler {
    /// Compile a source file.
    pub fn compile_file(&mut self, file: &SourceFile) -> WasmResult<()> {
        // First pass: process use declarations (for cross-module linking)
        // Including from nested modules
        self.collect_use_declarations(&file.items)?;

        // Second pass: collect type definitions (structs, enums)
        // Including from nested modules
        self.collect_all_type_defs(&file.items)?;

        // Third pass: pre-scan function bodies to add external imports
        // This must happen BEFORE collecting function signatures to preserve indices
        // Including from nested modules
        self.prescan_all_functions(&file.items)?;

        // Fourth pass: collect function signatures
        // Including from nested modules
        self.collect_all_function_sigs(&file.items)?;

        // Fifth pass: compile function bodies
        for item in &file.items {
            self.compile_item(&item.node)?;
        }

        Ok(())
    }

    /// Recursively collect use declarations from items and nested modules.
    fn collect_use_declarations(&mut self, items: &[crate::span::Spanned<Item>]) -> WasmResult<()> {
        for item in items {
            match &item.node {
                Item::Use(use_decl) => {
                    self.process_use_decl(use_decl)?;
                }
                Item::Module(module) => {
                    if let Some(nested_items) = &module.items {
                        self.module_path.push(module.name.name.clone());
                        self.collect_use_declarations(nested_items)?;
                        self.module_path.pop();
                    } else {
                        // File-based module - load and process
                        let module_name = module.name.name.clone();
                        // Load the module file FIRST, using current source_dir
                        let items = self.load_module_file(&module_name)?;
                        // THEN update source_dir for nested module resolution
                        let new_source_dir = self.get_module_source_dir(&module_name)
                            .unwrap_or_else(|| self.source_dir.clone());
                        let old_source_dir = std::mem::replace(&mut self.source_dir, new_source_dir);
                        self.module_path.push(module_name);
                        self.collect_use_declarations(&items)?;
                        self.module_path.pop();
                        self.source_dir = old_source_dir;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Recursively collect type definitions from items and nested modules.
    fn collect_all_type_defs(&mut self, items: &[crate::span::Spanned<Item>]) -> WasmResult<()> {
        for item in items {
            match &item.node {
                Item::Module(module) => {
                    if let Some(nested_items) = &module.items {
                        self.module_path.push(module.name.name.clone());
                        self.collect_all_type_defs(nested_items)?;
                        self.module_path.pop();
                    } else {
                        // File-based module - load and process
                        let module_name = module.name.name.clone();
                        // Load the module file FIRST, using current source_dir
                        let items = self.load_module_file(&module_name)?;
                        // THEN update source_dir for nested module resolution
                        let new_source_dir = self.get_module_source_dir(&module_name)
                            .unwrap_or_else(|| self.source_dir.clone());
                        let old_source_dir = std::mem::replace(&mut self.source_dir, new_source_dir);
                        self.module_path.push(module_name);
                        self.collect_all_type_defs(&items)?;
                        self.module_path.pop();
                        self.source_dir = old_source_dir;
                    }
                }
                _ => {
                    self.collect_type_def(&item.node)?;
                }
            }
        }
        Ok(())
    }

    /// Recursively pre-scan all function bodies for external calls.
    fn prescan_all_functions(&mut self, items: &[crate::span::Spanned<Item>]) -> WasmResult<()> {
        for item in items {
            match &item.node {
                Item::Function(func) => {
                    if let Some(body) = &func.body {
                        self.scan_for_external_calls(body)?;
                    }
                }
                Item::Module(module) => {
                    if let Some(nested_items) = &module.items {
                        self.module_path.push(module.name.name.clone());
                        self.prescan_all_functions(nested_items)?;
                        self.module_path.pop();
                    } else {
                        // File-based module - load and process
                        let module_name = module.name.name.clone();
                        // Load the module file FIRST, using current source_dir
                        let items = self.load_module_file(&module_name)?;
                        // THEN update source_dir for nested module resolution
                        let new_source_dir = self.get_module_source_dir(&module_name)
                            .unwrap_or_else(|| self.source_dir.clone());
                        let old_source_dir = std::mem::replace(&mut self.source_dir, new_source_dir);
                        self.module_path.push(module_name);
                        self.prescan_all_functions(&items)?;
                        self.module_path.pop();
                        self.source_dir = old_source_dir;
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Recursively collect function signatures from items and nested modules.
    fn collect_all_function_sigs(&mut self, items: &[crate::span::Spanned<Item>]) -> WasmResult<()> {
        for item in items {
            match &item.node {
                Item::Module(module) => {
                    if let Some(nested_items) = &module.items {
                        self.module_path.push(module.name.name.clone());
                        self.collect_all_function_sigs(nested_items)?;
                        self.module_path.pop();
                    } else {
                        // File-based module - load and process
                        self.load_and_collect_module_sigs(module)?;
                    }
                }
                _ => {
                    self.collect_function_sig(&item.node)?;
                }
            }
        }
        Ok(())
    }

    /// Load a file-based module and collect its function signatures.
    fn load_and_collect_module_sigs(&mut self, module: &Module) -> WasmResult<()> {
        let module_name = &module.name.name;

        // Load the module file FIRST, using current source_dir
        let items = self.load_module_file(module_name)?;

        // THEN update source_dir for nested module resolution
        let new_source_dir = self.get_module_source_dir(module_name)
            .unwrap_or_else(|| self.source_dir.clone());
        let old_source_dir = std::mem::replace(&mut self.source_dir, new_source_dir);

        self.module_path.push(module_name.clone());
        self.collect_all_function_sigs(&items)?;
        self.module_path.pop();

        self.source_dir = old_source_dir;
        Ok(())
    }

    /// Get the directory that should become source_dir for a module.
    /// For foo.sigil, returns the current source_dir.
    /// For foo/mod.sigil, returns the foo/ directory.
    fn get_module_source_dir(&self, module_name: &str) -> Option<std::path::PathBuf> {
        if self.source_dir.as_os_str().is_empty() {
            return None;
        }

        // Try foo.sigil first
        let file_path = self.source_dir.join(format!("{}.sigil", module_name));
        if file_path.exists() {
            // File-style module: source_dir stays the same
            return Some(self.source_dir.clone());
        }

        // Then try foo/mod.sigil
        let dir_path = self.source_dir.join(module_name).join("mod.sigil");
        if dir_path.exists() {
            // Directory-style module: source_dir becomes the module's directory
            return Some(self.source_dir.join(module_name));
        }

        None
    }

    /// Load a file-based module and return its parsed items.
    /// Handles both `foo.sigil` and `foo/mod.sigil` patterns.
    /// Uses caching to avoid re-parsing the same file multiple times.
    fn load_module_file(&mut self, module_name: &str) -> WasmResult<Vec<crate::span::Spanned<Item>>> {
        use std::fs;

        // If no source_dir is set, we can't resolve file modules
        if self.source_dir.as_os_str().is_empty() {
            return Err(WasmError::io(format!(
                "cannot resolve module '{}': no source directory set (use compile_from_path)",
                module_name
            )));
        }

        // Try foo.sigil first
        let file_path = self.source_dir.join(format!("{}.sigil", module_name));

        // Then try foo/mod.sigil
        let dir_path = self.source_dir.join(module_name).join("mod.sigil");

        let path = if file_path.exists() {
            file_path
        } else if dir_path.exists() {
            dir_path
        } else {
            return Err(WasmError::module_not_found(
                module_name,
                &[file_path.display().to_string(), dir_path.display().to_string()],
            ));
        };

        // Get canonical path for caching
        let canonical = path.canonicalize()
            .map_err(|e| WasmError::io(format!("cannot resolve path {}: {}", path.display(), e)))?;

        // Check cache first
        if let Some(items) = self.module_cache.get(&canonical) {
            return Ok(items.clone());
        }

        // Track this module to detect circular imports during initial loading
        if self.loaded_modules.contains(&canonical) {
            // Already loaded - return from cache (should have been cached)
            return self.module_cache.get(&canonical)
                .cloned()
                .ok_or_else(|| WasmError::internal("module marked as loaded but not cached"));
        }
        self.loaded_modules.insert(canonical.clone());

        // Read and parse the module file
        let source = fs::read_to_string(&path)
            .map_err(|e| WasmError::io(format!("cannot read {}: {}", path.display(), e)))?;

        let mut parser = Parser::new(&source);
        let file = parser.parse_file()
            .map_err(|e| WasmError::parse(format!("in {}: {}", path.display(), e)))?;

        // Cache the parsed items
        self.module_cache.insert(canonical, file.items.clone());

        Ok(file.items)
    }

    /// Store loaded module items for later compilation.
    /// Returns a reference to the stored items.
    fn get_or_load_module_items(&mut self, module_name: &str) -> WasmResult<Vec<crate::span::Spanned<Item>>> {
        // Note: In a more sophisticated implementation, we'd cache these.
        // For now, we reload each time (collection vs compilation phases).
        self.load_module_file(module_name)
    }

    /// Scan a block for external function calls and add them as WASM imports.
    fn scan_for_external_calls(&mut self, block: &crate::ast::Block) -> WasmResult<()> {
        use crate::ast::{Expr, Stmt};

        for stmt in &block.stmts {
            match stmt {
                Stmt::Let { init: Some(expr), .. }
                | Stmt::Expr(expr)
                | Stmt::Semi(expr) => {
                    self.scan_expr_for_external_calls(expr)?;
                }
                Stmt::LetElse { init, else_branch, .. } => {
                    self.scan_expr_for_external_calls(init)?;
                    self.scan_expr_for_external_calls(else_branch)?;
                }
                _ => {}
            }
        }

        if let Some(expr) = &block.expr {
            self.scan_expr_for_external_calls(expr)?;
        }

        Ok(())
    }

    /// Recursively scan an expression for external function calls.
    fn scan_expr_for_external_calls(&mut self, expr: &crate::ast::Expr) -> WasmResult<()> {
        use crate::ast::Expr;

        match expr {
            Expr::Call { func, args } => {
                // Check if this is a call to an external import
                if let Expr::Path(path) = func.as_ref() {
                    let simple_name = path.segments.first()
                        .map(|s| s.ident.name.as_str())
                        .unwrap_or("");

                    // If it's in external_imports, add the WASM import now
                    if let Some((module_name, _)) = self.external_imports.get(simple_name).cloned() {
                        self.get_or_add_external_import(&module_name, simple_name, args.len());
                    }
                }

                // Also scan the function expression and arguments
                self.scan_expr_for_external_calls(func)?;
                for arg in args {
                    self.scan_expr_for_external_calls(arg)?;
                }
            }
            Expr::Binary { left, right, .. } => {
                self.scan_expr_for_external_calls(left)?;
                self.scan_expr_for_external_calls(right)?;
            }
            Expr::Unary { expr, .. } => {
                self.scan_expr_for_external_calls(expr)?;
            }
            Expr::If { condition, then_branch, else_branch } => {
                self.scan_expr_for_external_calls(condition)?;
                self.scan_for_external_calls(then_branch)?;
                if let Some(else_expr) = else_branch {
                    self.scan_expr_for_external_calls(else_expr)?;
                }
            }
            Expr::Block(block) => {
                self.scan_for_external_calls(block)?;
            }
            Expr::Match { expr, arms } => {
                self.scan_expr_for_external_calls(expr)?;
                for arm in arms {
                    if let Some(guard) = &arm.guard {
                        self.scan_expr_for_external_calls(guard)?;
                    }
                    self.scan_expr_for_external_calls(&arm.body)?;
                }
            }
            Expr::While { condition, body, .. } => {
                self.scan_expr_for_external_calls(condition)?;
                self.scan_for_external_calls(body)?;
            }
            Expr::Loop { body, .. } => {
                self.scan_for_external_calls(body)?;
            }
            Expr::For { iter, body, .. } => {
                self.scan_expr_for_external_calls(iter)?;
                self.scan_for_external_calls(body)?;
            }
            Expr::Closure { body, .. } => {
                self.scan_expr_for_external_calls(body)?;
            }
            Expr::Return(Some(expr)) | Expr::Await { expr, .. } => {
                self.scan_expr_for_external_calls(expr)?;
            }
            Expr::Tuple(exprs) | Expr::Array(exprs) => {
                for e in exprs {
                    self.scan_expr_for_external_calls(e)?;
                }
            }
            Expr::Index { expr, index } => {
                self.scan_expr_for_external_calls(expr)?;
                self.scan_expr_for_external_calls(index)?;
            }
            Expr::Field { expr, .. } | Expr::Try(expr) => {
                self.scan_expr_for_external_calls(expr)?;
            }
            Expr::MethodCall { receiver, args, .. } => {
                self.scan_expr_for_external_calls(receiver)?;
                for arg in args {
                    self.scan_expr_for_external_calls(arg)?;
                }
            }
            Expr::Struct { fields, rest, .. } => {
                for field in fields {
                    if let Some(value) = &field.value {
                        self.scan_expr_for_external_calls(value)?;
                    }
                }
                if let Some(rest_expr) = rest {
                    self.scan_expr_for_external_calls(rest_expr)?;
                }
            }
            Expr::Range { start, end, .. } => {
                if let Some(s) = start {
                    self.scan_expr_for_external_calls(s)?;
                }
                if let Some(e) = end {
                    self.scan_expr_for_external_calls(e)?;
                }
            }
            Expr::Assign { target, value } => {
                self.scan_expr_for_external_calls(target)?;
                self.scan_expr_for_external_calls(value)?;
            }
            Expr::Cast { expr, .. } | Expr::AddrOf { expr, .. } | Expr::Deref(expr) => {
                self.scan_expr_for_external_calls(expr)?;
            }
            Expr::Pipe { expr, .. } => {
                self.scan_expr_for_external_calls(expr)?;
            }
            // Terminal expressions - no recursion needed
            Expr::Literal(_)
            | Expr::Path(_)
            | Expr::Break { .. }
            | Expr::Continue { .. }
            | Expr::Return(None) => {}
            // Other expressions - skip for now
            _ => {}
        }

        Ok(())
    }

    /// Process a use declaration to register external module imports.
    fn process_use_decl(&mut self, use_decl: &UseDecl) -> WasmResult<()> {
        self.process_use_tree(&use_decl.tree, &[])
    }

    /// Recursively process a use tree to extract module imports.
    fn process_use_tree(&mut self, tree: &UseTree, prefix: &[String]) -> WasmResult<()> {
        match tree {
            UseTree::Path { prefix: path_prefix, suffix } => {
                let mut new_prefix = prefix.to_vec();
                new_prefix.push(path_prefix.name.clone());
                self.process_use_tree(suffix, &new_prefix)
            }
            UseTree::Name(name) => {
                if !prefix.is_empty() {
                    // External import: use foo::bar::Baz
                    // module_name = first segment (e.g., "foo")
                    // qualified_name = full path (e.g., "foo::bar::Baz")
                    let module_name = prefix[0].clone();
                    let mut full_path = prefix.to_vec();
                    full_path.push(name.name.clone());
                    let qualified_name = full_path.join("::");
                    let simple_name = name.name.clone();

                    // Register the external import
                    self.external_imports.insert(
                        simple_name,
                        (module_name, qualified_name),
                    );
                }
                Ok(())
            }
            UseTree::Rename { name, alias } => {
                if !prefix.is_empty() {
                    let module_name = prefix[0].clone();
                    let mut full_path = prefix.to_vec();
                    full_path.push(name.name.clone());
                    let qualified_name = full_path.join("::");
                    let alias_name = alias.name.clone();

                    self.external_imports.insert(
                        alias_name,
                        (module_name, qualified_name),
                    );
                }
                Ok(())
            }
            UseTree::Glob => {
                // Glob imports (use foo::*) not supported for WASM linking
                // Would require parsing the external module
                Ok(())
            }
            UseTree::Group(trees) => {
                for subtree in trees {
                    self.process_use_tree(subtree, prefix)?;
                }
                Ok(())
            }
        }
    }

    /// Collect type definition from an item.
    fn collect_type_def(&mut self, item: &Item) -> WasmResult<()> {
        match item {
            Item::Struct(def) => self.register_struct(def),
            Item::Enum(def) => self.register_enum(def),
            Item::TypeAlias(alias) => {
                // Handle type aliases with inline struct/enum definitions
                // e.g., `type VNode = enum { Text(String), Element(VElement) }`
                self.register_type_alias(alias)
            }
            _ => Ok(()),
        }
    }

    /// Register a type alias, extracting inline struct/enum definitions.
    fn register_type_alias(&mut self, alias: &crate::ast::TypeAlias) -> WasmResult<()> {
        use crate::ast::TypeExpr;

        match &alias.ty {
            TypeExpr::InlineEnum { variants } => {
                // Create synthetic EnumDef
                let def = crate::ast::EnumDef {
                    doc_comments: Vec::new(),
                    outer_attrs: Vec::new(),
                    visibility: alias.visibility.clone(),
                    name: alias.name.clone(),
                    generics: alias.generics.clone(),
                    variants: variants.clone(),
                };
                self.register_enum(&def)
            }
            TypeExpr::InlineStruct { fields } => {
                // Create synthetic StructDef
                let def = crate::ast::StructDef {
                    doc_comments: Vec::new(),
                    visibility: alias.visibility.clone(),
                    attrs: crate::ast::StructAttrs::default(),
                    name: alias.name.clone(),
                    generics: alias.generics.clone(),
                    fields: crate::ast::StructFields::Named(fields.clone()),
                };
                self.register_struct(&def)
            }
            _ => Ok(()), // Regular type aliases don't need special handling
        }
    }

    /// Collect function signature from an item.
    fn collect_function_sig(&mut self, item: &Item) -> WasmResult<()> {
        match item {
            Item::Function(func) => {
                self.register_function_sig(func)?;
            }
            Item::Impl(impl_block) => {
                // Register impl methods with qualified names like Type::method
                let type_name = self.type_path_to_string(&impl_block.self_ty);
                self.module_path.push(type_name);
                for impl_item in &impl_block.items {
                    if let crate::ast::ImplItem::Function(func) = impl_item {
                        self.register_function_sig(func)?;
                    }
                }
                self.module_path.pop();
            }
            Item::ExternBlock(extern_block) => {
                // Register extern functions as WASM imports
                self.register_extern_block(extern_block)?;
            }
            Item::Actor(actor) => {
                // Register actor handlers and methods with qualified names
                let actor_name = actor.name.name.clone();
                self.module_path.push(actor_name.clone());

                // Register message handlers as ActorName::on_MessageName
                for handler in &actor.handlers {
                    let handler_name = format!("on_{}", handler.message.name);
                    self.register_handler_sig(&handler_name, handler)?;
                }

                // Register methods as ActorName::method_name
                for method in &actor.methods {
                    self.register_function_sig(method)?;
                }

                self.module_path.pop();
            }
            _ => {}
        }
        Ok(())
    }

    /// Register an extern block's functions as WASM imports.
    fn register_extern_block(&mut self, extern_block: &crate::ast::ExternBlock) -> WasmResult<()> {
        use crate::ast::ExternItem;

        // The ABI determines the import module name (e.g., "js" -> "env" or "js")
        let module_name = match extern_block.abi.as_str() {
            "js" | "JavaScript" => "env", // Standard WASM import module for JS
            "C" | "system" => "env",
            other => other,
        };

        for item in &extern_block.items {
            match item {
                ExternItem::Function(func) => {
                    self.register_extern_function(module_name, func)?;
                }
                ExternItem::Type(ty) => {
                    // Extern types are opaque - just track the name
                    // They're typically used as handles (represented as i64)
                    self.extern_types.insert(ty.name.name.clone());
                }
                ExternItem::Static(_) => {
                    // Extern statics could be global imports - not yet supported
                }
            }
        }
        Ok(())
    }

    /// Register a single extern function as a WASM import.
    fn register_extern_function(
        &mut self,
        module_name: &str,
        func: &crate::ast::ExternFunction,
    ) -> WasmResult<()> {
        use wasm_encoder::ValType;

        let func_name = &func.name.name;

        // Build parameter types - all extern params are i64 (handles/values)
        // Special case: &str params need i32 (pointer) + i32 (length) in some ABIs
        let param_count = func.params.len();
        let params: Vec<ValType> = vec![ValType::I64; param_count];

        // Return type - i64 for values, empty for void
        let results: Vec<ValType> = if func.return_type.is_some() {
            vec![ValType::I64]
        } else {
            vec![]
        };

        // Check if this is a method (first param is &self or &mut self)
        let is_method = func.params.first().map_or(false, |p| {
            match &p.pattern {
                crate::ast::Pattern::Ident { name, .. } => {
                    name.name == "this" || name.name == "self"
                }
                _ => false,
            }
        });

        // Register the import
        let import_idx = self.imports.add_import(module_name, func_name, params, results);

        // Also register with qualified name for method resolution
        // e.g., "Storage::get_item" for extern method
        if is_method {
            // Extract type name from first param (e.g., "&Storage" -> "Storage")
            if let Some(first_param) = func.params.first() {
                if let Some(type_name) = self.extract_type_name_from_param(first_param) {
                    let qualified_name = format!("{}::{}", type_name, func_name);
                    self.func_map.insert(qualified_name, import_idx);
                }
            }
        }

        // Register by simple name too
        self.func_map.insert(func_name.clone(), import_idx);

        Ok(())
    }

    /// Extract type name from a parameter (e.g., "&Storage" -> "Storage").
    fn extract_type_name_from_param(&self, param: &crate::ast::Param) -> Option<String> {
        use crate::ast::TypeExpr;

        let ty = &param.ty;
        match ty {
            TypeExpr::Reference { inner, .. } => {
                self.extract_type_name_from_type(inner)
            }
            _ => self.extract_type_name_from_type(ty),
        }
    }

    /// Extract type name from a type expression.
    fn extract_type_name_from_type(&self, ty: &crate::ast::TypeExpr) -> Option<String> {
        use crate::ast::TypeExpr;

        match ty {
            TypeExpr::Path(path) => {
                path.segments.last().map(|s| s.ident.name.clone())
            }
            TypeExpr::Reference { inner, .. } => {
                self.extract_type_name_from_type(inner)
            }
            _ => None,
        }
    }

    /// Convert a type path to a string for use in qualified names.
    fn type_path_to_string(&self, ty: &crate::ast::TypeExpr) -> String {
        match ty {
            crate::ast::TypeExpr::Path(path) => {
                path.segments.iter()
                    .map(|s| s.ident.name.as_str())
                    .collect::<Vec<_>>()
                    .join("::")
            }
            _ => "Unknown".to_string(),
        }
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
                // Push type name onto module path to match how we registered the functions
                let type_name = self.type_path_to_string(&impl_block.self_ty);
                self.module_path.push(type_name);
                for item in &impl_block.items {
                    if let ImplItem::Function(func) = item {
                        self.compile_function(func)?;
                    }
                }
                self.module_path.pop();
                Ok(())
            }
            Item::TypeAlias(_) => Ok(()), // Type aliases are compile-time only
            Item::Module(module) => self.compile_module(module),
            Item::Use(_) => Ok(()), // Use declarations are resolved during parsing
            Item::Actor(actor) => self.compile_actor(actor),
            Item::ExternBlock(_) => Ok(()), // Extern functions are imports
            Item::Macro(_) => Ok(()), // Macro definitions are compile-time only
            Item::MacroInvocation(mac) => self.compile_macro_invocation(mac),
            Item::Plurality(_) => Err(WasmError::unsupported("plurality items")),
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

        // Register with both simple and qualified names
        self.struct_layouts.insert(def.name.name.clone(), layout.clone());
        let qualified_name = self.qualify_name(&def.name.name);
        if qualified_name != def.name.name {
            self.struct_layouts.insert(qualified_name, layout);
        }
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

        // Register with both simple and qualified names
        self.enum_layouts.insert(def.name.name.clone(), layout.clone());
        let qualified_name = self.qualify_name(&def.name.name);
        if qualified_name != def.name.name {
            self.enum_layouts.insert(qualified_name, layout);
        }
        Ok(())
    }

    /// Register a function signature (without compiling body).
    fn register_function_sig(&mut self, func: &Function) -> WasmResult<()> {
        // Build qualified name
        let qualified_name = self.qualify_name(&func.name.name);

        // Skip if already registered
        if self.func_map.contains_key(&qualified_name) {
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

        // Record function index with both qualified and simple names
        self.func_map.insert(qualified_name.clone(), func_idx);
        // Also register simple name for backwards compatibility
        if !self.module_path.is_empty() {
            self.func_map.insert(func.name.name.clone(), func_idx);

            // For impl methods, also register with short Type::method format
            // This allows method resolution when the type is known without full path
            // e.g., VNode::attr in addition to qliphoth::core::vdom::VNode::attr
            if let Some(type_name) = self.module_path.last() {
                let short_qualified = format!("{}::{}", type_name, func.name.name);
                self.func_map.insert(short_qualified, func_idx);
            }
        }

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

    /// Register a message handler signature (for actors).
    fn register_handler_sig(&mut self, handler_name: &str, handler: &crate::ast::MessageHandler) -> WasmResult<()> {
        // Build qualified name
        let qualified_name = self.qualify_name(handler_name);

        // Skip if already registered
        if self.func_map.contains_key(&qualified_name) {
            return Ok(());
        }

        // Handler has implicit self parameter + explicit params
        // For WASM, we pass actor state as globals, so params are just the message params
        let param_types: Vec<ValType> = handler.params.iter().map(|_| ValType::I64).collect();

        // Result type
        let result_types = if handler.return_type.is_some() {
            vec![ValType::I64]
        } else {
            vec![] // Unit return
        };

        let type_idx = self.get_or_create_type(param_types.clone(), result_types.clone());
        let func_idx = self.imports.import_count() + self.functions.len() as u32;

        // Record function index with both qualified and simple names
        self.func_map.insert(qualified_name.clone(), func_idx);
        self.func_map.insert(handler_name.to_string(), func_idx);

        // Also register short qualified name (ActorName::on_Message)
        if let Some(actor_name) = self.module_path.last() {
            let short_qualified = format!("{}::{}", actor_name, handler_name);
            self.func_map.insert(short_qualified, func_idx);
        }

        // Create function placeholder
        let params_with_names: Vec<(String, ValType)> = handler
            .params
            .iter()
            .map(|p| (p.pattern_name().unwrap_or_default(), ValType::I64))
            .collect();

        let compiled_func = CompiledFunction::new(
            handler_name.to_string(),
            type_idx,
            func_idx,
            params_with_names,
            result_types,
            false, // Handlers are not exported directly
        );

        self.functions.push(compiled_func);

        Ok(())
    }

    /// Compile a function.
    fn compile_function(&mut self, func: &Function) -> WasmResult<()> {
        // Find the function index (try qualified name first, then simple name)
        let qualified_name = self.qualify_name(&func.name.name);
        let func_idx = self
            .func_map
            .get(&qualified_name)
            .or_else(|| self.func_map.get(&func.name.name))
            .copied()
            .ok_or_else(|| WasmError::internal(format!(
                "function not registered: '{}' (qualified: '{}')",
                func.name.name, qualified_name
            )))?;

        // Find the function in our list by matching func_idx
        // NOTE: We can't use (func_idx - import_count) because import_count may have
        // changed since registration due to dynamic import additions during compilation
        let fn_list_idx = self.functions
            .iter()
            .position(|f| f.func_idx == func_idx)
            .ok_or_else(|| WasmError::internal(format!(
                "function not found in list: func_idx={}, qualified='{}'",
                func_idx, qualified_name
            )))?;

        // Set as current function
        self.current_fn_idx = Some(fn_list_idx);

        // Track function in source map (if debug info enabled)
        if let Some(ref mut source_map) = self.source_map {
            source_map.begin_function(&func.name.name, func.name.span);
        }

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

        // End source map function tracking
        if let Some(ref mut source_map) = self.source_map {
            source_map.end_function();
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
        use crate::ast::{Expr, Literal};

        // Consts are inlined at use sites, but we need to evaluate them
        // For now, add as a global

        // Check if this is a string constant (plain, raw, or multi-line)
        let string_content = match &def.value {
            Expr::Literal(Literal::String(s)) => Some(s.as_str()),
            Expr::Literal(Literal::RawString(s)) => Some(s.as_str()),
            Expr::Literal(Literal::MultiLineString(s)) => Some(s.as_str()),
            _ => None,
        };
        if let Some(s) = string_content {
            // String constants are stored in the data segment
            // The global holds the offset into the data segment
            let offset = self.add_string(s);
            let idx = self.globals.len() as u32;
            self.globals.push((ValType::I64, false, offset as i64));
            // Register with both simple and qualified names
            self.global_map.insert(def.name.name.clone(), idx);
            let qualified_name = self.qualify_name(&def.name.name);
            if qualified_name != def.name.name {
                self.global_map.insert(qualified_name.clone(), idx);
            }
            // Also track this as a string constant for proper access
            self.string_consts.insert(def.name.name.clone(), offset);
            self.string_consts.insert(qualified_name, offset);
            return Ok(());
        }

        // Check for reference to string literal: &"string" or &str literal
        if let Expr::AddrOf { expr, .. } = &def.value {
            let string_content = match expr.as_ref() {
                Expr::Literal(Literal::String(s)) => Some(s.as_str()),
                Expr::Literal(Literal::RawString(s)) => Some(s.as_str()),
                Expr::Literal(Literal::MultiLineString(s)) => Some(s.as_str()),
                _ => None,
            };
            if let Some(s) = string_content {
                let offset = self.add_string(s);
                let idx = self.globals.len() as u32;
                self.globals.push((ValType::I64, false, offset as i64));
                // Register with both simple and qualified names
                self.global_map.insert(def.name.name.clone(), idx);
                let qualified_name = self.qualify_name(&def.name.name);
                if qualified_name != def.name.name {
                    self.global_map.insert(qualified_name.clone(), idx);
                }
                self.string_consts.insert(def.name.name.clone(), offset);
                self.string_consts.insert(qualified_name, offset);
                return Ok(());
            }
        }

        // Evaluate constant expression at compile time
        let const_val = self.eval_const_expr(&def.value)?;

        // Add as global (immutable)
        let idx = self.globals.len() as u32;
        self.globals.push((ValType::I64, false, const_val));
        // Register with both simple and qualified names
        self.global_map.insert(def.name.name.clone(), idx);
        let qualified_name = self.qualify_name(&def.name.name);
        if qualified_name != def.name.name {
            self.global_map.insert(qualified_name, idx);
        }
        Ok(())
    }

    /// Compile a static definition.
    fn compile_static(&mut self, def: &StaticDef) -> WasmResult<()> {
        let idx = self.globals.len() as u32;

        // Try to evaluate as constant expression first
        match self.eval_const_expr(&def.value) {
            Ok(init_val) => {
                // Constant initializer - add directly
                self.globals.push((ValType::I64, def.mutable, init_val));
            }
            Err(_) => {
                // Non-constant initializer - defer to __wasm_start
                // Initialize to 0 for now, actual init happens at runtime
                self.globals.push((ValType::I64, true, 0)); // Must be mutable for deferred init
                self.deferred_static_inits.push((idx, def.value.clone()));
            }
        }

        // Register with both simple and qualified names
        self.global_map.insert(def.name.name.clone(), idx);
        let qualified_name = self.qualify_name(&def.name.name);
        if qualified_name != def.name.name {
            self.global_map.insert(qualified_name, idx);
        }
        Ok(())
    }

    /// Compile a macro invocation at item level.
    /// Handles `thread_local! { ... }` by treating it as a static declaration
    /// (WASM is single-threaded, so thread-local storage is just a global).
    fn compile_macro_invocation(&mut self, mac: &MacroInvocation) -> WasmResult<()> {
        // Get the macro name from the path
        let macro_name = mac
            .path
            .segments
            .last()
            .map(|s| s.ident.name.as_str())
            .unwrap_or("");

        match macro_name {
            "thread_local" => {
                // Parse the tokens to extract static definitions
                // tokens looks like: "pub static RUNTIME: RefCell[Runtime] = RefCell·new(Runtime·new());"
                let tokens = &mac.tokens;

                // Create a synthetic source with a static declaration
                let source = tokens.trim().to_string();

                // Use the parser to parse this as an item
                let mut parser = Parser::new(&source);
                match parser.parse_file() {
                    Ok(file) => {
                        // Process each item in the macro body
                        for item in &file.items {
                            match &item.node {
                                Item::Static(def) => {
                                    self.compile_static(def)?;
                                }
                                _ => {
                                    // thread_local! can only contain static definitions
                                    return Err(WasmError::internal(format!(
                                        "thread_local! macro can only contain static definitions, got: {:?}",
                                        item.node
                                    )));
                                }
                            }
                        }
                        Ok(())
                    }
                    Err(e) => Err(WasmError::parse(format!(
                        "failed to parse thread_local! body: {}",
                        e
                    ))),
                }
            }
            _ => {
                // Unknown macro - return unsupported error
                Err(WasmError::unsupported(&format!(
                    "macro invocation: {}!",
                    macro_name
                )))
            }
        }
    }

    /// Compile a module (scroll declaration).
    /// Note: Type definitions and function signatures are already collected
    /// during the recursive collection passes. This only compiles the bodies.
    fn compile_module(&mut self, module: &Module) -> WasmResult<()> {
        // Push module name onto the path
        self.module_path.push(module.name.name.clone());

        if let Some(items) = &module.items {
            // Inline module - compile the provided items
            for item in items {
                self.compile_item(&item.node)?;
            }
        } else {
            // File-based module - load and compile
            let module_name = module.name.name.clone();

            // Load the module file FIRST, using current source_dir
            let items = self.get_or_load_module_items(&module_name)?;

            // THEN update source_dir for nested module resolution
            let new_source_dir = self.get_module_source_dir(&module_name)
                .unwrap_or_else(|| self.source_dir.clone());
            let old_source_dir = std::mem::replace(&mut self.source_dir, new_source_dir);

            for item in &items {
                self.compile_item(&item.node)?;
            }

            self.source_dir = old_source_dir;
        }

        // Pop module name from path
        self.module_path.pop();

        Ok(())
    }

    /// Compile an actor definition.
    ///
    /// Actors are compiled as:
    /// - State fields -> WASM globals (named ActorName_field)
    /// - Message handlers -> functions (named ActorName::on_Message)
    /// - Methods -> functions (named ActorName::method)
    fn compile_actor(&mut self, actor: &crate::ast::ActorDef) -> WasmResult<()> {
        let actor_name = actor.name.name.clone();

        // 1. Register state fields as globals
        for field in &actor.state {
            let global_name = format!("{}_{}", actor_name, field.name.name);
            let qualified_global = self.qualify_name(&global_name);

            // Get initial value (default to 0 if not provided)
            let init_val = if let Some(init_expr) = &field.default {
                self.eval_const_expr(init_expr).unwrap_or(0)
            } else {
                0
            };

            let idx = self.globals.len() as u32;
            self.globals.push((ValType::I64, true, init_val)); // Actor state is mutable

            // Register with both simple and qualified names
            self.global_map.insert(global_name.clone(), idx);
            if qualified_global != global_name {
                self.global_map.insert(qualified_global, idx);
            }
        }

        // 2. Push actor name onto module path for qualified method names
        self.module_path.push(actor_name.clone());

        // Track the current actor context for self resolution
        let prev_actor = self.current_actor.take();
        self.current_actor = Some(actor_name.clone());

        // 3. Compile message handlers
        for handler in &actor.handlers {
            let handler_name = format!("on_{}", handler.message.name);
            self.compile_handler(&actor_name, &handler_name, handler)?;
        }

        // 4. Compile methods
        for method in &actor.methods {
            self.compile_actor_method(&actor_name, method)?;
        }

        // Restore previous actor context
        self.current_actor = prev_actor;

        // Pop actor name from path
        self.module_path.pop();

        Ok(())
    }

    /// Compile a message handler.
    fn compile_handler(
        &mut self,
        actor_name: &str,
        handler_name: &str,
        handler: &crate::ast::MessageHandler,
    ) -> WasmResult<()> {
        // Find the handler function index
        let qualified_name = self.qualify_name(handler_name);
        let func_idx = self
            .func_map
            .get(&qualified_name)
            .or_else(|| self.func_map.get(handler_name))
            .copied()
            .ok_or_else(|| WasmError::internal(format!(
                "handler not registered: '{}' (qualified: '{}')",
                handler_name, qualified_name
            )))?;

        // Get the function list index
        let import_count = self.imports.import_count();
        let fn_list_idx = (func_idx - import_count) as usize;

        // Set current function context
        self.current_fn_idx = Some(fn_list_idx);

        // Track actor name for self.field resolution
        let prev_actor = self.current_actor.take();
        self.current_actor = Some(actor_name.to_string());

        // Parameters are already registered in register_handler_sig as params
        // They will be accessible via get_local by name

        // Compile the handler body
        self.compile_block(&handler.body)?;

        // If there's no explicit return and no trailing expression, push unit
        if handler.return_type.is_none() && handler.body.expr.is_none() {
            // No return value needed
        } else if handler.body.expr.is_none() {
            // Handler has return type but no trailing expression - push 0
            let func = self.current_function_mut()
                .ok_or_else(|| WasmError::internal("not in function context"))?;
            func.push(Instruction::I64Const(0));
        }

        // Add end instruction
        let func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        func.push(Instruction::End);

        // Restore previous actor context
        self.current_actor = prev_actor;

        // Clear function context
        self.current_fn_idx = None;

        Ok(())
    }

    /// Compile an actor method.
    fn compile_actor_method(&mut self, actor_name: &str, method: &Function) -> WasmResult<()> {
        // Track actor name for self.field resolution
        let prev_actor = self.current_actor.take();
        self.current_actor = Some(actor_name.to_string());

        // Compile as regular function - self references will resolve via current_actor
        self.compile_function(method)?;

        // Restore previous actor context
        self.current_actor = prev_actor;

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
                let name = path.segments.first().map(|s| s.ident.name.as_str()).unwrap_or("");
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
        use crate::ast::Pattern;
        extract_pattern_name(&self.pattern)
    }
}

/// Recursively extract the binding name from a pattern.
/// Handles &self, &mut self, ref self, and plain self patterns.
fn extract_pattern_name(pattern: &crate::ast::Pattern) -> Option<String> {
    use crate::ast::Pattern;
    match pattern {
        // Direct identifier: self, x, etc.
        Pattern::Ident { name, .. } => Some(name.name.clone()),
        // Reference pattern: &self, &mut self, &x
        Pattern::Ref { pattern: inner, .. } => extract_pattern_name(inner),
        // Ref binding pattern: ref x, ref mut x
        Pattern::RefBinding { name, .. } => Some(name.name.clone()),
        // Other patterns don't have simple names
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Block, BinOp, CrateConfig, EnumVariant, FieldDef, Ident, Literal, NumBase, StructAttrs, TypePath, UnaryOp};
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
                    attributes: vec![],
                    visibility: Visibility::Public,
                    name: make_ident("x"),
                    ty: crate::ast::TypeExpr::Path(TypePath { segments: vec![] }),
                    default: None,
                },
                FieldDef {
                    attributes: vec![],
                    visibility: Visibility::Public,
                    name: make_ident("y"),
                    ty: crate::ast::TypeExpr::Path(TypePath { segments: vec![] }),
                    default: None,
                },
            ]),
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
            doc_comments: Vec::new(),
            outer_attrs: Vec::new(),
            visibility: Visibility::Public,
            name: make_ident("Option"),
            generics: None,
            variants: vec![
                EnumVariant {
                    doc_comments: vec![],
                    attributes: vec![],
                    name: make_ident("None"),
                    fields: StructFields::Unit,
                    discriminant: None,
                },
                EnumVariant {
                    doc_comments: vec![],
                    attributes: vec![],
                    name: make_ident("Some"),
                    fields: StructFields::Tuple(vec![crate::ast::TypeExpr::Path(TypePath { segments: vec![] })]),
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

        let true_result = compiler.eval_const_expr(&crate::ast::Expr::Literal(Literal::Bool(true))).unwrap();
        assert_eq!(true_result, 1);

        let false_result = compiler.eval_const_expr(&crate::ast::Expr::Literal(Literal::Bool(false))).unwrap();
        assert_eq!(false_result, 0);
    }

    #[test]
    fn test_eval_const_expr_null() {
        let compiler = WasmCompiler::new();

        let null_result = compiler.eval_const_expr(&crate::ast::Expr::Literal(Literal::Null)).unwrap();
        assert_eq!(null_result, 0);

        let empty_result = compiler.eval_const_expr(&crate::ast::Expr::Literal(Literal::Empty)).unwrap();
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
            doc_comments: Vec::new(),
            outer_attrs: Vec::new(),
            visibility: Visibility::Public,
            name: make_ident("Shape"),
            generics: None,
            variants: vec![
                EnumVariant {
                    doc_comments: vec![],
                    attributes: vec![],
                    name: make_ident("Circle"),
                    fields: StructFields::Named(vec![FieldDef {
                        attributes: vec![],
                        visibility: Visibility::Public,
                        name: make_ident("radius"),
                        ty: crate::ast::TypeExpr::Path(TypePath { segments: vec![] }),
                        default: None,
                    }]),
                    discriminant: None,
                },
                EnumVariant {
                    doc_comments: vec![],
                    attributes: vec![],
                    name: make_ident("Rectangle"),
                    fields: StructFields::Named(vec![
                        FieldDef {
                            attributes: vec![],
                            visibility: Visibility::Public,
                            name: make_ident("width"),
                            ty: crate::ast::TypeExpr::Path(TypePath { segments: vec![] }),
                            default: None,
                        },
                        FieldDef {
                            attributes: vec![],
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
