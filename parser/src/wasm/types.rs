//! WASM compilation type definitions.
//!
//! Core data structures used throughout the WASM compiler.

use std::collections::HashMap;
use wasm_encoder::ValType;

/// Local variable information.
#[derive(Clone, Debug)]
pub struct LocalVar {
    /// Index in the function's local space
    pub index: u32,
    /// WASM value type
    pub ty: ValType,
    /// Whether this is a function parameter
    pub is_param: bool,
}

/// Compiled function representation.
#[derive(Clone, Debug)]
pub struct CompiledFunction {
    /// Function name
    pub name: String,
    /// Index into the type section
    pub type_idx: u32,
    /// Global function index (imports + local functions)
    pub func_idx: u32,
    /// Parameter names and types
    pub params: Vec<(String, ValType)>,
    /// Result types
    pub results: Vec<ValType>,
    /// Local variable map (name -> LocalVar)
    pub locals: HashMap<String, LocalVar>,
    /// Non-parameter local types
    pub local_types: Vec<ValType>,
    /// Compiled instructions
    pub instructions: Vec<wasm_encoder::Instruction<'static>>,
    /// Whether to export this function
    pub is_exported: bool,
}

impl CompiledFunction {
    /// Create a new compiled function.
    pub fn new(
        name: String,
        type_idx: u32,
        func_idx: u32,
        params: Vec<(String, ValType)>,
        results: Vec<ValType>,
        is_exported: bool,
    ) -> Self {
        let locals: HashMap<String, LocalVar> = params
            .iter()
            .enumerate()
            .map(|(i, (pname, ty))| {
                (
                    pname.clone(),
                    LocalVar {
                        index: i as u32,
                        ty: *ty,
                        is_param: true,
                    },
                )
            })
            .collect();

        Self {
            name,
            type_idx,
            func_idx,
            params,
            results,
            locals,
            local_types: Vec::new(),
            instructions: Vec::new(),
            is_exported,
        }
    }

    /// Allocate a new local variable.
    pub fn alloc_local(&mut self, name: String, ty: ValType) -> u32 {
        let index = self.params.len() as u32 + self.local_types.len() as u32;
        self.locals.insert(
            name,
            LocalVar {
                index,
                ty,
                is_param: false,
            },
        );
        self.local_types.push(ty);
        index
    }

    /// Get a local variable by name.
    pub fn get_local(&self, name: &str) -> Option<&LocalVar> {
        self.locals.get(name)
    }

    /// Push an instruction.
    pub fn push(&mut self, instr: wasm_encoder::Instruction<'static>) {
        self.instructions.push(instr);
    }
}

/// Import function information.
#[derive(Clone, Debug)]
pub struct ImportFn {
    /// Module name (e.g., "console", "dom")
    pub module: String,
    /// Function name
    pub name: String,
    /// Type index
    pub type_idx: u32,
}

impl ImportFn {
    pub fn new(module: impl Into<String>, name: impl Into<String>, type_idx: u32) -> Self {
        Self {
            module: module.into(),
            name: name.into(),
            type_idx,
        }
    }

    /// Get the full qualified name (module_name format).
    pub fn qualified_name(&self) -> String {
        format!("{}_{}", self.module, self.name)
    }
}

/// Loop context for break/continue.
#[derive(Clone, Debug)]
pub struct LoopContext {
    /// Label for break (outer block)
    pub break_label: u32,
    /// Label for continue (loop header)
    pub continue_label: u32,
    /// Optional named label for labeled break/continue (e.g., 'outer)
    pub name: Option<String>,
}

/// Closure information for capture analysis.
#[derive(Clone, Debug)]
pub struct ClosureInfo {
    /// Function index of the closure body
    pub func_idx: u32,
    /// Index in the function table
    pub table_idx: u32,
    /// Names of captured variables
    pub captures: Vec<String>,
    /// Size of environment in bytes
    pub env_size: u32,
}

/// Struct layout information.
#[derive(Clone, Debug)]
pub struct StructLayout {
    /// Struct name
    pub name: String,
    /// Fields: (name, byte offset)
    pub fields: Vec<(String, u32)>,
    /// Total size in bytes
    pub size: u32,
}

impl StructLayout {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            fields: Vec::new(),
            size: 0,
        }
    }

    /// Add a field (all fields are 8 bytes / i64).
    pub fn add_field(&mut self, name: impl Into<String>) {
        let offset = self.size;
        self.fields.push((name.into(), offset));
        self.size += 8;
    }

    /// Get field offset by name.
    pub fn field_offset(&self, name: &str) -> Option<u32> {
        self.fields
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, offset)| *offset)
    }
}

/// Enum layout information.
#[derive(Clone, Debug)]
pub struct EnumLayout {
    /// Enum name
    pub name: String,
    /// Variants: (name, tag, optional payload layout)
    pub variants: Vec<(String, u32, Option<StructLayout>)>,
}

impl EnumLayout {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            variants: Vec::new(),
        }
    }

    /// Add a unit variant (no payload).
    pub fn add_unit_variant(&mut self, name: impl Into<String>) {
        let tag = self.variants.len() as u32;
        self.variants.push((name.into(), tag, None));
    }

    /// Add a variant with payload.
    pub fn add_variant_with_payload(&mut self, name: impl Into<String>, payload: StructLayout) {
        let tag = self.variants.len() as u32;
        self.variants.push((name.into(), tag, Some(payload)));
    }

    /// Get variant tag by name.
    pub fn variant_tag(&self, name: &str) -> Option<u32> {
        self.variants
            .iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, tag, _)| *tag)
    }

    /// Get the inner type name for a tuple-struct variant's payload.
    /// Returns the StructLayout name stored for the variant's payload,
    /// which after the register_enum fix equals the inner type's simple name
    /// (e.g. "VElement" for the Element variant of VNode).
    pub fn variant_inner_type(&self, variant_name: &str) -> Option<&str> {
        self.variants
            .iter()
            .find(|(n, _, _)| n == variant_name)
            .and_then(|(_, _, payload)| payload.as_ref())
            .map(|layout| layout.name.as_str())
    }
}

/// Compilation scope for tracking variables.
#[derive(Clone, Debug, Default)]
pub struct Scope {
    /// Variables in this scope (name -> local index)
    pub vars: HashMap<String, u32>,
}

impl Scope {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn define(&mut self, name: String, index: u32) {
        self.vars.insert(name, index);
    }

    pub fn lookup(&self, name: &str) -> Option<u32> {
        self.vars.get(name).copied()
    }
}

/// Qualified item reference for module-level lookups.
#[derive(Clone, Debug)]
pub enum QualifiedItem {
    /// Function with its index
    Function(u32),
    /// Struct layout
    Struct(StructLayout),
    /// Enum layout
    Enum(EnumLayout),
    /// Constant with its global index
    Const(u32),
    /// Static with its global index
    Static(u32),
}

impl QualifiedItem {
    /// Get function index if this is a function.
    pub fn as_function(&self) -> Option<u32> {
        match self {
            QualifiedItem::Function(idx) => Some(*idx),
            _ => None,
        }
    }

    /// Get struct layout if this is a struct.
    pub fn as_struct(&self) -> Option<&StructLayout> {
        match self {
            QualifiedItem::Struct(layout) => Some(layout),
            _ => None,
        }
    }

    /// Get enum layout if this is an enum.
    pub fn as_enum(&self) -> Option<&EnumLayout> {
        match self {
            QualifiedItem::Enum(layout) => Some(layout),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_var_creation() {
        let local = LocalVar {
            index: 0,
            ty: ValType::I64,
            is_param: true,
        };
        assert_eq!(local.index, 0);
        assert!(local.is_param);
    }

    #[test]
    fn test_compiled_function_alloc_local() {
        let mut func = CompiledFunction::new(
            "test".to_string(),
            0,
            0,
            vec![("a".to_string(), ValType::I64)],
            vec![ValType::I64],
            false,
        );

        // Parameter should be at index 0
        assert_eq!(func.get_local("a").unwrap().index, 0);

        // New local should be at index 1
        let idx = func.alloc_local("x".to_string(), ValType::I64);
        assert_eq!(idx, 1);
        assert_eq!(func.get_local("x").unwrap().index, 1);
    }

    #[test]
    fn test_import_fn_qualified_name() {
        let import = ImportFn::new("console", "log", 0);
        assert_eq!(import.qualified_name(), "console_log");
    }

    #[test]
    fn test_struct_layout() {
        let mut layout = StructLayout::new("Point");
        layout.add_field("x");
        layout.add_field("y");

        assert_eq!(layout.size, 16);
        assert_eq!(layout.field_offset("x"), Some(0));
        assert_eq!(layout.field_offset("y"), Some(8));
        assert_eq!(layout.field_offset("z"), None);
    }

    #[test]
    fn test_enum_layout() {
        let mut layout = EnumLayout::new("Option");
        layout.add_unit_variant("None");

        let mut some_payload = StructLayout::new("Some");
        some_payload.add_field("value");
        layout.add_variant_with_payload("Some", some_payload);

        assert_eq!(layout.variant_tag("None"), Some(0));
        assert_eq!(layout.variant_tag("Some"), Some(1));
        assert!(layout.variants[1].2.is_some());
    }

    #[test]
    fn test_scope() {
        let mut scope = Scope::new();
        scope.define("x".to_string(), 0);
        scope.define("y".to_string(), 1);

        assert_eq!(scope.lookup("x"), Some(0));
        assert_eq!(scope.lookup("y"), Some(1));
        assert_eq!(scope.lookup("z"), None);
    }

    #[test]
    fn test_loop_context() {
        let ctx = LoopContext {
            break_label: 1,
            continue_label: 0,
            name: None,
        };
        assert_eq!(ctx.break_label, 1);
        assert_eq!(ctx.continue_label, 0);
        assert_eq!(ctx.name, None);

        let labeled_ctx = LoopContext {
            break_label: 1,
            continue_label: 0,
            name: Some("outer".to_string()),
        };
        assert_eq!(labeled_ctx.name, Some("outer".to_string()));
    }

    #[test]
    fn test_closure_info() {
        let info = ClosureInfo {
            func_idx: 5,
            table_idx: 0,
            captures: vec!["x".to_string(), "y".to_string()],
            env_size: 16,
        };
        assert_eq!(info.captures.len(), 2);
        assert_eq!(info.env_size, 16);
    }
}
