//! AI-Facing IR Emitter for Sigil
//!
//! This module provides JSON IR emission for AI agents and tooling.
//! The IR format is designed to be:
//! - Self-describing with embedded type information
//! - Easy for LLMs to parse, understand, and generate
//! - Bidirectional (can compile back to Sigil source)
//! - Evidentiality-aware with first-class evidence tracking

use crate::ast::*;
use crate::span::Span;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// IR format version
pub const IR_VERSION: &str = "1.0.0";

/// Top-level IR document
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrDocument {
    #[serde(rename = "$schema", skip_serializing_if = "Option::is_none")]
    pub schema: Option<String>,
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<SourceInfo>,
    pub module: IrModule,
    #[serde(skip_serializing_if = "HashMap::is_empty", default)]
    pub types: HashMap<String, IrTypeDef>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub evidence_constraints: Vec<EvidenceConstraint>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<IrMetadata>,
}

/// Source file information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceInfo {
    pub file: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hash: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
}

/// Module structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrModule {
    pub name: String,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub imports: Vec<IrImport>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub exports: Vec<String>,
    pub items: Vec<IrItem>,
}

/// Import declaration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrImport {
    pub path: String,
    pub items: Vec<String>,
}

/// Top-level item
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum IrItem {
    #[serde(rename = "function")]
    Function(IrFunction),
    #[serde(rename = "struct")]
    Struct(IrStruct),
    #[serde(rename = "enum")]
    Enum(IrEnum),
    #[serde(rename = "trait")]
    Trait(IrTrait),
    #[serde(rename = "impl")]
    Impl(IrImpl),
    #[serde(rename = "const")]
    Const(IrConst),
    #[serde(rename = "static")]
    Static(IrStatic),
    #[serde(rename = "type_alias")]
    TypeAlias(IrTypeAlias),
    #[serde(rename = "actor")]
    Actor(IrActor),
}

/// Function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrFunction {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visibility: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub generics: Vec<IrGenericParam>,
    pub params: Vec<IrFunctionParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_type: Option<IrType>,
    #[serde(rename = "async", skip_serializing_if = "is_false", default)]
    pub is_async: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<IrExpr>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub attributes: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span: Option<IrSpan>,
}

/// Function parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrFunctionParam {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: IrType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evidence: Option<String>,
}

/// Generic parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrGenericParam {
    pub name: String,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub bounds: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<IrType>,
}

/// Type representation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum IrType {
    #[serde(rename = "primitive")]
    Primitive { name: String },
    #[serde(rename = "array")]
    Array {
        element: Box<IrType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        size: Option<usize>,
    },
    #[serde(rename = "slice")]
    Slice { element: Box<IrType> },
    #[serde(rename = "tuple")]
    Tuple { elements: Vec<IrType> },
    #[serde(rename = "evidential")]
    Evidential {
        inner: Box<IrType>,
        evidence: String,
    },
    #[serde(rename = "function")]
    Function {
        params: Vec<IrType>,
        #[serde(rename = "return")]
        return_type: Box<IrType>,
    },
    #[serde(rename = "path")]
    Path {
        name: String,
        #[serde(skip_serializing_if = "Vec::is_empty", default)]
        args: Vec<IrType>,
    },
    #[serde(rename = "ref")]
    Ref {
        #[serde(skip_serializing_if = "is_false", default)]
        mutable: bool,
        inner: Box<IrType>,
    },
    #[serde(rename = "ptr")]
    Ptr {
        #[serde(skip_serializing_if = "is_false", default)]
        mutable: bool,
        inner: Box<IrType>,
    },
    #[serde(rename = "generic")]
    Generic { name: String },
    #[serde(rename = "simd")]
    Simd { element: Box<IrType>, lanes: u8 },
    #[serde(rename = "never")]
    Never,
    #[serde(rename = "unit")]
    Unit,
}

/// Expression IR
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum IrExpr {
    #[serde(rename = "literal")]
    Literal { value: IrLiteralValue },
    #[serde(rename = "path")]
    Path { name: String },
    #[serde(rename = "binary")]
    Binary {
        op: String,
        left: Box<IrExpr>,
        right: Box<IrExpr>,
    },
    #[serde(rename = "unary")]
    Unary { op: String, expr: Box<IrExpr> },
    #[serde(rename = "call")]
    Call { func: Box<IrExpr>, args: Vec<IrExpr> },
    #[serde(rename = "method_call")]
    MethodCall {
        receiver: Box<IrExpr>,
        method: String,
        args: Vec<IrExpr>,
    },
    #[serde(rename = "pipeline")]
    Pipeline {
        source: Box<IrExpr>,
        stages: Vec<IrPipelineStage>,
    },
    #[serde(rename = "closure")]
    Closure {
        params: Vec<IrClosureParam>,
        body: Box<IrExpr>,
    },
    #[serde(rename = "if")]
    If {
        condition: Box<IrExpr>,
        then_branch: Box<IrExpr>,
        #[serde(skip_serializing_if = "Option::is_none")]
        else_branch: Option<Box<IrExpr>>,
    },
    #[serde(rename = "match")]
    Match {
        scrutinee: Box<IrExpr>,
        arms: Vec<IrMatchArm>,
    },
    #[serde(rename = "block")]
    Block {
        statements: Vec<IrStatement>,
        #[serde(skip_serializing_if = "Option::is_none")]
        tail_expr: Option<Box<IrExpr>>,
    },
    #[serde(rename = "for")]
    For {
        pattern: IrPattern,
        iter: Box<IrExpr>,
        body: Box<IrExpr>,
    },
    #[serde(rename = "while")]
    While {
        condition: Box<IrExpr>,
        body: Box<IrExpr>,
    },
    #[serde(rename = "loop")]
    Loop { body: Box<IrExpr> },
    #[serde(rename = "return")]
    Return {
        #[serde(skip_serializing_if = "Option::is_none")]
        value: Option<Box<IrExpr>>,
    },
    #[serde(rename = "break")]
    Break {
        #[serde(skip_serializing_if = "Option::is_none")]
        value: Option<Box<IrExpr>>,
    },
    #[serde(rename = "continue")]
    Continue,
    #[serde(rename = "await")]
    Await { future: Box<IrExpr> },
    #[serde(rename = "field_access")]
    FieldAccess { object: Box<IrExpr>, field: String },
    #[serde(rename = "index")]
    Index {
        object: Box<IrExpr>,
        index: Box<IrExpr>,
    },
    #[serde(rename = "struct_literal")]
    StructLiteral {
        name: String,
        fields: Vec<IrFieldInit>,
    },
    #[serde(rename = "array_literal")]
    ArrayLiteral { elements: Vec<IrExpr> },
    #[serde(rename = "tuple_literal")]
    TupleLiteral { elements: Vec<IrExpr> },
    #[serde(rename = "morpheme")]
    Morpheme { symbol: String, body: Box<IrExpr> },
    #[serde(rename = "cast")]
    Cast {
        expr: Box<IrExpr>,
        #[serde(rename = "type")]
        ty: IrType,
    },
    #[serde(rename = "unsafe_block")]
    UnsafeBlock { body: Box<IrExpr> },
}

/// Literal value
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum IrLiteralValue {
    Int { int: i64 },
    Float { float: f64 },
    String { string: String },
    Bool { bool: bool },
    Char { char: String },
    Null { null: bool },
}

/// Pipeline stage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrPipelineStage {
    pub op: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub symbol: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mapper: Option<Box<IrExpr>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub predicate: Option<Box<IrExpr>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reducer: Option<Box<IrExpr>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<Box<IrExpr>>,
    #[serde(rename = "fn", skip_serializing_if = "Option::is_none")]
    pub fn_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<Vec<IrExpr>>,
}

/// Closure parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrClosureParam {
    pub name: String,
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    pub ty: Option<IrType>,
}

/// Match arm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrMatchArm {
    pub pattern: IrPattern,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guard: Option<IrExpr>,
    pub body: IrExpr,
}

/// Pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum IrPattern {
    #[serde(rename = "wildcard")]
    Wildcard,
    #[serde(rename = "binding")]
    Binding {
        name: String,
        #[serde(skip_serializing_if = "is_false", default)]
        mutable: bool,
    },
    #[serde(rename = "literal")]
    Literal { value: IrLiteralValue },
    #[serde(rename = "tuple")]
    Tuple { elements: Vec<IrPattern> },
    #[serde(rename = "struct")]
    Struct {
        struct_name: String,
        #[serde(skip_serializing_if = "Vec::is_empty", default)]
        fields: Vec<IrFieldPattern>,
    },
    #[serde(rename = "tuple_struct")]
    TupleStruct {
        struct_name: String,
        elements: Vec<IrPattern>,
    },
    #[serde(rename = "slice")]
    Slice { elements: Vec<IrPattern> },
    #[serde(rename = "or")]
    Or { patterns: Vec<IrPattern> },
}

/// Field pattern in struct pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrFieldPattern {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pattern: Option<IrPattern>,
}

/// Statement
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum IrStatement {
    #[serde(rename = "let")]
    Let {
        pattern: IrPattern,
        #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
        ty: Option<IrType>,
        #[serde(skip_serializing_if = "Option::is_none")]
        value: Option<IrExpr>,
    },
    #[serde(rename = "expr")]
    Expr { expr: IrExpr },
}

/// Field initialization in struct literal
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrFieldInit {
    pub name: String,
    pub value: IrExpr,
}

/// Struct definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrStruct {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visibility: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub generics: Vec<IrGenericParam>,
    pub fields: Vec<IrStructField>,
}

/// Struct field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrStructField {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: IrType,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visibility: Option<String>,
}

/// Enum definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrEnum {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visibility: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub generics: Vec<IrGenericParam>,
    pub variants: Vec<IrEnumVariant>,
}

/// Enum variant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrEnumVariant {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fields: Option<IrVariantFields>,
}

/// Variant fields
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum IrVariantFields {
    Tuple(Vec<IrType>),
    Struct(Vec<IrStructField>),
}

/// Trait definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrTrait {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visibility: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub generics: Vec<IrGenericParam>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub supertraits: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub methods: Vec<IrFunction>,
}

/// Impl block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrImpl {
    #[serde(rename = "trait", skip_serializing_if = "Option::is_none")]
    pub trait_name: Option<String>,
    pub target: IrType,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub generics: Vec<IrGenericParam>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub methods: Vec<IrFunction>,
}

/// Const definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrConst {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visibility: Option<String>,
    #[serde(rename = "type")]
    pub ty: IrType,
    pub value: IrExpr,
}

/// Static definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrStatic {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visibility: Option<String>,
    #[serde(skip_serializing_if = "is_false", default)]
    pub mutable: bool,
    #[serde(rename = "type")]
    pub ty: IrType,
    pub value: IrExpr,
}

/// Type alias
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrTypeAlias {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visibility: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub generics: Vec<IrGenericParam>,
    pub target: IrType,
}

/// Actor definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrActor {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub visibility: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub state: Vec<IrStructField>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub handlers: Vec<IrHandler>,
}

/// Message handler
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrHandler {
    pub message: String,
    pub params: Vec<IrFunctionParam>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub return_type: Option<IrType>,
    pub body: IrExpr,
}

/// Type definition (for types map)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum IrTypeDef {
    #[serde(rename = "struct")]
    Struct(IrStruct),
    #[serde(rename = "enum")]
    Enum(IrEnum),
    #[serde(rename = "type_alias")]
    TypeAlias(IrTypeAlias),
}

/// Evidence constraint
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceConstraint {
    pub location: String,
    pub required: String,
    pub actual: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resolution: Option<String>,
}

/// Source span
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrSpan {
    pub start: usize,
    pub end: usize,
}

/// Compilation metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub optimization_level: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
}

// Helper function for serde skip_serializing_if
fn is_false(b: &bool) -> bool {
    !*b
}

// ============================================================================
// IR Emitter Options
// ============================================================================

/// Options for IR emission
#[derive(Debug, Clone, Default)]
pub struct IrEmitterOptions {
    /// Include type information in expressions
    pub include_types: bool,
    /// Include source spans
    pub include_spans: bool,
    /// Pretty print JSON output
    pub pretty: bool,
}

// ============================================================================
// IR Emitter Implementation
// ============================================================================

/// Emits JSON IR from Sigil AST
pub struct IrEmitter {
    options: IrEmitterOptions,
    id_counter: usize,
}

impl IrEmitter {
    /// Create a new IR emitter with default options
    pub fn new() -> Self {
        Self {
            options: IrEmitterOptions::default(),
            id_counter: 0,
        }
    }

    /// Create a new IR emitter with custom options
    pub fn with_options(options: IrEmitterOptions) -> Self {
        Self {
            options,
            id_counter: 0,
        }
    }

    /// Generate a unique ID
    fn next_id(&mut self, prefix: &str) -> String {
        let id = format!("{}_{}", prefix, self.id_counter);
        self.id_counter += 1;
        id
    }

    /// Emit IR document from a source file
    pub fn emit(&mut self, source: &SourceFile, filename: &str) -> IrDocument {
        let module_name = filename
            .trim_end_matches(".sigil")
            .trim_end_matches(".sg")
            .split('/')
            .last()
            .unwrap_or("module")
            .to_string();

        let mut types = HashMap::new();
        let items: Vec<IrItem> = source
            .items
            .iter()
            .filter_map(|item| self.emit_item(&item.node, &mut types))
            .collect();

        IrDocument {
            schema: Some("https://sigil-lang.dev/schemas/ir/v1.json".to_string()),
            version: IR_VERSION.to_string(),
            source: Some(SourceInfo {
                file: filename.to_string(),
                hash: None,
                timestamp: None,
            }),
            module: IrModule {
                name: module_name,
                imports: vec![],
                exports: vec![],
                items,
            },
            types,
            evidence_constraints: vec![],
            metadata: Some(IrMetadata {
                optimization_level: None,
                target: None,
            }),
        }
    }

    /// Emit an item
    fn emit_item(&mut self, item: &Item, types: &mut HashMap<String, IrTypeDef>) -> Option<IrItem> {
        match item {
            Item::Function(f) => Some(IrItem::Function(self.emit_function(f))),
            Item::Struct(s) => {
                let ir_struct = self.emit_struct(s);
                types.insert(s.name.name.clone(), IrTypeDef::Struct(ir_struct.clone()));
                Some(IrItem::Struct(ir_struct))
            }
            Item::Enum(e) => {
                let ir_enum = self.emit_enum(e);
                types.insert(e.name.name.clone(), IrTypeDef::Enum(ir_enum.clone()));
                Some(IrItem::Enum(ir_enum))
            }
            Item::Trait(t) => Some(IrItem::Trait(self.emit_trait(t))),
            Item::Impl(i) => Some(IrItem::Impl(self.emit_impl(i))),
            Item::Const(c) => Some(IrItem::Const(self.emit_const(c))),
            Item::Static(s) => Some(IrItem::Static(self.emit_static(s))),
            Item::TypeAlias(t) => {
                let ir_alias = self.emit_type_alias(t);
                types.insert(t.name.name.clone(), IrTypeDef::TypeAlias(ir_alias.clone()));
                Some(IrItem::TypeAlias(ir_alias))
            }
            Item::Actor(a) => Some(IrItem::Actor(self.emit_actor(a))),
            Item::Module(_) | Item::Use(_) | Item::ExternBlock(_) => None,
        }
    }

    /// Emit a function
    fn emit_function(&mut self, f: &Function) -> IrFunction {
        IrFunction {
            id: Some(self.next_id("fn")),
            name: f.name.name.clone(),
            visibility: Some(self.emit_visibility(&f.visibility)),
            generics: f
                .generics
                .as_ref()
                .map(|g| g.params.iter().map(|p| self.emit_generic_param(p)).collect())
                .unwrap_or_default(),
            params: f.params.iter().map(|p| self.emit_param(p)).collect(),
            return_type: f.return_type.as_ref().map(|t| self.emit_type(t)),
            is_async: f.is_async,
            body: f.body.as_ref().map(|b| self.emit_block(b)),
            attributes: self.emit_function_attrs(&f.attrs),
            span: if self.options.include_spans {
                Some(self.emit_span(&f.name.span))
            } else {
                None
            },
        }
    }

    /// Emit visibility
    fn emit_visibility(&self, vis: &Visibility) -> String {
        match vis {
            Visibility::Public => "public".to_string(),
            Visibility::Private => "private".to_string(),
            Visibility::Crate => "crate".to_string(),
            Visibility::Super => "super".to_string(),
        }
    }

    /// Emit generic parameter
    fn emit_generic_param(&self, param: &GenericParam) -> IrGenericParam {
        match param {
            GenericParam::Type { name, bounds, .. } => IrGenericParam {
                name: name.name.clone(),
                bounds: bounds.iter().map(|b| self.type_to_string(b)).collect(),
                default: None,
            },
            GenericParam::Const { name, .. } => IrGenericParam {
                name: name.name.clone(),
                bounds: vec![],
                default: None,
            },
            GenericParam::Lifetime(name) => IrGenericParam {
                name: name.clone(),
                bounds: vec![],
                default: None,
            },
        }
    }

    /// Emit function parameter
    fn emit_param(&mut self, param: &Param) -> IrFunctionParam {
        IrFunctionParam {
            name: self.pattern_to_name(&param.pattern),
            ty: self.emit_type(&param.ty),
            evidence: None,
        }
    }

    /// Extract name from pattern
    fn pattern_to_name(&self, pattern: &Pattern) -> String {
        match pattern {
            Pattern::Ident { name, .. } => name.name.clone(),
            Pattern::Wildcard => "_".to_string(),
            _ => "_".to_string(),
        }
    }

    /// Emit type
    fn emit_type(&self, ty: &TypeExpr) -> IrType {
        match ty {
            TypeExpr::Path(path) => {
                let name = path
                    .segments
                    .iter()
                    .map(|s| s.ident.name.as_str())
                    .collect::<Vec<_>>()
                    .join("::");
                // Check for primitive types
                match name.as_str() {
                    "i8" | "i16" | "i32" | "i64" | "i128" | "isize" | "u8" | "u16" | "u32"
                    | "u64" | "u128" | "usize" | "f32" | "f64" | "bool" | "char" | "str" => {
                        IrType::Primitive { name }
                    }
                    "()" => IrType::Unit,
                    "!" => IrType::Never,
                    _ => IrType::Path {
                        name,
                        args: path
                            .segments
                            .last()
                            .and_then(|s| s.generics.as_ref())
                            .map(|gens| gens.iter().map(|g| self.emit_type(g)).collect())
                            .unwrap_or_default(),
                    },
                }
            }
            TypeExpr::Reference { mutable, inner } => IrType::Ref {
                mutable: *mutable,
                inner: Box::new(self.emit_type(inner)),
            },
            TypeExpr::Pointer { mutable, inner } => IrType::Ptr {
                mutable: *mutable,
                inner: Box::new(self.emit_type(inner)),
            },
            TypeExpr::Array { element, .. } => IrType::Array {
                element: Box::new(self.emit_type(element)),
                size: None,
            },
            TypeExpr::Slice(inner) => IrType::Slice {
                element: Box::new(self.emit_type(inner)),
            },
            TypeExpr::Tuple(elements) => IrType::Tuple {
                elements: elements.iter().map(|e| self.emit_type(e)).collect(),
            },
            TypeExpr::Function {
                params,
                return_type,
            } => IrType::Function {
                params: params.iter().map(|p| self.emit_type(p)).collect(),
                return_type: Box::new(
                    return_type
                        .as_ref()
                        .map(|t| self.emit_type(t))
                        .unwrap_or(IrType::Unit),
                ),
            },
            TypeExpr::Evidential { inner, evidentiality } => IrType::Evidential {
                inner: Box::new(self.emit_type(inner)),
                evidence: self.emit_evidence(evidentiality),
            },
            TypeExpr::Infer => IrType::Generic {
                name: "_".to_string(),
            },
            TypeExpr::Never => IrType::Never,
            TypeExpr::Simd { element, lanes } => IrType::Simd {
                element: Box::new(self.emit_type(element)),
                lanes: *lanes,
            },
            TypeExpr::Atomic(inner) => self.emit_type(inner),
            TypeExpr::Cycle { .. } => IrType::Path {
                name: "Cycle".to_string(),
                args: vec![],
            },
        }
    }

    /// Emit evidence level
    fn emit_evidence(&self, evidence: &Evidentiality) -> String {
        match evidence {
            Evidentiality::Known => "known".to_string(),
            Evidentiality::Uncertain => "uncertain".to_string(),
            Evidentiality::Reported => "reported".to_string(),
            Evidentiality::Paradox => "paradox".to_string(),
        }
    }

    /// Convert type to string (for bounds)
    fn type_to_string(&self, ty: &TypeExpr) -> String {
        match ty {
            TypeExpr::Path(path) => path
                .segments
                .iter()
                .map(|s| s.ident.name.as_str())
                .collect::<Vec<_>>()
                .join("::"),
            _ => format!("{:?}", ty),
        }
    }

    /// Emit function attributes
    fn emit_function_attrs(&self, attrs: &FunctionAttrs) -> Vec<String> {
        let mut result = vec![];
        if attrs.naked {
            result.push("#[naked]".to_string());
        }
        if let Some(ref hint) = attrs.inline {
            match hint {
                InlineHint::Hint => result.push("#[inline]".to_string()),
                InlineHint::Always => result.push("#[inline(always)]".to_string()),
                InlineHint::Never => result.push("#[inline(never)]".to_string()),
            }
        }
        if attrs.no_mangle {
            result.push("#[no_mangle]".to_string());
        }
        if attrs.cold {
            result.push("#[cold]".to_string());
        }
        if attrs.hot {
            result.push("#[hot]".to_string());
        }
        if attrs.test {
            result.push("#[test]".to_string());
        }
        result
    }

    /// Emit a block as expression
    fn emit_block(&mut self, block: &Block) -> IrExpr {
        let statements: Vec<IrStatement> = block
            .stmts
            .iter()
            .map(|s| self.emit_statement(s))
            .collect();
        let tail_expr = block.expr.as_ref().map(|e| Box::new(self.emit_expr(e)));

        IrExpr::Block {
            statements,
            tail_expr,
        }
    }

    /// Emit statement
    fn emit_statement(&mut self, stmt: &Stmt) -> IrStatement {
        match stmt {
            Stmt::Let { pattern, ty, init } => IrStatement::Let {
                pattern: self.emit_pattern(pattern),
                ty: ty.as_ref().map(|t| self.emit_type(t)),
                value: init.as_ref().map(|e| self.emit_expr(e)),
            },
            Stmt::Expr(expr) | Stmt::Semi(expr) => IrStatement::Expr {
                expr: self.emit_expr(expr),
            },
            Stmt::Item(_) => IrStatement::Expr {
                expr: IrExpr::Literal {
                    value: IrLiteralValue::Null { null: true },
                },
            },
        }
    }

    /// Emit expression
    fn emit_expr(&mut self, expr: &Expr) -> IrExpr {
        match expr {
            Expr::Literal(lit) => IrExpr::Literal {
                value: self.emit_literal(lit),
            },
            Expr::Path(path) => IrExpr::Path {
                name: path
                    .segments
                    .iter()
                    .map(|s| s.ident.name.as_str())
                    .collect::<Vec<_>>()
                    .join("::"),
            },
            Expr::Binary { left, op, right } => IrExpr::Binary {
                op: self.emit_binop(op),
                left: Box::new(self.emit_expr(left)),
                right: Box::new(self.emit_expr(right)),
            },
            Expr::Unary { op, expr } => IrExpr::Unary {
                op: self.emit_unaryop(op),
                expr: Box::new(self.emit_expr(expr)),
            },
            Expr::Call { func, args, .. } => IrExpr::Call {
                func: Box::new(self.emit_expr(func)),
                args: args.iter().map(|a| self.emit_expr(a)).collect(),
            },
            Expr::MethodCall {
                receiver,
                method,
                args,
                ..
            } => IrExpr::MethodCall {
                receiver: Box::new(self.emit_expr(receiver)),
                method: method.name.clone(),
                args: args.iter().map(|a| self.emit_expr(a)).collect(),
            },
            Expr::Pipe { expr, operations } => IrExpr::Pipeline {
                source: Box::new(self.emit_expr(expr)),
                stages: operations.iter().map(|op| self.emit_pipe_op(op)).collect(),
            },
            Expr::If {
                condition,
                then_branch,
                else_branch,
            } => IrExpr::If {
                condition: Box::new(self.emit_expr(condition)),
                then_branch: Box::new(self.emit_block(then_branch)),
                else_branch: else_branch.as_ref().map(|e| Box::new(self.emit_expr(e))),
            },
            Expr::Match { expr, arms } => IrExpr::Match {
                scrutinee: Box::new(self.emit_expr(expr)),
                arms: arms.iter().map(|a| self.emit_match_arm(a)).collect(),
            },
            Expr::Block(block) => self.emit_block(block),
            Expr::For {
                pattern,
                iter,
                body,
            } => IrExpr::For {
                pattern: self.emit_pattern(pattern),
                iter: Box::new(self.emit_expr(iter)),
                body: Box::new(self.emit_block(body)),
            },
            Expr::While { condition, body } => IrExpr::While {
                condition: Box::new(self.emit_expr(condition)),
                body: Box::new(self.emit_block(body)),
            },
            Expr::Loop(body) => IrExpr::Loop {
                body: Box::new(self.emit_block(body)),
            },
            Expr::Return(value) => IrExpr::Return {
                value: value.as_ref().map(|e| Box::new(self.emit_expr(e))),
            },
            Expr::Break(value) => IrExpr::Break {
                value: value.as_ref().map(|e| Box::new(self.emit_expr(e))),
            },
            Expr::Continue => IrExpr::Continue,
            Expr::Await(future) => IrExpr::Await {
                future: Box::new(self.emit_expr(future)),
            },
            Expr::Field { expr, field } => IrExpr::FieldAccess {
                object: Box::new(self.emit_expr(expr)),
                field: field.name.clone(),
            },
            Expr::Index { expr, index } => IrExpr::Index {
                object: Box::new(self.emit_expr(expr)),
                index: Box::new(self.emit_expr(index)),
            },
            Expr::Struct { path, fields, .. } => IrExpr::StructLiteral {
                name: path
                    .segments
                    .iter()
                    .map(|s| s.ident.name.as_str())
                    .collect::<Vec<_>>()
                    .join("::"),
                fields: fields.iter().map(|f| self.emit_field_init(f)).collect(),
            },
            Expr::Array(elements) => IrExpr::ArrayLiteral {
                elements: elements.iter().map(|e| self.emit_expr(e)).collect(),
            },
            Expr::Tuple(elements) => IrExpr::TupleLiteral {
                elements: elements.iter().map(|e| self.emit_expr(e)).collect(),
            },
            Expr::Closure { params, body } => IrExpr::Closure {
                params: params
                    .iter()
                    .map(|p| IrClosureParam {
                        name: self.pattern_to_name(&p.pattern),
                        ty: p.ty.as_ref().map(|t| self.emit_type(t)),
                    })
                    .collect(),
                body: Box::new(self.emit_expr(body)),
            },
            Expr::Cast { expr, ty } => IrExpr::Cast {
                expr: Box::new(self.emit_expr(expr)),
                ty: self.emit_type(ty),
            },
            Expr::Unsafe(block) => IrExpr::UnsafeBlock {
                body: Box::new(self.emit_block(block)),
            },
            Expr::Morpheme { kind, body } => IrExpr::Morpheme {
                symbol: self.emit_morpheme_kind(kind),
                body: Box::new(self.emit_expr(body)),
            },
            // Default case for unhandled expressions
            _ => IrExpr::Literal {
                value: IrLiteralValue::Null { null: true },
            },
        }
    }

    /// Emit literal
    fn emit_literal(&self, lit: &Literal) -> IrLiteralValue {
        match lit {
            Literal::Int { value, .. } => {
                let v: i64 = value.parse().unwrap_or(0);
                IrLiteralValue::Int { int: v }
            }
            Literal::Float { value, .. } => {
                let v: f64 = value.parse().unwrap_or(0.0);
                IrLiteralValue::Float { float: v }
            }
            Literal::String(s) => IrLiteralValue::String { string: s.clone() },
            Literal::Char(c) => IrLiteralValue::Char { char: c.to_string() },
            Literal::Bool(b) => IrLiteralValue::Bool { bool: *b },
            _ => IrLiteralValue::Null { null: true },
        }
    }

    /// Emit binary operator
    fn emit_binop(&self, op: &BinOp) -> String {
        match op {
            BinOp::Add => "+".to_string(),
            BinOp::Sub => "-".to_string(),
            BinOp::Mul => "*".to_string(),
            BinOp::Div => "/".to_string(),
            BinOp::Rem => "%".to_string(),
            BinOp::Pow => "**".to_string(),
            BinOp::Eq => "==".to_string(),
            BinOp::Ne => "!=".to_string(),
            BinOp::Lt => "<".to_string(),
            BinOp::Le => "<=".to_string(),
            BinOp::Gt => ">".to_string(),
            BinOp::Ge => ">=".to_string(),
            BinOp::And => "&&".to_string(),
            BinOp::Or => "||".to_string(),
            BinOp::BitAnd => "&".to_string(),
            BinOp::BitOr => "|".to_string(),
            BinOp::BitXor => "^".to_string(),
            BinOp::Shl => "<<".to_string(),
            BinOp::Shr => ">>".to_string(),
            BinOp::Concat => "++".to_string(),
        }
    }

    /// Emit unary operator
    fn emit_unaryop(&self, op: &UnaryOp) -> String {
        match op {
            UnaryOp::Neg => "-".to_string(),
            UnaryOp::Not => "!".to_string(),
            UnaryOp::Deref => "*".to_string(),
            UnaryOp::Ref => "&".to_string(),
            UnaryOp::RefMut => "&mut".to_string(),
        }
    }

    /// Emit pipe operation
    fn emit_pipe_op(&mut self, op: &PipeOp) -> IrPipelineStage {
        match op {
            PipeOp::Transform(mapper) => IrPipelineStage {
                op: "transform".to_string(),
                symbol: Some("τ".to_string()),
                mapper: Some(Box::new(self.emit_expr(mapper))),
                predicate: None,
                reducer: None,
                key: None,
                index: None,
                fn_name: None,
                args: None,
            },
            PipeOp::Filter(predicate) => IrPipelineStage {
                op: "filter".to_string(),
                symbol: Some("φ".to_string()),
                mapper: None,
                predicate: Some(Box::new(self.emit_expr(predicate))),
                reducer: None,
                key: None,
                index: None,
                fn_name: None,
                args: None,
            },
            PipeOp::Sort(key) => IrPipelineStage {
                op: "sort".to_string(),
                symbol: Some("σ".to_string()),
                mapper: None,
                predicate: None,
                reducer: None,
                key: key.as_ref().map(|k| k.name.clone()),
                index: None,
                fn_name: None,
                args: None,
            },
            PipeOp::Reduce(reducer) => IrPipelineStage {
                op: "reduce".to_string(),
                symbol: Some("ρ".to_string()),
                mapper: None,
                predicate: None,
                reducer: Some(Box::new(self.emit_expr(reducer))),
                key: None,
                index: None,
                fn_name: None,
                args: None,
            },
            PipeOp::First => IrPipelineStage {
                op: "first".to_string(),
                symbol: Some("α".to_string()),
                mapper: None,
                predicate: None,
                reducer: None,
                key: None,
                index: None,
                fn_name: None,
                args: None,
            },
            PipeOp::Last => IrPipelineStage {
                op: "last".to_string(),
                symbol: Some("ω".to_string()),
                mapper: None,
                predicate: None,
                reducer: None,
                key: None,
                index: None,
                fn_name: None,
                args: None,
            },
            PipeOp::Middle => IrPipelineStage {
                op: "middle".to_string(),
                symbol: Some("μ".to_string()),
                mapper: None,
                predicate: None,
                reducer: None,
                key: None,
                index: None,
                fn_name: None,
                args: None,
            },
            PipeOp::Choice => IrPipelineStage {
                op: "choice".to_string(),
                symbol: Some("χ".to_string()),
                mapper: None,
                predicate: None,
                reducer: None,
                key: None,
                index: None,
                fn_name: None,
                args: None,
            },
            PipeOp::Nth(index) => IrPipelineStage {
                op: "nth".to_string(),
                symbol: Some("ν".to_string()),
                mapper: None,
                predicate: None,
                reducer: None,
                key: None,
                index: Some(Box::new(self.emit_expr(index))),
                fn_name: None,
                args: None,
            },
            PipeOp::Next => IrPipelineStage {
                op: "next".to_string(),
                symbol: Some("ξ".to_string()),
                mapper: None,
                predicate: None,
                reducer: None,
                key: None,
                index: None,
                fn_name: None,
                args: None,
            },
            PipeOp::Method { name, args } => IrPipelineStage {
                op: "call".to_string(),
                symbol: None,
                mapper: None,
                predicate: None,
                reducer: None,
                key: None,
                index: None,
                fn_name: Some(name.name.clone()),
                args: Some(args.iter().map(|a| self.emit_expr(a)).collect()),
            },
            _ => IrPipelineStage {
                op: "unknown".to_string(),
                symbol: None,
                mapper: None,
                predicate: None,
                reducer: None,
                key: None,
                index: None,
                fn_name: None,
                args: None,
            },
        }
    }

    /// Emit morpheme kind
    fn emit_morpheme_kind(&self, kind: &MorphemeKind) -> String {
        match kind {
            MorphemeKind::Transform => "τ".to_string(),
            MorphemeKind::Filter => "φ".to_string(),
            MorphemeKind::Sort => "σ".to_string(),
            MorphemeKind::Reduce => "ρ".to_string(),
            MorphemeKind::Lambda => "λ".to_string(),
            MorphemeKind::Sum => "Σ".to_string(),
            MorphemeKind::Product => "Π".to_string(),
            MorphemeKind::First => "α".to_string(),
            MorphemeKind::Last => "ω".to_string(),
            MorphemeKind::Middle => "μ".to_string(),
            MorphemeKind::Choice => "χ".to_string(),
            MorphemeKind::Nth => "ν".to_string(),
            MorphemeKind::Next => "ξ".to_string(),
        }
    }

    /// Emit match arm
    fn emit_match_arm(&mut self, arm: &MatchArm) -> IrMatchArm {
        IrMatchArm {
            pattern: self.emit_pattern(&arm.pattern),
            guard: arm.guard.as_ref().map(|g| self.emit_expr(g)),
            body: self.emit_expr(&arm.body),
        }
    }

    /// Emit pattern
    fn emit_pattern(&self, pattern: &Pattern) -> IrPattern {
        match pattern {
            Pattern::Wildcard => IrPattern::Wildcard,
            Pattern::Ident { name, mutable, .. } => IrPattern::Binding {
                name: name.name.clone(),
                mutable: *mutable,
            },
            Pattern::Literal(lit) => IrPattern::Literal {
                value: self.emit_literal(lit),
            },
            Pattern::Tuple(patterns) => IrPattern::Tuple {
                elements: patterns.iter().map(|p| self.emit_pattern(p)).collect(),
            },
            Pattern::Struct { path, fields, .. } => IrPattern::Struct {
                struct_name: path
                    .segments
                    .iter()
                    .map(|s| s.ident.name.as_str())
                    .collect::<Vec<_>>()
                    .join("::"),
                fields: fields
                    .iter()
                    .map(|f| IrFieldPattern {
                        name: f.name.name.clone(),
                        pattern: f.pattern.as_ref().map(|p| self.emit_pattern(p)),
                    })
                    .collect(),
            },
            Pattern::TupleStruct { path, fields } => IrPattern::TupleStruct {
                struct_name: path
                    .segments
                    .iter()
                    .map(|s| s.ident.name.as_str())
                    .collect::<Vec<_>>()
                    .join("::"),
                elements: fields.iter().map(|p| self.emit_pattern(p)).collect(),
            },
            Pattern::Slice(patterns) => IrPattern::Slice {
                elements: patterns.iter().map(|p| self.emit_pattern(p)).collect(),
            },
            Pattern::Or(patterns) => IrPattern::Or {
                patterns: patterns.iter().map(|p| self.emit_pattern(p)).collect(),
            },
            _ => IrPattern::Wildcard,
        }
    }

    /// Emit field initialization
    fn emit_field_init(&mut self, field: &FieldInit) -> IrFieldInit {
        IrFieldInit {
            name: field.name.name.clone(),
            value: field
                .value
                .as_ref()
                .map(|e| self.emit_expr(e))
                .unwrap_or_else(|| IrExpr::Path {
                    name: field.name.name.clone(),
                }),
        }
    }

    /// Emit span
    fn emit_span(&self, span: &Span) -> IrSpan {
        IrSpan {
            start: span.start,
            end: span.end,
        }
    }

    /// Emit struct definition
    fn emit_struct(&mut self, s: &StructDef) -> IrStruct {
        IrStruct {
            name: s.name.name.clone(),
            visibility: Some(self.emit_visibility(&s.visibility)),
            generics: s
                .generics
                .as_ref()
                .map(|g| g.params.iter().map(|p| self.emit_generic_param(p)).collect())
                .unwrap_or_default(),
            fields: match &s.fields {
                StructFields::Named(fields) => {
                    fields.iter().map(|f| self.emit_field_def(f)).collect()
                }
                StructFields::Tuple(types) => types
                    .iter()
                    .enumerate()
                    .map(|(i, t)| IrStructField {
                        name: format!("{}", i),
                        ty: self.emit_type(t),
                        visibility: None,
                    })
                    .collect(),
                StructFields::Unit => vec![],
            },
        }
    }

    /// Emit field definition
    fn emit_field_def(&self, field: &FieldDef) -> IrStructField {
        IrStructField {
            name: field.name.name.clone(),
            ty: self.emit_type(&field.ty),
            visibility: Some(self.emit_visibility(&field.visibility)),
        }
    }

    /// Emit enum definition
    fn emit_enum(&mut self, e: &EnumDef) -> IrEnum {
        IrEnum {
            name: e.name.name.clone(),
            visibility: Some(self.emit_visibility(&e.visibility)),
            generics: e
                .generics
                .as_ref()
                .map(|g| g.params.iter().map(|p| self.emit_generic_param(p)).collect())
                .unwrap_or_default(),
            variants: e.variants.iter().map(|v| self.emit_variant(v)).collect(),
        }
    }

    /// Emit enum variant
    fn emit_variant(&mut self, variant: &EnumVariant) -> IrEnumVariant {
        IrEnumVariant {
            name: variant.name.name.clone(),
            fields: match &variant.fields {
                StructFields::Named(fields) => Some(IrVariantFields::Struct(
                    fields.iter().map(|f| self.emit_field_def(f)).collect(),
                )),
                StructFields::Tuple(types) => Some(IrVariantFields::Tuple(
                    types.iter().map(|t| self.emit_type(t)).collect(),
                )),
                StructFields::Unit => None,
            },
        }
    }

    /// Emit trait definition
    fn emit_trait(&mut self, t: &TraitDef) -> IrTrait {
        IrTrait {
            name: t.name.name.clone(),
            visibility: Some(self.emit_visibility(&t.visibility)),
            generics: t
                .generics
                .as_ref()
                .map(|g| g.params.iter().map(|p| self.emit_generic_param(p)).collect())
                .unwrap_or_default(),
            supertraits: t.supertraits.iter().map(|s| self.type_to_string(s)).collect(),
            methods: t
                .items
                .iter()
                .filter_map(|item| {
                    if let TraitItem::Function(f) = item {
                        Some(self.emit_function(f))
                    } else {
                        None
                    }
                })
                .collect(),
        }
    }

    /// Emit impl block
    fn emit_impl(&mut self, i: &ImplBlock) -> IrImpl {
        IrImpl {
            trait_name: i
                .trait_
                .as_ref()
                .map(|t| self.type_to_string(&TypeExpr::Path(t.clone()))),
            target: self.emit_type(&i.self_ty),
            generics: i
                .generics
                .as_ref()
                .map(|g| g.params.iter().map(|p| self.emit_generic_param(p)).collect())
                .unwrap_or_default(),
            methods: i
                .items
                .iter()
                .filter_map(|item| {
                    if let ImplItem::Function(f) = item {
                        Some(self.emit_function(f))
                    } else {
                        None
                    }
                })
                .collect(),
        }
    }

    /// Emit const definition
    fn emit_const(&mut self, c: &ConstDef) -> IrConst {
        IrConst {
            name: c.name.name.clone(),
            visibility: Some(self.emit_visibility(&c.visibility)),
            ty: self.emit_type(&c.ty),
            value: self.emit_expr(&c.value),
        }
    }

    /// Emit static definition
    fn emit_static(&mut self, s: &StaticDef) -> IrStatic {
        IrStatic {
            name: s.name.name.clone(),
            visibility: Some(self.emit_visibility(&s.visibility)),
            mutable: s.mutable,
            ty: self.emit_type(&s.ty),
            value: self.emit_expr(&s.value),
        }
    }

    /// Emit type alias
    fn emit_type_alias(&mut self, t: &TypeAlias) -> IrTypeAlias {
        IrTypeAlias {
            name: t.name.name.clone(),
            visibility: Some(self.emit_visibility(&t.visibility)),
            generics: t
                .generics
                .as_ref()
                .map(|g| g.params.iter().map(|p| self.emit_generic_param(p)).collect())
                .unwrap_or_default(),
            target: self.emit_type(&t.ty),
        }
    }

    /// Emit actor definition
    fn emit_actor(&mut self, a: &ActorDef) -> IrActor {
        IrActor {
            name: a.name.name.clone(),
            visibility: Some(self.emit_visibility(&a.visibility)),
            state: a.state.iter().map(|f| self.emit_field_def(f)).collect(),
            handlers: a
                .handlers
                .iter()
                .map(|h| IrHandler {
                    message: h.message.name.clone(),
                    params: h.params.iter().map(|p| self.emit_param(p)).collect(),
                    return_type: h.return_type.as_ref().map(|t| self.emit_type(t)),
                    body: self.emit_block(&h.body),
                })
                .collect(),
        }
    }

    /// Serialize to JSON string
    pub fn to_json(&self, doc: &IrDocument) -> Result<String, serde_json::Error> {
        if self.options.pretty {
            serde_json::to_string_pretty(doc)
        } else {
            serde_json::to_string(doc)
        }
    }
}

impl Default for IrEmitter {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Public API
// ============================================================================

/// Emit IR from a source file
pub fn emit_ir(source: &SourceFile, filename: &str, options: IrEmitterOptions) -> IrDocument {
    let mut emitter = IrEmitter::with_options(options);
    emitter.emit(source, filename)
}

/// Emit IR to JSON string
pub fn emit_ir_json(
    source: &SourceFile,
    filename: &str,
    pretty: bool,
) -> Result<String, serde_json::Error> {
    let options = IrEmitterOptions {
        include_types: true,
        include_spans: true,
        pretty,
    };
    let mut emitter = IrEmitter::with_options(options);
    let doc = emitter.emit(source, filename);
    emitter.to_json(&doc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Parser;

    #[test]
    fn test_emit_simple_function() {
        let source = r#"
            fn add(a: i32, b: i32) -> i32 {
                a + b
            }
        "#;
        let mut parser = Parser::new(source);
        let ast = parser.parse_file().unwrap();

        let json = emit_ir_json(&ast, "test.sigil", true).unwrap();
        assert!(json.contains("\"name\": \"add\""));
        assert!(json.contains("\"kind\": \"function\""));
    }
}
