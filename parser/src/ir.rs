//! AI-Facing Intermediate Representation (IR)
//!
//! This module provides a JSON-serializable IR designed for consumption by
//! AI agents, language servers, and external tooling. The IR captures:
//!
//! - Function definitions with typed parameters and bodies
//! - Pipeline operations (morphemes, transformations, forks)
//! - Evidentiality annotations throughout
//! - Type information including the evidentiality lattice
//! - Control flow and data flow structures
//! - Protocol operations with their trust boundaries

use serde::{Deserialize, Serialize};

/// The version of the IR schema
pub const IR_VERSION: &str = "1.0.0";

/// Top-level IR module representing a complete Sigil program
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrModule {
    /// Schema version
    pub version: String,
    /// Source file path
    pub source: String,
    /// All function definitions
    pub functions: Vec<IrFunction>,
    /// Type definitions (structs, enums)
    pub types: Vec<IrTypeDef>,
    /// Trait definitions
    pub traits: Vec<IrTraitDef>,
    /// Impl blocks
    pub impls: Vec<IrImplBlock>,
    /// Constants
    pub constants: Vec<IrConstant>,
    /// The evidentiality lattice structure
    pub evidentiality_lattice: EvidentialityLattice,
}

impl IrModule {
    /// Create a new empty IR module
    pub fn new(source: String) -> Self {
        Self {
            version: IR_VERSION.to_string(),
            source,
            functions: Vec::new(),
            types: Vec::new(),
            traits: Vec::new(),
            impls: Vec::new(),
            constants: Vec::new(),
            evidentiality_lattice: EvidentialityLattice::default(),
        }
    }
}

/// The evidentiality lattice with explicit join/meet rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidentialityLattice {
    /// The four evidence levels in order
    pub levels: Vec<EvidenceLevel>,
    /// Join rules (pessimistic combination)
    pub join_rules: Vec<LatticeRule>,
    /// Meet rules (optimistic combination)
    pub meet_rules: Vec<LatticeRule>,
}

impl Default for EvidentialityLattice {
    fn default() -> Self {
        Self {
            levels: vec![
                EvidenceLevel {
                    name: "known".to_string(),
                    symbol: "!".to_string(),
                    order: 0,
                    description: "Direct computation, verified".to_string(),
                },
                EvidenceLevel {
                    name: "uncertain".to_string(),
                    symbol: "?".to_string(),
                    order: 1,
                    description: "Inferred, possibly absent".to_string(),
                },
                EvidenceLevel {
                    name: "reported".to_string(),
                    symbol: "~".to_string(),
                    order: 2,
                    description: "External source, untrusted".to_string(),
                },
                EvidenceLevel {
                    name: "paradox".to_string(),
                    symbol: "‽".to_string(),
                    order: 3,
                    description: "Contradictory, trust boundary".to_string(),
                },
            ],
            join_rules: vec![
                LatticeRule::new("known", "uncertain", "uncertain"),
                LatticeRule::new("known", "reported", "reported"),
                LatticeRule::new("known", "paradox", "paradox"),
                LatticeRule::new("uncertain", "reported", "reported"),
                LatticeRule::new("uncertain", "paradox", "paradox"),
                LatticeRule::new("reported", "paradox", "paradox"),
            ],
            meet_rules: vec![
                LatticeRule::new("known", "uncertain", "known"),
                LatticeRule::new("known", "reported", "known"),
                LatticeRule::new("known", "paradox", "known"),
                LatticeRule::new("uncertain", "reported", "uncertain"),
                LatticeRule::new("uncertain", "paradox", "uncertain"),
                LatticeRule::new("reported", "paradox", "reported"),
            ],
        }
    }
}

/// An evidence level in the lattice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceLevel {
    pub name: String,
    pub symbol: String,
    pub order: u8,
    pub description: String,
}

/// A join/meet rule in the lattice
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatticeRule {
    pub left: String,
    pub right: String,
    pub result: String,
}

impl LatticeRule {
    pub fn new(left: &str, right: &str, result: &str) -> Self {
        Self {
            left: left.to_string(),
            right: right.to_string(),
            result: result.to_string(),
        }
    }
}

/// Evidence marker enum matching AST Evidentiality
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IrEvidence {
    Known,
    Uncertain,
    Reported,
    Paradox,
}

impl IrEvidence {
    /// Join two evidence levels (pessimistic - takes the higher/less certain)
    pub fn join(self, other: IrEvidence) -> IrEvidence {
        match (self, other) {
            (IrEvidence::Paradox, _) | (_, IrEvidence::Paradox) => IrEvidence::Paradox,
            (IrEvidence::Reported, _) | (_, IrEvidence::Reported) => IrEvidence::Reported,
            (IrEvidence::Uncertain, _) | (_, IrEvidence::Uncertain) => IrEvidence::Uncertain,
            (IrEvidence::Known, IrEvidence::Known) => IrEvidence::Known,
        }
    }

    /// Meet two evidence levels (optimistic - takes the lower/more certain)
    pub fn meet(self, other: IrEvidence) -> IrEvidence {
        match (self, other) {
            (IrEvidence::Known, _) | (_, IrEvidence::Known) => IrEvidence::Known,
            (IrEvidence::Uncertain, _) | (_, IrEvidence::Uncertain) => IrEvidence::Uncertain,
            (IrEvidence::Reported, _) | (_, IrEvidence::Reported) => IrEvidence::Reported,
            (IrEvidence::Paradox, IrEvidence::Paradox) => IrEvidence::Paradox,
        }
    }

    /// Get the symbol for this evidence level
    pub fn symbol(&self) -> &'static str {
        match self {
            IrEvidence::Known => "!",
            IrEvidence::Uncertain => "?",
            IrEvidence::Reported => "~",
            IrEvidence::Paradox => "‽",
        }
    }
}

impl Default for IrEvidence {
    fn default() -> Self {
        IrEvidence::Known
    }
}

/// Source location span
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrSpan {
    pub start: IrPosition,
    pub end: IrPosition,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrPosition {
    pub line: usize,
    pub column: usize,
}

/// Function visibility
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IrVisibility {
    Public,
    Private,
    Crate,
}

impl Default for IrVisibility {
    fn default() -> Self {
        IrVisibility::Private
    }
}

/// A function definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrFunction {
    /// Function name
    pub name: String,
    /// Unique identifier
    pub id: String,
    /// Visibility
    pub visibility: IrVisibility,
    /// Generic type parameters
    pub generics: Vec<IrGenericParam>,
    /// Parameters
    pub params: Vec<IrParam>,
    /// Return type
    pub return_type: IrType,
    /// Function body
    pub body: Option<IrOperation>,
    /// Attributes (inline, pure, async, etc.)
    pub attributes: Vec<String>,
    /// Whether this is an async function
    pub is_async: bool,
    /// Source span
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span: Option<IrSpan>,
}

/// A generic type parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrGenericParam {
    pub name: String,
    pub bounds: Vec<String>,
}

/// A function parameter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrParam {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: IrType,
    pub evidence: IrEvidence,
}

/// Type representation in IR
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum IrType {
    /// Primitive types (i32, f64, bool, str, etc.)
    #[serde(rename = "primitive")]
    Primitive { name: String },

    /// Unit type ()
    #[serde(rename = "unit")]
    Unit,

    /// Never type !
    #[serde(rename = "never")]
    Never,

    /// Array type [T; N]
    #[serde(rename = "array")]
    Array {
        element: Box<IrType>,
        size: Option<usize>,
    },

    /// Slice type [T]
    #[serde(rename = "slice")]
    Slice { element: Box<IrType> },

    /// Tuple type (A, B, C)
    #[serde(rename = "tuple")]
    Tuple { elements: Vec<IrType> },

    /// Named struct/enum type
    #[serde(rename = "named")]
    Named {
        name: String,
        generics: Vec<IrType>,
    },

    /// Reference &T or &mut T
    #[serde(rename = "reference")]
    Reference {
        mutable: bool,
        inner: Box<IrType>,
    },

    /// Pointer *const T or *mut T
    #[serde(rename = "pointer")]
    Pointer {
        mutable: bool,
        inner: Box<IrType>,
    },

    /// Function type fn(A, B) -> C
    #[serde(rename = "function")]
    Function {
        params: Vec<IrType>,
        return_type: Box<IrType>,
        is_async: bool,
    },

    /// Evidential wrapper E<T> with evidence level
    #[serde(rename = "evidential")]
    Evidential {
        inner: Box<IrType>,
        evidence: IrEvidence,
    },

    /// Cyclic arithmetic type Z/nZ
    #[serde(rename = "cycle")]
    Cycle { modulus: u64 },

    /// SIMD vector type
    #[serde(rename = "simd")]
    Simd {
        element: Box<IrType>,
        lanes: usize,
    },

    /// Atomic type
    #[serde(rename = "atomic")]
    Atomic { inner: Box<IrType> },

    /// Type variable (for inference)
    #[serde(rename = "var")]
    Var { id: String },

    /// Infer placeholder _
    #[serde(rename = "infer")]
    Infer,

    /// Error recovery type
    #[serde(rename = "error")]
    Error,
}

impl Default for IrType {
    fn default() -> Self {
        IrType::Unit
    }
}

/// Type definition (struct, enum)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum IrTypeDef {
    #[serde(rename = "struct_def")]
    Struct {
        name: String,
        generics: Vec<IrGenericParam>,
        fields: Vec<IrField>,
        #[serde(skip_serializing_if = "Option::is_none")]
        span: Option<IrSpan>,
    },
    #[serde(rename = "enum_def")]
    Enum {
        name: String,
        generics: Vec<IrGenericParam>,
        variants: Vec<IrVariant>,
        #[serde(skip_serializing_if = "Option::is_none")]
        span: Option<IrSpan>,
    },
    #[serde(rename = "type_alias")]
    TypeAlias {
        name: String,
        generics: Vec<IrGenericParam>,
        target: IrType,
        #[serde(skip_serializing_if = "Option::is_none")]
        span: Option<IrSpan>,
    },
}

/// A struct field
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrField {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: IrType,
    pub visibility: IrVisibility,
}

/// An enum variant
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrVariant {
    pub name: String,
    pub fields: Option<Vec<IrField>>,
    pub discriminant: Option<i64>,
}

/// Trait definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrTraitDef {
    pub name: String,
    pub generics: Vec<IrGenericParam>,
    pub super_traits: Vec<String>,
    pub methods: Vec<IrFunction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span: Option<IrSpan>,
}

/// Impl block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrImplBlock {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trait_name: Option<String>,
    pub target_type: IrType,
    pub generics: Vec<IrGenericParam>,
    pub methods: Vec<IrFunction>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span: Option<IrSpan>,
}

/// Constant definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrConstant {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: IrType,
    pub value: IrOperation,
    pub visibility: IrVisibility,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub span: Option<IrSpan>,
}

/// Core operation node - the heart of the IR
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum IrOperation {
    // === Literals ===
    #[serde(rename = "literal")]
    Literal {
        variant: LiteralVariant,
        value: serde_json::Value,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    // === Variables ===
    #[serde(rename = "var")]
    Var {
        name: String,
        id: String,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    // === Bindings ===
    #[serde(rename = "let")]
    Let {
        pattern: IrPattern,
        #[serde(skip_serializing_if = "Option::is_none")]
        type_annotation: Option<IrType>,
        init: Box<IrOperation>,
        evidence: IrEvidence,
    },

    // === Binary operations ===
    #[serde(rename = "binary")]
    Binary {
        operator: BinaryOp,
        left: Box<IrOperation>,
        right: Box<IrOperation>,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    // === Unary operations ===
    #[serde(rename = "unary")]
    Unary {
        operator: UnaryOp,
        operand: Box<IrOperation>,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    // === Function calls ===
    #[serde(rename = "call")]
    Call {
        function: String,
        function_id: String,
        args: Vec<IrOperation>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        type_args: Vec<IrType>,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    #[serde(rename = "method_call")]
    MethodCall {
        receiver: Box<IrOperation>,
        method: String,
        args: Vec<IrOperation>,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        type_args: Vec<IrType>,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    // === Closures ===
    #[serde(rename = "closure")]
    Closure {
        params: Vec<IrParam>,
        body: Box<IrOperation>,
        captures: Vec<String>,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    // === Control flow ===
    #[serde(rename = "if")]
    If {
        condition: Box<IrOperation>,
        then_branch: Box<IrOperation>,
        #[serde(skip_serializing_if = "Option::is_none")]
        else_branch: Option<Box<IrOperation>>,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    #[serde(rename = "match")]
    Match {
        scrutinee: Box<IrOperation>,
        arms: Vec<IrMatchArm>,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    #[serde(rename = "loop")]
    Loop {
        variant: LoopVariant,
        #[serde(skip_serializing_if = "Option::is_none")]
        condition: Option<Box<IrOperation>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        iterator: Option<IrForIterator>,
        body: Box<IrOperation>,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    #[serde(rename = "break")]
    Break {
        #[serde(skip_serializing_if = "Option::is_none")]
        value: Option<Box<IrOperation>>,
        evidence: IrEvidence,
    },

    #[serde(rename = "continue")]
    Continue { evidence: IrEvidence },

    #[serde(rename = "return")]
    Return {
        #[serde(skip_serializing_if = "Option::is_none")]
        value: Option<Box<IrOperation>>,
        evidence: IrEvidence,
    },

    // === Blocks ===
    #[serde(rename = "block")]
    Block {
        statements: Vec<IrOperation>,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    // === Pipeline operations (core Sigil feature) ===
    #[serde(rename = "pipeline")]
    Pipeline {
        input: Box<IrOperation>,
        steps: Vec<IrPipelineStep>,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    // === Morpheme operations ===
    #[serde(rename = "morpheme")]
    Morpheme {
        morpheme: MorphemeKind,
        symbol: String,
        input: Box<IrOperation>,
        #[serde(skip_serializing_if = "Option::is_none")]
        body: Option<Box<IrOperation>>,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    // === Fork/Join ===
    #[serde(rename = "fork")]
    Fork {
        branches: Vec<IrOperation>,
        join_strategy: JoinStrategy,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    #[serde(rename = "identity")]
    Identity {
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    // === Data structures ===
    #[serde(rename = "array")]
    Array {
        elements: Vec<IrOperation>,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    #[serde(rename = "tuple")]
    Tuple {
        elements: Vec<IrOperation>,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    #[serde(rename = "struct_init")]
    StructInit {
        name: String,
        fields: Vec<(String, IrOperation)>,
        #[serde(skip_serializing_if = "Option::is_none")]
        rest: Option<Box<IrOperation>>,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    // === Field/Index access ===
    #[serde(rename = "field")]
    Field {
        expr: Box<IrOperation>,
        field: String,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    #[serde(rename = "index")]
    Index {
        expr: Box<IrOperation>,
        index: Box<IrOperation>,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    // === Assignment ===
    #[serde(rename = "assign")]
    Assign {
        target: Box<IrOperation>,
        value: Box<IrOperation>,
        evidence: IrEvidence,
    },

    // === Evidentiality operations ===
    #[serde(rename = "evidence_coerce")]
    EvidenceCoerce {
        operation: EvidenceOp,
        expr: Box<IrOperation>,
        from_evidence: IrEvidence,
        to_evidence: IrEvidence,
        #[serde(rename = "type")]
        ty: IrType,
    },

    // === Incorporation (noun-verb fusion) ===
    #[serde(rename = "incorporation")]
    Incorporation {
        segments: Vec<IncorporationSegment>,
        args: Vec<IrOperation>,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    // === Affect (emotional markers) ===
    #[serde(rename = "affect")]
    Affect {
        expr: Box<IrOperation>,
        affect: IrAffect,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    // === Protocol operations ===
    #[serde(rename = "http_request")]
    HttpRequest {
        method: HttpMethod,
        url: Box<IrOperation>,
        #[serde(skip_serializing_if = "Option::is_none")]
        headers: Option<Box<IrOperation>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        body: Option<Box<IrOperation>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timeout: Option<Box<IrOperation>>,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    #[serde(rename = "grpc_call")]
    GrpcCall {
        service: String,
        method: String,
        message: Box<IrOperation>,
        #[serde(skip_serializing_if = "Option::is_none")]
        metadata: Option<Box<IrOperation>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        timeout: Option<Box<IrOperation>>,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    #[serde(rename = "websocket")]
    WebSocket {
        operation: WebSocketOp,
        url: Box<IrOperation>,
        #[serde(skip_serializing_if = "Option::is_none")]
        message: Option<Box<IrOperation>>,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    #[serde(rename = "kafka_op")]
    KafkaOp {
        operation: KafkaOpKind,
        topic: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        payload: Option<Box<IrOperation>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        key: Option<Box<IrOperation>>,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    // === Async ===
    #[serde(rename = "await")]
    Await {
        expr: Box<IrOperation>,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    // === Unsafe ===
    #[serde(rename = "unsafe")]
    Unsafe {
        body: Box<IrOperation>,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    // === Cast ===
    #[serde(rename = "cast")]
    Cast {
        expr: Box<IrOperation>,
        target_type: IrType,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },

    // === Try (? operator) ===
    #[serde(rename = "try")]
    Try {
        expr: Box<IrOperation>,
        #[serde(rename = "type")]
        ty: IrType,
        evidence: IrEvidence,
    },
}

/// Literal variants
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LiteralVariant {
    Int,
    Float,
    Bool,
    Char,
    String,
    Null,
}

/// Binary operators
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Rem,
    Pow,
    And,
    Or,
    BitAnd,
    BitOr,
    BitXor,
    Shl,
    Shr,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Concat,
}

/// Unary operators
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UnaryOp {
    Neg,
    Not,
    Deref,
    Ref,
    RefMut,
}

/// Loop variants
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LoopVariant {
    Infinite,
    While,
    For,
}

/// For loop iterator
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrForIterator {
    pub pattern: IrPattern,
    pub iterable: Box<IrOperation>,
}

/// Match arm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrMatchArm {
    pub pattern: IrPattern,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub guard: Option<IrOperation>,
    pub body: IrOperation,
}

/// Pattern in IR
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum IrPattern {
    #[serde(rename = "ident")]
    Ident {
        name: String,
        mutable: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        evidence: Option<IrEvidence>,
    },

    #[serde(rename = "tuple")]
    Tuple { elements: Vec<IrPattern> },

    #[serde(rename = "struct")]
    Struct {
        path: String,
        fields: Vec<(String, IrPattern)>,
        rest: bool,
    },

    #[serde(rename = "tuple_struct")]
    TupleStruct {
        path: String,
        fields: Vec<IrPattern>,
    },

    #[serde(rename = "slice")]
    Slice { elements: Vec<IrPattern> },

    #[serde(rename = "or")]
    Or { patterns: Vec<IrPattern> },

    #[serde(rename = "literal")]
    Literal { value: serde_json::Value },

    #[serde(rename = "range")]
    Range {
        #[serde(skip_serializing_if = "Option::is_none")]
        start: Option<Box<IrPattern>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        end: Option<Box<IrPattern>>,
        inclusive: bool,
    },

    #[serde(rename = "wildcard")]
    Wildcard,
}

/// Pipeline step
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum IrPipelineStep {
    #[serde(rename = "call")]
    Call {
        #[serde(rename = "fn")]
        function: String,
        args: Vec<IrOperation>,
    },

    #[serde(rename = "morpheme")]
    Morpheme {
        morpheme: MorphemeKind,
        symbol: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        body: Option<Box<IrOperation>>,
    },

    #[serde(rename = "fork")]
    Fork { branches: Vec<IrPipelineStep> },

    #[serde(rename = "identity")]
    Identity,

    #[serde(rename = "method")]
    Method {
        name: String,
        args: Vec<IrOperation>,
    },

    #[serde(rename = "await")]
    Await,

    #[serde(rename = "protocol")]
    Protocol {
        operation: ProtocolOp,
        #[serde(skip_serializing_if = "Option::is_none")]
        config: Option<Box<IrOperation>>,
    },
}

/// Morpheme kinds
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MorphemeKind {
    Transform,
    Filter,
    Sort,
    Reduce,
    Lambda,
    Sum,
    Product,
    First,
    Last,
    Middle,
    Choice,
    Nth,
    Next,
}

impl MorphemeKind {
    pub fn symbol(&self) -> &'static str {
        match self {
            MorphemeKind::Transform => "τ",
            MorphemeKind::Filter => "φ",
            MorphemeKind::Sort => "σ",
            MorphemeKind::Reduce => "ρ",
            MorphemeKind::Lambda => "λ",
            MorphemeKind::Sum => "Σ",
            MorphemeKind::Product => "Π",
            MorphemeKind::First => "α",
            MorphemeKind::Last => "ω",
            MorphemeKind::Middle => "μ",
            MorphemeKind::Choice => "χ",
            MorphemeKind::Nth => "ν",
            MorphemeKind::Next => "ξ",
        }
    }
}

/// Join strategy for fork operations
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JoinStrategy {
    Tuple,
    First,
    All,
    Race,
}

/// Evidence operations
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EvidenceOp {
    Trust,
    Verify,
    Mark,
}

/// Incorporation segment (noun-verb)
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum IncorporationSegment {
    #[serde(rename = "noun")]
    Noun { name: String },
    #[serde(rename = "verb")]
    Verb { name: String },
}

/// Affect markers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrAffect {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sentiment: Option<Sentiment>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intensity: Option<Intensity>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub formality: Option<Formality>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub emotion: Option<Emotion>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence: Option<Confidence>,
    #[serde(default)]
    pub sarcasm: bool,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Sentiment {
    Positive,
    Negative,
    Neutral,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Intensity {
    Up,
    Down,
    Max,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Formality {
    Formal,
    Informal,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Emotion {
    Joy,
    Sadness,
    Anger,
    Fear,
    Surprise,
    Love,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Confidence {
    High,
    Medium,
    Low,
}

/// HTTP methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum HttpMethod {
    Get,
    Post,
    Put,
    Delete,
    Patch,
    Head,
    Options,
    Connect,
    Trace,
}

/// WebSocket operations
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WebSocketOp {
    Connect,
    Send,
    Receive,
    Close,
}

/// Kafka operations
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KafkaOpKind {
    Produce,
    Consume,
    Subscribe,
    Unsubscribe,
    Commit,
    Seek,
}

/// Protocol operations for pipeline
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ProtocolOp {
    Send,
    Recv,
    Stream,
    Connect,
    Close,
    Timeout,
    Retry,
}

/// Configuration options for IR dump
#[derive(Debug, Clone, Default)]
pub struct IrDumpOptions {
    /// Pretty-print JSON output
    pub pretty: bool,
    /// Include full type information on every node
    pub full_types: bool,
    /// Include source spans
    pub include_spans: bool,
    /// Optimization level to apply before dumping
    pub opt_level: Option<u8>,
}

impl IrModule {
    /// Serialize to JSON string
    pub fn to_json(&self, pretty: bool) -> Result<String, serde_json::Error> {
        if pretty {
            serde_json::to_string_pretty(self)
        } else {
            serde_json::to_string(self)
        }
    }

    /// Deserialize from JSON string
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evidence_join() {
        assert_eq!(IrEvidence::Known.join(IrEvidence::Known), IrEvidence::Known);
        assert_eq!(
            IrEvidence::Known.join(IrEvidence::Uncertain),
            IrEvidence::Uncertain
        );
        assert_eq!(
            IrEvidence::Known.join(IrEvidence::Reported),
            IrEvidence::Reported
        );
        assert_eq!(
            IrEvidence::Known.join(IrEvidence::Paradox),
            IrEvidence::Paradox
        );
        assert_eq!(
            IrEvidence::Reported.join(IrEvidence::Paradox),
            IrEvidence::Paradox
        );
    }

    #[test]
    fn test_evidence_meet() {
        assert_eq!(
            IrEvidence::Paradox.meet(IrEvidence::Paradox),
            IrEvidence::Paradox
        );
        assert_eq!(
            IrEvidence::Paradox.meet(IrEvidence::Reported),
            IrEvidence::Reported
        );
        assert_eq!(
            IrEvidence::Paradox.meet(IrEvidence::Uncertain),
            IrEvidence::Uncertain
        );
        assert_eq!(
            IrEvidence::Paradox.meet(IrEvidence::Known),
            IrEvidence::Known
        );
    }

    #[test]
    fn test_ir_serialization() {
        let module = IrModule::new("test.sigil".to_string());
        let json = module.to_json(false).unwrap();
        let parsed: IrModule = IrModule::from_json(&json).unwrap();
        assert_eq!(parsed.version, IR_VERSION);
        assert_eq!(parsed.source, "test.sigil");
    }

    #[test]
    fn test_morpheme_symbols() {
        assert_eq!(MorphemeKind::Transform.symbol(), "τ");
        assert_eq!(MorphemeKind::Filter.symbol(), "φ");
        assert_eq!(MorphemeKind::Reduce.symbol(), "ρ");
    }
}
