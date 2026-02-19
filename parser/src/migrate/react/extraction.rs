//! React extraction types.
//!
//! These types represent the extracted structure from React/TSX files.
//! See docs/specs/REACT-MIGRATION.md Section 3.2 for specification.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use swc_common::{SourceMap, FilePathMapping, FileName, Spanned};
use swc_ecma_parser::{parse_file_as_module, Syntax, TsSyntax, EsSyntax};
use swc_ecma_ast::*;

// =============================================================================
// Core Extraction Types
// =============================================================================

/// Complete extraction from a React/TSX file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReactExtraction {
    pub file: FileInfo,
    pub components: Vec<ComponentExtraction>,
    pub custom_hooks: Vec<CustomHookExtraction>,
    pub types: Vec<TypeExtraction>,
    pub imports: Vec<ImportInfo>,
    pub exports: Vec<ExportInfo>,
    /// Helper functions at module scope (Phase 6.2)
    #[serde(default)]
    pub helper_functions: Vec<HelperFunctionExtraction>,
}

/// File metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub path: PathBuf,
    pub relative_path: String,
    pub language: Language,
    pub has_jsx: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    TypeScript,
    JavaScript,
}

// =============================================================================
// Component Extraction
// =============================================================================

/// Extracted React component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentExtraction {
    pub name: String,
    pub component_type: ComponentType,
    pub exported: bool,
    pub export_type: Option<ExportType>,
    pub location: SourceLocation,

    // Props
    pub props: Vec<PropExtraction>,
    pub props_type: Option<String>,

    // Hooks (functional components)
    pub hooks: Vec<HookUsage>,

    // Custom hooks (useChat, useAgent, etc.)
    pub custom_hooks: Vec<CustomHookUsage>,

    // Class components
    pub class_info: Option<ClassComponentInfo>,

    // JSX structure
    pub jsx: JsxTree,

    // Event handlers
    pub handlers: Vec<HandlerExtraction>,

    // Dependencies
    pub child_components: Vec<String>,

    /// Architecture recommendations (Phase 6.5)
    #[serde(default)]
    pub architecture: ArchitectureRecommendation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ComponentType {
    Functional,
    Class,
    ForwardRef,
    Memo,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExportType {
    Default,
    Named,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceLocation {
    pub start_line: u32,
    pub start_col: u32,
    pub end_line: u32,
    pub end_col: u32,
}

// =============================================================================
// Props Extraction
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropExtraction {
    pub name: String,
    pub type_annotation: Option<String>,
    pub required: bool,
    pub default_value: Option<String>,
    pub is_callback: bool,
    pub is_children: bool,
}

// =============================================================================
// Hook Extraction
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookUsage {
    pub hook_type: HookType,
    pub location: SourceLocation,

    // useState specific
    pub state_name: Option<String>,
    pub setter_name: Option<String>,
    pub initial_value: Option<String>,

    // useEffect specific
    pub dependencies: Option<Vec<String>>,
    pub has_cleanup: bool,

    // useCallback/useMemo specific
    pub memoized_deps: Option<Vec<String>>,

    // useRef specific
    pub ref_name: Option<String>,
    pub ref_type: Option<String>,

    // useContext specific
    pub context_name: Option<String>,

    // useReducer specific
    pub reducer_name: Option<String>,
    pub action_types: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum HookType {
    UseState,
    UseEffect,
    UseCallback,
    UseMemo,
    UseRef,
    UseContext,
    UseReducer,
    UseLayoutEffect,
    UseImperativeHandle,
    Custom(u32), // Index into custom_hooks
}

/// Extraction for custom hook definitions (defining a hook)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomHookExtraction {
    pub name: String,
    pub location: SourceLocation,
    pub parameters: Vec<String>,
    pub return_type: Option<String>,
    pub hooks_used: Vec<HookType>,
}

/// Extraction for custom hook usage (calling a hook like useChat, useAgent)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomHookUsage {
    /// Hook name (e.g., "useChat", "useAgent", "useInfernumStore")
    pub name: String,
    pub location: SourceLocation,
    /// Arguments passed to the hook (raw strings for backward compat)
    pub arguments: Vec<String>,
    /// Phase 6.4: Expanded argument information
    #[serde(default)]
    pub expanded_arguments: Vec<HookArgument>,
    /// Values destructured from the hook return
    /// e.g., { messages, isStreaming, addMessage } from useChat()
    pub returned_values: Vec<CustomHookReturnValue>,
    /// Whether the hook is a Zustand store selector
    pub is_zustand: bool,
}

/// Phase 6.4: Expanded hook argument with full structure
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum HookArgument {
    /// Simple expression (identifier, literal, etc.)
    Expression {
        value: String,
    },
    /// Object argument with properties (e.g., { onComplete: ..., options: ... })
    Object {
        properties: Vec<HookObjectProperty>,
    },
    /// Array argument
    Array {
        elements: Vec<String>,
    },
    /// Arrow/function expression
    Function {
        params: Vec<String>,
        body_summary: String,
        /// Calls made within the callback
        calls: Vec<HandlerCall>,
        /// Side effects in the callback
        side_effects: Vec<SideEffect>,
    },
}

/// A property in a hook object argument
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HookObjectProperty {
    /// Property name (e.g., "onComplete", "enabled")
    pub name: String,
    /// Value type/kind
    pub value_kind: HookPropertyValue,
}

/// Value of a property in a hook object argument
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum HookPropertyValue {
    /// Simple value (literal, identifier)
    Simple { value: String },
    /// Callback function
    Callback {
        params: Vec<String>,
        body_summary: String,
        calls: Vec<HandlerCall>,
        side_effects: Vec<SideEffect>,
    },
    /// Nested object
    Object { properties: Vec<HookObjectProperty> },
    /// Array value
    Array { elements: Vec<String> },
}

/// A value returned from a custom hook
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomHookReturnValue {
    /// The local variable name (e.g., "messages", "isStreaming")
    pub name: String,
    /// Whether this is a function (callback/setter) vs data
    pub is_function: bool,
    /// Original name if renamed (e.g., events: agentEvents → original: "events")
    pub original_name: Option<String>,
}

// =============================================================================
// Class Component Info
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassComponentInfo {
    pub state_type: Option<String>,
    pub state_initializer: Option<String>,
    pub lifecycle_methods: Vec<LifecycleMethod>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum LifecycleMethod {
    ComponentDidMount,
    ComponentDidUpdate,
    ComponentWillUnmount,
    ShouldComponentUpdate,
    GetDerivedStateFromProps,
    GetSnapshotBeforeUpdate,
    ComponentDidCatch,
}

// =============================================================================
// JSX Tree
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsxTree {
    pub root: Option<JsxNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsxNode {
    pub node_type: JsxNodeType,
    pub location: SourceLocation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum JsxNodeType {
    Element {
        tag: String,
        is_component: bool,
        attributes: Vec<JsxAttribute>,
        children: Vec<JsxNode>,
    },
    Fragment {
        children: Vec<JsxNode>,
    },
    Expression {
        code: String,
    },
    Text {
        value: String,
    },
    Conditional {
        condition: String,
        consequent: Box<JsxNode>,
        alternate: Option<Box<JsxNode>>,
    },
    Map {
        iterable: String,
        item_name: String,
        key_expr: Option<String>,
        body: Box<JsxNode>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsxAttribute {
    pub name: String,
    pub value: JsxAttributeValue,
    pub is_event_handler: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum JsxAttributeValue {
    String { value: String },
    Expression { code: String },
    Spread { name: String },
    True, // shorthand: <button disabled />
}

// =============================================================================
// Handler Extraction (Phase 6.3 - Handler Body Analysis)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandlerExtraction {
    pub name: String,
    pub event_type: Option<String>,
    pub is_async: bool,
    pub body_summary: String,
    /// Legacy: simple state mutation names
    pub state_mutations: Vec<String>,
    /// Legacy: simple API call names
    pub api_calls: Vec<String>,
    /// Phase 6.3: Detailed function calls with source tracking
    #[serde(default)]
    pub calls: Vec<HandlerCall>,
    /// Phase 6.3: Side effects detected in handler body
    #[serde(default)]
    pub side_effects: Vec<SideEffect>,
    /// Phase 6.3: Parameters with types
    #[serde(default)]
    pub parameters: Vec<HandlerParam>,
    /// Phase 6.3: Conditional branches detected
    #[serde(default)]
    pub has_conditionals: bool,
    /// Phase 6.3: Early returns detected
    #[serde(default)]
    pub has_early_return: bool,
}

/// A function call within a handler with source tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandlerCall {
    /// Name of the function being called
    pub name: String,
    /// Where this function comes from
    pub source: CallSource,
    /// Arguments passed (as source strings)
    pub arguments: Vec<String>,
    /// Whether this is an async call (awaited)
    pub is_async: bool,
}

/// Source of a function call within a handler
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CallSource {
    /// From a custom hook (e.g., addMessage from useChat)
    Hook { hook_name: String },
    /// From component props (e.g., onClick, onSubmit)
    Prop { prop_name: String },
    /// From a useState setter (e.g., setCount)
    StateSetter { state_name: String },
    /// Local helper function defined in same file
    LocalHelper { function_name: String },
    /// Imported function
    Import { module: String },
    /// Global/window function
    Global,
    /// Unknown source
    Unknown,
}

/// Handler parameter with type information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandlerParam {
    pub name: String,
    pub type_annotation: Option<String>,
}

// =============================================================================
// Helper Function Extraction (Phase 6.2)
// =============================================================================

/// Extracted helper function at module or component scope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelperFunctionExtraction {
    pub name: String,
    pub location: SourceLocation,
    pub exported: bool,
    /// Whether the function is async
    pub is_async: bool,
    /// Whether the function is a generator
    pub is_generator: bool,
    /// Function parameters with types
    pub parameters: Vec<FunctionParamExtraction>,
    /// Return type annotation if present
    pub return_type: Option<String>,
    /// Whether the function appears to be pure (no side effects detected)
    pub is_pure: bool,
    /// Side effects detected in the function body
    pub side_effects: Vec<SideEffect>,
    /// Other functions called within this function
    pub calls: Vec<String>,
    /// Components that use this helper function
    pub used_by: Vec<String>,
    /// Full source code
    pub source: String,
}

/// Function parameter with type information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionParamExtraction {
    pub name: String,
    pub type_annotation: Option<String>,
    pub optional: bool,
    pub default_value: Option<String>,
    pub is_rest: bool,
}

/// Detected side effect in a function
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum SideEffect {
    /// Mutation of external state (setState, dispatch, etc.)
    StateMutation { target: String },
    /// API/network call
    ApiCall { method: String },
    /// DOM manipulation
    DomMutation { operation: String },
    /// Console logging
    ConsoleLog,
    /// Writes to storage (localStorage, sessionStorage)
    StorageWrite { storage_type: String },
    /// Timer operations (setTimeout, setInterval)
    Timer { operation: String },
    /// Unknown side effect
    Unknown { description: String },
}

// =============================================================================
// Type Extraction (Phase 6.1 - Full Type Extraction)
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeExtraction {
    pub name: String,
    pub kind: TypeKind,
    pub location: SourceLocation,
    pub exported: bool,
    /// Full source definition (for reference)
    pub definition: String,
    /// Structured field information (sufficient for Qliphoth generation)
    pub fields: Vec<TypeFieldExtraction>,
    /// Type parameters (generics): e.g., ["T", "K extends string"]
    pub type_params: Vec<TypeParamExtraction>,
    /// Extended interfaces (for interface extends)
    pub extends: Vec<String>,
    /// Union variants (for type aliases that are unions)
    pub union_variants: Vec<String>,
    /// Doc comment if present
    pub doc_comment: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeFieldExtraction {
    pub name: String,
    /// Full type as string (e.g., "string", "'user' | 'admin'", "(e: Event) => void")
    pub type_annotation: String,
    /// Whether the field is optional (has ?)
    pub optional: bool,
    /// Whether this is a readonly field
    pub readonly: bool,
    /// Parsed type kind for easier mapping
    pub type_kind: TypeFieldKind,
    /// Doc comment for this field
    pub doc_comment: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind")]
pub enum TypeFieldKind {
    /// Primitive: string, number, boolean, null, undefined
    Primitive { name: String },
    /// Reference to another type: ButtonProps, React.ReactNode
    TypeRef { name: String, type_args: Vec<String> },
    /// Array type: string[], Array<T>
    Array { element_type: String },
    /// Union type: 'a' | 'b' | 'c'
    Union { variants: Vec<String> },
    /// Function type: (args) => return
    Function { params: Vec<FunctionParam>, return_type: String },
    /// Object/Record type: { [key: string]: value } or Record<K, V>
    Record { key_type: String, value_type: String },
    /// Tuple type: [string, number]
    Tuple { element_types: Vec<String> },
    /// Literal type: 'user', 42, true
    Literal { value: String },
    /// Complex/unknown type (fallback)
    Complex { raw: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionParam {
    pub name: Option<String>,
    pub type_annotation: String,
    pub optional: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeParamExtraction {
    pub name: String,
    /// Constraint: e.g., "extends string" or "extends keyof T"
    pub constraint: Option<String>,
    /// Default value: e.g., "= unknown"
    pub default: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TypeKind {
    Interface,
    TypeAlias,
    Enum,
}

// =============================================================================
// Architecture Mapping (Phase 6.5)
// =============================================================================

/// Architecture recommendations generated from hook and state analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ArchitectureRecommendation {
    /// Recommended service actors (from custom hooks like useChat, useAgent)
    pub service_actors: Vec<ServiceActorRecommendation>,
    /// State ownership mapping (which actor owns which state)
    pub state_ownership: Vec<StateOwnership>,
    /// Recommended communication patterns between actors
    pub communication_patterns: Vec<CommunicationPattern>,
    /// Zustand stores detected and their mapping
    pub zustand_stores: Vec<ZustandStoreMapping>,
}

/// A recommended service actor derived from hook patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceActorRecommendation {
    /// Suggested actor name (e.g., "ChatService", "AgentService")
    pub name: String,
    /// The hook(s) that suggested this actor
    pub derived_from: Vec<String>,
    /// Actor responsibilities inferred from hook usage
    pub responsibilities: Vec<String>,
    /// State this actor should own
    pub owned_state: Vec<String>,
    /// Messages this actor should handle
    pub messages: Vec<ActorMessage>,
}

/// A message type for actor communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActorMessage {
    /// Message name (e.g., "AddMessage", "RunAgent")
    pub name: String,
    /// Message type classification
    pub message_type: MessageType,
    /// Payload fields
    pub payload: Vec<String>,
    /// Whether this is async (returns a response)
    pub is_async: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MessageType {
    /// Command that mutates state
    Command,
    /// Query that reads state
    Query,
    /// Event notification
    Event,
    /// Request expecting response
    Request,
}

/// Mapping of state to owning actor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateOwnership {
    /// State variable name
    pub state_name: String,
    /// Recommended owning actor
    pub owner: String,
    /// How this state is accessed (local, prop, context)
    pub access_pattern: StateAccessPattern,
    /// Original source (useState, Zustand, context, etc.)
    pub source: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StateAccessPattern {
    /// Local actor state
    Local,
    /// Passed as prop from parent
    Prop,
    /// Shared via context/store
    Shared,
    /// Derived/computed from other state
    Derived,
}

/// Communication pattern between actors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationPattern {
    /// Source actor
    pub from: String,
    /// Target actor
    pub to: String,
    /// Pattern type
    pub pattern: CommunicationPatternType,
    /// Message names involved
    pub messages: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CommunicationPatternType {
    /// Direct message passing
    Direct,
    /// Publish/subscribe
    PubSub,
    /// Request/response
    RequestResponse,
    /// Broadcast to all
    Broadcast,
}

/// Mapping of Zustand store to Qliphoth pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZustandStoreMapping {
    /// Store hook name (e.g., "useInfernumStore")
    pub hook_name: String,
    /// Suggested actor name
    pub suggested_actor: String,
    /// Selectors used
    pub selectors_used: Vec<String>,
    /// Actions/mutations called
    pub actions_used: Vec<String>,
}

// =============================================================================
// Import/Export Info
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportInfo {
    pub source: String,
    pub specifiers: Vec<ImportSpecifier>,
    pub is_type_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportSpecifier {
    pub imported: String,
    pub local: String,
    pub is_default: bool,
    pub is_namespace: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportInfo {
    pub name: String,
    pub is_default: bool,
    pub is_type_only: bool,
    pub source: Option<String>, // re-export
}

// =============================================================================
// Extraction API
// =============================================================================

/// Error type for extraction failures.
#[derive(Debug, thiserror::Error)]
pub enum ExtractionError {
    #[error("Failed to read file: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Failed to parse: {0}")]
    ParseError(String),

    #[error("Unsupported file type: {0}")]
    UnsupportedFileType(String),
}

/// Extract React components from a source file.
pub fn extract_file(path: &std::path::Path, project_root: &std::path::Path) -> Result<ReactExtraction, ExtractionError> {
    let source = std::fs::read_to_string(path)?;
    let relative_path = path.strip_prefix(project_root)
        .unwrap_or(path)
        .to_string_lossy()
        .to_string();

    extract_source(&source, path, &relative_path)
}

/// Extract React components from source code.
pub fn extract_source(source: &str, path: &std::path::Path, relative_path: &str) -> Result<ReactExtraction, ExtractionError> {
    let extension = path.extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    let (language, syntax) = match extension {
        "tsx" => (Language::TypeScript, Syntax::Typescript(TsSyntax {
            tsx: true,
            ..Default::default()
        })),
        "ts" => (Language::TypeScript, Syntax::Typescript(TsSyntax {
            tsx: false,
            ..Default::default()
        })),
        "jsx" => (Language::JavaScript, Syntax::Es(EsSyntax {
            jsx: true,
            ..Default::default()
        })),
        "js" => (Language::JavaScript, Syntax::Es(EsSyntax {
            jsx: true, // Allow JSX in .js files
            ..Default::default()
        })),
        _ => return Err(ExtractionError::UnsupportedFileType(extension.to_string())),
    };

    let cm = SourceMap::new(FilePathMapping::empty());
    let fm = cm.new_source_file(
        FileName::Real(path.to_path_buf()).into(),
        source.to_string(),
    );

    let module = parse_file_as_module(
        &fm,
        syntax,
        Default::default(),
        None,
        &mut vec![],
    ).map_err(|e| ExtractionError::ParseError(format!("{:?}", e)))?;

    let extractor = Extractor::new(&cm, source, language, path.to_path_buf(), relative_path.to_string());
    extractor.extract_module(&module)
}

// =============================================================================
// Extractor Implementation
// =============================================================================

struct Extractor<'a> {
    cm: &'a SourceMap,
    source: &'a str,
    language: Language,
    path: PathBuf,
    relative_path: String,
    has_jsx: bool,
}

impl<'a> Extractor<'a> {
    fn new(cm: &'a SourceMap, source: &'a str, language: Language, path: PathBuf, relative_path: String) -> Self {
        Self {
            cm,
            source,
            language,
            path,
            relative_path,
            has_jsx: false,
        }
    }

    /// Extract source code for a span. This is the key method for preserving expression content.
    fn span_to_source(&self, span: swc_common::Span) -> String {
        // Get byte positions from the span
        let start = span.lo.0 as usize;
        let end = span.hi.0 as usize;

        // Adjust for the source file offset (spans are 1-indexed from file start)
        // swc spans include a base offset, so we need to find the actual position
        let source_file = self.cm.lookup_byte_offset(span.lo);
        let file_start = source_file.sf.start_pos.0 as usize;
        let local_start = start.saturating_sub(file_start);
        let local_end = end.saturating_sub(file_start);

        if local_end <= self.source.len() && local_start <= local_end {
            return self.source[local_start..local_end].to_string();
        }

        // Fallback: try direct byte positions (works for single-file parsing)
        if end <= self.source.len() + 1 && start >= 1 {
            let adjusted_start = start.saturating_sub(1);
            let adjusted_end = end.saturating_sub(1);
            if adjusted_end <= self.source.len() {
                return self.source[adjusted_start..adjusted_end].to_string();
            }
        }

        "/* source unavailable */".to_string()
    }

    /// Get span from an expression (handles Box<Expr>)
    fn expr_span(&self, expr: &Expr) -> swc_common::Span {
        use swc_common::Spanned;
        expr.span()
    }

    fn extract_module(mut self, module: &Module) -> Result<ReactExtraction, ExtractionError> {
        let mut components = Vec::new();
        let mut custom_hooks = Vec::new();
        let mut types = Vec::new();
        let mut imports = Vec::new();
        let mut exports = Vec::new();
        let mut helper_functions = Vec::new();

        for item in &module.body {
            match item {
                ModuleItem::ModuleDecl(decl) => {
                    self.process_module_decl(decl, &mut components, &mut types, &mut imports, &mut exports, &mut helper_functions);
                }
                ModuleItem::Stmt(stmt) => {
                    self.process_stmt(stmt, &mut components, &mut custom_hooks, &mut types, &mut helper_functions);
                }
            }
        }

        Ok(ReactExtraction {
            file: FileInfo {
                path: self.path,
                relative_path: self.relative_path,
                language: self.language,
                has_jsx: self.has_jsx,
            },
            components,
            custom_hooks,
            types,
            imports,
            exports,
            helper_functions,
        })
    }

    fn process_module_decl(
        &mut self,
        decl: &ModuleDecl,
        components: &mut Vec<ComponentExtraction>,
        types: &mut Vec<TypeExtraction>,
        imports: &mut Vec<ImportInfo>,
        exports: &mut Vec<ExportInfo>,
        helper_functions: &mut Vec<HelperFunctionExtraction>,
    ) {
        match decl {
            ModuleDecl::Import(import) => {
                imports.push(self.extract_import(import));
            }
            ModuleDecl::ExportDecl(export) => {
                if let Some(comp) = self.try_extract_component_from_decl(&export.decl, true, Some(ExportType::Named)) {
                    components.push(comp);
                } else if let Some(helper) = self.try_extract_helper_function_from_decl(&export.decl, true) {
                    helper_functions.push(helper);
                }
                if let Some(type_ext) = self.try_extract_type_from_decl(&export.decl, true) {
                    types.push(type_ext);
                }
            }
            ModuleDecl::ExportDefaultDecl(export) => {
                if let Some(comp) = self.try_extract_component_from_default_decl(&export.decl) {
                    let mut comp = comp;
                    comp.exported = true;
                    comp.export_type = Some(ExportType::Default);
                    components.push(comp);
                }
            }
            ModuleDecl::ExportDefaultExpr(export) => {
                if let Some(comp) = self.try_extract_component_from_expr(&export.expr) {
                    let mut comp = comp;
                    comp.exported = true;
                    comp.export_type = Some(ExportType::Default);
                    components.push(comp);
                }
            }
            ModuleDecl::ExportNamed(named) => {
                for spec in &named.specifiers {
                    if let ExportSpecifier::Named(named_spec) = spec {
                        let name = match &named_spec.exported {
                            Some(ModuleExportName::Ident(id)) => id.sym.to_string(),
                            Some(ModuleExportName::Str(s)) => s.value.as_str().unwrap_or("").to_string(),
                            None => match &named_spec.orig {
                                ModuleExportName::Ident(id) => id.sym.to_string(),
                                ModuleExportName::Str(s) => s.value.as_str().unwrap_or("").to_string(),
                            }
                        };
                        exports.push(ExportInfo {
                            name,
                            is_default: false,
                            is_type_only: named_spec.is_type_only,
                            source: named.src.as_ref().map(|s| s.value.as_str().unwrap_or("").to_string()),
                        });
                    }
                }
            }
            _ => {}
        }
    }

    fn process_stmt(
        &mut self,
        stmt: &Stmt,
        components: &mut Vec<ComponentExtraction>,
        _custom_hooks: &mut Vec<CustomHookExtraction>,
        types: &mut Vec<TypeExtraction>,
        helper_functions: &mut Vec<HelperFunctionExtraction>,
    ) {
        match stmt {
            Stmt::Decl(decl) => {
                if let Some(comp) = self.try_extract_component_from_decl(decl, false, None) {
                    components.push(comp);
                } else if let Some(helper) = self.try_extract_helper_function_from_decl(decl, false) {
                    helper_functions.push(helper);
                }
                if let Some(type_ext) = self.try_extract_type_from_decl(decl, false) {
                    types.push(type_ext);
                }
            }
            _ => {}
        }
    }

    fn extract_import(&self, import: &ImportDecl) -> ImportInfo {
        let mut specifiers = Vec::new();

        for spec in &import.specifiers {
            match spec {
                swc_ecma_ast::ImportSpecifier::Named(named) => {
                    let imported = match &named.imported {
                        Some(ModuleExportName::Ident(id)) => id.sym.to_string(),
                        Some(ModuleExportName::Str(s)) => s.value.as_str().unwrap_or("").to_string(),
                        None => named.local.sym.to_string(),
                    };
                    specifiers.push(ImportSpecifier {
                        imported,
                        local: named.local.sym.to_string(),
                        is_default: false,
                        is_namespace: false,
                    });
                }
                swc_ecma_ast::ImportSpecifier::Default(default) => {
                    specifiers.push(ImportSpecifier {
                        imported: "default".to_string(),
                        local: default.local.sym.to_string(),
                        is_default: true,
                        is_namespace: false,
                    });
                }
                swc_ecma_ast::ImportSpecifier::Namespace(ns) => {
                    specifiers.push(ImportSpecifier {
                        imported: "*".to_string(),
                        local: ns.local.sym.to_string(),
                        is_default: false,
                        is_namespace: true,
                    });
                }
            }
        }

        ImportInfo {
            source: import.src.value.as_str().unwrap_or("").to_string(),
            specifiers,
            is_type_only: import.type_only,
        }
    }

    fn try_extract_component_from_decl(&mut self, decl: &Decl, exported: bool, export_type: Option<ExportType>) -> Option<ComponentExtraction> {
        match decl {
            Decl::Fn(fn_decl) => {
                self.try_extract_functional_component(&fn_decl.ident.sym, &fn_decl.function, exported, export_type)
            }
            Decl::Var(var_decl) => {
                for decl in &var_decl.decls {
                    if let Some(Pat::Ident(ident)) = Some(&decl.name) {
                        if let Some(init) = &decl.init {
                            if let Some(comp) = self.try_extract_component_from_expr(init) {
                                let mut comp = comp;
                                comp.name = ident.id.sym.to_string();
                                comp.exported = exported;
                                comp.export_type = export_type;
                                return Some(comp);
                            }
                        }
                    }
                }
                None
            }
            Decl::Class(class_decl) => {
                self.try_extract_class_component(&class_decl.ident.sym, &class_decl.class, exported, export_type)
            }
            _ => None,
        }
    }

    fn try_extract_component_from_default_decl(&mut self, decl: &DefaultDecl) -> Option<ComponentExtraction> {
        match decl {
            DefaultDecl::Fn(fn_expr) => {
                let name = fn_expr.ident.as_ref()
                    .map(|i| i.sym.to_string())
                    .unwrap_or_else(|| "default".to_string());
                self.try_extract_functional_component(&name, &fn_expr.function, false, None)
            }
            DefaultDecl::Class(class_expr) => {
                let name = class_expr.ident.as_ref()
                    .map(|i| i.sym.to_string())
                    .unwrap_or_else(|| "default".to_string());
                self.try_extract_class_component(&name, &class_expr.class, false, None)
            }
            _ => None,
        }
    }

    fn try_extract_component_from_expr(&mut self, expr: &Expr) -> Option<ComponentExtraction> {
        match expr {
            Expr::Arrow(arrow) => {
                self.try_extract_arrow_component(arrow)
            }
            Expr::Fn(fn_expr) => {
                let name = fn_expr.ident.as_ref()
                    .map(|i| i.sym.to_string())
                    .unwrap_or_else(|| "Anonymous".to_string());
                self.try_extract_functional_component(&name, &fn_expr.function, false, None)
            }
            Expr::Call(call) => {
                // Check for memo(), forwardRef()
                if let Callee::Expr(callee) = &call.callee {
                    if let Expr::Ident(ident) = callee.as_ref() {
                        let name = ident.sym.as_ref();
                        if name == "memo" || name == "forwardRef" {
                            if let Some(arg) = call.args.first() {
                                if let Some(mut comp) = self.try_extract_component_from_expr(&arg.expr) {
                                    comp.component_type = if name == "memo" {
                                        ComponentType::Memo
                                    } else {
                                        ComponentType::ForwardRef
                                    };
                                    return Some(comp);
                                }
                            }
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }

    fn try_extract_functional_component(
        &mut self,
        name: &str,
        function: &Function,
        exported: bool,
        export_type: Option<ExportType>,
    ) -> Option<ComponentExtraction> {
        // Check if it looks like a React component (returns JSX)
        let jsx = self.extract_jsx_from_body(&function.body);
        if jsx.root.is_none() && !self.looks_like_component(name) {
            return None;
        }

        if jsx.root.is_some() {
            self.has_jsx = true;
        }

        let hooks = self.extract_hooks_from_body(&function.body);
        let mut handlers = self.extract_handlers_from_body(&function.body);
        let props = self.extract_props_from_params(&function.params);

        let custom_hooks = self.extract_custom_hooks_from_body(&function.body);

        // Link handler calls to their hook sources
        Self::link_handler_calls_to_hooks(&mut handlers, &custom_hooks);

        // Phase 6.5: Generate architecture recommendations
        let architecture = self.generate_architecture_recommendation(&hooks, &custom_hooks, &handlers);

        Some(ComponentExtraction {
            name: name.to_string(),
            component_type: ComponentType::Functional,
            exported,
            export_type,
            location: self.span_to_location(function.span),
            props,
            props_type: None, // TODO: extract from type annotation
            hooks,
            custom_hooks,
            class_info: None,
            jsx: jsx.clone(),
            handlers,
            child_components: self.extract_child_components_from_jsx(&jsx),
            architecture,
        })
    }

    fn try_extract_arrow_component(&mut self, arrow: &ArrowExpr) -> Option<ComponentExtraction> {
        let jsx = self.extract_jsx_from_arrow_body(&arrow.body);
        if jsx.root.is_none() {
            return None;
        }

        self.has_jsx = true;

        let hooks = self.extract_hooks_from_arrow_body(&arrow.body);
        let mut handlers = self.extract_handlers_from_arrow_body(&arrow.body);
        let props = self.extract_props_from_arrow_params(&arrow.params);

        let custom_hooks = match arrow.body.as_ref() {
            BlockStmtOrExpr::BlockStmt(block) => self.extract_custom_hooks_from_body(&Some(block.clone())),
            BlockStmtOrExpr::Expr(_) => Vec::new(),
        };

        // Link handler calls to their hook sources
        Self::link_handler_calls_to_hooks(&mut handlers, &custom_hooks);

        // Phase 6.5: Generate architecture recommendations
        let architecture = self.generate_architecture_recommendation(&hooks, &custom_hooks, &handlers);

        Some(ComponentExtraction {
            name: "Anonymous".to_string(),
            component_type: ComponentType::Functional,
            exported: false,
            export_type: None,
            location: self.span_to_location(arrow.span),
            props,
            props_type: None,
            hooks,
            custom_hooks,
            class_info: None,
            jsx: jsx.clone(),
            handlers,
            child_components: self.extract_child_components_from_jsx(&jsx),
            architecture,
        })
    }

    fn extract_handlers_from_arrow_body(&self, body: &BlockStmtOrExpr) -> Vec<HandlerExtraction> {
        match body {
            BlockStmtOrExpr::BlockStmt(block) => self.extract_handlers_from_body(&Some(block.clone())),
            BlockStmtOrExpr::Expr(_) => Vec::new(), // Arrow expressions don't have local handlers
        }
    }

    fn try_extract_class_component(
        &mut self,
        name: &str,
        class: &Class,
        exported: bool,
        export_type: Option<ExportType>,
    ) -> Option<ComponentExtraction> {
        // Check if it extends React.Component or Component
        let is_react_class = class.super_class.as_ref().map_or(false, |sc| {
            self.is_react_component_superclass(sc)
        });

        if !is_react_class {
            return None;
        }

        // Find render method
        let mut jsx = JsxTree { root: None };
        let mut lifecycle_methods = Vec::new();

        for member in &class.body {
            if let ClassMember::Method(method) = member {
                if let PropName::Ident(ident) = &method.key {
                    let method_name = ident.sym.as_ref();
                    match method_name {
                        "render" => {
                            jsx = self.extract_jsx_from_body(&method.function.body);
                        }
                        "componentDidMount" => lifecycle_methods.push(LifecycleMethod::ComponentDidMount),
                        "componentDidUpdate" => lifecycle_methods.push(LifecycleMethod::ComponentDidUpdate),
                        "componentWillUnmount" => lifecycle_methods.push(LifecycleMethod::ComponentWillUnmount),
                        "shouldComponentUpdate" => lifecycle_methods.push(LifecycleMethod::ShouldComponentUpdate),
                        _ => {}
                    }
                }
            }
        }

        if jsx.root.is_some() {
            self.has_jsx = true;
        }

        // Class components don't use hooks, so architecture is minimal
        let architecture = ArchitectureRecommendation::default();

        Some(ComponentExtraction {
            name: name.to_string(),
            component_type: ComponentType::Class,
            exported,
            export_type,
            location: self.span_to_location(class.span),
            props: Vec::new(), // TODO: extract from constructor
            props_type: None,
            hooks: Vec::new(), // Class components don't use hooks
            custom_hooks: Vec::new(), // Class components don't use hooks
            class_info: Some(ClassComponentInfo {
                state_type: None,
                state_initializer: None,
                lifecycle_methods,
            }),
            jsx: jsx.clone(),
            handlers: Vec::new(),
            child_components: self.extract_child_components_from_jsx(&jsx),
            architecture,
        })
    }

    /// Extract child component references from JSX tree
    fn extract_child_components_from_jsx(&self, jsx: &JsxTree) -> Vec<String> {
        let mut components = Vec::new();

        if let Some(root) = &jsx.root {
            self.collect_child_components(root, &mut components);
        }

        // Remove duplicates and sort
        components.sort();
        components.dedup();
        components
    }

    fn collect_child_components(&self, node: &JsxNode, components: &mut Vec<String>) {
        match &node.node_type {
            JsxNodeType::Element { tag, is_component, children, .. } => {
                if *is_component {
                    // Extract base component name (handle Namespace.Component)
                    let base_name = tag.split('.').next().unwrap_or(tag);
                    components.push(base_name.to_string());
                }
                for child in children {
                    self.collect_child_components(child, components);
                }
            }
            JsxNodeType::Fragment { children } => {
                for child in children {
                    self.collect_child_components(child, components);
                }
            }
            JsxNodeType::Conditional { consequent, alternate, .. } => {
                self.collect_child_components(consequent, components);
                if let Some(alt) = alternate {
                    self.collect_child_components(alt, components);
                }
            }
            JsxNodeType::Map { body, .. } => {
                self.collect_child_components(body, components);
            }
            _ => {}
        }
    }

    fn is_react_component_superclass(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Ident(ident) => {
                let name = ident.sym.as_ref();
                name == "Component" || name == "PureComponent"
            }
            Expr::Member(member) => {
                if let Expr::Ident(obj) = member.obj.as_ref() {
                    if obj.sym.as_ref() == "React" {
                        if let MemberProp::Ident(prop) = &member.prop {
                            let name = prop.sym.as_ref();
                            return name == "Component" || name == "PureComponent";
                        }
                    }
                }
                false
            }
            _ => false,
        }
    }

    fn looks_like_component(&self, name: &str) -> bool {
        // React components start with uppercase
        name.chars().next().map_or(false, |c| c.is_uppercase())
    }

    fn extract_jsx_from_body(&self, body: &Option<BlockStmt>) -> JsxTree {
        let body = match body {
            Some(b) => b,
            None => return JsxTree { root: None },
        };

        // Find the return statement that returns JSX
        for stmt in &body.stmts {
            if let Stmt::Return(ret) = stmt {
                if let Some(expr) = &ret.arg {
                    let result = self.extract_jsx_from_expr(expr);
                    if result.root.is_some() {
                        return result;
                    }
                }
            }
        }

        JsxTree { root: None }
    }

    fn extract_jsx_from_arrow_body(&self, body: &BlockStmtOrExpr) -> JsxTree {
        match body {
            BlockStmtOrExpr::Expr(expr) => self.extract_jsx_from_expr(expr),
            BlockStmtOrExpr::BlockStmt(block) => self.extract_jsx_from_body(&Some(block.clone())),
        }
    }

    fn extract_jsx_from_expr(&self, expr: &Expr) -> JsxTree {
        match expr {
            Expr::JSXElement(el) => {
                JsxTree {
                    root: Some(self.extract_jsx_element(el)),
                }
            }
            Expr::JSXFragment(frag) => {
                JsxTree {
                    root: Some(self.extract_jsx_fragment(frag)),
                }
            }
            Expr::Paren(paren) => self.extract_jsx_from_expr(&paren.expr),
            _ => JsxTree { root: None },
        }
    }

    fn extract_jsx_element(&self, el: &JSXElement) -> JsxNode {
        let tag = match &el.opening.name {
            JSXElementName::Ident(ident) => ident.sym.to_string(),
            JSXElementName::JSXMemberExpr(member) => {
                format!("{}.{}",
                    self.jsx_object_to_string(&member.obj),
                    member.prop.sym
                )
            }
            JSXElementName::JSXNamespacedName(ns) => {
                format!("{}:{}", ns.ns.sym, ns.name.sym)
            }
        };

        let is_component = tag.chars().next().map_or(false, |c| c.is_uppercase());

        let attributes: Vec<JsxAttribute> = el.opening.attrs.iter().filter_map(|attr| {
            match attr {
                JSXAttrOrSpread::JSXAttr(attr) => {
                    let name = match &attr.name {
                        JSXAttrName::Ident(ident) => ident.sym.to_string(),
                        JSXAttrName::JSXNamespacedName(ns) => {
                            format!("{}:{}", ns.ns.sym, ns.name.sym)
                        }
                    };
                    let is_event_handler = name.starts_with("on") && name.len() > 2 &&
                        name.chars().nth(2).map_or(false, |c| c.is_uppercase());

                    let value = match &attr.value {
                        Some(JSXAttrValue::Str(s)) => {
                            JsxAttributeValue::String { value: s.value.as_str().unwrap_or("").to_string() }
                        }
                        Some(JSXAttrValue::JSXExprContainer(container)) => {
                            match &container.expr {
                                JSXExpr::Expr(expr) => JsxAttributeValue::Expression {
                                    code: self.span_to_source((*expr).span())
                                },
                                JSXExpr::JSXEmptyExpr(_) => JsxAttributeValue::Expression {
                                    code: "".to_string()
                                },
                            }
                        }
                        Some(JSXAttrValue::JSXElement(el)) => {
                            JsxAttributeValue::Expression { code: self.span_to_source(el.span) }
                        }
                        Some(JSXAttrValue::JSXFragment(frag)) => {
                            JsxAttributeValue::Expression { code: self.span_to_source(frag.span) }
                        }
                        None => JsxAttributeValue::True,
                    };

                    Some(JsxAttribute {
                        name,
                        value,
                        is_event_handler,
                    })
                }
                JSXAttrOrSpread::SpreadElement(spread) => {
                    Some(JsxAttribute {
                        name: "...".to_string(),
                        value: JsxAttributeValue::Spread { name: self.span_to_source((*spread.expr).span()) },
                        is_event_handler: false,
                    })
                }
            }
        }).collect();

        let children: Vec<JsxNode> = el.children.iter().filter_map(|child| {
            match child {
                JSXElementChild::JSXElement(el) => Some(self.extract_jsx_element(el)),
                JSXElementChild::JSXFragment(frag) => Some(self.extract_jsx_fragment(frag)),
                JSXElementChild::JSXText(text) => {
                    let value = text.value.to_string().trim().to_string();
                    if value.is_empty() {
                        None
                    } else {
                        Some(JsxNode {
                            node_type: JsxNodeType::Text { value },
                            location: self.span_to_location(text.span),
                        })
                    }
                }
                JSXElementChild::JSXExprContainer(container) => {
                    match &container.expr {
                        JSXExpr::Expr(expr) => {
                            // Try to detect special expression patterns
                            Some(self.classify_jsx_expression(expr, container.span))
                        },
                        JSXExpr::JSXEmptyExpr(_) => None,
                    }
                }
                JSXElementChild::JSXSpreadChild(_) => None,
            }
        }).collect();

        JsxNode {
            node_type: JsxNodeType::Element {
                tag,
                is_component,
                attributes,
                children,
            },
            location: self.span_to_location(el.span),
        }
    }

    fn extract_jsx_fragment(&self, frag: &JSXFragment) -> JsxNode {
        let children: Vec<JsxNode> = frag.children.iter().filter_map(|child| {
            match child {
                JSXElementChild::JSXElement(el) => Some(self.extract_jsx_element(el)),
                JSXElementChild::JSXFragment(frag) => Some(self.extract_jsx_fragment(frag)),
                JSXElementChild::JSXText(text) => {
                    let value = text.value.to_string().trim().to_string();
                    if value.is_empty() {
                        None
                    } else {
                        Some(JsxNode {
                            node_type: JsxNodeType::Text { value },
                            location: self.span_to_location(text.span),
                        })
                    }
                }
                JSXElementChild::JSXExprContainer(container) => {
                    match &container.expr {
                        JSXExpr::Expr(expr) => Some(self.classify_jsx_expression(expr, container.span)),
                        JSXExpr::JSXEmptyExpr(_) => None,
                    }
                }
                JSXElementChild::JSXSpreadChild(_) => None,
            }
        }).collect();

        JsxNode {
            node_type: JsxNodeType::Fragment { children },
            location: self.span_to_location(frag.span),
        }
    }

    /// Classify a JSX expression into specific patterns (conditional, map, plain expression)
    fn classify_jsx_expression(&self, expr: &Expr, span: swc_common::Span) -> JsxNode {
        let location = self.span_to_location(span);

        // Check for conditional: {cond && <X/>} or {cond ? <A/> : <B/>}
        match expr {
            // Logical AND: {cond && <JSX/>}
            Expr::Bin(bin) if bin.op == BinaryOp::LogicalAnd => {
                let condition = self.span_to_source((*bin.left).span());
                if let Some(jsx_node) = self.try_extract_jsx_from_expr(&bin.right) {
                    return JsxNode {
                        node_type: JsxNodeType::Conditional {
                            condition,
                            consequent: Box::new(jsx_node),
                            alternate: None,
                        },
                        location,
                    };
                }
            }
            // Ternary: {cond ? <A/> : <B/>}
            Expr::Cond(cond) => {
                let condition = self.span_to_source((*cond.test).span());
                if let Some(cons) = self.try_extract_jsx_from_expr(&cond.cons) {
                    let alt = self.try_extract_jsx_from_expr(&cond.alt);
                    return JsxNode {
                        node_type: JsxNodeType::Conditional {
                            condition,
                            consequent: Box::new(cons),
                            alternate: alt.map(Box::new),
                        },
                        location,
                    };
                }
            }
            // Map: {items.map(item => <X/>)}
            Expr::Call(call) => {
                if let Some((iterable, item_name, key_expr, body)) = self.try_extract_map_pattern(call) {
                    return JsxNode {
                        node_type: JsxNodeType::Map {
                            iterable,
                            item_name,
                            key_expr,
                            body: Box::new(body),
                        },
                        location,
                    };
                }
            }
            _ => {}
        }

        // Default: plain expression
        JsxNode {
            node_type: JsxNodeType::Expression {
                code: self.span_to_source(expr.span())
            },
            location,
        }
    }

    /// Try to extract JSX from an expression (for conditional branches)
    fn try_extract_jsx_from_expr(&self, expr: &Expr) -> Option<JsxNode> {
        match expr {
            Expr::JSXElement(el) => Some(self.extract_jsx_element(el)),
            Expr::JSXFragment(frag) => Some(self.extract_jsx_fragment(frag)),
            Expr::Paren(p) => self.try_extract_jsx_from_expr(&p.expr),
            _ => None,
        }
    }

    /// Try to extract a .map() pattern: items.map(item => <X/>)
    fn try_extract_map_pattern(&self, call: &CallExpr) -> Option<(String, String, Option<String>, JsxNode)> {
        // Check if callee is xxx.map
        let (obj_span, method_name) = match &call.callee {
            Callee::Expr(expr) => {
                if let Expr::Member(member) = expr.as_ref() {
                    if let MemberProp::Ident(prop) = &member.prop {
                        if prop.sym.as_ref() == "map" {
                            ((*member.obj).span(), "map")
                        } else {
                            return None;
                        }
                    } else {
                        return None;
                    }
                } else {
                    return None;
                }
            }
            _ => return None,
        };

        if method_name != "map" {
            return None;
        }

        let iterable = self.span_to_source(obj_span);

        // Get the callback argument
        let callback = call.args.first()?;

        // Extract item name and body from callback
        let (item_name, body_expr) = match callback.expr.as_ref() {
            Expr::Arrow(arrow) => {
                let item = match arrow.params.first()? {
                    Pat::Ident(ident) => ident.id.sym.to_string(),
                    _ => return None,
                };
                let body = match &*arrow.body {
                    BlockStmtOrExpr::Expr(e) => e.as_ref(),
                    BlockStmtOrExpr::BlockStmt(block) => {
                        // Look for return statement
                        for stmt in &block.stmts {
                            if let Stmt::Return(ret) = stmt {
                                if let Some(arg) = &ret.arg {
                                    // Can't easily get ref here, just return None
                                    return None;
                                }
                            }
                        }
                        return None;
                    }
                };
                (item, body)
            }
            _ => return None,
        };

        // Try to extract JSX from body
        let jsx_body = self.try_extract_jsx_from_expr(body_expr)?;

        // Try to find key prop
        let key_expr = if let JsxNode { node_type: JsxNodeType::Element { attributes, .. }, .. } = &jsx_body {
            attributes.iter()
                .find(|a| a.name == "key")
                .and_then(|a| match &a.value {
                    JsxAttributeValue::Expression { code } => Some(code.clone()),
                    _ => None,
                })
        } else {
            None
        };

        Some((iterable, item_name, key_expr, jsx_body))
    }

    fn jsx_object_to_string(&self, obj: &JSXObject) -> String {
        match obj {
            JSXObject::Ident(ident) => ident.sym.to_string(),
            JSXObject::JSXMemberExpr(member) => {
                format!("{}.{}", self.jsx_object_to_string(&member.obj), member.prop.sym)
            }
        }
    }

    fn extract_hooks_from_body(&self, body: &Option<BlockStmt>) -> Vec<HookUsage> {
        let body = match body {
            Some(b) => b,
            None => return Vec::new(),
        };

        let mut hooks = Vec::new();

        for stmt in &body.stmts {
            self.find_hooks_in_stmt(stmt, &mut hooks);
        }

        hooks
    }

    fn extract_hooks_from_arrow_body(&self, body: &BlockStmtOrExpr) -> Vec<HookUsage> {
        match body {
            BlockStmtOrExpr::Expr(expr) => {
                let mut hooks = Vec::new();
                self.find_hooks_in_expr(expr, &mut hooks);
                hooks
            }
            BlockStmtOrExpr::BlockStmt(block) => self.extract_hooks_from_body(&Some(block.clone())),
        }
    }

    fn find_hooks_in_stmt(&self, stmt: &Stmt, hooks: &mut Vec<HookUsage>) {
        match stmt {
            Stmt::Decl(Decl::Var(var_decl)) => {
                for decl in &var_decl.decls {
                    if let Some(init) = &decl.init {
                        // Check if it's a hook call
                        if let Some(hook) = self.try_extract_hook_from_var_decl(&decl.name, init) {
                            hooks.push(hook);
                        }
                    }
                }
            }
            Stmt::Expr(expr_stmt) => {
                // Standalone hook calls like useEffect
                self.find_hooks_in_expr(&expr_stmt.expr, hooks);
            }
            _ => {}
        }
    }

    fn find_hooks_in_expr(&self, expr: &Expr, hooks: &mut Vec<HookUsage>) {
        if let Expr::Call(call) = expr {
            if let Some(hook) = self.try_extract_standalone_hook(call) {
                hooks.push(hook);
            }
        }
    }

    fn try_extract_hook_from_var_decl(&self, pattern: &Pat, init: &Expr) -> Option<HookUsage> {
        // Check if init is a hook call
        let call = match init {
            Expr::Call(c) => c,
            _ => return None,
        };

        let hook_name = self.get_callee_name(&call.callee)?;

        // Only process React hooks (start with "use")
        if !hook_name.starts_with("use") {
            return None;
        }

        match hook_name.as_str() {
            "useState" => self.extract_use_state(pattern, call),
            "useRef" => self.extract_use_ref(pattern, call),
            "useCallback" => self.extract_use_callback(call),
            "useMemo" => self.extract_use_memo(call),
            "useContext" => self.extract_use_context(pattern, call),
            "useReducer" => self.extract_use_reducer(pattern, call),
            _ => None, // Custom hook - could be extracted differently
        }
    }

    fn try_extract_standalone_hook(&self, call: &CallExpr) -> Option<HookUsage> {
        let hook_name = self.get_callee_name(&call.callee)?;

        match hook_name.as_str() {
            "useEffect" => self.extract_use_effect(call),
            "useLayoutEffect" => self.extract_use_layout_effect(call),
            _ => None,
        }
    }

    fn get_callee_name(&self, callee: &Callee) -> Option<String> {
        match callee {
            Callee::Expr(expr) => {
                match expr.as_ref() {
                    Expr::Ident(ident) => Some(ident.sym.to_string()),
                    Expr::Member(member) => {
                        // React.useState -> useState
                        if let MemberProp::Ident(prop) = &member.prop {
                            Some(prop.sym.to_string())
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    fn extract_use_state(&self, pattern: &Pat, call: &CallExpr) -> Option<HookUsage> {
        // Extract [state, setState] from pattern
        let (state_name, setter_name) = match pattern {
            Pat::Array(arr) if arr.elems.len() >= 2 => {
                let state = arr.elems.get(0).and_then(|e| e.as_ref())
                    .and_then(|p| self.get_ident_from_pattern(p));
                let setter = arr.elems.get(1).and_then(|e| e.as_ref())
                    .and_then(|p| self.get_ident_from_pattern(p));
                (state, setter)
            }
            _ => (None, None),
        };

        // Get initial value
        let initial_value = call.args.first()
            .map(|arg| self.expr_to_string(&arg.expr));

        Some(HookUsage {
            hook_type: HookType::UseState,
            location: self.span_to_location(call.span),
            state_name,
            setter_name,
            initial_value,
            dependencies: None,
            has_cleanup: false,
            memoized_deps: None,
            ref_name: None,
            ref_type: None,
            context_name: None,
            reducer_name: None,
            action_types: Vec::new(),
        })
    }

    fn extract_use_effect(&self, call: &CallExpr) -> Option<HookUsage> {
        // Get callback and check for cleanup
        let has_cleanup = call.args.first()
            .map(|arg| self.callback_has_cleanup(&arg.expr))
            .unwrap_or(false);

        // Get dependencies
        let dependencies = call.args.get(1)
            .and_then(|arg| self.extract_dependency_array(&arg.expr));

        Some(HookUsage {
            hook_type: HookType::UseEffect,
            location: self.span_to_location(call.span),
            state_name: None,
            setter_name: None,
            initial_value: None,
            dependencies,
            has_cleanup,
            memoized_deps: None,
            ref_name: None,
            ref_type: None,
            context_name: None,
            reducer_name: None,
            action_types: Vec::new(),
        })
    }

    fn extract_use_layout_effect(&self, call: &CallExpr) -> Option<HookUsage> {
        let mut hook = self.extract_use_effect(call)?;
        hook.hook_type = HookType::UseLayoutEffect;
        Some(hook)
    }

    fn extract_use_callback(&self, call: &CallExpr) -> Option<HookUsage> {
        let memoized_deps = call.args.get(1)
            .and_then(|arg| self.extract_dependency_array(&arg.expr));

        Some(HookUsage {
            hook_type: HookType::UseCallback,
            location: self.span_to_location(call.span),
            state_name: None,
            setter_name: None,
            initial_value: None,
            dependencies: None,
            has_cleanup: false,
            memoized_deps,
            ref_name: None,
            ref_type: None,
            context_name: None,
            reducer_name: None,
            action_types: Vec::new(),
        })
    }

    fn extract_use_memo(&self, call: &CallExpr) -> Option<HookUsage> {
        let mut hook = self.extract_use_callback(call)?;
        hook.hook_type = HookType::UseMemo;
        Some(hook)
    }

    fn extract_use_ref(&self, pattern: &Pat, call: &CallExpr) -> Option<HookUsage> {
        let ref_name = self.get_ident_from_pattern(pattern);

        // Try to get type from type argument: useRef<HTMLInputElement>
        let ref_type = call.type_args.as_ref()
            .and_then(|args| args.params.first())
            .map(|t| self.type_to_string(t));

        Some(HookUsage {
            hook_type: HookType::UseRef,
            location: self.span_to_location(call.span),
            state_name: None,
            setter_name: None,
            initial_value: None,
            dependencies: None,
            has_cleanup: false,
            memoized_deps: None,
            ref_name,
            ref_type,
            context_name: None,
            reducer_name: None,
            action_types: Vec::new(),
        })
    }

    fn extract_use_context(&self, _pattern: &Pat, call: &CallExpr) -> Option<HookUsage> {
        let context_name = call.args.first()
            .and_then(|arg| {
                if let Expr::Ident(ident) = arg.expr.as_ref() {
                    Some(ident.sym.to_string())
                } else {
                    None
                }
            });

        Some(HookUsage {
            hook_type: HookType::UseContext,
            location: self.span_to_location(call.span),
            state_name: None,
            setter_name: None,
            initial_value: None,
            dependencies: None,
            has_cleanup: false,
            memoized_deps: None,
            ref_name: None,
            ref_type: None,
            context_name,
            reducer_name: None,
            action_types: Vec::new(),
        })
    }

    fn extract_use_reducer(&self, pattern: &Pat, call: &CallExpr) -> Option<HookUsage> {
        // Get reducer name from first argument
        let reducer_name = call.args.first()
            .and_then(|arg| {
                if let Expr::Ident(ident) = arg.expr.as_ref() {
                    Some(ident.sym.to_string())
                } else {
                    None
                }
            });

        Some(HookUsage {
            hook_type: HookType::UseReducer,
            location: self.span_to_location(call.span),
            state_name: None,
            setter_name: None,
            initial_value: None,
            dependencies: None,
            has_cleanup: false,
            memoized_deps: None,
            ref_name: None,
            ref_type: None,
            context_name: None,
            reducer_name,
            action_types: Vec::new(), // Would need to analyze reducer
        })
    }

    fn get_ident_from_pattern(&self, pattern: &Pat) -> Option<String> {
        match pattern {
            Pat::Ident(ident) => Some(ident.id.sym.to_string()),
            _ => None,
        }
    }

    fn extract_dependency_array(&self, expr: &Expr) -> Option<Vec<String>> {
        if let Expr::Array(arr) = expr {
            let deps: Vec<String> = arr.elems.iter()
                .filter_map(|e| e.as_ref())
                .map(|elem| self.expr_to_string(&elem.expr))
                .collect();
            Some(deps)
        } else {
            None
        }
    }

    // =========================================================================
    // Custom Hook Extraction
    // =========================================================================

    /// Extract custom hook usages from function body
    /// Captures: useChat, useAgent, useInfernumStore, useQuery, etc.
    fn extract_custom_hooks_from_body(&self, body: &Option<BlockStmt>) -> Vec<CustomHookUsage> {
        let Some(block) = body else {
            return Vec::new();
        };

        let mut custom_hooks = Vec::new();

        for stmt in &block.stmts {
            if let Stmt::Decl(Decl::Var(var_decl)) = stmt {
                for decl in &var_decl.decls {
                    if let Some(init) = &decl.init {
                        if let Some(custom_hook) = self.try_extract_custom_hook(&decl.name, init) {
                            custom_hooks.push(custom_hook);
                        }
                    }
                }
            }
        }

        custom_hooks
    }

    /// Try to extract a custom hook from a variable declaration
    fn try_extract_custom_hook(&self, pattern: &Pat, init: &Expr) -> Option<CustomHookUsage> {
        let call = match init {
            Expr::Call(c) => c,
            _ => return None,
        };

        let hook_name = self.get_callee_name(&call.callee)?;

        // Must start with "use" but NOT be a standard React hook
        if !hook_name.starts_with("use") {
            return None;
        }

        // Skip standard React hooks (already extracted separately)
        // Includes React 18+ hooks: useId, useTransition, useDeferredValue,
        // useSyncExternalStore, useInsertionEffect
        let standard_hooks = [
            "useState", "useEffect", "useCallback", "useMemo",
            "useRef", "useContext", "useReducer", "useLayoutEffect",
            "useImperativeHandle", "useDebugValue",
            // React 18+ hooks
            "useId", "useTransition", "useDeferredValue",
            "useSyncExternalStore", "useInsertionEffect",
        ];
        if standard_hooks.contains(&hook_name.as_str()) {
            return None;
        }

        // Extract arguments passed to the hook (raw strings for backward compat)
        let arguments: Vec<String> = call.args.iter()
            .map(|arg| self.expr_to_string(&arg.expr))
            .collect();

        // Phase 6.4: Extract expanded arguments
        let expanded_arguments: Vec<HookArgument> = call.args.iter()
            .map(|arg| self.expand_hook_argument(&arg.expr))
            .collect();

        // Extract destructured return values
        let returned_values = self.extract_destructured_values(pattern);

        // Check if this looks like a Zustand store (has selector argument or name contains "Store")
        let is_zustand = hook_name.contains("Store") ||
            arguments.iter().any(|a| a.contains("select") || a.starts_with("(state)"));

        Some(CustomHookUsage {
            name: hook_name,
            location: self.span_to_location(call.span),
            arguments,
            expanded_arguments,
            returned_values,
            is_zustand,
        })
    }

    // =========================================================================
    // Phase 6.4: Hook Argument Expansion
    // =========================================================================

    /// Expand a hook argument to show its full structure
    fn expand_hook_argument(&self, expr: &Expr) -> HookArgument {
        match expr {
            Expr::Object(obj) => {
                let properties: Vec<HookObjectProperty> = obj.props.iter()
                    .filter_map(|prop| self.expand_object_property(prop))
                    .collect();
                HookArgument::Object { properties }
            }
            Expr::Array(arr) => {
                let elements: Vec<String> = arr.elems.iter()
                    .filter_map(|elem| elem.as_ref().map(|e| self.span_to_source(self.expr_span(&e.expr))))
                    .collect();
                HookArgument::Array { elements }
            }
            Expr::Arrow(arrow) => {
                let params: Vec<String> = arrow.params.iter()
                    .map(|p| self.span_to_source(p.span()))
                    .collect();
                let body_summary = self.span_to_source(arrow.body.span());
                let (calls, side_effects, _, _) = self.analyze_handler_body_detailed_arrow(&arrow.body);
                HookArgument::Function {
                    params,
                    body_summary,
                    calls,
                    side_effects,
                }
            }
            Expr::Fn(fn_expr) => {
                let params: Vec<String> = fn_expr.function.params.iter()
                    .map(|p| self.span_to_source(p.span))
                    .collect();
                let body_summary = fn_expr.function.body.as_ref()
                    .map(|b| self.span_to_source(b.span))
                    .unwrap_or_default();
                let (calls, side_effects, _, _) = self.analyze_handler_body_detailed(&fn_expr.function.body);
                HookArgument::Function {
                    params,
                    body_summary,
                    calls,
                    side_effects,
                }
            }
            _ => {
                HookArgument::Expression {
                    value: self.span_to_source(self.expr_span(expr)),
                }
            }
        }
    }

    /// Expand an object property for hook argument
    fn expand_object_property(&self, prop: &PropOrSpread) -> Option<HookObjectProperty> {
        match prop {
            PropOrSpread::Prop(p) => {
                match p.as_ref() {
                    Prop::KeyValue(kv) => {
                        let name = match &kv.key {
                            PropName::Ident(ident) => ident.sym.to_string(),
                            PropName::Str(s) => s.value.as_str().unwrap_or("").to_string(),
                            _ => return None,
                        };
                        let value_kind = self.expand_property_value(&kv.value);
                        Some(HookObjectProperty { name, value_kind })
                    }
                    Prop::Shorthand(ident) => {
                        Some(HookObjectProperty {
                            name: ident.sym.to_string(),
                            value_kind: HookPropertyValue::Simple {
                                value: ident.sym.to_string()
                            },
                        })
                    }
                    Prop::Method(method) => {
                        let name = match &method.key {
                            PropName::Ident(ident) => ident.sym.to_string(),
                            _ => return None,
                        };
                        let params: Vec<String> = method.function.params.iter()
                            .map(|p| self.span_to_source(p.span))
                            .collect();
                        let body_summary = method.function.body.as_ref()
                            .map(|b| self.span_to_source(b.span))
                            .unwrap_or_default();
                        let (calls, side_effects, _, _) = self.analyze_handler_body_detailed(&method.function.body);
                        Some(HookObjectProperty {
                            name,
                            value_kind: HookPropertyValue::Callback {
                                params,
                                body_summary,
                                calls,
                                side_effects,
                            },
                        })
                    }
                    _ => None,
                }
            }
            PropOrSpread::Spread(_) => None,
        }
    }

    /// Expand a property value in an object
    fn expand_property_value(&self, expr: &Expr) -> HookPropertyValue {
        match expr {
            Expr::Arrow(arrow) => {
                let params: Vec<String> = arrow.params.iter()
                    .map(|p| self.span_to_source(p.span()))
                    .collect();
                let body_summary = self.span_to_source(arrow.body.span());
                let (calls, side_effects, _, _) = self.analyze_handler_body_detailed_arrow(&arrow.body);
                HookPropertyValue::Callback {
                    params,
                    body_summary,
                    calls,
                    side_effects,
                }
            }
            Expr::Fn(fn_expr) => {
                let params: Vec<String> = fn_expr.function.params.iter()
                    .map(|p| self.span_to_source(p.span))
                    .collect();
                let body_summary = fn_expr.function.body.as_ref()
                    .map(|b| self.span_to_source(b.span))
                    .unwrap_or_default();
                let (calls, side_effects, _, _) = self.analyze_handler_body_detailed(&fn_expr.function.body);
                HookPropertyValue::Callback {
                    params,
                    body_summary,
                    calls,
                    side_effects,
                }
            }
            Expr::Object(obj) => {
                let properties: Vec<HookObjectProperty> = obj.props.iter()
                    .filter_map(|prop| self.expand_object_property(prop))
                    .collect();
                HookPropertyValue::Object { properties }
            }
            Expr::Array(arr) => {
                let elements: Vec<String> = arr.elems.iter()
                    .filter_map(|elem| elem.as_ref().map(|e| self.span_to_source(self.expr_span(&e.expr))))
                    .collect();
                HookPropertyValue::Array { elements }
            }
            _ => {
                HookPropertyValue::Simple {
                    value: self.span_to_source(self.expr_span(expr)),
                }
            }
        }
    }

    /// Extract destructured values from a pattern (object or array destructuring)
    fn extract_destructured_values(&self, pattern: &Pat) -> Vec<CustomHookReturnValue> {
        let mut values = Vec::new();

        match pattern {
            Pat::Object(obj) => {
                for prop in &obj.props {
                    match prop {
                        ObjectPatProp::KeyValue(kv) => {
                            // Handle renaming: { events: agentEvents }
                            let original_name = match &kv.key {
                                PropName::Ident(ident) => Some(ident.sym.to_string()),
                                _ => None,
                            };
                            if let Some(name) = self.get_ident_from_pattern(&kv.value) {
                                let is_function = self.looks_like_function_name(&name);
                                values.push(CustomHookReturnValue {
                                    name,
                                    is_function,
                                    original_name,
                                });
                            }
                        }
                        ObjectPatProp::Assign(assign) => {
                            // Handle shorthand: { messages }
                            let name = assign.key.sym.to_string();
                            let is_function = self.looks_like_function_name(&name);
                            values.push(CustomHookReturnValue {
                                name,
                                is_function,
                                original_name: None,
                            });
                        }
                        ObjectPatProp::Rest(rest) => {
                            // Handle rest: { ...rest }
                            if let Some(name) = self.get_ident_from_pattern(&rest.arg) {
                                values.push(CustomHookReturnValue {
                                    name,
                                    is_function: false,
                                    original_name: None,
                                });
                            }
                        }
                    }
                }
            }
            Pat::Array(arr) => {
                // Handle array destructuring: [state, setState]
                for (idx, elem) in arr.elems.iter().enumerate() {
                    if let Some(pat) = elem {
                        if let Some(name) = self.get_ident_from_pattern(pat) {
                            // Second element in [x, setX] pattern is usually a function
                            let is_function = idx == 1 || self.looks_like_function_name(&name);
                            values.push(CustomHookReturnValue {
                                name,
                                is_function,
                                original_name: None,
                            });
                        }
                    }
                }
            }
            Pat::Ident(ident) => {
                // Single value: const result = useHook()
                values.push(CustomHookReturnValue {
                    name: ident.id.sym.to_string(),
                    is_function: false,
                    original_name: None,
                });
            }
            _ => {}
        }

        values
    }

    /// Heuristic to detect if a name looks like a function
    fn looks_like_function_name(&self, name: &str) -> bool {
        // Common patterns for functions/callbacks
        name.starts_with("set") ||
        name.starts_with("add") ||
        name.starts_with("remove") ||
        name.starts_with("delete") ||
        name.starts_with("update") ||
        name.starts_with("toggle") ||
        name.starts_with("handle") ||
        name.starts_with("on") ||
        name.starts_with("run") ||
        name.starts_with("reset") ||
        name.starts_with("load") ||
        name.starts_with("save") ||
        name.starts_with("fetch") ||
        name.starts_with("submit") ||
        name.starts_with("export") ||
        name.starts_with("import") ||
        name.starts_with("new") ||     // newConversation, newItem, etc.
        name.starts_with("create") ||  // createUser, createSession, etc.
        name.starts_with("clear") ||   // clearCache, clearErrors, etc.
        name.starts_with("send") ||    // sendMessage, sendRequest, etc.
        name.starts_with("cancel") ||  // cancelRequest, cancelSubscription, etc.
        name.starts_with("refresh") || // refreshData, refreshToken, etc.
        name.ends_with("Mutation") ||
        name.ends_with("Callback")
    }

    fn callback_has_cleanup(&self, expr: &Expr) -> bool {
        // Check if arrow/function returns another function
        match expr {
            Expr::Arrow(arrow) => {
                match &*arrow.body {
                    BlockStmtOrExpr::BlockStmt(block) => {
                        // Check for return statement
                        block.stmts.iter().any(|stmt| {
                            matches!(stmt, Stmt::Return(ret) if ret.arg.is_some())
                        })
                    }
                    BlockStmtOrExpr::Expr(_) => false,
                }
            }
            Expr::Fn(fn_expr) => {
                fn_expr.function.body.as_ref().map_or(false, |block| {
                    block.stmts.iter().any(|stmt| {
                        matches!(stmt, Stmt::Return(ret) if ret.arg.is_some())
                    })
                })
            }
            _ => false,
        }
    }

    fn expr_to_string(&self, expr: &Expr) -> String {
        // Simple expression to string - basic implementation
        match expr {
            Expr::Ident(ident) => ident.sym.to_string(),
            Expr::Lit(Lit::Num(n)) => n.value.to_string(),
            Expr::Lit(Lit::Str(s)) => format!("\"{}\"", s.value.as_str().unwrap_or("")),
            Expr::Lit(Lit::Bool(b)) => b.value.to_string(),
            Expr::Lit(Lit::Null(_)) => "null".to_string(),
            Expr::Array(_) => "[]".to_string(),
            Expr::Object(_) => "∅".to_string(),
            _ => "None".to_string(),
        }
    }

    fn type_to_string(&self, ts_type: &TsType) -> String {
        // Simple type to string - basic implementation
        match ts_type {
            TsType::TsTypeRef(type_ref) => {
                match &type_ref.type_name {
                    TsEntityName::Ident(ident) => ident.sym.to_string(),
                    TsEntityName::TsQualifiedName(qn) => {
                        format!("{}.{}", self.ts_entity_name_to_string(&qn.left), qn.right.sym)
                    }
                }
            }
            TsType::TsKeywordType(kw) => format!("{:?}", kw.kind).to_lowercase(),
            _ => "Any".to_string(),
        }
    }

    fn ts_entity_name_to_string(&self, name: &TsEntityName) -> String {
        match name {
            TsEntityName::Ident(ident) => ident.sym.to_string(),
            TsEntityName::TsQualifiedName(qn) => {
                format!("{}.{}", self.ts_entity_name_to_string(&qn.left), qn.right.sym)
            }
        }
    }

    fn extract_handlers_from_body(&self, body: &Option<BlockStmt>) -> Vec<HandlerExtraction> {
        let body = match body {
            Some(b) => b,
            None => return Vec::new(),
        };

        let mut handlers = Vec::new();

        for stmt in &body.stmts {
            // Look for: const handleX = () => { ... }
            // or: const handleX = function() { ... }
            // or: function handleX() { ... }
            match stmt {
                Stmt::Decl(Decl::Var(var_decl)) => {
                    for decl in &var_decl.decls {
                        if let Pat::Ident(ident) = &decl.name {
                            let name = ident.id.sym.to_string();
                            // Convention: handlers start with "handle" or are event-like
                            if name.starts_with("handle") || name.starts_with("on") {
                                if let Some(init) = &decl.init {
                                    if let Some(handler) = self.extract_handler_from_expr(&name, init) {
                                        handlers.push(handler);
                                    }
                                }
                            }
                        }
                    }
                }
                Stmt::Decl(Decl::Fn(fn_decl)) => {
                    let name = fn_decl.ident.sym.to_string();
                    if name.starts_with("handle") || name.starts_with("on") {
                        handlers.push(self.extract_handler_from_function(&name, &fn_decl.function));
                    }
                }
                _ => {}
            }
        }

        handlers
    }

    fn extract_handler_from_expr(&self, name: &str, expr: &Expr) -> Option<HandlerExtraction> {
        match expr {
            Expr::Arrow(arrow) => {
                let is_async = arrow.is_async;
                let body_summary = self.span_to_source(arrow.span);
                let (state_mutations, api_calls) = self.analyze_handler_body_arrow(&arrow.body);

                // Phase 6.3: Enhanced analysis
                let (calls, side_effects, has_conditionals, has_early_return) =
                    self.analyze_handler_body_detailed_arrow(&arrow.body);
                let parameters = self.extract_handler_params_from_arrow(&arrow.params);

                Some(HandlerExtraction {
                    name: name.to_string(),
                    event_type: None,
                    is_async,
                    body_summary,
                    state_mutations,
                    api_calls,
                    calls,
                    side_effects,
                    parameters,
                    has_conditionals,
                    has_early_return,
                })
            }
            Expr::Fn(fn_expr) => {
                Some(self.extract_handler_from_function(name, &fn_expr.function))
            }
            _ => None,
        }
    }

    fn extract_handler_from_function(&self, name: &str, func: &Function) -> HandlerExtraction {
        let is_async = func.is_async;
        let body_summary = func.body.as_ref()
            .map(|b| self.span_to_source(b.span))
            .unwrap_or_default();
        let (state_mutations, api_calls) = self.analyze_handler_body(&func.body);

        // Phase 6.3: Enhanced analysis
        let (calls, side_effects, has_conditionals, has_early_return) =
            self.analyze_handler_body_detailed(&func.body);
        let parameters = self.extract_handler_params_from_function(&func.params);

        HandlerExtraction {
            name: name.to_string(),
            event_type: None,
            is_async,
            body_summary,
            state_mutations,
            api_calls,
            calls,
            side_effects,
            parameters,
            has_conditionals,
            has_early_return,
        }
    }

    fn analyze_handler_body(&self, body: &Option<BlockStmt>) -> (Vec<String>, Vec<String>) {
        let body = match body {
            Some(b) => b,
            None => return (Vec::new(), Vec::new()),
        };

        let mut state_mutations = Vec::new();
        let mut api_calls = Vec::new();

        for stmt in &body.stmts {
            self.find_mutations_and_calls_in_stmt(stmt, &mut state_mutations, &mut api_calls);
        }

        (state_mutations, api_calls)
    }

    fn analyze_handler_body_arrow(&self, body: &BlockStmtOrExpr) -> (Vec<String>, Vec<String>) {
        match body {
            BlockStmtOrExpr::BlockStmt(block) => self.analyze_handler_body(&Some(block.clone())),
            BlockStmtOrExpr::Expr(expr) => {
                let mut state_mutations = Vec::new();
                let mut api_calls = Vec::new();
                self.find_mutations_and_calls_in_expr(expr, &mut state_mutations, &mut api_calls);
                (state_mutations, api_calls)
            }
        }
    }

    fn find_mutations_and_calls_in_stmt(&self, stmt: &Stmt, state_mutations: &mut Vec<String>, api_calls: &mut Vec<String>) {
        match stmt {
            Stmt::Expr(expr_stmt) => {
                self.find_mutations_and_calls_in_expr(&expr_stmt.expr, state_mutations, api_calls);
            }
            Stmt::Return(ret) => {
                if let Some(arg) = &ret.arg {
                    self.find_mutations_and_calls_in_expr(arg, state_mutations, api_calls);
                }
            }
            Stmt::If(if_stmt) => {
                self.find_mutations_and_calls_in_expr(&if_stmt.test, state_mutations, api_calls);
                if let Stmt::Block(block) = &*if_stmt.cons {
                    for s in &block.stmts {
                        self.find_mutations_and_calls_in_stmt(s, state_mutations, api_calls);
                    }
                }
            }
            _ => {}
        }
    }

    fn find_mutations_and_calls_in_expr(&self, expr: &Expr, state_mutations: &mut Vec<String>, api_calls: &mut Vec<String>) {
        match expr {
            Expr::Call(call) => {
                if let Some(name) = self.get_callee_name(&call.callee) {
                    // Check for setState calls
                    if name.starts_with("set") && name.len() > 3 && name.chars().nth(3).map_or(false, |c| c.is_uppercase()) {
                        state_mutations.push(self.span_to_source(call.span));
                    }
                    // Check for API calls
                    if name == "fetch" || name == "axios" || name.contains("Api") || name.contains("api") {
                        api_calls.push(self.span_to_source(call.span));
                    }
                }
                // Also check member expression calls like api.get()
                if let Callee::Expr(callee_expr) = &call.callee {
                    if let Expr::Member(member) = callee_expr.as_ref() {
                        if let Expr::Ident(obj) = member.obj.as_ref() {
                            let obj_name = obj.sym.as_ref();
                            if obj_name == "fetch" || obj_name == "axios" || obj_name.contains("api") || obj_name.contains("Api") {
                                api_calls.push(self.span_to_source(call.span));
                            }
                        }
                    }
                }
            }
            Expr::Await(await_expr) => {
                self.find_mutations_and_calls_in_expr(&await_expr.arg, state_mutations, api_calls);
            }
            _ => {}
        }
    }

    // =========================================================================
    // Phase 6.3: Handler Body Analysis
    // =========================================================================

    /// Detailed analysis of function body for Phase 6.3
    fn analyze_handler_body_detailed(&self, body: &Option<BlockStmt>) -> (Vec<HandlerCall>, Vec<SideEffect>, bool, bool) {
        let Some(block) = body else {
            return (Vec::new(), Vec::new(), false, false);
        };

        let mut calls = Vec::new();
        let mut side_effects = Vec::new();
        let mut has_conditionals = false;
        let mut has_early_return = false;

        for stmt in &block.stmts {
            self.analyze_stmt_detailed(stmt, &mut calls, &mut side_effects, &mut has_conditionals, &mut has_early_return);
        }

        (calls, side_effects, has_conditionals, has_early_return)
    }

    /// Detailed analysis of arrow function body for Phase 6.3
    fn analyze_handler_body_detailed_arrow(&self, body: &BlockStmtOrExpr) -> (Vec<HandlerCall>, Vec<SideEffect>, bool, bool) {
        match body {
            BlockStmtOrExpr::BlockStmt(block) => self.analyze_handler_body_detailed(&Some(block.clone())),
            BlockStmtOrExpr::Expr(expr) => {
                let mut calls = Vec::new();
                let mut side_effects = Vec::new();
                self.analyze_expr_detailed(expr, &mut calls, &mut side_effects, false);
                (calls, side_effects, false, false)
            }
        }
    }

    /// Analyze a statement for detailed handler info
    fn analyze_stmt_detailed(
        &self,
        stmt: &Stmt,
        calls: &mut Vec<HandlerCall>,
        side_effects: &mut Vec<SideEffect>,
        has_conditionals: &mut bool,
        has_early_return: &mut bool,
    ) {
        match stmt {
            Stmt::Expr(expr_stmt) => {
                self.analyze_expr_detailed(&expr_stmt.expr, calls, side_effects, false);
            }
            Stmt::Return(ret) => {
                // Check if this is an early return (not the last statement)
                *has_early_return = true;
                if let Some(arg) = &ret.arg {
                    self.analyze_expr_detailed(arg, calls, side_effects, false);
                }
            }
            Stmt::If(if_stmt) => {
                *has_conditionals = true;
                self.analyze_expr_detailed(&if_stmt.test, calls, side_effects, false);
                if let Stmt::Block(block) = &*if_stmt.cons {
                    for s in &block.stmts {
                        self.analyze_stmt_detailed(s, calls, side_effects, has_conditionals, has_early_return);
                    }
                }
                if let Some(alt) = &if_stmt.alt {
                    if let Stmt::Block(block) = alt.as_ref() {
                        for s in &block.stmts {
                            self.analyze_stmt_detailed(s, calls, side_effects, has_conditionals, has_early_return);
                        }
                    }
                }
            }
            Stmt::Decl(Decl::Var(var_decl)) => {
                for decl in &var_decl.decls {
                    if let Some(init) = &decl.init {
                        self.analyze_expr_detailed(init, calls, side_effects, false);
                    }
                }
            }
            Stmt::Try(try_stmt) => {
                for s in &try_stmt.block.stmts {
                    self.analyze_stmt_detailed(s, calls, side_effects, has_conditionals, has_early_return);
                }
            }
            _ => {}
        }
    }

    /// Analyze an expression for detailed call/side-effect info
    fn analyze_expr_detailed(
        &self,
        expr: &Expr,
        calls: &mut Vec<HandlerCall>,
        side_effects: &mut Vec<SideEffect>,
        is_awaited: bool,
    ) {
        match expr {
            Expr::Call(call) => {
                let call_info = self.extract_call_info(call, is_awaited);
                if let Some(info) = call_info {
                    // Also add side effects based on call
                    match &info.source {
                        CallSource::StateSetter { .. } => {
                            side_effects.push(SideEffect::StateMutation { target: info.name.clone() });
                        }
                        _ => {
                            if info.name == "fetch" {
                                side_effects.push(SideEffect::ApiCall { method: "fetch".to_string() });
                            }
                        }
                    }
                    calls.push(info);
                }

                // Check for console.log, localStorage, etc.
                if let Callee::Expr(callee_expr) = &call.callee {
                    if let Expr::Member(member) = callee_expr.as_ref() {
                        if let Expr::Ident(obj) = member.obj.as_ref() {
                            let obj_name = obj.sym.as_ref();
                            if obj_name == "console" {
                                side_effects.push(SideEffect::ConsoleLog);
                            } else if obj_name == "localStorage" || obj_name == "sessionStorage" {
                                if let MemberProp::Ident(prop) = &member.prop {
                                    if prop.sym.as_ref() == "setItem" {
                                        side_effects.push(SideEffect::StorageWrite {
                                            storage_type: obj_name.to_string()
                                        });
                                    }
                                }
                            }
                        }
                    }
                }

                // Recurse into arguments
                for arg in &call.args {
                    self.analyze_expr_detailed(&arg.expr, calls, side_effects, false);
                }
            }
            Expr::Await(await_expr) => {
                self.analyze_expr_detailed(&await_expr.arg, calls, side_effects, true);
            }
            Expr::Bin(bin) => {
                self.analyze_expr_detailed(&bin.left, calls, side_effects, false);
                self.analyze_expr_detailed(&bin.right, calls, side_effects, false);
            }
            Expr::Cond(cond) => {
                self.analyze_expr_detailed(&cond.test, calls, side_effects, false);
                self.analyze_expr_detailed(&cond.cons, calls, side_effects, false);
                self.analyze_expr_detailed(&cond.alt, calls, side_effects, false);
            }
            Expr::Arrow(arrow) => {
                // Nested arrow function - analyze its body too
                let (nested_calls, nested_effects, _, _) = self.analyze_handler_body_detailed_arrow(&arrow.body);
                calls.extend(nested_calls);
                side_effects.extend(nested_effects);
            }
            _ => {}
        }
    }

    /// Extract detailed call information from a call expression
    fn extract_call_info(&self, call: &CallExpr, is_awaited: bool) -> Option<HandlerCall> {
        let (name, source) = self.get_call_name_and_source(&call.callee)?;

        let arguments: Vec<String> = call.args.iter()
            .map(|arg| self.span_to_source(self.expr_span(&arg.expr)))
            .collect();

        Some(HandlerCall {
            name,
            source,
            arguments,
            is_async: is_awaited,
        })
    }

    /// Get the name and source of a callee
    fn get_call_name_and_source(&self, callee: &Callee) -> Option<(String, CallSource)> {
        match callee {
            Callee::Expr(expr) => {
                match expr.as_ref() {
                    Expr::Ident(ident) => {
                        let name = ident.sym.to_string();
                        let source = self.classify_call_source(&name);
                        Some((name, source))
                    }
                    Expr::Member(member) => {
                        // e.g., api.get(), console.log()
                        if let MemberProp::Ident(prop) = &member.prop {
                            let method_name = prop.sym.to_string();
                            let obj_name = if let Expr::Ident(obj) = member.obj.as_ref() {
                                Some(obj.sym.to_string())
                            } else {
                                None
                            };

                            let full_name = if let Some(obj) = &obj_name {
                                format!("{}.{}", obj, method_name)
                            } else {
                                method_name.clone()
                            };

                            let source = if obj_name.as_deref() == Some("console") {
                                CallSource::Global
                            } else if obj_name.as_deref().map_or(false, |n| n.contains("api") || n.contains("Api")) {
                                CallSource::Import { module: obj_name.unwrap_or_default() }
                            } else {
                                CallSource::Unknown
                            };

                            Some((full_name, source))
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Classify the source of a function call by name
    fn classify_call_source(&self, name: &str) -> CallSource {
        // State setters: setX, setXxx
        if name.starts_with("set") && name.len() > 3 {
            let state_name = name[3..].to_string();
            // Convert first char to lowercase for state name
            let state_name = if let Some(first) = state_name.chars().next() {
                format!("{}{}", first.to_lowercase(), &state_name[1..])
            } else {
                state_name
            };
            return CallSource::StateSetter { state_name };
        }

        // Common global functions
        if matches!(name, "fetch" | "setTimeout" | "setInterval" | "clearTimeout" | "clearInterval" | "alert" | "confirm") {
            return CallSource::Global;
        }

        // Likely a hook return value if it looks like a function
        if name.chars().next().map_or(false, |c| c.is_lowercase()) {
            return CallSource::Unknown; // Could be from hook, prop, or local - need context
        }

        CallSource::Unknown
    }

    /// Link handler call sources to custom hooks that returned them
    ///
    /// After extracting handlers and custom hooks separately, this function
    /// updates CallSource::Unknown to CallSource::Hook where appropriate.
    fn link_handler_calls_to_hooks(
        handlers: &mut [HandlerExtraction],
        custom_hooks: &[CustomHookUsage],
    ) {
        // Build a map of function name -> hook name for all hook-returned functions
        let mut hook_functions: std::collections::HashMap<String, String> = std::collections::HashMap::new();

        for hook in custom_hooks {
            for ret_val in &hook.returned_values {
                if ret_val.is_function {
                    hook_functions.insert(ret_val.name.clone(), hook.name.clone());
                }
            }
        }

        // Update handler calls that match hook-returned functions
        for handler in handlers.iter_mut() {
            for call in handler.calls.iter_mut() {
                if matches!(call.source, CallSource::Unknown) {
                    if let Some(hook_name) = hook_functions.get(&call.name) {
                        call.source = CallSource::Hook { hook_name: hook_name.clone() };
                    }
                }
            }
        }
    }

    /// Extract handler parameters from function params
    fn extract_handler_params_from_function(&self, params: &[Param]) -> Vec<HandlerParam> {
        params.iter()
            .filter_map(|p| self.extract_handler_param_from_pat(&p.pat))
            .collect()
    }

    /// Extract handler parameters from arrow function params
    fn extract_handler_params_from_arrow(&self, params: &[Pat]) -> Vec<HandlerParam> {
        params.iter()
            .filter_map(|p| self.extract_handler_param_from_pat(p))
            .collect()
    }

    /// Extract a single handler param from a pattern
    fn extract_handler_param_from_pat(&self, pat: &Pat) -> Option<HandlerParam> {
        match pat {
            Pat::Ident(ident) => {
                Some(HandlerParam {
                    name: ident.id.sym.to_string(),
                    type_annotation: ident.type_ann.as_ref()
                        .map(|ann| self.span_to_source(ann.span)),
                })
            }
            Pat::Object(obj) => {
                Some(HandlerParam {
                    name: "{ ... }".to_string(),
                    type_annotation: obj.type_ann.as_ref()
                        .map(|ann| self.span_to_source(ann.span)),
                })
            }
            _ => None,
        }
    }

    fn extract_props_from_params(&self, params: &[Param]) -> Vec<PropExtraction> {
        let mut props = Vec::new();

        for param in params {
            self.extract_props_from_pattern(&param.pat, &mut props);
        }

        props
    }

    fn extract_props_from_arrow_params(&self, params: &[Pat]) -> Vec<PropExtraction> {
        let mut props = Vec::new();

        for pat in params {
            self.extract_props_from_pattern(pat, &mut props);
        }

        props
    }

    fn extract_props_from_pattern(&self, pat: &Pat, props: &mut Vec<PropExtraction>) {
        match pat {
            // Destructuring: ({ name, value, onChange })
            Pat::Object(obj) => {
                for prop in &obj.props {
                    match prop {
                        ObjectPatProp::KeyValue(kv) => {
                            if let PropName::Ident(key) = &kv.key {
                                let name = key.sym.to_string();
                                let is_callback = name.starts_with("on") && name.len() > 2 &&
                                    name.chars().nth(2).map_or(false, |c| c.is_uppercase());
                                let is_children = name == "children";

                                props.push(PropExtraction {
                                    name,
                                    type_annotation: None, // Would need type info from context
                                    required: true,
                                    default_value: None,
                                    is_callback,
                                    is_children,
                                });
                            }
                        }
                        ObjectPatProp::Assign(assign) => {
                            let name = assign.key.sym.to_string();
                            let default_value = assign.value.as_ref()
                                .map(|v| self.span_to_source((**v).span()));
                            let is_callback = name.starts_with("on") && name.len() > 2 &&
                                name.chars().nth(2).map_or(false, |c| c.is_uppercase());
                            let is_children = name == "children";

                            props.push(PropExtraction {
                                name,
                                type_annotation: None,
                                required: default_value.is_none(),
                                default_value,
                                is_callback,
                                is_children,
                            });
                        }
                        ObjectPatProp::Rest(rest) => {
                            if let Pat::Ident(ident) = &*rest.arg {
                                props.push(PropExtraction {
                                    name: format!("...{}", ident.id.sym),
                                    type_annotation: None,
                                    required: false,
                                    default_value: None,
                                    is_callback: false,
                                    is_children: false,
                                });
                            }
                        }
                    }
                }
            }
            // Simple parameter: (props)
            Pat::Ident(ident) => {
                // If first param is named "props", we can't know individual props
                // But we note it exists
                let name = ident.id.sym.to_string();
                if name == "props" || name == "p" {
                    // This is a props object, not destructured
                    props.push(PropExtraction {
                        name: "props".to_string(),
                        type_annotation: ident.type_ann.as_ref().map(|ann| self.span_to_source(ann.span)),
                        required: true,
                        default_value: None,
                        is_callback: false,
                        is_children: false,
                    });
                }
            }
            _ => {}
        }
    }

    fn try_extract_type_from_decl(&self, decl: &Decl, exported: bool) -> Option<TypeExtraction> {
        match decl {
            Decl::TsInterface(interface) => {
                Some(self.extract_interface(interface, exported))
            }
            Decl::TsTypeAlias(alias) => {
                Some(self.extract_type_alias(alias, exported))
            }
            Decl::TsEnum(enum_decl) => {
                Some(self.extract_enum(enum_decl, exported))
            }
            _ => None,
        }
    }

    // =========================================================================
    // Phase 6.1: Full Type Extraction
    // =========================================================================

    fn extract_interface(&self, interface: &TsInterfaceDecl, exported: bool) -> TypeExtraction {
        let name = interface.id.sym.to_string();
        let definition = self.span_to_source(interface.span);

        // Extract type parameters
        let type_params = interface.type_params.as_ref()
            .map(|params| self.extract_type_params(params))
            .unwrap_or_default();

        // Extract extends clauses
        let extends: Vec<String> = interface.extends.iter()
            .map(|ext| self.span_to_source(ext.span))
            .collect();

        // Extract fields from interface body
        let fields = self.extract_interface_fields(&interface.body);

        TypeExtraction {
            name,
            kind: TypeKind::Interface,
            location: self.span_to_location(interface.span),
            exported,
            definition,
            fields,
            type_params,
            extends,
            union_variants: Vec::new(),
            doc_comment: None, // TODO: extract from leading comments
        }
    }

    fn extract_type_alias(&self, alias: &TsTypeAliasDecl, exported: bool) -> TypeExtraction {
        let name = alias.id.sym.to_string();
        let definition = self.span_to_source(alias.span);

        // Extract type parameters
        let type_params = alias.type_params.as_ref()
            .map(|params| self.extract_type_params(params))
            .unwrap_or_default();

        // Check if this is a union type
        let (fields, union_variants) = match alias.type_ann.as_ref() {
            TsType::TsUnionOrIntersectionType(TsUnionOrIntersectionType::TsUnionType(union)) => {
                let variants: Vec<String> = union.types.iter()
                    .map(|t| self.ts_type_to_string(t))
                    .collect();
                (Vec::new(), variants)
            }
            TsType::TsTypeLit(lit) => {
                // Object type literal: type X = { field: string }
                let fields = self.extract_type_lit_fields(lit);
                (fields, Vec::new())
            }
            _ => (Vec::new(), Vec::new()),
        };

        TypeExtraction {
            name,
            kind: TypeKind::TypeAlias,
            location: self.span_to_location(alias.span),
            exported,
            definition,
            fields,
            type_params,
            extends: Vec::new(),
            union_variants,
            doc_comment: None,
        }
    }

    fn extract_enum(&self, enum_decl: &TsEnumDecl, exported: bool) -> TypeExtraction {
        let name = enum_decl.id.sym.to_string();
        let definition = self.span_to_source(enum_decl.span);

        // Extract enum members as union variants
        let union_variants: Vec<String> = enum_decl.members.iter()
            .map(|member| {
                let member_name = match &member.id {
                    TsEnumMemberId::Ident(ident) => ident.sym.to_string(),
                    TsEnumMemberId::Str(s) => s.value.as_str().unwrap_or("").to_string(),
                };
                if let Some(init) = &member.init {
                    format!("{} = {}", member_name, self.span_to_source(self.expr_span(init)))
                } else {
                    member_name
                }
            })
            .collect();

        TypeExtraction {
            name,
            kind: TypeKind::Enum,
            location: self.span_to_location(enum_decl.span),
            exported,
            definition,
            fields: Vec::new(),
            type_params: Vec::new(),
            extends: Vec::new(),
            union_variants,
            doc_comment: None,
        }
    }

    fn extract_type_params(&self, params: &TsTypeParamDecl) -> Vec<TypeParamExtraction> {
        params.params.iter()
            .map(|param| {
                TypeParamExtraction {
                    name: param.name.sym.to_string(),
                    constraint: param.constraint.as_ref()
                        .map(|c| self.ts_type_to_string(c)),
                    default: param.default.as_ref()
                        .map(|d| self.ts_type_to_string(d)),
                }
            })
            .collect()
    }

    fn extract_interface_fields(&self, body: &TsInterfaceBody) -> Vec<TypeFieldExtraction> {
        body.body.iter()
            .filter_map(|elem| self.extract_type_element(elem))
            .collect()
    }

    fn extract_type_lit_fields(&self, lit: &TsTypeLit) -> Vec<TypeFieldExtraction> {
        lit.members.iter()
            .filter_map(|elem| self.extract_type_element(elem))
            .collect()
    }

    fn extract_type_element(&self, elem: &TsTypeElement) -> Option<TypeFieldExtraction> {
        match elem {
            TsTypeElement::TsPropertySignature(prop) => {
                let name = match &*prop.key {
                    Expr::Ident(ident) => ident.sym.to_string(),
                    Expr::Lit(Lit::Str(s)) => s.value.as_str().unwrap_or("").to_string(),
                    _ => return None,
                };

                let (type_annotation, type_kind) = prop.type_ann.as_ref()
                    .map(|ann| {
                        let ts_type = &*ann.type_ann;
                        let annotation = self.ts_type_to_string(ts_type);
                        let kind = self.classify_ts_type(ts_type);
                        (annotation, kind)
                    })
                    .unwrap_or_else(|| ("unknown".to_string(), TypeFieldKind::Complex { raw: "unknown".to_string() }));

                Some(TypeFieldExtraction {
                    name,
                    type_annotation,
                    optional: prop.optional,
                    readonly: prop.readonly,
                    type_kind,
                    doc_comment: None,
                })
            }
            TsTypeElement::TsMethodSignature(method) => {
                let name = match &*method.key {
                    Expr::Ident(ident) => ident.sym.to_string(),
                    _ => return None,
                };

                // Extract method parameters
                let params: Vec<FunctionParam> = method.params.iter()
                    .map(|p| self.extract_fn_param(p))
                    .collect();

                let return_type = method.type_ann.as_ref()
                    .map(|ann| self.ts_type_to_string(&*ann.type_ann))
                    .unwrap_or_else(|| "void".to_string());

                let type_annotation = self.format_function_type(&params, &return_type);

                Some(TypeFieldExtraction {
                    name,
                    type_annotation: type_annotation.clone(),
                    optional: method.optional,
                    readonly: false,
                    type_kind: TypeFieldKind::Function { params, return_type },
                    doc_comment: None,
                })
            }
            TsTypeElement::TsIndexSignature(index) => {
                // { [key: string]: value }
                let key_type = index.params.first()
                    .map(|p| self.extract_fn_param(p).type_annotation)
                    .unwrap_or_else(|| "string".to_string());

                let value_type = index.type_ann.as_ref()
                    .map(|ann| self.ts_type_to_string(&*ann.type_ann))
                    .unwrap_or_else(|| "unknown".to_string());

                Some(TypeFieldExtraction {
                    name: format!("[key: {}]", key_type),
                    type_annotation: value_type.clone(),
                    optional: false,
                    readonly: index.readonly,
                    type_kind: TypeFieldKind::Record { key_type, value_type },
                    doc_comment: None,
                })
            }
            _ => None,
        }
    }

    fn extract_fn_param(&self, param: &TsFnParam) -> FunctionParam {
        match param {
            TsFnParam::Ident(ident) => {
                let type_annotation = ident.type_ann.as_ref()
                    .map(|ann| self.ts_type_to_string(&*ann.type_ann))
                    .unwrap_or_else(|| "unknown".to_string());
                FunctionParam {
                    name: Some(ident.id.sym.to_string()),
                    type_annotation,
                    optional: ident.id.optional,
                }
            }
            TsFnParam::Array(arr) => {
                let type_annotation = arr.type_ann.as_ref()
                    .map(|ann| self.ts_type_to_string(&*ann.type_ann))
                    .unwrap_or_else(|| "unknown[]".to_string());
                FunctionParam {
                    name: None,
                    type_annotation,
                    optional: false,
                }
            }
            TsFnParam::Object(obj) => {
                let type_annotation = obj.type_ann.as_ref()
                    .map(|ann| self.ts_type_to_string(&*ann.type_ann))
                    .unwrap_or_else(|| "object".to_string());
                FunctionParam {
                    name: None,
                    type_annotation,
                    optional: false,
                }
            }
            TsFnParam::Rest(rest) => {
                let type_annotation = rest.type_ann.as_ref()
                    .map(|ann| self.ts_type_to_string(&*ann.type_ann))
                    .unwrap_or_else(|| "unknown[]".to_string());
                FunctionParam {
                    name: self.get_ident_from_pattern(&rest.arg),
                    type_annotation,
                    optional: false,
                }
            }
        }
    }

    fn format_function_type(&self, params: &[FunctionParam], return_type: &str) -> String {
        let params_str: Vec<String> = params.iter()
            .map(|p| {
                let name = p.name.as_deref().unwrap_or("arg");
                let opt = if p.optional { "?" } else { "" };
                format!("{}{}: {}", name, opt, p.type_annotation)
            })
            .collect();
        format!("({}) => {}", params_str.join(", "), return_type)
    }

    fn classify_ts_type(&self, ts_type: &TsType) -> TypeFieldKind {
        match ts_type {
            TsType::TsKeywordType(kw) => {
                let name = match kw.kind {
                    TsKeywordTypeKind::TsStringKeyword => "string",
                    TsKeywordTypeKind::TsNumberKeyword => "number",
                    TsKeywordTypeKind::TsBooleanKeyword => "boolean",
                    TsKeywordTypeKind::TsNullKeyword => "null",
                    TsKeywordTypeKind::TsUndefinedKeyword => "undefined",
                    TsKeywordTypeKind::TsVoidKeyword => "void",
                    TsKeywordTypeKind::TsAnyKeyword => "any",
                    TsKeywordTypeKind::TsNeverKeyword => "never",
                    TsKeywordTypeKind::TsUnknownKeyword => "unknown",
                    TsKeywordTypeKind::TsObjectKeyword => "object",
                    TsKeywordTypeKind::TsBigIntKeyword => "bigint",
                    TsKeywordTypeKind::TsSymbolKeyword => "symbol",
                    TsKeywordTypeKind::TsIntrinsicKeyword => "intrinsic",
                };
                TypeFieldKind::Primitive { name: name.to_string() }
            }
            TsType::TsTypeRef(type_ref) => {
                let name = self.ts_entity_name_to_string(&type_ref.type_name);
                let type_args: Vec<String> = type_ref.type_params.as_ref()
                    .map(|params| params.params.iter()
                        .map(|p| self.ts_type_to_string(p))
                        .collect())
                    .unwrap_or_default();

                // Check for Array<T> pattern
                if name == "Array" && type_args.len() == 1 {
                    return TypeFieldKind::Array { element_type: type_args[0].clone() };
                }

                // Check for Record<K, V> pattern
                if name == "Record" && type_args.len() == 2 {
                    return TypeFieldKind::Record {
                        key_type: type_args[0].clone(),
                        value_type: type_args[1].clone()
                    };
                }

                TypeFieldKind::TypeRef { name, type_args }
            }
            TsType::TsArrayType(arr) => {
                let element_type = self.ts_type_to_string(&arr.elem_type);
                TypeFieldKind::Array { element_type }
            }
            TsType::TsUnionOrIntersectionType(TsUnionOrIntersectionType::TsUnionType(union)) => {
                let variants: Vec<String> = union.types.iter()
                    .map(|t| self.ts_type_to_string(t))
                    .collect();
                TypeFieldKind::Union { variants }
            }
            TsType::TsFnOrConstructorType(TsFnOrConstructorType::TsFnType(fn_type)) => {
                let params: Vec<FunctionParam> = fn_type.params.iter()
                    .map(|p| self.extract_fn_param(p))
                    .collect();
                let return_type = self.ts_type_to_string(&fn_type.type_ann.type_ann);
                TypeFieldKind::Function { params, return_type }
            }
            TsType::TsTupleType(tuple) => {
                let element_types: Vec<String> = tuple.elem_types.iter()
                    .map(|elem| self.ts_type_to_string(&elem.ty))
                    .collect();
                TypeFieldKind::Tuple { element_types }
            }
            TsType::TsLitType(lit) => {
                let value = match &lit.lit {
                    TsLit::Str(s) => format!("'{}'", s.value.as_str().unwrap_or("")),
                    TsLit::Number(n) => n.value.to_string(),
                    TsLit::Bool(b) => b.value.to_string(),
                    TsLit::BigInt(bi) => bi.value.to_string(),
                    TsLit::Tpl(_) => "template".to_string(),
                };
                TypeFieldKind::Literal { value }
            }
            TsType::TsTypeLit(_) => {
                TypeFieldKind::Complex { raw: self.span_to_source(ts_type.span()) }
            }
            _ => {
                TypeFieldKind::Complex { raw: self.span_to_source(ts_type.span()) }
            }
        }
    }

    fn ts_type_to_string(&self, ts_type: &TsType) -> String {
        // Use the source span for accurate representation
        self.span_to_source(ts_type.span())
    }

    fn span_to_location(&self, span: swc_common::Span) -> SourceLocation {
        let start = self.cm.lookup_char_pos(span.lo);
        let end = self.cm.lookup_char_pos(span.hi);
        SourceLocation {
            start_line: start.line as u32,
            start_col: start.col_display as u32,
            end_line: end.line as u32,
            end_col: end.col_display as u32,
        }
    }

    // =========================================================================
    // Phase 6.2: Helper Function Extraction
    // =========================================================================

    /// Try to extract a helper function from a declaration.
    /// Helper functions are module-scope functions that don't return JSX.
    fn try_extract_helper_function_from_decl(&self, decl: &Decl, exported: bool) -> Option<HelperFunctionExtraction> {
        match decl {
            Decl::Fn(fn_decl) => {
                let name = fn_decl.ident.sym.to_string();
                // Skip hooks and components (start with uppercase or "use")
                if name.starts_with("use") || name.chars().next().map_or(false, |c| c.is_uppercase()) {
                    return None;
                }
                self.extract_helper_function(&name, &fn_decl.function, exported)
            }
            Decl::Var(var_decl) => {
                for decl in &var_decl.decls {
                    if let Pat::Ident(ident) = &decl.name {
                        let name = ident.id.sym.to_string();
                        // Skip hooks and components
                        if name.starts_with("use") || name.chars().next().map_or(false, |c| c.is_uppercase()) {
                            continue;
                        }
                        if let Some(init) = &decl.init {
                            if let Some(helper) = self.try_extract_helper_from_expr(&name, init, exported) {
                                return Some(helper);
                            }
                        }
                    }
                }
                None
            }
            _ => None,
        }
    }

    /// Try to extract helper function from an arrow/function expression
    fn try_extract_helper_from_expr(&self, name: &str, expr: &Expr, exported: bool) -> Option<HelperFunctionExtraction> {
        match expr {
            Expr::Arrow(arrow) => {
                // Check if this arrow returns JSX - if so, it's likely a component
                let returns_jsx = match &*arrow.body {
                    BlockStmtOrExpr::Expr(e) => matches!(e.as_ref(), Expr::JSXElement(_) | Expr::JSXFragment(_)),
                    BlockStmtOrExpr::BlockStmt(block) => self.block_returns_jsx(block),
                };
                if returns_jsx {
                    return None;
                }

                let parameters = self.extract_arrow_params(&arrow.params);
                let return_type = arrow.return_type.as_ref()
                    .map(|rt| self.span_to_source(rt.span));
                let (side_effects, calls) = self.analyze_function_body_arrow_6_2(&arrow.body);
                let is_pure = side_effects.is_empty();

                Some(HelperFunctionExtraction {
                    name: name.to_string(),
                    location: self.span_to_location(arrow.span),
                    exported,
                    is_async: arrow.is_async,
                    is_generator: arrow.is_generator,
                    parameters,
                    return_type,
                    is_pure,
                    side_effects,
                    calls,
                    used_by: Vec::new(), // Would need cross-file analysis
                    source: self.span_to_source(arrow.span),
                })
            }
            Expr::Fn(fn_expr) => {
                self.extract_helper_function(name, &fn_expr.function, exported)
            }
            _ => None,
        }
    }

    /// Extract a helper function from a Function node
    fn extract_helper_function(&self, name: &str, func: &Function, exported: bool) -> Option<HelperFunctionExtraction> {
        // Check if function returns JSX - if so, it's likely a component
        if func.body.as_ref().map_or(false, |b| self.block_returns_jsx(b)) {
            return None;
        }

        let parameters = self.extract_fn_params_6_2(&func.params);
        let return_type = func.return_type.as_ref()
            .map(|rt| self.span_to_source(rt.span));
        let (side_effects, calls) = self.analyze_function_body_block_6_2(&func.body);
        let is_pure = side_effects.is_empty();

        Some(HelperFunctionExtraction {
            name: name.to_string(),
            location: self.span_to_location(func.span),
            exported,
            is_async: func.is_async,
            is_generator: func.is_generator,
            parameters,
            return_type,
            is_pure,
            side_effects,
            calls,
            used_by: Vec::new(),
            source: self.span_to_source(func.span),
        })
    }

    /// Check if a block returns JSX (used to distinguish components from helpers)
    fn block_returns_jsx(&self, block: &BlockStmt) -> bool {
        for stmt in &block.stmts {
            if let Stmt::Return(ret) = stmt {
                if let Some(arg) = &ret.arg {
                    if matches!(arg.as_ref(), Expr::JSXElement(_) | Expr::JSXFragment(_)) {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Extract function parameters for Phase 6.2
    fn extract_fn_params_6_2(&self, params: &[Param]) -> Vec<FunctionParamExtraction> {
        params.iter()
            .map(|p| self.extract_param_pattern(&p.pat))
            .collect()
    }

    /// Extract arrow function parameters
    fn extract_arrow_params(&self, params: &[Pat]) -> Vec<FunctionParamExtraction> {
        params.iter()
            .map(|p| self.extract_param_pattern(p))
            .collect()
    }

    /// Extract a single parameter from a pattern
    fn extract_param_pattern(&self, pat: &Pat) -> FunctionParamExtraction {
        match pat {
            Pat::Ident(ident) => {
                FunctionParamExtraction {
                    name: ident.id.sym.to_string(),
                    type_annotation: ident.type_ann.as_ref()
                        .map(|ann| self.span_to_source(ann.span)),
                    optional: ident.id.optional,
                    default_value: None,
                    is_rest: false,
                }
            }
            Pat::Assign(assign) => {
                let mut param = self.extract_param_pattern(&assign.left);
                param.default_value = Some(self.span_to_source(self.expr_span(&assign.right)));
                param.optional = true;
                param
            }
            Pat::Rest(rest) => {
                let mut param = self.extract_param_pattern(&rest.arg);
                param.is_rest = true;
                param
            }
            Pat::Object(obj) => {
                FunctionParamExtraction {
                    name: "{ ... }".to_string(),
                    type_annotation: obj.type_ann.as_ref()
                        .map(|ann| self.span_to_source(ann.span)),
                    optional: obj.optional,
                    default_value: None,
                    is_rest: false,
                }
            }
            Pat::Array(arr) => {
                FunctionParamExtraction {
                    name: "[ ... ]".to_string(),
                    type_annotation: arr.type_ann.as_ref()
                        .map(|ann| self.span_to_source(ann.span)),
                    optional: arr.optional,
                    default_value: None,
                    is_rest: false,
                }
            }
            _ => {
                FunctionParamExtraction {
                    name: "unknown".to_string(),
                    type_annotation: None,
                    optional: false,
                    default_value: None,
                    is_rest: false,
                }
            }
        }
    }

    /// Analyze function body for side effects and function calls (Phase 6.2)
    fn analyze_function_body_block_6_2(&self, body: &Option<BlockStmt>) -> (Vec<SideEffect>, Vec<String>) {
        let Some(block) = body else {
            return (Vec::new(), Vec::new());
        };

        let mut side_effects = Vec::new();
        let mut calls = Vec::new();

        for stmt in &block.stmts {
            self.analyze_stmt_for_side_effects(stmt, &mut side_effects, &mut calls);
        }

        (side_effects, calls)
    }

    /// Analyze arrow function body for side effects
    fn analyze_function_body_arrow_6_2(&self, body: &BlockStmtOrExpr) -> (Vec<SideEffect>, Vec<String>) {
        match body {
            BlockStmtOrExpr::BlockStmt(block) => self.analyze_function_body_block_6_2(&Some(block.clone())),
            BlockStmtOrExpr::Expr(expr) => {
                let mut side_effects = Vec::new();
                let mut calls = Vec::new();
                self.analyze_expr_for_side_effects(expr, &mut side_effects, &mut calls);
                (side_effects, calls)
            }
        }
    }

    /// Analyze a statement for side effects
    fn analyze_stmt_for_side_effects(&self, stmt: &Stmt, side_effects: &mut Vec<SideEffect>, calls: &mut Vec<String>) {
        match stmt {
            Stmt::Expr(expr_stmt) => {
                self.analyze_expr_for_side_effects(&expr_stmt.expr, side_effects, calls);
            }
            Stmt::Return(ret) => {
                if let Some(arg) = &ret.arg {
                    self.analyze_expr_for_side_effects(arg, side_effects, calls);
                }
            }
            Stmt::If(if_stmt) => {
                self.analyze_expr_for_side_effects(&if_stmt.test, side_effects, calls);
                if let Stmt::Block(block) = &*if_stmt.cons {
                    for s in &block.stmts {
                        self.analyze_stmt_for_side_effects(s, side_effects, calls);
                    }
                }
                if let Some(alt) = &if_stmt.alt {
                    if let Stmt::Block(block) = alt.as_ref() {
                        for s in &block.stmts {
                            self.analyze_stmt_for_side_effects(s, side_effects, calls);
                        }
                    }
                }
            }
            Stmt::Decl(Decl::Var(var_decl)) => {
                for decl in &var_decl.decls {
                    if let Some(init) = &decl.init {
                        self.analyze_expr_for_side_effects(init, side_effects, calls);
                    }
                }
            }
            Stmt::Try(try_stmt) => {
                // Analyze try block
                for s in &try_stmt.block.stmts {
                    self.analyze_stmt_for_side_effects(s, side_effects, calls);
                }
                // Analyze catch block if present
                if let Some(catch) = &try_stmt.handler {
                    for s in &catch.body.stmts {
                        self.analyze_stmt_for_side_effects(s, side_effects, calls);
                    }
                }
                // Analyze finally block if present
                if let Some(finally) = &try_stmt.finalizer {
                    for s in &finally.stmts {
                        self.analyze_stmt_for_side_effects(s, side_effects, calls);
                    }
                }
            }
            Stmt::Block(block) => {
                for s in &block.stmts {
                    self.analyze_stmt_for_side_effects(s, side_effects, calls);
                }
            }
            _ => {}
        }
    }

    /// Analyze an expression for side effects and function calls
    fn analyze_expr_for_side_effects(&self, expr: &Expr, side_effects: &mut Vec<SideEffect>, calls: &mut Vec<String>) {
        match expr {
            Expr::Call(call) => {
                // Check for console.log, localStorage, etc. first (before generic setX detection)
                let mut is_known_method_call = false;
                if let Callee::Expr(callee_expr) = &call.callee {
                    if let Expr::Member(member) = callee_expr.as_ref() {
                        if let Expr::Ident(obj) = member.obj.as_ref() {
                            let obj_name = obj.sym.as_ref();
                            is_known_method_call = matches!(obj_name, "console" | "localStorage" | "sessionStorage" | "document");
                            if obj_name == "console" {
                                side_effects.push(SideEffect::ConsoleLog);
                            } else if obj_name == "localStorage" || obj_name == "sessionStorage" {
                                if let MemberProp::Ident(prop) = &member.prop {
                                    if prop.sym.as_ref() == "setItem" {
                                        side_effects.push(SideEffect::StorageWrite {
                                            storage_type: obj_name.to_string()
                                        });
                                    }
                                }
                            } else if obj_name == "document" {
                                side_effects.push(SideEffect::DomMutation {
                                    operation: self.span_to_source(call.span)
                                });
                            }
                        }
                    }
                }

                // Recurse into arguments
                for arg in &call.args {
                    self.analyze_expr_for_side_effects(&arg.expr, side_effects, calls);
                }
            }
            Expr::Await(await_expr) => {
                self.analyze_expr_for_side_effects(&await_expr.arg, side_effects, calls);
            }
            Expr::Assign(assign) => {
                // Assignment to global or object property could be a side effect
                if let AssignTarget::Simple(SimpleAssignTarget::Member(_)) = &assign.left {
                    side_effects.push(SideEffect::Unknown {
                        description: "property assignment".to_string()
                    });
                }
                self.analyze_expr_for_side_effects(&assign.right, side_effects, calls);
            }
            Expr::Bin(bin) => {
                self.analyze_expr_for_side_effects(&bin.left, side_effects, calls);
                self.analyze_expr_for_side_effects(&bin.right, side_effects, calls);
            }
            Expr::Cond(cond) => {
                self.analyze_expr_for_side_effects(&cond.test, side_effects, calls);
                self.analyze_expr_for_side_effects(&cond.cons, side_effects, calls);
                self.analyze_expr_for_side_effects(&cond.alt, side_effects, calls);
            }
            _ => {}
        }
    }

    // =========================================================================
    // Phase 6.5: Architecture Mapping
    // =========================================================================

    /// Generate architecture recommendations from hooks and handlers
    fn generate_architecture_recommendation(
        &self,
        hooks: &[HookUsage],
        custom_hooks: &[CustomHookUsage],
        handlers: &[HandlerExtraction],
    ) -> ArchitectureRecommendation {
        let mut service_actors = Vec::new();
        let mut state_ownership = Vec::new();
        let mut communication_patterns = Vec::new();
        let mut zustand_stores = Vec::new();

        // Track unique hook-derived actors
        let mut hook_actors: std::collections::HashMap<String, ServiceActorRecommendation> = std::collections::HashMap::new();

        // Process custom hooks to identify service actors
        for hook in custom_hooks {
            if hook.is_zustand {
                // Handle Zustand stores
                let store_name = self.extract_store_name_from_hook(&hook.name);
                let suggested_actor = format!("{}Actor", store_name);

                // Collect selectors and actions
                let mut selectors_used = Vec::new();
                let mut actions_used = Vec::new();

                for ret_val in &hook.returned_values {
                    if ret_val.is_function {
                        actions_used.push(ret_val.name.clone());
                    } else {
                        selectors_used.push(ret_val.name.clone());
                    }
                }

                // Check expanded arguments for selector info
                for arg in &hook.expanded_arguments {
                    if let HookArgument::Function { body_summary, .. } = arg {
                        // Selector like (s) => s.serverStatus
                        if body_summary.starts_with("s.") || body_summary.starts_with("state.") {
                            let selector = body_summary.split('.').last().unwrap_or(body_summary).to_string();
                            if !selectors_used.contains(&selector) {
                                selectors_used.push(selector);
                            }
                        }
                    } else if let HookArgument::Expression { value } = arg {
                        // Named selector like selectIsModelLoaded
                        if value.starts_with("select") {
                            selectors_used.push(value.clone());
                        }
                    }
                }

                zustand_stores.push(ZustandStoreMapping {
                    hook_name: hook.name.clone(),
                    suggested_actor: suggested_actor.clone(),
                    selectors_used: selectors_used.clone(),
                    actions_used: actions_used.clone(),
                });

                // Create/update state ownership entries
                for selector in &selectors_used {
                    state_ownership.push(StateOwnership {
                        state_name: selector.clone(),
                        owner: suggested_actor.clone(),
                        access_pattern: StateAccessPattern::Shared,
                        source: hook.name.clone(),
                    });
                }
            } else {
                // Non-Zustand custom hook (useChat, useAgent, etc.)
                let actor_name = self.derive_actor_name_from_hook(&hook.name);

                let entry = hook_actors.entry(actor_name.clone()).or_insert_with(|| {
                    ServiceActorRecommendation {
                        name: actor_name.clone(),
                        derived_from: Vec::new(),
                        responsibilities: Vec::new(),
                        owned_state: Vec::new(),
                        messages: Vec::new(),
                    }
                });

                if !entry.derived_from.contains(&hook.name) {
                    entry.derived_from.push(hook.name.clone());
                }

                // Extract state from returned values
                for ret_val in &hook.returned_values {
                    if ret_val.is_function {
                        // Functions become messages
                        let msg_name = self.to_message_name(&ret_val.name);
                        if !entry.messages.iter().any(|m| m.name == msg_name) {
                            entry.messages.push(ActorMessage {
                                name: msg_name,
                                message_type: MessageType::Command,
                                payload: Vec::new(),
                                is_async: true, // Assume async by default
                            });
                        }
                    } else {
                        // Data becomes owned state
                        if !entry.owned_state.contains(&ret_val.name) {
                            entry.owned_state.push(ret_val.name.clone());
                        }

                        state_ownership.push(StateOwnership {
                            state_name: ret_val.name.clone(),
                            owner: actor_name.clone(),
                            access_pattern: StateAccessPattern::Shared,
                            source: hook.name.clone(),
                        });
                    }
                }
            }
        }

        // Process standard hooks for local state
        for hook in hooks {
            if let HookType::UseState = hook.hook_type {
                if let Some(ref state_name) = hook.state_name {
                    state_ownership.push(StateOwnership {
                        state_name: state_name.clone(),
                        owner: "Self".to_string(), // Component owns this state
                        access_pattern: StateAccessPattern::Local,
                        source: "useState".to_string(),
                    });
                }
            }

            if let HookType::UseContext = hook.hook_type {
                if let Some(ref context_name) = hook.context_name {
                    state_ownership.push(StateOwnership {
                        state_name: context_name.clone(),
                        owner: context_name.clone(),
                        access_pattern: StateAccessPattern::Shared,
                        source: "useContext".to_string(),
                    });
                }
            }
        }

        // Convert hook_actors map to vec
        service_actors.extend(hook_actors.into_values());

        // Analyze handlers for communication patterns
        for handler in handlers {
            for call in &handler.calls {
                match &call.source {
                    CallSource::Hook { hook_name } => {
                        // Component communicates with hook-derived actor
                        let actor_name = self.derive_actor_name_from_hook(hook_name);
                        communication_patterns.push(CommunicationPattern {
                            from: "Self".to_string(),
                            to: actor_name,
                            pattern: if call.is_async {
                                CommunicationPatternType::RequestResponse
                            } else {
                                CommunicationPatternType::Direct
                            },
                            messages: vec![self.to_message_name(&call.name)],
                        });
                    }
                    CallSource::StateSetter { state_name } => {
                        // State mutation - check if shared
                        if let Some(ownership) = state_ownership.iter().find(|s| &s.state_name == state_name) {
                            if ownership.owner != "Self" {
                                communication_patterns.push(CommunicationPattern {
                                    from: "Self".to_string(),
                                    to: ownership.owner.clone(),
                                    pattern: CommunicationPatternType::Direct,
                                    messages: vec![format!("Set{}", capitalize_first(state_name))],
                                });
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // Deduplicate communication patterns
        communication_patterns.sort_by(|a, b| {
            (&a.from, &a.to, &a.messages).cmp(&(&b.from, &b.to, &b.messages))
        });
        communication_patterns.dedup_by(|a, b| {
            a.from == b.from && a.to == b.to && a.messages == b.messages
        });

        ArchitectureRecommendation {
            service_actors,
            state_ownership,
            communication_patterns,
            zustand_stores,
        }
    }

    /// Extract store name from hook name (e.g., "useInfernumStore" → "Infernum")
    fn extract_store_name_from_hook(&self, hook_name: &str) -> String {
        let name = hook_name.strip_prefix("use").unwrap_or(hook_name);
        let name = name.strip_suffix("Store").unwrap_or(name);
        if name.is_empty() {
            "Store".to_string()
        } else {
            name.to_string()
        }
    }

    /// Derive actor name from hook name (e.g., "useChat" → "ChatService")
    fn derive_actor_name_from_hook(&self, hook_name: &str) -> String {
        let name = hook_name.strip_prefix("use").unwrap_or(hook_name);
        if name.is_empty() {
            "Service".to_string()
        } else {
            format!("{}Service", name)
        }
    }

    /// Convert function name to message name (e.g., "addMessage" → "AddMessage")
    fn to_message_name(&self, func_name: &str) -> String {
        capitalize_first(func_name)
    }
}

/// Capitalize the first letter of a string
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(first) => first.to_uppercase().collect::<String>() + chars.as_str(),
    }
}
