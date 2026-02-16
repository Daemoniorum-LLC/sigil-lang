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

    // Class components
    pub class_info: Option<ClassComponentInfo>,

    // JSX structure
    pub jsx: JsxTree,

    // Event handlers
    pub handlers: Vec<HandlerExtraction>,

    // Dependencies
    pub child_components: Vec<String>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomHookExtraction {
    pub name: String,
    pub location: SourceLocation,
    pub parameters: Vec<String>,
    pub return_type: Option<String>,
    pub hooks_used: Vec<HookType>,
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
// Handler Extraction
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandlerExtraction {
    pub name: String,
    pub event_type: Option<String>,
    pub is_async: bool,
    pub body_summary: String,
    pub state_mutations: Vec<String>,
    pub api_calls: Vec<String>,
}

// =============================================================================
// Type Extraction
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeExtraction {
    pub name: String,
    pub kind: TypeKind,
    pub location: SourceLocation,
    pub definition: String,
    pub exported: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TypeKind {
    Interface,
    TypeAlias,
    Enum,
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

        for item in &module.body {
            match item {
                ModuleItem::ModuleDecl(decl) => {
                    self.process_module_decl(decl, &mut components, &mut types, &mut imports, &mut exports);
                }
                ModuleItem::Stmt(stmt) => {
                    self.process_stmt(stmt, &mut components, &mut custom_hooks, &mut types);
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
        })
    }

    fn process_module_decl(
        &mut self,
        decl: &ModuleDecl,
        components: &mut Vec<ComponentExtraction>,
        types: &mut Vec<TypeExtraction>,
        imports: &mut Vec<ImportInfo>,
        exports: &mut Vec<ExportInfo>,
    ) {
        match decl {
            ModuleDecl::Import(import) => {
                imports.push(self.extract_import(import));
            }
            ModuleDecl::ExportDecl(export) => {
                if let Some(comp) = self.try_extract_component_from_decl(&export.decl, true, Some(ExportType::Named)) {
                    components.push(comp);
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
        custom_hooks: &mut Vec<CustomHookExtraction>,
        types: &mut Vec<TypeExtraction>,
    ) {
        match stmt {
            Stmt::Decl(decl) => {
                if let Some(comp) = self.try_extract_component_from_decl(decl, false, None) {
                    components.push(comp);
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
        let handlers = self.extract_handlers_from_body(&function.body);
        let props = self.extract_props_from_params(&function.params);

        Some(ComponentExtraction {
            name: name.to_string(),
            component_type: ComponentType::Functional,
            exported,
            export_type,
            location: self.span_to_location(function.span),
            props,
            props_type: None, // TODO: extract from type annotation
            hooks,
            class_info: None,
            jsx: jsx.clone(),
            handlers,
            child_components: self.extract_child_components_from_jsx(&jsx),
        })
    }

    fn try_extract_arrow_component(&mut self, arrow: &ArrowExpr) -> Option<ComponentExtraction> {
        let jsx = self.extract_jsx_from_arrow_body(&arrow.body);
        if jsx.root.is_none() {
            return None;
        }

        self.has_jsx = true;

        let hooks = self.extract_hooks_from_arrow_body(&arrow.body);
        let handlers = self.extract_handlers_from_arrow_body(&arrow.body);
        let props = self.extract_props_from_arrow_params(&arrow.params);

        Some(ComponentExtraction {
            name: "Anonymous".to_string(),
            component_type: ComponentType::Functional,
            exported: false,
            export_type: None,
            location: self.span_to_location(arrow.span),
            props,
            props_type: None,
            hooks,
            class_info: None,
            jsx: jsx.clone(),
            handlers,
            child_components: self.extract_child_components_from_jsx(&jsx),
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

        Some(ComponentExtraction {
            name: name.to_string(),
            component_type: ComponentType::Class,
            exported,
            export_type,
            location: self.span_to_location(class.span),
            props: Vec::new(), // TODO: extract from constructor
            props_type: None,
            hooks: Vec::new(), // Class components don't use hooks
            class_info: Some(ClassComponentInfo {
                state_type: None,
                state_initializer: None,
                lifecycle_methods,
            }),
            jsx: jsx.clone(),
            handlers: Vec::new(),
            child_components: self.extract_child_components_from_jsx(&jsx),
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
            Expr::Object(_) => "{}".to_string(),
            _ => "/* expr */".to_string(),
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
            _ => "/* type */".to_string(),
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

                Some(HandlerExtraction {
                    name: name.to_string(),
                    event_type: None,
                    is_async,
                    body_summary,
                    state_mutations,
                    api_calls,
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

        HandlerExtraction {
            name: name.to_string(),
            event_type: None,
            is_async,
            body_summary,
            state_mutations,
            api_calls,
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
                Some(TypeExtraction {
                    name: interface.id.sym.to_string(),
                    kind: TypeKind::Interface,
                    location: self.span_to_location(interface.span),
                    definition: "/* interface */".to_string(), // TODO: serialize
                    exported,
                })
            }
            Decl::TsTypeAlias(alias) => {
                Some(TypeExtraction {
                    name: alias.id.sym.to_string(),
                    kind: TypeKind::TypeAlias,
                    location: self.span_to_location(alias.span),
                    definition: "/* type alias */".to_string(), // TODO: serialize
                    exported,
                })
            }
            Decl::TsEnum(enum_decl) => {
                Some(TypeExtraction {
                    name: enum_decl.id.sym.to_string(),
                    kind: TypeKind::Enum,
                    location: self.span_to_location(enum_decl.span),
                    definition: "/* enum */".to_string(), // TODO: serialize
                    exported,
                })
            }
            _ => None,
        }
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
}
