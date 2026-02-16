//! React extraction types.
//!
//! These types represent the extracted structure from React/TSX files.
//! See docs/specs/REACT-MIGRATION.md Section 3.2 for specification.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

use swc_common::{SourceMap, FilePathMapping, FileName};
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

    let extractor = Extractor::new(&cm, language, path.to_path_buf(), relative_path.to_string());
    extractor.extract_module(&module)
}

// =============================================================================
// Extractor Implementation
// =============================================================================

struct Extractor<'a> {
    cm: &'a SourceMap,
    language: Language,
    path: PathBuf,
    relative_path: String,
    has_jsx: bool,
}

impl<'a> Extractor<'a> {
    fn new(cm: &'a SourceMap, language: Language, path: PathBuf, relative_path: String) -> Self {
        Self {
            cm,
            language,
            path,
            relative_path,
            has_jsx: false,
        }
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
            jsx,
            handlers,
            child_components: Vec::new(), // TODO: extract from JSX
        })
    }

    fn try_extract_arrow_component(&mut self, arrow: &ArrowExpr) -> Option<ComponentExtraction> {
        let jsx = self.extract_jsx_from_arrow_body(&arrow.body);
        if jsx.root.is_none() {
            return None;
        }

        self.has_jsx = true;

        let hooks = self.extract_hooks_from_arrow_body(&arrow.body);
        let handlers = Vec::new(); // TODO: extract from arrow body
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
            jsx,
            handlers,
            child_components: Vec::new(),
        })
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
            jsx,
            handlers: Vec::new(),
            child_components: Vec::new(),
        })
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
                                    code: "/* expression */".to_string() // TODO: serialize expr
                                },
                                JSXExpr::JSXEmptyExpr(_) => JsxAttributeValue::Expression {
                                    code: "".to_string()
                                },
                            }
                        }
                        Some(JSXAttrValue::JSXElement(_)) => {
                            JsxAttributeValue::Expression { code: "/* JSX element */".to_string() }
                        }
                        Some(JSXAttrValue::JSXFragment(_)) => {
                            JsxAttributeValue::Expression { code: "/* JSX fragment */".to_string() }
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
                        value: JsxAttributeValue::Spread { name: "/* spread */".to_string() },
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
                        JSXExpr::Expr(expr) => Some(JsxNode {
                            node_type: JsxNodeType::Expression {
                                code: "/* expression */".to_string()
                            },
                            location: self.span_to_location(container.span),
                        }),
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
                        JSXExpr::Expr(_) => Some(JsxNode {
                            node_type: JsxNodeType::Expression {
                                code: "/* expression */".to_string()
                            },
                            location: self.span_to_location(container.span),
                        }),
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
        // TODO: Implement handler extraction
        Vec::new()
    }

    fn extract_props_from_params(&self, params: &[Param]) -> Vec<PropExtraction> {
        // TODO: Implement props extraction from function params
        Vec::new()
    }

    fn extract_props_from_arrow_params(&self, params: &[Pat]) -> Vec<PropExtraction> {
        // TODO: Implement props extraction from arrow params
        Vec::new()
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
