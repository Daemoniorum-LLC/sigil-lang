//! Qliphoth code generation from MigrationSpec.
//!
//! Generates idiomatic Sigil/Qliphoth code from migration specifications:
//! - Actor structures with state fields
//! - Message enums and handlers
//! - VNode builder chains from JSX
//! - Qliphoth framework imports
//!
//! See docs/specs/REACT-MIGRATION.md Section 7 for Qliphoth mapping.

use super::ast_transform::{self, TransformConfig};
use super::extraction::*;
use super::spec::*;

// =============================================================================
// Generated Code Output
// =============================================================================

/// Complete generated Sigil file.
#[derive(Debug, Clone)]
pub struct GeneratedSigil {
    pub path: String,
    pub code: String,
    pub component_name: String,
}

/// Code generator for Qliphoth components.
pub struct QliphothGenerator<'a> {
    spec: &'a ComponentMigrationSpec,
    indent: usize,
    /// Whether we're generating for an actor (true) or pure function (false)
    is_actor: bool,
    /// Parameter names for pure functions (used for expression interpolation)
    param_names: Vec<String>,
}

/// Scope for tracking local variables (like map iterators) that shouldn't get self. prefix
#[derive(Default, Clone)]
struct VNodeScope {
    /// Local variable names (iterator vars, etc.)
    locals: Vec<String>,
}

impl<'a> QliphothGenerator<'a> {
    pub fn new(spec: &'a ComponentMigrationSpec) -> Self {
        let is_actor = spec.target.pattern == TargetPattern::Actor;
        let param_names: Vec<String> = spec.recommendations.props_handling.fields
            .iter()
            .map(|f| f.name.clone())
            .collect();

        Self {
            spec,
            indent: 0,
            is_actor,
            param_names,
        }
    }

    /// Generate complete Sigil file for the component.
    pub fn generate(&self) -> GeneratedSigil {
        let mut code = String::new();

        // Generate imports
        code.push_str(&self.generate_imports());
        code.push('\n');

        // Generate message enum if needed
        if !self.spec.recommendations.messages.is_empty() {
            code.push_str(&self.generate_message_enum());
            code.push('\n');
        }

        // Generate actor or function based on target pattern
        match self.spec.target.pattern {
            TargetPattern::Actor => {
                code.push_str(&self.generate_actor());
            }
            TargetPattern::Function => {
                code.push_str(&self.generate_function());
            }
        }

        GeneratedSigil {
            path: self.spec.target.suggested_path.clone(),
            code,
            component_name: self.spec.name.clone(),
        }
    }

    // =========================================================================
    // Import Generation
    // =========================================================================

    fn generate_imports(&self) -> String {
        let mut imports = Vec::new();

        // Always include qliphoth prelude
        imports.push("invoke qliphoth·prelude·*;".to_string());

        // Check for DOM/browser API usage (simplified detection)
        let needs_sys = self.spec.recommendations.effects.iter()
            .any(|e| e.reasoning.contains("document") || e.reasoning.contains("window"));

        if needs_sys {
            imports.push("invoke qliphoth_sys·*;".to_string());
        }

        // Note: Router detection can be added when router hooks are extracted

        imports.join("\n")
    }

    // =========================================================================
    // Message Enum Generation
    // =========================================================================

    fn generate_message_enum(&self) -> String {
        let name = format!("{}Msg", self.spec.name);
        let mut variants = Vec::new();

        for msg in &self.spec.recommendations.messages {
            if let Some(payload) = &msg.payload {
                variants.push(format!("    {} {},", msg.name, payload));
            } else {
                variants.push(format!("    {},", msg.name));
            }
        }

        format!(
            "ᛈ {} {{\n{}\n}}",
            name,
            variants.join("\n")
        )
    }

    // =========================================================================
    // Actor Generation
    // =========================================================================

    fn generate_actor(&self) -> String {
        let mut sections = Vec::new();

        // State fields
        let state_fields = self.generate_state_fields();
        if !state_fields.is_empty() {
            sections.push(state_fields);
        }

        // Constructor if props
        if !self.spec.recommendations.props_handling.fields.is_empty() {
            sections.push(self.generate_constructor());
        }

        // Message handlers
        let handlers = self.generate_message_handlers();
        if !handlers.is_empty() {
            sections.push(handlers);
        }

        // Lifecycle handlers (Mount/Unmount)
        let lifecycle = self.generate_lifecycle_handlers();
        if !lifecycle.is_empty() {
            sections.push(lifecycle);
        }

        // View method
        sections.push(self.generate_view_method());

        format!(
            "actor {} {{\n{}\n}}",
            self.spec.name,
            sections.join("\n\n")
        )
    }

    fn generate_state_fields(&self) -> String {
        let fields: Vec<String> = self.spec.recommendations.state_fields.iter()
            .map(|field| {
                // Convert JS null/undefined/placeholders to Sigil None
                let initial = match field.initial_value.as_str() {
                    "null" | "undefined" | "/* expr */" | "/* expression */" => "None".to_string(),
                    other if other.contains("/*") => "None".to_string(),
                    other => other.to_string(),
                };
                format!(
                    "    state {}: {}{} = {},",
                    field.to_field,
                    field.field_type,
                    field.evidentiality,
                    initial
                )
            })
            .collect();

        fields.join("\n")
    }

    fn generate_constructor(&self) -> String {
        let props = &self.spec.recommendations.props_handling;

        let params: Vec<String> = props.fields.iter()
            .map(|f| format!("{}: {}", f.name, f.field_type))
            .collect();

        let assignments: Vec<String> = props.fields.iter()
            .map(|f| format!("        self.{} = {};", f.name, f.name))
            .collect();

        format!(
            "    rite new({}) -> This! {{\n{}\n        This\n    }}",
            params.join(", "),
            assignments.join("\n")
        )
    }

    fn generate_message_handlers(&self) -> String {
        let handlers: Vec<String> = self.spec.recommendations.messages.iter()
            .map(|msg| {
                let mut body_parts: Vec<String> = Vec::new();

                // Add service calls (hook-returned function calls -> actor messages)
                for call in &msg.service_calls {
                    let args = if call.args.is_empty() {
                        String::new()
                    } else {
                        format!("({})", call.args.join(", "))
                    };
                    body_parts.push(format!("        {} ! {}{};", call.service, call.method, args));
                }

                // Add state changes
                for change in &msg.state_changes {
                    // Transform placeholder to valid expression
                    let change = change.replace("/* new value */", "msg.0");
                    body_parts.push(format!("        {};", change));
                }

                // Include inlined effects if any
                for effect in &self.spec.recommendations.effects {
                    if effect.strategy == EffectStrategy::Inline {
                        if effect.inline_in.as_ref().map(|i| i.contains(&msg.name.to_lowercase())).unwrap_or(false) {
                            body_parts.push(format!("        // Effect: {}", effect.reasoning));
                        }
                    }
                }

                // If no body parts, add TODO comment
                let body = if body_parts.is_empty() {
                    "        // TODO: implement".to_string()
                } else {
                    body_parts.join("\n")
                };

                format!("    on {} {{\n{}\n    }}", msg.name, body)
            })
            .collect();

        handlers.join("\n\n")
    }

    fn generate_lifecycle_handlers(&self) -> String {
        let mut handlers = Vec::new();

        for effect in &self.spec.recommendations.effects {
            if effect.strategy == EffectStrategy::Lifecycle {
                if let Some(event) = &effect.lifecycle_event {
                    handlers.push(format!(
                        "    on {} {{\n        // {}\n    }}",
                        event,
                        effect.reasoning
                    ));
                }
            }
        }

        handlers.join("\n\n")
    }

    fn generate_view_method(&self) -> String {
        let jsx = &self.spec.source.extraction.jsx;
        let scope = VNodeScope::default();

        let body = if let Some(root) = &jsx.root {
            self.generate_vnode(root, 2, &scope)
        } else {
            "        VNode·div()".to_string()
        };

        format!(
            "    rite view(self) -> VNode! {{\n{}\n    }}",
            body
        )
    }

    // =========================================================================
    // Function Generation (for pure components)
    // =========================================================================

    fn generate_function(&self) -> String {
        let props = &self.spec.recommendations.props_handling;

        let params = if props.fields.is_empty() {
            String::new()
        } else {
            props.fields.iter()
                .map(|f| {
                    // Handle rest parameters: ...props -> props: Vec<Any>
                    if f.name.starts_with("...") {
                        let name = &f.name[3..]; // Remove "..." prefix
                        format!("{}: Vec<Any>", name)
                    } else {
                        format!("{}: {}", f.name, f.field_type)
                    }
                })
                .collect::<Vec<_>>()
                .join(", ")
        };

        let jsx = &self.spec.source.extraction.jsx;
        let scope = VNodeScope::default();
        let body = if let Some(root) = &jsx.root {
            self.generate_vnode(root, 1, &scope)
        } else {
            "    VNode·div()".to_string()
        };

        format!(
            "rite {}({}) -> VNode! {{\n{}\n}}",
            to_snake_case(&self.spec.name),
            params,
            body
        )
    }

    // =========================================================================
    // VNode Builder Generation
    // =========================================================================

    fn generate_vnode(&self, node: &JsxNode, indent: usize, scope: &VNodeScope) -> String {
        let pad = "    ".repeat(indent);

        match &node.node_type {
            JsxNodeType::Element { tag, is_component, attributes, children } => {
                self.generate_element_vnode(tag, *is_component, attributes, children, indent, scope)
            }
            JsxNodeType::Fragment { children } => {
                self.generate_fragment_vnode(children, indent, scope)
            }
            JsxNodeType::Text { value } => {
                format!("{}·text_child(\"{}\")", pad, escape_string(value))
            }
            JsxNodeType::Expression { code } => {
                // Expression interpolation - convert to text_child with to_string
                let expr = self.transform_expression_scoped(code, scope);
                format!("{}·text_child({}·to_string())", pad, expr)
            }
            JsxNodeType::Conditional { condition, consequent, alternate } => {
                let cond_expr = self.transform_expression_scoped(condition, scope);
                let cons = self.generate_vnode(consequent, indent, scope);
                if let Some(alt) = alternate {
                    let alt_code = self.generate_vnode(alt, indent, scope);
                    format!("{}·when_else({}, {}, {})", pad, cond_expr, cons.trim(), alt_code.trim())
                } else {
                    format!("{}·when({}, {})", pad, cond_expr, cons.trim())
                }
            }
            JsxNodeType::Map { iterable, item_name, key_expr: _, body } => {
                let iter_expr = self.transform_expression_scoped(iterable, scope);
                // Create new scope with iterator variable
                let mut inner_scope = scope.clone();
                inner_scope.locals.push(item_name.clone());
                let body_code = self.generate_vnode(body, indent + 1, &inner_scope);
                format!(
                    "{pad}// Map: ∀ {item} ∈ {iter}\n{pad}·children({iter_expr}.iter().map(|{item}| {body}).collect())",
                    pad = pad,
                    item = item_name,
                    iter = iterable,
                    iter_expr = iter_expr,
                    body = body_code.trim()
                )
            }
        }
    }

    fn generate_element_vnode(
        &self,
        tag: &str,
        is_component: bool,
        attributes: &[JsxAttribute],
        children: &[JsxNode],
        indent: usize,
        scope: &VNodeScope,
    ) -> String {
        let pad = "    ".repeat(indent);
        let mut builder = if is_component {
            // Component reference
            format!("{}{}·view()", pad, tag)
        } else {
            // HTML element
            format!("{}VNode·{}()", pad, tag)
        };

        // Add attributes
        for attr in attributes {
            let attr_code = self.generate_attribute_scoped(attr, scope);
            if !attr_code.is_empty() {
                builder.push_str(&format!("\n{}    {}", pad, attr_code));
            }
        }

        // Add children
        for child in children {
            let child_code = self.generate_vnode(child, indent + 1, scope);
            match &child.node_type {
                JsxNodeType::Text { value } => {
                    // Text: append text_child directly
                    builder.push_str(&format!("\n{}    ·text_child(\"{}\")", pad, escape_string(value)));
                }
                JsxNodeType::Expression { .. } => {
                    // Expression: append the generated text_child directly (no wrapping)
                    builder.push_str(&format!("\n{}    {}", pad, child_code.trim()));
                }
                JsxNodeType::Map { .. } => {
                    // Map: append the generated ·children() directly (no wrapping)
                    builder.push_str(&format!("\n{}    {}", pad, child_code.trim_start()));
                }
                JsxNodeType::Conditional { .. } => {
                    // Conditional: append ·when() or ·when_else() directly
                    builder.push_str(&format!("\n{}    {}", pad, child_code.trim()));
                }
                _ => {
                    // Element, Fragment: wrap in ·child()
                    builder.push_str(&format!("\n{}    ·child(\n{}\n{}    )", pad, child_code, pad));
                }
            }
        }

        builder
    }

    fn generate_fragment_vnode(&self, children: &[JsxNode], indent: usize, scope: &VNodeScope) -> String {
        let pad = "    ".repeat(indent);
        let mut builder = format!("{}VNode·fragment()", pad);

        for child in children {
            let child_code = self.generate_vnode(child, indent + 1, scope);
            match &child.node_type {
                JsxNodeType::Text { value } => {
                    builder.push_str(&format!("\n{}    ·text_child(\"{}\")", pad, escape_string(value)));
                }
                JsxNodeType::Expression { .. } | JsxNodeType::Map { .. } | JsxNodeType::Conditional { .. } => {
                    builder.push_str(&format!("\n{}    {}", pad, child_code.trim()));
                }
                _ => {
                    builder.push_str(&format!("\n{}    ·child(\n{}\n{}    )", pad, child_code, pad));
                }
            }
        }

        builder
    }
}

// =============================================================================
// Public API
// =============================================================================

/// Generate Sigil code from a component migration spec.
pub fn generate_component(spec: &ComponentMigrationSpec) -> GeneratedSigil {
    let generator = QliphothGenerator::new(spec);
    generator.generate()
}

/// Generate Sigil code for all components in a migration spec.
pub fn generate_all(spec: &MigrationSpec) -> Vec<GeneratedSigil> {
    spec.components.iter()
        .map(|comp| generate_component(comp))
        .collect()
}

/// Generate Sigil code for a service actor.
pub fn generate_service_actor(actor: &ServiceActorSpec) -> GeneratedSigil {
    let generator = ServiceActorGenerator::new(actor);
    generator.generate()
}

/// Generate Sigil code for all service actors in a migration spec.
pub fn generate_all_service_actors(spec: &MigrationSpec) -> Vec<GeneratedSigil> {
    spec.service_actors.iter()
        .map(|actor| generate_service_actor(actor))
        .collect()
}

// =============================================================================
// Service Actor Generator (Phase 7)
// =============================================================================

/// Generator for service actor Sigil code.
struct ServiceActorGenerator<'a> {
    actor: &'a ServiceActorSpec,
}

impl<'a> ServiceActorGenerator<'a> {
    fn new(actor: &'a ServiceActorSpec) -> Self {
        Self { actor }
    }

    fn generate(&self) -> GeneratedSigil {
        let mut code = String::new();

        // Imports
        code.push_str("invoke qliphoth·prelude·*;\n\n");

        // Message enum
        code.push_str(&self.generate_message_enum());
        code.push_str("\n\n");

        // Actor definition
        code.push_str(&self.generate_actor());

        GeneratedSigil {
            path: format!("src/services/{}.sigil", to_snake_case(&self.actor.name)),
            code,
            component_name: self.actor.name.clone(),
        }
    }

    fn generate_message_enum(&self) -> String {
        if self.actor.messages.is_empty() {
            return String::new();
        }

        let name = format!("{}Msg", self.actor.name);
        let variants: Vec<String> = self.actor.messages.iter()
            .map(|msg| {
                if msg.parameters.is_empty() {
                    format!("    {},", msg.name)
                } else {
                    // Generate payload type from parameters
                    let payload = format!("{{ {} }}", msg.parameters.join(", "));
                    format!("    {} {},", msg.name, payload)
                }
            })
            .collect();

        format!("ᛈ {} {{\n{}\n}}", name, variants.join("\n"))
    }

    fn generate_actor(&self) -> String {
        let mut sections = Vec::new();

        // Comment showing derivation
        sections.push(format!("    // Derived from hook: {}", self.actor.derived_from));

        // State fields
        if !self.actor.state_fields.is_empty() {
            let fields: Vec<String> = self.actor.state_fields.iter()
                .map(|f| {
                    let evidentiality = if f.is_observable { "!" } else { "~" };
                    format!("    state {}: {}{} = /* initial */,", f.name, f.field_type, evidentiality)
                })
                .collect();
            sections.push(fields.join("\n"));
        }

        // Message handlers
        if !self.actor.messages.is_empty() {
            let handlers: Vec<String> = self.actor.messages.iter()
                .map(|msg| {
                    format!(
                        "    on {} {{\n        // TODO: implement {}\n    }}",
                        msg.name,
                        msg.original_name
                    )
                })
                .collect();
            sections.push(handlers.join("\n\n"));
        }

        // Query methods for state (observable fields become queryable)
        let queries: Vec<String> = self.actor.state_fields.iter()
            .filter(|f| f.is_observable)
            .map(|f| {
                format!(
                    "    rite {}(self) -> {}! {{\n        self.{}\n    }}",
                    f.name,
                    f.field_type,
                    f.name
                )
            })
            .collect();
        if !queries.is_empty() {
            sections.push(queries.join("\n\n"));
        }

        format!(
            "actor {} {{\n{}\n}}",
            self.actor.name,
            sections.join("\n\n")
        )
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

impl<'a> QliphothGenerator<'a> {
    /// Transform a React expression into Sigil syntax.
    /// For actors: prefix state with `self.`
    /// For pure functions: use parameters directly
    fn transform_expression(&self, code: &str) -> String {
        self.transform_expression_scoped(code, &VNodeScope::default())
    }

    /// Transform with scope awareness (don't prefix iterator variables with self.)
    fn transform_expression_scoped(&self, code: &str, scope: &VNodeScope) -> String {
        // Handle placeholder/invalid expressions
        if code.contains("/*") || code.is_empty() {
            return "None".to_string();
        }

        // Clean up the expression
        let code = code.trim();

        // Handle JSX in expressions - these need special handling
        // For now, replace with None as a placeholder
        if code.contains('<') && code.contains('>') {
            return "None".to_string();
        }

        // Build transformation config
        let state_fields: Vec<String> = self.spec.recommendations.state_fields
            .iter()
            .map(|f| f.to_field.clone())
            .collect();

        let config = TransformConfig {
            prefix_self: self.is_actor,
            state_fields,
            locals: scope.locals.clone(),
            props: self.param_names.clone(),
        };

        // Use AST-based transformation
        let result = ast_transform::transform_expression(code, &config);

        // Note: warnings are available in result.warnings if needed for debugging
        result.code
    }

    /// For actors, prefix state field references with self.
    fn prefix_state_variables(&self, code: &str) -> String {
        self.prefix_state_variables_scoped(code, &VNodeScope::default())
    }

    /// Prefix state variables, but not scoped locals
    fn prefix_state_variables_scoped(&self, code: &str, scope: &VNodeScope) -> String {
        let state_fields: Vec<String> = self.spec.recommendations.state_fields
            .iter()
            .map(|f| f.to_field.clone())
            .collect();

        let mut result = code.to_string();

        // Simple approach: prefix known state fields with self.
        // This is a basic implementation - a real one would parse the expression
        for field in &state_fields {
            // Skip if this is a scoped local variable
            if scope.locals.contains(field) {
                continue;
            }

            // Replace field name when it appears as a word boundary
            let patterns = [
                (format!("{}", field), format!("self.{}", field)),
            ];
            for (from, to) in &patterns {
                // Only replace if it's a standalone identifier (not part of a larger word)
                if result == *from {
                    result = to.clone();
                } else if result.starts_with(&format!("{}.", from)) {
                    result = format!("self.{}", result);
                    break;
                }
            }
        }

        // If no state field was found and it's a simple identifier that's not a local, prefix with self
        if !result.starts_with("self.") && is_simple_identifier(&result) && !scope.locals.iter().any(|l| result == *l) {
            result = format!("self.{}", result);
        }

        result
    }

    /// Generate attribute with scope awareness
    fn generate_attribute_scoped(&self, attr: &JsxAttribute, scope: &VNodeScope) -> String {
        // Handle special attributes
        match attr.name.as_str() {
            "className" | "class" => {
                match &attr.value {
                    JsxAttributeValue::String { value } => format!("·class(\"{}\")", value),
                    JsxAttributeValue::Expression { code } => {
                        let transformed = self.transform_expression_scoped(code, scope);
                        // Ensure class strings are properly quoted
                        if transformed.starts_with('"') && !transformed.ends_with('"') {
                            format!("·class({}\")", transformed)
                        } else if !transformed.starts_with('"') && !transformed.contains('"') {
                            // Simple identifier, wrap in quotes
                            format!("·class(\"{}\")", transformed)
                        } else {
                            format!("·class({})", transformed)
                        }
                    }
                    _ => String::new(),
                }
            }
            "id" => {
                match &attr.value {
                    JsxAttributeValue::String { value } => format!("·id(\"{}\")", value),
                    JsxAttributeValue::Expression { code } => {
                        let transformed = self.transform_expression_scoped(code, scope);
                        format!("·id({})", transformed)
                    }
                    _ => String::new(),
                }
            }
            "style" => {
                // Style needs special handling - simplified for now
                format!("·style(∅)")
            }
            name if attr.is_event_handler => {
                // Event handler -> message dispatch
                let event_name = name.strip_prefix("on").unwrap_or(name);
                let msg_name = to_pascal_case(event_name);
                format!("·on_{}({})", event_name.to_lowercase(), msg_name)
            }
            "disabled" | "checked" | "selected" | "readonly" => {
                // Boolean attributes
                match &attr.value {
                    JsxAttributeValue::True => format!("·attr(\"{}\", \"true\")", attr.name),
                    JsxAttributeValue::Expression { code } => {
                        let transformed = self.transform_expression_scoped(code, scope);
                        format!("·when({}, |n| n·attr(\"{}\", \"true\"))", transformed, attr.name)
                    }
                    _ => String::new(),
                }
            }
            "href" | "src" | "alt" | "type" | "name" | "value" | "placeholder" => {
                // Common attributes
                match &attr.value {
                    JsxAttributeValue::String { value } => {
                        format!("·attr(\"{}\", \"{}\")", attr.name, escape_string(value))
                    }
                    JsxAttributeValue::Expression { code } => {
                        let transformed = self.transform_expression_scoped(code, scope);
                        format!("·attr(\"{}\", {})", attr.name, transformed)
                    }
                    _ => String::new(),
                }
            }
            _ => {
                // Generic attribute
                match &attr.value {
                    JsxAttributeValue::String { value } => {
                        format!("·attr(\"{}\", \"{}\")", attr.name, escape_string(value))
                    }
                    JsxAttributeValue::Expression { code } => {
                        let transformed = self.transform_expression_scoped(code, scope);
                        format!("·attr(\"{}\", {})", attr.name, transformed)
                    }
                    JsxAttributeValue::Spread { name } => {
                        // Spread attributes can't be directly represented, skip
                        String::new()
                    }
                    JsxAttributeValue::True => {
                        format!("·attr(\"{}\", \"true\")", attr.name)
                    }
                }
            }
        }
    }
}

/// Check if a string is a simple identifier (no operators, dots, etc.)
fn is_simple_identifier(s: &str) -> bool {
    !s.is_empty() &&
    s.chars().next().map_or(false, |c| c.is_alphabetic() || c == '_') &&
    s.chars().all(|c| c.is_alphanumeric() || c == '_')
}

fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                result.push('_');
            }
            result.push(c.to_lowercase().next().unwrap());
        } else {
            result.push(c);
        }
    }
    result
}

fn to_pascal_case(s: &str) -> String {
    let mut result = String::new();
    let mut capitalize_next = true;
    for c in s.chars() {
        if c == '_' || c == '-' {
            capitalize_next = true;
        } else if capitalize_next {
            result.push(c.to_uppercase().next().unwrap());
            capitalize_next = false;
        } else {
            result.push(c);
        }
    }
    result
}

fn escape_string(s: &str) -> String {
    s.replace('\\', "\\\\")
     .replace('"', "\\\"")
     .replace('\n', "\\n")
     .replace('\t', "\\t")
}

// =============================================================================
// JavaScript to Sigil Expression Transformation
// =============================================================================

/// Transform JavaScript expression syntax to Sigil syntax.
/// Converts operators, method calls, and identifiers.
fn transform_js_to_sigil(code: &str) -> String {
    // FIRST: Handle template literals to prevent ${expr} from being misinterpreted as object literals
    // JS template literals: `foo` → "foo", ${expr} → stripped
    let mut result = transform_template_literals(code);

    // Handle JS object literals { key: value } or { foo, bar } → ∅
    // These can't be directly represented, so simplify
    result = transform_object_literals(&result);

    // Handle TypeScript type casts: (expr as Type) → expr
    result = transform_type_casts(&result);

    // Handle array indexing: arr[idx] → arr.get(idx)
    result = transform_array_indexing(&result);

    // First transform arrow functions to Sigil closures
    result = transform_arrow_functions(&result);

    // Transform ternary operator: a ? b : c → (if a { b } else { c })
    result = transform_ternary(&result);

    // Transform JS operators to Sigil operators
    // Order matters: longer patterns first to avoid partial matches

    // Logical operators
    result = result.replace(" && ", " ∧ ");
    result = result.replace(" || ", " ∨ ");
    result = result.replace("&&", " ∧ ");
    result = result.replace("||", " ∨ ");

    // Comparison operators (handle !== and === before != and ==)
    // Sigil uses == for equality (not =, which is assignment)
    result = result.replace("!==", "≠");
    result = result.replace("===", "==");
    result = result.replace("!=", "≠");
    // == stays as == in Sigil

    // Unary not: !foo → ¬foo (be careful not to replace != which is already handled)
    // Use a simple approach: replace ! at word boundaries
    result = transform_unary_not(&result);

    // JS method calls to Sigil
    result = result.replace(".length", ".len()");
    result = result.replace(".toString()", ".to_string()");
    result = result.replace(".trim()", ".trim()");
    result = result.replace(".toLowerCase()", ".to_lowercase()");
    result = result.replace(".toUpperCase()", ".to_uppercase()");
    result = result.replace(".includes(", ".contains(");
    result = result.replace(".indexOf(", ".find(");
    result = result.replace(".startsWith(", ".starts_with(");
    result = result.replace(".endsWith(", ".ends_with(");
    result = result.replace(".push(", ".append(");
    result = result.replace(".pop()", ".pop()");
    result = result.replace(".shift()", ".remove(0)");
    result = result.replace(".slice(", ".slice(");
    result = result.replace(".join(", ".join(");
    result = result.replace(".split(", ".split(");
    result = result.replace(".map(", ".map(");
    result = result.replace(".filter(", ".filter(");
    result = result.replace(".find(", ".find(");
    result = result.replace(".some(", ".any(");
    result = result.replace(".every(", ".all(");
    result = result.replace(".reduce(", ".fold(");

    // JS string quotes: 'foo' → "foo"
    result = transform_single_quotes(&result);

    // Note: Template literals are now handled at the start of this function
    // to prevent ${expr} from being misinterpreted as object literals

    // JS boolean literals: true → True, false → False
    // Must be done BEFORE snake_case transform to avoid true → True being converted to true
    result = transform_boolean_literals(&result);

    // Transform camelCase identifiers to snake_case in the expression
    result = transform_identifiers_to_snake_case(&result);

    result
}

/// Transform JavaScript arrow functions to Sigil closures.
/// Examples:
///   p => expr       →  |p| expr
///   (p) => expr     →  |p| expr
///   (a, b) => expr  →  |a, b| expr
fn transform_arrow_functions(code: &str) -> String {
    let mut result = code.to_string();

    // Pattern: identifier => expr (simple arrow function without parens)
    // e.g., "p => { id: p }" → "|p| { id: p }"
    let simple_arrow_re = regex::Regex::new(r"(\b)([a-zA-Z_][a-zA-Z0-9_]*)\s*=>\s*").unwrap();
    result = simple_arrow_re.replace_all(&result, "$1|$2| ").to_string();

    // Pattern: (params) => expr (arrow function with parens)
    // e.g., "(a, b) => a + b" → "|a, b| a + b"
    let paren_arrow_re = regex::Regex::new(r"\(([^)]*)\)\s*=>\s*").unwrap();
    result = paren_arrow_re.replace_all(&result, "|$1| ").to_string();

    result
}

/// Transform JavaScript ternary operators to Sigil if-else expressions.
/// For simple cases like standalone expressions: a ? b : c → if a { b } else { c }
/// For complex cases inside method calls: we simplify to just the first value (true branch)
fn transform_ternary(code: &str) -> String {
    // For ternaries inside method calls/arguments, use simpler transform
    // This avoids parser issues with (if...) inside method arguments
    // Just use the consequent (true branch) as a simplified fallback
    //
    // Pattern matches: condition ? true_branch : false_branch
    // The condition can be a complex expression (e.g., a == b, a && b, etc.)
    // We capture everything up to the first ? as condition, then true branch, then false branch
    //
    // Use non-greedy matching for the condition part, and greedy for branches
    let re = regex::Regex::new(r"(.+?)\s*\?\s*(.+?)\s*:\s*(.+)$").unwrap();
    re.replace_all(code, "$2").to_string()
}

/// Transform JavaScript template literals to Sigil strings.
/// Converts backticks to double quotes. For complex interpolations,
/// we simplify by just taking the static parts.
fn transform_template_literals(code: &str) -> String {
    let mut result = String::new();
    let chars: Vec<char> = code.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '`' {
            // Convert backtick to double quote
            result.push('"');
            i += 1;
        } else if chars[i] == '$' && i + 1 < chars.len() && chars[i + 1] == '{' {
            // Skip ${...} interpolation entirely (just remove it)
            i += 2; // Skip ${
            let mut brace_depth = 1;
            while i < chars.len() && brace_depth > 0 {
                if chars[i] == '{' {
                    brace_depth += 1;
                } else if chars[i] == '}' {
                    brace_depth -= 1;
                }
                i += 1;
            }
        } else {
            result.push(chars[i]);
            i += 1;
        }
    }

    result
}

/// Transform unary not operator: !foo → ¬foo
fn transform_unary_not(code: &str) -> String {
    let mut result = String::new();
    let chars: Vec<char> = code.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '!' {
            // Check if this is part of != or !== (already transformed to ≠)
            if i + 1 < chars.len() && chars[i + 1] == '=' {
                // This shouldn't happen since we already replaced != with ≠
                result.push(chars[i]);
            } else {
                // This is unary not
                result.push('¬');
            }
        } else {
            result.push(chars[i]);
        }
        i += 1;
    }

    result
}

/// Transform JavaScript boolean literals to Sigil format.
/// true → True, false → False
fn transform_boolean_literals(code: &str) -> String {
    // Use word boundary matching to avoid changing "trueValue" to "TrueValue"
    let true_re = regex::Regex::new(r"\btrue\b").unwrap();
    let result = true_re.replace_all(code, "True").to_string();

    let false_re = regex::Regex::new(r"\bfalse\b").unwrap();
    false_re.replace_all(&result, "False").to_string()
}

/// Transform JavaScript single quotes to double quotes.
fn transform_single_quotes(code: &str) -> String {
    let mut result = String::new();
    let mut in_single_quote = false;
    let mut in_double_quote = false;

    for c in code.chars() {
        match c {
            '\'' if !in_double_quote => {
                in_single_quote = !in_single_quote;
                result.push('"');
            }
            '"' if !in_single_quote => {
                in_double_quote = !in_double_quote;
                result.push('"');
            }
            _ => result.push(c),
        }
    }

    result
}

/// Transform camelCase identifiers to snake_case.
/// Only transforms identifiers, not strings or operators.
fn transform_identifiers_to_snake_case(code: &str) -> String {
    let mut result = String::new();
    let mut current_ident = String::new();
    let mut in_string = false;
    let mut string_char = '"';

    for c in code.chars() {
        // Track string state
        if (c == '"' || c == '\'') && !in_string {
            in_string = true;
            string_char = c;
            // Flush current identifier
            if !current_ident.is_empty() {
                result.push_str(&camel_to_snake(&current_ident));
                current_ident.clear();
            }
            result.push(c);
            continue;
        }
        if c == string_char && in_string {
            in_string = false;
            result.push(c);
            continue;
        }
        if in_string {
            result.push(c);
            continue;
        }

        // Build identifiers
        if c.is_alphanumeric() || c == '_' {
            current_ident.push(c);
        } else {
            // Flush current identifier
            if !current_ident.is_empty() {
                result.push_str(&camel_to_snake(&current_ident));
                current_ident.clear();
            }
            result.push(c);
        }
    }

    // Flush remaining identifier
    if !current_ident.is_empty() {
        result.push_str(&camel_to_snake(&current_ident));
    }

    result
}

/// Transform JavaScript object literals to Sigil.
/// { key: value, ... } or { foo, bar } → ∅ (simplified for now)
fn transform_object_literals(code: &str) -> String {
    let mut result = String::new();
    let chars: Vec<char> = code.chars().collect();
    let mut i = 0;
    let mut depth = 0;
    let mut brace_start = None;
    let mut in_string = false;
    let mut string_char = '"';

    while i < chars.len() {
        let c = chars[i];

        // Track string state
        if (c == '"' || c == '\'') && !in_string {
            in_string = true;
            string_char = c;
            result.push(c);
            i += 1;
            continue;
        }
        if c == string_char && in_string {
            in_string = false;
            result.push(c);
            i += 1;
            continue;
        }
        if in_string {
            result.push(c);
            i += 1;
            continue;
        }

        if c == '{' {
            if depth == 0 {
                brace_start = Some(result.len());
            }
            depth += 1;
            result.push(c);
        } else if c == '}' {
            depth -= 1;
            if depth == 0 {
                // Check if this looks like an object literal (not a block)
                if let Some(start) = brace_start {
                    let content = &result[start + 1..];
                    // Object literal indicators: contains ':', or is shorthand { foo, bar }
                    if content.contains(':') || (content.contains(',') && !content.contains(';')) {
                        // Replace the whole object literal with ∅
                        result.truncate(start);
                        result.push_str("∅");
                        brace_start = None;
                        i += 1;
                        continue;
                    }
                }
                brace_start = None;
            }
            result.push(c);
        } else {
            result.push(c);
        }
        i += 1;
    }

    result
}

/// Transform TypeScript type casts: (expr as Type) → expr, expr as Type → expr
fn transform_type_casts(code: &str) -> String {
    // Pattern: (expr as Type) → expr
    let paren_cast_re = regex::Regex::new(r"\(([^)]+)\s+as\s+[A-Za-z_][A-Za-z0-9_\[\]<>]*\)").unwrap();
    let result = paren_cast_re.replace_all(code, "$1").to_string();

    // Pattern: expr as Type → expr (without parens)
    let cast_re = regex::Regex::new(r"(\b[A-Za-z_][A-Za-z0-9_.()]*)\s+as\s+[A-Za-z_][A-Za-z0-9_\[\]<>]*").unwrap();
    cast_re.replace_all(&result, "$1").to_string()
}

/// Transform JavaScript array indexing: arr[idx] → arr.get(idx)
fn transform_array_indexing(code: &str) -> String {
    // Pattern: identifier[expr] → identifier.get(expr)
    // Be careful not to transform things like:
    // - Type[] or generic syntax
    // - CSS custom properties: text-[var(--accent-primary)]
    // - Tailwind CSS classes: bg-[#fff], w-[100px]
    //
    // Only transform if the bracket content looks like a simple identifier
    // (not CSS values which typically have dashes, hashes, or parens)
    let bracket_re = regex::Regex::new(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\[([a-zA-Z_][a-zA-Z0-9_]*)\]").unwrap();

    // First check if we're inside CSS (contains patterns like text-[, bg-[, etc.)
    // If so, don't transform anything
    if code.contains("-[") {
        return code.to_string();
    }

    bracket_re.replace_all(code, "$1.get($2)").to_string()
}

/// Convert a single camelCase identifier to snake_case.
/// Preserves all-caps abbreviations and numbers.
fn camel_to_snake(s: &str) -> String {
    // Don't transform Sigil boolean literals
    if s == "True" || s == "False" {
        return s.to_string();
    }

    // Don't transform keywords, operators, or already snake_case
    if s.contains('_') || s.chars().all(|c| c.is_lowercase() || c.is_numeric()) {
        return s.to_string();
    }

    // Don't transform if it's all uppercase (likely a constant)
    if s.chars().all(|c| c.is_uppercase() || c.is_numeric()) {
        return s.to_string();
    }

    // Don't transform short identifiers (likely intentional)
    if s.len() <= 2 {
        return s.to_string();
    }

    let mut result = String::new();
    let chars: Vec<char> = s.chars().collect();

    for (i, c) in chars.iter().enumerate() {
        if c.is_uppercase() && i > 0 {
            // Check if this is part of an acronym (multiple caps in a row)
            let prev_is_upper = chars.get(i.saturating_sub(1)).map(|c| c.is_uppercase()).unwrap_or(false);
            let next_is_lower = chars.get(i + 1).map(|c| c.is_lowercase()).unwrap_or(false);

            // Add underscore if transitioning from lowercase or at acronym boundary
            if !prev_is_upper || next_is_lower {
                result.push('_');
            }
        }
        result.push(c.to_lowercase().next().unwrap());
    }

    result
}
