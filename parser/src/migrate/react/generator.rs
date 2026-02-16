//! Qliphoth code generation from MigrationSpec.
//!
//! Generates idiomatic Sigil/Qliphoth code from migration specifications:
//! - Actor structures with state fields
//! - Message enums and handlers
//! - VNode builder chains from JSX
//! - Qliphoth framework imports
//!
//! See docs/specs/REACT-MIGRATION.md Section 7 for Qliphoth mapping.

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
            "☉ actor {} {{\n{}\n}}",
            self.spec.name,
            sections.join("\n\n")
        )
    }

    fn generate_state_fields(&self) -> String {
        let fields: Vec<String> = self.spec.recommendations.state_fields.iter()
            .map(|field| {
                format!(
                    "    state {}: {}{} = {},",
                    field.to_field,
                    field.field_type,
                    field.evidentiality,
                    field.initial_value
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
                let state_changes = if msg.state_changes.is_empty() {
                    "        // TODO: implement".to_string()
                } else {
                    msg.state_changes.iter()
                        .map(|c| format!("        {};", c))
                        .collect::<Vec<_>>()
                        .join("\n")
                };

                // Include inlined effects if any
                let inlined_effects: Vec<String> = self.spec.recommendations.effects.iter()
                    .filter(|e| e.strategy == EffectStrategy::Inline)
                    .filter(|e| e.inline_in.as_ref().map(|i| i.contains(&msg.name.to_lowercase())).unwrap_or(false))
                    .map(|e| format!("        // Effect: {}", e.reasoning))
                    .collect();

                let body = if inlined_effects.is_empty() {
                    state_changes
                } else {
                    format!("{}\n{}", state_changes, inlined_effects.join("\n"))
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

        let body = if let Some(root) = &jsx.root {
            self.generate_vnode(root, 2)
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
                .map(|f| format!("{}: {}", f.name, f.field_type))
                .collect::<Vec<_>>()
                .join(", ")
        };

        let jsx = &self.spec.source.extraction.jsx;
        let body = if let Some(root) = &jsx.root {
            self.generate_vnode(root, 1)
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

    fn generate_vnode(&self, node: &JsxNode, indent: usize) -> String {
        let pad = "    ".repeat(indent);

        match &node.node_type {
            JsxNodeType::Element { tag, is_component, attributes, children } => {
                self.generate_element_vnode(tag, *is_component, attributes, children, indent)
            }
            JsxNodeType::Fragment { children } => {
                self.generate_fragment_vnode(children, indent)
            }
            JsxNodeType::Text { value } => {
                format!("{}·text_child(\"{}\")", pad, escape_string(value))
            }
            JsxNodeType::Expression { code } => {
                // Expression interpolation - convert to text_child with to_string
                let expr = self.transform_expression(code);
                format!("{}·text_child({}·to_string())", pad, expr)
            }
            JsxNodeType::Conditional { condition, consequent, alternate } => {
                let cond_expr = self.transform_expression(condition);
                let cons = self.generate_vnode(consequent, indent);
                if let Some(alt) = alternate {
                    let alt_code = self.generate_vnode(alt, indent);
                    format!("{}·when_else({}, {}, {})", pad, cond_expr, cons.trim(), alt_code.trim())
                } else {
                    format!("{}·when({}, {})", pad, cond_expr, cons.trim())
                }
            }
            JsxNodeType::Map { iterable, item_name, body, .. } => {
                let iter_expr = self.transform_expression(iterable);
                let body_code = self.generate_vnode(body, indent + 1);
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
            let attr_code = self.generate_attribute(attr);
            if !attr_code.is_empty() {
                builder.push_str(&format!("\n{}    {}", pad, attr_code));
            }
        }

        // Add children
        for child in children {
            let child_code = self.generate_vnode(child, indent + 1);
            match &child.node_type {
                JsxNodeType::Text { value } => {
                    builder.push_str(&format!("\n{}    ·text_child(\"{}\")", pad, escape_string(value)));
                }
                _ => {
                    builder.push_str(&format!("\n{}    ·child(\n{}\n{}    )", pad, child_code, pad));
                }
            }
        }

        builder
    }

    fn generate_fragment_vnode(&self, children: &[JsxNode], indent: usize) -> String {
        let pad = "    ".repeat(indent);
        let mut builder = format!("{}VNode·fragment()", pad);

        for child in children {
            let child_code = self.generate_vnode(child, indent + 1);
            builder.push_str(&format!("\n{}    ·child(\n{}\n{}    )", pad, child_code, pad));
        }

        builder
    }

    fn generate_attribute(&self, attr: &JsxAttribute) -> String {
        // Handle special attributes
        match attr.name.as_str() {
            "className" | "class" => {
                match &attr.value {
                    JsxAttributeValue::String { value } => format!("·class(\"{}\")", value),
                    JsxAttributeValue::Expression { code } => format!("·class({})", code),
                    _ => String::new(),
                }
            }
            "id" => {
                match &attr.value {
                    JsxAttributeValue::String { value } => format!("·id(\"{}\")", value),
                    JsxAttributeValue::Expression { code } => format!("·id({})", code),
                    _ => String::new(),
                }
            }
            "style" => {
                // Style needs special handling - simplified for now
                format!("·style(/* style object */)")
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
                        format!("·when({}, |n| n·attr(\"{}\", \"true\"))", code, attr.name)
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
                        format!("·attr(\"{}\", {})", attr.name, code)
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
                        format!("·attr(\"{}\", {})", attr.name, code)
                    }
                    JsxAttributeValue::Spread { name } => {
                        format!("/* spread: {} */", name)
                    }
                    JsxAttributeValue::True => {
                        format!("·attr(\"{}\", \"true\")", attr.name)
                    }
                }
            }
        }
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

// =============================================================================
// Helper Functions
// =============================================================================

impl<'a> QliphothGenerator<'a> {
    /// Transform a React expression into Sigil syntax.
    /// For actors: prefix state with `self.`
    /// For pure functions: use parameters directly
    fn transform_expression(&self, code: &str) -> String {
        // Handle placeholder/invalid expressions
        if code.contains("/*") || code.is_empty() {
            return "/* expression */".to_string();
        }

        // Clean up the expression
        let code = code.trim();

        if self.is_actor {
            // For actors, simple identifiers become self.identifier
            // More complex expressions need smarter handling
            if is_simple_identifier(code) {
                format!("self.{}", code)
            } else {
                // For complex expressions, try to prefix state variables
                self.prefix_state_variables(code)
            }
        } else {
            // For pure functions, check if it's a prop parameter
            if self.param_names.iter().any(|p| code == p || code.starts_with(&format!("{}.", p))) {
                code.to_string()
            } else if is_simple_identifier(code) {
                // Might be a prop - just use as-is
                code.to_string()
            } else {
                // Complex expression - use as-is
                code.to_string()
            }
        }
    }

    /// For actors, prefix state field references with self.
    fn prefix_state_variables(&self, code: &str) -> String {
        let state_fields: Vec<String> = self.spec.recommendations.state_fields
            .iter()
            .map(|f| f.to_field.clone())
            .collect();

        let mut result = code.to_string();

        // Simple approach: prefix known state fields with self.
        // This is a basic implementation - a real one would parse the expression
        for field in &state_fields {
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

        // If no state field was found and it's a simple identifier, prefix with self
        if !result.starts_with("self.") && is_simple_identifier(&result) {
            result = format!("self.{}", result);
        }

        result
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
