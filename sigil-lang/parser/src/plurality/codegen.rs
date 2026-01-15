//! # Plurality Code Generation
//!
//! Code generation for plurality constructs. Transforms plurality AST nodes
//! into runtime Rust code that integrates with the DAEMONIORUM game engine.
//!
//! ## Generated Code Patterns
//!
//! ### Alter Definitions
//! ```text
//! // Generated from: alter Abaddon: Council { ... }
//! pub struct Abaddon {
//!     pub archetype: Archetype,
//!     pub preferred_reality: RealityLayer,
//!     pub abilities: Vec<Ability>,
//!     pub triggers: Vec<Trigger>,
//!     pub anima: AnimaState,
//!     pub state_machine: AlterStateMachine,
//! }
//!
//! impl Alter for Abaddon {
//!     fn category(&self) -> AlterCategory { AlterCategory::Council }
//!     fn can_front(&self) -> bool { true }
//!     // ...
//! }
//! ```
//!
//! ### Alter Blocks
//! ```text
//! // Generated from: alter Abaddon { ... }
//! {
//!     let _alter_guard = system.enter_alter::<Abaddon>();
//!     // ... block contents ...
//! } // AlterGuard drops, restores previous fronter
//! ```
//!
//! ### Switch Expressions
//! ```text
//! // Generated from: switch to Beleth { ... }
//! system.propose_switch(SwitchRequest {
//!     target: AlterId::Beleth,
//!     reason: SwitchReason::Combat,
//!     urgency: 0.8,
//!     requires: Consensus::Majority,
//! }).then(|result| {
//!     // then block
//! }).otherwise(|result| {
//!     // else block
//! })
//! ```

use std::fmt::Write;

use super::ast::*;
use crate::ast::Visibility;

// ============================================================================
// CODE GENERATOR
// ============================================================================

/// Plurality code generator
#[derive(Debug, Default)]
pub struct PluralityCodeGen {
    /// Output buffer
    output: String,
    /// Indentation level
    indent: usize,
    /// Module name for generated code
    module_name: String,
}

impl PluralityCodeGen {
    /// Create a new code generator
    pub fn new(module_name: &str) -> Self {
        Self {
            output: String::new(),
            indent: 0,
            module_name: module_name.to_string(),
        }
    }

    /// Generate code for a plurality item
    pub fn generate_item(&mut self, item: &PluralityItem) -> String {
        match item {
            PluralityItem::Alter(def) => self.generate_alter_def(def),
            PluralityItem::Headspace(def) => self.generate_headspace_def(def),
            PluralityItem::Reality(def) => self.generate_reality_def(def),
            PluralityItem::CoConChannel(channel) => self.generate_cocon_channel(channel),
            PluralityItem::TriggerHandler(handler) => self.generate_trigger_handler(handler),
        }
    }

    /// Generate code for an alter definition
    pub fn generate_alter_def(&mut self, def: &AlterDef) -> String {
        self.output.clear();

        // Generate struct
        self.write_visibility(&def.visibility);
        self.writeln(&format!("struct {} {{", def.name.name));
        self.indent += 1;

        // Core fields
        self.writeln("pub archetype: Option<Archetype>,");
        self.writeln("pub preferred_reality: RealityLayer,");
        self.writeln("pub abilities: Vec<Ability>,");
        self.writeln("pub triggers: Vec<TriggerDef>,");
        self.writeln("pub anima: AnimaState,");
        self.writeln("pub state_machine: AlterStateMachine,");
        self.writeln("pub state: AlterRuntimeState,");

        self.indent -= 1;
        self.writeln("}");
        self.writeln("");

        // Generate Default impl
        self.generate_alter_default(def);

        // Generate Alter trait impl
        self.generate_alter_trait_impl(def);

        // Generate methods
        if !def.body.methods.is_empty() {
            self.writeln(&format!("impl {} {{", def.name.name));
            self.indent += 1;

            for method in &def.body.methods {
                self.generate_alter_method(method);
            }

            self.indent -= 1;
            self.writeln("}");
        }

        self.output.clone()
    }

    /// Generate Default implementation for alter
    fn generate_alter_default(&mut self, def: &AlterDef) {
        self.writeln(&format!("impl Default for {} {{", def.name.name));
        self.indent += 1;
        self.writeln("fn default() -> Self {");
        self.indent += 1;
        self.writeln("Self {");
        self.indent += 1;

        // Archetype
        if let Some(archetype) = &def.body.archetype {
            self.writeln(&format!("archetype: Some({}),", expr_to_string(archetype)));
        } else {
            self.writeln("archetype: None,");
        }

        // Preferred reality
        if let Some(reality) = &def.body.preferred_reality {
            self.writeln(&format!("preferred_reality: {},", expr_to_string(reality)));
        } else {
            self.writeln("preferred_reality: RealityLayer::Grounded,");
        }

        // Abilities
        self.write("abilities: vec![");
        for (i, ability) in def.body.abilities.iter().enumerate() {
            if i > 0 {
                self.write(", ");
            }
            self.write(&expr_to_string(ability));
        }
        self.writeln("],");

        // Triggers
        self.write("triggers: vec![");
        for (i, trigger) in def.body.triggers.iter().enumerate() {
            if i > 0 {
                self.write(", ");
            }
            self.write(&expr_to_string(trigger));
        }
        self.writeln("],");

        // Anima
        self.generate_anima_init(&def.body.anima);

        // State machine
        self.generate_state_machine_init(&def.body.states);

        // Runtime state
        self.writeln("state: AlterRuntimeState::Dormant,");

        self.indent -= 1;
        self.writeln("}");
        self.indent -= 1;
        self.writeln("}");
        self.indent -= 1;
        self.writeln("}");
        self.writeln("");
    }

    /// Generate Alter trait implementation
    fn generate_alter_trait_impl(&mut self, def: &AlterDef) {
        self.writeln(&format!("impl Alter for {} {{", def.name.name));
        self.indent += 1;

        // Category
        let category = match def.category {
            AlterCategory::Council => "AlterCategory::Council",
            AlterCategory::Servant => "AlterCategory::Servant",
            AlterCategory::Fragment => "AlterCategory::Fragment",
            AlterCategory::Hidden => "AlterCategory::Hidden",
            AlterCategory::Persecutor => "AlterCategory::Persecutor",
            AlterCategory::Custom => "AlterCategory::Custom",
        };
        self.writeln(&format!(
            "fn category(&self) -> AlterCategory {{ {} }}",
            category
        ));

        // Can front
        let can_front = matches!(def.category, AlterCategory::Council);
        self.writeln(&format!("fn can_front(&self) -> bool {{ {} }}", can_front));

        // Name
        self.writeln(&format!(
            "fn name(&self) -> &'static str {{ \"{}\" }}",
            def.name.name
        ));

        // Archetype
        self.writeln("fn archetype(&self) -> Option<&Archetype> { self.archetype.as_ref() }");

        // Preferred reality
        self.writeln("fn preferred_reality(&self) -> RealityLayer { self.preferred_reality }");

        // Abilities
        self.writeln("fn abilities(&self) -> &[Ability] { &self.abilities }");

        // Triggers
        self.writeln("fn triggers(&self) -> &[TriggerDef] { &self.triggers }");

        // Anima
        self.writeln("fn anima(&self) -> &AnimaState { &self.anima }");
        self.writeln("fn anima_mut(&mut self) -> &mut AnimaState { &mut self.anima }");

        // State
        self.writeln("fn state(&self) -> AlterRuntimeState { self.state }");
        self.writeln("fn set_state(&mut self, state: AlterRuntimeState) { self.state = state; }");

        self.indent -= 1;
        self.writeln("}");
        self.writeln("");
    }

    /// Generate anima initialization
    fn generate_anima_init(&mut self, anima: &Option<AnimaConfig>) {
        if let Some(config) = anima {
            self.writeln("anima: AnimaState {");
            self.indent += 1;

            if let Some(arousal) = &config.base_arousal {
                self.writeln(&format!("arousal: {},", expr_to_string(arousal)));
            } else {
                self.writeln("arousal: 0.5,");
            }

            if let Some(dominance) = &config.base_dominance {
                self.writeln(&format!("dominance: {},", expr_to_string(dominance)));
            } else {
                self.writeln("dominance: 0.5,");
            }

            if let Some(expressiveness) = &config.expressiveness {
                self.writeln(&format!("expressiveness: {},", expr_to_string(expressiveness)));
            } else {
                self.writeln("expressiveness: 0.5,");
            }

            if let Some(susceptibility) = &config.susceptibility {
                self.writeln(&format!("susceptibility: {},", expr_to_string(susceptibility)));
            } else {
                self.writeln("susceptibility: 0.5,");
            }

            self.writeln("..Default::default()");
            self.indent -= 1;
            self.writeln("},");
        } else {
            self.writeln("anima: AnimaState::default(),");
        }
    }

    /// Generate state machine initialization
    fn generate_state_machine_init(&mut self, states: &Option<AlterStateMachine>) {
        if let Some(sm) = states {
            self.writeln("state_machine: AlterStateMachine {");
            self.indent += 1;
            self.writeln("transitions: vec![");
            self.indent += 1;

            for transition in &sm.transitions {
                self.writeln("AlterTransitionDef {");
                self.indent += 1;
                self.writeln(&format!(
                    "from: AlterRuntimeState::{:?},",
                    transition.from
                ));
                self.writeln(&format!(
                    "to: AlterRuntimeState::{:?},",
                    transition.to
                ));
                self.writeln(&format!(
                    "on: TriggerCondition::from({}),",
                    expr_to_string(&transition.on)
                ));

                if let Some(guard) = &transition.guard {
                    self.writeln(&format!(
                        "guard: Some(Box::new(|ctx| {})),",
                        expr_to_string(guard)
                    ));
                } else {
                    self.writeln("guard: None,");
                }

                if transition.action.is_some() {
                    self.writeln("action: Some(Box::new(|ctx| { /* action */ })),");
                } else {
                    self.writeln("action: None,");
                }

                self.indent -= 1;
                self.writeln("},");
            }

            self.indent -= 1;
            self.writeln("],");
            self.indent -= 1;
            self.writeln("},");
        } else {
            self.writeln("state_machine: AlterStateMachine::default(),");
        }
    }

    /// Generate an alter method
    fn generate_alter_method(&mut self, method: &AlterMethod) {
        self.write_visibility(&method.visibility);

        if method.is_async {
            self.write("async ");
        }

        self.write(&format!("fn {}(&", method.name.name));
        if method.params.iter().any(|p| is_self_pattern(&p.pattern)) {
            self.write("mut self");
        } else {
            self.write("self");
        }

        for param in &method.params {
            if !is_self_pattern(&param.pattern) {
                self.write(&format!(", {}", pattern_to_string(&param.pattern)));
                self.write(&format!(": {}", type_to_string(&param.ty)));
            }
        }
        self.write(")");

        if let Some(ret_ty) = &method.return_type {
            self.write(&format!(" -> {}", type_to_string(ret_ty)));
        }

        if let Some(_body) = &method.body {
            self.writeln(" {");
            self.indent += 1;
            self.writeln("// Method body");
            self.indent -= 1;
            self.writeln("}");
        } else {
            self.writeln(";");
        }
    }

    /// Generate code for a headspace definition
    pub fn generate_headspace_def(&mut self, def: &HeadspaceDef) -> String {
        self.output.clear();

        // Generate module
        self.write_visibility(&def.visibility);
        self.writeln(&format!("mod {} {{", def.name.name.to_lowercase()));
        self.indent += 1;
        self.writeln("use super::*;");
        self.writeln("");

        // Generate location structs
        for location in &def.locations {
            self.generate_location_struct(location);
        }

        // Generate headspace struct
        self.writeln(&format!("pub struct {} {{", def.name.name));
        self.indent += 1;
        for location in &def.locations {
            self.writeln(&format!(
                "pub {}: {},",
                location.name.name.to_lowercase(),
                location.name.name
            ));
        }
        self.indent -= 1;
        self.writeln("}");
        self.writeln("");

        // Generate impl with methods
        if !def.methods.is_empty() {
            self.writeln(&format!("impl {} {{", def.name.name));
            self.indent += 1;
            for method in &def.methods {
                self.generate_alter_method(method);
            }
            self.indent -= 1;
            self.writeln("}");
        }

        self.indent -= 1;
        self.writeln("}");

        self.output.clone()
    }

    /// Generate a location struct
    fn generate_location_struct(&mut self, location: &LocationDef) {
        self.writeln(&format!("pub struct {} {{", location.name.name));
        self.indent += 1;
        self.writeln(&format!(
            "pub location_type: {},",
            location.location_type.name
        ));

        for (field_name, _) in &location.fields {
            // Infer field type from field name or default to generic
            self.writeln(&format!("pub {}: LocationField,", field_name.name));
        }

        if !location.connections.is_empty() {
            self.writeln("pub connections: Vec<ConsciousnessStream>,");
        }

        if !location.hazards.is_empty() {
            self.writeln("pub hazards: Vec<Hazard>,");
        }

        self.indent -= 1;
        self.writeln("}");
        self.writeln("");
    }

    /// Generate code for a reality definition
    pub fn generate_reality_def(&mut self, def: &RealityDef) -> String {
        self.output.clear();

        // Generate superimposed entity struct
        self.write_visibility(&def.visibility);
        self.writeln(&format!("struct {} {{", def.name.name));
        self.indent += 1;

        for layer in &def.layers {
            self.writeln(&format!(
                "pub {}: {}Layer,",
                layer.name.name.to_lowercase(),
                layer.name.name
            ));
        }

        self.indent -= 1;
        self.writeln("}");
        self.writeln("");

        // Generate layer structs
        for layer in &def.layers {
            self.writeln(&format!("pub struct {}Layer {{", layer.name.name));
            self.indent += 1;
            for (field_name, _) in &layer.fields {
                self.writeln(&format!("pub {}: RealityValue,", field_name.name));
            }
            self.indent -= 1;
            self.writeln("}");
            self.writeln("");
        }

        // Generate Superimposed trait impl
        self.writeln(&format!("impl Superimposed for {} {{", def.name.name));
        self.indent += 1;

        self.writeln("fn current(&self, perception: &PerceptionState) -> &dyn RealityLayerView {");
        self.indent += 1;
        self.writeln("match perception.current_layer() {");
        self.indent += 1;

        for layer in &def.layers {
            self.writeln(&format!(
                "RealityLayer::{} => &self.{},",
                layer.name.name,
                layer.name.name.to_lowercase()
            ));
        }
        self.writeln("_ => &self.grounded,");

        self.indent -= 1;
        self.writeln("}");
        self.indent -= 1;
        self.writeln("}");

        // Generate transform logic
        if !def.transforms.is_empty() {
            self.writeln("");
            self.writeln("fn check_transform(&self, perception: &PerceptionState) -> Option<RealityLayer> {");
            self.indent += 1;

            for transform in &def.transforms {
                self.writeln(&format!(
                    "if perception.current_layer() == RealityLayer::{} && ({}) {{",
                    transform.from.name,
                    expr_to_string(&transform.condition)
                ));
                self.indent += 1;
                self.writeln(&format!(
                    "return Some(RealityLayer::{});",
                    transform.to.name
                ));
                self.indent -= 1;
                self.writeln("}");
            }

            self.writeln("None");
            self.indent -= 1;
            self.writeln("}");
        }

        self.indent -= 1;
        self.writeln("}");

        self.output.clone()
    }

    /// Generate code for a co-con channel
    pub fn generate_cocon_channel(&mut self, channel: &CoConChannel) -> String {
        self.output.clear();

        // Generate channel struct
        self.writeln(&format!("pub struct {}Channel {{", channel.name.name));
        self.indent += 1;
        self.writeln("participants: Vec<AlterId>,");
        self.writeln("active: bool,");
        self.indent -= 1;
        self.writeln("}");
        self.writeln("");

        // Generate impl
        self.writeln(&format!("impl {}Channel {{", channel.name.name));
        self.indent += 1;

        // Constructor
        self.writeln("pub fn new() -> Self {");
        self.indent += 1;
        self.writeln("Self {");
        self.indent += 1;
        self.write("participants: vec![");
        for (i, p) in channel.participants.iter().enumerate() {
            if i > 0 {
                self.write(", ");
            }
            self.write(&format!("AlterId::{}", p.name));
        }
        self.writeln("],");
        self.writeln("active: false,");
        self.indent -= 1;
        self.writeln("}");
        self.indent -= 1;
        self.writeln("}");
        self.writeln("");

        // Activate
        self.writeln("pub fn activate(&mut self, system: &PluralSystem) -> Result<(), ChannelError> {");
        self.indent += 1;
        self.writeln("for &p in &self.participants {");
        self.indent += 1;
        self.writeln("if !system.is_cocon(p) {");
        self.indent += 1;
        self.writeln("return Err(ChannelError::NotCoConscious(p));");
        self.indent -= 1;
        self.writeln("}");
        self.indent -= 1;
        self.writeln("}");
        self.writeln("self.active = true;");
        self.writeln("Ok(())");
        self.indent -= 1;
        self.writeln("}");

        self.indent -= 1;
        self.writeln("}");

        self.output.clone()
    }

    /// Generate code for a trigger handler
    pub fn generate_trigger_handler(&mut self, handler: &TriggerHandler) -> String {
        self.output.clear();

        // Generate handler function
        self.writeln(&format!(
            "pub fn handle_{}(ctx: &mut TriggerContext) -> TriggerResult {{",
            handler.pattern.trigger_type.name.to_lowercase()
        ));
        self.indent += 1;

        // Destructure trigger
        if !handler.pattern.fields.is_empty() {
            self.write(&format!(
                "let {} {{ ",
                handler.pattern.trigger_type.name
            ));
            for (i, (field, binding)) in handler.pattern.fields.iter().enumerate() {
                if i > 0 {
                    self.write(", ");
                }
                if field.name != binding.name {
                    self.write(&format!("{}: {}", field.name, binding.name));
                } else {
                    self.write(&binding.name);
                }
            }
            self.writeln(" } = ctx.trigger();");
        }

        // Guard condition
        if let Some(guard) = &handler.guard {
            self.writeln(&format!("if !({}) {{", expr_to_string(guard)));
            self.indent += 1;
            self.writeln("return TriggerResult::Ignored;");
            self.indent -= 1;
            self.writeln("}");
        }

        // Handler body
        self.writeln("// Handler body");
        self.writeln("TriggerResult::Handled");

        self.indent -= 1;
        self.writeln("}");

        self.output.clone()
    }

    /// Generate code for a switch expression
    pub fn generate_switch_expr(&mut self, expr: &SwitchExpr) -> String {
        let mut out = String::new();

        if expr.forced {
            write!(out, "system.force_switch(").unwrap();
        } else {
            write!(out, "system.propose_switch(").unwrap();
        }

        write!(out, "SwitchRequest {{").unwrap();
        write!(out, " target: {},", alter_expr_to_string(&expr.target)).unwrap();

        if let Some(reason) = &expr.config.reason {
            write!(out, " reason: {},", expr_to_string(reason)).unwrap();
        }

        if let Some(urgency) = &expr.config.urgency {
            write!(out, " urgency: {},", expr_to_string(urgency)).unwrap();
        }

        if let Some(requires) = &expr.config.requires {
            write!(out, " requires: {},", expr_to_string(requires)).unwrap();
        }

        write!(out, " bypass_deliberation: {},", expr.config.bypass_deliberation).unwrap();
        write!(out, " ..Default::default() }})").unwrap();

        // Add then/else closures
        if expr.config.then_block.is_some() || expr.config.else_block.is_some() {
            write!(out, ".handle(|result| match result {{ ").unwrap();
            if expr.config.then_block.is_some() {
                write!(out, "SwitchResult::Success => {{ /* then block */ }}, ").unwrap();
            }
            if expr.config.else_block.is_some() {
                write!(out, "SwitchResult::Denied(_) => {{ /* else block */ }}, ").unwrap();
            }
            if expr.config.emergency_block.is_some() {
                write!(out, "SwitchResult::Emergency => {{ /* emergency block */ }}, ").unwrap();
            }
            write!(out, "_ => {{}} }})").unwrap();
        }

        out
    }

    /// Generate code for a split expression
    pub fn generate_split_expr(&mut self, expr: &SplitExpr) -> String {
        let mut out = String::new();

        write!(out, "system.process_split(SplitRequest {{").unwrap();
        write!(out, " parent: {},", alter_expr_to_string(&expr.parent)).unwrap();

        if let Some(purpose) = &expr.config.purpose {
            write!(out, " purpose: {},", expr_to_string(purpose)).unwrap();
        }

        if let Some(memories) = &expr.config.memories {
            write!(out, " memories: {},", expr_to_string(memories)).unwrap();
        }

        if let Some(traits) = &expr.config.traits {
            write!(out, " traits: {},", expr_to_string(traits)).unwrap();
        }

        write!(out, " ..Default::default() }})").unwrap();

        out
    }

    /// Generate code for an alter block
    pub fn generate_alter_block(&mut self, block: &AlterBlock) -> String {
        let mut out = String::new();

        write!(out, "{{ let _alter_guard = system.enter_alter::<").unwrap();
        write!(out, "{}", alter_expr_to_string(&block.alter)).unwrap();
        write!(out, ">();").unwrap();
        write!(out, " /* block body */ }}").unwrap();

        out
    }

    // Helper methods

    fn write(&mut self, s: &str) {
        self.output.push_str(s);
    }

    fn writeln(&mut self, s: &str) {
        for _ in 0..self.indent {
            self.output.push_str("    ");
        }
        self.output.push_str(s);
        self.output.push('\n');
    }

    fn write_visibility(&mut self, vis: &Visibility) {
        match vis {
            Visibility::Public => self.write("pub "),
            Visibility::Crate => self.write("pub(crate) "),
            Visibility::Super => self.write("pub(super) "),
            Visibility::Private => {}
        }
    }
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Convert expression to string (placeholder)
fn expr_to_string(expr: &crate::ast::Expr) -> String {
    format!("{:?}", expr).replace('"', "'")
}

/// Convert type expression to string (placeholder)
fn type_to_string(ty: &crate::ast::TypeExpr) -> String {
    format!("{:?}", ty)
}

/// Convert alter expression to string
fn alter_expr_to_string(alter: &AlterExpr) -> String {
    match alter {
        AlterExpr::Named(ident) => ident.name.clone(),
        AlterExpr::CurrentFronter(_) => "system.current_fronter()".to_string(),
        AlterExpr::Expr(expr) => expr_to_string(expr),
    }
}

/// Check if a pattern is the "self" identifier
fn is_self_pattern(pattern: &crate::ast::Pattern) -> bool {
    match pattern {
        crate::ast::Pattern::Ident { name, .. } => name.name == "self",
        _ => false,
    }
}

/// Convert pattern to string (placeholder)
fn pattern_to_string(pattern: &crate::ast::Pattern) -> String {
    match pattern {
        crate::ast::Pattern::Ident { mutable, name, .. } => {
            if *mutable {
                format!("mut {}", name.name)
            } else {
                name.name.clone()
            }
        }
        crate::ast::Pattern::Tuple(patterns) => {
            let inner: Vec<_> = patterns.iter().map(pattern_to_string).collect();
            format!("({})", inner.join(", "))
        }
        _ => format!("{:?}", pattern),
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Expr, Ident, Literal};
    use crate::span::Span;

    fn mock_ident(name: &str) -> Ident {
        Ident {
            name: name.to_string(),
            evidentiality: None,
            affect: None,
            span: Span::default(),
        }
    }

    #[test]
    fn test_generate_alter_def() {
        let def = AlterDef {
            visibility: Visibility::Public,
            attrs: Vec::new(),
            name: mock_ident("Abaddon"),
            category: AlterCategory::Council,
            generics: None,
            where_clause: None,
            body: AlterBody {
                archetype: Some(Expr::Path(crate::ast::TypePath {
                    segments: vec![crate::ast::PathSegment {
                        ident: mock_ident("Goetia"),
                        generics: None,
                    }],
                })),
                preferred_reality: None,
                abilities: Vec::new(),
                triggers: Vec::new(),
                anima: None,
                states: None,
                special: Vec::new(),
                methods: Vec::new(),
                types: Vec::new(),
            },
            span: Span::default(),
        };

        let mut gen = PluralityCodeGen::new("test");
        let output = gen.generate_alter_def(&def);

        assert!(output.contains("struct Abaddon"));
        assert!(output.contains("impl Alter for Abaddon"));
        assert!(output.contains("AlterCategory::Council"));
    }

    #[test]
    fn test_generate_switch_expr() {
        let expr = SwitchExpr {
            forced: false,
            target: AlterExpr::Named(mock_ident("Beleth")),
            config: SwitchConfig {
                reason: Some(Expr::Literal(Literal::String("Combat".to_string()))),
                urgency: Some(Expr::Literal(Literal::Float {
                    value: "0.8".to_string(),
                    suffix: None,
                })),
                requires: None,
                then_block: None,
                else_block: None,
                emergency_block: None,
                bypass_deliberation: false,
            },
            span: Span::default(),
        };

        let mut gen = PluralityCodeGen::new("test");
        let output = gen.generate_switch_expr(&expr);

        assert!(output.contains("propose_switch"));
        assert!(output.contains("Beleth"));
    }

    #[test]
    fn test_generate_alter_block() {
        let block = AlterBlock {
            alter: AlterExpr::Named(mock_ident("Abaddon")),
            body: crate::ast::Block {
                stmts: Vec::new(),
                expr: None,
            },
            span: Span::default(),
        };

        let mut gen = PluralityCodeGen::new("test");
        let output = gen.generate_alter_block(&block);

        assert!(output.contains("enter_alter"));
        assert!(output.contains("Abaddon"));
    }
}
