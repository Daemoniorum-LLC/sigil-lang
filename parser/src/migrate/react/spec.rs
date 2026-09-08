//! Migration spec generation for React → Qliphoth.
//!
//! Takes a ReactExtraction and produces a MigrationSpec with:
//! - State field recommendations (from hooks)
//! - Message recommendations (from event handlers)
//! - Effect handling strategies
//! - Pattern examples
//! - Ambiguity detection
//!
//! See docs/specs/REACT-MIGRATION.md Section 4 for specification.

use serde::{Deserialize, Serialize};
use super::extraction::*;

// =============================================================================
// MigrationSpec - Top Level
// =============================================================================

/// Complete migration specification for a React project.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationSpec {
    pub version: String,
    pub generated_at: String,
    pub project_root: String,
    pub components: Vec<ComponentMigrationSpec>,
    pub types: Vec<TypeMigrationSpec>,
    /// Helper functions at module scope (Phase 6.2)
    #[serde(default)]
    pub helper_functions: Vec<HelperFunctionExtraction>,
    /// Service actors derived from custom hooks (Phase 7)
    #[serde(default)]
    pub service_actors: Vec<ServiceActorSpec>,
    pub state: MigrationState,
}

/// Migration state tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationState {
    pub total_components: usize,
    pub completed: usize,
    pub in_progress: usize,
    pub blocked: usize,
    pub last_updated: String,
}

// =============================================================================
// Component Migration Spec
// =============================================================================

/// Migration spec for a single component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentMigrationSpec {
    pub id: String,
    pub name: String,
    pub source: ComponentSource,
    pub target: TargetInfo,
    pub recommendations: Recommendations,
    pub patterns: Vec<PatternExample>,
    pub ambiguities: Vec<Ambiguity>,
    pub dependencies: Dependencies,
    pub complexity: Complexity,
    pub complexity_factors: Vec<String>,
    pub status: MigrationStatus,
}

/// JavaScript globals a component may name. They are not fields of anything, so
/// the unknown-identifier rule turned `Math.round(x)` into `self.math.round(x)`.
/// Listing them keeps the reference intact for a human to finish; Sigil has its
/// own spellings for most of these and none of them are `self.`.
const JS_GLOBALS: &[&str] = &[
    "Math", "JSON", "Object", "Array", "String", "Number", "Boolean", "Date",
    "RegExp", "Promise", "Map", "Set", "Error", "console", "window", "document",
    "navigator", "localStorage", "sessionStorage", "fetch", "URL",
    "URLSearchParams", "Intl", "parseInt", "parseFloat", "isNaN", "encodeURIComponent",
    "decodeURIComponent", "structuredClone", "queueMicrotask",
];

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentSource {
    pub path: String,
    pub code: String,
    pub extraction: ComponentExtraction,
    /// Names declared at module scope in the same file — helper functions and
    /// non-function `const`s.
    ///
    /// The generator needs these to leave a reference alone. Without them the
    /// unknown-identifier rule turned `fmtDate(x)` into `self.fmt_date(x)` and
    /// `DEFAULT_SORT` into `self.default_sort`, neither of which any actor
    /// declares — 108 references across the generated Lares client.
    #[serde(default)]
    pub module_scope: Vec<String>,
    /// Module-scope `const`s from the same file, so the generator can emit them.
    ///
    /// Knowing the name is not enough: `DEFAULT_SORT` stopped being rewritten to
    /// `self.default_sort` once it was in scope, but nothing declared it either.
    #[serde(default)]
    pub module_constants: Vec<ModuleConstantExtraction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TargetInfo {
    pub suggested_path: String,
    pub pattern: TargetPattern,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TargetPattern {
    Actor,    // Stateful component → actor
    Function, // Pure component → rite function
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Complexity {
    Simple,
    Moderate,
    Complex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MigrationStatus {
    Pending,
    InProgress,
    Completed,
    Blocked,
}

// =============================================================================
// Recommendations
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recommendations {
    pub state_fields: Vec<StateFieldRecommendation>,
    pub messages: Vec<MessageRecommendation>,
    pub effects: Vec<EffectRecommendation>,
    pub props_handling: PropsRecommendation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateFieldRecommendation {
    pub from_hook: String,         // "useState:count"
    pub to_field: String,          // "count"
    pub field_type: String,        // "i32"
    pub evidentiality: String,     // "!" | "?" | "~"
    pub initial_value: String,
    pub reasoning: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MessageRecommendation {
    pub name: String,              // "Increment"
    pub from_handler: String,      // "handleIncrement" or "onClick:button"
    pub payload: Option<String>,   // "{ amount: i32 }" or None
    pub state_changes: Vec<String>, // ["self.count += 1"]
    pub side_effects: Vec<String>,  // ["update document title"]
    /// Calls to service actor methods (from hook-returned functions)
    #[serde(default)]
    pub service_calls: Vec<ServiceCall>,
    /// The originating handler's parameter names, in order, for as many as the
    /// payload carries. The body's state changes are written in terms of these
    /// (`self.view = next`), so the generator has to bind them to `msg.N` first
    /// or the names resolve to nothing.
    #[serde(default)]
    pub param_bindings: Vec<String>,
    /// `state_changes` in structured form: the field written and the *untranslated*
    /// JavaScript that produces the value. `state_changes` holds the same thing
    /// flattened for the JSON spec, but flattening it meant the value never went
    /// through the expression transform — `setCollapsed(c => !c)` was emitted as
    /// a JavaScript arrow inside a Sigil actor, and seven files stopped parsing
    /// the moment plain-named handlers began contributing bodies.
    #[serde(default)]
    pub state_assignments: Vec<StateAssignment>,
    /// The React handler this came from had branches or an early return, and the
    /// mutation walker collects mutations from every branch into one flat list.
    /// The generated body therefore runs assignments unconditionally that React
    /// ran under a condition — a real semantic difference, and one a reader would
    /// otherwise have to diff against the TSX to notice.
    #[serde(default)]
    pub flattened_control_flow: bool,
}

/// A single `self.<field> = <value>` a message handler performs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StateAssignment {
    /// Sigil field name (already snake-cased and keyword-escaped).
    pub field: String,
    /// The value as React wrote it. Transformed by the generator, not here.
    pub value: String,
    /// Set when React used the functional updater form, `setX(prev => ...)`:
    /// the parameter name, which stands for the field's current value.
    #[serde(default)]
    pub updater_param: Option<String>,
}

/// A call to a service actor method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceCall {
    /// The service actor name (e.g., "ChatService")
    pub service: String,
    /// The method/message name (e.g., "AddMessage")
    pub method: String,
    /// Arguments to pass
    pub args: Vec<String>,
}

// =============================================================================
// Service Actor Spec (Phase 7)
// =============================================================================

/// Specification for a service actor derived from custom hook analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceActorSpec {
    /// Actor name (e.g., "ChatService")
    pub name: String,
    /// The custom hook this was derived from (e.g., "useChat")
    pub derived_from: String,
    /// State fields inferred from hook return values
    pub state_fields: Vec<ServiceStateField>,
    /// Messages the actor responds to
    pub messages: Vec<ServiceMessage>,
    /// Components that use this service
    pub used_by: Vec<String>,
}

/// A state field for a service actor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceStateField {
    /// Field name (e.g., "messages", "is_streaming")
    pub name: String,
    /// Original name from hook (before snake_case conversion)
    pub original_name: String,
    /// Inferred type
    pub field_type: String,
    /// Whether this is observable/reactive state
    pub is_observable: bool,
}

/// A message type for a service actor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMessage {
    /// Message name (e.g., "AddMessage")
    pub name: String,
    /// Original function name from hook (e.g., "addMessage")
    pub original_name: String,
    /// Parameter types/names inferred from call sites
    pub parameters: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectRecommendation {
    pub from_hook: String,         // "useEffect[count]"
    pub strategy: EffectStrategy,
    pub reasoning: String,
    pub inline_in: Option<String>, // message name if strategy is Inline
    pub lifecycle_event: Option<String>, // "Mount" | "Unmount"
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum EffectStrategy {
    Inline,    // Inline in message handler
    Message,   // Separate message
    Lifecycle, // Mount/Unmount
    Remove,    // Not needed in Qliphoth
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropsRecommendation {
    pub strategy: PropsStrategy,
    pub fields: Vec<PropsField>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PropsStrategy {
    Constructor, // Pass via constructor (rite new)
    Message,     // Pass via message
    None,        // No props
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PropsField {
    pub name: String,
    pub field_type: String,
    pub from_prop: String,
}

// =============================================================================
// Patterns and Ambiguities
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternExample {
    pub name: String,
    pub description: String,
    pub react: String,
    pub sigil: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ambiguity {
    pub id: String,
    pub category: AmbiguityCategory,
    pub question: String,
    pub options: Vec<AmbiguityOption>,
    pub default_choice: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AmbiguityCategory {
    EffectPlacement,
    StateType,
    EventMapping,
    ComponentStructure,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbiguityOption {
    pub label: String,
    pub description: String,
    pub recommended: bool,
}

// =============================================================================
// Dependencies
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dependencies {
    pub components: Vec<String>, // Other component IDs that must migrate first
    pub types: Vec<String>,      // Type IDs needed
}

// =============================================================================
// Type Migration
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeMigrationSpec {
    pub id: String,
    pub name: String,
    pub source: String,     // Original TypeScript
    pub target: String,     // Generated Sigil code
    pub manual_review_needed: bool,
    pub notes: Vec<String>,
    /// Extracted fields with full type information (Phase 6.1)
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub fields: Vec<TypeFieldSpec>,
    /// Type parameters for generics
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub type_params: Vec<String>,
    /// Extended types (for interfaces)
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub extends: Vec<String>,
    /// Union variants (for type aliases and enums)
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub union_variants: Vec<String>,
}

/// Extracted type field with full details for Qliphoth generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeFieldSpec {
    pub name: String,
    pub type_annotation: String,
    pub optional: bool,
    pub readonly: bool,
    /// Classified type kind for easier mapping
    pub type_kind: String,
}

// =============================================================================
// Pattern Library
// =============================================================================

/// Built-in pattern library for common React → Qliphoth transformations.
pub fn pattern_library() -> Vec<PatternExample> {
    vec![
        PatternExample {
            name: "useState_to_state".to_string(),
            description: "Convert useState hook to actor state field".to_string(),
            react: "const [count, setCount] = useState(0);".to_string(),
            sigil: "state count: i32! = 0,".to_string(),
        },
        PatternExample {
            name: "onClick_to_message".to_string(),
            description: "Convert onClick handler to message dispatch".to_string(),
            react: "<button onClick={() => setCount(c => c + 1)}>".to_string(),
            sigil: "VNode·button()·on_click(Increment)".to_string(),
        },
        PatternExample {
            name: "useEffect_mount".to_string(),
            description: "Convert mount-only useEffect to lifecycle".to_string(),
            react: "useEffect(() => { init(); }, []);".to_string(),
            sigil: "on Mount { self.init(); }".to_string(),
        },
        PatternExample {
            name: "useEffect_deps".to_string(),
            description: "Convert useEffect with deps to inline in message handler".to_string(),
            react: "useEffect(() => { save(count); }, [count]);".to_string(),
            sigil: "// Inline in the message that changes count:\non Increment { self.count += 1; self.save(); }".to_string(),
        },
        PatternExample {
            name: "conditional_render".to_string(),
            description: "Convert conditional JSX to .when()".to_string(),
            react: "{isVisible && <Modal />}".to_string(),
            sigil: "·when(self.is_visible, Modal·render())".to_string(),
        },
        PatternExample {
            name: "list_render".to_string(),
            description: "Convert .map() to explicit loop".to_string(),
            react: "{items.map(item => <Item key={item.id} item={item} />)}".to_string(),
            sigil: r#"≔ children: Vec<VNode>! = vec![];
∀ item ∈ self.items {
    children.push(Item·render(item));
}
·children(children)"#.to_string(),
        },
        PatternExample {
            name: "jsx_to_builder".to_string(),
            description: "Convert JSX element to VNode builder".to_string(),
            react: r#"<div className="container" id="main">
  <h1>Title</h1>
  <p>Content</p>
</div>"#.to_string(),
            sigil: r#"VNode·div()
    ·class("container")
    ·id("main")
    ·child(VNode·h1()·text("Title"))
    ·child(VNode·p()·text("Content"))"#.to_string(),
        },
        PatternExample {
            name: "input_controlled".to_string(),
            description: "Convert controlled input to message-based".to_string(),
            react: r#"<input
  value={text}
  onChange={e => setText(e.target.value)}
/>"#.to_string(),
            sigil: r#"VNode·input()
    ·attr("value", self.text·as_str())
    ·on_input(TextChanged)"#.to_string(),
        },
        PatternExample {
            name: "useRef_to_state".to_string(),
            description: "Convert useRef to non-reactive state field".to_string(),
            react: "const inputRef = useRef<HTMLInputElement>(null);".to_string(),
            sigil: "state input_ref: Option<Element>! = ∅,".to_string(),
        },
        PatternExample {
            name: "useCallback_remove".to_string(),
            description: "useCallback is not needed in actors".to_string(),
            react: "const handleClick = useCallback(() => { ... }, [dep]);".to_string(),
            sigil: "// No equivalent needed - actors don't re-render like React".to_string(),
        },
    ]
}

// =============================================================================
// Spec Generator
// =============================================================================

/// Generate a migration spec from a React extraction.
pub fn generate_spec(extraction: &ReactExtraction, source_code: &str) -> MigrationSpec {
    let generator = SpecGenerator::new(extraction, source_code);
    generator.generate()
}

struct SpecGenerator<'a> {
    extraction: &'a ReactExtraction,
    source_code: &'a str,
}

impl<'a> SpecGenerator<'a> {
    fn new(extraction: &'a ReactExtraction, source_code: &'a str) -> Self {
        Self { extraction, source_code }
    }

    fn generate(self) -> MigrationSpec {
        let components: Vec<ComponentMigrationSpec> = self.extraction.components.iter()
            .map(|comp| self.generate_component_spec(comp))
            .collect();

        let types: Vec<TypeMigrationSpec> = self.extraction.types.iter()
            .map(|t| self.generate_type_spec(t))
            .collect();

        // Collect service actors from custom hooks across all components
        let service_actors = self.collect_service_actors(&components);

        let total = components.len();

        MigrationSpec {
            version: "1.0".to_string(),
            generated_at: chrono_now(),
            project_root: self.extraction.file.path.parent()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default(),
            components,
            types,
            helper_functions: self.extraction.helper_functions.clone(),
            service_actors,
            state: MigrationState {
                total_components: total,
                completed: 0,
                in_progress: 0,
                blocked: 0,
                last_updated: chrono_now(),
            },
        }
    }

    fn generate_component_spec(&self, comp: &ComponentExtraction) -> ComponentMigrationSpec {
        let id = format!("{}:{}", self.extraction.file.relative_path, comp.name);
        let has_state = !comp.hooks.iter()
            .filter(|h| h.hook_type == HookType::UseState)
            .collect::<Vec<_>>()
            .is_empty();

        let target_pattern = if has_state || !comp.hooks.is_empty() {
            TargetPattern::Actor
        } else {
            TargetPattern::Function
        };

        let recommendations = self.generate_recommendations(comp);
        let patterns = self.select_patterns(comp);
        let ambiguities = self.detect_ambiguities(comp);
        let (complexity, factors) = self.calculate_complexity(comp);
        let dependencies = self.extract_dependencies(comp);

        ComponentMigrationSpec {
            id,
            name: comp.name.clone(),
            source: ComponentSource {
                path: self.extraction.file.path.to_string_lossy().to_string(),
                code: self.source_code.to_string(),
                extraction: comp.clone(),
                module_scope: self
                    .extraction
                    .helper_functions
                    .iter()
                    .map(|h| h.name.clone())
                    .chain(
                        self.extraction
                            .module_constants
                            .iter()
                            .map(|c| c.name.clone()),
                    )
                    // Imported names are module scope too, and most of the
                    // helpers a component actually uses come from a sibling
                    // module — `kbToGB` lives in `../format`, not in the file
                    // that calls it, and became `self.kb_to_gb`.
                    .chain(
                        self.extraction
                            .imports
                            .iter()
                            .filter(|i| !i.is_type_only)
                            .flat_map(|i| i.specifiers.iter().map(|s| s.local.clone())),
                    )
                    .chain(JS_GLOBALS.iter().map(|g| g.to_string()))
                    .collect(),
                module_constants: self.extraction.module_constants.clone(),
            },
            target: TargetInfo {
                suggested_path: format!("src/components/{}.sigil", to_snake_case(&comp.name)),
                pattern: target_pattern,
            },
            recommendations,
            patterns,
            ambiguities,
            dependencies,
            complexity,
            complexity_factors: factors,
            status: MigrationStatus::Pending,
        }
    }

    fn generate_recommendations(&self, comp: &ComponentExtraction) -> Recommendations {
        let state_fields = self.recommend_state_fields(comp);
        let messages = self.recommend_messages(comp);
        let effects = self.recommend_effects(comp);
        let props_handling = self.recommend_props(comp);

        Recommendations {
            state_fields,
            messages,
            effects,
            props_handling,
        }
    }

    fn recommend_state_fields(&self, comp: &ComponentExtraction) -> Vec<StateFieldRecommendation> {
        let mut fields = Vec::new();

        for hook in &comp.hooks {
            if hook.hook_type == HookType::UseState {
                if let Some(state_name) = &hook.state_name {
                    let initial = hook.initial_value.as_ref()
                        .map(|s| s.as_str())
                        .unwrap_or("∅");

                    let (field_type, evidentiality) = infer_type_from_value(initial);

                    fields.push(StateFieldRecommendation {
                        from_hook: format!("useState:{}", state_name),
                        to_field: state_name.clone(),
                        field_type,
                        evidentiality,
                        initial_value: initial.to_string(),
                        reasoning: format!("useState hook '{}' maps to actor state field", state_name),
                    });
                }
            } else if hook.hook_type == HookType::UseRef {
                if let Some(ref_name) = &hook.ref_name {
                    let ref_type = hook.ref_type.as_ref()
                        .map(|t| format!("Option<{}>", map_ts_type_to_sigil(t)))
                        .unwrap_or("Option<Element>".to_string());

                    fields.push(StateFieldRecommendation {
                        from_hook: format!("useRef:{}", ref_name),
                        to_field: ref_name.clone(),
                        field_type: ref_type,
                        evidentiality: "!".to_string(),
                        initial_value: "∅".to_string(),
                        reasoning: format!("useRef '{}' becomes non-reactive state (no re-render on change)", ref_name),
                    });
                }
            }
        }

        fields
    }

    fn recommend_messages(&self, comp: &ComponentExtraction) -> Vec<MessageRecommendation> {
        let mut messages = Vec::new();

        // Build state_fields mapping for transformation: (setter_name, field_name)
        let state_fields: Vec<(String, String)> = comp.hooks.iter()
            .filter(|h| h.hook_type == HookType::UseState)
            .filter_map(|h| {
                if let (Some(state_name), Some(setter_name)) = (&h.state_name, &h.setter_name) {
                    Some((setter_name.clone(), state_name.clone()))
                } else {
                    None
                }
            })
            .collect();

        // Generate messages from useState setters
        for hook in &comp.hooks {
            if hook.hook_type == HookType::UseState {
                if let (Some(state_name), Some(setter_name)) = (&hook.state_name, &hook.setter_name) {
                    // Common patterns: setCount -> Increment/Decrement, setVisible -> Toggle/Show/Hide
                    let msg_name = derive_message_name(state_name, setter_name);

                    // The body this generates is `self.<field> = msg.0`, so the
                    // variant has to declare the value it is reading. It used to be
                    // payload-less, which made every setter handler in the UI read a
                    // field off a message that had none.
                    messages.push(MessageRecommendation {
                        name: msg_name.clone(),
                        from_handler: setter_name.clone(),
                        payload: Some("(Any)".to_string()),
                        state_changes: vec![format!(
                            "self.{} = /* new value */",
                            to_snake_case(state_name)
                        )],
                        side_effects: vec![],
                        service_calls: vec![],
                        param_bindings: vec![],
                        state_assignments: vec![],
                        flattened_control_flow: false,
                    });
                }
            }
        }

        // Every JSX event in the component, with the message name the generator
        // will dispatch for it. Handlers are matched against this: a function
        // bound to an event is a handler, one that is not is a local helper.
        let jsx_events = collect_jsx_event_messages(&comp.jsx);
        let jsx_by_callback: Vec<(String, String, String)> = jsx_events
            .iter()
            .filter_map(|(msg, code)| {
                derive_invoked_callback(code).map(|cb| (cb, msg.clone(), code.clone()))
            })
            .collect();

        // Generate messages from event handlers
        for handler in &comp.handlers {
            // `comp.handlers` now holds every function-valued binding in the
            // component body, not only `handle*`/`on*` ones — that convention was
            // costing us the 66 plain-named handlers (`save`, `startNew`,
            // `copyBody`) that the Lares UI actually uses. Decide here which are
            // handlers: bound to a JSX event, or conventionally named. Anything
            // else is a helper the view calls directly and must not become a
            // message, or the enum fills with variants nothing dispatches.
            let jsx_hit = jsx_by_callback.iter().find(|(cb, _, _)| *cb == handler.name);
            let (msg_name, payload) = match jsx_hit {
                Some((_, msg, code)) => {
                    // Arity comes from the call site, exactly as the inline path
                    // below derives it, so declaration and dispatch agree.
                    let arity = derive_event_message_payload(code).len();
                    let payload = if arity == 0 {
                        None
                    } else {
                        Some(format!("({})", vec!["Any"; arity].join(", ")))
                    };
                    (msg.clone(), payload)
                }
                None if handler.name.starts_with("handle") || handler.name.starts_with("on") => {
                    (to_pascal_case(&handler.name.replace("handle", "")), None)
                }
                None => continue,
            };

            if messages.iter().any(|m| m.name == msg_name) {
                continue;
            }

            // Transform React state mutations to Sigil syntax
            let transformed_state_changes = transform_state_mutations_to_sigil(
                &handler.state_mutations,
                &state_fields,
            );
            let state_assignments =
                extract_state_assignments(&handler.state_mutations, &state_fields);
            let flattened_control_flow = !state_assignments.is_empty()
                && (handler.has_conditionals || handler.has_early_return);

            // Extract service calls from handler.calls (hook-returned functions)
            let service_calls = extract_service_calls(&handler.calls);

            // Only as many parameters as the payload actually carries; the rest
            // would bind `msg.N` for an N the variant does not have.
            let arity = payload
                .as_deref()
                .filter(|p| p.starts_with('('))
                .map(|p| p.trim_start_matches('(').trim_end_matches(')').split(',').count())
                .unwrap_or(0);
            let param_bindings: Vec<String> = handler
                .parameters
                .iter()
                .take(arity)
                .map(|p| p.name.clone())
                .collect();

            messages.push(MessageRecommendation {
                name: msg_name,
                from_handler: handler.name.clone(),
                payload,
                state_changes: transformed_state_changes,
                side_effects: handler.api_calls.clone(),
                service_calls,
                param_bindings,
                state_assignments,
                flattened_control_flow,
            });
        }

        // Inline JSX handlers (`onClick={() => onExpand(id)}`) are not in
        // comp.handlers, which only holds named handler functions. Without these the
        // generated view dispatched messages that the actor's enum never declared.
        for (msg_name, code) in jsx_events {
            if messages.iter().any(|m| m.name == msg_name) {
                continue;
            }
            // Carry the handler's arguments through as a tuple payload, so
            // `onExpand(slot.id)` declares `Expand(Any)` rather than throwing the
            // id away and leaving the actor unable to tell which row was clicked.
            let args = derive_event_message_payload(&code);

            // An inline handler that calls a useState setter is a state change, and
            // we know exactly which field: `onClick={() => setOpen(true)}` should
            // write `self.open = msg.0`, not leave a TODO next to a live message.
            let callback = derive_invoked_callback(&code);
            let state_changes = callback
                .as_ref()
                .and_then(|cb| state_fields.iter().find(|(setter, _)| setter == cb))
                .map(|(_, field)| {
                    vec![format!("self.{} = /* new value */", to_snake_case(field))]
                })
                .unwrap_or_default();

            // A body reading msg.0 needs a payload even when the call site had no
            // argument we could safely reproduce.
            let arity = if state_changes.is_empty() {
                args.len()
            } else {
                args.len().max(1)
            };
            let payload = if arity == 0 {
                None
            } else {
                Some(format!("({})", vec!["Any"; arity].join(", ")))
            };
            messages.push(MessageRecommendation {
                name: msg_name,
                from_handler: code,
                payload,
                state_changes,
                side_effects: vec![],
                service_calls: vec![],
                param_bindings: vec![],
                state_assignments: vec![],
                flattened_control_flow: false,
            });
        }

        // Invariant: a handler body that reads `msg.0` must belong to a variant that
        // declares one. The generator substitutes `/* new value */` with `msg.0`, so
        // that placeholder is the marker.
        for msg in messages.iter_mut() {
            if msg.payload.is_none()
                && msg.state_changes.iter().any(|c| c.contains("/* new value */"))
            {
                msg.payload = Some("(Any)".to_string());
            }
        }

        messages
    }

    fn recommend_effects(&self, comp: &ComponentExtraction) -> Vec<EffectRecommendation> {
        let mut effects = Vec::new();

        for hook in &comp.hooks {
            if hook.hook_type == HookType::UseEffect || hook.hook_type == HookType::UseLayoutEffect {
                let (strategy, lifecycle_event, inline_in, reasoning) = match &hook.dependencies {
                    Some(deps) if deps.is_empty() => {
                        // Empty deps = mount only
                        if hook.has_cleanup {
                            (EffectStrategy::Lifecycle, Some("Mount".to_string()), None,
                             "Empty deps with cleanup → Mount/Unmount lifecycle".to_string())
                        } else {
                            (EffectStrategy::Lifecycle, Some("Mount".to_string()), None,
                             "Empty deps → Mount lifecycle".to_string())
                        }
                    }
                    Some(deps) if !deps.is_empty() => {
                        // Has deps = inline in message handlers that change those deps
                        let deps_str = deps.join(", ");
                        (EffectStrategy::Inline, None, Some(deps.join("_")),
                         format!("Deps [{}] → inline in handlers that change these", deps_str))
                    }
                    None => {
                        // No deps = runs every render → usually should be removed or made explicit
                        (EffectStrategy::Remove, None, None,
                         "No dependency array → runs every render, usually not needed in actors".to_string())
                    }
                    _ => (EffectStrategy::Message, None, None, "Convert to explicit message".to_string()),
                };

                effects.push(EffectRecommendation {
                    from_hook: format!("useEffect[{}]",
                        hook.dependencies.as_ref()
                            .map(|d| d.join(","))
                            .unwrap_or("none".to_string())),
                    strategy,
                    reasoning,
                    inline_in,
                    lifecycle_event,
                });
            } else if hook.hook_type == HookType::UseCallback || hook.hook_type == HookType::UseMemo {
                effects.push(EffectRecommendation {
                    from_hook: format!("{:?}", hook.hook_type),
                    strategy: EffectStrategy::Remove,
                    reasoning: "useCallback/useMemo not needed in actors - no re-render optimization needed".to_string(),
                    inline_in: None,
                    lifecycle_event: None,
                });
            }
        }

        effects
    }

    fn recommend_props(&self, comp: &ComponentExtraction) -> PropsRecommendation {
        if comp.props.is_empty() {
            return PropsRecommendation {
                strategy: PropsStrategy::None,
                fields: vec![],
            };
        }

        let fields: Vec<PropsField> = comp.props.iter().map(|prop| {
            PropsField {
                name: to_snake_case(&prop.name),
                field_type: prop.type_annotation.as_ref()
                    .map(|t| map_ts_type_to_sigil(t))
                    .unwrap_or("Any".to_string()),
                from_prop: prop.name.clone(),
            }
        }).collect();

        PropsRecommendation {
            strategy: PropsStrategy::Constructor,
            fields,
        }
    }

    fn select_patterns(&self, comp: &ComponentExtraction) -> Vec<PatternExample> {
        let mut patterns = Vec::new();
        let library = pattern_library();

        // Select patterns based on what's in the component
        for hook in &comp.hooks {
            match hook.hook_type {
                HookType::UseState => {
                    if let Some(p) = library.iter().find(|p| p.name == "useState_to_state") {
                        if !patterns.iter().any(|x: &PatternExample| x.name == p.name) {
                            patterns.push(p.clone());
                        }
                    }
                }
                HookType::UseEffect => {
                    let pattern_name = if hook.dependencies.as_ref().map(|d| d.is_empty()).unwrap_or(false) {
                        "useEffect_mount"
                    } else {
                        "useEffect_deps"
                    };
                    if let Some(p) = library.iter().find(|p| p.name == pattern_name) {
                        if !patterns.iter().any(|x: &PatternExample| x.name == p.name) {
                            patterns.push(p.clone());
                        }
                    }
                }
                HookType::UseCallback => {
                    if let Some(p) = library.iter().find(|p| p.name == "useCallback_remove") {
                        if !patterns.iter().any(|x: &PatternExample| x.name == p.name) {
                            patterns.push(p.clone());
                        }
                    }
                }
                HookType::UseRef => {
                    if let Some(p) = library.iter().find(|p| p.name == "useRef_to_state") {
                        if !patterns.iter().any(|x: &PatternExample| x.name == p.name) {
                            patterns.push(p.clone());
                        }
                    }
                }
                _ => {}
            }
        }

        // Check JSX for patterns
        if comp.jsx.root.is_some() {
            if let Some(p) = library.iter().find(|p| p.name == "jsx_to_builder") {
                if !patterns.iter().any(|x: &PatternExample| x.name == p.name) {
                    patterns.push(p.clone());
                }
            }
        }

        // Check for event handlers
        if !comp.handlers.is_empty() || has_event_handlers(&comp.jsx) {
            if let Some(p) = library.iter().find(|p| p.name == "onClick_to_message") {
                if !patterns.iter().any(|x: &PatternExample| x.name == p.name) {
                    patterns.push(p.clone());
                }
            }
        }

        patterns
    }

    fn detect_ambiguities(&self, comp: &ComponentExtraction) -> Vec<Ambiguity> {
        let mut ambiguities = Vec::new();

        // Check for effects with deps that might have multiple placement options
        for (idx, hook) in comp.hooks.iter().enumerate() {
            if hook.hook_type == HookType::UseEffect {
                if let Some(deps) = &hook.dependencies {
                    if !deps.is_empty() {
                        ambiguities.push(Ambiguity {
                            id: format!("effect_{}", idx),
                            category: AmbiguityCategory::EffectPlacement,
                            question: format!(
                                "Where should the effect with deps [{}] be placed?",
                                deps.join(", ")
                            ),
                            options: vec![
                                AmbiguityOption {
                                    label: "Inline in handlers".to_string(),
                                    description: "Add effect logic to message handlers that change the dependencies".to_string(),
                                    recommended: true,
                                },
                                AmbiguityOption {
                                    label: "Separate message".to_string(),
                                    description: "Create a dedicated message for the effect logic".to_string(),
                                    recommended: false,
                                },
                            ],
                            default_choice: 0,
                        });
                    }
                }
            }
        }

        // Check for callback props that are passed down
        for prop in &comp.props {
            if prop.is_callback {
                ambiguities.push(Ambiguity {
                    id: format!("callback_prop_{}", prop.name),
                    category: AmbiguityCategory::EventMapping,
                    question: format!("How should callback prop '{}' be handled?", prop.name),
                    options: vec![
                        AmbiguityOption {
                            label: "Message dispatch".to_string(),
                            description: "Parent passes a message ID, child dispatches to parent".to_string(),
                            recommended: true,
                        },
                        AmbiguityOption {
                            label: "Actor reference".to_string(),
                            description: "Parent passes a reference, child sends messages to parent actor".to_string(),
                            recommended: false,
                        },
                    ],
                    default_choice: 0,
                });
            }
        }

        ambiguities
    }

    fn calculate_complexity(&self, comp: &ComponentExtraction) -> (Complexity, Vec<String>) {
        let mut factors = Vec::new();
        let mut score = 0;

        // State complexity
        let state_count = comp.hooks.iter()
            .filter(|h| h.hook_type == HookType::UseState)
            .count();
        if state_count > 5 {
            factors.push(format!("{} state variables", state_count));
            score += 2;
        } else if state_count > 2 {
            score += 1;
        }

        // Effect complexity
        let effect_count = comp.hooks.iter()
            .filter(|h| h.hook_type == HookType::UseEffect)
            .count();
        if effect_count > 3 {
            factors.push(format!("{} useEffect hooks", effect_count));
            score += 2;
        } else if effect_count > 0 {
            score += 1;
        }

        // Handler complexity
        if comp.handlers.len() > 5 {
            factors.push(format!("{} event handlers", comp.handlers.len()));
            score += 1;
        }

        // Props complexity
        if comp.props.len() > 10 {
            factors.push(format!("{} props", comp.props.len()));
            score += 1;
        }

        // Class component complexity
        if comp.component_type == ComponentType::Class {
            factors.push("Class component (lifecycle methods)".to_string());
            score += 2;
        }

        let complexity = if score >= 4 {
            Complexity::Complex
        } else if score >= 2 {
            Complexity::Moderate
        } else {
            Complexity::Simple
        };

        (complexity, factors)
    }

    fn extract_dependencies(&self, comp: &ComponentExtraction) -> Dependencies {
        let components: Vec<String> = comp.child_components.clone();

        let types: Vec<String> = if let Some(props_type) = &comp.props_type {
            vec![props_type.clone()]
        } else {
            vec![]
        };

        Dependencies { components, types }
    }

    /// Collect service actors from custom hooks across all components.
    fn collect_service_actors(&self, components: &[ComponentMigrationSpec]) -> Vec<ServiceActorSpec> {
        use std::collections::HashMap;

        // Map: hook_name -> (service_name, state_fields, messages, components_using)
        let mut service_map: HashMap<String, ServiceActorBuilder> = HashMap::new();

        // Iterate through all components and their custom hooks
        for comp in &self.extraction.components {
            for hook in &comp.custom_hooks {
                // Skip Zustand stores (handled differently)
                if hook.is_zustand {
                    continue;
                }

                let service_name = hook_name_to_service(&hook.name);
                let entry = service_map.entry(hook.name.clone()).or_insert_with(|| {
                    ServiceActorBuilder {
                        name: service_name,
                        derived_from: hook.name.clone(),
                        state_fields: HashMap::new(),
                        messages: HashMap::new(),
                        used_by: Vec::new(),
                    }
                });

                // Track which components use this service
                if !entry.used_by.contains(&comp.name) {
                    entry.used_by.push(comp.name.clone());
                }

                // Collect state fields from non-function return values
                for ret in &hook.returned_values {
                    if !ret.is_function {
                        let field_name = to_snake_case(&ret.name);
                        entry.state_fields.entry(field_name.clone()).or_insert_with(|| {
                            ServiceStateField {
                                name: field_name,
                                original_name: ret.name.clone(),
                                field_type: infer_type_from_name(&ret.name),
                                is_observable: true,
                            }
                        });
                    }
                }

                // Collect messages from function return values
                for ret in &hook.returned_values {
                    if ret.is_function {
                        let msg_name = to_pascal_case(&ret.name);
                        entry.messages.entry(msg_name.clone()).or_insert_with(|| {
                            ServiceMessage {
                                name: msg_name,
                                original_name: ret.name.clone(),
                                parameters: vec![], // Will be populated from call sites
                            }
                        });
                    }
                }
            }

            // Also look at handler calls to infer message parameters
            for handler in &comp.handlers {
                for call in &handler.calls {
                    if let CallSource::Hook { hook_name } = &call.source {
                        if let Some(entry) = service_map.get_mut(hook_name) {
                            let msg_name = to_pascal_case(&call.name);
                            if let Some(msg) = entry.messages.get_mut(&msg_name) {
                                // Merge parameters from call site
                                for arg in &call.arguments {
                                    if !msg.parameters.contains(arg) {
                                        msg.parameters.push(arg.clone());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Convert to final ServiceActorSpec
        service_map.into_values()
            .map(|builder| ServiceActorSpec {
                name: builder.name,
                derived_from: builder.derived_from,
                state_fields: builder.state_fields.into_values().collect(),
                messages: builder.messages.into_values().collect(),
                used_by: builder.used_by,
            })
            .collect()
    }

    fn generate_type_spec(&self, type_ext: &TypeExtraction) -> TypeMigrationSpec {
        let sigil_type = convert_ts_type_to_sigil(type_ext);

        // Convert extracted fields to spec format
        let fields: Vec<TypeFieldSpec> = type_ext.fields.iter()
            .map(|f| TypeFieldSpec {
                name: f.name.clone(),
                type_annotation: f.type_annotation.clone(),
                optional: f.optional,
                readonly: f.readonly,
                type_kind: format_type_kind(&f.type_kind),
            })
            .collect();

        // Convert type params
        let type_params: Vec<String> = type_ext.type_params.iter()
            .map(|p| {
                let mut s = p.name.clone();
                if let Some(constraint) = &p.constraint {
                    s.push_str(&format!(" extends {}", constraint));
                }
                if let Some(default) = &p.default {
                    s.push_str(&format!(" = {}", default));
                }
                s
            })
            .collect();

        TypeMigrationSpec {
            id: format!("{}:{}", self.extraction.file.relative_path, type_ext.name),
            name: type_ext.name.clone(),
            source: type_ext.definition.clone(),
            target: sigil_type,
            manual_review_needed: false,
            notes: vec![],
            fields,
            type_params,
            extends: type_ext.extends.clone(),
            union_variants: type_ext.union_variants.clone(),
        }
    }
}

/// Format TypeFieldKind as a simple string for JSON output
fn format_type_kind(kind: &TypeFieldKind) -> String {
    match kind {
        TypeFieldKind::Primitive { name } => format!("primitive:{}", name),
        TypeFieldKind::TypeRef { name, type_args } => {
            if type_args.is_empty() {
                format!("ref:{}", name)
            } else {
                format!("ref:{}<{}>", name, type_args.join(", "))
            }
        }
        TypeFieldKind::Array { element_type } => format!("array:{}", element_type),
        TypeFieldKind::Union { variants } => format!("union:[{}]", variants.join(" | ")),
        TypeFieldKind::Function { params, return_type } => {
            let params_str: Vec<String> = params.iter()
                .map(|p| {
                    let name = p.name.as_deref().unwrap_or("_");
                    let opt = if p.optional { "?" } else { "" };
                    format!("{}{}: {}", name, opt, p.type_annotation)
                })
                .collect();
            format!("fn:({}) => {}", params_str.join(", "), return_type)
        }
        TypeFieldKind::Record { key_type, value_type } => {
            format!("record:<{}, {}>", key_type, value_type)
        }
        TypeFieldKind::Tuple { element_types } => {
            format!("tuple:[{}]", element_types.join(", "))
        }
        TypeFieldKind::Literal { value } => format!("literal:{}", value),
        TypeFieldKind::Complex { raw } => format!("complex:{}", raw),
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Generate current UTC timestamp in ISO 8601 format.
pub fn chrono_now() -> String {
    // Use std::time for UTC timestamp
    use std::time::{SystemTime, UNIX_EPOCH};

    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    let secs = now.as_secs();

    // Calculate UTC datetime components
    let days_since_epoch = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Simple year/month/day calculation (doesn't handle leap years perfectly but close enough)
    let mut year = 1970;
    let mut remaining_days = days_since_epoch as i64;

    loop {
        let days_in_year = if year % 4 == 0 && (year % 100 != 0 || year % 400 == 0) { 366 } else { 365 };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        year += 1;
    }

    let is_leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days_in_months = if is_leap {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1;
    for days in days_in_months.iter() {
        if remaining_days < *days {
            break;
        }
        remaining_days -= *days;
        month += 1;
    }

    let day = remaining_days + 1;

    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", year, month, day, hours, minutes, seconds)
}

/// The one definition. `generator.rs` and `ast_transform.rs` delegate here;
/// they used to carry byte-identical copies, which is how a fix to one of them
/// could leave the other two mangling the same names.
/// snake_case WITHOUT keyword escaping.
///
/// A member or method name is not a binding, so it cannot collide with a
/// keyword and must not be escaped: `selected.body` became `selected.body_`,
/// `hunk.header` became `header_`, and `s.split(",")` became `s·split_(",")`,
/// which is a method Sigil does not have. Escaping belongs on names the
/// generated file *declares*.
pub(crate) fn to_snake_case_member(s: &str) -> String {
    let mut result = String::new();
    let chars: Vec<char> = s.chars().collect();
    for (i, c) in chars.iter().enumerate() {
        if c.is_uppercase() {
            let prev = if i > 0 { Some(chars[i - 1]) } else { None };
            let next = chars.get(i + 1).copied();
            let boundary = match prev {
                None => false,
                Some(p) if p.is_lowercase() || p.is_numeric() => true,
                Some(p) if p.is_uppercase() => next.is_some_and(|n| n.is_lowercase()),
                _ => false,
            };
            if boundary {
                result.push('_');
            }
            result.push(c.to_lowercase().next().unwrap());
        } else {
            result.push(*c);
        }
    }
    result
}

pub(crate) fn to_snake_case(s: &str) -> String {
    // Acronym-aware: a run of capitals is one word. Underscoring before every
    // capital turned `DEFAULT_SORT` into `d_e_f_a_u_l_t__s_o_r_t`, which is what
    // 16 module-level constants were referred to as across the generated client.
    let chars: Vec<char> = s.chars().collect();
    let mut result = String::new();
    for (i, c) in chars.iter().enumerate() {
        if c.is_uppercase() {
            let prev = if i > 0 { Some(chars[i - 1]) } else { None };
            let next = chars.get(i + 1).copied();
            let boundary = match prev {
                None => false,
                Some(p) if p.is_lowercase() || p.is_numeric() => true,
                // `URLPath` -> `url_path`: the last capital of a run starts a word.
                Some(p) if p.is_uppercase() => next.is_some_and(|n| n.is_lowercase()),
                _ => false,
            };
            if boundary {
                result.push('_');
            }
            result.push(c.to_lowercase().next().unwrap());
        } else {
            result.push(*c);
        }
    }
    // A prop named `ref` or `type` is a parse error, not a type error.
    escape_sigil_keyword(&result)
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

fn infer_type_from_value(value: &str) -> (String, String) {
    // Enhanced type inference from initial value
    let trimmed = value.trim();

    // Integer literals
    if trimmed == "0" || (trimmed.starts_with(|c: char| c.is_ascii_digit() || c == '-')
        && !trimmed.contains('.') && trimmed.parse::<i64>().is_ok()) {
        return ("i64".to_string(), "!".to_string());
    }

    // Float literals
    if trimmed == "0.0" || (trimmed.contains('.') && trimmed.parse::<f64>().is_ok()) {
        return ("f64".to_string(), "!".to_string());
    }

    // Boolean literals
    if trimmed == "true" || trimmed == "false" {
        return ("bool".to_string(), "!".to_string());
    }

    // String literals
    if (trimmed.starts_with('"') && trimmed.ends_with('"')) ||
       (trimmed.starts_with('\'') && trimmed.ends_with('\'')) ||
       (trimmed.starts_with('`') && trimmed.ends_with('`')) {
        return ("String".to_string(), "!".to_string());
    }

    // Null/undefined
    if trimmed == "null" || trimmed == "undefined" || trimmed == "∅" {
        return ("Option<Any>".to_string(), "~".to_string());
    }

    // Empty array
    if trimmed == "[]" {
        return ("Vec<Any>".to_string(), "!".to_string());
    }

    // Empty object
    if trimmed == "{}" {
        return ("Map<String, Any>".to_string(), "!".to_string());
    }

    // Array with elements - try to infer element type
    if trimmed.starts_with('[') && trimmed.ends_with(']') {
        let inner = trimmed[1..trimmed.len()-1].trim();
        if !inner.is_empty() {
            // Get first element
            let first = inner.split(',').next().unwrap_or("").trim();
            if !first.is_empty() {
                let (elem_type, _) = infer_type_from_value(first);
                return (format!("Vec<{}>", elem_type), "!".to_string());
            }
        }
        return ("Vec<Any>".to_string(), "!".to_string());
    }

    // Function call results - often need inference from context
    if trimmed.contains('(') && trimmed.ends_with(')') {
        // Common patterns
        if trimmed.starts_with("Date.now") || trimmed.contains("getTime") {
            return ("i64".to_string(), "!".to_string());
        }
        if trimmed.starts_with("new Date") {
            return ("DateTime".to_string(), "!".to_string());
        }
        if trimmed.starts_with("new Map") || trimmed.starts_with("new Set") {
            return ("Map<String, Any>".to_string(), "!".to_string());
        }
        // Unknown function - uncertain type
        return ("Any".to_string(), "~".to_string());
    }

    // Property access - often prop values, uncertain
    if trimmed.contains('.') {
        // Check for common suffixes
        if trimmed.ends_with(".length") {
            return ("i64".to_string(), "!".to_string());
        }
        return ("Any".to_string(), "~".to_string());
    }

    // Default: uncertain
    ("Any".to_string(), "~".to_string())
}

fn map_ts_type_to_sigil(ts_type: &str) -> String {
    // Strip TypeScript modifiers that have no Sigil equivalent. `readonly Foo[]`
    // otherwise became `Vec<readonly Foo>`, which does not parse.
    let t = ts_type
        .trim()
        .trim_start_matches("readonly ")
        .trim_start_matches("const ")
        .trim();
    match t {
        "number" => return "f64".to_string(),
        "string" => return "String".to_string(),
        "boolean" => return "bool".to_string(),
        "void" => return "()".to_string(),
        "null" | "undefined" => return "∅".to_string(),
        "any" | "unknown" => return "Any".to_string(),
        "HTMLInputElement" | "HTMLElement" | "Element" => return "Element".to_string(),
        _ => {}
    }

    // A function type — `(c: boolean) => void`, `() => void`. Callbacks become
    // messages in Qliphoth, so the prop itself is opaque.
    if t.contains("=>") {
        return "Any".to_string();
    }

    // A union. `T | null` and `T | undefined` are optionality, which Sigil spells
    // Option<T>. Any other union has no Sigil equivalent, so it degrades to Any
    // rather than emitting `A | B`, which does not parse.
    if t.contains('|') {
        let parts: Vec<&str> = t.split('|').map(|p| p.trim()).collect();
        let non_null: Vec<&str> = parts
            .iter()
            .copied()
            .filter(|p| *p != "null" && *p != "undefined")
            .collect();
        if non_null.len() == 1 && non_null.len() < parts.len() {
            return format!("Option<{}>", map_ts_type_to_sigil(non_null[0]));
        }
        return "Any".to_string();
    }

    // `T[]` array shorthand.
    if let Some(inner) = t.strip_suffix("[]") {
        return format!("Vec<{}>", map_ts_type_to_sigil(inner));
    }

    if let Some(inner) = t.strip_prefix("Array<").and_then(|x| x.strip_suffix('>')) {
        return format!("Vec<{}>", map_ts_type_to_sigil(inner));
    }

    // An inline object type has no name to refer to; keep it opaque rather than
    // emitting TypeScript into a Sigil signature.
    if t.starts_with('{') {
        return "Any".to_string();
    }

    // A generic application — `Record<string, DirState>`, `Readonly<Record<…>>`.
    // The ARGUMENTS have to be mapped as well. Sigil accepts a lowercase name as a
    // standalone type but rejects one as a type argument, so a bare `string` parsed
    // while `Record<string, X>` did not, and the whole file failed on it.
    if let Some((head, args)) = split_generic(t) {
        let mapped: Vec<String> = args.iter().map(|a| map_ts_type_to_sigil(a)).collect();
        return match head {
            // Erasable TS wrappers with no Sigil counterpart.
            "Readonly" | "Partial" | "Required" | "NonNullable" if mapped.len() == 1 => {
                mapped.into_iter().next().unwrap()
            }
            "Record" if mapped.len() == 2 => format!("Map<{}>", mapped.join(", ")),
            "ReadonlyArray" if mapped.len() == 1 => format!("Vec<{}>", mapped[0]),
            h if !h.is_empty()
                && h.chars().all(|c| c.is_alphanumeric() || c == '_') =>
            {
                format!("{}<{}>", h, mapped.join(", "))
            }
            _ => "Any".to_string(),
        };
    }

    // Custom named types pass through — but only if they are actually spellable in
    // Sigil. Anything still carrying TypeScript syntax (a colon, an arrow, a union
    // bar, a brace or paren, a leftover modifier) would be emitted verbatim into a
    // signature and fail to parse, so it degrades to Any instead. Being wrong in the
    // direction of Any costs type information; being wrong the other way costs a
    // file that will not compile at all.
    let spellable = t
        .chars()
        .all(|c| c.is_alphanumeric() || matches!(c, '_' | '<' | '>' | ',' | ' '))
        && !t.contains("=>");
    if spellable && !t.is_empty() {
        t.to_string()
    } else {
        "Any".to_string()
    }
}

/// Sigil's reserved words, taken from the lexer's `#[token("…")]` set.
///
/// React prop names collide with these more often than you would guess — `ref`,
/// `type`, `on`, `body`, `from`, `to`, `scope`, `location`. A prop named `ref`
/// emitted `rite new(ref: Any)`, which is a parse error, not a type error.
const SIGIL_KEYWORDS: &[&str] = &[
    "actor", "affine", "alter", "amqp", "anima", "as", "asm", "aspect", "async", "atomic",
    "await", "body", "broadcast", "close", "cocon", "connect", "consensus", "const",
    "derive", "distribute", "dyn", "each", "extern", "false", "forever", "from", "gather",
    "gpu", "graphql", "grpc", "header", "headspace", "http", "https", "interfere", "invoke",
    "kafka", "layer", "legion_field", "linear", "location", "loop", "macro", "macro_rules",
    "move", "mut", "naked", "nay", "no_grad", "null", "of", "on", "packed", "parallel",
    "reality", "recv", "ref", "relevant", "retry", "rite", "rune", "saga", "scope",
    "scroll", "self", "send", "sigil", "simd", "split", "states", "static", "stream", "super",
    "switch", "this", "timeout", "to", "tome", "trigger", "true", "type", "unsafe", "vary",
    "volatile", "where", "ws", "wss", "yay", "yea", "yield",
];

/// Make an identifier safe to emit. Applied inside every `to_snake_case`, so a field
/// declared `ref_` is also referenced as `ref_` — escaping one and not the other
/// would be worse than not escaping at all.
pub fn escape_sigil_keyword(name: &str) -> String {
    if SIGIL_KEYWORDS.contains(&name) {
        format!("{}_", name)
    } else {
        name.to_string()
    }
}

/// Split `Head<A, B>` into ("Head", ["A", "B"]). Returns None when the type is not a
/// generic application. Commas inside nested arguments do not split.
fn split_generic(t: &str) -> Option<(&str, Vec<String>)> {
    let open = t.find('<')?;
    if !t.ends_with('>') {
        return None;
    }
    let head = t[..open].trim();
    let inner = &t[open + 1..t.len() - 1];
    let mut args = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    for c in inner.chars() {
        match c {
            '<' => { depth += 1; cur.push(c); }
            '>' => { depth -= 1; cur.push(c); }
            ',' if depth == 0 => { args.push(cur.trim().to_string()); cur.clear(); }
            _ => cur.push(c),
        }
    }
    if !cur.trim().is_empty() {
        args.push(cur.trim().to_string());
    }
    if args.is_empty() { None } else { Some((head, args)) }
}

fn derive_message_name(state_name: &str, setter_name: &str) -> String {
    // setCount -> Update_count, or more specific based on common patterns
    if setter_name.starts_with("set") {
        format!("Update{}", to_pascal_case(&setter_name[3..]))
    } else {
        to_pascal_case(setter_name)
    }
}

fn has_event_handlers(jsx: &JsxTree) -> bool {
    fn check_node(node: &JsxNode) -> bool {
        match &node.node_type {
            JsxNodeType::Element { attributes, children, .. } => {
                if attributes.iter().any(|a| a.is_event_handler) {
                    return true;
                }
                children.iter().any(check_node)
            }
            JsxNodeType::Fragment { children } => children.iter().any(check_node),
            _ => false,
        }
    }

    jsx.root.as_ref().map(check_node).unwrap_or(false)
}

fn convert_ts_type_to_sigil(type_ext: &TypeExtraction) -> String {
    match type_ext.kind {
        TypeKind::Interface => {
            // Generate Σ struct with all fields
            let type_params = if type_ext.type_params.is_empty() {
                String::new()
            } else {
                format!("<{}>", type_ext.type_params.iter()
                    .map(|p| p.name.clone())
                    .collect::<Vec<_>>()
                    .join(", "))
            };

            if type_ext.fields.is_empty() {
                format!("Σ {}{} {{ }}", type_ext.name, type_params)
            } else {
                let fields: Vec<String> = type_ext.fields.iter()
                    .map(|f| {
                        let sigil_type = ts_type_to_sigil_type(&f.type_annotation, &f.type_kind);
                        let opt = if f.optional { "?" } else { "" };
                        format!("    {}{}: {}", f.name, opt, sigil_type)
                    })
                    .collect();
                format!("Σ {}{} {{\n{}\n}}", type_ext.name, type_params, fields.join(",\n"))
            }
        }
        TypeKind::TypeAlias => {
            // Handle union types specially
            if !type_ext.union_variants.is_empty() {
                let variants: Vec<String> = type_ext.union_variants.iter()
                    .map(|v| ts_type_to_sigil_type(v, &TypeFieldKind::Complex { raw: v.clone() }))
                    .collect();
                format!("type {} = {}", type_ext.name, variants.join(" | "))
            } else if !type_ext.fields.is_empty() {
                // Object type literal
                let fields: Vec<String> = type_ext.fields.iter()
                    .map(|f| {
                        let sigil_type = ts_type_to_sigil_type(&f.type_annotation, &f.type_kind);
                        let opt = if f.optional { "?" } else { "" };
                        format!("    {}{}: {}", f.name, opt, sigil_type)
                    })
                    .collect();
                format!("Σ {} {{\n{}\n}}", type_ext.name, fields.join(",\n"))
            } else {
                format!("type {} = /* TODO: map {} */", type_ext.name, type_ext.definition)
            }
        }
        TypeKind::Enum => {
            // Generate enum with variants
            if type_ext.union_variants.is_empty() {
                format!("ᛈ {} {{ }}", type_ext.name)
            } else {
                let variants: Vec<String> = type_ext.union_variants.iter()
                    .map(|v| format!("    {}", v))
                    .collect();
                format!("ᛈ {} {{\n{}\n}}", type_ext.name, variants.join(",\n"))
            }
        }
    }
}

/// Convert a TypeScript type annotation to Sigil type
fn ts_type_to_sigil_type(annotation: &str, kind: &TypeFieldKind) -> String {
    match kind {
        TypeFieldKind::Primitive { name } => match name.as_str() {
            "string" => "String".to_string(),
            "number" => "f64".to_string(),  // Or i64 depending on context
            "boolean" => "bool".to_string(),
            "null" | "undefined" => "∅".to_string(),
            "void" => "()".to_string(),
            "any" | "unknown" => "Any".to_string(),
            "never" => "!".to_string(),
            "object" => "Object".to_string(),
            "bigint" => "i128".to_string(),
            "symbol" => "Symbol".to_string(),
            _ => annotation.to_string(),
        },
        TypeFieldKind::TypeRef { name, type_args } => {
            // Map common React types
            let base = match name.as_str() {
                "ReactNode" | "React.ReactNode" => "VNode",
                "ReactElement" | "React.ReactElement" => "VNode",
                "JSX.Element" => "VNode",
                "HTMLElement" | "Element" => "DomRef",
                "CSSProperties" => "StyleMap",
                "Ref" => "Ref",
                "RefObject" => "Ref",
                "Promise" => "Future",
                "Date" => "DateTime",
                _ => name,
            };
            if type_args.is_empty() {
                base.to_string()
            } else {
                format!("{}<{}>", base, type_args.join(", "))
            }
        }
        TypeFieldKind::Array { element_type } => {
            format!("[{}]", ts_type_to_sigil_type(element_type, &TypeFieldKind::Complex { raw: element_type.clone() }))
        }
        TypeFieldKind::Union { variants } => {
            // Check for optional pattern: T | null | undefined
            let non_null: Vec<&String> = variants.iter()
                .filter(|v| v.as_str() != "null" && v.as_str() != "undefined")
                .collect();
            if non_null.len() == 1 && variants.len() > 1 {
                format!("Option<{}>", ts_type_to_sigil_type(non_null[0], &TypeFieldKind::Complex { raw: non_null[0].clone() }))
            } else {
                variants.join(" | ")
            }
        }
        TypeFieldKind::Function { params, return_type } => {
            let params_str: Vec<String> = params.iter()
                .map(|p| ts_type_to_sigil_type(&p.type_annotation, &TypeFieldKind::Complex { raw: p.type_annotation.clone() }))
                .collect();
            let ret = ts_type_to_sigil_type(return_type, &TypeFieldKind::Complex { raw: return_type.clone() });
            format!("rite({}) -> {}", params_str.join(", "), ret)
        }
        TypeFieldKind::Record { key_type, value_type } => {
            format!("Map<{}, {}>", key_type, value_type)
        }
        TypeFieldKind::Tuple { element_types } => {
            format!("({})", element_types.join(", "))
        }
        TypeFieldKind::Literal { value } => {
            // Literal types become the value itself
            value.clone()
        }
        TypeFieldKind::Complex { .. } => {
            // Fallback: use the annotation as-is but clean it up
            annotation.replace("React.", "")
        }
    }
}

// =============================================================================
// State Mutation Transformation (Phase 3)
// =============================================================================

/// Transform React state mutation calls to Sigil state assignments.
/// e.g., `setMessages([...messages, input])` → `self.messages = [...self.messages, self.input]`
fn transform_state_mutations_to_sigil(
    mutations: &[String],
    state_fields: &[(String, String)], // [(setter_name, field_name)]
) -> Vec<String> {
    mutations.iter()
        .filter_map(|mutation| transform_single_mutation(mutation, state_fields))
        .collect()
}

/// The same mutations, structured, with the value left as JavaScript so the
/// generator can run it through the expression transform.
fn extract_state_assignments(
    mutations: &[String],
    state_fields: &[(String, String)],
) -> Vec<StateAssignment> {
    mutations
        .iter()
        .filter_map(|mutation| {
            let mutation = mutation.trim();
            let (setter, field) = state_fields
                .iter()
                .find(|(setter, _)| mutation.starts_with(setter.as_str()))?;
            let _ = setter;
            let start = mutation.find('(')?;
            let end = find_matching_paren(mutation, start)?;
            let value = mutation[start + 1..end].trim();
            if value.is_empty() {
                return None;
            }
            // `setOpen(prev => !prev)` — the parameter stands for the current
            // value of the field, so the generator binds it before the body.
            let (value, updater_param) = match split_updater(value) {
                Some((param, body)) => (body, Some(param)),
                None => (value.to_string(), None),
            };
            Some(StateAssignment {
                field: to_snake_case(field),
                value,
                updater_param,
            })
        })
        .collect()
}

/// Split a functional-updater argument `prev => body` / `(prev) => body` into its
/// parameter and body. Returns None for anything else, including multi-parameter
/// arrows, which are not updaters.
fn split_updater(value: &str) -> Option<(String, String)> {
    let arrow = value.find("=>")?;
    let param = value[..arrow].trim().trim_start_matches('(').trim_end_matches(')').trim();
    if param.is_empty()
        || !param.chars().all(|c| c.is_alphanumeric() || c == '_' || c == '$')
        || param.chars().next().is_some_and(|c| c.is_numeric())
    {
        return None;
    }
    let body = value[arrow + 2..].trim();
    // A block body is a statement sequence, not an expression; the generator has
    // no spelling for it and would emit the braces verbatim.
    if body.starts_with('{') {
        return None;
    }
    Some((param.to_string(), body.to_string()))
}

/// Transform a single React state mutation to Sigil.
fn transform_single_mutation(
    mutation: &str,
    state_fields: &[(String, String)],
) -> Option<String> {
    let mutation = mutation.trim();

    // Look for setState(value) pattern
    for (setter, field) in state_fields {
        if mutation.starts_with(setter) {
            // Extract the value argument
            if let Some(start) = mutation.find('(') {
                let depth_start = start + 1;
                let end = find_matching_paren(mutation, start)?;
                let value = mutation[depth_start..end].trim();

                // Transform state references in the value
                let transformed_value = transform_state_references(value, state_fields);

                return Some(format!("self.{} = {}", to_snake_case(field), transformed_value));
            }
        }
    }

    // Not a recognized state setter. This used to fall through to
    // `transform_state_references`, which returns the JavaScript unchanged when
    // it recognises nothing — so a call to a prop setter or an imported helper
    // was emitted verbatim into the actor body. Harmless while only `handle*`
    // functions were extracted; once every function-valued binding became a
    // handler candidate it put raw JS in generated Sigil. Record what the React
    // did instead, on one line, and leave the translation to a human.
    let one_line: String = mutation.split_whitespace().collect::<Vec<_>>().join(" ");
    let one_line = if one_line.chars().count() > 100 {
        one_line.chars().take(97).collect::<String>() + "..."
    } else {
        one_line
    };
    Some(format!("// React: {}", one_line))
}

/// Find the matching closing parenthesis.
fn find_matching_paren(s: &str, open_pos: usize) -> Option<usize> {
    let bytes = s.as_bytes();
    let mut depth = 0;

    for (i, &b) in bytes.iter().enumerate().skip(open_pos) {
        match b {
            b'(' => depth += 1,
            b')' => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }

    None
}

/// Transform state variable references to use self. prefix.
fn transform_state_references(value: &str, state_fields: &[(String, String)]) -> String {
    let mut result = value.to_string();

    // Get all field names for prefixing
    let field_names: Vec<&str> = state_fields.iter()
        .map(|(_, field)| field.as_str())
        .collect();

    // Simple word boundary replacement for field names
    for field in &field_names {
        // Replace standalone field references with self.field
        // This is a simple approach - a full solution would use proper parsing
        let pattern = format!(r"\b{}\b", regex::escape(field));
        if let Ok(re) = regex::Regex::new(&pattern) {
            // Only replace if not already prefixed with self.
            let replacement = format!("self.{}", to_snake_case(field));

            // Avoid replacing self.field with self.self.field
            let mut new_result = String::new();
            let mut last_end = 0;

            for m in re.find_iter(&result) {
                // Check if preceded by "self."
                let prefix_start = m.start().saturating_sub(5);
                let prefix = &result[prefix_start..m.start()];

                new_result.push_str(&result[last_end..m.start()]);

                if prefix.ends_with("self.") {
                    // Already has self. prefix
                    new_result.push_str(m.as_str());
                } else {
                    new_result.push_str(&replacement);
                }

                last_end = m.end();
            }
            new_result.push_str(&result[last_end..]);
            result = new_result;
        }
    }

    result
}

// =============================================================================
// Service Call Extraction (Phase 3)
// =============================================================================

/// Extract service calls from handler calls that come from hooks.
/// Transforms hook-returned function calls into service actor messages.
fn extract_service_calls(calls: &[HandlerCall]) -> Vec<ServiceCall> {
    calls.iter()
        .filter_map(|call| {
            match &call.source {
                CallSource::Hook { hook_name } => {
                    // Convert hook name to service actor name
                    // e.g., "useChat" -> "ChatService"
                    let service = hook_name_to_service(hook_name);

                    // Convert function name to method name
                    // e.g., "addMessage" -> "AddMessage"
                    let method = to_pascal_case(&call.name);

                    Some(ServiceCall {
                        service,
                        method,
                        args: call.arguments.clone(),
                    })
                }
                _ => None, // Only hook-returned functions become service calls
            }
        })
        .collect()
}

/// Convert a hook name to a service actor name.
/// e.g., "useChat" -> "ChatService", "useAgent" -> "AgentService"
fn hook_name_to_service(hook_name: &str) -> String {
    let base = hook_name
        .strip_prefix("use")
        .unwrap_or(hook_name);

    format!("{}Service", to_pascal_case(base))
}

/// Builder for collecting service actor info across multiple components.
struct ServiceActorBuilder {
    name: String,
    derived_from: String,
    state_fields: std::collections::HashMap<String, ServiceStateField>,
    messages: std::collections::HashMap<String, ServiceMessage>,
    used_by: Vec<String>,
}

/// Infer type from a variable name (heuristic-based).
fn infer_type_from_name(name: &str) -> String {
    let lower = name.to_lowercase();

    // Boolean patterns
    if lower.starts_with("is_") || lower.starts_with("has_") ||
       lower.starts_with("can_") || lower.starts_with("should_") ||
       lower.starts_with("is") || lower.starts_with("has") ||
       lower.ends_with("ing") || lower.ends_with("ed") {
        return "bool".to_string();
    }

    // Array/list patterns
    if lower.ends_with("s") && !lower.ends_with("ss") && !lower.ends_with("us") {
        // Likely plural - probably a list
        return "Vec<Any>".to_string();
    }
    if lower.contains("list") || lower.contains("array") || lower.contains("items") {
        return "Vec<Any>".to_string();
    }

    // String patterns
    if lower.contains("name") || lower.contains("text") || lower.contains("message") ||
       lower.contains("content") || lower.contains("title") || lower.contains("description") ||
       lower.contains("id") || lower.contains("url") || lower.contains("path") {
        return "String".to_string();
    }

    // Number patterns
    if lower.contains("count") || lower.contains("index") || lower.contains("size") ||
       lower.contains("length") || lower.contains("num") || lower.contains("total") {
        return "i64".to_string();
    }

    // Default to Any
    "Any".to_string()
}

/// The callback an inline event handler actually invokes.
///
/// `() => onExpand(slot.id)` invokes `onExpand`; `onClick={handleSubmit}` invokes
/// `handleSubmit`. Shared by the message name, the message payload and the
/// handler body, so all three read the same call and cannot disagree about
/// which one the handler was for.
pub fn derive_invoked_callback(handler_code: &str) -> Option<String> {
    let body = match handler_code.find("=>") {
        Some(i) => &handler_code[i + 2..],
        None => handler_code,
    }
    .trim();

    // First identifier followed by '(' — the callback being invoked.
    let bytes: Vec<char> = body.chars().collect();
    let mut ident = String::new();
    let mut found = None;
    for (i, c) in bytes.iter().enumerate() {
        if c.is_alphanumeric() || *c == '_' || *c == '$' {
            ident.push(*c);
        } else {
            if *c == '(' && !ident.is_empty() {
                found = Some(ident.clone());
                break;
            }
            // a '.' continues a member chain; keep the LAST segment
            if *c == '.' {
                ident.clear();
            } else {
                ident.clear();
            }
        }
        if i + 1 == bytes.len() && !ident.is_empty() && found.is_none() {
            // bare reference, e.g. onClick={handleSubmit}
            found = Some(ident.clone());
        }
    }

    let name = found?;

    // `e => { e.preventDefault(); onSubmit(x) }` names the message after the real
    // action, not the event plumbing that happens to come first.
    if matches!(
        name.as_str(),
        "preventDefault" | "stopPropagation" | "stopImmediatePropagation"
    ) {
        return body.find(");").and_then(|i| {
            let rest = &body[i + 1..];
            let mut ident = String::new();
            let mut found2 = None;
            for c in rest.chars() {
                if c.is_alphanumeric() || c == '_' || c == '$' {
                    ident.push(c);
                } else {
                    if c == '(' && !ident.is_empty() {
                        found2 = Some(ident.clone());
                        break;
                    }
                    ident.clear();
                }
            }
            found2
        });
    }

    Some(name)
}

/// Derive a message name for an inline JSX event handler.
///
/// `onChange={e => onChange(e.target.checked)}` should not become the same message
/// as every other onChange in the file. The meaningful name is the callback the
/// handler actually invokes, so this looks past any arrow to the first call:
///
///   `e => onChange(e.target.checked)` -> Change
///   `() => onExpand(slot.id)`         -> Expand
///   `() => setOpen(true)`             -> UpdateOpen
///   `handleSubmit`                    -> HandleSubmit
///
/// Falls back to the event name when nothing better can be found.
pub fn derive_event_message_name(handler_code: &str, event_name: &str) -> String {
    let name = match derive_invoked_callback(handler_code) {
        Some(n) => n,
        None => return to_pascal_case(event_name.trim_start_matches("on")),
    };

    if let Some(rest) = name.strip_prefix("set") {
        if rest.chars().next().map_or(false, |c| c.is_uppercase()) {
            return format!("Update{}", to_pascal_case(rest));
        }
    }
    if let Some(rest) = name.strip_prefix("on") {
        if rest.chars().next().map_or(false, |c| c.is_uppercase()) {
            return to_pascal_case(rest);
        }
    }
    to_pascal_case(&name)
}

/// Collect the message names implied by inline event handlers in a JSX tree.
///
/// These never appear in `ComponentExtraction::handlers`, which only holds named
/// handler functions, so without this every inline handler dispatched a message
/// that was never declared in the actor's enum.
pub fn collect_jsx_event_messages(jsx: &JsxTree) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();

    fn walk(node: &JsxNode, out: &mut Vec<(String, String)>) {
        match &node.node_type {
            JsxNodeType::Element { attributes, children, .. } => {
                for a in attributes {
                    if !a.is_event_handler {
                        continue;
                    }
                    let code = match &a.value {
                        JsxAttributeValue::Expression { code } => code.clone(),
                        _ => String::new(),
                    };
                    let msg = derive_event_message_name(&code, &a.name);
                    if !out.iter().any(|(m, _)| *m == msg) {
                        out.push((msg, code));
                    }
                }
                for c in children {
                    walk(c, out);
                }
            }
            JsxNodeType::Fragment { children } => {
                for c in children {
                    walk(c, out);
                }
            }
            // Conditionals and maps hold the bulk of a real component's handlers —
            // rows rendered from `items.map(...)`, panels behind `cond && <div/>`.
            // Skipping them meant a large tab contributed no handler messages at all.
            JsxNodeType::Conditional { consequent, alternate, .. } => {
                walk(consequent, out);
                if let Some(alt) = alternate {
                    walk(alt, out);
                }
            }
            JsxNodeType::Map { body, .. } => walk(body, out),
            _ => {}
        }
    }

    if let Some(root) = &jsx.root {
        walk(root, &mut out);
    }
    out
}

/// The parameter names bound by a handler's arrow function, if it is one.
///
/// `(e: React.MouseEvent) => onPick(e.target.value)` binds `e`. Anything rooted
/// at one of these names cannot become a message payload: the dispatch site in
/// the generated Sigil is the node builder, not a closure, so the event object
/// is simply not in scope there.
fn arrow_param_names(handler_code: &str) -> Vec<String> {
    let head = match handler_code.find("=>") {
        Some(i) => handler_code[..i].trim(),
        None => return Vec::new(),
    };
    let head = head
        .trim()
        .trim_start_matches('(')
        .trim_end_matches(')')
        .trim();
    head.split(',')
        .filter_map(|p| {
            let ident: String = p
                .trim()
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '$')
                .collect();
            if ident.is_empty() { None } else { Some(ident) }
        })
        .collect()
}

/// Locate the first `ident(` in `body`, returning the name and the byte offset
/// of its opening paren.
fn find_invocation(body: &str) -> Option<(String, usize)> {
    let mut ident = String::new();
    let mut start = 0usize;
    for (i, c) in body.char_indices() {
        if c.is_alphanumeric() || c == '_' || c == '$' {
            if ident.is_empty() {
                start = i;
            }
            ident.push(c);
        } else {
            if c == '(' && !ident.is_empty() {
                let _ = start;
                return Some((ident, i));
            }
            ident.clear();
        }
    }
    None
}

/// Split the balanced argument list beginning at `open` (the index of `(`) into
/// its top-level comma-separated pieces.
fn split_call_args(body: &str, open: usize) -> Option<Vec<String>> {
    let mut depth = 0i32;
    let mut args: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut quote: Option<char> = None;
    for (i, c) in body.char_indices() {
        if i < open {
            continue;
        }
        if let Some(q) = quote {
            cur.push(c);
            if c == q {
                quote = None;
            }
            continue;
        }
        match c {
            '"' | '\'' | '`' => {
                quote = Some(c);
                cur.push(c);
            }
            '(' | '[' | '{' => {
                depth += 1;
                if depth > 1 {
                    cur.push(c);
                }
            }
            ')' | ']' | '}' => {
                depth -= 1;
                if depth == 0 {
                    if !cur.trim().is_empty() {
                        args.push(cur.trim().to_string());
                    }
                    return Some(args);
                }
                cur.push(c);
            }
            ',' if depth == 1 => {
                args.push(cur.trim().to_string());
                cur.clear();
            }
            _ => cur.push(c),
        }
    }
    None
}

/// Whether an argument expression can be carried verbatim into a message payload.
///
/// Conservative on purpose. A payload is emitted at the node-builder dispatch
/// site, so it may only reference bindings that are live there — a `map` item,
/// a prop, a signal — and must survive `transform_expression` unchanged in
/// shape. Calls, object/array literals, ternaries and arithmetic are all
/// rejected rather than mistranslated.
fn is_payloadable_arg(arg: &str, params: &[String]) -> bool {
    let a = arg.trim();
    if a.is_empty() || a.len() > 60 {
        return false;
    }
    if a.chars().any(|c| {
        matches!(
            c,
            '(' | ')' | '{' | '}' | '=' | '>' | '<' | '?' | ':' | '+' | '*' | '/' | '&' | '|'
                | '!' | '%' | '^' | '~' | ';'
        )
    }) {
        return false;
    }
    // String and numeric literals pass straight through.
    let first = a.chars().next().unwrap();
    if first == '"' || first == '\'' || first.is_ascii_digit() || first == '-' {
        return !a.contains('`');
    }
    if !(first.is_alphabetic() || first == '_' || first == '$') {
        return false;
    }
    let root: String = a
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == '$')
        .collect();
    if params.iter().any(|p| *p == root) {
        return false;
    }
    // `this`/`window`/`document` have no Sigil equivalent at the dispatch site.
    !matches!(root.as_str(), "this" | "window" | "document" | "globalThis")
}

/// The payload arguments implied by an inline event handler.
///
/// `() => onExpand(slot.id)` yields `["slot.id"]`, so the message can be
/// declared `Expand(Any)` and dispatched as `Expand(slot·id)` instead of losing
/// the one piece of information the handler was carrying. Returns an empty
/// vector when the handler takes no arguments, or when any of them is not
/// safely reproducible at the dispatch site.
pub fn derive_event_message_payload(handler_code: &str) -> Vec<String> {
    let body = match handler_code.find("=>") {
        Some(i) => &handler_code[i + 2..],
        None => handler_code,
    }
    .trim();

    let params = arrow_param_names(handler_code);

    let (name, open) = match find_invocation(body) {
        Some(v) => v,
        None => return Vec::new(),
    };

    // Skip the event plumbing, the same way the message name does, so payload
    // and name are always read off the same call.
    let (_, open) = if matches!(
        name.as_str(),
        "preventDefault" | "stopPropagation" | "stopImmediatePropagation"
    ) {
        match body.find(");").and_then(|i| {
            find_invocation(&body[i + 1..]).map(|(n, o)| (n, o + i + 1))
        }) {
            Some(v) => v,
            None => return Vec::new(),
        }
    } else {
        (name, open)
    };

    let args = match split_call_args(body, open) {
        Some(a) => a,
        None => return Vec::new(),
    };
    if args.is_empty() || !args.iter().all(|a| is_payloadable_arg(a, &params)) {
        return Vec::new();
    }
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SIGIL_KEYWORDS is transcribed from the lexer, and transcription drops entries:
    /// the first version of that list was missing exactly one word — `ref` — which was
    /// the one React actually uses, so two files still failed to parse after the fix
    /// that was supposed to handle them. Pin it to the lexer at compile time.
    #[test]
    fn keyword_list_matches_the_lexer() {
        let lexer = include_str!("../../lexer.rs");
        let mut from_lexer: Vec<&str> = lexer
            .match_indices("#[token(\"")
            .filter_map(|(i, _)| {
                let rest = &lexer[i + 9..];
                let end = rest.find('"')?;
                let word = &rest[..end];
                if rest[end..].starts_with("\")]")
                    && !word.is_empty()
                    && word.chars().all(|c| c.is_ascii_lowercase() || c == '_')
                {
                    Some(word)
                } else {
                    None
                }
            })
            .collect();
        from_lexer.sort_unstable();
        from_lexer.dedup();

        let missing: Vec<&&str> = from_lexer
            .iter()
            .filter(|w| !SIGIL_KEYWORDS.contains(w))
            .collect();
        assert!(missing.is_empty(), "keywords in the lexer but not escaped: {:?}", missing);
    }

    #[test]
    fn keywords_are_escaped_and_ordinary_names_are_not() {
        assert_eq!(escape_sigil_keyword("ref"), "ref_");
        assert_eq!(escape_sigil_keyword("type"), "type_");
        assert_eq!(escape_sigil_keyword("on_change"), "on_change");
        assert_eq!(escape_sigil_keyword("reference"), "reference");
    }

    /// Sigil accepts a lowercase name as a standalone type but rejects one as a type
    /// ARGUMENT, so mapping only the outer name left `Record<string, X>` unparseable.
    #[test]
    fn generic_arguments_are_mapped_too() {
        assert_eq!(map_ts_type_to_sigil("Record<string, DirState>"), "Map<String, DirState>");
        assert_eq!(map_ts_type_to_sigil("Readonly<Record<string, P>>"), "Map<String, P>");
        assert_eq!(map_ts_type_to_sigil("Array<number>"), "Vec<f64>");
        assert_eq!(map_ts_type_to_sigil("Option<string>"), "Option<String>");
    }

    #[test]
    fn payload_rejects_what_the_dispatch_site_cannot_reproduce() {
        // The dispatch site is a node builder, not a closure: `e` is not in scope.
        assert!(derive_event_message_payload("e => onPick(e.target.value)").is_empty());
        assert!(derive_event_message_payload("() => onSave(build())").is_empty());
        assert_eq!(derive_event_message_payload("() => onExpand(slot.id)"), vec!["slot.id"]);
        assert_eq!(derive_event_message_payload("() => onView(\"active\")"), vec!["\"active\""]);
    }
}
