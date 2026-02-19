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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentSource {
    pub path: String,
    pub code: String,
    pub extraction: ComponentExtraction,
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

                    messages.push(MessageRecommendation {
                        name: msg_name.clone(),
                        from_handler: setter_name.clone(),
                        payload: None,
                        state_changes: vec![format!("self.{} = /* new value */", state_name)],
                        side_effects: vec![],
                        service_calls: vec![],
                    });
                }
            }
        }

        // Generate messages from event handlers
        for handler in &comp.handlers {
            let msg_name = to_pascal_case(&handler.name.replace("handle", ""));

            // Transform React state mutations to Sigil syntax
            let transformed_state_changes = transform_state_mutations_to_sigil(
                &handler.state_mutations,
                &state_fields,
            );

            // Extract service calls from handler.calls (hook-returned functions)
            let service_calls = extract_service_calls(&handler.calls);

            messages.push(MessageRecommendation {
                name: msg_name,
                from_handler: handler.name.clone(),
                payload: None,
                state_changes: transformed_state_changes,
                side_effects: handler.api_calls.clone(),
                service_calls,
            });
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
    match ts_type {
        "number" => "f64".to_string(),
        "string" => "String".to_string(),
        "boolean" => "bool".to_string(),
        "void" => "()".to_string(),
        "null" | "undefined" => "∅".to_string(),
        "any" | "unknown" => "Any".to_string(),
        "HTMLInputElement" | "HTMLElement" | "Element" => "Element".to_string(),
        t if t.starts_with("Array<") => {
            let inner = &t[6..t.len()-1];
            format!("Vec<{}>", map_ts_type_to_sigil(inner))
        }
        t => t.to_string(), // Keep as-is for custom types
    }
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

                return Some(format!("self.{} = {}", field, transformed_value));
            }
        }
    }

    // Not a recognized state setter - return as-is with self prefix attempt
    Some(transform_state_references(mutation, state_fields))
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
            let replacement = format!("self.{}", field);

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
