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
    pub target: String,     // Suggested Sigil
    pub manual_review_needed: bool,
    pub notes: Vec<String>,
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

        let total = components.len();

        MigrationSpec {
            version: "1.0".to_string(),
            generated_at: chrono_now(),
            project_root: self.extraction.file.path.parent()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default(),
            components,
            types,
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
                    });
                }
            }
        }

        // Generate messages from event handlers
        for handler in &comp.handlers {
            let msg_name = to_pascal_case(&handler.name.replace("handle", ""));

            messages.push(MessageRecommendation {
                name: msg_name,
                from_handler: handler.name.clone(),
                payload: None,
                state_changes: handler.state_mutations.clone(),
                side_effects: handler.api_calls.clone(),
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

    fn generate_type_spec(&self, type_ext: &TypeExtraction) -> TypeMigrationSpec {
        let sigil_type = convert_ts_type_to_sigil(type_ext);

        TypeMigrationSpec {
            id: format!("{}:{}", self.extraction.file.relative_path, type_ext.name),
            name: type_ext.name.clone(),
            source: type_ext.definition.clone(),
            target: sigil_type,
            manual_review_needed: false,
            notes: vec![],
        }
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

fn chrono_now() -> String {
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
            format!("Σ {} {{ /* fields */ }}", type_ext.name)
        }
        TypeKind::TypeAlias => {
            format!("type {} = /* ... */", type_ext.name)
        }
        TypeKind::Enum => {
            format!("ᛈ {} {{ /* variants */ }}", type_ext.name)
        }
    }
}
