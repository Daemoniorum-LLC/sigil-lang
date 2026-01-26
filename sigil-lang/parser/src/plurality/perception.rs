//! # Perception System for DAEMONIORUM
//!
//! Handles reality layer perception (Grounded/Fractured/Shattered),
//! entity visibility, environmental transformation, and the interplay
//! between alter state and perceived reality.

use std::collections::HashMap;

use super::runtime::{AnimaState, PluralSystem, RealityLayer, Trigger, TriggerCategory};

// ============================================================================
// PERCEPTION STATE
// ============================================================================

/// The current perception state of the player
#[derive(Debug, Clone)]
pub struct PerceptionState {
    /// Current reality layer
    pub layer: RealityLayer,
    /// Perception intensity (0.0 = barely perceiving, 1.0 = fully immersed)
    pub intensity: f32,
    /// Stability of perception (0.0 = fluctuating, 1.0 = stable)
    pub stability: f32,
    /// Active perception modifiers
    pub modifiers: Vec<PerceptionModifier>,
    /// Transition state (if currently shifting)
    pub transition: Option<PerceptionTransition>,
    /// Entities currently visible in each layer
    pub visible_entities: HashMap<RealityLayer, Vec<String>>,
    /// Environmental overlays active
    pub overlays: Vec<EnvironmentalOverlay>,
}

impl Default for PerceptionState {
    fn default() -> Self {
        Self {
            layer: RealityLayer::Grounded,
            intensity: 0.5,
            stability: 0.8,
            modifiers: Vec::new(),
            transition: None,
            visible_entities: HashMap::new(),
            overlays: Vec::new(),
        }
    }
}

impl PerceptionState {
    /// Create a new perception state at the given layer
    pub fn at_layer(layer: RealityLayer) -> Self {
        Self {
            layer,
            ..Default::default()
        }
    }

    /// Update perception based on system state
    pub fn update_from_system(&mut self, system: &PluralSystem) {
        // Dissociation affects perception intensity
        self.intensity = 0.5 + system.dissociation * 0.4;

        // System stability affects perception stability
        self.stability = system.stability * 0.7 + 0.3;

        // High dissociation can force layer shifts
        if system.dissociation > 0.7 && self.layer == RealityLayer::Grounded {
            self.begin_transition(RealityLayer::Fractured, 0.5);
        }

        // Very high dissociation pushes to Shattered
        if system.dissociation > 0.9 && self.layer == RealityLayer::Fractured {
            self.begin_transition(RealityLayer::Shattered, 0.3);
        }

        // Grounding can restore normal perception
        if system.stability > 0.8 && system.dissociation < 0.3 {
            if self.layer == RealityLayer::Shattered {
                self.begin_transition(RealityLayer::Fractured, 0.7);
            } else if self.layer == RealityLayer::Fractured {
                self.begin_transition(RealityLayer::Grounded, 0.5);
            }
        }

        // Process any active transition
        if let Some(ref mut transition) = self.transition {
            transition.progress += transition.rate;
            if transition.progress >= 1.0 {
                self.layer = transition.target.clone();
                self.transition = None;
            }
        }
    }

    /// Begin transitioning to a new reality layer
    pub fn begin_transition(&mut self, target: RealityLayer, rate: f32) {
        if self.layer != target && self.transition.is_none() {
            self.transition = Some(PerceptionTransition {
                from: self.layer.clone(),
                target,
                progress: 0.0,
                rate,
                visual_effects: Vec::new(),
            });
        }
    }

    /// Force immediate layer change
    pub fn force_layer(&mut self, layer: RealityLayer) {
        self.transition = None;
        self.layer = layer;
    }

    /// Add a perception modifier
    pub fn add_modifier(&mut self, modifier: PerceptionModifier) {
        self.modifiers.push(modifier);
    }

    /// Remove expired modifiers
    pub fn tick_modifiers(&mut self) {
        self.modifiers.retain(|m| {
            if let Some(duration) = m.duration {
                duration > 0
            } else {
                true
            }
        });

        for modifier in &mut self.modifiers {
            if let Some(ref mut duration) = modifier.duration {
                *duration = duration.saturating_sub(1);
            }
        }
    }

    /// Get the effective perception intensity (with modifiers)
    pub fn effective_intensity(&self) -> f32 {
        let mut intensity = self.intensity;
        for modifier in &self.modifiers {
            if let PerceptionModifierType::IntensityChange(delta) = modifier.modifier_type {
                intensity += delta;
            }
        }
        intensity.clamp(0.0, 1.0)
    }

    /// Check if an entity is visible at current perception
    pub fn can_see_entity(&self, entity: &PerceivableEntity) -> bool {
        // Check if entity is visible in current layer
        if !entity.visible_in.contains(&self.layer) {
            return false;
        }

        // Check intensity requirements
        if let Some(min_intensity) = entity.min_perception_intensity {
            if self.effective_intensity() < min_intensity {
                return false;
            }
        }

        // Check modifier restrictions
        for modifier in &self.modifiers {
            if let PerceptionModifierType::BlockEntity(ref id) = modifier.modifier_type {
                if id == &entity.id {
                    return false;
                }
            }
        }

        true
    }

    /// Add an environmental overlay
    pub fn add_overlay(&mut self, overlay: EnvironmentalOverlay) {
        // Check if this overlay replaces an existing one
        if let Some(existing) = self.overlays.iter_mut().find(|o| o.id == overlay.id) {
            *existing = overlay;
        } else {
            self.overlays.push(overlay);
        }
    }

    /// Get active overlays for the current layer
    pub fn active_overlays(&self) -> Vec<&EnvironmentalOverlay> {
        self.overlays
            .iter()
            .filter(|o| o.applies_to.contains(&self.layer) || o.applies_to.is_empty())
            .collect()
    }
}

// ============================================================================
// PERCEPTION TRANSITION
// ============================================================================

/// A transition between reality layers
#[derive(Debug, Clone)]
pub struct PerceptionTransition {
    /// Layer transitioning from
    pub from: RealityLayer,
    /// Layer transitioning to
    pub target: RealityLayer,
    /// Transition progress (0.0 to 1.0)
    pub progress: f32,
    /// Rate of transition per tick
    pub rate: f32,
    /// Visual effects during transition
    pub visual_effects: Vec<TransitionEffect>,
}

impl PerceptionTransition {
    /// Get the current blend ratio for visual rendering
    pub fn blend_ratio(&self) -> (f32, f32) {
        (1.0 - self.progress, self.progress)
    }
}

/// Visual effects during layer transition
#[derive(Debug, Clone)]
pub enum TransitionEffect {
    /// Screen distortion
    Distortion { intensity: f32 },
    /// Color shift
    ColorShift { hue_offset: f32, saturation: f32 },
    /// Vignette/tunnel vision
    Vignette { radius: f32 },
    /// Static/noise
    Static { amount: f32 },
    /// Blur
    Blur { radius: f32 },
    /// Entity flickering
    EntityFlicker { ids: Vec<String> },
}

// ============================================================================
// PERCEPTION MODIFIERS
// ============================================================================

/// A modifier affecting perception
#[derive(Debug, Clone)]
pub struct PerceptionModifier {
    /// Modifier ID
    pub id: String,
    /// Display name
    pub name: String,
    /// Duration in ticks (None = permanent until removed)
    pub duration: Option<u32>,
    /// The actual modification
    pub modifier_type: PerceptionModifierType,
    /// Source of the modifier
    pub source: ModifierSource,
}

/// Types of perception modifications
#[derive(Debug, Clone)]
pub enum PerceptionModifierType {
    /// Change perception intensity
    IntensityChange(f32),
    /// Change perception stability
    StabilityChange(f32),
    /// Lock to a specific layer
    LayerLock(RealityLayer),
    /// Force layer transition
    ForceTransition(RealityLayer),
    /// Block seeing a specific entity
    BlockEntity(String),
    /// Reveal hidden entity
    RevealEntity(String),
    /// Add visual overlay
    AddOverlay(String),
    /// Custom effect
    Custom(String),
}

/// Source of a perception modifier
#[derive(Debug, Clone)]
pub enum ModifierSource {
    /// From an alter's influence
    Alter(String),
    /// From an item/ability
    Ability(String),
    /// From environment
    Environment,
    /// From a trigger event
    Trigger(String),
    /// From combat
    Combat,
    /// Unknown/system
    System,
}

// ============================================================================
// PERCEIVABLE ENTITIES
// ============================================================================

/// An entity that exists across reality layers
#[derive(Debug, Clone)]
pub struct PerceivableEntity {
    /// Entity ID
    pub id: String,
    /// Display name (may vary by layer)
    pub name: String,
    /// Entity type
    pub entity_type: EntityType,
    /// Which layers this entity is visible in
    pub visible_in: Vec<RealityLayer>,
    /// Minimum perception intensity to see
    pub min_perception_intensity: Option<f32>,
    /// Visual representations per layer
    pub layer_representations: HashMap<RealityLayer, EntityRepresentation>,
    /// Does this entity cause triggers when perceived?
    pub perception_triggers: Vec<PerceptionTrigger>,
    /// Symbolic meaning (for Fractured/Shattered layers)
    pub symbolic_meaning: Option<String>,
}

impl PerceivableEntity {
    /// Get the representation for a specific layer
    pub fn representation_for(&self, layer: &RealityLayer) -> Option<&EntityRepresentation> {
        self.layer_representations.get(layer)
    }

    /// Get the appropriate name for the current layer
    pub fn name_for_layer(&self, layer: &RealityLayer) -> &str {
        self.layer_representations
            .get(layer)
            .and_then(|r| r.display_name.as_deref())
            .unwrap_or(&self.name)
    }
}

/// Types of perceivable entities
#[derive(Debug, Clone, PartialEq)]
pub enum EntityType {
    /// A person/NPC
    Character,
    /// An object
    Object,
    /// A location/area
    Location,
    /// An environmental feature
    Environmental,
    /// A manifestation (trauma-based entity)
    Manifestation,
    /// UI/HUD element that varies by layer
    Interface,
    /// Abstract/symbolic entity
    Symbol,
}

/// How an entity appears in a specific layer
#[derive(Debug, Clone)]
pub struct EntityRepresentation {
    /// Override display name
    pub display_name: Option<String>,
    /// Visual description
    pub description: String,
    /// Sprite/model ID
    pub visual_id: String,
    /// Color tint
    pub color_tint: Option<(f32, f32, f32, f32)>,
    /// Scale modifier
    pub scale: f32,
    /// Opacity
    pub opacity: f32,
    /// Animation state
    pub animation: Option<String>,
    /// Additional visual effects
    pub effects: Vec<String>,
}

impl Default for EntityRepresentation {
    fn default() -> Self {
        Self {
            display_name: None,
            description: String::new(),
            visual_id: String::new(),
            color_tint: None,
            scale: 1.0,
            opacity: 1.0,
            animation: None,
            effects: Vec::new(),
        }
    }
}

/// A trigger that activates when an entity is perceived
#[derive(Debug, Clone)]
pub struct PerceptionTrigger {
    /// Trigger ID to fire
    pub trigger_id: String,
    /// Conditions for triggering
    pub conditions: Vec<PerceptionTriggerCondition>,
    /// Intensity of the trigger
    pub intensity: f32,
}

/// Conditions for perception triggers
#[derive(Debug, Clone)]
pub enum PerceptionTriggerCondition {
    /// Trigger when first perceived
    OnFirstSight,
    /// Trigger when perception starts (enter view)
    OnEnterView,
    /// Trigger when perception ends (leave view)
    OnLeaveView,
    /// Trigger when perceived for certain duration
    AfterDuration(u32),
    /// Trigger at specific perception intensity
    AtIntensity(f32),
    /// Trigger when specific alter is fronting
    WhenAlterFronting(String),
    /// Trigger when at specific layer
    AtLayer(RealityLayer),
}

// ============================================================================
// ENVIRONMENTAL OVERLAYS
// ============================================================================

/// A visual overlay applied to the environment
#[derive(Debug, Clone)]
pub struct EnvironmentalOverlay {
    /// Overlay ID
    pub id: String,
    /// Display name
    pub name: String,
    /// Which layers this overlay applies to (empty = all)
    pub applies_to: Vec<RealityLayer>,
    /// Overlay type
    pub overlay_type: OverlayType,
    /// Intensity (0.0 to 1.0)
    pub intensity: f32,
    /// Duration (None = permanent)
    pub duration: Option<u32>,
}

/// Types of environmental overlays
#[derive(Debug, Clone)]
pub enum OverlayType {
    /// Blood/gore overlay
    Blood { spread: f32, color: (f32, f32, f32) },
    /// Fire/burning overlay
    Fire { spread: f32 },
    /// Corruption/decay overlay
    Corruption { spread: f32, pattern: String },
    /// Symbolic pattern overlay
    Symbolic { symbol: String, repeating: bool },
    /// Memory fragment overlay
    Memory { memory_id: String, opacity: f32 },
    /// Text/graffiti overlay
    Text { content: String, font: String },
    /// Weather effect
    Weather { weather_type: String },
    /// Color filter
    ColorFilter {
        hue: f32,
        saturation: f32,
        brightness: f32,
    },
    /// Custom shader effect
    Shader {
        shader_id: String,
        params: HashMap<String, f32>,
    },
}

// ============================================================================
// PERCEPTION MANAGER
// ============================================================================

/// Manages perception updates and entity visibility
#[derive(Debug, Clone)]
pub struct PerceptionManager {
    /// Current perception state
    pub state: PerceptionState,
    /// All perceivable entities in the current scene
    pub entities: HashMap<String, PerceivableEntity>,
    /// Entity perception history (for trigger timing)
    pub perception_history: HashMap<String, PerceptionHistory>,
    /// Queued triggers to fire
    pub pending_triggers: Vec<Trigger>,
}

impl Default for PerceptionManager {
    fn default() -> Self {
        Self {
            state: PerceptionState::default(),
            entities: HashMap::new(),
            perception_history: HashMap::new(),
            pending_triggers: Vec::new(),
        }
    }
}

impl PerceptionManager {
    /// Create a new perception manager
    pub fn new() -> Self {
        Self::default()
    }

    /// Update perception state from the plural system
    pub fn update(&mut self, system: &PluralSystem) {
        self.state.update_from_system(system);
        self.update_entity_visibility();
        self.process_perception_triggers(system);
        self.state.tick_modifiers();
    }

    /// Add an entity to the scene
    pub fn add_entity(&mut self, entity: PerceivableEntity) {
        let id = entity.id.clone();
        self.entities.insert(id.clone(), entity);
        self.perception_history
            .insert(id, PerceptionHistory::default());
    }

    /// Remove an entity from the scene
    pub fn remove_entity(&mut self, id: &str) {
        self.entities.remove(id);
        self.perception_history.remove(id);
    }

    /// Update which entities are visible
    fn update_entity_visibility(&mut self) {
        let mut visible = HashMap::new();

        for (id, entity) in &self.entities {
            if self.state.can_see_entity(entity) {
                let layer = self.state.layer.clone();
                visible
                    .entry(layer)
                    .or_insert_with(Vec::new)
                    .push(id.clone());

                // Update perception history
                if let Some(history) = self.perception_history.get_mut(id) {
                    if !history.currently_visible {
                        history.currently_visible = true;
                        history.time_visible = 0;
                        history.times_seen += 1;
                    } else {
                        history.time_visible += 1;
                    }
                }
            } else {
                // Update history for non-visible
                if let Some(history) = self.perception_history.get_mut(id) {
                    if history.currently_visible {
                        history.currently_visible = false;
                        history.time_visible = 0;
                    }
                }
            }
        }

        self.state.visible_entities = visible;
    }

    /// Process perception triggers
    fn process_perception_triggers(&mut self, system: &PluralSystem) {
        for (id, entity) in &self.entities {
            let history = match self.perception_history.get(id) {
                Some(h) => h,
                None => continue,
            };

            for trigger_def in &entity.perception_triggers {
                let should_trigger = trigger_def.conditions.iter().all(|cond| match cond {
                    PerceptionTriggerCondition::OnFirstSight => {
                        history.times_seen == 1 && history.time_visible == 0
                    }
                    PerceptionTriggerCondition::OnEnterView => {
                        history.currently_visible && history.time_visible == 0
                    }
                    PerceptionTriggerCondition::OnLeaveView => {
                        !history.currently_visible && history.time_visible == 0
                    }
                    PerceptionTriggerCondition::AfterDuration(dur) => {
                        history.currently_visible && history.time_visible >= *dur
                    }
                    PerceptionTriggerCondition::AtIntensity(int) => {
                        self.state.effective_intensity() >= *int
                    }
                    PerceptionTriggerCondition::WhenAlterFronting(alter) => {
                        match &system.fronting {
                            super::runtime::FrontingState::Single(id) => id == alter,
                            super::runtime::FrontingState::Blended(ids) => ids.contains(alter),
                            _ => false,
                        }
                    }
                    PerceptionTriggerCondition::AtLayer(layer) => &self.state.layer == layer,
                });

                if should_trigger {
                    self.pending_triggers.push(Trigger {
                        id: trigger_def.trigger_id.clone(),
                        name: format!("Perception: {}", entity.name),
                        category: TriggerCategory::Environmental,
                        intensity: trigger_def.intensity,
                        context: HashMap::from([
                            ("entity_id".to_string(), id.clone()),
                            ("layer".to_string(), format!("{:?}", self.state.layer)),
                        ]),
                    });
                }
            }
        }
    }

    /// Get all currently visible entities
    pub fn visible_entities(&self) -> Vec<&PerceivableEntity> {
        let layer = &self.state.layer;
        self.state
            .visible_entities
            .get(layer)
            .map(|ids| ids.iter().filter_map(|id| self.entities.get(id)).collect())
            .unwrap_or_default()
    }

    /// Get pending triggers and clear the queue
    pub fn drain_triggers(&mut self) -> Vec<Trigger> {
        std::mem::take(&mut self.pending_triggers)
    }

    /// Force a layer shift
    pub fn force_layer_shift(&mut self, layer: RealityLayer) {
        self.state.force_layer(layer);
        self.update_entity_visibility();
    }

    /// Begin a gradual layer transition
    pub fn begin_layer_transition(&mut self, target: RealityLayer, rate: f32) {
        self.state.begin_transition(target, rate);
    }
}

/// History of perceiving an entity
#[derive(Debug, Clone, Default)]
pub struct PerceptionHistory {
    /// Is currently visible
    pub currently_visible: bool,
    /// Time visible (ticks)
    pub time_visible: u32,
    /// Number of times seen
    pub times_seen: u32,
}

// ============================================================================
// REALITY LAYER DEFINITIONS
// ============================================================================

/// Predefined characteristics for each reality layer
pub struct LayerCharacteristics {
    /// Base color grading
    pub color_grade: ColorGrade,
    /// Audio modifications
    pub audio_mod: AudioModification,
    /// Entity visibility rules
    pub entity_rules: Vec<EntityVisibilityRule>,
    /// Environmental modifications
    pub environment_mod: EnvironmentModification,
}

/// Color grading for a layer
#[derive(Debug, Clone)]
pub struct ColorGrade {
    pub saturation: f32,
    pub contrast: f32,
    pub brightness: f32,
    pub hue_shift: f32,
    pub vignette: f32,
    pub grain: f32,
}

impl ColorGrade {
    /// Grounded layer - normal, slightly warm
    pub fn grounded() -> Self {
        Self {
            saturation: 1.0,
            contrast: 1.0,
            brightness: 1.0,
            hue_shift: 0.0,
            vignette: 0.1,
            grain: 0.0,
        }
    }

    /// Fractured layer - desaturated, higher contrast, cold
    pub fn fractured() -> Self {
        Self {
            saturation: 0.6,
            contrast: 1.3,
            brightness: 0.9,
            hue_shift: -15.0,
            vignette: 0.3,
            grain: 0.1,
        }
    }

    /// Shattered layer - heavily stylized, surreal
    pub fn shattered() -> Self {
        Self {
            saturation: 0.3,
            contrast: 1.5,
            brightness: 0.7,
            hue_shift: 30.0,
            vignette: 0.5,
            grain: 0.3,
        }
    }
}

/// Audio modifications for a layer
#[derive(Debug, Clone)]
pub struct AudioModification {
    pub reverb: f32,
    pub low_pass_filter: f32,
    pub pitch_variation: f32,
    pub ambient_volume: f32,
    pub distortion: f32,
}

/// Rules for entity visibility in a layer
#[derive(Debug, Clone)]
pub struct EntityVisibilityRule {
    pub entity_type: EntityType,
    pub visible: bool,
    pub transform: Option<EntityTransform>,
}

/// Transform applied to entities in different layers
#[derive(Debug, Clone)]
pub struct EntityTransform {
    pub scale: f32,
    pub opacity: f32,
    pub color_shift: Option<(f32, f32, f32)>,
    pub effect: Option<String>,
}

/// Environmental modifications for a layer
#[derive(Debug, Clone)]
pub struct EnvironmentModification {
    pub fog_density: f32,
    pub shadow_intensity: f32,
    pub ambient_light: f32,
    pub weather_override: Option<String>,
    pub geometry_distortion: f32,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perception_state_default() {
        let state = PerceptionState::default();
        assert_eq!(state.layer, RealityLayer::Grounded);
        assert!((state.intensity - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_layer_transition() {
        let mut state = PerceptionState::default();
        state.begin_transition(RealityLayer::Fractured, 0.5);

        assert!(state.transition.is_some());
        let transition = state.transition.as_ref().unwrap();
        assert_eq!(transition.target, RealityLayer::Fractured);
    }

    #[test]
    fn test_entity_visibility() {
        let state = PerceptionState::at_layer(RealityLayer::Grounded);

        let visible_entity = PerceivableEntity {
            id: "test1".to_string(),
            name: "Test Entity".to_string(),
            entity_type: EntityType::Character,
            visible_in: vec![RealityLayer::Grounded, RealityLayer::Fractured],
            min_perception_intensity: None,
            layer_representations: HashMap::new(),
            perception_triggers: Vec::new(),
            symbolic_meaning: None,
        };

        let invisible_entity = PerceivableEntity {
            id: "test2".to_string(),
            name: "Fractured Only".to_string(),
            entity_type: EntityType::Manifestation,
            visible_in: vec![RealityLayer::Fractured],
            min_perception_intensity: None,
            layer_representations: HashMap::new(),
            perception_triggers: Vec::new(),
            symbolic_meaning: Some("Trauma manifestation".to_string()),
        };

        assert!(state.can_see_entity(&visible_entity));
        assert!(!state.can_see_entity(&invisible_entity));
    }

    #[test]
    fn test_perception_manager() {
        let mut manager = PerceptionManager::new();

        let entity = PerceivableEntity {
            id: "church".to_string(),
            name: "Church".to_string(),
            entity_type: EntityType::Location,
            visible_in: vec![RealityLayer::Grounded, RealityLayer::Fractured],
            min_perception_intensity: None,
            layer_representations: HashMap::from([
                (
                    RealityLayer::Grounded,
                    EntityRepresentation {
                        description: "A peaceful church".to_string(),
                        visual_id: "church_normal".to_string(),
                        ..Default::default()
                    },
                ),
                (
                    RealityLayer::Fractured,
                    EntityRepresentation {
                        description: "The church walls bleed".to_string(),
                        visual_id: "church_fractured".to_string(),
                        color_tint: Some((0.8, 0.2, 0.2, 1.0)),
                        ..Default::default()
                    },
                ),
            ]),
            perception_triggers: Vec::new(),
            symbolic_meaning: Some("Sanctuary lost to corruption".to_string()),
        };

        manager.add_entity(entity);

        let system = PluralSystem::default();
        manager.update(&system);

        let visible = manager.visible_entities();
        assert_eq!(visible.len(), 1);
        assert_eq!(visible[0].name, "Church");
    }
}
