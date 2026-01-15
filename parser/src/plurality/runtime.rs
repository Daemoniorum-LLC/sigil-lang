//! # Runtime Types for Plurality
//!
//! Runtime representations of plurality constructs for the DAEMONIORUM game engine.
//! These types represent the in-game state of the plural system.

use std::collections::{HashMap, HashSet};

// ============================================================================
// ANIMA STATE (PAD Model)
// ============================================================================

/// Anima state represents the emotional/psychological state of an alter or the system.
/// Based on the PAD (Pleasure-Arousal-Dominance) model of emotional states.
///
/// Each dimension ranges from -1.0 to 1.0:
/// - Pleasure: unhappy (-1) to happy (+1)
/// - Arousal: calm (-1) to excited (+1)
/// - Dominance: submissive (-1) to dominant (+1)
#[derive(Debug, Clone, PartialEq)]
pub struct AnimaState {
    /// Pleasure dimension: -1.0 (unhappy) to 1.0 (happy)
    pub pleasure: f32,
    /// Arousal dimension: -1.0 (calm) to 1.0 (excited)
    pub arousal: f32,
    /// Dominance dimension: -1.0 (submissive) to 1.0 (dominant)
    pub dominance: f32,
    /// Expressiveness: how visibly the emotion is displayed (0.0 to 1.0)
    pub expressiveness: f32,
    /// Stability: how stable the current emotional state is (0.0 to 1.0)
    pub stability: f32,
}

impl Default for AnimaState {
    fn default() -> Self {
        Self {
            pleasure: 0.0,
            arousal: 0.0,
            dominance: 0.0,
            expressiveness: 0.5,
            stability: 0.7,
        }
    }
}

impl AnimaState {
    /// Create a new AnimaState with the given PAD values
    pub fn new(pleasure: f32, arousal: f32, dominance: f32) -> Self {
        Self {
            pleasure: pleasure.clamp(-1.0, 1.0),
            arousal: arousal.clamp(-1.0, 1.0),
            dominance: dominance.clamp(-1.0, 1.0),
            expressiveness: 0.5,
            stability: 0.7,
        }
    }

    /// Create an anxious state (low pleasure, high arousal, low dominance)
    pub fn anxious() -> Self {
        Self::new(-0.5, 0.7, -0.4)
    }

    /// Create an angry state (low pleasure, high arousal, high dominance)
    pub fn angry() -> Self {
        Self::new(-0.7, 0.8, 0.6)
    }

    /// Create a calm state (neutral pleasure, low arousal, neutral dominance)
    pub fn calm() -> Self {
        Self::new(0.3, -0.6, 0.0)
    }

    /// Create a dissociated state (flat affect)
    pub fn dissociated() -> Self {
        Self {
            pleasure: 0.0,
            arousal: -0.3,
            dominance: -0.5,
            expressiveness: 0.1,
            stability: 0.3,
        }
    }

    /// Apply trauma response modifier
    pub fn apply_trauma_response(&mut self, intensity: f32) {
        self.arousal = (self.arousal + intensity * 0.5).clamp(-1.0, 1.0);
        self.stability -= intensity * 0.3;
        self.stability = self.stability.clamp(0.0, 1.0);
    }

    /// Blend two AnimaStates together
    pub fn blend(&self, other: &AnimaState, ratio: f32) -> AnimaState {
        let ratio = ratio.clamp(0.0, 1.0);
        let inv = 1.0 - ratio;
        AnimaState {
            pleasure: self.pleasure * inv + other.pleasure * ratio,
            arousal: self.arousal * inv + other.arousal * ratio,
            dominance: self.dominance * inv + other.dominance * ratio,
            expressiveness: self.expressiveness * inv + other.expressiveness * ratio,
            stability: (self.stability * inv + other.stability * ratio).min(self.stability).min(other.stability),
        }
    }

    /// Calculate the intensity/magnitude of the emotional state
    pub fn intensity(&self) -> f32 {
        (self.pleasure.powi(2) + self.arousal.powi(2) + self.dominance.powi(2)).sqrt() / 1.732
    }
}

// ============================================================================
// ALTER STATE
// ============================================================================

/// Runtime state of an alter
#[derive(Debug, Clone, PartialEq)]
pub enum AlterPresenceState {
    /// Alter is completely inactive
    Dormant,
    /// Alter is beginning to wake/activate
    Stirring,
    /// Alter is present but not fronting
    CoConscious,
    /// Alter is transitioning to front
    Emerging,
    /// Alter is currently in control
    Fronting,
    /// Alter is transitioning away from front
    Receding,
    /// Alter is in trauma response
    Triggered,
    /// Alter is disconnecting/going passive
    Dissociating,
}

/// Runtime representation of an alter
#[derive(Debug, Clone)]
pub struct Alter {
    /// Unique identifier for the alter
    pub id: String,
    /// Display name
    pub name: String,
    /// Category (Council, Servant, Fragment, etc.)
    pub category: AlterCategory,
    /// Current presence state
    pub state: AlterPresenceState,
    /// Current anima (emotional) state
    pub anima: AnimaState,
    /// Base arousal level (personality trait)
    pub base_arousal: f32,
    /// Base dominance level (personality trait)
    pub base_dominance: f32,
    /// Time since last fronting (in game time units)
    pub time_since_front: u64,
    /// Triggers that can activate this alter
    pub triggers: Vec<TriggerId>,
    /// Abilities unique to this alter
    pub abilities: HashSet<String>,
    /// Preferred reality layer
    pub preferred_reality: RealityLayer,
    /// Memory access level for this alter
    pub memory_access: MemoryAccess,
}

/// Alter category from the Council system
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AlterCategory {
    /// Core system member, full agency
    Council,
    /// Helper alter, limited scope
    Servant,
    /// Incomplete alter, specific function
    Fragment,
    /// External introject
    Introject,
    /// Persecutor alter
    Persecutor,
    /// Trauma holder
    TraumaHolder,
    /// Custom category
    Custom(String),
}

/// Memory access level
#[derive(Debug, Clone, PartialEq)]
pub enum MemoryAccess {
    /// Full access to all system memories
    Full,
    /// Partial access (specific memory sets)
    Partial(Vec<String>),
    /// Limited to own memories only
    Own,
    /// Amnesiac - no memory access
    None,
}

// ============================================================================
// PLURAL SYSTEM
// ============================================================================

/// The plural system as a whole
#[derive(Debug, Clone)]
pub struct PluralSystem {
    /// System name (if any)
    pub name: Option<String>,
    /// All alters in the system
    pub alters: HashMap<String, Alter>,
    /// Currently fronting alter(s)
    pub fronting: FrontingState,
    /// System-level anima state (blended from active alters)
    pub anima: AnimaState,
    /// Current reality perception layer
    pub reality_layer: RealityLayer,
    /// Active triggers being processed
    pub active_triggers: Vec<Trigger>,
    /// Headspace state
    pub headspace: HeadspaceState,
    /// Dissociation level (0.0 to 1.0)
    pub dissociation: f32,
    /// System stability (0.0 to 1.0)
    pub stability: f32,
}

impl Default for PluralSystem {
    fn default() -> Self {
        Self {
            name: None,
            alters: HashMap::new(),
            fronting: FrontingState::None,
            anima: AnimaState::default(),
            reality_layer: RealityLayer::Grounded,
            active_triggers: Vec::new(),
            headspace: HeadspaceState::default(),
            dissociation: 0.0,
            stability: 1.0,
        }
    }
}

impl PluralSystem {
    /// Create a new plural system with the given name
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            ..Default::default()
        }
    }

    /// Add an alter to the system
    pub fn add_alter(&mut self, alter: Alter) {
        self.alters.insert(alter.id.clone(), alter);
    }

    /// Get the currently fronting alter (if single fronter)
    pub fn current_fronter(&self) -> Option<&Alter> {
        match &self.fronting {
            FrontingState::Single(id) => self.alters.get(id),
            FrontingState::Blended(ids) if ids.len() == 1 => self.alters.get(&ids[0]),
            _ => None,
        }
    }

    /// Request a switch to a different alter
    pub fn request_switch(&mut self, target_id: &str, urgency: f32, forced: bool) -> SwitchResult {
        if !self.alters.contains_key(target_id) {
            return SwitchResult::Failed(SwitchFailReason::UnknownAlter);
        }

        // Check if switch is possible based on system state
        if self.dissociation > 0.8 && !forced {
            return SwitchResult::Failed(SwitchFailReason::TooDisassociated);
        }

        if self.stability < 0.2 && !forced {
            return SwitchResult::Failed(SwitchFailReason::SystemUnstable);
        }

        // Calculate switch difficulty
        let current_alter = self.current_fronter();
        let resistance = if let Some(current) = current_alter {
            // More dominant alters are harder to switch away from
            (current.anima.dominance + 1.0) / 2.0 * (1.0 - urgency)
        } else {
            0.0
        };

        if resistance > 0.7 && !forced {
            return SwitchResult::Resisted { resistance };
        }

        // Perform the switch
        if let Some(prev) = current_alter {
            let prev_id = prev.id.clone();
            if let Some(alter) = self.alters.get_mut(&prev_id) {
                alter.state = AlterPresenceState::Receding;
            }
        }

        if let Some(alter) = self.alters.get_mut(target_id) {
            alter.state = AlterPresenceState::Fronting;
            alter.time_since_front = 0;
        }

        self.fronting = FrontingState::Single(target_id.to_string());
        self.update_blended_anima();

        SwitchResult::Success
    }

    /// Update the system's blended anima state from active alters
    pub fn update_blended_anima(&mut self) {
        let mut total_influence = 0.0;
        let mut blended = AnimaState::default();

        for alter in self.alters.values() {
            let influence = match alter.state {
                AlterPresenceState::Fronting => 1.0,
                AlterPresenceState::CoConscious => 0.3,
                AlterPresenceState::Emerging => 0.5,
                AlterPresenceState::Receding => 0.2,
                AlterPresenceState::Triggered => 0.7,
                _ => 0.0,
            };

            if influence > 0.0 {
                blended.pleasure += alter.anima.pleasure * influence;
                blended.arousal += alter.anima.arousal * influence;
                blended.dominance += alter.anima.dominance * influence;
                total_influence += influence;
            }
        }

        if total_influence > 0.0 {
            blended.pleasure /= total_influence;
            blended.arousal /= total_influence;
            blended.dominance /= total_influence;
            blended.expressiveness = 0.5; // Average expressiveness
            blended.stability = self.stability;
        }

        self.anima = blended;
    }

    /// Process a trigger event
    pub fn process_trigger(&mut self, trigger: Trigger) -> TriggerResult {
        self.active_triggers.push(trigger.clone());

        // Find alters that respond to this trigger
        let responding_alters: Vec<String> = self.alters.values()
            .filter(|a| a.triggers.contains(&trigger.id))
            .map(|a| a.id.clone())
            .collect();

        if responding_alters.is_empty() {
            return TriggerResult::NoResponse;
        }

        // Calculate response intensity
        let intensity = trigger.intensity * (1.0 + self.dissociation);

        // Update responding alters
        for alter_id in &responding_alters {
            if let Some(alter) = self.alters.get_mut(alter_id) {
                if matches!(alter.state, AlterPresenceState::Dormant | AlterPresenceState::Stirring) {
                    alter.state = AlterPresenceState::Stirring;
                    alter.anima.apply_trauma_response(intensity);
                }
            }
        }

        // High intensity triggers can cause forced switches
        if intensity > 0.8 {
            if let Some(strongest) = responding_alters.first() {
                return TriggerResult::ForcedSwitch(strongest.clone());
            }
        }

        TriggerResult::Activation(responding_alters)
    }

    /// Shift reality perception layer
    pub fn shift_reality(&mut self, target: RealityLayer, perception_level: f32) {
        // Reality shifts are influenced by dissociation and trigger state
        let shift_threshold = match (&self.reality_layer, &target) {
            (RealityLayer::Grounded, RealityLayer::Fractured) => 0.3,
            (RealityLayer::Fractured, RealityLayer::Shattered) => 0.6,
            (RealityLayer::Shattered, RealityLayer::Fractured) => 0.4,
            (RealityLayer::Fractured, RealityLayer::Grounded) => 0.5,
            _ => 0.5,
        };

        if perception_level >= shift_threshold || self.dissociation > 0.7 {
            self.reality_layer = target;
        }
    }
}

// ============================================================================
// FRONTING STATE
// ============================================================================

/// Represents who is currently fronting
#[derive(Debug, Clone, PartialEq)]
pub enum FrontingState {
    /// No one is fronting (passive, autopilot)
    None,
    /// Single alter fronting
    Single(String),
    /// Multiple alters co-fronting
    Blended(Vec<String>),
    /// Rapid switching between alters
    Rapid(Vec<String>),
    /// Unknown/unclear who is fronting
    Unknown,
}

// ============================================================================
// REALITY LAYERS
// ============================================================================

/// Reality perception layer
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RealityLayer {
    /// Normal perception, grounded in consensus reality
    Grounded,
    /// Fractured perception - distorted, symbolic overlays
    Fractured,
    /// Completely shattered - full symbolic/hallucinatory experience
    Shattered,
    /// Custom reality layer
    Custom(String),
}

// ============================================================================
// TRIGGERS
// ============================================================================

/// Unique identifier for a trigger type
pub type TriggerId = String;

/// A trigger event that can affect the system
#[derive(Debug, Clone)]
pub struct Trigger {
    /// Unique identifier for this trigger type
    pub id: TriggerId,
    /// Display name
    pub name: String,
    /// Trigger category
    pub category: TriggerCategory,
    /// Intensity of the trigger (0.0 to 1.0)
    pub intensity: f32,
    /// Additional context/data
    pub context: HashMap<String, String>,
}

/// Categories of triggers
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TriggerCategory {
    /// Environmental trigger (sound, smell, place)
    Environmental,
    /// Social trigger (person, interaction type)
    Social,
    /// Internal trigger (thought, memory, emotion)
    Internal,
    /// Physical trigger (sensation, pain, touch)
    Physical,
    /// Temporal trigger (time of day, anniversary)
    Temporal,
    /// Combat trigger (threat, violence)
    Combat,
    /// Custom category
    Custom(String),
}

/// Result of processing a trigger
#[derive(Debug, Clone)]
pub enum TriggerResult {
    /// No alters responded to the trigger
    NoResponse,
    /// Trigger activated one or more alters
    Activation(Vec<String>),
    /// Trigger caused a forced switch
    ForcedSwitch(String),
    /// System dissociated in response
    Dissociation,
}

// ============================================================================
// SWITCH RESULT
// ============================================================================

/// Result of a switch attempt
#[derive(Debug, Clone)]
pub enum SwitchResult {
    /// Switch succeeded
    Success,
    /// Switch was resisted
    Resisted { resistance: f32 },
    /// Switch failed
    Failed(SwitchFailReason),
    /// Switch is in progress (async)
    InProgress { eta: u64 },
}

/// Reasons a switch can fail
#[derive(Debug, Clone, PartialEq)]
pub enum SwitchFailReason {
    /// Target alter doesn't exist
    UnknownAlter,
    /// System is too dissociated
    TooDisassociated,
    /// System is unstable
    SystemUnstable,
    /// Current fronter is refusing
    CurrentRefused,
    /// Target alter is unavailable
    TargetUnavailable,
    /// External barrier (game mechanic)
    Blocked(String),
}

// ============================================================================
// HEADSPACE
// ============================================================================

/// State of the internal headspace/inner world
#[derive(Debug, Clone, Default)]
pub struct HeadspaceState {
    /// Current active location
    pub current_location: Option<String>,
    /// Alters present at current location
    pub present_alters: Vec<String>,
    /// Active navigation path
    pub navigation_path: Vec<String>,
    /// Hazards in the current area
    pub active_hazards: Vec<String>,
    /// Weather/atmosphere
    pub atmosphere: HeadspaceAtmosphere,
}

/// Atmospheric conditions in the headspace
#[derive(Debug, Clone, Default)]
pub struct HeadspaceAtmosphere {
    /// Clarity (0.0 foggy to 1.0 clear)
    pub clarity: f32,
    /// Stability (0.0 chaotic to 1.0 stable)
    pub stability: f32,
    /// Lighting (0.0 dark to 1.0 bright)
    pub lighting: f32,
    /// Custom atmosphere effects
    pub effects: Vec<String>,
}

// ============================================================================
// CO-CONSCIOUSNESS CHANNEL
// ============================================================================

/// A communication channel between co-conscious alters
#[derive(Debug, Clone)]
pub struct CoConChannel {
    /// Participating alters
    pub participants: Vec<String>,
    /// Channel quality (0.0 to 1.0)
    pub quality: f32,
    /// Messages in the channel
    pub messages: Vec<CoConMessage>,
    /// Whether the channel is currently active
    pub active: bool,
}

/// A message in a co-con channel
#[derive(Debug, Clone)]
pub struct CoConMessage {
    /// Sender alter id
    pub from: String,
    /// Message content
    pub content: String,
    /// Certainty of the message (evidentiality)
    pub certainty: f32,
    /// Timestamp (game time)
    pub timestamp: u64,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anima_state_blend() {
        let anxious = AnimaState::anxious();
        let calm = AnimaState::calm();
        let blended = anxious.blend(&calm, 0.5);

        assert!(blended.pleasure > anxious.pleasure);
        assert!(blended.arousal < anxious.arousal);
    }

    #[test]
    fn test_plural_system_add_alter() {
        let mut system = PluralSystem::new("Test System");

        let alter = Alter {
            id: "abaddon".to_string(),
            name: "Abaddon".to_string(),
            category: AlterCategory::Council,
            state: AlterPresenceState::Dormant,
            anima: AnimaState::default(),
            base_arousal: 0.3,
            base_dominance: 0.6,
            time_since_front: 0,
            triggers: vec!["threat".to_string()],
            abilities: HashSet::from(["combat".to_string()]),
            preferred_reality: RealityLayer::Fractured,
            memory_access: MemoryAccess::Full,
        };

        system.add_alter(alter);
        assert!(system.alters.contains_key("abaddon"));
    }

    #[test]
    fn test_switch_request() {
        let mut system = PluralSystem::new("Test System");

        let alter1 = Alter {
            id: "host".to_string(),
            name: "Host".to_string(),
            category: AlterCategory::Council,
            state: AlterPresenceState::Fronting,
            anima: AnimaState::default(),
            base_arousal: 0.0,
            base_dominance: 0.0,
            time_since_front: 0,
            triggers: vec![],
            abilities: HashSet::new(),
            preferred_reality: RealityLayer::Grounded,
            memory_access: MemoryAccess::Full,
        };

        let alter2 = Alter {
            id: "protector".to_string(),
            name: "Protector".to_string(),
            category: AlterCategory::Council,
            state: AlterPresenceState::Dormant,
            anima: AnimaState::default(),
            base_arousal: 0.5,
            base_dominance: 0.7,
            time_since_front: 100,
            triggers: vec!["threat".to_string()],
            abilities: HashSet::from(["combat".to_string()]),
            preferred_reality: RealityLayer::Grounded,
            memory_access: MemoryAccess::Full,
        };

        system.add_alter(alter1);
        system.add_alter(alter2);
        system.fronting = FrontingState::Single("host".to_string());

        let result = system.request_switch("protector", 0.8, false);
        assert!(matches!(result, SwitchResult::Success));
        assert_eq!(system.fronting, FrontingState::Single("protector".to_string()));
    }
}
