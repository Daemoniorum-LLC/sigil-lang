//! # Dialogue System for DAEMONIORUM
//!
//! An alter-aware dialogue system that adapts responses based on
//! which alter is fronting, their relationships, and system state.

use std::collections::HashMap;

use super::runtime::{
    AlterCategory, AlterPresenceState, AnimaState, FrontingState, PluralSystem, RealityLayer,
};

// ============================================================================
// DIALOGUE TREE STRUCTURE
// ============================================================================

/// A complete dialogue tree for an NPC or interaction
#[derive(Debug, Clone)]
pub struct DialogueTree {
    /// Unique identifier
    pub id: String,
    /// All nodes in the tree
    pub nodes: HashMap<String, DialogueNode>,
    /// Entry point node
    pub entry_node: String,
    /// Variables set during dialogue
    pub variables: HashMap<String, DialogueValue>,
    /// Speaker information
    pub speaker: SpeakerInfo,
}

/// A single node in the dialogue tree
#[derive(Debug, Clone)]
pub struct DialogueNode {
    /// Node identifier
    pub id: String,
    /// The content of this node
    pub content: DialogueContent,
    /// Conditions for this node to be valid
    pub conditions: Vec<DialogueCondition>,
    /// Effects that trigger when this node is visited
    pub effects: Vec<DialogueEffect>,
    /// Possible responses/choices
    pub responses: Vec<DialogueResponse>,
    /// Next node if no responses (linear flow)
    pub next: Option<String>,
    /// Tags for filtering/searching
    pub tags: Vec<String>,
}

/// Content displayed in a dialogue node
#[derive(Debug, Clone)]
pub struct DialogueContent {
    /// Base text (fallback)
    pub base_text: String,
    /// Alter-specific text variations
    pub alter_variations: HashMap<String, AlterDialogueVariation>,
    /// Reality layer variations
    pub layer_variations: HashMap<RealityLayer, String>,
    /// Emotional state variations
    pub emotional_variations: Vec<EmotionalVariation>,
    /// Voice/expression cues
    pub voice_cues: Vec<String>,
    /// Animation triggers
    pub animations: Vec<String>,
}

/// Variation of dialogue for a specific alter
#[derive(Debug, Clone)]
pub struct AlterDialogueVariation {
    /// The text for this alter
    pub text: String,
    /// Unique observations this alter would make
    pub observations: Vec<String>,
    /// Emotional coloring
    pub tone: DialogueTone,
    /// Whether this alter recognizes something others wouldn't
    pub recognition: Option<RecognitionEvent>,
}

/// Emotional variation based on system state
#[derive(Debug, Clone)]
pub struct EmotionalVariation {
    /// Condition for this variation
    pub condition: EmotionalCondition,
    /// Text when condition is met
    pub text: String,
}

/// Emotional conditions for dialogue variation
#[derive(Debug, Clone)]
pub enum EmotionalCondition {
    /// High dissociation
    HighDissociation(f32),
    /// Low stability
    LowStability(f32),
    /// High arousal
    HighArousal(f32),
    /// Specific anima state
    AnimaState {
        pleasure: (f32, f32),
        arousal: (f32, f32),
        dominance: (f32, f32),
    },
    /// Recently triggered
    RecentTrigger(String),
}

/// Tone of dialogue
#[derive(Debug, Clone, PartialEq)]
pub enum DialogueTone {
    Neutral,
    Warm,
    Cold,
    Suspicious,
    Aggressive,
    Fearful,
    Curious,
    Analytical,
    Playful,
    Guarded,
}

/// Recognition event when an alter sees something familiar
#[derive(Debug, Clone)]
pub struct RecognitionEvent {
    /// What is recognized
    pub target: String,
    /// Type of recognition
    pub recognition_type: RecognitionType,
    /// Intensity (0-1)
    pub intensity: f32,
}

/// Types of recognition
#[derive(Debug, Clone)]
pub enum RecognitionType {
    /// Recognizes an abuser's traits
    Abuser,
    /// Recognizes a safe person's traits
    SafePerson,
    /// Recognizes a place from memory
    Place,
    /// Recognizes an object from trauma
    TraumaObject,
    /// Generic familiarity
    Familiar,
}

// ============================================================================
// DIALOGUE RESPONSES
// ============================================================================

/// A response option in dialogue
#[derive(Debug, Clone)]
pub struct DialogueResponse {
    /// Response identifier
    pub id: String,
    /// Displayed text
    pub text: String,
    /// Alter-specific response variations
    pub alter_variations: HashMap<String, String>,
    /// Conditions to show this response
    pub conditions: Vec<DialogueCondition>,
    /// Target node when selected
    pub target_node: String,
    /// Effects when selected
    pub effects: Vec<DialogueEffect>,
    /// Whether this is a "system" response (internal)
    pub internal: bool,
    /// Required alter traits to show
    pub required_traits: Vec<String>,
    /// Forbidden alter traits (hide if present)
    pub forbidden_traits: Vec<String>,
}

// ============================================================================
// CONDITIONS AND EFFECTS
// ============================================================================

/// Conditions for dialogue options
#[derive(Debug, Clone)]
pub enum DialogueCondition {
    /// Specific alter is fronting
    AlterFronting(String),
    /// Alter category is fronting
    CategoryFronting(AlterCategory),
    /// Specific alter is co-conscious
    AlterCoConscious(String),
    /// Reality layer requirement
    RealityLayer(RealityLayer),
    /// System stability threshold
    StabilityAbove(f32),
    /// Dissociation threshold
    DissociationBelow(f32),
    /// Dialogue variable check
    Variable {
        name: String,
        op: CompareOp,
        value: DialogueValue,
    },
    /// Flag is set
    FlagSet(String),
    /// Alter has trait
    AlterHasTrait(String),
    /// Anima state in range
    AnimaInRange {
        pleasure: (f32, f32),
        arousal: (f32, f32),
        dominance: (f32, f32),
    },
    /// Item in inventory
    HasItem(String),
    /// Ability unlocked
    HasAbility(String),
    /// Previous dialogue node visited
    NodeVisited(String),
    /// All conditions must be true
    All(Vec<DialogueCondition>),
    /// Any condition must be true
    Any(Vec<DialogueCondition>),
    /// Invert condition
    Not(Box<DialogueCondition>),
}

/// Comparison operators
#[derive(Debug, Clone)]
pub enum CompareOp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
}

/// Dialogue variable values
#[derive(Debug, Clone, PartialEq)]
pub enum DialogueValue {
    Bool(bool),
    Int(i32),
    Float(f32),
    String(String),
}

/// Effects triggered by dialogue
#[derive(Debug, Clone)]
pub enum DialogueEffect {
    /// Set a dialogue variable
    SetVariable { name: String, value: DialogueValue },
    /// Set a global flag
    SetFlag(String),
    /// Clear a flag
    ClearFlag(String),
    /// Modify anima state
    ModifyAnima {
        pleasure: f32,
        arousal: f32,
        dominance: f32,
    },
    /// Modify system stability
    ModifyStability(f32),
    /// Modify dissociation
    ModifyDissociation(f32),
    /// Trigger a switch request
    RequestSwitch { alter_id: String, urgency: f32 },
    /// Activate a trigger
    ActivateTrigger(String),
    /// Add item to inventory
    GiveItem(String),
    /// Remove item from inventory
    TakeItem(String),
    /// Unlock an ability
    UnlockAbility(String),
    /// Shift reality layer
    ShiftReality { target: RealityLayer, rate: f32 },
    /// Play sound
    PlaySound(String),
    /// Start cutscene
    StartCutscene(String),
    /// Grant experience/insight
    GrantInsight { id: String, amount: f32 },
    /// End dialogue
    EndDialogue,
}

// ============================================================================
// SPEAKER INFORMATION
// ============================================================================

/// Information about who is speaking
#[derive(Debug, Clone)]
pub struct SpeakerInfo {
    /// Speaker identifier
    pub id: String,
    /// Display name
    pub name: String,
    /// Portrait/avatar
    pub portrait: String,
    /// Alter-specific portraits (if speaker appears differently)
    pub alter_portraits: HashMap<String, String>,
    /// Reality layer portraits
    pub layer_portraits: HashMap<RealityLayer, String>,
}

// ============================================================================
// DIALOGUE MANAGER
// ============================================================================

/// Manages active dialogue sessions
pub struct DialogueManager {
    /// Currently active dialogue tree
    pub current_tree: Option<DialogueTree>,
    /// Current node ID
    pub current_node: Option<String>,
    /// History of visited nodes
    pub visited_nodes: Vec<String>,
    /// Dialogue variables
    pub variables: HashMap<String, DialogueValue>,
    /// Global flags
    pub flags: HashMap<String, bool>,
    /// Loaded dialogue trees
    trees: HashMap<String, DialogueTree>,
}

impl DialogueManager {
    /// Create a new dialogue manager
    pub fn new() -> Self {
        Self {
            current_tree: None,
            current_node: None,
            visited_nodes: Vec::new(),
            variables: HashMap::new(),
            flags: HashMap::new(),
            trees: HashMap::new(),
        }
    }

    /// Load a dialogue tree
    pub fn load_tree(&mut self, tree: DialogueTree) {
        self.trees.insert(tree.id.clone(), tree);
    }

    /// Start a dialogue
    pub fn start_dialogue(&mut self, tree_id: &str) -> Result<(), DialogueError> {
        let tree = self
            .trees
            .get(tree_id)
            .ok_or_else(|| DialogueError::TreeNotFound(tree_id.to_string()))?
            .clone();

        self.current_node = Some(tree.entry_node.clone());
        self.current_tree = Some(tree);
        self.visited_nodes.clear();

        Ok(())
    }

    /// Get the current dialogue content, adapted for the system state
    pub fn get_current_content(
        &self,
        system: &PluralSystem,
    ) -> Result<ResolvedDialogue, DialogueError> {
        let tree = self
            .current_tree
            .as_ref()
            .ok_or(DialogueError::NoActiveDialogue)?;

        let node_id = self
            .current_node
            .as_ref()
            .ok_or(DialogueError::NoActiveDialogue)?;

        let node = tree
            .nodes
            .get(node_id)
            .ok_or_else(|| DialogueError::NodeNotFound(node_id.clone()))?;

        // Resolve the content based on system state
        let text = self.resolve_content(&node.content, system, &tree.speaker);
        let responses = self.resolve_responses(&node.responses, system);
        let speaker = self.resolve_speaker(&tree.speaker, system);

        Ok(ResolvedDialogue {
            node_id: node_id.clone(),
            text,
            responses,
            speaker,
            voice_cues: node.content.voice_cues.clone(),
            animations: node.content.animations.clone(),
        })
    }

    /// Select a response and advance the dialogue
    pub fn select_response(
        &mut self,
        response_id: &str,
        system: &mut PluralSystem,
    ) -> Result<DialogueResult, DialogueError> {
        // Extract data we need before mutable borrow
        let (response_effects, response_conditions, response_target, node_id_clone) = {
            let tree = self
                .current_tree
                .as_ref()
                .ok_or(DialogueError::NoActiveDialogue)?;

            let node_id = self
                .current_node
                .as_ref()
                .ok_or(DialogueError::NoActiveDialogue)?;

            let node = tree
                .nodes
                .get(node_id)
                .ok_or_else(|| DialogueError::NodeNotFound(node_id.clone()))?;

            // Find the response
            let response = node
                .responses
                .iter()
                .find(|r| r.id == response_id)
                .ok_or_else(|| DialogueError::ResponseNotFound(response_id.to_string()))?;

            (
                response.effects.clone(),
                response.conditions.clone(),
                response.target_node.clone(),
                node_id.clone(),
            )
        };

        // Check conditions
        if !self.check_conditions(&response_conditions, system) {
            return Err(DialogueError::ConditionsNotMet);
        }

        // Apply effects
        let effects = self.apply_effects(&response_effects, system)?;

        // Track visited
        self.visited_nodes.push(node_id_clone);

        // Check for dialogue end
        if effects
            .iter()
            .any(|e| matches!(e, AppliedEffect::EndDialogue))
        {
            self.end_dialogue();
            return Ok(DialogueResult::Ended);
        }

        // Move to next node
        self.current_node = Some(response_target);

        Ok(DialogueResult::Continue(effects))
    }

    /// Advance to next node (for linear dialogue)
    pub fn advance(&mut self, system: &mut PluralSystem) -> Result<DialogueResult, DialogueError> {
        // Extract data we need before mutable borrow
        let (node_effects, node_next, has_responses, node_id_clone) = {
            let tree = self
                .current_tree
                .as_ref()
                .ok_or(DialogueError::NoActiveDialogue)?;

            let node_id = self
                .current_node
                .as_ref()
                .ok_or(DialogueError::NoActiveDialogue)?;

            let node = tree
                .nodes
                .get(node_id)
                .ok_or_else(|| DialogueError::NodeNotFound(node_id.clone()))?;

            (
                node.effects.clone(),
                node.next.clone(),
                !node.responses.is_empty(),
                node_id.clone(),
            )
        };

        // Apply node effects
        let effects = self.apply_effects(&node_effects, system)?;

        // Track visited
        self.visited_nodes.push(node_id_clone);

        // Check for dialogue end
        if effects
            .iter()
            .any(|e| matches!(e, AppliedEffect::EndDialogue))
        {
            self.end_dialogue();
            return Ok(DialogueResult::Ended);
        }

        // Move to next node
        match node_next {
            Some(next_id) => {
                self.current_node = Some(next_id);
                Ok(DialogueResult::Continue(effects))
            }
            None if !has_responses => {
                self.end_dialogue();
                Ok(DialogueResult::Ended)
            }
            None => Ok(DialogueResult::AwaitingChoice(effects)),
        }
    }

    /// End the current dialogue
    pub fn end_dialogue(&mut self) {
        self.current_tree = None;
        self.current_node = None;
    }

    /// Check if dialogue is active
    pub fn is_active(&self) -> bool {
        self.current_tree.is_some()
    }

    // ========================================================================
    // CONTENT RESOLUTION
    // ========================================================================

    /// Resolve dialogue content based on system state
    fn resolve_content(
        &self,
        content: &DialogueContent,
        system: &PluralSystem,
        _speaker: &SpeakerInfo,
    ) -> String {
        // Check for alter-specific variation
        if let Some(fronter_id) = self.get_fronter_id(system) {
            if let Some(variation) = content.alter_variations.get(&fronter_id) {
                return variation.text.clone();
            }
        }

        // Check for reality layer variation
        if let Some(layer_text) = content.layer_variations.get(&system.reality_layer) {
            return layer_text.clone();
        }

        // Check for emotional variations
        for variation in &content.emotional_variations {
            if self.check_emotional_condition(&variation.condition, system) {
                return variation.text.clone();
            }
        }

        // Default to base text
        content.base_text.clone()
    }

    /// Resolve available responses based on system state
    fn resolve_responses(
        &self,
        responses: &[DialogueResponse],
        system: &PluralSystem,
    ) -> Vec<ResolvedResponse> {
        let fronter_id = self.get_fronter_id(system);

        responses
            .iter()
            .filter(|r| self.check_conditions(&r.conditions, system))
            .filter(|r| self.check_trait_requirements(r, system))
            .map(|r| {
                // Get alter-specific text if available
                let text = fronter_id
                    .as_ref()
                    .and_then(|id| r.alter_variations.get(id))
                    .cloned()
                    .unwrap_or_else(|| r.text.clone());

                ResolvedResponse {
                    id: r.id.clone(),
                    text,
                    internal: r.internal,
                }
            })
            .collect()
    }

    /// Resolve speaker appearance based on system state
    fn resolve_speaker(&self, speaker: &SpeakerInfo, system: &PluralSystem) -> ResolvedSpeaker {
        // Check for alter-specific portrait
        let portrait = self
            .get_fronter_id(system)
            .and_then(|id| speaker.alter_portraits.get(&id))
            .cloned()
            // Check for reality layer portrait
            .or_else(|| speaker.layer_portraits.get(&system.reality_layer).cloned())
            // Default to base portrait
            .unwrap_or_else(|| speaker.portrait.clone());

        ResolvedSpeaker {
            name: speaker.name.clone(),
            portrait,
        }
    }

    // ========================================================================
    // CONDITION CHECKING
    // ========================================================================

    /// Check all conditions
    fn check_conditions(&self, conditions: &[DialogueCondition], system: &PluralSystem) -> bool {
        conditions.iter().all(|c| self.check_condition(c, system))
    }

    /// Check a single condition
    fn check_condition(&self, condition: &DialogueCondition, system: &PluralSystem) -> bool {
        match condition {
            DialogueCondition::AlterFronting(id) => {
                self.get_fronter_id(system).as_ref() == Some(id)
            }
            DialogueCondition::CategoryFronting(category) => {
                if let Some(fronter_id) = self.get_fronter_id(system) {
                    system
                        .alters
                        .get(&fronter_id)
                        .map(|a| &a.category == category)
                        .unwrap_or(false)
                } else {
                    false
                }
            }
            DialogueCondition::AlterCoConscious(id) => system
                .alters
                .get(id)
                .map(|a| matches!(a.state, AlterPresenceState::CoConscious))
                .unwrap_or(false),
            DialogueCondition::RealityLayer(layer) => &system.reality_layer == layer,
            DialogueCondition::StabilityAbove(threshold) => system.stability >= *threshold,
            DialogueCondition::DissociationBelow(threshold) => system.dissociation < *threshold,
            DialogueCondition::Variable { name, op, value } => self
                .variables
                .get(name)
                .map(|v| self.compare_values(v, value, op))
                .unwrap_or(false),
            DialogueCondition::FlagSet(flag) => self.flags.get(flag).copied().unwrap_or(false),
            DialogueCondition::AlterHasTrait(trait_name) => {
                if let Some(fronter_id) = self.get_fronter_id(system) {
                    system
                        .alters
                        .get(&fronter_id)
                        .map(|a| a.abilities.contains(trait_name))
                        .unwrap_or(false)
                } else {
                    false
                }
            }
            DialogueCondition::AnimaInRange {
                pleasure,
                arousal,
                dominance,
            } => {
                let anima = &system.anima;
                anima.pleasure >= pleasure.0
                    && anima.pleasure <= pleasure.1
                    && anima.arousal >= arousal.0
                    && anima.arousal <= arousal.1
                    && anima.dominance >= dominance.0
                    && anima.dominance <= dominance.1
            }
            DialogueCondition::HasItem(_item_id) => {
                // Would check inventory - simplified for now
                true
            }
            DialogueCondition::HasAbility(_ability_id) => {
                // Would check abilities - simplified for now
                true
            }
            DialogueCondition::NodeVisited(node_id) => self.visited_nodes.contains(node_id),
            DialogueCondition::All(conditions) => {
                conditions.iter().all(|c| self.check_condition(c, system))
            }
            DialogueCondition::Any(conditions) => {
                conditions.iter().any(|c| self.check_condition(c, system))
            }
            DialogueCondition::Not(condition) => !self.check_condition(condition, system),
        }
    }

    /// Check trait requirements for a response
    fn check_trait_requirements(&self, response: &DialogueResponse, system: &PluralSystem) -> bool {
        let fronter_id = match self.get_fronter_id(system) {
            Some(id) => id,
            None => return response.required_traits.is_empty(),
        };

        let alter = match system.alters.get(&fronter_id) {
            Some(a) => a,
            None => return response.required_traits.is_empty(),
        };

        // Check required traits
        let has_required = response
            .required_traits
            .iter()
            .all(|t| alter.abilities.contains(t));

        // Check forbidden traits
        let has_forbidden = response
            .forbidden_traits
            .iter()
            .any(|t| alter.abilities.contains(t));

        has_required && !has_forbidden
    }

    /// Check emotional condition
    fn check_emotional_condition(
        &self,
        condition: &EmotionalCondition,
        system: &PluralSystem,
    ) -> bool {
        match condition {
            EmotionalCondition::HighDissociation(threshold) => system.dissociation >= *threshold,
            EmotionalCondition::LowStability(threshold) => system.stability < *threshold,
            EmotionalCondition::HighArousal(threshold) => system.anima.arousal >= *threshold,
            EmotionalCondition::AnimaState {
                pleasure,
                arousal,
                dominance,
            } => {
                let anima = &system.anima;
                anima.pleasure >= pleasure.0
                    && anima.pleasure <= pleasure.1
                    && anima.arousal >= arousal.0
                    && anima.arousal <= arousal.1
                    && anima.dominance >= dominance.0
                    && anima.dominance <= dominance.1
            }
            EmotionalCondition::RecentTrigger(trigger_id) => {
                system.active_triggers.iter().any(|t| &t.id == trigger_id)
            }
        }
    }

    /// Compare dialogue values
    fn compare_values(&self, a: &DialogueValue, b: &DialogueValue, op: &CompareOp) -> bool {
        match (a, b) {
            (DialogueValue::Bool(a), DialogueValue::Bool(b)) => match op {
                CompareOp::Eq => a == b,
                CompareOp::Ne => a != b,
                _ => false,
            },
            (DialogueValue::Int(a), DialogueValue::Int(b)) => match op {
                CompareOp::Eq => a == b,
                CompareOp::Ne => a != b,
                CompareOp::Lt => a < b,
                CompareOp::Le => a <= b,
                CompareOp::Gt => a > b,
                CompareOp::Ge => a >= b,
            },
            (DialogueValue::Float(a), DialogueValue::Float(b)) => match op {
                CompareOp::Eq => (a - b).abs() < f32::EPSILON,
                CompareOp::Ne => (a - b).abs() >= f32::EPSILON,
                CompareOp::Lt => a < b,
                CompareOp::Le => a <= b,
                CompareOp::Gt => a > b,
                CompareOp::Ge => a >= b,
            },
            (DialogueValue::String(a), DialogueValue::String(b)) => match op {
                CompareOp::Eq => a == b,
                CompareOp::Ne => a != b,
                _ => false,
            },
            _ => false,
        }
    }

    // ========================================================================
    // EFFECT APPLICATION
    // ========================================================================

    /// Apply dialogue effects
    fn apply_effects(
        &mut self,
        effects: &[DialogueEffect],
        system: &mut PluralSystem,
    ) -> Result<Vec<AppliedEffect>, DialogueError> {
        let mut applied = Vec::new();

        for effect in effects {
            match effect {
                DialogueEffect::SetVariable { name, value } => {
                    self.variables.insert(name.clone(), value.clone());
                    applied.push(AppliedEffect::VariableSet(name.clone()));
                }
                DialogueEffect::SetFlag(flag) => {
                    self.flags.insert(flag.clone(), true);
                    applied.push(AppliedEffect::FlagSet(flag.clone()));
                }
                DialogueEffect::ClearFlag(flag) => {
                    self.flags.insert(flag.clone(), false);
                    applied.push(AppliedEffect::FlagCleared(flag.clone()));
                }
                DialogueEffect::ModifyAnima {
                    pleasure,
                    arousal,
                    dominance,
                } => {
                    system.anima.pleasure = (system.anima.pleasure + pleasure).clamp(-1.0, 1.0);
                    system.anima.arousal = (system.anima.arousal + arousal).clamp(-1.0, 1.0);
                    system.anima.dominance = (system.anima.dominance + dominance).clamp(-1.0, 1.0);
                    applied.push(AppliedEffect::AnimaModified);
                }
                DialogueEffect::ModifyStability(delta) => {
                    system.stability = (system.stability + delta).clamp(0.0, 1.0);
                    applied.push(AppliedEffect::StabilityModified);
                }
                DialogueEffect::ModifyDissociation(delta) => {
                    system.dissociation = (system.dissociation + delta).clamp(0.0, 1.0);
                    applied.push(AppliedEffect::DissociationModified);
                }
                DialogueEffect::RequestSwitch { alter_id, urgency } => {
                    system.request_switch(alter_id, *urgency, false);
                    applied.push(AppliedEffect::SwitchRequested(alter_id.clone()));
                }
                DialogueEffect::ActivateTrigger(trigger_id) => {
                    // Would create and add trigger
                    applied.push(AppliedEffect::TriggerActivated(trigger_id.clone()));
                }
                DialogueEffect::GiveItem(item_id) => {
                    applied.push(AppliedEffect::ItemGiven(item_id.clone()));
                }
                DialogueEffect::TakeItem(item_id) => {
                    applied.push(AppliedEffect::ItemTaken(item_id.clone()));
                }
                DialogueEffect::UnlockAbility(ability_id) => {
                    applied.push(AppliedEffect::AbilityUnlocked(ability_id.clone()));
                }
                DialogueEffect::ShiftReality { target, rate } => {
                    // Would begin reality transition
                    applied.push(AppliedEffect::RealityShifted(target.clone()));
                }
                DialogueEffect::PlaySound(sound_id) => {
                    applied.push(AppliedEffect::SoundPlayed(sound_id.clone()));
                }
                DialogueEffect::StartCutscene(cutscene_id) => {
                    applied.push(AppliedEffect::CutsceneStarted(cutscene_id.clone()));
                }
                DialogueEffect::GrantInsight { id, amount } => {
                    applied.push(AppliedEffect::InsightGranted(id.clone(), *amount));
                }
                DialogueEffect::EndDialogue => {
                    applied.push(AppliedEffect::EndDialogue);
                }
            }
        }

        Ok(applied)
    }

    // ========================================================================
    // HELPERS
    // ========================================================================

    /// Get the currently fronting alter's ID
    fn get_fronter_id(&self, system: &PluralSystem) -> Option<String> {
        match &system.fronting {
            FrontingState::Single(id) => Some(id.clone()),
            FrontingState::Blended(ids) => ids.first().cloned(),
            _ => None,
        }
    }
}

impl Default for DialogueManager {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// RESULT TYPES
// ============================================================================

/// Resolved dialogue for display
#[derive(Debug, Clone)]
pub struct ResolvedDialogue {
    /// Current node ID
    pub node_id: String,
    /// Resolved text to display
    pub text: String,
    /// Available responses
    pub responses: Vec<ResolvedResponse>,
    /// Speaker information
    pub speaker: ResolvedSpeaker,
    /// Voice cues for audio
    pub voice_cues: Vec<String>,
    /// Animations to trigger
    pub animations: Vec<String>,
}

/// Resolved response option
#[derive(Debug, Clone)]
pub struct ResolvedResponse {
    /// Response ID
    pub id: String,
    /// Display text
    pub text: String,
    /// Whether this is internal (system-only)
    pub internal: bool,
}

/// Resolved speaker info
#[derive(Debug, Clone)]
pub struct ResolvedSpeaker {
    /// Display name
    pub name: String,
    /// Portrait to show
    pub portrait: String,
}

/// Result of dialogue action
#[derive(Debug)]
pub enum DialogueResult {
    /// Continue to next node
    Continue(Vec<AppliedEffect>),
    /// Awaiting player choice
    AwaitingChoice(Vec<AppliedEffect>),
    /// Dialogue ended
    Ended,
}

/// Applied effect for tracking
#[derive(Debug, Clone)]
pub enum AppliedEffect {
    VariableSet(String),
    FlagSet(String),
    FlagCleared(String),
    AnimaModified,
    StabilityModified,
    DissociationModified,
    SwitchRequested(String),
    TriggerActivated(String),
    ItemGiven(String),
    ItemTaken(String),
    AbilityUnlocked(String),
    RealityShifted(RealityLayer),
    SoundPlayed(String),
    CutsceneStarted(String),
    InsightGranted(String, f32),
    EndDialogue,
}

// ============================================================================
// ERRORS
// ============================================================================

/// Dialogue system errors
#[derive(Debug)]
pub enum DialogueError {
    /// Tree not found
    TreeNotFound(String),
    /// No active dialogue
    NoActiveDialogue,
    /// Node not found
    NodeNotFound(String),
    /// Response not found
    ResponseNotFound(String),
    /// Conditions not met for action
    ConditionsNotMet,
}

impl std::fmt::Display for DialogueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DialogueError::TreeNotFound(id) => write!(f, "Dialogue tree not found: {}", id),
            DialogueError::NoActiveDialogue => write!(f, "No active dialogue"),
            DialogueError::NodeNotFound(id) => write!(f, "Dialogue node not found: {}", id),
            DialogueError::ResponseNotFound(id) => write!(f, "Response not found: {}", id),
            DialogueError::ConditionsNotMet => write!(f, "Conditions not met for this action"),
        }
    }
}

impl std::error::Error for DialogueError {}

// ============================================================================
// BUILDERS
// ============================================================================

/// Builder for creating dialogue trees
pub struct DialogueTreeBuilder {
    id: String,
    nodes: HashMap<String, DialogueNode>,
    entry_node: Option<String>,
    speaker: Option<SpeakerInfo>,
}

impl DialogueTreeBuilder {
    /// Create a new builder
    pub fn new(id: &str) -> Self {
        Self {
            id: id.to_string(),
            nodes: HashMap::new(),
            entry_node: None,
            speaker: None,
        }
    }

    /// Set the entry node
    pub fn entry(mut self, node_id: &str) -> Self {
        self.entry_node = Some(node_id.to_string());
        self
    }

    /// Set the speaker
    pub fn speaker(mut self, speaker: SpeakerInfo) -> Self {
        self.speaker = Some(speaker);
        self
    }

    /// Add a node
    pub fn node(mut self, node: DialogueNode) -> Self {
        self.nodes.insert(node.id.clone(), node);
        self
    }

    /// Build the dialogue tree
    pub fn build(self) -> Result<DialogueTree, String> {
        let entry_node = self.entry_node.ok_or("No entry node specified")?;
        let speaker = self.speaker.ok_or("No speaker specified")?;

        if !self.nodes.contains_key(&entry_node) {
            return Err(format!("Entry node '{}' not found in nodes", entry_node));
        }

        Ok(DialogueTree {
            id: self.id,
            nodes: self.nodes,
            entry_node,
            variables: HashMap::new(),
            speaker,
        })
    }
}

/// Builder for creating dialogue nodes
pub struct DialogueNodeBuilder {
    id: String,
    base_text: String,
    alter_variations: HashMap<String, AlterDialogueVariation>,
    layer_variations: HashMap<RealityLayer, String>,
    emotional_variations: Vec<EmotionalVariation>,
    conditions: Vec<DialogueCondition>,
    effects: Vec<DialogueEffect>,
    responses: Vec<DialogueResponse>,
    next: Option<String>,
    voice_cues: Vec<String>,
    animations: Vec<String>,
    tags: Vec<String>,
}

impl DialogueNodeBuilder {
    /// Create a new builder
    pub fn new(id: &str, text: &str) -> Self {
        Self {
            id: id.to_string(),
            base_text: text.to_string(),
            alter_variations: HashMap::new(),
            layer_variations: HashMap::new(),
            emotional_variations: Vec::new(),
            conditions: Vec::new(),
            effects: Vec::new(),
            responses: Vec::new(),
            next: None,
            voice_cues: Vec::new(),
            animations: Vec::new(),
            tags: Vec::new(),
        }
    }

    /// Add an alter-specific variation
    pub fn alter_variation(mut self, alter_id: &str, text: &str, tone: DialogueTone) -> Self {
        self.alter_variations.insert(
            alter_id.to_string(),
            AlterDialogueVariation {
                text: text.to_string(),
                observations: Vec::new(),
                tone,
                recognition: None,
            },
        );
        self
    }

    /// Add a reality layer variation
    pub fn layer_variation(mut self, layer: RealityLayer, text: &str) -> Self {
        self.layer_variations.insert(layer, text.to_string());
        self
    }

    /// Add a condition
    pub fn condition(mut self, condition: DialogueCondition) -> Self {
        self.conditions.push(condition);
        self
    }

    /// Add an effect
    pub fn effect(mut self, effect: DialogueEffect) -> Self {
        self.effects.push(effect);
        self
    }

    /// Add a response
    pub fn response(mut self, response: DialogueResponse) -> Self {
        self.responses.push(response);
        self
    }

    /// Set next node for linear flow
    pub fn next(mut self, node_id: &str) -> Self {
        self.next = Some(node_id.to_string());
        self
    }

    /// Add a voice cue
    pub fn voice_cue(mut self, cue: &str) -> Self {
        self.voice_cues.push(cue.to_string());
        self
    }

    /// Add an animation
    pub fn animation(mut self, anim: &str) -> Self {
        self.animations.push(anim.to_string());
        self
    }

    /// Add a tag
    pub fn tag(mut self, tag: &str) -> Self {
        self.tags.push(tag.to_string());
        self
    }

    /// Build the node
    pub fn build(self) -> DialogueNode {
        DialogueNode {
            id: self.id,
            content: DialogueContent {
                base_text: self.base_text,
                alter_variations: self.alter_variations,
                layer_variations: self.layer_variations,
                emotional_variations: self.emotional_variations,
                voice_cues: self.voice_cues,
                animations: self.animations,
            },
            conditions: self.conditions,
            effects: self.effects,
            responses: self.responses,
            next: self.next,
            tags: self.tags,
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_system() -> PluralSystem {
        let mut system = PluralSystem::new("Test System");
        system.add_alter(super::super::runtime::Alter {
            id: "host".to_string(),
            name: "Host".to_string(),
            category: AlterCategory::Council,
            state: AlterPresenceState::Fronting,
            anima: AnimaState::default(),
            base_arousal: 0.0,
            base_dominance: 0.0,
            time_since_front: 0,
            triggers: Vec::new(),
            abilities: std::collections::HashSet::from(["analysis".to_string()]),
            preferred_reality: RealityLayer::Grounded,
            memory_access: super::super::runtime::MemoryAccess::Full,
        });
        system.fronting = FrontingState::Single("host".to_string());
        system
    }

    fn create_test_tree() -> DialogueTree {
        let speaker = SpeakerInfo {
            id: "marcus".to_string(),
            name: "Father Marcus".to_string(),
            portrait: "marcus_default".to_string(),
            alter_portraits: HashMap::new(),
            layer_portraits: HashMap::new(),
        };

        let node1 = DialogueNodeBuilder::new("start", "Greetings, traveler.")
            .alter_variation(
                "host",
                "Welcome, child. I sense... complexity within you.",
                DialogueTone::Warm,
            )
            .response(DialogueResponse {
                id: "ask_help".to_string(),
                text: "I need your help.".to_string(),
                alter_variations: HashMap::new(),
                conditions: Vec::new(),
                target_node: "help_offered".to_string(),
                effects: Vec::new(),
                internal: false,
                required_traits: Vec::new(),
                forbidden_traits: Vec::new(),
            })
            .response(DialogueResponse {
                id: "analyze".to_string(),
                text: "[Analyze] Something seems off about you...".to_string(),
                alter_variations: HashMap::new(),
                conditions: vec![DialogueCondition::AlterHasTrait("analysis".to_string())],
                target_node: "analysis_result".to_string(),
                effects: Vec::new(),
                internal: false,
                required_traits: vec!["analysis".to_string()],
                forbidden_traits: Vec::new(),
            })
            .build();

        let node2 = DialogueNodeBuilder::new("help_offered", "How may I assist you?")
            .effect(DialogueEffect::EndDialogue)
            .build();

        let node3 =
            DialogueNodeBuilder::new("analysis_result", "You... you can see it, can't you?")
                .layer_variation(
                    RealityLayer::Fractured,
                    "The priest's form wavers. Behind his smile, shadow teeth.",
                )
                .effect(DialogueEffect::ModifyAnima {
                    pleasure: -0.1,
                    arousal: 0.2,
                    dominance: 0.0,
                })
                .effect(DialogueEffect::EndDialogue)
                .build();

        DialogueTreeBuilder::new("marcus_greeting")
            .entry("start")
            .speaker(speaker)
            .node(node1)
            .node(node2)
            .node(node3)
            .build()
            .unwrap()
    }

    #[test]
    fn test_dialogue_manager_creation() {
        let manager = DialogueManager::new();
        assert!(!manager.is_active());
    }

    #[test]
    fn test_start_dialogue() {
        let mut manager = DialogueManager::new();
        manager.load_tree(create_test_tree());

        let result = manager.start_dialogue("marcus_greeting");
        assert!(result.is_ok());
        assert!(manager.is_active());
    }

    #[test]
    fn test_get_current_content() {
        let mut manager = DialogueManager::new();
        manager.load_tree(create_test_tree());
        manager.start_dialogue("marcus_greeting").unwrap();

        let system = create_test_system();
        let content = manager.get_current_content(&system).unwrap();

        // Should get alter variation since host is fronting
        assert!(content.text.contains("Welcome, child"));
        assert_eq!(content.responses.len(), 2);
    }

    #[test]
    fn test_select_response() {
        let mut manager = DialogueManager::new();
        manager.load_tree(create_test_tree());
        manager.start_dialogue("marcus_greeting").unwrap();

        let mut system = create_test_system();
        // Selecting response moves to target node
        let result = manager.select_response("ask_help", &mut system);
        assert!(matches!(result, Ok(DialogueResult::Continue(_))));

        // Advancing executes the target node's effects (EndDialogue)
        let result = manager.advance(&mut system);
        assert!(matches!(result, Ok(DialogueResult::Ended)));
    }

    #[test]
    fn test_condition_checking() {
        let manager = DialogueManager::new();
        let system = create_test_system();

        // Test alter fronting condition
        let condition = DialogueCondition::AlterFronting("host".to_string());
        assert!(manager.check_condition(&condition, &system));

        // Test alter has trait condition
        let condition = DialogueCondition::AlterHasTrait("analysis".to_string());
        assert!(manager.check_condition(&condition, &system));

        // Test reality layer condition
        let condition = DialogueCondition::RealityLayer(RealityLayer::Grounded);
        assert!(manager.check_condition(&condition, &system));
    }

    #[test]
    fn test_trait_filtered_responses() {
        let mut manager = DialogueManager::new();
        manager.load_tree(create_test_tree());
        manager.start_dialogue("marcus_greeting").unwrap();

        let system = create_test_system();
        let content = manager.get_current_content(&system).unwrap();

        // Should see the analyze response since host has "analysis" trait
        let has_analyze = content.responses.iter().any(|r| r.id == "analyze");
        assert!(has_analyze);
    }
}
