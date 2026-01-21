//! # Game Loop for DAEMONIORUM
//!
//! The core game loop integrating plural system, combat, perception,
//! and narrative progression.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use super::combat::{CombatResult, CombatState, Enemy};
use super::perception::PerceptionManager;
use super::runtime::{
    Alter, AlterCategory, AlterPresenceState, AnimaState, FrontingState, MemoryAccess,
    PluralSystem, RealityLayer, Trigger, TriggerCategory,
};

// ============================================================================
// GAME STATE
// ============================================================================

/// The complete game state
#[derive(Debug)]
pub struct GameState {
    /// The plural system (player)
    pub system: PluralSystem,
    /// Current perception state
    pub perception: PerceptionManager,
    /// Active combat (if any)
    pub combat: Option<CombatState>,
    /// Current scene/location
    pub scene: Scene,
    /// Game time (in-game seconds)
    pub game_time: u64,
    /// Real time tracking
    pub real_time: Instant,
    /// Game phase
    pub phase: GamePhase,
    /// Event queue
    pub events: Vec<GameEvent>,
    /// Narrative flags
    pub flags: HashMap<String, FlagValue>,
    /// Player inventory
    pub inventory: Vec<Item>,
    /// Unlocked abilities
    pub unlocked_abilities: Vec<String>,
    /// Processed traumas (for progression)
    pub processed_traumas: Vec<String>,
}

impl GameState {
    /// Create a new game with initial setup
    pub fn new() -> Self {
        let mut system = PluralSystem::new("The Council");

        // Initialize with starter alters
        system.add_alter(create_host_alter());
        system.add_alter(create_protector_alter());
        system.fronting = FrontingState::Single("host".to_string());

        Self {
            system,
            perception: PerceptionManager::new(),
            combat: None,
            scene: Scene::default(),
            game_time: 0,
            real_time: Instant::now(),
            phase: GamePhase::Exploration,
            events: Vec::new(),
            flags: HashMap::new(),
            inventory: Vec::new(),
            unlocked_abilities: Vec::new(),
            processed_traumas: Vec::new(),
        }
    }

    /// Load game from save data
    pub fn from_save(save: SaveData) -> Self {
        Self {
            system: save.system,
            perception: PerceptionManager::new(),
            combat: None,
            scene: save.scene,
            game_time: save.game_time,
            real_time: Instant::now(),
            phase: GamePhase::Exploration,
            events: Vec::new(),
            flags: save.flags,
            inventory: save.inventory,
            unlocked_abilities: save.unlocked_abilities,
            processed_traumas: save.processed_traumas,
        }
    }

    /// Create save data
    pub fn to_save(&self) -> SaveData {
        SaveData {
            system: self.system.clone(),
            scene: self.scene.clone(),
            game_time: self.game_time,
            flags: self.flags.clone(),
            inventory: self.inventory.clone(),
            unlocked_abilities: self.unlocked_abilities.clone(),
            processed_traumas: self.processed_traumas.clone(),
        }
    }
}

/// Game phases
#[derive(Debug, Clone, PartialEq)]
pub enum GamePhase {
    /// Free exploration
    Exploration,
    /// In dialogue
    Dialogue(DialogueState),
    /// In combat
    Combat,
    /// Cutscene/narrative
    Cutscene,
    /// Headspace navigation
    Headspace,
    /// Paused
    Paused,
    /// Game over
    GameOver(GameOverReason),
}

/// Reasons for game over
#[derive(Debug, Clone, PartialEq)]
pub enum GameOverReason {
    /// System completely destabilized
    SystemCollapse,
    /// Player chose to end
    PlayerChoice,
    /// Story ending reached
    Ending(String),
}

// ============================================================================
// GAME LOOP
// ============================================================================

/// The main game loop
pub struct GameLoop {
    /// Current game state
    pub state: GameState,
    /// Target updates per second
    pub target_ups: u32,
    /// Is the game running?
    pub running: bool,
    /// Pending input
    pub input_queue: Vec<PlayerInput>,
}

impl GameLoop {
    /// Create a new game loop
    pub fn new() -> Self {
        Self {
            state: GameState::new(),
            target_ups: 60,
            running: true,
            input_queue: Vec::new(),
        }
    }

    /// Run the game loop
    pub fn run(&mut self) {
        let frame_duration = Duration::from_secs(1) / self.target_ups;
        let mut last_update = Instant::now();

        while self.running {
            let now = Instant::now();
            let delta = now.duration_since(last_update);

            if delta >= frame_duration {
                self.update(delta);
                last_update = now;
            }

            // Small sleep to prevent CPU spinning
            std::thread::sleep(Duration::from_millis(1));
        }
    }

    /// Single update tick
    pub fn update(&mut self, delta: Duration) {
        // Process input
        self.process_input();

        // Update based on phase
        match &self.state.phase {
            GamePhase::Exploration => self.update_exploration(delta),
            GamePhase::Dialogue(_) => self.update_dialogue(delta),
            GamePhase::Combat => self.update_combat(delta),
            GamePhase::Cutscene => self.update_cutscene(delta),
            GamePhase::Headspace => self.update_headspace(delta),
            GamePhase::Paused => {}
            GamePhase::GameOver(_) => {}
        }

        // Always update core systems
        self.update_plural_system(delta);
        self.update_perception(delta);
        self.process_events();

        // Update game time
        self.state.game_time += delta.as_millis() as u64 / 100; // 10x faster than real time

        // Check for game over conditions
        self.check_game_over();
    }

    /// Process player input
    fn process_input(&mut self) {
        while let Some(input) = self.input_queue.pop() {
            match input {
                PlayerInput::Move(direction) => {
                    if self.state.phase == GamePhase::Exploration {
                        self.handle_movement(direction);
                    }
                }
                PlayerInput::Interact => {
                    self.handle_interaction();
                }
                PlayerInput::OpenMenu(menu) => {
                    self.open_menu(menu);
                }
                PlayerInput::SwitchAlter(alter_id) => {
                    self.request_alter_switch(&alter_id);
                }
                PlayerInput::UseAbility(ability_id) => {
                    self.use_ability(&ability_id);
                }
                PlayerInput::Ground => {
                    self.attempt_grounding();
                }
                PlayerInput::Pause => {
                    self.toggle_pause();
                }
                PlayerInput::DialogueChoice(choice) => {
                    if let GamePhase::Dialogue(ref mut dialogue) = self.state.phase {
                        self.select_dialogue_choice(choice);
                    }
                }
                PlayerInput::CombatAction(action) => {
                    if self.state.phase == GamePhase::Combat {
                        self.execute_combat_action(action);
                    }
                }
            }
        }
    }

    /// Update during exploration phase
    fn update_exploration(&mut self, delta: Duration) {
        // Update scene entities
        for entity in &mut self.state.scene.entities {
            entity.update(delta);
        }

        // Check for trigger zones
        self.check_trigger_zones();

        // Check for combat encounters
        if let Some(enemy) = self.check_enemy_encounter() {
            self.start_combat(enemy);
        }

        // Check for interactive objects
        self.update_interactables();
    }

    /// Update during combat
    fn update_combat(&mut self, _delta: Duration) {
        // Get combat result if combat is over
        let combat_result = if let Some(ref mut combat) = self.state.combat {
            // Combat is turn-based, so this mainly handles animations/effects
            combat.check_combat_end();
            if combat.is_over {
                combat.result.clone()
            } else {
                None
            }
        } else {
            None
        };

        // End combat if result is set
        if let Some(result) = combat_result {
            self.end_combat(Some(result));
        }
    }

    /// Update during dialogue
    fn update_dialogue(&mut self, _delta: Duration) {
        // Dialogue is event-driven, minimal updates needed
    }

    /// Update during cutscene
    fn update_cutscene(&mut self, delta: Duration) {
        // Advance cutscene based on timing
        if let Some(cutscene) = &mut self.state.scene.active_cutscene {
            cutscene.advance(delta);
            if cutscene.is_complete() {
                self.end_cutscene();
            }
        }
    }

    /// Update headspace navigation
    fn update_headspace(&mut self, delta: Duration) {
        // Update headspace-specific logic
        self.update_inner_world(delta);
    }

    /// Update the plural system
    fn update_plural_system(&mut self, delta: Duration) {
        let delta_secs = delta.as_secs_f32();

        // Update alter time tracking
        for alter in self.state.system.alters.values_mut() {
            if !matches!(alter.state, AlterPresenceState::Fronting) {
                alter.time_since_front += (delta_secs * 1000.0) as u64;
            }
        }

        // Natural stability recovery (when safe)
        if self.state.phase == GamePhase::Exploration && self.state.scene.safety_level > 0.5 {
            self.state.system.stability += delta_secs * 0.01;
            self.state.system.stability = self.state.system.stability.min(1.0);
        }

        // Natural dissociation decay
        if self.state.system.dissociation > 0.0 {
            self.state.system.dissociation -= delta_secs * 0.005;
            self.state.system.dissociation = self.state.system.dissociation.max(0.0);
        }

        // Update blended anima
        self.state.system.update_blended_anima();

        // Process any active triggers
        let triggers: Vec<_> = self.state.system.active_triggers.drain(..).collect();
        for trigger in triggers {
            let result = self.state.system.process_trigger(trigger);
            self.handle_trigger_result(result);
        }
    }

    /// Update perception system
    fn update_perception(&mut self, _delta: Duration) {
        self.state.perception.update(&self.state.system);

        // Process perception triggers
        let triggers = self.state.perception.drain_triggers();
        for trigger in triggers {
            self.state.system.active_triggers.push(trigger);
        }

        // Reality layer affects gameplay
        match self.state.perception.state.layer {
            RealityLayer::Grounded => {
                // Normal gameplay
            }
            RealityLayer::Fractured => {
                // Reveal hidden entities/clues
                self.reveal_fractured_content();
            }
            RealityLayer::Shattered => {
                // Potential danger, unique interactions
                self.handle_shattered_reality();
            }
            RealityLayer::Custom(_) => {}
        }
    }

    /// Process queued events
    fn process_events(&mut self) {
        let events: Vec<_> = self.state.events.drain(..).collect();
        for event in events {
            self.handle_event(event);
        }
    }

    /// Check for game over conditions
    fn check_game_over(&mut self) {
        // System collapse
        if self.state.system.stability <= 0.0 && self.state.system.dissociation >= 1.0 {
            self.state.phase = GamePhase::GameOver(GameOverReason::SystemCollapse);
        }
    }

    // ========================================================================
    // ACTION HANDLERS
    // ========================================================================

    /// Handle movement input
    fn handle_movement(&mut self, direction: Direction) {
        // Update player position in scene
        self.state.scene.move_player(direction);

        // Check for zone transitions
        if let Some(transition) = self.state.scene.check_transition() {
            self.transition_scene(transition);
        }
    }

    /// Handle interaction with nearby objects
    fn handle_interaction(&mut self) {
        // Clone the interactable to avoid borrow issues
        let interaction = self.state.scene.nearest_interactable().cloned();

        if let Some(target) = interaction {
            match &target.interaction_type {
                InteractionType::Examine => {
                    self.show_examination(&target);
                }
                InteractionType::Talk(npc_id) => {
                    self.start_dialogue(npc_id.clone());
                }
                InteractionType::PickUp(item) => {
                    self.pick_up_item(item.clone());
                }
                InteractionType::Use => {
                    self.use_object(&target);
                }
                InteractionType::Enter(scene_id) => {
                    self.transition_scene(scene_id.clone());
                }
            }
        }
    }

    /// Request an alter switch
    fn request_alter_switch(&mut self, alter_id: &str) {
        let urgency = if self.state.phase == GamePhase::Combat {
            0.8
        } else {
            0.5
        };
        let result = self.state.system.request_switch(alter_id, urgency, false);

        match result {
            super::runtime::SwitchResult::Success => {
                self.state
                    .events
                    .push(GameEvent::AlterSwitched(alter_id.to_string()));

                // Reality might shift based on new fronter
                if let Some(alter) = self.state.system.alters.get(alter_id) {
                    self.state.perception.state.layer = alter.preferred_reality.clone();
                }
            }
            super::runtime::SwitchResult::Resisted { resistance } => {
                self.state
                    .events
                    .push(GameEvent::SwitchResisted(resistance));
            }
            super::runtime::SwitchResult::Failed(reason) => {
                self.state
                    .events
                    .push(GameEvent::SwitchFailed(format!("{:?}", reason)));
            }
            _ => {}
        }
    }

    /// Attempt a grounding exercise
    fn attempt_grounding(&mut self) {
        // Grounding helps with dissociation and reality stability
        let success_chance = 0.5 + self.state.system.stability * 0.3;

        if rand_float() < success_chance {
            self.state.system.dissociation -= 0.2;
            self.state.system.dissociation = self.state.system.dissociation.max(0.0);

            // Move toward grounded reality
            if self.state.perception.state.layer != RealityLayer::Grounded {
                self.state
                    .perception
                    .begin_layer_transition(RealityLayer::Grounded, 0.1);
            }

            self.state.events.push(GameEvent::GroundingSuccess);
        } else {
            self.state.events.push(GameEvent::GroundingFailed);
        }
    }

    /// Start combat with an enemy
    fn start_combat(&mut self, enemy: Enemy) {
        let combat = CombatState::new(&self.state.system, vec![enemy]);
        self.state.combat = Some(combat);
        self.state.phase = GamePhase::Combat;
        self.state.events.push(GameEvent::CombatStarted);
    }

    /// End combat
    fn end_combat(&mut self, result: Option<CombatResult>) {
        self.state.combat = None;
        self.state.phase = GamePhase::Exploration;

        match result {
            Some(CombatResult::Victory) => {
                self.state.events.push(GameEvent::CombatVictory);
            }
            Some(CombatResult::Defeat) => {
                // Handle defeat (not game over, system protects itself)
                self.handle_combat_defeat();
            }
            Some(CombatResult::Flee) => {
                self.state.events.push(GameEvent::CombatFled);
            }
            _ => {}
        }
    }

    /// Handle combat defeat
    fn handle_combat_defeat(&mut self) {
        // Emergency switch to protector
        if let Some(protector) = self.find_protector() {
            self.state.system.request_switch(&protector, 1.0, true);
        }

        // Increase dissociation
        self.state.system.dissociation += 0.3;

        // Retreat to safer area
        self.state.events.push(GameEvent::EmergencyRetreat);
    }

    /// Start dialogue with an NPC
    fn start_dialogue(&mut self, npc_id: String) {
        let dialogue_state = DialogueState {
            npc_id,
            current_node: "start".to_string(),
            history: Vec::new(),
        };
        self.state.phase = GamePhase::Dialogue(dialogue_state);
        self.state.events.push(GameEvent::DialogueStarted);
    }

    /// Handle trigger result
    fn handle_trigger_result(&mut self, result: super::runtime::TriggerResult) {
        match result {
            super::runtime::TriggerResult::ForcedSwitch(alter_id) => {
                self.state.events.push(GameEvent::ForcedSwitch(alter_id));
            }
            super::runtime::TriggerResult::Activation(alters) => {
                for alter_id in alters {
                    self.state.events.push(GameEvent::AlterActivated(alter_id));
                }
            }
            super::runtime::TriggerResult::Dissociation => {
                self.state.system.dissociation += 0.2;
                self.state.events.push(GameEvent::DissociationSpike);
            }
            super::runtime::TriggerResult::NoResponse => {}
        }
    }

    /// Handle a game event
    fn handle_event(&mut self, event: GameEvent) {
        // Events can be processed by UI, narrative system, etc.
        // This is the hook for external systems
        match event {
            GameEvent::AlterSwitched(id) => {
                // Update UI, play animation, etc.
            }
            GameEvent::CombatStarted => {
                // Transition music, UI, etc.
            }
            GameEvent::TraumaProcessed(id) => {
                self.state.processed_traumas.push(id);
            }
            _ => {}
        }
    }

    // ========================================================================
    // HELPER METHODS
    // ========================================================================

    fn find_protector(&self) -> Option<String> {
        self.state
            .system
            .alters
            .values()
            .find(|a| a.abilities.contains("protection") || a.abilities.contains("combat"))
            .map(|a| a.id.clone())
    }

    fn check_trigger_zones(&mut self) {
        // Check if player is in a trigger zone
        for zone in &self.state.scene.trigger_zones {
            if zone.contains(self.state.scene.player_position) {
                let trigger = Trigger {
                    id: zone.trigger_id.clone(),
                    name: zone.name.clone(),
                    category: TriggerCategory::Environmental,
                    intensity: zone.intensity,
                    context: HashMap::new(),
                };
                self.state.system.active_triggers.push(trigger);
            }
        }
    }

    fn check_enemy_encounter(&self) -> Option<Enemy> {
        // Check for enemy encounters in current scene
        None // Placeholder
    }

    fn reveal_fractured_content(&mut self) {
        // In Fractured reality, hidden content becomes visible
    }

    fn handle_shattered_reality(&mut self) {
        // Shattered reality has unique dangers
        if rand_float() < 0.01 {
            self.state.system.dissociation += 0.05;
        }
    }

    fn update_inner_world(&mut self, _delta: Duration) {
        // Headspace-specific updates
    }

    fn show_examination(&mut self, _target: &Interactable) {}
    fn pick_up_item(&mut self, item: Item) {
        self.state.inventory.push(item);
    }
    fn use_object(&mut self, _target: &Interactable) {}
    fn transition_scene(&mut self, _scene_id: String) {}
    fn update_interactables(&mut self) {}
    fn open_menu(&mut self, _menu: MenuType) {}
    fn use_ability(&mut self, _ability_id: &str) {}
    fn toggle_pause(&mut self) {
        if self.state.phase == GamePhase::Paused {
            self.state.phase = GamePhase::Exploration;
        } else {
            self.state.phase = GamePhase::Paused;
        }
    }
    fn select_dialogue_choice(&mut self, _choice: usize) {}
    fn execute_combat_action(&mut self, _action: CombatAction) {}
    fn end_cutscene(&mut self) {
        self.state.phase = GamePhase::Exploration;
    }
}

// ============================================================================
// SUPPORTING TYPES
// ============================================================================

/// Player input types
#[derive(Debug, Clone)]
pub enum PlayerInput {
    Move(Direction),
    Interact,
    OpenMenu(MenuType),
    SwitchAlter(String),
    UseAbility(String),
    Ground,
    Pause,
    DialogueChoice(usize),
    CombatAction(CombatAction),
}

/// Movement directions
#[derive(Debug, Clone, Copy)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

/// Menu types
#[derive(Debug, Clone)]
pub enum MenuType {
    System,
    Inventory,
    Alters,
    Map,
}

/// Combat actions
#[derive(Debug, Clone)]
pub enum CombatAction {
    Attack(usize),
    Defend,
    UseItem(usize),
    Switch(String),
    Flee,
}

/// Game events for external systems
#[derive(Debug, Clone)]
pub enum GameEvent {
    AlterSwitched(String),
    AlterActivated(String),
    ForcedSwitch(String),
    SwitchResisted(f32),
    SwitchFailed(String),
    CombatStarted,
    CombatVictory,
    CombatDefeat,
    CombatFled,
    EmergencyRetreat,
    DialogueStarted,
    DialogueEnded,
    GroundingSuccess,
    GroundingFailed,
    DissociationSpike,
    RealityShift(RealityLayer),
    TraumaProcessed(String),
    ItemAcquired(String),
    AbilityUnlocked(String),
}

/// Dialogue state
#[derive(Debug, Clone, PartialEq)]
pub struct DialogueState {
    pub npc_id: String,
    pub current_node: String,
    pub history: Vec<String>,
}

/// Scene data
#[derive(Debug, Clone, Default)]
pub struct Scene {
    pub id: String,
    pub name: String,
    pub safety_level: f32,
    pub entities: Vec<SceneEntity>,
    pub trigger_zones: Vec<TriggerZone>,
    pub interactables: Vec<Interactable>,
    pub player_position: (f32, f32),
    pub active_cutscene: Option<Cutscene>,
}

impl Scene {
    fn move_player(&mut self, direction: Direction) {
        let speed = 5.0;
        match direction {
            Direction::Up => self.player_position.1 -= speed,
            Direction::Down => self.player_position.1 += speed,
            Direction::Left => self.player_position.0 -= speed,
            Direction::Right => self.player_position.0 += speed,
        }
    }

    fn check_transition(&self) -> Option<String> {
        None
    }

    fn nearest_interactable(&self) -> Option<&Interactable> {
        None
    }
}

/// Scene entity
#[derive(Debug, Clone)]
pub struct SceneEntity {
    pub id: String,
    pub position: (f32, f32),
    pub entity_type: String,
}

impl SceneEntity {
    fn update(&mut self, _delta: Duration) {}
}

/// Trigger zone in scene
#[derive(Debug, Clone)]
pub struct TriggerZone {
    pub name: String,
    pub trigger_id: String,
    pub bounds: ((f32, f32), (f32, f32)),
    pub intensity: f32,
}

impl TriggerZone {
    fn contains(&self, pos: (f32, f32)) -> bool {
        pos.0 >= self.bounds.0 .0
            && pos.0 <= self.bounds.1 .0
            && pos.1 >= self.bounds.0 .1
            && pos.1 <= self.bounds.1 .1
    }
}

/// Interactable object
#[derive(Debug, Clone)]
pub struct Interactable {
    pub id: String,
    pub position: (f32, f32),
    pub interaction_type: InteractionType,
}

/// Interaction types
#[derive(Debug, Clone)]
pub enum InteractionType {
    Examine,
    Talk(String),
    PickUp(Item),
    Use,
    Enter(String),
}

/// Inventory item
#[derive(Debug, Clone)]
pub struct Item {
    pub id: String,
    pub name: String,
    pub description: String,
    pub item_type: ItemType,
}

/// Item types
#[derive(Debug, Clone)]
pub enum ItemType {
    Key,
    Consumable,
    Document,
    Memento,
}

/// Cutscene data
#[derive(Debug, Clone)]
pub struct Cutscene {
    pub id: String,
    pub duration: Duration,
    pub elapsed: Duration,
}

impl Cutscene {
    fn advance(&mut self, delta: Duration) {
        self.elapsed += delta;
    }

    fn is_complete(&self) -> bool {
        self.elapsed >= self.duration
    }
}

/// Save data
#[derive(Debug, Clone)]
pub struct SaveData {
    pub system: PluralSystem,
    pub scene: Scene,
    pub game_time: u64,
    pub flags: HashMap<String, FlagValue>,
    pub inventory: Vec<Item>,
    pub unlocked_abilities: Vec<String>,
    pub processed_traumas: Vec<String>,
}

/// Flag values for narrative state
#[derive(Debug, Clone)]
pub enum FlagValue {
    Bool(bool),
    Int(i32),
    String(String),
}

// ============================================================================
// INITIALIZATION HELPERS
// ============================================================================

fn create_host_alter() -> Alter {
    use std::collections::HashSet;
    Alter {
        id: "host".to_string(),
        name: "Host".to_string(),
        category: AlterCategory::Council,
        state: AlterPresenceState::Fronting,
        anima: AnimaState::default(),
        base_arousal: 0.0,
        base_dominance: 0.0,
        time_since_front: 0,
        triggers: vec![],
        abilities: HashSet::from(["perception".to_string()]),
        preferred_reality: RealityLayer::Grounded,
        memory_access: MemoryAccess::Partial(vec!["recent".to_string()]),
    }
}

fn create_protector_alter() -> Alter {
    use std::collections::HashSet;
    Alter {
        id: "protector".to_string(),
        name: "Protector".to_string(),
        category: AlterCategory::Council,
        state: AlterPresenceState::Dormant,
        anima: AnimaState::new(0.0, 0.5, 0.7),
        base_arousal: 0.5,
        base_dominance: 0.7,
        time_since_front: 1000,
        triggers: vec!["threat".to_string(), "danger".to_string()],
        abilities: HashSet::from(["combat".to_string(), "protection".to_string()]),
        preferred_reality: RealityLayer::Grounded,
        memory_access: MemoryAccess::Full,
    }
}

/// Placeholder random function
fn rand_float() -> f32 {
    // In real implementation, use proper RNG
    0.5
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_game_state_new() {
        let state = GameState::new();
        assert!(state.system.alters.contains_key("host"));
        assert!(state.system.alters.contains_key("protector"));
        assert_eq!(state.phase, GamePhase::Exploration);
    }

    #[test]
    fn test_game_loop_creation() {
        let game_loop = GameLoop::new();
        assert!(game_loop.running);
        assert_eq!(game_loop.target_ups, 60);
    }

    #[test]
    fn test_alter_switch_in_game() {
        let mut game_loop = GameLoop::new();
        game_loop.request_alter_switch("protector");

        // Check that an event was generated
        assert!(!game_loop.state.events.is_empty());
    }

    #[test]
    fn test_grounding_mechanic() {
        let mut game_loop = GameLoop::new();
        game_loop.state.system.dissociation = 0.5;
        game_loop.state.system.stability = 0.8;

        game_loop.attempt_grounding();

        // Event should be generated
        assert!(!game_loop.state.events.is_empty());
    }
}
