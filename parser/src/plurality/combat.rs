//! # Combat System for DAEMONIORUM
//!
//! Combat mechanics that integrate with plurality, alter switching,
//! and reality perception systems.

use std::collections::HashMap;

use super::runtime::{
    Alter, AlterPresenceState, AnimaState, FrontingState, PluralSystem, RealityLayer,
    SwitchResult, Trigger, TriggerCategory, TriggerResult,
};

// ============================================================================
// COMBAT STATE
// ============================================================================

/// The current combat encounter state
#[derive(Debug, Clone)]
pub struct CombatState {
    /// Active combatants
    pub combatants: Vec<Combatant>,
    /// Current turn order
    pub turn_order: Vec<usize>,
    /// Current active combatant index
    pub current_turn: usize,
    /// Combat round number
    pub round: u32,
    /// Environmental factors
    pub environment: CombatEnvironment,
    /// Active combat effects
    pub effects: Vec<CombatEffect>,
    /// Combat phase
    pub phase: CombatPhase,
    /// Is combat over?
    pub is_over: bool,
    /// Victory/defeat result (if over)
    pub result: Option<CombatResult>,
}

impl CombatState {
    /// Create a new combat encounter
    pub fn new(player_system: &PluralSystem, enemies: Vec<Enemy>) -> Self {
        let mut combatants = vec![Combatant::Player(PlayerCombatant::from_system(player_system))];
        combatants.extend(enemies.into_iter().map(Combatant::Enemy));

        let mut state = Self {
            combatants,
            turn_order: Vec::new(),
            current_turn: 0,
            round: 1,
            environment: CombatEnvironment::default(),
            effects: Vec::new(),
            phase: CombatPhase::PreCombat,
            is_over: false,
            result: None,
        };

        state.calculate_turn_order();
        state
    }

    /// Calculate turn order based on initiative
    fn calculate_turn_order(&mut self) {
        let mut order: Vec<(usize, f32)> = self
            .combatants
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.initiative()))
            .collect();

        // Sort by initiative (highest first)
        order.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        self.turn_order = order.into_iter().map(|(i, _)| i).collect();
    }

    /// Get the current active combatant
    pub fn current_combatant(&self) -> Option<&Combatant> {
        self.turn_order
            .get(self.current_turn)
            .and_then(|&idx| self.combatants.get(idx))
    }

    /// Get the current active combatant mutably
    pub fn current_combatant_mut(&mut self) -> Option<&mut Combatant> {
        let idx = *self.turn_order.get(self.current_turn)?;
        self.combatants.get_mut(idx)
    }

    /// Advance to the next turn
    pub fn next_turn(&mut self) {
        self.current_turn += 1;
        if self.current_turn >= self.turn_order.len() {
            self.current_turn = 0;
            self.round += 1;
            self.phase = CombatPhase::Combat;
        }
    }

    /// Check if combat should end
    pub fn check_combat_end(&mut self) {
        let player_alive = self.combatants.iter().any(|c| {
            matches!(c, Combatant::Player(p) if p.health > 0.0)
        });

        let enemies_alive = self.combatants.iter().any(|c| {
            matches!(c, Combatant::Enemy(e) if e.health > 0.0)
        });

        if !player_alive {
            self.is_over = true;
            self.result = Some(CombatResult::Defeat);
        } else if !enemies_alive {
            self.is_over = true;
            self.result = Some(CombatResult::Victory);
        }
    }
}

/// Combat phases
#[derive(Debug, Clone, PartialEq)]
pub enum CombatPhase {
    /// Before combat starts (preparation)
    PreCombat,
    /// Active combat
    Combat,
    /// Special event during combat
    Event,
    /// Combat is ending
    PostCombat,
}

/// Combat result
#[derive(Debug, Clone, PartialEq)]
pub enum CombatResult {
    Victory,
    Defeat,
    Flee,
    Negotiated,
}

// ============================================================================
// COMBATANTS
// ============================================================================

/// A participant in combat
#[derive(Debug, Clone)]
pub enum Combatant {
    Player(PlayerCombatant),
    Enemy(Enemy),
}

impl Combatant {
    /// Get the combatant's initiative value
    pub fn initiative(&self) -> f32 {
        match self {
            Combatant::Player(p) => p.initiative(),
            Combatant::Enemy(e) => e.initiative,
        }
    }

    /// Get the combatant's name
    pub fn name(&self) -> &str {
        match self {
            Combatant::Player(p) => &p.fronting_alter,
            Combatant::Enemy(e) => &e.name,
        }
    }
}

/// Player combatant (the plural system)
#[derive(Debug, Clone)]
pub struct PlayerCombatant {
    /// Currently fronting alter name
    pub fronting_alter: String,
    /// Current health (0.0 to max_health)
    pub health: f32,
    /// Maximum health
    pub max_health: f32,
    /// Current psyche (mental/emotional resource)
    pub psyche: f32,
    /// Maximum psyche
    pub max_psyche: f32,
    /// Current anima state
    pub anima: AnimaState,
    /// Current reality perception
    pub reality: RealityLayer,
    /// Available abilities for current fronter
    pub abilities: Vec<CombatAbility>,
    /// Status effects
    pub status_effects: Vec<StatusEffect>,
    /// Alter switching cooldown (turns remaining)
    pub switch_cooldown: u32,
    /// Available alters that can switch in
    pub available_alters: Vec<String>,
    /// Combat-specific triggers active
    pub active_triggers: Vec<String>,
}

impl PlayerCombatant {
    /// Create from a plural system
    pub fn from_system(system: &PluralSystem) -> Self {
        let fronting = match &system.fronting {
            FrontingState::Single(id) => id.clone(),
            FrontingState::Blended(ids) => ids.first().cloned().unwrap_or_default(),
            _ => "Unknown".to_string(),
        };

        let fronter = system.alters.get(&fronting);
        let abilities = fronter
            .map(|a| generate_combat_abilities(a))
            .unwrap_or_default();

        let available: Vec<String> = system
            .alters
            .values()
            .filter(|a| {
                !matches!(a.state, AlterPresenceState::Dormant | AlterPresenceState::Dissociating)
            })
            .map(|a| a.id.clone())
            .collect();

        Self {
            fronting_alter: fronting,
            health: 100.0,
            max_health: 100.0,
            psyche: 50.0,
            max_psyche: 50.0,
            anima: system.anima.clone(),
            reality: system.reality_layer.clone(),
            abilities,
            status_effects: Vec::new(),
            switch_cooldown: 0,
            available_alters: available,
            active_triggers: Vec::new(),
        }
    }

    /// Calculate initiative based on current state
    pub fn initiative(&self) -> f32 {
        let base = 10.0;
        let arousal_bonus = self.anima.arousal * 5.0; // Higher arousal = faster
        let dominance_bonus = self.anima.dominance * 2.0;
        base + arousal_bonus + dominance_bonus
    }

    /// Execute an ability
    pub fn execute_ability(&mut self, ability_index: usize, target: &mut Combatant) -> AbilityResult {
        if ability_index >= self.abilities.len() {
            return AbilityResult::Failed("Invalid ability".to_string());
        }

        let ability = &self.abilities[ability_index].clone();

        // Check psyche cost
        if self.psyche < ability.psyche_cost {
            return AbilityResult::Failed("Not enough psyche".to_string());
        }

        // Check reality requirements
        if let Some(ref required) = ability.reality_requirement {
            if &self.reality != required {
                return AbilityResult::Failed("Wrong reality layer".to_string());
            }
        }

        // Pay cost
        self.psyche -= ability.psyche_cost;

        // Calculate damage/effect
        let mut damage = ability.base_damage;

        // Apply anima modifiers
        match ability.damage_type {
            DamageType::Physical => {
                damage *= 1.0 + self.anima.dominance * 0.3;
            }
            DamageType::Psychic => {
                damage *= 1.0 + self.anima.arousal * 0.3;
            }
            DamageType::Emotional => {
                damage *= 1.0 + (1.0 - self.anima.stability) * 0.5;
            }
            DamageType::Reality => {
                if matches!(self.reality, RealityLayer::Fractured | RealityLayer::Shattered) {
                    damage *= 1.5;
                }
            }
        }

        // Apply to target
        match target {
            Combatant::Enemy(e) => {
                e.health = (e.health - damage).max(0.0);
            }
            Combatant::Player(p) => {
                p.health = (p.health - damage).max(0.0);
            }
        }

        // Apply status effects
        for effect in &ability.applies_effects {
            match target {
                Combatant::Enemy(e) => e.status_effects.push(effect.clone()),
                Combatant::Player(p) => p.status_effects.push(effect.clone()),
            }
        }

        AbilityResult::Success { damage, effects: ability.applies_effects.clone() }
    }

    /// Attempt to switch alters mid-combat
    pub fn combat_switch(&mut self, target_alter: &str, system: &mut PluralSystem) -> CombatSwitchResult {
        if self.switch_cooldown > 0 {
            return CombatSwitchResult::OnCooldown(self.switch_cooldown);
        }

        if !self.available_alters.contains(&target_alter.to_string()) {
            return CombatSwitchResult::NotAvailable;
        }

        // Attempt the switch
        let urgency = 0.5 + self.anima.arousal * 0.3; // Higher arousal = easier combat switches
        let result = system.request_switch(target_alter, urgency, false);

        match result {
            SwitchResult::Success => {
                self.fronting_alter = target_alter.to_string();
                self.switch_cooldown = 2; // 2 turn cooldown

                // Update abilities for new fronter
                if let Some(alter) = system.alters.get(target_alter) {
                    self.abilities = generate_combat_abilities(alter);
                    self.anima = alter.anima.clone();
                }

                // Reality might shift based on alter preference
                if let Some(alter) = system.alters.get(target_alter) {
                    self.reality = alter.preferred_reality.clone();
                }

                CombatSwitchResult::Success
            }
            SwitchResult::Resisted { resistance } => {
                CombatSwitchResult::Resisted(resistance)
            }
            SwitchResult::Failed(reason) => {
                CombatSwitchResult::Failed(format!("{:?}", reason))
            }
            SwitchResult::InProgress { eta } => {
                CombatSwitchResult::InProgress(eta)
            }
        }
    }

    /// Process a combat trigger
    pub fn process_combat_trigger(&mut self, trigger: Trigger, system: &mut PluralSystem) -> CombatTriggerResult {
        let trigger_result = system.process_trigger(trigger.clone());

        match trigger_result {
            TriggerResult::ForcedSwitch(alter_id) => {
                // Forced switch ignores cooldown
                self.switch_cooldown = 0;
                let switch_result = self.combat_switch(&alter_id, system);
                CombatTriggerResult::ForcedSwitch(alter_id, switch_result)
            }
            TriggerResult::Activation(alters) => {
                // Activation makes alters more available
                for alter_id in &alters {
                    if !self.available_alters.contains(alter_id) {
                        self.available_alters.push(alter_id.clone());
                    }
                }
                CombatTriggerResult::AltersActivated(alters)
            }
            TriggerResult::Dissociation => {
                self.anima.apply_trauma_response(0.5);
                CombatTriggerResult::Dissociation
            }
            TriggerResult::NoResponse => {
                CombatTriggerResult::NoEffect
            }
        }
    }
}

/// Result of a combat switch attempt
#[derive(Debug, Clone)]
pub enum CombatSwitchResult {
    Success,
    OnCooldown(u32),
    NotAvailable,
    Resisted(f32),
    Failed(String),
    InProgress(u64),
}

/// Result of a combat trigger
#[derive(Debug, Clone)]
pub enum CombatTriggerResult {
    NoEffect,
    ForcedSwitch(String, CombatSwitchResult),
    AltersActivated(Vec<String>),
    Dissociation,
}

// ============================================================================
// ENEMIES
// ============================================================================

/// An enemy combatant
#[derive(Debug, Clone)]
pub struct Enemy {
    /// Enemy ID
    pub id: String,
    /// Display name
    pub name: String,
    /// Enemy type
    pub enemy_type: EnemyType,
    /// Current health
    pub health: f32,
    /// Maximum health
    pub max_health: f32,
    /// Initiative value
    pub initiative: f32,
    /// Available abilities
    pub abilities: Vec<CombatAbility>,
    /// Status effects
    pub status_effects: Vec<StatusEffect>,
    /// Trigger behaviors (what triggers this enemy causes)
    pub trigger_behaviors: Vec<TriggerBehavior>,
    /// Reality perception (affects which layer player sees them in)
    pub reality_visibility: Vec<RealityLayer>,
}

/// Types of enemies
#[derive(Debug, Clone, PartialEq)]
pub enum EnemyType {
    /// Normal human threat
    Human,
    /// Symbolic/metaphorical threat (trauma manifestation)
    Manifestation,
    /// Environmental hazard
    Hazard,
    /// Internal threat (persecutor alter, intrusive thought)
    Internal,
    /// Supernatural entity
    Entity,
}

/// Behavior that creates triggers
#[derive(Debug, Clone)]
pub struct TriggerBehavior {
    /// Trigger ID this enemy can cause
    pub trigger_id: String,
    /// Condition for triggering
    pub condition: TriggerCondition,
    /// Intensity when triggered
    pub intensity: f32,
}

/// Conditions for enemy triggers
#[derive(Debug, Clone)]
pub enum TriggerCondition {
    /// Always trigger at start of enemy turn
    OnTurn,
    /// Trigger when enemy attacks
    OnAttack,
    /// Trigger when enemy health drops below threshold
    HealthBelow(f32),
    /// Trigger when player is at low health
    PlayerLowHealth,
    /// Trigger when specific ability is used
    OnAbility(String),
    /// Random chance per turn
    Random(f32),
}

// ============================================================================
// ABILITIES
// ============================================================================

/// A combat ability
#[derive(Debug, Clone)]
pub struct CombatAbility {
    /// Ability ID
    pub id: String,
    /// Display name
    pub name: String,
    /// Description
    pub description: String,
    /// Base damage/healing
    pub base_damage: f32,
    /// Damage type
    pub damage_type: DamageType,
    /// Psyche cost
    pub psyche_cost: f32,
    /// Cooldown in turns
    pub cooldown: u32,
    /// Current cooldown remaining
    pub current_cooldown: u32,
    /// Required reality layer (if any)
    pub reality_requirement: Option<RealityLayer>,
    /// Required alter category (if any)
    pub alter_requirement: Option<String>,
    /// Status effects applied
    pub applies_effects: Vec<StatusEffect>,
    /// Is this a defensive ability?
    pub is_defensive: bool,
}

/// Types of damage
#[derive(Debug, Clone, PartialEq)]
pub enum DamageType {
    Physical,
    Psychic,
    Emotional,
    Reality, // Affects reality perception
}

/// Result of using an ability
#[derive(Debug, Clone)]
pub enum AbilityResult {
    Success { damage: f32, effects: Vec<StatusEffect> },
    Failed(String),
    Blocked,
}

// ============================================================================
// STATUS EFFECTS
// ============================================================================

/// A status effect that modifies combat
#[derive(Debug, Clone)]
pub struct StatusEffect {
    /// Effect ID
    pub id: String,
    /// Display name
    pub name: String,
    /// Duration in turns (None = permanent)
    pub duration: Option<u32>,
    /// Effect type
    pub effect_type: StatusEffectType,
    /// Intensity/magnitude
    pub intensity: f32,
}

/// Types of status effects
#[derive(Debug, Clone)]
pub enum StatusEffectType {
    /// Damage over time
    Bleed,
    /// Psyche drain
    PsycheDrain,
    /// Cannot switch alters
    SwitchLocked,
    /// Reality perception forced
    RealityLocked(RealityLayer),
    /// Damage reduction
    Shielded,
    /// Increased damage output
    Empowered,
    /// Reduced accuracy/effectiveness
    Disoriented,
    /// Cannot use abilities
    Silenced,
    /// Specific trigger is active
    Triggered(String),
    /// Healing over time
    Regenerating,
}

// ============================================================================
// COMBAT ENVIRONMENT
// ============================================================================

/// Environmental factors in combat
#[derive(Debug, Clone, Default)]
pub struct CombatEnvironment {
    /// Current reality layer visibility
    pub reality_layers: Vec<RealityLayer>,
    /// Environmental hazards
    pub hazards: Vec<EnvironmentHazard>,
    /// Cover/defensive positions available
    pub defensive_positions: u32,
    /// Light level (affects perception)
    pub light_level: f32,
    /// Ambient trigger intensity
    pub ambient_trigger_intensity: f32,
}

/// Environmental hazards
#[derive(Debug, Clone)]
pub struct EnvironmentHazard {
    pub name: String,
    pub damage_per_turn: f32,
    pub affects_reality: Option<RealityLayer>,
}

// ============================================================================
// COMBAT EFFECTS
// ============================================================================

/// Active effects in combat
#[derive(Debug, Clone)]
pub struct CombatEffect {
    pub name: String,
    pub duration: u32,
    pub effect: CombatEffectType,
}

#[derive(Debug, Clone)]
pub enum CombatEffectType {
    /// Reality is shifting
    RealityFlux,
    /// Forced switching is occurring
    SystemInstability,
    /// Environmental change
    EnvironmentChange(String),
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/// Generate combat abilities for an alter based on their traits
fn generate_combat_abilities(alter: &Alter) -> Vec<CombatAbility> {
    let mut abilities = Vec::new();

    // Basic attack available to all
    abilities.push(CombatAbility {
        id: "basic_attack".to_string(),
        name: "Strike".to_string(),
        description: "A basic physical attack".to_string(),
        base_damage: 10.0,
        damage_type: DamageType::Physical,
        psyche_cost: 0.0,
        cooldown: 0,
        current_cooldown: 0,
        reality_requirement: None,
        alter_requirement: None,
        applies_effects: Vec::new(),
        is_defensive: false,
    });

    // Add abilities based on alter's abilities set
    for ability_name in &alter.abilities {
        match ability_name.as_str() {
            "combat" | "combat_master" => {
                abilities.push(CombatAbility {
                    id: "power_strike".to_string(),
                    name: "Power Strike".to_string(),
                    description: "A devastating physical attack".to_string(),
                    base_damage: 25.0,
                    damage_type: DamageType::Physical,
                    psyche_cost: 10.0,
                    cooldown: 2,
                    current_cooldown: 0,
                    reality_requirement: None,
                    alter_requirement: None,
                    applies_effects: Vec::new(),
                    is_defensive: false,
                });
            }
            "perception" | "perceive" => {
                abilities.push(CombatAbility {
                    id: "reality_sight".to_string(),
                    name: "Reality Sight".to_string(),
                    description: "Pierce the veil between layers".to_string(),
                    base_damage: 0.0,
                    damage_type: DamageType::Reality,
                    psyche_cost: 15.0,
                    cooldown: 3,
                    current_cooldown: 0,
                    reality_requirement: None,
                    alter_requirement: None,
                    applies_effects: Vec::new(),
                    is_defensive: true,
                });
            }
            "protection" | "shield" => {
                abilities.push(CombatAbility {
                    id: "protective_barrier".to_string(),
                    name: "Protective Barrier".to_string(),
                    description: "Shield the system from harm".to_string(),
                    base_damage: 0.0,
                    damage_type: DamageType::Psychic,
                    psyche_cost: 20.0,
                    cooldown: 3,
                    current_cooldown: 0,
                    reality_requirement: None,
                    alter_requirement: None,
                    applies_effects: vec![StatusEffect {
                        id: "shielded".to_string(),
                        name: "Shielded".to_string(),
                        duration: Some(2),
                        effect_type: StatusEffectType::Shielded,
                        intensity: 0.5,
                    }],
                    is_defensive: true,
                });
            }
            "trauma_processing" => {
                abilities.push(CombatAbility {
                    id: "trauma_strike".to_string(),
                    name: "Trauma Strike".to_string(),
                    description: "Channel trauma into devastating force".to_string(),
                    base_damage: 35.0,
                    damage_type: DamageType::Emotional,
                    psyche_cost: 25.0,
                    cooldown: 4,
                    current_cooldown: 0,
                    reality_requirement: Some(RealityLayer::Fractured),
                    alter_requirement: None,
                    applies_effects: Vec::new(),
                    is_defensive: false,
                });
            }
            _ => {}
        }
    }

    // Add reality-specific ability if alter prefers fractured
    if matches!(alter.preferred_reality, RealityLayer::Fractured | RealityLayer::Shattered) {
        abilities.push(CombatAbility {
            id: "fractured_assault".to_string(),
            name: "Fractured Assault".to_string(),
            description: "Attack from the fractured realm".to_string(),
            base_damage: 20.0,
            damage_type: DamageType::Reality,
            psyche_cost: 15.0,
            cooldown: 2,
            current_cooldown: 0,
            reality_requirement: Some(RealityLayer::Fractured),
            alter_requirement: None,
            applies_effects: Vec::new(),
            is_defensive: false,
        });
    }

    abilities
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn create_test_system() -> PluralSystem {
        let mut system = PluralSystem::new("Test System");

        let alter = Alter {
            id: "protector".to_string(),
            name: "Protector".to_string(),
            category: super::super::runtime::AlterCategory::Council,
            state: AlterPresenceState::Fronting,
            anima: AnimaState::new(0.0, 0.5, 0.7),
            base_arousal: 0.5,
            base_dominance: 0.7,
            time_since_front: 0,
            triggers: vec!["threat".to_string()],
            abilities: HashSet::from(["combat".to_string(), "protection".to_string()]),
            preferred_reality: RealityLayer::Grounded,
            memory_access: super::super::runtime::MemoryAccess::Full,
        };

        system.add_alter(alter);
        system.fronting = FrontingState::Single("protector".to_string());
        system
    }

    #[test]
    fn test_combat_state_creation() {
        let system = create_test_system();
        let enemy = Enemy {
            id: "enemy1".to_string(),
            name: "Shadow".to_string(),
            enemy_type: EnemyType::Manifestation,
            health: 50.0,
            max_health: 50.0,
            initiative: 8.0,
            abilities: Vec::new(),
            status_effects: Vec::new(),
            trigger_behaviors: Vec::new(),
            reality_visibility: vec![RealityLayer::Fractured],
        };

        let combat = CombatState::new(&system, vec![enemy]);
        assert_eq!(combat.combatants.len(), 2);
        assert_eq!(combat.round, 1);
        assert!(!combat.is_over);
    }

    #[test]
    fn test_ability_execution() {
        let system = create_test_system();
        let mut player = PlayerCombatant::from_system(&system);
        let mut enemy = Combatant::Enemy(Enemy {
            id: "enemy1".to_string(),
            name: "Target".to_string(),
            enemy_type: EnemyType::Human,
            health: 100.0,
            max_health: 100.0,
            initiative: 5.0,
            abilities: Vec::new(),
            status_effects: Vec::new(),
            trigger_behaviors: Vec::new(),
            reality_visibility: vec![RealityLayer::Grounded],
        });

        // Basic attack should work
        let result = player.execute_ability(0, &mut enemy);
        assert!(matches!(result, AbilityResult::Success { .. }));

        // Enemy should have taken damage
        if let Combatant::Enemy(e) = &enemy {
            assert!(e.health < 100.0);
        }
    }
}
