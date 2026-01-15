//! # Save System for DAEMONIORUM
//!
//! Handles serialization and persistence of game state, including
//! plural system state, progression, and narrative flags.

use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::path::Path;

use super::game_loop::{FlagValue, GameState, Item, ItemType, SaveData, Scene};
use super::runtime::{
    Alter, AlterCategory, AlterPresenceState, AnimaState, FrontingState,
    MemoryAccess, PluralSystem, RealityLayer, Trigger, TriggerCategory,
};

// ============================================================================
// SAVE FORMAT VERSION
// ============================================================================

/// Current save format version for compatibility checking
pub const SAVE_FORMAT_VERSION: u32 = 1;

/// Magic bytes to identify save files
pub const SAVE_MAGIC: &[u8; 8] = b"DAEMNGM\0";

// ============================================================================
// SAVE FILE STRUCTURE
// ============================================================================

/// Complete save file structure
#[derive(Debug, Clone)]
pub struct SaveFile {
    /// Save format version
    pub version: u32,
    /// Save slot metadata
    pub metadata: SaveMetadata,
    /// The actual save data
    pub data: SaveData,
}

/// Metadata about the save
#[derive(Debug, Clone)]
pub struct SaveMetadata {
    /// Save slot name
    pub name: String,
    /// Unix timestamp of save creation
    pub timestamp: u64,
    /// Play time in seconds
    pub play_time: u64,
    /// Current location name for display
    pub location_name: String,
    /// Current fronter name for display
    pub fronter_name: String,
    /// Completion percentage (0-100)
    pub completion: u8,
}

// ============================================================================
// SERIALIZATION FORMATS
// ============================================================================

/// Serializable representation of PluralSystem
#[derive(Debug, Clone)]
pub struct SerializedPluralSystem {
    pub name: Option<String>,
    pub alters: Vec<SerializedAlter>,
    pub fronting: SerializedFrontingState,
    pub anima: SerializedAnimaState,
    pub reality_layer: String,
    pub dissociation: f32,
    pub stability: f32,
}

/// Serializable representation of Alter
#[derive(Debug, Clone)]
pub struct SerializedAlter {
    pub id: String,
    pub name: String,
    pub category: String,
    pub state: String,
    pub anima: SerializedAnimaState,
    pub base_arousal: f32,
    pub base_dominance: f32,
    pub time_since_front: u64,
    pub triggers: Vec<String>,
    pub abilities: Vec<String>,
    pub preferred_reality: String,
    pub memory_access: SerializedMemoryAccess,
}

/// Serializable AnimaState
#[derive(Debug, Clone)]
pub struct SerializedAnimaState {
    pub pleasure: f32,
    pub arousal: f32,
    pub dominance: f32,
    pub expressiveness: f32,
    pub stability: f32,
}

/// Serializable FrontingState
#[derive(Debug, Clone)]
pub enum SerializedFrontingState {
    None,
    Single(String),
    Blended(Vec<String>),
    Rapid(Vec<String>),
    Unknown,
}

/// Serializable MemoryAccess
#[derive(Debug, Clone)]
pub enum SerializedMemoryAccess {
    Full,
    Partial(Vec<String>),
    Own,
    None,
}

// ============================================================================
// SAVE MANAGER
// ============================================================================

/// Manages save operations
pub struct SaveManager {
    /// Base directory for saves
    save_dir: String,
    /// Number of available slots
    num_slots: usize,
    /// Auto-save enabled
    auto_save_enabled: bool,
    /// Auto-save interval in seconds
    auto_save_interval: u64,
}

impl SaveManager {
    /// Create a new save manager
    pub fn new(save_dir: &str) -> Self {
        Self {
            save_dir: save_dir.to_string(),
            num_slots: 3,
            auto_save_enabled: true,
            auto_save_interval: 300, // 5 minutes
        }
    }

    /// Get the path for a save slot
    fn slot_path(&self, slot: usize) -> String {
        format!("{}/save_{}.daem", self.save_dir, slot)
    }

    /// Get the auto-save path
    fn auto_save_path(&self) -> String {
        format!("{}/autosave.daem", self.save_dir)
    }

    /// Save game to a slot
    pub fn save_to_slot(&self, slot: usize, state: &GameState) -> Result<(), SaveError> {
        if slot >= self.num_slots {
            return Err(SaveError::InvalidSlot(slot));
        }

        let save_file = self.create_save_file(state, &format!("Save {}", slot + 1))?;
        let path = self.slot_path(slot);
        self.write_save_file(&path, &save_file)
    }

    /// Load game from a slot
    pub fn load_from_slot(&self, slot: usize) -> Result<GameState, SaveError> {
        if slot >= self.num_slots {
            return Err(SaveError::InvalidSlot(slot));
        }

        let path = self.slot_path(slot);
        let save_file = self.read_save_file(&path)?;
        self.create_game_state(save_file)
    }

    /// Auto-save the game
    pub fn auto_save(&self, state: &GameState) -> Result<(), SaveError> {
        if !self.auto_save_enabled {
            return Ok(());
        }

        let save_file = self.create_save_file(state, "Auto Save")?;
        let path = self.auto_save_path();
        self.write_save_file(&path, &save_file)
    }

    /// Load auto-save
    pub fn load_auto_save(&self) -> Result<GameState, SaveError> {
        let path = self.auto_save_path();
        let save_file = self.read_save_file(&path)?;
        self.create_game_state(save_file)
    }

    /// Check if a slot has save data
    pub fn slot_exists(&self, slot: usize) -> bool {
        let path = self.slot_path(slot);
        Path::new(&path).exists()
    }

    /// Get metadata for a slot without loading full data
    pub fn get_slot_metadata(&self, slot: usize) -> Result<SaveMetadata, SaveError> {
        if slot >= self.num_slots {
            return Err(SaveError::InvalidSlot(slot));
        }

        let path = self.slot_path(slot);
        let save_file = self.read_save_file(&path)?;
        Ok(save_file.metadata)
    }

    /// Delete a save slot
    pub fn delete_slot(&self, slot: usize) -> Result<(), SaveError> {
        if slot >= self.num_slots {
            return Err(SaveError::InvalidSlot(slot));
        }

        let path = self.slot_path(slot);
        std::fs::remove_file(&path).map_err(|e| SaveError::IoError(e.to_string()))
    }

    /// Create a SaveFile from GameState
    fn create_save_file(&self, state: &GameState, name: &str) -> Result<SaveFile, SaveError> {
        let fronter_name = match &state.system.fronting {
            FrontingState::Single(id) => state.system.alters.get(id)
                .map(|a| a.name.clone())
                .unwrap_or_else(|| "Unknown".to_string()),
            FrontingState::Blended(ids) => {
                if ids.is_empty() {
                    "None".to_string()
                } else {
                    format!("{} & others", ids.first().unwrap())
                }
            }
            _ => "Unknown".to_string(),
        };

        let metadata = SaveMetadata {
            name: name.to_string(),
            timestamp: current_timestamp(),
            play_time: state.game_time,
            location_name: state.scene.name.clone(),
            fronter_name,
            completion: calculate_completion(state),
        };

        Ok(SaveFile {
            version: SAVE_FORMAT_VERSION,
            metadata,
            data: state.to_save(),
        })
    }

    /// Create GameState from SaveFile
    fn create_game_state(&self, save_file: SaveFile) -> Result<GameState, SaveError> {
        // Version check
        if save_file.version > SAVE_FORMAT_VERSION {
            return Err(SaveError::VersionMismatch {
                expected: SAVE_FORMAT_VERSION,
                found: save_file.version,
            });
        }

        Ok(GameState::from_save(save_file.data))
    }

    /// Write save file to disk
    fn write_save_file(&self, path: &str, save_file: &SaveFile) -> Result<(), SaveError> {
        // Ensure directory exists
        if let Some(parent) = Path::new(path).parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| SaveError::IoError(e.to_string()))?;
        }

        // Serialize to binary format
        let data = serialize_save_file(save_file)?;

        // Write to file
        let mut file = std::fs::File::create(path)
            .map_err(|e| SaveError::IoError(e.to_string()))?;

        file.write_all(&data)
            .map_err(|e| SaveError::IoError(e.to_string()))?;

        Ok(())
    }

    /// Read save file from disk
    fn read_save_file(&self, path: &str) -> Result<SaveFile, SaveError> {
        // Read from file
        let mut file = std::fs::File::open(path)
            .map_err(|e| SaveError::IoError(e.to_string()))?;

        let mut data = Vec::new();
        file.read_to_end(&mut data)
            .map_err(|e| SaveError::IoError(e.to_string()))?;

        // Deserialize from binary format
        deserialize_save_file(&data)
    }
}

// ============================================================================
// SERIALIZATION
// ============================================================================

/// Serialize a SaveFile to binary
fn serialize_save_file(save_file: &SaveFile) -> Result<Vec<u8>, SaveError> {
    let mut buffer = Vec::new();

    // Write magic bytes
    buffer.extend_from_slice(SAVE_MAGIC);

    // Write version
    buffer.extend_from_slice(&save_file.version.to_le_bytes());

    // Write metadata
    write_string(&mut buffer, &save_file.metadata.name);
    buffer.extend_from_slice(&save_file.metadata.timestamp.to_le_bytes());
    buffer.extend_from_slice(&save_file.metadata.play_time.to_le_bytes());
    write_string(&mut buffer, &save_file.metadata.location_name);
    write_string(&mut buffer, &save_file.metadata.fronter_name);
    buffer.push(save_file.metadata.completion);

    // Write data
    serialize_save_data(&mut buffer, &save_file.data)?;

    Ok(buffer)
}

/// Deserialize a SaveFile from binary
fn deserialize_save_file(data: &[u8]) -> Result<SaveFile, SaveError> {
    let mut cursor = 0;

    // Read and verify magic bytes
    if data.len() < 8 || &data[0..8] != SAVE_MAGIC {
        return Err(SaveError::InvalidFormat("Invalid magic bytes".to_string()));
    }
    cursor += 8;

    // Read version
    let version = read_u32(data, &mut cursor)?;

    // Read metadata
    let name = read_string(data, &mut cursor)?;
    let timestamp = read_u64(data, &mut cursor)?;
    let play_time = read_u64(data, &mut cursor)?;
    let location_name = read_string(data, &mut cursor)?;
    let fronter_name = read_string(data, &mut cursor)?;
    let completion = read_u8(data, &mut cursor)?;

    let metadata = SaveMetadata {
        name,
        timestamp,
        play_time,
        location_name,
        fronter_name,
        completion,
    };

    // Read data
    let save_data = deserialize_save_data(data, &mut cursor)?;

    Ok(SaveFile {
        version,
        metadata,
        data: save_data,
    })
}

/// Serialize SaveData
fn serialize_save_data(buffer: &mut Vec<u8>, data: &SaveData) -> Result<(), SaveError> {
    // Serialize PluralSystem
    serialize_plural_system(buffer, &data.system)?;

    // Serialize Scene
    serialize_scene(buffer, &data.scene);

    // Game time
    buffer.extend_from_slice(&data.game_time.to_le_bytes());

    // Flags
    write_u32(buffer, data.flags.len() as u32);
    for (key, value) in &data.flags {
        write_string(buffer, key);
        serialize_flag_value(buffer, value);
    }

    // Inventory
    write_u32(buffer, data.inventory.len() as u32);
    for item in &data.inventory {
        serialize_item(buffer, item);
    }

    // Unlocked abilities
    write_u32(buffer, data.unlocked_abilities.len() as u32);
    for ability in &data.unlocked_abilities {
        write_string(buffer, ability);
    }

    // Processed traumas
    write_u32(buffer, data.processed_traumas.len() as u32);
    for trauma in &data.processed_traumas {
        write_string(buffer, trauma);
    }

    Ok(())
}

/// Deserialize SaveData
fn deserialize_save_data(data: &[u8], cursor: &mut usize) -> Result<SaveData, SaveError> {
    // Deserialize PluralSystem
    let system = deserialize_plural_system(data, cursor)?;

    // Deserialize Scene
    let scene = deserialize_scene(data, cursor)?;

    // Game time
    let game_time = read_u64(data, cursor)?;

    // Flags
    let flag_count = read_u32(data, cursor)? as usize;
    let mut flags = HashMap::new();
    for _ in 0..flag_count {
        let key = read_string(data, cursor)?;
        let value = deserialize_flag_value(data, cursor)?;
        flags.insert(key, value);
    }

    // Inventory
    let inventory_count = read_u32(data, cursor)? as usize;
    let mut inventory = Vec::new();
    for _ in 0..inventory_count {
        inventory.push(deserialize_item(data, cursor)?);
    }

    // Unlocked abilities
    let ability_count = read_u32(data, cursor)? as usize;
    let mut unlocked_abilities = Vec::new();
    for _ in 0..ability_count {
        unlocked_abilities.push(read_string(data, cursor)?);
    }

    // Processed traumas
    let trauma_count = read_u32(data, cursor)? as usize;
    let mut processed_traumas = Vec::new();
    for _ in 0..trauma_count {
        processed_traumas.push(read_string(data, cursor)?);
    }

    Ok(SaveData {
        system,
        scene,
        game_time,
        flags,
        inventory,
        unlocked_abilities,
        processed_traumas,
    })
}

/// Serialize PluralSystem
fn serialize_plural_system(buffer: &mut Vec<u8>, system: &PluralSystem) -> Result<(), SaveError> {
    // Name
    write_option_string(buffer, &system.name);

    // Alters
    write_u32(buffer, system.alters.len() as u32);
    for alter in system.alters.values() {
        serialize_alter(buffer, alter)?;
    }

    // Fronting state
    serialize_fronting_state(buffer, &system.fronting);

    // Anima
    serialize_anima_state(buffer, &system.anima);

    // Reality layer
    serialize_reality_layer(buffer, &system.reality_layer);

    // Dissociation and stability
    buffer.extend_from_slice(&system.dissociation.to_le_bytes());
    buffer.extend_from_slice(&system.stability.to_le_bytes());

    Ok(())
}

/// Deserialize PluralSystem
fn deserialize_plural_system(data: &[u8], cursor: &mut usize) -> Result<PluralSystem, SaveError> {
    let name = read_option_string(data, cursor)?;

    let alter_count = read_u32(data, cursor)? as usize;
    let mut alters = HashMap::new();
    for _ in 0..alter_count {
        let alter = deserialize_alter(data, cursor)?;
        alters.insert(alter.id.clone(), alter);
    }

    let fronting = deserialize_fronting_state(data, cursor)?;
    let anima = deserialize_anima_state(data, cursor)?;
    let reality_layer = deserialize_reality_layer(data, cursor)?;
    let dissociation = read_f32(data, cursor)?;
    let stability = read_f32(data, cursor)?;

    Ok(PluralSystem {
        name,
        alters,
        fronting,
        anima,
        reality_layer,
        active_triggers: Vec::new(), // Not persisted
        headspace: super::runtime::HeadspaceState::default(),
        dissociation,
        stability,
    })
}

/// Serialize Alter
fn serialize_alter(buffer: &mut Vec<u8>, alter: &Alter) -> Result<(), SaveError> {
    write_string(buffer, &alter.id);
    write_string(buffer, &alter.name);
    serialize_alter_category(buffer, &alter.category);
    serialize_alter_state(buffer, &alter.state);
    serialize_anima_state(buffer, &alter.anima);
    buffer.extend_from_slice(&alter.base_arousal.to_le_bytes());
    buffer.extend_from_slice(&alter.base_dominance.to_le_bytes());
    buffer.extend_from_slice(&alter.time_since_front.to_le_bytes());

    // Triggers
    write_u32(buffer, alter.triggers.len() as u32);
    for trigger in &alter.triggers {
        write_string(buffer, trigger);
    }

    // Abilities
    write_u32(buffer, alter.abilities.len() as u32);
    for ability in &alter.abilities {
        write_string(buffer, ability);
    }

    serialize_reality_layer(buffer, &alter.preferred_reality);
    serialize_memory_access(buffer, &alter.memory_access);

    Ok(())
}

/// Deserialize Alter
fn deserialize_alter(data: &[u8], cursor: &mut usize) -> Result<Alter, SaveError> {
    let id = read_string(data, cursor)?;
    let name = read_string(data, cursor)?;
    let category = deserialize_alter_category(data, cursor)?;
    let state = deserialize_alter_state(data, cursor)?;
    let anima = deserialize_anima_state(data, cursor)?;
    let base_arousal = read_f32(data, cursor)?;
    let base_dominance = read_f32(data, cursor)?;
    let time_since_front = read_u64(data, cursor)?;

    let trigger_count = read_u32(data, cursor)? as usize;
    let mut triggers = Vec::new();
    for _ in 0..trigger_count {
        triggers.push(read_string(data, cursor)?);
    }

    let ability_count = read_u32(data, cursor)? as usize;
    let mut abilities = HashSet::new();
    for _ in 0..ability_count {
        abilities.insert(read_string(data, cursor)?);
    }

    let preferred_reality = deserialize_reality_layer(data, cursor)?;
    let memory_access = deserialize_memory_access(data, cursor)?;

    Ok(Alter {
        id,
        name,
        category,
        state,
        anima,
        base_arousal,
        base_dominance,
        time_since_front,
        triggers,
        abilities,
        preferred_reality,
        memory_access,
    })
}

// ============================================================================
// SERIALIZATION HELPERS
// ============================================================================

fn serialize_anima_state(buffer: &mut Vec<u8>, anima: &AnimaState) {
    buffer.extend_from_slice(&anima.pleasure.to_le_bytes());
    buffer.extend_from_slice(&anima.arousal.to_le_bytes());
    buffer.extend_from_slice(&anima.dominance.to_le_bytes());
    buffer.extend_from_slice(&anima.expressiveness.to_le_bytes());
    buffer.extend_from_slice(&anima.stability.to_le_bytes());
}

fn deserialize_anima_state(data: &[u8], cursor: &mut usize) -> Result<AnimaState, SaveError> {
    Ok(AnimaState {
        pleasure: read_f32(data, cursor)?,
        arousal: read_f32(data, cursor)?,
        dominance: read_f32(data, cursor)?,
        expressiveness: read_f32(data, cursor)?,
        stability: read_f32(data, cursor)?,
    })
}

fn serialize_fronting_state(buffer: &mut Vec<u8>, state: &FrontingState) {
    match state {
        FrontingState::None => buffer.push(0),
        FrontingState::Single(id) => {
            buffer.push(1);
            write_string(buffer, id);
        }
        FrontingState::Blended(ids) => {
            buffer.push(2);
            write_u32(buffer, ids.len() as u32);
            for id in ids {
                write_string(buffer, id);
            }
        }
        FrontingState::Rapid(ids) => {
            buffer.push(3);
            write_u32(buffer, ids.len() as u32);
            for id in ids {
                write_string(buffer, id);
            }
        }
        FrontingState::Unknown => buffer.push(4),
    }
}

fn deserialize_fronting_state(data: &[u8], cursor: &mut usize) -> Result<FrontingState, SaveError> {
    let tag = read_u8(data, cursor)?;
    match tag {
        0 => Ok(FrontingState::None),
        1 => Ok(FrontingState::Single(read_string(data, cursor)?)),
        2 => {
            let count = read_u32(data, cursor)? as usize;
            let mut ids = Vec::new();
            for _ in 0..count {
                ids.push(read_string(data, cursor)?);
            }
            Ok(FrontingState::Blended(ids))
        }
        3 => {
            let count = read_u32(data, cursor)? as usize;
            let mut ids = Vec::new();
            for _ in 0..count {
                ids.push(read_string(data, cursor)?);
            }
            Ok(FrontingState::Rapid(ids))
        }
        4 => Ok(FrontingState::Unknown),
        _ => Err(SaveError::InvalidFormat(format!("Unknown FrontingState tag: {}", tag))),
    }
}

fn serialize_reality_layer(buffer: &mut Vec<u8>, layer: &RealityLayer) {
    match layer {
        RealityLayer::Grounded => buffer.push(0),
        RealityLayer::Fractured => buffer.push(1),
        RealityLayer::Shattered => buffer.push(2),
        RealityLayer::Custom(name) => {
            buffer.push(3);
            write_string(buffer, name);
        }
    }
}

fn deserialize_reality_layer(data: &[u8], cursor: &mut usize) -> Result<RealityLayer, SaveError> {
    let tag = read_u8(data, cursor)?;
    match tag {
        0 => Ok(RealityLayer::Grounded),
        1 => Ok(RealityLayer::Fractured),
        2 => Ok(RealityLayer::Shattered),
        3 => Ok(RealityLayer::Custom(read_string(data, cursor)?)),
        _ => Err(SaveError::InvalidFormat(format!("Unknown RealityLayer tag: {}", tag))),
    }
}

fn serialize_alter_category(buffer: &mut Vec<u8>, category: &AlterCategory) {
    match category {
        AlterCategory::Council => buffer.push(0),
        AlterCategory::Servant => buffer.push(1),
        AlterCategory::Fragment => buffer.push(2),
        AlterCategory::Introject => buffer.push(3),
        AlterCategory::Persecutor => buffer.push(4),
        AlterCategory::TraumaHolder => buffer.push(5),
        AlterCategory::Custom(name) => {
            buffer.push(6);
            write_string(buffer, name);
        }
    };
}

fn deserialize_alter_category(data: &[u8], cursor: &mut usize) -> Result<AlterCategory, SaveError> {
    let tag = read_u8(data, cursor)?;
    match tag {
        0 => Ok(AlterCategory::Council),
        1 => Ok(AlterCategory::Servant),
        2 => Ok(AlterCategory::Fragment),
        3 => Ok(AlterCategory::Introject),
        4 => Ok(AlterCategory::Persecutor),
        5 => Ok(AlterCategory::TraumaHolder),
        6 => Ok(AlterCategory::Custom(read_string(data, cursor)?)),
        _ => Err(SaveError::InvalidFormat(format!("Unknown AlterCategory tag: {}", tag))),
    }
}

fn serialize_alter_state(buffer: &mut Vec<u8>, state: &AlterPresenceState) {
    let tag = match state {
        AlterPresenceState::Dormant => 0,
        AlterPresenceState::Stirring => 1,
        AlterPresenceState::CoConscious => 2,
        AlterPresenceState::Emerging => 3,
        AlterPresenceState::Fronting => 4,
        AlterPresenceState::Receding => 5,
        AlterPresenceState::Triggered => 6,
        AlterPresenceState::Dissociating => 7,
    };
    buffer.push(tag);
}

fn deserialize_alter_state(data: &[u8], cursor: &mut usize) -> Result<AlterPresenceState, SaveError> {
    let tag = read_u8(data, cursor)?;
    match tag {
        0 => Ok(AlterPresenceState::Dormant),
        1 => Ok(AlterPresenceState::Stirring),
        2 => Ok(AlterPresenceState::CoConscious),
        3 => Ok(AlterPresenceState::Emerging),
        4 => Ok(AlterPresenceState::Fronting),
        5 => Ok(AlterPresenceState::Receding),
        6 => Ok(AlterPresenceState::Triggered),
        7 => Ok(AlterPresenceState::Dissociating),
        _ => Err(SaveError::InvalidFormat(format!("Unknown AlterPresenceState tag: {}", tag))),
    }
}

fn serialize_memory_access(buffer: &mut Vec<u8>, access: &MemoryAccess) {
    match access {
        MemoryAccess::Full => buffer.push(0),
        MemoryAccess::Partial(ids) => {
            buffer.push(1);
            write_u32(buffer, ids.len() as u32);
            for id in ids {
                write_string(buffer, id);
            }
        }
        MemoryAccess::Own => buffer.push(2),
        MemoryAccess::None => buffer.push(3),
    }
}

fn deserialize_memory_access(data: &[u8], cursor: &mut usize) -> Result<MemoryAccess, SaveError> {
    let tag = read_u8(data, cursor)?;
    match tag {
        0 => Ok(MemoryAccess::Full),
        1 => {
            let count = read_u32(data, cursor)? as usize;
            let mut ids = Vec::new();
            for _ in 0..count {
                ids.push(read_string(data, cursor)?);
            }
            Ok(MemoryAccess::Partial(ids))
        }
        2 => Ok(MemoryAccess::Own),
        3 => Ok(MemoryAccess::None),
        _ => Err(SaveError::InvalidFormat(format!("Unknown MemoryAccess tag: {}", tag))),
    }
}

fn serialize_scene(buffer: &mut Vec<u8>, scene: &Scene) {
    write_string(buffer, &scene.id);
    write_string(buffer, &scene.name);
    buffer.extend_from_slice(&scene.safety_level.to_le_bytes());
    buffer.extend_from_slice(&scene.player_position.0.to_le_bytes());
    buffer.extend_from_slice(&scene.player_position.1.to_le_bytes());
    // Other scene data would be loaded from scene definitions, not saved
}

fn deserialize_scene(data: &[u8], cursor: &mut usize) -> Result<Scene, SaveError> {
    let id = read_string(data, cursor)?;
    let name = read_string(data, cursor)?;
    let safety_level = read_f32(data, cursor)?;
    let x = read_f32(data, cursor)?;
    let y = read_f32(data, cursor)?;

    Ok(Scene {
        id,
        name,
        safety_level,
        player_position: (x, y),
        entities: Vec::new(), // Loaded from scene definitions
        trigger_zones: Vec::new(),
        interactables: Vec::new(),
        active_cutscene: None,
    })
}

fn serialize_flag_value(buffer: &mut Vec<u8>, value: &FlagValue) {
    match value {
        FlagValue::Bool(b) => {
            buffer.push(0);
            buffer.push(if *b { 1 } else { 0 });
        }
        FlagValue::Int(i) => {
            buffer.push(1);
            buffer.extend_from_slice(&i.to_le_bytes());
        }
        FlagValue::String(s) => {
            buffer.push(2);
            write_string(buffer, s);
        }
    }
}

fn deserialize_flag_value(data: &[u8], cursor: &mut usize) -> Result<FlagValue, SaveError> {
    let tag = read_u8(data, cursor)?;
    match tag {
        0 => Ok(FlagValue::Bool(read_u8(data, cursor)? != 0)),
        1 => Ok(FlagValue::Int(read_i32(data, cursor)?)),
        2 => Ok(FlagValue::String(read_string(data, cursor)?)),
        _ => Err(SaveError::InvalidFormat(format!("Unknown FlagValue tag: {}", tag))),
    }
}

fn serialize_item(buffer: &mut Vec<u8>, item: &Item) {
    write_string(buffer, &item.id);
    write_string(buffer, &item.name);
    write_string(buffer, &item.description);
    let type_tag = match item.item_type {
        ItemType::Key => 0,
        ItemType::Consumable => 1,
        ItemType::Document => 2,
        ItemType::Memento => 3,
    };
    buffer.push(type_tag);
}

fn deserialize_item(data: &[u8], cursor: &mut usize) -> Result<Item, SaveError> {
    let id = read_string(data, cursor)?;
    let name = read_string(data, cursor)?;
    let description = read_string(data, cursor)?;
    let type_tag = read_u8(data, cursor)?;
    let item_type = match type_tag {
        0 => ItemType::Key,
        1 => ItemType::Consumable,
        2 => ItemType::Document,
        3 => ItemType::Memento,
        _ => return Err(SaveError::InvalidFormat(format!("Unknown ItemType tag: {}", type_tag))),
    };

    Ok(Item {
        id,
        name,
        description,
        item_type,
    })
}

// ============================================================================
// BINARY READ/WRITE PRIMITIVES
// ============================================================================

fn write_u32(buffer: &mut Vec<u8>, value: u32) {
    buffer.extend_from_slice(&value.to_le_bytes());
}

fn read_u8(data: &[u8], cursor: &mut usize) -> Result<u8, SaveError> {
    if *cursor >= data.len() {
        return Err(SaveError::UnexpectedEof);
    }
    let value = data[*cursor];
    *cursor += 1;
    Ok(value)
}

fn read_u32(data: &[u8], cursor: &mut usize) -> Result<u32, SaveError> {
    if *cursor + 4 > data.len() {
        return Err(SaveError::UnexpectedEof);
    }
    let bytes: [u8; 4] = data[*cursor..*cursor + 4].try_into().unwrap();
    *cursor += 4;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i32(data: &[u8], cursor: &mut usize) -> Result<i32, SaveError> {
    if *cursor + 4 > data.len() {
        return Err(SaveError::UnexpectedEof);
    }
    let bytes: [u8; 4] = data[*cursor..*cursor + 4].try_into().unwrap();
    *cursor += 4;
    Ok(i32::from_le_bytes(bytes))
}

fn read_u64(data: &[u8], cursor: &mut usize) -> Result<u64, SaveError> {
    if *cursor + 8 > data.len() {
        return Err(SaveError::UnexpectedEof);
    }
    let bytes: [u8; 8] = data[*cursor..*cursor + 8].try_into().unwrap();
    *cursor += 8;
    Ok(u64::from_le_bytes(bytes))
}

fn read_f32(data: &[u8], cursor: &mut usize) -> Result<f32, SaveError> {
    if *cursor + 4 > data.len() {
        return Err(SaveError::UnexpectedEof);
    }
    let bytes: [u8; 4] = data[*cursor..*cursor + 4].try_into().unwrap();
    *cursor += 4;
    Ok(f32::from_le_bytes(bytes))
}

fn write_string(buffer: &mut Vec<u8>, s: &str) {
    let bytes = s.as_bytes();
    write_u32(buffer, bytes.len() as u32);
    buffer.extend_from_slice(bytes);
}

fn read_string(data: &[u8], cursor: &mut usize) -> Result<String, SaveError> {
    let len = read_u32(data, cursor)? as usize;
    if *cursor + len > data.len() {
        return Err(SaveError::UnexpectedEof);
    }
    let bytes = &data[*cursor..*cursor + len];
    *cursor += len;
    String::from_utf8(bytes.to_vec())
        .map_err(|e| SaveError::InvalidFormat(e.to_string()))
}

fn write_option_string(buffer: &mut Vec<u8>, opt: &Option<String>) {
    match opt {
        Some(s) => {
            buffer.push(1);
            write_string(buffer, s);
        }
        None => buffer.push(0),
    }
}

fn read_option_string(data: &[u8], cursor: &mut usize) -> Result<Option<String>, SaveError> {
    let has_value = read_u8(data, cursor)?;
    if has_value != 0 {
        Ok(Some(read_string(data, cursor)?))
    } else {
        Ok(None)
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

fn current_timestamp() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn calculate_completion(state: &GameState) -> u8 {
    // Simple completion calculation based on processed traumas
    // In a real game, this would be more sophisticated
    let trauma_progress = (state.processed_traumas.len() as f32 / 10.0 * 100.0).min(100.0);
    trauma_progress as u8
}

// ============================================================================
// ERRORS
// ============================================================================

/// Errors that can occur during save operations
#[derive(Debug)]
pub enum SaveError {
    /// Invalid save slot
    InvalidSlot(usize),
    /// I/O error
    IoError(String),
    /// Invalid save format
    InvalidFormat(String),
    /// Version mismatch
    VersionMismatch { expected: u32, found: u32 },
    /// Unexpected end of file
    UnexpectedEof,
    /// Save file corrupted
    Corrupted(String),
}

impl std::fmt::Display for SaveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SaveError::InvalidSlot(slot) => write!(f, "Invalid save slot: {}", slot),
            SaveError::IoError(msg) => write!(f, "I/O error: {}", msg),
            SaveError::InvalidFormat(msg) => write!(f, "Invalid save format: {}", msg),
            SaveError::VersionMismatch { expected, found } => {
                write!(f, "Save version mismatch: expected {}, found {}", expected, found)
            }
            SaveError::UnexpectedEof => write!(f, "Unexpected end of save file"),
            SaveError::Corrupted(msg) => write!(f, "Save file corrupted: {}", msg),
        }
    }
}

impl std::error::Error for SaveError {}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_anima_state() {
        let anima = AnimaState::new(0.5, 0.3, 0.7);
        let mut buffer = Vec::new();
        serialize_anima_state(&mut buffer, &anima);

        let mut cursor = 0;
        let result = deserialize_anima_state(&buffer, &mut cursor).unwrap();

        assert!((result.pleasure - 0.5).abs() < 0.001);
        assert!((result.arousal - 0.3).abs() < 0.001);
        assert!((result.dominance - 0.7).abs() < 0.001);
    }

    #[test]
    fn test_serialize_fronting_state() {
        let states = vec![
            FrontingState::None,
            FrontingState::Single("host".to_string()),
            FrontingState::Blended(vec!["a".to_string(), "b".to_string()]),
            FrontingState::Unknown,
        ];

        for state in states {
            let mut buffer = Vec::new();
            serialize_fronting_state(&mut buffer, &state);

            let mut cursor = 0;
            let result = deserialize_fronting_state(&buffer, &mut cursor).unwrap();

            match (&state, &result) {
                (FrontingState::None, FrontingState::None) => {}
                (FrontingState::Single(a), FrontingState::Single(b)) => assert_eq!(a, b),
                (FrontingState::Blended(a), FrontingState::Blended(b)) => assert_eq!(a, b),
                (FrontingState::Unknown, FrontingState::Unknown) => {}
                _ => panic!("Mismatch: {:?} vs {:?}", state, result),
            }
        }
    }

    #[test]
    fn test_serialize_reality_layer() {
        let layers = vec![
            RealityLayer::Grounded,
            RealityLayer::Fractured,
            RealityLayer::Shattered,
            RealityLayer::Custom("nightmare".to_string()),
        ];

        for layer in layers {
            let mut buffer = Vec::new();
            serialize_reality_layer(&mut buffer, &layer);

            let mut cursor = 0;
            let result = deserialize_reality_layer(&buffer, &mut cursor).unwrap();

            match (&layer, &result) {
                (RealityLayer::Grounded, RealityLayer::Grounded) => {}
                (RealityLayer::Fractured, RealityLayer::Fractured) => {}
                (RealityLayer::Shattered, RealityLayer::Shattered) => {}
                (RealityLayer::Custom(a), RealityLayer::Custom(b)) => assert_eq!(a, b),
                _ => panic!("Mismatch: {:?} vs {:?}", layer, result),
            }
        }
    }

    #[test]
    fn test_save_file_roundtrip() {
        let state = GameState::new();
        let manager = SaveManager::new("/tmp/test_saves");

        let save_file = manager.create_save_file(&state, "Test Save").unwrap();
        let serialized = serialize_save_file(&save_file).unwrap();
        let deserialized = deserialize_save_file(&serialized).unwrap();

        assert_eq!(save_file.version, deserialized.version);
        assert_eq!(save_file.metadata.name, deserialized.metadata.name);
        assert_eq!(save_file.data.game_time, deserialized.data.game_time);
    }

    #[test]
    fn test_save_manager_slot_validation() {
        let manager = SaveManager::new("/tmp/test_saves");

        // Invalid slot should fail
        let result = manager.save_to_slot(10, &GameState::new());
        assert!(matches!(result, Err(SaveError::InvalidSlot(_))));
    }
}
