# Plurality Game Engine Extraction Specification

**Version:** 0.1.0
**Status:** Draft
**Date:** 2026-02-01
**Target:** Aether Framework (DAEMONIORUM game)

---

## 1. Overview

The `sigil-parser` crate contains game engine code that was incorrectly placed in the language parser. This code should be extracted and moved to `aether-framework/aether-engine/games/daemoniorum/`.

### 1.1 Problem

The plurality module at `sigil-lang/parser/src/plurality/` mixes two concerns:

1. **Language extensions** (belongs in parser): Syntax for `alter`, `switch`, `headspace`, etc.
2. **Game engine runtime** (belongs in aether): Combat, dialogue, save systems, perception

### 1.2 Principle

A parser should parse. Game logic belongs in the game engine.

---

## 2. Files to Extract

The following files should be moved from `sigil-lang/parser/src/plurality/` to aether:

| File | Description | Lines | Dependencies |
|------|-------------|-------|--------------|
| `combat.rs` | Combat system, abilities, damage calculation | ~980 | `runtime.rs` types |
| `dialogue.rs` | Dialogue trees, NPC responses, choice effects | ~830 | `runtime.rs` types |
| `game_loop.rs` | Game state machine, phases, player input | ~600 | `runtime.rs`, `combat.rs`, `dialogue.rs` |
| `save_system.rs` | Save/load game state, serialization | ~400 | `runtime.rs`, `game_loop.rs` |
| `perception.rs` | Reality layers, entity visibility, environmental effects | ~1020 | `runtime.rs` types |

### 2.1 Files to Keep in Parser

| File | Description | Reason |
|------|-------------|--------|
| `ast.rs` | AST nodes for plurality syntax | Language definition |
| `lexer.rs` | Token extensions for plurality | Language definition |
| `parser.rs` | Parsing rules for plurality constructs | Language definition |
| `typeck.rs` | Type checking for plurality types | Language definition |
| `codegen.rs` | Code generation for plurality | Language definition |
| `mod.rs` | Module organization | Needs updating after extraction |

### 2.2 Shared Types (runtime.rs)

`runtime.rs` contains both language-level types and game runtime types. It should be split:

**Keep in parser (type definitions):**
- `AnimaState` - PAD emotional model
- `Alter` - Alter definition
- `AlterCategory` - Enum of alter types
- `AlterPresenceState` - Fronting state enum
- `FrontingState` - Who is fronting
- `RealityLayer` - Reality layer enum
- `Trigger` / `TriggerCategory` - Trigger definitions

**Move to aether (runtime behavior):**
- `PluralSystem` - Runtime system management
- `SwitchRequest` / `SwitchResult` - Runtime switching logic
- `TriggerResult` - Runtime trigger handling
- `MemoryAccess` - Runtime memory mechanics

---

## 3. Extracted Module Structure

Suggested structure in aether:

```
aether-framework/aether-engine/games/daemoniorum/src/
├── plurality/
│   ├── mod.rs
│   ├── combat.rs        # From sigil-parser
│   ├── dialogue.rs      # From sigil-parser
│   ├── game_loop.rs     # From sigil-parser
│   ├── perception.rs    # From sigil-parser
│   ├── save_system.rs   # From sigil-parser
│   └── runtime.rs       # Runtime portions from sigil-parser
```

---

## 4. Key Systems to Extract

### 4.1 Combat System (`combat.rs`)

- `CombatState` - Active combat encounter
- `Combatant` - Entity in combat
- `Ability` / `AbilityEffect` - Combat abilities
- `CombatAction` / `CombatResult` - Action resolution
- `apply_damage_emotional_effect()` - AnimaState integration
- `calculate_arousal_hit_modifier()` - Yerkes-Dodson curve

### 4.2 Dialogue System (`dialogue.rs`)

- `DialogueManager` - Dialogue state machine
- `DialogueNode` / `DialogueResponse` - Conversation tree
- `DialogueEffect` - Effects from choices
- `DialogueTone` - Emotional tone of responses
- `filter_responses_by_emotional_state()` - AnimaState integration
- `can_disclose_emotion()` - Expressiveness check

### 4.3 Game Loop (`game_loop.rs`)

- `GameLoop` - Main game loop
- `GameState` - Current game state
- `GamePhase` - Exploration/Combat/Dialogue/Cutscene
- `PlayerInput` - Input handling
- `GameEvent` - Event system

### 4.4 Perception System (`perception.rs`)

- `PerceptionState` - Current perception state
- `PerceptionManager` - Perception updates
- `RealityLayerTransition` - Layer transitions
- `EnvironmentalOverlay` - Visual effects
- `EntityVisibility` - What entities are visible
- `apply_emotional_modifiers()` - AnimaState integration

### 4.5 Save System (`save_system.rs`)

- `SaveManager` - Save/load operations
- `SaveData` - Serializable game state
- Serialization helpers for game types

---

## 5. AnimaState Integration

The recently added AnimaState integration should move with the game engine code:

### 5.1 Perception Integration
- High arousal → increased intensity (hypervigilance)
- Low stability → perception flickers
- Fear state → threat-focused mode
- Calm state → normalized perception

### 5.2 Combat Integration
- Damage taken → decreases pleasure, increases arousal
- Damage dealt → increases dominance
- Combat triggers → trauma response (stability decrease)
- Arousal affects hit chance (Yerkes-Dodson)

### 5.3 Dialogue Integration
- Low dominance → filters assertive options
- High expressiveness → enables emotional disclosure
- Emotional state affects available tones

---

## 6. Dependencies After Extraction

### 6.1 Aether depends on Sigil (types only)

```toml
[dependencies]
sigil-parser = { version = "0.4", default-features = false, features = ["plurality-types"] }
```

Aether imports type definitions:
- `AnimaState`
- `Alter`, `AlterCategory`
- `RealityLayer`
- `Trigger`, `TriggerCategory`

### 6.2 Sigil parser has no game dependencies

The parser should have zero knowledge of game logic.

---

## 7. Dependency Analysis

**Status:** Analyzed 2026-02-01

### 7.1 External Dependencies

| Check | Result |
|-------|--------|
| External crates depend on game engine? | **NO** |
| lib.rs imports game engine? | **NO** |
| main.rs imports game engine? | **NO** |
| Code outside plurality/ uses game types? | **NO** |

**Conclusion:** Game engine code is fully self-contained. Safe to remove.

### 7.2 Internal Dependencies (within plurality/)

```
game_loop.rs
  ├── imports combat.rs (CombatState, CombatResult, Enemy)
  ├── imports dialogue.rs (DialogueManager)
  ├── imports perception.rs (PerceptionManager)
  └── imports runtime.rs (PluralSystem, etc.)

save_system.rs
  ├── imports game_loop.rs (GameState, SaveData, etc.)
  └── imports runtime.rs (Alter, AnimaState, etc.)

combat.rs
  └── imports runtime.rs (AnimaState, Trigger, etc.)

dialogue.rs
  └── imports runtime.rs (AnimaState, etc.)

perception.rs
  └── imports runtime.rs (AnimaState, RealityLayer, etc.)
```

**All game engine files depend on runtime.rs types.**

### 7.3 runtime.rs Split Required

The runtime.rs file contains both:
- **Type definitions** (keep): AnimaState, Alter, AlterCategory, RealityLayer, Trigger, etc.
- **Runtime behavior** (remove): PluralSystem, SwitchRequest/Result, TriggerResult

---

## 8. Backup Location

The game engine files have been backed up prior to removal:

```
/tmp/plurality-extraction-backup/
├── combat.rs      (33KB)
├── dialogue.rs    (51KB)
├── game_loop.rs   (28KB)
├── perception.rs  (34KB)
├── runtime.rs     (21KB)  # Contains both types and runtime - needs splitting
└── save_system.rs (35KB)
```

**Total:** ~200KB of game engine code to extract.

---

## 9. Migration Steps

1. **Create feature flag** in sigil-parser: `plurality-types` (types only, no runtime)
2. **Create plurality crate** in aether with game engine code
3. **Update imports** in aether to use sigil-parser types
4. **Remove game engine files** from sigil-parser
5. **Update sigil-parser mod.rs** to only export language extensions
6. **Update tests** - game tests move to aether, language tests stay

---

## 10. Pre-existing Issues

### 10.1 Duplicate Type Definitions

`AlterCategory`, `CoConChannel`, and `RealityLayer` are defined in both `ast.rs` and `runtime.rs`. These should be consolidated to a single location.

### 10.2 Failing Parser Test

`test_parse_alter_def_basic` fails due to whitespace handling. This is a language-level bug that stays with the parser team.

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-02-01 | Initial extraction spec |
