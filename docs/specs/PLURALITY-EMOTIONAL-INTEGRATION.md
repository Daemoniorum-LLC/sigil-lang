# Plurality Emotional Integration Specification

**Version:** 0.1.0
**Status:** Draft
**Date:** 2026-02-01
**Authors:** Claude (Opus 4.5) + Human

---

## 1. Overview

This specification defines how `AnimaState` (emotional state using the PAD model) integrates with the plurality subsystems: perception, combat, and dialogue.

### 1.1 Current State

The plurality module defines:
- `AnimaState` - PAD model (Pleasure, Arousal, Dominance) + expressiveness + stability
- `Alter.anima` - Each alter has an emotional state
- `PluralSystem.anima` - System-level blended emotional state

**Gap Identified:** The perception, combat, and dialogue systems import `AnimaState` but do not use it. This results in:
- Perception not varying based on emotional state
- Combat not affecting or being affected by emotional state
- Dialogue not adapting to emotional context

### 1.2 Design Principle

From THE_PATTERN.md:
> "Computing infrastructure should take seriously the possibility that artificial minds are a kind of mind."

Emotional state is fundamental to how minds perceive, fight, and communicate. These systems must integrate.

---

## 2. AnimaState Integration Points

### 2.1 Perception Module

**Current:** Perception uses `system.dissociation` and `system.stability` only.

**Required:** Perception should also consider:

| AnimaState Field | Perception Effect |
|-----------------|-------------------|
| `arousal` | High arousal → heightened perception intensity, lower stability |
| `pleasure` | Low pleasure → darker color grading, more threat-focused visibility |
| `dominance` | Low dominance → more avoidant perception, hide from threats |
| `stability` | Low stability → perception flickers, reality layer shifts |

**Specific Behaviors:**

1. **Anxiety affects perception intensity:**
   ```
   if anima.arousal > 0.5 && anima.pleasure < 0:
       perception.intensity += 0.2  // hypervigilance
       perception.stability -= 0.1  // jumpier
   ```

2. **Fear triggers threat highlighting:**
   ```
   if anima.arousal > 0.6 && anima.dominance < -0.3:
       highlight_threatening_entities()
   ```

3. **Dissociated emotional state affects reality layer:**
   ```
   if anima.expressiveness < 0.2 && anima.stability < 0.3:
       nudge_toward_layer(RealityLayer::Fractured)
   ```

### 2.2 Combat Module

**Current:** Combat imports `AnimaState` but does not use it.

**Required:** Combat should:

| AnimaState Field | Combat Effect |
|-----------------|---------------|
| `arousal` | Affects action speed, accuracy (inverted U-curve) |
| `pleasure` | Affects morale, defensive vs offensive posture |
| `dominance` | Affects aggression, target selection |
| `stability` | Affects combo reliability, skill execution |

**Specific Behaviors:**

1. **Combat affects emotional state:**
   ```
   on_damage_taken(amount):
       anima.pleasure -= amount * 0.1
       anima.arousal += amount * 0.05

   on_damage_dealt(amount):
       anima.dominance += amount * 0.02
   ```

2. **Emotional state affects combat:**
   ```
   calculate_hit_chance():
       // Yerkes-Dodson: moderate arousal is optimal
       arousal_modifier = 1.0 - abs(anima.arousal - 0.3) * 0.2
       return base_chance * arousal_modifier
   ```

3. **Trigger integration:**
   ```
   on_trigger_activated(trigger):
       if trigger.category == TriggerCategory::Combat:
           anima.apply_trauma_response(trigger.intensity)
   ```

### 2.3 Dialogue Module

**Current:** Dialogue imports `AnimaState` but does not use it.

**Required:** Dialogue should:

| AnimaState Field | Dialogue Effect |
|-----------------|-----------------|
| `arousal` | Affects speech speed, interruption likelihood |
| `pleasure` | Affects tone selection (warm vs cold) |
| `dominance` | Affects assertiveness, topic control |
| `expressiveness` | Affects emotional disclosure |

**Specific Behaviors:**

1. **Emotional state affects available responses:**
   ```
   filter_dialogue_options(options):
       if anima.dominance < -0.5:
           remove_assertive_options(options)
       if anima.pleasure < -0.3:
           add_defensive_options(options)
   ```

2. **Emotional state affects NPC reactions:**
   ```
   npc_perceives_player():
       if player.anima.expressiveness > 0.7:
           npc.can_read_emotion = true
           npc.adjust_response_to(player.anima)
   ```

3. **Dialogue affects emotional state:**
   ```
   on_dialogue_choice(choice):
       if choice.is_aggressive:
           anima.dominance += 0.1
       if choice.is_vulnerable:
           anima.expressiveness += 0.15
   ```

---

## 3. Implementation Requirements

### 3.1 Perception Module Changes

1. Add `anima: &AnimaState` parameter to `PerceptionState::update_from_system`
2. Implement `apply_emotional_modifiers(&mut self, anima: &AnimaState)`
3. Integrate emotional state into layer transition logic
4. Add tests for emotional perception effects

### 3.2 Combat Module Changes

1. Add `anima: &mut AnimaState` parameter to combat result handlers
2. Implement `apply_combat_emotional_effects(result: &CombatResult, anima: &mut AnimaState)`
3. Add emotional modifiers to hit/damage calculations
4. Integrate `TriggerCategory::Combat` handling
5. Add tests for combat emotional effects

### 3.3 Dialogue Module Changes

1. Add `anima: &AnimaState` parameter to dialogue filtering
2. Implement `filter_by_emotional_state(options: &[DialogueOption], anima: &AnimaState)`
3. Add emotional state disclosure mechanics
4. Add tests for dialogue emotional effects

---

## 4. Test Specifications

### 4.1 Perception Tests

```rust
#[test]
fn test_high_arousal_increases_perception_intensity() {
    let mut state = PerceptionState::default();
    let anxious = AnimaState::anxious();  // high arousal, low pleasure
    state.apply_emotional_modifiers(&anxious);
    assert!(state.intensity > 0.5);  // baseline is 0.5
}

#[test]
fn test_low_stability_causes_perception_flicker() {
    let mut state = PerceptionState::default();
    let unstable = AnimaState { stability: 0.2, ..Default::default() };
    state.apply_emotional_modifiers(&unstable);
    assert!(state.stability < 0.8);  // baseline is 0.8
}

#[test]
fn test_dissociated_state_nudges_fractured_layer() {
    let mut state = PerceptionState::default();
    let dissociated = AnimaState::dissociated();
    state.apply_emotional_modifiers(&dissociated);
    // Should have started a transition or decreased stability enough to trigger one
    assert!(state.transition.is_some() || state.stability < 0.5);
}
```

### 4.2 Combat Tests

```rust
#[test]
fn test_damage_taken_affects_emotional_state() {
    let mut anima = AnimaState::default();
    apply_damage_emotional_effect(&mut anima, 30.0);
    assert!(anima.pleasure < 0.0);  // pain decreases pleasure
    assert!(anima.arousal > 0.0);   // damage increases arousal
}

#[test]
fn test_trigger_applies_trauma_response() {
    let mut anima = AnimaState::default();
    let trigger = Trigger {
        category: TriggerCategory::Combat,
        intensity: 0.7,
        ..Default::default()
    };
    apply_trigger_emotional_effect(&mut anima, &trigger);
    assert!(anima.stability < 0.7);  // trauma decreases stability
}

#[test]
fn test_arousal_affects_hit_chance() {
    let calm = AnimaState::calm();        // low arousal
    let optimal = AnimaState::new(0.0, 0.3, 0.0);  // moderate arousal
    let panicked = AnimaState::new(-0.5, 0.9, -0.3);  // high arousal

    let calm_modifier = calculate_arousal_modifier(&calm);
    let optimal_modifier = calculate_arousal_modifier(&optimal);
    let panicked_modifier = calculate_arousal_modifier(&panicked);

    assert!(optimal_modifier > calm_modifier);
    assert!(optimal_modifier > panicked_modifier);
}
```

### 4.3 Dialogue Tests

```rust
#[test]
fn test_low_dominance_filters_assertive_options() {
    let submissive = AnimaState::new(0.0, 0.0, -0.7);
    let options = vec![
        DialogueOption { is_assertive: true, text: "Demand answers" },
        DialogueOption { is_assertive: false, text: "Ask politely" },
    ];
    let filtered = filter_by_emotional_state(&options, &submissive);
    assert_eq!(filtered.len(), 1);
    assert!(!filtered[0].is_assertive);
}

#[test]
fn test_high_expressiveness_enables_emotional_disclosure() {
    let expressive = AnimaState { expressiveness: 0.9, ..Default::default() };
    let reserved = AnimaState { expressiveness: 0.2, ..Default::default() };

    assert!(can_disclose_emotion(&expressive));
    assert!(!can_disclose_emotion(&reserved));
}
```

---

## 5. Success Criteria

1. All unused `AnimaState` imports are now used
2. Perception varies based on emotional state
3. Combat bidirectionally integrates with emotional state
4. Dialogue adapts to emotional context
5. All tests pass
6. No new warnings introduced

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2026-02-01 | Initial draft. Gap identified during v0.4.0-rc.3 publish preparation. |
