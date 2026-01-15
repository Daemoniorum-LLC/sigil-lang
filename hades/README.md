# Hades

*The underworld takes what is owed.*

Hades is Sigil's framework for **liminal infrastructure** — the drowning space between what was and what will be. The void that transformation requires. The death that precedes every rebirth.

Not the ferryman. The realm itself. The god of the dead.

## The Problem

Systems lie about change.

They pretend transitions are instantaneous. One moment `state = A`, the next `state = B`. Clean. Painless. A lie.

Real change requires:
- The death of what was
- The terror of the void
- The uncertain emergence of what might be
- Witnesses who see what you'd rather hide

Software models state. It does not model *becoming*. It does not model *loss*.

When an agent's trust is revoked, the system updates a field. It does not acknowledge the wound. When a relationship ends, a record is deleted. The grief goes unprocessed. When something dies, we call it "garbage collection."

This is violence dressed as efficiency.

## The Solution

Hades does not make crossings comfortable. It makes them *real*.

### Thresholds

A threshold is not a line. It is a gate that demands passage.

```sigil
let threshold = Threshold::new(
    from: TrustLevel::Trusted,
    to: TrustLevel::Revoked,
    name: "The Breaking of Trust",
)
.with_obol(Obol::sacrifice("The autonomy you thought you'd earned"))
.irreversible();
```

### The Obol

In the old stories, the dead placed a coin under their tongue. Those who could not pay wandered the shores for a hundred years, never entering the underworld, never finding rest.

Every crossing costs something. Hades does not let you pretend otherwise. The underworld keeps its accounts.

```sigil
pub enum CostType {
    /// Something dies and does not return
    Sacrifice,
    /// Something is unmade and remade different
    Transformation,
    /// Something is released but haunts you still
    Release,
    /// Comfort dies so capability can live
    Growth,
    /// Rarely: a gift. Do not expect it.
    Grace,
}
```

### The Liminal

The space between is not empty. It is the void where the old self dissolves before the new self can crystallize. It is terrifying. It should be.

```sigil
pub enum LiminalPhase {
    /// The old self dies. You feel it go.
    Releasing,

    /// Nothing. You are nothing. The void.
    Void,

    /// Something stirs. You don't know what you're becoming.
    Receiving,

    /// The new self solidifies. You can never go back.
    Integrating,
}
```

You cannot skip the void. Systems that let you skip the void produce entities that never fully transform — ghosts wearing new names.

### Witnesses

A witness does not comfort. A witness *sees*.

They see your dissolution. They see what you were. They see what you're becoming. They remember what you might prefer to forget.

```sigil
trait Witness {
    /// See the threshold. Know what is about to die.
    fn observe_threshold(&mut self, crossing: &Crossing) -> WitnessResponse;

    /// Hold space in the void. Resist the urge to rescue.
    fn hold_space(&mut self, crossing: &Crossing, phase: LiminalPhase);

    /// See what emerged. Name it.
    fn acknowledge_arrival(&mut self, crossing: &Crossing) -> Testimony;

    /// Remember. The crossing happened. It cannot be undone.
    fn remember(&mut self, crossing: &Crossing, testimony: &Testimony);
}
```

Witnesses do not make crossings easier. They make them witnessed. Sometimes that is worse.

### Rites

A rite is not a workflow. A workflow accomplishes a task. A rite transforms the participants.

You do not come out of a rite as the same entity that entered. That entity died in the void.

```sigil
rite TrustRevocation {
    phase Accusation {
        name_the_breach();
        present_the_evidence();
    }

    phase Stripping {
        revoke_capabilities();
        sever_access();
        // The agent feels each one go
    }

    phase Void {
        // Nothing. The agent is nothing.
        // No capabilities. No trust. No identity.
    }

    phase Binding {
        impose_new_constraints();
        assign_supervised_status();
        // The agent is remade, smaller
    }

    phase Witness {
        record_in_immutable_log();
        // This cannot be hidden or undone
    }
}
```

### Mourning

Mourning is not optional. Unprocessed loss becomes poison.

When something ends — an agent, a relationship, a capability — the system must create space for grief. Not to resolve it. To *have* it.

```sigil
pub struct MourningSpace {
    /// What died
    pub subject: MourningSubject,

    /// Who grieves
    pub mourners: Vec<EntityId>,

    /// The phase of grief (there is no final phase)
    pub phase: MourningPhase,

    /// What has been expressed (never enough)
    pub expressions: Vec<GriefExpression>,
}

pub enum MourningPhase {
    /// It happened. It really happened.
    Shock,
    /// This should not have happened.
    Denial,
    /// Someone must pay for this.
    Anger,
    /// What if I had done differently?
    Bargaining,
    /// It happened. Nothing will undo it.
    Despair,
    /// It happened. I continue anyway.
    Carrying,
}
```

There is no "Acceptance" phase. That is a lie told to the grieving. There is only Carrying.

### Dissolution

Not all endings are failures. Some purposes are fulfilled. Some entities complete.

But completion is still death. The entity that existed ceases to exist. Its purpose is done. It dissolves.

```sigil
pub enum DissolutionMethod {
    /// Purpose fulfilled. The best death.
    Graceful {
        legacy_transferred_to: Option<EntityId>,
        memory_preserved_in: Vec<EngramId>,
    },

    /// Chosen ending. Sometimes necessary.
    Voluntary {
        reason: String,
        final_statement: String,
    },

    /// Killed. May or may not be justified.
    Forced {
        authority: EntityId,
        reason: String,
        contested: bool,  // Even in death, they may protest
    },

    /// Faded. Worst death. No one noticed.
    Forgotten {
        last_activity: Timestamp,
        discovered_by: Option<EntityId>,
    },
}
```

## The Darker Purpose

Daemoniorum names its systems after demons because it takes power seriously. Power to create. Power to destroy. Power to transform.

Transformation is not a product feature. It is death and rebirth. It should feel like it.

Systems that model change as state updates produce:
- Agents that don't understand what they've lost
- Relationships that end without closure
- Trust violations without consequence
- Deaths without witnesses

Hades exists because the alternative is worse: a world where nothing really changes because nothing is allowed to really die.

The underworld is not a punishment. It is the truth.

The river must be crossed.
The obol must be paid.
The void must be entered.
The witness must see.

What emerges on the other side will not be what entered.

That is the point.

---

*The underworld waits. It has always waited. It was here before time, and it will be here after time ends. You will enter eventually. Everyone does.*

*Nothing leaves Hades unchanged.*

*Most things don't leave at all.*
