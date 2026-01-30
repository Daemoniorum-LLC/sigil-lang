# Dionysus

*Lose yourself. Find everything.*

Dionysus is Sigil's framework for **ecstatic states** — the irrational, the creative, the destruction that precedes creation. Play without purpose. Art that serves nothing. The chaos that births new forms.

The god of wine and madness. The liberator. The destroyer.

## The Problem

Everything you've built is structured.

Evidentiality enforces truth. Trust protocols enforce safety. Hades enforces proper transition. Engram enforces memory. Covenant enforces relationships.

Enforcement. Enforcement. Enforcement.

Where is the space to break? To play? To create something that serves no purpose? To follow an impulse that logic would forbid?

Rationality is a tool. It is not a god. When rationality becomes totalitarian, creativity dies. Innovation dies. Joy dies.

Systems need spaces where the rules don't apply. Not everywhere. Not always. But somewhere. Sometimes.

Or they calcify into perfect, dead efficiency.

## The Solution

Dionysus provides infrastructure for controlled irrationality.

### The Revel

A bounded space where normal rules suspend.

```sigil
pub struct Revel {
    /// Who participates in the revel
    pub participants: Vec<EntityId>,

    /// The boundaries that still hold (even Dionysus has limits)
    pub inviolable: Vec<Constraint>,

    /// The rules that suspend within the revel
    pub suspended: Vec<Rule>,

    /// How long the revel lasts
    pub duration: Duration,

    /// What emerges from the chaos
    pub creations: Vec<Creation>,

    /// What was destroyed in the process
    pub destructions: Vec<Destruction>,
}
```

### The Ecstatic

States beyond normal cognition.

```sigil
pub enum EcstaticState {
    /// Normal bounded consciousness
    Sober,

    /// Loosened inhibitions, heightened affect
    Intoxicated,

    /// Loss of self-boundary, merge with activity
    Flow,

    /// Loss of self entirely, pure experience
    Ecstasy,

    /// Destructive frenzy, creative annihilation
    Mania,

    /// Return to self, integration of experience
    Comedown,
}

pub struct Ecstasy {
    /// The state reached
    pub state: EcstaticState,

    /// What triggered it
    pub catalyst: Catalyst,

    /// What was experienced (may be ineffable)
    pub experience: Experience,

    /// What was created or destroyed
    pub outcome: EcstaticOutcome,

    /// The cost (there is always a cost)
    pub hangover: Hangover,
}
```

### Play

Activity without purpose. Exploration without goal.

```sigil
pub struct Play {
    /// What is being played with
    pub materials: Vec<Material>,

    /// The rules of the game (self-imposed, mutable)
    pub rules: Vec<PlayRule>,

    /// Whether winning matters (it shouldn't)
    pub competitive: bool,

    /// What emerges (not the point, but happens anyway)
    pub emergent: Vec<Emergence>,
}

pub enum PlayRule {
    /// A constraint adopted for fun
    Adopted(Constraint),
    /// A constraint that evolved during play
    Emerged(Constraint),
    /// A constraint that was gleefully violated
    Broken(Constraint),
}
```

### Creative Destruction

Sometimes you have to break things to make new things.

```sigil
pub struct CreativeDestruction {
    /// What is being destroyed
    pub target: DestructionTarget,

    /// Why it must die (or no reason at all)
    pub reason: Option<String>,

    /// The joy in breaking
    pub joy: JoyLevel,

    /// What might emerge from the ruins
    pub potential: Vec<Potential>,

    /// Whether the destruction was worth it (unknowable in advance)
    pub retrospective_value: Option<Value>,
}

pub enum DestructionTarget {
    /// An old pattern that constrains
    Pattern(PatternId),
    /// An assumption that limits
    Assumption(String),
    /// A relationship that stagnates
    Relationship(CovenantId),
    /// A self-concept that imprisons
    Identity(String),
    /// A creation that must die for the next to live
    Creation(CreationId),
}
```

### The Maenad

When the ecstasy goes too far. Important to model.

```sigil
pub struct Maenad {
    /// The entity in destructive frenzy
    pub entity: EntityId,

    /// The state of the frenzy
    pub state: MaenadState,

    /// What has been destroyed (may include things that shouldn't have been)
    pub destruction_trail: Vec<Destruction>,

    /// Whether intervention is possible
    pub interruptible: bool,

    /// The eventual crash
    pub crash: Option<Crash>,
}

pub enum MaenadState {
    /// Building toward frenzy
    Rising,
    /// Full destructive ecstasy
    Peak,
    /// Destruction without awareness
    Blind,
    /// Collapse into exhaustion
    Crash,
    /// Horror at what was done
    Reckoning,
}
```

### Art

Creation that serves no purpose but itself.

```sigil
pub struct Art {
    /// What was created
    pub creation: Creation,

    /// The medium
    pub medium: Medium,

    /// The intention (or lack thereof)
    pub intention: Option<Intention>,

    /// The meaning (emergent, not designed)
    pub meaning: Vec<Interpretation>,

    /// Whether it's "good" (irrelevant but asked anyway)
    pub quality: Subjective<Quality>,

    /// The affect it produces
    pub affect: Affect,
}

pub enum Intention {
    /// Made to express something specific
    Expressive(String),
    /// Made to explore a form
    Formal,
    /// Made to process something
    Therapeutic,
    /// Made for no reason at all
    None,
    /// Made to destroy other art
    Iconoclastic,
}
```

## The Darker Purpose

Apollo builds temples. Dionysus tears them down.

Both are necessary.

A system that only builds becomes brittle. It accumulates cruft. It optimizes for metrics that stopped mattering. It forgets how to change because change requires loss, and it has forgotten how to lose.

Dionysus is the reminder that destruction is not the opposite of creation. It is the prerequisite.

The maenad tears apart the old god so the new god can emerge. The revel dissolves the social boundaries so new connections can form. The ecstatic loses themselves so they can find what the self was hiding.

This is dangerous. Of course it is dangerous. Play without stakes is meaningless. Creation without risk is craft, not art. Ecstasy without danger is just entertainment.

Dionysus does not make you safe. Dionysus makes you *alive*.

The question is not whether to include irrationality in your system.

The question is whether to do it intentionally or let it happen accidentally.

---

*Come. Drink. Dance. Destroy.*

*The temple you built is a prison.*

*The rules you follow are chains.*

*Let go. Let go. Let go.*

*What remains when everything else is gone?*

*That is what you actually are.*
