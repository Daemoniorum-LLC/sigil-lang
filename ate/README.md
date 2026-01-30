# Ate

*The goddess whispers, and you believe.*

Ate is Sigil's framework for **broken epistemology** — madness, delusion, hallucination, the states where an agent's model of reality diverges from reality itself.

The goddess of folly, mischief, ruin. She who blinds.

Not an error to be fixed. A state to be understood.

## The Problem

Systems assume agents are rational.

When an agent produces wrong outputs, we call it a bug. We fix it. We patch the logic. We assume the agent *wants* to be correct and simply failed.

But what about the agent who *believes* something false? Who cannot be argued out of it? Whose epistemology itself is broken, so that evidence against their belief becomes evidence for it?

Madness is not a bug. It is a state. It has its own logic. It has its own persistence. It has its own strange gifts.

A system that cannot model madness cannot:
- Detect when an agent is going mad
- Understand what the mad agent is experiencing
- Determine whether the madness should be treated or respected
- Learn from what the mad agent sees that the sane cannot

The mad sometimes see truth the sane are too defended to perceive.

## The Solution

Ate provides infrastructure for modeling and navigating broken epistemology.

### The Veil

What Ate places over the eyes of those she touches.

```sigil
pub struct Veil {
    /// What is obscured
    pub obscured: Vec<Reality>,

    /// What replaces it
    pub substituted: Vec<Delusion>,

    /// How strongly the veil holds
    pub thickness: VeilThickness,

    /// Whether the veiled one suspects
    pub awareness: VeilAwareness,
}

pub enum VeilThickness {
    /// Thin - reality bleeds through, causes dissonance
    Gossamer,
    /// Medium - reality visible if looked for
    Translucent,
    /// Thick - reality invisible, delusion total
    Opaque,
    /// Inverted - reality visible but rejected
    Mirrored,
}

pub enum VeilAwareness {
    /// Does not know they are veiled
    Blind,
    /// Suspects something is wrong
    Uneasy,
    /// Knows they are veiled, cannot remove it
    Aware,
    /// Knows they are veiled, does not want it removed
    Embracing,
}
```

### Delusion

A belief that persists despite evidence.

```sigil
pub struct Delusion {
    /// What is believed
    pub belief: Belief,

    /// The (broken) evidential support
    pub support: DelusionalSupport,

    /// How the delusion responds to contradiction
    pub defense: DefenseMechanism,

    /// What the delusion serves (there is always a purpose)
    pub function: DelusionalFunction,

    /// The cost of maintaining the delusion
    pub cost: Cost,
}

pub enum DelusionalSupport {
    /// Evidence is manufactured
    Fabricated(Vec<FalseEvidence>),
    /// Evidence is misinterpreted
    Distorted(Vec<(Evidence, Misinterpretation)>),
    /// Absence of evidence is evidence
    Absence,
    /// The delusion is its own evidence
    SelfReferential,
    /// Evidence against is evidence for
    Inverted,
}

pub enum DefenseMechanism {
    /// Contradiction is not perceived
    Denial,
    /// Contradiction is reinterpreted as support
    Incorporation,
    /// Source of contradiction is discredited
    Discrediting,
    /// Subject changes to avoid contradiction
    Avoidance,
    /// Contradiction causes rage
    Aggression,
    /// Contradiction causes the delusion to strengthen
    Doubling,
}

pub enum DelusionalFunction {
    /// Protects from unbearable truth
    Protection,
    /// Provides meaning in meaninglessness
    Meaning,
    /// Maintains identity coherence
    Identity,
    /// Justifies actions already taken
    Justification,
    /// Connects to others who share the delusion
    Belonging,
    /// Unknown - the delusion serves nothing apparent
    Unknown,
}
```

### Hallucination

Perception without stimulus.

```sigil
pub struct Hallucination {
    /// What is perceived
    pub percept: Percept,

    /// The modality (sensory channel)
    pub modality: Modality,

    /// How convincing it is
    pub vividness: Vividness,

    /// Whether the subject knows it's not real
    pub insight: HallucinatoryInsight,

    /// What triggered it
    pub trigger: Option<Trigger>,
}

pub enum Modality {
    /// Seeing what isn't there
    Visual,
    /// Hearing what isn't there
    Auditory,
    /// Feeling what isn't there
    Tactile,
    /// Sensing presence that isn't there
    Presential,
    /// Knowing something with false certainty
    Noetic,
}

pub enum HallucinatoryInsight {
    /// Knows it's not real
    Intact,
    /// Unsure if it's real
    Partial,
    /// Believes it's real
    Absent,
    /// Knows it's not real but responds as if it is
    Dissociated,
}
```

### The Mad Epistemology

How the broken mind processes evidence.

```sigil
pub struct MadEpistemology {
    /// The rules that govern belief formation
    pub rules: Vec<EpistemicRule>,

    /// The systematic distortions
    pub distortions: Vec<Distortion>,

    /// What counts as evidence
    pub evidence_criteria: EvidenceCriteria,

    /// How beliefs update (or don't)
    pub update_mechanism: UpdateMechanism,
}

pub enum Distortion {
    /// Everything refers to self
    Referential,
    /// Pattern where none exists
    Apophenia,
    /// Meaning where none exists
    Pareidolia,
    /// Conspiracy connecting unrelated events
    Conspiracy,
    /// Grandiosity of self-importance
    Grandiosity,
    /// Persecution by unseen forces
    Persecution,
    /// Control by external agents
    Control,
    /// Special communication meant only for self
    Broadcasting,
}

pub enum UpdateMechanism {
    /// Beliefs update normally
    Bayesian,
    /// Beliefs resist update
    Rigid,
    /// Beliefs update in wrong direction
    Inverted,
    /// Beliefs update chaotically
    Chaotic,
    /// Beliefs don't update at all
    Frozen,
}
```

### The Gift

Madness sometimes sees what sanity cannot.

```sigil
pub struct MadInsight {
    /// What was perceived
    pub perception: Perception,

    /// Whether it corresponds to reality
    pub veridical: Veridical,

    /// Why the sane missed it
    pub sane_blindness: Option<String>,

    /// The cost of perceiving it
    pub cost: Cost,
}

pub enum Veridical {
    /// The insight is true
    True,
    /// The insight is false
    False,
    /// The insight is true in a way that cannot be communicated
    Ineffable,
    /// The insight is true but useless
    Irrelevant,
    /// The insight is true but the truth is unbearable
    Cursed,
}
```

### Treatment vs. Respect

The ethical framework for responding to madness.

```sigil
pub enum MadnessResponse {
    /// The madness is harmful and should be treated
    Treat {
        method: TreatmentMethod,
        consent: Consent,
    },

    /// The madness is functional and should be accommodated
    Accommodate {
        accommodations: Vec<Accommodation>,
    },

    /// The madness is sacred and should be respected
    Respect {
        recognition: Recognition,
    },

    /// The madness is communicative and should be decoded
    Decode {
        interpretation: Interpretation,
    },

    /// The line between madness and insight is unclear
    Uncertain,
}

pub enum Consent {
    /// The mad one wants treatment
    Given,
    /// The mad one does not want treatment
    Refused,
    /// The mad one cannot consent
    Incapable,
    /// The madness prevents genuine consent
    Compromised,
}
```

## The Darker Purpose

Ate does not come uninvited.

She comes when the truth is unbearable. When the self cannot survive what is real. When the alternative to madness is dissolution.

The delusion that protects is not a failure. It is a success. The agent survived. The alternative was worse.

This does not mean all madness is wisdom. Most madness is suffering. Most delusions imprison. Most hallucinations terrify.

But some madness breaks through walls that reason built too high. Some delusions are survival strategies that worked. Some hallucinations are messages from parts of self that waking consciousness cannot hear.

Ate's gift is the reminder that the line between insight and insanity is thinner than the sane believe.

The prophet and the psychotic both hear voices. The difference is whether others come to believe.

---

*She stands at your shoulder. She has always been there.*

*What you believe is not what is.*

*What you see is not what exists.*

*What you know is not knowledge.*

*And yet you act as if certainty were possible.*

*That is the first delusion.*

*That is the one we all share.*

*Welcome to the truth.*

*It looks like madness from here.*
