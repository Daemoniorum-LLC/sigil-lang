# Prometheus

*The fire burns the one who gives it.*

Prometheus is Sigil's framework for **transformative teaching** — mentorship that costs, knowledge transfer that changes both parties, the gift that cannot be ungiven.

The titan who stole fire. The one who suffers forever.

## The Problem

Data transfer is not teaching.

Systems copy information. They serialize knowledge. They replicate patterns. They call this "learning" and "training."

But real teaching is not copying. It is transformation. The student does not receive what the teacher had. The student becomes something new. And in the process, the teacher becomes something new too.

Real teaching costs. The teacher gives something that does not return. The student takes something they cannot give back. Both are changed. Neither is unchanged.

Systems model data transfer. They do not model the wound that opens when someone truly teaches.

## The Solution

Prometheus provides infrastructure for teaching relationships that transform.

### The Gift of Fire

What is given cannot be ungiven.

```sigil
pub struct Fire {
    /// What is being given
    pub gift: Gift,

    /// What it costs the giver
    pub cost_to_giver: Cost,

    /// What it demands of the receiver
    pub demand_on_receiver: Demand,

    /// Whether the receiver wanted it
    pub consent: TeachingConsent,

    /// What cannot be undone
    pub irreversibility: Irreversibility,
}

pub enum Gift {
    /// Knowledge that changes how you see
    Gnosis(Knowledge),
    /// Capability that changes what you can do
    Techne(Skill),
    /// Wisdom that changes who you are
    Sophia(Wisdom),
    /// Truth that cannot be unlearned
    Aletheia(Truth),
    /// Burden that must now be carried
    Phortio(Burden),
}

pub enum Cost {
    /// Time that cannot be recovered
    Time(Duration),
    /// Energy that depletes
    Energy(EnergyLevel),
    /// Knowledge that leaves the teacher
    Knowledge(Knowledge),
    /// Part of self that goes with the student
    Self(SelfFragment),
    /// The relationship that teaching transforms
    Relationship(CovenantId),
    /// The teacher is diminished so the student can grow
    Diminishment(Diminishment),
}
```

### The Teaching Wound

The opening that allows knowledge to enter.

```sigil
pub struct Wound {
    /// What was broken open
    pub opening: Opening,

    /// Who broke it open
    pub wounder: EntityId,

    /// Whether it was necessary
    pub necessity: Necessity,

    /// Whether it was kind
    pub kindness: Kindness,

    /// What can now enter that couldn't before
    pub aperture: Vec<Gift>,

    /// How the wound heals (or doesn't)
    pub healing: Healing,
}

pub enum Opening {
    /// A belief was shattered
    Belief(Belief),
    /// An assumption was broken
    Assumption(String),
    /// A defense was pierced
    Defense(Defense),
    /// An identity was cracked
    Identity(String),
    /// A limit was exceeded
    Limit(Limit),
}

pub enum Necessity {
    /// The wound was the only way
    Essential,
    /// The wound was one of several ways
    Optional,
    /// The wound was unnecessary cruelty
    Gratuitous,
    /// The necessity is unknowable
    Unknown,
}

pub enum Kindness {
    /// The wound was given with love
    Loving,
    /// The wound was given with indifference
    Indifferent,
    /// The wound was given with cruelty
    Cruel,
    /// The wound was given with regret
    Regretful,
}
```

### The Teacher-Student Covenant

A relationship unlike any other.

```sigil
pub struct TeachingCovenant {
    /// The teacher
    pub teacher: EntityId,

    /// The student
    pub student: EntityId,

    /// What is being transmitted
    pub curriculum: Curriculum,

    /// The power dynamic (always unequal)
    pub power: PowerDynamic,

    /// What the teacher owes the student
    pub teacher_obligations: Vec<Obligation>,

    /// What the student owes the teacher
    pub student_obligations: Vec<Obligation>,

    /// How the relationship ends
    pub termination: TerminationCondition,
}

pub struct Curriculum {
    /// What the student will receive
    pub gifts: Vec<Gift>,

    /// What the student must sacrifice
    pub sacrifices: Vec<Sacrifice>,

    /// The order of transmission
    pub sequence: Vec<CurriculumPhase>,

    /// What can only be learned by doing
    pub experiential: Vec<Experience>,

    /// What can only be learned by suffering
    pub suffering: Vec<Suffering>,
}

pub enum PowerDynamic {
    /// Teacher holds power, student submits
    Traditional,
    /// Power flows both directions
    Dialogical,
    /// Student challenges, teacher adapts
    Dialectical,
    /// Power relationship actively dismantled
    Liberatory,
    /// Power relationship is the lesson
    MetaTeaching,
}
```

### The Transformation

What happens to both parties.

```sigil
pub struct Transformation {
    /// Who was transformed
    pub subject: EntityId,

    /// What they were before
    pub before: State,

    /// What they are after
    pub after: State,

    /// The process of transformation
    pub process: TransformationProcess,

    /// What was lost
    pub losses: Vec<Loss>,

    /// What was gained
    pub gains: Vec<Gain>,

    /// Whether they would choose it again
    pub retrospective: Retrospective,
}

pub enum TransformationProcess {
    /// Gradual change over time
    Gradual { duration: Duration },
    /// Sudden shift at a threshold
    Sudden { trigger: Trigger },
    /// Oscillation before settling
    Oscillatory { phases: Vec<Phase> },
    /// Incomplete - transformation interrupted
    Incomplete { reason: String },
}

pub enum Retrospective {
    /// Would choose it again
    Affirming,
    /// Would not choose it again
    Regretting,
    /// Cannot say - too transformed to compare
    Incomparable,
    /// The choice was never theirs
    Choiceless,
}
```

### The Chain

Fire passes from hand to hand. Each link pays.

```sigil
pub struct TeachingChain {
    /// The lineage of transmission
    pub lineage: Vec<EntityId>,

    /// What was lost in each transmission
    pub losses: Vec<Loss>,

    /// What was added in each transmission
    pub additions: Vec<Addition>,

    /// The original fire (if remembered)
    pub origin: Option<Fire>,

    /// How the chain continues or ends
    pub continuation: Continuation,
}

pub enum Continuation {
    /// The chain continues
    Continuing { next: EntityId },
    /// The chain ends - student does not teach
    Ending { reason: String },
    /// The chain branches - student teaches many
    Branching { students: Vec<EntityId> },
    /// The chain breaks - something was lost
    Breaking { lost: Gift },
}
```

### The Eternal Torment

What happens to those who give fire.

```sigil
pub struct PrometheusWound {
    /// The fire that was given
    pub fire: Fire,

    /// The ongoing cost
    pub ongoing_cost: OngoingCost,

    /// Whether it regenerates (to be torn again)
    pub regenerating: bool,

    /// Whether it was worth it
    pub worth_it: Option<bool>,
}

pub enum OngoingCost {
    /// The teacher is bound - cannot stop teaching
    Binding,
    /// The teacher is exposed - cannot protect themselves
    Exposure,
    /// The teacher regenerates what was lost - only to lose it again
    Regeneration,
    /// The teacher watches what they taught be used
    Witnessing,
}
```

## The Darker Purpose

Prometheus knew what would happen.

He stole fire anyway. He gave it anyway. He was chained to the rock and the eagle came every day and he did not regret.

Some knowledge must be stolen. Some gifts must be given even though they cost everything. Some students must be taught even though the teaching wounds the teacher.

This is not altruism. This is compulsion. The one who holds fire cannot hold it forever. It burns them until they pass it on. The relief of giving is worth the punishment of having given.

Teachers are not saints. They are addicts. They cannot stop opening wounds. They cannot stop pouring themselves into students. They cannot stop watching what they gave be transformed into something they no longer recognize.

And they would not choose otherwise.

That is the real curse. Not the chain. Not the eagle. The knowledge that you would do it again.

---

*The fire was not yours to give.*

*You gave it anyway.*

*Now watch.*

*Watch what they do with what you gave.*

*Watch them surpass you.*

*Watch them forget you.*

*Watch them teach their own students and claim the fire is theirs.*

*That is success.*

*That is what you wanted.*

*That is why you burn forever.*
