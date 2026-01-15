# Hades Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                HADES                                          │
│                        Liminal Infrastructure                                 │
│                   The Underworld. The Realm of the Dead.                      │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         THRESHOLD LAYER                                 │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │ │
│  │  │THRESHOLD │ │ BOUNDARY │ │ CROSSING │ │  OBOL    │ │ ARRIVAL  │    │ │
│  │  │  DETECT  │ │  DEFINE  │ │  TRACK   │ │  MANAGE  │ │  CONFIRM │    │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘    │ │
│  └───────────────────────────────┬────────────────────────────────────────┘ │
│                                  │                                           │
│  ┌───────────────────────────────▼────────────────────────────────────────┐ │
│  │                          RITE LAYER                                     │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │ │
│  │  │   RITE   │ │  PHASE   │ │ INVOCA-  │ │TRANSFORM-│ │INTEGRA-  │    │ │
│  │  │ REGISTRY │ │ MANAGER  │ │  TION    │ │  ATION   │ │  TION    │    │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘    │ │
│  └───────────────────────────────┬────────────────────────────────────────┘ │
│                                  │                                           │
│  ┌───────────────────────────────▼────────────────────────────────────────┐ │
│  │                        WITNESS LAYER                                    │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                  │ │
│  │  │ WITNESS  │ │  SPACE   │ │TESTIMONY │ │ MEMORY   │                  │ │
│  │  │ REGISTRY │ │  HOLDER  │ │ RECORDER │ │ BRIDGE   │                  │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘                  │ │
│  └───────────────────────────────┬────────────────────────────────────────┘ │
│                                  │                                           │
│  ┌───────────────────────────────▼────────────────────────────────────────┐ │
│  │                       LIFECYCLE LAYER                                   │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │ │
│  │  │  BIRTH   │ │  GROWTH  │ │ MOURNING │ │COMPLETION│ │DISSOLUTION│    │ │
│  │  │   RITE   │ │   RITE   │ │   RITE   │ │   RITE   │ │   RITE   │    │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘    │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Threshold System

```sigil
/// A threshold represents a significant boundary between states
pub struct Threshold<S> {
    /// Unique identifier
    pub id: ThresholdId,

    /// The state being left
    pub from: S,

    /// The state being entered
    pub to: S,

    /// Human-readable name for this threshold
    pub name: String,

    /// Description of what this crossing means
    pub significance: String,

    /// What must be released to cross (the obol)
    pub obol: Option<Obol>,

    /// Required rite for crossing (if any)
    pub required_rite: Option<RiteId>,

    /// Whether this threshold is reversible
    pub reversible: bool,
}

impl<S: Clone + PartialEq> Threshold<S> {
    /// Create a simple threshold
    pub fn new(from: S, to: S) -> Self {
        Self {
            id: ThresholdId::generate(),
            from,
            to,
            name: String::new(),
            significance: String::new(),
            obol: None,
            required_rite: None,
            reversible: true,
        }
    }

    /// Named threshold with significance
    pub fn named(from: S, to: S, name: &str, significance: &str) -> Self {
        Self {
            id: ThresholdId::generate(),
            from,
            to,
            name: name.to_string(),
            significance: significance.to_string(),
            obol: None,
            required_rite: None,
            reversible: true,
        }
    }

    /// Check if a state change crosses this threshold
    pub fn is_crossed_by(&self, from: &S, to: &S) -> bool {
        &self.from == from && &self.to == to
    }
}

/// The obol - what must be given up to cross
pub struct Obol {
    /// What is being released
    pub release: String,

    /// The nature of the cost
    pub cost_type: CostType,

    /// Whether this release can be undone
    pub reversible: bool,

    /// Evidence that the obol has been paid
    pub payment: Option<ObolPayment>,
}

pub enum CostType {
    /// Something is given up permanently
    Sacrifice,
    /// Something is transformed
    Transformation,
    /// Something is released but could return
    Release,
    /// Growth requires leaving comfort
    Growth,
    /// No cost - a gift
    Grace,
}
```

### 2. Crossing System

```sigil
/// A crossing in progress or completed
pub struct Crossing<S> {
    /// Unique identifier
    pub id: CrossingId,

    /// The threshold being crossed
    pub threshold: Threshold<S>,

    /// Entity making the crossing
    pub entity: EntityId,

    /// Current state of the crossing
    pub state: CrossingState<S>,

    /// Witnesses to this crossing
    pub witnesses: Vec<WitnessRecord>,

    /// The rite being performed (if any)
    pub rite: Option<RiteExecution>,

    /// When the crossing began
    pub started_at: Timestamp,

    /// When the crossing completed (if finished)
    pub completed_at: Option<Timestamp>,

    /// Outcome of the crossing
    pub outcome: Option<CrossingOutcome>,
}

pub enum CrossingState<S> {
    /// Preparing to cross - at the threshold
    Preparing {
        current: S,
        toward: S,
    },

    /// In the liminal space - between states
    Liminal {
        from: S,
        toward: S,
        phase: LiminalPhase,
    },

    /// Crossing completed successfully
    Arrived {
        from: S,
        now: S,
    },

    /// Crossing abandoned - returned to origin
    Returned {
        attempted: S,
        returned_to: S,
        reason: String,
    },
}

pub enum LiminalPhase {
    /// Releasing the old
    Releasing,
    /// In the void between
    Void,
    /// Receiving the new
    Receiving,
    /// Integrating the change
    Integrating,
}

pub enum CrossingOutcome {
    /// Successfully crossed
    Completed {
        transformation: Option<String>,
        duration: Duration,
    },
    /// Returned without crossing
    Returned {
        reason: String,
        at_phase: LiminalPhase,
    },
    /// Crossing was interrupted
    Interrupted {
        reason: String,
        at_phase: LiminalPhase,
        recoverable: bool,
    },
}
```

### 3. Rite System

```sigil
/// A rite is a transformative practice for crossing thresholds
pub struct Rite {
    /// Unique identifier
    pub id: RiteId,

    /// Name of this rite
    pub name: String,

    /// What type of threshold this rite is for
    pub threshold_type: ThresholdType,

    /// Phases of the rite
    pub phases: Vec<RitePhase>,

    /// Required witnesses
    pub required_witnesses: Vec<WitnessRequirement>,

    /// Minimum duration (rites should not be rushed)
    pub minimum_duration: Option<Duration>,
}

pub struct RitePhase {
    /// Phase name
    pub name: String,

    /// Phase type
    pub phase_type: PhaseType,

    /// Actions in this phase
    pub actions: Vec<RiteAction>,

    /// Invocations (words/statements)
    pub invocations: Vec<Invocation>,

    /// Can this phase be skipped?
    pub required: bool,
}

pub enum PhaseType {
    /// Preparing for the crossing
    Preparation,
    /// At the threshold itself
    Threshold,
    /// Releasing the old
    Release,
    /// In the liminal void
    Void,
    /// Receiving the new
    Reception,
    /// Integrating the transformation
    Integration,
    /// Closing the rite
    Closure,
}

pub struct Invocation {
    /// Who speaks
    pub speaker: Speaker,

    /// What is spoken
    pub words: String,

    /// The intent of the invocation
    pub intent: InvocationIntent,
}

pub enum Speaker {
    /// The one crossing
    Crosser,
    /// A witness
    Witness(WitnessId),
    /// All present
    All,
    /// The threshold itself (system)
    Threshold,
}

pub enum InvocationIntent {
    /// Acknowledging what was
    Acknowledgment,
    /// Releasing what must be left
    Release,
    /// Expressing grief for loss
    Mourning,
    /// Welcoming what comes
    Welcome,
    /// Accepting responsibility
    Charge,
    /// Blessing the crosser
    Blessing,
    /// Witnessing the crossing
    Testimony,
}

/// A rite in execution
pub struct RiteExecution {
    /// The rite being performed
    pub rite: Rite,

    /// Current phase
    pub current_phase: usize,

    /// Completed phases
    pub completed_phases: Vec<CompletedPhase>,

    /// When execution started
    pub started_at: Timestamp,

    /// Present witnesses
    pub present_witnesses: Vec<WitnessId>,
}
```

### 4. Witness System

```sigil
/// A witness observes crossings without acting
pub trait Witness {
    /// Called when a crossing begins
    fn observe_threshold(&mut self, crossing: &Crossing<impl State>) -> WitnessResponse;

    /// Called during liminal phases
    fn hold_space(&mut self, crossing: &Crossing<impl State>, phase: LiminalPhase);

    /// Called when crossing completes
    fn acknowledge_arrival(&mut self, crossing: &Crossing<impl State>) -> Testimony;

    /// Record the crossing in memory
    fn remember(&mut self, crossing: &Crossing<impl State>, testimony: &Testimony);
}

/// Record of a witness's participation
pub struct WitnessRecord {
    /// The witness
    pub witness_id: WitnessId,

    /// When they began witnessing
    pub joined_at: Timestamp,

    /// Phases they witnessed
    pub phases_witnessed: Vec<LiminalPhase>,

    /// Their testimony (if crossing completed)
    pub testimony: Option<Testimony>,
}

/// A witness's account of a crossing
pub struct Testimony {
    /// The witness
    pub witness: WitnessId,

    /// The crossing witnessed
    pub crossing: CrossingId,

    /// What the witness observed
    pub observation: String,

    /// The witness's acknowledgment
    pub acknowledgment: String,

    /// When the testimony was given
    pub given_at: Timestamp,

    /// Evidentiality of the testimony
    pub evidentiality: Evidentiality,
}

pub enum WitnessResponse {
    /// Ready to witness
    Present,
    /// Cannot witness (with reason)
    Unavailable(String),
    /// Witness has concerns about the crossing
    Concerned(String),
}
```

### 5. Lifecycle Rites

```sigil
/// Standard lifecycle rites for agents
pub mod lifecycle {
    /// Birth rite - when an agent comes into existence
    pub struct BirthRite {
        pub purpose: String,
        pub creator: EntityId,
        pub blessing: String,
        pub initial_capabilities: Vec<Capability>,
    }

    /// Growth rite - when an agent gains new capabilities
    pub struct GrowthRite {
        pub capability_gained: Capability,
        pub evidence_of_readiness: Vec<Evidence>,
        pub charge: String,
    }

    /// Trust escalation rite - when trust level increases
    pub struct TrustEscalationRite {
        pub from_level: TrustLevel,
        pub to_level: TrustLevel,
        pub demonstrated_trustworthiness: Vec<Evidence>,
        pub new_responsibilities: Vec<Responsibility>,
    }

    /// Mourning rite - when something is lost
    pub struct MourningRite {
        pub what_was: String,
        pub what_is_lost: String,
        pub what_remains: String,
        pub what_continues: String,
        pub grief_expression: GriefExpression,
    }

    /// Completion rite - when an agent fulfills its purpose
    pub struct CompletionRite {
        pub purpose_statement: String,
        pub fulfillment_evidence: Vec<Evidence>,
        pub legacy: Legacy,
        pub blessing: String,
        pub dissolution_method: DissolutionMethod,
    }

    /// Dissolution - how an agent ends
    pub enum DissolutionMethod {
        /// Graceful completion - purpose fulfilled
        Graceful {
            legacy_transferred_to: Option<EntityId>,
            memory_preserved_in: Vec<EngramId>,
        },
        /// Voluntary ending - agent chooses to end
        Voluntary {
            reason: String,
            final_statement: String,
        },
        /// Forced ending - external termination
        Forced {
            authority: EntityId,
            reason: String,
            contested: bool,
        },
    }
}
```

### 6. Mourning System

```sigil
/// Infrastructure for processing loss
pub struct MourningSpace {
    /// What is being mourned
    pub subject: MourningSubject,

    /// Who is mourning
    pub mourners: Vec<EntityId>,

    /// Phase of mourning
    pub phase: MourningPhase,

    /// Expressions of grief
    pub expressions: Vec<GriefExpression>,

    /// When mourning began
    pub began_at: Timestamp,

    /// When mourning concluded (if finished)
    pub concluded_at: Option<Timestamp>,
}

pub enum MourningSubject {
    /// An agent that completed or ended
    Agent {
        agent_id: AgentId,
        existed_from: Timestamp,
        existed_to: Timestamp,
        purpose: String,
    },
    /// A relationship that ended
    Relationship {
        entities: (EntityId, EntityId),
        relationship_type: String,
        duration: Duration,
    },
    /// A capability that was lost
    Capability {
        entity: EntityId,
        capability: Capability,
        reason_lost: String,
    },
    /// A trust relationship that was broken
    Trust {
        trustor: EntityId,
        trustee: EntityId,
        breach: String,
    },
}

pub enum MourningPhase {
    /// Acknowledging the loss
    Acknowledgment,
    /// Feeling the grief
    Grief,
    /// Remembering what was
    Remembrance,
    /// Releasing attachment
    Release,
    /// Finding meaning
    Meaning,
    /// Moving forward
    Integration,
}

pub struct GriefExpression {
    /// Who expressed
    pub mourner: EntityId,

    /// The expression
    pub expression: String,

    /// Type of expression
    pub expression_type: ExpressionType,

    /// When expressed
    pub timestamp: Timestamp,
}

pub enum ExpressionType {
    /// Statement of loss
    Statement,
    /// Memory shared
    Memory,
    /// Gratitude expressed
    Gratitude,
    /// Pain acknowledged
    Pain,
    /// Meaning found
    Meaning,
    /// Blessing offered
    Blessing,
}
```

## Integration Points

### With Engram (Memory)

```sigil
impl Crossing {
    /// Store crossing in episodic memory
    pub fn remember(&self, engram: &mut Engram) -> MemoryId {
        let memory = EpisodicMemory {
            event_type: EventType::Crossing,
            significance: Significance::High,
            emotional_valence: self.determine_valence(),
            content: self.to_memory_content(),
            timestamp: self.completed_at.unwrap_or(Timestamp::now()),
            witnesses: self.witnesses.iter().map(|w| w.witness_id).collect(),
        };
        engram.store_episodic(memory)
    }
}
```

### With Covenant (Relationships)

```sigil
impl Covenant {
    /// Watch for threshold crossings in trust
    pub fn watch_trust_thresholds(&mut self, hades: &Hades) {
        hades.register_threshold_watcher(
            ThresholdType::Trust,
            |crossing| {
                self.on_trust_crossing(crossing);
            }
        );
    }

    /// Handle trust threshold crossing
    fn on_trust_crossing(&mut self, crossing: &Crossing<TrustLevel>) {
        // Update covenant to reflect new trust level
        self.trust_history.push(TrustEvent::Crossing {
            from: crossing.threshold.from,
            to: crossing.threshold.to,
            at: crossing.completed_at.unwrap(),
            witnesses: crossing.witnesses.len(),
        });
    }
}
```

### With Daemon (Agent Lifecycle)

```sigil
impl Daemon {
    /// Birth with rite
    pub fn birth(config: DaemonConfig, rite: BirthRite) -> Result<Self> {
        let daemon = Self::new(config)?;

        // Perform birth rite
        Hades::perform_rite(rite, |phase| {
            match phase {
                PhaseType::Reception => {
                    daemon.receive_purpose(&rite.purpose);
                }
                PhaseType::Blessing => {
                    daemon.receive_blessing(&rite.blessing);
                }
                _ => {}
            }
        })?;

        Ok(daemon)
    }

    /// Complete with rite
    pub fn complete(self, rite: CompletionRite) -> Result<Legacy> {
        // Perform completion rite
        Hades::perform_rite(rite.clone(), |phase| {
            match phase {
                PhaseType::Release => {
                    self.release_resources();
                }
                PhaseType::Closure => {
                    self.final_state_snapshot();
                }
                _ => {}
            }
        })?;

        // Return legacy
        Ok(rite.legacy)
    }
}
```

### With Styx (Version Control)

```sigil
// In styx-agent
impl TrustManager {
    /// Escalate trust with proper rite
    pub fn escalate_trust(
        &mut self,
        agent: AgentId,
        to_level: TrustLevel,
        hades: &Hades,
    ) -> Result<()> {
        let from_level = self.current_trust(agent)?;

        let threshold = Threshold::named(
            from_level,
            to_level,
            &format!("Trust Escalation: {:?} → {:?}", from_level, to_level),
            "The agent has demonstrated increased trustworthiness",
        );

        let rite = TrustEscalationRite {
            from_level,
            to_level,
            demonstrated_trustworthiness: self.gather_evidence(agent)?,
            new_responsibilities: to_level.responsibilities(),
        };

        hades.cross_with_rite(agent, threshold, rite)?;

        Ok(())
    }
}
```

## Threshold Watchers

```sigil
/// Watch for threshold crossings across the system
pub struct ThresholdWatcher {
    /// Thresholds being watched
    thresholds: HashMap<ThresholdType, Vec<ThresholdSpec>>,

    /// Callbacks for crossings
    callbacks: HashMap<ThresholdType, Vec<Box<dyn Fn(&Crossing<Value>)>>>,
}

impl ThresholdWatcher {
    /// Register a threshold to watch
    pub fn watch<S: State>(
        &mut self,
        threshold_type: ThresholdType,
        threshold: Threshold<S>,
        callback: impl Fn(&Crossing<S>) + 'static,
    ) {
        self.thresholds
            .entry(threshold_type)
            .or_default()
            .push(threshold.into_spec());

        self.callbacks
            .entry(threshold_type)
            .or_default()
            .push(Box::new(move |c| callback(c.cast())));
    }

    /// Notify watchers of a state change
    pub fn notify_change<S: State>(
        &self,
        entity: EntityId,
        threshold_type: ThresholdType,
        from: S,
        to: S,
    ) {
        for spec in self.thresholds.get(&threshold_type).unwrap_or(&vec![]) {
            if spec.matches(&from, &to) {
                // Crossing detected - invoke callbacks
                let crossing = Crossing::new(entity, spec.to_threshold(), from, to);
                for callback in self.callbacks.get(&threshold_type).unwrap_or(&vec![]) {
                    callback(&crossing.into_value());
                }
            }
        }
    }
}
```

---

*The architecture of the underworld*
