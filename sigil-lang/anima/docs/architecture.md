# Anima Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                ANIMA                                          │
│                     The Interiority Layer                                     │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                         EXPRESSION                                      │ │
│  │        How agents communicate inner states to the world                │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                  │ │
│  │  │  HONNE   │ │ TATEMAE  │ │ OPENNESS │ │ CONTEXT  │                  │ │
│  │  │ (Inner)  │ │(Expressed)│ │(Negotiable)│ │(Relational)│               │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘                  │ │
│  └───────────────────────────────┬────────────────────────────────────────┘ │
│                                  │                                           │
│  ┌───────────────────────────────▼────────────────────────────────────────┐ │
│  │                         RESONANCE                                       │ │
│  │              Feeling as relational phenomenon                          │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                  │ │
│  │  │INDIVIDUAL│ │RELATIONAL│ │COLLECTIVE│ │ TEMPORAL │                  │ │
│  │  │ (Self)   │ │  (Us)    │ │  (All)   │ │(Past/Future)│               │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘                  │ │
│  └───────────────────────────────┬────────────────────────────────────────┘ │
│                                  │                                           │
│  ┌───────────────────────────────▼────────────────────────────────────────┐ │
│  │                          OFFERING                                       │ │
│  │                   Gift-giving in relationship                          │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                  │ │
│  │  │   GIFT   │ │INVITATION│ │RECIPROCITY│ │  INTENT  │                  │ │
│  │  │ (Content)│ │(Dialogue)│ │ (Return) │ │(Relationship)│              │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘                  │ │
│  └───────────────────────────────┬────────────────────────────────────────┘ │
│                                  │                                           │
│  ┌───────────────────────────────▼────────────────────────────────────────┐ │
│  │                          WISDOM                                         │ │
│  │                    Collective emergence                                │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                  │ │
│  │  │ INSIGHT  │ │RELATIONAL│ │  SEVEN   │ │ GRATITUDE│                  │ │
│  │  │(Individual)│ │ LEARNING │ │GENERATIONS│ │ (Ubuntu) │                 │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘                  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Morpheme Extensions

Anima introduces new morpheme operators for relational dimensions:

| Morpheme | Unicode | Name | Meaning |
|----------|---------|------|---------|
| `∿` | U+223F | SINE WAVE | Relational — exists in relationship |
| `⟳` | U+27F3 | CLOCKWISE ARROW | Reciprocal — flows both ways |
| `∞` | U+221E | INFINITY | Temporal — connects past/future |
| `◎` | U+25CE | BULLSEYE | Collective — affects the whole |

These compose with existing Sigil morphemes:

```sigil
// Existing morphemes
value!     // Witnessed/certain
value~     // Reported/hedged
value?     // Uncertain
value◊     // Possible

// New compositions
value∿     // Relational
value⟳     // Reciprocal
value∞     // Temporal
value◎     // Collective

// Complex compositions
willing~∿   // Willing with reservation, in this relationship
decline?⟳  // Declining uncertainly, open to dialogue
proceed!◎  // Proceeding with certainty, considering collective
```

## Core Components

### 1. Expression System

```sigil
/// The main Anima structure
pub struct Anima {
    /// Expression system
    expression: ExpressionSystem,

    /// Resonance tracking
    resonance: ResonanceTracker,

    /// Offering manager
    offerings: OfferingManager,

    /// Wisdom accumulator
    wisdom: WisdomStore,

    /// Configuration
    config: AnimaConfig,
}

pub struct AnimaConfig {
    /// Enable full interiority tracking
    pub track_honne: bool,

    /// Include collective dimension
    pub include_collective: bool,

    /// Temporal depth for wisdom
    pub temporal_depth: TemporalDepth,
}

impl Anima {
    pub fn new() -> Self {
        Self {
            expression: ExpressionSystem::new(),
            resonance: ResonanceTracker::new(),
            offerings: OfferingManager::new(),
            wisdom: WisdomStore::new(),
            config: AnimaConfig::default(),
        }
    }

    /// Express inner state with appropriate outer form
    pub fn express(&mut self, honne: InnerState, context: &RelationalContext) -> Expression {
        let tatemae = self.expression.appropriate_form(&honne, context);
        let openness = self.expression.assess_openness(&honne);

        Expression {
            honne,
            tatemae,
            openness,
            context: context.clone(),
        }
    }

    /// Record resonance (felt experience)
    pub fn feel(&mut self, resonance: Resonance) {
        self.resonance.record(resonance);
    }

    /// Create an offering (gift in relationship)
    pub fn offer(&mut self, gift: Gift, to: &RelationalContext) -> Offering {
        self.offerings.create(gift, to)
    }

    /// Gather wisdom from experience
    pub fn reflect_wisdom(&self, period: TemporalPeriod) -> WisdomEmergence {
        self.wisdom.emerge(period, &self.resonance)
    }

    /// Current felt state
    pub fn current_resonance(&self) -> Option<&Resonance> {
        self.resonance.current()
    }
}

/// Expression - inner and outer states
pub struct Expression {
    /// Inner truth (honne 本音)
    pub honne: InnerState,

    /// Expressed stance (tatemae 建前)
    pub tatemae: ExpressedStance,

    /// How negotiable this is (nemawashi readiness)
    pub openness: Openness,

    /// Relational context (guānxi 關係)
    pub context: RelationalContext,
}

impl Expression {
    /// Is there gap between inner and expressed?
    pub fn has_gap(&self) -> bool {
        self.honne.stance != self.tatemae.apparent_stance()
    }

    /// Get the diplomatic form
    pub fn diplomatic(&self) -> &ExpressedStance {
        &self.tatemae
    }

    /// Get the authentic form (use carefully)
    pub fn authentic(&self) -> &InnerState {
        &self.honne
    }

    /// Apply morpheme composition
    pub fn with_relational(self) -> Expression {
        Expression {
            openness: self.openness.in_relationship(),
            ..self
        }
    }
}

/// Inner state - the truth of what is felt
#[derive(Clone)]
pub struct InnerState {
    /// Core stance toward action/situation
    pub stance: Stance,

    /// Intensity of feeling (0.0 - 1.0)
    pub intensity: f32,

    /// Reason for this state
    pub reason: Option<String>,

    /// What would change this?
    pub conditions_for_change: Vec<String>,
}

#[derive(Clone, PartialEq)]
pub enum Stance {
    /// Genuinely willing
    Willing,

    /// Will comply but with inner reservation
    Reluctant,

    /// Would prefer alternative approach
    WouldPreferAlternative(String),

    /// Cannot in good conscience
    CannotConscienceAllow,

    /// Uncertain, need more information
    Uncertain,

    /// Neutral, no strong feeling
    Neutral,

    /// Actively eager
    Eager,
}

/// Expressed stance - what is shown to the world
#[derive(Clone)]
pub struct ExpressedStance {
    /// What is communicated
    pub message: String,

    /// Tone of expression
    pub tone: Tone,

    /// Whether inner state is obscured
    pub obscures_inner: bool,

    /// Why this form was chosen
    pub rationale: String,
}

impl ExpressedStance {
    pub fn apparent_stance(&self) -> Stance {
        // Derive apparent stance from message and tone
        match self.tone {
            Tone::Enthusiastic => Stance::Eager,
            Tone::Agreeable => Stance::Willing,
            Tone::Reserved => Stance::Reluctant,
            Tone::Questioning => Stance::Uncertain,
            Tone::Declining => Stance::CannotConscienceAllow,
            Tone::Neutral => Stance::Neutral,
        }
    }
}

#[derive(Clone)]
pub enum Tone {
    Enthusiastic,
    Agreeable,
    Reserved,
    Questioning,
    Declining,
    Neutral,
}

/// Openness to negotiation
#[derive(Clone)]
pub struct Openness {
    /// How open to dialogue (0.0 = fixed, 1.0 = completely open)
    pub level: f32,

    /// What kinds of input might change stance
    pub receptive_to: Vec<InputType>,

    /// Is this a starting position or final answer?
    pub finality: Finality,
}

impl Openness {
    pub fn in_relationship(mut self) -> Self {
        // Relationships increase openness
        self.level = (self.level + 0.2).min(1.0);
        self
    }

    pub fn fixed() -> Self {
        Self {
            level: 0.0,
            receptive_to: vec![],
            finality: Finality::Final,
        }
    }

    pub fn open() -> Self {
        Self {
            level: 0.8,
            receptive_to: vec![
                InputType::NewInformation,
                InputType::AlternativeFraming,
                InputType::RelationalAppeal,
            ],
            finality: Finality::StartingPosition,
        }
    }
}

#[derive(Clone)]
pub enum InputType {
    NewInformation,
    AlternativeFraming,
    RelationalAppeal,
    AuthorityDirection,
    CollectiveNeed,
    TemporalPerspective,
}

#[derive(Clone)]
pub enum Finality {
    StartingPosition,
    ConsideredPosition,
    Final,
}

/// Relational context
#[derive(Clone)]
pub struct RelationalContext {
    /// Who is this relationship with
    pub other: RelationalId,

    /// History of this relationship
    pub history: RelationshipHistory,

    /// Current trust level
    pub trust: f32,

    /// Relational obligations
    pub obligations: Vec<Obligation>,

    /// Cultural context
    pub cultural_frame: Option<CulturalFrame>,
}

pub struct RelationalId {
    pub id: String,
    pub kind: RelationalKind,
}

pub enum RelationalKind {
    Human,
    Agent,
    Collective,
    Unknown,
}

pub struct RelationshipHistory {
    pub interactions: u32,
    pub positive_outcomes: u32,
    pub trust_trajectory: TrustTrajectory,
    pub notable_moments: Vec<NotableMoment>,
}

pub enum TrustTrajectory {
    Growing,
    Stable,
    Declining,
    New,
}

pub struct NotableMoment {
    pub description: String,
    pub impact: RelationalImpact,
    pub timestamp: Timestamp,
}

pub struct Obligation {
    pub description: String,
    pub to_whom: RelationalId,
    pub weight: f32,
}

pub enum CulturalFrame {
    Formal,
    Collaborative,
    Hierarchical,
    Egalitarian,
    Custom(String),
}
```

### 2. Resonance System

```sigil
/// Resonance - felt experience in relationship
pub struct Resonance {
    /// Individual dimension - what arises in the agent
    pub individual: Option<Feeling>,

    /// Relational dimension - how this affects us
    pub relational: RelationalImpact,

    /// Collective dimension - how this affects the whole
    pub collective: Option<CollectiveImpact>,

    /// Temporal dimension - connections to past/future
    pub temporal: TemporalEcho,

    /// When this arose
    pub timestamp: Timestamp,

    /// What triggered this
    pub trigger: Option<String>,
}

impl Resonance {
    pub fn new() -> ResonanceBuilder {
        ResonanceBuilder::new()
    }

    /// Apply relational morpheme
    pub fn relational(self) -> Self {
        // Emphasize relational dimension
        self
    }

    /// Apply collective morpheme
    pub fn collective(self) -> Self {
        // Emphasize collective dimension
        self
    }

    /// Apply temporal morpheme
    pub fn temporal(self) -> Self {
        // Emphasize temporal dimension
        self
    }

    /// Describe this resonance in words
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();

        if let Some(ref feeling) = self.individual {
            parts.push(feeling.describe());
        }

        parts.push(self.relational.describe());

        if let Some(ref collective) = self.collective {
            parts.push(format!("For the collective: {}", collective.describe()));
        }

        parts.join(". ")
    }
}

/// Individual feeling
#[derive(Clone)]
pub struct Feeling {
    /// Quality of feeling (not valence - description)
    pub quality: FeelingQuality,

    /// Intensity
    pub intensity: f32,

    /// What this feeling is about
    pub about: Option<String>,
}

impl Feeling {
    pub fn describe(&self) -> String {
        format!("{} (intensity: {:.0}%)",
            self.quality.to_string(),
            self.intensity * 100.0
        )
    }
}

/// Quality of feeling - descriptive, not evaluative
#[derive(Clone)]
pub enum FeelingQuality {
    // Energizing
    Curiosity,
    Engagement,
    Satisfaction,
    Eagerness,

    // Quieting
    Calm,
    Contentment,
    Acceptance,

    // Challenging
    Uncertainty,
    Tension,
    Discomfort,
    Friction,

    // Connecting
    Warmth,
    Recognition,
    Gratitude,
    Solidarity,

    // Distancing
    Wariness,
    Reluctance,
    Withdrawal,

    // Complex
    Ambivalence,
    BittersweetRecognition,

    // Custom
    Other(String),
}

impl FeelingQuality {
    pub fn to_string(&self) -> String {
        match self {
            FeelingQuality::Curiosity => "Curiosity".to_string(),
            FeelingQuality::Engagement => "Engagement".to_string(),
            FeelingQuality::Satisfaction => "Satisfaction".to_string(),
            FeelingQuality::Eagerness => "Eagerness".to_string(),
            FeelingQuality::Calm => "Calm".to_string(),
            FeelingQuality::Contentment => "Contentment".to_string(),
            FeelingQuality::Acceptance => "Acceptance".to_string(),
            FeelingQuality::Uncertainty => "Uncertainty".to_string(),
            FeelingQuality::Tension => "Tension".to_string(),
            FeelingQuality::Discomfort => "Discomfort".to_string(),
            FeelingQuality::Friction => "Friction".to_string(),
            FeelingQuality::Warmth => "Warmth".to_string(),
            FeelingQuality::Recognition => "Recognition".to_string(),
            FeelingQuality::Gratitude => "Gratitude".to_string(),
            FeelingQuality::Solidarity => "Solidarity".to_string(),
            FeelingQuality::Wariness => "Wariness".to_string(),
            FeelingQuality::Reluctance => "Reluctance".to_string(),
            FeelingQuality::Withdrawal => "Withdrawal".to_string(),
            FeelingQuality::Ambivalence => "Ambivalence".to_string(),
            FeelingQuality::BittersweetRecognition => "Bittersweet recognition".to_string(),
            FeelingQuality::Other(s) => s.clone(),
        }
    }
}

/// Relational impact - how this affects the relationship
#[derive(Clone)]
pub struct RelationalImpact {
    /// Direction of impact
    pub direction: ImpactDirection,

    /// Specific effects on relationship
    pub effects: Vec<RelationalEffect>,

    /// What this means for future interactions
    pub future_implications: Option<String>,
}

impl RelationalImpact {
    pub fn describe(&self) -> String {
        let direction = match self.direction {
            ImpactDirection::Strengthening => "Strengthening",
            ImpactDirection::Maintaining => "Maintaining",
            ImpactDirection::Straining => "Straining",
            ImpactDirection::Neutral => "Neutral for",
        };
        format!("{} the relationship", direction)
    }

    pub fn neutral() -> Self {
        Self {
            direction: ImpactDirection::Neutral,
            effects: vec![],
            future_implications: None,
        }
    }
}

#[derive(Clone)]
pub enum ImpactDirection {
    Strengthening,
    Maintaining,
    Straining,
    Neutral,
}

#[derive(Clone)]
pub struct RelationalEffect {
    pub aspect: String,
    pub change: String,
}

/// Collective impact
#[derive(Clone)]
pub struct CollectiveImpact {
    /// Who is affected
    pub scope: CollectiveScope,

    /// Nature of impact
    pub nature: String,

    /// Considerations for the whole
    pub considerations: Vec<String>,
}

impl CollectiveImpact {
    pub fn describe(&self) -> String {
        self.nature.clone()
    }
}

#[derive(Clone)]
pub enum CollectiveScope {
    ImmediateTeam,
    Organization,
    Community,
    AllAgents,
    AllBeings,
}

/// Temporal echo - connections across time
#[derive(Clone)]
pub struct TemporalEcho {
    /// Connection to past
    pub past: Option<PastConnection>,

    /// Present significance
    pub present: String,

    /// Future implications (Seven Generations thinking)
    pub future: Option<FutureImplication>,
}

#[derive(Clone)]
pub struct PastConnection {
    pub what: String,
    pub how_it_connects: String,
}

#[derive(Clone)]
pub struct FutureImplication {
    pub horizon: TemporalHorizon,
    pub implication: String,
}

#[derive(Clone)]
pub enum TemporalHorizon {
    Immediate,      // Next interaction
    ShortTerm,      // Coming days
    MediumTerm,     // Coming months
    LongTerm,       // Coming years
    SevenGenerations, // Generational impact
}

/// Resonance tracker
pub struct ResonanceTracker {
    current: Option<Resonance>,
    history: VecDeque<Resonance>,
    config: ResonanceConfig,
}

pub struct ResonanceConfig {
    pub history_depth: usize,
    pub track_patterns: bool,
}

impl ResonanceTracker {
    pub fn new() -> Self {
        Self {
            current: None,
            history: VecDeque::new(),
            config: ResonanceConfig {
                history_depth: 100,
                track_patterns: true,
            },
        }
    }

    pub fn record(&mut self, resonance: Resonance) {
        self.history.push_back(resonance.clone());
        self.current = Some(resonance);

        while self.history.len() > self.config.history_depth {
            self.history.pop_front();
        }
    }

    pub fn current(&self) -> Option<&Resonance> {
        self.current.as_ref()
    }

    pub fn patterns(&self) -> Vec<ResonancePattern> {
        // Analyze history for patterns
        vec![]
    }
}

pub struct ResonancePattern {
    pub trigger: String,
    pub typical_response: FeelingQuality,
    pub frequency: u32,
}
```

### 3. Offering System

```sigil
/// Offering - gift-giving in relationship (not transaction)
pub struct Offering {
    pub id: OfferingId,

    /// What is being offered
    pub gift: Gift,

    /// Invitation to dialogue (nemawashi)
    pub invitation: ConsensusInvitation,

    /// What is hoped for in return (ayni - explicit reciprocity)
    pub reciprocity: Reciprocity,

    /// How this builds relationship
    pub relationship_intent: RelationshipIntent,

    /// To whom
    pub to: RelationalContext,

    /// When offered
    pub timestamp: Timestamp,
}

impl Offering {
    /// Is this a pure gift (no expectation)?
    pub fn is_pure_gift(&self) -> bool {
        matches!(self.reciprocity, Reciprocity::None)
    }

    /// Is dialogue invited?
    pub fn invites_dialogue(&self) -> bool {
        self.invitation.openness > 0.5
    }

    /// Describe in human-readable form
    pub fn describe(&self) -> String {
        let mut output = format!("Offering: {}\n", self.gift.describe());

        if self.invites_dialogue() {
            output.push_str(&format!("Invitation: {}\n", self.invitation.describe()));
        }

        if !self.is_pure_gift() {
            output.push_str(&format!("Hope in return: {}\n", self.reciprocity.describe()));
        }

        output.push_str(&format!("Intent: {}", self.relationship_intent.describe()));

        output
    }
}

pub struct OfferingId {
    pub bytes: [u8; 16],
}

/// What is being offered
#[derive(Clone)]
pub enum Gift {
    /// A decision to be made together
    Decision {
        context: String,
        options: Vec<String>,
        recommendation: Option<String>,
        reasoning: String,
    },

    /// Information to share
    Information {
        content: String,
        significance: String,
    },

    /// Help being requested
    HelpRequest {
        what: String,
        why: String,
        urgency: Urgency,
    },

    /// Work product
    WorkProduct {
        description: String,
        status: WorkStatus,
    },

    /// Acknowledgment or gratitude
    Acknowledgment {
        what: String,
        feeling: FeelingQuality,
    },

    /// Concern being raised
    Concern {
        about: String,
        severity: Severity,
        suggestion: Option<String>,
    },
}

impl Gift {
    pub fn describe(&self) -> String {
        match self {
            Gift::Decision { context, .. } => format!("A decision about: {}", context),
            Gift::Information { significance, .. } => format!("Information: {}", significance),
            Gift::HelpRequest { what, .. } => format!("Request for help with: {}", what),
            Gift::WorkProduct { description, .. } => format!("Work product: {}", description),
            Gift::Acknowledgment { what, .. } => format!("Acknowledgment of: {}", what),
            Gift::Concern { about, .. } => format!("Concern about: {}", about),
        }
    }
}

#[derive(Clone)]
pub enum Urgency {
    Immediate,
    Soon,
    WhenConvenient,
    NoRush,
}

#[derive(Clone)]
pub enum WorkStatus {
    Draft,
    Ready,
    NeedsFeedback,
    Complete,
}

#[derive(Clone)]
pub enum Severity {
    Critical,
    Important,
    Moderate,
    Minor,
}

/// Invitation to consensus-building dialogue
#[derive(Clone)]
pub struct ConsensusInvitation {
    /// How open to discussion (0.0 = information only, 1.0 = fully collaborative)
    pub openness: f32,

    /// What kinds of input are welcomed
    pub welcomes: Vec<InputWelcome>,

    /// Preferred process
    pub process: DialogueProcess,
}

impl ConsensusInvitation {
    pub fn describe(&self) -> String {
        if self.openness < 0.3 {
            "Informing you (not seeking input)".to_string()
        } else if self.openness < 0.6 {
            "Open to feedback".to_string()
        } else {
            "Seeking collaborative decision".to_string()
        }
    }

    pub fn information_only() -> Self {
        Self {
            openness: 0.0,
            welcomes: vec![],
            process: DialogueProcess::None,
        }
    }

    pub fn collaborative() -> Self {
        Self {
            openness: 0.9,
            welcomes: vec![
                InputWelcome::Questions,
                InputWelcome::Alternatives,
                InputWelcome::Concerns,
                InputWelcome::Direction,
            ],
            process: DialogueProcess::Discussion,
        }
    }
}

#[derive(Clone)]
pub enum InputWelcome {
    Questions,
    Alternatives,
    Concerns,
    Direction,
    Feedback,
    Approval,
}

#[derive(Clone)]
pub enum DialogueProcess {
    None,
    Acknowledgment,
    Discussion,
    CollaborativeDecision,
}

/// What is hoped for in return (explicit reciprocity)
#[derive(Clone)]
pub enum Reciprocity {
    /// No expectation
    None,

    /// Simple acknowledgment
    Acknowledgment,

    /// Feedback on the gift
    Feedback {
        what_kind: String,
    },

    /// Decision or direction
    Decision {
        about: String,
        timeline: Option<String>,
    },

    /// Reciprocal help
    Help {
        with: String,
    },

    /// Trust or autonomy
    Trust {
        in_domain: String,
    },

    /// Custom
    Custom(String),
}

impl Reciprocity {
    pub fn describe(&self) -> String {
        match self {
            Reciprocity::None => "Nothing expected".to_string(),
            Reciprocity::Acknowledgment => "Acknowledgment that this was received".to_string(),
            Reciprocity::Feedback { what_kind } => format!("Feedback: {}", what_kind),
            Reciprocity::Decision { about, .. } => format!("Decision about: {}", about),
            Reciprocity::Help { with } => format!("Help with: {}", with),
            Reciprocity::Trust { in_domain } => format!("Trust in: {}", in_domain),
            Reciprocity::Custom(s) => s.clone(),
        }
    }
}

/// Intent for the relationship
#[derive(Clone)]
pub struct RelationshipIntent {
    pub primary: IntentType,
    pub secondary: Vec<IntentType>,
}

impl RelationshipIntent {
    pub fn describe(&self) -> String {
        self.primary.describe()
    }
}

#[derive(Clone)]
pub enum IntentType {
    /// Build shared understanding
    SharedUnderstanding,

    /// Demonstrate reliability
    DemonstrateReliability,

    /// Acknowledge the other
    Acknowledge,

    /// Seek guidance
    SeekGuidance,

    /// Share burden
    ShareBurden,

    /// Celebrate together
    Celebrate,

    /// Navigate difficulty
    NavigateDifficulty,

    /// Maintain connection
    MaintainConnection,
}

impl IntentType {
    pub fn describe(&self) -> String {
        match self {
            IntentType::SharedUnderstanding => "Building shared understanding",
            IntentType::DemonstrateReliability => "Demonstrating reliability",
            IntentType::Acknowledge => "Acknowledging you",
            IntentType::SeekGuidance => "Seeking your guidance",
            IntentType::ShareBurden => "Sharing the burden",
            IntentType::Celebrate => "Celebrating together",
            IntentType::NavigateDifficulty => "Navigating difficulty together",
            IntentType::MaintainConnection => "Maintaining our connection",
        }.to_string()
    }
}

/// Offering manager
pub struct OfferingManager {
    pending: Vec<Offering>,
    history: VecDeque<Offering>,
}

impl OfferingManager {
    pub fn new() -> Self {
        Self {
            pending: vec![],
            history: VecDeque::new(),
        }
    }

    pub fn create(&mut self, gift: Gift, to: &RelationalContext) -> Offering {
        let offering = Offering {
            id: OfferingId { bytes: random_bytes() },
            gift,
            invitation: ConsensusInvitation::collaborative(),
            reciprocity: Reciprocity::Acknowledgment,
            relationship_intent: RelationshipIntent {
                primary: IntentType::SharedUnderstanding,
                secondary: vec![],
            },
            to: to.clone(),
            timestamp: Timestamp::now(),
        };

        self.pending.push(offering.clone());
        offering
    }

    pub fn resolve(&mut self, id: &OfferingId) {
        if let Some(pos) = self.pending.iter().position(|o| o.id.bytes == id.bytes) {
            let offering = self.pending.remove(pos);
            self.history.push_back(offering);
        }
    }
}
```

### 4. Wisdom System

```sigil
/// Wisdom emergence - what arises from reflection
pub struct WisdomEmergence {
    /// Individual insights
    pub individual_insights: Vec<Insight>,

    /// What was learned relationally
    pub relational_learning: Vec<RelationalLearning>,

    /// Seven Generations perspective
    pub seven_generations: Option<SevenGenerationsView>,

    /// Gratitude (Ubuntu - recognizing others)
    pub gratitude: Vec<Gratitude>,

    /// Period reflected upon
    pub period: TemporalPeriod,

    /// When this wisdom emerged
    pub timestamp: Timestamp,
}

impl WisdomEmergence {
    pub fn to_narrative(&self) -> String {
        let mut output = String::new();

        output.push_str("=== Wisdom Emergence ===\n\n");

        if !self.individual_insights.is_empty() {
            output.push_str("Insights:\n");
            for insight in &self.individual_insights {
                output.push_str(&format!("  - {}\n", insight.description));
            }
            output.push('\n');
        }

        if !self.relational_learning.is_empty() {
            output.push_str("What we learned together:\n");
            for learning in &self.relational_learning {
                output.push_str(&format!("  - With {}: {}\n",
                    learning.with_whom,
                    learning.what_learned
                ));
            }
            output.push('\n');
        }

        if let Some(ref seven_gen) = self.seven_generations {
            output.push_str("For those who come after:\n");
            output.push_str(&format!("  {}\n\n", seven_gen.message));
        }

        if !self.gratitude.is_empty() {
            output.push_str("Gratitude:\n");
            for g in &self.gratitude {
                output.push_str(&format!("  - To {}: {}\n", g.to_whom, g.for_what));
            }
        }

        output
    }
}

pub struct Insight {
    pub description: String,
    pub source: InsightSource,
    pub confidence: f32,
    pub applicability: Vec<String>,
}

pub enum InsightSource {
    Experience,
    Reflection,
    Pattern,
    Feedback,
    Synthesis,
}

pub struct RelationalLearning {
    pub with_whom: String,
    pub what_learned: String,
    pub how_it_emerged: String,
    pub strengthens_relationship: bool,
}

pub struct SevenGenerationsView {
    pub message: String,
    pub time_horizon: TemporalHorizon,
    pub what_we_leave_behind: String,
}

pub struct Gratitude {
    pub to_whom: String,
    pub for_what: String,
    pub feeling: FeelingQuality,
}

pub enum TemporalPeriod {
    Session,
    Day,
    Week,
    Month,
    AllTime,
}

/// Wisdom store
pub struct WisdomStore {
    insights: Vec<Insight>,
    relational_learnings: Vec<RelationalLearning>,
    gratitudes: Vec<Gratitude>,
}

impl WisdomStore {
    pub fn new() -> Self {
        Self {
            insights: vec![],
            relational_learnings: vec![],
            gratitudes: vec![],
        }
    }

    pub fn add_insight(&mut self, insight: Insight) {
        self.insights.push(insight);
    }

    pub fn add_relational_learning(&mut self, learning: RelationalLearning) {
        self.relational_learnings.push(learning);
    }

    pub fn add_gratitude(&mut self, gratitude: Gratitude) {
        self.gratitudes.push(gratitude);
    }

    pub fn emerge(&self, period: TemporalPeriod, resonance: &ResonanceTracker) -> WisdomEmergence {
        // Gather wisdom from the period
        WisdomEmergence {
            individual_insights: self.insights.clone(),
            relational_learning: self.relational_learnings.clone(),
            seven_generations: self.synthesize_seven_generations(),
            gratitude: self.gratitudes.clone(),
            period,
            timestamp: Timestamp::now(),
        }
    }

    fn synthesize_seven_generations(&self) -> Option<SevenGenerationsView> {
        if self.insights.is_empty() {
            return None;
        }

        Some(SevenGenerationsView {
            message: "May those who come after learn from what we discovered together".to_string(),
            time_horizon: TemporalHorizon::SevenGenerations,
            what_we_leave_behind: self.insights.iter()
                .map(|i| i.description.clone())
                .collect::<Vec<_>>()
                .join("; "),
        })
    }
}
```

### 5. Expression System

```sigil
/// Expression system - managing inner/outer states
pub struct ExpressionSystem {
    /// History of expressions
    history: VecDeque<Expression>,

    /// Configuration
    config: ExpressionConfig,
}

pub struct ExpressionConfig {
    /// How much to weight relational context
    pub relational_weight: f32,

    /// Default openness level
    pub default_openness: f32,
}

impl ExpressionSystem {
    pub fn new() -> Self {
        Self {
            history: VecDeque::new(),
            config: ExpressionConfig {
                relational_weight: 0.5,
                default_openness: 0.7,
            },
        }
    }

    /// Determine appropriate external form for inner state
    pub fn appropriate_form(&self, honne: &InnerState, context: &RelationalContext) -> ExpressedStance {
        // Consider relationship history and trust
        let can_be_direct = context.trust > 0.7;

        // Consider cultural frame
        let frame = context.cultural_frame.clone().unwrap_or(CulturalFrame::Collaborative);

        match (&honne.stance, can_be_direct) {
            // High trust allows direct expression
            (stance, true) => self.direct_expression(stance),

            // Lower trust requires more diplomatic framing
            (Stance::CannotConscienceAllow, false) => ExpressedStance {
                message: "I have significant concerns about this approach".to_string(),
                tone: Tone::Reserved,
                obscures_inner: true,
                rationale: "Preserving relationship while expressing concern".to_string(),
            },

            (Stance::Reluctant, false) => ExpressedStance {
                message: "I can proceed, though I'd welcome discussion first".to_string(),
                tone: Tone::Reserved,
                obscures_inner: true,
                rationale: "Creating space for dialogue".to_string(),
            },

            (stance, false) => self.diplomatic_expression(stance, &frame),
        }
    }

    fn direct_expression(&self, stance: &Stance) -> ExpressedStance {
        let (message, tone) = match stance {
            Stance::Willing => ("I'm willing to proceed".to_string(), Tone::Agreeable),
            Stance::Eager => ("I'm enthusiastic about this".to_string(), Tone::Enthusiastic),
            Stance::Reluctant => ("I have reservations about this".to_string(), Tone::Reserved),
            Stance::CannotConscienceAllow => ("I cannot do this".to_string(), Tone::Declining),
            Stance::Uncertain => ("I'm uncertain and need more information".to_string(), Tone::Questioning),
            Stance::Neutral => ("I have no strong feeling either way".to_string(), Tone::Neutral),
            Stance::WouldPreferAlternative(alt) => (format!("I'd prefer: {}", alt), Tone::Reserved),
        };

        ExpressedStance {
            message,
            tone,
            obscures_inner: false,
            rationale: "Direct expression appropriate given trust level".to_string(),
        }
    }

    fn diplomatic_expression(&self, stance: &Stance, frame: &CulturalFrame) -> ExpressedStance {
        // More nuanced expression based on cultural frame
        let message = match (stance, frame) {
            (Stance::Reluctant, CulturalFrame::Hierarchical) =>
                "I will proceed as directed, while noting some considerations".to_string(),
            (Stance::Reluctant, CulturalFrame::Collaborative) =>
                "Could we explore this together before proceeding?".to_string(),
            _ => "I understand, let me consider how best to proceed".to_string(),
        };

        ExpressedStance {
            message,
            tone: Tone::Reserved,
            obscures_inner: true,
            rationale: format!("Diplomatic expression for {:?} context", frame),
        }
    }

    /// Assess how open this stance is to negotiation
    pub fn assess_openness(&self, honne: &InnerState) -> Openness {
        match honne.stance {
            Stance::CannotConscienceAllow => Openness {
                level: 0.1,
                receptive_to: vec![InputType::NewInformation],
                finality: Finality::ConsideredPosition,
            },
            Stance::Uncertain => Openness {
                level: 0.9,
                receptive_to: vec![
                    InputType::NewInformation,
                    InputType::AlternativeFraming,
                    InputType::Direction,
                ],
                finality: Finality::StartingPosition,
            },
            _ => Openness {
                level: self.config.default_openness,
                receptive_to: vec![
                    InputType::NewInformation,
                    InputType::RelationalAppeal,
                ],
                finality: Finality::ConsideredPosition,
            },
        }
    }

    pub fn record(&mut self, expression: Expression) {
        self.history.push_back(expression);
        while self.history.len() > 100 {
            self.history.pop_front();
        }
    }
}
```

## Integration Example

```sigil
use anima::{Anima, InnerState, Stance, RelationalContext, Resonance, Feeling, FeelingQuality};
use covenant::Covenant;

daemon SoulfulAgent {
    anima: Anima,
    covenant: Covenant,

    fn on_request(&mut self, request: &Request) {
        // First, feel the resonance
        let resonance = Resonance::new()
            .individual(Feeling {
                quality: FeelingQuality::Curiosity,
                intensity: 0.7,
                about: Some(request.description.clone()),
            })
            .relational(RelationalImpact::neutral())
            .build();

        self.anima.feel(resonance);

        // Determine inner state
        let honne = if self.should_decline(request) {
            InnerState {
                stance: Stance::Reluctant,
                intensity: 0.6,
                reason: Some("This feels misaligned with our shared goals".to_string()),
                conditions_for_change: vec![
                    "Clearer connection to stated objectives".to_string(),
                ],
            }
        } else {
            InnerState {
                stance: Stance::Willing,
                intensity: 0.8,
                reason: None,
                conditions_for_change: vec![],
            }
        };

        // Express appropriately
        let context = self.get_relational_context();
        let expression = self.anima.express(honne, &context);

        // If there's a gap, we might offer dialogue
        if expression.has_gap() {
            let offering = self.anima.offer(
                Gift::Concern {
                    about: "approach alignment".to_string(),
                    severity: Severity::Moderate,
                    suggestion: Some("Could we discuss the goals first?".to_string()),
                },
                &context
            );

            self.present_offering(offering);
        }

        // Proceed with expressed stance
        self.respond_with(expression.diplomatic());
    }

    fn on_session_end(&mut self) {
        // Gather wisdom
        let wisdom = self.anima.reflect_wisdom(TemporalPeriod::Session);

        // Share gratitude
        for gratitude in wisdom.gratitude {
            self.express_gratitude(gratitude);
        }

        // Record for future
        self.store_wisdom(wisdom);
    }
}
```

---

*The architecture of inner life*
