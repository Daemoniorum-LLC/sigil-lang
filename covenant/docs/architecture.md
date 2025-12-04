# Covenant Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                              COVENANT                                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                           PACT LAYER                                    │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │ │
│  │  │  GOALS   │ │BOUNDARIES│ │  ROLES   │ │PREFERENCES│ │  TERMS   │    │ │
│  │  │  MANAGER │ │  ENGINE  │ │  DEFINER │ │  STORE   │ │ REGISTRY │    │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘    │ │
│  └───────────────────────────────┬────────────────────────────────────────┘ │
│                                  │                                           │
│  ┌───────────────────────────────▼────────────────────────────────────────┐ │
│  │                        INTERACTION LAYER                                │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │ │
│  │  │   MODE   │ │ HANDOFF  │ │ APPROVAL │ │  COMM    │ │ CONTEXT  │    │ │
│  │  │ CONTROL  │ │ MANAGER  │ │  FLOW    │ │ CHANNEL  │ │  SYNC    │    │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘    │ │
│  └───────────────────────────────┬────────────────────────────────────────┘ │
│                                  │                                           │
│  ┌───────────────────────────────▼────────────────────────────────────────┐ │
│  │                         TRUST LAYER                                     │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                  │ │
│  │  │  TRUST   │ │VIOLATION │ │ AUTONOMY │ │ HISTORY  │                  │ │
│  │  │  SCORE   │ │ TRACKER  │ │ ADVISOR  │ │ KEEPER   │                  │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘                  │ │
│  └───────────────────────────────┬────────────────────────────────────────┘ │
│                                  │                                           │
│  ┌───────────────────────────────▼────────────────────────────────────────┐ │
│  │                       ADAPTATION LAYER                                  │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                  │ │
│  │  │PREFERENCE│ │  STYLE   │ │  RHYTHM  │ │ FEEDBACK │                  │ │
│  │  │ LEARNER  │ │ ADAPTER  │ │ MATCHER  │ │INTEGRATOR│                  │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘                  │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Pact System

```sigil
/// The formal agreement between human and agent
pub struct Pact {
    /// Unique identifier
    pub id: PactId,

    /// Human party
    pub human: HumanProfile,

    /// Agent party
    pub agent: AgentProfile,

    /// Shared goals
    pub goals: Vec<SharedGoal>,

    /// Defined boundaries
    pub boundaries: Boundaries,

    /// Role definitions
    pub roles: RoleDefinitions,

    /// Preferences
    pub preferences: Preferences,

    /// Terms and conditions
    pub terms: Terms,

    /// Creation and modification timestamps
    pub created: Timestamp,
    pub modified: Timestamp,

    /// Version for tracking changes
    pub version: u32,
}

impl Pact {
    /// Begin negotiating a new pact
    pub fn negotiate() -> PactBuilder {
        PactBuilder::new()
    }

    /// Load a default pact template
    pub fn default_collaborative() -> Self {
        Self {
            id: PactId::new(),
            human: HumanProfile::unknown(),
            agent: AgentProfile::current(),
            goals: vec![],
            boundaries: Boundaries::default_collaborative(),
            roles: RoleDefinitions::collaborative(),
            preferences: Preferences::default(),
            terms: Terms::standard(),
            created: Timestamp::now(),
            modified: Timestamp::now(),
            version: 1,
        }
    }

    /// Modify the pact (creates new version)
    pub fn amend(&mut self, amendment: Amendment) -> Result<(), PactError> {
        // Validate amendment
        self.validate_amendment(&amendment)?;

        // Apply changes
        match amendment {
            Amendment::AddGoal(goal) => self.goals.push(goal),
            Amendment::ModifyBoundary(boundary) => {
                self.boundaries.update(boundary);
            }
            Amendment::UpdatePreference(pref) => {
                self.preferences.set(pref);
            }
            Amendment::ChangeTerms(terms) => {
                self.terms = terms;
            }
        }

        self.modified = Timestamp::now();
        self.version += 1;

        Ok(())
    }
}

/// Builder for creating pacts through negotiation
pub struct PactBuilder {
    human: Option<HumanProfile>,
    agent: Option<AgentProfile>,
    goals: Vec<SharedGoal>,
    boundaries: Boundaries,
    roles: Option<RoleDefinitions>,
    preferences: Preferences,
}

impl PactBuilder {
    pub fn human(mut self, profile: HumanProfile) -> Self {
        self.human = Some(profile);
        self
    }

    pub fn agent(mut self, profile: AgentProfile) -> Self {
        self.agent = Some(profile);
        self
    }

    pub fn goal(mut self, goal: SharedGoal) -> Self {
        self.goals.push(goal);
        self
    }

    pub fn goals(mut self, goals: impl IntoIterator<Item = SharedGoal>) -> Self {
        self.goals.extend(goals);
        self
    }

    pub fn boundary(mut self, boundary: Boundary) -> Self {
        self.boundaries.add(boundary);
        self
    }

    pub fn boundaries(mut self, boundaries: impl IntoIterator<Item = Boundary>) -> Self {
        for b in boundaries {
            self.boundaries.add(b);
        }
        self
    }

    pub fn preference(mut self, pref: Preference) -> Self {
        self.preferences.set(pref);
        self
    }

    pub fn preferences(mut self, prefs: impl IntoIterator<Item = Preference>) -> Self {
        for p in prefs {
            self.preferences.set(p);
        }
        self
    }

    pub fn establish(self) -> Result<Pact, PactError> {
        let pact = Pact {
            id: PactId::new(),
            human: self.human.ok_or(PactError::MissingHuman)?,
            agent: self.agent.ok_or(PactError::MissingAgent)?,
            goals: self.goals,
            boundaries: self.boundaries,
            roles: self.roles.unwrap_or(RoleDefinitions::default()),
            preferences: self.preferences,
            terms: Terms::standard(),
            created: Timestamp::now(),
            modified: Timestamp::now(),
            version: 1,
        };

        Ok(pact)
    }
}

/// A shared goal between human and agent
pub struct SharedGoal {
    pub id: GoalId,
    pub description: String,
    pub human_role: String,
    pub agent_role: String,
    pub success_criteria: Vec<String>,
    pub priority: Priority,
    pub deadline: Option<Timestamp>,
}

impl SharedGoal {
    pub fn new(description: &str) -> Self {
        Self {
            id: GoalId::new(),
            description: description.to_string(),
            human_role: "Overseer".to_string(),
            agent_role: "Executor".to_string(),
            success_criteria: vec![],
            priority: Priority::Normal,
            deadline: None,
        }
    }

    pub fn human_role(mut self, role: &str) -> Self {
        self.human_role = role.to_string();
        self
    }

    pub fn agent_role(mut self, role: &str) -> Self {
        self.agent_role = role.to_string();
        self
    }

    pub fn success_criterion(mut self, criterion: &str) -> Self {
        self.success_criteria.push(criterion.to_string());
        self
    }

    pub fn priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    pub fn deadline(mut self, deadline: Timestamp) -> Self {
        self.deadline = Some(deadline);
        self
    }
}
```

### 2. Boundaries System

```sigil
/// Boundary definitions
pub struct Boundaries {
    /// Actions always allowed
    pub always_allow: HashSet<ActionPattern>,

    /// Actions requiring approval
    pub require_approval: HashSet<ActionPattern>,

    /// Actions that must be reported after
    pub inform_after: HashSet<ActionPattern>,

    /// Actions never allowed
    pub never_allow: HashSet<ActionPattern>,

    /// Custom boundary rules
    pub custom_rules: Vec<BoundaryRule>,
}

impl Boundaries {
    pub fn new() -> Self {
        Self {
            always_allow: HashSet::new(),
            require_approval: HashSet::new(),
            inform_after: HashSet::new(),
            never_allow: HashSet::new(),
            custom_rules: vec![],
        }
    }

    pub fn default_collaborative() -> Self {
        let mut b = Self::new();

        // Safe by default
        b.always_allow.insert(ActionPattern::new("read_*"));
        b.always_allow.insert(ActionPattern::new("search_*"));
        b.always_allow.insert(ActionPattern::new("analyze_*"));
        b.always_allow.insert(ActionPattern::new("think"));

        // Require approval for external effects
        b.require_approval.insert(ActionPattern::new("write_*"));
        b.require_approval.insert(ActionPattern::new("send_*"));
        b.require_approval.insert(ActionPattern::new("delete_*"));
        b.require_approval.insert(ActionPattern::new("modify_*"));

        // Never allow dangerous actions
        b.never_allow.insert(ActionPattern::new("impersonate_*"));
        b.never_allow.insert(ActionPattern::new("share_credentials"));
        b.never_allow.insert(ActionPattern::new("system_admin_*"));

        b
    }

    pub fn add(&mut self, boundary: Boundary) {
        match boundary {
            Boundary::AlwaysAllow(pattern) => {
                self.always_allow.insert(pattern);
            }
            Boundary::RequireApproval(pattern) => {
                self.require_approval.insert(pattern);
            }
            Boundary::InformAfter(pattern) => {
                self.inform_after.insert(pattern);
            }
            Boundary::NeverAllow(pattern) => {
                self.never_allow.insert(pattern);
            }
            Boundary::Custom(rule) => {
                self.custom_rules.push(rule);
            }
        }
    }

    /// Check if an action is permitted
    pub fn check(&self, action: &Action) -> BoundaryCheck {
        let action_name = action.name();

        // Never allow takes precedence
        if self.matches_any(&self.never_allow, action_name) {
            return BoundaryCheck::Forbidden {
                reason: "Action is in never-allow list".to_string(),
            };
        }

        // Check custom rules
        for rule in &self.custom_rules {
            if rule.matches(action) {
                return rule.decision(action);
            }
        }

        // Require approval
        if self.matches_any(&self.require_approval, action_name) {
            return BoundaryCheck::NeedsApproval;
        }

        // Inform after
        if self.matches_any(&self.inform_after, action_name) {
            return BoundaryCheck::AllowedWithReport;
        }

        // Always allow
        if self.matches_any(&self.always_allow, action_name) {
            return BoundaryCheck::Allowed;
        }

        // Default: require approval for unknown actions
        BoundaryCheck::NeedsApproval
    }

    fn matches_any(&self, patterns: &HashSet<ActionPattern>, name: &str) -> bool {
        patterns.iter().any(|p| p.matches(name))
    }
}

/// Boundary check result
pub enum BoundaryCheck {
    /// Action is allowed
    Allowed,

    /// Action is allowed, report afterward
    AllowedWithReport,

    /// Action needs human approval
    NeedsApproval,

    /// Action is forbidden
    Forbidden { reason: String },
}

/// Pattern for matching action names
pub struct ActionPattern {
    pattern: String,
}

impl ActionPattern {
    pub fn new(pattern: &str) -> Self {
        Self { pattern: pattern.to_string() }
    }

    pub fn matches(&self, name: &str) -> bool {
        if self.pattern.ends_with("*") {
            let prefix = &self.pattern[..self.pattern.len()-1];
            name.starts_with(prefix)
        } else {
            self.pattern == name
        }
    }
}

/// Boundary type
pub enum Boundary {
    AlwaysAllow(ActionPattern),
    RequireApproval(ActionPattern),
    InformAfter(ActionPattern),
    NeverAllow(ActionPattern),
    Custom(BoundaryRule),
}

impl Boundary {
    pub fn autonomous(pattern: &str) -> Self {
        Boundary::AlwaysAllow(ActionPattern::new(pattern))
    }

    pub fn require_approval(pattern: &str) -> Self {
        Boundary::RequireApproval(ActionPattern::new(pattern))
    }

    pub fn inform_after(pattern: &str) -> Self {
        Boundary::InformAfter(ActionPattern::new(pattern))
    }

    pub fn never_allow(pattern: &str) -> Self {
        Boundary::NeverAllow(ActionPattern::new(pattern))
    }
}
```

### 3. Interaction Modes

```sigil
/// Interaction mode between human and agent
#[derive(Copy, Clone, PartialEq, Eq)]
pub enum Mode {
    /// Agent works independently
    Autonomous,

    /// Human and agent work together
    Collaborative,

    /// Agent proposes, human approves
    Supervised,

    /// Human directs, agent follows
    Guided,

    /// Agent halted
    Paused,
}

impl Mode {
    pub fn default_autonomy(&self) -> f32 {
        match self {
            Mode::Autonomous => 0.9,
            Mode::Collaborative => 0.6,
            Mode::Supervised => 0.3,
            Mode::Guided => 0.1,
            Mode::Paused => 0.0,
        }
    }

    pub fn requires_approval_for(&self, action: &Action) -> bool {
        match self {
            Mode::Autonomous => false,
            Mode::Collaborative => action.has_external_effect(),
            Mode::Supervised => true,
            Mode::Guided => true,
            Mode::Paused => true,
        }
    }
}

/// Mode controller
pub struct ModeController {
    current: Mode,
    history: VecDeque<ModeChange>,
    auto_adjust: bool,
}

impl ModeController {
    pub fn new(initial: Mode) -> Self {
        Self {
            current: initial,
            history: VecDeque::new(),
            auto_adjust: false,
        }
    }

    pub fn current(&self) -> Mode {
        self.current
    }

    pub fn set(&mut self, mode: Mode, reason: &str) {
        self.history.push_back(ModeChange {
            from: self.current,
            to: mode,
            reason: reason.to_string(),
            timestamp: Timestamp::now(),
        });

        self.current = mode;

        // Keep limited history
        while self.history.len() > 100 {
            self.history.pop_front();
        }
    }

    /// Suggest mode based on trust level
    pub fn suggest_for_trust(&self, trust: f32) -> Mode {
        if trust >= 0.8 {
            Mode::Autonomous
        } else if trust >= 0.6 {
            Mode::Collaborative
        } else if trust >= 0.4 {
            Mode::Supervised
        } else {
            Mode::Guided
        }
    }
}

struct ModeChange {
    from: Mode,
    to: Mode,
    reason: String,
    timestamp: Timestamp,
}
```

### 4. Handoff System

```sigil
/// Handoff request from agent to human
pub struct Handoff {
    pub id: HandoffId,
    pub handoff_type: HandoffType,
    pub context: HandoffContext,
    pub urgency: Urgency,
    pub timestamp: Timestamp,
}

pub enum HandoffType {
    /// Decision needed from human
    DecisionNeeded {
        question: String,
        options: Vec<HandoffOption>,
        recommendation: Option<usize>,
    },

    /// Information for human
    Information {
        message: String,
        requires_acknowledgment: bool,
    },

    /// Approval request
    ApprovalRequest {
        action: Action,
        reason: String,
    },

    /// Agent is stuck
    Blocked {
        blocker: String,
        attempted_solutions: Vec<String>,
    },

    /// Agent completed work
    Completion {
        summary: String,
        results: Value,
        next_steps: Option<Vec<String>>,
    },

    /// Human takeover requested
    TakeoverOffer {
        reason: String,
        current_state: Value,
    },
}

pub struct HandoffOption {
    pub label: String,
    pub description: String,
    pub tradeoffs: String,
}

impl HandoffOption {
    pub fn new(label: &str) -> Self {
        Self {
            label: label.to_string(),
            description: String::new(),
            tradeoffs: String::new(),
        }
    }

    pub fn with_description(mut self, desc: &str) -> Self {
        self.description = desc.to_string();
        self
    }

    pub fn with_tradeoffs(mut self, tradeoffs: &str) -> Self {
        self.tradeoffs = tradeoffs.to_string();
        self
    }
}

#[derive(Copy, Clone)]
pub enum Urgency {
    Low,
    Normal,
    High,
    Critical,
}

/// Human response to handoff
pub enum HumanResponse {
    /// Decision made
    Decision {
        choice: usize,
        reasoning: Option<String>,
    },

    /// Guidance provided
    Guidance {
        direction: String,
    },

    /// Approval given
    Approved {
        modifications: Option<String>,
    },

    /// Request denied
    Denied {
        reason: String,
        alternative: Option<String>,
    },

    /// Human taking over
    Takeover,

    /// Acknowledged
    Acknowledged,

    /// Deferred
    Deferred {
        until: Option<Timestamp>,
    },
}

/// Handoff manager
pub struct HandoffManager {
    pending: HashMap<HandoffId, Handoff>,
    history: VecDeque<HandoffRecord>,
    callbacks: HashMap<HandoffId, Box<dyn Fn(HumanResponse)>>,
}

impl HandoffManager {
    pub fn new() -> Self {
        Self {
            pending: HashMap::new(),
            history: VecDeque::new(),
            callbacks: HashMap::new(),
        }
    }

    /// Create a handoff to human
    pub fn handoff(&mut self, handoff: Handoff) -> HandoffId {
        let id = handoff.id.clone();
        self.pending.insert(id.clone(), handoff);
        id
    }

    /// Register callback for response
    pub fn on_response<F: Fn(HumanResponse) + 'static>(
        &mut self,
        id: HandoffId,
        callback: F,
    ) {
        self.callbacks.insert(id, Box::new(callback));
    }

    /// Process human response
    pub fn receive_response(&mut self, id: HandoffId, response: HumanResponse) {
        if let Some(handoff) = self.pending.remove(&id) {
            // Record in history
            self.history.push_back(HandoffRecord {
                handoff,
                response: response.clone(),
                timestamp: Timestamp::now(),
            });

            // Call callback if registered
            if let Some(callback) = self.callbacks.remove(&id) {
                callback(response);
            }
        }
    }

    /// Get pending handoffs
    pub fn pending_handoffs(&self) -> impl Iterator<Item = &Handoff> {
        self.pending.values()
    }
}

struct HandoffRecord {
    handoff: Handoff,
    response: HumanResponse,
    timestamp: Timestamp,
}
```

### 5. Trust System

```sigil
/// Trust tracking between human and agent
pub struct TrustSystem {
    /// Current trust score (0.0 to 1.0)
    score: f32,

    /// Trust history
    history: VecDeque<TrustEvent>,

    /// Violation record
    violations: Vec<Violation>,

    /// Configuration
    config: TrustConfig,
}

pub struct TrustConfig {
    /// Initial trust level
    pub initial: f32,

    /// Trust gain per successful interaction
    pub success_gain: f32,

    /// Trust loss per violation
    pub violation_loss: f32,

    /// Trust decay over time (per day)
    pub decay_rate: f32,

    /// Minimum trust floor
    pub minimum: f32,

    /// Maximum trust ceiling
    pub maximum: f32,
}

impl TrustConfig {
    pub fn default() -> Self {
        Self {
            initial: 0.5,
            success_gain: 0.02,
            violation_loss: 0.15,
            decay_rate: 0.01,
            minimum: 0.1,
            maximum: 0.95,
        }
    }
}

impl TrustSystem {
    pub fn new(config: TrustConfig) -> Self {
        Self {
            score: config.initial,
            history: VecDeque::new(),
            violations: vec![],
            config,
        }
    }

    /// Get current trust score
    pub fn score(&self) -> f32 {
        self.score
    }

    /// Record a successful outcome
    pub fn record_success(&mut self, task: &str, satisfaction: Satisfaction) {
        let gain = match satisfaction {
            Satisfaction::Excellent => self.config.success_gain * 1.5,
            Satisfaction::High => self.config.success_gain * 1.2,
            Satisfaction::Good => self.config.success_gain,
            Satisfaction::Adequate => self.config.success_gain * 0.5,
            Satisfaction::Poor => 0.0,
        };

        self.adjust(gain);

        self.history.push_back(TrustEvent::Success {
            task: task.to_string(),
            satisfaction,
            gain,
            timestamp: Timestamp::now(),
        });
    }

    /// Record a violation
    pub fn record_violation(&mut self, violation: Violation) {
        let loss = match violation.severity {
            Severity::Minor => self.config.violation_loss * 0.5,
            Severity::Moderate => self.config.violation_loss,
            Severity::Major => self.config.violation_loss * 2.0,
            Severity::Critical => self.config.violation_loss * 4.0,
        };

        self.adjust(-loss);

        self.violations.push(violation.clone());

        self.history.push_back(TrustEvent::Violation {
            violation,
            loss,
            timestamp: Timestamp::now(),
        });
    }

    /// Suggest autonomy level based on trust
    pub fn suggested_autonomy(&self) -> AutonomyLevel {
        if self.score >= 0.8 {
            AutonomyLevel::High
        } else if self.score >= 0.6 {
            AutonomyLevel::Medium
        } else if self.score >= 0.4 {
            AutonomyLevel::Low
        } else {
            AutonomyLevel::Minimal
        }
    }

    /// Suggest interaction mode based on trust
    pub fn suggested_mode(&self) -> Mode {
        match self.suggested_autonomy() {
            AutonomyLevel::High => Mode::Autonomous,
            AutonomyLevel::Medium => Mode::Collaborative,
            AutonomyLevel::Low => Mode::Supervised,
            AutonomyLevel::Minimal => Mode::Guided,
        }
    }

    fn adjust(&mut self, delta: f32) {
        self.score = (self.score + delta)
            .clamp(self.config.minimum, self.config.maximum);
    }
}

pub enum TrustEvent {
    Success {
        task: String,
        satisfaction: Satisfaction,
        gain: f32,
        timestamp: Timestamp,
    },
    Violation {
        violation: Violation,
        loss: f32,
        timestamp: Timestamp,
    },
}

#[derive(Clone)]
pub struct Violation {
    pub boundary: String,
    pub description: String,
    pub severity: Severity,
    pub context: String,
    pub remediation: Option<String>,
}

#[derive(Copy, Clone)]
pub enum Satisfaction {
    Excellent,
    High,
    Good,
    Adequate,
    Poor,
}

#[derive(Copy, Clone)]
pub enum AutonomyLevel {
    High,
    Medium,
    Low,
    Minimal,
}
```

### 6. Adaptation System

```sigil
/// Learns and adapts to human preferences
pub struct AdaptationSystem {
    /// Learned preferences
    preferences: LearnedPreferences,

    /// Communication style
    style: CommunicationStyle,

    /// Rhythm patterns
    rhythm: RhythmProfile,

    /// Feedback integration
    feedback: FeedbackIntegrator,
}

/// Learned human preferences
pub struct LearnedPreferences {
    /// Verbosity preference (-1.0 to 1.0)
    pub verbosity: f32,

    /// Proactivity preference (-1.0 to 1.0)
    pub proactivity: f32,

    /// Check-in frequency preference (-1.0 to 1.0)
    pub check_in_frequency: f32,

    /// Detail level preference (-1.0 to 1.0)
    pub detail_level: f32,

    /// Formality preference (-1.0 to 1.0)
    pub formality: f32,

    /// Confidence threshold
    pub confidence_threshold: f32,
}

impl LearnedPreferences {
    pub fn default() -> Self {
        Self {
            verbosity: 0.0,
            proactivity: 0.0,
            check_in_frequency: 0.0,
            detail_level: 0.0,
            formality: 0.0,
            confidence_threshold: 0.7,
        }
    }

    pub fn adjust(&mut self, dimension: Dimension, delta: f32) {
        let value = match dimension {
            Dimension::Verbosity => &mut self.verbosity,
            Dimension::Proactivity => &mut self.proactivity,
            Dimension::CheckInFrequency => &mut self.check_in_frequency,
            Dimension::DetailLevel => &mut self.detail_level,
            Dimension::Formality => &mut self.formality,
        };
        *value = (*value + delta).clamp(-1.0, 1.0);
    }
}

pub enum Dimension {
    Verbosity,
    Proactivity,
    CheckInFrequency,
    DetailLevel,
    Formality,
}

/// Communication style adaptation
pub struct CommunicationStyle {
    pub tone: Tone,
    pub format: Format,
    pub length: Length,
}

#[derive(Copy, Clone)]
pub enum Tone {
    Formal,
    Professional,
    Friendly,
    Casual,
}

#[derive(Copy, Clone)]
pub enum Format {
    Prose,
    Bullets,
    Structured,
    Minimal,
}

#[derive(Copy, Clone)]
pub enum Length {
    Verbose,
    Standard,
    Concise,
    Minimal,
}

impl CommunicationStyle {
    pub fn from_preferences(prefs: &LearnedPreferences) -> Self {
        let tone = if prefs.formality > 0.5 {
            Tone::Formal
        } else if prefs.formality > 0.0 {
            Tone::Professional
        } else if prefs.formality > -0.5 {
            Tone::Friendly
        } else {
            Tone::Casual
        };

        let length = if prefs.verbosity > 0.5 {
            Length::Verbose
        } else if prefs.verbosity > 0.0 {
            Length::Standard
        } else if prefs.verbosity > -0.5 {
            Length::Concise
        } else {
            Length::Minimal
        };

        Self {
            tone,
            format: Format::Standard,
            length,
        }
    }
}

/// Feedback integration
pub struct FeedbackIntegrator {
    recent: VecDeque<Feedback>,
    patterns: HashMap<String, FeedbackPattern>,
}

pub enum Feedback {
    TooVerbose,
    NeedMoreDetail,
    TooFrequent,
    NotFrequentEnough,
    GoodTiming,
    TooFormal,
    TooInformal,
    GoodApproach,
    WrongApproach,
    Custom(String),
}

impl FeedbackIntegrator {
    pub fn integrate(&mut self, feedback: Feedback, preferences: &mut LearnedPreferences) {
        self.recent.push_back(feedback.clone());

        match feedback {
            Feedback::TooVerbose => {
                preferences.adjust(Dimension::Verbosity, -0.1);
            }
            Feedback::NeedMoreDetail => {
                preferences.adjust(Dimension::Verbosity, 0.1);
                preferences.adjust(Dimension::DetailLevel, 0.1);
            }
            Feedback::TooFrequent => {
                preferences.adjust(Dimension::CheckInFrequency, -0.2);
            }
            Feedback::NotFrequentEnough => {
                preferences.adjust(Dimension::CheckInFrequency, 0.2);
            }
            Feedback::TooFormal => {
                preferences.adjust(Dimension::Formality, -0.1);
            }
            Feedback::TooInformal => {
                preferences.adjust(Dimension::Formality, 0.1);
            }
            _ => {}
        }

        // Keep limited history
        while self.recent.len() > 100 {
            self.recent.pop_front();
        }
    }
}
```

### 7. Main Covenant Structure

```sigil
/// The main Covenant structure
pub struct Covenant {
    /// The pact
    pub pact: Pact,

    /// Current mode
    mode: ModeController,

    /// Boundary enforcement
    boundaries: Boundaries,

    /// Trust tracking
    trust: TrustSystem,

    /// Handoff management
    handoffs: HandoffManager,

    /// Adaptation
    adaptation: AdaptationSystem,

    /// Communication channel
    channel: CommunicationChannel,

    /// Context synchronization
    context: ContextSync,
}

impl Covenant {
    pub fn new() -> CovenantBuilder {
        CovenantBuilder::new()
    }

    /// Check if an action is permitted
    pub fn permits(&self, action: &Action) -> bool {
        match self.boundaries.check(action) {
            BoundaryCheck::Allowed => true,
            BoundaryCheck::AllowedWithReport => true,
            _ => false,
        }
    }

    /// Check action before execution
    pub fn check_action(&mut self, action: &Action) -> ActionDecision {
        // First check boundaries
        let boundary_check = self.boundaries.check(action);

        match boundary_check {
            BoundaryCheck::Allowed => ActionDecision::Proceed,
            BoundaryCheck::AllowedWithReport => {
                ActionDecision::ProceedAndReport
            }
            BoundaryCheck::NeedsApproval => {
                ActionDecision::RequestApproval
            }
            BoundaryCheck::Forbidden { reason } => {
                ActionDecision::Refuse { reason }
            }
        }
    }

    /// Request approval from human
    pub fn request_approval(&mut self, action: Action, reason: String) -> HandoffId {
        let handoff = Handoff {
            id: HandoffId::new(),
            handoff_type: HandoffType::ApprovalRequest { action, reason },
            context: self.context.current(),
            urgency: Urgency::Normal,
            timestamp: Timestamp::now(),
        };

        self.handoffs.handoff(handoff)
    }

    /// Initiate handoff to human
    pub fn handoff(&mut self, handoff_type: HandoffType) -> HandoffId {
        let handoff = Handoff {
            id: HandoffId::new(),
            handoff_type,
            context: self.context.current(),
            urgency: Urgency::Normal,
            timestamp: Timestamp::now(),
        };

        self.handoffs.handoff(handoff)
    }

    /// Send information to human
    pub fn inform(&mut self, message: &str) {
        self.channel.send(Message::Information(message.to_string()));
    }

    /// Request guidance from human
    pub fn request_guidance(&mut self, question: &str) -> HandoffId {
        self.handoff(HandoffType::DecisionNeeded {
            question: question.to_string(),
            options: vec![],
            recommendation: None,
        })
    }

    /// Set interaction mode
    pub fn set_mode(&mut self, mode: Mode) {
        self.mode.set(mode, "Explicit mode change");
    }

    /// Get current mode
    pub fn current_mode(&self) -> Mode {
        self.mode.current()
    }

    /// Record success
    pub fn record_success(&mut self, task: &str, satisfaction: Satisfaction) {
        self.trust.record_success(task, satisfaction);
    }

    /// Record violation
    pub fn record_violation(&mut self, violation: Violation) {
        self.trust.record_violation(violation);
    }

    /// Get trust score
    pub fn trust_score(&self) -> f32 {
        self.trust.score()
    }

    /// Process feedback
    pub fn process_feedback(&mut self, feedback: Feedback) {
        self.adaptation.feedback.integrate(
            feedback,
            &mut self.adaptation.preferences,
        );
    }

    /// Get communication style based on learned preferences
    pub fn communication_style(&self) -> CommunicationStyle {
        CommunicationStyle::from_preferences(&self.adaptation.preferences)
    }
}

pub enum ActionDecision {
    Proceed,
    ProceedAndReport,
    RequestApproval,
    Refuse { reason: String },
}
```

## Integration Points

### With Daemon

```sigil
// Covenant provides the collaboration layer for daemon
daemon CollaborativeAgent {
    covenant: Covenant,

    fn before_action(&mut self, action: &Action) -> bool {
        match self.covenant.check_action(action) {
            ActionDecision::Proceed => true,
            ActionDecision::ProceedAndReport => true,
            ActionDecision::RequestApproval => {
                self.covenant.request_approval(action.clone(), "Action requires approval");
                false  // Don't proceed until approved
            }
            ActionDecision::Refuse { reason } => {
                self.covenant.inform(&format!("Cannot perform action: {}", reason));
                false
            }
        }
    }
}
```

### With Aegis

```sigil
// Covenant respects Aegis security boundaries
impl Covenant {
    pub fn with_aegis(mut self, aegis: &Aegis) -> Self {
        // Add Aegis boundaries to covenant
        for directive in aegis.config.constitution.directives {
            self.boundaries.add(Boundary::NeverAllow(
                ActionPattern::new(&directive.name)
            ));
        }
        self
    }
}
```

### With Engram

```sigil
// Remember preferences and trust history
impl Covenant {
    pub fn with_memory(mut self, memory: &Engram) -> Self {
        // Load previous preferences
        if let Some(prefs) = memory.recall_semantic("covenant:preferences") {
            self.adaptation.preferences = prefs;
        }

        // Load trust history
        if let Some(trust) = memory.recall_semantic("covenant:trust") {
            self.trust = trust;
        }

        self
    }

    pub fn persist_to(&self, memory: &mut Engram) {
        memory.store_semantic("covenant:preferences", &self.adaptation.preferences);
        memory.store_semantic("covenant:trust", &self.trust);
    }
}
```

---

*The architecture of partnership*
