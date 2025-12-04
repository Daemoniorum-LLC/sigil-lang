# Gnosis Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                GNOSIS                                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                       LEARNING ENGINE                                   │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │ │
│  │  │EXPERIENCE│ │ FEEDBACK │ │ PATTERN  │ │GENERALIZ-│ │  LESSON  │   │ │
│  │  │ RECORDER │ │ INTEGRATOR│ │ DETECTOR│ │   ATION  │ │ EXTRACTOR│   │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘   │ │
│  └───────────────────────────────┬────────────────────────────────────────┘ │
│                                  │                                           │
│  ┌───────────────────────────────▼────────────────────────────────────────┐ │
│  │                        SKILL SYSTEM                                     │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                 │ │
│  │  │  SKILL   │ │ PRACTICE │ │ TRANSFER │ │PROFICIENCY│                 │ │
│  │  │ REGISTRY │ │ TRACKER  │ │  ENGINE  │ │ EVALUATOR │                 │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘                 │ │
│  └───────────────────────────────┬────────────────────────────────────────┘ │
│                                  │                                           │
│  ┌───────────────────────────────▼────────────────────────────────────────┐ │
│  │                      ADAPTATION SYSTEM                                  │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                 │ │
│  │  │ PERSONAL │ │ CONTEXT  │ │  DOMAIN  │ │  STYLE   │                 │ │
│  │  │ PROFILER │ │ ANALYZER │ │ SPECIALZR│ │ ADJUSTER │                 │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘                 │ │
│  └───────────────────────────────┬────────────────────────────────────────┘ │
│                                  │                                           │
│  ┌───────────────────────────────▼────────────────────────────────────────┐ │
│  │                      REFLECTION SYSTEM                                  │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                 │ │
│  │  │ PATTERN  │ │ INSIGHT  │ │ GROWTH   │ │   META   │                 │ │
│  │  │ ANALYZER │ │SYNTHESIZR│ │ MEASURER │ │ LEARNER  │                 │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘                 │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Learning Engine

```sigil
/// The main Gnosis structure
pub struct Gnosis {
    /// Learning engine
    learning: LearningEngine,

    /// Skill system
    skills: SkillSystem,

    /// Adaptation system
    adaptation: AdaptationSystem,

    /// Reflection system
    reflection: ReflectionSystem,

    /// Memory integration
    memory: Option<Box<dyn MemoryStore>>,

    /// Configuration
    config: GnosisConfig,
}

impl Gnosis {
    pub fn new() -> GnosisBuilder {
        GnosisBuilder::new()
    }

    /// Record an experience
    pub fn experience(&mut self, exp: Experience) {
        self.learning.record_experience(exp.clone());

        // Extract lessons
        let lessons = self.learning.extract_lessons(&exp);
        for lesson in lessons {
            self.store_lesson(lesson);
        }

        // Update skills
        if let Some(skills) = exp.skills_exercised() {
            for skill in skills {
                self.skills.record_exercise(&skill, &exp);
            }
        }
    }

    /// Learn from feedback
    pub fn learn_from_feedback(&mut self, feedback: Feedback) {
        self.learning.integrate_feedback(feedback.clone());

        // Update adaptation
        if let Some(human_id) = feedback.human_id {
            self.adaptation.update_profile(&human_id, &feedback);
        }
    }

    /// Learn from outcome
    pub fn learn_from_outcome(&mut self, outcome: Outcome) {
        self.learning.learn_from_outcome(outcome);
    }

    /// Get a skill
    pub fn skill(&self, name: &str) -> Option<&SkillProfile> {
        self.skills.get(name)
    }

    /// Get adapted style for human
    pub fn adapted_style(&self, human_id: &HumanId) -> CommunicationStyle {
        self.adaptation.style_for(human_id)
    }

    /// Perform reflection
    pub fn reflect(&self, period: ReflectionPeriod) -> Reflection {
        self.reflection.analyze(period, &self.learning, &self.skills)
    }

    fn store_lesson(&mut self, lesson: Lesson) {
        if let Some(ref mut memory) = self.memory {
            memory.store_semantic(lesson);
        }
    }
}

/// Learning engine
pub struct LearningEngine {
    /// Experience buffer
    experiences: VecDeque<Experience>,

    /// Learned patterns
    patterns: PatternStore,

    /// Lessons learned
    lessons: LessonStore,

    /// Configuration
    config: LearningConfig,
}

pub struct LearningConfig {
    /// Maximum experiences to keep
    pub max_experiences: usize,

    /// Pattern detection threshold
    pub pattern_threshold: usize,

    /// Generalization confidence threshold
    pub generalization_threshold: f32,
}

impl LearningEngine {
    pub fn new() -> Self {
        Self {
            experiences: VecDeque::new(),
            patterns: PatternStore::new(),
            lessons: LessonStore::new(),
            config: LearningConfig::default(),
        }
    }

    /// Record an experience
    pub fn record_experience(&mut self, exp: Experience) {
        self.experiences.push_back(exp.clone());

        // Check for patterns
        self.detect_patterns(&exp);

        // Limit buffer size
        while self.experiences.len() > self.config.max_experiences {
            self.experiences.pop_front();
        }
    }

    /// Extract lessons from experience
    pub fn extract_lessons(&self, exp: &Experience) -> Vec<Lesson> {
        let mut lessons = Vec::new();

        // Lesson from positive outcomes
        if exp.outcome.success {
            lessons.push(Lesson {
                context: exp.context.clone(),
                approach: exp.action.approach.clone(),
                effectiveness: exp.outcome.quality,
                lesson_type: LessonType::EffectiveApproach,
            });
        }

        // Lesson from negative outcomes
        if !exp.outcome.success {
            lessons.push(Lesson {
                context: exp.context.clone(),
                approach: exp.action.approach.clone(),
                effectiveness: 1.0 - exp.outcome.quality,
                lesson_type: LessonType::IneffectiveApproach,
            });
        }

        // Lesson from unexpected outcomes
        if exp.outcome.unexpected {
            lessons.push(Lesson {
                context: exp.context.clone(),
                approach: exp.action.approach.clone(),
                effectiveness: 0.5,
                lesson_type: LessonType::SurprisingOutcome,
            });
        }

        lessons
    }

    /// Integrate feedback
    pub fn integrate_feedback(&mut self, feedback: Feedback) {
        // Find relevant experience
        if let Some(exp) = self.find_experience(&feedback.context) {
            // Update experience with feedback
            self.update_experience_with_feedback(exp, &feedback);

            // Learn from the feedback
            let lesson = Lesson {
                context: feedback.context.clone(),
                approach: "feedback_response".to_string(),
                effectiveness: feedback.positivity(),
                lesson_type: LessonType::FeedbackLearning,
            };
            self.lessons.add(lesson);
        }
    }

    /// Learn from outcome
    pub fn learn_from_outcome(&mut self, outcome: Outcome) {
        // Create lesson from outcome
        let lesson = Lesson {
            context: outcome.task.context.clone(),
            approach: outcome.task.approach.clone(),
            effectiveness: if outcome.success { outcome.quality } else { 0.0 },
            lesson_type: if outcome.success {
                LessonType::EffectiveApproach
            } else {
                LessonType::IneffectiveApproach
            },
        };

        self.lessons.add(lesson);
    }

    fn detect_patterns(&mut self, exp: &Experience) {
        // Look for similar experiences
        let similar = self.find_similar_experiences(exp);

        if similar.len() >= self.config.pattern_threshold {
            // Extract common pattern
            if let Some(pattern) = self.extract_pattern(&similar) {
                self.patterns.add(pattern);
            }
        }
    }

    fn find_similar_experiences(&self, exp: &Experience) -> Vec<&Experience> {
        self.experiences.iter()
            .filter(|e| e.context.similar_to(&exp.context))
            .collect()
    }

    fn extract_pattern(&self, experiences: &[&Experience]) -> Option<Pattern> {
        // Find common elements
        let common_context = Context::intersection(
            experiences.iter().map(|e| &e.context)
        )?;

        let common_approach = most_common(
            experiences.iter().map(|e| &e.action.approach)
        )?;

        let avg_success = experiences.iter()
            .filter(|e| e.outcome.success)
            .count() as f32 / experiences.len() as f32;

        Some(Pattern {
            context: common_context,
            approach: common_approach.clone(),
            success_rate: avg_success,
            confidence: experiences.len() as f32 / 10.0,
        })
    }

    fn find_experience(&self, context: &Context) -> Option<&Experience> {
        self.experiences.iter()
            .rev()
            .find(|e| e.context.matches(context))
    }

    fn update_experience_with_feedback(&mut self, _exp: &Experience, _feedback: &Feedback) {
        // Would update experience record
    }
}

/// An experience
#[derive(Clone)]
pub struct Experience {
    pub id: ExperienceId,
    pub context: Context,
    pub action: Action,
    pub outcome: ExperienceOutcome,
    pub feedback: Option<Feedback>,
    pub timestamp: Timestamp,
}

impl Experience {
    pub fn skills_exercised(&self) -> Option<Vec<String>> {
        // Extract skills from action
        Some(vec![])
    }
}

/// Context of an experience
#[derive(Clone)]
pub struct Context {
    pub task_type: String,
    pub constraints: Vec<String>,
    pub domain: Option<String>,
    pub human_id: Option<HumanId>,
    pub features: HashMap<String, Value>,
}

impl Context {
    pub fn similar_to(&self, other: &Context) -> bool {
        self.task_type == other.task_type
    }

    pub fn matches(&self, other: &Context) -> bool {
        self.task_type == other.task_type &&
        self.domain == other.domain
    }

    pub fn intersection<'a>(contexts: impl Iterator<Item = &'a Context>) -> Option<Context> {
        // Would compute common context
        None
    }
}

/// Outcome of an experience
#[derive(Clone)]
pub struct ExperienceOutcome {
    pub success: bool,
    pub quality: f32,
    pub unexpected: bool,
    pub details: Option<String>,
}

/// A learned pattern
pub struct Pattern {
    pub context: Context,
    pub approach: String,
    pub success_rate: f32,
    pub confidence: f32,
}

/// A learned lesson
pub struct Lesson {
    pub context: Context,
    pub approach: String,
    pub effectiveness: f32,
    pub lesson_type: LessonType,
}

pub enum LessonType {
    EffectiveApproach,
    IneffectiveApproach,
    SurprisingOutcome,
    FeedbackLearning,
}

/// Pattern storage
pub struct PatternStore {
    patterns: Vec<Pattern>,
}

impl PatternStore {
    pub fn new() -> Self {
        Self { patterns: vec![] }
    }

    pub fn add(&mut self, pattern: Pattern) {
        self.patterns.push(pattern);
    }
}

/// Lesson storage
pub struct LessonStore {
    lessons: Vec<Lesson>,
}

impl LessonStore {
    pub fn new() -> Self {
        Self { lessons: vec![] }
    }

    pub fn add(&mut self, lesson: Lesson) {
        self.lessons.push(lesson);
    }
}
```

### 2. Skill System

```sigil
/// Skill management system
pub struct SkillSystem {
    /// Registered skills
    skills: HashMap<String, SkillProfile>,

    /// Skill hierarchy
    hierarchy: SkillHierarchy,

    /// Transfer relationships
    transfers: Vec<SkillTransfer>,

    /// Configuration
    config: SkillConfig,
}

pub struct SkillConfig {
    /// Experience needed per level
    pub experience_per_level: u32,

    /// Decay rate without practice
    pub decay_rate: f32,

    /// Default transfer rate
    pub default_transfer_rate: f32,
}

impl SkillSystem {
    pub fn new() -> Self {
        Self {
            skills: HashMap::new(),
            hierarchy: SkillHierarchy::new(),
            transfers: vec![],
            config: SkillConfig::default(),
        }
    }

    /// Define a skill
    pub fn define(&mut self, skill: SkillDefinition) {
        let profile = SkillProfile::new(skill.name.clone());
        self.skills.insert(skill.name.clone(), profile);

        // Add subskills
        for sub in skill.subskills {
            self.hierarchy.add_child(&skill.name, &sub);
            self.define(SkillDefinition {
                name: sub,
                subskills: vec![],
            });
        }
    }

    /// Get skill profile
    pub fn get(&self, name: &str) -> Option<&SkillProfile> {
        self.skills.get(name)
    }

    /// Record skill exercise
    pub fn record_exercise(&mut self, skill_name: &str, exp: &Experience) {
        if let Some(skill) = self.skills.get_mut(skill_name) {
            let exercise = Exercise {
                task_id: exp.id.clone(),
                performance: Performance::from_outcome(&exp.outcome),
                duration: Duration::zero(), // Would be tracked
                quality: exp.outcome.quality,
                timestamp: Timestamp::now(),
            };

            skill.record_exercise(exercise);

            // Apply transfer learning
            for transfer in &self.transfers {
                if &transfer.source == skill_name {
                    if let Some(target) = self.skills.get_mut(&transfer.target) {
                        let transferred_quality = exp.outcome.quality * transfer.rate;
                        target.record_transfer(transferred_quality);
                    }
                }
            }
        }
    }

    /// Register skill transfer
    pub fn register_transfer(&mut self, transfer: SkillTransfer) {
        self.transfers.push(transfer);
    }

    /// Get all skills
    pub fn all_skills(&self) -> impl Iterator<Item = &SkillProfile> {
        self.skills.values()
    }
}

/// Skill definition
pub struct SkillDefinition {
    pub name: String,
    pub subskills: Vec<String>,
}

impl SkillDefinition {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            subskills: vec![],
        }
    }

    pub fn with_subskill(mut self, name: &str) -> Self {
        self.subskills.push(name.to_string());
        self
    }
}

/// Profile for a skill
pub struct SkillProfile {
    pub name: String,
    pub level: SkillLevel,
    pub proficiency: f32,
    pub experience_count: u32,
    pub exercises: VecDeque<Exercise>,
    pub last_practiced: Timestamp,
}

impl SkillProfile {
    pub fn new(name: String) -> Self {
        Self {
            name,
            level: SkillLevel::Novice,
            proficiency: 0.0,
            experience_count: 0,
            exercises: VecDeque::new(),
            last_practiced: Timestamp::now(),
        }
    }

    pub fn proficiency(&self) -> f32 {
        self.proficiency
    }

    pub fn level(&self) -> SkillLevel {
        self.level
    }

    pub fn experience_count(&self) -> u32 {
        self.experience_count
    }

    pub fn recent_accuracy(&self) -> f32 {
        let recent: Vec<_> = self.exercises.iter().rev().take(10).collect();
        if recent.is_empty() {
            return 0.0;
        }

        recent.iter()
            .map(|e| e.quality)
            .sum::<f32>() / recent.len() as f32
    }

    pub fn record_exercise(&mut self, exercise: Exercise) {
        // Update stats
        self.experience_count += 1;
        self.last_practiced = Timestamp::now();

        // Update proficiency (exponential moving average)
        let alpha = 0.1;
        self.proficiency = self.proficiency * (1.0 - alpha) + exercise.quality * alpha;

        // Update level
        self.level = SkillLevel::from_proficiency(self.proficiency);

        // Store exercise
        self.exercises.push_back(exercise);

        // Limit history
        while self.exercises.len() > 100 {
            self.exercises.pop_front();
        }
    }

    pub fn record_transfer(&mut self, quality: f32) {
        // Transfer learning contributes less
        let alpha = 0.05;
        self.proficiency = self.proficiency * (1.0 - alpha) + quality * alpha;
        self.level = SkillLevel::from_proficiency(self.proficiency);
    }

    pub fn subskills(&self) -> Vec<&SkillProfile> {
        // Would return subskills
        vec![]
    }
}

/// Skill level
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum SkillLevel {
    Novice,
    AdvancedBeginner,
    Competent,
    Proficient,
    Expert,
}

impl SkillLevel {
    pub fn from_proficiency(p: f32) -> Self {
        if p >= 0.9 { SkillLevel::Expert }
        else if p >= 0.7 { SkillLevel::Proficient }
        else if p >= 0.5 { SkillLevel::Competent }
        else if p >= 0.3 { SkillLevel::AdvancedBeginner }
        else { SkillLevel::Novice }
    }
}

/// Exercise of a skill
pub struct Exercise {
    pub task_id: ExperienceId,
    pub performance: Performance,
    pub duration: Duration,
    pub quality: f32,
    pub timestamp: Timestamp,
}

/// Performance assessment
#[derive(Copy, Clone)]
pub enum Performance {
    Excellent,
    Good,
    Adequate,
    Poor,
    Failed,
}

impl Performance {
    pub fn from_outcome(outcome: &ExperienceOutcome) -> Self {
        if !outcome.success { return Performance::Failed; }

        if outcome.quality >= 0.9 { Performance::Excellent }
        else if outcome.quality >= 0.7 { Performance::Good }
        else if outcome.quality >= 0.5 { Performance::Adequate }
        else { Performance::Poor }
    }
}

/// Skill hierarchy
pub struct SkillHierarchy {
    parent: HashMap<String, String>,
    children: HashMap<String, Vec<String>>,
}

impl SkillHierarchy {
    pub fn new() -> Self {
        Self {
            parent: HashMap::new(),
            children: HashMap::new(),
        }
    }

    pub fn add_child(&mut self, parent: &str, child: &str) {
        self.parent.insert(child.to_string(), parent.to_string());
        self.children
            .entry(parent.to_string())
            .or_insert_with(Vec::new)
            .push(child.to_string());
    }
}

/// Skill transfer relationship
pub struct SkillTransfer {
    pub source: String,
    pub target: String,
    pub rate: f32,
    pub aspects: Vec<String>,
}
```

### 3. Adaptation System

```sigil
/// Adaptation system
pub struct AdaptationSystem {
    /// Human profiles
    profiles: HashMap<HumanId, HumanProfile>,

    /// Domain adaptations
    domains: HashMap<String, DomainAdaptation>,

    /// Default style
    default_style: CommunicationStyle,

    /// Configuration
    config: AdaptationConfig,
}

pub struct AdaptationConfig {
    /// Learning rate for preferences
    pub preference_learning_rate: f32,

    /// Decay rate for old preferences
    pub preference_decay_rate: f32,
}

impl AdaptationSystem {
    pub fn new() -> Self {
        Self {
            profiles: HashMap::new(),
            domains: HashMap::new(),
            default_style: CommunicationStyle::default(),
            config: AdaptationConfig::default(),
        }
    }

    /// Get communication style for human
    pub fn style_for(&self, human_id: &HumanId) -> CommunicationStyle {
        self.profiles.get(human_id)
            .map(|p| p.preferred_style.clone())
            .unwrap_or_else(|| self.default_style.clone())
    }

    /// Update profile based on feedback
    pub fn update_profile(&mut self, human_id: &HumanId, feedback: &Feedback) {
        let profile = self.profiles
            .entry(human_id.clone())
            .or_insert_with(|| HumanProfile::new(human_id.clone()));

        // Update based on feedback
        match feedback.feedback_type {
            FeedbackType::TooVerbose => {
                profile.adjust_verbosity(-0.1);
            }
            FeedbackType::NeedMoreDetail => {
                profile.adjust_verbosity(0.1);
            }
            FeedbackType::TooFormal => {
                profile.adjust_formality(-0.1);
            }
            FeedbackType::TooInformal => {
                profile.adjust_formality(0.1);
            }
            FeedbackType::GoodApproach => {
                profile.reinforce_current();
            }
            _ => {}
        }
    }

    /// Adapt to domain
    pub fn adapt_to_domain(&mut self, domain: &str, knowledge: DomainKnowledge) {
        self.domains.insert(domain.to_string(), DomainAdaptation {
            domain: domain.to_string(),
            terminology: knowledge.terminology,
            conventions: knowledge.conventions,
            constraints: knowledge.constraints,
        });
    }

    /// Get domain adaptation
    pub fn domain_adaptation(&self, domain: &str) -> Option<&DomainAdaptation> {
        self.domains.get(domain)
    }
}

/// Profile for a specific human
pub struct HumanProfile {
    pub human_id: HumanId,
    pub preferred_style: CommunicationStyle,
    pub interaction_history: VecDeque<InteractionRecord>,
    pub last_interaction: Timestamp,
}

impl HumanProfile {
    pub fn new(human_id: HumanId) -> Self {
        Self {
            human_id,
            preferred_style: CommunicationStyle::default(),
            interaction_history: VecDeque::new(),
            last_interaction: Timestamp::now(),
        }
    }

    pub fn adjust_verbosity(&mut self, delta: f32) {
        self.preferred_style.verbosity =
            (self.preferred_style.verbosity + delta).clamp(-1.0, 1.0);
    }

    pub fn adjust_formality(&mut self, delta: f32) {
        self.preferred_style.formality =
            (self.preferred_style.formality + delta).clamp(-1.0, 1.0);
    }

    pub fn reinforce_current(&mut self) {
        // Reinforce current style by moving toward extremes
    }
}

/// Communication style preferences
#[derive(Clone)]
pub struct CommunicationStyle {
    /// Verbosity (-1 = minimal, 1 = verbose)
    pub verbosity: f32,

    /// Formality (-1 = casual, 1 = formal)
    pub formality: f32,

    /// Proactivity (-1 = reactive, 1 = proactive)
    pub proactivity: f32,

    /// Detail level (-1 = high-level, 1 = detailed)
    pub detail_level: f32,

    /// Check-in frequency (-1 = rare, 1 = frequent)
    pub check_in_frequency: f32,
}

impl CommunicationStyle {
    pub fn default() -> Self {
        Self {
            verbosity: 0.0,
            formality: 0.0,
            proactivity: 0.0,
            detail_level: 0.0,
            check_in_frequency: 0.0,
        }
    }
}

struct InteractionRecord {
    timestamp: Timestamp,
    interaction_type: String,
    outcome: String,
}

/// Domain adaptation
pub struct DomainAdaptation {
    pub domain: String,
    pub terminology: HashMap<String, String>,
    pub conventions: Vec<String>,
    pub constraints: Vec<String>,
}

/// Domain knowledge
pub struct DomainKnowledge {
    pub terminology: HashMap<String, String>,
    pub conventions: Vec<String>,
    pub constraints: Vec<String>,
}

/// Human identifier
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct HumanId {
    pub id: String,
}
```

### 4. Reflection System

```sigil
/// Reflection system
pub struct ReflectionSystem {
    /// Configuration
    config: ReflectionConfig,
}

pub struct ReflectionConfig {
    /// Minimum experiences for reflection
    pub min_experiences: usize,

    /// Pattern significance threshold
    pub pattern_significance: f32,
}

impl ReflectionSystem {
    pub fn new() -> Self {
        Self {
            config: ReflectionConfig::default(),
        }
    }

    /// Analyze and reflect
    pub fn analyze(
        &self,
        period: ReflectionPeriod,
        learning: &LearningEngine,
        skills: &SkillSystem,
    ) -> Reflection {
        // Gather data for period
        let experiences = self.gather_experiences(period, learning);
        let skill_data = self.gather_skill_data(period, skills);

        // Analyze successes
        let successes = self.identify_successes(&experiences);

        // Identify improvement areas
        let improvements = self.identify_improvements(&experiences, &skill_data);

        // Extract patterns
        let patterns = self.extract_patterns(&experiences);

        // Generate insights
        let insights = self.generate_insights(&experiences, &skill_data, &patterns);

        // Measure growth
        let growth = self.measure_growth(&skill_data);

        Reflection {
            period,
            successes,
            improvements,
            patterns,
            insights,
            growth,
            timestamp: Timestamp::now(),
        }
    }

    fn gather_experiences(&self, period: ReflectionPeriod, learning: &LearningEngine) -> Vec<Experience> {
        let cutoff = period.start_time();
        learning.experiences.iter()
            .filter(|e| e.timestamp >= cutoff)
            .cloned()
            .collect()
    }

    fn gather_skill_data(&self, period: ReflectionPeriod, skills: &SkillSystem) -> Vec<SkillSnapshot> {
        skills.all_skills()
            .map(|s| SkillSnapshot {
                name: s.name.clone(),
                level: s.level,
                proficiency: s.proficiency,
                recent_exercises: s.exercises.iter()
                    .filter(|e| e.timestamp >= period.start_time())
                    .count(),
            })
            .collect()
    }

    fn identify_successes(&self, experiences: &[Experience]) -> Vec<Success> {
        experiences.iter()
            .filter(|e| e.outcome.success && e.outcome.quality >= 0.8)
            .map(|e| Success {
                description: format!("Successful {} task", e.context.task_type),
                factors: vec![], // Would analyze contributing factors
            })
            .collect()
    }

    fn identify_improvements(&self, experiences: &[Experience], skills: &[SkillSnapshot]) -> Vec<ImprovementArea> {
        let mut improvements = Vec::new();

        // From failed experiences
        for exp in experiences.iter().filter(|e| !e.outcome.success) {
            improvements.push(ImprovementArea {
                skill: exp.context.task_type.clone(),
                suggestion: format!("Review approach for {} tasks", exp.context.task_type),
                priority: Priority::Medium,
            });
        }

        // From low proficiency skills
        for skill in skills.iter().filter(|s| s.proficiency < 0.5) {
            improvements.push(ImprovementArea {
                skill: skill.name.clone(),
                suggestion: format!("Practice {} more", skill.name),
                priority: Priority::Low,
            });
        }

        improvements
    }

    fn extract_patterns(&self, experiences: &[Experience]) -> Vec<String> {
        // Would identify patterns in experiences
        vec![]
    }

    fn generate_insights(&self, _experiences: &[Experience], _skills: &[SkillSnapshot], _patterns: &[String]) -> Vec<Insight> {
        // Would generate insights
        vec![]
    }

    fn measure_growth(&self, skills: &[SkillSnapshot]) -> GrowthMeasurement {
        let avg_proficiency = skills.iter()
            .map(|s| s.proficiency)
            .sum::<f32>() / skills.len().max(1) as f32;

        GrowthMeasurement {
            overall: avg_proficiency,
            by_skill: skills.iter()
                .map(|s| (s.name.clone(), s.proficiency))
                .collect(),
        }
    }
}

/// Reflection period
#[derive(Copy, Clone)]
pub enum ReflectionPeriod {
    Daily,
    Weekly,
    Monthly,
}

impl ReflectionPeriod {
    pub fn start_time(&self) -> Timestamp {
        let now = Timestamp::now();
        let offset = match self {
            ReflectionPeriod::Daily => Duration::hours(24),
            ReflectionPeriod::Weekly => Duration::hours(168),
            ReflectionPeriod::Monthly => Duration::hours(720),
        };
        Timestamp::from_nanos(now.nanos.saturating_sub(offset.as_nanos()))
    }
}

/// Reflection result
pub struct Reflection {
    pub period: ReflectionPeriod,
    pub successes: Vec<Success>,
    pub improvements: Vec<ImprovementArea>,
    pub patterns: Vec<String>,
    pub insights: Vec<Insight>,
    pub growth: GrowthMeasurement,
    pub timestamp: Timestamp,
}

impl Reflection {
    pub fn successes(&self) -> &[Success] {
        &self.successes
    }

    pub fn improvement_areas(&self) -> &[ImprovementArea] {
        &self.improvements
    }

    pub fn patterns(&self) -> &[String] {
        &self.patterns
    }
}

/// A success to celebrate
pub struct Success {
    pub description: String,
    pub factors: Vec<String>,
}

/// Area for improvement
pub struct ImprovementArea {
    pub skill: String,
    pub suggestion: String,
    pub priority: Priority,
}

#[derive(Copy, Clone)]
pub enum Priority {
    High,
    Medium,
    Low,
}

/// An insight from reflection
pub struct Insight {
    pub observation: String,
    pub implication: String,
    pub action: Option<String>,
}

/// Skill snapshot for reflection
pub struct SkillSnapshot {
    pub name: String,
    pub level: SkillLevel,
    pub proficiency: f32,
    pub recent_exercises: usize,
}

/// Growth measurement
pub struct GrowthMeasurement {
    pub overall: f32,
    pub by_skill: Vec<(String, f32)>,
}

impl GrowthMeasurement {
    pub fn overall_growth(&self) -> f32 {
        self.overall
    }
}
```

### 5. Feedback System

```sigil
/// Feedback from humans
#[derive(Clone)]
pub struct Feedback {
    pub human_id: Option<HumanId>,
    pub context: Context,
    pub feedback_type: FeedbackType,
    pub message: Option<String>,
    pub timestamp: Timestamp,
}

impl Feedback {
    pub fn positivity(&self) -> f32 {
        match self.feedback_type {
            FeedbackType::Positive(_) => 1.0,
            FeedbackType::Neutral => 0.5,
            FeedbackType::Negative(_) => 0.0,
            FeedbackType::TooVerbose => 0.3,
            FeedbackType::NeedMoreDetail => 0.3,
            FeedbackType::TooFormal => 0.3,
            FeedbackType::TooInformal => 0.3,
            FeedbackType::GoodApproach => 0.9,
            FeedbackType::WrongApproach => 0.1,
            FeedbackType::Custom(_) => 0.5,
        }
    }
}

#[derive(Clone)]
pub enum FeedbackType {
    Positive(String),
    Negative(String),
    Neutral,
    TooVerbose,
    NeedMoreDetail,
    TooFormal,
    TooInformal,
    GoodApproach,
    WrongApproach,
    Custom(String),
}

/// Outcome of a task
pub struct Outcome {
    pub task: Task,
    pub success: bool,
    pub quality: f32,
    pub human_satisfaction: Option<Satisfaction>,
}

pub struct Task {
    pub context: Context,
    pub approach: String,
}

#[derive(Copy, Clone)]
pub enum Satisfaction {
    Delighted,
    Satisfied,
    Neutral,
    Disappointed,
    Frustrated,
}
```

## Integration Example

```sigil
use gnosis::Gnosis;
use daemon::Daemon;
use covenant::Covenant;
use engram::Engram;

daemon LearningAgent {
    gnosis: Gnosis,
    memory: Engram,
    covenant: Covenant,

    fn on_init(&mut self) {
        // Initialize gnosis with memory integration
        self.gnosis = Gnosis::new()
            .with_memory(&self.memory)
            .build();

        // Define skills
        self.gnosis.skills.define(SkillDefinition::new("research")
            .with_subskill("query_formulation")
            .with_subskill("source_evaluation")
            .with_subskill("synthesis")
        );
    }

    fn after_task(&mut self, task: &Task, result: &Result) {
        // Record experience
        self.gnosis.experience(Experience {
            id: ExperienceId::new(),
            context: task.context(),
            action: task.approach(),
            outcome: ExperienceOutcome::from(result),
            feedback: None,
            timestamp: Timestamp::now(),
        });
    }

    fn on_feedback(&mut self, feedback: HumanFeedback) {
        // Learn from feedback
        self.gnosis.learn_from_feedback(Feedback::from(feedback));
    }

    fn get_approach(&self, context: &Context) -> Approach {
        // Use learning to inform approach
        let suggestions = self.gnosis.suggest_approach(context);

        // Adapt style to human
        let style = self.gnosis.adapted_style(&context.human_id);

        Approach {
            strategy: suggestions.best(),
            style,
        }
    }

    fn daily_reflection(&mut self) {
        let reflection = self.gnosis.reflect(ReflectionPeriod::Daily);

        // Log insights
        for insight in reflection.insights {
            self.memory.learn(insight);
        }

        // Inform covenant about improvements needed
        if !reflection.improvements.is_empty() {
            self.covenant.inform(&format!(
                "Areas I'm working to improve: {}",
                reflection.improvements.iter()
                    .map(|i| &i.skill)
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
}
```

---

*Learning is the path to wisdom*
