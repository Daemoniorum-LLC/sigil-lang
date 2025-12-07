# Oracle Architecture

## System Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                ORACLE                                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                       EXPLANATION ENGINE                                │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │ │
│  │  │ SUMMARY  │ │REASONING │ │ EVIDENCE │ │CONFIDENCE│ │ALTERNATIVE│   │ │
│  │  │GENERATOR │ │EXTRACTOR │ │ATTRIBUTOR│ │ ANALYZER │ │ ANALYZER  │   │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘   │ │
│  └───────────────────────────────┬────────────────────────────────────────┘ │
│                                  │                                           │
│  ┌───────────────────────────────▼────────────────────────────────────────┐ │
│  │                         TRACE SYSTEM                                    │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                 │ │
│  │  │  TRACE   │ │  STEP    │ │ CAUSAL   │ │PROVENANCE│                 │ │
│  │  │COLLECTOR │ │ RECORDER │ │  GRAPH   │ │ TRACKER  │                 │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘                 │ │
│  └───────────────────────────────┬────────────────────────────────────────┘ │
│                                  │                                           │
│  ┌───────────────────────────────▼────────────────────────────────────────┐ │
│  │                    COUNTERFACTUAL ENGINE                                │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐                 │ │
│  │  │ WHAT-IF  │ │ALTERNATIVE│ │ CONTRAST │ │SENSITIVITY│                 │ │
│  │  │ ANALYZER │ │ GENERATOR │ │ BUILDER  │ │ ANALYZER │                 │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘                 │ │
│  └───────────────────────────────┬────────────────────────────────────────┘ │
│                                  │                                           │
│  ┌───────────────────────────────▼────────────────────────────────────────┐ │
│  │                      PRESENTATION LAYER                                 │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐   │ │
│  │  │  HUMAN   │ │  GRAPH   │ │ ANALOGY  │ │INTERACTIVE│ │  EXPORT  │   │ │
│  │  │ RENDERER │ │VISUALIZER│ │  FINDER  │ │ EXPLORER │ │  ENGINE  │   │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘   │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Explanation System

```sigil
/// The main Oracle structure
pub struct Oracle {
    /// Trace collector
    trace: TraceCollector,

    /// Explanation engine
    explainer: ExplanationEngine,

    /// Counterfactual engine
    counterfactual: CounterfactualEngine,

    /// Presentation layer
    presenter: Presenter,

    /// Configuration
    config: OracleConfig,

    /// Integration points
    memory: Option<Box<dyn MemoryAccess>>,
}

impl Oracle {
    pub fn new() -> Self {
        Self {
            trace: TraceCollector::new(),
            explainer: ExplanationEngine::new(),
            counterfactual: CounterfactualEngine::new(),
            presenter: Presenter::new(),
            config: OracleConfig::default(),
            memory: None,
        }
    }

    /// Explain a decision
    pub fn explain<T: Explainable>(&self, decision: &T, level: ExplanationLevel) -> Explanation {
        // Get trace for this decision
        let trace = self.trace.get_trace(decision.id());

        // Build explanation
        self.explainer.explain(decision, trace, level)
    }

    /// Start tracing
    pub fn trace_on(&mut self) {
        self.trace.enable();
    }

    /// Stop tracing
    pub fn trace_off(&mut self) {
        self.trace.disable();
    }

    /// Record a reasoning step
    pub fn record_step(&mut self, step: ReasoningStep) {
        self.trace.record(step);
    }

    /// Analyze counterfactual
    pub fn counterfactual(&self, actual: &Action, alternative: &Action) -> Counterfactual {
        self.counterfactual.analyze(actual, alternative, &self.trace)
    }

    /// Get confidence breakdown
    pub fn confidence_breakdown<T: Explainable>(&self, decision: &T) -> ConfidenceBreakdown {
        self.explainer.analyze_confidence(decision)
    }

    /// Create interactive explainer
    pub fn interactive_explainer<T: Explainable>(&self, decision: &T) -> InteractiveExplainer {
        InteractiveExplainer::new(decision, self)
    }
}

/// Explanation level
#[derive(Copy, Clone)]
pub enum ExplanationLevel {
    /// Brief summary
    Brief,
    /// Standard explanation
    Standard,
    /// Full detailed explanation
    Full,
    /// Technical/debugging level
    Technical,
}

/// Trait for things that can be explained
pub trait Explainable {
    fn id(&self) -> ExplainableId;
    fn description(&self) -> String;
    fn factors(&self) -> Vec<Factor>;
}
```

### 2. Explanation Structure

```sigil
/// A complete explanation
pub struct Explanation {
    /// What was decided
    pub decision: String,

    /// Summary of reasoning
    pub summary: String,

    /// Detailed reasoning chain
    pub reasoning: ReasoningChain,

    /// Supporting evidence
    pub evidence: Vec<Evidence>,

    /// Confidence assessment
    pub confidence: ConfidenceAssessment,

    /// Alternatives considered
    pub alternatives: Vec<Alternative>,

    /// Uncertainties and caveats
    pub uncertainties: Vec<Uncertainty>,

    /// Level of explanation
    pub level: ExplanationLevel,
}

impl Explanation {
    /// Get human-readable summary
    pub fn summary(&self) -> &str {
        &self.summary
    }

    /// Get detailed reasoning
    pub fn reasoning(&self) -> String {
        self.reasoning.to_prose()
    }

    /// Get evidence list
    pub fn evidence(&self) -> &[Evidence] {
        &self.evidence
    }

    /// Get alternatives considered
    pub fn alternatives_considered(&self) -> &[Alternative] {
        &self.alternatives
    }

    /// Get confidence
    pub fn confidence(&self) -> f32 {
        self.confidence.overall
    }

    /// Render as human-readable text
    pub fn to_human_readable(&self) -> String {
        self.render(Format::Prose)
    }

    /// Render in specified format
    pub fn render(&self, format: Format) -> String {
        match format {
            Format::Prose => self.render_prose(),
            Format::Structured => self.render_structured(),
            Format::Bullets => self.render_bullets(),
            Format::Json => self.render_json(),
        }
    }

    fn render_prose(&self) -> String {
        let mut output = String::new();

        // Summary
        output.push_str(&format!("{}\n\n", self.summary));

        // Reasoning (if standard or above)
        if self.level as u8 >= ExplanationLevel::Standard as u8 {
            output.push_str("Reasoning:\n");
            output.push_str(&self.reasoning.to_prose());
            output.push_str("\n\n");
        }

        // Evidence (if standard or above)
        if self.level as u8 >= ExplanationLevel::Standard as u8 && !self.evidence.is_empty() {
            output.push_str("Based on:\n");
            for e in &self.evidence {
                output.push_str(&format!("  - {} ({})\n", e.content, e.source));
            }
            output.push_str("\n");
        }

        // Confidence
        output.push_str(&format!("Confidence: {:.0}%", self.confidence.overall * 100.0));

        // Uncertainties (if any)
        if !self.uncertainties.is_empty() {
            output.push_str("\n\nUncertainties:\n");
            for u in &self.uncertainties {
                output.push_str(&format!("  - {}\n", u.description));
            }
        }

        // Alternatives (if full level)
        if self.level as u8 >= ExplanationLevel::Full as u8 && !self.alternatives.is_empty() {
            output.push_str("\nAlternatives considered:\n");
            for alt in &self.alternatives {
                output.push_str(&format!("  - {}: {} (rejected: {})\n",
                    alt.option, alt.description, alt.rejection_reason
                ));
            }
        }

        output
    }

    fn render_structured(&self) -> String {
        // YAML-like structured output
        let mut output = String::new();
        output.push_str(&format!("decision: {}\n", self.decision));
        output.push_str(&format!("summary: {}\n", self.summary));
        output.push_str(&format!("confidence: {:.2}\n", self.confidence.overall));
        output.push_str("reasoning:\n");
        for step in self.reasoning.steps() {
            output.push_str(&format!("  - {}\n", step.description));
        }
        output
    }

    fn render_bullets(&self) -> String {
        let mut output = String::new();
        output.push_str(&format!("• Decision: {}\n", self.decision));
        output.push_str(&format!("• Summary: {}\n", self.summary));
        output.push_str(&format!("• Confidence: {:.0}%\n", self.confidence.overall * 100.0));
        if !self.evidence.is_empty() {
            output.push_str("• Evidence:\n");
            for e in &self.evidence {
                output.push_str(&format!("  • {}\n", e.content));
            }
        }
        output
    }

    fn render_json(&self) -> String {
        // Would serialize to JSON
        "{}".to_string()
    }
}

/// Reasoning chain
pub struct ReasoningChain {
    steps: Vec<ReasoningStep>,
}

impl ReasoningChain {
    pub fn new() -> Self {
        Self { steps: vec![] }
    }

    pub fn add(&mut self, step: ReasoningStep) {
        self.steps.push(step);
    }

    pub fn steps(&self) -> &[ReasoningStep] {
        &self.steps
    }

    pub fn to_prose(&self) -> String {
        self.steps.iter()
            .enumerate()
            .map(|(i, s)| format!("{}. {}", i + 1, s.description))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Single reasoning step
pub struct ReasoningStep {
    /// Step identifier
    pub id: StepId,

    /// Step type
    pub step_type: StepType,

    /// Human-readable description
    pub description: String,

    /// Inputs to this step
    pub inputs: Vec<StepInput>,

    /// Output of this step
    pub output: Option<StepOutput>,

    /// Reasoning applied
    pub reasoning: String,

    /// Confidence in this step
    pub confidence: f32,

    /// Timestamp
    pub timestamp: Timestamp,
}

pub enum StepType {
    Observation,
    Retrieval,
    Inference,
    Evaluation,
    Decision,
    Planning,
    Action,
}

pub struct StepInput {
    pub name: String,
    pub value: Value,
    pub source: Option<String>,
}

pub struct StepOutput {
    pub value: Value,
    pub confidence: f32,
}

/// Evidence supporting a decision
pub struct Evidence {
    /// Evidence content
    pub content: String,

    /// Source of evidence
    pub source: String,

    /// Type of evidence
    pub evidence_type: EvidenceType,

    /// Weight in the decision
    pub weight: f32,

    /// Epistemic status
    pub epistemic: Epistemic,

    /// Timestamp
    pub timestamp: Timestamp,
}

pub enum EvidenceType {
    DirectObservation,
    Memory,
    Inference,
    ExternalSource,
    UserStatement,
}

pub enum Epistemic {
    Certain,
    HighConfidence,
    MediumConfidence,
    LowConfidence,
    Uncertain,
}

/// Alternative considered but not chosen
pub struct Alternative {
    /// The option
    pub option: String,

    /// Description
    pub description: String,

    /// Why it was rejected
    pub rejection_reason: String,

    /// Score if applicable
    pub score: Option<f32>,
}

/// Uncertainty or caveat
pub struct Uncertainty {
    /// Description of uncertainty
    pub description: String,

    /// What it affects
    pub affects: Vec<String>,

    /// How it could change the decision
    pub impact: String,
}
```

### 3. Trace System

```sigil
/// Collects reasoning traces
pub struct TraceCollector {
    /// Active tracing
    enabled: bool,

    /// Current trace
    current: Option<Trace>,

    /// Completed traces
    traces: HashMap<ExplainableId, Trace>,

    /// Configuration
    config: TraceConfig,
}

pub struct TraceConfig {
    /// Maximum steps to keep
    pub max_steps: usize,

    /// Record inputs/outputs
    pub record_io: bool,

    /// Record timing
    pub record_timing: bool,
}

impl TraceCollector {
    pub fn new() -> Self {
        Self {
            enabled: false,
            current: None,
            traces: HashMap::new(),
            config: TraceConfig::default(),
        }
    }

    pub fn enable(&mut self) {
        self.enabled = true;
        self.current = Some(Trace::new());
    }

    pub fn disable(&mut self) {
        self.enabled = false;
    }

    pub fn record(&mut self, step: ReasoningStep) {
        if self.enabled {
            if let Some(ref mut trace) = self.current {
                trace.add_step(step);
            }
        }
    }

    pub fn complete(&mut self, id: ExplainableId) {
        if let Some(trace) = self.current.take() {
            self.traces.insert(id, trace);
        }
    }

    pub fn get_trace(&self, id: &ExplainableId) -> Option<&Trace> {
        self.traces.get(id)
    }
}

/// A complete reasoning trace
pub struct Trace {
    /// Steps in order
    steps: Vec<ReasoningStep>,

    /// Causal graph
    causal_graph: CausalGraph,

    /// Start time
    started: Timestamp,

    /// End time
    ended: Option<Timestamp>,
}

impl Trace {
    pub fn new() -> Self {
        Self {
            steps: vec![],
            causal_graph: CausalGraph::new(),
            started: Timestamp::now(),
            ended: None,
        }
    }

    pub fn add_step(&mut self, step: ReasoningStep) {
        // Add to causal graph
        self.causal_graph.add_node(step.id.clone(), &step);

        // Link to previous step if any
        if let Some(prev) = self.steps.last() {
            self.causal_graph.add_edge(&prev.id, &step.id);
        }

        self.steps.push(step);
    }

    pub fn complete(&mut self) {
        self.ended = Some(Timestamp::now());
    }

    pub fn steps(&self) -> &[ReasoningStep] {
        &self.steps
    }

    pub fn causal_graph(&self) -> &CausalGraph {
        &self.causal_graph
    }
}

/// Causal graph of reasoning
pub struct CausalGraph {
    nodes: HashMap<StepId, CausalNode>,
    edges: Vec<(StepId, StepId)>,
}

struct CausalNode {
    id: StepId,
    step_type: StepType,
    description: String,
}

impl CausalGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: vec![],
        }
    }

    pub fn add_node(&mut self, id: StepId, step: &ReasoningStep) {
        self.nodes.insert(id.clone(), CausalNode {
            id,
            step_type: step.step_type.clone(),
            description: step.description.clone(),
        });
    }

    pub fn add_edge(&mut self, from: &StepId, to: &StepId) {
        self.edges.push((from.clone(), to.clone()));
    }

    /// Export as DOT format for Graphviz
    pub fn to_dot(&self) -> String {
        let mut output = String::from("digraph reasoning {\n");

        for (id, node) in &self.nodes {
            output.push_str(&format!(
                "  {} [label=\"{}\"];\n",
                id.to_short(), node.description.truncate(30)
            ));
        }

        for (from, to) in &self.edges {
            output.push_str(&format!(
                "  {} -> {};\n",
                from.to_short(), to.to_short()
            ));
        }

        output.push_str("}\n");
        output
    }

    /// Export as Mermaid diagram
    pub fn to_mermaid(&self) -> String {
        let mut output = String::from("graph TD\n");

        for (id, node) in &self.nodes {
            output.push_str(&format!(
                "  {}[{}]\n",
                id.to_short(), node.description.truncate(30)
            ));
        }

        for (from, to) in &self.edges {
            output.push_str(&format!(
                "  {} --> {}\n",
                from.to_short(), to.to_short()
            ));
        }

        output
    }

    /// Export as ASCII art
    pub fn to_ascii(&self) -> String {
        // Simplified ASCII representation
        let mut output = String::new();

        for (i, (id, node)) in self.nodes.iter().enumerate() {
            if i > 0 {
                output.push_str("    │\n    ▼\n");
            }
            output.push_str(&format!("┌─ {} ─┐\n", node.description.truncate(40)));
        }

        output
    }
}
```

### 4. Counterfactual Engine

```sigil
/// Analyzes counterfactuals
pub struct CounterfactualEngine {
    /// Configuration
    config: CounterfactualConfig,
}

pub struct CounterfactualConfig {
    /// Maximum depth of analysis
    pub max_depth: usize,

    /// Include probability estimates
    pub estimate_probabilities: bool,
}

impl CounterfactualEngine {
    pub fn new() -> Self {
        Self {
            config: CounterfactualConfig::default(),
        }
    }

    /// Analyze why actual was chosen over alternative
    pub fn analyze(
        &self,
        actual: &Action,
        alternative: &Action,
        trace: &TraceCollector,
    ) -> Counterfactual {
        // Find decision point
        let decision_step = self.find_decision_step(actual, trace);

        // Generate reasons for difference
        let reasons = self.generate_reasons(actual, alternative, decision_step.as_ref());

        // Estimate what would have happened
        let consequences = self.project_consequences(alternative);

        Counterfactual {
            actual: actual.description(),
            alternative: alternative.description(),
            reasons,
            decision_point: decision_step.map(|s| s.description.clone()),
            projected_consequences: consequences,
        }
    }

    /// Analyze sensitivity to factors
    pub fn sensitivity_analysis<T: Explainable>(&self, decision: &T) -> SensitivityAnalysis {
        let factors = decision.factors();

        let mut sensitivities = Vec::new();
        for factor in factors {
            let sensitivity = self.measure_sensitivity(decision, &factor);
            sensitivities.push(FactorSensitivity {
                factor: factor.clone(),
                sensitivity,
                threshold: self.find_threshold(decision, &factor),
            });
        }

        SensitivityAnalysis { factors: sensitivities }
    }

    fn find_decision_step(&self, action: &Action, trace: &TraceCollector) -> Option<&ReasoningStep> {
        // Would search trace for relevant decision step
        None
    }

    fn generate_reasons(&self, actual: &Action, alternative: &Action, step: Option<&ReasoningStep>) -> Vec<String> {
        let mut reasons = Vec::new();

        // Compare actions
        if actual.expected_benefit() > alternative.expected_benefit() {
            reasons.push(format!(
                "{} has higher expected benefit",
                actual.description()
            ));
        }

        if actual.risk_level() < alternative.risk_level() {
            reasons.push(format!(
                "{} has lower risk",
                actual.description()
            ));
        }

        // Use step reasoning if available
        if let Some(s) = step {
            reasons.push(s.reasoning.clone());
        }

        reasons
    }

    fn project_consequences(&self, action: &Action) -> Vec<String> {
        // Would project what might have happened
        vec![format!("If {} were chosen...", action.description())]
    }

    fn measure_sensitivity<T: Explainable>(&self, decision: &T, factor: &Factor) -> f32 {
        // Would measure how sensitive the decision is to this factor
        0.5
    }

    fn find_threshold<T: Explainable>(&self, decision: &T, factor: &Factor) -> Option<f32> {
        // Would find the threshold where decision would change
        None
    }
}

/// Counterfactual analysis result
pub struct Counterfactual {
    /// What was chosen
    pub actual: String,

    /// What the alternative was
    pub alternative: String,

    /// Reasons for the choice
    pub reasons: Vec<String>,

    /// Where the decision was made
    pub decision_point: Option<String>,

    /// What might have happened otherwise
    pub projected_consequences: Vec<String>,
}

impl Counterfactual {
    pub fn to_prose(&self) -> String {
        let mut output = String::new();

        output.push_str(&format!(
            "I chose {} instead of {} because:\n",
            self.actual, self.alternative
        ));

        for (i, reason) in self.reasons.iter().enumerate() {
            output.push_str(&format!("{}. {}\n", i + 1, reason));
        }

        if !self.projected_consequences.is_empty() {
            output.push_str("\nIf the alternative had been chosen:\n");
            for consequence in &self.projected_consequences {
                output.push_str(&format!("  - {}\n", consequence));
            }
        }

        output
    }
}

/// Sensitivity analysis result
pub struct SensitivityAnalysis {
    pub factors: Vec<FactorSensitivity>,
}

pub struct FactorSensitivity {
    pub factor: Factor,
    pub sensitivity: f32, // How much decision depends on this
    pub threshold: Option<f32>, // Value at which decision would change
}

pub struct Factor {
    pub name: String,
    pub value: f32,
    pub description: String,
}

impl Clone for Factor {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            value: self.value,
            description: self.description.clone(),
        }
    }
}
```

### 5. Confidence System

```sigil
/// Confidence assessment
pub struct ConfidenceAssessment {
    /// Overall confidence
    pub overall: f32,

    /// Evidence strength
    pub evidence_strength: f32,

    /// Reasoning validity
    pub reasoning_validity: f32,

    /// Experience relevance
    pub experience_relevance: f32,

    /// Specific uncertainties
    pub uncertainties: Vec<UncertaintyFactor>,
}

pub struct UncertaintyFactor {
    pub factor: String,
    pub impact: f32,
    pub description: String,
}

/// Confidence analyzer
pub struct ConfidenceAnalyzer {
    config: ConfidenceConfig,
}

pub struct ConfidenceConfig {
    /// Weight for evidence in overall confidence
    pub evidence_weight: f32,
    /// Weight for reasoning
    pub reasoning_weight: f32,
    /// Weight for experience
    pub experience_weight: f32,
}

impl ConfidenceAnalyzer {
    pub fn analyze<T: Explainable>(&self, decision: &T, trace: Option<&Trace>) -> ConfidenceAssessment {
        // Analyze evidence strength
        let evidence_strength = self.analyze_evidence(decision);

        // Analyze reasoning validity
        let reasoning_validity = self.analyze_reasoning(trace);

        // Analyze experience relevance
        let experience_relevance = self.analyze_experience(decision);

        // Identify uncertainties
        let uncertainties = self.identify_uncertainties(decision);

        // Calculate overall
        let overall = (
            evidence_strength * self.config.evidence_weight +
            reasoning_validity * self.config.reasoning_weight +
            experience_relevance * self.config.experience_weight
        ) / (
            self.config.evidence_weight +
            self.config.reasoning_weight +
            self.config.experience_weight
        );

        // Reduce for uncertainties
        let uncertainty_penalty: f32 = uncertainties.iter()
            .map(|u| u.impact)
            .sum::<f32>()
            .min(0.5);

        ConfidenceAssessment {
            overall: (overall - uncertainty_penalty).max(0.0),
            evidence_strength,
            reasoning_validity,
            experience_relevance,
            uncertainties,
        }
    }

    fn analyze_evidence<T: Explainable>(&self, decision: &T) -> f32 {
        // Would analyze quality and quantity of evidence
        0.7
    }

    fn analyze_reasoning(&self, trace: Option<&Trace>) -> f32 {
        // Would analyze reasoning chain validity
        match trace {
            Some(t) => {
                // Check for logical consistency, valid inferences, etc.
                0.8
            }
            None => 0.5 // No trace = less confidence in reasoning
        }
    }

    fn analyze_experience<T: Explainable>(&self, decision: &T) -> f32 {
        // Would check for similar past decisions
        0.6
    }

    fn identify_uncertainties<T: Explainable>(&self, decision: &T) -> Vec<UncertaintyFactor> {
        // Would identify specific uncertainties
        vec![]
    }
}

/// Breakdown of confidence for human consumption
pub struct ConfidenceBreakdown {
    pub overall: f32,
    pub evidence: f32,
    pub reasoning: f32,
    pub experience: f32,
    pub uncertainties: Vec<String>,
}
```

### 6. Interactive Exploration

```sigil
/// Interactive explanation explorer
pub struct InteractiveExplainer<'a, T: Explainable> {
    decision: &'a T,
    oracle: &'a Oracle,
    current_level: ExplanationLevel,
    history: Vec<ExplainerAction>,
}

enum ExplainerAction {
    DrillDown,
    Counterfactual(String),
    ShowEvidence,
    ExplainConfidence,
}

impl<'a, T: Explainable> InteractiveExplainer<'a, T> {
    pub fn new(decision: &'a T, oracle: &'a Oracle) -> Self {
        Self {
            decision,
            oracle,
            current_level: ExplanationLevel::Standard,
            history: vec![],
        }
    }

    /// Get current explanation
    pub fn current_explanation(&self) -> Explanation {
        self.oracle.explain(self.decision, self.current_level)
    }

    /// Drill down for more detail
    pub fn drill_down(&mut self) -> Explanation {
        self.history.push(ExplainerAction::DrillDown);

        self.current_level = match self.current_level {
            ExplanationLevel::Brief => ExplanationLevel::Standard,
            ExplanationLevel::Standard => ExplanationLevel::Full,
            ExplanationLevel::Full => ExplanationLevel::Technical,
            ExplanationLevel::Technical => ExplanationLevel::Technical,
        };

        self.current_explanation()
    }

    /// Explore a counterfactual
    pub fn counterfactual(&self, what_if: &str) -> String {
        format!("If {}, then the decision might have been different because...", what_if)
    }

    /// Show evidence details
    pub fn show_evidence(&self) -> Vec<Evidence> {
        let explanation = self.oracle.explain(self.decision, ExplanationLevel::Full);
        explanation.evidence
    }

    /// Explain confidence
    pub fn explain_confidence(&self) -> ConfidenceBreakdown {
        self.oracle.confidence_breakdown(self.decision)
    }

    /// Ask a follow-up question
    pub fn ask(&self, question: &str) -> String {
        // Would process natural language questions
        format!("Regarding '{}': ...", question)
    }
}
```

### 7. Presentation Layer

```sigil
/// Presentation and formatting
pub struct Presenter {
    /// Default format
    default_format: Format,

    /// Analogy database
    analogies: AnalogyFinder,
}

#[derive(Copy, Clone)]
pub enum Format {
    Prose,
    Structured,
    Bullets,
    Json,
}

impl Presenter {
    pub fn new() -> Self {
        Self {
            default_format: Format::Prose,
            analogies: AnalogyFinder::new(),
        }
    }

    /// Render explanation in format
    pub fn render(&self, explanation: &Explanation, format: Format) -> String {
        explanation.render(format)
    }

    /// Find analogy for concept
    pub fn find_analogy(&self, concept: &str, audience: Audience) -> Option<Analogy> {
        self.analogies.find(concept, audience)
    }
}

/// Analogy for explaining concepts
pub struct Analogy {
    pub concept: String,
    pub analog: String,
    pub mapping: String,
    pub limitations: Vec<String>,
}

/// Analogy finder
pub struct AnalogyFinder {
    analogies: HashMap<String, Vec<Analogy>>,
}

impl AnalogyFinder {
    pub fn new() -> Self {
        Self {
            analogies: HashMap::new(),
        }
    }

    pub fn find(&self, concept: &str, audience: Audience) -> Option<Analogy> {
        // Would find appropriate analogy for audience
        self.analogies.get(concept)
            .and_then(|list| list.first())
            .cloned()
    }
}

impl Clone for Analogy {
    fn clone(&self) -> Self {
        Self {
            concept: self.concept.clone(),
            analog: self.analog.clone(),
            mapping: self.mapping.clone(),
            limitations: self.limitations.clone(),
        }
    }
}

/// Target audience for explanations
pub enum Audience {
    General,
    Technical,
    Expert,
    Child,
}
```

## Integration Example

```sigil
use oracle::{Oracle, ExplanationLevel};
use daemon::Daemon;
use covenant::Covenant;

daemon ExplainableAgent {
    oracle: Oracle,
    covenant: Covenant,

    fn deliberate(&mut self, context: Context) -> Action {
        // Enable tracing
        self.oracle.trace_on();

        // Record reasoning steps
        self.oracle.record_step(ReasoningStep {
            id: StepId::new(),
            step_type: StepType::Observation,
            description: "Observed user request".to_string(),
            inputs: vec![],
            output: None,
            reasoning: "User asked for help with task".to_string(),
            confidence: 1.0,
            timestamp: Timestamp::now(),
        });

        // Make decision with recorded reasoning
        let action = self.decide(context);

        // Complete trace
        self.oracle.trace.complete(action.id());

        // Explain if needed
        if self.covenant.requires_explanation(&action) {
            let explanation = self.oracle.explain(&action, ExplanationLevel::Standard);
            self.covenant.inform(&explanation.to_human_readable());
        }

        action
    }
}
```

---

*Making the invisible visible*
