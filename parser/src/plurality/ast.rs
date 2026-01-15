//! # AST Extensions for Plurality
//!
//! New AST nodes for alter definitions, switch expressions, headspace
//! navigation, and other plurality-specific constructs.

use crate::ast::{
    Attribute, Block, Expr, Generics, Ident, Param, TypeExpr, Visibility, WhereClause,
};
use crate::span::Span;

// ============================================================================
// ALTER SOURCE MARKERS
// ============================================================================

/// Alter-source markers extend evidentiality to track which alter
/// perceives or controls data.
///
/// These combine with standard evidentiality (!~?‽) to create
/// compound types like `Perception@Abaddon!` (certain, from Abaddon's view).
#[derive(Debug, Clone, PartialEq)]
pub enum AlterSource {
    /// Data from the currently fronting alter (authoritative)
    Fronting,
    /// Data from a co-conscious alter (reported, may differ)
    CoConscious(Option<Ident>),
    /// Data from a dormant alter (uncertain access)
    Dormant(Option<Ident>),
    /// Data from multiple alters (potentially contradictory)
    Blended(Vec<Ident>),
    /// Data from a specific named alter
    Named(Ident),
    /// Data from any alter matching a trait bound
    Bound(AlterBound),
}

/// Trait bound for alter-generic code
#[derive(Debug, Clone, PartialEq)]
pub struct AlterBound {
    /// The type parameter name (e.g., `A` in `fn foo<A: Alter>`)
    pub name: Ident,
    /// Required traits (e.g., `Alter`, `Council`, `Combat`)
    pub bounds: Vec<Ident>,
}

// ============================================================================
// ALTER DEFINITION
// ============================================================================

/// An alter definition - a first-class identity within the System.
///
/// ```sigil
/// alter Abaddon: Council {
///     archetype: Goetia::Abaddon,
///     preferred_reality: RealityLayer::Fractured,
///     abilities: [...],
///     triggers: [...],
///     anima: { ... },
///     states: { ... }
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct AlterDef {
    /// Visibility (pub, crate, etc.)
    pub visibility: Visibility,
    /// Attributes (#[derive(...)])
    pub attrs: Vec<Attribute>,
    /// Alter name
    pub name: Ident,
    /// Alter category (Council, Servant, Fragment, etc.)
    pub category: AlterCategory,
    /// Generic parameters
    pub generics: Option<Generics>,
    /// Where clause
    pub where_clause: Option<WhereClause>,
    /// Alter body
    pub body: AlterBody,
    /// Source span
    pub span: Span,
}

/// Category of alter (determines capabilities)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlterCategory {
    /// Can front and take control
    Council,
    /// Maintains internal functions, cannot front
    Servant,
    /// Incomplete alter, may evolve
    Fragment,
    /// Hidden until discovered
    Hidden,
    /// Hostile or exiled alter
    Persecutor,
    /// Custom category
    Custom,
}

/// Body of an alter definition
#[derive(Debug, Clone, PartialEq)]
pub struct AlterBody {
    /// Archetype (Goetia demon, mythological figure, etc.)
    pub archetype: Option<Expr>,
    /// Preferred reality layer
    pub preferred_reality: Option<Expr>,
    /// Available abilities
    pub abilities: Vec<Expr>,
    /// Trigger conditions
    pub triggers: Vec<Expr>,
    /// Anima configuration (base emotional state)
    pub anima: Option<AnimaConfig>,
    /// State machine
    pub states: Option<AlterStateMachine>,
    /// Special capabilities
    pub special: Vec<SpecialCapability>,
    /// Methods and associated functions
    pub methods: Vec<AlterMethod>,
    /// Associated types
    pub types: Vec<AlterTypeAlias>,
}

/// Anima configuration for an alter
#[derive(Debug, Clone, PartialEq)]
pub struct AnimaConfig {
    /// Base arousal level (0.0 - 1.0)
    pub base_arousal: Option<Expr>,
    /// Base dominance level (0.0 - 1.0)
    pub base_dominance: Option<Expr>,
    /// How much this alter broadcasts emotions
    pub expressiveness: Option<Expr>,
    /// How much this alter picks up others' emotions
    pub susceptibility: Option<Expr>,
    /// Additional anima fields
    pub extra: Vec<(Ident, Expr)>,
}

/// State machine for alter lifecycle
#[derive(Debug, Clone, PartialEq)]
pub struct AlterStateMachine {
    /// State transitions
    pub transitions: Vec<AlterTransition>,
}

/// A state transition in the alter lifecycle
#[derive(Debug, Clone, PartialEq)]
pub struct AlterTransition {
    /// Source state
    pub from: AlterState,
    /// Target state
    pub to: AlterState,
    /// Trigger condition
    pub on: Expr,
    /// Guard condition (optional)
    pub guard: Option<Expr>,
    /// Action to perform (optional)
    pub action: Option<Block>,
}

/// Possible states for an alter
#[derive(Debug, Clone, PartialEq)]
pub enum AlterState {
    Dormant,
    Stirring,
    CoConscious,
    Emerging,
    Fronting,
    Receding,
    Triggered,
    Dissociating,
    /// Custom state
    Custom(Ident),
}

/// Special capability unique to an alter
#[derive(Debug, Clone, PartialEq)]
pub struct SpecialCapability {
    pub name: Ident,
    pub params: Vec<(Ident, Expr)>,
}

/// Method defined on an alter
#[derive(Debug, Clone, PartialEq)]
pub struct AlterMethod {
    pub visibility: Visibility,
    pub is_async: bool,
    pub name: Ident,
    pub params: Vec<Param>,
    pub return_type: Option<TypeExpr>,
    pub body: Option<Block>,
}

/// Type alias for an alter
#[derive(Debug, Clone, PartialEq)]
pub struct AlterTypeAlias {
    pub visibility: Visibility,
    pub name: Ident,
    pub ty: TypeExpr,
}

// ============================================================================
// ALTER BLOCK
// ============================================================================

/// An alter block - scoped fronting for a code section.
///
/// ```sigil
/// alter Abaddon {
///     // Inside here, 'self' refers to Abaddon
///     // Perception uses Abaddon's view
///     let threat = perceive_entity(enemy);
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct AlterBlock {
    /// Which alter is fronting for this block
    pub alter: AlterExpr,
    /// The code to execute as that alter
    pub body: Block,
    /// Source span
    pub span: Span,
}

/// Expression that resolves to an alter
#[derive(Debug, Clone, PartialEq)]
pub enum AlterExpr {
    /// Named alter: `alter Abaddon { ... }`
    Named(Ident),
    /// Current fronter: `alter council·fronter() { ... }`
    CurrentFronter(Box<Expr>),
    /// Expression that evaluates to an alter
    Expr(Box<Expr>),
}

// ============================================================================
// SWITCH EXPRESSION
// ============================================================================

/// A switch expression - deliberated or forced switching.
///
/// ```sigil
/// let result = switch to Beleth {
///     reason: SwitchReason::TacticalNeed,
///     urgency: 0.8,
///     requires: Consensus::Majority,
///     then: { ... },
///     else: { ... },
///     emergency: { force_switch() }
/// };
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SwitchExpr {
    /// Is this a forced switch (switch!)
    pub forced: bool,
    /// Target alter
    pub target: AlterExpr,
    /// Switch configuration
    pub config: SwitchConfig,
    /// Source span
    pub span: Span,
}

/// Configuration for a switch
#[derive(Debug, Clone, PartialEq)]
pub struct SwitchConfig {
    /// Reason for the switch
    pub reason: Option<Expr>,
    /// Urgency level (0.0 - 1.0)
    pub urgency: Option<Expr>,
    /// Required consensus level
    pub requires: Option<Expr>,
    /// Block to execute on success
    pub then_block: Option<Block>,
    /// Block to execute on failure
    pub else_block: Option<Block>,
    /// Emergency override block
    pub emergency_block: Option<Block>,
    /// Whether to bypass deliberation
    pub bypass_deliberation: bool,
}

// ============================================================================
// CO-CONSCIOUS CHANNEL
// ============================================================================

/// A co-conscious communication channel.
///
/// ```sigil
/// cocon<Stolas, Paimon> knowledge_share {
///     fn share_discovery(info: Knowledge!) -> Acknowledgment~ {
///         Paimon.receive(info~)
///     }
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct CoConChannel {
    /// Participating alters
    pub participants: Vec<Ident>,
    /// Channel name
    pub name: Ident,
    /// Channel body
    pub body: Block,
    /// Source span
    pub span: Span,
}

// ============================================================================
// REALITY LAYER DEFINITION
// ============================================================================

/// A reality layer definition for superimposed entities.
///
/// ```sigil
/// reality entity Church {
///     layer Grounded {
///         name: "First Ward Chapel",
///         threat: 0.2,
///     }
///     layer Fractured {
///         name: "The White Throne's Outpost",
///         threat: 0.8,
///     }
///     transform Grounded -> Fractured: on perception > 0.5,
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RealityDef {
    /// Visibility
    pub visibility: Visibility,
    /// Entity name
    pub name: Ident,
    /// Layer definitions
    pub layers: Vec<RealityLayer>,
    /// Transform rules
    pub transforms: Vec<RealityTransform>,
    /// Source span
    pub span: Span,
}

/// A single reality layer
#[derive(Debug, Clone, PartialEq)]
pub struct RealityLayer {
    /// Layer name (Grounded, Fractured, Psychological, Cosmic)
    pub name: Ident,
    /// Layer fields
    pub fields: Vec<(Ident, Expr)>,
}

/// A transform rule between reality layers
#[derive(Debug, Clone, PartialEq)]
pub struct RealityTransform {
    /// Source layer
    pub from: Ident,
    /// Target layer
    pub to: Ident,
    /// Condition for transform
    pub condition: Expr,
}

// ============================================================================
// HEADSPACE DEFINITION
// ============================================================================

/// A headspace (Inner World) definition.
///
/// ```sigil
/// headspace InnerWorld {
///     location Citadel: Sanctuary {
///         biome: Biome::Citadel,
///         connections: [...]
///     }
///
///     fn navigate(from: LocationId, to: LocationId) -> NavigationResult@? { ... }
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct HeadspaceDef {
    /// Visibility
    pub visibility: Visibility,
    /// Headspace name
    pub name: Ident,
    /// Location definitions
    pub locations: Vec<LocationDef>,
    /// Methods
    pub methods: Vec<AlterMethod>,
    /// Source span
    pub span: Span,
}

/// A location in the headspace
#[derive(Debug, Clone, PartialEq)]
pub struct LocationDef {
    /// Location name
    pub name: Ident,
    /// Location type (Sanctuary, PersonalDomain, etc.)
    pub location_type: Ident,
    /// Location fields
    pub fields: Vec<(Ident, Expr)>,
    /// Connections (stream definitions)
    pub connections: Vec<StreamDef>,
    /// Hazards
    pub hazards: Vec<Expr>,
}

/// A consciousness stream connection
#[derive(Debug, Clone, PartialEq)]
pub struct StreamDef {
    /// Target location
    pub target: Ident,
    /// Stream content type
    pub content: Option<Expr>,
    /// Is bidirectional
    pub bidirectional: bool,
    /// Is locked
    pub locked: bool,
}

// ============================================================================
// SPLIT EXPRESSION
// ============================================================================

/// A split expression - trauma-based alter creation.
///
/// ```sigil
/// let new_alter = split! from trauma.primary_holder {
///     purpose: SplitPurpose::TraumaHolder,
///     memories: inherited_memories(),
///     traits: possibly_inverted_traits(),
/// };
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SplitExpr {
    /// Parent alter to split from
    pub parent: AlterExpr,
    /// Split configuration
    pub config: SplitConfig,
    /// Source span
    pub span: Span,
}

/// Configuration for a split
#[derive(Debug, Clone, PartialEq)]
pub struct SplitConfig {
    /// Purpose of the new alter
    pub purpose: Option<Expr>,
    /// Memories to inherit
    pub memories: Option<Expr>,
    /// Traits to inherit (may be inverted)
    pub traits: Option<Expr>,
    /// Additional fields
    pub extra: Vec<(Ident, Expr)>,
}

// ============================================================================
// TYPE EXTENSIONS
// ============================================================================

/// Extended type expression with alter-source
#[derive(Debug, Clone, PartialEq)]
pub struct AlterSourcedType {
    /// Base type
    pub inner: TypeExpr,
    /// Alter source
    pub alter_source: AlterSource,
    /// Source span
    pub span: Span,
}

// ============================================================================
// TRIGGER DEFINITION
// ============================================================================

/// A trigger handler definition.
///
/// ```sigil
/// on trigger ThreatDetected { level: threat } where threat > 0.9 {
///     switch! to Abaddon {
///         reason: SwitchReason::Emergency,
///         bypass_deliberation: true,
///     }
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct TriggerHandler {
    /// Trigger pattern to match
    pub pattern: TriggerPattern,
    /// Guard condition
    pub guard: Option<Expr>,
    /// Handler body
    pub body: Block,
    /// Source span
    pub span: Span,
}

/// Pattern for matching triggers
#[derive(Debug, Clone, PartialEq)]
pub struct TriggerPattern {
    /// Trigger type name
    pub trigger_type: Ident,
    /// Destructured fields
    pub fields: Vec<(Ident, Ident)>,
}

// ============================================================================
// PLURALITY ITEM (TOP-LEVEL)
// ============================================================================

/// Top-level plurality items
#[derive(Debug, Clone, PartialEq)]
pub enum PluralityItem {
    /// Alter definition
    Alter(AlterDef),
    /// Headspace definition
    Headspace(HeadspaceDef),
    /// Reality entity definition
    Reality(RealityDef),
    /// Co-conscious channel
    CoConChannel(CoConChannel),
    /// Trigger handler
    TriggerHandler(TriggerHandler),
}

// ============================================================================
// PLURALITY EXPRESSION
// ============================================================================

/// Plurality-specific expressions
#[derive(Debug, Clone, PartialEq)]
pub enum PluralityExpr {
    /// Alter block
    AlterBlock(AlterBlock),
    /// Switch expression
    Switch(SwitchExpr),
    /// Split expression
    Split(SplitExpr),
    /// Alter-sourced value access
    AlterSourced {
        expr: Box<Expr>,
        source: AlterSource,
        span: Span,
    },
}

// ============================================================================
// CONVERSION TRAITS
// ============================================================================

impl From<AlterDef> for PluralityItem {
    fn from(def: AlterDef) -> Self {
        PluralityItem::Alter(def)
    }
}

impl From<HeadspaceDef> for PluralityItem {
    fn from(def: HeadspaceDef) -> Self {
        PluralityItem::Headspace(def)
    }
}

impl From<RealityDef> for PluralityItem {
    fn from(def: RealityDef) -> Self {
        PluralityItem::Reality(def)
    }
}

impl AlterState {
    /// Parse from identifier
    pub fn from_ident(ident: &Ident) -> Self {
        match ident.name.as_str() {
            "Dormant" => AlterState::Dormant,
            "Stirring" => AlterState::Stirring,
            "CoConscious" => AlterState::CoConscious,
            "Emerging" => AlterState::Emerging,
            "Fronting" => AlterState::Fronting,
            "Receding" => AlterState::Receding,
            "Triggered" => AlterState::Triggered,
            "Dissociating" => AlterState::Dissociating,
            _ => AlterState::Custom(ident.clone()),
        }
    }
}

impl AlterCategory {
    /// Parse from identifier
    pub fn from_ident(ident: &Ident) -> Self {
        match ident.name.as_str() {
            "Council" => AlterCategory::Council,
            "Servant" => AlterCategory::Servant,
            "Fragment" => AlterCategory::Fragment,
            "Hidden" => AlterCategory::Hidden,
            "Persecutor" => AlterCategory::Persecutor,
            _ => AlterCategory::Custom,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alter_state_parsing() {
        let ident = Ident {
            name: "Fronting".to_string(),
            evidentiality: None,
            affect: None,
            span: Span::default(),
        };
        assert_eq!(AlterState::from_ident(&ident), AlterState::Fronting);
    }

    #[test]
    fn test_alter_category_parsing() {
        let ident = Ident {
            name: "Council".to_string(),
            evidentiality: None,
            affect: None,
            span: Span::default(),
        };
        assert_eq!(AlterCategory::from_ident(&ident), AlterCategory::Council);
    }
}
