//! # Plurality Type Checking
//!
//! Type system extensions for plurality constructs. Extends Sigil's evidentiality
//! type system to track alter-source information.
//!
//! ## Core Type Extensions
//!
//! Sigil's evidentiality markers (`!`, `~`, `?`, `‽`) are extended with alter-source:
//!
//! | Marker | Base Meaning | Alter-Source Extension |
//! |--------|--------------|------------------------|
//! | `@!`   | Certain      | Fronting alter's authoritative view |
//! | `@~`   | Reported     | Co-conscious alter's shared perception |
//! | `@?`   | Uncertain    | Dormant alter's cached memory |
//! | `@‽`   | Paradox      | Blended state from multiple alters |
//!
//! ## Type Compatibility Rules
//!
//! 1. `T@!` (fronting) is always compatible with `T`
//! 2. `T@~` (co-con) can be assigned to `T~` or `T?`
//! 3. `T@?` (dormant) can only be assigned to `T?`
//! 4. `T@‽` (blended) requires explicit resolution before use
//!
//! ## Alter-Polymorphism
//!
//! Functions can be polymorphic over fronting alter:
//! ```sigil
//! fn perceive<A: Alter>(entity: &Entity) -> Perception@A
//! ```

use std::collections::HashMap;
use std::fmt;

use super::ast::{AlterCategory, AlterDef, AlterSource, AlterState};
use crate::ast::{Evidentiality, Ident, TypeExpr};
use crate::span::Span;

// ============================================================================
// PLURALITY TYPE SYSTEM
// ============================================================================

/// Extended type information for plurality
#[derive(Debug, Clone, PartialEq)]
pub struct PluralType {
    /// The base type expression
    pub base: TypeExpr,
    /// Base evidentiality (!~?‽)
    pub evidentiality: Option<Evidentiality>,
    /// Alter-source information
    pub alter_source: Option<AlterSource>,
    /// Is this type inside an alter block?
    pub in_alter_context: bool,
    /// The span for error reporting
    pub span: Span,
}

impl PluralType {
    /// Create a new plural type from a base type
    pub fn from_type(base: TypeExpr, span: Span) -> Self {
        Self {
            base,
            evidentiality: None,
            alter_source: None,
            in_alter_context: false,
            span,
        }
    }

    /// Add alter-source information
    pub fn with_alter_source(mut self, source: AlterSource) -> Self {
        self.alter_source = Some(source);
        self
    }

    /// Add evidentiality marker
    pub fn with_evidentiality(mut self, ev: Evidentiality) -> Self {
        self.evidentiality = Some(ev);
        self
    }

    /// Check if this type is in a fronting context
    pub fn is_fronting(&self) -> bool {
        matches!(self.alter_source, Some(AlterSource::Fronting))
    }

    /// Check if this type is from co-conscious context
    pub fn is_cocon(&self) -> bool {
        matches!(self.alter_source, Some(AlterSource::CoConscious(_)))
    }

    /// Check if this type is from dormant context
    pub fn is_dormant(&self) -> bool {
        matches!(self.alter_source, Some(AlterSource::Dormant(_)))
    }

    /// Check if this type is in blended state
    pub fn is_blended(&self) -> bool {
        matches!(self.alter_source, Some(AlterSource::Blended(_)))
    }
}

// ============================================================================
// ALTER TYPE CONTEXT
// ============================================================================

/// Context for type checking within alter blocks
#[derive(Debug, Clone)]
pub struct AlterContext {
    /// Currently fronting alter (if in alter block)
    pub current_alter: Option<Ident>,
    /// Stack of alter blocks (for nested contexts)
    pub alter_stack: Vec<Ident>,
    /// Known alter definitions
    pub alter_defs: HashMap<String, AlterDef>,
    /// Current alter's category
    pub current_category: Option<AlterCategory>,
    /// Current alter's state
    pub current_state: Option<AlterState>,
}

impl Default for AlterContext {
    fn default() -> Self {
        Self {
            current_alter: None,
            alter_stack: Vec::new(),
            alter_defs: HashMap::new(),
            current_category: None,
            current_state: None,
        }
    }
}

impl AlterContext {
    /// Create a new alter context
    pub fn new() -> Self {
        Self::default()
    }

    /// Register an alter definition
    pub fn register_alter(&mut self, def: AlterDef) {
        self.alter_defs.insert(def.name.name.clone(), def);
    }

    /// Enter an alter block context
    pub fn enter_alter(&mut self, alter: Ident) {
        if let Some(current) = &self.current_alter {
            self.alter_stack.push(current.clone());
        }
        self.current_alter = Some(alter.clone());

        // Update category from definition if known
        if let Some(def) = self.alter_defs.get(&alter.name) {
            self.current_category = Some(def.category);
        }
    }

    /// Exit current alter block
    pub fn exit_alter(&mut self) {
        self.current_alter = self.alter_stack.pop();
        self.current_category = self.current_alter.as_ref().and_then(|a| {
            self.alter_defs
                .get(&a.name)
                .map(|def| def.category)
        });
    }

    /// Check if we're currently in an alter block
    pub fn in_alter_block(&self) -> bool {
        self.current_alter.is_some()
    }

    /// Get the current fronting alter's name
    pub fn fronter_name(&self) -> Option<&str> {
        self.current_alter.as_ref().map(|a| a.name.as_str())
    }

    /// Check if an alter can front based on category
    pub fn can_front(&self, alter_name: &str) -> bool {
        if let Some(def) = self.alter_defs.get(alter_name) {
            matches!(def.category, AlterCategory::Council)
        } else {
            true // Unknown alters assumed to be able to front
        }
    }

    /// Check if a switch to an alter is valid
    pub fn validate_switch(&self, target: &str) -> SwitchValidation {
        if let Some(def) = self.alter_defs.get(target) {
            match def.category {
                AlterCategory::Council => SwitchValidation::Allowed,
                AlterCategory::Servant => SwitchValidation::Denied {
                    reason: "Servants cannot front".to_string(),
                },
                AlterCategory::Fragment => SwitchValidation::Warning {
                    reason: "Fragments are unstable and may not maintain fronting".to_string(),
                },
                AlterCategory::Hidden => SwitchValidation::Warning {
                    reason: "Hidden alters may not respond to switch requests".to_string(),
                },
                AlterCategory::Persecutor => SwitchValidation::Warning {
                    reason: "Switching to a Persecutor may be destabilizing".to_string(),
                },
                AlterCategory::Custom => SwitchValidation::Allowed,
            }
        } else {
            SwitchValidation::Unknown {
                alter: target.to_string(),
            }
        }
    }
}

/// Result of validating a switch
#[derive(Debug, Clone, PartialEq)]
pub enum SwitchValidation {
    Allowed,
    Warning { reason: String },
    Denied { reason: String },
    Unknown { alter: String },
}

// ============================================================================
// TYPE COMPATIBILITY
// ============================================================================

/// Rules for alter-source type compatibility
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlterSourceCompatibility {
    /// Types are compatible
    Compatible,
    /// Types are compatible with implicit coercion
    CoercibleWith(AlterSourceCoercion),
    /// Types are incompatible
    Incompatible,
    /// Requires explicit resolution
    RequiresResolution,
}

/// Coercion type for alter-source
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AlterSourceCoercion {
    /// Upgrade certainty (e.g., @~ -> @!)
    UpgradeCertainty,
    /// Downgrade certainty (e.g., @! -> @~)
    DowngradeCertainty,
    /// Transfer between alters
    TransferAlter,
    /// Blend multiple sources
    Blend,
}

/// Check compatibility between two plural types
pub fn check_compatibility(from: &PluralType, to: &PluralType) -> AlterSourceCompatibility {
    // Same source is always compatible
    if from.alter_source == to.alter_source {
        return AlterSourceCompatibility::Compatible;
    }

    match (&from.alter_source, &to.alter_source) {
        // Fronting is compatible with anything (authoritative source)
        (Some(AlterSource::Fronting), _) => AlterSourceCompatibility::Compatible,

        // Co-con to fronting requires upgrade
        (Some(AlterSource::CoConscious(_)), Some(AlterSource::Fronting)) => {
            AlterSourceCompatibility::CoercibleWith(AlterSourceCoercion::UpgradeCertainty)
        }

        // Co-con to co-con is compatible
        (Some(AlterSource::CoConscious(_)), Some(AlterSource::CoConscious(_))) => {
            AlterSourceCompatibility::Compatible
        }

        // Co-con to dormant is a downgrade
        (Some(AlterSource::CoConscious(_)), Some(AlterSource::Dormant(_))) => {
            AlterSourceCompatibility::CoercibleWith(AlterSourceCoercion::DowngradeCertainty)
        }

        // Dormant can only be assigned to dormant or weaker
        (Some(AlterSource::Dormant(_)), Some(AlterSource::Dormant(_))) => {
            AlterSourceCompatibility::Compatible
        }

        // Dormant to anything else is not allowed without explicit verification
        (Some(AlterSource::Dormant(_)), _) => AlterSourceCompatibility::Incompatible,

        // Blended state requires explicit resolution
        (Some(AlterSource::Blended(_)), _) | (_, Some(AlterSource::Blended(_))) => {
            AlterSourceCompatibility::RequiresResolution
        }

        // Named alter to fronting needs verification
        (Some(AlterSource::Named(_)), Some(AlterSource::Fronting)) => {
            AlterSourceCompatibility::CoercibleWith(AlterSourceCoercion::TransferAlter)
        }

        // Bound alters need trait checking
        (Some(AlterSource::Bound(_)), _) => {
            // This would involve trait resolution
            AlterSourceCompatibility::Compatible
        }

        // No alter source means default fronting context
        (None, _) | (_, None) => AlterSourceCompatibility::Compatible,

        // Default case
        _ => AlterSourceCompatibility::Incompatible,
    }
}

// ============================================================================
// TYPE ERRORS
// ============================================================================

/// Plurality-specific type errors
#[derive(Debug, Clone)]
pub enum PluralityTypeError {
    /// Attempted to access dormant data without verification
    DormantAccessWithoutVerification {
        alter: String,
        data_type: String,
        span: Span,
    },

    /// Blended state requires resolution
    UnresolvedBlendedState {
        alters: Vec<String>,
        data_type: String,
        span: Span,
    },

    /// Servant alter cannot front
    ServantCannotFront {
        alter: String,
        span: Span,
    },

    /// Alter not found in system
    AlterNotFound {
        name: String,
        span: Span,
    },

    /// Switch to alter not allowed
    SwitchDenied {
        target: String,
        reason: String,
        span: Span,
    },

    /// Alter-source mismatch
    AlterSourceMismatch {
        expected: String,
        found: String,
        span: Span,
    },

    /// Co-con channel participants not co-conscious
    CoCOnNotActive {
        alter: String,
        span: Span,
    },

    /// Reality layer mismatch
    RealityLayerMismatch {
        expected: String,
        found: String,
        span: Span,
    },

    /// Cannot split from non-existent alter
    SplitSourceNotFound {
        alter: String,
        span: Span,
    },

    /// Trigger handler without matching trigger
    TriggerNotDefined {
        trigger: String,
        span: Span,
    },
}

impl fmt::Display for PluralityTypeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PluralityTypeError::DormantAccessWithoutVerification { alter, data_type, .. } => {
                write!(
                    f,
                    "Cannot access dormant alter '{}' data of type '{}' without verification. \
                     Use `verify_access` or add `@?` uncertainty marker.",
                    alter, data_type
                )
            }
            PluralityTypeError::UnresolvedBlendedState { alters, data_type, .. } => {
                write!(
                    f,
                    "Blended state from alters [{}] for type '{}' must be resolved. \
                     Use `resolve_blend` or explicit alter selection.",
                    alters.join(", "),
                    data_type
                )
            }
            PluralityTypeError::ServantCannotFront { alter, .. } => {
                write!(
                    f,
                    "Alter '{}' is a Servant and cannot front. \
                     Servants maintain internal functions only.",
                    alter
                )
            }
            PluralityTypeError::AlterNotFound { name, .. } => {
                write!(f, "Alter '{}' not found in system definition.", name)
            }
            PluralityTypeError::SwitchDenied { target, reason, .. } => {
                write!(f, "Switch to '{}' denied: {}", target, reason)
            }
            PluralityTypeError::AlterSourceMismatch { expected, found, .. } => {
                write!(
                    f,
                    "Alter-source mismatch: expected '{}', found '{}'",
                    expected, found
                )
            }
            PluralityTypeError::CoCOnNotActive { alter, .. } => {
                write!(
                    f,
                    "Alter '{}' is not co-conscious and cannot participate in channel.",
                    alter
                )
            }
            PluralityTypeError::RealityLayerMismatch { expected, found, .. } => {
                write!(
                    f,
                    "Reality layer mismatch: expected '{}', found '{}'",
                    expected, found
                )
            }
            PluralityTypeError::SplitSourceNotFound { alter, .. } => {
                write!(f, "Cannot split from '{}': alter not found.", alter)
            }
            PluralityTypeError::TriggerNotDefined { trigger, .. } => {
                write!(f, "Trigger '{}' is not defined in system.", trigger)
            }
        }
    }
}

impl std::error::Error for PluralityTypeError {}

// ============================================================================
// TYPE CHECKER EXTENSION
// ============================================================================

/// Plurality type checker
#[derive(Debug)]
pub struct PluralityTypeChecker {
    /// Alter context
    context: AlterContext,
    /// Collected errors
    errors: Vec<PluralityTypeError>,
    /// Collected warnings
    warnings: Vec<String>,
}

impl Default for PluralityTypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl PluralityTypeChecker {
    /// Create a new type checker
    pub fn new() -> Self {
        Self {
            context: AlterContext::new(),
            errors: Vec::new(),
            warnings: Vec::new(),
        }
    }

    /// Register an alter definition
    pub fn register_alter(&mut self, def: AlterDef) {
        self.context.register_alter(def);
    }

    /// Enter an alter block
    pub fn enter_alter_block(&mut self, alter: &Ident) {
        self.context.enter_alter(alter.clone());
    }

    /// Exit alter block
    pub fn exit_alter_block(&mut self) {
        self.context.exit_alter();
    }

    /// Check if we're in an alter block
    pub fn in_alter_block(&self) -> bool {
        self.context.in_alter_block()
    }

    /// Validate a switch expression
    pub fn validate_switch(&mut self, target: &str, span: Span) -> bool {
        match self.context.validate_switch(target) {
            SwitchValidation::Allowed => true,
            SwitchValidation::Warning { reason } => {
                self.warnings.push(format!("Switch to '{}': {}", target, reason));
                true
            }
            SwitchValidation::Denied { reason } => {
                self.errors.push(PluralityTypeError::SwitchDenied {
                    target: target.to_string(),
                    reason,
                    span,
                });
                false
            }
            SwitchValidation::Unknown { alter } => {
                self.warnings.push(format!(
                    "Switch to unknown alter '{}'. Consider defining it.",
                    alter
                ));
                true
            }
        }
    }

    /// Check type compatibility
    pub fn check_assignment(
        &mut self,
        from: &PluralType,
        to: &PluralType,
        span: Span,
    ) -> bool {
        match check_compatibility(from, to) {
            AlterSourceCompatibility::Compatible => true,
            AlterSourceCompatibility::CoercibleWith(coercion) => {
                match coercion {
                    AlterSourceCoercion::UpgradeCertainty => {
                        self.warnings.push(format!(
                            "Implicit certainty upgrade from co-conscious to fronting at {:?}",
                            span
                        ));
                    }
                    AlterSourceCoercion::DowngradeCertainty => {
                        self.warnings.push(format!(
                            "Certainty downgrade - data may be less reliable at {:?}",
                            span
                        ));
                    }
                    AlterSourceCoercion::TransferAlter => {
                        self.warnings.push(format!(
                            "Cross-alter data transfer at {:?}",
                            span
                        ));
                    }
                    AlterSourceCoercion::Blend => {
                        self.warnings.push(format!(
                            "Blending data from multiple alters at {:?}",
                            span
                        ));
                    }
                }
                true
            }
            AlterSourceCompatibility::Incompatible => {
                let from_str = format!("{:?}", from.alter_source);
                let to_str = format!("{:?}", to.alter_source);
                self.errors.push(PluralityTypeError::AlterSourceMismatch {
                    expected: to_str,
                    found: from_str,
                    span,
                });
                false
            }
            AlterSourceCompatibility::RequiresResolution => {
                if let Some(AlterSource::Blended(alters)) = &from.alter_source {
                    self.errors.push(PluralityTypeError::UnresolvedBlendedState {
                        alters: alters.iter().map(|a| a.name.clone()).collect(),
                        data_type: format!("{:?}", from.base),
                        span,
                    });
                }
                false
            }
        }
    }

    /// Check dormant access
    pub fn check_dormant_access(&mut self, alter: &str, data_type: &str, span: Span) -> bool {
        // In a fronting context, dormant access requires explicit verification
        if self.context.in_alter_block() {
            if let Some(fronter) = self.context.fronter_name() {
                if fronter != alter {
                    self.errors.push(PluralityTypeError::DormantAccessWithoutVerification {
                        alter: alter.to_string(),
                        data_type: data_type.to_string(),
                        span,
                    });
                    return false;
                }
            }
        }
        true
    }

    /// Get collected errors
    pub fn errors(&self) -> &[PluralityTypeError] {
        &self.errors
    }

    /// Get collected warnings
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Check if there are any errors
    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    /// Clear errors and warnings
    pub fn clear(&mut self) {
        self.errors.clear();
        self.warnings.clear();
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::TypeExpr;

    fn mock_alter_def(name: &str, category: AlterCategory) -> AlterDef {
        AlterDef {
            visibility: crate::ast::Visibility::Public,
            attrs: Vec::new(),
            name: Ident {
                name: name.to_string(),
                evidentiality: None,
                affect: None,
                span: Span::default(),
            },
            category,
            generics: None,
            where_clause: None,
            body: super::super::ast::AlterBody {
                archetype: None,
                preferred_reality: None,
                abilities: Vec::new(),
                triggers: Vec::new(),
                anima: None,
                states: None,
                special: Vec::new(),
                methods: Vec::new(),
                types: Vec::new(),
            },
            span: Span::default(),
        }
    }

    #[test]
    fn test_alter_context_registration() {
        let mut ctx = AlterContext::new();
        let def = mock_alter_def("Abaddon", AlterCategory::Council);
        ctx.register_alter(def);

        assert!(ctx.can_front("Abaddon"));
    }

    #[test]
    fn test_servant_cannot_front() {
        let mut ctx = AlterContext::new();
        let def = mock_alter_def("Watcher", AlterCategory::Servant);
        ctx.register_alter(def);

        assert!(!ctx.can_front("Watcher"));
    }

    #[test]
    fn test_switch_validation() {
        let mut ctx = AlterContext::new();
        ctx.register_alter(mock_alter_def("Abaddon", AlterCategory::Council));
        ctx.register_alter(mock_alter_def("Watcher", AlterCategory::Servant));

        assert_eq!(ctx.validate_switch("Abaddon"), SwitchValidation::Allowed);
        assert!(matches!(
            ctx.validate_switch("Watcher"),
            SwitchValidation::Denied { .. }
        ));
    }

    #[test]
    fn test_alter_source_compatibility() {
        let fronting_type = PluralType {
            base: TypeExpr::Path(crate::ast::TypePath {
                segments: vec![],
            }),
            evidentiality: None,
            alter_source: Some(AlterSource::Fronting),
            in_alter_context: true,
            span: Span::default(),
        };

        let cocon_type = PluralType {
            base: TypeExpr::Path(crate::ast::TypePath {
                segments: vec![],
            }),
            evidentiality: None,
            alter_source: Some(AlterSource::CoConscious(None)),
            in_alter_context: true,
            span: Span::default(),
        };

        // Fronting to anything is compatible
        assert_eq!(
            check_compatibility(&fronting_type, &cocon_type),
            AlterSourceCompatibility::Compatible
        );

        // Co-con to fronting needs upgrade
        assert!(matches!(
            check_compatibility(&cocon_type, &fronting_type),
            AlterSourceCompatibility::CoercibleWith(AlterSourceCoercion::UpgradeCertainty)
        ));
    }

    #[test]
    fn test_blended_requires_resolution() {
        let blended = PluralType {
            base: TypeExpr::Path(crate::ast::TypePath {
                segments: vec![],
            }),
            evidentiality: None,
            alter_source: Some(AlterSource::Blended(Vec::new())),
            in_alter_context: true,
            span: Span::default(),
        };

        let fronting = PluralType {
            base: TypeExpr::Path(crate::ast::TypePath {
                segments: vec![],
            }),
            evidentiality: None,
            alter_source: Some(AlterSource::Fronting),
            in_alter_context: true,
            span: Span::default(),
        };

        assert_eq!(
            check_compatibility(&blended, &fronting),
            AlterSourceCompatibility::RequiresResolution
        );
    }
}
