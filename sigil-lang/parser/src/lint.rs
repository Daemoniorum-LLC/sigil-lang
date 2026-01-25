//! Linter for Sigil source code.
//!
//! Provides static analysis to catch common mistakes, style issues,
//! and Sigil-specific patterns that may cause problems.
//!
//! # Configuration
//!
//! The linter can be configured via `.sigillint.toml`:
//!
//! ```toml
//! [lint]
//! suggest_unicode = true
//! check_naming = true
//! max_nesting_depth = 6
//!
//! [lint.levels]
//! unused_variable = "allow"    # allow, warn, or deny
//! shadowing = "warn"
//! deep_nesting = "deny"
//! ```

use crate::ast::*;
use crate::diagnostic::{Diagnostic, Diagnostics, FixSuggestion, Severity};
use crate::parser::ParseError;
use crate::span::Span;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

// ============================================
// Lint Configuration
// ============================================

/// TOML-serializable configuration for the linter.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LintConfigFile {
    /// Lint settings
    pub lint: LintSettings,
}

/// Lint settings section of config file.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LintSettings {
    /// Whether to suggest Unicode morphemes
    pub suggest_unicode: bool,
    /// Whether to check naming conventions
    pub check_naming: bool,
    /// Maximum nesting depth before warning
    pub max_nesting_depth: usize,
    /// Lint level overrides by lint name
    pub levels: HashMap<String, String>,
}

impl Default for LintSettings {
    fn default() -> Self {
        Self {
            suggest_unicode: true,
            check_naming: true,
            max_nesting_depth: 6,
            levels: HashMap::new(),
        }
    }
}

impl Default for LintConfigFile {
    fn default() -> Self {
        Self {
            lint: LintSettings::default(),
        }
    }
}

/// Runtime configuration for the linter.
#[derive(Debug, Clone)]
pub struct LintConfig {
    /// Lint level overrides by lint ID
    pub levels: HashMap<String, LintLevel>,
    /// Whether to suggest Unicode morphemes
    pub suggest_unicode: bool,
    /// Whether to check naming conventions
    pub check_naming: bool,
    /// Reserved identifiers to warn about
    pub reserved_words: HashSet<String>,
    /// Maximum nesting depth before warning
    pub max_nesting_depth: usize,
}

impl Default for LintConfig {
    fn default() -> Self {
        let mut reserved = HashSet::new();
        for word in &[
            "from", "split", "ref", "location", "save", "type", "move", "match",
            "loop", "if", "else", "while", "for", "in", "return", "break",
            "continue", "fn", "let", "mut", "const", "static", "struct", "enum",
            "trait", "impl", "pub", "mod", "use", "as", "where", "async", "await",
            "dyn", "unsafe", "extern", "crate", "self", "super", "true", "false",
        ] {
            reserved.insert(word.to_string());
        }

        Self {
            levels: HashMap::new(),
            suggest_unicode: true,
            check_naming: true,
            reserved_words: reserved,
            max_nesting_depth: 6,
        }
    }
}

impl LintConfig {
    /// Load configuration from a TOML file.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config file: {}", e))?;
        Self::from_toml(&content)
    }

    /// Parse configuration from TOML string.
    pub fn from_toml(content: &str) -> Result<Self, String> {
        let file: LintConfigFile = toml::from_str(content)
            .map_err(|e| format!("Failed to parse config: {}", e))?;

        let mut config = Self::default();
        config.suggest_unicode = file.lint.suggest_unicode;
        config.check_naming = file.lint.check_naming;
        config.max_nesting_depth = file.lint.max_nesting_depth;

        // Convert string levels to LintLevel
        for (name, level_str) in file.lint.levels {
            let level = match level_str.to_lowercase().as_str() {
                "allow" => LintLevel::Allow,
                "warn" => LintLevel::Warn,
                "deny" => LintLevel::Deny,
                _ => return Err(format!("Invalid lint level '{}' for '{}'", level_str, name)),
            };
            config.levels.insert(name, level);
        }

        Ok(config)
    }

    /// Find and load config from current directory or ancestors.
    pub fn find_and_load() -> Self {
        let config_names = [".sigillint.toml", "sigillint.toml"];

        if let Ok(mut dir) = std::env::current_dir() {
            loop {
                for name in &config_names {
                    let config_path = dir.join(name);
                    if config_path.exists() {
                        if let Ok(config) = Self::from_file(&config_path) {
                            return config;
                        }
                    }
                }
                if !dir.pop() {
                    break;
                }
            }
        }

        Self::default()
    }

    /// Generate a default config file as TOML string.
    pub fn default_toml() -> String {
        r#"# Sigil Linter Configuration
# Place this file as .sigillint.toml in your project root

[lint]
# Suggest Unicode morphemes (→ instead of ->, etc.)
suggest_unicode = true

# Check naming conventions (PascalCase, snake_case, etc.)
check_naming = true

# Maximum nesting depth before warning (default: 6)
max_nesting_depth = 6

# Lint level overrides (allow, warn, or deny)
[lint.levels]
# unused_variable = "allow"
# shadowing = "warn"
# deep_nesting = "deny"
# empty_block = "warn"
# bool_comparison = "warn"
"#.to_string()
    }
}

/// Lint severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LintLevel {
    Allow,
    Warn,
    Deny,
}

/// Lint rule categories for grouping and bulk enable/disable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum LintCategory {
    /// Code correctness issues that may cause bugs
    Correctness,
    /// Code style and formatting preferences
    Style,
    /// Performance-related suggestions
    Performance,
    /// Code complexity and maintainability
    Complexity,
    /// Sigil-specific features (evidentiality, morphemes)
    Sigil,
}

// ============================================
// Lint Rule Definitions
// ============================================

/// Unique identifier for a lint rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LintId {
    ReservedIdentifier,      // W0101
    NestedGenerics,          // W0104
    PreferUnicodeMorpheme,   // W0200
    NamingConvention,        // W0201
    UnusedVariable,          // W0202
    UnusedImport,            // W0203
    Shadowing,               // W0204
    DeepNesting,             // W0205
    EmptyBlock,              // W0206
    BoolComparison,          // W0207
    RedundantElse,           // W0208
    UnusedParameter,         // W0209
    MagicNumber,             // W0210
    MissingDocComment,       // W0211
    HighComplexity,          // W0212
    ConstantCondition,       // W0213
    PreferIfLet,             // W0214
    TodoWithoutIssue,        // W0215
    LongFunction,            // W0216
    TooManyParameters,       // W0217
    NeedlessReturn,          // W0218
    MissingReturn,           // W0300
    PreferMorphemePipeline,  // W0500
    EvidentialityViolation,  // E0600
    UnvalidatedExternalData, // E0601
    CertaintyDowngrade,      // E0602
    UnreachableCode,         // E0700
    InfiniteLoop,            // E0701
    DivisionByZero,          // E0702

    // === Aether 2.0 Enhanced Linter Rules ===

    // Enhanced Evidentiality Rules (E06xx series)
    EvidentialityMismatch,       // E0603 - Assignment between different evidence levels
    UncertaintyUnhandled,        // E0604 - Using ? values without error handling
    ReportedWithoutAttribution,  // E0605 - Using ~ without source attribution

    // Morpheme Pipeline Rules (W05xx series)
    BrokenMorphemePipeline,      // W0501 - Invalid morpheme chain
    MorphemeTypeIncompatibility, // W0502 - Type mismatch in pipeline
    InconsistentMorphemeStyle,   // W0503 - Mixing |τ{} and method chains

    // Domain Validation Rules (W06xx series - Aether/esoteric patterns)
    InvalidHexagramNumber,       // W0600 - I Ching hexagram outside 1-64
    InvalidTarotNumber,          // W0601 - Major Arcana outside 0-21
    InvalidChakraIndex,          // W0602 - Chakra index outside 0-6
    InvalidZodiacIndex,          // W0603 - Zodiac sign outside 0-11
    InvalidGematriaValue,        // W0604 - Negative or overflow gematria
    FrequencyOutOfRange,         // W0605 - Audio frequency outside audible range

    // Enhanced Pattern Rules (W07xx series)
    MissingEvidentialityMarker,  // W0700 - Type without !, ?, or ~ marker
    PreferNamedEsotericConstant, // W0701 - Magic numbers in esoteric contexts
    EmotionIntensityOutOfRange,  // W0702 - Emotion intensity outside valid range
}

impl LintId {
    pub fn code(&self) -> &'static str {
        match self {
            LintId::ReservedIdentifier => "W0101",
            LintId::NestedGenerics => "W0104",
            LintId::PreferUnicodeMorpheme => "W0200",
            LintId::NamingConvention => "W0201",
            LintId::UnusedVariable => "W0202",
            LintId::UnusedImport => "W0203",
            LintId::Shadowing => "W0204",
            LintId::DeepNesting => "W0205",
            LintId::EmptyBlock => "W0206",
            LintId::BoolComparison => "W0207",
            LintId::RedundantElse => "W0208",
            LintId::UnusedParameter => "W0209",
            LintId::MagicNumber => "W0210",
            LintId::MissingDocComment => "W0211",
            LintId::HighComplexity => "W0212",
            LintId::ConstantCondition => "W0213",
            LintId::PreferIfLet => "W0214",
            LintId::TodoWithoutIssue => "W0215",
            LintId::LongFunction => "W0216",
            LintId::TooManyParameters => "W0217",
            LintId::NeedlessReturn => "W0218",
            LintId::MissingReturn => "W0300",
            LintId::PreferMorphemePipeline => "W0500",
            LintId::EvidentialityViolation => "E0600",
            LintId::UnvalidatedExternalData => "E0601",
            LintId::CertaintyDowngrade => "E0602",
            LintId::UnreachableCode => "E0700",
            LintId::InfiniteLoop => "E0701",
            LintId::DivisionByZero => "E0702",

            // Aether 2.0 Enhanced Rules
            LintId::EvidentialityMismatch => "E0603",
            LintId::UncertaintyUnhandled => "E0604",
            LintId::ReportedWithoutAttribution => "E0605",
            LintId::BrokenMorphemePipeline => "W0501",
            LintId::MorphemeTypeIncompatibility => "W0502",
            LintId::InconsistentMorphemeStyle => "W0503",
            LintId::InvalidHexagramNumber => "W0600",
            LintId::InvalidTarotNumber => "W0601",
            LintId::InvalidChakraIndex => "W0602",
            LintId::InvalidZodiacIndex => "W0603",
            LintId::InvalidGematriaValue => "W0604",
            LintId::FrequencyOutOfRange => "W0605",
            LintId::MissingEvidentialityMarker => "W0700",
            LintId::PreferNamedEsotericConstant => "W0701",
            LintId::EmotionIntensityOutOfRange => "W0702",
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            LintId::ReservedIdentifier => "reserved_identifier",
            LintId::NestedGenerics => "nested_generics_unsupported",
            LintId::PreferUnicodeMorpheme => "prefer_unicode_morpheme",
            LintId::NamingConvention => "naming_convention",
            LintId::UnusedVariable => "unused_variable",
            LintId::UnusedImport => "unused_import",
            LintId::Shadowing => "shadowing",
            LintId::DeepNesting => "deep_nesting",
            LintId::EmptyBlock => "empty_block",
            LintId::BoolComparison => "bool_comparison",
            LintId::RedundantElse => "redundant_else",
            LintId::UnusedParameter => "unused_parameter",
            LintId::MagicNumber => "magic_number",
            LintId::MissingDocComment => "missing_doc_comment",
            LintId::HighComplexity => "high_complexity",
            LintId::ConstantCondition => "constant_condition",
            LintId::PreferIfLet => "prefer_if_let",
            LintId::TodoWithoutIssue => "todo_without_issue",
            LintId::LongFunction => "long_function",
            LintId::TooManyParameters => "too_many_parameters",
            LintId::NeedlessReturn => "needless_return",
            LintId::MissingReturn => "missing_return",
            LintId::PreferMorphemePipeline => "prefer_morpheme_pipeline",
            LintId::EvidentialityViolation => "evidentiality_violation",
            LintId::UnvalidatedExternalData => "unvalidated_external_data",
            LintId::CertaintyDowngrade => "certainty_downgrade",
            LintId::UnreachableCode => "unreachable_code",
            LintId::InfiniteLoop => "infinite_loop",
            LintId::DivisionByZero => "division_by_zero",

            // Aether 2.0 Enhanced Rules
            LintId::EvidentialityMismatch => "evidentiality_mismatch",
            LintId::UncertaintyUnhandled => "uncertainty_unhandled",
            LintId::ReportedWithoutAttribution => "reported_without_attribution",
            LintId::BrokenMorphemePipeline => "broken_morpheme_pipeline",
            LintId::MorphemeTypeIncompatibility => "morpheme_type_incompatibility",
            LintId::InconsistentMorphemeStyle => "inconsistent_morpheme_style",
            LintId::InvalidHexagramNumber => "invalid_hexagram_number",
            LintId::InvalidTarotNumber => "invalid_tarot_number",
            LintId::InvalidChakraIndex => "invalid_chakra_index",
            LintId::InvalidZodiacIndex => "invalid_zodiac_index",
            LintId::InvalidGematriaValue => "invalid_gematria_value",
            LintId::FrequencyOutOfRange => "frequency_out_of_range",
            LintId::MissingEvidentialityMarker => "missing_evidentiality_marker",
            LintId::PreferNamedEsotericConstant => "prefer_named_esoteric_constant",
            LintId::EmotionIntensityOutOfRange => "emotion_intensity_out_of_range",
        }
    }

    pub fn default_level(&self) -> LintLevel {
        match self {
            LintId::ReservedIdentifier => LintLevel::Warn,
            LintId::NestedGenerics => LintLevel::Warn,
            LintId::PreferUnicodeMorpheme => LintLevel::Allow,
            LintId::NamingConvention => LintLevel::Warn,
            LintId::UnusedVariable => LintLevel::Warn,
            LintId::UnusedImport => LintLevel::Warn,
            LintId::Shadowing => LintLevel::Warn,
            LintId::DeepNesting => LintLevel::Warn,
            LintId::EmptyBlock => LintLevel::Warn,
            LintId::BoolComparison => LintLevel::Warn,
            LintId::RedundantElse => LintLevel::Warn,
            LintId::UnusedParameter => LintLevel::Warn,
            LintId::MagicNumber => LintLevel::Allow, // Off by default, can be noisy
            LintId::MissingDocComment => LintLevel::Allow, // Off by default
            LintId::HighComplexity => LintLevel::Warn,
            LintId::ConstantCondition => LintLevel::Warn,
            LintId::PreferIfLet => LintLevel::Allow, // Style preference
            LintId::TodoWithoutIssue => LintLevel::Allow, // Off by default
            LintId::LongFunction => LintLevel::Warn,
            LintId::TooManyParameters => LintLevel::Warn,
            LintId::NeedlessReturn => LintLevel::Allow, // Style preference
            LintId::MissingReturn => LintLevel::Warn,
            LintId::PreferMorphemePipeline => LintLevel::Allow, // Stylistic suggestion
            LintId::EvidentialityViolation => LintLevel::Deny,
            LintId::UnvalidatedExternalData => LintLevel::Deny,
            LintId::CertaintyDowngrade => LintLevel::Warn,
            LintId::UnreachableCode => LintLevel::Warn,
            LintId::InfiniteLoop => LintLevel::Warn,
            LintId::DivisionByZero => LintLevel::Deny,

            // Aether 2.0 Enhanced Rules
            LintId::EvidentialityMismatch => LintLevel::Deny,      // Critical: type safety
            LintId::UncertaintyUnhandled => LintLevel::Warn,       // Should handle uncertain data
            LintId::ReportedWithoutAttribution => LintLevel::Warn, // Attribution expected
            LintId::BrokenMorphemePipeline => LintLevel::Deny,     // Critical: syntax error
            LintId::MorphemeTypeIncompatibility => LintLevel::Deny,// Critical: type safety
            LintId::InconsistentMorphemeStyle => LintLevel::Allow, // Stylistic preference
            LintId::InvalidHexagramNumber => LintLevel::Warn,      // Domain validation
            LintId::InvalidTarotNumber => LintLevel::Warn,         // Domain validation
            LintId::InvalidChakraIndex => LintLevel::Warn,         // Domain validation
            LintId::InvalidZodiacIndex => LintLevel::Warn,         // Domain validation
            LintId::InvalidGematriaValue => LintLevel::Warn,       // Domain validation
            LintId::FrequencyOutOfRange => LintLevel::Warn,        // Domain validation
            LintId::MissingEvidentialityMarker => LintLevel::Allow,// Opt-in strictness
            LintId::PreferNamedEsotericConstant => LintLevel::Allow,// Stylistic preference
            LintId::EmotionIntensityOutOfRange => LintLevel::Warn, // Domain validation
        }
    }

    pub fn description(&self) -> &'static str {
        match self {
            LintId::ReservedIdentifier => "This identifier is a reserved word in Sigil",
            LintId::NestedGenerics => "Nested generic parameters may not parse correctly",
            LintId::PreferUnicodeMorpheme => "Consider using Unicode morphemes for idiomatic Sigil",
            LintId::NamingConvention => "Identifier does not follow Sigil naming conventions",
            LintId::UnusedVariable => "Variable is declared but never used",
            LintId::UnusedImport => "Import is never used",
            LintId::Shadowing => "Variable shadows another variable from an outer scope",
            LintId::DeepNesting => "Code has excessive nesting depth, consider refactoring",
            LintId::EmptyBlock => "Empty block does nothing, consider adding code or removing",
            LintId::BoolComparison => "Comparison to boolean literal is redundant",
            LintId::RedundantElse => "Else branch after return/break/continue is redundant",
            LintId::UnusedParameter => "Function parameter is never used",
            LintId::MagicNumber => "Consider using a named constant instead of magic number",
            LintId::MissingDocComment => "Public item should have a documentation comment",
            LintId::HighComplexity => "Function has high cyclomatic complexity, consider refactoring",
            LintId::ConstantCondition => "Condition is always true or always false",
            LintId::PreferIfLet => "Consider using if-let instead of match with single arm",
            LintId::TodoWithoutIssue => "TODO comment without issue reference",
            LintId::LongFunction => "Function exceeds maximum line count",
            LintId::TooManyParameters => "Function has too many parameters",
            LintId::NeedlessReturn => "Unnecessary return statement at end of function",
            LintId::MissingReturn => "Function may not return a value on all code paths",
            LintId::PreferMorphemePipeline => "Consider using morpheme pipeline (|τ{}, |φ{}) instead of method chain",
            LintId::EvidentialityViolation => "Evidence level mismatch in assignment or call",
            LintId::UnvalidatedExternalData => "External data (~) used without validation",
            LintId::CertaintyDowngrade => "Certain (!) data being downgraded to uncertain (?)",
            LintId::UnreachableCode => "Code will never be executed",
            LintId::InfiniteLoop => "Loop has no exit condition",
            LintId::DivisionByZero => "Division by zero detected",

            // Aether 2.0 Enhanced Rules
            LintId::EvidentialityMismatch => "Assigning between incompatible evidentiality levels (!, ?, ~)",
            LintId::UncertaintyUnhandled => "Uncertain (?) value used without error handling or unwrap",
            LintId::ReportedWithoutAttribution => "Reported (~) data lacks source attribution",
            LintId::BrokenMorphemePipeline => "Morpheme pipeline has invalid or missing operators",
            LintId::MorphemeTypeIncompatibility => "Type mismatch between morpheme pipeline stages",
            LintId::InconsistentMorphemeStyle => "Mixing morpheme pipeline (|τ{}) with method chain (.map())",
            LintId::InvalidHexagramNumber => "I Ching hexagram number must be between 1 and 64",
            LintId::InvalidTarotNumber => "Major Arcana number must be between 0 and 21",
            LintId::InvalidChakraIndex => "Chakra index must be between 0 and 6",
            LintId::InvalidZodiacIndex => "Zodiac sign index must be between 0 and 11",
            LintId::InvalidGematriaValue => "Gematria value is negative or exceeds maximum",
            LintId::FrequencyOutOfRange => "Audio frequency outside audible range (20Hz-20kHz)",
            LintId::MissingEvidentialityMarker => "Type declaration lacks evidentiality marker (!, ?, ~)",
            LintId::PreferNamedEsotericConstant => "Use named constant for esoteric value (e.g., GOLDEN_RATIO)",
            LintId::EmotionIntensityOutOfRange => "Emotion intensity must be between 0.0 and 1.0",
        }
    }

    /// Get the category for this lint rule.
    pub fn category(&self) -> LintCategory {
        match self {
            // Correctness - things that are likely bugs
            LintId::DivisionByZero => LintCategory::Correctness,
            LintId::InfiniteLoop => LintCategory::Correctness,
            LintId::UnreachableCode => LintCategory::Correctness,
            LintId::ConstantCondition => LintCategory::Correctness,

            // Style - code style preferences
            LintId::NamingConvention => LintCategory::Style,
            LintId::BoolComparison => LintCategory::Style,
            LintId::RedundantElse => LintCategory::Style,
            LintId::EmptyBlock => LintCategory::Style,
            LintId::PreferIfLet => LintCategory::Style,
            LintId::MissingDocComment => LintCategory::Style,
            LintId::NeedlessReturn => LintCategory::Style,

            // Correctness - control flow
            LintId::MissingReturn => LintCategory::Correctness,

            // Sigil idioms
            LintId::PreferMorphemePipeline => LintCategory::Sigil,

            // Complexity - maintainability concerns
            LintId::DeepNesting => LintCategory::Complexity,
            LintId::HighComplexity => LintCategory::Complexity,
            LintId::MagicNumber => LintCategory::Complexity,
            LintId::LongFunction => LintCategory::Complexity,
            LintId::TooManyParameters => LintCategory::Complexity,
            LintId::TodoWithoutIssue => LintCategory::Complexity,

            // Performance - unused code, wasteful patterns
            LintId::UnusedVariable => LintCategory::Performance,
            LintId::UnusedImport => LintCategory::Performance,
            LintId::UnusedParameter => LintCategory::Performance,
            LintId::Shadowing => LintCategory::Performance,

            // Sigil-specific features
            LintId::ReservedIdentifier => LintCategory::Sigil,
            LintId::NestedGenerics => LintCategory::Sigil,
            LintId::PreferUnicodeMorpheme => LintCategory::Sigil,
            LintId::EvidentialityViolation => LintCategory::Sigil,
            LintId::UnvalidatedExternalData => LintCategory::Sigil,
            LintId::CertaintyDowngrade => LintCategory::Sigil,

            // Aether 2.0 Enhanced Rules - Evidentiality
            LintId::EvidentialityMismatch => LintCategory::Sigil,
            LintId::UncertaintyUnhandled => LintCategory::Sigil,
            LintId::ReportedWithoutAttribution => LintCategory::Sigil,

            // Aether 2.0 Enhanced Rules - Morphemes
            LintId::BrokenMorphemePipeline => LintCategory::Sigil,
            LintId::MorphemeTypeIncompatibility => LintCategory::Sigil,
            LintId::InconsistentMorphemeStyle => LintCategory::Style,

            // Aether 2.0 Enhanced Rules - Domain Validation
            LintId::InvalidHexagramNumber => LintCategory::Correctness,
            LintId::InvalidTarotNumber => LintCategory::Correctness,
            LintId::InvalidChakraIndex => LintCategory::Correctness,
            LintId::InvalidZodiacIndex => LintCategory::Correctness,
            LintId::InvalidGematriaValue => LintCategory::Correctness,
            LintId::FrequencyOutOfRange => LintCategory::Correctness,
            LintId::EmotionIntensityOutOfRange => LintCategory::Correctness,

            // Aether 2.0 Enhanced Rules - Style
            LintId::MissingEvidentialityMarker => LintCategory::Sigil,
            LintId::PreferNamedEsotericConstant => LintCategory::Complexity,
        }
    }

    /// Get extended documentation for this lint rule.
    pub fn extended_docs(&self) -> &'static str {
        match self {
            LintId::ReservedIdentifier => r#"
This lint detects use of identifiers that are reserved words in Sigil.
Reserved words have special meaning in the language and cannot be used
as variable, function, or type names.

Example:
    let location = "here";  // Error: 'location' is reserved

Fix:
    let place = "here";     // Use an alternative name

Common alternatives:
  - location -> place
  - save -> slot, store
  - from -> source, origin
"#,
            LintId::NestedGenerics => r#"
This lint warns about nested generic parameters which may not parse
correctly in the current version of Sigil.

Example:
    fn process(data: Vec<Option<i32>>) { }  // May not parse

Fix:
    type OptInt = Option<i32>;
    fn process(data: Vec<OptInt>) { }  // Use type alias
"#,
            LintId::UnusedVariable => r#"
This lint detects variables that are declared but never used.
Unused variables may indicate incomplete code or typos.

Example:
    let x = 42;
    println(y);  // 'x' is never used, 'y' may be a typo

Fix:
    let x = 42;
    println(x);  // Use the variable

    // Or prefix with underscore to indicate intentionally unused:
    let _x = 42;
"#,
            LintId::Shadowing => r#"
This lint warns when a variable shadows another variable from an
outer scope. While sometimes intentional, shadowing can make code
harder to understand.

Example:
    let x = 1;
    {
        let x = 2;  // Shadows outer 'x'
    }

Fix:
    let x = 1;
    {
        let x_inner = 2;  // Use distinct name
    }

    // Or prefix with underscore if intentional:
    let _x = 2;
"#,
            LintId::DeepNesting => r#"
This lint warns about excessively nested code structures.
Deep nesting makes code hard to read and maintain.

Example:
    if a {
        if b {
            if c {
                if d {  // Too deep!
                }
            }
        }
    }

Fix:
    // Use early returns
    if !a { return; }
    if !b { return; }
    if !c { return; }
    if d { ... }

    // Or extract into functions
    fn check_conditions() { ... }
"#,
            LintId::HighComplexity => r#"
This lint warns about functions with high cyclomatic complexity.
High complexity makes code harder to test and maintain.

Complexity is calculated by counting:
  - Each if/while/for/loop adds 1
  - Each match arm (except first) adds 1
  - Each && or || operator adds 1
  - Each guard condition adds 1

Fix:
    // Extract complex logic into smaller functions
    // Use early returns to reduce nesting
    // Consider using match instead of if-else chains
"#,
            LintId::DivisionByZero => r#"
This lint detects division by a literal zero, which will cause
a runtime panic.

Example:
    let result = x / 0;  // Will panic!

Fix:
    if divisor != 0 {
        let result = x / divisor;
    }
"#,
            LintId::ConstantCondition => r#"
This lint detects conditions that are always true or always false,
indicating likely bugs or unnecessary code.

Example:
    if true { ... }      // Always executes
    while false { ... }  // Never executes

Fix:
    // Remove unnecessary conditions
    // Or use the correct variable in the condition
"#,
            LintId::TodoWithoutIssue => r#"
This lint warns about TODO comments that don't reference an issue tracker.

Example:
    // TODO: fix this later

Fix:
    // TODO(#123): fix this later
    // TODO(GH-456): address edge case

Configure via .sigillint.toml:
    [lint.levels]
    todo_without_issue = "warn"
"#,
            LintId::LongFunction => r#"
This lint warns about functions that exceed a maximum line count.
Long functions are harder to understand, test, and maintain.

Default threshold: 50 lines

Fix:
    // Break into smaller, focused functions
    // Extract helper functions for distinct operations
    // Use early returns to reduce nesting
"#,
            LintId::TooManyParameters => r#"
This lint warns about functions with too many parameters.
Many parameters indicate a function may be doing too much.

Default threshold: 7 parameters

Fix:
    // Group related parameters into a struct
    // Use builder pattern for complex construction
    // Consider if function should be split
"#,
            LintId::NeedlessReturn => r#"
This lint suggests removing unnecessary return statements.
In Sigil, the last expression is the return value.

Example:
    fn add(a: i32, b: i32) -> i32 {
        return a + b;  // Unnecessary return
    }

Fix:
    fn add(a: i32, b: i32) -> i32 {
        a + b  // Implicit return
    }
"#,
            LintId::MissingReturn => r#"
This lint warns when a function with a return type may not return
a value on all execution paths.

Example:
    fn maybe_return(x: i32) -> i32 {
        if x > 0 {
            return x;
        }
        // Missing return for x <= 0!
    }

Fix:
    fn maybe_return(x: i32) -> i32 {
        if x > 0 {
            x
        } else {
            0  // Default value
        }
    }

The linter checks:
  - If all branches return a value
  - If match arms all produce values
  - If loops with breaks produce consistent values
"#,
            LintId::PreferMorphemePipeline => r#"
This lint suggests using Sigil's morpheme pipeline syntax instead
of method chains. Morpheme pipelines are more idiomatic in Sigil
and provide clearer data flow semantics.

Example (method chain):
    let result = data.iter().map(|x| x * 2).filter(|x| *x > 10).collect();

Preferred (morpheme pipeline):
    let result = data
        |τ{_ * 2}       // τ (tau) = transform/map
        |φ{_ > 10}      // φ (phi) = filter
        |σ;             // σ (sigma) = collect/sort

Common morpheme operators:
  - τ (tau)   : Transform/map
  - φ (phi)   : Filter
  - σ (sigma) : Sort/collect/sum
  - ρ (rho)   : Reduce/fold
  - α (alpha) : First element
  - ω (omega) : Last element
  - ζ (zeta)  : Zip/combine

This lint is off by default. Enable with:
    [lint.levels]
    prefer_morpheme_pipeline = "warn"
"#,
            _ => self.description(),
        }
    }

    /// Get all lint IDs.
    pub fn all() -> &'static [LintId] {
        &[
            LintId::ReservedIdentifier,
            LintId::NestedGenerics,
            LintId::PreferUnicodeMorpheme,
            LintId::NamingConvention,
            LintId::UnusedVariable,
            LintId::UnusedImport,
            LintId::Shadowing,
            LintId::DeepNesting,
            LintId::EmptyBlock,
            LintId::BoolComparison,
            LintId::RedundantElse,
            LintId::UnusedParameter,
            LintId::MagicNumber,
            LintId::MissingDocComment,
            LintId::HighComplexity,
            LintId::ConstantCondition,
            LintId::PreferIfLet,
            LintId::TodoWithoutIssue,
            LintId::LongFunction,
            LintId::TooManyParameters,
            LintId::NeedlessReturn,
            LintId::MissingReturn,
            LintId::PreferMorphemePipeline,
            LintId::EvidentialityViolation,
            LintId::UnvalidatedExternalData,
            LintId::CertaintyDowngrade,
            LintId::UnreachableCode,
            LintId::InfiniteLoop,
            LintId::DivisionByZero,

            // Aether 2.0 Enhanced Rules
            LintId::EvidentialityMismatch,
            LintId::UncertaintyUnhandled,
            LintId::ReportedWithoutAttribution,
            LintId::BrokenMorphemePipeline,
            LintId::MorphemeTypeIncompatibility,
            LintId::InconsistentMorphemeStyle,
            LintId::InvalidHexagramNumber,
            LintId::InvalidTarotNumber,
            LintId::InvalidChakraIndex,
            LintId::InvalidZodiacIndex,
            LintId::InvalidGematriaValue,
            LintId::FrequencyOutOfRange,
            LintId::MissingEvidentialityMarker,
            LintId::PreferNamedEsotericConstant,
            LintId::EmotionIntensityOutOfRange,
        ]
    }

    /// Find a lint by code (e.g., "W0101") or name (e.g., "reserved_identifier").
    pub fn from_str(s: &str) -> Option<LintId> {
        for lint in Self::all() {
            if lint.code() == s || lint.name() == s {
                return Some(*lint);
            }
        }
        None
    }
}

// ============================================
// Inline Suppression Comments
// ============================================

/// A parsed inline suppression directive.
#[derive(Debug, Clone)]
pub struct Suppression {
    /// Line number (1-indexed) where the suppression applies
    pub line: usize,
    /// Lint IDs to suppress (empty means all)
    pub lints: Vec<LintId>,
    /// Whether this suppression applies to the next line only
    pub next_line: bool,
}

/// Parse inline suppression comments from source code.
///
/// Supports two formats:
/// - `// sigil-lint: allow(W0201, unused_variable)` - suppress on current/next line
/// - `// sigil-lint: allow-next-line(W0201)` - suppress on next line only
pub fn parse_suppressions(source: &str) -> Vec<Suppression> {
    let mut suppressions = Vec::new();

    for (line_num, line) in source.lines().enumerate() {
        let line_1indexed = line_num + 1;

        // Find suppression comment
        if let Some(comment_start) = line.find("// sigil-lint:") {
            let comment = &line[comment_start + 14..].trim();

            if let Some(rest) = comment.strip_prefix("allow-next-line") {
                // Suppress next line only
                if let Some(lints) = parse_lint_list(rest) {
                    suppressions.push(Suppression {
                        line: line_1indexed + 1,
                        lints,
                        next_line: true,
                    });
                }
            } else if let Some(rest) = comment.strip_prefix("allow") {
                // Suppress current line (or next line if at end of line)
                if let Some(lints) = parse_lint_list(rest) {
                    suppressions.push(Suppression {
                        line: line_1indexed,
                        lints,
                        next_line: false,
                    });
                }
            }
        }
    }

    suppressions
}

/// Parse a lint list like "(W0201, unused_variable)".
fn parse_lint_list(s: &str) -> Option<Vec<LintId>> {
    let s = s.trim();
    if !s.starts_with('(') || !s.contains(')') {
        return Some(Vec::new()); // No list = suppress all
    }

    let start = s.find('(')? + 1;
    let end = s.find(')')?;
    let list = &s[start..end];

    let mut lints = Vec::new();
    for item in list.split(',') {
        let item = item.trim();
        if !item.is_empty() {
            if let Some(lint) = LintId::from_str(item) {
                lints.push(lint);
            }
        }
    }

    Some(lints)
}

// ============================================
// Lint Statistics
// ============================================

/// Statistics about a lint run.
#[derive(Debug, Clone, Default)]
pub struct LintStats {
    /// Count of each lint type encountered
    pub lint_counts: HashMap<LintId, usize>,
    /// Count per category
    pub category_counts: HashMap<LintCategory, usize>,
    /// Total diagnostics emitted
    pub total_diagnostics: usize,
    /// Diagnostics suppressed by inline comments
    pub suppressed: usize,
    /// Time taken to lint (in microseconds)
    pub duration_us: u64,
}

impl LintStats {
    /// Record a lint occurrence.
    pub fn record(&mut self, lint: LintId) {
        *self.lint_counts.entry(lint).or_insert(0) += 1;
        *self.category_counts.entry(lint.category()).or_insert(0) += 1;
        self.total_diagnostics += 1;
    }

    /// Record a suppressed lint.
    pub fn record_suppressed(&mut self) {
        self.suppressed += 1;
    }
}

// ============================================
// Baseline Support
// ============================================

/// A single baseline entry representing a known lint issue.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct BaselineEntry {
    /// File path (relative to project root)
    pub file: String,
    /// Lint rule code (e.g., "W0202")
    pub code: String,
    /// Line number (1-indexed, 0 means unknown)
    pub line: usize,
    /// Hash of the diagnostic message for matching
    pub message_hash: u64,
    /// Original message (for human readability)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
}

impl BaselineEntry {
    /// Create a baseline entry from a diagnostic.
    pub fn from_diagnostic(file: &str, diag: &Diagnostic, source: &str) -> Self {
        let line = Self::offset_to_line(diag.span.start, source);
        let message_hash = Self::hash_message(&diag.message);

        Self {
            file: file.to_string(),
            code: diag.code.clone().unwrap_or_default(),
            line,
            message_hash,
            message: Some(diag.message.clone()),
        }
    }

    /// Calculate line number from byte offset.
    fn offset_to_line(offset: usize, source: &str) -> usize {
        source[..offset.min(source.len())]
            .chars()
            .filter(|&c| c == '\n')
            .count() + 1
    }

    /// Simple hash of a message for comparison.
    fn hash_message(message: &str) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        message.hash(&mut hasher);
        hasher.finish()
    }

    /// Check if this entry matches a diagnostic (fuzzy match).
    pub fn matches(&self, file: &str, diag: &Diagnostic, source: &str) -> bool {
        // Must match file and code
        if self.file != file {
            return false;
        }
        if let Some(ref code) = diag.code {
            if &self.code != code {
                return false;
            }
        }

        // Try exact message hash match first
        let msg_hash = Self::hash_message(&diag.message);
        if self.message_hash == msg_hash {
            return true;
        }

        // Fall back to line-based match if message changed slightly
        let diag_line = Self::offset_to_line(diag.span.start, source);
        if self.line > 0 && diag_line > 0 {
            // Allow ±3 lines tolerance for code movement
            let line_diff = (self.line as i64 - diag_line as i64).abs();
            if line_diff <= 3 && self.code == diag.code.as_deref().unwrap_or("") {
                return true;
            }
        }

        false
    }
}

/// A baseline file containing known lint issues.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Baseline {
    /// Schema version for forward compatibility
    pub version: u32,
    /// Timestamp when baseline was created/updated
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created: Option<String>,
    /// Number of entries
    pub count: usize,
    /// Baseline entries grouped by file
    pub entries: HashMap<String, Vec<BaselineEntry>>,
}

impl Baseline {
    /// Create a new empty baseline.
    pub fn new() -> Self {
        Self {
            version: 1,
            created: Some(chrono_lite_now()),
            count: 0,
            entries: HashMap::new(),
        }
    }

    /// Load baseline from a JSON file.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read baseline file: {}", e))?;
        Self::from_json(&content)
    }

    /// Parse baseline from JSON string.
    pub fn from_json(content: &str) -> Result<Self, String> {
        serde_json::from_str(content)
            .map_err(|e| format!("Failed to parse baseline: {}", e))
    }

    /// Save baseline to a JSON file.
    pub fn to_file(&self, path: &Path) -> Result<(), String> {
        let content = self.to_json()?;
        std::fs::write(path, content)
            .map_err(|e| format!("Failed to write baseline file: {}", e))
    }

    /// Convert baseline to JSON string.
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize baseline: {}", e))
    }

    /// Add a diagnostic to the baseline.
    pub fn add(&mut self, file: &str, diag: &Diagnostic, source: &str) {
        let entry = BaselineEntry::from_diagnostic(file, diag, source);
        self.entries
            .entry(file.to_string())
            .or_default()
            .push(entry);
        self.count += 1;
    }

    /// Check if a diagnostic is in the baseline.
    pub fn contains(&self, file: &str, diag: &Diagnostic, source: &str) -> bool {
        if let Some(entries) = self.entries.get(file) {
            entries.iter().any(|e| e.matches(file, diag, source))
        } else {
            false
        }
    }

    /// Filter diagnostics, removing those in the baseline.
    /// Returns (filtered_diagnostics, baseline_matches).
    pub fn filter(&self, file: &str, diagnostics: &Diagnostics, source: &str) -> (Diagnostics, usize) {
        let mut filtered = Diagnostics::new();
        let mut baseline_matches = 0;

        for diag in diagnostics.iter() {
            if self.contains(file, diag, source) {
                baseline_matches += 1;
            } else {
                filtered.add(diag.clone());
            }
        }

        (filtered, baseline_matches)
    }

    /// Create a baseline from directory lint results.
    pub fn from_directory_result(result: &DirectoryLintResult, sources: &HashMap<String, String>) -> Self {
        let mut baseline = Self::new();

        for (path, diagnostics) in &result.files {
            if let Some(source) = sources.get(path) {
                for diag in diagnostics.iter() {
                    baseline.add(path, diag, source);
                }
            }
        }

        baseline
    }

    /// Update baseline: keep existing entries that still match, add new issues.
    pub fn update(&mut self, file: &str, diagnostics: &Diagnostics, source: &str) {
        let mut new_entries = Vec::new();

        // Keep entries that still match current diagnostics
        if let Some(old_entries) = self.entries.get(file) {
            for old in old_entries {
                // Check if any diagnostic still matches this baseline entry
                let still_exists = diagnostics.iter().any(|d| old.matches(file, d, source));
                if still_exists {
                    new_entries.push(old.clone());
                }
            }
        }

        // Add new diagnostics not already in baseline
        for diag in diagnostics.iter() {
            let already_exists = new_entries.iter().any(|e| e.matches(file, diag, source));
            if !already_exists {
                new_entries.push(BaselineEntry::from_diagnostic(file, diag, source));
            }
        }

        // Update count
        let old_count = self.entries.get(file).map(|v| v.len()).unwrap_or(0);
        self.count = self.count - old_count + new_entries.len();

        if new_entries.is_empty() {
            self.entries.remove(file);
        } else {
            self.entries.insert(file.to_string(), new_entries);
        }

        self.created = Some(chrono_lite_now());
    }

    /// Get summary statistics.
    pub fn summary(&self) -> BaselineSummary {
        let mut by_code: HashMap<String, usize> = HashMap::new();

        for entries in self.entries.values() {
            for entry in entries {
                *by_code.entry(entry.code.clone()).or_insert(0) += 1;
            }
        }

        BaselineSummary {
            total_files: self.entries.len(),
            total_issues: self.count,
            by_code,
        }
    }
}

/// Summary of baseline contents.
#[derive(Debug, Clone)]
pub struct BaselineSummary {
    /// Number of files with baselined issues
    pub total_files: usize,
    /// Total number of baselined issues
    pub total_issues: usize,
    /// Issues grouped by lint code
    pub by_code: HashMap<String, usize>,
}

/// Simple timestamp function (no chrono dependency).
fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();

    // Convert to simple ISO-8601 format
    let days = secs / 86400;
    let years = 1970 + days / 365;
    let remaining_days = days % 365;
    let months = remaining_days / 30 + 1;
    let day = remaining_days % 30 + 1;
    let hours = (secs % 86400) / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;

    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
            years, months.min(12), day.min(31), hours, minutes, seconds)
}

/// Find and load baseline from standard locations.
///
/// Searches for:
/// - `.sigillint-baseline.json`
/// - `sigillint-baseline.json`
/// - `.lint-baseline.json`
pub fn find_baseline() -> Option<Baseline> {
    let baseline_names = [
        ".sigillint-baseline.json",
        "sigillint-baseline.json",
        ".lint-baseline.json",
    ];

    if let Ok(mut dir) = std::env::current_dir() {
        loop {
            for name in &baseline_names {
                let path = dir.join(name);
                if path.exists() {
                    if let Ok(baseline) = Baseline::from_file(&path) {
                        return Some(baseline);
                    }
                }
            }
            if !dir.pop() {
                break;
            }
        }
    }

    None
}

/// Result of linting with baseline filtering.
#[derive(Debug)]
pub struct BaselineLintResult {
    /// New issues (not in baseline)
    pub new_issues: Diagnostics,
    /// Issues that matched baseline (suppressed)
    pub baseline_matches: usize,
    /// Total issues before filtering
    pub total_before: usize,
}

/// Lint with baseline filtering.
pub fn lint_with_baseline(
    source: &str,
    filename: &str,
    config: LintConfig,
    baseline: &Baseline,
) -> BaselineLintResult {
    let diagnostics = lint_source_with_config(source, filename, config);
    let total_before = diagnostics.iter().count();
    let (new_issues, baseline_matches) = baseline.filter(filename, &diagnostics, source);

    BaselineLintResult {
        new_issues,
        baseline_matches,
        total_before,
    }
}

// ============================================
// CLI Severity Overrides
// ============================================

/// Command-line overrides for lint levels.
///
/// Allows users to pass `--deny`, `--allow`, and `--warn` flags
/// to override lint levels without modifying config files.
///
/// # Priority
/// CLI overrides take highest priority, overriding both:
/// 1. Default lint levels
/// 2. Config file settings
///
/// # Usage
/// ```text
/// sigil lint --deny unused_variable --warn magic_number --allow W0211
/// sigil lint --deny-category correctness --allow-category style
/// ```
#[derive(Debug, Clone, Default)]
pub struct CliOverrides {
    /// Lints to set to Deny level
    pub deny: Vec<String>,
    /// Lints to set to Warn level
    pub warn: Vec<String>,
    /// Lints to set to Allow level
    pub allow: Vec<String>,
    /// Categories to set to Deny level
    pub deny_category: Vec<LintCategory>,
    /// Categories to set to Warn level
    pub warn_category: Vec<LintCategory>,
    /// Categories to set to Allow level
    pub allow_category: Vec<LintCategory>,
}

impl CliOverrides {
    /// Create a new empty set of overrides.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a lint to deny.
    pub fn deny(mut self, lint: impl Into<String>) -> Self {
        self.deny.push(lint.into());
        self
    }

    /// Add a lint to warn.
    pub fn warn(mut self, lint: impl Into<String>) -> Self {
        self.warn.push(lint.into());
        self
    }

    /// Add a lint to allow.
    pub fn allow(mut self, lint: impl Into<String>) -> Self {
        self.allow.push(lint.into());
        self
    }

    /// Add a category to deny.
    pub fn deny_cat(mut self, category: LintCategory) -> Self {
        self.deny_category.push(category);
        self
    }

    /// Add a category to warn.
    pub fn warn_cat(mut self, category: LintCategory) -> Self {
        self.warn_category.push(category);
        self
    }

    /// Add a category to allow.
    pub fn allow_cat(mut self, category: LintCategory) -> Self {
        self.allow_category.push(category);
        self
    }

    /// Apply overrides to a LintConfig.
    ///
    /// Overrides are applied in this order:
    /// 1. Category-level overrides (less specific)
    /// 2. Individual lint overrides (more specific, takes precedence)
    pub fn apply(&self, config: &mut LintConfig) {
        // First, apply category overrides
        for cat in &self.allow_category {
            for lint in LintId::all() {
                if lint.category() == *cat {
                    config.levels.insert(lint.name().to_string(), LintLevel::Allow);
                }
            }
        }
        for cat in &self.warn_category {
            for lint in LintId::all() {
                if lint.category() == *cat {
                    config.levels.insert(lint.name().to_string(), LintLevel::Warn);
                }
            }
        }
        for cat in &self.deny_category {
            for lint in LintId::all() {
                if lint.category() == *cat {
                    config.levels.insert(lint.name().to_string(), LintLevel::Deny);
                }
            }
        }

        // Then, apply individual lint overrides (takes precedence)
        for lint_str in &self.allow {
            if let Some(lint) = LintId::from_str(lint_str) {
                config.levels.insert(lint.name().to_string(), LintLevel::Allow);
            } else {
                // Try as a name directly
                config.levels.insert(lint_str.clone(), LintLevel::Allow);
            }
        }
        for lint_str in &self.warn {
            if let Some(lint) = LintId::from_str(lint_str) {
                config.levels.insert(lint.name().to_string(), LintLevel::Warn);
            } else {
                config.levels.insert(lint_str.clone(), LintLevel::Warn);
            }
        }
        for lint_str in &self.deny {
            if let Some(lint) = LintId::from_str(lint_str) {
                config.levels.insert(lint.name().to_string(), LintLevel::Deny);
            } else {
                config.levels.insert(lint_str.clone(), LintLevel::Deny);
            }
        }
    }

    /// Parse a category from string.
    pub fn parse_category(s: &str) -> Option<LintCategory> {
        match s.to_lowercase().as_str() {
            "correctness" => Some(LintCategory::Correctness),
            "style" => Some(LintCategory::Style),
            "performance" => Some(LintCategory::Performance),
            "complexity" => Some(LintCategory::Complexity),
            "sigil" => Some(LintCategory::Sigil),
            _ => None,
        }
    }

    /// Check if any overrides are set.
    pub fn is_empty(&self) -> bool {
        self.deny.is_empty()
            && self.warn.is_empty()
            && self.allow.is_empty()
            && self.deny_category.is_empty()
            && self.warn_category.is_empty()
            && self.allow_category.is_empty()
    }
}

/// Create a LintConfig with CLI overrides applied.
pub fn config_with_overrides(base: LintConfig, overrides: &CliOverrides) -> LintConfig {
    let mut config = base;
    overrides.apply(&mut config);
    config
}

/// Lint source with CLI overrides.
pub fn lint_source_with_overrides(
    source: &str,
    filename: &str,
    overrides: &CliOverrides,
) -> Diagnostics {
    let mut config = LintConfig::find_and_load();
    overrides.apply(&mut config);
    lint_source_with_config(source, filename, config)
}

// ============================================
// File Hash Caching for Incremental Linting
// ============================================

/// Cache for storing file hashes and lint results.
///
/// Enables incremental linting by skipping unchanged files.
/// Cache is stored as JSON and can be persisted to disk.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LintCache {
    /// Schema version for forward compatibility
    pub version: u32,
    /// Config hash - invalidate cache if config changes
    pub config_hash: u64,
    /// Cached file entries: path -> CacheEntry
    pub entries: HashMap<String, CacheEntry>,
}

/// A cached lint result for a single file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    /// BLAKE3 hash of file contents
    pub content_hash: String,
    /// Modification timestamp (Unix epoch seconds)
    pub mtime: u64,
    /// File size in bytes
    pub size: u64,
    /// Cached diagnostic count (for quick stats)
    pub warning_count: usize,
    /// Cached error count
    pub error_count: usize,
    /// Serialized diagnostics (for avoiding re-lint)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<Vec<CachedDiagnostic>>,
}

/// Minimal diagnostic representation for caching.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedDiagnostic {
    pub code: Option<String>,
    pub message: String,
    pub severity: String,
    pub start: usize,
    pub end: usize,
}

impl CachedDiagnostic {
    /// Convert from a full Diagnostic.
    pub fn from_diagnostic(diag: &Diagnostic) -> Self {
        Self {
            code: diag.code.clone(),
            message: diag.message.clone(),
            severity: format!("{:?}", diag.severity),
            start: diag.span.start,
            end: diag.span.end,
        }
    }

    /// Convert back to a full Diagnostic.
    pub fn to_diagnostic(&self) -> Diagnostic {
        let severity = match self.severity.as_str() {
            "Error" => Severity::Error,
            "Warning" => Severity::Warning,
            "Info" => Severity::Info,
            "Hint" => Severity::Hint,
            _ => Severity::Warning,
        };

        Diagnostic {
            severity,
            code: self.code.clone(),
            message: self.message.clone(),
            span: Span::new(self.start, self.end),
            labels: Vec::new(),
            notes: Vec::new(),
            suggestions: Vec::new(),
            related: Vec::new(),
        }
    }
}

impl LintCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self {
            version: 1,
            config_hash: 0,
            entries: HashMap::new(),
        }
    }

    /// Create a cache with a specific config hash.
    pub fn with_config(config: &LintConfig) -> Self {
        Self {
            version: 1,
            config_hash: Self::hash_config(config),
            entries: HashMap::new(),
        }
    }

    /// Hash a LintConfig for change detection.
    fn hash_config(config: &LintConfig) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();

        // Hash key config fields
        config.suggest_unicode.hash(&mut hasher);
        config.check_naming.hash(&mut hasher);
        config.max_nesting_depth.hash(&mut hasher);

        // Hash level overrides (sorted for consistency)
        let mut levels: Vec<_> = config.levels.iter().collect();
        levels.sort_by_key(|(k, _)| *k);
        for (name, level) in levels {
            name.hash(&mut hasher);
            std::mem::discriminant(level).hash(&mut hasher);
        }

        hasher.finish()
    }

    /// Load cache from a JSON file.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read cache file: {}", e))?;
        Self::from_json(&content)
    }

    /// Parse cache from JSON string.
    pub fn from_json(content: &str) -> Result<Self, String> {
        serde_json::from_str(content)
            .map_err(|e| format!("Failed to parse cache: {}", e))
    }

    /// Save cache to a JSON file.
    pub fn to_file(&self, path: &Path) -> Result<(), String> {
        let content = self.to_json()?;
        std::fs::write(path, content)
            .map_err(|e| format!("Failed to write cache file: {}", e))
    }

    /// Convert cache to JSON string.
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string(self)
            .map_err(|e| format!("Failed to serialize cache: {}", e))
    }

    /// Compute BLAKE3 hash of file contents.
    pub fn hash_content(content: &str) -> String {
        let hash = blake3::hash(content.as_bytes());
        hash.to_hex().to_string()
    }

    /// Check if a file needs re-linting.
    ///
    /// Returns `true` if:
    /// - File is not in cache
    /// - File content has changed (different hash)
    /// - File metadata suggests change (mtime/size)
    pub fn needs_lint(&self, path: &str, content: &str, metadata: Option<&std::fs::Metadata>) -> bool {
        let Some(entry) = self.entries.get(path) else {
            return true; // Not in cache
        };

        // Quick check: file size
        if let Some(meta) = metadata {
            if entry.size != meta.len() {
                return true;
            }
        }

        // Content hash check (definitive)
        let current_hash = Self::hash_content(content);
        entry.content_hash != current_hash
    }

    /// Get cached diagnostics for a file if valid.
    pub fn get_cached(&self, path: &str, content: &str) -> Option<Diagnostics> {
        let entry = self.entries.get(path)?;

        // Verify content hash
        let current_hash = Self::hash_content(content);
        if entry.content_hash != current_hash {
            return None;
        }

        // Convert cached diagnostics back
        let cached = entry.diagnostics.as_ref()?;
        let mut diagnostics = Diagnostics::new();
        for cd in cached {
            diagnostics.add(cd.to_diagnostic());
        }

        Some(diagnostics)
    }

    /// Update cache entry for a file.
    pub fn update(
        &mut self,
        path: &str,
        content: &str,
        diagnostics: &Diagnostics,
        metadata: Option<&std::fs::Metadata>,
    ) {
        let content_hash = Self::hash_content(content);

        let mtime = metadata
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let size = metadata.map(|m| m.len()).unwrap_or(0);

        let warning_count = diagnostics.iter()
            .filter(|d| d.severity == Severity::Warning)
            .count();
        let error_count = diagnostics.iter()
            .filter(|d| d.severity == Severity::Error)
            .count();

        let cached_diags: Vec<CachedDiagnostic> = diagnostics
            .iter()
            .map(CachedDiagnostic::from_diagnostic)
            .collect();

        self.entries.insert(path.to_string(), CacheEntry {
            content_hash,
            mtime,
            size,
            warning_count,
            error_count,
            diagnostics: Some(cached_diags),
        });
    }

    /// Remove stale entries (files that no longer exist).
    pub fn prune(&mut self, existing_files: &HashSet<String>) {
        self.entries.retain(|path, _| existing_files.contains(path));
    }

    /// Check if cache is valid for given config.
    pub fn is_valid_for(&self, config: &LintConfig) -> bool {
        self.config_hash == Self::hash_config(config)
    }

    /// Get cache statistics.
    pub fn stats(&self) -> CacheStats {
        let mut total_warnings = 0;
        let mut total_errors = 0;

        for entry in self.entries.values() {
            total_warnings += entry.warning_count;
            total_errors += entry.error_count;
        }

        CacheStats {
            cached_files: self.entries.len(),
            total_warnings,
            total_errors,
        }
    }
}

/// Statistics about the lint cache.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of files in cache
    pub cached_files: usize,
    /// Total warnings across cached files
    pub total_warnings: usize,
    /// Total errors across cached files
    pub total_errors: usize,
}

/// Default cache file name.
pub const CACHE_FILE: &str = ".sigillint-cache.json";

/// Find and load cache from standard location.
pub fn find_cache() -> Option<LintCache> {
    if let Ok(dir) = std::env::current_dir() {
        let cache_path = dir.join(CACHE_FILE);
        if cache_path.exists() {
            return LintCache::from_file(&cache_path).ok();
        }
    }
    None
}

/// Result of incremental linting.
#[derive(Debug)]
pub struct IncrementalLintResult {
    /// Directory lint result (combined)
    pub result: DirectoryLintResult,
    /// Files that were actually linted (not cached)
    pub linted_files: usize,
    /// Files retrieved from cache
    pub cached_files: usize,
    /// Updated cache (should be saved)
    pub cache: LintCache,
}

/// Lint a directory with caching for incremental performance.
///
/// This function:
/// 1. Loads existing cache (if valid for current config)
/// 2. Skips unchanged files (returns cached results)
/// 3. Lints changed files
/// 4. Updates cache with new results
pub fn lint_directory_incremental(
    dir: &Path,
    config: LintConfig,
    cache: Option<LintCache>,
) -> IncrementalLintResult {
    use rayon::prelude::*;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    let files = collect_sigil_files(dir);

    // Check if existing cache is valid
    let mut cache = cache
        .filter(|c| c.is_valid_for(&config))
        .unwrap_or_else(|| LintCache::with_config(&config));

    let linted_count = AtomicUsize::new(0);
    let cached_count = AtomicUsize::new(0);
    let total_warnings = AtomicUsize::new(0);
    let total_errors = AtomicUsize::new(0);
    let parse_errors = AtomicUsize::new(0);

    // Collect cache updates: (path, source, cached_diagnostics, metadata)
    let cache_updates: Mutex<Vec<(String, String, Vec<CachedDiagnostic>, Option<std::fs::Metadata>)>> = Mutex::new(Vec::new());

    let file_results: Vec<(String, Diagnostics)> = files
        .par_iter()
        .filter_map(|path| {
            let source = fs::read_to_string(path).ok()?;
            let path_str = path.display().to_string();
            let metadata = fs::metadata(path).ok();

            // Check cache first
            if let Some(cached_diags) = cache.get_cached(&path_str, &source) {
                cached_count.fetch_add(1, Ordering::Relaxed);
                let warnings = cached_diags.iter()
                    .filter(|d| d.severity == Severity::Warning)
                    .count();
                let errors = cached_diags.iter()
                    .filter(|d| d.severity == Severity::Error)
                    .count();
                total_warnings.fetch_add(warnings, Ordering::Relaxed);
                total_errors.fetch_add(errors, Ordering::Relaxed);
                return Some((path_str, cached_diags));
            }

            // Need to lint
            linted_count.fetch_add(1, Ordering::Relaxed);
            let diagnostics = lint_source_with_config(&source, &path_str, config.clone());

            let warnings = diagnostics.iter()
                .filter(|d| d.severity == Severity::Warning)
                .count();
            let errors = diagnostics.iter()
                .filter(|d| d.severity == Severity::Error)
                .count();

            // Parse errors are detected by code prefix P0xx
            let has_parse_error = diagnostics.iter()
                .any(|d| d.code.as_ref().map_or(false, |c| c.starts_with("P0")));
            if has_parse_error {
                parse_errors.fetch_add(1, Ordering::Relaxed);
            }

            total_warnings.fetch_add(warnings, Ordering::Relaxed);
            total_errors.fetch_add(errors, Ordering::Relaxed);

            // Collect cached diagnostics for cache update
            let cached_diags: Vec<CachedDiagnostic> = diagnostics
                .iter()
                .map(CachedDiagnostic::from_diagnostic)
                .collect();

            // Queue cache update
            if let Ok(mut updates) = cache_updates.lock() {
                updates.push((path_str.clone(), source.clone(), cached_diags, metadata));
            }

            Some((path_str, diagnostics))
        })
        .collect();

    // Apply cache updates
    if let Ok(updates) = cache_updates.into_inner() {
        for (path, source, cached_diags, meta) in updates {
            // Reconstruct diagnostics from cached form for the update
            let mut diagnostics = Diagnostics::new();
            for cd in &cached_diags {
                diagnostics.add(cd.to_diagnostic());
            }
            cache.update(&path, &source, &diagnostics, meta.as_ref());
        }
    }

    // Prune stale cache entries
    let existing: HashSet<String> = file_results.iter().map(|(p, _)| p.clone()).collect();
    cache.prune(&existing);

    IncrementalLintResult {
        result: DirectoryLintResult {
            files: file_results,
            total_warnings: total_warnings.load(Ordering::Relaxed),
            total_errors: total_errors.load(Ordering::Relaxed),
            parse_errors: parse_errors.load(Ordering::Relaxed),
        },
        linted_files: linted_count.load(Ordering::Relaxed),
        cached_files: cached_count.load(Ordering::Relaxed),
        cache,
    }
}

// ============================================
// Linter Implementation
// ============================================

/// The main linter struct.
pub struct Linter {
    config: LintConfig,
    diagnostics: Diagnostics,
    declared_vars: HashMap<String, (Span, bool)>,
    declared_imports: HashMap<String, (Span, bool)>,
    /// Scope stack for shadowing detection: each scope has a set of variable names
    scope_stack: Vec<HashSet<String>>,
    /// Current nesting depth for complexity checking
    nesting_depth: usize,
    /// Function parameters for current function: (name, span, used)
    current_fn_params: HashMap<String, (Span, bool)>,
    /// Cyclomatic complexity counter for current function
    current_complexity: usize,
    /// Maximum complexity threshold (configurable)
    max_complexity: usize,
    /// Maximum function length in lines
    max_function_lines: usize,
    /// Maximum number of function parameters
    max_parameters: usize,
    /// Current function line count
    current_fn_lines: usize,
    /// Source code for comment checking
    source_text: String,
    /// Inline suppressions from source
    suppressions: Vec<Suppression>,
    /// Lint statistics
    stats: LintStats,
}

impl Linter {
    pub fn new(config: LintConfig) -> Self {
        Self {
            config,
            diagnostics: Diagnostics::new(),
            declared_vars: HashMap::new(),
            declared_imports: HashMap::new(),
            scope_stack: vec![HashSet::new()], // Start with global scope
            nesting_depth: 0,
            current_fn_params: HashMap::new(),
            current_complexity: 0,
            max_complexity: 10, // Default: warn if complexity > 10
            max_function_lines: 50, // Default: warn if function > 50 lines
            max_parameters: 7, // Default: warn if > 7 parameters
            current_fn_lines: 0,
            source_text: String::new(),
            suppressions: Vec::new(),
            stats: LintStats::default(),
        }
    }

    /// Create a linter with parsed suppressions from source.
    pub fn with_suppressions(config: LintConfig, source: &str) -> Self {
        let mut linter = Self::new(config);
        linter.suppressions = parse_suppressions(source);
        linter.source_text = source.to_string();
        linter
    }

    /// Get lint statistics after linting.
    pub fn stats(&self) -> &LintStats {
        &self.stats
    }

    /// Check if a lint is suppressed at the given line.
    fn is_suppressed(&self, lint: LintId, line: usize) -> bool {
        for suppression in &self.suppressions {
            if suppression.line == line {
                if suppression.lints.is_empty() || suppression.lints.contains(&lint) {
                    return true;
                }
            }
        }
        false
    }

    /// Get line number from a span (1-indexed).
    fn span_to_line(&self, span: Span) -> usize {
        // For now, return 0 (unknown) - would need source text for accurate line calculation
        // Spans contain byte offsets, we'd need to count newlines
        0
    }

    /// Enter a new scope (for shadowing detection)
    fn push_scope(&mut self) {
        self.scope_stack.push(HashSet::new());
    }

    /// Exit current scope
    fn pop_scope(&mut self) {
        self.scope_stack.pop();
    }

    /// Check if a variable would shadow an outer scope variable
    fn check_shadowing(&mut self, name: &str, span: Span) {
        // Skip _prefixed variables (intentional shadowing)
        if name.starts_with('_') {
            return;
        }

        // Check all outer scopes (excluding current)
        for scope in self.scope_stack.iter().rev().skip(1) {
            if scope.contains(name) {
                self.emit(
                    LintId::Shadowing,
                    format!("`{}` shadows a variable from an outer scope", name),
                    span,
                );
                break;
            }
        }

        // Add to current scope
        if let Some(current_scope) = self.scope_stack.last_mut() {
            current_scope.insert(name.to_string());
        }
    }

    /// Enter a nesting level (if, loop, match, etc.)
    fn push_nesting(&mut self, span: Span) {
        self.nesting_depth += 1;
        let max_depth = self.config.max_nesting_depth;
        if self.nesting_depth > max_depth {
            self.emit(
                LintId::DeepNesting,
                format!("nesting depth {} exceeds maximum of {}", self.nesting_depth, max_depth),
                span,
            );
        }
    }

    /// Exit a nesting level
    fn pop_nesting(&mut self) {
        self.nesting_depth = self.nesting_depth.saturating_sub(1);
    }

    pub fn lint(&mut self, file: &SourceFile, source: &str) -> &Diagnostics {
        // Store source for TODO checking
        self.source_text = source.to_string();

        self.visit_source_file(file);
        self.check_unused();

        // Check for TODO comments without issue references
        self.check_todo_comments();

        &self.diagnostics
    }

    fn lint_level(&self, lint: LintId) -> LintLevel {
        self.config
            .levels
            .get(lint.name())
            .copied()
            .unwrap_or_else(|| lint.default_level())
    }

    fn emit(&mut self, lint: LintId, message: impl Into<String>, span: Span) {
        let level = self.lint_level(lint);
        if level == LintLevel::Allow {
            return;
        }

        // Check inline suppressions
        let line = self.span_to_line(span);
        if line > 0 && self.is_suppressed(lint, line) {
            self.stats.record_suppressed();
            return;
        }

        // Record statistics
        self.stats.record(lint);

        let severity = match level {
            LintLevel::Allow => return,
            LintLevel::Warn => Severity::Warning,
            LintLevel::Deny => Severity::Error,
        };

        let diag = Diagnostic {
            severity,
            code: Some(lint.code().to_string()),
            message: message.into(),
            span,
            labels: Vec::new(),
            notes: vec![lint.description().to_string()],
            suggestions: Vec::new(),
            related: Vec::new(),
        };

        self.diagnostics.add(diag);
    }

    fn emit_with_fix(
        &mut self,
        lint: LintId,
        message: impl Into<String>,
        span: Span,
        fix_message: impl Into<String>,
        replacement: impl Into<String>,
    ) {
        let level = self.lint_level(lint);
        if level == LintLevel::Allow {
            return;
        }

        // Check inline suppressions
        let line = self.span_to_line(span);
        if line > 0 && self.is_suppressed(lint, line) {
            self.stats.record_suppressed();
            return;
        }

        // Record statistics
        self.stats.record(lint);

        let severity = match level {
            LintLevel::Allow => return,
            LintLevel::Warn => Severity::Warning,
            LintLevel::Deny => Severity::Error,
        };

        let diag = Diagnostic {
            severity,
            code: Some(lint.code().to_string()),
            message: message.into(),
            span,
            labels: Vec::new(),
            notes: vec![lint.description().to_string()],
            suggestions: vec![FixSuggestion {
                message: fix_message.into(),
                span,
                replacement: replacement.into(),
            }],
            related: Vec::new(),
        };

        self.diagnostics.add(diag);
    }

    fn check_unused(&mut self) {
        let mut unused_vars: Vec<(String, Span)> = Vec::new();
        let mut unused_imports: Vec<(String, Span)> = Vec::new();

        for (name, (span, used)) in &self.declared_vars {
            if !used && !name.starts_with('_') {
                unused_vars.push((name.clone(), *span));
            }
        }

        for (name, (span, used)) in &self.declared_imports {
            if !used {
                unused_imports.push((name.clone(), *span));
            }
        }

        for (name, span) in unused_vars {
            self.emit(
                LintId::UnusedVariable,
                format!("unused variable: `{}`", name),
                span,
            );
        }

        for (name, span) in unused_imports {
            self.emit(
                LintId::UnusedImport,
                format!("unused import: `{}`", name),
                span,
            );
        }
    }

    fn check_reserved(&mut self, name: &str, span: Span) {
        let reserved_suggestions: &[(&str, &str)] = &[
            ("location", "place"),
            ("save", "slot"),
            ("from", "source"),
            ("split", "divide"),
        ];

        for (reserved, suggestion) in reserved_suggestions {
            if name == *reserved {
                self.emit_with_fix(
                    LintId::ReservedIdentifier,
                    format!("`{}` is a reserved word in Sigil", reserved),
                    span,
                    format!("rename to `{}`", suggestion),
                    suggestion.to_string(),
                );
                return;
            }
        }
    }

    fn check_nested_generics(&mut self, ty: &TypeExpr, span: Span) {
        if let TypeExpr::Path(path) = ty {
            for segment in &path.segments {
                if let Some(ref generics) = segment.generics {
                    for arg in generics {
                        if let TypeExpr::Path(inner_path) = arg {
                            for inner_seg in &inner_path.segments {
                                if inner_seg.generics.is_some() {
                                    self.emit(
                                        LintId::NestedGenerics,
                                        "nested generic parameters may not parse correctly",
                                        span,
                                    );
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn check_division(&mut self, op: &BinOp, right: &Expr, span: Span) {
        if let BinOp::Div = op {
            if let Expr::Literal(Literal::Int { value, .. }) = right {
                if value == "0" {
                    self.emit(LintId::DivisionByZero, "division by zero", span);
                }
            }
        }
    }

    fn check_infinite_loop(&mut self, body: &Block, span: Span) {
        if !Self::block_contains_break(body) {
            self.emit(
                LintId::InfiniteLoop,
                "loop has no `break` statement and may run forever",
                span,
            );
        }
    }

    fn block_contains_break(block: &Block) -> bool {
        for stmt in &block.stmts {
            if Self::stmt_contains_break(stmt) {
                return true;
            }
        }
        if let Some(ref expr) = block.expr {
            if Self::expr_contains_break(expr) {
                return true;
            }
        }
        false
    }

    fn stmt_contains_break(stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Expr(e) | Stmt::Semi(e) => Self::expr_contains_break(e),
            Stmt::Let { init, .. } => init.as_ref().map_or(false, Self::expr_contains_break),
            Stmt::LetElse { init, else_branch, .. } => {
                Self::expr_contains_break(init) || Self::expr_contains_break(else_branch)
            }
            Stmt::Item(_) => false,
        }
    }

    fn expr_contains_break(expr: &Expr) -> bool {
        match expr {
            Expr::Break { .. } => true,
            Expr::Return(_) => true,
            Expr::Block(b) => Self::block_contains_break(b),
            Expr::If { then_branch, else_branch, .. } => {
                Self::block_contains_break(then_branch)
                    || else_branch.as_ref().map_or(false, |e| Self::expr_contains_break(e))
            }
            Expr::Match { arms, .. } => arms.iter().any(|arm| Self::expr_contains_break(&arm.body)),
            Expr::Loop { .. } | Expr::While { .. } | Expr::For { .. } => false,
            _ => false,
        }
    }

    /// Check for empty blocks (W0206)
    fn check_empty_block(&mut self, block: &Block, span: Span) {
        if block.stmts.is_empty() && block.expr.is_none() {
            self.emit(
                LintId::EmptyBlock,
                "empty block",
                span,
            );
        }
    }

    /// Check for comparison to boolean literals (W0207)
    /// e.g., `if x == true` should be `if x`
    fn check_bool_comparison(&mut self, op: &BinOp, left: &Expr, right: &Expr, span: Span) {
        let is_eq_or_ne = matches!(op, BinOp::Eq | BinOp::Ne);
        if !is_eq_or_ne {
            return;
        }

        let has_bool_literal = |expr: &Expr| -> Option<bool> {
            if let Expr::Literal(Literal::Bool(value)) = expr {
                Some(*value)
            } else {
                None
            }
        };

        if let Some(val) = has_bool_literal(right) {
            let suggestion = match (op, val) {
                (BinOp::Eq, true) | (BinOp::Ne, false) => "use the expression directly",
                (BinOp::Eq, false) | (BinOp::Ne, true) => "use `!expr` instead",
                _ => "simplify the comparison",
            };
            self.emit(
                LintId::BoolComparison,
                format!("comparison to `{}` is redundant; {}", val, suggestion),
                span,
            );
        } else if let Some(val) = has_bool_literal(left) {
            let suggestion = match (op, val) {
                (BinOp::Eq, true) | (BinOp::Ne, false) => "use the expression directly",
                (BinOp::Eq, false) | (BinOp::Ne, true) => "use `!expr` instead",
                _ => "simplify the comparison",
            };
            self.emit(
                LintId::BoolComparison,
                format!("comparison to `{}` is redundant; {}", val, suggestion),
                span,
            );
        }
    }

    /// Check for redundant else after terminating statement (W0208)
    /// e.g., `if cond { return x; } else { y }` - the else is redundant
    fn check_redundant_else(&mut self, then_branch: &Block, else_branch: &Option<Box<Expr>>, span: Span) {
        if else_branch.is_none() {
            return;
        }

        // Check if then_branch ends with a terminating statement
        let then_terminates = if let Some(ref expr) = then_branch.expr {
            Self::expr_terminates(expr).is_some()
        } else if let Some(last) = then_branch.stmts.last() {
            Self::stmt_terminates(last).is_some()
        } else {
            false
        };

        if then_terminates {
            self.emit(
                LintId::RedundantElse,
                "else branch is redundant after return/break/continue",
                span,
            );
        }
    }

    /// Check for magic numbers (numeric literals that should be constants).
    /// Allows common values: 0, 1, 2, -1, 10, 100, 1000, etc.
    fn check_magic_number(&mut self, value: &str, span: Span) {
        // Common allowed values
        let allowed = ["0", "1", "2", "-1", "10", "100", "1000", "0.0", "1.0", "0.5"];
        if allowed.contains(&value) {
            return;
        }

        // Skip small integers (0-10)
        if let Ok(n) = value.parse::<i64>() {
            if n >= 0 && n <= 10 {
                return;
            }
        }

        self.emit(
            LintId::MagicNumber,
            format!("magic number `{}` should be a named constant", value),
            span,
        );
    }

    /// Increment complexity counter for branching constructs.
    fn add_complexity(&mut self, amount: usize) {
        self.current_complexity += amount;
    }

    /// Check if complexity exceeds threshold and emit warning.
    fn check_complexity(&mut self, func_name: &str, span: Span) {
        if self.current_complexity > self.max_complexity {
            self.emit(
                LintId::HighComplexity,
                format!(
                    "function `{}` has cyclomatic complexity of {} (max: {})",
                    func_name, self.current_complexity, self.max_complexity
                ),
                span,
            );
        }
    }

    /// Check for unused function parameters.
    fn check_unused_params(&mut self) {
        // Collect unused params first to avoid borrow issues
        let unused: Vec<(String, Span)> = self.current_fn_params
            .iter()
            .filter(|(name, (_, used))| !name.starts_with('_') && !used)
            .map(|(name, (span, _))| (name.clone(), *span))
            .collect();

        for (name, span) in unused {
            self.emit_with_fix(
                LintId::UnusedParameter,
                format!("parameter `{}` is never used", name),
                span,
                "prefix with underscore to indicate intentionally unused",
                format!("_{}", name),
            );
        }
    }

    /// Mark a parameter as used.
    fn mark_param_used(&mut self, name: &str) {
        if let Some((_, used)) = self.current_fn_params.get_mut(name) {
            *used = true;
        }
    }

    /// Check for missing doc comments on public items.
    fn check_missing_doc(&mut self, vis: &Visibility, name: &str, span: Span) {
        // Only check pub items
        if !matches!(vis, Visibility::Public) {
            return;
        }

        // This would need access to doc comments in the AST
        // For now, we emit for all public items without attached docs
        // The parser would need to preserve doc comments for full implementation
        self.emit(
            LintId::MissingDocComment,
            format!("public item `{}` should have a documentation comment", name),
            span,
        );
    }

    /// Check for TODO comments without issue references.
    fn check_todo_comments(&mut self) {
        // Pattern: TODO without (#123) or (GH-123) or (ISSUE-123)
        let issue_pattern = regex::Regex::new(r"TODO\s*\([#A-Z]+-?\d+\)").unwrap();
        let todo_pattern = regex::Regex::new(r"//.*\bTODO\b").unwrap();

        // Clone source text to avoid borrow conflict
        let source = self.source_text.clone();
        for line in source.lines() {
            if todo_pattern.is_match(line) && !issue_pattern.is_match(line) {
                // Found a TODO without issue reference
                self.emit(
                    LintId::TodoWithoutIssue,
                    "TODO comment should reference an issue (e.g., TODO(#123):)",
                    Span::default(),
                );
            }
        }
    }

    /// Check function length.
    fn check_function_length(&mut self, func_name: &str, span: Span, line_count: usize) {
        if line_count > self.max_function_lines {
            self.emit(
                LintId::LongFunction,
                format!(
                    "function `{}` has {} lines (max: {})",
                    func_name, line_count, self.max_function_lines
                ),
                span,
            );
        }
    }

    /// Check parameter count.
    fn check_parameter_count(&mut self, func_name: &str, span: Span, param_count: usize) {
        if param_count > self.max_parameters {
            self.emit(
                LintId::TooManyParameters,
                format!(
                    "function `{}` has {} parameters (max: {})",
                    func_name, param_count, self.max_parameters
                ),
                span,
            );
        }
    }

    /// Check for needless return at end of function.
    fn check_needless_return(&mut self, body: &Block, span: Span) {
        // Check if the last statement/expression is an unnecessary return
        if let Some(ref expr) = body.expr {
            if let Expr::Return(Some(_)) = &**expr {
                self.emit(
                    LintId::NeedlessReturn,
                    "unnecessary return statement; the last expression is automatically returned",
                    span,
                );
            }
        } else if let Some(last) = body.stmts.last() {
            match last {
                Stmt::Semi(Expr::Return(Some(_))) | Stmt::Expr(Expr::Return(Some(_))) => {
                    self.emit(
                        LintId::NeedlessReturn,
                        "unnecessary return statement; the last expression is automatically returned",
                        span,
                    );
                }
                _ => {}
            }
        }
    }

    /// Check for missing return in functions with return types (W0300).
    ///
    /// A function may not return a value on all paths if:
    /// - An if without else doesn't return in all branches
    /// - A match doesn't cover all cases with returns
    /// - Early returns leave some paths without values
    fn check_missing_return(&mut self, body: &Block, has_return_type: bool, func_name: &str, span: Span) {
        if !has_return_type {
            return; // Unit functions don't need return checks
        }

        // Check if the body always produces a value
        if !Self::block_always_returns(body) {
            self.emit(
                LintId::MissingReturn,
                format!("function `{}` may not return a value on all code paths", func_name),
                span,
            );
        }
    }

    /// Check if a block always produces a value (returns or evaluates to expression).
    fn block_always_returns(block: &Block) -> bool {
        // If block has a trailing expression, it returns (unless it's a unit-producing expr)
        if let Some(ref expr) = block.expr {
            return Self::expr_always_returns(expr);
        }

        // Otherwise check if all paths through statements lead to returns
        // Check if any statement terminates
        for stmt in &block.stmts {
            if Self::stmt_always_returns(stmt) {
                return true;
            }
        }

        false
    }

    /// Check if a statement always returns.
    fn stmt_always_returns(stmt: &Stmt) -> bool {
        match stmt {
            Stmt::Expr(e) | Stmt::Semi(e) => Self::expr_always_returns(e),
            _ => false,
        }
    }

    /// Check if an expression always produces a value or terminates.
    fn expr_always_returns(expr: &Expr) -> bool {
        match expr {
            // Direct terminators
            Expr::Return(_) => true,
            Expr::Break { .. } => true, // In loop context
            Expr::Continue { .. } => true, // In loop context

            // Block: check if block returns
            Expr::Block(b) => Self::block_always_returns(b),

            // If: both branches must return
            Expr::If { then_branch, else_branch, .. } => {
                if let Some(ref else_expr) = else_branch {
                    Self::block_always_returns(then_branch) && Self::expr_always_returns(else_expr)
                } else {
                    false // No else means it might not produce a value
                }
            }

            // Match: all arms must return (or be unreachable)
            Expr::Match { arms, .. } => {
                if arms.is_empty() {
                    false
                } else {
                    arms.iter().all(|arm| Self::expr_always_returns(&arm.body))
                }
            }

            // Loop is more complex - we assume it might not return
            // (proper analysis would check break values)
            Expr::Loop { .. } => false,
            Expr::While { .. } => false,
            Expr::For { .. } => false,

            // Most expressions produce values
            Expr::Literal(_) => true,
            Expr::Path(_) => true,
            Expr::Binary { .. } => true,
            Expr::Unary { .. } => true,
            Expr::Call { .. } => true,
            Expr::MethodCall { .. } => true,
            Expr::Field { .. } => true,
            Expr::Index { .. } => true,
            Expr::Array(_) => true,
            Expr::Tuple(_) => true,
            Expr::Struct { .. } => true,
            Expr::Range { .. } => true,
            Expr::Cast { .. } => true,
            Expr::AddrOf { .. } => true,
            Expr::Deref(_) => true,
            Expr::Closure { .. } => true,
            Expr::Await { .. } => true,
            Expr::Try(_) => true,
            Expr::Morpheme { .. } => true,
            Expr::Pipe { .. } => true,
            Expr::Unsafe(b) => Self::block_always_returns(b),
            Expr::Evidential { .. } => true,
            Expr::Incorporation { .. } => true,
            Expr::Let { .. } => true,

            // Assign produces unit, not a value
            Expr::Assign { .. } => false,

            // Default: assume it might not return
            _ => false,
        }
    }

    /// Check for method chains that could use morpheme pipeline syntax (W0500).
    ///
    /// Detects patterns like: data.iter().map(...).filter(...).collect()
    /// And suggests: data |τ{...} |φ{...} |σ
    fn check_prefer_morpheme_pipeline(&mut self, expr: &Expr, span: Span) {
        // Count consecutive method calls
        let chain_length = Self::method_chain_length(expr);

        // Suggest morpheme pipeline for chains of 2+ transformations
        if chain_length >= 2 {
            // Check if any methods are transformable to morphemes
            let transformable_methods = Self::count_transformable_methods(expr);
            if transformable_methods >= 2 {
                self.emit(
                    LintId::PreferMorphemePipeline,
                    format!(
                        "consider using morpheme pipeline (|τ{{}}, |φ{{}}) for this {}-method chain",
                        chain_length
                    ),
                    span,
                );
            }
        }
    }

    /// Count the length of a method call chain.
    fn method_chain_length(expr: &Expr) -> usize {
        match expr {
            Expr::MethodCall { receiver, .. } => {
                1 + Self::method_chain_length(receiver)
            }
            _ => 0,
        }
    }

    /// Count methods in a chain that could be replaced with morpheme operators.
    fn count_transformable_methods(expr: &Expr) -> usize {
        let transformable = ["map", "filter", "fold", "reduce", "collect", "sort", "first", "last", "zip", "iter"];

        match expr {
            Expr::MethodCall { receiver, method, .. } => {
                let count = if transformable.contains(&method.name.as_str()) { 1 } else { 0 };
                count + Self::count_transformable_methods(receiver)
            }
            _ => 0,
        }
    }

    /// Check for constant conditions (if true, while false, etc.).
    fn check_constant_condition(&mut self, condition: &Expr, span: Span) {
        let is_constant = match condition {
            Expr::Literal(Literal::Bool(val)) => Some(*val),
            Expr::Path(p) if p.segments.len() == 1 => {
                let name = &p.segments[0].ident.name;
                if name == "true" {
                    Some(true)
                } else if name == "false" {
                    Some(false)
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(val) = is_constant {
            self.emit(
                LintId::ConstantCondition,
                format!("condition is always `{}`", val),
                span,
            );
        }
    }

    /// Check for match expressions that could be if-let.
    fn check_prefer_if_let(&mut self, arms: &[MatchArm], span: Span) {
        // If match has exactly 2 arms and one is a wildcard, suggest if-let
        if arms.len() == 2 {
            let has_wildcard = arms.iter().any(|arm| {
                matches!(&arm.pattern, Pattern::Wildcard)
            });
            if has_wildcard {
                self.emit(
                    LintId::PreferIfLet,
                    "consider using `if let` instead of `match` with wildcard",
                    span,
                );
            }
        }
    }

    // ============================================
    // Aether 2.0 Enhanced Lint Checks
    // ============================================

    /// Check for I Ching hexagram number validity (1-64).
    fn check_hexagram_number(&mut self, value: i64, span: Span) {
        if value < 1 || value > 64 {
            self.emit(
                LintId::InvalidHexagramNumber,
                format!("hexagram number {} is invalid (must be 1-64)", value),
                span,
            );
        }
    }

    /// Check for Tarot Major Arcana number validity (0-21).
    fn check_tarot_number(&mut self, value: i64, span: Span) {
        if value < 0 || value > 21 {
            self.emit(
                LintId::InvalidTarotNumber,
                format!("Major Arcana number {} is invalid (must be 0-21)", value),
                span,
            );
        }
    }

    /// Check for chakra index validity (0-6).
    fn check_chakra_index(&mut self, value: i64, span: Span) {
        if value < 0 || value > 6 {
            self.emit(
                LintId::InvalidChakraIndex,
                format!("chakra index {} is invalid (must be 0-6)", value),
                span,
            );
        }
    }

    /// Check for zodiac sign index validity (0-11).
    fn check_zodiac_index(&mut self, value: i64, span: Span) {
        if value < 0 || value > 11 {
            self.emit(
                LintId::InvalidZodiacIndex,
                format!("zodiac index {} is invalid (must be 0-11)", value),
                span,
            );
        }
    }

    /// Check for gematria value validity (non-negative).
    fn check_gematria_value(&mut self, value: i64, span: Span) {
        if value < 0 {
            self.emit(
                LintId::InvalidGematriaValue,
                format!("gematria value {} is invalid (must be non-negative)", value),
                span,
            );
        }
    }

    /// Check for audio frequency range (20Hz-20kHz audible range).
    fn check_frequency_range(&mut self, value: f64, span: Span) {
        if value < 20.0 || value > 20000.0 {
            self.emit(
                LintId::FrequencyOutOfRange,
                format!("frequency {:.2}Hz is outside audible range (20Hz-20kHz)", value),
                span,
            );
        }
    }

    /// Check for emotion intensity range (0.0-1.0).
    fn check_emotion_intensity(&mut self, value: f64, span: Span) {
        if value < 0.0 || value > 1.0 {
            self.emit(
                LintId::EmotionIntensityOutOfRange,
                format!("emotion intensity {:.2} is invalid (must be 0.0-1.0)", value),
                span,
            );
        }
    }

    /// Check for esoteric magic numbers that should be named constants.
    fn check_esoteric_constant(&mut self, value: &str, span: Span) {
        // Common esoteric constants
        let esoteric_values = [
            ("1.618", "GOLDEN_RATIO or PHI"),
            ("0.618", "GOLDEN_RATIO_INVERSE"),
            ("1.414", "SQRT_2 or SILVER_RATIO"),
            ("2.414", "SILVER_RATIO"),
            ("3.14159", "PI"),
            ("2.71828", "E or EULER"),
            ("432", "VERDI_PITCH or A432"),
            ("440", "CONCERT_PITCH or A440"),
            ("528", "SOLFEGGIO_MI or LOVE_FREQUENCY"),
            ("396", "SOLFEGGIO_UT"),
            ("639", "SOLFEGGIO_FA"),
            ("741", "SOLFEGGIO_SOL"),
            ("852", "SOLFEGGIO_LA"),
            ("963", "SOLFEGGIO_SI"),
        ];

        for (pattern, suggestion) in esoteric_values {
            if value.starts_with(pattern) {
                self.emit(
                    LintId::PreferNamedEsotericConstant,
                    format!("consider using named constant {} instead of {}", suggestion, value),
                    span,
                );
                return;
            }
        }
    }

    /// Check for inconsistent morpheme style (mixing |τ{} with method chains).
    fn check_morpheme_style_consistency(&mut self, expr: &Expr, span: Span) {
        let has_morpheme = Self::has_morpheme_pipeline(expr);
        let has_method_chain = Self::method_chain_length(expr) >= 2;

        if has_morpheme && has_method_chain {
            self.emit(
                LintId::InconsistentMorphemeStyle,
                "mixing morpheme pipeline (|τ{}) with method chains; prefer one style",
                span,
            );
        }
    }

    /// Check if expression contains morpheme pipeline operators.
    fn has_morpheme_pipeline(expr: &Expr) -> bool {
        match expr {
            Expr::Morpheme { .. } => true,
            Expr::Pipe { .. } => true,
            Expr::MethodCall { receiver, .. } => Self::has_morpheme_pipeline(receiver),
            Expr::Binary { left, right, .. } => {
                Self::has_morpheme_pipeline(left) || Self::has_morpheme_pipeline(right)
            }
            _ => false,
        }
    }

    /// Detect domain-specific numeric literals for validation.
    fn check_domain_literal(&mut self, func_name: &str, value: i64, span: Span) {
        // Detect by function/context naming patterns
        let name_lower = func_name.to_lowercase();

        if name_lower.contains("hexagram") || name_lower.contains("iching") {
            self.check_hexagram_number(value, span);
        } else if name_lower.contains("arcana") || name_lower.contains("tarot") {
            self.check_tarot_number(value, span);
        } else if name_lower.contains("chakra") {
            self.check_chakra_index(value, span);
        } else if name_lower.contains("zodiac") || name_lower.contains("sign") {
            self.check_zodiac_index(value, span);
        } else if name_lower.contains("gematria") {
            self.check_gematria_value(value, span);
        }
    }

    /// Check for domain-specific float literals.
    fn check_domain_float_literal(&mut self, func_name: &str, value: f64, span: Span) {
        let name_lower = func_name.to_lowercase();

        if name_lower.contains("frequency") || name_lower.contains("hz") || name_lower.contains("hertz") {
            self.check_frequency_range(value, span);
        } else if name_lower.contains("intensity") || name_lower.contains("emotion") {
            self.check_emotion_intensity(value, span);
        }
    }

    // ============================================
    // Evidentiality Checking
    // ============================================

    /// External data sources that require evidentiality markers.
    /// Returns (pattern, suggested_marker, marker_symbol, rationale)
    const EXTERNAL_DATA_SOURCES: &'static [(&'static str, &'static str, &'static str, &'static str)] = &[
        // HTTP/Network - data from external systems
        ("Http·get", "Reported", "~", "HTTP responses come from external systems"),
        ("Http·post", "Reported", "~", "HTTP responses come from external systems"),
        ("Http·request", "Reported", "~", "HTTP responses come from external systems"),
        ("HttpClient·get", "Reported", "~", "HTTP responses come from external systems"),
        ("HttpClient·post", "Reported", "~", "HTTP responses come from external systems"),
        ("WebSocket·connect", "Reported", "~", "WebSocket data comes from external systems"),
        ("WebSocket·recv", "Reported", "~", "WebSocket data comes from external systems"),
        ("TcpStream·connect", "Reported", "~", "Network data comes from external systems"),
        ("TcpStream·read", "Reported", "~", "Network data comes from external systems"),
        ("UdpSocket·recv", "Reported", "~", "Network data comes from external systems"),

        // File I/O - data from filesystem
        ("File·read", "Reported", "~", "file contents may have changed externally"),
        ("File·open", "Reported", "~", "file existence/contents are external state"),
        ("File·read_to_string", "Reported", "~", "file contents may have changed externally"),
        ("Fs·read", "Reported", "~", "file contents may have changed externally"),
        ("Fs·read_dir", "Reported", "~", "directory contents are external state"),

        // User input - unverified data
        ("stdin·read", "Uncertain", "?", "user input is unverified"),
        ("stdin·read_line", "Uncertain", "?", "user input is unverified"),
        ("Stdin·read", "Uncertain", "?", "user input is unverified"),
        ("Env·var", "Uncertain", "?", "environment variables are external input"),
        ("Env·args", "Uncertain", "?", "command line arguments are external input"),

        // Database - external persistent state
        ("Db·query", "Reported", "~", "database contents are external state"),
        ("Db·execute", "Reported", "~", "database results reflect external state"),
        ("Sql·query", "Reported", "~", "database contents are external state"),
        ("Redis·get", "Reported", "~", "cache contents are external state"),

        // System calls - external system state
        ("Sys·read", "Reported", "~", "system call returns external data"),
        ("Sys·recv", "Reported", "~", "network data is external"),
        ("Sys·recvfrom", "Reported", "~", "network data is external"),

        // Time - external world state
        ("Time·now", "Reported", "~", "current time is external state"),
        ("Instant·now", "Reported", "~", "current time is external state"),
        ("SystemTime·now", "Reported", "~", "current time is external state"),

        // Random - non-deterministic
        ("Random·next", "Uncertain", "?", "random values are non-deterministic"),
        ("Rng·gen", "Uncertain", "?", "random values are non-deterministic"),
        ("rand", "Uncertain", "?", "random values are non-deterministic"),

        // ML/AI predictions
        ("Model·predict", "Predicted", "◊", "ML predictions are probabilistic"),
        ("Model·infer", "Predicted", "◊", "ML inference is probabilistic"),
        ("Llm·complete", "Predicted", "◊", "LLM outputs are probabilistic"),
        ("Llm·chat", "Predicted", "◊", "LLM outputs are probabilistic"),

        // JSON/Parsing - may fail or be malformed
        ("Json·parse", "Uncertain", "?", "parsed data may be malformed"),
        ("Toml·parse", "Uncertain", "?", "parsed data may be malformed"),
        ("Yaml·parse", "Uncertain", "?", "parsed data may be malformed"),
        ("Xml·parse", "Uncertain", "?", "parsed data may be malformed"),
    ];

    /// Check if a function call is to an external data source and emit lint if unmarked.
    #[allow(dead_code)]
    fn check_external_data_source(&mut self, func_name: &str, has_evidentiality: bool, span: Span) {
        if has_evidentiality {
            return; // Already marked, nothing to do
        }

        for (pattern, marker_name, marker_symbol, rationale) in Self::EXTERNAL_DATA_SOURCES {
            if func_name == *pattern || func_name.ends_with(&format!("·{}", pattern.split('·').last().unwrap_or(pattern))) {
                self.emit_with_fix(
                    LintId::UnvalidatedExternalData,
                    format!(
                        "external data source `{}` requires evidentiality marker",
                        func_name
                    ),
                    span,
                    format!(
                        "add `{}` ({}) marker: {}",
                        marker_symbol, marker_name, rationale
                    ),
                    format!("{}  // mark result with {}", func_name, marker_symbol),
                );
                return;
            }
        }
    }

    /// Check a let binding for missing evidentiality on external data.
    fn check_let_evidentiality(&mut self, var_name: &str, var_evidentiality: Option<&crate::ast::Evidentiality>, init_expr: &Expr, span: Span) {
        // Check if the init expression is a call to an external data source
        if let Some(func_name) = Self::extract_call_name(init_expr) {
            for (pattern, marker_name, marker_symbol, rationale) in Self::EXTERNAL_DATA_SOURCES {
                if func_name == *pattern || func_name.contains(pattern) {
                    let expected_marker = Self::symbol_to_evidentiality(marker_symbol);

                    match var_evidentiality {
                        None => {
                            // Missing marker - emit error with fix
                            self.emit_with_fix(
                                LintId::UnvalidatedExternalData,
                                format!(
                                    "variable `{}` receives external data from `{}` without evidentiality marker",
                                    var_name, func_name
                                ),
                                span,
                                format!(
                                    "mark variable with `{}` ({}) suffix: `{}{}`\n   = note: {}",
                                    marker_symbol, marker_name, var_name, marker_symbol, rationale
                                ),
                                format!("{}{}", var_name, marker_symbol),
                            );
                        }
                        Some(actual_marker) => {
                            // Marker present - check if it's correct
                            if let Some(expected) = expected_marker {
                                if *actual_marker != expected {
                                    let actual_symbol = Self::evidentiality_to_symbol(actual_marker);
                                    let actual_name = Self::evidentiality_to_name(actual_marker);
                                    self.emit_with_fix(
                                        LintId::EvidentialityMismatch,
                                        format!(
                                            "variable `{}` has incorrect evidentiality marker `{}` ({}) for data from `{}`",
                                            var_name, actual_symbol, actual_name, func_name
                                        ),
                                        span,
                                        format!(
                                            "change marker to `{}` ({}): `{}{}`\n   = note: {}\n   = note: `{}` implies {} but {} data is {}",
                                            marker_symbol, marker_name, var_name, marker_symbol, rationale,
                                            actual_symbol, actual_name, pattern, marker_name
                                        ),
                                        format!("{}{}", var_name, marker_symbol),
                                    );
                                }
                            }
                        }
                    }
                    return;
                }
            }
        }
    }

    /// Convert evidentiality symbol to enum variant
    fn symbol_to_evidentiality(symbol: &str) -> Option<crate::ast::Evidentiality> {
        match symbol {
            "!" => Some(crate::ast::Evidentiality::Known),
            "?" => Some(crate::ast::Evidentiality::Uncertain),
            "~" => Some(crate::ast::Evidentiality::Reported),
            "◊" => Some(crate::ast::Evidentiality::Predicted),
            "‽" => Some(crate::ast::Evidentiality::Paradox),
            _ => None,
        }
    }

    /// Convert evidentiality enum to symbol
    fn evidentiality_to_symbol(ev: &crate::ast::Evidentiality) -> &'static str {
        match ev {
            crate::ast::Evidentiality::Known => "!",
            crate::ast::Evidentiality::Uncertain => "?",
            crate::ast::Evidentiality::Reported => "~",
            crate::ast::Evidentiality::Predicted => "◊",
            crate::ast::Evidentiality::Paradox => "‽",
        }
    }

    /// Convert evidentiality enum to human-readable name
    fn evidentiality_to_name(ev: &crate::ast::Evidentiality) -> &'static str {
        match ev {
            crate::ast::Evidentiality::Known => "Known/Verified",
            crate::ast::Evidentiality::Uncertain => "Uncertain/Unverified",
            crate::ast::Evidentiality::Reported => "Reported/External",
            crate::ast::Evidentiality::Predicted => "Predicted/Speculative",
            crate::ast::Evidentiality::Paradox => "Paradox/Contradiction",
        }
    }

    /// Extract the function name from a call expression.
    fn extract_call_name(expr: &Expr) -> Option<String> {
        match expr {
            Expr::Call { func, .. } => {
                match func.as_ref() {
                    Expr::Path(path) => {
                        Some(path.segments.iter()
                            .map(|s| s.ident.name.clone())
                            .collect::<Vec<_>>()
                            .join("·"))
                    }
                    Expr::Field { expr: base, field, .. } => {
                        if let Some(base_name) = Self::extract_call_name(base) {
                            Some(format!("{}·{}", base_name, field.name))
                        } else {
                            Some(field.name.clone())
                        }
                    }
                    _ => None,
                }
            }
            Expr::MethodCall { receiver, method, .. } => {
                if let Some(receiver_type) = Self::extract_type_name(receiver) {
                    Some(format!("{}·{}", receiver_type, method.name))
                } else {
                    Some(method.name.clone())
                }
            }
            Expr::Await { expr: inner, .. } => Self::extract_call_name(inner),
            Expr::Try(inner) => Self::extract_call_name(inner),
            _ => None,
        }
    }

    /// Try to extract a type name from an expression (for method calls).
    fn extract_type_name(expr: &Expr) -> Option<String> {
        match expr {
            Expr::Path(path) => {
                Some(path.segments.iter()
                    .map(|s| s.ident.name.clone())
                    .collect::<Vec<_>>()
                    .join("·"))
            }
            Expr::Call { func, .. } => Self::extract_type_name(func),
            _ => None,
        }
    }

    // AST Visitor methods
    fn visit_source_file(&mut self, file: &SourceFile) {
        for item in &file.items {
            self.visit_item(&item.node);
        }
    }

    fn visit_item(&mut self, item: &Item) {
        match item {
            Item::Function(f) => self.visit_function(f),
            Item::Struct(s) => self.visit_struct(s),
            Item::Module(m) => self.visit_module(m),
            _ => {}
        }
    }

    fn visit_function(&mut self, func: &Function) {
        self.check_reserved(&func.name.name, func.name.span);

        // Check for missing doc comment on public functions
        self.check_missing_doc(&func.visibility, &func.name.name, func.name.span);

        // Check parameter count
        self.check_parameter_count(&func.name.name, func.name.span, func.params.len());

        // Reset complexity counter for this function
        self.current_complexity = 1; // Base complexity is 1

        // Clear and populate parameter tracking
        self.current_fn_params.clear();

        // Push function scope for parameters
        self.push_scope();

        for param in &func.params {
            // Add parameter to scope (for shadowing detection in body)
            if let Pattern::Ident { name, .. } = &param.pattern {
                if let Some(scope) = self.scope_stack.last_mut() {
                    scope.insert(name.name.clone());
                }
                // Track parameter for unused detection
                self.current_fn_params.insert(name.name.clone(), (name.span, false));
            }
            self.visit_pattern(&param.pattern);
        }

        if let Some(ref body) = func.body {
            // Check for needless return
            self.check_needless_return(body, func.name.span);

            // Check for missing return (function has return type but may not return on all paths)
            let has_return_type = func.return_type.is_some();
            self.check_missing_return(body, has_return_type, &func.name.name, func.name.span);

            // Line count estimate based on statements
            let line_estimate = body.stmts.len() + if body.expr.is_some() { 1 } else { 0 } + 2; // +2 for fn signature and closing brace
            self.check_function_length(&func.name.name, func.name.span, line_estimate);

            self.visit_block(body);
        }

        // Check for unused parameters
        self.check_unused_params();

        // Check complexity threshold
        self.check_complexity(&func.name.name, func.name.span);

        self.pop_scope();
    }

    fn visit_struct(&mut self, s: &StructDef) {
        self.check_reserved(&s.name.name, s.name.span);

        if let StructFields::Named(ref fields) = s.fields {
            for field in fields {
                self.check_reserved(&field.name.name, field.name.span);
                self.check_nested_generics(&field.ty, field.name.span);
            }
        }
    }

    fn visit_module(&mut self, m: &Module) {
        if let Some(ref items) = m.items {
            for item in items {
                self.visit_item(&item.node);
            }
        }
    }

    fn visit_block(&mut self, block: &Block) {
        self.push_scope();

        let mut found_terminator = false;

        for stmt in &block.stmts {
            // Check for unreachable code
            if found_terminator {
                if let Some(span) = Self::stmt_span(stmt) {
                    self.emit(
                        LintId::UnreachableCode,
                        "unreachable statement after return/break/continue",
                        span,
                    );
                }
            }

            self.visit_stmt(stmt);

            // Check if this statement terminates control flow
            if !found_terminator {
                if Self::stmt_terminates(stmt).is_some() {
                    found_terminator = true;
                }
            }
        }

        // Check trailing expression for unreachability
        if let Some(ref expr) = block.expr {
            if found_terminator {
                if let Some(span) = Self::expr_span(expr) {
                    self.emit(
                        LintId::UnreachableCode,
                        "unreachable expression after return/break/continue",
                        span,
                    );
                }
            }
            self.visit_expr(expr);
        }

        self.pop_scope();
    }

    /// Get span from a statement if possible
    fn stmt_span(stmt: &Stmt) -> Option<Span> {
        match stmt {
            Stmt::Let { pattern, .. } => {
                if let Pattern::Ident { name, .. } = pattern {
                    Some(name.span)
                } else {
                    None
                }
            }
            Stmt::LetElse { pattern, .. } => {
                if let Pattern::Ident { name, .. } = pattern {
                    Some(name.span)
                } else {
                    None
                }
            }
            Stmt::Expr(e) | Stmt::Semi(e) => Self::expr_span(e),
            Stmt::Item(_) => None,
        }
    }

    /// Get span from an expression if possible
    fn expr_span(expr: &Expr) -> Option<Span> {
        match expr {
            Expr::Return(_) => Some(Span::default()),
            Expr::Break { .. } => Some(Span::default()),
            Expr::Continue { .. } => Some(Span::default()),
            Expr::Path(p) if !p.segments.is_empty() => Some(p.segments[0].ident.span),
            // Literals don't have spans in AST, use default
            Expr::Literal(_) => Some(Span::default()),
            _ => Some(Span::default()), // Default span for other expressions
        }
    }

    /// Check if a statement terminates control flow, return the span if so
    fn stmt_terminates(stmt: &Stmt) -> Option<Span> {
        match stmt {
            Stmt::Expr(e) | Stmt::Semi(e) => Self::expr_terminates(e),
            _ => None,
        }
    }

    /// Check if an expression terminates control flow
    fn expr_terminates(expr: &Expr) -> Option<Span> {
        match expr {
            Expr::Return(_) => Some(Span::default()),
            Expr::Break { .. } => Some(Span::default()),
            Expr::Continue { .. } => Some(Span::default()),
            Expr::Block(b) => {
                // Block terminates if it ends with a terminating expression
                if let Some(ref e) = b.expr {
                    Self::expr_terminates(e)
                } else if let Some(last) = b.stmts.last() {
                    Self::stmt_terminates(last)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn visit_stmt(&mut self, stmt: &Stmt) {
        match stmt {
            Stmt::Let { pattern, init, .. } => {
                if let Pattern::Ident { name, evidentiality, .. } = pattern {
                    self.check_reserved(&name.name, name.span);
                    self.check_shadowing(&name.name, name.span);
                    self.declared_vars.insert(name.name.clone(), (name.span, false));

                    // Check for external data sources without evidentiality markers
                    // Note: evidentiality can be stored in Pattern::Ident.evidentiality OR in Ident.evidentiality
                    // The parser stores unambiguous markers (~, ◊, ‽) in Ident.evidentiality
                    // and ambiguous markers (!, ?) in Pattern.evidentiality
                    let effective_evidentiality = evidentiality.as_ref().or(name.evidentiality.as_ref());
                    if let Some(ref init_expr) = init {
                        self.check_let_evidentiality(&name.name, effective_evidentiality, init_expr, name.span);
                    }
                }
                self.visit_pattern(pattern);
                if let Some(ref e) = init {
                    self.visit_expr(e);
                }
            }
            Stmt::LetElse { pattern, init, else_branch, .. } => {
                if let Pattern::Ident { name, evidentiality, .. } = pattern {
                    self.check_reserved(&name.name, name.span);
                    self.check_shadowing(&name.name, name.span);
                    self.declared_vars.insert(name.name.clone(), (name.span, false));

                    // Check for external data sources without evidentiality markers
                    // Note: evidentiality can be in Pattern or Ident (see Stmt::Let comment)
                    let effective_evidentiality = evidentiality.as_ref().or(name.evidentiality.as_ref());
                    self.check_let_evidentiality(&name.name, effective_evidentiality, init, name.span);
                }
                self.visit_pattern(pattern);
                self.visit_expr(init);
                self.visit_expr(else_branch);
            }
            Stmt::Expr(e) | Stmt::Semi(e) => self.visit_expr(e),
            Stmt::Item(item) => self.visit_item(item),
        }
    }

    fn visit_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Path(path) => {
                if path.segments.len() == 1 {
                    let name = &path.segments[0].ident.name;
                    if let Some((_, used)) = self.declared_vars.get_mut(name) {
                        *used = true;
                    }
                    // Also mark parameters as used
                    self.mark_param_used(name);
                }
            }
            Expr::Literal(lit) => {
                // Check for magic numbers
                match lit {
                    Literal::Int { value, .. } => {
                        self.check_magic_number(value, Span::default());
                    }
                    Literal::Float { value, .. } => {
                        self.check_magic_number(value, Span::default());
                    }
                    _ => {}
                }
            }
            Expr::Binary { op, left, right, .. } => {
                self.check_division(op, right, Span::default());
                self.check_bool_comparison(op, left, right, Span::default());
                // Count && and || as complexity points
                if matches!(op, BinOp::And | BinOp::Or) {
                    self.add_complexity(1);
                }
                self.visit_expr(left);
                self.visit_expr(right);
            }
            Expr::Loop { body, .. } => {
                self.push_nesting(Span::default());
                self.add_complexity(1); // Loop adds complexity
                self.check_infinite_loop(body, Span::default());
                self.check_empty_block(body, Span::default());
                self.visit_block(body);
                self.pop_nesting();
            }
            Expr::Block(b) => {
                self.check_empty_block(b, Span::default());
                self.visit_block(b);
            }
            Expr::If { condition, then_branch, else_branch, .. } => {
                self.push_nesting(Span::default());
                self.add_complexity(1); // If adds complexity
                self.check_constant_condition(condition, Span::default());
                self.check_redundant_else(then_branch, else_branch, Span::default());
                self.check_empty_block(then_branch, Span::default());
                self.visit_expr(condition);
                self.visit_block(then_branch);
                if let Some(ref e) = else_branch {
                    self.visit_expr(e);
                }
                self.pop_nesting();
            }
            Expr::Match { expr: match_expr, arms, .. } => {
                self.push_nesting(Span::default());
                self.check_prefer_if_let(arms, Span::default());
                // Each match arm adds complexity (minus 1 for the base)
                if !arms.is_empty() {
                    self.add_complexity(arms.len().saturating_sub(1));
                }
                self.visit_expr(match_expr);
                for arm in arms {
                    self.visit_pattern(&arm.pattern);
                    if let Some(ref guard) = arm.guard {
                        self.add_complexity(1); // Guard adds complexity
                        self.visit_expr(guard);
                    }
                    self.visit_expr(&arm.body);
                }
                self.pop_nesting();
            }
            Expr::While { condition, body, .. } => {
                self.push_nesting(Span::default());
                self.add_complexity(1); // While adds complexity
                self.check_constant_condition(condition, Span::default());
                self.visit_expr(condition);
                self.visit_block(body);
                self.pop_nesting();
            }
            Expr::For { pattern, iter, body, .. } => {
                self.push_nesting(Span::default());
                self.add_complexity(1); // For adds complexity
                self.visit_pattern(pattern);
                self.visit_expr(iter);
                self.visit_block(body);
                self.pop_nesting();
            }
            Expr::Call { func, args, .. } => {
                self.visit_expr(func);
                for arg in args {
                    self.visit_expr(arg);
                }
            }
            Expr::MethodCall { receiver, args, .. } => {
                // Check for method chains that could be morpheme pipelines
                self.check_prefer_morpheme_pipeline(expr, Span::default());
                self.visit_expr(receiver);
                for arg in args {
                    self.visit_expr(arg);
                }
            }
            Expr::Field { expr: field_expr, .. } => self.visit_expr(field_expr),
            Expr::Index { expr: idx_expr, index, .. } => {
                self.visit_expr(idx_expr);
                self.visit_expr(index);
            }
            Expr::Array(elements) | Expr::Tuple(elements) => {
                for e in elements {
                    self.visit_expr(e);
                }
            }
            Expr::Struct { fields, rest, .. } => {
                for field in fields {
                    if let Some(ref value) = field.value {
                        self.visit_expr(value);
                    }
                }
                if let Some(ref b) = rest {
                    self.visit_expr(b);
                }
            }
            Expr::Range { start, end, .. } => {
                if let Some(ref s) = start {
                    self.visit_expr(s);
                }
                if let Some(ref e) = end {
                    self.visit_expr(e);
                }
            }
            Expr::Return(e) => {
                if let Some(ref ret_expr) = e {
                    self.visit_expr(ret_expr);
                }
            }
            Expr::Break { value, .. } => {
                if let Some(ref brk_expr) = value {
                    self.visit_expr(brk_expr);
                }
            }
            Expr::Assign { target, value, .. } => {
                self.visit_expr(target);
                self.visit_expr(value);
            }
            Expr::AddrOf { expr: addr_expr, .. } => self.visit_expr(addr_expr),
            Expr::Deref(e) => self.visit_expr(e),
            Expr::Cast { expr: cast_expr, .. } => self.visit_expr(cast_expr),
            Expr::Closure { params, body, .. } => {
                for param in params {
                    self.visit_pattern(&param.pattern);
                }
                self.visit_expr(body);
            }
            Expr::Await { expr: await_expr, .. } => self.visit_expr(await_expr),
            Expr::Try(e) => self.visit_expr(e),
            Expr::Morpheme { body, .. } => self.visit_expr(body),
            Expr::Pipe { expr: pipe_expr, .. } => self.visit_expr(pipe_expr),
            Expr::Unsafe(block) => self.visit_block(block),
            Expr::Async { block, .. } => self.visit_block(block),
            Expr::Unary { expr: unary_expr, .. } => self.visit_expr(unary_expr),
            Expr::Evidential { expr: ev_expr, .. } => self.visit_expr(ev_expr),
            Expr::Let { value, pattern, .. } => {
                self.visit_pattern(pattern);
                self.visit_expr(value);
            }
            Expr::Incorporation { segments } => {
                for seg in segments {
                    if let Some(ref args) = seg.args {
                        for arg in args {
                            self.visit_expr(arg);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn visit_pattern(&mut self, _pattern: &Pattern) {}
}

// ============================================
// Convenience Functions
// ============================================

/// Lint a source file with default configuration.
pub fn lint_file(file: &SourceFile, source: &str) -> Diagnostics {
    let mut linter = Linter::new(LintConfig::default());
    linter.lint(file, source);
    linter.diagnostics
}

/// Convert a ParseError into a rich Diagnostic for LSP/CLI display.
fn parse_error_to_diagnostic(error: &ParseError, source_len: usize) -> Diagnostic {
    match error {
        ParseError::DeprecatedRustSyntax { rust, sigil, span } => {
            let code = match rust.as_str() {
                "fn" | "let" | "mut" | "struct" | "impl" | "trait" | "enum" => "P001",
                "pub" | "mod" | "use" => "P002",
                "if" | "else" | "match" | "while" | "for" | "in" => "P003",
                "return" | "break" | "continue" => "P004",
                "&mut" => "P005",
                "::" => "P006",
                _ => "P000",
            };

            Diagnostic::error(format!("Deprecated Rust syntax: `{}`", rust), *span)
                .with_code(code)
                .with_label(*span, format!("Rust syntax not supported"))
                .with_note(format!("Sigil has its own native syntax. Use: {}", sigil))
                .with_note("Run `sigil migrate <file>` to auto-convert Rust syntax to Sigil".to_string())
        }
        ParseError::UnexpectedToken { expected, found, span } => {
            Diagnostic::error(format!("Unexpected token: expected {}, found {:?}", expected, found), *span)
                .with_code("P010")
                .with_label(*span, format!("expected {}", expected))
        }
        ParseError::UnexpectedEof => {
            let span = Span::new(source_len.saturating_sub(1), source_len);
            Diagnostic::error("Unexpected end of file".to_string(), span)
                .with_code("P011")
                .with_note("The file ended unexpectedly. Check for missing closing braces, parentheses, or semicolons.".to_string())
        }
        ParseError::InvalidNumber(msg) => {
            let span = Span::new(0, 1);
            Diagnostic::error(format!("Invalid number literal: {}", msg), span)
                .with_code("P012")
        }
        ParseError::Custom(msg) => {
            let span = Span::new(0, 1);
            Diagnostic::error(msg.clone(), span)
                .with_code("P099")
        }
    }
}

/// Lint source code string (parses and lints).
///
/// Parse errors are returned as diagnostics rather than Err, allowing
/// them to be displayed in LSP and CLI with full context.
pub fn lint_source(source: &str, _filename: &str) -> Diagnostics {
    use crate::parser::Parser;

    let mut parser = Parser::new(source);

    match parser.parse_file() {
        Ok(file) => lint_file(&file, source),
        Err(e) => {
            let mut diagnostics = Diagnostics::new();
            diagnostics.add(parse_error_to_diagnostic(&e, source.len()));
            diagnostics
        }
    }
}

/// Lint source code with custom configuration.
///
/// Parse errors are returned as diagnostics rather than Err, allowing
/// them to be displayed in LSP and CLI with full context.
pub fn lint_source_with_config(source: &str, _filename: &str, config: LintConfig) -> Diagnostics {
    use crate::parser::Parser;

    let mut parser = Parser::new(source);

    match parser.parse_file() {
        Ok(file) => {
            let mut linter = Linter::new(config);
            linter.lint(&file, source);
            linter.diagnostics
        }
        Err(e) => {
            let mut diagnostics = Diagnostics::new();
            diagnostics.add(parse_error_to_diagnostic(&e, source.len()));
            diagnostics
        }
    }
}

/// Result of linting a directory.
#[derive(Debug)]
pub struct DirectoryLintResult {
    /// Results per file: (path, diagnostics)
    pub files: Vec<(String, Diagnostics)>,
    /// Total warnings across all files
    pub total_warnings: usize,
    /// Total errors across all files
    pub total_errors: usize,
    /// Files with parse errors (included in diagnostics with has_errors())
    pub parse_errors: usize,
}

/// Collect all Sigil files in a directory recursively.
fn collect_sigil_files(dir: &Path) -> Vec<std::path::PathBuf> {
    use std::fs;
    let mut files = Vec::new();

    fn visit_dir(dir: &Path, files: &mut Vec<std::path::PathBuf>) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    visit_dir(&path, files);
                } else if path.extension().map_or(false, |ext| ext == "sigil" || ext == "sg") {
                    files.push(path);
                }
            }
        }
    }

    visit_dir(dir, &mut files);
    files
}

/// Lint all Sigil files in a directory recursively (sequential).
pub fn lint_directory(dir: &Path, config: LintConfig) -> DirectoryLintResult {
    use std::fs;

    let files = collect_sigil_files(dir);
    let mut result = DirectoryLintResult {
        files: Vec::new(),
        total_warnings: 0,
        total_errors: 0,
        parse_errors: 0,
    };

    for path in files {
        if let Ok(source) = fs::read_to_string(&path) {
            let path_str = path.display().to_string();
            let diagnostics = lint_source_with_config(&source, &path_str, config.clone());

            let warnings = diagnostics.iter()
                .filter(|d| d.severity == crate::diagnostic::Severity::Warning)
                .count();
            let errors = diagnostics.iter()
                .filter(|d| d.severity == crate::diagnostic::Severity::Error)
                .count();

            // Parse errors are detected by code prefix P0xx
            let has_parse_error = diagnostics.iter()
                .any(|d| d.code.as_ref().map_or(false, |c| c.starts_with("P0")));
            if has_parse_error {
                result.parse_errors += 1;
            }

            result.total_warnings += warnings;
            result.total_errors += errors;
            result.files.push((path_str, diagnostics));
        }
    }

    result
}

/// Lint all Sigil files in a directory recursively (parallel).
///
/// Uses rayon for parallel processing, providing significant speedups
/// for large codebases.
pub fn lint_directory_parallel(dir: &Path, config: LintConfig) -> DirectoryLintResult {
    use rayon::prelude::*;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let files = collect_sigil_files(dir);
    let total_warnings = AtomicUsize::new(0);
    let total_errors = AtomicUsize::new(0);
    let parse_errors = AtomicUsize::new(0);

    let file_results: Vec<(String, Diagnostics)> = files
        .par_iter()
        .filter_map(|path| {
            let source = fs::read_to_string(path).ok()?;
            let path_str = path.display().to_string();
            let diagnostics = lint_source_with_config(&source, &path_str, config.clone());

            let warnings = diagnostics.iter()
                .filter(|d| d.severity == crate::diagnostic::Severity::Warning)
                .count();
            let errors = diagnostics.iter()
                .filter(|d| d.severity == crate::diagnostic::Severity::Error)
                .count();

            // Parse errors are detected by code prefix P0xx
            let has_parse_error = diagnostics.iter()
                .any(|d| d.code.as_ref().map_or(false, |c| c.starts_with("P0")));
            if has_parse_error {
                parse_errors.fetch_add(1, Ordering::Relaxed);
            }

            total_warnings.fetch_add(warnings, Ordering::Relaxed);
            total_errors.fetch_add(errors, Ordering::Relaxed);
            Some((path_str, diagnostics))
        })
        .collect();

    DirectoryLintResult {
        files: file_results,
        total_warnings: total_warnings.load(Ordering::Relaxed),
        total_errors: total_errors.load(Ordering::Relaxed),
        parse_errors: parse_errors.load(Ordering::Relaxed),
    }
}

/// Watch mode configuration.
#[derive(Debug, Clone)]
pub struct WatchConfig {
    /// Polling interval in milliseconds
    pub poll_interval_ms: u64,
    /// Whether to clear terminal before each run
    pub clear_screen: bool,
    /// Whether to run on startup before first change
    pub run_on_start: bool,
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            poll_interval_ms: 500,
            clear_screen: true,
            run_on_start: true,
        }
    }
}

/// Result of a watch iteration.
#[derive(Debug)]
pub struct WatchResult {
    /// Files that changed
    pub changed_files: Vec<String>,
    /// Lint result for changed files
    pub lint_result: DirectoryLintResult,
}

/// Watch a directory for changes and lint on each change.
///
/// Returns an iterator that yields results whenever files change.
/// Uses polling-based change detection.
pub fn watch_directory(
    dir: &Path,
    config: LintConfig,
    watch_config: WatchConfig,
) -> impl Iterator<Item = WatchResult> {
    use std::collections::HashMap;
    use std::fs;
    use std::time::{Duration, SystemTime};

    let dir = dir.to_path_buf();
    let poll_interval = Duration::from_millis(watch_config.poll_interval_ms);
    let mut file_times: HashMap<std::path::PathBuf, SystemTime> = HashMap::new();
    let mut first_run = watch_config.run_on_start;

    std::iter::from_fn(move || {
        loop {
            let files = collect_sigil_files(&dir);
            let mut changed = Vec::new();

            for path in &files {
                if let Ok(metadata) = fs::metadata(path) {
                    if let Ok(modified) = metadata.modified() {
                        let prev = file_times.get(path);
                        if prev.is_none() || prev.is_some_and(|t| *t != modified) {
                            changed.push(path.display().to_string());
                            file_times.insert(path.clone(), modified);
                        }
                    }
                }
            }

            // Check for deleted files
            let current_paths: std::collections::HashSet<_> = files.iter().collect();
            file_times.retain(|p, _| current_paths.contains(p));

            if first_run || !changed.is_empty() {
                first_run = false;
                let lint_result = lint_directory_parallel(&dir, config.clone());
                return Some(WatchResult {
                    changed_files: changed,
                    lint_result,
                });
            }

            std::thread::sleep(poll_interval);
        }
    })
}

// ============================================
// Auto-Fix Application
// ============================================

/// Result of applying fixes to source code.
#[derive(Debug)]
pub struct FixResult {
    /// The modified source code
    pub source: String,
    /// Number of fixes applied
    pub fixes_applied: usize,
    /// Fixes that could not be applied (conflicting spans, etc.)
    pub fixes_skipped: usize,
}

/// Apply fix suggestions from diagnostics to source code.
///
/// Returns the modified source and count of applied/skipped fixes.
/// Fixes are applied in reverse order to preserve span validity.
pub fn apply_fixes(source: &str, diagnostics: &Diagnostics) -> FixResult {
    // Collect all fixes with their spans
    let mut fixes: Vec<(&FixSuggestion, Span)> = diagnostics
        .iter()
        .flat_map(|d| d.suggestions.iter().map(move |s| (s, s.span)))
        .collect();

    // Sort by span start in reverse order (apply from end to start)
    fixes.sort_by(|a, b| b.1.start.cmp(&a.1.start));

    let mut result = source.to_string();
    let mut applied = 0;
    let mut skipped = 0;
    let mut last_end = usize::MAX;

    for (fix, span) in fixes {
        // Skip overlapping fixes
        if span.end > last_end {
            skipped += 1;
            continue;
        }

        // Validate span bounds
        if span.start > span.end || span.end > result.len() {
            skipped += 1;
            continue;
        }

        // Apply the fix
        let before = &result[..span.start];
        let after = &result[span.end..];
        result = format!("{}{}{}", before, fix.replacement, after);

        applied += 1;
        last_end = span.start;
    }

    FixResult {
        source: result,
        fixes_applied: applied,
        fixes_skipped: skipped,
    }
}

/// Lint and optionally apply fixes to source code.
///
/// Returns (fixed_source, diagnostics, fix_result).
pub fn lint_and_fix(source: &str, filename: &str, config: LintConfig) -> (String, Diagnostics, FixResult) {
    let diagnostics = lint_source_with_config(source, filename, config);
    let fix_result = apply_fixes(source, &diagnostics);
    (fix_result.source.clone(), diagnostics, fix_result)
}

// ============================================
// SARIF Output Format
// ============================================

/// SARIF (Static Analysis Results Interchange Format) output.
///
/// SARIF is a standard JSON format for static analysis tools,
/// supported by IDEs like VS Code and CI systems like GitHub Actions.
#[derive(Debug, Clone, Serialize)]
pub struct SarifReport {
    #[serde(rename = "$schema")]
    pub schema: String,
    pub version: String,
    pub runs: Vec<SarifRun>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SarifRun {
    pub tool: SarifTool,
    pub results: Vec<SarifResult>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SarifTool {
    pub driver: SarifDriver,
}

#[derive(Debug, Clone, Serialize)]
pub struct SarifDriver {
    pub name: String,
    pub version: String,
    #[serde(rename = "informationUri")]
    pub information_uri: String,
    pub rules: Vec<SarifRule>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SarifRule {
    pub id: String,
    pub name: String,
    #[serde(rename = "shortDescription")]
    pub short_description: SarifMessage,
    #[serde(rename = "fullDescription")]
    pub full_description: SarifMessage,
    #[serde(rename = "defaultConfiguration")]
    pub default_configuration: SarifConfiguration,
    pub properties: SarifRuleProperties,
}

#[derive(Debug, Clone, Serialize)]
pub struct SarifMessage {
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SarifConfiguration {
    pub level: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SarifRuleProperties {
    pub category: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SarifResult {
    #[serde(rename = "ruleId")]
    pub rule_id: String,
    pub level: String,
    pub message: SarifMessage,
    pub locations: Vec<SarifLocation>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub fixes: Vec<SarifFix>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SarifLocation {
    #[serde(rename = "physicalLocation")]
    pub physical_location: SarifPhysicalLocation,
}

#[derive(Debug, Clone, Serialize)]
pub struct SarifPhysicalLocation {
    #[serde(rename = "artifactLocation")]
    pub artifact_location: SarifArtifactLocation,
    pub region: SarifRegion,
}

#[derive(Debug, Clone, Serialize)]
pub struct SarifArtifactLocation {
    pub uri: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SarifRegion {
    #[serde(rename = "startLine")]
    pub start_line: usize,
    #[serde(rename = "startColumn")]
    pub start_column: usize,
    #[serde(rename = "endLine")]
    pub end_line: usize,
    #[serde(rename = "endColumn")]
    pub end_column: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct SarifFix {
    pub description: SarifMessage,
    #[serde(rename = "artifactChanges")]
    pub artifact_changes: Vec<SarifArtifactChange>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SarifArtifactChange {
    #[serde(rename = "artifactLocation")]
    pub artifact_location: SarifArtifactLocation,
    pub replacements: Vec<SarifReplacement>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SarifReplacement {
    #[serde(rename = "deletedRegion")]
    pub deleted_region: SarifRegion,
    #[serde(rename = "insertedContent")]
    pub inserted_content: SarifContent,
}

#[derive(Debug, Clone, Serialize)]
pub struct SarifContent {
    pub text: String,
}

impl SarifReport {
    /// Create a new SARIF report with all lint rules.
    pub fn new() -> Self {
        let rules: Vec<SarifRule> = LintId::all()
            .iter()
            .map(|lint| SarifRule {
                id: lint.code().to_string(),
                name: lint.name().to_string(),
                short_description: SarifMessage {
                    text: lint.description().to_string(),
                },
                full_description: SarifMessage {
                    text: lint.extended_docs().trim().to_string(),
                },
                default_configuration: SarifConfiguration {
                    level: match lint.default_level() {
                        LintLevel::Allow => "none".to_string(),
                        LintLevel::Warn => "warning".to_string(),
                        LintLevel::Deny => "error".to_string(),
                    },
                },
                properties: SarifRuleProperties {
                    category: format!("{:?}", lint.category()),
                },
            })
            .collect();

        Self {
            schema: "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json".to_string(),
            version: "2.1.0".to_string(),
            runs: vec![SarifRun {
                tool: SarifTool {
                    driver: SarifDriver {
                        name: "sigil-lint".to_string(),
                        version: env!("CARGO_PKG_VERSION").to_string(),
                        information_uri: "https://github.com/Daemoniorum-LLC/styx".to_string(),
                        rules,
                    },
                },
                results: Vec::new(),
            }],
        }
    }

    /// Add diagnostics from a file to the report.
    pub fn add_file(&mut self, filename: &str, diagnostics: &Diagnostics, source: &str) {
        let line_starts: Vec<usize> = std::iter::once(0)
            .chain(source.match_indices('\n').map(|(i, _)| i + 1))
            .collect();

        let offset_to_line_col = |offset: usize| -> (usize, usize) {
            let line = line_starts.partition_point(|&start| start <= offset);
            let col = if line > 0 {
                offset - line_starts[line - 1] + 1
            } else {
                offset + 1
            };
            (line.max(1), col)
        };

        for diag in diagnostics.iter() {
            let (start_line, start_col) = offset_to_line_col(diag.span.start);
            let (end_line, end_col) = offset_to_line_col(diag.span.end);

            let level = match diag.severity {
                Severity::Error => "error",
                Severity::Warning => "warning",
                Severity::Info | Severity::Hint => "note",
            };

            let fixes: Vec<SarifFix> = diag.suggestions.iter().map(|fix| {
                let (fix_start_line, fix_start_col) = offset_to_line_col(fix.span.start);
                let (fix_end_line, fix_end_col) = offset_to_line_col(fix.span.end);

                SarifFix {
                    description: SarifMessage {
                        text: fix.message.clone(),
                    },
                    artifact_changes: vec![SarifArtifactChange {
                        artifact_location: SarifArtifactLocation {
                            uri: filename.to_string(),
                        },
                        replacements: vec![SarifReplacement {
                            deleted_region: SarifRegion {
                                start_line: fix_start_line,
                                start_column: fix_start_col,
                                end_line: fix_end_line,
                                end_column: fix_end_col,
                            },
                            inserted_content: SarifContent {
                                text: fix.replacement.clone(),
                            },
                        }],
                    }],
                }
            }).collect();

            if let Some(ref mut run) = self.runs.first_mut() {
                run.results.push(SarifResult {
                    rule_id: diag.code.clone().unwrap_or_default(),
                    level: level.to_string(),
                    message: SarifMessage {
                        text: diag.message.clone(),
                    },
                    locations: vec![SarifLocation {
                        physical_location: SarifPhysicalLocation {
                            artifact_location: SarifArtifactLocation {
                                uri: filename.to_string(),
                            },
                            region: SarifRegion {
                                start_line,
                                start_column: start_col,
                                end_line,
                                end_column: end_col,
                            },
                        },
                    }],
                    fixes,
                });
            }
        }
    }

    /// Convert to JSON string.
    pub fn to_json(&self) -> Result<String, String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize SARIF: {}", e))
    }
}

impl Default for SarifReport {
    fn default() -> Self {
        Self::new()
    }
}

/// Generate a SARIF report for linting results.
pub fn generate_sarif(filename: &str, diagnostics: &Diagnostics, source: &str) -> SarifReport {
    let mut report = SarifReport::new();
    report.add_file(filename, diagnostics, source);
    report
}

/// Generate a SARIF report for directory linting results.
pub fn generate_sarif_for_directory(result: &DirectoryLintResult, sources: &HashMap<String, String>) -> SarifReport {
    let mut report = SarifReport::new();

    for (path, diagnostics) in &result.files {
        if let Some(source) = sources.get(path) {
            report.add_file(path, diagnostics, source);
        }
    }

    report
}

// ============================================
// Explain Mode
// ============================================

/// Print detailed documentation for a lint rule.
pub fn explain_lint(lint: LintId) -> String {
    format!(
        r#"
╔══════════════════════════════════════════════════════════════╗
║  {code}: {name}
╠══════════════════════════════════════════════════════════════╣
║  Category:    {category:?}
║  Default:     {level:?}
╚══════════════════════════════════════════════════════════════╝

{description}

{extended}

Configuration:
  In .sigillint.toml:
    [lint.levels]
    {name} = "allow"  # or "warn" or "deny"

  Inline suppression:
    // sigil-lint: allow({code})
    let code = here;

  Next-line suppression:
    // sigil-lint: allow-next-line({code})
    let code = here;
"#,
        code = lint.code(),
        name = lint.name(),
        category = lint.category(),
        level = lint.default_level(),
        description = lint.description(),
        extended = lint.extended_docs().trim(),
    )
}

/// List all available lint rules grouped by category.
pub fn list_lints() -> String {
    use std::collections::BTreeMap;

    let mut by_category: BTreeMap<LintCategory, Vec<LintId>> = BTreeMap::new();

    for lint in LintId::all() {
        by_category.entry(lint.category()).or_default().push(*lint);
    }

    let mut output = String::from("\n╔══════════════════════════════════════════════════════════════╗\n");
    output.push_str("║              Sigil Linter Rules                              ║\n");
    output.push_str("╚══════════════════════════════════════════════════════════════╝\n\n");

    for (category, lints) in by_category {
        output.push_str(&format!("── {:?} ──\n", category));
        for lint in lints {
            let level_char = match lint.default_level() {
                LintLevel::Allow => '○',
                LintLevel::Warn => '◐',
                LintLevel::Deny => '●',
            };
            output.push_str(&format!(
                "  {} {} {}: {}\n",
                level_char,
                lint.code(),
                lint.name(),
                lint.description()
            ));
        }
        output.push('\n');
    }

    output.push_str("Legend: ○ = allow by default, ◐ = warn by default, ● = deny by default\n");
    output
}

// ============================================
// Phase 8: LSP Server Support
// ============================================

/// LSP diagnostic severity mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LspSeverity {
    Error = 1,
    Warning = 2,
    Information = 3,
    Hint = 4,
}

impl From<Severity> for LspSeverity {
    fn from(sev: Severity) -> Self {
        match sev {
            Severity::Error => LspSeverity::Error,
            Severity::Warning => LspSeverity::Warning,
            Severity::Info => LspSeverity::Information,
            Severity::Hint => LspSeverity::Hint,
        }
    }
}

/// LSP-compatible diagnostic.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspDiagnostic {
    /// Line number (0-indexed)
    pub line: u32,
    /// Character offset (0-indexed)
    pub character: u32,
    /// End line
    pub end_line: u32,
    /// End character
    pub end_character: u32,
    /// Severity (1=error, 2=warning, 3=info, 4=hint)
    pub severity: u32,
    /// Diagnostic code
    pub code: Option<String>,
    /// Source identifier
    pub source: String,
    /// Message
    pub message: String,
    /// Related information
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub related_information: Vec<LspRelatedInfo>,
    /// Code actions available
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub code_actions: Vec<LspCodeAction>,
}

/// Related diagnostic information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspRelatedInfo {
    pub uri: String,
    pub line: u32,
    pub character: u32,
    pub message: String,
}

/// Code action for quick fixes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspCodeAction {
    pub title: String,
    pub kind: String,
    pub edit: LspTextEdit,
}

/// Text edit for code actions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspTextEdit {
    pub line: u32,
    pub character: u32,
    pub end_line: u32,
    pub end_character: u32,
    pub new_text: String,
}

impl LspDiagnostic {
    /// Convert from internal Diagnostic to LSP format.
    pub fn from_diagnostic(diag: &Diagnostic, source: &str) -> Self {
        let (line, character) = offset_to_position(diag.span.start, source);
        let (end_line, end_character) = offset_to_position(diag.span.end, source);

        let mut code_actions = Vec::new();

        // Convert fix suggestions to code actions
        for suggestion in &diag.suggestions {
            let (fix_line, fix_char) = offset_to_position(suggestion.span.start, source);
            let (fix_end_line, fix_end_char) = offset_to_position(suggestion.span.end, source);

            code_actions.push(LspCodeAction {
                title: suggestion.message.clone(),
                kind: "quickfix".to_string(),
                edit: LspTextEdit {
                    line: fix_line,
                    character: fix_char,
                    end_line: fix_end_line,
                    end_character: fix_end_char,
                    new_text: suggestion.replacement.clone(),
                },
            });
        }

        Self {
            line,
            character,
            end_line,
            end_character,
            severity: LspSeverity::from(diag.severity) as u32,
            code: diag.code.clone(),
            source: "sigil-lint".to_string(),
            message: diag.message.clone(),
            related_information: Vec::new(),
            code_actions,
        }
    }
}

/// Convert byte offset to line/character position.
fn offset_to_position(offset: usize, source: &str) -> (u32, u32) {
    let mut line = 0u32;
    let mut col = 0u32;

    for (i, ch) in source.char_indices() {
        if i >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 0;
        } else {
            col += 1;
        }
    }

    (line, col)
}

/// Result of LSP lint operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LspLintResult {
    /// URI of the file
    pub uri: String,
    /// Version of the document
    pub version: Option<i32>,
    /// Diagnostics
    pub diagnostics: Vec<LspDiagnostic>,
}

/// Lint for LSP integration.
pub fn lint_for_lsp(source: &str, uri: &str, config: LintConfig) -> LspLintResult {
    let diags = lint_source_with_config(source, uri, config);
    let diagnostics = diags
        .iter()
        .map(|d| LspDiagnostic::from_diagnostic(d, source))
        .collect();

    LspLintResult {
        uri: uri.to_string(),
        version: None,
        diagnostics,
    }
}

/// LSP server state (for use with tower-lsp).
#[derive(Debug, Default)]
pub struct LspServerState {
    /// Open documents: URI -> (version, content)
    pub documents: HashMap<String, (i32, String)>,
    /// Lint configuration
    pub config: LintConfig,
    /// Baseline (if loaded)
    pub baseline: Option<Baseline>,
}

impl LspServerState {
    /// Create new LSP server state.
    pub fn new() -> Self {
        Self {
            documents: HashMap::new(),
            config: LintConfig::find_and_load(),
            baseline: find_baseline(),
        }
    }

    /// Update document content.
    pub fn update_document(&mut self, uri: &str, version: i32, content: String) {
        self.documents.insert(uri.to_string(), (version, content));
    }

    /// Remove document.
    pub fn remove_document(&mut self, uri: &str) {
        self.documents.remove(uri);
    }

    /// Lint a document.
    pub fn lint_document(&self, uri: &str) -> Option<LspLintResult> {
        let (version, content) = self.documents.get(uri)?;

        let mut result = lint_for_lsp(content, uri, self.config.clone());
        result.version = Some(*version);

        // Filter against baseline if present
        if let Some(ref baseline) = self.baseline {
            result.diagnostics.retain(|lsp_diag| {
                // Convert back to check against baseline
                let span = Span::new(0, 0); // Simplified - baseline uses line matching
                let diag = Diagnostic {
                    severity: match lsp_diag.severity {
                        1 => Severity::Error,
                        2 => Severity::Warning,
                        3 => Severity::Info,
                        _ => Severity::Hint,
                    },
                    code: lsp_diag.code.clone(),
                    message: lsp_diag.message.clone(),
                    span,
                    labels: Vec::new(),
                    notes: Vec::new(),
                    suggestions: Vec::new(),
                    related: Vec::new(),
                };
                !baseline.contains(uri, &diag, content)
            });
        }

        Some(result)
    }
}

// ============================================
// Phase 9: Git Integration
// ============================================

/// Git integration for linting only changed files.
#[derive(Debug, Clone)]
pub struct GitIntegration {
    /// Repository root path
    pub repo_root: PathBuf,
}

impl GitIntegration {
    /// Create new git integration from current directory.
    pub fn from_current_dir() -> Result<Self, String> {
        let output = std::process::Command::new("git")
            .args(["rev-parse", "--show-toplevel"])
            .output()
            .map_err(|e| format!("Failed to run git: {}", e))?;

        if !output.status.success() {
            return Err("Not in a git repository".to_string());
        }

        let root = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(Self {
            repo_root: PathBuf::from(root),
        })
    }

    /// Get list of changed files (staged and unstaged).
    pub fn get_changed_files(&self) -> Result<Vec<PathBuf>, String> {
        let mut files = HashSet::new();

        // Get staged changes
        let staged = self.run_git(&["diff", "--cached", "--name-only"])?;
        for line in staged.lines() {
            if line.ends_with(".sigil") {
                files.insert(self.repo_root.join(line));
            }
        }

        // Get unstaged changes
        let unstaged = self.run_git(&["diff", "--name-only"])?;
        for line in unstaged.lines() {
            if line.ends_with(".sigil") {
                files.insert(self.repo_root.join(line));
            }
        }

        // Get untracked files
        let untracked = self.run_git(&["ls-files", "--others", "--exclude-standard"])?;
        for line in untracked.lines() {
            if line.ends_with(".sigil") {
                files.insert(self.repo_root.join(line));
            }
        }

        Ok(files.into_iter().collect())
    }

    /// Get files changed since a specific commit/branch.
    pub fn get_changed_since(&self, base: &str) -> Result<Vec<PathBuf>, String> {
        let output = self.run_git(&["diff", "--name-only", base])?;
        let files: Vec<PathBuf> = output
            .lines()
            .filter(|line| line.ends_with(".sigil"))
            .map(|line| self.repo_root.join(line))
            .collect();
        Ok(files)
    }

    /// Run a git command and return stdout.
    fn run_git(&self, args: &[&str]) -> Result<String, String> {
        let output = std::process::Command::new("git")
            .current_dir(&self.repo_root)
            .args(args)
            .output()
            .map_err(|e| format!("Failed to run git: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Git command failed: {}", stderr));
        }

        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
}

/// Lint only changed files (git diff mode).
pub fn lint_changed_files(config: LintConfig) -> Result<DirectoryLintResult, String> {
    let git = GitIntegration::from_current_dir()?;
    let changed = git.get_changed_files()?;

    if changed.is_empty() {
        return Ok(DirectoryLintResult {
            files: Vec::new(),
            total_warnings: 0,
            total_errors: 0,
            parse_errors: 0,
        });
    }

    Ok(lint_files(&changed, config))
}

/// Lint files changed since a base ref.
pub fn lint_changed_since(base: &str, config: LintConfig) -> Result<DirectoryLintResult, String> {
    let git = GitIntegration::from_current_dir()?;
    let changed = git.get_changed_since(base)?;

    if changed.is_empty() {
        return Ok(DirectoryLintResult {
            files: Vec::new(),
            total_warnings: 0,
            total_errors: 0,
            parse_errors: 0,
        });
    }

    Ok(lint_files(&changed, config))
}

/// Lint a list of specific files.
pub fn lint_files(files: &[PathBuf], config: LintConfig) -> DirectoryLintResult {
    use std::fs;

    let mut total_warnings = 0;
    let mut total_errors = 0;
    let mut parse_errors = 0;
    let mut results = Vec::new();

    for path in files {
        let path_str = path.display().to_string();

        if let Ok(source) = fs::read_to_string(path) {
            let diagnostics = lint_source_with_config(&source, &path_str, config.clone());

            for diag in diagnostics.iter() {
                match diag.severity {
                    Severity::Error => total_errors += 1,
                    Severity::Warning => total_warnings += 1,
                    _ => {}
                }
            }

            // Parse errors are detected by code prefix P0xx
            let has_parse_error = diagnostics.iter()
                .any(|d| d.code.as_ref().map_or(false, |c| c.starts_with("P0")));
            if has_parse_error {
                parse_errors += 1;
            }

            results.push((path_str, diagnostics));
        }
        // Skip files that can't be read
    }

    DirectoryLintResult {
        files: results,
        total_warnings,
        total_errors,
        parse_errors,
    }
}

/// Pre-commit hook script content.
pub const PRE_COMMIT_HOOK: &str = r#"#!/bin/sh
# Sigil lint pre-commit hook
# Generated by sigil lint --generate-hook

# Get list of staged .sigil files
STAGED_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep '\.sigil$')

if [ -z "$STAGED_FILES" ]; then
    exit 0
fi

echo "Running Sigil linter on staged files..."

# Run linter on staged files
RESULT=0
for FILE in $STAGED_FILES; do
    if [ -f "$FILE" ]; then
        sigil lint "$FILE"
        if [ $? -ne 0 ]; then
            RESULT=1
        fi
    fi
done

if [ $RESULT -ne 0 ]; then
    echo ""
    echo "Commit blocked: Please fix lint errors before committing."
    echo "Use 'git commit --no-verify' to bypass (not recommended)."
    exit 1
fi

exit 0
"#;

/// Generate pre-commit hook.
pub fn generate_pre_commit_hook() -> Result<PathBuf, String> {
    let git = GitIntegration::from_current_dir()?;
    let hook_path = git.repo_root.join(".git/hooks/pre-commit");

    std::fs::write(&hook_path, PRE_COMMIT_HOOK)
        .map_err(|e| format!("Failed to write hook: {}", e))?;

    // Make executable on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = std::fs::metadata(&hook_path)
            .map_err(|e| format!("Failed to get permissions: {}", e))?
            .permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&hook_path, perms)
            .map_err(|e| format!("Failed to set permissions: {}", e))?;
    }

    Ok(hook_path)
}

// ============================================
// Phase 10: Custom Rules
// ============================================

/// Custom lint rule definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomRule {
    /// Rule identifier (e.g., "custom_001")
    pub id: String,
    /// Rule name (e.g., "no_print_statements")
    pub name: String,
    /// Description
    pub description: String,
    /// Severity level
    pub level: LintLevel,
    /// Category
    pub category: String,
    /// Pattern type
    pub pattern: CustomPattern,
    /// Suggested fix (optional)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<String>,
    /// Extended documentation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub docs: Option<String>,
}

/// Pattern matching for custom rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum CustomPattern {
    /// Match a regex pattern in source
    Regex { pattern: String },
    /// Match function calls by name
    FunctionCall { names: Vec<String> },
    /// Match identifiers
    Identifier { names: Vec<String> },
    /// Match imports
    Import { modules: Vec<String> },
    /// Match string literals containing pattern
    StringContains { patterns: Vec<String> },
    /// Match based on AST node type
    AstNode { node_type: String, conditions: HashMap<String, String> },
}

/// Custom rules configuration file.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CustomRulesFile {
    /// Schema version
    #[serde(default = "default_version")]
    pub version: u32,
    /// Custom rules
    #[serde(default)]
    pub rules: Vec<CustomRule>,
    /// Rule sets (named groups of rules)
    #[serde(default)]
    pub rulesets: HashMap<String, Vec<String>>,
}

fn default_version() -> u32 { 1 }

impl CustomRulesFile {
    /// Load custom rules from file.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read custom rules: {}", e))?;

        if path.extension().map(|e| e == "json").unwrap_or(false) {
            serde_json::from_str(&content)
                .map_err(|e| format!("Failed to parse JSON: {}", e))
        } else {
            toml::from_str(&content)
                .map_err(|e| format!("Failed to parse TOML: {}", e))
        }
    }

    /// Find and load custom rules from standard locations.
    pub fn find_and_load() -> Option<Self> {
        let names = [
            ".sigillint-rules.toml",
            ".sigillint-rules.json",
            "sigillint-rules.toml",
        ];

        if let Ok(mut dir) = std::env::current_dir() {
            loop {
                for name in &names {
                    let path = dir.join(name);
                    if path.exists() {
                        if let Ok(rules) = Self::from_file(&path) {
                            return Some(rules);
                        }
                    }
                }
                if !dir.pop() {
                    break;
                }
            }
        }

        None
    }
}

/// Result of applying a custom rule.
#[derive(Debug)]
pub struct CustomRuleMatch {
    /// Rule that matched
    pub rule_id: String,
    /// Span of the match
    pub span: Span,
    /// Match details
    pub matched_text: String,
}

/// Custom rule checker.
pub struct CustomRuleChecker {
    rules: Vec<CustomRule>,
    compiled_patterns: HashMap<String, regex::Regex>,
}

impl CustomRuleChecker {
    /// Create a new custom rule checker.
    pub fn new(rules: Vec<CustomRule>) -> Self {
        let mut compiled = HashMap::new();

        for rule in &rules {
            if let CustomPattern::Regex { pattern } = &rule.pattern {
                if let Ok(re) = regex::Regex::new(pattern) {
                    compiled.insert(rule.id.clone(), re);
                }
            }
        }

        Self {
            rules,
            compiled_patterns: compiled,
        }
    }

    /// Check source code against custom rules.
    pub fn check(&self, source: &str) -> Vec<(CustomRule, CustomRuleMatch)> {
        let mut matches = Vec::new();

        for rule in &self.rules {
            match &rule.pattern {
                CustomPattern::Regex { .. } => {
                    if let Some(re) = self.compiled_patterns.get(&rule.id) {
                        for m in re.find_iter(source) {
                            matches.push((
                                rule.clone(),
                                CustomRuleMatch {
                                    rule_id: rule.id.clone(),
                                    span: Span::new(m.start(), m.end()),
                                    matched_text: m.as_str().to_string(),
                                },
                            ));
                        }
                    }
                }
                CustomPattern::FunctionCall { names } => {
                    for name in names {
                        let pattern = format!(r"\b{}\s*\(", regex::escape(name));
                        if let Ok(re) = regex::Regex::new(&pattern) {
                            for m in re.find_iter(source) {
                                matches.push((
                                    rule.clone(),
                                    CustomRuleMatch {
                                        rule_id: rule.id.clone(),
                                        span: Span::new(m.start(), m.end() - 1),
                                        matched_text: name.clone(),
                                    },
                                ));
                            }
                        }
                    }
                }
                CustomPattern::Identifier { names } => {
                    for name in names {
                        let pattern = format!(r"\b{}\b", regex::escape(name));
                        if let Ok(re) = regex::Regex::new(&pattern) {
                            for m in re.find_iter(source) {
                                matches.push((
                                    rule.clone(),
                                    CustomRuleMatch {
                                        rule_id: rule.id.clone(),
                                        span: Span::new(m.start(), m.end()),
                                        matched_text: name.clone(),
                                    },
                                ));
                            }
                        }
                    }
                }
                CustomPattern::StringContains { patterns } => {
                    // Match string literals containing the patterns
                    let string_re = regex::Regex::new(r#""([^"\\]|\\.)*""#).unwrap();
                    for string_match in string_re.find_iter(source) {
                        let string_content = string_match.as_str();
                        for pattern in patterns {
                            if string_content.contains(pattern) {
                                matches.push((
                                    rule.clone(),
                                    CustomRuleMatch {
                                        rule_id: rule.id.clone(),
                                        span: Span::new(string_match.start(), string_match.end()),
                                        matched_text: string_content.to_string(),
                                    },
                                ));
                                break;
                            }
                        }
                    }
                }
                CustomPattern::Import { modules } => {
                    for module in modules {
                        let pattern = format!(r"use\s+{}", regex::escape(module));
                        if let Ok(re) = regex::Regex::new(&pattern) {
                            for m in re.find_iter(source) {
                                matches.push((
                                    rule.clone(),
                                    CustomRuleMatch {
                                        rule_id: rule.id.clone(),
                                        span: Span::new(m.start(), m.end()),
                                        matched_text: module.clone(),
                                    },
                                ));
                            }
                        }
                    }
                }
                CustomPattern::AstNode { .. } => {
                    // AST-based matching would require parsing - skip for text-based check
                }
            }
        }

        matches
    }

    /// Convert matches to diagnostics.
    pub fn to_diagnostics(&self, source: &str) -> Diagnostics {
        let mut diagnostics = Diagnostics::new();

        for (rule, m) in self.check(source) {
            let severity = match rule.level {
                LintLevel::Deny => Severity::Error,
                LintLevel::Warn => Severity::Warning,
                LintLevel::Allow => continue,
            };

            let mut diag = Diagnostic {
                severity,
                code: Some(format!("CUSTOM:{}", rule.id)),
                message: rule.description.clone(),
                span: m.span,
                labels: Vec::new(),
                notes: Vec::new(),
                suggestions: Vec::new(),
                related: Vec::new(),
            };

            if let Some(ref suggestion) = rule.suggestion {
                diag.notes.push(format!("Suggestion: {}", suggestion));
            }

            diagnostics.add(diag);
        }

        diagnostics
    }
}

/// Lint with custom rules.
pub fn lint_with_custom_rules(
    source: &str,
    filename: &str,
    config: LintConfig,
    custom_rules: &[CustomRule],
) -> Diagnostics {
    // Run standard linting
    let mut diagnostics = lint_source_with_config(source, filename, config);

    // Run custom rules
    let checker = CustomRuleChecker::new(custom_rules.to_vec());
    let custom_diags = checker.to_diagnostics(source);

    // Merge diagnostics
    for diag in custom_diags.iter() {
        diagnostics.add(diag.clone());
    }

    diagnostics
}

// ============================================
// Phase 11: Ignore Patterns
// ============================================

/// Ignore pattern configuration.
#[derive(Debug, Clone, Default)]
pub struct IgnorePatterns {
    /// Compiled glob patterns
    patterns: Vec<globset::GlobMatcher>,
    /// Raw patterns (for debugging)
    raw_patterns: Vec<String>,
}

impl IgnorePatterns {
    /// Create empty ignore patterns.
    pub fn new() -> Self {
        Self::default()
    }

    /// Load from .sigillintignore file.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read ignore file: {}", e))?;
        Self::from_string(&content)
    }

    /// Parse ignore patterns from string.
    pub fn from_string(content: &str) -> Result<Self, String> {
        let mut patterns = Vec::new();
        let mut raw_patterns = Vec::new();

        for line in content.lines() {
            let line = line.trim();

            // Skip empty lines and comments
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Build glob
            match globset::Glob::new(line) {
                Ok(glob) => {
                    patterns.push(glob.compile_matcher());
                    raw_patterns.push(line.to_string());
                }
                Err(e) => {
                    return Err(format!("Invalid pattern '{}': {}", line, e));
                }
            }
        }

        Ok(Self { patterns, raw_patterns })
    }

    /// Find and load ignore file from standard locations.
    pub fn find_and_load() -> Option<Self> {
        let names = [
            ".sigillintignore",
            ".lintignore",
        ];

        if let Ok(mut dir) = std::env::current_dir() {
            loop {
                for name in &names {
                    let path = dir.join(name);
                    if path.exists() {
                        if let Ok(patterns) = Self::from_file(&path) {
                            return Some(patterns);
                        }
                    }
                }
                if !dir.pop() {
                    break;
                }
            }
        }

        None
    }

    /// Check if a path should be ignored.
    pub fn is_ignored(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();

        for pattern in &self.patterns {
            if pattern.is_match(path) || pattern.is_match(path_str.as_ref()) {
                return true;
            }
        }

        false
    }

    /// Check if a path string should be ignored.
    pub fn is_ignored_str(&self, path: &str) -> bool {
        self.is_ignored(Path::new(path))
    }

    /// Get raw patterns for display.
    pub fn patterns(&self) -> &[String] {
        &self.raw_patterns
    }

    /// Check if any patterns are defined.
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }
}

/// Filter files based on ignore patterns.
pub fn filter_ignored(files: Vec<PathBuf>, ignore: &IgnorePatterns) -> Vec<PathBuf> {
    files
        .into_iter()
        .filter(|f| !ignore.is_ignored(f))
        .collect()
}

/// Collect sigil files respecting ignore patterns.
pub fn collect_sigil_files_filtered(dir: &Path, ignore: &IgnorePatterns) -> Vec<PathBuf> {
    let all_files = collect_sigil_files(dir);
    filter_ignored(all_files, ignore)
}

/// Lint directory with ignore patterns.
pub fn lint_directory_filtered(
    dir: &Path,
    config: LintConfig,
    ignore: Option<&IgnorePatterns>,
) -> DirectoryLintResult {
    let files = if let Some(patterns) = ignore {
        collect_sigil_files_filtered(dir, patterns)
    } else if let Some(loaded) = IgnorePatterns::find_and_load() {
        collect_sigil_files_filtered(dir, &loaded)
    } else {
        collect_sigil_files(dir)
    };

    // Use the existing parallel implementation
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let total_warnings = AtomicUsize::new(0);
    let total_errors = AtomicUsize::new(0);
    let parse_errors = AtomicUsize::new(0);

    let file_results: Vec<(String, Diagnostics)> = files
        .par_iter()
        .filter_map(|path| {
            let source = std::fs::read_to_string(path).ok()?;
            let path_str = path.display().to_string();
            let diagnostics = lint_source_with_config(&source, &path_str, config.clone());

            let warnings = diagnostics.iter()
                .filter(|d| d.severity == Severity::Warning)
                .count();
            let errors = diagnostics.iter()
                .filter(|d| d.severity == Severity::Error)
                .count();

            // Parse errors are detected by code prefix P0xx
            let has_parse_error = diagnostics.iter()
                .any(|d| d.code.as_ref().map_or(false, |c| c.starts_with("P0")));
            if has_parse_error {
                parse_errors.fetch_add(1, Ordering::Relaxed);
            }

            total_warnings.fetch_add(warnings, Ordering::Relaxed);
            total_errors.fetch_add(errors, Ordering::Relaxed);
            Some((path_str, diagnostics))
        })
        .collect();

    DirectoryLintResult {
        files: file_results,
        total_warnings: total_warnings.load(Ordering::Relaxed),
        total_errors: total_errors.load(Ordering::Relaxed),
        parse_errors: parse_errors.load(Ordering::Relaxed),
    }
}

// ============================================
// Phase 12: HTML Reports and Trend Tracking
// ============================================

/// Lint report for trend tracking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LintReport {
    /// Report timestamp
    pub timestamp: String,
    /// Git commit hash (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    /// Git branch (if available)
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    /// Total files linted
    pub total_files: usize,
    /// Total warnings
    pub total_warnings: usize,
    /// Total errors
    pub total_errors: usize,
    /// Parse errors
    pub parse_errors: usize,
    /// Issues by rule
    pub by_rule: HashMap<String, usize>,
    /// Issues by category
    pub by_category: HashMap<String, usize>,
    /// Issues by file (top N)
    pub by_file: Vec<(String, usize)>,
}

impl LintReport {
    /// Create report from directory lint result.
    pub fn from_result(result: &DirectoryLintResult) -> Self {
        let mut by_rule: HashMap<String, usize> = HashMap::new();
        let mut by_category: HashMap<String, usize> = HashMap::new();
        let mut by_file: Vec<(String, usize)> = Vec::new();

        for (path, diagnostics) in &result.files {
            let count = diagnostics.iter().count();
            if count > 0 {
                by_file.push((path.clone(), count));
            }

            for diag in diagnostics.iter() {
                if let Some(ref code) = diag.code {
                    *by_rule.entry(code.clone()).or_insert(0usize) += 1;

                    // Infer category from code
                    let category = if code.starts_with('E') {
                        "error"
                    } else if code.starts_with('W') {
                        match &code[1..3] {
                            "01" | "02" => "style",
                            "03" | "04" | "05" => "correctness",
                            _ => "other",
                        }
                    } else {
                        "other"
                    };
                    *by_category.entry(category.to_string()).or_insert(0usize) += 1;
                }
            }
        }

        // Sort by_file by count (descending)
        by_file.sort_by(|a, b| b.1.cmp(&a.1));
        by_file.truncate(20); // Keep top 20

        // Get git info
        let (commit, branch) = Self::get_git_info();

        Self {
            timestamp: chrono_lite_now(),
            commit,
            branch,
            total_files: result.files.len(),
            total_warnings: result.total_warnings,
            total_errors: result.total_errors,
            parse_errors: result.parse_errors,
            by_rule,
            by_category,
            by_file,
        }
    }

    /// Get current git commit and branch.
    fn get_git_info() -> (Option<String>, Option<String>) {
        let commit = std::process::Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());

        let branch = std::process::Command::new("git")
            .args(["rev-parse", "--abbrev-ref", "HEAD"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());

        (commit, branch)
    }

    /// Save report to JSON file.
    pub fn save_json(&self, path: &Path) -> Result<(), String> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize report: {}", e))?;
        std::fs::write(path, content)
            .map_err(|e| format!("Failed to write report: {}", e))
    }

    /// Load report from JSON file.
    pub fn load_json(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read report: {}", e))?;
        serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse report: {}", e))
    }
}

/// Trend data for multiple reports.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrendData {
    /// Historical reports
    pub reports: Vec<LintReport>,
    /// Maximum reports to keep
    pub max_reports: usize,
}

impl TrendData {
    /// Create new trend tracker.
    pub fn new(max_reports: usize) -> Self {
        Self {
            reports: Vec::new(),
            max_reports,
        }
    }

    /// Load from file.
    pub fn from_file(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read trend data: {}", e))?;
        serde_json::from_str(&content)
            .map_err(|e| format!("Failed to parse trend data: {}", e))
    }

    /// Save to file.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let content = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize trend data: {}", e))?;
        std::fs::write(path, content)
            .map_err(|e| format!("Failed to write trend data: {}", e))
    }

    /// Add a report to the trend.
    pub fn add_report(&mut self, report: LintReport) {
        self.reports.push(report);

        // Keep only max_reports
        if self.reports.len() > self.max_reports {
            self.reports.remove(0);
        }
    }

    /// Get trend summary.
    pub fn summary(&self) -> TrendSummary {
        if self.reports.is_empty() {
            return TrendSummary::default();
        }

        let latest = self.reports.last().unwrap();
        let previous = if self.reports.len() > 1 {
            Some(&self.reports[self.reports.len() - 2])
        } else {
            None
        };

        let warning_delta = previous
            .map(|p| latest.total_warnings as i64 - p.total_warnings as i64)
            .unwrap_or(0);
        let error_delta = previous
            .map(|p| latest.total_errors as i64 - p.total_errors as i64)
            .unwrap_or(0);

        TrendSummary {
            total_reports: self.reports.len(),
            latest_warnings: latest.total_warnings,
            latest_errors: latest.total_errors,
            warning_delta,
            error_delta,
            trend_direction: if warning_delta + error_delta < 0 {
                TrendDirection::Improving
            } else if warning_delta + error_delta > 0 {
                TrendDirection::Degrading
            } else {
                TrendDirection::Stable
            },
        }
    }
}

/// Trend direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
}

/// Trend summary.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrendSummary {
    pub total_reports: usize,
    pub latest_warnings: usize,
    pub latest_errors: usize,
    pub warning_delta: i64,
    pub error_delta: i64,
    pub trend_direction: TrendDirection,
}

impl Default for TrendDirection {
    fn default() -> Self {
        TrendDirection::Stable
    }
}

/// Generate HTML report.
pub fn generate_html_report(result: &DirectoryLintResult, title: &str) -> String {
    let report = LintReport::from_result(result);

    let mut html = String::new();

    // HTML header
    html.push_str(&format!(r#"<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{} - Sigil Lint Report</title>
    <style>
        :root {{
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-card: #0f3460;
            --text-primary: #eee;
            --text-secondary: #aaa;
            --accent: #e94560;
            --success: #4ecca3;
            --warning: #ffc107;
            --error: #e94560;
        }}
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: var(--accent); margin-bottom: 0.5rem; }}
        .meta {{ color: var(--text-secondary); margin-bottom: 2rem; }}
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .stat-card {{
            background: var(--bg-card);
            padding: 1.5rem;
            border-radius: 8px;
            text-align: center;
        }}
        .stat-value {{ font-size: 2.5rem; font-weight: bold; }}
        .stat-label {{ color: var(--text-secondary); }}
        .stat-value.errors {{ color: var(--error); }}
        .stat-value.warnings {{ color: var(--warning); }}
        .stat-value.success {{ color: var(--success); }}
        .section {{ margin-bottom: 2rem; }}
        .section h2 {{
            color: var(--accent);
            border-bottom: 2px solid var(--bg-card);
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--bg-secondary);
            border-radius: 8px;
            overflow: hidden;
        }}
        th, td {{
            padding: 0.75rem 1rem;
            text-align: left;
            border-bottom: 1px solid var(--bg-card);
        }}
        th {{ background: var(--bg-card); color: var(--accent); }}
        tr:hover {{ background: var(--bg-card); }}
        .bar {{
            height: 8px;
            background: var(--bg-card);
            border-radius: 4px;
            overflow: hidden;
        }}
        .bar-fill {{
            height: 100%;
            background: var(--accent);
            transition: width 0.3s ease;
        }}
        .chart {{
            display: flex;
            align-items: flex-end;
            gap: 0.5rem;
            height: 150px;
            padding: 1rem;
            background: var(--bg-secondary);
            border-radius: 8px;
        }}
        .chart-bar {{
            flex: 1;
            background: var(--accent);
            border-radius: 4px 4px 0 0;
            min-width: 20px;
            position: relative;
        }}
        .chart-bar:hover {{ opacity: 0.8; }}
        .chart-label {{
            position: absolute;
            bottom: -1.5rem;
            left: 50%;
            transform: translateX(-50%);
            font-size: 0.75rem;
            color: var(--text-secondary);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🔮 {}</h1>
        <p class="meta">Generated: {} | Commit: {} | Branch: {}</p>

        <div class="stats">
            <div class="stat-card">
                <div class="stat-value">{}</div>
                <div class="stat-label">Files Analyzed</div>
            </div>
            <div class="stat-card">
                <div class="stat-value errors">{}</div>
                <div class="stat-label">Errors</div>
            </div>
            <div class="stat-card">
                <div class="stat-value warnings">{}</div>
                <div class="stat-label">Warnings</div>
            </div>
            <div class="stat-card">
                <div class="stat-value success">{}</div>
                <div class="stat-label">Clean Files</div>
            </div>
        </div>
"#,
        title,
        title,
        report.timestamp,
        report.commit.as_deref().unwrap_or("N/A"),
        report.branch.as_deref().unwrap_or("N/A"),
        report.total_files,
        report.total_errors,
        report.total_warnings,
        report.total_files - report.by_file.len()
    ));

    // Issues by Rule
    if !report.by_rule.is_empty() {
        let max_count = *report.by_rule.values().max().unwrap_or(&1);
        let mut rules: Vec<_> = report.by_rule.iter().collect();
        rules.sort_by(|a, b| b.1.cmp(a.1));

        html.push_str(r#"        <div class="section">
            <h2>Issues by Rule</h2>
            <table>
                <thead>
                    <tr><th>Rule</th><th>Count</th><th>Distribution</th></tr>
                </thead>
                <tbody>
"#);

        for (rule, count) in rules.iter().take(15) {
            let pct = (**count as f64 / max_count as f64) * 100.0;
            html.push_str(&format!(
                r#"                    <tr>
                        <td><code>{}</code></td>
                        <td>{}</td>
                        <td><div class="bar"><div class="bar-fill" style="width: {:.1}%"></div></div></td>
                    </tr>
"#,
                rule, count, pct
            ));
        }

        html.push_str("                </tbody>\n            </table>\n        </div>\n\n");
    }

    // Top Files with Issues
    if !report.by_file.is_empty() {
        html.push_str(r#"        <div class="section">
            <h2>Files with Most Issues</h2>
            <table>
                <thead>
                    <tr><th>File</th><th>Issues</th></tr>
                </thead>
                <tbody>
"#);

        for (file, count) in report.by_file.iter().take(10) {
            let short_file = if file.len() > 60 {
                format!("...{}", &file[file.len() - 57..])
            } else {
                file.clone()
            };
            html.push_str(&format!(
                "                    <tr><td><code>{}</code></td><td>{}</td></tr>\n",
                short_file, count
            ));
        }

        html.push_str("                </tbody>\n            </table>\n        </div>\n\n");
    }

    // Footer
    html.push_str(r#"        <div class="section" style="text-align: center; color: var(--text-secondary); margin-top: 3rem;">
            <p>Generated by Sigil Linter v0.2.1</p>
        </div>
    </div>
</body>
</html>
"#);

    html
}

/// Save HTML report to file.
pub fn save_html_report(result: &DirectoryLintResult, path: &Path, title: &str) -> Result<(), String> {
    let html = generate_html_report(result, title);
    std::fs::write(path, html)
        .map_err(|e| format!("Failed to write HTML report: {}", e))
}

/// CI annotation format (for GitHub Actions, etc).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CiFormat {
    /// GitHub Actions annotations
    GitHub,
    /// GitLab CI format
    GitLab,
    /// Azure DevOps format
    AzureDevOps,
    /// Generic format
    Generic,
}

/// Generate CI annotations from lint result.
pub fn generate_ci_annotations(result: &DirectoryLintResult, format: CiFormat) -> String {
    let mut output = String::new();

    for (path, diagnostics) in &result.files {
        for diag in diagnostics.iter() {
            let line = 1; // Would need source to calculate exact line

            match format {
                CiFormat::GitHub => {
                    let level = match diag.severity {
                        Severity::Error => "error",
                        Severity::Warning => "warning",
                        _ => "notice",
                    };
                    output.push_str(&format!(
                        "::{} file={},line={}::{}\n",
                        level,
                        path,
                        line,
                        diag.message.replace('\n', "%0A")
                    ));
                }
                CiFormat::GitLab => {
                    output.push_str(&format!(
                        "{}:{}:{}: {}\n",
                        path,
                        line,
                        if diag.severity == Severity::Error { "error" } else { "warning" },
                        diag.message
                    ));
                }
                CiFormat::AzureDevOps => {
                    let level = match diag.severity {
                        Severity::Error => "error",
                        Severity::Warning => "warning",
                        _ => "debug",
                    };
                    output.push_str(&format!(
                        "##vso[task.logissue type={};sourcepath={};linenumber={}]{}\n",
                        level, path, line, diag.message
                    ));
                }
                CiFormat::Generic => {
                    output.push_str(&format!(
                        "{}:{}: {}: {}\n",
                        path,
                        line,
                        if diag.severity == Severity::Error { "error" } else { "warning" },
                        diag.message
                    ));
                }
            }
        }
    }

    output
}

// ============================================
// Tests
// ============================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lint_level_defaults() {
        assert_eq!(LintId::ReservedIdentifier.default_level(), LintLevel::Warn);
        assert_eq!(LintId::EvidentialityViolation.default_level(), LintLevel::Deny);
        assert_eq!(LintId::PreferUnicodeMorpheme.default_level(), LintLevel::Allow);
    }

    #[test]
    fn test_lint_codes() {
        assert_eq!(LintId::ReservedIdentifier.code(), "W0101");
        assert_eq!(LintId::EvidentialityViolation.code(), "E0600");
    }

    #[test]
    fn test_reserved_words() {
        let config = LintConfig::default();
        assert!(config.reserved_words.contains("location"));
        assert!(config.reserved_words.contains("save"));
        assert!(config.reserved_words.contains("from"));
    }

    // ============================================
    // Aether 2.0 Enhanced Rule Tests
    // ============================================

    #[test]
    fn test_aether_lint_codes() {
        // Evidentiality rules (E06xx)
        assert_eq!(LintId::EvidentialityMismatch.code(), "E0603");
        assert_eq!(LintId::UncertaintyUnhandled.code(), "E0604");
        assert_eq!(LintId::ReportedWithoutAttribution.code(), "E0605");

        // Morpheme rules (W05xx)
        assert_eq!(LintId::BrokenMorphemePipeline.code(), "W0501");
        assert_eq!(LintId::MorphemeTypeIncompatibility.code(), "W0502");
        assert_eq!(LintId::InconsistentMorphemeStyle.code(), "W0503");

        // Domain validation rules (W06xx)
        assert_eq!(LintId::InvalidHexagramNumber.code(), "W0600");
        assert_eq!(LintId::InvalidTarotNumber.code(), "W0601");
        assert_eq!(LintId::InvalidChakraIndex.code(), "W0602");
        assert_eq!(LintId::InvalidZodiacIndex.code(), "W0603");
        assert_eq!(LintId::InvalidGematriaValue.code(), "W0604");
        assert_eq!(LintId::FrequencyOutOfRange.code(), "W0605");

        // Enhanced pattern rules (W07xx)
        assert_eq!(LintId::MissingEvidentialityMarker.code(), "W0700");
        assert_eq!(LintId::PreferNamedEsotericConstant.code(), "W0701");
        assert_eq!(LintId::EmotionIntensityOutOfRange.code(), "W0702");
    }

    #[test]
    fn test_aether_lint_names() {
        assert_eq!(LintId::EvidentialityMismatch.name(), "evidentiality_mismatch");
        assert_eq!(LintId::InvalidHexagramNumber.name(), "invalid_hexagram_number");
        assert_eq!(LintId::FrequencyOutOfRange.name(), "frequency_out_of_range");
        assert_eq!(LintId::PreferNamedEsotericConstant.name(), "prefer_named_esoteric_constant");
    }

    #[test]
    fn test_aether_lint_levels() {
        // Critical rules should be Deny
        assert_eq!(LintId::EvidentialityMismatch.default_level(), LintLevel::Deny);
        assert_eq!(LintId::BrokenMorphemePipeline.default_level(), LintLevel::Deny);
        assert_eq!(LintId::MorphemeTypeIncompatibility.default_level(), LintLevel::Deny);

        // Domain validation should be Warn
        assert_eq!(LintId::InvalidHexagramNumber.default_level(), LintLevel::Warn);
        assert_eq!(LintId::InvalidTarotNumber.default_level(), LintLevel::Warn);
        assert_eq!(LintId::InvalidChakraIndex.default_level(), LintLevel::Warn);
        assert_eq!(LintId::InvalidZodiacIndex.default_level(), LintLevel::Warn);
        assert_eq!(LintId::FrequencyOutOfRange.default_level(), LintLevel::Warn);

        // Style suggestions should be Allow
        assert_eq!(LintId::InconsistentMorphemeStyle.default_level(), LintLevel::Allow);
        assert_eq!(LintId::MissingEvidentialityMarker.default_level(), LintLevel::Allow);
        assert_eq!(LintId::PreferNamedEsotericConstant.default_level(), LintLevel::Allow);
    }

    #[test]
    fn test_aether_lint_categories() {
        // Sigil-specific rules
        assert_eq!(LintId::EvidentialityMismatch.category(), LintCategory::Sigil);
        assert_eq!(LintId::UncertaintyUnhandled.category(), LintCategory::Sigil);
        assert_eq!(LintId::BrokenMorphemePipeline.category(), LintCategory::Sigil);
        assert_eq!(LintId::MissingEvidentialityMarker.category(), LintCategory::Sigil);

        // Domain validation as correctness
        assert_eq!(LintId::InvalidHexagramNumber.category(), LintCategory::Correctness);
        assert_eq!(LintId::InvalidTarotNumber.category(), LintCategory::Correctness);
        assert_eq!(LintId::FrequencyOutOfRange.category(), LintCategory::Correctness);

        // Style rules
        assert_eq!(LintId::InconsistentMorphemeStyle.category(), LintCategory::Style);
    }

    #[test]
    fn test_aether_lint_descriptions() {
        // Descriptions should not be empty
        assert!(!LintId::EvidentialityMismatch.description().is_empty());
        assert!(!LintId::InvalidHexagramNumber.description().is_empty());
        assert!(!LintId::FrequencyOutOfRange.description().is_empty());

        // Descriptions should contain relevant keywords
        assert!(LintId::InvalidHexagramNumber.description().contains("1") &&
                LintId::InvalidHexagramNumber.description().contains("64"));
        assert!(LintId::InvalidTarotNumber.description().contains("0") &&
                LintId::InvalidTarotNumber.description().contains("21"));
        assert!(LintId::FrequencyOutOfRange.description().contains("20Hz") ||
                LintId::FrequencyOutOfRange.description().contains("20kHz"));
    }

    #[test]
    fn test_all_includes_aether_rules() {
        let all = LintId::all();

        // Check that new rules are included
        assert!(all.contains(&LintId::EvidentialityMismatch));
        assert!(all.contains(&LintId::UncertaintyUnhandled));
        assert!(all.contains(&LintId::ReportedWithoutAttribution));
        assert!(all.contains(&LintId::BrokenMorphemePipeline));
        assert!(all.contains(&LintId::InvalidHexagramNumber));
        assert!(all.contains(&LintId::InvalidTarotNumber));
        assert!(all.contains(&LintId::InvalidChakraIndex));
        assert!(all.contains(&LintId::InvalidZodiacIndex));
        assert!(all.contains(&LintId::FrequencyOutOfRange));
        assert!(all.contains(&LintId::PreferNamedEsotericConstant));
        assert!(all.contains(&LintId::EmotionIntensityOutOfRange));
    }

    #[test]
    fn test_lint_count() {
        // Should now have 44 lint rules (30 original + 14 Aether rules)
        let all = LintId::all();
        assert_eq!(all.len(), 44);
    }

    #[test]
    fn test_from_str_aether_rules() {
        // Should find by code
        assert_eq!(LintId::from_str("E0603"), Some(LintId::EvidentialityMismatch));
        assert_eq!(LintId::from_str("W0600"), Some(LintId::InvalidHexagramNumber));
        assert_eq!(LintId::from_str("W0605"), Some(LintId::FrequencyOutOfRange));

        // Should find by name
        assert_eq!(LintId::from_str("evidentiality_mismatch"), Some(LintId::EvidentialityMismatch));
        assert_eq!(LintId::from_str("invalid_hexagram_number"), Some(LintId::InvalidHexagramNumber));
        assert_eq!(LintId::from_str("frequency_out_of_range"), Some(LintId::FrequencyOutOfRange));
    }
}
