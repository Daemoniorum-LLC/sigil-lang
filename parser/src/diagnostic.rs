//! Rich diagnostic reporting for Sigil.
//!
//! Provides Rust-quality error messages with:
//! - Colored output with source context
//! - "Did you mean?" suggestions
//! - Fix suggestions
//! - Multi-span support for related information
//! - JSON output for AI agent consumption

use ariadne::{Color, ColorGenerator, Config, Fmt, Label, Report, ReportKind, Source};
use serde::{Deserialize, Serialize};
use std::ops::Range;
use strsim::jaro_winkler;

use crate::span::Span;
use crate::lexer::Token;

/// Severity level for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Error,
    Warning,
    Info,
    Hint,
}

impl Severity {
    fn to_report_kind(self) -> ReportKind<'static> {
        match self {
            Severity::Error => ReportKind::Error,
            Severity::Warning => ReportKind::Warning,
            Severity::Info => ReportKind::Advice,
            Severity::Hint => ReportKind::Advice,
        }
    }

    fn color(self) -> Color {
        match self {
            Severity::Error => Color::Red,
            Severity::Warning => Color::Yellow,
            Severity::Info => Color::Blue,
            Severity::Hint => Color::Cyan,
        }
    }
}

/// A fix suggestion that can be applied automatically.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FixSuggestion {
    pub message: String,
    pub span: Span,
    pub replacement: String,
}

/// A related location with additional context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedInfo {
    pub message: String,
    pub span: Span,
}

/// A label at a specific location.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiagnosticLabel {
    pub span: Span,
    pub message: String,
}

/// A rich diagnostic with all context needed for beautiful error reporting.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostic {
    pub severity: Severity,
    pub code: Option<String>,
    pub message: String,
    pub span: Span,
    #[serde(skip)]
    pub labels: Vec<(Span, String)>,
    pub notes: Vec<String>,
    pub suggestions: Vec<FixSuggestion>,
    pub related: Vec<RelatedInfo>,
}

/// JSON-serializable diagnostic output for AI agents.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonDiagnostic {
    pub severity: Severity,
    pub code: Option<String>,
    pub message: String,
    pub file: String,
    pub span: Span,
    pub line: u32,
    pub column: u32,
    pub end_line: u32,
    pub end_column: u32,
    pub labels: Vec<DiagnosticLabel>,
    pub notes: Vec<String>,
    pub suggestions: Vec<FixSuggestion>,
    pub related: Vec<RelatedInfo>,
}

/// JSON output wrapper for multiple diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonDiagnosticsOutput {
    pub file: String,
    pub diagnostics: Vec<JsonDiagnostic>,
    pub error_count: usize,
    pub warning_count: usize,
    pub success: bool,
}

impl Diagnostic {
    /// Create a new error diagnostic.
    pub fn error(message: impl Into<String>, span: Span) -> Self {
        Self {
            severity: Severity::Error,
            code: None,
            message: message.into(),
            span,
            labels: Vec::new(),
            notes: Vec::new(),
            suggestions: Vec::new(),
            related: Vec::new(),
        }
    }

    /// Create a new warning diagnostic.
    pub fn warning(message: impl Into<String>, span: Span) -> Self {
        Self {
            severity: Severity::Warning,
            code: None,
            message: message.into(),
            span,
            labels: Vec::new(),
            notes: Vec::new(),
            suggestions: Vec::new(),
            related: Vec::new(),
        }
    }

    /// Add an error code (e.g., "E0001").
    pub fn with_code(mut self, code: impl Into<String>) -> Self {
        self.code = Some(code.into());
        self
    }

    /// Add a label at a specific span.
    pub fn with_label(mut self, span: Span, message: impl Into<String>) -> Self {
        self.labels.push((span, message.into()));
        self
    }

    /// Add a note (shown at the bottom).
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.notes.push(note.into());
        self
    }

    /// Add a fix suggestion.
    pub fn with_suggestion(mut self, message: impl Into<String>, span: Span, replacement: impl Into<String>) -> Self {
        self.suggestions.push(FixSuggestion {
            message: message.into(),
            span,
            replacement: replacement.into(),
        });
        self
    }

    /// Add related information.
    pub fn with_related(mut self, message: impl Into<String>, span: Span) -> Self {
        self.related.push(RelatedInfo {
            message: message.into(),
            span,
        });
        self
    }

    /// Render this diagnostic to a string with colors.
    pub fn render(&self, filename: &str, source: &str) -> String {
        let mut output = Vec::new();
        self.write_to(&mut output, filename, source);
        String::from_utf8(output).unwrap_or_else(|_| self.message.clone())
    }

    /// Write this diagnostic to a writer.
    pub fn write_to<W: std::io::Write>(&self, writer: W, filename: &str, source: &str) {
        let span_range: Range<usize> = self.span.start..self.span.end;

        let mut colors = ColorGenerator::new();
        let primary_color = self.severity.color();

        let mut builder = Report::build(self.severity.to_report_kind(), filename, self.span.start)
            .with_config(Config::default().with_cross_gap(true))
            .with_message(&self.message);

        // Add error code if present
        if let Some(ref code) = self.code {
            builder = builder.with_code(code);
        }

        // Primary label
        builder = builder.with_label(
            Label::new((filename, span_range.clone()))
                .with_message(&self.message)
                .with_color(primary_color),
        );

        // Additional labels
        for (span, msg) in &self.labels {
            let color = colors.next();
            builder = builder.with_label(
                Label::new((filename, span.start..span.end))
                    .with_message(msg)
                    .with_color(color),
            );
        }

        // Notes
        for note in &self.notes {
            builder = builder.with_note(note);
        }

        // Suggestions as notes
        for suggestion in &self.suggestions {
            let help_msg = format!(
                "help: {}: `{}`",
                suggestion.message,
                suggestion.replacement.clone().fg(Color::Green)
            );
            builder = builder.with_help(help_msg);
        }

        builder
            .finish()
            .write((filename, Source::from(source)), writer)
            .unwrap();
    }

    /// Print this diagnostic to stderr.
    pub fn eprint(&self, filename: &str, source: &str) {
        self.write_to(std::io::stderr(), filename, source);
    }

    /// Convert to JSON-serializable format with computed line/column positions.
    pub fn to_json(&self, filename: &str, source: &str) -> JsonDiagnostic {
        let (line, column) = offset_to_line_col(source, self.span.start);
        let (end_line, end_column) = offset_to_line_col(source, self.span.end);

        JsonDiagnostic {
            severity: self.severity,
            code: self.code.clone(),
            message: self.message.clone(),
            file: filename.to_string(),
            span: self.span,
            line,
            column,
            end_line,
            end_column,
            labels: self.labels.iter().map(|(span, msg)| DiagnosticLabel {
                span: *span,
                message: msg.clone(),
            }).collect(),
            notes: self.notes.clone(),
            suggestions: self.suggestions.clone(),
            related: self.related.clone(),
        }
    }
}

/// Convert byte offset to line/column (1-indexed).
fn offset_to_line_col(source: &str, offset: usize) -> (u32, u32) {
    let mut line = 1u32;
    let mut col = 1u32;

    for (i, ch) in source.char_indices() {
        if i >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }

    (line, col)
}

/// Diagnostic builder for common error patterns.
pub struct DiagnosticBuilder;

impl DiagnosticBuilder {
    /// Create an "unexpected token" error with suggestions.
    pub fn unexpected_token(
        expected: &str,
        found: &Token,
        span: Span,
        source: &str,
    ) -> Diagnostic {
        let found_str = format!("{:?}", found);
        let message = format!("expected {}, found {}", expected, found_str);

        let mut diag = Diagnostic::error(message, span)
            .with_code("E0001");

        // Add context about what was expected
        diag = diag.with_label(span, format!("expected {} here", expected));

        // Add suggestions for common mistakes
        if let Some(suggestion) = Self::suggest_token_fix(expected, found, source, span) {
            diag = diag.with_suggestion(
                suggestion.0,
                span,
                suggestion.1,
            );
        }

        diag
    }

    /// Create an "undefined variable" error with "did you mean?" suggestions.
    pub fn undefined_variable(
        name: &str,
        span: Span,
        known_names: &[&str],
    ) -> Diagnostic {
        let message = format!("cannot find value `{}` in this scope", name);
        let mut diag = Diagnostic::error(message, span)
            .with_code("E0425")
            .with_label(span, "not found in this scope");

        // Find similar names
        if let Some(suggestion) = Self::find_similar(name, known_names) {
            diag = diag.with_suggestion(
                format!("a local variable with a similar name exists"),
                span,
                suggestion.to_string(),
            );
        }

        diag
    }

    /// Create a "type mismatch" error.
    pub fn type_mismatch(
        expected: &str,
        found: &str,
        span: Span,
        expected_span: Option<Span>,
    ) -> Diagnostic {
        let message = format!("mismatched types: expected `{}`, found `{}`", expected, found);
        let mut diag = Diagnostic::error(message, span)
            .with_code("E0308")
            .with_label(span, format!("expected `{}`", expected));

        if let Some(exp_span) = expected_span {
            diag = diag.with_related("expected due to this", exp_span);
        }

        diag
    }

    /// Create an "evidentiality mismatch" error.
    pub fn evidentiality_mismatch(
        expected: &str,
        found: &str,
        span: Span,
    ) -> Diagnostic {
        let message = format!(
            "evidentiality mismatch: expected `{}`, found `{}`",
            expected, found
        );

        Diagnostic::error(message, span)
            .with_code("E0600")
            .with_label(span, format!("has evidentiality `{}`", found))
            .with_note(format!(
                "values with `{}` evidentiality cannot be used where `{}` is required",
                found, expected
            ))
            .with_note(Self::evidentiality_help(expected, found))
    }

    /// Create an "untrusted data" error for evidentiality violations.
    pub fn untrusted_data_used(
        span: Span,
        source_span: Option<Span>,
    ) -> Diagnostic {
        let mut diag = Diagnostic::error(
            "cannot use reported (~) data without validation",
            span,
        )
        .with_code("E0601")
        .with_label(span, "untrusted data used here")
        .with_note("data from external sources must be validated before use")
        .with_suggestion(
            "validate the data first",
            span,
            "value|validate!{...}",
        );

        if let Some(src) = source_span {
            diag = diag.with_related("data originates from external source here", src);
        }

        diag
    }

    /// Create a "missing morpheme" error with ASCII alternatives.
    pub fn unknown_morpheme(
        found: &str,
        span: Span,
    ) -> Diagnostic {
        let message = format!("unknown morpheme `{}`", found);
        let mut diag = Diagnostic::error(message, span)
            .with_code("E0100");

        // Suggest similar morphemes
        let morphemes = [
            ("τ", "tau", "transform/map"),
            ("φ", "phi", "filter"),
            ("σ", "sigma", "sort"),
            ("ρ", "rho", "reduce"),
            ("λ", "lambda", "anonymous function"),
            ("Σ", "sum", "sum all"),
            ("Π", "pi", "product"),
            ("α", "alpha", "first element"),
            ("ω", "omega", "last element"),
            ("μ", "mu", "middle element"),
            ("χ", "chi", "random choice"),
            ("ν", "nu", "nth element"),
            ("ξ", "xi", "next in sequence"),
        ];

        if let Some((greek, _ascii, desc)) = morphemes.iter().find(|(g, a, _)| {
            jaro_winkler(found, g) > 0.8 || jaro_winkler(found, a) > 0.8
        }) {
            diag = diag.with_suggestion(
                format!("did you mean the {} morpheme?", desc),
                span,
                greek.to_string(),
            );
        }

        diag = diag.with_note("transform morphemes: τ (map), φ (filter), σ (sort), ρ (reduce), Σ (sum), Π (product)");
        diag = diag.with_note("access morphemes: α (first), ω (last), μ (middle), χ (choice), ν (nth), ξ (next)");

        diag
    }

    /// Suggest a Unicode symbol for an ASCII operator or name.
    /// Returns (unicode_symbol, description) if a suggestion exists.
    pub fn suggest_unicode_symbol(ascii: &str) -> Option<(&'static str, &'static str)> {
        match ascii {
            // Logic operators
            "&&" => Some(("∧", "logical AND")),
            "||" => Some(("∨", "logical OR")),
            "^^" => Some(("⊻", "logical XOR")),

            // Bitwise operators
            "&" => Some(("⋏", "bitwise AND")),
            "|" => Some(("⋎", "bitwise OR")),

            // Set operations
            "union" => Some(("∪", "set union")),
            "intersect" | "intersection" => Some(("∩", "set intersection")),
            "subset" => Some(("⊂", "proper subset")),
            "superset" => Some(("⊃", "proper superset")),
            "in" | "element_of" => Some(("∈", "element of")),
            "not_in" => Some(("∉", "not element of")),

            // Math symbols
            "sqrt" => Some(("√", "square root")),
            "cbrt" => Some(("∛", "cube root")),
            "infinity" | "inf" => Some(("∞", "infinity")),
            "pi" => Some(("π", "pi constant")),
            "sum" => Some(("Σ", "summation")),
            "product" => Some(("Π", "product")),
            "integral" => Some(("∫", "integral/cumulative sum")),
            "partial" | "derivative" => Some(("∂", "partial/derivative")),

            // Morphemes
            "tau" | "map" | "transform" => Some(("τ", "transform morpheme")),
            "phi" | "filter" => Some(("φ", "filter morpheme")),
            "sigma" | "sort" => Some(("σ", "sort morpheme")),
            "rho" | "reduce" | "fold" => Some(("ρ", "reduce morpheme")),
            "lambda" => Some(("λ", "lambda")),
            "alpha" | "first" => Some(("α", "first element")),
            "omega" | "last" => Some(("ω", "last element")),
            "mu" | "middle" | "median" => Some(("μ", "middle element")),
            "chi" | "choice" | "random" => Some(("χ", "random choice")),
            "nu" | "nth" => Some(("ν", "nth element")),
            "xi" | "next" => Some(("ξ", "next in sequence")),
            "delta" | "diff" | "change" => Some(("δ", "delta/change")),
            "epsilon" | "empty" => Some(("ε", "epsilon/empty")),
            "zeta" | "zip" => Some(("ζ", "zeta/zip")),

            // Category theory
            "compose" => Some(("∘", "function composition")),
            "tensor" => Some(("⊗", "tensor product")),
            "direct_sum" | "xor" => Some(("⊕", "direct sum/XOR")),

            // Special values
            "null" | "void" | "nothing" => Some(("∅", "empty set")),
            "true" | "top" | "any" => Some(("⊤", "top/true")),
            "false" | "bottom" | "never" => Some(("⊥", "bottom/false")),

            // Quantifiers
            "forall" | "for_all" => Some(("∀", "universal quantifier")),
            "exists" => Some(("∃", "existential quantifier")),

            // Data operations
            "join" | "zip_with" => Some(("⋈", "join/zip with")),
            "flatten" => Some(("⋳", "flatten")),
            "max" | "supremum" => Some(("⊔", "supremum/max")),
            "min" | "infimum" => Some(("⊓", "infimum/min")),

            _ => None,
        }
    }

    /// Create a hint diagnostic suggesting a Unicode symbol.
    pub fn suggest_symbol_upgrade(
        ascii: &str,
        span: Span,
    ) -> Option<Diagnostic> {
        Self::suggest_unicode_symbol(ascii).map(|(unicode, desc)| {
            Diagnostic::warning(
                format!("consider using Unicode symbol `{}` for {}", unicode, desc),
                span,
            )
            .with_code("W0200")
            .with_suggestion(
                format!("use `{}` for clearer, more idiomatic Sigil", unicode),
                span,
                unicode.to_string(),
            )
            .with_note(format!(
                "Sigil supports Unicode symbols. `{}` → `{}` ({})",
                ascii, unicode, desc
            ))
        })
    }

    /// Get all available symbol mappings for documentation/completion.
    pub fn all_symbol_mappings() -> Vec<(&'static str, &'static str, &'static str)> {
        vec![
            // (ascii, unicode, description)
            ("&&", "∧", "logical AND"),
            ("||", "∨", "logical OR"),
            ("^^", "⊻", "logical XOR"),
            ("&", "⋏", "bitwise AND"),
            ("|", "⋎", "bitwise OR"),
            ("union", "∪", "set union"),
            ("intersect", "∩", "set intersection"),
            ("subset", "⊂", "proper subset"),
            ("superset", "⊃", "proper superset"),
            ("in", "∈", "element of"),
            ("not_in", "∉", "not element of"),
            ("sqrt", "√", "square root"),
            ("cbrt", "∛", "cube root"),
            ("infinity", "∞", "infinity"),
            ("tau", "τ", "transform"),
            ("phi", "φ", "filter"),
            ("sigma", "σ", "sort"),
            ("rho", "ρ", "reduce"),
            ("lambda", "λ", "lambda"),
            ("alpha", "α", "first"),
            ("omega", "ω", "last"),
            ("mu", "μ", "middle"),
            ("chi", "χ", "choice"),
            ("nu", "ν", "nth"),
            ("xi", "ξ", "next"),
            ("sum", "Σ", "sum"),
            ("product", "Π", "product"),
            ("compose", "∘", "compose"),
            ("tensor", "⊗", "tensor"),
            ("xor", "⊕", "direct sum"),
            ("forall", "∀", "for all"),
            ("exists", "∃", "exists"),
            ("null", "∅", "empty"),
            ("true", "⊤", "top/true"),
            ("false", "⊥", "bottom/false"),
        ]
    }

    /// Find a similar name using Jaro-Winkler distance.
    fn find_similar<'a>(name: &str, candidates: &[&'a str]) -> Option<&'a str> {
        candidates
            .iter()
            .filter(|c| jaro_winkler(name, c) > 0.8)
            .max_by(|a, b| {
                jaro_winkler(name, a)
                    .partial_cmp(&jaro_winkler(name, b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
    }

    /// Suggest fixes for common token mistakes.
    fn suggest_token_fix(
        expected: &str,
        found: &Token,
        _source: &str,
        _span: Span,
    ) -> Option<(String, String)> {
        // Common mistake patterns
        match (expected, found) {
            ("`;`", Token::RBrace) => Some((
                "you might be missing a semicolon".to_string(),
                ";".to_string(),
            )),
            ("`{`", Token::Arrow) => Some((
                "you might want a block here".to_string(),
                "{ ... }".to_string(),
            )),
            ("`)`", Token::Comma) => Some((
                "unexpected comma, maybe close the parenthesis first".to_string(),
                ")".to_string(),
            )),
            _ => None,
        }
    }

    /// Generate help text for evidentiality issues.
    fn evidentiality_help(expected: &str, found: &str) -> String {
        match (expected, found) {
            ("!", "~") => "use `value|validate!{...}` to promote reported data to known".to_string(),
            ("!", "?") => "handle the uncertain case with `match` or unwrap with `value!`".to_string(),
            ("?", "~") => "reported data is already uncertain, no conversion needed".to_string(),
            _ => format!("evidentiality flows: ! (known) < ? (uncertain) < ~ (reported) < ‽ (paradox)"),
        }
    }
}

/// Collection of diagnostics for a compilation unit.
#[derive(Debug, Default)]
pub struct Diagnostics {
    items: Vec<Diagnostic>,
    has_errors: bool,
}

impl Diagnostics {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, diagnostic: Diagnostic) {
        if diagnostic.severity == Severity::Error {
            self.has_errors = true;
        }
        self.items.push(diagnostic);
    }

    pub fn has_errors(&self) -> bool {
        self.has_errors
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &Diagnostic> {
        self.items.iter()
    }

    /// Render all diagnostics.
    pub fn render_all(&self, filename: &str, source: &str) -> String {
        let mut output = String::new();
        for diag in &self.items {
            output.push_str(&diag.render(filename, source));
            output.push('\n');
        }
        output
    }

    /// Print all diagnostics to stderr.
    pub fn eprint_all(&self, filename: &str, source: &str) {
        for diag in &self.items {
            diag.eprint(filename, source);
        }
    }

    /// Get error count.
    pub fn error_count(&self) -> usize {
        self.items.iter().filter(|d| d.severity == Severity::Error).count()
    }

    /// Get warning count.
    pub fn warning_count(&self) -> usize {
        self.items.iter().filter(|d| d.severity == Severity::Warning).count()
    }

    /// Print summary.
    pub fn print_summary(&self) {
        let errors = self.error_count();
        let warnings = self.warning_count();

        if errors > 0 || warnings > 0 {
            eprint!("\n");
            if errors > 0 {
                eprintln!(
                    "{}: aborting due to {} previous error{}",
                    "error".fg(Color::Red),
                    errors,
                    if errors == 1 { "" } else { "s" }
                );
            }
            if warnings > 0 {
                eprintln!(
                    "{}: {} warning{} emitted",
                    "warning".fg(Color::Yellow),
                    warnings,
                    if warnings == 1 { "" } else { "s" }
                );
            }
        }
    }

    /// Convert all diagnostics to JSON output format.
    ///
    /// Returns a structured JSON object suitable for machine consumption,
    /// with line/column positions, fix suggestions, and metadata.
    pub fn to_json_output(&self, filename: &str, source: &str) -> JsonDiagnosticsOutput {
        let diagnostics: Vec<JsonDiagnostic> = self.items
            .iter()
            .map(|d| d.to_json(filename, source))
            .collect();

        let error_count = self.error_count();
        let warning_count = self.warning_count();

        JsonDiagnosticsOutput {
            file: filename.to_string(),
            diagnostics,
            error_count,
            warning_count,
            success: error_count == 0,
        }
    }

    /// Render diagnostics as JSON string.
    ///
    /// For AI agent consumption - provides structured, machine-readable output.
    pub fn to_json_string(&self, filename: &str, source: &str) -> String {
        let output = self.to_json_output(filename, source);
        serde_json::to_string_pretty(&output).unwrap_or_else(|_| "{}".to_string())
    }

    /// Render diagnostics as compact JSON (single line).
    ///
    /// For piping to other tools or streaming output.
    pub fn to_json_compact(&self, filename: &str, source: &str) -> String {
        let output = self.to_json_output(filename, source);
        serde_json::to_string(&output).unwrap_or_else(|_| "{}".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_undefined_variable_suggestion() {
        let known = vec!["counter", "count", "total", "sum"];
        let diag = DiagnosticBuilder::undefined_variable("countr", Span::new(10, 16), &known);

        assert!(diag.suggestions.iter().any(|s| s.replacement == "counter"));
    }

    #[test]
    fn test_evidentiality_mismatch() {
        let diag = DiagnosticBuilder::evidentiality_mismatch("!", "~", Span::new(0, 5));

        assert!(diag.notes.iter().any(|n| n.contains("validate")));
    }

    #[test]
    fn test_unicode_symbol_suggestions() {
        // Logic operators
        assert_eq!(DiagnosticBuilder::suggest_unicode_symbol("&&"), Some(("∧", "logical AND")));
        assert_eq!(DiagnosticBuilder::suggest_unicode_symbol("||"), Some(("∨", "logical OR")));

        // Bitwise operators
        assert_eq!(DiagnosticBuilder::suggest_unicode_symbol("&"), Some(("⋏", "bitwise AND")));
        assert_eq!(DiagnosticBuilder::suggest_unicode_symbol("|"), Some(("⋎", "bitwise OR")));

        // Morphemes
        assert_eq!(DiagnosticBuilder::suggest_unicode_symbol("tau"), Some(("τ", "transform morpheme")));
        assert_eq!(DiagnosticBuilder::suggest_unicode_symbol("filter"), Some(("φ", "filter morpheme")));
        assert_eq!(DiagnosticBuilder::suggest_unicode_symbol("alpha"), Some(("α", "first element")));

        // Math
        assert_eq!(DiagnosticBuilder::suggest_unicode_symbol("sqrt"), Some(("√", "square root")));
        assert_eq!(DiagnosticBuilder::suggest_unicode_symbol("infinity"), Some(("∞", "infinity")));

        // Unknown
        assert_eq!(DiagnosticBuilder::suggest_unicode_symbol("foobar"), None);
    }

    #[test]
    fn test_symbol_upgrade_diagnostic() {
        let diag = DiagnosticBuilder::suggest_symbol_upgrade("&&", Span::new(0, 2));
        assert!(diag.is_some());

        let d = diag.unwrap();
        assert!(d.suggestions.iter().any(|s| s.replacement == "∧"));
        assert!(d.notes.iter().any(|n| n.contains("logical AND")));
    }

    #[test]
    fn test_unknown_morpheme_with_access_morphemes() {
        let diag = DiagnosticBuilder::unknown_morpheme("alph", Span::new(0, 4));

        // Should suggest alpha
        assert!(diag.suggestions.iter().any(|s| s.replacement == "α"));
        // Should have notes about both transform and access morphemes
        assert!(diag.notes.iter().any(|n| n.contains("transform morphemes")));
        assert!(diag.notes.iter().any(|n| n.contains("access morphemes")));
    }

    #[test]
    fn test_all_symbol_mappings() {
        let mappings = DiagnosticBuilder::all_symbol_mappings();
        assert!(!mappings.is_empty());

        // Check that essential mappings exist
        assert!(mappings.iter().any(|(a, u, _)| *a == "&&" && *u == "∧"));
        assert!(mappings.iter().any(|(a, u, _)| *a == "tau" && *u == "τ"));
        assert!(mappings.iter().any(|(a, u, _)| *a == "sqrt" && *u == "√"));
    }
}
