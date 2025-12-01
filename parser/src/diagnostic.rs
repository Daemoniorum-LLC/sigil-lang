//! Rich diagnostic reporting for Sigil.
//!
//! Provides Rust-quality error messages with:
//! - Colored output with source context
//! - "Did you mean?" suggestions
//! - Fix suggestions
//! - Multi-span support for related information

use ariadne::{Color, ColorGenerator, Config, Fmt, Label, Report, ReportKind, Source};
use std::ops::Range;
use strsim::jaro_winkler;

use crate::span::Span;
use crate::lexer::Token;

/// Severity level for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone)]
pub struct FixSuggestion {
    pub message: String,
    pub span: Span,
    pub replacement: String,
}

/// A related location with additional context.
#[derive(Debug, Clone)]
pub struct RelatedInfo {
    pub message: String,
    pub span: Span,
}

/// A rich diagnostic with all context needed for beautiful error reporting.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub severity: Severity,
    pub code: Option<String>,
    pub message: String,
    pub span: Span,
    pub labels: Vec<(Span, String)>,
    pub notes: Vec<String>,
    pub suggestions: Vec<FixSuggestion>,
    pub related: Vec<RelatedInfo>,
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

        diag = diag.with_note("available morphemes: τ (map), φ (filter), σ (sort), ρ (reduce), λ (lambda), Σ (sum), Π (product)");

        diag
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
}
