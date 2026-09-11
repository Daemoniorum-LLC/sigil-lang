//! Sigil Parser Library
//!
//! A polysynthetic programming language with evidentiality types.
//!
//! This crate provides:
//! - Lexer and parser for Sigil source code
//! - Tree-walking interpreter for development/debugging
//! - JIT compiler using Cranelift for native performance
//! - Comprehensive optimization passes (O0-O3)
//! - Rich diagnostic reporting with colored output
//! - AI-facing IR for tooling and agent integration

use std::sync::atomic::{AtomicBool, Ordering};

/// Global verbose flag for debug output control.
/// Set via `set_verbose(true)` or `--verbose` CLI flag.
static VERBOSE: AtomicBool = AtomicBool::new(false);

/// Enable or disable verbose debug output.
pub fn set_verbose(enabled: bool) {
    VERBOSE.store(enabled, Ordering::SeqCst);
}

/// Check if verbose mode is enabled.
pub fn is_verbose() -> bool {
    VERBOSE.load(Ordering::SeqCst)
}

/// Print debug message only when verbose mode is enabled.
#[macro_export]
macro_rules! sigil_debug {
    ($($arg:tt)*) => {
        if $crate::is_verbose() {
            eprintln!($($arg)*);
        }
    };
}

/// Print warning message only when verbose mode is enabled.
#[macro_export]
macro_rules! sigil_warn {
    ($($arg:tt)*) => {
        if $crate::is_verbose() {
            eprintln!($($arg)*);
        }
    };
}

/// Global switch for the `.sigil`-extension deprecation warning (LARES-363).
/// Set via `set_deprecation_warnings(false)` or the `--no-deprecation-warnings` CLI flag.
static DEPRECATION_WARNINGS: AtomicBool = AtomicBool::new(true);

/// Enable or disable the `.sigil` deprecation warning process-wide.
pub fn set_deprecation_warnings(enabled: bool) {
    DEPRECATION_WARNINGS.store(enabled, Ordering::SeqCst);
}

/// Check whether the `.sigil` deprecation warning is enabled.
pub fn deprecation_warnings_enabled() -> bool {
    DEPRECATION_WARNINGS.load(Ordering::SeqCst)
}

/// Whether `path` carries the deprecated `.sigil` extension. Pure and
/// independent of the enabled/dedup state, so the extension rule itself is
/// unit-testable without touching global state or stderr.
pub fn is_deprecated_sigil_extension<P: AsRef<std::path::Path>>(path: P) -> bool {
    path.as_ref().extension().map_or(false, |ext| ext == "sigil")
}

/// Warn once per process, per path, when a `.sigil` source file is read.
///
/// `.sg` is the canonical extension; `.sigil` is deprecated but keeps
/// compiling (LARES-363) — this only names the path and points at the
/// replacement. A `.sg` file (or anything else) triggers nothing.
/// Suppressible via `--no-deprecation-warnings` or `set_deprecation_warnings(false)`.
pub fn warn_if_deprecated_extension<P: AsRef<std::path::Path>>(path: P) {
    use std::collections::HashSet;
    use std::sync::Mutex;

    if !deprecation_warnings_enabled() {
        return;
    }

    let path = path.as_ref();
    if !is_deprecated_sigil_extension(path) {
        return;
    }

    static WARNED: Mutex<Option<HashSet<std::path::PathBuf>>> = Mutex::new(None);
    let mut guard = WARNED.lock().unwrap();
    let seen = guard.get_or_insert_with(HashSet::new);
    if seen.insert(path.to_path_buf()) {
        eprintln!(
            "warning: `{}` uses the deprecated `.sigil` extension; rename to `.sg` (suppress with --no-deprecation-warnings)",
            path.display()
        );
    }
}

#[cfg(test)]
mod deprecation_warning_tests {
    use super::*;

    // AC: "A `.sigil` file emits the warning... A `.sg` file emits NOTHING.
    // A warning that fires on both is noise." This is the rule itself, kept
    // separate from the printing/dedup side effects so it can be asserted
    // directly rather than by scraping stderr.
    #[test]
    fn sigil_extension_is_flagged() {
        assert!(is_deprecated_sigil_extension("foo.sigil"));
        assert!(is_deprecated_sigil_extension("dir/nested/lib.sigil"));
    }

    #[test]
    fn sg_extension_is_not_flagged() {
        assert!(!is_deprecated_sigil_extension("foo.sg"));
        assert!(!is_deprecated_sigil_extension("dir/nested/lib.sg"));
    }

    #[test]
    fn unrelated_extensions_are_not_flagged() {
        assert!(!is_deprecated_sigil_extension("foo.rs"));
        assert!(!is_deprecated_sigil_extension("Sigil.toml"));
        assert!(!is_deprecated_sigil_extension("noext"));
    }

    // The only test that touches the global suppression switch; every other
    // test in this module must leave it alone, since `cargo test` runs tests
    // in parallel within one process and the flag is process-wide.
    #[test]
    fn suppression_flag_round_trips() {
        let original = deprecation_warnings_enabled();
        set_deprecation_warnings(false);
        assert!(!deprecation_warnings_enabled());
        set_deprecation_warnings(true);
        assert!(deprecation_warnings_enabled());
        set_deprecation_warnings(original);
    }
}

pub mod ast;
pub mod cfg;
pub mod diagnostic;
pub mod ffi;
pub mod impl_registry;
pub mod interpreter;
pub mod ir;
pub mod lexer;
pub mod lower;
pub mod monomorph;
pub mod const_eval;
pub mod optimize;
pub mod parser;
pub mod plurality;
pub mod span;
pub mod stdlib;
pub mod typeck;
pub mod holographic;

#[cfg(feature = "native")]
pub mod lint;
#[cfg(feature = "native")]
pub mod tree_sitter_support;
#[cfg(feature = "native")]
pub mod fmt;

// New v0.4.0 features
#[cfg(feature = "lsp")]
pub mod lsp;
pub mod tome;

#[cfg(feature = "jit")]
pub mod codegen;

#[cfg(feature = "llvm")]
pub mod llvm_codegen;

pub mod rust_codegen;

pub mod async_transform;

#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(feature = "protocol-core")]
pub mod protocol;

#[cfg(feature = "websocket")]
pub mod websocket;

#[cfg(feature = "playground")]
pub mod playground_api;

#[cfg(feature = "react-migrate")]
pub mod migrate;

pub use ast::*;
pub use diagnostic::{Diagnostic, DiagnosticBuilder, Diagnostics, FixSuggestion, Severity};
pub use interpreter::{Evidence, Function, Interpreter, RuntimeError, Value};
pub use ir::{IrDumpOptions, IrEvidence, IrFunction, IrModule, IrOperation, IrType};
pub use lexer::{Lexer, Token};
#[cfg(feature = "native")]
pub use lint::{
    lint_file, lint_source, lint_source_with_config, lint_directory, lint_directory_parallel,
    lint_and_fix, apply_fixes, watch_directory,
    LintConfig, LintConfigFile, LintSettings, LintId, LintLevel, LintCategory, Linter,
    DirectoryLintResult, FixResult, WatchConfig, WatchResult,
    // Phase 6: Suppressions, SARIF, Stats, Explain
    Suppression, parse_suppressions, LintStats,
    SarifReport, generate_sarif, generate_sarif_for_directory,
    explain_lint, list_lints,
    // Phase 7: Baseline support, CLI overrides, caching
    Baseline, BaselineEntry, BaselineSummary, BaselineLintResult,
    find_baseline, lint_with_baseline,
    CliOverrides, config_with_overrides, lint_source_with_overrides,
    LintCache, CacheEntry, CachedDiagnostic, CacheStats, IncrementalLintResult,
    find_cache, lint_directory_incremental, CACHE_FILE,
    // Phase 8: LSP support
    LspSeverity, LspDiagnostic, LspRelatedInfo, LspCodeAction, LspTextEdit,
    LspLintResult, LspServerState, lint_for_lsp,
    // Phase 9: Git integration
    GitIntegration, lint_changed_files, lint_changed_since, lint_files,
    generate_pre_commit_hook, PRE_COMMIT_HOOK,
    // Phase 10: Custom rules
    CustomRule, CustomPattern, CustomRulesFile, CustomRuleMatch, CustomRuleChecker,
    lint_with_custom_rules,
    // Phase 11: Ignore patterns
    IgnorePatterns, filter_ignored, collect_sigil_files_filtered, lint_directory_filtered,
    // Phase 12: HTML reports and trend tracking
    LintReport, TrendData, TrendDirection, TrendSummary,
    generate_html_report, save_html_report, CiFormat, generate_ci_annotations,
};
pub use lower::lower_source_file;
pub use optimize::{optimize, OptLevel, OptStats, Optimizer};
pub use parser::Parser;
pub use span::Span;
pub use stdlib::register_stdlib;
pub use typeck::{EvidenceLevel, Type, TypeChecker, TypeError};

#[cfg(feature = "jit")]
pub use codegen::JitCompiler;

#[cfg(feature = "llvm")]
pub use llvm_codegen::llvm::{CompileMode, LlvmCompiler};

#[cfg(feature = "wasm")]
pub use wasm::WasmCompiler;

pub use rust_codegen::{RustCompiler, RustCodegenOptions, RustCodegenError, RustEdition};
