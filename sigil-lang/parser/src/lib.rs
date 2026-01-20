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

pub mod ast;
pub mod diagnostic;
pub mod ffi;
pub mod interpreter;
pub mod ir;
pub mod lexer;
pub mod lint;
pub mod lower;
pub mod optimize;
pub mod parser;
pub mod plurality;
pub mod span;
pub mod stdlib;
pub mod typeck;
pub mod tree_sitter_support;

// New v0.4.0 features
pub mod fmt;
pub mod lsp;
pub mod tome;

#[cfg(feature = "jit")]
pub mod codegen;

#[cfg(feature = "llvm")]
pub mod llvm_codegen;

#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(feature = "protocol-core")]
pub mod protocol;

pub use ast::*;
pub use diagnostic::{Diagnostic, DiagnosticBuilder, Diagnostics, FixSuggestion, Severity};
pub use interpreter::{Evidence, Function, Interpreter, RuntimeError, Value};
pub use ir::{IrDumpOptions, IrEvidence, IrFunction, IrModule, IrOperation, IrType};
pub use lexer::{Lexer, Token};
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
