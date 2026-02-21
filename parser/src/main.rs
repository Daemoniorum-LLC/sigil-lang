//! Sigil CLI - Parse, check, and run Sigil source files.

use sigil_parser::lower::lower_source_file;
#[cfg(feature = "lsp")]
use sigil_parser::lsp::start_lsp;
use sigil_parser::span::Span;
use sigil_parser::typeck::TypeChecker;
#[cfg(feature = "jit")]
use sigil_parser::JitCompiler;
use sigil_parser::{register_stdlib, set_verbose, Diagnostic, Diagnostics, Interpreter, Lexer, Parser, Token};
#[cfg(feature = "llvm")]
use sigil_parser::{CompileMode, LlvmCompiler, OptLevel};
#[cfg(feature = "wasm")]
use sigil_parser::WasmCompiler;
use std::borrow::Cow;
use std::env;
use std::fs;
use std::process::ExitCode;

/// Output format for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq)]
enum OutputFormat {
    Human,   // Pretty-printed with colors
    Json,    // JSON for AI agents (pretty)
    Compact, // JSON single-line for piping
    Sarif,   // SARIF for IDE/CI integration
}

use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::{Config, Editor, Helper};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();

    // Handle --verbose / -v flag globally
    if args.iter().any(|a| a == "--verbose" || a == "-v") {
        set_verbose(true);
    }

    // Filter out global flags for command processing
    let args: Vec<String> = args.into_iter()
        .filter(|a| a != "--verbose" && a != "-v")
        .collect();

    if args.len() < 2 {
        eprintln!("Sigil v0.1.0 - A polysynthetic programming language");
        eprintln!();
        eprintln!("Usage: sigil <command> [file.sigil] [options]");
        eprintln!();
        eprintln!("Commands:");
        eprintln!("  run <file>      Execute a Sigil file (interpreted)");
        eprintln!("  run-dir <dir>   Execute all .sg/.sigil files in dir (multi-module)");
        eprintln!("  run-ws [bin]    Run a workspace (reads Sigil.toml, optional bin crate name)");
        eprintln!("  jit <file>      Execute a Sigil file (JIT compiled, fast)");
        eprintln!("  llvm <file>     Execute a Sigil file (LLVM backend, fastest)");
        eprintln!("  compile <file>  Compile to native executable (AOT, --lto for LTO)");
        eprintln!("  rust <file>     Transpile to Rust source code");
        eprintln!("  check <file>    Type-check and validate (for AI agents: --format=json)");
        eprintln!("  lint <path>     Run linter on file or directory (--format=json for AI)");
        eprintln!("  dump-ir <file>  Dump AI-facing IR as JSON (for agents/tooling)");
        eprintln!("  doc-extract <file>  Extract SGDOC documentation from source file");
        eprintln!("  parse <file>    Parse and check a Sigil file");
        eprintln!("  lex <file>      Tokenize a Sigil file");
        eprintln!("  repl            Start interactive REPL");
        eprintln!("  lsp             Start Language Server Protocol server");
        eprintln!();
        eprintln!("Project Commands:");
        eprintln!("  new <name>      Create a new Sigil project");
        eprintln!("  init            Initialize a Sigil project in current directory");
        eprintln!("  test            Run tests in the current project");
        eprintln!("  build           Build the current project");
        eprintln!("  migrate <file|dir>  Convert Rust syntax to native Sigil");
        eprintln!("                      Options: --dry-run, --backup, --workspace");
        eprintln!();
        eprintln!("AI Agent Options (for 'check' command):");
        eprintln!("  --format=json       Output diagnostics as JSON (pretty-printed)");
        eprintln!("  --format=compact    Output diagnostics as single-line JSON");
        eprintln!("  --quiet             Exit code only, no output (for fast validation)");
        eprintln!("  --apply-suggestions Auto-apply fix suggestions (alias: --fix)");
        eprintln!();
        eprintln!("AI IR Options (for 'dump-ir' command):");
        eprintln!("  --pretty            Pretty-print JSON output (default)");
        eprintln!("  --compact           Single-line JSON output");
        eprintln!("  -o <file>           Write IR to file instead of stdout");
        return ExitCode::from(1);
    }

    match args[1].as_str() {
        "run" => {
            if args.len() < 3 {
                eprintln!("Error: missing file argument");
                return ExitCode::from(1);
            }
            // Check for --no-typecheck flag
            let skip_typecheck = args.iter().any(|a| a == "--no-typecheck");
            // Get the file argument (first non-flag arg after "run")
            let file_arg = args[2..].iter()
                .find(|a| !a.starts_with("--") && !a.starts_with("-"))
                .map(|s| s.as_str())
                .unwrap_or(&args[2]);
            // Collect program args (after --)
            let program_args: Vec<String> = if let Some(pos) = args.iter().position(|a| a == "--") {
                args[pos+1..].to_vec()
            } else {
                vec![]
            };
            run_file(file_arg, &program_args, skip_typecheck)
        }
        "run-dir" => {
            if args.len() < 3 {
                eprintln!("Error: missing directory argument");
                return ExitCode::from(1);
            }
            // Collect program args (after --)
            let program_args: Vec<String> = if let Some(pos) = args.iter().position(|a| a == "--") {
                args[pos+1..].to_vec()
            } else {
                vec![]
            };
            run_directory(&args[2], &program_args)
        }
        "run-ws" => {
            // Optional: specify which binary crate to run (defaults to first found)
            let bin_name = if args.len() >= 3 && !args[2].starts_with('-') {
                Some(args[2].as_str())
            } else {
                None
            };
            // Collect program args (after --)
            let program_args: Vec<String> = if let Some(pos) = args.iter().position(|a| a == "--") {
                args[pos+1..].to_vec()
            } else {
                vec![]
            };
            run_workspace(bin_name, &program_args)
        }
        #[cfg(feature = "jit")]
        "jit" => {
            if args.len() < 3 {
                eprintln!("Error: missing file argument");
                return ExitCode::from(1);
            }
            jit_file(&args[2])
        }
        #[cfg(not(feature = "jit"))]
        "jit" => {
            eprintln!("Error: JIT compilation not available (compile with --features jit)");
            ExitCode::from(1)
        }
        #[cfg(feature = "llvm")]
        "llvm" => {
            if args.len() < 3 {
                eprintln!("Error: missing file argument");
                return ExitCode::from(1);
            }
            llvm_file(&args[2])
        }
        #[cfg(not(feature = "llvm"))]
        "llvm" => {
            eprintln!("Error: LLVM backend not available (compile with --features llvm)");
            ExitCode::from(1)
        }
        #[cfg(feature = "llvm")]
        "compile" => {
            if args.len() < 3 {
                eprintln!("Error: missing file argument");
                eprintln!("Usage: sigil compile <file.sigil> [-o output] [--lib] [--lto] [--tls] [--cuda] [--native-runtime] [-O0|-O1|-O2|-O3|-Os]");
                return ExitCode::from(1);
            }
            // Parse flags
            let use_lto = args.iter().any(|a| a == "--lto");
            let use_tls = args.iter().any(|a| a == "--tls");
            let use_cuda = args.iter().any(|a| a == "--cuda");
            let use_native_runtime = args.iter().any(|a| a == "--native-runtime" || a == "--native");
            let is_library = args.iter().any(|a| a == "--lib" || a == "--shared");
            // Parse optimization level
            let opt_level = if args.iter().any(|a| a == "-O0" || a == "-Onone") {
                OptLevel::None
            } else if args.iter().any(|a| a == "-O1" || a == "-Obasic") {
                OptLevel::Basic
            } else if args.iter().any(|a| a == "-O2" || a == "-Ostandard") {
                OptLevel::Standard
            } else if args.iter().any(|a| a == "-Os" || a == "-Osize") {
                OptLevel::Size
            } else if args.iter().any(|a| a == "-O3" || a == "-Oaggressive") {
                OptLevel::Aggressive
            } else {
                // Default: Standard (-O2) - O3 can crash LLVM on complex nested loops
                OptLevel::Standard
            };
            let output = if let Some(pos) = args.iter().position(|a| a == "-o") {
                if pos + 1 < args.len() {
                    args[pos + 1].clone()
                } else {
                    args[2]
                        .trim_end_matches(".sigil")
                        .trim_end_matches(".sg")
                        .to_string()
                }
            } else {
                // Default output name: strip extension
                args[2]
                    .trim_end_matches(".sigil")
                    .trim_end_matches(".sg")
                    .to_string()
            };
            compile_file(&args[2], &output, use_lto, use_tls, use_cuda, use_native_runtime, is_library, opt_level)
        }
        #[cfg(not(feature = "llvm"))]
        "compile" => {
            eprintln!("Error: AOT compilation requires LLVM (compile with --features llvm)");
            ExitCode::from(1)
        }
        #[cfg(feature = "wasm")]
        "wasm" => {
            if args.len() < 3 {
                eprintln!("Error: missing file argument");
                eprintln!("Usage: sigil wasm <file.sigil> [-o output.wasm]");
                return ExitCode::from(1);
            }
            let output = if let Some(pos) = args.iter().position(|a| a == "-o") {
                if pos + 1 < args.len() {
                    args[pos + 1].clone()
                } else {
                    args[2]
                        .trim_end_matches(".sigil")
                        .trim_end_matches(".sg")
                        .to_string() + ".wasm"
                }
            } else {
                args[2]
                    .trim_end_matches(".sigil")
                    .trim_end_matches(".sg")
                    .to_string() + ".wasm"
            };
            wasm_compile_file(&args[2], &output)
        }
        #[cfg(not(feature = "wasm"))]
        "wasm" => {
            eprintln!("Error: WASM compilation requires --features wasm");
            ExitCode::from(1)
        }
        "rust" => {
            if args.len() < 3 {
                eprintln!("Error: missing file argument");
                eprintln!("Usage: sigil rust <file.sigil|dir> [-o output] [--preserve-evidence] [--no-std] [--emit-cargo] [--workspace]");
                return ExitCode::from(1);
            }
            let preserve_evidence = args.iter().any(|a| a == "--preserve-evidence");
            let no_std = args.iter().any(|a| a == "--no-std");
            let emit_comments = args.iter().any(|a| a == "--emit-comments");
            let emit_cargo = args.iter().any(|a| a == "--emit-cargo");
            let workspace = args.iter().any(|a| a == "--workspace");
            let output = if let Some(pos) = args.iter().position(|a| a == "-o") {
                if pos + 1 < args.len() {
                    Some(args[pos + 1].clone())
                } else {
                    None
                }
            } else {
                None
            };

            // Check if path is a directory (workspace mode)
            let path = &args[2];
            if std::path::Path::new(path).is_dir() || workspace {
                rust_compile_workspace(path, output.as_deref(), preserve_evidence, no_std, emit_comments, emit_cargo)
            } else {
                rust_compile_file(path, output.as_deref(), preserve_evidence, no_std, emit_comments)
            }
        }
        "check" => {
            if args.len() < 3 {
                eprintln!("Error: missing file argument");
                eprintln!("Usage: sigil check <file.sigil> [--format=json|compact] [--quiet] [--apply-suggestions]");
                return ExitCode::from(1);
            }
            // Parse format option
            let format = if args.iter().any(|a| a == "--format=json") {
                OutputFormat::Json
            } else if args.iter().any(|a| a == "--format=compact") {
                OutputFormat::Compact
            } else {
                OutputFormat::Human
            };
            let quiet = args.iter().any(|a| a == "--quiet");
            let apply_fixes = args
                .iter()
                .any(|a| a == "--apply-suggestions" || a == "--fix");
            check_file(&args[2], format, quiet, apply_fixes)
        }
        "lint" => {
            // Handle --init flag to generate default config
            if args.iter().any(|a| a == "--init") {
                return lint_init();
            }

            // Handle --list flag to show all lint rules
            if args.iter().any(|a| a == "--list" || a == "--list-lints") {
                return lint_list_rules();
            }

            // Handle --explain=<RULE> flag
            if let Some(explain_arg) = args.iter().find(|a| a.starts_with("--explain")) {
                let rule = if explain_arg.contains('=') {
                    explain_arg.strip_prefix("--explain=").unwrap_or("")
                } else if let Some(idx) = args.iter().position(|a| a == "--explain") {
                    if idx + 1 < args.len() {
                        &args[idx + 1]
                    } else {
                        ""
                    }
                } else {
                    ""
                };
                return lint_explain(rule);
            }

            if args.len() < 3 {
                eprintln!("Error: missing path argument");
                eprintln!("Usage: sigil lint <file.sigil|directory> [options]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --format=json|compact   Output format for AI agents");
                eprintln!("  --format=sarif          SARIF output for IDE/CI integration");
                eprintln!("  --config=<path>         Use specific config file");
                eprintln!("  --fix                   Apply auto-fix suggestions");
                eprintln!("  --watch                 Watch for changes and re-lint");
                eprintln!("  --parallel              Use parallel linting for directories");
                eprintln!("  --stats                 Show lint statistics");
                eprintln!("  --init                  Generate default .sigillint.toml");
                eprintln!("  --list                  List all available lint rules");
                eprintln!("  --explain=<RULE>        Show detailed docs for a rule (code or name)");
                eprintln!("  --evidentiality         Enable strict evidentiality checking");
                return ExitCode::from(1);
            }
            let format = if args.iter().any(|a| a == "--format=json") {
                OutputFormat::Json
            } else if args.iter().any(|a| a == "--format=compact") {
                OutputFormat::Compact
            } else if args.iter().any(|a| a == "--format=sarif") {
                OutputFormat::Sarif
            } else {
                OutputFormat::Human
            };

            // Get config path if specified
            let config_path = args.iter()
                .find(|a| a.starts_with("--config="))
                .map(|a| a.strip_prefix("--config=").unwrap());

            // Check for flags
            let apply_fix = args.iter().any(|a| a == "--fix");
            let watch_mode = args.iter().any(|a| a == "--watch");
            let parallel = args.iter().any(|a| a == "--parallel");
            let show_stats = args.iter().any(|a| a == "--stats");
            let evidentiality_mode = args.iter().any(|a| a == "--evidentiality");

            // Find the path argument (first non-flag after "lint")
            let path = args.iter().skip(2)
                .find(|a| !a.starts_with("--") && !a.starts_with("-c"))
                .map(|s| s.as_str());

            let path = match path {
                Some(p) => p,
                None => {
                    eprintln!("Error: missing file or directory argument");
                    eprintln!("Usage: sigil lint <path> [options]");
                    return ExitCode::from(1);
                }
            };

            if watch_mode {
                lint_watch(path, format, config_path)
            } else {
                lint_path(path, format, config_path, apply_fix, parallel, show_stats, evidentiality_mode)
            }
        }
        "dump-ir" => {
            if args.len() < 3 {
                eprintln!("Error: missing file argument");
                eprintln!(
                    "Usage: sigil dump-ir <file.sigil> [--pretty|--compact] [-o output.json]"
                );
                return ExitCode::from(1);
            }
            let pretty = !args.iter().any(|a| a == "--compact");
            let output = if let Some(pos) = args.iter().position(|a| a == "-o") {
                if pos + 1 < args.len() {
                    Some(args[pos + 1].clone())
                } else {
                    None
                }
            } else {
                None
            };
            dump_ir_file(&args[2], pretty, output.as_deref())
        }
        "doc-extract" => {
            if args.len() < 3 {
                eprintln!("Error: missing file argument");
                eprintln!("Usage: sigil doc-extract <file.sg> [--format=<json|markdown|html>] [-o output]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  --format=json      Output as JSON (default)");
                eprintln!("  --format=markdown  Output as Markdown");
                eprintln!("  --format=html      Output as HTML");
                eprintln!("  -o <file>          Write to file instead of stdout");
                return ExitCode::from(1);
            }
            let format = if args.iter().any(|a| a == "--format=markdown" || a == "--format=md") {
                "markdown"
            } else if args.iter().any(|a| a == "--format=html") {
                "html"
            } else {
                "json"
            };
            let output = if let Some(pos) = args.iter().position(|a| a == "-o") {
                if pos + 1 < args.len() {
                    Some(args[pos + 1].clone())
                } else {
                    None
                }
            } else {
                None
            };
            doc_extract_file(&args[2], format, output.as_deref())
        }
        "parse" => {
            if args.len() < 3 {
                eprintln!("Error: missing file argument");
                return ExitCode::from(1);
            }
            parse_file(&args[2])
        }
        "lex" => {
            if args.len() < 3 {
                eprintln!("Error: missing file argument");
                return ExitCode::from(1);
            }
            lex_file(&args[2])
        }
        "repl" => repl(),
        #[cfg(feature = "lsp")]
        "lsp" => start_lsp(),
        #[cfg(not(feature = "lsp"))]
        "lsp" => {
            eprintln!("Error: LSP support not enabled. Rebuild with --features lsp");
            ExitCode::from(1)
        }
        "new" => {
            if args.len() < 3 {
                eprintln!("Error: missing project name");
                eprintln!("Usage: sigil new <project-name>");
                return ExitCode::from(1);
            }
            new_project(&args[2])
        }
        "init" => init_project(),
        "test" => run_tests(),
        "build" => build_project(),
        "migrate" => {
            let dry_run = args.iter().any(|a| a == "--dry-run");
            let backup = args.iter().any(|a| a == "--backup");
            let evidentiality = args.iter().any(|a| a == "--evidentiality");
            let workspace = args.iter().any(|a| a == "--workspace");
            // Parse output directory option: -o <dir> or --output <dir>
            let output_dir: Option<String> = args.iter()
                .position(|a| a == "-o" || a == "--output")
                .and_then(|pos| args.get(pos + 1).cloned());

            // React migration (feature-gated)
            #[cfg(feature = "react-migrate")]
            if args.iter().any(|a| a == "--from-react") {
                return match sigil_parser::migrate::react::run_react_migrate(&args[2..]) {
                    Ok(()) => ExitCode::SUCCESS,
                    Err(e) => {
                        eprintln!("React migration error: {}", e);
                        ExitCode::from(1)
                    }
                };
            }

            // egui migration (feature-gated)
            #[cfg(feature = "egui-migrate")]
            if args.iter().any(|a| a == "--from-egui") {
                use sigil_parser::migrate::egui::{parse_egui_migrate_args, run_egui_migration};
                return match parse_egui_migrate_args(&args[2..]) {
                    Err(e) => {
                        eprintln!("egui migration argument error: {}", e);
                        ExitCode::from(1)
                    }
                    Ok(config) => match run_egui_migration(&config) {
                        Ok(summary) => {
                            eprintln!(
                                "[egui-migrate] Done: {} file(s), {} actor(s), {} spec(s) written",
                                summary.files_processed, summary.actors_found, summary.specs_written
                            );
                            ExitCode::SUCCESS
                        }
                        Err(e) => {
                            eprintln!("egui migration error: {}", e);
                            ExitCode::from(1)
                        }
                    },
                };
            }

            if workspace {
                // Migrate entire workspace from Sigil.toml
                migrate_workspace(dry_run, backup, evidentiality)
            } else if args.len() < 3 || args[2].starts_with('-') {
                eprintln!("Usage: sigil migrate <file|directory> [options]");
                eprintln!("       sigil migrate <file|directory> -o <output_dir> [options]");
                eprintln!("       sigil migrate --workspace [options]");
                #[cfg(feature = "react-migrate")]
                eprintln!("       sigil migrate --from-react <dir> [options]");
                #[cfg(feature = "egui-migrate")]
                eprintln!("       sigil migrate --from-egui <file|dir> [options]");
                eprintln!();
                eprintln!("Options:");
                eprintln!("  -o, --output     Output directory (writes .sg files, preserves structure)");
                eprintln!("  --dry-run        Show changes without applying");
                eprintln!("  --backup         Create .bak backup before modifying");
                eprintln!("  --evidentiality  Add evidentiality markers to external data sources");
                eprintln!("  --workspace      Migrate all files in workspace (reads Sigil.toml)");
                #[cfg(feature = "react-migrate")]
                eprintln!("  --from-react     Migrate React/TSX to Qliphoth actors");
                #[cfg(feature = "egui-migrate")]
                eprintln!("  --from-egui      Migrate egui Rust sources to Qliphoth actors");
                eprintln!();
                eprintln!("When -o is specified, .rs files are converted to .sg files in output dir.");
                eprintln!("Without -o, files are modified in-place (must be .sg or .sigil).");
                return ExitCode::from(1);
            } else {
                let path = std::path::Path::new(&args[2]);
                if path.is_dir() {
                    migrate_directory(&args[2], output_dir.as_deref(), dry_run, backup, evidentiality)
                } else {
                    migrate_file(&args[2], output_dir.as_deref(), dry_run, backup, evidentiality)
                }
            }
        }
        _ => {
            // Treat as file if it ends with .sigil or .sg
            if args[1].ends_with(".sigil") || args[1].ends_with(".sg") {
                // Collect program args (after --)
                let program_args: Vec<String> = if let Some(pos) = args.iter().position(|a| a == "--") {
                    args[pos+1..].to_vec()
                } else {
                    vec![]
                };
                run_file(&args[1], &program_args, false)
            } else {
                eprintln!("Unknown command: {}", args[1]);
                ExitCode::from(1)
            }
        }
    }
}

fn run_file(path: &str, program_args: &[String], skip_typecheck: bool) -> ExitCode {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", path, e);
            return ExitCode::from(1);
        }
    };

    // Parse
    let mut parser = Parser::new(&source);
    let ast = match parser.parse_file() {
        Ok(ast) => ast,
        Err(e) => {
            eprintln!("Parse error in '{}': {}", path, e);
            return ExitCode::from(1);
        }
    };

    // Type check (can be skipped with --no-typecheck for bootstrapping/testing)
    if !skip_typecheck {
        let mut type_checker = TypeChecker::new();
        if let Err(type_errors) = type_checker.check_file(&ast) {
            for err in type_errors {
                eprintln!("Type error in '{}': {}", path, err.message);
                for note in &err.notes {
                    eprintln!("  note: {}", note);
                }
            }
            return ExitCode::from(1);
        }
    }

    // Execute with full stdlib
    let mut interpreter = Interpreter::new();
    register_stdlib(&mut interpreter);

    // Set source code for span-to-line conversion in IR export
    interpreter.set_source_code(source);

    // Set source directory for module resolution
    if let Some(parent) = std::path::Path::new(path).parent() {
        let source_dir = parent.to_string_lossy().to_string();
        interpreter.set_current_source_dir(Some(source_dir.clone()));

        // Discover project (find Sigil.toml and parse workspace members)
        if let Err(e) = interpreter.discover_project(&source_dir) {
            eprintln!("Warning: failed to discover project: {}", e);
        }
    }
    
    // Set program arguments (program name + actual args)
    let program_name = std::path::Path::new(path)
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "sigil".to_string());
    let mut full_args = vec![program_name];
    full_args.extend(program_args.iter().cloned());
    interpreter.set_program_args(full_args);
    
    match interpreter.execute(&ast) {
        Ok(value) => {
            // Only print result if it's not null
            if !matches!(value, sigil_parser::Value::Null) {
                println!("{}", value);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Runtime error in '{}': {}", path, e);
            ExitCode::from(1)
        }
    }
}

/// Run all .sg/.sigil files in a directory as a multi-module program
fn run_directory(dir_path: &str, program_args: &[String]) -> ExitCode {
    use std::path::Path;

    // Collect all .sg and .sigil files
    let dir = match fs::read_dir(dir_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error reading directory '{}': {}", dir_path, e);
            return ExitCode::from(1);
        }
    };

    let mut files: Vec<String> = dir
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "sg" || ext == "sigil"))
        .map(|e| e.path().to_string_lossy().to_string())
        .collect();

    // Sort files to ensure proper load order:
    // 1. lib.sigil first (defines the module's public interface)
    // 2. Other modules in alphabetical order
    // 3. main.sigil last (uses definitions from other modules)
    files.sort_by(|a, b| {
        let a_name = Path::new(a).file_name().and_then(|n| n.to_str()).unwrap_or("");
        let b_name = Path::new(b).file_name().and_then(|n| n.to_str()).unwrap_or("");
        match (a_name, b_name) {
            ("lib.sigil", _) | ("lib.sg", _) => std::cmp::Ordering::Less,
            (_, "lib.sigil") | (_, "lib.sg") => std::cmp::Ordering::Greater,
            ("main.sigil", _) | ("main.sg", _) => std::cmp::Ordering::Greater,
            (_, "main.sigil") | (_, "main.sg") => std::cmp::Ordering::Less,
            _ => a_name.cmp(b_name),
        }
    });

    if files.is_empty() {
        eprintln!("No .sg or .sigil files found in '{}'", dir_path);
        return ExitCode::from(1);
    }

    eprintln!("Loading {} modules from '{}':", files.len(), dir_path);
    for f in &files {
        let name = Path::new(f).file_name().unwrap_or_default().to_string_lossy();
        eprintln!("  - {}", name);
    }

    // Create interpreter and register stdlib
    let mut interpreter = Interpreter::new();
    register_stdlib(&mut interpreter);

    // Get absolute path for source directory
    let abs_dir = match Path::new(dir_path).canonicalize() {
        Ok(p) => p,
        Err(_) => Path::new(dir_path).to_path_buf(),
    };

    // Derive crate name from directory name
    // If directory is "src", use parent directory name instead
    let dir_name = abs_dir
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("crate");

    let crate_name = if dir_name == "src" {
        // Use parent directory name
        abs_dir
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("crate")
            .to_string()
    } else {
        dir_name.to_string()
    };

    eprintln!("Crate name: {}", crate_name);

    // Set up interpreter state for multi-module project
    interpreter.set_current_source_dir(Some(abs_dir.to_string_lossy().to_string()));
    // Note: crate_name and crate_alias tracked in main.rs for multi-module compilation

    // Parse and execute each file to register its definitions
    for file_path in &files {
        let source = match fs::read_to_string(file_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error reading '{}': {}", file_path, e);
                return ExitCode::from(1);
            }
        };

        // Get module name from filename (without extension)
        let module_name = Path::new(file_path)
            .file_stem()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown");

        // Set current module context for this file
        // For lib.sg, use the crate name itself; for others, use the module name
        if module_name == "lib" {
            interpreter.set_current_module(None);
        } else {
            // Set module name so module-scoped items work
            interpreter.set_current_module(Some(module_name.to_string()));
        }

        let mut parser = Parser::new(&source);
        let ast = match parser.parse_file() {
            Ok(ast) => ast,
            Err(e) => {
                eprintln!("Parse error in '{}': {}", file_path, e);
                return ExitCode::from(1);
            }
        };

        // Execute to register all definitions
        if let Err(e) = interpreter.execute(&ast) {
            eprintln!("Error loading '{}': {}", file_path, e);
            return ExitCode::from(1);
        }
    }

    // Create program args array
    let args_value = sigil_parser::Value::Array(
        std::rc::Rc::new(std::cell::RefCell::new(
            program_args.iter()
                .map(|s| sigil_parser::Value::String(std::rc::Rc::new(s.clone())))
                .collect()
        ))
    );

    // Try to call main with args
    match interpreter.call_function_by_name("main", vec![args_value]) {
        Ok(value) => {
            // Check if result is an exit code
            match &value {
                sigil_parser::Value::Int(code) => ExitCode::from(*code as u8),
                sigil_parser::Value::Null => ExitCode::SUCCESS,
                _ => {
                    println!("{}", value);
                    ExitCode::SUCCESS
                }
            }
        }
        Err(e) => {
            eprintln!("Runtime error: {}", e);
            ExitCode::from(1)
        }
    }
}

/// Run a workspace project by loading all crates and running a binary crate
fn run_workspace(bin_name: Option<&str>, program_args: &[String]) -> ExitCode {
    use std::path::Path;
    use toml::Value as TomlValue;

    // Look for Sigil.toml in current directory
    let manifest_path = Path::new("Sigil.toml");
    if !manifest_path.exists() {
        // Also try sigil.toml (lowercase)
        let manifest_path = Path::new("sigil.toml");
        if !manifest_path.exists() {
            eprintln!("Error: No Sigil.toml found in current directory");
            eprintln!("Run this command from a Sigil workspace root");
            return ExitCode::from(1);
        }
    }

    let manifest_content = match fs::read_to_string("Sigil.toml")
        .or_else(|_| fs::read_to_string("sigil.toml")) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading Sigil.toml: {}", e);
            return ExitCode::from(1);
        }
    };

    let manifest: TomlValue = match manifest_content.parse() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error parsing Sigil.toml: {}", e);
            return ExitCode::from(1);
        }
    };

    // Get project name
    let project_name = manifest
        .get("project")
        .and_then(|p| p.get("name"))
        .and_then(|n| n.as_str())
        .unwrap_or("unnamed");

    eprintln!("Loading workspace: {}", project_name);

    // Get workspace members
    let members: Vec<String> = manifest
        .get("workspace")
        .and_then(|w| w.get("members"))
        .and_then(|m| m.as_array())
        .map(|arr| arr.iter()
            .filter_map(|v| v.as_str())
            .map(|s| s.to_string())
            .collect())
        .unwrap_or_default();

    if members.is_empty() {
        eprintln!("Error: No workspace members found in Sigil.toml");
        return ExitCode::from(1);
    }

    // Get dependencies (external path-based crates)
    let dependencies: Vec<(String, String)> = manifest
        .get("dependencies")
        .and_then(|d| d.as_table())
        .map(|table| {
            table.iter()
                .filter_map(|(name, value)| {
                    // Handle path dependencies: { path = "../../some/path" }
                    value.get("path")
                        .and_then(|p| p.as_str())
                        .map(|path| (name.replace('-', "_"), path.to_string()))
                })
                .collect()
        })
        .unwrap_or_default();

    if !dependencies.is_empty() {
        eprintln!("Found {} dependencies:", dependencies.len());
        for (name, path) in &dependencies {
            eprintln!("  - {} ({})", name, path);
        }
    }

    eprintln!("Found {} crates:", members.len());
    for member in &members {
        eprintln!("  - {}", member);
    }

    // Create interpreter and register stdlib
    let mut interpreter = Interpreter::new();
    register_stdlib(&mut interpreter);

    // Set program arguments (simulating what the program would see via env::args)
    // First arg is the "program name" (binary crate name), rest are actual args
    let binary_name = bin_name.unwrap_or("samael").to_string();
    let mut full_args = vec![binary_name];
    full_args.extend(program_args.iter().cloned());
    interpreter.set_program_args(full_args);

    // Track which crate we're loading for namespacing
    let mut binary_crate: Option<String> = None;

    // Load dependencies first (external path-based crates)
    for (dep_name, dep_path) in &dependencies {
        let crate_path = Path::new(dep_path);
        let src_path = crate_path.join("src");

        // Find all .sigil files in src/
        let src_dir = match fs::read_dir(&src_path) {
            Ok(d) => d,
            Err(_) => {
                eprintln!("  Warning: Could not read dependency {}/src/", dep_path);
                continue;
            }
        };

        let mut files: Vec<String> = src_dir
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "sigil" || ext == "sg"))
            .map(|e| e.path().to_string_lossy().to_string())
            .collect();

        // Sort files: lib.sigil first, main.sigil last
        files.sort_by(|a, b| {
            let a_name = Path::new(a).file_name().and_then(|n| n.to_str()).unwrap_or("");
            let b_name = Path::new(b).file_name().and_then(|n| n.to_str()).unwrap_or("");
            match (a_name, b_name) {
                ("lib.sigil", _) | ("lib.sg", _) => std::cmp::Ordering::Less,
                (_, "lib.sigil") | (_, "lib.sg") => std::cmp::Ordering::Greater,
                ("main.sigil", _) | ("main.sg", _) => std::cmp::Ordering::Greater,
                (_, "main.sigil") | (_, "main.sg") => std::cmp::Ordering::Less,
                _ => a_name.cmp(b_name),
            }
        });

        eprintln!("  Loading dependency {} ({} files)...", dep_name, files.len());

        // Load each file in the dependency
        for file_path in &files {
            let file_name = Path::new(file_path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?");
            eprintln!("    - {}", file_name);

            let module_name = if file_name != "lib.sigil" && file_name != "main.sigil"
                && file_name != "lib.sg" && file_name != "main.sg" {
                Path::new(file_name)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string())
            } else {
                None
            };
            interpreter.set_current_module(module_name);

            let source = match fs::read_to_string(file_path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("      Error reading: {}", e);
                    continue;
                }
            };

            let mut parser = Parser::new(&source);
            match parser.parse_file() {
                Ok(ast) => {
                    if let Err(e) = interpreter.execute_definitions(&ast) {
                        eprintln!("      Error in {}: {}", file_name, e);
                    }
                }
                Err(e) => {
                    eprintln!("      Parse error in {}: {:?}", file_name, e);
                }
            }
        }
    }

    // Load each crate in order (members should be in dependency order)
    for member in &members {
        let crate_path = Path::new(member);
        let src_path = crate_path.join("src");

        // Get crate name from path (last component, with - replaced by _)
        let crate_name = crate_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .replace('-', "_");

        // Check if this is a binary crate (has main.sigil)
        let main_path = src_path.join("main.sigil");
        let is_binary = main_path.exists();

        if is_binary {
            if bin_name.is_none() || bin_name == Some(crate_name.as_str()) {
                binary_crate = Some(crate_name.clone());
            }
        }

        // Find all .sigil files in src/ (including subdirectories)
        fn collect_sigil_files(dir: &std::path::Path, files: &mut Vec<String>) {
            if let Ok(entries) = fs::read_dir(dir) {
                for entry in entries.filter_map(|e| e.ok()) {
                    let path = entry.path();
                    if path.is_dir() {
                        // Recurse into subdirectories
                        collect_sigil_files(&path, files);
                    } else if path.extension().map_or(false, |ext| ext == "sigil" || ext == "sg") {
                        files.push(path.to_string_lossy().to_string());
                    }
                }
            }
        }

        let mut files: Vec<String> = Vec::new();
        collect_sigil_files(&src_path, &mut files);

        if files.is_empty() {
            eprintln!("  Warning: No .sigil files found in {}/src/", member);
            continue;
        }

        // Sort files to ensure proper load order:
        // 1. Root lib.sigil first (defines the crate's public interface)
        // 2. Root-level non-lib/main files
        // 3. Subdirectory mod.sigil files (in order of depth, then alphabetically)
        // 4. Subdirectory non-mod files
        // 5. main.sigil last (uses definitions from other modules)
        files.sort_by(|a, b| {
            let a_path = Path::new(a);
            let b_path = Path::new(b);
            let a_name = a_path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            let b_name = b_path.file_name().and_then(|n| n.to_str()).unwrap_or("");

            // Count directory depth relative to src/
            let a_depth = a.matches('/').count();
            let b_depth = b.matches('/').count();

            // Root-level lib.sigil always first
            let a_is_root_lib = a_depth <= 3 && (a_name == "lib.sigil" || a_name == "lib.sg");
            let b_is_root_lib = b_depth <= 3 && (b_name == "lib.sigil" || b_name == "lib.sg");
            if a_is_root_lib && !b_is_root_lib { return std::cmp::Ordering::Less; }
            if b_is_root_lib && !a_is_root_lib { return std::cmp::Ordering::Greater; }

            // main.sigil always last
            let a_is_main = a_name == "main.sigil" || a_name == "main.sg";
            let b_is_main = b_name == "main.sigil" || b_name == "main.sg";
            if a_is_main && !b_is_main { return std::cmp::Ordering::Greater; }
            if b_is_main && !a_is_main { return std::cmp::Ordering::Less; }

            // Sort by depth first (shallower files first)
            if a_depth != b_depth { return a_depth.cmp(&b_depth); }

            // Within same depth, mod.sigil comes first
            let a_is_mod = a_name == "mod.sigil" || a_name == "mod.sg";
            let b_is_mod = b_name == "mod.sigil" || b_name == "mod.sg";
            if a_is_mod && !b_is_mod { return std::cmp::Ordering::Less; }
            if b_is_mod && !a_is_mod { return std::cmp::Ordering::Greater; }

            // Otherwise alphabetical by full path
            a.cmp(b)
        });

        eprintln!("  Loading {} ({} files)...", crate_name, files.len());

        // Load each file in the crate
        for file_path in &files {
            let file_path_obj = Path::new(file_path);
            let file_name = file_path_obj
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?");

            // Get relative path from src/ for subdirectory files
            let relative_display = file_path_obj
                .strip_prefix(&src_path)
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_else(|_| file_name.to_string());
            eprintln!("    - {}", relative_display);

            // Set current module based on file path (for module-qualified function names)
            // e.g., "analyze.sigil" -> module name "analyze"
            //       "router/mod.sigil" -> module name "router"
            //       "router/types.sigil" -> module name "router·types"
            // Skip for lib.sigil and main.sigil as they are the crate root
            let module_name = if file_name != "lib.sigil" && file_name != "main.sigil"
                && file_name != "lib.sg" && file_name != "main.sg" {
                // Build module path from relative directory path + file stem
                let relative_path = file_path_obj.strip_prefix(&src_path).ok();
                let module_parts: Vec<String> = if let Some(rel) = relative_path {
                    let parent_parts: Vec<&str> = rel.parent()
                        .map(|p| p.iter().filter_map(|c| c.to_str()).collect())
                        .unwrap_or_default();
                    let stem = rel.file_stem().and_then(|s| s.to_str()).unwrap_or("");

                    // For mod.sigil files, use just the parent directory path
                    if stem == "mod" {
                        parent_parts.iter().map(|s| s.to_string()).collect()
                    } else {
                        // For other files, include parent path + file stem
                        let mut parts: Vec<String> = parent_parts.iter().map(|s| s.to_string()).collect();
                        parts.push(stem.to_string());
                        parts
                    }
                } else {
                    vec![file_path_obj.file_stem().and_then(|s| s.to_str()).unwrap_or("").to_string()]
                };

                if module_parts.is_empty() || (module_parts.len() == 1 && module_parts[0].is_empty()) {
                    None
                } else {
                    Some(module_parts.join("·"))
                }
            } else {
                None
            };
            interpreter.set_current_module(module_name);

            let source = match fs::read_to_string(file_path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Error reading '{}': {}", file_path, e);
                    return ExitCode::from(1);
                }
            };

            let mut parser = Parser::new(&source);
            let ast = match parser.parse_file() {
                Ok(ast) => ast,
                Err(e) => {
                    eprintln!("Parse error in '{}': {}", file_path, e);
                    return ExitCode::from(1);
                }
            };

            // Execute to register all definitions (but don't auto-call main yet)
            if let Err(e) = interpreter.execute_definitions(&ast) {
                eprintln!("Error loading '{}': {}", file_path, e);
                return ExitCode::from(1);
            }
        }

        // Clear module context after loading crate
        interpreter.set_current_module(None);
    }

    eprintln!("All crates loaded successfully.");

    // Check we found a binary crate to run
    let binary_crate = match binary_crate {
        Some(name) => name,
        None => {
            if let Some(name) = bin_name {
                eprintln!("Error: Binary crate '{}' not found in workspace", name);
            } else {
                eprintln!("Error: No binary crate found in workspace (no main.sigil)");
            }
            return ExitCode::from(1);
        }
    };

    eprintln!("Running binary: {}\n", binary_crate);

    // Create program args array
    let args_value = sigil_parser::Value::Array(
        std::rc::Rc::new(std::cell::RefCell::new(
            program_args.iter()
                .map(|s| sigil_parser::Value::String(std::rc::Rc::new(s.clone())))
                .collect()
        ))
    );

    // Try to call main
    match interpreter.call_function_by_name("main", vec![args_value.clone()]) {
        Ok(value) => {
            match &value {
                sigil_parser::Value::Int(code) => ExitCode::from(*code as u8),
                sigil_parser::Value::Null => ExitCode::SUCCESS,
                _ => {
                    println!("{}", value);
                    ExitCode::SUCCESS
                }
            }
        }
        Err(e) => {
            // Try calling main with no args
            match interpreter.call_function_by_name("main", vec![]) {
                Ok(value) => {
                    match &value {
                        sigil_parser::Value::Int(code) => ExitCode::from(*code as u8),
                        sigil_parser::Value::Null => ExitCode::SUCCESS,
                        _ => {
                            println!("{}", value);
                            ExitCode::SUCCESS
                        }
                    }
                }
                Err(e2) => {
                    eprintln!("Runtime error: {}", e2);
                    ExitCode::from(1)
                }
            }
        }
    }
}

#[cfg(feature = "jit")]
fn jit_file(path: &str) -> ExitCode {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", path, e);
            return ExitCode::from(1);
        }
    };

    // Create JIT compiler
    let mut jit = match JitCompiler::new() {
        Ok(jit) => jit,
        Err(e) => {
            eprintln!("Failed to initialize JIT compiler: {}", e);
            return ExitCode::from(1);
        }
    };

    // Compile
    if let Err(e) = jit.compile(&source) {
        eprintln!("Compilation error in '{}': {}", path, e);
        return ExitCode::from(1);
    }

    // Run
    match jit.run() {
        Ok(result) => {
            if result != 0 {
                println!("{}", result);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Runtime error in '{}': {}", path, e);
            ExitCode::from(1)
        }
    }
}

#[cfg(feature = "llvm")]
fn llvm_file(path: &str) -> ExitCode {
    use inkwell::context::Context;

    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", path, e);
            return ExitCode::from(1);
        }
    };

    // Create LLVM context and compiler
    let context = Context::create();
    let mut compiler = match LlvmCompiler::new(&context, OptLevel::Aggressive) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to initialize LLVM compiler: {}", e);
            return ExitCode::from(1);
        }
    };

    // Set source path to enable tome loading
    let source_path = std::path::Path::new(path);
    if let Ok(abs_path) = source_path.canonicalize() {
        if let Err(e) = compiler.set_source_path(&abs_path) {
            eprintln!("Warning: failed to set source path: {}", e);
        }
    }

    // Compile
    if let Err(e) = compiler.compile(&source) {
        eprintln!("Compilation error in '{}': {}", path, e);
        return ExitCode::from(1);
    }

    // Debug: print IR
    if std::env::var("SIGIL_DEBUG_IR").is_ok() {
        eprintln!("Generated LLVM IR:\n{}", compiler.get_ir());
    }

    // Run
    match compiler.run() {
        Ok(result) => {
            if result != 0 {
                println!("{}", result);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Runtime error in '{}': {}", path, e);
            ExitCode::from(1)
        }
    }
}

#[cfg(feature = "llvm")]
fn compile_file(path: &str, output: &str, use_lto: bool, use_tls: bool, use_cuda: bool, use_native_runtime: bool, is_library: bool, opt_level: OptLevel) -> ExitCode {
    use inkwell::context::Context;
    use std::path::Path;
    use std::process::Command;

    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", path, e);
            return ExitCode::from(1);
        }
    };

    // Create LLVM context and compiler in AOT mode
    let context = Context::create();
    let mut compiler =
        match LlvmCompiler::with_mode(&context, opt_level, CompileMode::Aot) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Failed to initialize LLVM compiler: {}", e);
                return ExitCode::from(1);
            }
        };

    // Set source path to enable tome loading
    let source_path = std::path::Path::new(path);
    if let Ok(abs_path) = source_path.canonicalize() {
        if let Err(e) = compiler.set_source_path(&abs_path) {
            eprintln!("Warning: failed to set source path: {}", e);
        }
    }

    // Compile
    if let Err(e) = compiler.compile(&source) {
        eprintln!("Compilation error in '{}': {}", path, e);
        return ExitCode::from(1);
    }

    // Get link libraries from #[link("lib")] attributes on extern blocks
    let link_libs: Vec<String> = compiler.get_link_libraries()
        .iter()
        .map(|lib| format!("-l{}", lib))
        .collect();

    // Debug: print IR
    if std::env::var("SIGIL_DEBUG_IR").is_ok() {
        eprintln!("Generated LLVM IR:\n{}", compiler.get_ir());
    }

    // Write object file
    let obj_path = format!("{}.o", output);
    if let Err(e) = compiler.write_object_file(Path::new(&obj_path)) {
        eprintln!("Failed to write object file: {}", e);
        return ExitCode::from(1);
    }

    // Library mode: create shared library without runtime (no main function required)
    if is_library {
        return compile_as_library(&obj_path, output, &link_libs);
    }

    // Native runtime mode: link with pure assembly runtime, no libc
    if use_native_runtime {
        return link_native_runtime(&obj_path, output);
    }

    // Find the runtime (static library or C source)
    let runtime_result = find_runtime(use_lto, use_tls, use_cuda);
    if runtime_result.is_none() {
        eprintln!("Error: Could not find sigil runtime");
        eprintln!("Expected locations:");
        if use_cuda {
            eprintln!("  - ./runtime/libsigil_runtime_cuda.a (CUDA-enabled runtime)");
            eprintln!("Run 'make cuda' in the runtime directory to build.");
        } else if use_tls {
            eprintln!("  - ./runtime/libsigil_runtime_tls.a (TLS-enabled runtime)");
            eprintln!("Run 'make tls' in the runtime directory to build.");
        } else {
            eprintln!("  - ./runtime/libsigil_runtime.a (pre-built, faster)");
            eprintln!("  - ./runtime/sigil_runtime.c (source, required for --lto)");
            eprintln!("Run 'make' in the runtime directory to build.");
        }
        return ExitCode::from(1);
    }
    let (runtime, should_use_lto) = runtime_result.unwrap();
    let is_static_lib = runtime.ends_with(".a");

    // Link with clang/gcc
    let mode_str = if use_cuda {
        "CUDA enabled"
    } else if should_use_lto {
        "with LTO"
    } else if is_static_lib {
        "pre-built runtime"
    } else {
        "compiling from source"
    };
    println!("Compiling {} -> {} ({})", path, output, mode_str);
    let linker = find_linker();

    // Build linker arguments based on runtime type and LTO
    let mut args = vec![&obj_path as &str, &runtime, "-o", output, "-lm"];
    if should_use_lto {
        // Add LTO flags for cross-module optimization
        args.insert(0, "-flto");
        args.insert(0, "-O3");
    } else if !is_static_lib {
        // Add optimization flag when compiling C source without LTO
        args.insert(0, "-O3");
    }

    // Add TLS/OpenSSL libraries when using --tls
    if use_tls {
        args.push("-lssl");
        args.push("-lcrypto");
    }

    // Add CUDA libraries when using --cuda
    // NOTE: -L paths MUST come before -l flags for the linker to find libraries
    if use_cuda {
        // Search paths first
        args.push("-L/usr/lib/wsl/lib");  // WSL2 CUDA driver (libcuda.so)
        args.push("-L/usr/lib/x86_64-linux-gnu");  // System libs (libnvrtc.so)
        args.push("-L/usr/local/cuda/lib64");  // CUDA toolkit (if installed)
        // Then library names
        args.push("-lcuda");
        args.push("-lnvrtc");
        // Set rpath for runtime library loading
        args.push("-Wl,-rpath,/usr/lib/wsl/lib");
        args.push("-Wl,-rpath,/usr/lib/x86_64-linux-gnu");
        args.push("-Wl,-rpath,/usr/local/cuda/lib64");
    }

    // Add libraries from #[link("lib")] attributes on extern blocks
    for lib_flag in &link_libs {
        args.push(lib_flag);
    }

    let link_result = Command::new(&linker).args(&args).status();

    // Clean up object file
    let _ = std::fs::remove_file(&obj_path);

    match link_result {
        Ok(status) if status.success() => {
            println!("Successfully compiled to: {}", output);
            ExitCode::SUCCESS
        }
        Ok(status) => {
            eprintln!("Linker failed with status: {}", status);
            ExitCode::from(1)
        }
        Err(e) => {
            eprintln!("Failed to run linker '{}': {}", linker, e);
            ExitCode::from(1)
        }
    }
}

/// Link with the native assembly runtime (no libc dependency)
#[cfg(feature = "llvm")]
fn link_native_runtime(obj_path: &str, output: &str) -> ExitCode {
    use std::process::Command;

    // Find the native runtime library
    let native_lib = find_native_runtime();
    if native_lib.is_none() {
        eprintln!("Error: Could not find native runtime library");
        eprintln!("Expected: libsigil_native.a");
        eprintln!("Run './build_native.sh' in the runtime directory to build.");
        let _ = std::fs::remove_file(obj_path);
        return ExitCode::from(1);
    }
    let runtime = native_lib.unwrap();

    println!("Compiling with native runtime (no libc) -> {}", output);

    // Link with ld directly, no libc
    let link_result = Command::new("ld")
        .args([
            obj_path,
            &runtime,
            "-o", output,
            "-nostdlib",
            "-static",
        ])
        .status();

    // Clean up object file
    let _ = std::fs::remove_file(obj_path);

    match link_result {
        Ok(status) if status.success() => {
            println!("Successfully compiled to: {} (native runtime, no libc)", output);
            ExitCode::SUCCESS
        }
        Ok(status) => {
            eprintln!("Linker failed with status: {}", status);
            ExitCode::from(1)
        }
        Err(e) => {
            eprintln!("Failed to run linker 'ld': {}", e);
            ExitCode::from(1)
        }
    }
}

/// Compile as a shared library (no runtime, no main function required)
#[cfg(feature = "llvm")]
fn compile_as_library(obj_path: &str, output: &str, link_libs: &[String]) -> ExitCode {
    use std::process::Command;

    let linker = find_linker();

    // Determine output type based on extension
    let is_static = output.ends_with(".a");
    let output_path = if output.ends_with(".so") || output.ends_with(".a") {
        output.to_string()
    } else {
        format!("{}.so", output)
    };

    if is_static {
        // Create static library with ar
        println!("Creating static library {} -> {}", obj_path, output_path);

        let ar_result = Command::new("ar")
            .args(["rcs", &output_path, obj_path])
            .status();

        // Clean up object file
        let _ = std::fs::remove_file(obj_path);

        match ar_result {
            Ok(status) if status.success() => {
                println!("Successfully created static library: {}", output_path);
                ExitCode::SUCCESS
            }
            Ok(status) => {
                eprintln!("ar failed with status: {}", status);
                ExitCode::from(1)
            }
            Err(e) => {
                eprintln!("Failed to run ar: {}", e);
                ExitCode::from(1)
            }
        }
    } else {
        // Create shared library with -shared flag
        println!("Creating shared library {} -> {}", obj_path, output_path);

        let mut args = vec![
            "-shared",
            "-fPIC",
            obj_path,
            "-o", &output_path,
            "-lm",
        ];

        // Add libraries from #[link("lib")] attributes
        for lib_flag in link_libs {
            args.push(lib_flag);
        }

        let link_result = Command::new(&linker).args(&args).status();

        // Clean up object file
        let _ = std::fs::remove_file(obj_path);

        match link_result {
            Ok(status) if status.success() => {
                println!("Successfully created shared library: {}", output_path);
                ExitCode::SUCCESS
            }
            Ok(status) => {
                eprintln!("Linker failed with status: {}", status);
                ExitCode::from(1)
            }
            Err(e) => {
                eprintln!("Failed to run linker '{}': {}", linker, e);
                ExitCode::from(1)
            }
        }
    }
}

/// Find the native runtime library (libsigil_native.a)
#[cfg(feature = "llvm")]
fn find_native_runtime() -> Option<String> {
    let candidates = [
        "runtime/libsigil_native.a",
        "../runtime/libsigil_native.a",
        "parser/runtime/libsigil_native.a",
        "../parser/runtime/libsigil_native.a",
    ];

    for candidate in candidates {
        if std::path::Path::new(candidate).exists() {
            return Some(candidate.to_string());
        }
    }

    // Try relative to executable
    if let Ok(exe) = std::env::current_exe() {
        if let Some(exe_dir) = exe.parent() {
            let lib_path = exe_dir.join("runtime/libsigil_native.a");
            if lib_path.exists() {
                return Some(lib_path.to_string_lossy().to_string());
            }
            let lib_path = exe_dir.join("../runtime/libsigil_native.a");
            if lib_path.exists() {
                return Some(lib_path.to_string_lossy().to_string());
            }
        }
    }

    None
}

#[cfg(feature = "llvm")]
fn find_runtime(use_lto: bool, use_tls: bool, use_cuda: bool) -> Option<(String, bool)> {
    // For CUDA, use the CUDA-enabled runtime
    if use_cuda {
        let cuda_lib_candidates = [
            "runtime/libsigil_runtime_cuda.a",
            "../runtime/libsigil_runtime_cuda.a",
            "sigil/parser/runtime/libsigil_runtime_cuda.a",
        ];

        for candidate in cuda_lib_candidates {
            if std::path::Path::new(candidate).exists() {
                return Some((candidate.to_string(), false));
            }
        }

        // Try relative to executable
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                let lib_path = dir.join("runtime/libsigil_runtime_cuda.a");
                if lib_path.exists() {
                    return Some((lib_path.to_string_lossy().into_owned(), false));
                }
            }
        }

        return None;
    }

    // For TLS, we must use the pre-compiled TLS runtime (can't compile from source easily)
    if use_tls {
        let tls_lib_candidates = [
            "runtime/libsigil_runtime_tls.a",
            "../runtime/libsigil_runtime_tls.a",
            "sigil/parser/runtime/libsigil_runtime_tls.a",
        ];

        for candidate in tls_lib_candidates {
            if std::path::Path::new(candidate).exists() {
                return Some((candidate.to_string(), false));
            }
        }

        // Try relative to executable
        if let Ok(exe) = std::env::current_exe() {
            if let Some(dir) = exe.parent() {
                let lib_path = dir.join("runtime/libsigil_runtime_tls.a");
                if lib_path.exists() {
                    return Some((lib_path.to_string_lossy().into_owned(), false));
                }
            }
        }

        return None;
    }

    // For LTO, prefer C source so it can be compiled with -flto
    // This enables cross-module optimization between Sigil code and runtime
    if use_lto {
        let source_candidates = [
            "runtime/sigil_runtime.c",
            "../runtime/sigil_runtime.c",
            "sigil/parser/runtime/sigil_runtime.c",
        ];

        for candidate in source_candidates {
            if std::path::Path::new(candidate).exists() {
                return Some((candidate.to_string(), true));
            }
        }
    }

    // Prefer pre-compiled static library for faster linking
    let static_lib_candidates = [
        "runtime/libsigil_runtime.a",
        "../runtime/libsigil_runtime.a",
        "sigil/parser/runtime/libsigil_runtime.a",
    ];

    for candidate in static_lib_candidates {
        if std::path::Path::new(candidate).exists() {
            return Some((candidate.to_string(), false));
        }
    }

    // Fall back to C source (slower but works without pre-build)
    let source_candidates = [
        "runtime/sigil_runtime.c",
        "../runtime/sigil_runtime.c",
        "sigil/parser/runtime/sigil_runtime.c",
    ];

    for candidate in source_candidates {
        if std::path::Path::new(candidate).exists() {
            return Some((candidate.to_string(), false));
        }
    }

    // Try relative to executable — check multiple ancestor levels.
    // The binary may live in target/release/ (2 levels below the runtime dir).
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            // Candidates relative to the executable directory
            let search_dirs = [
                dir.to_path_buf(),                     // <exe_dir>/
                dir.join(".."),                        // <exe_dir>/../
                dir.join("../.."),                     // <exe_dir>/../../  (covers target/release -> parser)
                dir.join("../../.."),                  // deeper fallback
            ];
            for base in &search_dirs {
                let lib_path = base.join("runtime/libsigil_runtime.a");
                if lib_path.exists() {
                    return Some((lib_path.to_string_lossy().into_owned(), false));
                }
                let src_path = base.join("runtime/sigil_runtime.c");
                if src_path.exists() {
                    return Some((src_path.to_string_lossy().into_owned(), use_lto));
                }
            }
        }
    }

    None
}

#[cfg(feature = "llvm")]
fn find_linker() -> String {
    // Prefer clang, fall back to gcc
    for linker in ["clang", "gcc", "cc"] {
        if std::process::Command::new(linker)
            .arg("--version")
            .output()
            .is_ok()
        {
            return linker.to_string();
        }
    }
    "cc".to_string()
}

/// Transpile a Sigil source file to Rust source code.
fn rust_compile_file(
    path: &str,
    output: Option<&str>,
    preserve_evidence: bool,
    no_std: bool,
    emit_comments: bool,
) -> ExitCode {
    use sigil_parser::{RustCompiler, RustCodegenOptions, RustEdition};

    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", path, e);
            return ExitCode::from(1);
        }
    };

    // Configure options
    let options = RustCodegenOptions {
        preserve_evidence,
        emit_comments,
        edition: RustEdition::Edition2021,
        no_std,
        indent_spaces: 4,
    };

    // Parse
    let mut parser = Parser::new(&source);
    let source_file = match parser.parse_file() {
        Ok(sf) => sf,
        Err(e) => {
            eprintln!("Parse error in '{}': {}", path, e);
            return ExitCode::from(1);
        }
    };

    // Extract items from SourceFile
    let items: Vec<_> = source_file.items.iter().map(|s| s.node.clone()).collect();

    // Generate Rust code
    let mut compiler = RustCompiler::with_options(options);
    let rust_code = match compiler.compile(&items) {
        Ok(code) => code,
        Err(e) => {
            eprintln!("Rust codegen error in '{}': {}", path, e);
            return ExitCode::from(1);
        }
    };

    // Output
    if let Some(output_path) = output {
        if let Err(e) = fs::write(output_path, &rust_code) {
            eprintln!("Error writing output file '{}': {}", output_path, e);
            return ExitCode::from(1);
        }
        println!("Generated: {} ({} bytes)", output_path, rust_code.len());
    } else {
        // Print to stdout
        print!("{}", rust_code);
    }

    ExitCode::SUCCESS
}

/// Compile a Sigil workspace to Rust with Cargo.toml generation.
fn rust_compile_workspace(
    path: &str,
    output: Option<&str>,
    preserve_evidence: bool,
    no_std: bool,
    emit_comments: bool,
    emit_cargo: bool,
) -> ExitCode {
    use sigil_parser::{RustCompiler, RustCodegenOptions, RustEdition};
    use std::path::Path;

    let workspace_path = Path::new(path);
    let output_dir = output.unwrap_or("rust-out");

    // Look for Sigil.toml in the workspace root
    let sigil_toml_path = workspace_path.join("Sigil.toml");
    let workspace_config = if sigil_toml_path.exists() {
        match fs::read_to_string(&sigil_toml_path) {
            Ok(content) => Some(content),
            Err(e) => {
                eprintln!("Warning: Could not read {}: {}", sigil_toml_path.display(), e);
                None
            }
        }
    } else {
        None
    };

    // Parse workspace members from Sigil.toml
    let crate_dirs: Vec<String> = if let Some(ref config) = workspace_config {
        parse_workspace_members(config)
    } else {
        // Fallback: find all directories with .sigil or .sg files
        find_sigil_crates(workspace_path)
    };

    if crate_dirs.is_empty() {
        eprintln!("No Sigil crates found in workspace");
        return ExitCode::from(1);
    }

    // Create output directory
    if let Err(e) = fs::create_dir_all(output_dir) {
        eprintln!("Error creating output directory '{}': {}", output_dir, e);
        return ExitCode::from(1);
    }

    println!("Compiling {} crates to Rust...", crate_dirs.len());

    // Configure codegen options
    let options = RustCodegenOptions {
        preserve_evidence,
        emit_comments,
        edition: RustEdition::Edition2021,
        no_std,
        indent_spaces: 4,
    };

    let mut success_count = 0;
    let mut crate_names = Vec::new();

    for crate_dir in &crate_dirs {
        let crate_path = workspace_path.join(crate_dir);
        let crate_name = Path::new(crate_dir)
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| crate_dir.replace('/', "-"));

        // Create output crate directory
        let out_crate_dir = Path::new(output_dir).join(&crate_name);
        let out_src_dir = out_crate_dir.join("src");
        if let Err(e) = fs::create_dir_all(&out_src_dir) {
            eprintln!("  Error creating {}: {}", out_src_dir.display(), e);
            continue;
        }

        // Find all .sigil and .sg files in src/
        let src_dir = crate_path.join("src");
        if !src_dir.is_dir() {
            eprintln!("  Skipping {}: no src/ directory", crate_name);
            continue;
        }

        let mut has_lib = false;
        let mut file_count = 0;

        if let Ok(entries) = fs::read_dir(&src_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                let ext = path.extension().and_then(|e| e.to_str());

                if ext != Some("sigil") && ext != Some("sg") {
                    continue;
                }

                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown");
                let out_name = if stem == "lib" {
                    has_lib = true;
                    "lib.rs".to_string()
                } else {
                    format!("{}.rs", stem)
                };

                // Read and compile
                let source = match fs::read_to_string(&path) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("    Error reading {}: {}", path.display(), e);
                        continue;
                    }
                };

                let mut parser = Parser::new(&source);
                let source_file = match parser.parse_file() {
                    Ok(sf) => sf,
                    Err(e) => {
                        eprintln!("    Parse error in {}: {}", path.display(), e);
                        continue;
                    }
                };

                let items: Vec<_> = source_file.items.iter().map(|s| s.node.clone()).collect();
                let mut compiler = RustCompiler::with_options(options.clone());
                let rust_code = match compiler.compile(&items) {
                    Ok(code) => code,
                    Err(e) => {
                        eprintln!("    Codegen error in {}: {}", stem, e);
                        continue;
                    }
                };

                // Write output file
                let out_path = out_src_dir.join(&out_name);
                if let Err(e) = fs::write(&out_path, &rust_code) {
                    eprintln!("    Error writing {}: {}", out_path.display(), e);
                    continue;
                }

                file_count += 1;
            }
        }

        if !has_lib {
            eprintln!("  Skipping {}: no lib.sigil or lib.sg", crate_name);
            continue;
        }

        // Generate Cargo.toml if requested
        if emit_cargo {
            let cargo_toml = generate_cargo_toml(&crate_name, crate_dir, workspace_config.as_deref());
            let cargo_path = out_crate_dir.join("Cargo.toml");
            if let Err(e) = fs::write(&cargo_path, &cargo_toml) {
                eprintln!("  Error writing {}: {}", cargo_path.display(), e);
            }
        }

        println!("  {} ({} files) -> {}", crate_name, file_count, out_src_dir.display());
        crate_names.push(crate_name);
        success_count += 1;
    }

    // Generate workspace Cargo.toml
    if emit_cargo && !crate_names.is_empty() {
        let workspace_cargo = generate_workspace_cargo(&crate_names);
        let workspace_cargo_path = Path::new(output_dir).join("Cargo.toml");
        if let Err(e) = fs::write(&workspace_cargo_path, &workspace_cargo) {
            eprintln!("Error writing workspace Cargo.toml: {}", e);
        } else {
            println!("Generated workspace: {}", workspace_cargo_path.display());
        }
    }

    println!("\nCompiled {}/{} crates successfully", success_count, crate_dirs.len());

    if success_count > 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(1)
    }
}

/// Parse workspace.members from Sigil.toml
fn parse_workspace_members(toml_content: &str) -> Vec<String> {
    let mut members = Vec::new();
    let mut in_members = false;

    for line in toml_content.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("members = [") {
            in_members = true;
            continue;
        }

        if in_members {
            if trimmed == "]" {
                break;
            }
            // Extract path from quoted string
            if let Some(start) = trimmed.find('"') {
                if let Some(end) = trimmed[start + 1..].find('"') {
                    let path = &trimmed[start + 1..start + 1 + end];
                    members.push(path.to_string());
                }
            }
        }
    }

    members
}

/// Find directories containing .sigil or .sg files
fn find_sigil_crates(workspace_path: &std::path::Path) -> Vec<String> {
    let mut crates = Vec::new();
    let crates_dir = workspace_path.join("crates");

    if crates_dir.is_dir() {
        if let Ok(entries) = fs::read_dir(&crates_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let src_dir = path.join("src");
                    if src_dir.join("lib.sigil").exists() || src_dir.join("lib.sg").exists() {
                        if let Some(name) = path.file_name() {
                            crates.push(format!("crates/{}", name.to_string_lossy()));
                        }
                    }
                }
            }
        }
    }

    crates
}

/// Generate a Cargo.toml for a single crate
fn generate_cargo_toml(crate_name: &str, _crate_path: &str, _workspace_config: Option<&str>) -> String {
    // Normalize crate name (replace hyphens with underscores for Rust)
    let lib_name = crate_name.replace('-', "_");

    // Infer dependencies based on crate name patterns
    let deps = infer_crate_dependencies(crate_name);

    format!(
        r#"[package]
name = "{crate_name}"
version = "0.1.0"
edition = "2021"

[lib]
name = "{lib_name}"
path = "src/lib.rs"

[dependencies]
{deps}

[features]
default = []
cuda = []
"#,
        crate_name = crate_name,
        lib_name = lib_name,
        deps = deps,
    )
}

/// Infer dependencies for a crate based on naming conventions
fn infer_crate_dependencies(crate_name: &str) -> String {
    // Nihil dependency graph (based on Sigil.toml structure)
    let deps: Vec<&str> = match crate_name {
        "nihil-core" => vec![],
        "nihil-memory" => vec!["nihil-core"],
        "nihil-ops" => vec!["nihil-core", "nihil-memory"],
        "nihil-cpu" => vec!["nihil-core", "nihil-ops"],
        "nihil-cuda" => vec!["nihil-core", "nihil-ops", "nihil-memory"],
        "nihil-autograd" => vec!["nihil-core", "nihil-ops"],
        "nihil-einsum" => vec!["nihil-core", "nihil-ops"],
        "nihil-linalg" => vec!["nihil-core", "nihil-ops", "nihil-einsum"],
        "nihil-nn" => vec!["nihil-core", "nihil-ops", "nihil-autograd"],
        "nihil-optim" => vec!["nihil-nn", "nihil-autograd"],
        "nihil-transformer" => vec!["nihil-nn", "nihil-ops"],
        "nihil-io" => vec!["nihil-core"],
        "nihil-models" => vec!["nihil-transformer", "nihil-io"],
        "nihil-quant" => vec!["nihil-core", "nihil-ops"],
        "nihil-distributed" => vec!["nihil-core", "nihil-ops", "nihil-nn"],
        "nihil-dispatch" => vec!["nihil-core", "nihil-cpu", "nihil-cuda"],
        "nihil-compile" => vec!["nihil-core", "nihil-ops"],
        "nihil-test" => vec!["nihil-core", "nihil-ops", "nihil-nn"],
        "nihil-bench" => vec!["nihil-core", "nihil-ops"],
        "nihil-embed" => vec!["nihil-transformer", "nihil-models"],
        "pynihil" => vec!["nihil-core", "nihil-ops", "nihil-nn"],
        "nihil" => vec![
            "nihil-core", "nihil-ops", "nihil-nn", "nihil-autograd",
            "nihil-transformer", "nihil-io", "nihil-models",
        ],
        _ => vec![],
    };

    if deps.is_empty() {
        String::new()
    } else {
        deps.iter()
            .map(|d| format!("{} = {{ path = \"../{}\" }}", d, d))
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Generate a workspace Cargo.toml
fn generate_workspace_cargo(crate_names: &[String]) -> String {
    let members: Vec<String> = crate_names.iter().map(|n| format!("    \"{}\",", n)).collect();

    format!(
        r#"[workspace]
resolver = "2"
members = [
{}
]

[workspace.package]
version = "0.1.0"
edition = "2021"

[profile.release]
opt-level = 3
lto = true
codegen-units = 1
"#,
        members.join("\n")
    )
}

/// Compile a Sigil source file or project to WebAssembly.
#[cfg(feature = "wasm")]
fn wasm_compile_file(path: &str, output: &str) -> ExitCode {
    use std::path::Path;

    let path = Path::new(path);

    // Check if this is a directory with sigil.toml or Sigil.toml (project compilation)
    let is_project = if path.is_dir() {
        path.join("sigil.toml").exists() || path.join("Sigil.toml").exists()
    } else if let Some(parent) = path.parent() {
        parent.join("sigil.toml").exists() || parent.join("Sigil.toml").exists()
    } else {
        false
    };

    if is_project {
        // Project compilation with dependencies
        let project_dir = if path.is_dir() {
            path.to_path_buf()
        } else {
            path.parent().unwrap().to_path_buf()
        };

        println!("Compiling project {} -> {} (WebAssembly with dependencies)",
                 project_dir.display(), output);

        match WasmCompiler::compile_project(&project_dir) {
            Ok(wasm_bytes) => {
                if let Err(e) = fs::write(output, &wasm_bytes) {
                    eprintln!("Error writing output file '{}': {}", output, e);
                    return ExitCode::from(1);
                }

                let size = wasm_bytes.len();
                let size_str = format_size(size);
                println!("Successfully compiled to: {} ({})", output, size_str);
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("Compilation error: {}", e);
                ExitCode::from(1)
            }
        }
    } else {
        // Single file compilation
        println!("Compiling {} -> {} (WebAssembly)", path.display(), output);

        let mut compiler = WasmCompiler::new();
        match compiler.compile_from_path(path) {
            Ok(wasm_bytes) => {
                if let Err(e) = fs::write(output, &wasm_bytes) {
                    eprintln!("Error writing output file '{}': {}", output, e);
                    return ExitCode::from(1);
                }

                let size = wasm_bytes.len();
                let size_str = format_size(size);
                println!("Successfully compiled to: {} ({})", output, size_str);
                ExitCode::SUCCESS
            }
            Err(e) => {
                eprintln!("Compilation error in '{}': {}", path.display(), e);
                ExitCode::from(1)
            }
        }
    }
}

/// Format file size for display.
#[cfg(feature = "wasm")]
fn format_size(size: usize) -> String {
    if size < 1024 {
        format!("{} bytes", size)
    } else if size < 1024 * 1024 {
        format!("{:.1} KB", size as f64 / 1024.0)
    } else {
        format!("{:.1} MB", size as f64 / (1024.0 * 1024.0))
    }
}

/// Check a file and output diagnostics.
///
/// This is the primary interface for AI agents - provides structured
/// JSON output with all information needed for self-correction.
///
/// With `--apply-suggestions`, automatically applies fix suggestions
/// and rewrites the file.
fn check_file(path: &str, format: OutputFormat, quiet: bool, apply_fixes: bool) -> ExitCode {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            if format == OutputFormat::Human && !quiet {
                eprintln!("Error reading file '{}': {}", path, e);
            } else if !quiet {
                // Output error as JSON for AI agents
                let error_json = serde_json::json!({
                    "file": path,
                    "diagnostics": [{
                        "severity": "error",
                        "message": format!("Failed to read file: {}", e),
                        "code": "E0001",
                        "line": 1,
                        "column": 1,
                    }],
                    "error_count": 1,
                    "warning_count": 0,
                    "success": false
                });
                if format == OutputFormat::Json {
                    println!("{}", serde_json::to_string_pretty(&error_json).unwrap());
                } else {
                    println!("{}", serde_json::to_string(&error_json).unwrap());
                }
            }
            return ExitCode::from(1);
        }
    };

    // Collect diagnostics during parsing
    let mut diagnostics = Diagnostics::new();
    let mut parser = Parser::new(&source);

    match parser.parse_file() {
        Ok(ast) => {
            // Run type checker with evidence enforcement
            let mut type_checker = TypeChecker::new();
            if let Err(type_errors) = type_checker.check_file(&ast) {
                for err in type_errors {
                    let mut diag =
                        Diagnostic::error(&err.message, err.span.unwrap_or(Span::default()))
                            .with_code("E0003");
                    for note in &err.notes {
                        diag = diag.with_note(note);
                    }
                    diagnostics.add(diag);
                }
            }
        }
        Err(e) => {
            // Convert parse error to diagnostic
            let diag = Diagnostic::error(format!("{}", e), Span::default()).with_code("E0002");
            diagnostics.add(diag);
        }
    }

    // Apply fixes if requested
    let source = if apply_fixes && !diagnostics.is_empty() {
        let suggestions: Vec<_> = diagnostics
            .iter()
            .flat_map(|d| d.suggestions.iter())
            .collect();

        if !suggestions.is_empty() {
            // Apply fixes in reverse order to preserve byte positions
            let mut fixed = source.clone();
            let mut applied_fixes = Vec::new();

            // Sort by span start descending so we apply from end to start
            let mut sorted_suggestions: Vec<_> = suggestions.iter().collect();
            sorted_suggestions.sort_by(|a, b| b.span.start.cmp(&a.span.start));

            for suggestion in sorted_suggestions {
                // Apply the fix
                if suggestion.span.end <= fixed.len() {
                    fixed.replace_range(
                        suggestion.span.start..suggestion.span.end,
                        &suggestion.replacement,
                    );
                    applied_fixes.push(suggestion.message.clone());
                }
            }

            // Write the fixed file
            if !applied_fixes.is_empty() {
                if let Err(e) = fs::write(path, &fixed) {
                    if format == OutputFormat::Human {
                        eprintln!("Error writing fixed file: {}", e);
                    }
                    return ExitCode::from(1);
                }

                if format == OutputFormat::Human && !quiet {
                    println!(
                        "Applied {} fix{}:",
                        applied_fixes.len(),
                        if applied_fixes.len() == 1 { "" } else { "es" }
                    );
                    for fix in &applied_fixes {
                        println!("  • {}", fix);
                    }
                } else if format != OutputFormat::Human && !quiet {
                    // Include applied fixes in JSON output
                    let fix_json = serde_json::json!({
                        "file": path,
                        "applied_fixes": applied_fixes,
                        "fix_count": applied_fixes.len(),
                        "recheck_needed": true
                    });
                    if format == OutputFormat::Json {
                        println!("{}", serde_json::to_string_pretty(&fix_json).unwrap());
                    } else {
                        println!("{}", serde_json::to_string(&fix_json).unwrap());
                    }
                }

                // Re-check with fixed source
                return check_file(path, format, quiet, false);
            }
            source // No fixes applied, use original
        } else {
            source // No suggestions available
        }
    } else {
        source
    };

    // Output in requested format
    if quiet {
        // Exit code only - no output
        return if diagnostics.has_errors() {
            ExitCode::from(1)
        } else {
            ExitCode::SUCCESS
        };
    }

    match format {
        OutputFormat::Human => {
            if diagnostics.is_empty() {
                println!("✓ {} - no errors", path);
            } else {
                diagnostics.eprint_all(path, &source);
                diagnostics.print_summary();
            }
        }
        OutputFormat::Json => {
            println!("{}", diagnostics.to_json_string(path, &source));
        }
        OutputFormat::Compact => {
            println!("{}", diagnostics.to_json_compact(path, &source));
        }
        OutputFormat::Sarif => {
            // SARIF format not applicable for check command - use lint instead
            eprintln!("Note: SARIF format is only available for 'sigil lint', using JSON");
            println!("{}", diagnostics.to_json_string(path, &source));
        }
    }

    if diagnostics.has_errors() {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

/// List all available lint rules.
fn lint_list_rules() -> ExitCode {
    use sigil_parser::lint::list_lints;
    println!("{}", list_lints());
    ExitCode::SUCCESS
}

/// Show detailed documentation for a lint rule.
fn lint_explain(rule: &str) -> ExitCode {
    use sigil_parser::lint::{explain_lint, LintId};

    if rule.is_empty() {
        eprintln!("Error: --explain requires a rule code or name");
        eprintln!("Usage: sigil lint --explain=W0101");
        eprintln!("       sigil lint --explain unused_variable");
        eprintln!();
        eprintln!("Use 'sigil lint --list' to see all available rules");
        return ExitCode::from(1);
    }

    match LintId::from_str(rule) {
        Some(lint) => {
            println!("{}", explain_lint(lint));
            ExitCode::SUCCESS
        }
        None => {
            eprintln!("Error: unknown lint rule '{}'", rule);
            eprintln!();
            eprintln!("Use 'sigil lint --list' to see all available rules");
            ExitCode::from(1)
        }
    }
}

/// Generate a default .sigillint.toml configuration file.
fn lint_init() -> ExitCode {
    use sigil_parser::lint::LintConfig;
    use std::path::Path;

    let config_path = Path::new(".sigillint.toml");
    if config_path.exists() {
        eprintln!("Error: .sigillint.toml already exists");
        return ExitCode::from(1);
    }

    let default_config = LintConfig::default_toml();
    match fs::write(config_path, &default_config) {
        Ok(_) => {
            println!("✓ Created .sigillint.toml");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Error writing config file: {}", e);
            ExitCode::from(1)
        }
    }
}

/// Run the linter on a file or directory.
///
/// The linter performs static analysis to catch:
/// - Reserved word usage (W0101)
/// - Nested generics compatibility (W0104)
/// - Unused variables and imports (W0202, W0203)
/// - Variable shadowing (W0204)
/// - Deep nesting complexity (W0205)
/// - Empty blocks (W0206)
/// - Bool comparison (W0207)
/// - Redundant else (W0208)
/// - Evidentiality violations (E0600)
/// - Unreachable code (E0700)
/// - Infinite loops (E0701)
/// - Division by zero (E0702)
fn lint_path(path: &str, format: OutputFormat, config_path: Option<&str>, apply_fix: bool, parallel: bool, show_stats: bool, evidentiality_mode: bool) -> ExitCode {
    use sigil_parser::lint::{lint_source_with_config, lint_directory, lint_directory_parallel, apply_fixes, generate_sarif, LintConfig, LintLevel};
    use std::path::Path;

    // Load config
    let mut config = if let Some(cfg_path) = config_path {
        match LintConfig::from_file(Path::new(cfg_path)) {
            Ok(cfg) => cfg,
            Err(e) => {
                eprintln!("Error loading config '{}': {}", cfg_path, e);
                return ExitCode::from(1);
            }
        }
    } else {
        LintConfig::find_and_load()
    };

    // Enable strict evidentiality checking if requested
    if evidentiality_mode {
        eprintln!("Evidentiality mode enabled: enforcing data provenance markers");
        // Set all evidentiality-related lints to Deny
        config.levels.insert("unvalidated_external_data".to_string(), LintLevel::Deny);
        config.levels.insert("evidentiality_violation".to_string(), LintLevel::Deny);
        config.levels.insert("certainty_downgrade".to_string(), LintLevel::Deny);
        config.levels.insert("evidentiality_mismatch".to_string(), LintLevel::Deny);
        config.levels.insert("uncertainty_unhandled".to_string(), LintLevel::Warn);
        config.levels.insert("reported_without_attribution".to_string(), LintLevel::Warn);
        config.levels.insert("missing_evidentiality_marker".to_string(), LintLevel::Warn);
    }

    let target = Path::new(path);

    // Directory linting (--fix not supported for directories yet)
    if target.is_dir() {
        if apply_fix {
            eprintln!("Warning: --fix is not yet supported for directory linting");
        }
        let result = if parallel {
            lint_directory_parallel(target, config)
        } else {
            lint_directory(target, config)
        };

        match format {
            OutputFormat::Human => {
                for (file_path, diagnostics) in &result.files {
                    if diagnostics.is_empty() {
                        println!("✓ {} - no issues", file_path);
                    } else {
                        let source = fs::read_to_string(file_path).unwrap_or_default();
                        diagnostics.eprint_all(file_path, &source);
                    }
                }
                println!();
                println!(
                    "Linted {} file(s): {} warning(s), {} error(s), {} parse error(s)",
                    result.files.len(),
                    result.total_warnings,
                    result.total_errors,
                    result.parse_errors
                );
            }
            OutputFormat::Json | OutputFormat::Compact => {
                let json_result = serde_json::json!({
                    "directory": path,
                    "files": result.files.iter().map(|(p, diags)| {
                        let has_parse_error = diags.iter()
                            .any(|d| d.code.as_ref().map_or(false, |c| c.starts_with("P0")));
                        serde_json::json!({
                            "file": p,
                            "success": !has_parse_error,
                            "warning_count": diags.iter()
                                .filter(|d| d.severity == sigil_parser::diagnostic::Severity::Warning)
                                .count(),
                            "error_count": diags.iter()
                                .filter(|d| d.severity == sigil_parser::diagnostic::Severity::Error)
                                .count(),
                        })
                    }).collect::<Vec<_>>(),
                    "total_warnings": result.total_warnings,
                    "total_errors": result.total_errors,
                    "parse_errors": result.parse_errors,
                });
                if format == OutputFormat::Json {
                    println!("{}", serde_json::to_string_pretty(&json_result).unwrap());
                } else {
                    println!("{}", serde_json::to_string(&json_result).unwrap());
                }
            }
            OutputFormat::Sarif => {
                use sigil_parser::lint::SarifReport;
                let mut sarif = SarifReport::new();
                for (file_path, diagnostics) in &result.files {
                    if let Ok(source) = fs::read_to_string(file_path) {
                        sarif.add_file(file_path, diagnostics, &source);
                    }
                }
                match sarif.to_json() {
                    Ok(json) => println!("{}", json),
                    Err(e) => eprintln!("Error generating SARIF: {}", e),
                }
            }
        }

        if result.total_errors > 0 || result.parse_errors > 0 {
            ExitCode::from(1)
        } else {
            ExitCode::SUCCESS
        }
    } else {
        // Single file linting
        let source = match fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                if format == OutputFormat::Human {
                    eprintln!("Error reading file '{}': {}", path, e);
                } else {
                    let error_json = serde_json::json!({
                        "file": path,
                        "diagnostics": [{
                            "severity": "error",
                            "message": format!("Failed to read file: {}", e),
                            "code": "E0001",
                            "line": 1,
                            "column": 1,
                        }],
                        "error_count": 1,
                        "warning_count": 0,
                        "success": false
                    });
                    if format == OutputFormat::Json {
                        println!("{}", serde_json::to_string_pretty(&error_json).unwrap());
                    } else {
                        println!("{}", serde_json::to_string(&error_json).unwrap());
                    }
                }
                return ExitCode::from(1);
            }
        };

        // Run the linter with config
        let diagnostics = lint_source_with_config(&source, path, config);
        let warning_count = diagnostics.iter()
            .filter(|d| d.severity == sigil_parser::diagnostic::Severity::Warning)
            .count();
        let error_count = diagnostics.iter()
            .filter(|d| d.severity == sigil_parser::diagnostic::Severity::Error)
            .count();

        // Apply fixes if requested
        if apply_fix {
            let fix_result = apply_fixes(&source, &diagnostics);
            if fix_result.fixes_applied > 0 {
                // Write fixed source back to file
                if let Err(e) = fs::write(path, &fix_result.source) {
                    eprintln!("Error writing fixes to '{}': {}", path, e);
                    return ExitCode::from(1);
                }
                if format == OutputFormat::Human {
                    println!("✓ {} - applied {} fix(es)", path, fix_result.fixes_applied);
                    if fix_result.fixes_skipped > 0 {
                        println!("  ({} fix(es) skipped due to conflicts)", fix_result.fixes_skipped);
                    }
                }
            } else if format == OutputFormat::Human {
                println!("✓ {} - no fixes to apply", path);
            }
            return ExitCode::SUCCESS;
        }

        match format {
            OutputFormat::Human => {
                if diagnostics.is_empty() {
                    println!("✓ {} - no issues found", path);
                } else {
                    diagnostics.eprint_all(path, &source);
                    println!();
                    println!("Found {} warning(s), {} error(s)", warning_count, error_count);
                }
                if show_stats {
                    println!();
                    println!("── Statistics ──");
                    println!("  Total: {} diagnostics", warning_count + error_count);
                }
            }
            OutputFormat::Json => {
                println!("{}", diagnostics.to_json_string(path, &source));
            }
            OutputFormat::Compact => {
                println!("{}", diagnostics.to_json_compact(path, &source));
            }
            OutputFormat::Sarif => {
                let sarif = generate_sarif(path, &diagnostics, &source);
                match sarif.to_json() {
                    Ok(json) => println!("{}", json),
                    Err(e) => eprintln!("Error generating SARIF: {}", e),
                }
            }
        }

        if error_count > 0 {
            ExitCode::from(1)
        } else {
            ExitCode::SUCCESS
        }
    }
}

/// Watch a directory for changes and continuously lint.
fn lint_watch(path: &str, format: OutputFormat, config_path: Option<&str>) -> ExitCode {
    use sigil_parser::lint::{watch_directory, LintConfig, WatchConfig};
    use std::path::Path;

    let target = Path::new(path);
    if !target.is_dir() {
        eprintln!("Error: --watch requires a directory path");
        return ExitCode::from(1);
    }

    // Load config
    let config = if let Some(cfg_path) = config_path {
        match LintConfig::from_file(Path::new(cfg_path)) {
            Ok(cfg) => cfg,
            Err(e) => {
                eprintln!("Error loading config '{}': {}", cfg_path, e);
                return ExitCode::from(1);
            }
        }
    } else {
        LintConfig::find_and_load()
    };

    let watch_config = WatchConfig::default();

    println!("Watching {} for changes (Ctrl+C to stop)...", path);
    println!();

    for watch_result in watch_directory(target, config, watch_config) {
        // Clear screen if human format
        if format == OutputFormat::Human {
            print!("\x1b[2J\x1b[H"); // ANSI clear screen
            println!("=== Lint Results ===");
            println!();
        }

        let result = &watch_result.lint_result;

        match format {
            OutputFormat::Human => {
                for (file_path, diagnostics) in &result.files {
                    if diagnostics.is_empty() {
                        println!("✓ {}", file_path);
                    } else {
                        let source = fs::read_to_string(file_path).unwrap_or_default();
                        diagnostics.eprint_all(file_path, &source);
                    }
                }
                println!();
                println!(
                    "Total: {} file(s), {} warning(s), {} error(s)",
                    result.files.len(),
                    result.total_warnings,
                    result.total_errors
                );
                if !watch_result.changed_files.is_empty() {
                    println!("Changed: {}", watch_result.changed_files.join(", "));
                }
            }
            OutputFormat::Json | OutputFormat::Compact => {
                let json_result = serde_json::json!({
                    "changed_files": watch_result.changed_files,
                    "total_files": result.files.len(),
                    "total_warnings": result.total_warnings,
                    "total_errors": result.total_errors,
                    "parse_errors": result.parse_errors,
                });
                if format == OutputFormat::Json {
                    println!("{}", serde_json::to_string_pretty(&json_result).unwrap());
                } else {
                    println!("{}", serde_json::to_string(&json_result).unwrap());
                }
            }
            OutputFormat::Sarif => {
                use sigil_parser::lint::SarifReport;
                let mut sarif = SarifReport::new();
                for (file_path, diagnostics) in &result.files {
                    if let Ok(source) = fs::read_to_string(file_path) {
                        sarif.add_file(file_path, diagnostics, &source);
                    }
                }
                if let Ok(json) = sarif.to_json() {
                    println!("{}", json);
                }
            }
        }
    }

    ExitCode::SUCCESS
}

/// Dump the AI-facing IR to JSON.
///
/// This is the primary interface for AI agents and tooling to inspect
/// Sigil programs in a structured, semantically-rich format.
///
/// The IR includes:
/// - Function definitions with typed parameters
/// - Pipeline operations (morphemes, transformations, forks)
/// - Evidentiality annotations throughout
/// - The evidentiality lattice structure
fn dump_ir_file(path: &str, pretty: bool, output: Option<&str>) -> ExitCode {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", path, e);
            return ExitCode::from(1);
        }
    };

    // Parse the source
    let mut parser = Parser::new(&source);
    let ast = match parser.parse_file() {
        Ok(ast) => ast,
        Err(e) => {
            eprintln!("Parse error in '{}': {}", path, e);
            return ExitCode::from(1);
        }
    };

    // Lower AST to IR
    let ir = lower_source_file(path, &ast);

    // Serialize to JSON
    let json = match ir.to_json(pretty) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("Error serializing IR: {}", e);
            return ExitCode::from(1);
        }
    };

    // Output
    match output {
        Some(output_path) => {
            if let Err(e) = fs::write(output_path, &json) {
                eprintln!("Error writing to '{}': {}", output_path, e);
                return ExitCode::from(1);
            }
            eprintln!("IR written to: {}", output_path);
        }
        None => {
            println!("{}", json);
        }
    }

    ExitCode::SUCCESS
}

/// Extract SGDOC documentation from a Sigil source file
fn doc_extract_file(path: &str, format: &str, output: Option<&str>) -> ExitCode {
    use sigil_parser::ast::{DocComment, Evidentiality, Item};
    use std::collections::HashMap;

    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", path, e);
            return ExitCode::from(1);
        }
    };

    // Parse the source
    let mut parser = Parser::new(&source);
    let ast = match parser.parse_file() {
        Ok(ast) => ast,
        Err(e) => {
            eprintln!("Parse error in '{}': {}", path, e);
            return ExitCode::from(1);
        }
    };

    // Helper to convert evidentiality to string
    fn ev_to_str(ev: &Evidentiality) -> &'static str {
        match ev {
            Evidentiality::Known => "verified",
            Evidentiality::Reported => "reported",
            Evidentiality::Uncertain => "uncertain",
            Evidentiality::Predicted => "predicted",
            Evidentiality::Chaos => "chaotic",
            Evidentiality::Paradox => "paradox",
        }
    }

    fn type_expr_to_string(ty: &sigil_parser::ast::TypeExpr) -> String {
        use sigil_parser::ast::TypeExpr;
        match ty {
            TypeExpr::Path(path) => path.segments.iter()
                .map(|seg| seg.ident.name.clone())
                .collect::<Vec<_>>()
                .join("::"),
            _ => "?".to_string(),
        }
    }

    fn type_path_to_string(path: &sigil_parser::ast::TypePath) -> String {
        path.segments.iter()
            .map(|seg| seg.ident.name.clone())
            .collect::<Vec<_>>()
            .join("::")
    }

    // Extract documented items
    #[derive(Debug)]
    struct DocSection {
        name: String,
        item_type: String,
        claims: Vec<(String, String, bool, usize)>, // (content, evidentiality, is_inner, line)
    }

    let mut sections = Vec::new();

    for spanned_item in &ast.items {
        let item = &spanned_item.node;
        let (doc_comments, name, item_type): (&Vec<DocComment>, String, &str) = match item {
            Item::Function(f) => (&f.doc_comments, f.name.name.clone(), "function"),
            Item::Struct(s) => (&s.doc_comments, s.name.name.clone(), "struct"),
            Item::Enum(e) => (&e.doc_comments, e.name.name.clone(), "enum"),
            Item::Trait(t) => (&t.doc_comments, t.name.name.clone(), "trait"),
            Item::Impl(i) => {
                let name = match &i.trait_ {
                    Some(t) => format!("{}::{}", type_path_to_string(t), type_expr_to_string(&i.self_ty)),
                    None => format!("impl {}", type_expr_to_string(&i.self_ty)),
                };
                (&i.doc_comments, name, "impl")
            }
            Item::Module(m) => (&m.doc_comments, m.name.name.clone(), "module"),
            Item::Const(c) => (&c.doc_comments, c.name.name.clone(), "const"),
            Item::Static(s) => (&s.doc_comments, s.name.name.clone(), "static"),
            _ => continue,
        };

        if doc_comments.is_empty() {
            continue;
        }

        let claims: Vec<(String, String, bool, usize)> = doc_comments.iter().map(|dc| {
            (dc.content.clone(), ev_to_str(&dc.evidentiality).to_string(), dc.is_inner, dc.span.start)
        }).collect();

        sections.push(DocSection {
            name,
            item_type: item_type.to_string(),
            claims,
        });
    }

    // Format output
    let file_name = std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("untitled");

    let result = match format {
        "markdown" | "md" => {
            let mut md = String::new();
            md.push_str(&format!("# Documentation: {}\n\n", file_name));
            md.push_str(&format!("Auto-extracted documentation from {}\n\n", path));

            for (i, section) in sections.iter().enumerate() {
                let section_id = format!("{}.{}", i + 1, section.item_type.chars().next().unwrap_or('x'));
                md.push_str(&format!("## {} {}\n\n", section_id, section.name));

                for (content, ev, _is_inner, _line) in &section.claims {
                    let badge = match ev.as_str() {
                        "verified" => "✓",
                        "reported" => "○",
                        "uncertain" => "?",
                        "predicted" => "◊",
                        "paradox" => "‽",
                        _ => "-",
                    };
                    md.push_str(&format!("- {} {}\n", badge, content));
                }
                md.push('\n');
            }
            md
        }
        "html" => {
            let mut html = String::new();
            html.push_str("<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n");
            html.push_str("<meta charset=\"UTF-8\">\n");
            html.push_str(&format!("<title>Documentation: {}</title>\n", file_name));
            html.push_str("<style>\n");
            html.push_str("body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; max-width: 800px; margin: 0 auto; padding: 2em; }\n");
            html.push_str(".claim { padding: 0.5em 1em; margin: 0.25em 0; border-left: 4px solid; }\n");
            html.push_str(".verified { border-color: #22c55e; background: #f0fdf4; }\n");
            html.push_str(".reported { border-color: #3b82f6; background: #eff6ff; }\n");
            html.push_str(".uncertain { border-color: #f59e0b; background: #fffbeb; }\n");
            html.push_str(".predicted { border-color: #8b5cf6; background: #f5f3ff; }\n");
            html.push_str(".paradox { border-color: #ef4444; background: #fef2f2; }\n");
            html.push_str(".badge { font-weight: bold; margin-right: 0.5em; }\n");
            html.push_str("</style>\n</head>\n<body>\n");
            html.push_str(&format!("<h1>Documentation: {}</h1>\n", file_name));
            html.push_str(&format!("<p>Auto-extracted documentation from {}</p>\n", path));

            for (i, section) in sections.iter().enumerate() {
                let section_id = format!("{}.{}", i + 1, section.item_type.chars().next().unwrap_or('x'));
                html.push_str(&format!("<h2 id=\"section-{}\">{} {}</h2>\n", section_id, section_id, section.name));
                html.push_str("<div class=\"claims\">\n");

                for (content, ev, _is_inner, _line) in &section.claims {
                    let (css_class, badge) = match ev.as_str() {
                        "verified" => ("verified", "✓"),
                        "reported" => ("reported", "○"),
                        "uncertain" => ("uncertain", "?"),
                        "predicted" => ("predicted", "◊"),
                        "paradox" => ("paradox", "‽"),
                        _ => ("", "-"),
                    };
                    html.push_str(&format!(
                        "<div class=\"claim {}\"><span class=\"badge\">{}</span>{}</div>\n",
                        css_class, badge, content
                    ));
                }
                html.push_str("</div>\n");
            }
            html.push_str("</body>\n</html>");
            html
        }
        _ => {
            // JSON format (default)
            let mut json = String::new();
            json.push_str("{\n");
            json.push_str(&format!("  \"title\": \"Documentation: {}\",\n", file_name));
            json.push_str(&format!("  \"source\": \"{}\",\n", path));
            json.push_str("  \"sections\": [\n");

            for (i, section) in sections.iter().enumerate() {
                if i > 0 { json.push_str(",\n"); }
                json.push_str("    {\n");
                json.push_str(&format!("      \"name\": \"{}\",\n", section.name));
                json.push_str(&format!("      \"type\": \"{}\",\n", section.item_type));
                json.push_str("      \"claims\": [\n");

                for (j, (content, ev, is_inner, line)) in section.claims.iter().enumerate() {
                    if j > 0 { json.push_str(",\n"); }
                    json.push_str("        {\n");
                    json.push_str(&format!("          \"content\": \"{}\",\n", content.replace('\\', "\\\\").replace('"', "\\\"")));
                    json.push_str(&format!("          \"evidentiality\": \"{}\",\n", ev));
                    json.push_str(&format!("          \"is_inner\": {},\n", is_inner));
                    json.push_str(&format!("          \"line\": {}\n", line));
                    json.push_str("        }");
                }
                json.push_str("\n      ]\n");
                json.push_str("    }");
            }
            json.push_str("\n  ]\n");
            json.push_str("}\n");
            json
        }
    };

    // Output
    match output {
        Some(output_path) => {
            if let Err(e) = fs::write(output_path, &result) {
                eprintln!("Error writing to '{}': {}", output_path, e);
                return ExitCode::from(1);
            }
            eprintln!("Documentation written to: {}", output_path);
        }
        None => {
            println!("{}", result);
        }
    }

    ExitCode::SUCCESS
}

fn parse_file(path: &str) -> ExitCode {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", path, e);
            return ExitCode::from(1);
        }
    };

    let mut parser = Parser::new(&source);
    match parser.parse_file() {
        Ok(ast) => {
            println!("Successfully parsed '{}'", path);
            println!("Found {} top-level items:", ast.items.len());
            for item in &ast.items {
                print_item_summary(&item.node);
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("Parse error in '{}': {}", path, e);
            ExitCode::from(1)
        }
    }
}

fn print_item_summary(item: &sigil_parser::Item) {
    use sigil_parser::Item;
    match item {
        Item::Function(f) => {
            let async_str = if f.is_async { "async " } else { "" };
            let params: Vec<_> = f.params.iter().map(|_| "_").collect();
            println!("  {}fn {}({})", async_str, f.name.name, params.join(", "));
        }
        Item::Struct(s) => {
            println!("  struct {}", s.name.name);
        }
        Item::Enum(e) => {
            println!("  enum {} ({} variants)", e.name.name, e.variants.len());
        }
        Item::Trait(t) => {
            println!("  trait {} ({} items)", t.name.name, t.items.len());
        }
        Item::Impl(i) => {
            if let Some(ref trait_) = i.trait_ {
                let trait_name = trait_
                    .segments
                    .iter()
                    .map(|s| s.ident.name.as_str())
                    .collect::<Vec<_>>()
                    .join("::");
                println!("  impl {} for ...", trait_name);
            } else {
                println!("  impl ...");
            }
        }
        Item::TypeAlias(t) => {
            println!("  type {}", t.name.name);
        }
        Item::Module(m) => {
            let inline = if m.items.is_some() { " { ... }" } else { "" };
            println!("  mod {}{}", m.name.name, inline);
        }
        Item::Use(u) => {
            println!("  use ...");
            let _ = u;
        }
        Item::Const(c) => {
            println!("  const {}", c.name.name);
        }
        Item::Static(s) => {
            let mut_str = if s.mutable { "mut " } else { "" };
            println!("  static {}{}", mut_str, s.name.name);
        }
        Item::Actor(a) => {
            println!("  actor {} ({} handlers)", a.name.name, a.handlers.len());
        }
        Item::ExternBlock(e) => {
            println!("  extern \"{}\" ({} items)", e.abi, e.items.len());
        }
        Item::Macro(m) => {
            println!("  macro {}", m.name.name);
        }
        Item::MacroInvocation(m) => {
            let path_str: String = m.path.segments.iter()
                .map(|s| s.ident.name.as_str())
                .collect::<Vec<_>>()
                .join("::");
            println!("  {}! {{ ... }}", path_str);
        }
        Item::Plurality(p) => {
            use sigil_parser::plurality::PluralityItem;
            match p {
                PluralityItem::Alter(a) => {
                    println!("  alter {} ({:?})", a.name.name, a.category);
                }
                PluralityItem::Headspace(h) => {
                    println!("  headspace {} ({} locations)", h.name.name, h.locations.len());
                }
                PluralityItem::Reality(r) => {
                    println!("  reality {} ({} layers)", r.name.name, r.layers.len());
                }
                PluralityItem::CoConChannel(c) => {
                    println!("  cocon {} ({} participants)", c.name.name, c.participants.len());
                }
                PluralityItem::TriggerHandler(_) => {
                    println!("  trigger handler");
                }
            }
        }
    }
}

fn lex_file(path: &str) -> ExitCode {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", path, e);
            return ExitCode::from(1);
        }
    };

    println!("Tokens in '{}':", path);
    let mut lexer = Lexer::new(&source);
    while let Some((token, span)) = lexer.next_token() {
        println!("  {:?} @ {}", token, span);
    }

    ExitCode::SUCCESS
}

// ============================================================================
// REPL with Syntax Highlighting
// ============================================================================

/// ANSI color codes for syntax highlighting
mod colors {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";

    // Status colors
    pub const GREEN: &str = "\x1b[38;5;114m"; // Green for success
    pub const ERROR: &str = "\x1b[38;5;203m"; // Red for errors

    // Semantic colors for Sigil
    pub const KEYWORD: &str = "\x1b[38;5;198m"; // Magenta/pink for keywords
    pub const MORPHEME: &str = "\x1b[38;5;51m"; // Cyan for morphemes (τ, φ, σ, ρ, Λ)
    pub const EVIDENCE: &str = "\x1b[38;5;214m"; // Orange for evidentiality (!, ?, ~, ‽)
    pub const STRING: &str = "\x1b[38;5;114m"; // Green for strings
    pub const NUMBER: &str = "\x1b[38;5;141m"; // Purple for numbers
    pub const COMMENT: &str = "\x1b[38;5;245m"; // Gray for comments
    pub const OPERATOR: &str = "\x1b[38;5;252m"; // Light gray for operators
    pub const FUNCTION: &str = "\x1b[38;5;81m"; // Blue for function names
    pub const TYPE: &str = "\x1b[38;5;222m"; // Yellow for types
    pub const SPECIAL: &str = "\x1b[38;5;203m"; // Red for special symbols
}

/// Sigil syntax highlighter using the lexer
struct SigilHighlighter;

impl SigilHighlighter {
    fn highlight_token(token: &Token) -> &'static str {
        match token {
            // Keywords
            Token::Fn
            | Token::Let
            | Token::Mut
            | Token::If
            | Token::Else
            | Token::While
            | Token::ForAll
            | Token::ElementOf
            | Token::Match
            | Token::Return
            | Token::Tensor
            | Token::CycleArrow
            | Token::Struct
            | Token::Enum
            | Token::Impl
            | Token::Trait
            | Token::Pub
            | Token::Use
            | Token::Mod
            | Token::Const
            | Token::Static
            | Token::Type
            | Token::Where
            | Token::Async
            | Token::Await
            | Token::Actor
            | Token::SelfLower
            | Token::SelfUpper
            | Token::True
            | Token::False
            | Token::LambdaExpr => colors::KEYWORD,

            // Morphemes (polysynthetic operators) - including new access morphemes
            Token::Tau
            | Token::Phi
            | Token::Sigma
            | Token::Rho
            | Token::Lambda
            | Token::Pi
            | Token::Alpha
            | Token::Omega
            | Token::Mu
            | Token::Chi
            | Token::Nu
            | Token::Xi => colors::MORPHEME,

            // Aspect morphemes
            Token::AspectProgressive
            | Token::AspectPerfective
            | Token::AspectPotential
            | Token::AspectResultative => colors::MORPHEME,

            // Bitwise operators (Unicode)
            Token::BitwiseAndSymbol | Token::BitwiseOrSymbol => colors::OPERATOR,

            // Data operation symbols
            Token::Bowtie
            | Token::ElementSmallVerticalBar
            | Token::SquareCup
            | Token::SquareCap => colors::SPECIAL,

            // Evidentiality markers
            Token::Bang | Token::Question | Token::Tilde | Token::Interrobang => colors::EVIDENCE,

            // Special symbols
            Token::Circle
            | Token::Empty
            | Token::Infinity => colors::SPECIAL,

            // Strings and chars
            Token::StringLit(_) | Token::CharLit(_) => colors::STRING,

            // Numbers
            Token::IntLit(_)
            | Token::FloatLit(_)
            | Token::BinaryLit(_)
            | Token::OctalLit(_)
            | Token::HexLit(_)
            | Token::DuodecimalLit(_)
            | Token::SexagesimalLit(_)
            | Token::VigesimalLit(_) => colors::NUMBER,

            // Comments
            Token::LineComment(_) | Token::DocComment(_) => colors::COMMENT,

            // Operators
            Token::Plus
            | Token::Minus
            | Token::Star
            | Token::Slash
            | Token::Percent
            | Token::StarStar
            | Token::PlusPlus
            | Token::Eq
            | Token::EqEq
            | Token::NotEq
            | Token::Lt
            | Token::LtEq
            | Token::Gt
            | Token::GtEq
            | Token::AndAnd
            | Token::OrOr
            | Token::Amp
            | Token::Caret
            | Token::Shl
            | Token::Shr
            | Token::Pipe
            | Token::Arrow
            | Token::FatArrow
            | Token::MiddleDot => colors::OPERATOR,

            // Identifiers - check for common stdlib functions
            Token::Ident(name) => {
                // Built-in functions get special coloring
                match name.as_str() {
                    "print" | "println" | "dbg" | "assert" | "panic" | "todo" | "sqrt" | "pow"
                    | "sin" | "cos" | "tan" | "abs" | "floor" | "ceil" | "len" | "push" | "pop"
                    | "sum" | "product" | "sort" | "reverse" | "known" | "uncertain"
                    | "reported" | "paradox" | "evidence_of" | "upper" | "lower" | "trim"
                    | "split" | "join" | "random" | "shuffle" | "now" | "sleep" => colors::FUNCTION,
                    // Type names (capitalized)
                    _ if name
                        .chars()
                        .next()
                        .map(|c| c.is_uppercase())
                        .unwrap_or(false) =>
                    {
                        colors::TYPE
                    }
                    _ => colors::RESET,
                }
            }

            // Default - no special color
            _ => colors::RESET,
        }
    }
}

/// REPL helper that provides highlighting and completion
#[derive(Helper)]
struct SigilHelper {
    /// Keywords and stdlib functions for completion
    completions: Vec<String>,
}

impl SigilHelper {
    fn new() -> Self {
        let completions = vec![
            // Keywords
            "fn",
            "let",
            "mut",
            "if",
            "else",
            "while",
            "for",
            "in",
            "match",
            "return",
            "break",
            "continue",
            "struct",
            "enum",
            "impl",
            "trait",
            "pub",
            "use",
            "mod",
            "const",
            "static",
            "type",
            "where",
            "async",
            "await",
            "actor",
            "handler",
            "receive",
            "send",
            "true",
            "false",
            "null",
            // Transform morphemes (Greek letters)
            "τ",
            "Τ",
            "φ",
            "Φ",
            "σ",
            "Σ",
            "ρ",
            "Ρ",
            "λ",
            "Λ",
            "Π",
            // Access morphemes
            "α",
            "ω",
            "Ω",
            "μ",
            "Μ",
            "χ",
            "Χ",
            "ν",
            "Ν",
            "ξ",
            "Ξ",
            // Other Greek letters
            "δ",
            "Δ",
            "ε",
            "ζ",
            // Aspect suffixes
            "·ing",
            "·ed",
            "·able",
            "·ive",
            // Logic operators (Unicode)
            "∧",
            "∨",
            "¬",
            "⊻",
            "⊤",
            "⊥",
            // Bitwise operators (Unicode)
            "⋏",
            "⋎",
            // Set operators
            "∪",
            "∩",
            "∖",
            "⊂",
            "⊆",
            "⊃",
            "⊇",
            "∈",
            "∉",
            // Math operators
            "∘",
            "⊗",
            "⊕",
            "∫",
            "∂",
            "√",
            "∛",
            // Data operations
            "⋈",
            "⋳",
            "⊔",
            "⊓",
            // Special literals
            "∅",
            "∞",
            "◯",
            // Quantifiers
            "∀",
            "∃",
            // Evidentiality markers
            "!",
            "?",
            "~",
            "‽",
            // Stdlib functions
            "print",
            "println",
            "dbg",
            "assert",
            "panic",
            "todo",
            "unreachable",
            "clone",
            "id",
            "default",
            "abs",
            "sqrt",
            "pow",
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "sinh",
            "cosh",
            "tanh",
            "exp",
            "ln",
            "log",
            "floor",
            "ceil",
            "round",
            "min",
            "max",
            "clamp",
            "sign",
            "gcd",
            "lcm",
            "is_prime",
            "fibonacci",
            "len",
            "push",
            "pop",
            "first",
            "last",
            "get",
            "set",
            "contains",
            "index_of",
            "reverse",
            "sort",
            "unique",
            "flatten",
            "zip",
            "enumerate",
            "chunks",
            "windows",
            "take",
            "skip",
            "concat",
            "chars",
            "bytes",
            "split",
            "join",
            "trim",
            "upper",
            "lower",
            "replace",
            "starts_with",
            "ends_with",
            "substring",
            "repeat",
            "known",
            "uncertain",
            "reported",
            "paradox",
            "evidence_of",
            "is_known",
            "strip_evidence",
            "trust",
            "verify",
            "sum",
            "product",
            "mean",
            "median",
            "min_of",
            "max_of",
            "any",
            "all",
            "none",
            "read_file",
            "write_file",
            "file_exists",
            "env",
            "cwd",
            "args",
            "now",
            "now_secs",
            "sleep",
            "random",
            "random_int",
            "shuffle",
            "sample",
            "to_string",
            "to_int",
            "to_float",
            "hex",
            "oct",
            "bin",
            "parse_int",
            "cycle",
            "mod_add",
            "mod_sub",
            "mod_mul",
            "mod_pow",
            "mod_inv",
            "octave",
            "interval",
            "cents",
            "freq",
            "midi",
            // New stdlib functions
            "middle",
            "choice",
            "nth",
            "next",
            "peek",
            "zip_with",
            "supremum",
            "infimum",
            // REPL commands
            ":help",
            ":ast",
            ":exit",
            ":quit",
            ":clear",
            ":type",
            ":symbols",
        ]
        .into_iter()
        .map(String::from)
        .collect();

        Self { completions }
    }
}

impl Completer for SigilHelper {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> rustyline::Result<(usize, Vec<Pair>)> {
        // Find the start of the current word
        let start = line[..pos]
            .rfind(|c: char| !c.is_alphanumeric() && c != '_' && c != ':')
            .map(|i| i + 1)
            .unwrap_or(0);

        let prefix = &line[start..pos];

        let matches: Vec<Pair> = self
            .completions
            .iter()
            .filter(|s| s.starts_with(prefix))
            .map(|s| Pair {
                display: s.clone(),
                replacement: s.clone(),
            })
            .collect();

        Ok((start, matches))
    }
}

impl Hinter for SigilHelper {
    type Hint = String;

    fn hint(&self, line: &str, pos: usize, _ctx: &rustyline::Context<'_>) -> Option<String> {
        // Find the current word
        let start = line[..pos]
            .rfind(|c: char| !c.is_alphanumeric() && c != '_')
            .map(|i| i + 1)
            .unwrap_or(0);

        let prefix = &line[start..pos];

        if prefix.len() < 2 {
            return None;
        }

        // Find first matching completion
        self.completions
            .iter()
            .find(|s| s.starts_with(prefix) && s.len() > prefix.len())
            .map(|s| format!("{}{}", colors::DIM, &s[prefix.len()..]))
    }
}

impl Highlighter for SigilHelper {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> Cow<'l, str> {
        let mut result = String::with_capacity(line.len() * 2);
        let mut lexer = Lexer::new(line);
        let mut last_end = 0;

        while let Some((token, span)) = lexer.next_token() {
            // Add any text between tokens (whitespace)
            if span.start > last_end {
                result.push_str(&line[last_end..span.start]);
            }

            // Add highlighted token
            let color = SigilHighlighter::highlight_token(&token);
            let token_text = &line[span.start..span.end];
            result.push_str(color);
            result.push_str(token_text);
            result.push_str(colors::RESET);

            last_end = span.end;
        }

        // Add any remaining text
        if last_end < line.len() {
            result.push_str(&line[last_end..]);
        }

        Cow::Owned(result)
    }

    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(
        &'s self,
        _prompt: &'p str,
        _default: bool,
    ) -> Cow<'b, str> {
        Cow::Owned(format!(
            "{}{}λ>{} ",
            colors::BOLD,
            colors::MORPHEME,
            colors::RESET
        ))
    }

    fn highlight_char(&self, _line: &str, _pos: usize, _forced: bool) -> bool {
        true // Always re-highlight
    }
}

impl Validator for SigilHelper {}

// =============================================================================
// Project Management Commands
// =============================================================================

/// Default sigil.toml content for a new project
fn default_manifest(name: &str) -> String {
    format!(
        r#"[package]
name = "{name}"
version = "0.1.0"
authors = []
description = ""

# Evidentiality configuration
[evidentiality]
# Require explicit evidence markers on function boundaries
strict = false
# Default evidence level ∀ unmarked external data
external_default = "reported"

[dependencies]
# Add dependencies here
# example = {{ git = "https://github.com/user/example" }}

[dev-dependencies]
# Add test dependencies here

[[bin]]
name = "{name}"
path = "src/main.sigil"
"#
    )
}

/// Default main.sigil for a new project
fn default_main(name: &str) -> String {
    format!(
        r#"// {name} - A Sigil project
//
// Run with: sigil run src/main.sigil
// Or: sigil build && ./{name}

rite main() {{
    print("Hello from {name}!");

    // Sigil's evidentiality system tracks data provenance:
    // - known (!)    : computed locally, verified
    // - uncertain (?) : may be absent
    // - reported (~)  : external source, untrusted
    // - paradox (‽)   : trust boundary crossing

    // Example pipeline using morphemes:
    // τ (tau) = transform/map
    // φ (phi) = filter
    // σ (sigma) = sort
    ≔ data = [1, 2, 3, 4, 5];
    ≔ result = data
        |τ{{_ * 2}}
        |φ{{_ > 5}}
        |σ;

    print("Processed: ");
    print(result);

    ⤺ 0;
}}
"#
    )
}

/// Default test file for a new project
fn default_test() -> String {
    r#"// Tests ∀ the project
//
// Run with: sigil test

#[test]
rite test_example() {
    ≔ result = 2 + 2;
    assert_eq(result, 4);
}

#[test]
rite test_morpheme_pipeline() {
    ≔ data = [1, 2, 3];
    ≔ doubled = data|τ{_ * 2};
    assert_eq(doubled, [2, 4, 6]);
}
"#
    .to_string()
}

/// Create a new Sigil project in a new directory
fn new_project(name: &str) -> ExitCode {
    use std::path::Path;

    let project_dir = Path::new(name);

    // Check if directory already exists
    if project_dir.exists() {
        eprintln!("Error: directory '{}' already exists", name);
        return ExitCode::from(1);
    }

    // Create directory structure
    let src_dir = project_dir.join("src");
    let tests_dir = project_dir.join("tests");

    if let Err(e) = fs::create_dir_all(&src_dir) {
        eprintln!("Error creating src directory: {}", e);
        return ExitCode::from(1);
    }

    if let Err(e) = fs::create_dir_all(&tests_dir) {
        eprintln!("Error creating tests directory: {}", e);
        return ExitCode::from(1);
    }

    // Write sigil.toml
    let manifest_path = project_dir.join("sigil.toml");
    if let Err(e) = fs::write(&manifest_path, default_manifest(name)) {
        eprintln!("Error writing sigil.toml: {}", e);
        return ExitCode::from(1);
    }

    // Write src/main.sigil
    let main_path = src_dir.join("main.sigil");
    if let Err(e) = fs::write(&main_path, default_main(name)) {
        eprintln!("Error writing src/main.sigil: {}", e);
        return ExitCode::from(1);
    }

    // Write tests/test_main.sigil
    let test_path = tests_dir.join("test_main.sigil");
    if let Err(e) = fs::write(&test_path, default_test()) {
        eprintln!("Error writing tests/test_main.sigil: {}", e);
        return ExitCode::from(1);
    }

    // Write .gitignore
    let gitignore_path = project_dir.join(".gitignore");
    let gitignore_content = r#"# Build artifacts
/target/
*.o
*.a

# Editor files
.vscode/
.idea/
*.swp
*~

# OS files
.DS_Store
Thumbs.db
"#;
    if let Err(e) = fs::write(&gitignore_path, gitignore_content) {
        eprintln!("Error writing .gitignore: {}", e);
        return ExitCode::from(1);
    }

    println!("✓ Created Sigil project '{}'", name);
    println!();
    println!(
        "  {}sigil.toml{}       Project manifest",
        colors::DIM,
        colors::RESET
    );
    println!(
        "  {}src/main.sigil{} Entry point",
        colors::DIM,
        colors::RESET
    );
    println!(
        "  {}tests/{}          Test directory",
        colors::DIM,
        colors::RESET
    );
    println!();
    println!("Get started:");
    println!("  cd {}", name);
    println!("  sigil run src/main.sigil");
    println!();
    println!("Or build and run:");
    println!("  sigil build");
    println!("  ./{}", name);

    ExitCode::SUCCESS
}

/// Initialize a Sigil project in the current directory
fn init_project() -> ExitCode {
    use std::path::Path;

    let manifest_path = Path::new("sigil.toml");

    // Check if already initialized
    if manifest_path.exists() {
        eprintln!("Error: sigil.toml already exists in this directory");
        return ExitCode::from(1);
    }

    // Determine project name from current directory
    let name = std::env::current_dir()
        .ok()
        .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
        .unwrap_or_else(|| "my_project".to_string());

    // Create directories
    let src_dir = Path::new("src");
    let tests_dir = Path::new("tests");

    if !src_dir.exists() {
        if let Err(e) = fs::create_dir_all(src_dir) {
            eprintln!("Error creating src directory: {}", e);
            return ExitCode::from(1);
        }
    }

    if !tests_dir.exists() {
        if let Err(e) = fs::create_dir_all(tests_dir) {
            eprintln!("Error creating tests directory: {}", e);
            return ExitCode::from(1);
        }
    }

    // Write sigil.toml
    if let Err(e) = fs::write(manifest_path, default_manifest(&name)) {
        eprintln!("Error writing sigil.toml: {}", e);
        return ExitCode::from(1);
    }

    // Write src/main.sigil if it doesn't exist
    let main_path = src_dir.join("main.sigil");
    if !main_path.exists() {
        if let Err(e) = fs::write(&main_path, default_main(&name)) {
            eprintln!("Error writing src/main.sigil: {}", e);
            return ExitCode::from(1);
        }
    }

    println!("✓ Initialized Sigil project '{}'", name);
    println!();
    println!("Run your project:");
    println!("  sigil run src/main.sigil");

    ExitCode::SUCCESS
}

/// Recursively collect test function names from AST items.
/// Looks for both top-level #[test]/`//@ rune: test` functions and functions
/// inside `scroll tests {}` modules.
fn collect_test_fn_names(
    items: &[sigil_parser::span::Spanned<sigil_parser::ast::Item>],
    test_fn_names: &mut Vec<String>,
    module_prefix: Option<&str>,
) {
    use sigil_parser::ast::Item;

    for item in items {
        match &item.node {
            Item::Function(func) => {
                if func.attrs.test {
                    let name = match module_prefix {
                        // Use middledot (·) as separator - matches how interpreter registers module functions
                        Some(prefix) => format!("{}·{}", prefix, func.name.name),
                        None => func.name.name.clone(),
                    };
                    test_fn_names.push(name);
                }
            }
            Item::Module(m) => {
                // Check if this is a `tests` module (either named "tests" or has #[cfg(test)])
                let is_test_module = m.name.name == "tests" || m.name.name == "test";

                if let Some(ref inner_items) = m.items {
                    if is_test_module {
                        // Look for test functions inside the tests module
                        let new_prefix = match module_prefix {
                            Some(prefix) => format!("{}·{}", prefix, m.name.name),
                            None => m.name.name.clone(),
                        };
                        collect_test_fn_names(inner_items, test_fn_names, Some(&new_prefix));
                    } else {
                        // Recursively check other modules for nested test modules
                        let new_prefix = match module_prefix {
                            Some(prefix) => format!("{}·{}", prefix, m.name.name),
                            None => m.name.name.clone(),
                        };
                        collect_test_fn_names(inner_items, test_fn_names, Some(&new_prefix));
                    }
                }
            }
            _ => {}
        }
    }
}

/// Run tests in the current project
fn collect_test_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    if dir.exists() {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path
                    .extension()
                    .map(|e| e == "sigil" || e == "sg")
                    .unwrap_or(false)
                {
                    files.push(path);
                }
            }
        }
    }
    files.sort();
    files
}

fn collect_src_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();
    collect_src_files_inner(dir, &mut files);
    files.sort();
    files
}

fn collect_src_files_inner(dir: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                collect_src_files_inner(&path, files);
            } else if path
                .extension()
                .map(|e| e == "sigil" || e == "sg")
                .unwrap_or(false)
            {
                files.push(path);
            }
        }
    }
}

fn run_test_files(
    test_files: &[std::path::PathBuf],
    label: Option<&str>,
) -> (usize, usize, usize) {
    let mut total_tests = 0;
    let mut passed_tests = 0;
    let mut failed_tests = 0;

    for test_file in test_files {
        let source = match fs::read_to_string(test_file) {
            Ok(s) => s,
            Err(e) => {
                println!(
                    "  {}✗{} {} - read error: {}",
                    colors::ERROR,
                    colors::RESET,
                    test_file.display(),
                    e
                );
                failed_tests += 1;
                total_tests += 1;
                continue;
            }
        };

        let mut parser = Parser::new(&source);
        let ast = match parser.parse_file() {
            Ok(ast) => ast,
            Err(e) => {
                println!(
                    "  {}✗{} {} - parse error: {}",
                    colors::ERROR,
                    colors::RESET,
                    test_file.display(),
                    e
                );
                failed_tests += 1;
                total_tests += 1;
                continue;
            }
        };

        // Type check
        let mut type_checker = TypeChecker::new();
        if let Err(errors) = type_checker.check_file(&ast) {
            println!(
                "  {}✗{} {} - type error:",
                colors::ERROR,
                colors::RESET,
                test_file.display()
            );
            for err in &errors {
                println!("      {}", err.message);
            }
            failed_tests += 1;
            total_tests += 1;
            continue;
        }

        // Collect test function names (both top-level and from `scroll tests {}` modules)
        let mut test_fn_names: Vec<String> = Vec::new();
        collect_test_fn_names(&ast.items, &mut test_fn_names, None);

        if !test_fn_names.is_empty() {
            let mut interpreter = Interpreter::new();
            register_stdlib(&mut interpreter);

            // Register all definitions (structs, impls, functions) without calling main
            match interpreter.execute_definitions(&ast) {
                Ok(_) => {
                    // Now invoke each #[test] function individually
                    let mut file_passed = 0;
                    let mut file_failed = 0;
                    let mut fail_messages: Vec<(String, String)> = Vec::new();

                    for test_name in &test_fn_names {
                        match interpreter.call_function_by_name(test_name, vec![]) {
                            Ok(_) => {
                                file_passed += 1;
                            }
                            Err(e) => {
                                file_failed += 1;
                                fail_messages.push((test_name.clone(), e.to_string()));
                            }
                        }
                    }

                    let file_label = test_file.file_stem().unwrap_or_default().to_string_lossy();
                    if file_failed == 0 {
                        println!(
                            "  {}✓{} {} ({} tests)",
                            colors::GREEN,
                            colors::RESET,
                            file_label,
                            file_passed
                        );
                    } else {
                        println!(
                            "  {}✗{} {} ({} passed, {} failed)",
                            colors::ERROR,
                            colors::RESET,
                            file_label,
                            file_passed,
                            file_failed
                        );
                        for (name, msg) in &fail_messages {
                            println!("      {} - {}", name, msg);
                        }
                    }
                    passed_tests += file_passed;
                    failed_tests += file_failed;
                    total_tests += file_passed + file_failed;
                }
                Err(e) => {
                    // File-level registration error (e.g., impl block fails)
                    println!(
                        "  {}✗{} {} - registration error: {}",
                        colors::ERROR,
                        colors::RESET,
                        test_file.file_stem().unwrap_or_default().to_string_lossy(),
                        e
                    );
                    failed_tests += test_fn_names.len();
                    total_tests += test_fn_names.len();
                }
            }
        }
    }

    (total_tests, passed_tests, failed_tests)
}

fn run_tests() -> ExitCode {
    use std::path::Path;

    // Check for workspace: if Sigil.toml exists, run workspace-level tests
    let manifest_path = Path::new("Sigil.toml");
    let manifest_alt = Path::new("sigil.toml");

    if manifest_path.exists() || manifest_alt.exists() {
        return run_tests_workspace();
    }

    // Single-project mode: find test files in tests/ directory
    let test_files = collect_test_files(Path::new("tests"));

    if test_files.is_empty() {
        println!("No test files found in tests/");
        println!();
        println!("Create a test file:");
        println!("  tests/test_main.sigil");
        println!();
        println!("With test functions:");
        println!("  #[test]");
        println!("  fn test_something() {{");
        println!("      assert_eq(1 + 1, 2);");
        println!("  }}");
        return ExitCode::SUCCESS;
    }

    println!("Running tests...");
    println!();

    let (total_tests, passed_tests, failed_tests) = run_test_files(&test_files, None);

    println!();
    print_test_summary(total_tests, passed_tests, failed_tests)
}

fn run_tests_workspace() -> ExitCode {
    use std::path::Path;

    // Parse Sigil.toml
    let manifest_content = match fs::read_to_string("Sigil.toml")
        .or_else(|_| fs::read_to_string("sigil.toml"))
    {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading Sigil.toml: {}", e);
            return ExitCode::from(1);
        }
    };

    let manifest: toml::Value = match manifest_content.parse() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error parsing Sigil.toml: {}", e);
            return ExitCode::from(1);
        }
    };

    let project_name = manifest
        .get("project")
        .or_else(|| manifest.get("package"))
        .and_then(|p| p.get("name"))
        .and_then(|n| n.as_str())
        .unwrap_or("unnamed");

    let members: Vec<String> = manifest
        .get("workspace")
        .and_then(|w| w.get("members"))
        .and_then(|m| m.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default();

    // Package manifest (no workspace members): test this package directly.
    if members.is_empty() {
        return run_tests_package(project_name);
    }

    println!("Testing workspace: {}", project_name);
    println!();

    let mut grand_total = 0;
    let mut grand_passed = 0;
    let mut grand_failed = 0;
    let mut crates_tested = 0;
    let mut crates_failed = 0;

    for member in &members {
        let tests_dir = Path::new(member).join("tests");
        let test_files = collect_test_files(&tests_dir);

        if test_files.is_empty() {
            continue;
        }

        let crate_name = Path::new(member)
            .file_name()
            .unwrap_or_default()
            .to_string_lossy();

        println!("  {} {}", crate_name, "─".repeat(40 - crate_name.len().min(39)));

        let (total, passed, failed) = run_test_files(&test_files, Some(&crate_name));

        grand_total += total;
        grand_passed += passed;
        grand_failed += failed;
        crates_tested += 1;
        if failed > 0 {
            crates_failed += 1;
        }
        println!();
    }

    // Print workspace summary
    println!("═══════════════════════════════════════════════");
    println!(
        "Workspace: {} crates tested, {} total tests",
        crates_tested, grand_total
    );
    print_test_summary(grand_total, grand_passed, grand_failed)
}

fn run_tests_package(package_name: &str) -> ExitCode {
    use std::path::Path;

    // Collect all source files (for inline #[cfg(test)] blocks) and dedicated test files.
    let src_files = collect_src_files(Path::new("src"));
    let test_files = collect_test_files(Path::new("tests"));

    if src_files.is_empty() && test_files.is_empty() {
        println!("No source or test files found.");
        return ExitCode::SUCCESS;
    }

    println!("Testing package: {}", package_name);
    println!();

    // Build one shared interpreter context so cross-module references resolve.
    let mut interpreter = Interpreter::new();
    register_stdlib(&mut interpreter);

    let mut all_test_fn_names: Vec<String> = Vec::new();

    // Phase 1: parse all source files, collect inline test names, register definitions.
    let mut src_asts = Vec::new();
    for src_file in &src_files {
        let source = match fs::read_to_string(src_file) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error reading {}: {}", src_file.display(), e);
                continue;
            }
        };
        let mut parser = Parser::new(&source);
        match parser.parse_file() {
            Ok(ast) => {
                collect_test_fn_names(&ast.items, &mut all_test_fn_names, None);
                src_asts.push(ast);
            }
            Err(e) => {
                eprintln!("Parse error in {}: {}", src_file.display(), e);
            }
        }
    }
    for ast in &src_asts {
        if let Err(e) = interpreter.execute_definitions(ast) {
            eprintln!("Definition error: {}", e);
        }
    }

    // Phase 2: parse dedicated test files, collect test names, register definitions.
    let mut test_asts = Vec::new();
    for test_file in &test_files {
        let source = match fs::read_to_string(test_file) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("Error reading {}: {}", test_file.display(), e);
                continue;
            }
        };
        let mut parser = Parser::new(&source);
        match parser.parse_file() {
            Ok(ast) => {
                collect_test_fn_names(&ast.items, &mut all_test_fn_names, None);
                test_asts.push((test_file.clone(), ast));
            }
            Err(e) => {
                eprintln!(
                    "  {}✗{} {} - parse error: {}",
                    colors::ERROR,
                    colors::RESET,
                    test_file.display(),
                    e
                );
            }
        }
    }
    for (_, ast) in &test_asts {
        if let Err(e) = interpreter.execute_definitions(ast) {
            eprintln!("Test definition error: {}", e);
        }
    }

    if all_test_fn_names.is_empty() {
        println!("No test functions found (functions with #[test] attribute)");
        return ExitCode::SUCCESS;
    }

    // Phase 3: run all discovered tests in the shared context.
    let mut passed = 0;
    let mut failed = 0;
    let mut fail_messages: Vec<(String, String)> = Vec::new();

    for test_name in &all_test_fn_names {
        match interpreter.call_function_by_name(test_name, vec![]) {
            Ok(_) => passed += 1,
            Err(e) => {
                failed += 1;
                fail_messages.push((test_name.clone(), e.to_string()));
            }
        }
    }

    let total = passed + failed;
    if failed == 0 {
        println!(
            "  {}✓{} {} ({} tests)",
            colors::GREEN,
            colors::RESET,
            package_name,
            total
        );
    } else {
        println!(
            "  {}✗{} {} ({} passed, {} failed)",
            colors::ERROR,
            colors::RESET,
            package_name,
            passed,
            failed
        );
        for (name, msg) in &fail_messages {
            println!("      {} - {}", name, msg);
        }
    }
    println!();

    print_test_summary(total, passed, failed)
}

fn print_test_summary(total: usize, passed: usize, failed: usize) -> ExitCode {
    if total == 0 {
        println!("No test functions found (functions with #[test] attribute)");
        ExitCode::SUCCESS
    } else if failed == 0 {
        println!(
            "{}All {} tests passed!{}",
            colors::GREEN,
            total,
            colors::RESET
        );
        ExitCode::SUCCESS
    } else {
        println!(
            "{}{} passed, {} failed{}",
            colors::ERROR,
            passed,
            failed,
            colors::RESET
        );
        ExitCode::from(1)
    }
}

// ============================================================================
// Build System: Manifest Parsing and Dependency Resolution
// ============================================================================

/// Parsed manifest information
#[derive(Debug, Clone)]
struct Manifest {
    name: String,
    version: String,
    has_lib: bool,
    has_bin: bool,
    dependencies: Vec<Dependency>,
    workspace_members: Vec<String>,  // [workspace] members list
    native_backend: Option<NativeBackend>,
}

/// A dependency reference
#[derive(Debug, Clone)]
struct Dependency {
    name: String,
    path: std::path::PathBuf,
}

/// A native Rust backend to build and link (from [native] section in sigil.toml)
#[derive(Debug, Clone)]
struct NativeBackend {
    /// Backend name, e.g. "wgpu" or "gtk4"
    name: String,
    /// Path to the Cargo project root for this backend
    path: std::path::PathBuf,
}

/// Parse a sigil.toml manifest file
fn parse_manifest(manifest_path: &std::path::Path) -> Result<Manifest, String> {
    let content = fs::read_to_string(manifest_path)
        .map_err(|e| format!("Failed to read manifest: {}", e))?;

    // Parse name
    let name = content
        .lines()
        .find(|l| l.trim().starts_with("name"))
        .and_then(|l| l.split('=').nth(1))
        .map(|s| s.trim().trim_matches('"').to_string())
        .ok_or_else(|| "Missing 'name' in manifest".to_string())?;

    // Parse version (optional, default to 0.1.0)
    let version = content
        .lines()
        .find(|l| l.trim().starts_with("version"))
        .and_then(|l| l.split('=').nth(1))
        .map(|s| s.trim().trim_matches('"').to_string())
        .unwrap_or_else(|| "0.1.0".to_string());

    // Check for lib.sigil and main.sigil
    let manifest_dir = manifest_path.parent().unwrap_or(std::path::Path::new("."));
    let has_lib = manifest_dir.join("src/lib.sigil").exists();
    let has_bin = manifest_dir.join("src/main.sigil").exists();

    // Parse dependencies
    let dependencies = parse_dependencies(&content, manifest_dir);

    // Parse workspace members
    let workspace_members = parse_workspace_members(&content);

    // Parse [native] section
    let native_backend = parse_native_backend(&content, manifest_dir);

    Ok(Manifest {
        name,
        version,
        has_lib,
        has_bin,
        dependencies,
        workspace_members,
        native_backend,
    })
}

/// Parse [dependencies] section from manifest content
fn parse_dependencies(content: &str, manifest_dir: &std::path::Path) -> Vec<Dependency> {
    let mut deps = Vec::new();
    let mut in_deps_section = false;
    let debug = std::env::var("SIGIL_DEBUG_DEPS").is_ok();

    if debug {
        eprintln!("DEBUG: parse_dependencies called, manifest_dir={}", manifest_dir.display());
    }

    for line in content.lines() {
        let trimmed = line.trim();

        // Check for section headers
        if trimmed.starts_with('[') {
            in_deps_section = trimmed == "[dependencies]";
            if debug && in_deps_section {
                eprintln!("DEBUG: Found [dependencies] section");
            }
            continue;
        }

        // Skip if not in dependencies section
        if !in_deps_section {
            continue;
        }

        // Skip empty lines and comments
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        // Parse dependency line: name = { path = "...", optional = true }
        if let Some((name, rest)) = trimmed.split_once('=') {
            let name = name.trim().to_string();
            let rest = rest.trim();

            // Skip optional dependencies
            if rest.contains("optional") && rest.contains("true") {
                if debug {
                    eprintln!("DEBUG: Skipping optional dep '{}'", name);
                }
                continue;
            }

            // Extract path from { path = "..." }
            if let Some(path_start) = rest.find("path") {
                let after_path = &rest[path_start + 4..];
                if let Some(eq_pos) = after_path.find('=') {
                    let path_value = after_path[eq_pos + 1..].trim();
                    // Extract the quoted path
                    if let Some(start) = path_value.find('"') {
                        if let Some(end) = path_value[start + 1..].find('"') {
                            let path_str = &path_value[start + 1..start + 1 + end];
                            let resolved_path = manifest_dir.join(path_str);
                            if debug {
                                eprintln!("DEBUG: Found dep '{}' at '{}'", name, resolved_path.display());
                            }
                            deps.push(Dependency {
                                name,
                                path: resolved_path,
                            });
                        }
                    }
                }
            }
        }
    }

    if debug {
        eprintln!("DEBUG: Parsed {} dependencies", deps.len());
    }
    deps
}

/// Parse [native] section from sigil.toml.
///
/// Expected format:
///   [native]
///   backend = "wgpu"
///   path = "../../qliphoth/runtime/native/wgpu"
fn parse_native_backend(content: &str, manifest_dir: &std::path::Path) -> Option<NativeBackend> {
    let mut in_native = false;
    let mut name: Option<String> = None;
    let mut path: Option<std::path::PathBuf> = None;

    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('[') {
            in_native = trimmed == "[native]";
            continue;
        }
        if !in_native { continue; }
        if trimmed.is_empty() || trimmed.starts_with('#') { continue; }

        if let Some((key, val)) = trimmed.split_once('=') {
            let key = key.trim();
            let val = val.trim().trim_matches('"');
            match key {
                "backend" => name = Some(val.to_string()),
                "path"    => path = Some(manifest_dir.join(val)),
                _ => {}
            }
        }
    }

    match (name, path) {
        (Some(n), Some(p)) => Some(NativeBackend { name: n, path: p }),
        _ => None,
    }
}

/// Build a native Rust backend crate with cargo and return link args:
/// [path/to/libname.so, -Wl,-rpath,/path/to/dir]
fn build_native_backend(backend: &NativeBackend) -> Result<Vec<std::path::PathBuf>, String> {
    use std::process::Command;

    println!("Building native backend '{}' at {} ...", backend.name, backend.path.display());

    let status = Command::new("cargo")
        .args(["build"])
        .current_dir(&backend.path)
        .status()
        .map_err(|e| format!("Failed to invoke cargo for native backend: {}", e))?;

    if !status.success() {
        return Err(format!("cargo build failed for native backend '{}'", backend.name));
    }

    // Library name: libqliphoth_native_<name>.so  (cdylib output)
    let lib_filename = format!("libqliphoth_native_{}.so", backend.name.replace('-', "_"));
    let lib_dir = backend.path.join("target/debug");
    let lib_path = lib_dir.join(&lib_filename);

    if !lib_path.exists() {
        // Try the staticlib fallback (.a)
        let a_filename = format!("libqliphoth_native_{}.a", backend.name.replace('-', "_"));
        let a_path = lib_dir.join(&a_filename);
        if a_path.exists() {
            println!("  Linking native backend (static): {}", a_path.display());
            return Ok(vec![a_path]);
        }
        return Err(format!(
            "Native backend library not found.\n  Looked for: {}\n  In: {}",
            lib_filename, lib_dir.display()
        ));
    }

    // Dynamic library: pass the .so directly and add rpath so the binary finds it at runtime.
    let rpath_flag = format!("-Wl,-rpath,{}", lib_dir.display());
    println!("  Linking native backend (dynamic): {}", lib_path.display());
    Ok(vec![lib_path, std::path::PathBuf::from(rpath_flag)])
}

/// Build dependencies in the correct order (dependencies first)
fn build_dependencies(deps: &[Dependency], built: &mut std::collections::HashSet<String>) -> Result<Vec<std::path::PathBuf>, String> {
    use std::sync::LazyLock;
    // Track in-progress builds for cycle detection
    static IN_PROGRESS: LazyLock<std::sync::Mutex<std::collections::HashSet<String>>> =
        LazyLock::new(|| std::sync::Mutex::new(std::collections::HashSet::new()));

    let mut lib_paths = Vec::new();

    for dep in deps {
        // Check for circular dependency
        {
            let in_progress = IN_PROGRESS.lock().unwrap();
            if in_progress.contains(&dep.name) {
                eprintln!("Note: Skipping circular dependency '{}'", dep.name);
                continue;
            }
        }

        // Skip if already built
        if built.contains(&dep.name) {
            // Find the already-built library
            let lib_name = format!("lib{}.a", dep.name.replace('-', "_"));
            let lib_path = dep.path.join("target").join(&lib_name);
            if lib_path.exists() {
                lib_paths.push(lib_path);
            }
            continue;
        }

        // Check if dependency directory exists
        if !dep.path.exists() {
            // Skip external dependencies that don't exist locally
            // (e.g., system libraries, packages to be installed separately)
            eprintln!("Note: Skipping external dependency '{}' (path not found)", dep.name);
            built.insert(dep.name.clone());
            continue;
        }

        // Check for sigil.toml in dependency
        let dep_manifest = dep.path.join("sigil.toml");
        if !dep_manifest.exists() {
            // Skip non-Sigil dependencies (e.g., Rust/Cargo packages, FFI libs)
            eprintln!("Note: Skipping non-Sigil dependency '{}' (no sigil.toml)", dep.name);
            built.insert(dep.name.clone());
            continue;
        }

        // Parse dependency's manifest
        let dep_info = parse_manifest(&dep_manifest)?;

        // Recursively build this dependency's dependencies first
        if !dep_info.dependencies.is_empty() {
            let sub_libs = build_dependencies(&dep_info.dependencies, built)?;
            lib_paths.extend(sub_libs);
        }

        // Build this dependency (library only)
        if dep_info.has_lib {
            // Canonicalize paths to avoid relative path issues
            let dep_path = match dep.path.canonicalize() {
                Ok(p) => p,
                Err(_) => dep.path.clone(), // Fall back to original if canonicalize fails
            };

            // Check if library already exists (avoid rebuilding)
            let lib_name = format!("lib{}.a", dep.name.replace('-', "_"));
            let lib_path = dep_path.join("target").join(&lib_name);

            if lib_path.exists() {
                // Library already built
                lib_paths.push(lib_path);
            } else {
                println!("  Building dependency: {}", dep.name);

                // Mark as in-progress to detect cycles
                {
                    let mut in_progress = IN_PROGRESS.lock().unwrap();
                    in_progress.insert(dep.name.clone());
                }

                // Spawn a subprocess to build each dependency
                // This isolates LLVM contexts and prevents memory corruption
                let build_result = std::process::Command::new(std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("sigil")))
                    .arg("build")
                    .current_dir(&dep_path)
                    .status();

                // Remove from in-progress
                {
                    let mut in_progress = IN_PROGRESS.lock().unwrap();
                    in_progress.remove(&dep.name);
                }

                match build_result {
                    Ok(status) if status.success() => {
                        // Build succeeded
                        lib_paths.push(lib_path);
                    }
                    Ok(status) => {
                        return Err(format!("Failed to build dependency '{}': exit code {:?}", dep.name, status.code()));
                    }
                    Err(e) => {
                        return Err(format!("Failed to run build for dependency '{}': {}", dep.name, e));
                    }
                }
            }
        }

        built.insert(dep.name.clone());
    }

    Ok(lib_paths)
}

/// Build the current project
fn build_project() -> ExitCode {
    use std::path::Path;
    use std::collections::HashSet;

    // Check for sigil.toml
    let manifest_path = Path::new("sigil.toml");
    if !manifest_path.exists() {
        eprintln!("Error: no sigil.toml found in current directory");
        eprintln!("Run 'sigil init' to create a project, or 'sigil new <name>' for a new one.");
        return ExitCode::from(1);
    }

    // Parse manifest using the new parser
    let manifest = match parse_manifest(manifest_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("Error parsing sigil.toml: {}", e);
            return ExitCode::from(1);
        }
    };

    let debug = std::env::var("SIGIL_DEBUG_DEPS").is_ok();
    if debug {
        eprintln!("DEBUG: Manifest name={}, has_lib={}, has_bin={}, deps={}, workspace_members={}",
            manifest.name, manifest.has_lib, manifest.has_bin, manifest.dependencies.len(), manifest.workspace_members.len());
        for dep in &manifest.dependencies {
            eprintln!("DEBUG:   dep: {} at {}", dep.name, dep.path.display());
        }
    }

    // Check if this is a workspace manifest
    if !manifest.workspace_members.is_empty() {
        return build_workspace(&manifest);
    }

    if !manifest.has_lib && !manifest.has_bin {
        eprintln!("Error: no src/lib.sigil or src/main.sigil found");
        eprintln!("A tome must have at least one of these files.");
        return ExitCode::from(1);
    }

    // Create target directory
    let target_dir = Path::new("target");
    if !target_dir.exists() {
        if let Err(e) = fs::create_dir_all(target_dir) {
            eprintln!("Error creating target directory: {}", e);
            return ExitCode::from(1);
        }
    }

    // Build dependencies first
    let mut built: HashSet<String> = HashSet::new();
    let mut dep_libs = if !manifest.dependencies.is_empty() {
        println!("Building dependencies...");
        match build_dependencies(&manifest.dependencies, &mut built) {
            Ok(libs) => libs,
            Err(e) => {
                eprintln!("Error building dependencies: {}", e);
                return ExitCode::from(1);
            }
        }
    } else {
        Vec::new()
    };

    // Build native backend (Rust crate) if [native] section is present
    if let Some(ref backend) = manifest.native_backend {
        match build_native_backend(backend) {
            Ok(native_libs) => dep_libs.extend(native_libs),
            Err(e) => {
                eprintln!("Error building native backend: {}", e);
                return ExitCode::from(1);
            }
        }
    }

    // Build library if present
    let lib_file = Path::new("src/lib.sigil");
    if manifest.has_lib {
        println!("Building {} (library)...", manifest.name);
        let result = build_library(&manifest.name, lib_file, target_dir);
        if result != ExitCode::SUCCESS {
            return result;
        }
    }

    // Build binary if present
    let main_file = Path::new("src/main.sigil");
    if manifest.has_bin {
        println!("Building {} (binary)...", manifest.name);
        let result = build_binary_with_deps(&manifest.name, main_file, target_dir, &dep_libs);
        if result != ExitCode::SUCCESS {
            return result;
        }
    }

    ExitCode::SUCCESS
}

/// Build all tomes in a workspace
fn build_workspace(manifest: &Manifest) -> ExitCode {
    use std::path::Path;

    println!("Building workspace '{}' with {} members...", manifest.name, manifest.workspace_members.len());

    let mut success_count = 0;
    let mut fail_count = 0;

    for member_path in &manifest.workspace_members {
        let member_dir = Path::new(member_path);

        // Check if member directory exists
        if !member_dir.exists() {
            eprintln!("  Warning: member '{}' not found, skipping", member_path);
            continue;
        }

        // Check for sigil.toml in member
        let member_manifest = member_dir.join("sigil.toml");
        if !member_manifest.exists() {
            eprintln!("  Warning: no sigil.toml in '{}', skipping", member_path);
            continue;
        }

        // Get member name from manifest
        let member_name = match parse_manifest(&member_manifest) {
            Ok(m) => m.name,
            Err(_) => member_path.split('/').last().unwrap_or(member_path).to_string(),
        };

        // Check if library already built
        let lib_name = format!("lib{}.a", member_name.replace('-', "_"));
        let lib_path = member_dir.join("target").join(&lib_name);

        if lib_path.exists() {
            println!("  {} (already built)", member_name);
            success_count += 1;
            continue;
        }

        // Spawn subprocess to build member (isolates LLVM contexts)
        print!("  {} ... ", member_name);

        let build_result = std::process::Command::new(std::env::current_exe().unwrap_or_else(|_| std::path::PathBuf::from("sigil")))
            .arg("build")
            .current_dir(member_dir)
            .stdout(std::process::Stdio::null())  // Suppress verbose output
            .stderr(std::process::Stdio::piped())
            .output();

        match build_result {
            Ok(output) if output.status.success() => {
                println!("ok");
                success_count += 1;
            }
            Ok(output) => {
                println!("FAILED");
                if !output.stderr.is_empty() {
                    eprintln!("    {}", String::from_utf8_lossy(&output.stderr).lines().next().unwrap_or(""));
                }
                fail_count += 1;
            }
            Err(e) => {
                println!("FAILED ({})", e);
                fail_count += 1;
            }
        }
    }

    println!("\nWorkspace build complete: {}/{} tomes succeeded",
        success_count, success_count + fail_count);

    if fail_count > 0 {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

/// Build a library tome (produces .a static library)
fn build_library(name: &str, lib_file: &std::path::Path, target_dir: &std::path::Path) -> ExitCode {
    // Read and parse source
    let source = match fs::read_to_string(lib_file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading {}: {}", lib_file.display(), e);
            return ExitCode::from(1);
        }
    };

    let mut parser = Parser::new(&source);
    let ast = match parser.parse_file() {
        Ok(ast) => ast,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            return ExitCode::from(1);
        }
    };

    // Type check
    let mut type_checker = TypeChecker::new();
    if let Err(errors) = type_checker.check_file(&ast) {
        for err in errors {
            eprintln!("Error: {}", err.message);
            for note in &err.notes {
                eprintln!("  note: {}", note);
            }
        }
        return ExitCode::from(1);
    }

    // Compile to object file and archive
    #[cfg(feature = "llvm")]
    {
        return compile_library(&lib_file.to_string_lossy(), name, target_dir);
    }

    #[cfg(not(feature = "llvm"))]
    {
        println!("✓ Type check passed (library)");
        println!("Note: Native compilation requires LLVM support.");
        ExitCode::SUCCESS
    }
}

/// Build a binary tome (produces executable)
fn build_binary(name: &str, main_file: &std::path::Path, target_dir: &std::path::Path) -> ExitCode {
    // Read and parse source
    let source = match fs::read_to_string(main_file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading {}: {}", main_file.display(), e);
            return ExitCode::from(1);
        }
    };

    let mut parser = Parser::new(&source);
    let ast = match parser.parse_file() {
        Ok(ast) => ast,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            return ExitCode::from(1);
        }
    };

    // Type check
    let mut type_checker = TypeChecker::new();
    if let Err(errors) = type_checker.check_file(&ast) {
        for err in errors {
            eprintln!("Error: {}", err.message);
            for note in &err.notes {
                eprintln!("  note: {}", note);
            }
        }
        return ExitCode::from(1);
    }

    // Compile to native executable
    #[cfg(feature = "llvm")]
    {
        let output_path = target_dir.join(name);
        let output_str = output_path.to_string_lossy();
        println!("Compiling to native executable...");
        return compile_file(&main_file.to_string_lossy(), &output_str, false, false, false, false, false, OptLevel::Standard);
    }

    #[cfg(not(feature = "llvm"))]
    {
        println!("✓ Type check passed (binary)");
        println!("Note: Native compilation requires LLVM support.");
        ExitCode::SUCCESS
    }
}

/// Build a binary tome with dependency libraries (produces executable linked with deps)
fn build_binary_with_deps(name: &str, main_file: &std::path::Path, target_dir: &std::path::Path, dep_libs: &[std::path::PathBuf]) -> ExitCode {
    // Read and parse source
    let source = match fs::read_to_string(main_file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading {}: {}", main_file.display(), e);
            return ExitCode::from(1);
        }
    };

    let mut parser = Parser::new(&source);
    let ast = match parser.parse_file() {
        Ok(ast) => ast,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            return ExitCode::from(1);
        }
    };

    // Type check
    let mut type_checker = TypeChecker::new();
    if let Err(errors) = type_checker.check_file(&ast) {
        for err in errors {
            eprintln!("Error: {}", err.message);
            for note in &err.notes {
                eprintln!("  note: {}", note);
            }
        }
        return ExitCode::from(1);
    }

    // Compile to native executable with dependency linking
    #[cfg(feature = "llvm")]
    {
        let output_path = target_dir.join(name);
        let output_str = output_path.to_string_lossy();
        println!("Compiling to native executable...");
        if !dep_libs.is_empty() {
            println!("  Linking with {} dependencies", dep_libs.len());
        }
        return compile_file_with_deps(&main_file.to_string_lossy(), &output_str, dep_libs, OptLevel::Standard);
    }

    #[cfg(not(feature = "llvm"))]
    {
        println!("✓ Type check passed (binary)");
        println!("Note: Native compilation requires LLVM support.");
        ExitCode::SUCCESS
    }
}

/// Compile a file to an executable, linking with dependency libraries
#[cfg(feature = "llvm")]
fn compile_file_with_deps(path: &str, output: &str, dep_libs: &[std::path::PathBuf], opt_level: OptLevel) -> ExitCode {
    use inkwell::context::Context;
    use std::path::Path;
    use std::process::Command;

    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", path, e);
            return ExitCode::from(1);
        }
    };

    // Create LLVM context and compiler in AOT mode
    let context = Context::create();
    let mut compiler =
        match LlvmCompiler::with_mode(&context, opt_level, CompileMode::Aot) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("Failed to initialize LLVM compiler: {}", e);
                return ExitCode::from(1);
            }
        };

    // Set source path to enable tome loading
    let source_path = std::path::Path::new(path);
    if let Ok(abs_path) = source_path.canonicalize() {
        if let Err(e) = compiler.set_source_path(&abs_path) {
            eprintln!("Warning: failed to set source path: {}", e);
        }
    }

    // Compile
    if let Err(e) = compiler.compile(&source) {
        eprintln!("Compilation error in '{}': {}", path, e);
        return ExitCode::from(1);
    }

    // Get link libraries from #[link("lib")] attributes on extern blocks
    let link_libs: Vec<String> = compiler.get_link_libraries()
        .iter()
        .map(|lib| format!("-l{}", lib))
        .collect();

    // Write object file
    let obj_path = format!("{}.o", output);
    if let Err(e) = compiler.write_object_file(Path::new(&obj_path)) {
        eprintln!("Failed to write object file: {}", e);
        return ExitCode::from(1);
    }

    // Find the runtime
    let runtime_result = find_runtime(false, false, false);
    if runtime_result.is_none() {
        eprintln!("Error: Could not find sigil runtime");
        eprintln!("Expected: ./runtime/libsigil_runtime.a");
        let _ = std::fs::remove_file(&obj_path);
        return ExitCode::from(1);
    }
    let (runtime, _) = runtime_result.unwrap();

    // Build linker arguments
    let linker = find_linker();
    let mut args: Vec<String> = vec![
        obj_path.clone(),
        runtime,
        "-o".to_string(),
        output.to_string(),
        "-lm".to_string(),
    ];

    // Add dependency libraries
    for lib_path in dep_libs {
        args.push(lib_path.to_string_lossy().to_string());
    }

    // Add libraries from #[link("lib")] attributes on extern blocks
    for lib_flag in &link_libs {
        args.push(lib_flag.clone());
    }

    let args_refs: Vec<&str> = args.iter().map(|s| s.as_str()).collect();
    let link_result = Command::new(&linker).args(&args_refs).status();

    // Clean up object file
    let _ = std::fs::remove_file(&obj_path);

    match link_result {
        Ok(status) if status.success() => {
            println!("Successfully compiled to: {}", output);
            ExitCode::SUCCESS
        }
        Ok(status) => {
            eprintln!("Linker failed with status: {}", status);
            ExitCode::from(1)
        }
        Err(e) => {
            eprintln!("Failed to run linker '{}': {}", linker, e);
            ExitCode::from(1)
        }
    }
}

/// Compile a library to a static archive (.a file)
#[cfg(feature = "llvm")]
fn compile_library(path: &str, name: &str, target_dir: &std::path::Path) -> ExitCode {
    use inkwell::context::Context;
    use std::process::Command;

    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading file '{}': {}", path, e);
            return ExitCode::from(1);
        }
    };

    // Create LLVM context and compiler in AOT mode
    let context = Context::create();
    let mut compiler = match LlvmCompiler::with_mode(&context, OptLevel::Standard, CompileMode::Aot) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to initialize LLVM compiler: {}", e);
            return ExitCode::from(1);
        }
    };

    // Set source path to enable tome loading
    let source_path = std::path::Path::new(path);
    if let Ok(abs_path) = source_path.canonicalize() {
        if let Err(e) = compiler.set_source_path(&abs_path) {
            eprintln!("Warning: failed to set source path: {}", e);
        }
    }

    // Compile (generates LLVM IR without requiring main)
    if let Err(e) = compiler.compile(&source) {
        eprintln!("Compilation error in '{}': {}", path, e);
        return ExitCode::from(1);
    }

    // Write object file
    let obj_path = target_dir.join(format!("{}.o", name));
    if let Err(e) = compiler.write_object_file(&obj_path) {
        eprintln!("Failed to write object file: {}", e);
        return ExitCode::from(1);
    }

    // Create static archive (.a file) using ar
    let lib_name = format!("lib{}.a", name.replace('-', "_"));
    let lib_path = target_dir.join(&lib_name);

    println!("Creating static library: {}", lib_path.display());

    let ar_result = Command::new("ar")
        .args(["rcs", &lib_path.to_string_lossy(), &obj_path.to_string_lossy()])
        .status();

    // Clean up object file
    let _ = std::fs::remove_file(&obj_path);

    match ar_result {
        Ok(status) if status.success() => {
            println!("Successfully built library: {}", lib_path.display());
            ExitCode::SUCCESS
        }
        Ok(status) => {
            eprintln!("ar failed with status: {}", status);
            ExitCode::from(1)
        }
        Err(e) => {
            eprintln!("Failed to run ar: {}", e);
            eprintln!("Make sure 'ar' is installed (part of binutils).");
            ExitCode::from(1)
        }
    }
}

/// Recursively collect all .sg and .sigil files from a directory.
fn collect_migrate_files(dir: &std::path::Path) -> Vec<std::path::PathBuf> {
    collect_migrate_files_with_rs(dir, false)
}

/// Recursively collect all .sg, .sigil, and optionally .rs files from a directory.
fn collect_migrate_files_with_rs(dir: &std::path::Path, include_rs: bool) -> Vec<std::path::PathBuf> {
    let mut files = Vec::new();

    fn visit_dir(dir: &std::path::Path, files: &mut Vec<std::path::PathBuf>, include_rs: bool) {
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    // Skip target directories (Rust build artifacts)
                    if path.file_name().map_or(false, |n| n == "target") {
                        continue;
                    }
                    visit_dir(&path, files, include_rs);
                } else {
                    let ext_ok = path.extension().map_or(false, |ext| {
                        ext == "sigil" || ext == "sg" || (include_rs && ext == "rs")
                    });
                    if ext_ok {
                        // Skip backup files
                        if !path.to_string_lossy().ends_with(".bak") {
                            files.push(path);
                        }
                    }
                }
            }
        }
    }

    visit_dir(dir, &mut files, include_rs);
    files.sort();
    files
}

/// Migrate all .sg/.sigil/.rs files in a directory (recursive).
/// With output_dir, preserves directory structure and converts .rs → .sg.
fn migrate_directory(dir_path: &str, output_dir: Option<&str>, dry_run: bool, backup: bool, evidentiality: bool) -> ExitCode {
    let input_path = std::path::Path::new(dir_path);
    if !input_path.exists() {
        eprintln!("Error: directory '{}' does not exist", dir_path);
        return ExitCode::from(1);
    }

    let files = collect_migrate_files_with_rs(input_path, output_dir.is_some());
    if files.is_empty() {
        if output_dir.is_some() {
            eprintln!("No .sg, .sigil, or .rs files found in '{}'", dir_path);
        } else {
            eprintln!("No .sg or .sigil files found in '{}'", dir_path);
        }
        return ExitCode::from(1);
    }

    println!("Migrating {} files in '{}'...", files.len(), dir_path);
    if let Some(out_dir) = output_dir {
        println!("Output directory: {}", out_dir);
    }
    println!();

    let mut total_files_changed = 0;
    let mut errors = 0;

    for file in &files {
        let file_str = file.to_string_lossy();
        // Read file to count changes before calling migrate_file
        if let Ok(source) = fs::read_to_string(file) {
            // Quick check: does this file have any Rust syntax or attributes?
            let has_rust = source.contains("pub ") || source.contains("fn ") || source.contains("let ")
                || source.contains("struct ") || source.contains("impl ") || source.contains("trait ")
                || source.contains("enum ") || source.contains("match ") || source.contains("::")
                || source.contains("#[");
            if !has_rust && !evidentiality && output_dir.is_none() {
                continue; // Skip already-native files silently (in-place mode only)
            }
        }

        // Compute output subdirectory preserving structure
        let file_output_dir = if let Some(out_dir) = output_dir {
            // Get relative path from input directory
            let rel_path = file.strip_prefix(input_path).unwrap_or(file);
            if let Some(parent) = rel_path.parent() {
                let parent_str = parent.to_string_lossy();
                if parent_str.is_empty() {
                    Some(out_dir.to_string())
                } else {
                    Some(format!("{}/{}", out_dir, parent_str))
                }
            } else {
                Some(out_dir.to_string())
            }
        } else {
            None
        };

        let result = migrate_file(&file_str, file_output_dir.as_deref(), dry_run, backup, evidentiality);
        if result == ExitCode::SUCCESS {
            total_files_changed += 1;
        } else {
            errors += 1;
        }
    }

    println!();
    println!("=== Migration Summary ===");
    println!("  Files scanned: {}", files.len());
    println!("  Files migrated: {}", total_files_changed);
    if errors > 0 {
        println!("  Errors: {}", errors);
    }

    if errors > 0 { ExitCode::from(1) } else { ExitCode::SUCCESS }
}

/// Migrate all files in a workspace (reads Sigil.toml).
fn migrate_workspace(dry_run: bool, backup: bool, evidentiality: bool) -> ExitCode {
    use toml::Value as TomlValue;

    // Look for Sigil.toml
    let manifest_content = match fs::read_to_string("Sigil.toml")
        .or_else(|_| fs::read_to_string("sigil.toml"))
    {
        Ok(s) => s,
        Err(_) => {
            eprintln!("Error: No Sigil.toml found in current directory");
            eprintln!("Run this command from a Sigil workspace root, or use:");
            eprintln!("  sigil migrate <directory>");
            return ExitCode::from(1);
        }
    };

    let manifest: TomlValue = match manifest_content.parse() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("Error parsing Sigil.toml: {}", e);
            return ExitCode::from(1);
        }
    };

    let project_name = manifest
        .get("project")
        .and_then(|p| p.get("name"))
        .and_then(|n| n.as_str())
        .unwrap_or("unnamed");

    // Get workspace members
    let members: Vec<String> = manifest
        .get("workspace")
        .and_then(|w| w.get("members"))
        .and_then(|m| m.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str())
                .map(|s| s.to_string())
                .collect()
        })
        .unwrap_or_default();

    if members.is_empty() {
        eprintln!("Error: No workspace members found in Sigil.toml");
        return ExitCode::from(1);
    }

    println!("Migrating workspace '{}' ({} members)...", project_name, members.len());
    println!();

    // Collect all files from all workspace members
    let mut all_files = Vec::new();
    for member in &members {
        let member_path = std::path::Path::new(member);
        if member_path.exists() {
            let files = collect_migrate_files(member_path);
            all_files.extend(files);
        } else {
            eprintln!("  Warning: workspace member '{}' not found, skipping", member);
        }
    }

    // Also check for top-level src/ and examples/
    for extra_dir in &["src", "examples"] {
        let extra_path = std::path::Path::new(extra_dir);
        if extra_path.exists() {
            all_files.extend(collect_migrate_files(extra_path));
        }
    }

    if all_files.is_empty() {
        println!("No .sg or .sigil files found in workspace members");
        return ExitCode::SUCCESS;
    }

    println!("Found {} files across {} workspace members", all_files.len(), members.len());
    println!();

    let mut files_migrated = 0;
    let mut files_skipped = 0;
    let mut errors = 0;

    for file in &all_files {
        let file_str = file.to_string_lossy();
        // Quick check for Rust syntax
        if let Ok(source) = fs::read_to_string(file) {
            let has_rust = source.contains("pub ") || source.contains("fn ") || source.contains("let ")
                || source.contains("struct ") || source.contains("impl ") || source.contains("trait ")
                || source.contains("enum ") || source.contains("match ") || source.contains("::");
            if !has_rust && !evidentiality {
                files_skipped += 1;
                continue;
            }
        }

        let result = migrate_file(&file_str, None, dry_run, backup, evidentiality);
        if result == ExitCode::SUCCESS {
            files_migrated += 1;
        } else {
            errors += 1;
        }
    }

    println!();
    println!("=== Workspace Migration Summary ===");
    println!("  Workspace: {}", project_name);
    println!("  Files scanned: {}", all_files.len());
    println!("  Files migrated: {}", files_migrated);
    if files_skipped > 0 {
        println!("  Files already native: {}", files_skipped);
    }
    if errors > 0 {
        println!("  Errors: {}", errors);
    }

    if errors > 0 { ExitCode::from(1) } else { ExitCode::SUCCESS }
}

/// Migrate a file from Rust syntax to native Sigil syntax.
///
/// Converts deprecated Rust keywords to their Sigil equivalents:
/// - fn → rite (function declaration)
/// - let → ≔ (definition)
/// - mut → Δ (delta/mutable)
/// - struct → Σ (sigma)
/// - impl → ⊢ (turnstile)
/// - trait → Θ (theta)
/// - enum → ᛈ (perthro rune)
/// - pub → ☉ (sun/public)
/// - mod → scroll
/// - use → invoke
/// - if → ⎇ (branch)
/// - else → ⎉ (alternative)
/// - match → ⌥ (option)
/// - while → ⟳ (cycle)
/// - for → ∀ (forall)
/// - in → ∈ (element-of)
/// - return → ⤺ (return-arrow)
/// - break → ⊗ (tensor/break)
/// - continue → ↻ (cycle-arrow)
/// - :: → · (middledot)
/// - &mut → &Δ
///
/// With --evidentiality flag, also adds evidentiality markers to external data sources:
/// - Http·get/post/request → ~ (Reported)
/// - File·read/write → ~ (Reported)
/// - Env·var → ? (Uncertain)
/// - stdin/readline → ? (Uncertain)
/// - Model·predict/infer → ◊ (Predicted)
/// - rand/random → ? (Uncertain)
/// - Time·now → ~ (Reported)
///
/// With output_dir, writes to that directory (converting .rs → .sg).
/// Without output_dir, modifies files in-place.
fn migrate_file(path: &str, output_dir: Option<&str>, dry_run: bool, backup: bool, evidentiality: bool) -> ExitCode {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Error reading '{}': {}", path, e);
            return ExitCode::from(1);
        }
    };

    // Compute output path
    let output_path = if let Some(out_dir) = output_dir {
        let input_path = std::path::Path::new(path);
        let file_name = input_path.file_name().unwrap_or_default().to_string_lossy();
        // Change .rs extension to .sg
        let new_name = if file_name.ends_with(".rs") {
            file_name.trim_end_matches(".rs").to_string() + ".sg"
        } else {
            file_name.to_string()
        };
        format!("{}/{}", out_dir, new_name)
    } else {
        path.to_string()
    };

    // Simple replacements (not keywords - always replace)
    let simple_replacements = [
        ("::", "·"),
    ];

    // Keyword replacements - require word boundary before (start of line, whitespace, or punctuation)
    // Format: (keyword, replacement, suffixes_to_match)
    let keyword_replacements: &[(&str, &str, &[&str])] = &[
        ("&mut", "&Δ", &[" ", "\t", "("]),
        ("fn", "rite", &[" ", "("]),
        ("let", "≔", &[" ", "\t"]),
        ("mut", "Δ", &[" ", ","]),
        ("struct", "Σ", &[" "]),
        ("impl", "⊢", &[" ", "<"]),
        ("trait", "Θ", &[" "]),
        ("enum", "ᛈ", &[" "]),
        ("pub", "☉", &[" ", "("]),
        ("mod", "scroll", &[" "]),
        ("use", "invoke", &[" "]),
        ("if", "⎇", &[" ", "("]),
        ("else", "⎉", &[" ", "{"]),
        ("match", "⌥", &[" ", "("]),
        ("while", "⟳", &[" ", "("]),
        ("for", "∀", &[" ", "("]),
        ("return", "⤺", &[" ", ";", "(", ","]),
        ("break", "⊗", &[";", " ", ","]),
        ("continue", "↻", &[";", " ", ","]),
    ];

    let mut result = source.clone();
    let mut changes = 0;
    let mut ev_changes = 0;

    // Apply simple replacements
    for (from, to) in &simple_replacements {
        let count = result.matches(from).count();
        if count > 0 {
            changes += count;
            result = result.replace(from, to);
        }
    }

    // Convert Rust attributes #[...] and #![...] to Sigil rune comments //@ rune: ...
    // Only converts line-starting attributes (not inline ones like #[from] in enum variants)
    // Inner attributes #![...] become //@ rune!: ...
    let mut attr_result = String::with_capacity(result.len());
    let mut attr_changes = 0;
    let mut chars = result.chars().peekable();
    let mut line_start = true;

    while let Some(c) = chars.next() {
        if c == '\n' {
            attr_result.push(c);
            line_start = true;
            continue;
        }

        if line_start && (c == ' ' || c == '\t') {
            attr_result.push(c);
            continue;
        }

        if line_start && c == '#' {
            // Check for #![ (inner attribute) or #[ (outer attribute)
            let is_inner = chars.peek() == Some(&'!');
            if is_inner {
                chars.next(); // consume '!'
            }

            if chars.peek() == Some(&'[') {
                chars.next(); // consume '['
                // Collect the attribute content until matching ']'
                // Skip bracket counting inside string literals
                let mut attr_content = String::new();
                let mut bracket_depth = 1;
                let mut in_string = false;
                let mut escape_next = false;
                while let Some(ac) = chars.next() {
                    if escape_next {
                        escape_next = false;
                        attr_content.push(ac);
                        continue;
                    }
                    if ac == '\\' && in_string {
                        escape_next = true;
                        attr_content.push(ac);
                        continue;
                    }
                    if ac == '"' {
                        in_string = !in_string;
                        attr_content.push(ac);
                        continue;
                    }
                    if !in_string {
                        if ac == '[' {
                            bracket_depth += 1;
                        } else if ac == ']' {
                            bracket_depth -= 1;
                            if bracket_depth == 0 {
                                break;
                            }
                        }
                    }
                    attr_content.push(ac);
                }

                // Convert the attribute content
                // Replace :: with · in attribute names (e.g., tokio::test → tokio·test)
                let converted_attr = attr_content.replace("::", "·");

                // Known Sigil rune attributes (convert to //@ rune:)
                // Unknown attributes become regular comments (// #[...])
                let attr_name = converted_attr.split('(').next().unwrap_or(&converted_attr).trim();
                let known_runes = [
                    // Core derive and testing
                    "derive", "test", "default", "ignore",
                    // Error handling (thiserror)
                    "error", "from", "source",
                    // Serde
                    "serde",
                    // Lints (kept as regular comments since Sigil has different lint system)
                    // "allow", "warn", "deny",
                    // Documentation
                    "doc", "deprecated",
                    // Async
                    "tokio·test", "async_trait",
                    // Note: Python bindings (pyo3, pyfunction, etc.) are Rust-specific FFI
                    // Note: cfg, cfg_attr, inline, must_use, repr are Rust-specific
                ];

                let is_known_rune = known_runes.iter().any(|&r| attr_name == r || attr_name.starts_with(&format!("{}(", r)));

                if is_known_rune {
                    if is_inner {
                        attr_result.push_str("//@ rune!: ");
                    } else {
                        attr_result.push_str("//@ rune: ");
                    }
                    attr_result.push_str(&converted_attr);
                } else {
                    // Unknown attribute - keep as regular comment without #[ prefix
                    // (Sigil parser warns about comments containing #[...])
                    attr_result.push_str("// ");
                    attr_result.push_str(&converted_attr);
                }
                attr_changes += 1;
                line_start = false;
                continue;
            } else if is_inner {
                // Was #! but not #![, put it back
                attr_result.push('#');
                attr_result.push('!');
                line_start = false;
                continue;
            }
        }

        line_start = false;
        attr_result.push(c);
    }

    result = attr_result;
    changes += attr_changes;

    // Apply keyword replacements with word boundary check
    for (keyword, replacement, suffixes) in keyword_replacements {
        for suffix in *suffixes {
            let pattern = format!("{}{}", keyword, suffix);
            let replacement_str = format!("{}{}", replacement, suffix);

            // Only replace if preceded by word boundary (start, whitespace, newline, or certain punctuation)
            let mut new_result = String::with_capacity(result.len());
            let mut last_end = 0;

            for (idx, _) in result.match_indices(&pattern) {
                // Check if this is at a word boundary
                let is_word_boundary = if idx == 0 {
                    true
                } else {
                    let prev_char = result[..idx].chars().last().unwrap();
                    !prev_char.is_alphanumeric() && prev_char != '_'
                };

                if is_word_boundary {
                    new_result.push_str(&result[last_end..idx]);
                    new_result.push_str(&replacement_str);
                    last_end = idx + pattern.len();
                    changes += 1;
                }
            }
            new_result.push_str(&result[last_end..]);
            result = new_result;
        }
    }

    // Special case: " in " -> " ∈ " (always safe, has spaces on both sides)
    let in_count = result.matches(" in ").count();
    if in_count > 0 {
        changes += in_count;
        result = result.replace(" in ", " ∈ ");
    }

    // Apply evidentiality markers if requested
    if evidentiality {
        // External data source patterns and their markers
        // Format: (pattern, marker_symbol, marker_name)
        let external_data_sources: &[(&str, &str, &str)] = &[
            // HTTP/Network - Reported (~)
            ("Http·get", "~", "Reported"),
            ("Http·post", "~", "Reported"),
            ("Http·put", "~", "Reported"),
            ("Http·delete", "~", "Reported"),
            ("Http·request", "~", "Reported"),
            ("fetch(", "~", "Reported"),
            ("TcpStream·connect", "~", "Reported"),
            ("Socket·recv", "~", "Reported"),
            ("WebSocket·receive", "~", "Reported"),

            // File I/O - Reported (~)
            ("File·read", "~", "Reported"),
            ("File·open", "~", "Reported"),
            ("fs·read", "~", "Reported"),
            ("std·fs·read", "~", "Reported"),

            // User Input - Uncertain (?)
            ("stdin", "?", "Uncertain"),
            ("readline", "?", "Uncertain"),
            ("Env·var", "?", "Uncertain"),
            ("Env·get", "?", "Uncertain"),
            ("std·env·var", "?", "Uncertain"),
            ("args()", "?", "Uncertain"),

            // Database - Reported (~)
            ("Db·query", "~", "Reported"),
            ("Db·execute", "~", "Reported"),
            ("query(", "~", "Reported"),

            // System - Reported (~)
            ("Time·now", "~", "Reported"),
            ("Instant·now", "~", "Reported"),
            ("SystemTime·now", "~", "Reported"),

            // Random - Uncertain (?)
            ("rand·", "?", "Uncertain"),
            ("random(", "?", "Uncertain"),
            ("Rng·", "?", "Uncertain"),

            // ML/AI - Predicted (◊)
            ("Model·predict", "◊", "Predicted"),
            ("Model·infer", "◊", "Predicted"),
            ("model·forward", "◊", "Predicted"),
            ("llm·complete", "◊", "Predicted"),
            ("llm·chat", "◊", "Predicted"),
            ("embed(", "◊", "Predicted"),

            // JSON/Parsing - Uncertain (?)
            ("json·parse", "?", "Uncertain"),
            ("serde·deserialize", "?", "Uncertain"),
            ("from_str(", "?", "Uncertain"),
        ];

        // Look for let bindings with external data sources
        // Pattern: ≔ <varname> = <external_call>
        // We need to add marker after varname if not present
        for (pattern, marker, _name) in external_data_sources {
            // Find lines with this pattern in a let binding
            let lines: Vec<&str> = result.lines().collect();
            let mut new_lines: Vec<String> = Vec::with_capacity(lines.len());

            for line in lines {
                let mut new_line = line.to_string();

                // Check if line contains the external data pattern
                if line.contains(pattern) {
                    // Check if it's a let binding: ≔ varname = ...pattern...
                    if let Some(let_pos) = line.find("≔ ") {
                        let after_let = &line[let_pos + "≔ ".len()..];

                        // Find the variable name (up to = or :)
                        if let Some(eq_pos) = after_let.find(" = ") {
                            let var_part = &after_let[..eq_pos];
                            let var_name = var_part.trim();

                            // Check if variable already has an evidentiality marker
                            let has_marker = var_name.ends_with('!')
                                || var_name.ends_with('?')
                                || var_name.ends_with('~')
                                || var_name.ends_with('◊')
                                || var_name.ends_with('‽');

                            if !has_marker && !var_name.contains(':') {
                                // Add the marker after the variable name
                                let old_pattern = format!("≔ {} = ", var_name);
                                let new_pattern = format!("≔ {}{} = ", var_name, marker);

                                if line.contains(&old_pattern) {
                                    new_line = line.replace(&old_pattern, &new_pattern);
                                    ev_changes += 1;
                                }
                            }
                        }
                    }
                }

                new_lines.push(new_line);
            }

            result = new_lines.join("\n");
            // Preserve trailing newline if original had one
            if source.ends_with('\n') && !result.ends_with('\n') {
                result.push('\n');
            }
        }
    }

    let total_changes = changes + ev_changes;

    if total_changes == 0 {
        if evidentiality {
            println!("✓ {} - no migrations needed (syntax and evidentiality up to date)", path);
        } else {
            println!("✓ {} - no Rust syntax found, already native Sigil", path);
        }
        return ExitCode::SUCCESS;
    }

    if dry_run {
        println!("=== Dry run: {} changes would be made to {} ===", total_changes, path);
        if changes > 0 {
            println!("  Syntax changes: {}", changes);
        }
        if ev_changes > 0 {
            println!("  Evidentiality markers added: {}", ev_changes);
        }
        println!();
        println!("{}", result);
        println!();
        println!("Run without --dry-run to apply changes.");
        return ExitCode::SUCCESS;
    }

    // Create backup if requested (only for in-place migration)
    if backup && output_dir.is_none() {
        let backup_path = format!("{}.bak", path);
        if let Err(e) = fs::write(&backup_path, &source) {
            eprintln!("Error creating backup '{}': {}", backup_path, e);
            return ExitCode::from(1);
        }
        println!("Created backup: {}", backup_path);
    }

    // Create output directory if needed
    if let Some(out_dir) = output_dir {
        if let Err(e) = fs::create_dir_all(out_dir) {
            eprintln!("Error creating output directory '{}': {}", out_dir, e);
            return ExitCode::from(1);
        }
    }

    // Write the migrated file
    if let Err(e) = fs::write(&output_path, &result) {
        eprintln!("Error writing '{}': {}", output_path, e);
        return ExitCode::from(1);
    }

    if output_dir.is_some() {
        println!("✓ Migrated {} → {} ({} replacements)", path, output_path, total_changes);
    } else {
        println!("✓ Migrated {} ({} replacements)", path, total_changes);
    }
    println!();
    if changes > 0 {
        println!("Converted Rust syntax to native Sigil:");
        println!("  rite (fn), ≔ (let), Δ (mut), Σ (struct), ⊢ (impl)");
        println!("  Θ (trait), ᛈ (enum), ☉ (pub), · (::)");
        println!("  ⎇/⎉ (if/else), ⌥ (match), ⟳ (while), ∀/∈ (for/in)");
        println!("  ⤺ (return), ⊗ (break), ↻ (continue)");
    }
    if ev_changes > 0 {
        println!();
        println!("Added evidentiality markers ({} variables):", ev_changes);
        println!("  ~ (Reported) - HTTP, File I/O, Database, Time");
        println!("  ? (Uncertain) - User input, Environment, Random, Parsing");
        println!("  ◊ (Predicted) - ML models, LLM completions");
    }

    ExitCode::SUCCESS
}

fn repl() -> ExitCode {
    println!(
        "{}{}Sigil REPL v0.1.0{}",
        colors::BOLD,
        colors::MORPHEME,
        colors::RESET
    );
    println!("A polysynthetic language with evidentiality types.");
    println!();
    println!("{}Commands:{}", colors::DIM, colors::RESET);
    println!("  {}:help{}     Show help", colors::KEYWORD, colors::RESET);
    println!(
        "  {}:ast{}      Toggle AST display",
        colors::KEYWORD,
        colors::RESET
    );
    println!(
        "  {}:clear{}    Clear screen",
        colors::KEYWORD,
        colors::RESET
    );
    println!("  {}:exit{}     Exit REPL", colors::KEYWORD, colors::RESET);
    println!();
    println!(
        "{}Press Tab for completions, ↑/↓ for history{}",
        colors::DIM,
        colors::RESET
    );
    println!();

    let config = Config::builder()
        .history_ignore_space(true)
        .completion_type(rustyline::CompletionType::List)
        .build();

    let helper = SigilHelper::new();
    let mut rl: Editor<SigilHelper, _> = match Editor::with_config(config) {
        Ok(rl) => rl,
        Err(e) => {
            eprintln!("Error initializing REPL: {}", e);
            return ExitCode::from(1);
        }
    };
    rl.set_helper(Some(helper));

    // Load history
    let history_path = dirs_home().map(|h| h.join(".sigil_history"));
    if let Some(ref path) = history_path {
        let _ = rl.load_history(path);
    }

    let mut interpreter = Interpreter::new();
    register_stdlib(&mut interpreter);
    let mut show_ast = false;

    loop {
        let readline = rl.readline("");
        match readline {
            Ok(line) => {
                let input = line.trim();
                if input.is_empty() {
                    continue;
                }

                let _ = rl.add_history_entry(input);

                // Handle REPL commands
                match input {
                    ":exit" | ":quit" | "exit" | "quit" => break,
                    ":clear" => {
                        print!("\x1b[2J\x1b[H"); // Clear screen
                        continue;
                    }
                    ":help" => {
                        print_help();
                        continue;
                    }
                    ":symbols" => {
                        print_symbols();
                        continue;
                    }
                    ":ast" => {
                        show_ast = !show_ast;
                        println!(
                            "AST display: {}{}{}",
                            colors::KEYWORD,
                            if show_ast { "on" } else { "off" },
                            colors::RESET
                        );
                        continue;
                    }
                    _ => {}
                }

                // Try to parse and evaluate
                evaluate_input(&mut interpreter, input, show_ast);
            }
            Err(ReadlineError::Interrupted) => {
                println!("{}^C{}", colors::DIM, colors::RESET);
                continue;
            }
            Err(ReadlineError::Eof) => {
                break;
            }
            Err(err) => {
                eprintln!("{}Error: {}{}", colors::SPECIAL, err, colors::RESET);
                break;
            }
        }
    }

    // Save history
    if let Some(ref path) = history_path {
        let _ = rl.save_history(path);
    }

    println!("{}Goodbye!{}", colors::DIM, colors::RESET);
    ExitCode::SUCCESS
}

fn dirs_home() -> Option<std::path::PathBuf> {
    std::env::var_os("HOME").map(std::path::PathBuf::from)
}

fn print_help() {
    println!(
        "{}{}Sigil REPL Commands:{}",
        colors::BOLD,
        colors::MORPHEME,
        colors::RESET
    );
    println!();
    println!(
        "  {}:help{}      Show this help",
        colors::KEYWORD,
        colors::RESET
    );
    println!(
        "  {}:symbols{}   Show all Unicode symbols",
        colors::KEYWORD,
        colors::RESET
    );
    println!(
        "  {}:ast{}       Toggle AST display mode",
        colors::KEYWORD,
        colors::RESET
    );
    println!(
        "  {}:clear{}     Clear the screen",
        colors::KEYWORD,
        colors::RESET
    );
    println!(
        "  {}:exit{}      Exit the REPL",
        colors::KEYWORD,
        colors::RESET
    );
    println!();
    println!("{}{}Examples:{}", colors::BOLD, colors::TYPE, colors::RESET);
    println!();
    println!("  {}// Arithmetic{}", colors::COMMENT, colors::RESET);
    println!(
        "  {}1{} + {}2{} * {}3{}",
        colors::NUMBER,
        colors::RESET,
        colors::NUMBER,
        colors::RESET,
        colors::NUMBER,
        colors::RESET
    );
    println!();
    println!("  {}// Variables{}", colors::COMMENT, colors::RESET);
    println!(
        "  {}let{} x = {}42{};",
        colors::KEYWORD,
        colors::RESET,
        colors::NUMBER,
        colors::RESET
    );
    println!();
    println!(
        "  {}// Pipe transforms (polysynthetic){}",
        colors::COMMENT,
        colors::RESET
    );
    println!(
        "  [{}1{}, {}2{}, {}3{}]|{}τ{}{{_ * {}2{}}}  {}// [2, 4, 6]{}",
        colors::NUMBER,
        colors::RESET,
        colors::NUMBER,
        colors::RESET,
        colors::NUMBER,
        colors::RESET,
        colors::MORPHEME,
        colors::RESET,
        colors::NUMBER,
        colors::RESET,
        colors::COMMENT,
        colors::RESET
    );
    println!();
    println!(
        "  {}// Evidentiality markers{}",
        colors::COMMENT,
        colors::RESET
    );
    println!(
        "  {}known{}({}42{})    {}// Certain value (!){}",
        colors::FUNCTION,
        colors::RESET,
        colors::NUMBER,
        colors::RESET,
        colors::COMMENT,
        colors::RESET
    );
    println!(
        "  {}uncertain{}(x) {}// Uncertain value (?){}",
        colors::FUNCTION,
        colors::RESET,
        colors::COMMENT,
        colors::RESET
    );
    println!();
    println!("  {}// Functions{}", colors::COMMENT, colors::RESET);
    println!(
        "  {}fn{} add(a, b) {{ a + b }}",
        colors::KEYWORD,
        colors::RESET
    );
    println!();
    println!(
        "{}{}Morphemes:{}",
        colors::BOLD,
        colors::MORPHEME,
        colors::RESET
    );
    println!(
        "  {}τ{} transform  {}φ{} filter  {}σ{} sort  {}ρ{} reduce  {}Λ{} lambda",
        colors::MORPHEME,
        colors::RESET,
        colors::MORPHEME,
        colors::RESET,
        colors::MORPHEME,
        colors::RESET,
        colors::MORPHEME,
        colors::RESET,
        colors::MORPHEME,
        colors::RESET
    );
    println!();
    println!(
        "Type {}:symbols{} for a complete symbol reference.",
        colors::KEYWORD,
        colors::RESET
    );
    println!();
}

fn print_symbols() {
    println!(
        "{}{}Sigil Unicode Symbols Reference{}",
        colors::BOLD,
        colors::MORPHEME,
        colors::RESET
    );
    println!();

    // Transform Morphemes
    println!(
        "{}{}Transform Morphemes (Pipe Syntax: data|morpheme):{}",
        colors::BOLD,
        colors::MORPHEME,
        colors::RESET
    );
    println!(
        "  {}τ{}/{}Τ{}  transform/map   data|τ{{_ * 2}}",
        colors::MORPHEME,
        colors::RESET,
        colors::MORPHEME,
        colors::RESET
    );
    println!(
        "  {}φ{}/{}Φ{}  filter          data|φ{{_ > 0}}",
        colors::MORPHEME,
        colors::RESET,
        colors::MORPHEME,
        colors::RESET
    );
    println!(
        "  {}σ{}     sort            data|σ",
        colors::MORPHEME,
        colors::RESET
    );
    println!(
        "  {}Σ{}     sum             data|Σ",
        colors::MORPHEME,
        colors::RESET
    );
    println!(
        "  {}ρ{}/{}Ρ{}  reduce/fold     data|ρ{{acc + _}}",
        colors::MORPHEME,
        colors::RESET,
        colors::MORPHEME,
        colors::RESET
    );
    println!(
        "  {}λ{}/{}Λ{}  lambda/closure   λ(x) {{ x + 1 }}",
        colors::MORPHEME,
        colors::RESET,
        colors::MORPHEME,
        colors::RESET
    );
    println!(
        "  {}Π{}     product         data|Π",
        colors::MORPHEME,
        colors::RESET
    );
    println!();

    // Access Morphemes
    println!(
        "{}{}Access Morphemes (Pipe Syntax: data|morpheme):{}",
        colors::BOLD,
        colors::MORPHEME,
        colors::RESET
    );
    println!(
        "  {}α{}     first element   data|α",
        colors::MORPHEME,
        colors::RESET
    );
    println!(
        "  {}ω{}/{}Ω{}  last element    data|ω",
        colors::MORPHEME,
        colors::RESET,
        colors::MORPHEME,
        colors::RESET
    );
    println!(
        "  {}μ{}/{}Μ{}  middle/median   data|μ",
        colors::MORPHEME,
        colors::RESET,
        colors::MORPHEME,
        colors::RESET
    );
    println!(
        "  {}χ{}/{}Χ{}  random choice   data|χ",
        colors::MORPHEME,
        colors::RESET,
        colors::MORPHEME,
        colors::RESET
    );
    println!(
        "  {}ν{}/{}Ν{}  nth element     data|ν{{2}}",
        colors::MORPHEME,
        colors::RESET,
        colors::MORPHEME,
        colors::RESET
    );
    println!(
        "  {}ξ{}/{}Ξ{}  next in iter    data|ξ",
        colors::MORPHEME,
        colors::RESET,
        colors::MORPHEME,
        colors::RESET
    );
    println!();

    // Evidentiality
    println!(
        "{}{}Evidentiality Markers:{}",
        colors::BOLD,
        colors::EVIDENCE,
        colors::RESET
    );
    println!(
        "  {}!{}  known/direct     let x{}!{} = verified();",
        colors::EVIDENCE,
        colors::RESET,
        colors::EVIDENCE,
        colors::RESET
    );
    println!(
        "  {}?{}  uncertain        let x{}?{} = maybe_get();",
        colors::EVIDENCE,
        colors::RESET,
        colors::EVIDENCE,
        colors::RESET
    );
    println!(
        "  {}~{}  reported         let x{}~{} = fetch_api();",
        colors::EVIDENCE,
        colors::RESET,
        colors::EVIDENCE,
        colors::RESET
    );
    println!(
        "  {}‽{}  paradox          let x{}‽{} = contradict();",
        colors::EVIDENCE,
        colors::RESET,
        colors::EVIDENCE,
        colors::RESET
    );
    println!();

    // Logic Operators
    println!(
        "{}{}Logic Operators:{}",
        colors::BOLD,
        colors::OPERATOR,
        colors::RESET
    );
    println!(
        "  {}∧{}  AND (&&)        a ∧ b",
        colors::OPERATOR,
        colors::RESET
    );
    println!(
        "  {}∨{}  OR (||)         a ∨ b",
        colors::OPERATOR,
        colors::RESET
    );
    println!(
        "  {}¬{}  NOT (!)         ¬a",
        colors::OPERATOR,
        colors::RESET
    );
    println!(
        "  {}⊻{}  XOR             a ⊻ b",
        colors::OPERATOR,
        colors::RESET
    );
    println!("  {}⊤{}  true/top", colors::OPERATOR, colors::RESET);
    println!("  {}⊥{}  false/bottom", colors::OPERATOR, colors::RESET);
    println!();

    // Bitwise Operators
    println!(
        "{}{}Bitwise Operators:{}",
        colors::BOLD,
        colors::OPERATOR,
        colors::RESET
    );
    println!(
        "  {}⋏{}  AND (&)         a ⋏ b",
        colors::OPERATOR,
        colors::RESET
    );
    println!(
        "  {}⋎{}  OR              a ⋎ b",
        colors::OPERATOR,
        colors::RESET
    );
    println!();

    // Set Operators
    println!(
        "{}{}Set Operators:{}",
        colors::BOLD,
        colors::OPERATOR,
        colors::RESET
    );
    println!(
        "  {}∪{}  union           a ∪ b",
        colors::OPERATOR,
        colors::RESET
    );
    println!(
        "  {}∩{}  intersection    a ∩ b",
        colors::OPERATOR,
        colors::RESET
    );
    println!(
        "  {}∖{}  difference      a ∖ b",
        colors::OPERATOR,
        colors::RESET
    );
    println!(
        "  {}⊂{}  proper subset   a ⊂ b",
        colors::OPERATOR,
        colors::RESET
    );
    println!(
        "  {}⊆{}  subset/equal    a ⊆ b",
        colors::OPERATOR,
        colors::RESET
    );
    println!(
        "  {}∈{}  element of      x ∈ set",
        colors::OPERATOR,
        colors::RESET
    );
    println!(
        "  {}∉{}  not element of  x ∉ set",
        colors::OPERATOR,
        colors::RESET
    );
    println!();

    // Data Operations
    println!(
        "{}{}Data Operations:{}",
        colors::BOLD,
        colors::SPECIAL,
        colors::RESET
    );
    println!(
        "  {}⋈{}  zip with op     zip_with(a, b, \"add\")",
        colors::SPECIAL,
        colors::RESET
    );
    println!(
        "  {}⋳{}  flatten         flatten(nested)",
        colors::SPECIAL,
        colors::RESET
    );
    println!(
        "  {}⊔{}  supremum/max    supremum(a, b)",
        colors::SPECIAL,
        colors::RESET
    );
    println!(
        "  {}⊓{}  infimum/min     infimum(a, b)",
        colors::SPECIAL,
        colors::RESET
    );
    println!();

    // Math Operations
    println!(
        "{}{}Math Operations:{}",
        colors::BOLD,
        colors::OPERATOR,
        colors::RESET
    );
    println!(
        "  {}∘{}  compose         f ∘ g",
        colors::OPERATOR,
        colors::RESET
    );
    println!(
        "  {}⊗{}  tensor          a ⊗ b",
        colors::OPERATOR,
        colors::RESET
    );
    println!(
        "  {}⊕{}  direct sum      a ⊕ b",
        colors::OPERATOR,
        colors::RESET
    );
    println!("  {}∫{}  integral/cumsum", colors::OPERATOR, colors::RESET);
    println!("  {}∂{}  partial/deriv", colors::OPERATOR, colors::RESET);
    println!(
        "  {}√{}  sqrt            √x",
        colors::OPERATOR,
        colors::RESET
    );
    println!(
        "  {}∛{}  cbrt            ∛x",
        colors::OPERATOR,
        colors::RESET
    );
    println!();

    // Special Literals
    println!(
        "{}{}Special Literals:{}",
        colors::BOLD,
        colors::SPECIAL,
        colors::RESET
    );
    println!(
        "  {}∅{}  empty/void      let x = ∅;",
        colors::SPECIAL,
        colors::RESET
    );
    println!(
        "  {}∞{}  infinity        let x = ∞;",
        colors::SPECIAL,
        colors::RESET
    );
    println!("  {}◯{}  geometric zero", colors::SPECIAL, colors::RESET);
    println!();

    // Quantifiers
    println!(
        "{}{}Quantifiers:{}",
        colors::BOLD,
        colors::OPERATOR,
        colors::RESET
    );
    println!("  {}∀{}  for all", colors::OPERATOR, colors::RESET);
    println!("  {}∃{}  exists", colors::OPERATOR, colors::RESET);
    println!();

    // Aspect Morphemes
    println!(
        "{}{}Aspect Suffixes (Function naming):{}",
        colors::BOLD,
        colors::MORPHEME,
        colors::RESET
    );
    println!(
        "  {}·ing{}  progressive   fn read·ing() -> Stream",
        colors::MORPHEME,
        colors::RESET
    );
    println!(
        "  {}·ed{}   perfective    fn process·ed() -> Result",
        colors::MORPHEME,
        colors::RESET
    );
    println!(
        "  {}·able{} potential     fn parse·able() -> Bool",
        colors::MORPHEME,
        colors::RESET
    );
    println!(
        "  {}·ive{}  resultative   fn destruct·ive() -> Parts",
        colors::MORPHEME,
        colors::RESET
    );
    println!();
}

fn evaluate_input(interpreter: &mut Interpreter, input: &str, show_ast: bool) {
    // First, try as a top-level item (fn, struct, etc.)
    let mut parser = Parser::new(input);
    let result = parser.parse_file();

    match result {
        Ok(ast) if !ast.items.is_empty() => {
            if show_ast {
                for item in &ast.items {
                    println!("{}{:#?}{}", colors::DIM, item.node, colors::RESET);
                }
            } else {
                match interpreter.execute(&ast) {
                    Ok(value) => {
                        if !matches!(value, sigil_parser::Value::Null) {
                            println!("{}=> {}{}", colors::DIM, colors::RESET, value);
                        }
                    }
                    Err(e) => eprintln!("{}Error: {}{}", colors::SPECIAL, e, colors::RESET),
                }
            }
        }
        Ok(_) => {
            // Empty file, try parsing as expression
            let wrapped = format!("fn __repl__() {{ {} }}", input);
            let mut parser = Parser::new(&wrapped);
            match parser.parse_file() {
                Ok(ast) => {
                    if show_ast {
                        if let Some(item) = ast.items.first() {
                            println!("{}{:#?}{}", colors::DIM, item.node, colors::RESET);
                        }
                    } else {
                        match interpreter.execute(&ast) {
                            Ok(_) => {
                                // Call __repl__ to get the result
                                let repl_fn =
                                    interpreter.globals.borrow().get("__repl__").and_then(|v| {
                                        if let sigil_parser::Value::Function(f) = v {
                                            Some(f.clone())
                                        } else {
                                            None
                                        }
                                    });
                                if let Some(f) = repl_fn {
                                    match interpreter.call_function(&f, vec![]) {
                                        Ok(value) => {
                                            if !matches!(value, sigil_parser::Value::Null) {
                                                println!(
                                                    "{}=> {}{}",
                                                    colors::DIM,
                                                    colors::RESET,
                                                    value
                                                );
                                            }
                                        }
                                        Err(e) => eprintln!(
                                            "{}Error: {}{}",
                                            colors::SPECIAL,
                                            e,
                                            colors::RESET
                                        ),
                                    }
                                }
                            }
                            Err(e) => eprintln!("{}Error: {}{}", colors::SPECIAL, e, colors::RESET),
                        }
                    }
                }
                Err(e) => eprintln!("{}Parse error: {}{}", colors::SPECIAL, e, colors::RESET),
            }
        }
        Err(e) => eprintln!("{}Parse error: {}{}", colors::SPECIAL, e, colors::RESET),
    }
}
