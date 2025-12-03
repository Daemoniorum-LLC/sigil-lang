//! Sigil CLI - Parse, check, and run Sigil source files.

use sigil_parser::{Parser, Interpreter, register_stdlib, Lexer, Token, Diagnostics, Diagnostic, IrEmitterOptions};
use sigil_parser::span::Span;
#[cfg(feature = "jit")]
use sigil_parser::JitCompiler;
#[cfg(feature = "llvm")]
use sigil_parser::{LlvmCompiler, CompileMode, OptLevel};
use std::env;
use std::fs;
use std::process::ExitCode;
use std::borrow::Cow;

/// Output format for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq)]
enum OutputFormat {
    Human,   // Pretty-printed with colors
    Json,    // JSON for AI agents (pretty)
    Compact, // JSON single-line for piping
}

use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::{Config, Editor, Helper};

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Sigil v0.1.0 - A polysynthetic programming language");
        eprintln!();
        eprintln!("Usage: sigil <command> [file.sigil] [options]");
        eprintln!();
        eprintln!("Commands:");
        eprintln!("  run <file>      Execute a Sigil file (interpreted)");
        eprintln!("  jit <file>      Execute a Sigil file (JIT compiled, fast)");
        eprintln!("  llvm <file>     Execute a Sigil file (LLVM backend, fastest)");
        eprintln!("  compile <file>  Compile to native executable (AOT, --lto for LTO)");
        eprintln!("  check <file>    Type-check and validate (for AI agents: --format=json)");
        eprintln!("  ir <file>       Emit AI-facing JSON IR (for AI agents/tooling)");
        eprintln!("  parse <file>    Parse and check a Sigil file");
        eprintln!("  lex <file>      Tokenize a Sigil file");
        eprintln!("  repl            Start interactive REPL");
        eprintln!();
        eprintln!("AI Agent Options:");
        eprintln!("  --format=json       Output as JSON (pretty-printed)");
        eprintln!("  --format=compact    Output as single-line JSON");
        eprintln!("  --quiet             Exit code only, no output (for fast validation)");
        eprintln!("  --apply-suggestions Auto-apply fix suggestions (alias: --fix)");
        eprintln!();
        eprintln!("IR Emission Options (for 'ir' command):");
        eprintln!("  --compact           Output as single-line JSON");
        eprintln!("  --include-spans     Include source spans in output");
        eprintln!("  --include-types     Include type annotations");
        return ExitCode::from(1);
    }

    match args[1].as_str() {
        "run" => {
            if args.len() < 3 {
                eprintln!("Error: missing file argument");
                return ExitCode::from(1);
            }
            run_file(&args[2])
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
                eprintln!("Usage: sigil compile <file.sigil> [-o output] [--lto]");
                return ExitCode::from(1);
            }
            // Parse flags
            let use_lto = args.iter().any(|a| a == "--lto");
            let output = if let Some(pos) = args.iter().position(|a| a == "-o") {
                if pos + 1 < args.len() {
                    args[pos + 1].clone()
                } else {
                    args[2].trim_end_matches(".sigil").trim_end_matches(".sg").to_string()
                }
            } else {
                // Default output name: strip extension
                args[2].trim_end_matches(".sigil").trim_end_matches(".sg").to_string()
            };
            compile_file(&args[2], &output, use_lto)
        }
        #[cfg(not(feature = "llvm"))]
        "compile" => {
            eprintln!("Error: AOT compilation requires LLVM (compile with --features llvm)");
            ExitCode::from(1)
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
            let apply_fixes = args.iter().any(|a| a == "--apply-suggestions" || a == "--fix");
            check_file(&args[2], format, quiet, apply_fixes)
        }
        "ir" => {
            if args.len() < 3 {
                eprintln!("Error: missing file argument");
                eprintln!("Usage: sigil ir <file.sigil> [--compact] [--include-spans] [--include-types]");
                return ExitCode::from(1);
            }
            let compact = args.iter().any(|a| a == "--compact");
            let include_spans = args.iter().any(|a| a == "--include-spans");
            let include_types = args.iter().any(|a| a == "--include-types");
            emit_ir_file(&args[2], !compact, include_spans, include_types)
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
        "repl" => {
            repl()
        }
        _ => {
            // Treat as file if it ends with .sigil or .sg
            if args[1].ends_with(".sigil") || args[1].ends_with(".sg") {
                run_file(&args[1])
            } else {
                eprintln!("Unknown command: {}", args[1]);
                ExitCode::from(1)
            }
        }
    }
}

fn run_file(path: &str) -> ExitCode {
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

    // Execute with full stdlib
    let mut interpreter = Interpreter::new();
    register_stdlib(&mut interpreter);
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
fn compile_file(path: &str, output: &str, use_lto: bool) -> ExitCode {
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
    let mut compiler = match LlvmCompiler::with_mode(&context, OptLevel::Aggressive, CompileMode::Aot) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("Failed to initialize LLVM compiler: {}", e);
            return ExitCode::from(1);
        }
    };

    // Compile
    if let Err(e) = compiler.compile(&source) {
        eprintln!("Compilation error in '{}': {}", path, e);
        return ExitCode::from(1);
    }

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

    // Find the runtime (static library or C source)
    let runtime_result = find_runtime(use_lto);
    if runtime_result.is_none() {
        eprintln!("Error: Could not find sigil runtime");
        eprintln!("Expected locations:");
        eprintln!("  - ./runtime/libsigil_runtime.a (pre-built, faster)");
        eprintln!("  - ./runtime/sigil_runtime.c (source, required for --lto)");
        eprintln!("Run 'make' in the runtime directory to build.");
        return ExitCode::from(1);
    }
    let (runtime, should_use_lto) = runtime_result.unwrap();
    let is_static_lib = runtime.ends_with(".a");

    // Link with clang/gcc
    let mode_str = if should_use_lto {
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

    let link_result = Command::new(&linker)
        .args(&args)
        .status();

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

#[cfg(feature = "llvm")]
fn find_runtime(use_lto: bool) -> Option<(String, bool)> {
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

    // Try relative to executable
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            // Try static library
            let lib_path = dir.join("runtime/libsigil_runtime.a");
            if lib_path.exists() {
                return Some((lib_path.to_string_lossy().into_owned(), false));
            }
            // Fall back to source
            let src_path = dir.join("runtime/sigil_runtime.c");
            if src_path.exists() {
                return Some((src_path.to_string_lossy().into_owned(), use_lto));
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
        Ok(_ast) => {
            // TODO: Add semantic analysis / type checking here
            // For now, just report successful parse
        }
        Err(e) => {
            // Convert parse error to diagnostic
            let diag = Diagnostic::error(format!("{}", e), Span::default())
                .with_code("E0002");
            diagnostics.add(diag);
        }
    }

    // Apply fixes if requested
    let source = if apply_fixes && !diagnostics.is_empty() {
        let suggestions: Vec<_> = diagnostics.iter()
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
                    fixed.replace_range(suggestion.span.start..suggestion.span.end, &suggestion.replacement);
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
                    println!("Applied {} fix{}:", applied_fixes.len(),
                        if applied_fixes.len() == 1 { "" } else { "es" });
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
    }

    if diagnostics.has_errors() {
        ExitCode::from(1)
    } else {
        ExitCode::SUCCESS
    }
}

/// Emit AI-facing JSON IR for a source file.
///
/// This is the primary interface for AI agents - provides structured
/// JSON output that can be consumed by LLMs and tooling.
fn emit_ir_file(path: &str, pretty: bool, include_spans: bool, include_types: bool) -> ExitCode {
    let source = match fs::read_to_string(path) {
        Ok(s) => s,
        Err(e) => {
            // Output error as JSON for consistency
            let error_json = serde_json::json!({
                "error": format!("Failed to read file: {}", e),
                "file": path
            });
            if pretty {
                eprintln!("{}", serde_json::to_string_pretty(&error_json).unwrap());
            } else {
                eprintln!("{}", serde_json::to_string(&error_json).unwrap());
            }
            return ExitCode::from(1);
        }
    };

    let mut parser = Parser::new(&source);
    match parser.parse_file() {
        Ok(ast) => {
            let options = IrEmitterOptions {
                include_types,
                include_spans,
                pretty,
            };
            let mut emitter = sigil_parser::ir::IrEmitter::with_options(options);
            let doc = emitter.emit(&ast, path);

            match emitter.to_json(&doc) {
                Ok(json) => {
                    println!("{}", json);
                    ExitCode::SUCCESS
                }
                Err(e) => {
                    eprintln!("Error serializing IR: {}", e);
                    ExitCode::from(1)
                }
            }
        }
        Err(e) => {
            // Output parse error as JSON
            let error_json = serde_json::json!({
                "error": format!("Parse error: {}", e),
                "file": path
            });
            if pretty {
                eprintln!("{}", serde_json::to_string_pretty(&error_json).unwrap());
            } else {
                eprintln!("{}", serde_json::to_string(&error_json).unwrap());
            }
            ExitCode::from(1)
        }
    }
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
                let trait_name = trait_.segments.iter()
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

    // Semantic colors for Sigil
    pub const KEYWORD: &str = "\x1b[38;5;198m";      // Magenta/pink for keywords
    pub const MORPHEME: &str = "\x1b[38;5;51m";      // Cyan for morphemes (τ, φ, σ, ρ, λ)
    pub const EVIDENCE: &str = "\x1b[38;5;214m";     // Orange for evidentiality (!, ?, ~, ‽)
    pub const STRING: &str = "\x1b[38;5;114m";       // Green for strings
    pub const NUMBER: &str = "\x1b[38;5;141m";       // Purple for numbers
    pub const COMMENT: &str = "\x1b[38;5;245m";      // Gray for comments
    pub const OPERATOR: &str = "\x1b[38;5;252m";     // Light gray for operators
    pub const FUNCTION: &str = "\x1b[38;5;81m";      // Blue for function names
    pub const TYPE: &str = "\x1b[38;5;222m";         // Yellow for types
    pub const SPECIAL: &str = "\x1b[38;5;203m";      // Red for special symbols
}

/// Sigil syntax highlighter using the lexer
struct SigilHighlighter;

impl SigilHighlighter {
    fn highlight_token(token: &Token) -> &'static str {
        match token {
            // Keywords
            Token::Fn | Token::Let | Token::Mut | Token::If | Token::Else |
            Token::While | Token::For | Token::In | Token::Match | Token::Return |
            Token::Break | Token::Continue | Token::Struct | Token::Enum |
            Token::Impl | Token::Trait | Token::Pub | Token::Use | Token::Mod |
            Token::Const | Token::Static | Token::Type | Token::Where |
            Token::Async | Token::Await | Token::Actor |
            Token::SelfLower | Token::SelfUpper |
            Token::True | Token::False => colors::KEYWORD,

            // Morphemes (polysynthetic operators) - including new access morphemes
            Token::Tau | Token::Phi | Token::Sigma | Token::Rho | Token::Lambda |
            Token::Pi | Token::Alpha | Token::Omega | Token::Mu | Token::Chi |
            Token::Nu | Token::Xi => colors::MORPHEME,

            // Aspect morphemes
            Token::AspectProgressive | Token::AspectPerfective |
            Token::AspectPotential | Token::AspectResultative => colors::MORPHEME,

            // Bitwise operators (Unicode)
            Token::BitwiseAndSymbol | Token::BitwiseOrSymbol => colors::OPERATOR,

            // Data operation symbols
            Token::Bowtie | Token::ElementSmallVerticalBar |
            Token::SquareCup | Token::SquareCap => colors::SPECIAL,

            // Evidentiality markers
            Token::Bang | Token::Question | Token::Tilde | Token::Interrobang => colors::EVIDENCE,

            // Special symbols
            Token::Hourglass | Token::Circle | Token::Empty | Token::Infinity |
            Token::MiddleDot => colors::SPECIAL,

            // Strings and chars
            Token::StringLit(_) | Token::CharLit(_) => colors::STRING,

            // Numbers
            Token::IntLit(_) | Token::FloatLit(_) | Token::BinaryLit(_) |
            Token::OctalLit(_) | Token::HexLit(_) | Token::DuodecimalLit(_) |
            Token::SexagesimalLit(_) | Token::VigesimalLit(_) => colors::NUMBER,

            // Comments
            Token::LineComment(_) | Token::DocComment(_) => colors::COMMENT,

            // Operators
            Token::Plus | Token::Minus | Token::Star | Token::Slash | Token::Percent |
            Token::StarStar | Token::PlusPlus | Token::Eq | Token::EqEq | Token::NotEq |
            Token::Lt | Token::LtEq | Token::Gt | Token::GtEq | Token::AndAnd |
            Token::OrOr | Token::Amp | Token::Caret | Token::Shl | Token::Shr |
            Token::Pipe | Token::Arrow | Token::FatArrow | Token::ColonColon => colors::OPERATOR,

            // Identifiers - check for common stdlib functions
            Token::Ident(name) => {
                // Built-in functions get special coloring
                match name.as_str() {
                    "print" | "println" | "dbg" | "assert" | "panic" | "todo" |
                    "sqrt" | "pow" | "sin" | "cos" | "tan" | "abs" | "floor" | "ceil" |
                    "len" | "push" | "pop" | "sum" | "product" | "sort" | "reverse" |
                    "known" | "uncertain" | "reported" | "paradox" | "evidence_of" |
                    "upper" | "lower" | "trim" | "split" | "join" |
                    "random" | "shuffle" | "now" | "sleep" => colors::FUNCTION,
                    // Type names (capitalized)
                    _ if name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) => colors::TYPE,
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
            "fn", "let", "mut", "if", "else", "while", "for", "in", "match",
            "return", "break", "continue", "struct", "enum", "impl", "trait",
            "pub", "use", "mod", "const", "static", "type", "where", "async", "await",
            "actor", "handler", "receive", "send", "true", "false", "null",

            // Transform morphemes (Greek letters)
            "τ", "Τ", "φ", "Φ", "σ", "Σ", "ρ", "Ρ", "λ", "Λ", "Π",

            // Access morphemes
            "α", "ω", "Ω", "μ", "Μ", "χ", "Χ", "ν", "Ν", "ξ", "Ξ",

            // Other Greek letters
            "δ", "Δ", "ε", "ζ",

            // Aspect suffixes
            "·ing", "·ed", "·able", "·ive",

            // Logic operators (Unicode)
            "∧", "∨", "¬", "⊻", "⊤", "⊥",

            // Bitwise operators (Unicode)
            "⋏", "⋎",

            // Set operators
            "∪", "∩", "∖", "⊂", "⊆", "⊃", "⊇", "∈", "∉",

            // Math operators
            "∘", "⊗", "⊕", "∫", "∂", "√", "∛",

            // Data operations
            "⋈", "⋳", "⊔", "⊓",

            // Special literals
            "∅", "∞", "◯",

            // Quantifiers
            "∀", "∃",

            // Evidentiality markers
            "!", "?", "~", "‽",

            // Stdlib functions
            "print", "println", "dbg", "assert", "panic", "todo", "unreachable",
            "clone", "id", "default",
            "abs", "sqrt", "pow", "sin", "cos", "tan", "asin", "acos", "atan",
            "sinh", "cosh", "tanh", "exp", "ln", "log", "floor", "ceil", "round",
            "min", "max", "clamp", "sign", "gcd", "lcm", "is_prime", "fibonacci",
            "len", "push", "pop", "first", "last", "get", "set", "contains",
            "index_of", "reverse", "sort", "unique", "flatten", "zip", "enumerate",
            "chunks", "windows", "take", "skip", "concat",
            "chars", "bytes", "split", "join", "trim", "upper", "lower",
            "replace", "starts_with", "ends_with", "substring", "repeat",
            "known", "uncertain", "reported", "paradox", "evidence_of",
            "is_known", "strip_evidence", "trust", "verify",
            "sum", "product", "mean", "median", "min_of", "max_of", "any", "all", "none",
            "read_file", "write_file", "file_exists", "env", "cwd", "args",
            "now", "now_secs", "sleep",
            "random", "random_int", "shuffle", "sample",
            "to_string", "to_int", "to_float", "hex", "oct", "bin", "parse_int",
            "cycle", "mod_add", "mod_sub", "mod_mul", "mod_pow", "mod_inv",
            "octave", "interval", "cents", "freq", "midi",

            // New stdlib functions
            "middle", "choice", "nth", "next", "peek",
            "zip_with", "supremum", "infimum",

            // REPL commands
            ":help", ":ast", ":exit", ":quit", ":clear", ":type", ":symbols",
        ].into_iter().map(String::from).collect();

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

        let matches: Vec<Pair> = self.completions
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
        Cow::Owned(format!("{}{}λ>{} ", colors::BOLD, colors::MORPHEME, colors::RESET))
    }

    fn highlight_char(&self, _line: &str, _pos: usize, _forced: bool) -> bool {
        true // Always re-highlight
    }
}

impl Validator for SigilHelper {}

fn repl() -> ExitCode {
    println!("{}{}Sigil REPL v0.1.0{}", colors::BOLD, colors::MORPHEME, colors::RESET);
    println!("A polysynthetic language with evidentiality types.");
    println!();
    println!("{}Commands:{}", colors::DIM, colors::RESET);
    println!("  {}:help{}     Show help", colors::KEYWORD, colors::RESET);
    println!("  {}:ast{}      Toggle AST display", colors::KEYWORD, colors::RESET);
    println!("  {}:clear{}    Clear screen", colors::KEYWORD, colors::RESET);
    println!("  {}:exit{}     Exit REPL", colors::KEYWORD, colors::RESET);
    println!();
    println!("{}Press Tab for completions, ↑/↓ for history{}", colors::DIM, colors::RESET);
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
                        println!("AST display: {}{}{}",
                            colors::KEYWORD,
                            if show_ast { "on" } else { "off" },
                            colors::RESET);
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
    println!("{}{}Sigil REPL Commands:{}", colors::BOLD, colors::MORPHEME, colors::RESET);
    println!();
    println!("  {}:help{}      Show this help", colors::KEYWORD, colors::RESET);
    println!("  {}:symbols{}   Show all Unicode symbols", colors::KEYWORD, colors::RESET);
    println!("  {}:ast{}       Toggle AST display mode", colors::KEYWORD, colors::RESET);
    println!("  {}:clear{}     Clear the screen", colors::KEYWORD, colors::RESET);
    println!("  {}:exit{}      Exit the REPL", colors::KEYWORD, colors::RESET);
    println!();
    println!("{}{}Examples:{}", colors::BOLD, colors::TYPE, colors::RESET);
    println!();
    println!("  {}// Arithmetic{}", colors::COMMENT, colors::RESET);
    println!("  {}1{} + {}2{} * {}3{}",
        colors::NUMBER, colors::RESET, colors::NUMBER, colors::RESET, colors::NUMBER, colors::RESET);
    println!();
    println!("  {}// Variables{}", colors::COMMENT, colors::RESET);
    println!("  {}let{} x = {}42{};", colors::KEYWORD, colors::RESET, colors::NUMBER, colors::RESET);
    println!();
    println!("  {}// Pipe transforms (polysynthetic){}", colors::COMMENT, colors::RESET);
    println!("  [{}1{}, {}2{}, {}3{}]|{}τ{}{{_ * {}2{}}}  {}// [2, 4, 6]{}",
        colors::NUMBER, colors::RESET, colors::NUMBER, colors::RESET, colors::NUMBER, colors::RESET,
        colors::MORPHEME, colors::RESET, colors::NUMBER, colors::RESET,
        colors::COMMENT, colors::RESET);
    println!();
    println!("  {}// Evidentiality markers{}", colors::COMMENT, colors::RESET);
    println!("  {}known{}({}42{})    {}// Certain value (!){}",
        colors::FUNCTION, colors::RESET, colors::NUMBER, colors::RESET, colors::COMMENT, colors::RESET);
    println!("  {}uncertain{}(x) {}// Uncertain value (?){}",
        colors::FUNCTION, colors::RESET, colors::COMMENT, colors::RESET);
    println!();
    println!("  {}// Functions{}", colors::COMMENT, colors::RESET);
    println!("  {}fn{} add(a, b) {{ a + b }}", colors::KEYWORD, colors::RESET);
    println!();
    println!("{}{}Morphemes:{}", colors::BOLD, colors::MORPHEME, colors::RESET);
    println!("  {}τ{} transform  {}φ{} filter  {}σ{} sort  {}ρ{} reduce  {}λ{} lambda",
        colors::MORPHEME, colors::RESET, colors::MORPHEME, colors::RESET,
        colors::MORPHEME, colors::RESET, colors::MORPHEME, colors::RESET,
        colors::MORPHEME, colors::RESET);
    println!();
    println!("Type {}:symbols{} for a complete symbol reference.", colors::KEYWORD, colors::RESET);
    println!();
}

fn print_symbols() {
    println!("{}{}Sigil Unicode Symbols Reference{}", colors::BOLD, colors::MORPHEME, colors::RESET);
    println!();

    // Transform Morphemes
    println!("{}{}Transform Morphemes (Pipe Syntax: data|morpheme):{}", colors::BOLD, colors::MORPHEME, colors::RESET);
    println!("  {}τ{}/{}Τ{}  transform/map   data|τ{{_ * 2}}", colors::MORPHEME, colors::RESET, colors::MORPHEME, colors::RESET);
    println!("  {}φ{}/{}Φ{}  filter          data|φ{{_ > 0}}", colors::MORPHEME, colors::RESET, colors::MORPHEME, colors::RESET);
    println!("  {}σ{}     sort            data|σ", colors::MORPHEME, colors::RESET);
    println!("  {}Σ{}     sum             data|Σ", colors::MORPHEME, colors::RESET);
    println!("  {}ρ{}/{}Ρ{}  reduce/fold     data|ρ{{acc + _}}", colors::MORPHEME, colors::RESET, colors::MORPHEME, colors::RESET);
    println!("  {}λ{}/{}Λ{}  lambda          λ x -> x + 1", colors::MORPHEME, colors::RESET, colors::MORPHEME, colors::RESET);
    println!("  {}Π{}     product         data|Π", colors::MORPHEME, colors::RESET);
    println!();

    // Access Morphemes
    println!("{}{}Access Morphemes (Pipe Syntax: data|morpheme):{}", colors::BOLD, colors::MORPHEME, colors::RESET);
    println!("  {}α{}     first element   data|α", colors::MORPHEME, colors::RESET);
    println!("  {}ω{}/{}Ω{}  last element    data|ω", colors::MORPHEME, colors::RESET, colors::MORPHEME, colors::RESET);
    println!("  {}μ{}/{}Μ{}  middle/median   data|μ", colors::MORPHEME, colors::RESET, colors::MORPHEME, colors::RESET);
    println!("  {}χ{}/{}Χ{}  random choice   data|χ", colors::MORPHEME, colors::RESET, colors::MORPHEME, colors::RESET);
    println!("  {}ν{}/{}Ν{}  nth element     data|ν{{2}}", colors::MORPHEME, colors::RESET, colors::MORPHEME, colors::RESET);
    println!("  {}ξ{}/{}Ξ{}  next in iter    data|ξ", colors::MORPHEME, colors::RESET, colors::MORPHEME, colors::RESET);
    println!();

    // Evidentiality
    println!("{}{}Evidentiality Markers:{}", colors::BOLD, colors::EVIDENCE, colors::RESET);
    println!("  {}!{}  known/direct     let x{}!{} = verified();", colors::EVIDENCE, colors::RESET, colors::EVIDENCE, colors::RESET);
    println!("  {}?{}  uncertain        let x{}?{} = maybe_get();", colors::EVIDENCE, colors::RESET, colors::EVIDENCE, colors::RESET);
    println!("  {}~{}  reported         let x{}~{} = fetch_api();", colors::EVIDENCE, colors::RESET, colors::EVIDENCE, colors::RESET);
    println!("  {}‽{}  paradox          let x{}‽{} = contradict();", colors::EVIDENCE, colors::RESET, colors::EVIDENCE, colors::RESET);
    println!();

    // Logic Operators
    println!("{}{}Logic Operators:{}", colors::BOLD, colors::OPERATOR, colors::RESET);
    println!("  {}∧{}  AND (&&)        a ∧ b", colors::OPERATOR, colors::RESET);
    println!("  {}∨{}  OR (||)         a ∨ b", colors::OPERATOR, colors::RESET);
    println!("  {}¬{}  NOT (!)         ¬a", colors::OPERATOR, colors::RESET);
    println!("  {}⊻{}  XOR             a ⊻ b", colors::OPERATOR, colors::RESET);
    println!("  {}⊤{}  true/top", colors::OPERATOR, colors::RESET);
    println!("  {}⊥{}  false/bottom", colors::OPERATOR, colors::RESET);
    println!();

    // Bitwise Operators
    println!("{}{}Bitwise Operators:{}", colors::BOLD, colors::OPERATOR, colors::RESET);
    println!("  {}⋏{}  AND (&)         a ⋏ b", colors::OPERATOR, colors::RESET);
    println!("  {}⋎{}  OR              a ⋎ b", colors::OPERATOR, colors::RESET);
    println!();

    // Set Operators
    println!("{}{}Set Operators:{}", colors::BOLD, colors::OPERATOR, colors::RESET);
    println!("  {}∪{}  union           a ∪ b", colors::OPERATOR, colors::RESET);
    println!("  {}∩{}  intersection    a ∩ b", colors::OPERATOR, colors::RESET);
    println!("  {}∖{}  difference      a ∖ b", colors::OPERATOR, colors::RESET);
    println!("  {}⊂{}  proper subset   a ⊂ b", colors::OPERATOR, colors::RESET);
    println!("  {}⊆{}  subset/equal    a ⊆ b", colors::OPERATOR, colors::RESET);
    println!("  {}∈{}  element of      x ∈ set", colors::OPERATOR, colors::RESET);
    println!("  {}∉{}  not element of  x ∉ set", colors::OPERATOR, colors::RESET);
    println!();

    // Data Operations
    println!("{}{}Data Operations:{}", colors::BOLD, colors::SPECIAL, colors::RESET);
    println!("  {}⋈{}  zip with op     zip_with(a, b, \"add\")", colors::SPECIAL, colors::RESET);
    println!("  {}⋳{}  flatten         flatten(nested)", colors::SPECIAL, colors::RESET);
    println!("  {}⊔{}  supremum/max    supremum(a, b)", colors::SPECIAL, colors::RESET);
    println!("  {}⊓{}  infimum/min     infimum(a, b)", colors::SPECIAL, colors::RESET);
    println!();

    // Math Operations
    println!("{}{}Math Operations:{}", colors::BOLD, colors::OPERATOR, colors::RESET);
    println!("  {}∘{}  compose         f ∘ g", colors::OPERATOR, colors::RESET);
    println!("  {}⊗{}  tensor          a ⊗ b", colors::OPERATOR, colors::RESET);
    println!("  {}⊕{}  direct sum      a ⊕ b", colors::OPERATOR, colors::RESET);
    println!("  {}∫{}  integral/cumsum", colors::OPERATOR, colors::RESET);
    println!("  {}∂{}  partial/deriv", colors::OPERATOR, colors::RESET);
    println!("  {}√{}  sqrt            √x", colors::OPERATOR, colors::RESET);
    println!("  {}∛{}  cbrt            ∛x", colors::OPERATOR, colors::RESET);
    println!();

    // Special Literals
    println!("{}{}Special Literals:{}", colors::BOLD, colors::SPECIAL, colors::RESET);
    println!("  {}∅{}  empty/void      let x = ∅;", colors::SPECIAL, colors::RESET);
    println!("  {}∞{}  infinity        let x = ∞;", colors::SPECIAL, colors::RESET);
    println!("  {}◯{}  geometric zero", colors::SPECIAL, colors::RESET);
    println!();

    // Quantifiers
    println!("{}{}Quantifiers:{}", colors::BOLD, colors::OPERATOR, colors::RESET);
    println!("  {}∀{}  for all", colors::OPERATOR, colors::RESET);
    println!("  {}∃{}  exists", colors::OPERATOR, colors::RESET);
    println!();

    // Aspect Morphemes
    println!("{}{}Aspect Suffixes (Function naming):{}", colors::BOLD, colors::MORPHEME, colors::RESET);
    println!("  {}·ing{}  progressive   fn read·ing() -> Stream", colors::MORPHEME, colors::RESET);
    println!("  {}·ed{}   perfective    fn process·ed() -> Result", colors::MORPHEME, colors::RESET);
    println!("  {}·able{} potential     fn parse·able() -> Bool", colors::MORPHEME, colors::RESET);
    println!("  {}·ive{}  resultative   fn destruct·ive() -> Parts", colors::MORPHEME, colors::RESET);
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
                                let repl_fn = interpreter.globals.borrow()
                                    .get("__repl__")
                                    .and_then(|v| {
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
                                                println!("{}=> {}{}", colors::DIM, colors::RESET, value);
                                            }
                                        }
                                        Err(e) => eprintln!("{}Error: {}{}", colors::SPECIAL, e, colors::RESET),
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
