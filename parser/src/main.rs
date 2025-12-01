//! Sigil CLI - Parse, check, and run Sigil source files.

use sigil_parser::{Parser, Interpreter, register_stdlib, Lexer, Token};
#[cfg(feature = "jit")]
use sigil_parser::JitCompiler;
#[cfg(feature = "llvm")]
use sigil_parser::{LlvmCompiler, CompileMode, OptLevel};
use std::env;
use std::fs;
use std::process::ExitCode;
use std::borrow::Cow;

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
        eprintln!("Usage: sigil <command> [file.sigil]");
        eprintln!();
        eprintln!("Commands:");
        eprintln!("  run <file>      Execute a Sigil file (interpreted)");
        eprintln!("  jit <file>      Execute a Sigil file (JIT compiled, fast)");
        eprintln!("  llvm <file>     Execute a Sigil file (LLVM backend, fastest)");
        eprintln!("  compile <file>  Compile to native executable (AOT)");
        eprintln!("  parse <file>    Parse and check a Sigil file");
        eprintln!("  lex <file>      Tokenize a Sigil file");
        eprintln!("  repl            Start interactive REPL");
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
                eprintln!("Usage: sigil compile <file.sigil> [-o output]");
                return ExitCode::from(1);
            }
            let output = if args.len() >= 5 && args[3] == "-o" {
                args[4].clone()
            } else {
                // Default output name: strip extension and add nothing (executable)
                args[2].trim_end_matches(".sigil").trim_end_matches(".sg").to_string()
            };
            compile_file(&args[2], &output)
        }
        #[cfg(not(feature = "llvm"))]
        "compile" => {
            eprintln!("Error: AOT compilation requires LLVM (compile with --features llvm)");
            ExitCode::from(1)
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
fn compile_file(path: &str, output: &str) -> ExitCode {
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

    // Find the runtime directory (relative to the binary or in known locations)
    let runtime_c = find_runtime();
    if runtime_c.is_none() {
        eprintln!("Error: Could not find sigil_runtime.c");
        eprintln!("Expected locations:");
        eprintln!("  - ./runtime/sigil_runtime.c");
        eprintln!("  - ../runtime/sigil_runtime.c");
        return ExitCode::from(1);
    }
    let runtime_c = runtime_c.unwrap();

    // Link with clang/gcc
    println!("Compiling {} -> {}", path, output);
    let linker = find_linker();
    let link_result = Command::new(&linker)
        .args([
            "-O3",
            &obj_path,
            &runtime_c,
            "-o", output,
            "-lm",  // Link math library
        ])
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
fn find_runtime() -> Option<String> {
    let candidates = [
        "runtime/sigil_runtime.c",
        "../runtime/sigil_runtime.c",
        "sigil/parser/runtime/sigil_runtime.c",
    ];

    for candidate in candidates {
        if std::path::Path::new(candidate).exists() {
            return Some(candidate.to_string());
        }
    }

    // Try relative to executable
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let path = dir.join("runtime/sigil_runtime.c");
            if path.exists() {
                return path.to_string_lossy().into_owned().into();
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

            // Morphemes (polysynthetic operators)
            Token::Tau | Token::Phi | Token::Sigma | Token::Rho | Token::Lambda |
            Token::Pi => colors::MORPHEME,

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
            "actor", "handler", "receive", "send", "true", "false",
            // Morphemes
            "τ", "φ", "σ", "ρ", "λ", "Σ", "Π",
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
            // REPL commands
            ":help", ":ast", ":exit", ":quit", ":clear", ":type",
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
