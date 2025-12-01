//! Sigil CLI - Parse and check Sigil source files.

use sigil_parser::Parser;
use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Sigil Parser v0.1.0");
        eprintln!("Usage: sigil <command> [file.sigil]");
        eprintln!();
        eprintln!("Commands:");
        eprintln!("  parse <file>    Parse and check a Sigil file");
        eprintln!("  lex <file>      Tokenize a Sigil file");
        eprintln!("  repl            Start interactive REPL");
        return ExitCode::from(1);
    }

    match args[1].as_str() {
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
                parse_file(&args[1])
            } else {
                eprintln!("Unknown command: {}", args[1]);
                ExitCode::from(1)
            }
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
    }
}

fn lex_file(path: &str) -> ExitCode {
    use sigil_parser::Lexer;

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

fn repl() -> ExitCode {
    use std::io::{self, Write};

    println!("Sigil REPL v0.1.0");
    println!("Type expressions or statements. Type 'exit' to quit.");
    println!();

    loop {
        print!("sigil> ");
        io::stdout().flush().unwrap();

        let mut input = String::new();
        match io::stdin().read_line(&mut input) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                continue;
            }
        }

        let input = input.trim();
        if input.is_empty() {
            continue;
        }
        if input == "exit" || input == "quit" {
            break;
        }

        // Try to parse as expression wrapped in a function
        let wrapped = format!("fn __repl__() {{ {} }}", input);
        let mut parser = Parser::new(&wrapped);
        match parser.parse_file() {
            Ok(ast) => {
                println!("Parsed successfully");
                if let Some(item) = ast.items.first() {
                    println!("{:#?}", item.node);
                }
            }
            Err(e) => {
                eprintln!("Error: {}", e);
            }
        }
    }

    println!("Goodbye!");
    ExitCode::SUCCESS
}
