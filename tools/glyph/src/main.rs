//! Glyph - Code formatter for Sigil
//!
//! A simple token-based formatter that enforces consistent style:
//! - 4-space indentation
//! - Proper spacing around operators
//! - Consistent brace placement
//! - Line length limits

use clap::Parser as ClapParser;
use sigil_parser::{Lexer, Token};
use sigil_parser::span::Span;
use std::fs;
use std::io::{self, Read, Write};
use std::path::PathBuf;
use walkdir::WalkDir;

/// Glyph - Sigil Code Formatter
#[derive(ClapParser, Debug)]
#[command(name = "sigil-glyph")]
#[command(author = "Daemoniorum, Inc.")]
#[command(version = "0.1.0")]
#[command(about = "Format Sigil source code", long_about = None)]
struct Args {
    /// Files or directories to format
    #[arg(default_value = ".")]
    paths: Vec<PathBuf>,

    /// Check if files are formatted without modifying them
    #[arg(short, long)]
    check: bool,

    /// Print the diff instead of modifying files
    #[arg(short, long)]
    diff: bool,

    /// Maximum line width
    #[arg(long, default_value = "100")]
    max_width: usize,

    /// Number of spaces for indentation
    #[arg(long, default_value = "4")]
    indent: usize,

    /// Read from stdin and write to stdout
    #[arg(long)]
    stdin: bool,
}

/// Formatter configuration.
#[derive(Clone)]
struct FormatConfig {
    max_width: usize,
    indent_size: usize,
}

impl Default for FormatConfig {
    fn default() -> Self {
        Self {
            max_width: 100,
            indent_size: 4,
        }
    }
}

/// Token-based Sigil code formatter.
struct Formatter {
    config: FormatConfig,
    output: String,
    indent_level: i32,
    line_length: usize,
    last_token: Option<Token>,
    in_pipe_chain: bool,
    paren_depth: i32,
    brace_depth: i32,
    bracket_depth: i32,
}

impl Formatter {
    fn new(config: FormatConfig) -> Self {
        Self {
            config,
            output: String::new(),
            indent_level: 0,
            line_length: 0,
            last_token: None,
            in_pipe_chain: false,
            paren_depth: 0,
            brace_depth: 0,
            bracket_depth: 0,
        }
    }

    /// Format source code.
    fn format(&mut self, source: &str) -> String {
        self.output.clear();
        self.indent_level = 0;
        self.line_length = 0;
        self.last_token = None;
        self.in_pipe_chain = false;
        self.paren_depth = 0;
        self.brace_depth = 0;
        self.bracket_depth = 0;

        let mut lexer = Lexer::new(source);

        while let Some((token, span)) = lexer.next_token() {
            self.format_token(&token, source, span);
            self.last_token = Some(token);
        }

        // Ensure file ends with newline
        if !self.output.ends_with('\n') {
            self.output.push('\n');
        }

        std::mem::take(&mut self.output)
    }

    fn format_token(&mut self, token: &Token, _source: &str, _span: Span) {
        match token {
            // Comments - preserve as-is with proper spacing
            Token::LineComment(s) | Token::DocComment(s) => {
                self.ensure_space_before_if_needed();
                self.write(s);
                self.newline();
            }

            // Opening braces
            Token::LBrace => {
                self.ensure_space_before();
                self.write("{");
                self.newline();
                self.indent_level += 1;
                self.brace_depth += 1;
            }

            // Closing braces
            Token::RBrace => {
                self.indent_level -= 1;
                self.brace_depth -= 1;
                if !self.output.ends_with('\n') {
                    self.newline();
                }
                self.write_indent();
                self.write("}");
            }

            // Parentheses
            Token::LParen => {
                self.write("(");
                self.paren_depth += 1;
            }
            Token::RParen => {
                self.write(")");
                self.paren_depth -= 1;
            }

            // Brackets
            Token::LBracket => {
                self.write("[");
                self.bracket_depth += 1;
            }
            Token::RBracket => {
                self.write("]");
                self.bracket_depth -= 1;
            }

            // Semicolons
            Token::Semi => {
                self.write(";");
                // Newline after semicolon at statement level
                if self.paren_depth == 0 && self.bracket_depth == 0 {
                    self.newline();
                }
            }

            // Commas
            Token::Comma => {
                self.write(",");
                self.write(" ");
            }

            // Pipe operator - special handling for chains
            Token::Pipe => {
                // Break long pipe chains
                if self.in_pipe_chain && self.line_length > self.config.max_width / 2 {
                    self.newline();
                    self.write_indent();
                    self.write("    "); // Extra indent for continuation
                }
                self.write("|");
                self.in_pipe_chain = true;
            }

            // Binary operators - space around
            Token::Plus | Token::Minus | Token::Star | Token::Slash | Token::Percent |
            Token::EqEq | Token::NotEq | Token::Lt | Token::LtEq | Token::Gt | Token::GtEq |
            Token::AndAnd | Token::OrOr | Token::Eq | Token::PlusEq | Token::MinusEq |
            Token::StarEq | Token::SlashEq => {
                self.ensure_space_before();
                self.write(&self.token_to_string(token));
                self.write(" ");
            }

            // Arrow operators
            Token::Arrow => {
                self.ensure_space_before();
                self.write("->");
                self.write(" ");
            }
            Token::FatArrow => {
                self.ensure_space_before();
                self.write("=>");
                self.write(" ");
            }

            // Morphemes - tight spacing
            Token::Tau => self.write("τ"),
            Token::Phi => self.write("φ"),
            Token::Sigma => self.write("σ"),
            Token::Rho => self.write("ρ"),
            Token::Lambda => {
                self.ensure_space_before();
                self.write("λ");
                self.write(" ");
            }
            Token::Pi => self.write("Π"),
            Token::Hourglass => self.write("⌛"),

            // Evidentiality markers - no space before
            Token::Bang | Token::Question | Token::Tilde | Token::Interrobang => {
                self.write(&self.token_to_string(token));
            }

            // Middle dot (incorporation)
            Token::MiddleDot => {
                self.write("·");
            }

            // Keywords that start blocks
            Token::Fn | Token::Async | Token::Struct | Token::Enum | Token::Trait |
            Token::Impl | Token::Mod | Token::Actor => {
                if !self.output.ends_with('\n') && !self.output.is_empty() {
                    // Blank line before top-level items
                    if self.brace_depth == 0 {
                        self.newline();
                    }
                }
                self.write_indent();
                self.write(&self.token_to_string(token));
                self.write(" ");
                self.in_pipe_chain = false;
            }

            // Control flow keywords
            Token::If | Token::Else | Token::Match | Token::While | Token::For |
            Token::Loop | Token::Return | Token::Break | Token::Continue => {
                self.write_indent_if_line_start();
                self.write(&self.token_to_string(token));
                if !matches!(token, Token::Else) || !matches!(self.last_token.as_ref(), Some(Token::RBrace)) {
                    self.write(" ");
                }
                self.in_pipe_chain = false;
            }

            // Other keywords
            Token::Let | Token::Mut | Token::Const | Token::Static | Token::Pub |
            Token::Use | Token::Type | Token::Where | Token::In | Token::As => {
                self.write_indent_if_line_start();
                self.write(&self.token_to_string(token));
                self.write(" ");
            }

            // Identifiers
            Token::Ident(s) => {
                self.space_before_ident_if_needed();
                self.write(s);
            }

            // Literals
            Token::IntLit(n) => {
                self.space_before_ident_if_needed();
                self.write(&n.to_string());
            }
            Token::FloatLit(f) => {
                self.space_before_ident_if_needed();
                self.write(&f.to_string());
            }
            Token::StringLit(s) => {
                self.space_before_ident_if_needed();
                self.write("\"");
                self.write(s);
                self.write("\"");
            }
            Token::CharLit(c) => {
                self.space_before_ident_if_needed();
                self.write("'");
                self.write(&c.to_string());
                self.write("'");
            }
            Token::True => {
                self.space_before_ident_if_needed();
                self.write("true");
            }
            Token::False => {
                self.space_before_ident_if_needed();
                self.write("false");
            }

            // Colon
            Token::Colon => {
                self.write(":");
                self.write(" ");
            }

            // Double colon
            Token::ColonColon => {
                self.write("::");
            }

            // Dot
            Token::Dot => {
                self.write(".");
            }

            // Range operators
            Token::DotDot => {
                self.write("..");
            }
            Token::DotDotEq => {
                self.write("..=");
            }

            // Other tokens - default handling
            _ => {
                let text = self.token_to_string(token);
                if !text.is_empty() {
                    self.write(&text);
                }
            }
        }
    }

    fn token_to_string(&self, token: &Token) -> String {
        match token {
            Token::Fn => "fn".to_string(),
            Token::Async => "async".to_string(),
            Token::Let => "let".to_string(),
            Token::Mut => "mut".to_string(),
            Token::Const => "const".to_string(),
            Token::Type => "type".to_string(),
            Token::Struct => "struct".to_string(),
            Token::Enum => "enum".to_string(),
            Token::Trait => "trait".to_string(),
            Token::Impl => "impl".to_string(),
            Token::Mod => "mod".to_string(),
            Token::Use => "use".to_string(),
            Token::Pub => "pub".to_string(),
            Token::Actor => "actor".to_string(),
            Token::If => "if".to_string(),
            Token::Else => "else".to_string(),
            Token::Match => "match".to_string(),
            Token::Loop => "loop".to_string(),
            Token::While => "while".to_string(),
            Token::For => "for".to_string(),
            Token::In => "in".to_string(),
            Token::Break => "break".to_string(),
            Token::Continue => "continue".to_string(),
            Token::Return => "return".to_string(),
            Token::Where => "where".to_string(),
            Token::As => "as".to_string(),
            Token::Static => "static".to_string(),
            Token::Plus => "+".to_string(),
            Token::Minus => "-".to_string(),
            Token::Star => "*".to_string(),
            Token::Slash => "/".to_string(),
            Token::Percent => "%".to_string(),
            Token::Eq => "=".to_string(),
            Token::EqEq => "==".to_string(),
            Token::NotEq => "!=".to_string(),
            Token::Lt => "<".to_string(),
            Token::LtEq => "<=".to_string(),
            Token::Gt => ">".to_string(),
            Token::GtEq => ">=".to_string(),
            Token::AndAnd => "&&".to_string(),
            Token::OrOr => "||".to_string(),
            Token::Bang => "!".to_string(),
            Token::Question => "?".to_string(),
            Token::Tilde => "~".to_string(),
            Token::Interrobang => "‽".to_string(),
            Token::PlusEq => "+=".to_string(),
            Token::MinusEq => "-=".to_string(),
            Token::StarEq => "*=".to_string(),
            Token::SlashEq => "/=".to_string(),
            Token::Ident(s) => s.clone(),
            Token::IntLit(n) => n.to_string(),
            Token::FloatLit(f) => f.to_string(),
            Token::True => "true".to_string(),
            Token::False => "false".to_string(),
            _ => String::new(),
        }
    }

    fn write(&mut self, s: &str) {
        self.output.push_str(s);
        self.line_length += s.len();
    }

    fn write_indent(&mut self) {
        let indent_str = " ".repeat(self.config.indent_size);
        let indent = indent_str.repeat(self.indent_level.max(0) as usize);
        self.output.push_str(&indent);
        self.line_length = indent.len();
    }

    fn write_indent_if_line_start(&mut self) {
        if self.output.ends_with('\n') || self.output.is_empty() {
            self.write_indent();
        }
    }

    fn newline(&mut self) {
        // Remove trailing whitespace
        while self.output.ends_with(' ') || self.output.ends_with('\t') {
            self.output.pop();
        }
        self.output.push('\n');
        self.line_length = 0;
        self.in_pipe_chain = false;
    }

    fn ensure_space_before(&mut self) {
        if !self.output.is_empty()
            && !self.output.ends_with(' ')
            && !self.output.ends_with('\n')
            && !self.output.ends_with('(')
            && !self.output.ends_with('[')
            && !self.output.ends_with('{')
        {
            self.write(" ");
        }
    }

    fn ensure_space_before_if_needed(&mut self) {
        if !self.output.is_empty() && !self.output.ends_with('\n') {
            if !self.output.ends_with(' ') {
                self.write(" ");
            }
        } else if self.output.ends_with('\n') {
            self.write_indent();
        }
    }

    fn space_before_ident_if_needed(&mut self) {
        if let Some(ref last) = self.last_token {
            match last {
                Token::Let | Token::Mut | Token::Fn | Token::Async |
                Token::Struct | Token::Enum | Token::Trait | Token::Impl |
                Token::Type | Token::Const | Token::Static | Token::Pub |
                Token::Mod | Token::Use | Token::If | Token::Else |
                Token::While | Token::For | Token::Match | Token::Return |
                Token::In | Token::As | Token::Where | Token::Colon |
                Token::Arrow | Token::FatArrow | Token::Eq |
                Token::Plus | Token::Minus | Token::Star | Token::Slash |
                Token::EqEq | Token::NotEq | Token::Lt | Token::LtEq |
                Token::Gt | Token::GtEq | Token::AndAnd | Token::OrOr => {
                    // Space already added
                }
                Token::Comma => {
                    // Space already added
                }
                Token::LParen | Token::LBracket | Token::LBrace |
                Token::Pipe | Token::MiddleDot | Token::Dot |
                Token::Bang | Token::Question | Token::Tilde | Token::Interrobang |
                Token::Tau | Token::Phi | Token::Sigma | Token::Rho => {
                    // No space needed
                }
                _ => {
                    if !self.output.ends_with(' ') && !self.output.ends_with('\n') && !self.output.is_empty() {
                        self.write(" ");
                    }
                }
            }
        }
        self.write_indent_if_line_start();
    }
}

fn format_file(path: &PathBuf, config: &FormatConfig, check: bool, diff: bool) -> io::Result<bool> {
    let content = fs::read_to_string(path)?;
    let mut formatter = Formatter::new(config.clone());
    let formatted = formatter.format(&content);

    if content == formatted {
        return Ok(true); // Already formatted
    }

    if check {
        eprintln!("Would reformat: {}", path.display());
        return Ok(false);
    }

    if diff {
        eprintln!("--- {}", path.display());
        eprintln!("+++ {} (formatted)", path.display());
        // Simplified: just show that files differ
        eprintln!("(files differ)");
        return Ok(false);
    }

    fs::write(path, &formatted)?;
    eprintln!("Reformatted: {}", path.display());
    Ok(true)
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    let config = FormatConfig {
        max_width: args.max_width,
        indent_size: args.indent,
    };

    if args.stdin {
        let mut content = String::new();
        io::stdin().read_to_string(&mut content)?;

        let mut formatter = Formatter::new(config);
        let formatted = formatter.format(&content);
        io::stdout().write_all(formatted.as_bytes())?;
        return Ok(());
    }

    let mut all_ok = true;
    let mut file_count = 0;

    for path in &args.paths {
        if path.is_file() {
            if path.extension().map(|e| e == "sigil" || e == "sg").unwrap_or(false) {
                all_ok &= format_file(path, &config, args.check, args.diff)?;
                file_count += 1;
            }
        } else if path.is_dir() {
            for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
                let p = entry.path();
                if p.is_file() && p.extension().map(|e| e == "sigil" || e == "sg").unwrap_or(false) {
                    all_ok &= format_file(&p.to_path_buf(), &config, args.check, args.diff)?;
                    file_count += 1;
                }
            }
        }
    }

    if file_count == 0 {
        eprintln!("No Sigil files found");
    } else {
        eprintln!("Checked {} file(s)", file_count);
        if args.check && !all_ok {
            std::process::exit(1);
        }
    }

    Ok(())
}
