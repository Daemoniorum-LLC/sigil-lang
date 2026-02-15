//! Lexer for the Sigil programming language.
//!
//! Handles polysynthetic morphemes, evidentiality markers, and multi-base numerals.

use crate::span::Span;
use logos::Logos;

/// Process escape sequences in a string literal.
/// Converts \n, \t, \r, \\, \", \', \0, \xNN, \u{NNNN} to their actual characters.
/// Also handles line continuation: \<newline><whitespace> is stripped entirely.
fn process_escape_sequences(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('\n') => {
                    // Line continuation: skip newline and any leading whitespace
                    while let Some(&c) = chars.peek() {
                        if c == ' ' || c == '\t' {
                            chars.next();
                        } else {
                            break;
                        }
                    }
                }
                Some('n') => result.push('\n'),
                Some('t') => result.push('\t'),
                Some('r') => result.push('\r'),
                Some('\\') => result.push('\\'),
                Some('"') => result.push('"'),
                Some('\'') => result.push('\''),
                Some('0') => result.push('\0'),
                Some('x') => {
                    // \xNN - two hex digits
                    let mut hex = String::new();
                    for _ in 0..2 {
                        if let Some(&c) = chars.peek() {
                            if c.is_ascii_hexdigit() {
                                hex.push(chars.next().unwrap());
                            }
                        }
                    }
                    if let Ok(val) = u8::from_str_radix(&hex, 16) {
                        result.push(val as char);
                    }
                }
                Some('u') => {
                    // \u{NNNN} - Unicode code point
                    if chars.peek() == Some(&'{') {
                        chars.next(); // consume '{'
                        let mut hex = String::new();
                        while let Some(&c) = chars.peek() {
                            if c == '}' {
                                chars.next();
                                break;
                            }
                            if c.is_ascii_hexdigit() {
                                hex.push(chars.next().unwrap());
                            } else {
                                break;
                            }
                        }
                        if let Ok(val) = u32::from_str_radix(&hex, 16) {
                            if let Some(c) = char::from_u32(val) {
                                result.push(c);
                            }
                        }
                    }
                }
                Some(other) => {
                    // Unknown escape, keep as-is
                    result.push('\\');
                    result.push(other);
                }
                None => result.push('\\'),
            }
        } else {
            result.push(c);
        }
    }
    result
}

/// Process escape sequences in byte string literals, returning bytes.
fn process_byte_escape_sequences(s: &str) -> Vec<u8> {
    let mut result = Vec::with_capacity(s.len());
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('\n') => {
                    // Line continuation: skip newline and any leading whitespace
                    while let Some(&c) = chars.peek() {
                        if c == ' ' || c == '\t' {
                            chars.next();
                        } else {
                            break;
                        }
                    }
                }
                Some('n') => result.push(b'\n'),
                Some('t') => result.push(b'\t'),
                Some('r') => result.push(b'\r'),
                Some('\\') => result.push(b'\\'),
                Some('"') => result.push(b'"'),
                Some('\'') => result.push(b'\''),
                Some('0') => result.push(0),
                Some('x') => {
                    // \xNN - two hex digits
                    let mut hex = String::new();
                    for _ in 0..2 {
                        if let Some(&c) = chars.peek() {
                            if c.is_ascii_hexdigit() {
                                hex.push(chars.next().unwrap());
                            }
                        }
                    }
                    if let Ok(val) = u8::from_str_radix(&hex, 16) {
                        result.push(val);
                    }
                }
                Some(other) => {
                    // Unknown escape, keep as-is
                    result.push(b'\\');
                    if other.is_ascii() {
                        result.push(other as u8);
                    }
                }
                None => result.push(b'\\'),
            }
        } else if c.is_ascii() {
            result.push(c as u8);
        }
        // Non-ASCII in byte strings is ignored (Rust doesn't allow it)
    }
    result
}

/// Callback for delimited raw strings (r#"..."#).
/// Reads until the closing "# is found.
/// Callback for block comments: /* ... */
/// Consumes characters until */ is found
fn block_comment_callback(lex: &mut logos::Lexer<'_, Token>) -> Option<String> {
    let remainder = lex.remainder();

    // Find the closing */
    if let Some(end_pos) = remainder.find("*/") {
        let content = &remainder[..end_pos];
        // Bump past content and closing */ (2 chars)
        lex.bump(end_pos + 2);
        Some(content.to_string())
    } else {
        // No closing */ found - consume rest as comment
        let len = remainder.len();
        lex.bump(len);
        Some(remainder.to_string())
    }
}

fn raw_string_delimited_callback(lex: &mut logos::Lexer<'_, Token>) -> Option<String> {
    let remainder = lex.remainder();

    // Find the closing "#
    if let Some(end_pos) = remainder.find("\"#") {
        let content = &remainder[..end_pos];
        // Bump past content and closing "# (2 chars)
        lex.bump(end_pos + 2);
        Some(content.to_string())
    } else {
        None
    }
}

/// Callback for multi-line string literals.
/// Reads from """ until the next """ is found.
fn multiline_string_callback(lex: &mut logos::Lexer<'_, Token>) -> Option<String> {
    let remainder = lex.remainder();

    // Find the closing """
    if let Some(end_pos) = remainder.find("\"\"\"") {
        let content = &remainder[..end_pos];
        // Bump the lexer past the content and closing quotes
        lex.bump(end_pos + 3);
        Some(process_escape_sequences(content))
    } else {
        // No closing """ found - skip to end and return what we have
        None
    }
}

/// Process escape sequences in a character literal.
fn process_char_escape(s: &str) -> char {
    let mut chars = s.chars();
    match chars.next() {
        Some('\\') => match chars.next() {
            Some('n') => '\n',
            Some('t') => '\t',
            Some('r') => '\r',
            Some('\\') => '\\',
            Some('"') => '"',
            Some('\'') => '\'',
            Some('0') => '\0',
            Some('x') => {
                let hex: String = chars.take(2).collect();
                u8::from_str_radix(&hex, 16)
                    .map(|v| v as char)
                    .unwrap_or('?')
            }
            Some('u') => {
                if chars.next() == Some('{') {
                    let hex: String = chars.take_while(|&c| c != '}').collect();
                    u32::from_str_radix(&hex, 16)
                        .ok()
                        .and_then(char::from_u32)
                        .unwrap_or('?')
                } else {
                    '?'
                }
            }
            Some(c) => c,
            None => '?',
        },
        Some(c) => c,
        None => '?',
    }
}

/// Process escape sequences in a byte character literal (b'x').
fn process_byte_char_escape(s: &str) -> u8 {
    let mut chars = s.chars();
    match chars.next() {
        Some('\\') => match chars.next() {
            Some('n') => b'\n',
            Some('t') => b'\t',
            Some('r') => b'\r',
            Some('\\') => b'\\',
            Some('"') => b'"',
            Some('\'') => b'\'',
            Some('0') => b'\0',
            Some('x') => {
                let hex: String = chars.take(2).collect();
                u8::from_str_radix(&hex, 16).unwrap_or(b'?')
            }
            Some(c) => c as u8,
            None => b'?',
        },
        Some(c) => c as u8,
        None => b'?',
    }
}

/// Token types for Sigil.
#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\r\n\f]+")]
pub enum Token {
    // === Evidential Doc Comments (SGDOC) ===
    // These must come before LineComment to match first
    // Format: //<marker> <content> or //<marker><marker> <content> for inner docs
    // Inner docs (//!!, //~~, etc.) have priority 10 to match before outer docs

    /// Verified doc comment: //! content (backed by test)
    /// Inner variant: //!! documents the containing module
    #[regex(r"//!![^\n]*", priority = 10, callback = |lex| lex.slice()[4..].trim().to_string())]
    DocCommentVerifiedInner(String),
    #[regex(r"//![^!\n][^\n]*", priority = 8, callback = |lex| lex.slice()[3..].trim().to_string())]
    DocCommentVerified(String),

    /// Reported doc comment: //~ content (from spec/author, default)
    /// Inner variant: //~~ documents the containing module
    #[regex(r"//~~[^\n]*", priority = 10, callback = |lex| lex.slice()[4..].trim().to_string())]
    DocCommentReportedInner(String),
    #[regex(r"//~[^~\n][^\n]*", priority = 8, callback = |lex| lex.slice()[3..].trim().to_string())]
    DocCommentReported(String),

    /// Uncertain doc comment: //? content (needs investigation)
    /// Inner variant: //?? documents the containing module
    #[regex(r"//\?\?[^\n]*", priority = 10, callback = |lex| lex.slice()[4..].trim().to_string())]
    DocCommentUncertainInner(String),
    #[regex(r"//\?[^\?\n][^\n]*", priority = 8, callback = |lex| lex.slice()[3..].trim().to_string())]
    DocCommentUncertain(String),

    /// Predicted doc comment: //◊ content (planned feature)
    /// Inner variant: //◊◊ documents the containing module
    #[regex(r"//◊◊[^\n]*", priority = 10, callback = |lex| {
        // ◊ is multi-byte UTF-8, use trim_start_matches
        lex.slice().trim_start_matches("//◊◊").trim().to_string()
    })]
    DocCommentPredictedInner(String),
    #[regex(r"//◊[^◊\n][^\n]*", priority = 8, callback = |lex| {
        lex.slice().trim_start_matches("//◊").trim().to_string()
    })]
    DocCommentPredicted(String),

    /// Paradox doc comment: //‽ content (known inconsistency)
    /// Inner variant: //‽‽ documents the containing module
    #[regex(r"//‽‽[^\n]*", priority = 10, callback = |lex| {
        // ‽ is multi-byte UTF-8, use trim_start_matches
        lex.slice().trim_start_matches("//‽‽").trim().to_string()
    })]
    DocCommentParadoxInner(String),
    #[regex(r"//‽[^‽\n][^\n]*", priority = 8, callback = |lex| {
        lex.slice().trim_start_matches("//‽").trim().to_string()
    })]
    DocCommentParadox(String),

    // === Regular Comments ===
    // Legacy doc comment (maps to Reported for backwards compat)
    // Must have priority > LineComment to match /// before //
    #[regex(r"///[^\n]*", priority = 5, callback = |lex| lex.slice()[3..].trim().to_string())]
    DocComment(String),

    // Standard line comment (must be lower priority than doc comments)
    #[regex(r"//[^\n]*", priority = 1, callback = |lex| lex.slice().to_string())]
    LineComment(String),

    // Tilde comment style: ~~ ... ~~
    #[regex(r"~~[^\n]*", |lex| lex.slice().to_string())]
    TildeComment(String),

    // Block comment: /* ... */ (non-nested)
    #[token("/*", block_comment_callback)]
    BlockComment(String),

    // === Deprecated Rust Syntax ===
    // These tokens capture Rust-like syntax for helpful error messages
    // The parser maps these to Sigil equivalents in error messages
    // Deprecated Rust keywords - emit as deprecated for error messaging
    // Note: All common Rust keywords are now handled as aliases for backwards compatibility.
    // This variant is kept for future deprecation warnings if needed.
    DeprecatedRustKeyword(String),

    // Rust mutable reference &mut - use &Δ in Sigil
    #[token("&mut")]
    DeprecatedAmpMut,

    // === Keywords (Native Sigil Syntax) ===
    #[token("rite")]  // Rite - named function declaration
    Fn,
    #[token("async")]
    #[token("⌛")]  // Hourglass - time/waiting (native symbol alternative)
    Async,
    #[token("≔")]  // Definition assignment
    Let,
    #[token("Δ")]  // Delta - change/mutable
    #[token("vary")]  // Vary - mutable/changing (Sigil prose alternative)
    Mut,
    #[token("const")]
    #[token("◆")]  // Diamond - solid, fixed (native symbol alternative)
    Const,
    #[token("linear")]
    Linear,
    #[token("affine")]
    Affine,
    #[token("relevant")]
    Relevant,
    #[token("type")]
    Type,
    #[token("sigil")]
    #[token("Σ")]
    Struct,
    #[token("ᛈ")]  // Perthro rune - lot cup, choices/fate
    Enum,
    #[token("Θ")]  // Theta - theory, aspect
    #[token("aspect")]  // Aspect - trait/interface (Sigil prose alternative)
    Trait,
    #[token("⊢")]  // Turnstile - proves/provides
    Impl,
    #[token("scroll")]
    Mod,
    #[token("invoke")]
    Use,
    #[token("☉")]  // Sun - visible, public
    Pub,
    #[token("actor")]
    Actor,
    #[token("saga")]
    Saga,
    #[token("scope")]
    Scope,
    #[token("rune")]
    Rune,
    #[token("macro")]
    Macro,
    #[token("macro_rules")]
    MacroRules,

    // Control flow (Native Sigil Syntax)
    // Note: ∀ (ForAll token) is used contextually as `for` by parser
    // Note: ∈ (ElementOf token) is used contextually as `in` by parser
    // Note: ⊗ (Tensor token) is used contextually as `break` by parser
    // Note: ↻ (CycleArrow token) is used contextually as `continue` by parser
    // Note: ∞ (Infinity token) is used contextually as `loop` by parser
    #[token("⎇")]  // ISO branch symbol
    If,
    #[token("⎉")]  // ISO alternative symbol
    Else,
    #[token("⌥")]  // Option key - choices
    Match,
    #[token("loop")]
    #[token("forever")]  // Prose alternative for loop
    Loop, // Legacy - parser also handles ∞ (Infinity token) for loop
    #[token("⟳")]  // Cycle arrow
    While,
    For,
    In,
    Break,
    Continue,
    #[token("⤺")]  // Return arrow (U+2940 anticlockwise closed circle arrow)
    #[token("↩")]  // Alternative return arrow (U+21A9 leftwards arrow with hook)
    Return,
    #[token("yield")]
    Yield,
    #[token("await")]
    Await,

    // Other keywords
    #[token("self")]
    #[token("this")]  // This - current instance (Sigil prose alternative)
    SelfLower,
    // Note: ⊙ kept for Hadamard product - self uses keyword only
    #[token("Self")]
    #[token("This")]  // This - current type (Sigil prose alternative)
    SelfUpper,
    #[token("super")]
    Super,
    #[token("tome")]
    Crate,
    #[token("where")]
    #[token("∋")]  // Contains as member - native where clause
    Where,
    #[token("as")]
    As,
    #[token("dyn")]
    Dyn,
    #[token("move")]
    Move,
    #[token("ref")]
    Ref,
    #[token("static")]
    Static,
    #[token("unsafe")]
    Unsafe,
    #[token("no_grad")]
    NoGrad,
    #[token("extern")]
    Extern,
    #[token("asm")]
    Asm,
    #[token("volatile")]
    Volatile,
    #[token("naked")]
    Naked,
    #[token("packed")]
    Packed,
    #[token("simd")]
    Simd,
    #[token("atomic")]
    Atomic,
    #[token("derive")]
    Derive,
    #[token("on")]
    On,

    // Plurality keywords (DAEMONIORUM extensions)
    #[token("alter")]
    Alter,
    #[token("switch")]
    Switch,
    #[token("headspace")]
    Headspace,
    #[token("cocon")]
    CoCon,
    #[token("reality")]
    Reality,
    #[token("split")]
    Split,
    #[token("trigger")]
    Trigger,
    #[token("layer")]
    Layer,
    #[token("location")]
    Location,
    #[token("states")]
    States,
    #[token("anima")]
    Anima,
    #[token("to")]
    To,
    #[token("from")]
    From,

    // Alter-source markers (compound tokens)
    #[token("@!")]
    AlterSourceFronting,
    #[token("@~")]
    AlterSourceCoCon,
    #[token("@?")]
    AlterSourceDormant,
    #[token("@‽")]
    AlterSourceBlended,

    // Boolean literals
    #[token("true")]
    #[token("yay")]
    #[token("yea")]
    True,
    #[token("false")]
    #[token("nay")]
    False,

    // Null literal
    #[token("null")]
    Null,

    // === Morphemes (Greek letters) ===
    #[token("τ")]
    #[token("Τ")]
    Tau, // Transform/map

    #[token("φ")]
    #[token("Φ")]
    Phi, // Filter

    #[token("σ")]
    Sigma, // Sort morpheme (lowercase only - uppercase Σ is struct keyword)

    #[token("ρ")]
    #[token("Ρ")]
    Rho, // Reduce

    #[token("Λ")]
    Lambda, // Lambda morpheme (uppercase Λ only - used in pipe morpheme contexts)

    #[token("λ")]
    LambdaExpr, // Lambda closure expression: λ(params) [→ RetType] { body }

    #[token("Π")]
    Pi, // Product

    // Note: ⌛ (hourglass) is now mapped to Async keyword

    // Additional morphemes
    #[token("δ")]
    Delta, // Difference/change morpheme (lowercase only - uppercase Δ is mut keyword)

    #[token("ε")]
    Epsilon, // Empty/null

    #[token("ω")]
    #[token("Ω")]
    Omega, // End/terminal

    #[token("α")]
    Alpha, // First element

    #[token("ζ")]
    Zeta, // Zip/combine

    // === Additional Access Morphemes ===
    #[token("μ")]
    #[token("Μ")]
    Mu, // Middle/median element

    #[token("χ")]
    #[token("Χ")]
    Chi, // Random/choice (from chaos)

    #[token("ν")]
    #[token("Ν")]
    Nu, // Nth element (ordinal)

    #[token("ξ")]
    #[token("Ξ")]
    Xi, // Next in sequence

    #[token("ψ")]
    #[token("Ψ")]
    Psi, // Psychological/mental state

    #[token("θ")]
    Theta, // Threshold/angle morpheme (lowercase only - uppercase Θ is trait keyword)

    #[token("κ")]
    #[token("Κ")]
    Kappa, // Callback/continuation

    // === Parallel/Concurrency Morphemes ===
    #[token("∥")]
    #[token("parallel")]
    Parallel, // Parallel execution (U+2225)

    #[token("gpu")]
    Gpu, // GPU compute shader

    #[token("⊛")]
    Convolve, // Convolution/merge operator (U+229B - circled asterisk)

    // === Quantifiers (for AI-native set operations) ===
    #[token("∀")]
    #[token("each")]  // Prose alternative for impl blocks: ⊢ Trait each Type
    ForAll, // Universal quantification (parser handles contextual use as `for` keyword)

    #[token("∃")]
    Exists, // Existential quantification

    #[token("∈")]
    #[token("of")]   // Prose alternative: each x of iter { }
    ElementOf, // Membership test (parser handles contextual use as `in` keyword)

    #[token("∉")]
    NotElementOf, // Non-membership

    // === Set Operations ===
    #[token("∪")]
    Union, // Set union

    #[token("∩")]
    Intersection, // Set intersection

    #[token("∖")]
    SetMinus, // Set difference

    #[token("⊂")]
    Subset, // Proper subset

    #[token("⊆")]
    SubsetEq, // Subset or equal

    #[token("⊃")]
    Superset, // Proper superset

    #[token("⊇")]
    SupersetEq, // Superset or equal

    // === Logic Operators ===
    #[token("∧")]
    LogicAnd, // Logical conjunction

    #[token("∨")]
    LogicOr, // Logical disjunction

    #[token("¬")]
    LogicNot, // Logical negation

    #[token("⊻")]
    LogicXor, // Exclusive or

    #[token("⊤")]
    Top, // True/any type

    #[token("⊥")]
    Bottom, // False/never type

    // === Bitwise Operators (Unicode) ===
    #[token("⋏")]
    BitwiseAndSymbol, // Bitwise AND (U+22CF)

    #[token("⋎")]
    BitwiseOrSymbol, // Bitwise OR (U+22CE)

    #[token("⊙")]
    CircledDot, // Hadamard product / element-wise multiply (U+2299)

    // Note: ⊗ (tensor product) is already defined as Token::Tensor below

    // === Type Theory ===
    #[token("∷")]
    TypeAnnotation, // Type annotation (alternative to :)

    // === Analysis/Calculus ===
    #[token("∫")]
    Integral, // Cumulative sum

    #[token("∂")]
    Partial, // Discrete derivative

    #[token("√")]
    Sqrt, // Square root

    #[token("∛")]
    Cbrt, // Cube root

    #[token("∇")]
    Nabla, // Gradient (U+2207)

    // === APL-Inspired Symbols ===
    #[token("⍋")]
    GradeUp, // Sort ascending (U+234B)

    #[token("⍒")]
    GradeDown, // Sort descending (U+2352)

    #[token("⌽")]
    Rotate, // Reverse/rotate (U+233D)

    #[token("↻")]
    #[token("⊳")]  // Right triangle - prose alternative for continue
    CycleArrow, // Cycle/repeat (U+21BB)

    #[token("⌺")]
    QuadDiamond, // Windows/stencil (U+233A)

    #[token("⊞")]
    SquaredPlus, // Chunks (U+229E)

    #[token("⍳")]
    Iota, // Enumerate/index (U+2373)

    // === Category Theory ===
    #[token("∘")]
    Compose, // Function composition

    #[token("⊗")]
    #[token("⊲")]  // Left triangle - prose alternative for break
    Tensor, // Tensor product (parser also handles as break keyword)

    #[token("⊕")]
    DirectSum, // Direct sum / XOR

    // === Data Operations ===
    #[token("⋈")]
    Bowtie, // Join/zip combining (U+22C8)

    #[token("⋳")]
    ElementSmallVerticalBar, // Flatten (U+22F3)

    #[token("⊔")]
    SquareCup, // Lattice join / supremum (U+2294)

    #[token("⊓")]
    SquareCap, // Lattice meet / infimum (U+2293)

    // === Evidentiality Markers ===
    // Note: These are handled contextually since ! and ? have other uses
    #[token("‽")]
    Interrobang, // Paradox/trust boundary (U+203D)

    #[token("◊")]
    Lozenge, // Predicted/speculative (U+25CA) - Token◊

    #[token("⁂")]
    Asterism, // Chaos/entropic (U+2042) - random⁂

    #[token("□")]
    BoxSquare, // Necessity/verification (U+25A1) - |□verify

    // === Legion Morphemes (Holographic Agent Collective) ===
    // From Infernum 2.0 - distributed memory and multi-agent coordination

    #[token("∿")]
    #[token("legion_field")]
    LegionField, // Collective memory substrate (U+223F sine wave) - memory∿

    #[token("⫰")]
    #[token("interfere")]
    Interfere, // Interference query (U+2AF0) - query ⫰ field∿

    #[token("⟁")]
    #[token("distribute")]
    Distribute, // Holographic distribution (U+27C1) - task ⟁ 8

    #[token("⟀")]
    #[token("gather")]
    Gather, // Interference gathering (U+27C0) - fragments ⟀

    #[token("↠")]
    #[token("broadcast")]
    Broadcast, // One-to-many broadcast (U+21A0) - signal ↠ legion

    #[token("⇢")]
    #[token("consensus")]
    Consensus, // Many-to-one consensus (U+21E2) - contributions ⇢

    // Compound Legion operators
    #[token("⊕=")]
    DirectSumEq, // Superposition assign - field∿ ⊕= pattern

    #[token("∂=")]
    PartialEq_, // Decay assign - field∿ ∂= 0.95 (renamed to avoid std conflict)

    #[token("⫰=")]
    InterfereEq, // Interference assign

    // === Affective Markers (Sentiment & Emotion) ===
    // Sentiment polarity
    #[token("⊖")]
    AffectNegative, // Negative sentiment (U+2296 Circled Minus)

    #[token("⊜")]
    AffectNeutral, // Neutral sentiment (U+229C Circled Equals)

    // Note: ⊕ (U+2295) is already DirectSum - we'll use it dual-purpose for positive sentiment

    // Sarcasm/Irony
    #[token("⸮")]
    IronyMark, // Irony/sarcasm marker (U+2E2E - historical percontation point!)

    // Intensity modifiers
    #[token("↑")]
    IntensityUp, // Intensifier (U+2191)

    #[token("↓")]
    IntensityDown, // Dampener (U+2193)

    #[token("⇈")]
    IntensityMax, // Maximum intensity (U+21C8)

    // Formality register
    #[token("♔")]
    FormalRegister, // Formal (U+2654 White King)

    #[token("♟")]
    InformalRegister, // Informal (U+265F Black Pawn)

    // Emotion markers (Plutchik's wheel)
    #[token("☺")]
    EmotionJoy, // Joy (U+263A)

    #[token("☹")]
    EmotionSadness, // Sadness (U+2639)

    #[token("⚡")]
    EmotionAnger, // Anger (U+26A1)

    #[token("❄")]
    EmotionFear, // Fear (U+2744)

    #[token("✦")]
    EmotionSurprise, // Surprise (U+2726)

    #[token("♡")]
    EmotionLove, // Love/Trust (U+2661)

    // Confidence markers
    #[token("◉")]
    ConfidenceHigh, // High confidence (U+25C9)

    #[token("◎")]
    ConfidenceMedium, // Medium confidence (U+25CE)

    #[token("○")]
    ConfidenceLow, // Low confidence (U+25CB)

    // === Aspect Morphemes (verb aspects) ===
    #[token("·ing")]
    AspectProgressive, // Ongoing/streaming aspect

    #[token("·ed")]
    AspectPerfective, // Completed aspect

    #[token("·able")]
    AspectPotential, // Capability aspect

    #[token("·ive")]
    AspectResultative, // Result-producing aspect

    // === Operators ===
    #[token("|")]
    Pipe,
    #[token("·")]
    MiddleDot, // Incorporation / Path separator
    #[token("->")]
    #[token("→")]  // Unicode arrow (U+2192) - native Sigil syntax
    Arrow,
    #[token("=>")]
    FatArrow,
    #[token("<-")]
    LeftArrow,
    #[token("==")]
    EqEq,
    #[token("!=")]
    NotEq,
    #[token("<=")]
    LtEq,
    #[token(">=")]
    GtEq,
    #[token("<")]
    Lt,
    #[token(">")]
    Gt,
    #[token("+")]
    Plus,
    #[token("-")]
    Minus,
    #[token("*")]
    Star,
    #[token("/")]
    Slash,
    #[token("%")]
    Percent,
    #[token("**")]
    StarStar, // Exponentiation
    #[token("&&")]
    AndAnd,
    #[token("||")]
    OrOr,
    #[token("!")]
    Bang, // Evidentiality: known / logical not
    #[token("?")]
    Question, // Evidentiality: uncertain / try
    #[token("~")]
    Tilde, // Evidentiality: reported
    #[token("&")]
    Amp,
    #[token("^")]
    Caret,
    #[token("<<=")]
    ShlEq,
    #[token(">>=")]
    ShrEq,
    #[token("<<")]
    Shl,
    #[token(">>")]
    Shr,
    #[token("=")]
    Eq,
    #[token("+=")]
    PlusEq,
    #[token("-=")]
    MinusEq,
    #[token("*=")]
    StarEq,
    #[token("/=")]
    SlashEq,
    #[token("%=")]
    PercentEq,
    #[token("|=")]
    PipeEq,
    #[token("&=")]
    AmpEq,
    #[token("^=")]
    CaretEq,
    #[token("..")]
    DotDot,
    #[token("..=")]
    DotDotEq,
    #[token("++")]
    PlusPlus, // Concatenation
    #[token(":")]
    Colon,
    #[token(";")]
    Semi,
    #[token(",")]
    Comma,
    #[token(".")]
    Dot,
    #[token("@")]
    At,
    #[token("$")]
    Dollar, // For macro parameters like $n, $__pipe
    #[token("#!")]
    HashBang, // Inner attribute prefix #![...]
    #[token("#")]
    Hash,
    #[token("_", priority = 3)]
    Underscore,

    // === Delimiters ===
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token("{")]
    LBrace,
    #[token("}")]
    RBrace,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,

    // === Special symbols ===
    #[token("∅")]
    Empty, // Void/emptiness (śūnya)
    #[token("◯")]
    Circle, // Geometric zero
    #[token("∞")]
    Infinity, // Infinite value (parser handles contextual use for loop keyword)

    // === Protocol Operations (Sigil-native networking) ===
    #[token("⇒")]
    ProtoSend, // Send data (U+21D2 - rightwards double arrow)

    #[token("⇐")]
    ProtoRecv, // Receive data (U+21D0 - leftwards double arrow)

    #[token("≋")]
    ProtoStream, // Stream data (U+224B - triple tilde)

    #[token("⊸")]
    ProtoConnect, // Connect/lollipop (U+22B8 - multimap)

    #[token("⏱")]
    ProtoTimeout, // Timeout (U+23F1 - stopwatch)

    // Note: ⊗ (Tensor) is used for close in protocol contexts

    // Protocol keywords for ASCII fallback
    #[token("send")]
    Send,
    #[token("recv")]
    Recv,
    #[token("stream")]
    Stream,
    #[token("connect")]
    Connect,
    #[token("close")]
    Close,
    #[token("timeout")]
    Timeout,
    #[token("retry")]
    Retry,
    #[token("header")]
    Header,
    #[token("body")]
    Body,

    // Protocol type identifiers (for incorporation: http·, ws·, grpc·, kafka·)
    #[token("http")]
    Http,
    #[token("https")]
    Https,
    #[token("ws")]
    Ws,
    #[token("wss")]
    Wss,
    #[token("grpc")]
    Grpc,
    #[token("kafka")]
    Kafka,
    #[token("amqp")]
    Amqp,
    #[token("graphql")]
    GraphQL,

    // === Numbers ===
    // Binary: 0b... with optional type suffix
    #[regex(r"0b[01_]+(i8|i16|i32|i64|i128|isize|u8|u16|u32|u64|u128|usize)?", |lex| lex.slice().to_string())]
    BinaryLit(String),

    // Octal: 0o... with optional type suffix
    #[regex(r"0o[0-7_]+(i8|i16|i32|i64|i128|isize|u8|u16|u32|u64|u128|usize)?", |lex| lex.slice().to_string())]
    OctalLit(String),

    // Hex: 0x... with optional type suffix
    #[regex(r"0x[0-9a-fA-F_]+(i8|i16|i32|i64|i128|isize|u8|u16|u32|u64|u128|usize)?", |lex| lex.slice().to_string())]
    HexLit(String),

    // Vigesimal: 0v... (base 20)
    #[regex(r"0v[0-9a-jA-J_]+", |lex| lex.slice().to_string())]
    VigesimalLit(String),

    // Sexagesimal: 0s... (base 60)
    #[regex(r"0s[0-9a-zA-Z_]+", |lex| lex.slice().to_string())]
    SexagesimalLit(String),

    // Duodecimal: 0z... (base 12)
    #[regex(r"0z[0-9a-bA-B_]+", |lex| lex.slice().to_string())]
    DuodecimalLit(String),

    // Float: 123.456 or 1.23e10 or 1e-15 (with or without decimal point if exponent present)
    // Optional type suffix: f16, f32, f64, f128 (with optional underscore separator, e.g. 2.0_f64)
    #[regex(r"([0-9][0-9_]*\.[0-9][0-9_]*([eE][+-]?[0-9_]+)?|[0-9][0-9_]*[eE][+-]?[0-9_]+)_?(f16|f32|f64|f128)?", |lex| lex.slice().to_string())]
    FloatLit(String),

    // Integer: 123 with optional type suffix (i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize)
    // Optional underscore separator before suffix (e.g. 42_i32)
    #[regex(r"[0-9][0-9_]*_?(i8|i16|i32|i64|i128|isize|u8|u16|u32|u64|u128|usize)?", |lex| lex.slice().to_string())]
    IntLit(String),

    // === Strings ===
    // Regular string with escape sequence processing
    // Note: \\(.|\n) handles both regular escapes and line continuation (\ at end of line)
    #[regex(r#""([^"\\]|\\(.|\n))*""#, |lex| {
        let s = lex.slice();
        let inner = &s[1..s.len()-1];
        process_escape_sequences(inner)
    })]
    StringLit(String),

    // Multi-line string (triple-quoted) - handled via callback
    #[token(r#"""""#, multiline_string_callback)]
    MultiLineStringLit(String),

    // Byte string literal
    #[regex(r#"b"([^"\\]|\\.)*""#, |lex| {
        let s = lex.slice();
        let inner = &s[2..s.len()-1];
        process_byte_escape_sequences(inner)
    })]
    ByteStringLit(Vec<u8>),

    // Interpolated string (will be parsed further for expressions)
    #[regex(r#"f"([^"\\]|\\.)*""#, |lex| {
        let s = lex.slice();
        let inner = &s[2..s.len()-1];
        process_escape_sequences(inner)
    })]
    InterpolatedStringLit(String),

    // Sigil string - SQL template (σ prefix)
    #[regex(r#"σ"([^"\\]|\\.)*""#, |lex| {
        let s = lex.slice();
        // Get byte index after the σ character (which is 2 bytes in UTF-8)
        let start = "σ".len() + 1; // σ + opening quote
        let inner = &s[start..s.len()-1];
        process_escape_sequences(inner)
    })]
    SigilStringSql(String),

    // Sigil string - Route template (ρ prefix)
    #[regex(r#"ρ"([^"\\]|\\.)*""#, |lex| {
        let s = lex.slice();
        // Get byte index after the ρ character (which is 2 bytes in UTF-8)
        let start = "ρ".len() + 1; // ρ + opening quote
        let inner = &s[start..s.len()-1];
        process_escape_sequences(inner)
    })]
    SigilStringRoute(String),

    // Char literal with escape sequence processing
    // Matches: single char, hex escape \xNN, unicode escape \u{N...}, or simple escape \c
    #[regex(r"'([^'\\]|\\x[0-9a-fA-F]{2}|\\u\{[0-9a-fA-F]{1,6}\}|\\.)'", |lex| {
        let s = lex.slice();
        let inner = &s[1..s.len()-1];
        process_char_escape(inner)
    })]
    CharLit(char),

    // Byte char literal (b'x' or b'\n')
    #[regex(r"b'([^'\\]|\\x[0-9a-fA-F]{2}|\\.)'", |lex| {
        let s = lex.slice();
        // Extract the character between b' and '
        let inner = &s[2..s.len()-1];
        process_byte_char_escape(inner)
    })]
    ByteCharLit(u8),

    // Raw string (no escape processing, but allows \" for literal quotes in patterns)
    #[regex(r#"r"([^"\\]|\\.)*""#, |lex| {
        let s = lex.slice();
        s[2..s.len()-1].to_string()
    })]
    RawStringLit(String),

    // Raw string with delimiter (r#"..."# style) - handles internal quotes
    #[token(r##"r#""##, raw_string_delimited_callback)]
    RawStringDelimited(String),

    // === Lifetime/Label (for loop labels like 'outer: loop { break 'outer }) ===
    #[regex(r"'[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice()[1..].to_string())]
    Lifetime(String),

    // === Identifiers ===
    // Includes Greek letters for polysynthetic identifiers like compute_ψ_state
    // Greek letters (both cases): αΑ, βΒ, γΓ, δΔ, εΕ, ζΖ, ηΗ, θΘ, ιΙ, κΚ, λΛ, μΜ, νΝ, ξΞ, οΟ, πΠ, ρΡ, σΣ, τΤ, υΥ, φΦ, χΧ, ψΨ, ωΩ
    #[regex(r"[a-zA-Z_αΑβΒγΓδΔεΕζΖηΗθΘιΙκΚλΛμΜνΝξΞοΟπΠρΡσΣτΤυΥφΦχΧψΨωΩ][a-zA-Z0-9_αΑβΒγΓδΔεΕζΖηΗθΘιΙκΚλΛμΜνΝξΞοΟπΠρΡσΣτΤυΥφΦχΧψΨωΩ]*", |lex| lex.slice().to_string())]
    Ident(String),

    // === Rune annotation ===
    // Match //@ rune: followed by name and optional (args) - capture entire annotation
    // Examples: //@ rune: test, //@ rune: should_panic(expected = "error")
    #[regex(r"//@\s*rune:\s*[a-zA-Z_][a-zA-Z0-9_]*(\([^)\n]*\))?", |lex| lex.slice().to_string())]
    RuneAnnotation(String),
}

impl Token {
    pub fn is_keyword(&self) -> bool {
        matches!(
            self,
            Token::Fn
                | Token::Async
                | Token::Let
                | Token::Mut
                | Token::Const
                | Token::Linear
                | Token::Affine
                | Token::Relevant
                | Token::Type
                | Token::Struct
                | Token::Enum
                | Token::Trait
                | Token::Impl
                | Token::Mod
                | Token::Use
                | Token::Pub
                | Token::Actor
                | Token::Saga
                | Token::Scope
                | Token::Rune
                | Token::If
                | Token::Else
                | Token::Match
                | Token::Loop
                | Token::While
                | Token::ForAll      // ∀ - used as for keyword
                | Token::ElementOf   // ∈ - used as in keyword
                | Token::Tensor      // ⊗ - used as break keyword
                | Token::CycleArrow  // ↻ - used as continue keyword
                | Token::Return
                | Token::Yield
                | Token::Await
        ) || self.is_plurality_keyword()
    }

    pub fn is_plurality_keyword(&self) -> bool {
        matches!(
            self,
            Token::Alter
                | Token::Switch
                | Token::Headspace
                | Token::CoCon
                | Token::Reality
                | Token::Split
                | Token::Trigger
                | Token::Layer
                | Token::Location
                | Token::States
                | Token::Anima
                | Token::To
                | Token::From
        )
    }

    pub fn is_alter_source(&self) -> bool {
        matches!(
            self,
            Token::AlterSourceFronting
                | Token::AlterSourceCoCon
                | Token::AlterSourceDormant
                | Token::AlterSourceBlended
        )
    }

    pub fn is_morpheme(&self) -> bool {
        matches!(
            self,
            Token::Tau | Token::Phi | Token::Sigma | Token::Rho |
            Token::Lambda | Token::Pi | Token::Async |
            Token::Delta | Token::Epsilon | Token::Omega | Token::Alpha | Token::Zeta |
            Token::Mu | Token::Chi | Token::Nu | Token::Xi |  // Access morphemes
            Token::Parallel | Token::Gpu |  // Concurrency morphemes
            Token::Integral | Token::Partial | Token::Sqrt | Token::Cbrt |
            Token::Compose
        )
    }

    pub fn is_aspect(&self) -> bool {
        matches!(
            self,
            Token::AspectProgressive
                | Token::AspectPerfective
                | Token::AspectPotential
                | Token::AspectResultative
        )
    }

    pub fn is_data_op(&self) -> bool {
        matches!(
            self,
            Token::Bowtie | Token::ElementSmallVerticalBar | Token::SquareCup | Token::SquareCap
        )
    }

    pub fn is_bitwise_symbol(&self) -> bool {
        matches!(self, Token::BitwiseAndSymbol | Token::BitwiseOrSymbol)
    }

    pub fn is_quantifier(&self) -> bool {
        matches!(
            self,
            Token::ForAll | Token::Exists | Token::ElementOf | Token::NotElementOf
        )
    }

    pub fn is_set_op(&self) -> bool {
        matches!(
            self,
            Token::Union
                | Token::Intersection
                | Token::SetMinus
                | Token::Subset
                | Token::SubsetEq
                | Token::Superset
                | Token::SupersetEq
        )
    }

    pub fn is_logic_op(&self) -> bool {
        matches!(
            self,
            Token::LogicAnd
                | Token::LogicOr
                | Token::LogicNot
                | Token::LogicXor
                | Token::Top
                | Token::Bottom
        )
    }

    pub fn is_evidentiality(&self) -> bool {
        matches!(
            self,
            Token::Bang | Token::Question | Token::Tilde | Token::Interrobang | Token::Lozenge
        )
    }

    pub fn is_legion_morpheme(&self) -> bool {
        matches!(
            self,
            Token::LegionField      // ∿ - collective memory
                | Token::DirectSum  // ⊕ - superposition
                | Token::Interfere  // ⫰ - interference
                | Token::ConfidenceHigh  // ◉ - resonance (dual-purpose)
                | Token::Distribute // ⟁ - holographic distribution
                | Token::Gather     // ⟀ - interference gathering
                | Token::Broadcast  // ↠ - one-to-many
                | Token::Consensus  // ⇢ - many-to-one
                | Token::Partial    // ∂ - decay
        )
    }

    pub fn is_legion_assign(&self) -> bool {
        matches!(
            self,
            Token::DirectSumEq | Token::PartialEq_ | Token::InterfereEq
        )
    }

    pub fn is_affective(&self) -> bool {
        matches!(
            self,
            // Sentiment
            Token::DirectSum |  // ⊕ positive (dual-purpose with DirectSum)
            Token::AffectNegative |  // ⊖ negative
            Token::AffectNeutral |  // ⊜ neutral
            // Sarcasm
            Token::IronyMark |  // ⸮ irony/sarcasm
            // Intensity
            Token::IntensityUp |  // ↑
            Token::IntensityDown |  // ↓
            Token::IntensityMax |  // ⇈
            // Formality
            Token::FormalRegister |  // ♔
            Token::InformalRegister |  // ♟
            // Emotions
            Token::EmotionJoy |  // ☺
            Token::EmotionSadness |  // ☹
            Token::EmotionAnger |  // ⚡
            Token::EmotionFear |  // ❄
            Token::EmotionSurprise |  // ✦
            Token::EmotionLove |  // ♡
            // Confidence
            Token::ConfidenceHigh |  // ◉
            Token::ConfidenceMedium |  // ◎
            Token::ConfidenceLow // ○
        )
    }

    pub fn is_sentiment(&self) -> bool {
        matches!(
            self,
            Token::DirectSum | Token::AffectNegative | Token::AffectNeutral
        )
    }

    pub fn is_emotion(&self) -> bool {
        matches!(
            self,
            Token::EmotionJoy
                | Token::EmotionSadness
                | Token::EmotionAnger
                | Token::EmotionFear
                | Token::EmotionSurprise
                | Token::EmotionLove
        )
    }

    pub fn is_intensity(&self) -> bool {
        matches!(
            self,
            Token::IntensityUp | Token::IntensityDown | Token::IntensityMax
        )
    }

    /// Returns true if this token is any kind of doc comment (evidential or legacy)
    pub fn is_doc_comment(&self) -> bool {
        matches!(
            self,
            Token::DocCommentVerified(_)
                | Token::DocCommentVerifiedInner(_)
                | Token::DocCommentReported(_)
                | Token::DocCommentReportedInner(_)
                | Token::DocCommentUncertain(_)
                | Token::DocCommentUncertainInner(_)
                | Token::DocCommentPredicted(_)
                | Token::DocCommentPredictedInner(_)
                | Token::DocCommentParadox(_)
                | Token::DocCommentParadoxInner(_)
                | Token::DocComment(_)  // Legacy /// comments map to Reported
        )
    }

    /// Returns true if this is an inner doc comment (documents the enclosing item)
    pub fn is_inner_doc_comment(&self) -> bool {
        matches!(
            self,
            Token::DocCommentVerifiedInner(_)
                | Token::DocCommentReportedInner(_)
                | Token::DocCommentUncertainInner(_)
                | Token::DocCommentPredictedInner(_)
                | Token::DocCommentParadoxInner(_)
        )
    }

    /// Returns the evidentiality marker for a doc comment (!, ~, ?, ◊, ‽)
    /// Returns '~' (Reported) for legacy /// comments as default
    pub fn doc_comment_evidentiality(&self) -> Option<char> {
        match self {
            Token::DocCommentVerified(_) | Token::DocCommentVerifiedInner(_) => Some('!'),
            Token::DocCommentReported(_) | Token::DocCommentReportedInner(_) => Some('~'),
            Token::DocCommentUncertain(_) | Token::DocCommentUncertainInner(_) => Some('?'),
            Token::DocCommentPredicted(_) | Token::DocCommentPredictedInner(_) => Some('◊'),
            Token::DocCommentParadox(_) | Token::DocCommentParadoxInner(_) => Some('‽'),
            Token::DocComment(_) => Some('~'),  // Legacy defaults to Reported
            _ => None,
        }
    }

    /// Extracts the content string from a doc comment token
    pub fn doc_comment_content(&self) -> Option<&str> {
        match self {
            Token::DocCommentVerified(s)
            | Token::DocCommentVerifiedInner(s)
            | Token::DocCommentReported(s)
            | Token::DocCommentReportedInner(s)
            | Token::DocCommentUncertain(s)
            | Token::DocCommentUncertainInner(s)
            | Token::DocCommentPredicted(s)
            | Token::DocCommentPredictedInner(s)
            | Token::DocCommentParadox(s)
            | Token::DocCommentParadoxInner(s)
            | Token::DocComment(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

/// Lexer wrapping Logos for Sigil.
pub struct Lexer<'a> {
    inner: logos::Lexer<'a, Token>,
    /// Buffer for lookahead tokens (supports multi-token peek)
    buffer: Vec<Option<(Token, Span)>>,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Self {
            inner: Token::lexer(source),
            buffer: Vec::new(),
        }
    }

    /// Read the next token from the underlying logos lexer
    fn read_next(&mut self) -> Option<(Token, Span)> {
        match self.inner.next() {
            Some(Ok(token)) => {
                let span = self.inner.span();
                Some((token, Span::new(span.start, span.end)))
            }
            Some(Err(_)) => {
                // Skip invalid tokens and try next
                self.read_next()
            }
            None => None,
        }
    }

    pub fn next_token(&mut self) -> Option<(Token, Span)> {
        if !self.buffer.is_empty() {
            // Return from buffer (front = next token)
            // Each buffer element is Option<(Token, Span)> where None = EOF
            return self.buffer.remove(0);
        }
        self.read_next()
    }

    pub fn peek(&mut self) -> Option<&(Token, Span)> {
        self.peek_n(0)
    }

    /// Peek n tokens ahead (0 = next token, 1 = token after that, etc.)
    pub fn peek_n(&mut self, n: usize) -> Option<&(Token, Span)> {
        // Fill buffer up to position n
        while self.buffer.len() <= n {
            let token = self.read_next();
            self.buffer.push(token);
        }
        self.buffer.get(n).and_then(|opt| opt.as_ref())
    }

    pub fn span(&self) -> Span {
        let span = self.inner.span();
        Span::new(span.start, span.end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morphemes() {
        // Note: Case matters for Greek letters in Sigil:
        // - lowercase λ = Token::LambdaExpr (closure expression), uppercase Λ = Token::Lambda (morpheme)
        // - lowercase σ = Token::Sigma (sort morpheme), uppercase Σ = Token::Struct (keyword)
        // - lowercase π = identifier, uppercase Π = Token::Pi (product morpheme)
        // This test verifies uppercase morphemes that are NOT keywords
        let mut lexer = Lexer::new("τ φ σ ρ Λ Π ⌛");
        assert!(matches!(lexer.next_token(), Some((Token::Tau, _))));      // τ = Tau (transform)
        assert!(matches!(lexer.next_token(), Some((Token::Phi, _))));      // φ = Phi (filter)
        assert!(matches!(lexer.next_token(), Some((Token::Sigma, _))));    // σ = Sigma (sort/sum)
        assert!(matches!(lexer.next_token(), Some((Token::Rho, _))));      // ρ = Rho (reduce)
        assert!(matches!(lexer.next_token(), Some((Token::Lambda, _))));   // Λ = Lambda (uppercase)
        assert!(matches!(lexer.next_token(), Some((Token::Pi, _))));       // Π = Pi (product)
        assert!(matches!(lexer.next_token(), Some((Token::Async, _))));    // ⌛ = Async
    }

    #[test]
    fn test_evidentiality() {
        let mut lexer = Lexer::new("value! uncertain? reported~ paradox‽");
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "value"));
        assert!(matches!(lexer.next_token(), Some((Token::Bang, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "uncertain"));
        assert!(matches!(lexer.next_token(), Some((Token::Question, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "reported"));
        assert!(matches!(lexer.next_token(), Some((Token::Tilde, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "paradox"));
        assert!(matches!(lexer.next_token(), Some((Token::Interrobang, _))));
    }

    #[test]
    fn test_pipe_chain() {
        let mut lexer = Lexer::new("data|τ{f}|φ{p}|σ");
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "data"));
        assert!(matches!(lexer.next_token(), Some((Token::Pipe, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Tau, _))));
        assert!(matches!(lexer.next_token(), Some((Token::LBrace, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "f"));
        assert!(matches!(lexer.next_token(), Some((Token::RBrace, _))));
    }

    #[test]
    fn test_numbers() {
        let mut lexer = Lexer::new("42 0b1010 0o52 0x2A 0v22 0s42 3.14");
        assert!(matches!(lexer.next_token(), Some((Token::IntLit(s), _)) if s == "42"));
        assert!(matches!(lexer.next_token(), Some((Token::BinaryLit(s), _)) if s == "0b1010"));
        assert!(matches!(lexer.next_token(), Some((Token::OctalLit(s), _)) if s == "0o52"));
        assert!(matches!(lexer.next_token(), Some((Token::HexLit(s), _)) if s == "0x2A"));
        assert!(matches!(lexer.next_token(), Some((Token::VigesimalLit(s), _)) if s == "0v22"));
        assert!(matches!(lexer.next_token(), Some((Token::SexagesimalLit(s), _)) if s == "0s42"));
        assert!(matches!(lexer.next_token(), Some((Token::FloatLit(s), _)) if s == "3.14"));
    }

    #[test]
    fn test_incorporation() {
        let mut lexer = Lexer::new("file·open·read");
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "file"));
        assert!(matches!(lexer.next_token(), Some((Token::MiddleDot, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "open"));
        assert!(matches!(lexer.next_token(), Some((Token::MiddleDot, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "read"));
    }

    #[test]
    fn test_special_symbols() {
        let mut lexer = Lexer::new("∅ ◯ ∞");
        assert!(matches!(lexer.next_token(), Some((Token::Empty, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Circle, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Infinity, _))));
    }

    #[test]
    fn test_quantifiers() {
        let mut lexer = Lexer::new("∀x ∃y x∈S y∉T");
        assert!(matches!(lexer.next_token(), Some((Token::ForAll, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "x"));
        assert!(matches!(lexer.next_token(), Some((Token::Exists, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "y"));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "x"));
        assert!(matches!(lexer.next_token(), Some((Token::ElementOf, _))));
    }

    #[test]
    fn test_set_operations() {
        let mut lexer = Lexer::new("A∪B A∩B A∖B A⊂B A⊆B");
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "A"));
        assert!(matches!(lexer.next_token(), Some((Token::Union, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "B"));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "A"));
        assert!(matches!(lexer.next_token(), Some((Token::Intersection, _))));
    }

    #[test]
    fn test_logic_operators() {
        let mut lexer = Lexer::new("p∧q p∨q ¬p p⊻q ⊤ ⊥");
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "p"));
        assert!(matches!(lexer.next_token(), Some((Token::LogicAnd, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "q"));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "p"));
        assert!(matches!(lexer.next_token(), Some((Token::LogicOr, _))));
    }

    #[test]
    fn test_analysis_operators() {
        let mut lexer = Lexer::new("∫f ∂g √x ∛y f∘g");
        assert!(matches!(lexer.next_token(), Some((Token::Integral, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "f"));
        assert!(matches!(lexer.next_token(), Some((Token::Partial, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Ident(s), _)) if s == "g"));
        assert!(matches!(lexer.next_token(), Some((Token::Sqrt, _))));
    }

    #[test]
    fn test_additional_morphemes() {
        let mut lexer = Lexer::new("δ ε ω α ζ");
        assert!(matches!(lexer.next_token(), Some((Token::Delta, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Epsilon, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Omega, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Alpha, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Zeta, _))));
    }

    #[test]
    fn test_ffi_keywords() {
        let mut lexer = Lexer::new("extern unsafe");
        assert!(matches!(lexer.next_token(), Some((Token::Extern, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Unsafe, _))));
    }

    #[test]
    fn test_parallel_morphemes() {
        // Note: ⊛ is Token::Convolve (convolution operator), not Gpu
        // gpu (keyword) produces Token::Gpu
        let mut lexer = Lexer::new("∥ parallel ⊛ gpu");
        assert!(matches!(lexer.next_token(), Some((Token::Parallel, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Parallel, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Convolve, _))));  // ⊛ = Convolve
        assert!(matches!(lexer.next_token(), Some((Token::Gpu, _))));       // gpu = Gpu
    }

    #[test]
    fn test_lifetime_labels() {
        // Test loop labels
        let mut lexer = Lexer::new("'outer: loop { break 'outer }");
        assert!(matches!(lexer.next_token(), Some((Token::Lifetime(s), _)) if s == "outer"));
        assert!(matches!(lexer.next_token(), Some((Token::Colon, _))));
        assert!(matches!(lexer.next_token(), Some((Token::Loop, _))));
        assert!(matches!(lexer.next_token(), Some((Token::LBrace, _))));
        assert!(matches!(lexer.next_token(), Some((Token::DeprecatedRustKeyword(s), _)) if s == "break"));
        assert!(matches!(lexer.next_token(), Some((Token::Lifetime(s), _)) if s == "outer"));
        assert!(matches!(lexer.next_token(), Some((Token::RBrace, _))));
    }

    // ==================== STRING LITERAL TESTS ====================

    #[test]
    fn test_string_escape_sequences() {
        // Test basic escape sequences
        let mut lexer = Lexer::new(r#""hello\nworld""#);
        match lexer.next_token() {
            Some((Token::StringLit(s), _)) => assert_eq!(s, "hello\nworld"),
            other => panic!("Expected StringLit, got {:?}", other),
        }

        // Test tab escape
        let mut lexer = Lexer::new(r#""hello\tworld""#);
        match lexer.next_token() {
            Some((Token::StringLit(s), _)) => assert_eq!(s, "hello\tworld"),
            other => panic!("Expected StringLit, got {:?}", other),
        }

        // Test carriage return
        let mut lexer = Lexer::new(r#""hello\rworld""#);
        match lexer.next_token() {
            Some((Token::StringLit(s), _)) => assert_eq!(s, "hello\rworld"),
            other => panic!("Expected StringLit, got {:?}", other),
        }

        // Test escaped backslash
        let mut lexer = Lexer::new(r#""hello\\world""#);
        match lexer.next_token() {
            Some((Token::StringLit(s), _)) => assert_eq!(s, "hello\\world"),
            other => panic!("Expected StringLit, got {:?}", other),
        }

        // Test escaped quote
        let mut lexer = Lexer::new(r#""hello\"world""#);
        match lexer.next_token() {
            Some((Token::StringLit(s), _)) => assert_eq!(s, "hello\"world"),
            other => panic!("Expected StringLit, got {:?}", other),
        }

        // Test null character
        let mut lexer = Lexer::new(r#""hello\0world""#);
        match lexer.next_token() {
            Some((Token::StringLit(s), _)) => assert_eq!(s, "hello\0world"),
            other => panic!("Expected StringLit, got {:?}", other),
        }
    }

    #[test]
    fn test_string_hex_escape() {
        // Test \xNN hex escape
        let mut lexer = Lexer::new(r#""hello\x41world""#);
        match lexer.next_token() {
            Some((Token::StringLit(s), _)) => assert_eq!(s, "helloAworld"),
            other => panic!("Expected StringLit, got {:?}", other),
        }
    }

    #[test]
    fn test_string_unicode_escape() {
        // Test \u{NNNN} Unicode escape
        let mut lexer = Lexer::new(r#""hello\u{1F600}world""#);
        match lexer.next_token() {
            Some((Token::StringLit(s), _)) => assert_eq!(s, "hello😀world"),
            other => panic!("Expected StringLit, got {:?}", other),
        }

        // Test Greek letter
        let mut lexer = Lexer::new(r#""\u{03C4}""#);
        match lexer.next_token() {
            Some((Token::StringLit(s), _)) => assert_eq!(s, "τ"),
            other => panic!("Expected StringLit, got {:?}", other),
        }
    }

    #[test]
    fn test_char_escape_sequences() {
        let mut lexer = Lexer::new(r"'\n'");
        match lexer.next_token() {
            Some((Token::CharLit(c), _)) => assert_eq!(c, '\n'),
            other => panic!("Expected CharLit, got {:?}", other),
        }

        let mut lexer = Lexer::new(r"'\t'");
        match lexer.next_token() {
            Some((Token::CharLit(c), _)) => assert_eq!(c, '\t'),
            other => panic!("Expected CharLit, got {:?}", other),
        }

        let mut lexer = Lexer::new(r"'\\'");
        match lexer.next_token() {
            Some((Token::CharLit(c), _)) => assert_eq!(c, '\\'),
            other => panic!("Expected CharLit, got {:?}", other),
        }
    }

    #[test]
    fn test_raw_string() {
        // Raw strings should NOT process escapes
        let mut lexer = Lexer::new(r#"r"hello\nworld""#);
        match lexer.next_token() {
            Some((Token::RawStringLit(s), _)) => assert_eq!(s, r"hello\nworld"),
            other => panic!("Expected RawStringLit, got {:?}", other),
        }
    }

    #[test]
    fn test_raw_string_delimited() {
        // r#"..."# style
        let mut lexer = Lexer::new(r##"r#"hello "world""#"##);
        match lexer.next_token() {
            Some((Token::RawStringDelimited(s), _)) => assert_eq!(s, r#"hello "world""#),
            other => panic!("Expected RawStringDelimited, got {:?}", other),
        }
    }

    #[test]
    fn test_byte_string() {
        let mut lexer = Lexer::new(r#"b"hello""#);
        match lexer.next_token() {
            Some((Token::ByteStringLit(bytes), _)) => {
                assert_eq!(bytes, vec![104, 101, 108, 108, 111]); // "hello" in ASCII
            }
            other => panic!("Expected ByteStringLit, got {:?}", other),
        }
    }

    #[test]
    fn test_interpolated_string() {
        let mut lexer = Lexer::new(r#"f"hello {name}""#);
        match lexer.next_token() {
            Some((Token::InterpolatedStringLit(s), _)) => assert_eq!(s, "hello {name}"),
            other => panic!("Expected InterpolatedStringLit, got {:?}", other),
        }
    }

    #[test]
    fn test_sigil_string_sql() {
        let mut lexer = Lexer::new(r#"σ"SELECT * FROM {table}""#);
        match lexer.next_token() {
            Some((Token::SigilStringSql(s), _)) => assert_eq!(s, "SELECT * FROM {table}"),
            other => panic!("Expected SigilStringSql, got {:?}", other),
        }
    }

    #[test]
    fn test_sigil_string_route() {
        let mut lexer = Lexer::new(r#"ρ"/api/v1/{resource}/{id}""#);
        match lexer.next_token() {
            Some((Token::SigilStringRoute(s), _)) => assert_eq!(s, "/api/v1/{resource}/{id}"),
            other => panic!("Expected SigilStringRoute, got {:?}", other),
        }
    }

    #[test]
    fn test_unicode_in_strings() {
        // Test direct Unicode in strings
        let mut lexer = Lexer::new(r#""τφσρ 你好 🦀""#);
        match lexer.next_token() {
            Some((Token::StringLit(s), _)) => assert_eq!(s, "τφσρ 你好 🦀"),
            other => panic!("Expected StringLit, got {:?}", other),
        }
    }

    #[test]
    fn test_empty_string() {
        let mut lexer = Lexer::new(r#""""#);
        match lexer.next_token() {
            Some((Token::StringLit(s), _)) => assert_eq!(s, ""),
            other => panic!("Expected empty StringLit, got {:?}", other),
        }
    }

    #[test]
    fn test_escape_sequence_helper() {
        // Unit test the helper function directly
        assert_eq!(process_escape_sequences(r"hello\nworld"), "hello\nworld");
        assert_eq!(process_escape_sequences(r"hello\tworld"), "hello\tworld");
        assert_eq!(process_escape_sequences(r"hello\\world"), "hello\\world");
        assert_eq!(process_escape_sequences(r#"hello\"world"#), "hello\"world");
        assert_eq!(process_escape_sequences(r"hello\x41world"), "helloAworld");
        assert_eq!(
            process_escape_sequences(r"hello\u{1F600}world"),
            "hello😀world"
        );
    }
}
