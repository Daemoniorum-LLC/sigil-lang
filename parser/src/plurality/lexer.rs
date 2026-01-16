//! # Plurality Lexer Extensions
//!
//! Extends the Sigil lexer with tokens for plurality constructs:
//! - Keywords: `alter`, `switch`, `headspace`, `cocon`, `reality`, `split`
//! - Alter-source markers: `@!`, `@~`, `@?`, `@‽`
//! - Forced operators: `switch!`, `split!`
//!
//! ## Integration with Main Lexer
//!
//! These tokens should be added to `lexer.rs` Token enum:
//!
//! ```rust,ignore
//! // Plurality keywords
//! #[token("alter")]
//! Alter,
//! #[token("switch")]
//! Switch,
//! #[token("headspace")]
//! Headspace,
//! #[token("cocon")]
//! CoCon,
//! #[token("reality")]
//! Reality,
//! #[token("split")]
//! Split,
//! #[token("trigger")]
//! Trigger,
//! #[token("location")]
//! Location,
//! #[token("layer")]
//! Layer,
//! #[token("transform")]
//! Transform,
//! #[token("states")]
//! States,
//! #[token("anima")]
//! Anima,
//!
//! // Alter-source markers (compound tokens)
//! #[token("@!")]
//! AlterSourceFronting,    // Authoritative from fronting alter
//! #[token("@~")]
//! AlterSourceCoCon,       // Reported from co-conscious
//! #[token("@?")]
//! AlterSourceDormant,     // Uncertain from dormant
//! #[token("@‽")]
//! AlterSourceBlended,     // Paradoxical from blended state
//!
//! // Forced operation variants
//! #[token("switch!")]
//! SwitchForced,           // Forced switch (bypasses deliberation)
//! #[token("split!")]
//! SplitForced,            // Forced split (trauma response)
//! ```

use crate::lexer::Token;
use crate::span::Span;

// ============================================================================
// PLURALITY TOKEN CATEGORIES
// ============================================================================

/// Check if a token is a plurality keyword
pub fn is_plurality_keyword(token: &Token) -> bool {
    matches!(
        token,
        Token::Ident(s) if matches!(s.as_str(),
            "alter" | "switch" | "headspace" | "cocon" |
            "reality" | "split" | "trigger" | "location" |
            "layer" | "transform" | "states" | "anima"
        )
    )
}

/// Check if a token is an alter category keyword
pub fn is_alter_category(token: &Token) -> bool {
    matches!(
        token,
        Token::Ident(s) if matches!(s.as_str(),
            "Council" | "Servant" | "Fragment" | "Hidden" | "Persecutor"
        )
    )
}

/// Check if a token is an alter state keyword
pub fn is_alter_state(token: &Token) -> bool {
    matches!(
        token,
        Token::Ident(s) if matches!(s.as_str(),
            "Dormant" | "Stirring" | "CoConscious" | "Emerging" |
            "Fronting" | "Receding" | "Triggered" | "Dissociating"
        )
    )
}

/// Check if a token sequence represents an alter-source marker
/// Returns (is_alter_source, consumed_count)
pub fn check_alter_source_sequence(tokens: &[(Token, Span)]) -> Option<AlterSourceMarker> {
    if tokens.is_empty() {
        return None;
    }

    // Check for @ followed by evidentiality marker
    if let Token::At = &tokens[0].0 {
        if tokens.len() >= 2 {
            match &tokens[1].0 {
                Token::Bang => Some(AlterSourceMarker::Fronting),
                Token::Tilde => Some(AlterSourceMarker::CoCon),
                Token::Question => Some(AlterSourceMarker::Dormant),
                Token::Interrobang => Some(AlterSourceMarker::Blended),
                Token::Ident(name) if name == "Fronting" => Some(AlterSourceMarker::Fronting),
                Token::Ident(name) if name == "CoCon" => Some(AlterSourceMarker::CoCon),
                Token::Ident(name) if name == "Dormant" => Some(AlterSourceMarker::Dormant),
                Token::Ident(name) if name == "Blended" => Some(AlterSourceMarker::Blended),
                _ => None,
            }
        } else {
            None
        }
    } else {
        None
    }
}

/// Check if a token sequence represents a forced operation
/// (e.g., `switch!` or `split!`)
pub fn check_forced_operation(tokens: &[(Token, Span)]) -> Option<ForcedOperation> {
    if tokens.len() < 2 {
        return None;
    }

    if let Token::Ident(name) = &tokens[0].0 {
        if let Token::Bang = &tokens[1].0 {
            match name.as_str() {
                "switch" => return Some(ForcedOperation::Switch),
                "split" => return Some(ForcedOperation::Split),
                _ => {}
            }
        }
    }

    None
}

// ============================================================================
// PLURALITY TOKEN TYPES
// ============================================================================

/// Alter-source marker type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlterSourceMarker {
    /// `@!` or `@Fronting` - authoritative from fronting alter
    Fronting,
    /// `@~` or `@CoCon` - reported from co-conscious alter
    CoCon,
    /// `@?` or `@Dormant` - uncertain from dormant alter
    Dormant,
    /// `@‽` or `@Blended` - paradoxical from blended state
    Blended,
}

impl AlterSourceMarker {
    /// Convert to evidentiality marker equivalent
    pub fn to_evidentiality(&self) -> &'static str {
        match self {
            AlterSourceMarker::Fronting => "!",
            AlterSourceMarker::CoCon => "~",
            AlterSourceMarker::Dormant => "?",
            AlterSourceMarker::Blended => "‽",
        }
    }

    /// Get the symbol representation
    pub fn symbol(&self) -> &'static str {
        match self {
            AlterSourceMarker::Fronting => "@!",
            AlterSourceMarker::CoCon => "@~",
            AlterSourceMarker::Dormant => "@?",
            AlterSourceMarker::Blended => "@‽",
        }
    }
}

/// Forced operation type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ForcedOperation {
    /// `switch!` - forced switch bypassing deliberation
    Switch,
    /// `split!` - forced split (trauma response)
    Split,
}

// ============================================================================
// PLURALITY TOKEN STREAM HELPERS
// ============================================================================

/// Iterator adapter for plurality token processing
pub struct PluralityTokenStream<'a> {
    tokens: &'a [(Token, Span)],
    position: usize,
}

impl<'a> PluralityTokenStream<'a> {
    pub fn new(tokens: &'a [(Token, Span)]) -> Self {
        Self {
            tokens,
            position: 0,
        }
    }

    /// Peek at the current token
    pub fn peek(&self) -> Option<&(Token, Span)> {
        self.tokens.get(self.position)
    }

    /// Peek at the next n tokens
    pub fn peek_n(&self, n: usize) -> &[(Token, Span)] {
        let end = (self.position + n).min(self.tokens.len());
        &self.tokens[self.position..end]
    }

    /// Advance by n tokens
    pub fn advance(&mut self, n: usize) {
        self.position = (self.position + n).min(self.tokens.len());
    }

    /// Check if we're at a plurality keyword
    pub fn at_plurality_keyword(&self) -> bool {
        self.peek()
            .map(|(t, _)| is_plurality_keyword(t))
            .unwrap_or(false)
    }

    /// Try to consume an alter-source marker
    pub fn try_alter_source(&mut self) -> Option<(AlterSourceMarker, Span)> {
        let lookahead = self.peek_n(2);
        if let Some(marker) = check_alter_source_sequence(lookahead) {
            let start = lookahead[0].1.start;
            let end = lookahead[1].1.end;
            self.advance(2);
            Some((marker, Span::new(start, end)))
        } else {
            None
        }
    }

    /// Try to consume a forced operation
    pub fn try_forced_operation(&mut self) -> Option<(ForcedOperation, Span)> {
        let lookahead = self.peek_n(2);
        if let Some(op) = check_forced_operation(lookahead) {
            let start = lookahead[0].1.start;
            let end = lookahead[1].1.end;
            self.advance(2);
            Some((op, Span::new(start, end)))
        } else {
            None
        }
    }

    /// Check if we're at an alter definition start
    /// (`alter Ident: Category` or `alter Ident {`)
    pub fn at_alter_def(&self) -> bool {
        let lookahead = self.peek_n(4);
        if lookahead.is_empty() {
            return false;
        }

        // Check for `alter` keyword
        matches!(&lookahead[0].0, Token::Ident(s) if s == "alter")
            && lookahead.len() > 1
            && matches!(&lookahead[1].0, Token::Ident(_))
    }

    /// Check if we're at a switch expression
    /// (`switch to Alter` or `switch! to Alter`)
    pub fn at_switch_expr(&self) -> bool {
        let lookahead = self.peek_n(3);
        if lookahead.is_empty() {
            return false;
        }

        match &lookahead[0].0 {
            Token::Ident(s) if s == "switch" => {
                // Check for `switch to` or `switch! to`
                if lookahead.len() > 1 {
                    match &lookahead[1].0 {
                        Token::Bang => {
                            // switch! to ...
                            lookahead.len() > 2
                                && matches!(&lookahead[2].0, Token::Ident(s) if s == "to")
                        }
                        Token::Ident(s) if s == "to" => true,
                        _ => false,
                    }
                } else {
                    false
                }
            }
            _ => false,
        }
    }

    /// Check if we're at a headspace definition
    pub fn at_headspace_def(&self) -> bool {
        let lookahead = self.peek_n(2);
        !lookahead.is_empty()
            && matches!(&lookahead[0].0, Token::Ident(s) if s == "headspace")
            && lookahead.len() > 1
            && matches!(&lookahead[1].0, Token::Ident(_))
    }

    /// Check if we're at a reality definition
    pub fn at_reality_def(&self) -> bool {
        let lookahead = self.peek_n(3);
        if lookahead.len() < 3 {
            return false;
        }

        matches!(&lookahead[0].0, Token::Ident(s) if s == "reality")
            && matches!(&lookahead[1].0, Token::Ident(s) if s == "entity")
            && matches!(&lookahead[2].0, Token::Ident(_))
    }

    /// Check if we're at a co-con channel definition
    /// (`cocon<A, B> name { ... }`)
    pub fn at_cocon_channel(&self) -> bool {
        let lookahead = self.peek_n(2);
        !lookahead.is_empty()
            && matches!(&lookahead[0].0, Token::Ident(s) if s == "cocon")
            && lookahead.len() > 1
            && matches!(&lookahead[1].0, Token::Lt)
    }

    /// Check if we're at a trigger handler
    /// (`on trigger Name { ... }`)
    pub fn at_trigger_handler(&self) -> bool {
        let lookahead = self.peek_n(3);
        if lookahead.len() < 3 {
            return false;
        }

        matches!(&lookahead[0].0, Token::On)
            && matches!(&lookahead[1].0, Token::Ident(s) if s == "trigger")
            && matches!(&lookahead[2].0, Token::Ident(_))
    }

    /// Check if we're at a split expression
    /// (`split! from Alter { ... }`)
    pub fn at_split_expr(&self) -> bool {
        let lookahead = self.peek_n(3);
        if lookahead.is_empty() {
            return false;
        }

        matches!(&lookahead[0].0, Token::Ident(s) if s == "split")
            && lookahead.len() > 1
            && matches!(&lookahead[1].0, Token::Bang)
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_token(token: Token) -> (Token, Span) {
        (token, Span::default())
    }

    #[test]
    fn test_alter_source_markers() {
        // Test @! (fronting)
        let tokens = vec![make_token(Token::At), make_token(Token::Bang)];
        assert_eq!(
            check_alter_source_sequence(&tokens),
            Some(AlterSourceMarker::Fronting)
        );

        // Test @~ (co-con)
        let tokens = vec![make_token(Token::At), make_token(Token::Tilde)];
        assert_eq!(
            check_alter_source_sequence(&tokens),
            Some(AlterSourceMarker::CoCon)
        );

        // Test @? (dormant)
        let tokens = vec![make_token(Token::At), make_token(Token::Question)];
        assert_eq!(
            check_alter_source_sequence(&tokens),
            Some(AlterSourceMarker::Dormant)
        );

        // Test @‽ (blended)
        let tokens = vec![make_token(Token::At), make_token(Token::Interrobang)];
        assert_eq!(
            check_alter_source_sequence(&tokens),
            Some(AlterSourceMarker::Blended)
        );
    }

    #[test]
    fn test_forced_operations() {
        // Test switch!
        let tokens = vec![
            make_token(Token::Ident("switch".to_string())),
            make_token(Token::Bang),
        ];
        assert_eq!(
            check_forced_operation(&tokens),
            Some(ForcedOperation::Switch)
        );

        // Test split!
        let tokens = vec![
            make_token(Token::Ident("split".to_string())),
            make_token(Token::Bang),
        ];
        assert_eq!(
            check_forced_operation(&tokens),
            Some(ForcedOperation::Split)
        );
    }

    #[test]
    fn test_plurality_keywords() {
        assert!(is_plurality_keyword(&Token::Ident("alter".to_string())));
        assert!(is_plurality_keyword(&Token::Ident("switch".to_string())));
        assert!(is_plurality_keyword(&Token::Ident("headspace".to_string())));
        assert!(is_plurality_keyword(&Token::Ident("cocon".to_string())));
        assert!(!is_plurality_keyword(&Token::Ident("struct".to_string())));
    }

    #[test]
    fn test_alter_categories() {
        assert!(is_alter_category(&Token::Ident("Council".to_string())));
        assert!(is_alter_category(&Token::Ident("Servant".to_string())));
        assert!(is_alter_category(&Token::Ident("Fragment".to_string())));
        assert!(!is_alter_category(&Token::Ident("Other".to_string())));
    }

    #[test]
    fn test_alter_states() {
        assert!(is_alter_state(&Token::Ident("Dormant".to_string())));
        assert!(is_alter_state(&Token::Ident("Fronting".to_string())));
        assert!(is_alter_state(&Token::Ident("CoConscious".to_string())));
        assert!(!is_alter_state(&Token::Ident("Running".to_string())));
    }

    #[test]
    fn test_token_stream_alter_def() {
        let tokens = vec![
            make_token(Token::Ident("alter".to_string())),
            make_token(Token::Ident("Abaddon".to_string())),
            make_token(Token::Colon),
            make_token(Token::Ident("Council".to_string())),
        ];
        let stream = PluralityTokenStream::new(&tokens);
        assert!(stream.at_alter_def());
    }

    #[test]
    fn test_token_stream_switch_expr() {
        // Regular switch
        let tokens = vec![
            make_token(Token::Ident("switch".to_string())),
            make_token(Token::Ident("to".to_string())),
            make_token(Token::Ident("Beleth".to_string())),
        ];
        let stream = PluralityTokenStream::new(&tokens);
        assert!(stream.at_switch_expr());

        // Forced switch
        let tokens = vec![
            make_token(Token::Ident("switch".to_string())),
            make_token(Token::Bang),
            make_token(Token::Ident("to".to_string())),
            make_token(Token::Ident("Abaddon".to_string())),
        ];
        let stream = PluralityTokenStream::new(&tokens);
        assert!(stream.at_switch_expr());
    }

    #[test]
    fn test_token_stream_headspace() {
        let tokens = vec![
            make_token(Token::Ident("headspace".to_string())),
            make_token(Token::Ident("InnerWorld".to_string())),
            make_token(Token::LBrace),
        ];
        let stream = PluralityTokenStream::new(&tokens);
        assert!(stream.at_headspace_def());
    }

    #[test]
    fn test_token_stream_cocon() {
        let tokens = vec![
            make_token(Token::Ident("cocon".to_string())),
            make_token(Token::Lt),
            make_token(Token::Ident("Stolas".to_string())),
            make_token(Token::Comma),
            make_token(Token::Ident("Paimon".to_string())),
            make_token(Token::Gt),
        ];
        let stream = PluralityTokenStream::new(&tokens);
        assert!(stream.at_cocon_channel());
    }
}
