//! # Plurality Parser Extensions
//!
//! Parser rules for plurality constructs:
//! - Alter definitions: `alter Name: Category { ... }`
//! - Alter blocks: `alter Name { ... }`
//! - Switch expressions: `switch to Alter { ... }` or `switch! to Alter { ... }`
//! - Headspace definitions: `headspace Name { ... }`
//! - Co-con channels: `cocon<A, B> name { ... }`
//! - Reality definitions: `reality entity Name { ... }`
//! - Split expressions: `split! from Alter { ... }`
//! - Trigger handlers: `on trigger Name { ... } where condition { ... }`

use crate::ast::{Block, Expr, Generics, Ident, Param, TypeExpr, Visibility, WhereClause};
use crate::lexer::Token;
use crate::parser::{ParseError, ParseResult, Parser};
use crate::span::Span;

use super::ast::*;
use super::lexer::{AlterSourceMarker, ForcedOperation, PluralityTokenStream};

// ============================================================================
// PARSER EXTENSION TRAIT
// ============================================================================

/// Extension trait for parsing plurality constructs
pub trait PluralityParser {
    // Top-level items
    fn parse_plurality_item(&mut self) -> ParseResult<PluralityItem>;
    fn parse_alter_def(&mut self, visibility: Visibility) -> ParseResult<AlterDef>;
    fn parse_headspace_def(&mut self, visibility: Visibility) -> ParseResult<HeadspaceDef>;
    fn parse_reality_def(&mut self, visibility: Visibility) -> ParseResult<RealityDef>;
    fn parse_cocon_channel(&mut self) -> ParseResult<CoConChannel>;
    fn parse_trigger_handler(&mut self) -> ParseResult<TriggerHandler>;

    // Expressions
    fn parse_plurality_expr(&mut self) -> ParseResult<PluralityExpr>;
    fn parse_alter_block(&mut self) -> ParseResult<AlterBlock>;
    fn parse_switch_expr(&mut self) -> ParseResult<SwitchExpr>;
    fn parse_split_expr(&mut self) -> ParseResult<SplitExpr>;

    // Type extensions
    fn parse_alter_sourced_type(&mut self) -> ParseResult<AlterSourcedType>;

    // Helpers
    fn parse_alter_expr(&mut self) -> ParseResult<AlterExpr>;
    fn parse_alter_source(&mut self) -> ParseResult<Option<AlterSource>>;
    fn try_parse_alter_source_marker(&mut self) -> Option<AlterSourceMarker>;
}

impl<'a> PluralityParser for Parser<'a> {
    // ========================================================================
    // TOP-LEVEL ITEMS
    // ========================================================================

    /// Parse a top-level plurality item
    fn parse_plurality_item(&mut self) -> ParseResult<PluralityItem> {
        let visibility = self.parse_visibility()?;

        match self.current_token() {
            Some(Token::Alter) => {
                Ok(PluralityItem::Alter(self.parse_alter_def(visibility)?))
            }
            Some(Token::Headspace) => {
                Ok(PluralityItem::Headspace(self.parse_headspace_def(visibility)?))
            }
            Some(Token::Reality) => {
                Ok(PluralityItem::Reality(self.parse_reality_def(visibility)?))
            }
            Some(Token::CoCon) => {
                Ok(PluralityItem::CoConChannel(self.parse_cocon_channel()?))
            }
            Some(Token::On) => Ok(PluralityItem::TriggerHandler(self.parse_trigger_handler()?)),
            Some(t) => Err(ParseError::UnexpectedToken {
                expected: "plurality item (alter, headspace, reality, cocon, or trigger handler)"
                    .to_string(),
                found: t.clone(),
                span: self.current_span(),
            }),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    /// Parse an alter definition
    ///
    /// ```sigil
    /// alter Abaddon: Council {
    ///     archetype: Goetia::Abaddon,
    ///     preferred_reality: RealityLayer::Fractured,
    ///     abilities: [...],
    ///     triggers: [...],
    ///     anima: { ... },
    ///     states: { ... }
    /// }
    /// ```
    fn parse_alter_def(&mut self, visibility: Visibility) -> ParseResult<AlterDef> {
        let start = self.current_span();

        // Expect 'alter' keyword
        self.expect(Token::Alter)?;

        // Parse alter name
        let name = self.parse_ident()?;

        // Parse optional category (: Council, : Servant, etc.)
        let category = if self.consume_if(&Token::Colon) {
            let cat_ident = self.parse_ident()?;
            AlterCategory::from_ident(&cat_ident)
        } else {
            AlterCategory::Custom
        };

        // Parse optional generics
        let generics = self.parse_generics_opt()?;

        // Parse optional where clause
        let where_clause = self.parse_where_clause_opt()?;

        // Parse body
        self.expect(Token::LBrace)?;
        let body = self.parse_alter_body()?;
        self.expect(Token::RBrace)?;

        let end = self.current_span();

        Ok(AlterDef {
            visibility,
            attrs: Vec::new(),
            name,
            category,
            generics,
            where_clause,
            body,
            span: start.merge(end),
        })
    }

    /// Parse a headspace definition
    ///
    /// ```sigil
    /// headspace InnerWorld {
    ///     location Citadel: Sanctuary { ... }
    ///     fn navigate(from: Location, to: Location) -> NavigationResult@? { ... }
    /// }
    /// ```
    fn parse_headspace_def(&mut self, visibility: Visibility) -> ParseResult<HeadspaceDef> {
        let start = self.current_span();

        self.expect(Token::Headspace)?;
        let name = self.parse_ident()?;

        self.expect(Token::LBrace)?;

        let mut locations = Vec::new();
        let mut methods = Vec::new();

        while !self.check(&Token::RBrace) && !self.is_eof() {
            self.skip_comments();

            if self.check(&Token::Location) {
                locations.push(self.parse_location_def()?);
            } else if self.check(&Token::Fn) || self.check(&Token::Async) {
                methods.push(self.parse_alter_method()?);
            } else if self.check(&Token::RBrace) {
                break;
            } else {
                return Err(ParseError::UnexpectedToken {
                    expected: "location or method".to_string(),
                    found: self.current_token().cloned().unwrap_or(Token::RBrace),
                    span: self.current_span(),
                });
            }
        }

        self.expect(Token::RBrace)?;
        let end = self.current_span();

        Ok(HeadspaceDef {
            visibility,
            name,
            locations,
            methods,
            span: start.merge(end),
        })
    }

    /// Parse a reality definition
    ///
    /// ```sigil
    /// reality entity Church {
    ///     layer Grounded { ... }
    ///     layer Fractured { ... }
    ///     transform Grounded -> Fractured: on perception > 0.5,
    /// }
    /// ```
    fn parse_reality_def(&mut self, visibility: Visibility) -> ParseResult<RealityDef> {
        let start = self.current_span();

        self.expect(Token::Reality)?;
        self.expect_ident("entity")?;
        let name = self.parse_ident()?;

        self.expect(Token::LBrace)?;

        let mut layers = Vec::new();
        let mut transforms = Vec::new();

        while !self.check(&Token::RBrace) && !self.is_eof() {
            self.skip_comments();

            if self.check(&Token::Layer) {
                layers.push(self.parse_reality_layer()?);
            } else if self.check_ident("transform") {
                transforms.push(self.parse_reality_transform()?);
            } else if self.check(&Token::RBrace) {
                break;
            } else {
                return Err(ParseError::UnexpectedToken {
                    expected: "layer or transform".to_string(),
                    found: self.current_token().cloned().unwrap_or(Token::RBrace),
                    span: self.current_span(),
                });
            }
        }

        self.expect(Token::RBrace)?;
        let end = self.current_span();

        Ok(RealityDef {
            visibility,
            name,
            layers,
            transforms,
            span: start.merge(end),
        })
    }

    /// Parse a co-conscious channel
    ///
    /// ```sigil
    /// cocon<Stolas, Paimon> knowledge_share {
    ///     fn share_discovery(info: Knowledge!) -> Acknowledgment~ { ... }
    /// }
    /// ```
    fn parse_cocon_channel(&mut self) -> ParseResult<CoConChannel> {
        let start = self.current_span();

        self.expect(Token::CoCon)?;
        self.expect(Token::Lt)?;

        // Parse participating alters
        let mut participants = Vec::new();
        participants.push(self.parse_ident()?);
        while self.consume_if(&Token::Comma) {
            if self.check(&Token::Gt) {
                break;
            }
            participants.push(self.parse_ident()?);
        }
        self.expect_gt()?;

        let name = self.parse_ident()?;
        let body = self.parse_block()?;
        let end = self.current_span();

        Ok(CoConChannel {
            participants,
            name,
            body,
            span: start.merge(end),
        })
    }

    /// Parse a trigger handler
    ///
    /// ```sigil
    /// on trigger ThreatDetected { level: threat } where threat > 0.9 {
    ///     switch! to Abaddon { ... }
    /// }
    /// ```
    fn parse_trigger_handler(&mut self) -> ParseResult<TriggerHandler> {
        let start = self.current_span();

        self.expect(Token::On)?;
        self.expect(Token::Trigger)?;

        // Parse trigger pattern
        let pattern = self.parse_trigger_pattern()?;

        // Parse optional guard
        let guard = if self.consume_if(&Token::Where) {
            Some(self.parse_expr()?)
        } else {
            None
        };

        // Parse handler body
        let body = self.parse_block()?;
        let end = self.current_span();

        Ok(TriggerHandler {
            pattern,
            guard,
            body,
            span: start.merge(end),
        })
    }

    // ========================================================================
    // EXPRESSIONS
    // ========================================================================

    /// Parse a plurality-specific expression
    fn parse_plurality_expr(&mut self) -> ParseResult<PluralityExpr> {
        match self.current_token() {
            Some(Token::Alter) => {
                Ok(PluralityExpr::AlterBlock(self.parse_alter_block()?))
            }
            Some(Token::Switch) => {
                Ok(PluralityExpr::Switch(self.parse_switch_expr()?))
            }
            Some(Token::Split) => {
                Ok(PluralityExpr::Split(self.parse_split_expr()?))
            }
            Some(t) => Err(ParseError::UnexpectedToken {
                expected: "plurality expression (alter, switch, or split)".to_string(),
                found: t.clone(),
                span: self.current_span(),
            }),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    /// Parse an alter block (scoped fronting)
    ///
    /// ```sigil
    /// alter Abaddon {
    ///     let threat = perceive_entity(enemy);
    /// }
    /// ```
    fn parse_alter_block(&mut self) -> ParseResult<AlterBlock> {
        let start = self.current_span();

        self.expect(Token::Alter)?;
        let alter = self.parse_alter_expr()?;
        let body = self.parse_block()?;
        let end = self.current_span();

        Ok(AlterBlock {
            alter,
            body,
            span: start.merge(end),
        })
    }

    /// Parse a switch expression
    ///
    /// ```sigil
    /// switch to Beleth {
    ///     reason: SwitchReason::TacticalNeed,
    ///     urgency: 0.8,
    ///     then: { ... },
    ///     else: { ... },
    /// }
    /// ```
    fn parse_switch_expr(&mut self) -> ParseResult<SwitchExpr> {
        let start = self.current_span();

        self.expect(Token::Switch)?;

        // Check for forced switch (switch!)
        let forced = self.consume_if(&Token::Bang);

        self.expect(Token::To)?;
        let target = self.parse_alter_expr()?;

        // Parse switch configuration block
        let config = if self.check(&Token::LBrace) {
            self.parse_switch_config()?
        } else {
            SwitchConfig::default()
        };

        let end = self.current_span();

        Ok(SwitchExpr {
            forced,
            target,
            config,
            span: start.merge(end),
        })
    }

    /// Parse a split expression
    ///
    /// ```sigil
    /// split! from trauma.primary_holder {
    ///     purpose: SplitPurpose::TraumaHolder,
    ///     memories: inherited_memories(),
    ///     traits: possibly_inverted_traits(),
    /// }
    /// ```
    fn parse_split_expr(&mut self) -> ParseResult<SplitExpr> {
        let start = self.current_span();

        self.expect(Token::Split)?;
        self.expect(Token::Bang)?; // split! is always forced
        self.expect(Token::From)?;

        let parent = self.parse_alter_expr()?;
        let config = self.parse_split_config()?;
        let end = self.current_span();

        Ok(SplitExpr {
            parent,
            config,
            span: start.merge(end),
        })
    }

    // ========================================================================
    // TYPE EXTENSIONS
    // ========================================================================

    /// Parse a type with optional alter-source marker
    ///
    /// ```sigil
    /// WorldState@!     // Fronting alter's view
    /// Vec<Voice>@~     // Co-conscious report
    /// Memory@?         // Dormant alter's uncertain memory
    /// State@‽          // Blended paradoxical state
    /// ```
    fn parse_alter_sourced_type(&mut self) -> ParseResult<AlterSourcedType> {
        let start = self.current_span();
        let inner = self.parse_type()?;

        // Check for alter-source marker
        if let Some(marker) = self.try_parse_alter_source_marker() {
            let end = self.current_span();
            let alter_source = match marker {
                AlterSourceMarker::Fronting => AlterSource::Fronting,
                AlterSourceMarker::CoCon => AlterSource::CoConscious(None),
                AlterSourceMarker::Dormant => AlterSource::Dormant(None),
                AlterSourceMarker::Blended => AlterSource::Blended(Vec::new()),
            };

            Ok(AlterSourcedType {
                inner,
                alter_source,
                span: start.merge(end),
            })
        } else {
            // No alter-source, return with Fronting as default
            let end = self.current_span();
            Ok(AlterSourcedType {
                inner,
                alter_source: AlterSource::Fronting,
                span: start.merge(end),
            })
        }
    }

    // ========================================================================
    // HELPERS
    // ========================================================================

    /// Parse an alter expression (reference to an alter)
    fn parse_alter_expr(&mut self) -> ParseResult<AlterExpr> {
        match self.current_token() {
            Some(Token::Ident(_)) => {
                let ident = self.parse_ident()?;

                // Check for method call like `council·fronter()`
                if self.check(&Token::MiddleDot) {
                    self.advance();
                    let method = self.parse_ident()?;
                    if self.check(&Token::LParen) {
                        // Parse method call arguments
                        self.advance(); // consume (
                        let mut args = Vec::new();
                        while !self.check(&Token::RParen) && !self.is_eof() {
                            args.push(self.parse_expr()?);
                            if !self.consume_if(&Token::Comma) {
                                break;
                            }
                        }
                        self.expect(Token::RParen)?;

                        // Build method call expression
                        let receiver_path = crate::ast::TypePath {
                            segments: vec![crate::ast::PathSegment {
                                ident: ident.clone(),
                                generics: None,
                            }],
                        };
                        let call_expr = Expr::MethodCall {
                            receiver: Box::new(Expr::Path(receiver_path)),
                            method,
                            type_args: None,
                            args,
                        };
                        return Ok(AlterExpr::CurrentFronter(Box::new(call_expr)));
                    }
                    // Reconstruct as path expression with combined name
                    let combined_path = crate::ast::TypePath {
                        segments: vec![crate::ast::PathSegment {
                            ident: Ident {
                                name: format!("{}.{}", ident.name, method.name),
                                evidentiality: None,
                                affect: None,
                                span: ident.span.merge(method.span),
                            },
                            generics: None,
                        }],
                    };
                    return Ok(AlterExpr::Expr(Box::new(Expr::Path(combined_path))));
                }

                Ok(AlterExpr::Named(ident))
            }
            Some(_) => {
                // Parse as general expression
                let expr = self.parse_expr()?;
                Ok(AlterExpr::Expr(Box::new(expr)))
            }
            None => Err(ParseError::UnexpectedEof),
        }
    }

    /// Parse an alter-source annotation (@!, @~, @?, @‽, or @AlterName)
    fn parse_alter_source(&mut self) -> ParseResult<Option<AlterSource>> {
        if !self.check(&Token::At) {
            return Ok(None);
        }

        self.advance(); // consume @

        match self.current_token() {
            Some(Token::Bang) => {
                self.advance();
                Ok(Some(AlterSource::Fronting))
            }
            Some(Token::Tilde) => {
                self.advance();
                Ok(Some(AlterSource::CoConscious(None)))
            }
            Some(Token::Question) => {
                self.advance();
                Ok(Some(AlterSource::Dormant(None)))
            }
            Some(Token::Interrobang) => {
                self.advance();
                Ok(Some(AlterSource::Blended(Vec::new())))
            }
            Some(Token::Ident(_)) => {
                let ident = self.parse_ident()?;
                match ident.name.as_str() {
                    "Fronting" => Ok(Some(AlterSource::Fronting)),
                    "CoCon" => Ok(Some(AlterSource::CoConscious(None))),
                    "Dormant" => Ok(Some(AlterSource::Dormant(None))),
                    "Blended" => Ok(Some(AlterSource::Blended(Vec::new()))),
                    _ => Ok(Some(AlterSource::Named(ident))),
                }
            }
            _ => Ok(None),
        }
    }

    /// Try to parse an alter-source marker if present
    fn try_parse_alter_source_marker(&mut self) -> Option<AlterSourceMarker> {
        if !self.check(&Token::At) {
            return None;
        }

        // Peek at next token
        let next = self.peek_next()?;
        let marker = match next {
            Token::Bang => Some(AlterSourceMarker::Fronting),
            Token::Tilde => Some(AlterSourceMarker::CoCon),
            Token::Question => Some(AlterSourceMarker::Dormant),
            Token::Interrobang => Some(AlterSourceMarker::Blended),
            _ => None,
        };

        if marker.is_some() {
            self.advance(); // consume @
            self.advance(); // consume marker
        }

        marker
    }
}

// ============================================================================
// PRIVATE PARSER HELPERS
// ============================================================================

impl<'a> Parser<'a> {
    /// Check if current token is a specific identifier
    fn check_ident(&self, name: &str) -> bool {
        matches!(self.current_token(), Some(Token::Ident(s)) if s == name)
    }

    /// Expect a specific identifier
    fn expect_ident(&mut self, name: &str) -> ParseResult<Span> {
        match self.current_token() {
            Some(Token::Ident(s)) if s == name => {
                let span = self.current_span();
                self.advance();
                Ok(span)
            }
            Some(t) => Err(ParseError::UnexpectedToken {
                expected: format!("identifier '{}'", name),
                found: t.clone(),
                span: self.current_span(),
            }),
            None => Err(ParseError::UnexpectedEof),
        }
    }

    /// Parse the body of an alter definition
    fn parse_alter_body(&mut self) -> ParseResult<AlterBody> {
        let mut body = AlterBody {
            archetype: None,
            preferred_reality: None,
            abilities: Vec::new(),
            triggers: Vec::new(),
            anima: None,
            states: None,
            special: Vec::new(),
            methods: Vec::new(),
            types: Vec::new(),
        };

        while !self.check(&Token::RBrace) && !self.is_eof() {
            self.skip_comments();

            match self.current_token() {
                // Handle keyword tokens for anima and states
                Some(Token::Anima) => {
                    self.advance();
                    self.expect(Token::Colon)?;
                    body.anima = Some(self.parse_anima_config()?);
                    self.consume_if(&Token::Comma);
                }
                Some(Token::States) => {
                    self.advance();
                    self.expect(Token::Colon)?;
                    body.states = Some(self.parse_alter_state_machine()?);
                    self.consume_if(&Token::Comma);
                }
                // Handle identifier-based fields
                Some(Token::Ident(s)) => match s.as_str() {
                    "archetype" => {
                        self.advance();
                        self.expect(Token::Colon)?;
                        body.archetype = Some(self.parse_expr()?);
                        self.consume_if(&Token::Comma);
                    }
                    "preferred_reality" => {
                        self.advance();
                        self.expect(Token::Colon)?;
                        body.preferred_reality = Some(self.parse_expr()?);
                        self.consume_if(&Token::Comma);
                    }
                    "abilities" => {
                        self.advance();
                        self.expect(Token::Colon)?;
                        body.abilities = self.parse_bracketed_expr_list()?;
                        self.consume_if(&Token::Comma);
                    }
                    "triggers" => {
                        self.advance();
                        self.expect(Token::Colon)?;
                        body.triggers = self.parse_bracketed_expr_list()?;
                        self.consume_if(&Token::Comma);
                    }
                    _ => {
                        // Unknown field - error
                        return Err(ParseError::UnexpectedToken {
                            expected: "alter body field".to_string(),
                            found: Token::Ident(s.clone()),
                            span: self.current_span(),
                        });
                    }
                },
                Some(Token::Fn) | Some(Token::Async) => {
                    body.methods.push(self.parse_alter_method()?);
                }
                Some(Token::Type) => {
                    body.types.push(self.parse_alter_type_alias()?);
                }
                Some(Token::RBrace) => break,
                Some(t) => {
                    return Err(ParseError::UnexpectedToken {
                        expected: "alter body field or method".to_string(),
                        found: t.clone(),
                        span: self.current_span(),
                    });
                }
                None => return Err(ParseError::UnexpectedEof),
            }
        }

        Ok(body)
    }

    /// Parse anima configuration
    fn parse_anima_config(&mut self) -> ParseResult<AnimaConfig> {
        self.expect(Token::LBrace)?;

        let mut config = AnimaConfig {
            base_arousal: None,
            base_dominance: None,
            expressiveness: None,
            susceptibility: None,
            extra: Vec::new(),
        };

        while !self.check(&Token::RBrace) && !self.is_eof() {
            self.skip_comments();

            let field_name = self.parse_ident()?;
            self.expect(Token::Colon)?;
            let value = self.parse_expr()?;

            match field_name.name.as_str() {
                "base_arousal" => config.base_arousal = Some(value),
                "base_dominance" => config.base_dominance = Some(value),
                "expressiveness" => config.expressiveness = Some(value),
                "susceptibility" => config.susceptibility = Some(value),
                _ => config.extra.push((field_name, value)),
            }

            self.consume_if(&Token::Comma);
        }

        self.expect(Token::RBrace)?;
        Ok(config)
    }

    /// Parse alter state machine
    fn parse_alter_state_machine(&mut self) -> ParseResult<AlterStateMachine> {
        self.expect(Token::LBrace)?;

        let mut transitions = Vec::new();

        while !self.check(&Token::RBrace) && !self.is_eof() {
            self.skip_comments();
            transitions.push(self.parse_alter_transition()?);
            self.consume_if(&Token::Comma);
        }

        self.expect(Token::RBrace)?;
        Ok(AlterStateMachine { transitions })
    }

    /// Parse a single state transition
    /// `Dormant -> Stirring: on Trigger::match`
    fn parse_alter_transition(&mut self) -> ParseResult<AlterTransition> {
        let from_ident = self.parse_ident()?;
        let from = AlterState::from_ident(&from_ident);

        self.expect(Token::Arrow)?;

        let to_ident = self.parse_ident()?;
        let to = AlterState::from_ident(&to_ident);

        self.expect(Token::Colon)?;
        self.expect(Token::On)?;

        let on = self.parse_expr()?;

        // Optional guard
        let guard = if self.check(&Token::Where) {
            self.advance();
            Some(self.parse_expr()?)
        } else {
            None
        };

        // Optional action block
        let action = if self.check(&Token::LBrace) {
            Some(self.parse_block()?)
        } else {
            None
        };

        Ok(AlterTransition {
            from,
            to,
            on,
            guard,
            action,
        })
    }

    /// Parse an alter method
    fn parse_alter_method(&mut self) -> ParseResult<AlterMethod> {
        let visibility = self.parse_visibility()?;
        let is_async = self.consume_if(&Token::Async);
        self.expect(Token::Fn)?;
        let name = self.parse_ident()?;

        self.expect(Token::LParen)?;
        let params = self.parse_params()?;
        self.expect(Token::RParen)?;

        let return_type = if self.consume_if(&Token::Arrow) {
            Some(self.parse_type()?)
        } else {
            None
        };

        let body = if self.check(&Token::LBrace) {
            Some(self.parse_block()?)
        } else {
            self.consume_if(&Token::Semi);
            None
        };

        Ok(AlterMethod {
            visibility,
            is_async,
            name,
            params,
            return_type,
            body,
        })
    }

    /// Parse an alter type alias
    fn parse_alter_type_alias(&mut self) -> ParseResult<AlterTypeAlias> {
        let visibility = self.parse_visibility()?;
        self.expect(Token::Type)?;
        let name = self.parse_ident()?;
        self.expect(Token::Eq)?;
        let ty = self.parse_type()?;
        self.consume_if(&Token::Semi);

        Ok(AlterTypeAlias {
            visibility,
            name,
            ty,
        })
    }

    /// Parse a location definition in headspace
    fn parse_location_def(&mut self) -> ParseResult<LocationDef> {
        self.expect(Token::Location)?;
        let name = self.parse_ident()?;
        self.expect(Token::Colon)?;
        let location_type = self.parse_ident()?;

        self.expect(Token::LBrace)?;

        let mut fields = Vec::new();
        let mut connections = Vec::new();
        let mut hazards = Vec::new();

        while !self.check(&Token::RBrace) && !self.is_eof() {
            self.skip_comments();

            let field_name = self.parse_ident()?;
            self.expect(Token::Colon)?;

            match field_name.name.as_str() {
                "connections" => {
                    // Parse array of stream definitions
                    self.expect(Token::LBracket)?;
                    while !self.check(&Token::RBracket) {
                        connections.push(self.parse_stream_def()?);
                        self.consume_if(&Token::Comma);
                    }
                    self.expect(Token::RBracket)?;
                }
                "hazards" => {
                    hazards = self.parse_bracketed_expr_list()?;
                }
                _ => {
                    let value = self.parse_expr()?;
                    fields.push((field_name, value));
                }
            }

            self.consume_if(&Token::Comma);
        }

        self.expect(Token::RBrace)?;

        Ok(LocationDef {
            name,
            location_type,
            fields,
            connections,
            hazards,
        })
    }

    /// Parse a stream definition
    /// `stream!(Target, bidirectional: true, locked: false)`
    fn parse_stream_def(&mut self) -> ParseResult<StreamDef> {
        // Could be `stream!(...)` macro or inline definition
        let target = self.parse_ident()?;

        let mut bidirectional = false;
        let mut locked = false;
        let mut content = None;

        // Check for options
        if self.consume_if(&Token::Comma) {
            while !self.check(&Token::RParen)
                && !self.check(&Token::RBracket)
                && !self.check(&Token::Comma)
                && !self.is_eof()
            {
                let opt_name = self.parse_ident()?;
                self.expect(Token::Colon)?;

                match opt_name.name.as_str() {
                    "bidirectional" => {
                        bidirectional = self.consume_if(&Token::True);
                        if !bidirectional {
                            self.consume_if(&Token::False);
                        }
                    }
                    "locked" => {
                        locked = self.consume_if(&Token::True);
                        if !locked {
                            self.consume_if(&Token::False);
                        }
                    }
                    "content" => {
                        content = Some(self.parse_expr()?);
                    }
                    _ => {}
                }

                if !self.consume_if(&Token::Comma) {
                    break;
                }
            }
        }

        Ok(StreamDef {
            target,
            content,
            bidirectional,
            locked,
        })
    }

    /// Parse a reality layer
    fn parse_reality_layer(&mut self) -> ParseResult<RealityLayer> {
        self.expect(Token::Layer)?;
        let name = self.parse_ident()?;

        self.expect(Token::LBrace)?;

        let mut fields = Vec::new();

        while !self.check(&Token::RBrace) && !self.is_eof() {
            self.skip_comments();

            let field_name = self.parse_ident()?;
            self.expect(Token::Colon)?;
            let value = self.parse_expr()?;
            fields.push((field_name, value));

            self.consume_if(&Token::Comma);
        }

        self.expect(Token::RBrace)?;

        Ok(RealityLayer { name, fields })
    }

    /// Parse a reality transform rule
    fn parse_reality_transform(&mut self) -> ParseResult<RealityTransform> {
        self.expect_ident("transform")?;
        let from = self.parse_ident()?;
        self.expect(Token::Arrow)?;
        let to = self.parse_ident()?;
        self.expect(Token::Colon)?;
        self.expect(Token::On)?;
        let condition = self.parse_expr()?;
        self.consume_if(&Token::Comma);

        Ok(RealityTransform { from, to, condition })
    }

    /// Parse switch configuration
    fn parse_switch_config(&mut self) -> ParseResult<SwitchConfig> {
        self.expect(Token::LBrace)?;

        let mut config = SwitchConfig {
            reason: None,
            urgency: None,
            requires: None,
            then_block: None,
            else_block: None,
            emergency_block: None,
            bypass_deliberation: false,
        };

        while !self.check(&Token::RBrace) && !self.is_eof() {
            self.skip_comments();

            let field_name = self.parse_ident()?;
            self.expect(Token::Colon)?;

            match field_name.name.as_str() {
                "reason" => config.reason = Some(self.parse_expr()?),
                "urgency" => config.urgency = Some(self.parse_expr()?),
                "requires" => config.requires = Some(self.parse_expr()?),
                "then" => config.then_block = Some(self.parse_block()?),
                "else" => config.else_block = Some(self.parse_block()?),
                "emergency" => config.emergency_block = Some(self.parse_block()?),
                "bypass_deliberation" => {
                    config.bypass_deliberation = self.consume_if(&Token::True);
                    if !config.bypass_deliberation {
                        self.consume_if(&Token::False);
                    }
                }
                _ => {
                    // Skip unknown fields
                    let _ = self.parse_expr()?;
                }
            }

            self.consume_if(&Token::Comma);
        }

        self.expect(Token::RBrace)?;
        Ok(config)
    }

    /// Parse split configuration
    fn parse_split_config(&mut self) -> ParseResult<SplitConfig> {
        self.expect(Token::LBrace)?;

        let mut config = SplitConfig {
            purpose: None,
            memories: None,
            traits: None,
            extra: Vec::new(),
        };

        while !self.check(&Token::RBrace) && !self.is_eof() {
            self.skip_comments();

            let field_name = self.parse_ident()?;
            self.expect(Token::Colon)?;
            let value = self.parse_expr()?;

            match field_name.name.as_str() {
                "purpose" => config.purpose = Some(value),
                "memories" => config.memories = Some(value),
                "traits" => config.traits = Some(value),
                _ => config.extra.push((field_name, value)),
            }

            self.consume_if(&Token::Comma);
        }

        self.expect(Token::RBrace)?;
        Ok(config)
    }

    /// Parse trigger pattern
    fn parse_trigger_pattern(&mut self) -> ParseResult<TriggerPattern> {
        let trigger_type = self.parse_ident()?;

        let mut fields = Vec::new();

        if self.consume_if(&Token::LBrace) {
            while !self.check(&Token::RBrace) && !self.is_eof() {
                let field_name = self.parse_ident()?;
                self.expect(Token::Colon)?;
                let binding = self.parse_ident()?;
                fields.push((field_name, binding));

                if !self.consume_if(&Token::Comma) {
                    break;
                }
            }
            self.expect(Token::RBrace)?;
        }

        Ok(TriggerPattern {
            trigger_type,
            fields,
        })
    }

    /// Parse a list of expressions in brackets
    fn parse_bracketed_expr_list(&mut self) -> ParseResult<Vec<Expr>> {
        self.expect(Token::LBracket)?;

        let mut exprs = Vec::new();

        while !self.check(&Token::RBracket) && !self.is_eof() {
            exprs.push(self.parse_expr()?);
            if !self.consume_if(&Token::Comma) {
                break;
            }
        }

        self.expect(Token::RBracket)?;
        Ok(exprs)
    }
}

// ============================================================================
// DEFAULT IMPLEMENTATIONS
// ============================================================================

impl Default for SwitchConfig {
    fn default() -> Self {
        Self {
            reason: None,
            urgency: None,
            requires: None,
            then_block: None,
            else_block: None,
            emergency_block: None,
            bypass_deliberation: false,
        }
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_plurality(source: &str) -> ParseResult<PluralityItem> {
        let mut parser = Parser::new(source);
        parser.parse_plurality_item()
    }

    #[test]
    fn test_parse_alter_def_basic() {
        let source = r#"
            alter Abaddon: Council {
                archetype: Goetia·Abaddon,
            }
        "#;
        let result = parse_plurality(source);
        assert!(result.is_ok());
        if let Ok(PluralityItem::Alter(def)) = result {
            assert_eq!(def.name.name, "Abaddon");
            assert_eq!(def.category, AlterCategory::Council);
        }
    }

    #[test]
    fn test_parse_switch_expr() {
        let source = r#"
            alter test {
                switch to Beleth {
                    reason: SwitchReason·Combat,
                    urgency: 0.8,
                }
            }
        "#;
        // This would need full parser integration to test
    }
}
