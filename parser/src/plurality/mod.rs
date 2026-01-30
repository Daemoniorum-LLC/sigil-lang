//! # Plurality Extensions for Sigil
//!
//! This module extends Sigil's type system to support plurality mechanics
//! for the DAEMONIORUM game engine. It builds on Sigil's existing evidentiality
//! system (!~?‽) to add alter-source tracking.
//!
//! ## Core Concepts
//!
//! - **Alter-Sourcing**: Track which alter perceives/controls data
//! - **Fronting State**: First-class tracking of who is fronting
//! - **Co-consciousness**: Channels between alters
//! - **Split Mechanics**: Trauma-based alter creation
//!
//! ## New Syntax
//!
//! ```sigil
//! // Alter definition
//! alter Abaddon: Council { ... }
//!
//! // Alter block (scoped fronting)
//! alter Abaddon { ... }
//!
//! // Switch expression
//! switch to Beleth { reason: ..., then: ..., else: ... }
//!
//! // Co-conscious channel
//! cocon<Stolas, Paimon> { ... }
//!
//! // Reality layer
//! reality entity Church { layer Grounded { ... }, layer Fractured { ... } }
//!
//! // Headspace navigation
//! headspace InnerWorld { location Citadel { ... } }
//!
//! // Trauma split
//! split! from Abaddon { purpose: ..., memories: ... }
//! ```

pub mod ast;
pub mod codegen;
pub mod combat;
pub mod dialogue;
pub mod game_loop;
pub mod lexer;
pub mod parser;
pub mod perception;
pub mod runtime;
pub mod save_system;
pub mod typeck;

pub use ast::*;
pub use codegen::*;
pub use combat::*;
pub use dialogue::*;
pub use game_loop::*;
pub use lexer::*;
pub use parser::*;
pub use perception::*;
pub use runtime::*;
pub use save_system::*;
pub use typeck::*;
