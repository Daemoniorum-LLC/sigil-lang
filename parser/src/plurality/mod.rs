// Allow ambiguous glob re-exports for duplicate types (AlterCategory, CoConChannel, RealityLayer)
// defined in both ast and runtime modules. These are intentionally the same types.
// TODO(design): Consolidate duplicate type definitions to a single canonical location.
#![allow(ambiguous_glob_reexports)]

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

// Language extension modules (parser components)
pub mod ast;
pub mod codegen;
pub mod lexer;
pub mod parser;
pub mod runtime;
pub mod typeck;

// Re-exports
pub use ast::*;
pub use codegen::*;
pub use lexer::*;
pub use parser::*;
pub use runtime::*;
pub use typeck::*;

// NOTE: Game engine modules (combat, dialogue, game_loop, perception, save_system)
// have been extracted to aether-framework. See docs/specs/PLURALITY-EXTRACTION-SPEC.md
