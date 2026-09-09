//! React → Qliphoth migration.
//!
//! Extracts React/TSX component structure and generates migration specs
//! for agent-assisted migration to Qliphoth actors.
//!
//! # Architecture
//!
//! 1. **Extraction**: Parse React/TSX with swc, produce `ReactExtraction`
//! 2. **Spec Generation**: Enrich extraction with Qliphoth patterns, produce `MigrationSpec`
//! 3. **Code Generation**: Generate idiomatic Sigil/Qliphoth code from specs
//! 4. **MCP Interface**: Serve specs to agents, validate output
//!
//! See docs/specs/REACT-MIGRATION.md for full specification.

pub(crate) mod ast_transform;
mod cli;
pub(crate) mod extraction;
mod generator;
mod mcp;
mod spec;
pub(crate) mod stmt_transform;
#[cfg(test)]
mod tests;

pub use cli::*;
pub use extraction::*;
pub use generator::*;
pub use mcp::*;
pub use spec::*;
