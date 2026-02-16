//! React → Qliphoth migration.
//!
//! Extracts React/TSX component structure and generates migration specs
//! for agent-assisted migration to Qliphoth actors.
//!
//! # Architecture
//!
//! 1. **Extraction**: Parse React/TSX with swc, produce `ReactExtraction`
//! 2. **Spec Generation**: Enrich extraction with Qliphoth patterns, produce `MigrationSpec`
//! 3. **MCP Interface**: Serve specs to agents, validate output
//!
//! See docs/specs/REACT-MIGRATION.md for full specification.

mod extraction;
#[cfg(test)]
mod tests;

pub use extraction::*;
