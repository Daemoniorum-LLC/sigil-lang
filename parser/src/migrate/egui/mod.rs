//! egui → Qliphoth migration.
//!
//! Extracts egui/Rust widget structure and generates migration specs
//! for agent-assisted migration to Qliphoth actors.
//!
//! # Architecture
//!
//! 1. **Extraction**: Parse Rust source with `syn`, produce `EguiExtraction`
//! 2. **Spec Generation**: Enrich extraction with Qliphoth patterns, produce `EguiMigrationSpec`
//! 3. **Code Generation**: Generate idiomatic Sigil/Qliphoth actor skeletons from specs
//! 4. **CLI Interface**: Single-file and batch modes, JSON spec output, dry-run
//!
//! # Usage
//!
//! ```bash
//! sigil migrate --from-egui crates/ide-gui/src/notifications.rs --output migration/
//! sigil migrate --from-egui crates/ide-gui/src/ --output migration/
//! sigil migrate --from-egui crates/ide-gui/src/ --dry-run
//! sigil migrate --status --output migration/
//! ```
//!
//! See the plan at `.claude/plans/curious-swimming-feather.md` for full specification.

mod cli;
mod extraction;
mod generator;
mod patterns;
mod spec;
#[cfg(test)]
mod tests;

pub use cli::*;
pub use extraction::*;
pub use generator::*;
pub use patterns::*;
pub use spec::*;
