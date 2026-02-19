//! Async State Machine Transformation
//!
//! Transforms async functions with multiple await points into explicit state machines.
//! This is a frontend pass that produces `StateMachineIR`, which can be compiled by
//! any backend (WASM, LLVM, interpreter).
//!
//! The transformation pipeline is:
//! 1. **Flatten** - Hoist complex await expressions to simple let-bindings
//! 2. **Transform** - Convert to state machine IR
//!
//! See: docs/specs/ASYNC-STATE-MACHINE-SPEC.md
//! See: docs/specs/AWAIT-EXPRESSION-FLATTENING-SPEC.md

mod flatten;
mod ir;
mod transform;

pub use flatten::*;
pub use ir::*;
pub use transform::*;

#[cfg(test)]
mod tests;
