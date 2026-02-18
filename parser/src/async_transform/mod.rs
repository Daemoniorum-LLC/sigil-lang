//! Async State Machine Transformation
//!
//! Transforms async functions with multiple await points into explicit state machines.
//! This is a frontend pass that produces `StateMachineIR`, which can be compiled by
//! any backend (WASM, LLVM, interpreter).
//!
//! See: docs/specs/ASYNC-STATE-MACHINE-SPEC.md

mod ir;
mod transform;

pub use ir::*;
pub use transform::*;

#[cfg(test)]
mod tests;
