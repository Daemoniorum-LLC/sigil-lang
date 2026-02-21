//! Migration tools for converting code to Sigil/Qliphoth.
//!
//! This module provides:
//! - Rust → Sigil migration (existing, in main.rs - to be moved here)
//! - React → Qliphoth migration (new, requires `react-migrate` feature)
//! - egui → Qliphoth migration (new, requires `egui-migrate` feature)

#[cfg(feature = "react-migrate")]
pub mod react;

#[cfg(feature = "egui-migrate")]
pub mod egui;
