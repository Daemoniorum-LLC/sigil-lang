//! What `@[cfg(…)]` is evaluated against.
//!
//! Conditional compilation used to be answered by asking the *host* — the
//! machine running `sigil`, through Rust's own `cfg!` — so `@[cfg(target_arch =
//! "wasm32")]` was false while compiling to WebAssembly on an x86 laptop, and
//! `@[cfg(feature = "gtk")]` was answered by guessing. A target is a property of
//! the compilation, not of the compiler, so it lives here and the CLI sets it
//! once per invocation.

use std::collections::BTreeSet;
use std::sync::{OnceLock, RwLock};

/// The set of `cfg` predicates that hold for this compilation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CfgContext {
    /// `target_family`: "wasm", "unix" or "windows".
    pub target_family: String,
    /// `target_arch`: "wasm32", "x86_64", "aarch64", …
    pub target_arch: String,
    /// `target_os`: "unknown" for wasm, else "linux" / "macos" / "windows".
    pub target_os: String,
    /// Features enabled for this build — `@[cfg(feature = "gtk")]`.
    pub features: BTreeSet<String>,
    /// `debug_assertions`.
    pub debug_assertions: bool,
    /// `test` — set only when building a test harness, as in Rust.
    pub test: bool,
}

impl CfgContext {
    /// The machine running the compiler. What `sigil run`, `sigil jit` and
    /// `sigil compile` target unless told otherwise.
    pub fn host() -> Self {
        let target_os = if cfg!(target_os = "linux") {
            "linux"
        } else if cfg!(target_os = "macos") {
            "macos"
        } else if cfg!(target_os = "windows") {
            "windows"
        } else {
            "unknown"
        };
        let target_arch = if cfg!(target_arch = "x86_64") {
            "x86_64"
        } else if cfg!(target_arch = "aarch64") {
            "aarch64"
        } else {
            "unknown"
        };
        let target_family = if target_os == "windows" { "windows" } else { "unix" };
        Self {
            target_family: target_family.to_string(),
            target_arch: target_arch.to_string(),
            target_os: target_os.to_string(),
            features: BTreeSet::new(),
            debug_assertions: true,
            test: false,
        }
    }

    /// What `sigil wasm` targets.
    pub fn wasm32() -> Self {
        Self {
            target_family: "wasm".to_string(),
            target_arch: "wasm32".to_string(),
            target_os: "unknown".to_string(),
            features: BTreeSet::new(),
            debug_assertions: true,
            test: false,
        }
    }

    pub fn with_features<I, S>(mut self, features: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.features = features.into_iter().map(Into::into).collect();
        self
    }

    /// Whether a bare predicate — `@[cfg(unix)]`, `@[cfg(test)]` — holds.
    ///
    /// A name this compiler does not know is **not set**, the same as in Rust.
    /// Answering "true" for anything unrecognised, which is what this used to
    /// do, means `@[cfg(gtk_backend)]` and `@[cfg(not(gtk_backend))]` are both
    /// included and the second definition wins.
    pub fn holds_ident(&self, name: &str) -> bool {
        match name {
            "debug_assertions" => self.debug_assertions,
            "test" => self.test,
            "unix" | "windows" | "wasm" => self.target_family == name,
            _ => false,
        }
    }

    /// Whether `key = "value"` holds.
    pub fn holds_key_value(&self, key: &str, value: &str) -> bool {
        match key {
            "target_family" => self.target_family == value,
            "target_arch" => self.target_arch == value,
            "target_os" => self.target_os == value,
            "feature" => self.features.contains(value),
            _ => false,
        }
    }
}

impl Default for CfgContext {
    fn default() -> Self {
        Self::host()
    }
}

fn slot() -> &'static RwLock<CfgContext> {
    static ACTIVE: OnceLock<RwLock<CfgContext>> = OnceLock::new();
    ACTIVE.get_or_init(|| RwLock::new(CfgContext::host()))
}

/// The context every `@[cfg(…)]` in this process is evaluated against.
///
/// Process-global rather than threaded through the parser because one
/// invocation of `sigil` compiles for one target, and the parser is constructed
/// in more than a hundred places — a parameter none of them could answer.
pub fn active() -> CfgContext {
    slot().read().map(|c| c.clone()).unwrap_or_else(|_| CfgContext::host())
}

/// Set the target for this compilation. Called once, by the CLI, before parsing.
pub fn set_active(cfg: CfgContext) {
    if let Ok(mut slot) = slot().write() {
        *slot = cfg;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unknown_predicates_are_not_set() {
        let cfg = CfgContext::wasm32();
        assert!(!cfg.holds_ident("gtk_backend"));
        assert!(!cfg.holds_key_value("target_pointer_width", "32"));
    }

    #[test]
    fn wasm_target_is_not_the_host() {
        let cfg = CfgContext::wasm32();
        assert!(cfg.holds_key_value("target_arch", "wasm32"));
        assert!(!cfg.holds_key_value("target_os", "linux"));
        assert!(cfg.holds_ident("wasm"));
        assert!(!cfg.holds_ident("unix"));
    }

    #[test]
    fn features_come_from_the_build() {
        let cfg = CfgContext::wasm32().with_features(["gtk"]);
        assert!(cfg.holds_key_value("feature", "gtk"));
        assert!(!cfg.holds_key_value("feature", "ssr"));
    }
}
