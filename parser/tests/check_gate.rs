//! What `sigil check` is allowed to let through.
//!
//! `sigil check` is a gate — the Sigil migration work gates on it, and a green
//! check is read as "this will run". LARES-453 filed two cases where it was
//! green on code that cannot run:
//!
//! ```text
//! rite main() { ≔ x = totally_undefined_fn("a"); print(x); }
//!   sigil check → rc=0 "no errors"      sigil run → rc=1 [R0003] undefined variable
//! ```
//!
//! and a crate whose entry declared `☉ scroll registry;` over a `registry.sigil`
//! with a deliberate syntax error in it — `check registry.sigil` rc=1,
//! `check lib.sigil` rc=0, because the entry check never opened the file.
//!
//! Nothing tested that `check` caught anything, which is why both survived a
//! release and a prior pass over this very code (LARES-360). These tests are
//! that gate: each one fails against the behaviour as filed.

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

const SIGIL: &str = env!("CARGO_BIN_EXE_sigil");

/// A scratch crate directory that cleans itself up.
struct Crate {
    dir: PathBuf,
}

impl Crate {
    fn new(name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "sigil-check-gate-{}-{}-{:?}",
            name,
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("create scratch crate");
        Crate { dir }
    }

    fn file(&self, name: &str, contents: &str) -> PathBuf {
        let path = self.dir.join(name);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create module directory");
        }
        std::fs::write(&path, contents).expect("write source file");
        path
    }
}

impl Drop for Crate {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn check(path: &Path, extra: &[&str]) -> Output {
    Command::new(SIGIL)
        .arg("check")
        .arg(path)
        .args(extra)
        .output()
        .expect("run sigil check")
}

fn run(path: &Path) -> Output {
    Command::new(SIGIL)
        .arg("run")
        .arg(path)
        .output()
        .expect("run sigil run")
}

fn combined(out: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    )
}

// ---------------------------------------------------------------------------
// Call targets
// ---------------------------------------------------------------------------

/// The one-line reproduction from the ticket, verbatim.
#[test]
fn a_call_to_a_function_that_does_not_exist_is_an_error() {
    let c = Crate::new("undefined-call");
    let f = c.file(
        "main.sigil",
        "rite main() { ≔ x = totally_undefined_fn(\"a\"); print(x); }\n",
    );

    let out = check(&f, &[]);
    assert!(
        !out.status.success(),
        "check passed a call to a function that exists nowhere:\n{}",
        combined(&out)
    );
    assert!(
        combined(&out).contains("totally_undefined_fn"),
        "the diagnostic should name the unresolvable call:\n{}",
        combined(&out)
    );
}

/// The half of the ticket that makes the other half a defect rather than a
/// preference: the runtime already rejects this, so check disagreeing with it
/// is what turns a typo into a production failure.
#[test]
fn check_and_run_agree_that_an_undefined_call_is_wrong() {
    let c = Crate::new("check-run-agree");
    let f = c.file(
        "main.sigil",
        "rite main() { ≔ x = totally_undefined_fn(\"a\"); print(x); }\n",
    );

    let checked = check(&f, &[]);
    let ran = run(&f);
    assert!(
        !ran.status.success(),
        "expected the runtime to reject this; the premise of the test has changed:\n{}",
        combined(&ran)
    );
    assert_eq!(
        checked.status.success(),
        ran.status.success(),
        "check and run disagree about the same file\ncheck:\n{}\nrun:\n{}",
        combined(&checked),
        combined(&ran)
    );
}

/// Adding an import of a crate we cannot see must not be a way to silence the
/// check for names that file never imports — but see
/// `a_glob_import_holds_call_resolution`, which is the deliberate exception.
#[test]
fn a_named_import_does_not_excuse_an_unrelated_undefined_call() {
    let c = Crate::new("named-import");
    let f = c.file(
        "main.sigil",
        "invoke qliphoth·prelude·VNode;\nrite main() { ≔ x = totally_undefined_fn(\"a\"); print(x); }\n",
    );

    let out = check(&f, &[]);
    assert!(
        !out.status.success(),
        "an unrelated import silenced call resolution:\n{}",
        combined(&out)
    );
}

/// A call into a sibling module is not a typo. Resolution has to see the whole
/// crate before it is allowed to call anything undefined — this is the reason
/// name resolution was opt-in (`--strict`) before the module graph existed.
#[test]
fn a_call_into_a_sibling_module_resolves() {
    let c = Crate::new("sibling-call");
    c.file("helpers.sigil", "☉ rite helper(x: i64) -> i64 { ⤺ x + 1; }\n");
    let lib = c.file(
        "lib.sigil",
        "☉ scroll helpers;\n\nrite main() { ≔ y = helper(1); print(y); }\n",
    );

    let out = check(&lib, &[]);
    assert!(
        out.status.success(),
        "a call to a function defined in a sibling module was reported as undefined:\n{}",
        combined(&out)
    );
}

/// A locally bound closure is a legitimate call target.
#[test]
fn a_local_closure_is_a_call_target() {
    let c = Crate::new("closure-call");
    let f = c.file(
        "main.sigil",
        "rite main() { ≔ double = |x| x * 2; ≔ y = double(4); print(y); }\n",
    );

    let out = check(&f, &[]);
    assert!(
        out.status.success(),
        "a call to a closure bound in the same function was reported as undefined:\n{}",
        combined(&out)
    );
}

/// A glob import pulls in names this check cannot enumerate. Reporting calls as
/// unresolvable against a crate we only half know would blame working code, so
/// resolution stands down — deliberately, and only for calls.
#[test]
fn a_glob_import_holds_call_resolution() {
    let c = Crate::new("glob-import");
    let f = c.file(
        "main.sigil",
        "invoke qliphoth·prelude·*;\nrite main() { ≔ x = something_from_the_glob(\"a\"); print(x); }\n",
    );

    let out = check(&f, &[]);
    assert!(
        out.status.success(),
        "call resolution fired on a file that glob-imports names it cannot see:\n{}",
        combined(&out)
    );
}

/// The escape hatch has to actually restore the old behaviour, or it is not one.
#[test]
fn no_resolve_calls_restores_the_old_leniency() {
    let c = Crate::new("no-resolve-flag");
    let f = c.file(
        "main.sigil",
        "rite main() { ≔ x = totally_undefined_fn(\"a\"); print(x); }\n",
    );

    let out = check(&f, &["--no-resolve-calls"]);
    assert!(
        out.status.success(),
        "--no-resolve-calls still reported the call:\n{}",
        combined(&out)
    );
}

// ---------------------------------------------------------------------------
// Module traversal
// ---------------------------------------------------------------------------

/// The ticket as filed: `check registry.sigil` rc=1, `check lib.sigil` rc=0,
/// same error, same file.
#[test]
fn a_syntax_error_behind_a_scroll_fails_the_entry_check() {
    let c = Crate::new("scroll-syntax");
    let registry = c.file("registry.sigil", "rite broken( {\n    ≔ x = 1;\n}\n");
    let lib = c.file("lib.sigil", "☉ scroll registry;\n\nrite main() {\n    print(\"hi\");\n}\n");

    let direct = check(&registry, &[]);
    assert!(
        !direct.status.success(),
        "the premise has changed: the module no longer fails on its own:\n{}",
        combined(&direct)
    );

    let entry = check(&lib, &[]);
    assert!(
        !entry.status.success(),
        "checking the crate entry passed a syntax error one file below it:\n{}",
        combined(&entry)
    );
    assert!(
        combined(&entry).contains("registry.sigil"),
        "the diagnostic should name the file the error is in:\n{}",
        combined(&entry)
    );
}

/// An unresolvable call inside a module, reached through the entry — the
/// `zzzmod·zzzfn(&body)` case from the ticket, reduced.
#[test]
fn an_undefined_call_behind_a_scroll_fails_the_entry_check() {
    let c = Crate::new("scroll-call");
    c.file(
        "registry.sigil",
        "☉ rite register() { ≔ r = definitely_not_a_function(1); print(r); }\n",
    );
    let lib = c.file("lib.sigil", "☉ scroll registry;\n\nrite main() { print(\"hi\"); }\n");

    let out = check(&lib, &[]);
    assert!(
        !out.status.success(),
        "an undefined call one module below the entry passed the crate check:\n{}",
        combined(&out)
    );
}

/// `invoke tome·registry·…` reaches a sibling file just as `scroll` does, and
/// the interpreter loads it the same way, so the check has to follow both.
#[test]
fn invoke_tome_is_traversed_too() {
    let c = Crate::new("invoke-tome");
    c.file("registry.sigil", "rite broken( {\n    ≔ x = 1;\n}\n");
    let lib = c.file(
        "lib.sigil",
        "invoke tome·registry·Thing;\n\nrite main() { print(\"hi\"); }\n",
    );

    let out = check(&lib, &[]);
    assert!(
        !out.status.success(),
        "a module reached by `invoke tome·` went unchecked:\n{}",
        combined(&out)
    );
}

/// A module directory — `registry/mod.sigil` — is the other spelling.
#[test]
fn a_module_directory_is_traversed() {
    let c = Crate::new("module-dir");
    c.file("registry/mod.sigil", "rite broken( {\n    ≔ x = 1;\n}\n");
    let lib = c.file("lib.sigil", "☉ scroll registry;\n\nrite main() { print(\"hi\"); }\n");

    let out = check(&lib, &[]);
    assert!(
        !out.status.success(),
        "a module in its own directory went unchecked:\n{}",
        combined(&out)
    );
}

/// Modules two levels down are no less part of the crate.
#[test]
fn traversal_reaches_a_module_of_a_module() {
    let c = Crate::new("nested-module");
    c.file("deep.sigil", "rite broken( {\n    ≔ x = 1;\n}\n");
    c.file("mid.sigil", "☉ scroll deep;\n");
    let lib = c.file("lib.sigil", "☉ scroll mid;\n\nrite main() { print(\"hi\"); }\n");

    let out = check(&lib, &[]);
    assert!(
        !out.status.success(),
        "traversal stopped one level short of the broken file:\n{}",
        combined(&out)
    );
}

/// Two modules that declare each other must not hang the check.
#[test]
fn a_module_cycle_terminates() {
    let c = Crate::new("module-cycle");
    c.file("a.sigil", "☉ scroll b;\n");
    c.file("b.sigil", "☉ scroll a;\n");
    let lib = c.file("lib.sigil", "☉ scroll a;\n\nrite main() { print(\"hi\"); }\n");

    let out = check(&lib, &[]);
    assert!(
        out.status.success(),
        "a cycle between two clean modules should still check clean:\n{}",
        combined(&out)
    );
}

/// A `scroll` with no file beside it is a warning, not an error: cfg-gated
/// modules (`@[cfg(unix)] ☉ scroll native;`) legitimately have no file on this
/// target, and the interpreter carries on regardless.
#[test]
fn a_module_with_no_file_warns_without_failing() {
    let c = Crate::new("missing-module");
    let lib = c.file("lib.sigil", "☉ scroll nowhere;\n\nrite main() { print(\"hi\"); }\n");

    let out = check(&lib, &[]);
    assert!(
        out.status.success(),
        "a missing module file should warn, not fail the check:\n{}",
        combined(&out)
    );
    assert!(
        combined(&out).contains("nowhere"),
        "the missing module should at least be reported:\n{}",
        combined(&out)
    );
}

/// A clean crate stays clean — the gate is worthless if it cries wolf.
#[test]
fn a_clean_crate_checks_clean() {
    let c = Crate::new("clean-crate");
    c.file("helpers.sigil", "☉ rite helper(x: i64) -> i64 { ⤺ x + 1; }\n");
    let lib = c.file(
        "lib.sigil",
        "☉ scroll helpers;\n\nrite main() { ≔ y = helper(1); print(y); }\n",
    );

    let out = check(&lib, &[]);
    assert!(
        out.status.success(),
        "a crate with nothing wrong with it failed the check:\n{}",
        combined(&out)
    );
}

/// `--no-traverse` puts the check back on one file, for the case where
/// following a module makes things worse.
#[test]
fn no_traverse_checks_only_the_named_file() {
    let c = Crate::new("no-traverse-flag");
    c.file("registry.sigil", "rite broken( {\n    ≔ x = 1;\n}\n");
    let lib = c.file("lib.sigil", "☉ scroll registry;\n\nrite main() { print(\"hi\"); }\n");

    let out = check(&lib, &["--no-traverse"]);
    assert!(
        out.status.success(),
        "--no-traverse still followed the scroll:\n{}",
        combined(&out)
    );
}

// ---------------------------------------------------------------------------
// Output
// ---------------------------------------------------------------------------

/// JSON output is what agents read. Every diagnostic has to say which file it
/// is about, now that a check spans more than one.
#[test]
fn json_output_attributes_each_diagnostic_to_its_file() {
    let c = Crate::new("json-attribution");
    c.file("registry.sigil", "rite broken( {\n    ≔ x = 1;\n}\n");
    let lib = c.file("lib.sigil", "☉ scroll registry;\n\nrite main() { print(\"hi\"); }\n");

    let out = check(&lib, &["--format=compact"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let parsed: serde_json::Value =
        serde_json::from_str(stdout.trim()).expect("check emits valid JSON");

    assert_eq!(parsed["success"], serde_json::json!(false));
    let diagnostics = parsed["diagnostics"]
        .as_array()
        .expect("diagnostics is an array");
    assert!(
        diagnostics
            .iter()
            .any(|d| d["file"].as_str().unwrap_or("").ends_with("registry.sigil")),
        "no diagnostic was attributed to the module that is actually broken: {}",
        stdout
    );
    let files = parsed["files_checked"]
        .as_array()
        .expect("files_checked is an array");
    assert_eq!(
        files.len(),
        2,
        "both files should be reported as checked: {}",
        stdout
    );
}
