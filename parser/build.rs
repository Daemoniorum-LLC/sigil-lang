// Bake this binary's own provenance in at build time.
//
// A shared build output is an undeclared input. `sigil` had no --version at
// all, so there was no way to ask a binary what it was, and the obvious path
// on a workstation can be a build from a branch 49 commits ahead of and 173
// behind develop with uncommitted changes. That happened, and it produced two
// confident corrections against work that was correct (#153).
//
// Everything here resolves from CARGO_MANIFEST_DIR -- the source tree being
// compiled -- and never from the current directory. Resolving from cwd would
// report whatever repository the *caller* happens to stand in, which is not
// merely useless: it would have described the wrong branch confidently, which
// is worse than reporting nothing.

use std::process::Command;

fn main() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default();

    let commit = git(&manifest_dir, &["rev-parse", "HEAD"]);
    let short = git(&manifest_dir, &["rev-parse", "--short", "HEAD"]);
    let branch = git(&manifest_dir, &["rev-parse", "--abbrev-ref", "HEAD"]);

    // Dirty means: tracked files differ from the commit named above, so that
    // commit does not describe this binary. Untracked files are excluded --
    // they do not change what was compiled, and counting them would make
    // almost every working tree permanently dirty, which would train people to
    // ignore the marker.
    let dirty = match git_raw(&manifest_dir, &["status", "--porcelain", "--untracked-files=no"]) {
        Some(out) => {
            let n = out.lines().filter(|l| !l.trim().is_empty()).count();
            if n > 0 { n.to_string() } else { "0".to_string() }
        }
        // Not a git repository, or git unavailable. Report unknown rather than
        // failing the build: a source tarball or a vendored copy is a
        // legitimate way to build this, and refusing would make it unbuildable.
        None => "unknown".to_string(),
    };

    emit("SIGIL_BUILD_COMMIT", commit.as_deref().unwrap_or("unknown"));
    emit("SIGIL_BUILD_COMMIT_SHORT", short.as_deref().unwrap_or("unknown"));
    emit("SIGIL_BUILD_BRANCH", branch.as_deref().unwrap_or("unknown"));
    emit("SIGIL_BUILD_DIRTY", &dirty);
    emit("SIGIL_BUILD_PROFILE", &std::env::var("PROFILE").unwrap_or_else(|_| "unknown".into()));
    emit(
        "SIGIL_BUILD_FEATURES",
        &enabled_features(&manifest_dir).unwrap_or_else(|| "unknown".into()),
    );

    // Rebuild when HEAD moves or the index changes, so the baked values cannot
    // go stale behind an incremental build -- a stale provenance string is the
    // defect this exists to prevent.
    if let Some(git_dir) = git(&manifest_dir, &["rev-parse", "--git-dir"]) {
        let git_dir = std::path::Path::new(&manifest_dir).join(&git_dir);
        for f in ["HEAD", "index"] {
            let p = git_dir.join(f);
            if p.exists() {
                println!("cargo:rerun-if-changed={}", p.display());
            }
        }
    }
    println!("cargo:rerun-if-changed=build.rs");
}

fn emit(key: &str, value: &str) {
    println!("cargo:rustc-env={}={}", key, value);
}

/// Which of the crate's *own* features are on, so a count measured under a
/// narrower build is identifiable as such. `--no-default-features --features
/// minimal,protocols` silently drops 45 tests relative to a default build; the
/// number is true and not comparable, and nothing in it says so.
///
/// Only the names declared in this crate's `[features]` table are reported.
/// CARGO_FEATURE_* also carries every transitive dependency feature, which
/// runs to dozens of entries and would make the line unreadable -- and a
/// provenance field nobody reads is not a provenance field.
fn enabled_features(manifest_dir: &str) -> Option<String> {
    let declared = declared_features(manifest_dir)?;
    let mut on: Vec<String> = std::env::vars()
        .filter_map(|(k, v)| {
            let name = k.strip_prefix("CARGO_FEATURE_")?;
            (v == "1").then(|| name.to_lowercase().replace('_', "-"))
        })
        .filter(|f| declared.contains(f))
        .collect();
    if on.is_empty() {
        return None;
    }
    on.sort();
    Some(on.join(","))
}

/// The feature names this crate declares, read from its own Cargo.toml.
fn declared_features(manifest_dir: &str) -> Option<Vec<String>> {
    let toml = std::fs::read_to_string(std::path::Path::new(manifest_dir).join("Cargo.toml")).ok()?;
    let mut out = Vec::new();
    let mut in_features = false;
    for line in toml.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            in_features = t == "[features]";
            continue;
        }
        if !in_features || t.starts_with('#') || t.is_empty() {
            continue;
        }
        if let Some((name, _)) = t.split_once('=') {
            let name = name.trim();
            if !name.is_empty() && !name.starts_with('"') {
                out.push(name.to_lowercase().replace('_', "-"));
            }
        }
    }
    (!out.is_empty()).then_some(out)
}

fn git(dir: &str, args: &[&str]) -> Option<String> {
    git_raw(dir, args).map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

fn git_raw(dir: &str, args: &[&str]) -> Option<String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout).ok()
}
