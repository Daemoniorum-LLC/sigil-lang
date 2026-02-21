//! CLI interface for egui → Qliphoth migration.
//!
//! Mirrors `migrate/react/cli.rs` structure. Provides:
//! - `MigrateEguiConfig` — configuration parsed from CLI args
//! - `parse_egui_migrate_args` — argument parsing (called from main.rs `migrate` dispatch)
//! - `run_egui_migration` — main entry point for `sigil migrate --from-egui`

use serde_json;
use std::fs;
use std::path::{Path, PathBuf};

use super::extraction::extract_file;
use super::generator::generate_sigil;
use super::spec::{build_manifest, build_spec};

// =============================================================================
// Configuration
// =============================================================================

/// Configuration for egui migration. Mirrors `MigrateReactConfig`.
#[derive(Debug, Clone)]
pub struct MigrateEguiConfig {
    /// Source file or directory containing `.rs` egui files.
    pub source: PathBuf,
    /// Output directory for `.migration.json` specs and (optionally) `.sigil` files.
    pub output_dir: PathBuf,
    /// Overwrite existing output files.
    pub force: bool,
    /// Print what would be done without writing any files.
    pub dry_run: bool,
    /// Show migration status summary for an existing output directory.
    pub show_status: bool,
    /// Also write `.sigil` skeleton files (not just JSON specs).
    pub generate_code: bool,
    /// Patterns to exclude (default: `test`, `tests`, `benches`, `build.rs`).
    pub exclude_patterns: Vec<String>,
}

impl Default for MigrateEguiConfig {
    fn default() -> Self {
        Self {
            source: PathBuf::new(),
            output_dir: PathBuf::from("migration"),
            force: false,
            dry_run: false,
            show_status: false,
            generate_code: false,
            exclude_patterns: vec![
                "tests".to_string(),
                "test".to_string(),
                "benches".to_string(),
                "build.rs".to_string(),
                "main.rs".to_string(),
            ],
        }
    }
}

// =============================================================================
// Argument parsing (mirrors react/cli.rs parse_react_migrate_args)
// =============================================================================

/// Parse CLI arguments for egui migration.
///
/// Consumes args from the point after `--from-egui` has already been
/// consumed by the outer `migrate` dispatcher.
pub fn parse_egui_migrate_args(args: &[String]) -> Result<MigrateEguiConfig, String> {
    let mut config = MigrateEguiConfig::default();
    let mut i = 0;

    while i < args.len() {
        match args[i].as_str() {
            "--from-egui" => {
                if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                    i += 1;
                    config.source = PathBuf::from(&args[i]);
                }
            }
            "-o" | "--output" => {
                i += 1;
                if i >= args.len() {
                    return Err("-o/--output requires a path argument".to_string());
                }
                config.output_dir = PathBuf::from(&args[i]);
            }
            "--force" => {
                config.force = true;
            }
            "--dry-run" => {
                config.dry_run = true;
            }
            "--status" => {
                config.show_status = true;
            }
            "--generate" | "-g" => {
                config.generate_code = true;
            }
            "--exclude" => {
                i += 1;
                if i >= args.len() {
                    return Err("--exclude requires a pattern argument".to_string());
                }
                config.exclude_patterns.push(args[i].clone());
            }
            _ if !args[i].starts_with('-') => {
                if config.source.as_os_str().is_empty() {
                    config.source = PathBuf::from(&args[i]);
                }
            }
            _ => {} // ignore unknown flags
        }

        i += 1;
    }

    if !config.show_status && config.source.as_os_str().is_empty() {
        return Err(
            "Source file or directory required. Use: sigil migrate --from-egui <path>".to_string()
        );
    }

    Ok(config)
}

// =============================================================================
// File discovery (mirrors react/cli.rs discover_react_files)
// =============================================================================

/// Discover `.rs` files under a source path, respecting exclude patterns.
pub fn discover_egui_files(config: &MigrateEguiConfig) -> Result<Vec<PathBuf>, String> {
    if !config.source.exists() {
        return Err(format!("Source not found: {:?}", config.source));
    }

    if config.source.is_file() {
        return Ok(vec![config.source.clone()]);
    }

    let mut files = Vec::new();
    discover_recursive(&config.source, &config.exclude_patterns, &mut files)?;
    files.sort();
    Ok(files)
}

fn discover_recursive(
    dir: &Path,
    exclude: &[String],
    out: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let entries = fs::read_dir(dir)
        .map_err(|e| format!("Cannot read {:?}: {}", dir, e))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("Entry error: {}", e))?;
        let path = entry.path();
        let name = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");

        // Skip excluded names
        if exclude.iter().any(|p| name == p || name.ends_with(&format!("_{}", p))) {
            continue;
        }

        // Skip hidden files/dirs
        if name.starts_with('.') {
            continue;
        }

        if path.is_dir() {
            discover_recursive(&path, exclude, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }

    Ok(())
}

// =============================================================================
// Main migration runner
// =============================================================================

/// Run the full egui migration for the given config.
pub fn run_egui_migration(config: &MigrateEguiConfig) -> Result<MigrationSummary, String> {
    if config.show_status {
        return show_status(config);
    }

    let files = discover_egui_files(config)?;
    eprintln!("[egui-migrate] Found {} Rust files", files.len());

    let source_root = if config.source.is_dir() {
        config.source.clone()
    } else {
        config.source.parent().unwrap_or(Path::new(".")).to_path_buf()
    };

    if !config.dry_run {
        fs::create_dir_all(&config.output_dir)
            .map_err(|e| format!("Cannot create output dir: {}", e))?;

        if config.generate_code {
            let sigil_dir = config.output_dir.join("sigil");
            fs::create_dir_all(&sigil_dir)
                .map_err(|e| format!("Cannot create sigil dir: {}", e))?;
        }
    }

    let mut all_specs = Vec::new();
    let mut file_errors: Vec<(PathBuf, String)> = Vec::new();

    for path in &files {
        match process_file(path, &source_root, config) {
            Ok(specs) => {
                eprintln!(
                    "[egui-migrate] {} → {} actor(s)",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    specs.len()
                );
                all_specs.extend(specs);
            }
            Err(e) => {
                eprintln!("[egui-migrate] WARN: skipping {:?}: {}", path, e);
                file_errors.push((path.clone(), e));
            }
        }
    }

    // Write per-file JSON specs
    if !config.dry_run {
        for spec in &all_specs {
            write_spec_json(spec, &source_root, &config.output_dir, config.force)?;

            if config.generate_code {
                write_sigil_file(spec, &config.output_dir, config.force)?;
            }
        }
    }

    // Write manifest
    let manifest = build_manifest(all_specs.clone(), &source_root);
    let manifest_path = config.output_dir.join("migration-manifest.json");

    if !config.dry_run {
        let json = serde_json::to_string_pretty(&manifest)
            .map_err(|e| format!("JSON serialization error: {}", e))?;
        fs::write(&manifest_path, json)
            .map_err(|e| format!("Cannot write manifest: {}", e))?;
        eprintln!("[egui-migrate] Wrote manifest → {:?}", manifest_path);
    }

    Ok(MigrationSummary {
        files_processed: files.len() - file_errors.len(),
        files_errored: file_errors.len(),
        actors_found: all_specs.len(),
        specs_written: if config.dry_run { 0 } else { all_specs.len() },
        sigil_written: if config.dry_run || !config.generate_code { 0 } else { all_specs.len() },
    })
}

// =============================================================================
// Per-file processing
// =============================================================================

fn process_file(
    path: &Path,
    source_root: &Path,
    _config: &MigrateEguiConfig,
) -> Result<Vec<super::spec::EguiMigrationSpec>, String> {
    let extraction = extract_file(path, source_root)?;

    // Build a spec for each public struct in the file
    let specs: Vec<_> = extraction.structs.iter()
        .filter(|s| s.is_pub || !s.fields.is_empty())
        .map(|s| build_spec(&extraction, s, source_root))
        .collect();

    Ok(specs)
}

// =============================================================================
// Output writers
// =============================================================================

fn write_spec_json(
    spec: &super::spec::EguiMigrationSpec,
    source_root: &Path,
    output_dir: &Path,
    force: bool,
) -> Result<(), String> {
    // Derive output filename from source path:
    // `crates/ide-gui/src/notifications.rs:Notifications`
    // → `notifications--notifications.migration.json`
    let stem = spec.id
        .split(':')
        .next()
        .and_then(|p| Path::new(p).file_stem().and_then(|s| s.to_str()).map(|s| s.to_string()))
        .unwrap_or_else(|| "unknown".to_string());
    let actor = to_snake_case(&spec.name);
    let filename = format!("{}--{}.migration.json", stem, actor);
    let out_path = output_dir.join(filename);

    if out_path.exists() && !force {
        eprintln!("[egui-migrate] Skipping {:?} (already exists, use --force to overwrite)", out_path);
        return Ok(());
    }

    let json = serde_json::to_string_pretty(spec)
        .map_err(|e| format!("JSON error: {}", e))?;
    fs::write(&out_path, json)
        .map_err(|e| format!("Write error {:?}: {}", out_path, e))?;

    Ok(())
}

fn write_sigil_file(
    spec: &super::spec::EguiMigrationSpec,
    output_dir: &Path,
    force: bool,
) -> Result<(), String> {
    let generated = generate_sigil(spec);
    let sigil_dir = output_dir.join("sigil");
    let filename = Path::new(&generated.path)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    let out_path = sigil_dir.join(filename);

    if out_path.exists() && !force {
        eprintln!("[egui-migrate] Skipping {:?} (exists)", out_path);
        return Ok(());
    }

    fs::write(&out_path, &generated.code)
        .map_err(|e| format!("Write error {:?}: {}", out_path, e))?;

    Ok(())
}

// =============================================================================
// Status command
// =============================================================================

fn show_status(config: &MigrateEguiConfig) -> Result<MigrationSummary, String> {
    let manifest_path = config.output_dir.join("migration-manifest.json");

    if !manifest_path.exists() {
        eprintln!(
            "[egui-migrate] No manifest at {:?} — run migration first",
            manifest_path
        );
        return Ok(MigrationSummary::default());
    }

    let json = fs::read_to_string(&manifest_path)
        .map_err(|e| format!("Cannot read manifest: {}", e))?;
    let manifest: super::spec::EguiMigrationManifest = serde_json::from_str(&json)
        .map_err(|e| format!("Invalid manifest JSON: {}", e))?;

    let s = &manifest.state;
    eprintln!("[egui-migrate] Status for {:?}", config.output_dir);
    eprintln!("  Total components : {}", s.total_components);
    eprintln!("  Completed        : {}", s.completed);
    eprintln!("  In progress      : {}", s.in_progress);
    eprintln!("  Blocked          : {}", s.blocked);
    eprintln!("  Pending          : {}", s.total_components - s.completed - s.in_progress - s.blocked);
    eprintln!("  Last updated     : {}", s.last_updated);

    Ok(MigrationSummary {
        files_processed: 0,
        files_errored: 0,
        actors_found: s.total_components,
        specs_written: s.completed,
        sigil_written: 0,
    })
}

// =============================================================================
// Summary type
// =============================================================================

/// Summary of a migration run.
#[derive(Debug, Default)]
pub struct MigrationSummary {
    pub files_processed: usize,
    pub files_errored: usize,
    pub actors_found: usize,
    pub specs_written: usize,
    pub sigil_written: usize,
}

// =============================================================================
// Helpers
// =============================================================================

fn to_snake_case(s: &str) -> String {
    let mut result = String::new();
    for (i, ch) in s.chars().enumerate() {
        if ch.is_uppercase() && i > 0 {
            result.push('_');
        }
        result.push(ch.to_lowercase().next().unwrap());
    }
    result
}
