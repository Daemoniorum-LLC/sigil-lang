//! CLI interface for React→Qliphoth migration.
//!
//! Provides command-line tools for migrating React projects to Qliphoth.
//!
//! See docs/specs/REACT-MIGRATION.md Section 6 for specification.

use super::extraction::*;
use super::spec::*;
use super::generator::*;
use super::mcp::*;
use serde_json;
use std::path::{Path, PathBuf};
use std::fs;

// =============================================================================
// CLI Configuration
// =============================================================================

/// Configuration for React migration CLI.
#[derive(Debug, Clone)]
pub struct MigrateReactConfig {
    /// Source directory containing React files
    pub source_dir: PathBuf,

    /// Output directory for migration specs and generated code
    pub output_dir: PathBuf,

    /// Glob patterns for files to include (default: *.tsx, *.jsx, *.ts, *.js)
    pub include_patterns: Vec<String>,

    /// Glob patterns for files to exclude (default: *.test.*, *.spec.*, node_modules)
    pub exclude_patterns: Vec<String>,

    /// Overwrite existing files
    pub force: bool,

    /// Show what would be done without writing
    pub dry_run: bool,

    /// Start MCP server mode
    pub serve: bool,

    /// Validate a single Sigil file
    pub validate_file: Option<PathBuf>,

    /// Show migration status
    pub show_status: bool,

    /// Generate Sigil code (not just specs)
    pub generate_code: bool,

    /// Qliphoth source files to vendor into the generated project.
    ///
    /// The project needs the VNode builder its views are written against.
    /// Depending on the whole `qliphoth` package does not work yet — its
    /// `qliphoth-sys` is written against `web_sys` and does not compile — so
    /// the modules that are needed are named and copied in.
    pub vendor: Vec<PathBuf>,

    /// Emit a Sigil project — shared module scope once, one module per
    /// component, a `sigil.toml` and a `lib.sigil` — instead of 93
    /// self-contained files.
    pub project: bool,

    /// Extra directories to parse for cross-module resolution.
    ///
    /// A bundler alias is configuration the migrator has no bundler to read:
    /// the Lares client imports `@shared/identity`, which Vite maps onto a
    /// sibling directory outside the source root. Six components referenced
    /// names that live there, so the roots have to be given explicitly.
    /// Components found here are migrated too — these are ordinary source
    /// directories, just ones no import path names relatively.
    pub extra_source_dirs: Vec<PathBuf>,
}

impl Default for MigrateReactConfig {
    fn default() -> Self {
        Self {
            source_dir: PathBuf::new(),
            output_dir: PathBuf::from("migration-specs"),
            include_patterns: vec![
                "**/*.tsx".to_string(),
                "**/*.jsx".to_string(),
            ],
            exclude_patterns: vec![
                "**/node_modules/**".to_string(),
                "**/*.test.*".to_string(),
                "**/*.spec.*".to_string(),
                "**/__tests__/**".to_string(),
                "**/__mocks__/**".to_string(),
            ],
            force: false,
            dry_run: false,
            serve: false,
            validate_file: None,
            show_status: false,
            generate_code: false,
            project: false,
            vendor: Vec::new(),
            extra_source_dirs: Vec::new(),
        }
    }
}

// =============================================================================
// Argument Parsing
// =============================================================================

/// Parse CLI arguments for React migration.
pub fn parse_react_migrate_args(args: &[String]) -> Result<MigrateReactConfig, String> {
    let mut config = MigrateReactConfig::default();
    let mut i = 0;

    while i < args.len() {
        let arg = &args[i];

        match arg.as_str() {
            "--from-react" => {
                // Next arg could be source directory (if not another flag)
                if i + 1 < args.len() && !args[i + 1].starts_with('-') {
                    i += 1;
                    config.source_dir = PathBuf::from(&args[i]);
                }
                // If no path follows, it will be picked up as a positional arg
            }
            "-o" | "--output" => {
                i += 1;
                if i >= args.len() {
                    return Err("-o/--output requires a path argument".to_string());
                }
                config.output_dir = PathBuf::from(&args[i]);
            }
            "--vendor" => {
                i += 1;
                if i >= args.len() {
                    return Err("--vendor requires a file path".to_string());
                }
                config.vendor.push(PathBuf::from(&args[i]));
            }
            "--project" => {
                config.project = true;
                config.generate_code = true;
            }
            "--include-source" => {
                // The loop advances by one at the bottom; every other
                // value-taking flag consumes its argument with a single `i += 1`
                // and lets the tail do the rest. Advancing by two here ate the
                // flag that followed — `--generate`, which is how the
                // regeneration silently stopped generating.
                i += 1;
                if i >= args.len() {
                    return Err("--include-source requires a directory argument".to_string());
                }
                config.extra_source_dirs.push(PathBuf::from(&args[i]));
            }
            "--include" => {
                i += 1;
                if i >= args.len() {
                    return Err("--include requires a pattern argument".to_string());
                }
                config.include_patterns.push(args[i].clone());
            }
            "--exclude" => {
                i += 1;
                if i >= args.len() {
                    return Err("--exclude requires a pattern argument".to_string());
                }
                config.exclude_patterns.push(args[i].clone());
            }
            "--force" => {
                config.force = true;
            }
            "--dry-run" => {
                config.dry_run = true;
            }
            "--serve" => {
                config.serve = true;
            }
            "--validate" => {
                i += 1;
                if i >= args.len() {
                    return Err("--validate requires a file path".to_string());
                }
                config.validate_file = Some(PathBuf::from(&args[i]));
            }
            "--status" => {
                config.show_status = true;
            }
            "--generate" | "-g" => {
                config.generate_code = true;
            }
            _ if !arg.starts_with('-') => {
                // Positional argument - could be source dir
                if config.source_dir.as_os_str().is_empty() {
                    config.source_dir = PathBuf::from(arg);
                }
            }
            _ => {
                // Ignore unknown flags (might be for other parts of CLI)
            }
        }

        i += 1;
    }

    // Validate required arguments
    if config.validate_file.is_none() && !config.show_status && config.source_dir.as_os_str().is_empty() {
        return Err("Source directory required. Use: sigil migrate --from-react <path>".to_string());
    }

    Ok(config)
}

// =============================================================================
// File Discovery
// =============================================================================

/// Find React/TSX files in a directory.
pub fn discover_react_files(config: &MigrateReactConfig) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();

    if !config.source_dir.exists() {
        return Err(format!("Source directory not found: {:?}", config.source_dir));
    }

    if !config.source_dir.is_dir() {
        // Single file
        files.push(config.source_dir.clone());
        return Ok(files);
    }

    // Walk directory recursively
    discover_files_recursive(&config.source_dir, &config.include_patterns, &config.exclude_patterns, &mut files)?;

    for extra in &config.extra_source_dirs {
        if !extra.exists() {
            return Err(format!("Source directory not found: {:?}", extra));
        }
        if extra.is_dir() {
            discover_files_recursive(
                extra,
                &config.include_patterns,
                &config.exclude_patterns,
                &mut files,
            )?;
        } else {
            files.push(extra.clone());
        }
    }

    Ok(files)
}

fn discover_files_recursive(
    dir: &Path,
    include: &[String],
    exclude: &[String],
    files: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let entries = fs::read_dir(dir)
        .map_err(|e| format!("Cannot read directory {:?}: {}", dir, e))?;

    for entry in entries {
        let entry = entry.map_err(|e| format!("Error reading entry: {}", e))?;
        let path = entry.path();
        let path_str = path.to_string_lossy();

        // Skip excluded paths
        if should_exclude(&path_str, exclude) {
            continue;
        }

        if path.is_dir() {
            // Recurse into subdirectory
            discover_files_recursive(&path, include, exclude, files)?;
        } else if should_include(&path, include) {
            files.push(path);
        }
    }

    Ok(())
}

fn should_exclude(path: &str, patterns: &[String]) -> bool {
    for pattern in patterns {
        if pattern.contains("node_modules") && path.contains("node_modules") {
            return true;
        }
        if pattern.contains("__tests__") && path.contains("__tests__") {
            return true;
        }
        if pattern.contains("__mocks__") && path.contains("__mocks__") {
            return true;
        }
        if pattern.contains(".test.") && path.contains(".test.") {
            return true;
        }
        if pattern.contains(".spec.") && path.contains(".spec.") {
            return true;
        }
    }
    false
}

fn should_include(path: &Path, patterns: &[String]) -> bool {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

    // Default: include .tsx, .jsx, .ts, .js
    if patterns.is_empty() || patterns.iter().any(|p| p.contains("**/*.tsx") || p.contains("**/*.jsx")) {
        return matches!(ext, "tsx" | "jsx" | "ts" | "js");
    }

    // Check explicit patterns
    for pattern in patterns {
        if pattern.ends_with(&format!("*.{}", ext)) {
            return true;
        }
    }

    false
}

// =============================================================================
// Output Generation
// =============================================================================

/// Output structure for migration specs.
pub struct MigrationOutput {
    /// Root output directory
    pub root: PathBuf,

    /// Path to manifest.json
    pub manifest_path: PathBuf,

    /// Component spec paths
    pub component_paths: Vec<PathBuf>,

    /// Type spec paths
    pub type_paths: Vec<PathBuf>,

    /// Generated Sigil file paths
    pub output_paths: Vec<PathBuf>,
}


/// Emit a Sigil project: shared module scope once, one module per component.
///
/// The per-file output is deliberately self-contained — every component carries
/// copies of the constants and helpers it references — which is what lets one
/// be compiled on its own and what makes all of them impossible to compile
/// together: concatenated they redefine `get_json` 84 times. This is the other
/// arrangement. `sigil wasm <dir>` on the result links the whole thing, so a
/// component calling a sibling component resolves instead of being stubbed.
fn write_project(
    components: &[&crate::migrate::react::spec::ComponentMigrationSpec],
    root: &Path,
    config: &MigrateReactConfig,
) -> Result<(), String> {
    use crate::migrate::react::generator::{component_prop_map, generate_component_only, generate_shared_module};
    use crate::migrate::react::spec::to_snake_case;

    let project_dir = root.join("project");
    let src_dir = project_dir.join("src");
    if config.dry_run {
        println!("Dry run - would write project to {:?}", project_dir);
        return Ok(());
    }
    fs::create_dir_all(&src_dir).map_err(|e| format!("Cannot create project src dir: {}", e))?;

    println!();
    println!("Generating Sigil project:");

    fs::write(
        project_dir.join("sigil.toml"),
        "[tome]\nname = \"lares_client\"\nversion = \"0.1.0\"\n",
    )
    .map_err(|e| format!("Cannot write sigil.toml: {}", e))?;

    // Vendored Qliphoth modules, in the order given.
    let mut vendored: Vec<String> = Vec::new();
    for v in &config.vendor {
        let text = fs::read_to_string(v)
            .map_err(|e| format!("Cannot read {:?}: {}", v, e))?;
        let stem = v
            .file_stem()
            .map(|s| to_snake_case(&s.to_string_lossy()))
            .unwrap_or_else(|| "vendored".to_string());
        // Its own `invoke` lines name a package this project does not depend
        // on; everything it needs is in this module tree.
        let body: String = text
            .lines()
            .filter(|l| !l.starts_with("invoke "))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(
            src_dir.join(format!("{}.sigil", stem)),
            format!("// Vendored from {}.\n\n{}", v.display(), body),
        )
        .map_err(|e| format!("Cannot write {}.sigil: {}", stem, e))?;
        println!("  project/src/{}.sigil  (vendored)", stem);
        vendored.push(stem);
    }

    let shared = generate_shared_module(components);
    let mut shared_invokes = String::new();
    for v in &vendored {
        shared_invokes.push_str(&format!("invoke tome·{}·*;\n", v));
    }
    let shared = format!("{}\n{}", shared_invokes, shared);
    fs::write(src_dir.join("shared.sigil"), &shared)
        .map_err(|e| format!("Cannot write shared.sigil: {}", e))?;
    println!("  project/src/shared.sigil");

    // One module per component. Names are the module path, so they have to be
    // valid identifiers — kebab-case file names are not.
    let props = component_prop_map(components);
    let mut modules: Vec<String> = Vec::new();
    for comp in components {
        let generated = generate_component_only(comp, props.clone());
        let module = to_snake_case(&generated.component_name);
        if modules.contains(&module) {
            continue;
        }
        let mut invokes = String::from("invoke tome·shared·*;\n");
        for v in &vendored {
            invokes.push_str(&format!("invoke tome·{}·*;\n", v));
        }
        let body = format!(
            "// Generated from {}.\n{}\n{}",
            comp.source.path,
            invokes,
            generated.code.replace("invoke qliphoth·prelude·*;\n", "")
        );
        fs::write(src_dir.join(format!("{}.sigil", module)), body)
            .map_err(|e| format!("Cannot write {}.sigil: {}", module, e))?;
        modules.push(module);
    }
    println!("  project/src/*.sigil  ({} components)", modules.len());

    let mut lib = String::from(
        "// Every component and the module scope they share. `sigil wasm .` from\n         // this directory links them into one module.\n\n",
    );
    for v in &vendored {
        lib.push_str(&format!("scroll {};\n", v));
    }
    lib.push_str("scroll shared;\n");
    for m in &modules {
        lib.push_str(&format!("scroll {};\n", m));
    }
    lib.push('\n');
    for v in &vendored {
        lib.push_str(&format!("invoke {}·*;\n", v));
    }
    lib.push_str("invoke shared·*;\n");
    for m in &modules {
        lib.push_str(&format!("invoke {}·*;\n", m));
    }
    fs::write(src_dir.join("lib.sigil"), lib)
        .map_err(|e| format!("Cannot write lib.sigil: {}", e))?;
    println!("  project/src/lib.sigil");
    println!("  project/sigil.toml");

    Ok(())
}

/// Write migration specs to disk.
pub fn write_migration_output(
    session: &MigrationSession,
    config: &MigrateReactConfig,
) -> Result<MigrationOutput, String> {
    let root = &config.output_dir;

    // Create directory structure
    let components_dir = root.join("components");
    let types_dir = root.join("types");
    let patterns_dir = root.join("patterns");
    let output_dir = root.join("output");

    if !config.dry_run {
        fs::create_dir_all(&components_dir)
            .map_err(|e| format!("Cannot create components dir: {}", e))?;
        fs::create_dir_all(&types_dir)
            .map_err(|e| format!("Cannot create types dir: {}", e))?;
        fs::create_dir_all(&patterns_dir)
            .map_err(|e| format!("Cannot create patterns dir: {}", e))?;
        fs::create_dir_all(&output_dir)
            .map_err(|e| format!("Cannot create output dir: {}", e))?;
    }

    let spec = session.spec();
    let mut result = MigrationOutput {
        root: root.clone(),
        manifest_path: root.join("manifest.json"),
        component_paths: Vec::new(),
        type_paths: Vec::new(),
        output_paths: Vec::new(),
    };

    // Write manifest.json
    if !config.dry_run {
        let manifest_json = serde_json::to_string_pretty(spec)
            .map_err(|e| format!("Cannot serialize manifest: {}", e))?;
        fs::write(&result.manifest_path, manifest_json)
            .map_err(|e| format!("Cannot write manifest: {}", e))?;
    }
    println!("  manifest.json");

    // Write individual component specs
    // Use component id (file:name) to avoid collisions between same-named components
    for comp in &spec.components {
        let filename = format!("{}.json", id_to_filename(&comp.id));
        let comp_path = components_dir.join(&filename);

        if !config.dry_run {
            let comp_json = serde_json::to_string_pretty(comp)
                .map_err(|e| format!("Cannot serialize component {}: {}", comp.name, e))?;
            fs::write(&comp_path, comp_json)
                .map_err(|e| format!("Cannot write component {}: {}", comp.name, e))?;
        }

        println!("  components/{}", filename);
        result.component_paths.push(comp_path);
    }

    // Write type specs
    // Use type id (file:name) to avoid collisions between same-named types
    for type_spec in &spec.types {
        let filename = format!("{}.json", id_to_filename(&type_spec.id));
        let type_path = types_dir.join(&filename);

        if !config.dry_run {
            let type_json = serde_json::to_string_pretty(type_spec)
                .map_err(|e| format!("Cannot serialize type {}: {}", type_spec.name, e))?;
            fs::write(&type_path, type_json)
                .map_err(|e| format!("Cannot write type {}: {}", type_spec.name, e))?;
        }

        println!("  types/{}", filename);
        result.type_paths.push(type_path);
    }

    // Write pattern library
    let patterns = session.resource_patterns();
    let patterns_path = patterns_dir.join("library.json");
    if !config.dry_run {
        let patterns_json = serde_json::to_string_pretty(&patterns)
            .map_err(|e| format!("Cannot serialize patterns: {}", e))?;
        fs::write(&patterns_path, patterns_json)
            .map_err(|e| format!("Cannot write patterns: {}", e))?;
    }
    println!("  patterns/library.json");

    // Generate Sigil code if requested
    if config.project {
        write_project(&spec.components.iter().collect::<Vec<_>>(), root, config)?;
    }

    if config.generate_code {
        println!();
        println!("Generating Sigil code:");

        // Generate component actors
        let all: Vec<&crate::migrate::react::spec::ComponentMigrationSpec> =
            spec.components.iter().collect();
        let props = crate::migrate::react::generator::component_prop_map(&all);
        for comp_spec in &spec.components {
            let generated = crate::migrate::react::generator::generate_component_with(
                comp_spec,
                props.clone(),
            );
            let filename = format!("{}.sigil", to_kebab_case(&generated.component_name));
            let sigil_path = output_dir.join(&filename);

            if !config.dry_run {
                fs::write(&sigil_path, &generated.code)
                    .map_err(|e| format!("Cannot write {}: {}", filename, e))?;
            }

            println!("  output/{}", filename);
            result.output_paths.push(sigil_path);
        }

        // Generate service actors from custom hooks
        if !spec.service_actors.is_empty() {
            println!();
            println!("Generating service actors:");

            // Create services subdirectory
            let services_dir = output_dir.join("services");
            if !config.dry_run && !services_dir.exists() {
                fs::create_dir_all(&services_dir)
                    .map_err(|e| format!("Cannot create services directory: {}", e))?;
            }

            for actor_spec in &spec.service_actors {
                let generated = generate_service_actor(actor_spec);
                let filename = format!("{}.sigil", to_kebab_case(&generated.component_name));
                let sigil_path = services_dir.join(&filename);

                if !config.dry_run {
                    fs::write(&sigil_path, &generated.code)
                        .map_err(|e| format!("Cannot write {}: {}", filename, e))?;
                }

                println!("  output/services/{} (from {})", filename, actor_spec.derived_from);
                result.output_paths.push(sigil_path);
            }
        }
    }

    Ok(result)
}

/// Convert PascalCase to kebab-case.
fn to_kebab_case(s: &str) -> String {
    let mut result = String::new();
    for (i, c) in s.chars().enumerate() {
        if c.is_uppercase() && i > 0 {
            result.push('-');
        }
        result.push(c.to_ascii_lowercase());
    }
    result
}

/// Convert component id (path:name) to a unique filename.
/// Example: "components/chat/ChatPanel.tsx:ChatPanel" -> "chat--chat-panel"
fn id_to_filename(id: &str) -> String {
    // Split on ':' to get path and component name
    let parts: Vec<&str> = id.splitn(2, ':').collect();

    if parts.len() == 2 {
        let path = parts[0];
        let name = parts[1];

        // Extract parent directory from path (e.g., "components/chat/ChatPanel.tsx" -> "chat")
        let path_parts: Vec<&str> = path.split('/').collect();
        let parent = if path_parts.len() >= 2 {
            // Use second-to-last part (the directory name)
            path_parts[path_parts.len() - 2]
        } else {
            "root"
        };

        // Combine parent directory and component name
        format!("{}--{}", to_kebab_case(parent), to_kebab_case(name))
    } else {
        // Fallback to just kebab-casing the whole id
        to_kebab_case(id)
    }
}

// =============================================================================
// Main Entry Point
// =============================================================================

/// Run the React migration CLI.
pub fn run_react_migrate(args: &[String]) -> Result<(), String> {
    let config = parse_react_migrate_args(args)?;

    // Handle --validate
    if let Some(ref file_path) = config.validate_file {
        return run_validate(file_path);
    }

    // Handle --status
    if config.show_status {
        return run_status(&config.output_dir);
    }

    // Normal migration flow
    println!("React → Qliphoth Migration");
    println!("==========================");
    println!();

    // Discover files
    println!("Discovering React files in {:?}...", config.source_dir);
    let files = discover_react_files(&config)?;

    if files.is_empty() {
        println!("No React files found.");
        return Ok(());
    }

    println!("Found {} files to process.", files.len());
    println!();

    // Create migration session
    let mut session = MigrationSession::new(&config.source_dir, &config.output_dir)
        .map_err(|e| format!("Cannot create session: {}", e))?;

    // Process each file
    println!("Extracting components...");
    let mut total_components = 0;

    for file_path in &files {
        let relative = file_path.strip_prefix(&config.source_dir)
            .unwrap_or(file_path);
        print!("  {:?}... ", relative);

        let source = fs::read_to_string(file_path)
            .map_err(|e| format!("Cannot read {:?}: {}", file_path, e))?;

        match session.add_file(file_path, &source) {
            Ok(()) => {
                let count = session.list_migrations().len() - total_components;
                total_components = session.list_migrations().len();
                println!("{} components", count);
            }
            Err(e) => {
                println!("SKIP ({})", e);
            }
        }
    }

    // Every file is in now, so a component's imports can be matched against the
    // constants other files export.
    session.resolve_cross_module_imports();

    println!();
    println!("Extracted {} total components.", total_components);
    println!();

    // Write output
    if config.dry_run {
        println!("Dry run - would write to {:?}:", config.output_dir);
    } else {
        println!("Writing migration specs to {:?}:", config.output_dir);
    }

    let output = write_migration_output(&session, &config)?;

    println!();
    if config.generate_code {
        println!("Migration complete!");
        println!();
        println!("Generated files:");
        println!("  Specs:  {:?}", output.root.join("components"));
        println!("  Output: {:?}", output.root.join("output"));
        println!();
        println!("Next steps:");
        println!("  1. Review generated Sigil code in output/");
        println!("  2. Copy to your Qliphoth project and adjust as needed");
    } else {
        println!("Migration specs generated successfully!");
        println!();
        println!("Next steps:");
        println!("  1. Review component specs in {:?}", output.root.join("components"));
        println!("  2. Generate Sigil code: sigil migrate --from-react {:?} --generate", config.source_dir);
        println!("  3. Or use MCP tools for interactive migration");
    }

    // Handle --serve
    if config.serve {
        println!();
        println!("Starting MCP server mode...");
        // MCP server would be implemented here
        println!("(MCP server not yet implemented)");
    }

    Ok(())
}

/// Validate a single Sigil file.
fn run_validate(file_path: &Path) -> Result<(), String> {
    println!("Validating {:?}...", file_path);

    let code = fs::read_to_string(file_path)
        .map_err(|e| format!("Cannot read file: {}", e))?;

    // Create a minimal session for validation
    let session = MigrationSession::new(".", ".")
        .map_err(|e| format!("Cannot create session: {}", e))?;

    let result = session.validate_sigil(&code);

    if result.valid {
        println!("✓ Valid Sigil code");
        if !result.warnings.is_empty() {
            println!();
            println!("Warnings:");
            for warning in &result.warnings {
                println!("  {}:{}: {}", warning.line, warning.column, warning.message);
            }
        }
        Ok(())
    } else {
        println!("✗ Invalid Sigil code");
        println!();
        println!("Errors:");
        for error in &result.errors {
            println!("  {}:{}: {}", error.line, error.column, error.message);
            if let Some(ref suggestion) = error.suggestion {
                println!("    Suggestion: {}", suggestion);
            }
        }
        Err("Validation failed".to_string())
    }
}

/// Show migration status.
fn run_status(output_dir: &Path) -> Result<(), String> {
    let manifest_path = output_dir.join("manifest.json");

    if !manifest_path.exists() {
        return Err(format!("No migration found at {:?}", output_dir));
    }

    let manifest_json = fs::read_to_string(&manifest_path)
        .map_err(|e| format!("Cannot read manifest: {}", e))?;

    let spec: MigrationSpec = serde_json::from_str(&manifest_json)
        .map_err(|e| format!("Cannot parse manifest: {}", e))?;

    println!("Migration Status");
    println!("================");
    println!();
    println!("Project: {}", spec.project_root);
    println!("Generated: {}", spec.generated_at);
    println!();
    println!("Components: {}", spec.state.total_components);
    println!("  Completed:   {}", spec.state.completed);
    println!("  In Progress: {}", spec.state.in_progress);
    println!("  Blocked:     {}", spec.state.blocked);
    println!("  Pending:     {}", spec.state.total_components - spec.state.completed - spec.state.in_progress - spec.state.blocked);
    println!();

    if spec.state.completed < spec.state.total_components {
        println!("Next to migrate:");
        for comp in spec.components.iter().take(5) {
            if comp.status == MigrationStatus::Pending {
                println!("  - {} ({:?})", comp.name, comp.complexity);
            }
        }
    } else {
        println!("✓ All components migrated!");
    }

    Ok(())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_from_react() {
        let args = vec![
            "migrate".to_string(),
            "--from-react".to_string(),
            "./src".to_string(),
        ];
        let config = parse_react_migrate_args(&args).unwrap();
        assert_eq!(config.source_dir, PathBuf::from("./src"));
    }

    #[test]
    fn test_parse_dry_run() {
        let args = vec![
            "migrate".to_string(),
            "--from-react".to_string(),
            "./src".to_string(),
            "--dry-run".to_string(),
        ];
        let config = parse_react_migrate_args(&args).unwrap();
        assert!(config.dry_run);
    }

    #[test]
    fn test_parse_output() {
        let args = vec![
            "migrate".to_string(),
            "--from-react".to_string(),
            "./src".to_string(),
            "-o".to_string(),
            "./out".to_string(),
        ];
        let config = parse_react_migrate_args(&args).unwrap();
        assert_eq!(config.output_dir, PathBuf::from("./out"));
    }

    #[test]
    fn test_parse_serve() {
        let args = vec![
            "migrate".to_string(),
            "--from-react".to_string(),
            "./src".to_string(),
            "--serve".to_string(),
        ];
        let config = parse_react_migrate_args(&args).unwrap();
        assert!(config.serve);
    }

    #[test]
    fn test_parse_include_exclude() {
        let args = vec![
            "migrate".to_string(),
            "--from-react".to_string(),
            "./src".to_string(),
            "--include".to_string(),
            "**/*.tsx".to_string(),
            "--exclude".to_string(),
            "**/*.test.tsx".to_string(),
        ];
        let config = parse_react_migrate_args(&args).unwrap();
        assert!(config.include_patterns.contains(&"**/*.tsx".to_string()));
        assert!(config.exclude_patterns.contains(&"**/*.test.tsx".to_string()));
    }

    #[test]
    fn test_parse_validate() {
        let args = vec![
            "migrate".to_string(),
            "--from-react".to_string(),
            "--validate".to_string(),
            "./counter.sigil".to_string(),
        ];
        let config = parse_react_migrate_args(&args).unwrap();
        assert_eq!(config.validate_file, Some(PathBuf::from("./counter.sigil")));
    }

    #[test]
    fn test_parse_status() {
        let args = vec![
            "migrate".to_string(),
            "--from-react".to_string(),
            "--status".to_string(),
            "-o".to_string(),
            "./migration-specs".to_string(),
        ];
        let config = parse_react_migrate_args(&args).unwrap();
        assert!(config.show_status);
    }

    #[test]
    fn test_to_kebab_case() {
        assert_eq!(to_kebab_case("Counter"), "counter");
        assert_eq!(to_kebab_case("TodoList"), "todo-list");
        assert_eq!(to_kebab_case("UserProfileCard"), "user-profile-card");
    }

    #[test]
    fn test_id_to_filename() {
        // Normal case: directory/file.tsx:ComponentName
        assert_eq!(
            id_to_filename("components/chat/ChatPanel.tsx:ChatPanel"),
            "chat--chat-panel"
        );
        // Disambiguates same-named components from different files
        assert_eq!(
            id_to_filename("components/studio/StudioPanel.tsx:StatCard"),
            "studio--stat-card"
        );
        assert_eq!(
            id_to_filename("components/metrics/MetricsDashboard.tsx:StatCard"),
            "metrics--stat-card"
        );
        // Single directory
        assert_eq!(
            id_to_filename("src/App.tsx:App"),
            "src--app"
        );
        // Fallback for malformed id
        assert_eq!(
            id_to_filename("NoColonHere"),
            "no-colon-here"
        );
    }

    #[test]
    fn test_should_exclude_node_modules() {
        let patterns = vec!["**/node_modules/**".to_string()];
        assert!(should_exclude("/project/node_modules/react/index.js", &patterns));
        assert!(!should_exclude("/project/src/App.tsx", &patterns));
    }

    #[test]
    fn test_should_exclude_test_files() {
        let patterns = vec!["**/*.test.*".to_string()];
        assert!(should_exclude("/project/src/App.test.tsx", &patterns));
        assert!(!should_exclude("/project/src/App.tsx", &patterns));
    }

    #[test]
    fn test_should_include_tsx() {
        let patterns = vec!["**/*.tsx".to_string()];
        assert!(should_include(Path::new("App.tsx"), &patterns));
        assert!(should_include(Path::new("App.jsx"), &patterns));
        assert!(!should_include(Path::new("App.css"), &patterns));
    }
}
