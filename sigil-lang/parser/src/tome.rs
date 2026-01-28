//! Tome - The Sigil Package Manager
//!
//! Manages Sigil tomes (packages) through ritual commands:
//!
//! Commands:
//!   sigil conjure <name>     Summon a new tome into existence
//!   sigil inscribe           Mark current directory as a tome
//!   sigil summon <tome>      Call forth a dependency
//!   sigil banish <tome>      Cast out a dependency
//!   sigil attune             Realign with latest binding versions
//!   sigil forge              Shape the tome into being
//!   sigil consecrate         Enshrine tome in the Grimoire registry
//!
//! Manifest: Grimoire.toml
//!
//! The Grimoire is the central registry of all consecrated tomes.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// The manifest file name
pub const GRIMOIRE_TOML: &str = "Grimoire.toml";

/// Lock file for reproducible builds
pub const GRIMOIRE_LOCK: &str = "Grimoire.lock";

/// Directory for cached tomes
pub const TOMES_DIR: &str = ".tomes";

/// The Grimoire registry URL
pub const REGISTRY_URL: &str = "https://www.sigil-lang.com/grimoire";

/// Registry index file
pub const REGISTRY_INDEX: &str = "index.json";

// ============================================================================
// Registry Types
// ============================================================================

/// The registry index structure
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RegistryIndex {
    /// Map of tome name to metadata
    pub tomes: HashMap<String, RegistryTome>,
}

/// Metadata for a tome in the registry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegistryTome {
    /// Short description
    #[serde(default)]
    pub description: String,
    /// Available versions (newest first)
    pub versions: Vec<String>,
    /// Latest stable version
    pub latest: String,
    /// Yanked versions (should not be used)
    #[serde(default)]
    pub yanked: Vec<String>,
}

// ============================================================================
// Grimoire.toml Structure
// ============================================================================

/// Root structure of Grimoire.toml
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Grimoire {
    /// Tome metadata
    pub tome: TomeMetadata,
    /// Dependencies (bindings)
    #[serde(default)]
    pub bindings: HashMap<String, Binding>,
    /// Dev dependencies
    #[serde(default)]
    pub dev_bindings: HashMap<String, Binding>,
    /// Custom rites (scripts)
    #[serde(default)]
    pub rites: HashMap<String, String>,
    /// Workspace configuration (for multi-tome projects)
    #[serde(default)]
    pub workspace: Option<Workspace>,
}

/// Tome metadata section
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TomeMetadata {
    /// Tome name
    pub name: String,
    /// Version (semver)
    pub version: String,
    /// Authors
    #[serde(default)]
    pub authors: Vec<String>,
    /// Edition year
    #[serde(default)]
    pub edition: Option<String>,
    /// Description
    #[serde(default)]
    pub description: Option<String>,
    /// License
    #[serde(default)]
    pub license: Option<String>,
    /// Repository URL
    #[serde(default)]
    pub repository: Option<String>,
    /// Homepage URL
    #[serde(default)]
    pub homepage: Option<String>,
    /// Keywords for discovery
    #[serde(default)]
    pub keywords: Vec<String>,
    /// Categories
    #[serde(default)]
    pub categories: Vec<String>,
}

/// A binding (dependency) specification
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Binding {
    /// Simple version string: aegis = "0.1"
    Version(String),
    /// Detailed specification
    Detailed(BindingSpec),
}

/// Detailed binding specification
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BindingSpec {
    /// Version requirement
    #[serde(default)]
    pub version: Option<String>,
    /// Local path
    #[serde(default)]
    pub path: Option<String>,
    /// Git repository
    #[serde(default)]
    pub git: Option<String>,
    /// Git branch
    #[serde(default)]
    pub branch: Option<String>,
    /// Git tag
    #[serde(default)]
    pub tag: Option<String>,
    /// Git revision
    #[serde(default)]
    pub rev: Option<String>,
    /// Optional dependency
    #[serde(default)]
    pub optional: bool,
    /// Features to enable
    #[serde(default)]
    pub features: Vec<String>,
}

/// Workspace configuration for multi-tome projects
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Workspace {
    /// Member tome paths
    #[serde(default)]
    pub members: Vec<String>,
    /// Excluded paths
    #[serde(default)]
    pub exclude: Vec<String>,
}

// ============================================================================
// Lock File Structure
// ============================================================================

/// Lock file for reproducible builds
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GrimoireLock {
    /// Lock file version
    pub version: u32,
    /// Locked bindings
    #[serde(default)]
    pub bindings: Vec<LockedBinding>,
}

/// A locked binding with exact version/source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LockedBinding {
    /// Tome name
    pub name: String,
    /// Exact version
    pub version: String,
    /// Source (registry, git, path)
    pub source: String,
    /// Checksum for verification
    #[serde(default)]
    pub checksum: Option<String>,
    /// Dependencies of this binding
    #[serde(default)]
    pub dependencies: Vec<String>,
}

// ============================================================================
// Implementation
// ============================================================================

impl Grimoire {
    /// Load Grimoire.toml from a path
    pub fn load(path: &Path) -> Result<Self, String> {
        let grimoire_path = if path.is_dir() {
            path.join(GRIMOIRE_TOML)
        } else {
            path.to_path_buf()
        };

        let content = fs::read_to_string(&grimoire_path)
            .map_err(|e| format!("Failed to read {}: {}", grimoire_path.display(), e))?;

        toml::from_str(&content)
            .map_err(|e| format!("Failed to parse {}: {}", grimoire_path.display(), e))
    }

    /// Save Grimoire.toml to a path
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let grimoire_path = if path.is_dir() {
            path.join(GRIMOIRE_TOML)
        } else {
            path.to_path_buf()
        };

        let content = toml::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize Grimoire: {}", e))?;

        fs::write(&grimoire_path, content)
            .map_err(|e| format!("Failed to write {}: {}", grimoire_path.display(), e))
    }

    /// Find Grimoire.toml in current or ancestor directories
    pub fn find() -> Option<PathBuf> {
        let mut current = std::env::current_dir().ok()?;
        loop {
            let grimoire_path = current.join(GRIMOIRE_TOML);
            if grimoire_path.exists() {
                return Some(grimoire_path);
            }
            if !current.pop() {
                return None;
            }
        }
    }

    /// Add a binding to the grimoire
    pub fn summon(&mut self, name: &str, binding: Binding) {
        self.bindings.insert(name.to_string(), binding);
    }

    /// Remove a binding from the grimoire
    pub fn banish(&mut self, name: &str) -> Option<Binding> {
        self.bindings.remove(name)
    }

    /// Check if a binding exists
    pub fn has_binding(&self, name: &str) -> bool {
        self.bindings.contains_key(name)
    }
}

impl Binding {
    /// Get the version requirement
    pub fn version(&self) -> Option<&str> {
        match self {
            Binding::Version(v) => Some(v),
            Binding::Detailed(spec) => spec.version.as_deref(),
        }
    }

    /// Check if this is a path binding
    pub fn is_path(&self) -> bool {
        matches!(self, Binding::Detailed(spec) if spec.path.is_some())
    }

    /// Check if this is a git binding
    pub fn is_git(&self) -> bool {
        matches!(self, Binding::Detailed(spec) if spec.git.is_some())
    }

    /// Get the path if this is a path binding
    pub fn path(&self) -> Option<&str> {
        match self {
            Binding::Detailed(spec) => spec.path.as_deref(),
            _ => None,
        }
    }

    /// Get the git URL if this is a git binding
    pub fn git(&self) -> Option<&str> {
        match self {
            Binding::Detailed(spec) => spec.git.as_deref(),
            _ => None,
        }
    }
}

// ============================================================================
// Tome Operations
// ============================================================================

/// Conjure a new tome (create new project)
pub fn conjure(name: &str, path: Option<&Path>) -> Result<PathBuf, String> {
    let project_path = path
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from(name));

    if project_path.exists() {
        return Err(format!(
            "Cannot conjure '{}': path already exists",
            project_path.display()
        ));
    }

    // Create directory structure
    fs::create_dir_all(&project_path).map_err(|e| format!("Failed to create directory: {}", e))?;
    fs::create_dir_all(project_path.join("src"))
        .map_err(|e| format!("Failed to create src directory: {}", e))?;

    // Create Grimoire.toml
    let grimoire = Grimoire {
        tome: TomeMetadata {
            name: name.to_string(),
            version: "0.1.0".to_string(),
            authors: get_git_author().map(|a| vec![a]).unwrap_or_default(),
            edition: Some("2026".to_string()),
            description: None,
            license: None,
            repository: None,
            homepage: None,
            keywords: vec![],
            categories: vec![],
        },
        bindings: HashMap::new(),
        dev_bindings: HashMap::new(),
        rites: HashMap::new(),
        workspace: None,
    };
    grimoire.save(&project_path)?;

    // Create main.sg
    let main_content = format!(
        r#"// {name} - A Sigil Tome
//
// Conjured with `sigil conjure {name}`

fn main() {{
    println("Hello from {name}!");
}}
"#
    );
    fs::write(project_path.join("src/main.sg"), main_content)
        .map_err(|e| format!("Failed to create main.sg: {}", e))?;

    // Create .gitignore
    let gitignore = r#"# Sigil build artifacts
/target/
/.tomes/

# Lock file (include for applications, exclude for libraries)
# Grimoire.lock
"#;
    fs::write(project_path.join(".gitignore"), gitignore)
        .map_err(|e| format!("Failed to create .gitignore: {}", e))?;

    Ok(project_path)
}

/// Inscribe current directory as a tome (init in existing directory)
pub fn inscribe(path: &Path) -> Result<(), String> {
    let grimoire_path = path.join(GRIMOIRE_TOML);
    if grimoire_path.exists() {
        return Err(format!(
            "Directory already inscribed: {} exists",
            GRIMOIRE_TOML
        ));
    }

    // Derive name from directory
    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unnamed")
        .to_string();

    let grimoire = Grimoire {
        tome: TomeMetadata {
            name,
            version: "0.1.0".to_string(),
            authors: get_git_author().map(|a| vec![a]).unwrap_or_default(),
            edition: Some("2026".to_string()),
            ..Default::default()
        },
        ..Default::default()
    };

    grimoire.save(path)?;

    // Create src directory if it doesn't exist
    let src_dir = path.join("src");
    if !src_dir.exists() {
        fs::create_dir_all(&src_dir)
            .map_err(|e| format!("Failed to create src directory: {}", e))?;
    }

    Ok(())
}

/// Summon a binding (add dependency)
pub fn summon(path: &Path, name: &str, spec: &str) -> Result<(), String> {
    let mut grimoire = Grimoire::load(path)?;

    let binding = parse_binding_spec(spec)?;
    grimoire.summon(name, binding);
    grimoire.save(path)?;

    Ok(())
}

/// Banish a binding (remove dependency)
pub fn banish(path: &Path, name: &str) -> Result<(), String> {
    let mut grimoire = Grimoire::load(path)?;

    if grimoire.banish(name).is_none() {
        return Err(format!("Binding '{}' not found in Grimoire", name));
    }

    grimoire.save(path)?;
    Ok(())
}

/// Attune bindings (update/resolve dependencies)
pub fn attune(path: &Path) -> Result<AttuneResult, String> {
    let grimoire = Grimoire::load(path)?;
    let mut result = AttuneResult::default();

    // Create .tomes directory
    let tomes_dir = path.join(TOMES_DIR);
    if !tomes_dir.exists() {
        fs::create_dir_all(&tomes_dir)
            .map_err(|e| format!("Failed to create .tomes directory: {}", e))?;
    }

    // Resolve each binding
    for (name, binding) in &grimoire.bindings {
        match resolve_binding(name, binding, &tomes_dir) {
            Ok(resolved) => {
                result.resolved.push(resolved);
            }
            Err(e) => {
                result.errors.push((name.clone(), e));
            }
        }
    }

    // Generate lock file
    if result.errors.is_empty() {
        let lock = generate_lock_file(&result.resolved);
        let lock_content = toml::to_string_pretty(&lock)
            .map_err(|e| format!("Failed to serialize lock file: {}", e))?;
        fs::write(path.join(GRIMOIRE_LOCK), lock_content)
            .map_err(|e| format!("Failed to write lock file: {}", e))?;
    }

    Ok(result)
}

/// Result of attunement
#[derive(Debug, Default)]
pub struct AttuneResult {
    pub resolved: Vec<ResolvedBinding>,
    pub errors: Vec<(String, String)>,
}

/// A resolved binding with its location
#[derive(Debug, Clone)]
pub struct ResolvedBinding {
    pub name: String,
    pub version: String,
    pub path: PathBuf,
    pub source: BindingSource,
}

/// Source type for a binding
#[derive(Debug, Clone)]
pub enum BindingSource {
    Registry,
    Path,
    Git { url: String, reference: String },
}

/// Forge the tome (build with dependencies)
pub fn forge(path: &Path) -> Result<ForgeResult, String> {
    let grimoire = Grimoire::load(path)?;
    let mut result = ForgeResult::default();

    // First, attune if needed
    let lock_path = path.join(GRIMOIRE_LOCK);
    if !lock_path.exists() {
        eprintln!("Attuning bindings...");
        let attune_result = attune(path)?;
        if !attune_result.errors.is_empty() {
            for (name, err) in &attune_result.errors {
                eprintln!("  Failed to resolve {}: {}", name, err);
            }
            return Err("Failed to attune bindings".to_string());
        }
    }

    // Check if this is a workspace
    if let Some(workspace) = &grimoire.workspace {
        if !workspace.members.is_empty() {
            return forge_workspace(path, &grimoire, workspace);
        }
    }

    // Find main source file for single-tome project
    let main_file = find_main_source(path)?;

    result.main_file = Some(main_file);
    result.tome_name = grimoire.tome.name.clone();
    result.version = grimoire.tome.version.clone();

    Ok(result)
}

/// Find the main source file in a tome directory
fn find_main_source(path: &Path) -> Result<PathBuf, String> {
    let src_dir = path.join("src");

    // Check standard locations
    for filename in &["main.sg", "main.sigil", "lib.sg", "lib.sigil"] {
        let file_path = src_dir.join(filename);
        if file_path.exists() {
            return Ok(file_path);
        }
    }

    Err(format!(
        "No main.sg, main.sigil, lib.sg, or lib.sigil found in {}/src/",
        path.display()
    ))
}

/// Forge a workspace with multiple member tomes
fn forge_workspace(
    workspace_path: &Path,
    _grimoire: &Grimoire,
    workspace: &Workspace,
) -> Result<ForgeResult, String> {
    let mut result = ForgeResult::default();
    result.tome_name = "workspace".to_string();

    eprintln!(
        "Forging workspace with {} members...",
        workspace.members.len()
    );

    for member_path in &workspace.members {
        let member_full_path = workspace_path.join(member_path);

        // Check if member has a Grimoire.toml
        let member_grimoire_path = member_full_path.join(GRIMOIRE_TOML);
        if !member_grimoire_path.exists() {
            eprintln!("  Skipping {}: no Grimoire.toml found", member_path);
            continue;
        }

        // Load and forge the member
        match Grimoire::load(&member_full_path) {
            Ok(member_grimoire) => {
                eprintln!("  Forging {}...", member_grimoire.tome.name);

                // Find the main source file for this member
                match find_main_source(&member_full_path) {
                    Ok(main_file) => {
                        result.artifacts.push(main_file);
                    }
                    Err(e) => {
                        eprintln!("    Warning: {}", e);
                    }
                }
            }
            Err(e) => {
                eprintln!("  Warning: Failed to load {}: {}", member_path, e);
            }
        }
    }

    if result.artifacts.is_empty() {
        return Err("No buildable members found in workspace".to_string());
    }

    eprintln!("Found {} buildable members", result.artifacts.len());
    Ok(result)
}

/// Result of forging
#[derive(Debug, Default)]
pub struct ForgeResult {
    pub tome_name: String,
    pub version: String,
    pub main_file: Option<PathBuf>,
    pub artifacts: Vec<PathBuf>,
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Parse a binding specification string
fn parse_binding_spec(spec: &str) -> Result<Binding, String> {
    // Check for path: prefix
    if spec.starts_with("path:") {
        let path = spec.strip_prefix("path:").unwrap().trim();
        return Ok(Binding::Detailed(BindingSpec {
            path: Some(path.to_string()),
            ..Default::default()
        }));
    }

    // Check for git: prefix
    if spec.starts_with("git:") {
        let url = spec.strip_prefix("git:").unwrap().trim();
        return Ok(Binding::Detailed(BindingSpec {
            git: Some(url.to_string()),
            ..Default::default()
        }));
    }

    // Otherwise, treat as version
    Ok(Binding::Version(spec.to_string()))
}

/// Resolve a binding to a local path
fn resolve_binding(
    name: &str,
    binding: &Binding,
    tomes_dir: &Path,
) -> Result<ResolvedBinding, String> {
    match binding {
        Binding::Version(version) => {
            // Registry binding - fetch from Grimoire
            resolve_registry_binding(name, version, tomes_dir)
        }
        Binding::Detailed(spec) => {
            if let Some(path) = &spec.path {
                // Local path binding
                let resolved_path = PathBuf::from(path);
                if !resolved_path.exists() {
                    return Err(format!("Path does not exist: {}", path));
                }
                Ok(ResolvedBinding {
                    name: name.to_string(),
                    version: spec.version.clone().unwrap_or_else(|| "0.0.0".to_string()),
                    path: resolved_path,
                    source: BindingSource::Path,
                })
            } else if let Some(git_url) = &spec.git {
                // Git binding
                let reference = spec
                    .branch
                    .clone()
                    .or_else(|| spec.tag.clone())
                    .or_else(|| spec.rev.clone())
                    .unwrap_or_else(|| "main".to_string());

                let clone_dir = tomes_dir.join(name);
                clone_git_repo(git_url, &reference, &clone_dir)?;

                Ok(ResolvedBinding {
                    name: name.to_string(),
                    version: spec
                        .version
                        .clone()
                        .unwrap_or_else(|| "0.0.0-git".to_string()),
                    path: clone_dir,
                    source: BindingSource::Git {
                        url: git_url.clone(),
                        reference,
                    },
                })
            } else if let Some(version) = &spec.version {
                // Registry binding with version in detailed spec
                resolve_registry_binding(name, version, tomes_dir)
            } else {
                Err(format!(
                    "Invalid binding for '{}': must specify version, path, or git",
                    name
                ))
            }
        }
    }
}

/// Clone a git repository
fn clone_git_repo(url: &str, reference: &str, dest: &Path) -> Result<(), String> {
    use std::process::Command;

    if dest.exists() {
        // Pull instead of clone
        let output = Command::new("git")
            .args(["pull", "--ff-only"])
            .current_dir(dest)
            .output()
            .map_err(|e| format!("Failed to run git pull: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("git pull failed: {}", stderr));
        }
    } else {
        // Clone
        let output = Command::new("git")
            .args(["clone", "--depth", "1", "--branch", reference, url])
            .arg(dest)
            .output()
            .map_err(|e| format!("Failed to run git clone: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("git clone failed: {}", stderr));
        }
    }

    Ok(())
}

/// Generate a lock file from resolved bindings
fn generate_lock_file(resolved: &[ResolvedBinding]) -> GrimoireLock {
    let bindings = resolved
        .iter()
        .map(|r| LockedBinding {
            name: r.name.clone(),
            version: r.version.clone(),
            source: match &r.source {
                BindingSource::Registry => "registry".to_string(),
                BindingSource::Path => format!("path:{}", r.path.display()),
                BindingSource::Git { url, reference } => format!("git:{}#{}", url, reference),
            },
            checksum: None,
            dependencies: vec![],
        })
        .collect();

    GrimoireLock {
        version: 1,
        bindings,
    }
}

/// Get git author from git config
fn get_git_author() -> Option<String> {
    use std::process::Command;

    let name = Command::new("git")
        .args(["config", "user.name"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())?;

    let email = Command::new("git")
        .args(["config", "user.email"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())?;

    if name.is_empty() {
        None
    } else if email.is_empty() {
        Some(name)
    } else {
        Some(format!("{} <{}>", name, email))
    }
}

// ============================================================================
// Registry Operations
// ============================================================================

/// Cache for registry index (avoids repeated fetches during attune)
static REGISTRY_CACHE: std::sync::OnceLock<RegistryIndex> = std::sync::OnceLock::new();

/// Fetch the registry index from the Grimoire
pub fn fetch_registry_index() -> Result<&'static RegistryIndex, String> {
    // Return cached index if available
    if let Some(index) = REGISTRY_CACHE.get() {
        return Ok(index);
    }

    // Fetch from registry
    let url = format!("{}/{}", REGISTRY_URL, REGISTRY_INDEX);
    let index = fetch_and_parse_index(&url)?;

    // Cache and return
    Ok(REGISTRY_CACHE.get_or_init(|| index))
}

/// Fetch and parse the index from a URL
fn fetch_and_parse_index(url: &str) -> Result<RegistryIndex, String> {
    use std::process::Command;

    // Use curl to fetch (available on most systems)
    let output = Command::new("curl")
        .args(["-sSf", "--max-time", "30", url])
        .output()
        .map_err(|e| format!("Failed to fetch registry index: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!(
            "Failed to fetch registry index from {}: {}",
            url, stderr
        ));
    }

    let json = String::from_utf8(output.stdout)
        .map_err(|e| format!("Invalid UTF-8 in registry index: {}", e))?;

    serde_json::from_str(&json).map_err(|e| format!("Failed to parse registry index: {}", e))
}

/// Find a tome in the registry
pub fn find_tome_in_registry(name: &str) -> Result<&'static RegistryTome, String> {
    let index = fetch_registry_index()?;
    index
        .tomes
        .get(name)
        .ok_or_else(|| format!("Tome '{}' not found in Grimoire registry", name))
}

/// Resolve a version requirement to a specific version
pub fn resolve_version(tome: &RegistryTome, requirement: &str) -> Result<String, String> {
    // Simple version matching for now
    // Supports: exact ("1.0.0"), caret ("^1.0"), tilde ("~1.0"), wildcard ("*")

    let requirement = requirement.trim();

    // Wildcard - use latest
    if requirement == "*" || requirement.is_empty() {
        return Ok(tome.latest.clone());
    }

    // Exact version
    if !requirement.starts_with('^')
        && !requirement.starts_with('~')
        && !requirement.starts_with('>')
        && !requirement.starts_with('<')
    {
        // Check if exact version exists
        if tome.versions.contains(&requirement.to_string()) {
            if tome.yanked.contains(&requirement.to_string()) {
                return Err(format!("Version {} has been yanked", requirement));
            }
            return Ok(requirement.to_string());
        }
        return Err(format!(
            "Version {} not found. Available: {}",
            requirement,
            tome.versions.join(", ")
        ));
    }

    // Caret version (^1.0 means >=1.0.0 <2.0.0)
    if let Some(base) = requirement.strip_prefix('^') {
        let parts: Vec<&str> = base.split('.').collect();
        let major: u32 = parts
            .first()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);

        // Find highest matching version
        for version in &tome.versions {
            if tome.yanked.contains(version) {
                continue;
            }
            let v_parts: Vec<&str> = version.split('.').collect();
            let v_major: u32 = v_parts
                .first()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);

            if v_major == major && version_satisfies(version, base) {
                return Ok(version.clone());
            }
        }
        return Err(format!("No version satisfies {}", requirement));
    }

    // Tilde version (~1.0 means >=1.0.0 <1.1.0)
    if let Some(base) = requirement.strip_prefix('~') {
        let parts: Vec<&str> = base.split('.').collect();
        let major: u32 = parts.first().and_then(|s| s.parse().ok()).unwrap_or(0);
        let minor: u32 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);

        for version in &tome.versions {
            if tome.yanked.contains(version) {
                continue;
            }
            let v_parts: Vec<&str> = version.split('.').collect();
            let v_major: u32 = v_parts.first().and_then(|s| s.parse().ok()).unwrap_or(0);
            let v_minor: u32 = v_parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(0);

            if v_major == major && v_minor == minor && version_satisfies(version, base) {
                return Ok(version.clone());
            }
        }
        return Err(format!("No version satisfies {}", requirement));
    }

    // Fallback to latest for unrecognized formats
    Ok(tome.latest.clone())
}

/// Check if version >= base
fn version_satisfies(version: &str, base: &str) -> bool {
    let v_parts: Vec<u32> = version
        .split('.')
        .filter_map(|s| s.parse().ok())
        .collect();
    let b_parts: Vec<u32> = base.split('.').filter_map(|s| s.parse().ok()).collect();

    for i in 0..3 {
        let v = v_parts.get(i).copied().unwrap_or(0);
        let b = b_parts.get(i).copied().unwrap_or(0);
        if v > b {
            return true;
        }
        if v < b {
            return false;
        }
    }
    true // Equal
}

/// Download and extract a tome from the registry
pub fn download_tome(
    name: &str,
    version: &str,
    tomes_dir: &Path,
) -> Result<PathBuf, String> {
    let tome_dir = tomes_dir.join(format!("{}-{}", name, version));

    // Skip if already downloaded
    if tome_dir.exists() {
        let grimoire_path = tome_dir.join(GRIMOIRE_TOML);
        if grimoire_path.exists() {
            return Ok(tome_dir);
        }
        // Incomplete download, remove and retry
        fs::remove_dir_all(&tome_dir)
            .map_err(|e| format!("Failed to clean incomplete download: {}", e))?;
    }

    // Download tarball
    let tarball_url = format!("{}/tomes/{}/{}.tar.gz", REGISTRY_URL, name, version);
    let tarball_path = tomes_dir.join(format!("{}-{}.tar.gz", name, version));

    download_file(&tarball_url, &tarball_path)?;

    // Extract tarball
    extract_tarball(&tarball_path, tomes_dir)?;

    // Clean up tarball
    let _ = fs::remove_file(&tarball_path);

    // Verify extraction
    if !tome_dir.exists() {
        return Err(format!(
            "Extraction failed: expected directory {} not found",
            tome_dir.display()
        ));
    }

    Ok(tome_dir)
}

/// Download a file from URL to path
fn download_file(url: &str, dest: &Path) -> Result<(), String> {
    use std::process::Command;

    // Ensure parent directory exists
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create directory: {}", e))?;
    }

    let output = Command::new("curl")
        .args([
            "-sSfL",
            "--max-time",
            "120",
            "-o",
            dest.to_str().unwrap_or(""),
            url,
        ])
        .output()
        .map_err(|e| format!("Failed to download {}: {}", url, e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Download failed for {}: {}", url, stderr));
    }

    Ok(())
}

/// Extract a tarball to a directory
fn extract_tarball(tarball: &Path, dest: &Path) -> Result<(), String> {
    use std::process::Command;

    let output = Command::new("tar")
        .args([
            "-xzf",
            tarball.to_str().unwrap_or(""),
            "-C",
            dest.to_str().unwrap_or(""),
        ])
        .output()
        .map_err(|e| format!("Failed to extract tarball: {}", e))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("Extraction failed: {}", stderr));
    }

    Ok(())
}

/// Resolve a registry binding (version string) to a local path
fn resolve_registry_binding(
    name: &str,
    version: &str,
    tomes_dir: &Path,
) -> Result<ResolvedBinding, String> {
    // Find tome in registry
    let tome = find_tome_in_registry(name)?;

    // Resolve version requirement to specific version
    let resolved_version = resolve_version(tome, version)?;

    // Download if needed
    let tome_path = download_tome(name, &resolved_version, tomes_dir)?;

    Ok(ResolvedBinding {
        name: name.to_string(),
        version: resolved_version,
        path: tome_path,
        source: BindingSource::Registry,
    })
}

/// List available rites (scripts)
pub fn list_rites(path: &Path) -> Result<Vec<(String, String)>, String> {
    let grimoire = Grimoire::load(path)?;
    Ok(grimoire.rites.into_iter().collect())
}

/// Run a rite (script)
pub fn invoke_rite(path: &Path, rite_name: &str) -> Result<(), String> {
    let grimoire = Grimoire::load(path)?;

    let command = grimoire
        .rites
        .get(rite_name)
        .ok_or_else(|| format!("Unknown rite: {}", rite_name))?;

    use std::process::Command;

    let status = if cfg!(windows) {
        Command::new("cmd")
            .args(["/C", command])
            .current_dir(path)
            .status()
    } else {
        Command::new("sh")
            .args(["-c", command])
            .current_dir(path)
            .status()
    };

    match status {
        Ok(s) if s.success() => Ok(()),
        Ok(s) => Err(format!("Rite failed with exit code: {:?}", s.code())),
        Err(e) => Err(format!("Failed to invoke rite: {}", e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_binding_version() {
        let binding = parse_binding_spec("0.1.0").unwrap();
        assert!(matches!(binding, Binding::Version(v) if v == "0.1.0"));
    }

    #[test]
    fn test_parse_binding_path() {
        let binding = parse_binding_spec("path:../chorus").unwrap();
        assert!(binding.is_path());
        assert_eq!(binding.path(), Some("../chorus"));
    }

    #[test]
    fn test_parse_binding_git() {
        let binding = parse_binding_spec("git:https://github.com/example/repo").unwrap();
        assert!(binding.is_git());
        assert_eq!(binding.git(), Some("https://github.com/example/repo"));
    }

    #[test]
    fn test_grimoire_serialization() {
        let grimoire = Grimoire {
            tome: TomeMetadata {
                name: "test-tome".to_string(),
                version: "0.1.0".to_string(),
                ..Default::default()
            },
            bindings: {
                let mut b = HashMap::new();
                b.insert("aegis".to_string(), Binding::Version("0.1".to_string()));
                b
            },
            ..Default::default()
        };

        let toml = toml::to_string_pretty(&grimoire).unwrap();
        assert!(toml.contains("[tome]"));
        assert!(toml.contains("name = \"test-tome\""));
        assert!(toml.contains("[bindings]"));
    }

    // Registry tests

    #[test]
    fn test_version_satisfies() {
        // Equal versions
        assert!(version_satisfies("1.0.0", "1.0.0"));
        assert!(version_satisfies("1.2.3", "1.2.3"));

        // Greater versions
        assert!(version_satisfies("1.1.0", "1.0.0"));
        assert!(version_satisfies("2.0.0", "1.0.0"));
        assert!(version_satisfies("1.0.1", "1.0.0"));

        // Lesser versions
        assert!(!version_satisfies("0.9.0", "1.0.0"));
        assert!(!version_satisfies("1.0.0", "1.0.1"));
    }

    #[test]
    fn test_resolve_version_exact() {
        let tome = RegistryTome {
            description: "Test".to_string(),
            versions: vec!["1.0.0".to_string(), "0.9.0".to_string()],
            latest: "1.0.0".to_string(),
            yanked: vec![],
        };

        // Exact match
        assert_eq!(resolve_version(&tome, "1.0.0").unwrap(), "1.0.0");
        assert_eq!(resolve_version(&tome, "0.9.0").unwrap(), "0.9.0");

        // Version not found
        assert!(resolve_version(&tome, "2.0.0").is_err());
    }

    #[test]
    fn test_resolve_version_wildcard() {
        let tome = RegistryTome {
            description: "Test".to_string(),
            versions: vec!["1.0.0".to_string()],
            latest: "1.0.0".to_string(),
            yanked: vec![],
        };

        // Wildcard returns latest
        assert_eq!(resolve_version(&tome, "*").unwrap(), "1.0.0");
        assert_eq!(resolve_version(&tome, "").unwrap(), "1.0.0");
    }

    #[test]
    fn test_resolve_version_caret() {
        let tome = RegistryTome {
            description: "Test".to_string(),
            versions: vec![
                "1.2.0".to_string(),
                "1.1.0".to_string(),
                "1.0.0".to_string(),
                "0.9.0".to_string(),
            ],
            latest: "1.2.0".to_string(),
            yanked: vec![],
        };

        // Caret version ^1.0 means >=1.0.0 <2.0.0
        assert_eq!(resolve_version(&tome, "^1.0").unwrap(), "1.2.0");
        assert_eq!(resolve_version(&tome, "^1.1").unwrap(), "1.2.0");

        // No matching version
        assert!(resolve_version(&tome, "^2.0").is_err());
    }

    #[test]
    fn test_resolve_version_yanked() {
        let tome = RegistryTome {
            description: "Test".to_string(),
            versions: vec!["1.0.0".to_string(), "0.9.0".to_string()],
            latest: "1.0.0".to_string(),
            yanked: vec!["1.0.0".to_string()],
        };

        // Yanked version returns error
        assert!(resolve_version(&tome, "1.0.0").is_err());

        // Non-yanked version works
        assert_eq!(resolve_version(&tome, "0.9.0").unwrap(), "0.9.0");
    }

    #[test]
    fn test_registry_index_parsing() {
        let json = r#"{
            "tomes": {
                "aegis": {
                    "description": "Security primitives",
                    "versions": ["0.2.0", "0.1.0"],
                    "latest": "0.2.0",
                    "yanked": []
                }
            }
        }"#;

        let index: RegistryIndex = serde_json::from_str(json).unwrap();
        assert!(index.tomes.contains_key("aegis"));
        let aegis = &index.tomes["aegis"];
        assert_eq!(aegis.latest, "0.2.0");
        assert_eq!(aegis.versions.len(), 2);
    }

    #[test]
    fn test_registry_tome_default_yanked() {
        // Test that yanked defaults to empty when not present
        let json = r#"{
            "description": "Test",
            "versions": ["1.0.0"],
            "latest": "1.0.0"
        }"#;

        let tome: RegistryTome = serde_json::from_str(json).unwrap();
        assert!(tome.yanked.is_empty());
    }
}
