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
            // Registry binding - placeholder for future Grimoire registry
            Err(format!(
                "Registry bindings not yet implemented. Use path: or git: for '{}'",
                name
            ))
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
                // Registry binding with version
                Err(format!(
                    "Registry bindings not yet implemented. Use path: or git: for '{}'",
                    name
                ))
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
}
