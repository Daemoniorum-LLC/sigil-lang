//! Dependency resolution for multi-crate WASM compilation.
//!
//! Parses sigil.toml files and resolves dependency paths for bundled compilation.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};

use super::error::{WasmError, WasmResult};

/// A parsed sigil.toml dependency specification.
#[derive(Debug, Clone)]
pub struct Dependency {
    pub name: String,
    pub path: Option<PathBuf>,
    pub version: Option<String>,
}

/// Project manifest parsed from sigil.toml.
#[derive(Debug, Clone)]
pub struct ProjectManifest {
    pub name: String,
    pub version: String,
    pub lib_path: PathBuf,
    pub dependencies: Vec<Dependency>,
    pub root_dir: PathBuf,
}

impl ProjectManifest {
    /// Parse a sigil.toml or Sigil.toml file from the given directory.
    pub fn from_dir(dir: &Path) -> WasmResult<Self> {
        // Try both sigil.toml and Sigil.toml (case-insensitive)
        let toml_path = dir.join("sigil.toml");
        let toml_path_cap = dir.join("Sigil.toml");

        let actual_path = if toml_path.exists() {
            toml_path
        } else if toml_path_cap.exists() {
            toml_path_cap
        } else {
            return Err(WasmError::io(format!(
                "no sigil.toml or Sigil.toml found in {}",
                dir.display()
            )));
        };

        let content = fs::read_to_string(&actual_path)
            .map_err(|e| WasmError::io(format!("cannot read {}: {}", actual_path.display(), e)))?;

        Self::parse(&content, dir)
    }

    /// Parse sigil.toml content.
    fn parse(content: &str, root_dir: &Path) -> WasmResult<Self> {
        let mut name = String::new();
        let mut version = String::new();
        let mut lib_path = PathBuf::from("src/lib.sigil");
        let mut dependencies = Vec::new();

        let mut in_package = false;
        let mut in_dependencies = false;
        let mut current_dep_name: Option<String> = None;
        let mut current_dep_path: Option<String> = None;

        for line in content.lines() {
            let line = line.trim();

            // Skip comments and empty lines
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Section headers (support both [package] and [project])
            if line == "[package]" || line == "[project]" {
                in_package = true;
                in_dependencies = false;
                continue;
            }
            if line == "[dependencies]" {
                in_package = false;
                in_dependencies = true;
                continue;
            }
            if line.starts_with('[') {
                in_package = false;
                in_dependencies = false;
                continue;
            }

            // Parse key-value pairs
            if let Some((key, value)) = line.split_once('=') {
                let key = key.trim();
                let value = value.trim().trim_matches('"');

                if in_package {
                    match key {
                        "name" => name = value.to_string(),
                        "version" => version = value.to_string(),
                        _ => {}
                    }
                } else if in_dependencies {
                    // Handle inline table: dep = { path = "..." }
                    if value.starts_with('{') {
                        let inner = value.trim_start_matches('{').trim_end_matches('}');
                        let mut dep_path = None;
                        let mut dep_version = None;

                        for part in inner.split(',') {
                            if let Some((k, v)) = part.split_once('=') {
                                let k = k.trim();
                                let v = v.trim().trim_matches('"');
                                match k {
                                    "path" => dep_path = Some(v.to_string()),
                                    "version" => dep_version = Some(v.to_string()),
                                    _ => {}
                                }
                            }
                        }

                        dependencies.push(Dependency {
                            name: key.to_string(),
                            path: dep_path.map(PathBuf::from),
                            version: dep_version,
                        });
                    } else {
                        // Simple version string: dep = "1.0"
                        dependencies.push(Dependency {
                            name: key.to_string(),
                            path: None,
                            version: Some(value.to_string()),
                        });
                    }
                }
            }
        }

        // Check for lib entry point
        let lib_sigil = root_dir.join("src/lib.sigil");
        let mod_sigil = root_dir.join("src/mod.sigil");

        if lib_sigil.exists() {
            lib_path = lib_sigil;
        } else if mod_sigil.exists() {
            lib_path = mod_sigil;
        }

        Ok(ProjectManifest {
            name,
            version,
            lib_path,
            dependencies,
            root_dir: root_dir.to_path_buf(),
        })
    }
}

/// Dependency graph for topological ordering.
pub struct DependencyGraph {
    manifests: HashMap<String, ProjectManifest>,
    order: Vec<String>,
}

impl DependencyGraph {
    /// Build a dependency graph starting from a root project.
    pub fn from_project(root_dir: &Path) -> WasmResult<Self> {
        let mut graph = DependencyGraph {
            manifests: HashMap::new(),
            order: Vec::new(),
        };

        let mut visited = HashSet::new();
        let mut stack = HashSet::new();

        graph.resolve_recursive(root_dir, &mut visited, &mut stack)?;

        Ok(graph)
    }

    fn resolve_recursive(
        &mut self,
        dir: &Path,
        visited: &mut HashSet<PathBuf>,
        stack: &mut HashSet<PathBuf>,
    ) -> WasmResult<()> {
        let canonical = dir.canonicalize()
            .map_err(|e| WasmError::io(format!("cannot resolve {}: {}", dir.display(), e)))?;

        // Check for circular dependency
        if stack.contains(&canonical) {
            return Err(WasmError::internal(format!(
                "circular dependency detected at {}",
                dir.display()
            )));
        }

        // Already processed
        if visited.contains(&canonical) {
            return Ok(());
        }

        stack.insert(canonical.clone());

        // Parse manifest
        let manifest = ProjectManifest::from_dir(dir)?;
        let name = manifest.name.clone();

        // Process dependencies first (post-order traversal)
        for dep in &manifest.dependencies {
            if let Some(ref path) = dep.path {
                let dep_dir = dir.join(path);
                if dep_dir.exists() {
                    self.resolve_recursive(&dep_dir, visited, stack)?;
                }
            }
        }

        // Add this manifest after its dependencies
        self.manifests.insert(name.clone(), manifest);
        self.order.push(name);

        stack.remove(&canonical);
        visited.insert(canonical);

        Ok(())
    }

    /// Get manifests in dependency order (dependencies first).
    pub fn iter_in_order(&self) -> impl Iterator<Item = &ProjectManifest> {
        self.order.iter().filter_map(|name| self.manifests.get(name))
    }

    /// Get the root project manifest.
    pub fn root(&self) -> Option<&ProjectManifest> {
        self.order.last().and_then(|name| self.manifests.get(name))
    }

    /// Get all manifests.
    pub fn manifests(&self) -> &HashMap<String, ProjectManifest> {
        &self.manifests
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_manifest() {
        let content = r#"
[package]
name = "my-app"
version = "0.1.0"

[dependencies]
qliphoth = { path = "../qliphoth" }
qliphoth-sys = { path = "../qliphoth/packages/qliphoth-sys" }
"#;

        let manifest = ProjectManifest::parse(content, Path::new("/tmp")).unwrap();
        assert_eq!(manifest.name, "my-app");
        assert_eq!(manifest.version, "0.1.0");
        assert_eq!(manifest.dependencies.len(), 2);
        assert_eq!(manifest.dependencies[0].name, "qliphoth");
        assert!(manifest.dependencies[0].path.is_some());
    }
}
