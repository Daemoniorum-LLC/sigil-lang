//! MCP (Model Context Protocol) interface for React migration.
//!
//! Provides tools and resources for AI agents to interact with the migration system:
//! - Tools: list_migrations, get_migration, validate_sigil, complete_migration
//! - Resources: migrations://pending, migrations://patterns, migrations://component/{id}
//!
//! See docs/specs/REACT-MIGRATION.md Section 5 for specification.

use super::extraction::*;
use super::spec::*;
use super::generator::*;
use crate::parser::{Parser, ParseError};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs;

// =============================================================================
// MCP Response Types
// =============================================================================

/// Summary of a migration for listing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MigrationSummary {
    pub id: String,
    pub name: String,
    pub status: MigrationStatus,
    pub complexity: Complexity,
    pub blocked_by: Vec<String>,
}

/// Result of validating Sigil code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub valid: bool,
    pub errors: Vec<ValidationError>,
    pub warnings: Vec<ValidationWarning>,
}

/// A validation error in generated Sigil code
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationError {
    pub line: usize,
    pub column: usize,
    pub message: String,
    pub suggestion: Option<String>,
}

/// A validation warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationWarning {
    pub line: usize,
    pub column: usize,
    pub message: String,
}

/// Result of completing a migration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResult {
    pub success: bool,
    pub output_path: String,
    pub next_suggested: Vec<String>,
}

/// Filter for pattern queries
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PatternFilter {
    pub name: Option<String>,
    pub category: Option<String>,
}

// =============================================================================
// Session State (for persistence)
// =============================================================================

/// Serializable session state for save/load.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SessionState {
    spec: MigrationSpec,
    status: HashMap<String, MigrationStatus>,
    completed: HashMap<String, String>,
    resolved_ambiguities: HashMap<String, HashMap<String, usize>>,
    output_dir: PathBuf,
}

// =============================================================================
// Migration Session
// =============================================================================

/// Maintains state for a migration session.
/// This is the main entry point for MCP tools.
pub struct MigrationSession {
    /// Root path of the React project being migrated
    project_root: PathBuf,

    /// Output directory for generated Sigil files
    output_dir: PathBuf,

    /// The full migration spec
    spec: MigrationSpec,

    /// Status of each component (keyed by ID)
    status: HashMap<String, MigrationStatus>,

    /// Generated code for completed migrations
    completed: HashMap<String, String>,

    /// Resolved ambiguities
    resolved_ambiguities: HashMap<String, HashMap<String, usize>>,

    /// Every `export const` seen anywhere in the project, by exported name.
    ///
    /// A component's module constants come from its own file, but React code
    /// keeps shared tables in sibling modules — `INVENTORY_KINDS` in
    /// `inventory-logic.ts`, `ANIMA_DIMENSIONS` in `types.ts`. Those names are
    /// import specifiers, so they are module scope and never get a `self.`
    /// prefix, and nothing then declares them: the generated component
    /// referenced a binding that did not exist. The session already parses
    /// every `.ts` in the tree, so the definitions are in hand — they just were
    /// not carried across the file boundary.
    exported_constants: HashMap<String, ModuleConstantExtraction>,

    /// Every exported helper function seen anywhere in the project, by name.
    /// The companion to `exported_constants`, and resolved the same way.
    exported_helpers: HashMap<String, HelperFunctionExtraction>,

    /// Every helper function seen anywhere, exported or not.
    ///
    /// The companion to `all_constants`, and missing for exactly as long as
    /// that one was. A file-private helper cannot be an import specifier, so
    /// nothing seeds the queue with it — but an EXPORTED helper's body calls
    /// it, and that call is how it has to travel. `isMrList` is exported and
    /// arrived; the `isMr` it calls and the `isObj` that calls in turn are
    /// file-private, and both were dropped. `isObj` alone blocked 15 helpers
    /// in the Lares client, each of which blocked more.
    all_helpers: HashMap<String, HelperFunctionExtraction>,

    /// Every module constant seen anywhere, exported or not.
    ///
    /// Only reachable transitively: a name that is not exported cannot be an
    /// import specifier, so nothing seeds the queue with it. But it can be what
    /// an exported constant is made of — `export const DEFAULT_PROJECT =
    /// PROJECT_KEY` beside a file-private `const PROJECT_KEY = …` — and pulling
    /// the first without the second just moves the undefined name.
    all_constants: HashMap<String, ModuleConstantExtraction>,
}

impl MigrationSession {
    /// Create a new migration session from a React project.
    pub fn new(project_root: impl AsRef<Path>, output_dir: impl AsRef<Path>) -> Result<Self, McpError> {
        let project_root = project_root.as_ref().to_path_buf();
        let output_dir = output_dir.as_ref().to_path_buf();

        // Create empty spec - will be populated via extract_project
        let spec = MigrationSpec {
            version: "1.0".to_string(),
            generated_at: chrono_now(),
            project_root: project_root.display().to_string(),
            components: Vec::new(),
            types: Vec::new(),
            helper_functions: Vec::new(),
            service_actors: Vec::new(),
            state: MigrationState {
                total_components: 0,
                completed: 0,
                in_progress: 0,
                blocked: 0,
                last_updated: chrono_now(),
            },
        };

        Ok(Self {
            project_root,
            output_dir,
            spec,
            status: HashMap::new(),
            completed: HashMap::new(),
            resolved_ambiguities: HashMap::new(),
            exported_constants: HashMap::new(),
            exported_helpers: HashMap::new(),
            all_helpers: HashMap::new(),
            all_constants: HashMap::new(),
        })
    }

    /// Create a session from an existing MigrationSpec.
    pub fn from_spec(spec: MigrationSpec, output_dir: impl AsRef<Path>) -> Self {
        let project_root = PathBuf::from(&spec.project_root);
        let output_dir = output_dir.as_ref().to_path_buf();

        // Initialize status from spec
        let mut status = HashMap::new();
        for comp in &spec.components {
            status.insert(comp.id.clone(), comp.status);
        }

        Self {
            project_root,
            output_dir,
            spec,
            status,
            completed: HashMap::new(),
            resolved_ambiguities: HashMap::new(),
            exported_constants: HashMap::new(),
            exported_helpers: HashMap::new(),
            all_helpers: HashMap::new(),
            all_constants: HashMap::new(),
        }
    }

    /// Extract and add a single React file to the session.
    pub fn add_file(&mut self, path: impl AsRef<Path>, source: &str) -> Result<(), McpError> {
        let path = path.as_ref();
        let relative_path = path.strip_prefix(&self.project_root)
            .unwrap_or(path)
            .display()
            .to_string();

        // Extract the React file
        let extraction = extract_source(source, path, &relative_path)
            .map_err(|e| McpError::ExtractionError(format!("{:?}", e)))?;

        for c in &extraction.module_constants {
            if c.exported {
                self.exported_constants
                    .entry(c.name.clone())
                    .or_insert_with(|| c.clone());
            }
            self.all_constants
                .entry(c.name.clone())
                .or_insert_with(|| c.clone());
        }
        for h in &extraction.helper_functions {
            if h.exported {
                self.exported_helpers
                    .entry(h.name.clone())
                    .or_insert_with(|| h.clone());
            }
            self.all_helpers
                .entry(h.name.clone())
                .or_insert_with(|| h.clone());
        }

        // Generate spec for each component
        let component_specs = generate_spec(&extraction, source);

        // Add components to our spec
        for comp_spec in component_specs.components {
            let id = comp_spec.id.clone();
            self.status.insert(id.clone(), MigrationStatus::Pending);
            self.spec.components.push(comp_spec);
        }

        // Add types
        for type_spec in component_specs.types {
            self.spec.types.push(type_spec);
        }

        // Add helper functions (Phase 6.2)
        for helper in component_specs.helper_functions {
            self.spec.helper_functions.push(helper);
        }

        // Update state
        self.update_state();

        Ok(())
    }

    /// Pull in every exported constant a component imports from another file.
    ///
    /// Run once, after all files are added — a component can import from a
    /// module that had not been parsed yet when its own spec was generated.
    /// Transitive: `DEFAULT_PROJECT = PROJECT_KEY` needs `PROJECT_KEY` too, and
    /// emitting the first without the second just moves the undefined name.
    pub fn resolve_cross_module_imports(&mut self) {
        let table = self.exported_constants.clone();
        let deep = self.all_constants.clone();
        let helpers = self.exported_helpers.clone();
        let deep_helpers = self.all_helpers.clone();
        for comp in &mut self.spec.components {
            let mut have: std::collections::HashSet<String> = comp
                .source
                .module_constants
                .iter()
                .map(|c| c.name.clone())
                .collect();

            // `module_scope` already carries every import specifier this
            // component's file brings in, alongside its own helpers and
            // constants — the latter two are in `have` and skip themselves.
            let mut queue: Vec<String> = comp.source.module_scope.clone();

            // …and every name mentioned by a helper this component will pull
            // in. A file-private constant that only an imported helper's body
            // names is in nobody's `module_scope`, so nothing queued it:
            // `mirrorPrefix`, `builtinThemes`, `checkpointLabels` and
            // `terminalTokenKeys` all reached the generator as undeclared
            // names, and the helpers that read them stayed untranslated.
            //
            // The helper walk below decides which helpers those are; this needs
            // the same reachable set, so it is computed once here.
            let reachable_helpers =
                Self::reachable_helper_set(&comp.source, &helpers, &deep_helpers);
            for h in &reachable_helpers {
                for word in h
                    .source
                    .split(|ch: char| !(ch.is_alphanumeric() || ch == '_' || ch == '$'))
                {
                    if !word.is_empty() && (table.contains_key(word) || deep.contains_key(word)) {
                        queue.push(word.to_string());
                    }
                }
            }

            let mut added: Vec<ModuleConstantExtraction> = Vec::new();
            let mut guard = 0;
            while let Some(name) = queue.pop() {
                guard += 1;
                if guard > 512 {
                    break;
                }
                if have.contains(&name) {
                    continue;
                }
                // Seeded names must be exported (they came from an import
                // specifier); anything reached from a constant's own body may
                // be file-private.
                let c = match table.get(&name).or_else(|| deep.get(&name)) {
                    Some(c) => c,
                    None => continue,
                };
                have.insert(name.clone());
                // Whatever bare identifiers the initialiser mentions may
                // themselves be exported constants from elsewhere.
                for word in c
                    .init
                    .split(|ch: char| !(ch.is_alphanumeric() || ch == '_' || ch == '$'))
                {
                    if !word.is_empty() && (table.contains_key(word) || deep.contains_key(word)) {
                        queue.push(word.to_string());
                    }
                }
                added.push(c.clone());
            }

            // Dependencies before dependents. Reversing discovery order is not
            // enough: a constant can be reached both directly (it is imported
            // too) and as something another constant is made of, and then it is
            // discovered first and reversing puts it last. `DEFAULT_PROJECT` is
            // `PROJECT_KEY` is `identity.name`, and `identity` is imported by
            // the same file — so the three came out in exactly the wrong order.
            // Sort them by what they actually reference.
            added = topo_sort_constants(added);
            for c in &added {
                if !comp.source.module_scope.contains(&c.name) {
                    comp.source.module_scope.push(c.name.clone());
                }
            }
            // Imported constants go FIRST. A module-scope `≔` binding cannot be
            // read before it is written, and the file's own constants are the
            // ones that reference the imported ones, not the other way round.
            let mut merged = added;
            merged.append(&mut comp.source.module_constants);
            comp.source.module_constants = merged;

            // constant -> helper. The walks above cover helper->helper,
            // constant->constant and helper->constant; this is the fourth edge
            // and it was missing. A helper that only a module constant's
            // INITIALISER calls is in nobody's `module_scope`, and no helper
            // body names it either, so nothing queued it:
            //
            //   export const BUILTIN_THEMES = {
            //     "daemoniorum": { modes: { dark: daemoniorumTheme("dark") } },
            //     …
            //   }
            //
            // `BUILTIN_THEMES` is exported, reached and emitted; the five theme
            // builders it calls were not, and each call compiled to a constant
            // 0. This is S119 in the other direction — that one added the
            // constants a helper's body names, and left the helpers a
            // constant's initialiser names.
            let mut reachable_helpers = reachable_helpers;
            {
                let mut have_h: std::collections::HashSet<String> = reachable_helpers
                    .iter()
                    .map(|h| h.name.clone())
                    .chain(comp.source.helpers.iter().map(|h| h.name.clone()))
                    .collect();
                let mut seeds: Vec<String> = Vec::new();
                // The component's full constant set, imported and own: a
                // constant declared in the component's own file has the same
                // edge, and `module_constants` is the merged list by here.
                for c in &comp.source.module_constants {
                    for word in c
                        .init
                        .split(|ch: char| !(ch.is_alphanumeric() || ch == '_' || ch == '$'))
                    {
                        if !word.is_empty()
                            && !have_h.contains(word)
                            && (helpers.contains_key(word) || deep_helpers.contains_key(word))
                        {
                            have_h.insert(word.to_string());
                            seeds.push(word.to_string());
                        }
                    }
                }
                if !seeds.is_empty() {
                    let from_constants = Self::reachable_helpers_from(
                        seeds,
                        comp.source.helpers.iter().map(|h| h.name.clone()).collect(),
                        &helpers,
                        &deep_helpers,
                    );
                    for h in from_constants {
                        if !reachable_helpers.iter().any(|e| e.name == h.name) {
                            reachable_helpers.push(h);
                        }
                    }
                }
            }

            // Helpers, computed above: a helper's own body may reference
            // another helper, so that walk is transitive too.
            let added_h = reachable_helpers;
            for h in &added_h {
                if !comp.source.module_scope.contains(&h.name) {
                    comp.source.module_scope.push(h.name.clone());
                }
                if !comp.source.module_functions.contains(&h.name) {
                    comp.source.module_functions.push(h.name.clone());
                }
            }
            comp.source.helpers.extend(added_h);
        }
    }

    /// Every helper a component reaches, transitively, that it does not already
    /// have.
    ///
    /// Seeded names must be exported — they came from an import specifier.
    /// Anything reached from a helper's own body may be file-private, exactly
    /// as for constants: `isMrList` is exported and arrived, and the `isMr` it
    /// reaches through `v.every(isMr)` was dropped with the `isObj` behind it.
    ///
    /// The helper's SOURCE, not its `calls` list — that is empty on everything
    /// the extractor produces, and would miss a bare reference passed as an
    /// argument even if it were not.
    fn reachable_helper_set(
        source: &crate::migrate::react::spec::ComponentSource,
        exported: &HashMap<String, HelperFunctionExtraction>,
        all: &HashMap<String, HelperFunctionExtraction>,
    ) -> Vec<HelperFunctionExtraction> {
        let have: std::collections::HashSet<String> =
            source.helpers.iter().map(|h| h.name.clone()).collect();
        Self::reachable_helpers_from(source.module_scope.clone(), have, exported, all)
    }

    /// The same walk from an explicit seed list, with an explicit "already
    /// have" set.
    ///
    /// Split out for the constant→helper edge: a helper that only a module
    /// constant's initialiser calls is in nobody's `module_scope`, so the
    /// seeded walk above never reaches it.
    fn reachable_helpers_from(
        seeds: Vec<String>,
        mut have: std::collections::HashSet<String>,
        exported: &HashMap<String, HelperFunctionExtraction>,
        all: &HashMap<String, HelperFunctionExtraction>,
    ) -> Vec<HelperFunctionExtraction> {
        let mut queue: Vec<String> = seeds;
        let mut added: Vec<HelperFunctionExtraction> = Vec::new();
        let mut guard = 0;
        while let Some(name) = queue.pop() {
            guard += 1;
            if guard > 512 {
                break;
            }
            if have.contains(&name) {
                continue;
            }
            let Some(h) = exported.get(&name).or_else(|| all.get(&name)) else {
                continue;
            };
            have.insert(name.clone());
            for word in h
                .source
                .split(|ch: char| !(ch.is_alphanumeric() || ch == '_' || ch == '$'))
            {
                if !word.is_empty()
                    && word != name
                    && (exported.contains_key(word) || all.contains_key(word))
                {
                    queue.push(word.to_string());
                }
            }
            added.push(h.clone());
        }
        added
    }

    /// Update the migration state counts.
    fn update_state(&mut self) {
        let mut completed = 0;
        let mut in_progress = 0;
        let mut blocked = 0;

        for status in self.status.values() {
            match status {
                MigrationStatus::Completed => completed += 1,
                MigrationStatus::InProgress => in_progress += 1,
                MigrationStatus::Blocked => blocked += 1,
                MigrationStatus::Pending => {}
            }
        }

        self.spec.state = MigrationState {
            total_components: self.spec.components.len(),
            completed,
            in_progress,
            blocked,
            last_updated: chrono_now(),
        };
    }

    // =========================================================================
    // MCP Tools
    // =========================================================================

    /// List all migrations with their status.
    pub fn list_migrations(&self) -> Vec<MigrationSummary> {
        self.spec.components.iter().map(|comp| {
            let status = self.status.get(&comp.id)
                .copied()
                .unwrap_or(MigrationStatus::Pending);

            // Find components that block this one
            let blocked_by: Vec<String> = comp.dependencies.components.iter()
                .filter(|dep_id| {
                    self.status.get(*dep_id)
                        .map(|s| *s != MigrationStatus::Completed)
                        .unwrap_or(true)
                })
                .cloned()
                .collect();

            MigrationSummary {
                id: comp.id.clone(),
                name: comp.name.clone(),
                status,
                complexity: comp.complexity,
                blocked_by,
            }
        }).collect()
    }

    /// Get full spec for a specific component.
    pub fn get_migration(&self, component_id: &str) -> Result<&ComponentMigrationSpec, McpError> {
        self.spec.components.iter()
            .find(|c| c.id == component_id)
            .ok_or_else(|| McpError::NotFound(component_id.to_string()))
    }

    /// Validate generated Sigil code.
    ///
    /// Performs both heuristic checks (for migration-specific issues) and
    /// full Sigil parser validation (for syntax errors).
    pub fn validate_sigil(&self, code: &str) -> ValidationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();

        // Phase 1: Heuristic checks for migration-specific issues
        let lines: Vec<&str> = code.lines().collect();

        for (i, line) in lines.iter().enumerate() {
            let line_num = i + 1;

            // Check for placeholder expressions (migration artifact)
            if line.contains("/* expression */") {
                errors.push(ValidationError {
                    line: line_num,
                    column: line.find("/*").unwrap_or(0) + 1,
                    message: "Placeholder expression not replaced".to_string(),
                    suggestion: Some("Replace with actual expression".to_string()),
                });
            }

            // Check for TODO comments
            if line.contains("// TODO:") {
                warnings.push(ValidationWarning {
                    line: line_num,
                    column: line.find("// TODO:").unwrap_or(0) + 1,
                    message: "TODO comment present".to_string(),
                });
            }

            // Check for missing semicolons after state changes
            if line.trim().starts_with("self.") && !line.trim().ends_with(';') && !line.trim().ends_with('{') {
                warnings.push(ValidationWarning {
                    line: line_num,
                    column: line.len(),
                    message: "Statement may be missing semicolon".to_string(),
                });
            }
        }

        // Check for required imports
        if !code.contains("invoke qliphoth") {
            errors.push(ValidationError {
                line: 1,
                column: 1,
                message: "Missing qliphoth imports".to_string(),
                suggestion: Some("Add 'invoke qliphoth·prelude·*;' at the top".to_string()),
            });
        }

        // Check actor structure
        if code.contains("actor ") && !code.contains("rite view(") {
            warnings.push(ValidationWarning {
                line: 1,
                column: 1,
                message: "Actor missing view method".to_string(),
            });
        }

        // Phase 2: Full Sigil parser validation
        // Only run if no heuristic errors (parser errors may be confusing if code has placeholders)
        if errors.is_empty() {
            let mut parser = Parser::new(code);
            if let Err(parse_error) = parser.parse_file() {
                let (line, column) = byte_offset_to_line_col(code, &parse_error);
                let (message, suggestion) = format_parse_error(&parse_error);

                errors.push(ValidationError {
                    line,
                    column,
                    message,
                    suggestion,
                });
            }
        }

        ValidationResult {
            valid: errors.is_empty(),
            errors,
            warnings,
        }
    }

    /// Complete a migration by writing the generated code.
    pub fn complete_migration(&mut self, component_id: &str, sigil_code: &str) -> Result<CompletionResult, McpError> {
        // Validate the code first
        let validation = self.validate_sigil(sigil_code);
        if !validation.valid {
            return Err(McpError::ValidationFailed(validation.errors));
        }

        // Get the component spec
        let comp = self.spec.components.iter()
            .find(|c| c.id == component_id)
            .ok_or_else(|| McpError::NotFound(component_id.to_string()))?;

        // Determine output path
        let output_path = self.output_dir.join(&comp.target.suggested_path);

        // Create output directory if needed
        if let Some(parent) = output_path.parent() {
            fs::create_dir_all(parent)
                .map_err(|e| McpError::IoError(e.to_string()))?;
        }

        // Write the file
        fs::write(&output_path, sigil_code)
            .map_err(|e| McpError::IoError(e.to_string()))?;

        // Update status
        self.status.insert(component_id.to_string(), MigrationStatus::Completed);
        self.completed.insert(component_id.to_string(), sigil_code.to_string());

        // Find next suggested migrations (unblocked pending)
        let next_suggested = self.find_unblocked_pending();

        // Update state
        self.update_state();

        Ok(CompletionResult {
            success: true,
            output_path: output_path.display().to_string(),
            next_suggested,
        })
    }

    /// Get patterns, optionally filtered.
    pub fn get_patterns(&self, filter: Option<PatternFilter>) -> Vec<PatternExample> {
        let patterns = pattern_library();

        match filter {
            Some(f) => {
                patterns.into_iter()
                    .filter(|p| {
                        let name_match = f.name.as_ref()
                            .map(|n| p.name.contains(n))
                            .unwrap_or(true);
                        // Categories are embedded in pattern names for now
                        let cat_match = f.category.as_ref()
                            .map(|c| p.name.contains(c) || p.description.contains(c))
                            .unwrap_or(true);
                        name_match && cat_match
                    })
                    .collect()
            }
            None => patterns,
        }
    }

    /// Resolve an ambiguity for a component.
    pub fn resolve_ambiguity(
        &mut self,
        component_id: &str,
        ambiguity_id: &str,
        choice: usize,
    ) -> Result<(), McpError> {
        // Verify the component exists
        let comp = self.spec.components.iter()
            .find(|c| c.id == component_id)
            .ok_or_else(|| McpError::NotFound(component_id.to_string()))?;

        // Verify the ambiguity exists
        let ambiguity = comp.ambiguities.iter()
            .find(|a| a.id == ambiguity_id)
            .ok_or_else(|| McpError::NotFound(format!("ambiguity:{}", ambiguity_id)))?;

        // Verify the choice is valid
        if choice >= ambiguity.options.len() {
            return Err(McpError::InvalidChoice(choice, ambiguity.options.len()));
        }

        // Store the resolution
        self.resolved_ambiguities
            .entry(component_id.to_string())
            .or_default()
            .insert(ambiguity_id.to_string(), choice);

        Ok(())
    }

    /// Mark a migration as in progress.
    pub fn start_migration(&mut self, component_id: &str) -> Result<(), McpError> {
        if !self.status.contains_key(component_id) {
            return Err(McpError::NotFound(component_id.to_string()));
        }
        self.status.insert(component_id.to_string(), MigrationStatus::InProgress);
        self.update_state();
        Ok(())
    }

    // =========================================================================
    // MCP Resources
    // =========================================================================

    /// Get resource: migrations://pending
    pub fn resource_pending(&self) -> Vec<MigrationSummary> {
        self.list_migrations().into_iter()
            .filter(|m| m.status == MigrationStatus::Pending)
            .collect()
    }

    /// Get resource: migrations://patterns
    pub fn resource_patterns(&self) -> Vec<PatternExample> {
        self.get_patterns(None)
    }

    /// Get resource: migrations://component/{id}
    pub fn resource_component(&self, id: &str) -> Result<&ComponentMigrationSpec, McpError> {
        self.get_migration(id)
    }

    /// Get resource: migrations://overview
    pub fn resource_overview(&self) -> &MigrationState {
        &self.spec.state
    }

    // =========================================================================
    // Helpers
    // =========================================================================

    /// Find pending migrations that are not blocked.
    fn find_unblocked_pending(&self) -> Vec<String> {
        self.spec.components.iter()
            .filter(|comp| {
                let status = self.status.get(&comp.id).copied().unwrap_or(MigrationStatus::Pending);
                if status != MigrationStatus::Pending {
                    return false;
                }

                // Check if all dependencies are completed
                comp.dependencies.components.iter().all(|dep_id| {
                    self.status.get(dep_id)
                        .map(|s| *s == MigrationStatus::Completed)
                        .unwrap_or(false)
                })
            })
            .map(|c| c.id.clone())
            .collect()
    }

    /// Get the full spec.
    pub fn spec(&self) -> &MigrationSpec {
        &self.spec
    }

    /// Generate Sigil code for a component using the generator.
    pub fn generate_code(&self, component_id: &str) -> Result<GeneratedSigil, McpError> {
        let comp = self.get_migration(component_id)?;
        Ok(generate_component(comp))
    }

    /// Get the generated code for a completed migration.
    /// Returns None if the migration hasn't been completed.
    pub fn get_completed_code(&self, component_id: &str) -> Option<&String> {
        self.completed.get(component_id)
    }

    // =========================================================================
    // State Persistence
    // =========================================================================

    /// Save session state to a JSON file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), McpError> {
        let state = SessionState {
            spec: self.spec.clone(),
            status: self.status.clone(),
            completed: self.completed.clone(),
            resolved_ambiguities: self.resolved_ambiguities.clone(),
            output_dir: self.output_dir.clone(),
        };

        let json = serde_json::to_string_pretty(&state)
            .map_err(|e| McpError::SerializationError(e.to_string()))?;

        fs::write(path, json)
            .map_err(|e| McpError::IoError(e.to_string()))?;

        Ok(())
    }

    /// Load session state from a JSON file.
    pub fn load(path: impl AsRef<Path>, output_dir: impl AsRef<Path>) -> Result<Self, McpError> {
        let json = fs::read_to_string(path)
            .map_err(|e| McpError::IoError(e.to_string()))?;

        let state: SessionState = serde_json::from_str(&json)
            .map_err(|e| McpError::SerializationError(e.to_string()))?;

        let project_root = PathBuf::from(&state.spec.project_root);

        Ok(Self {
            project_root,
            output_dir: output_dir.as_ref().to_path_buf(),
            spec: state.spec,
            status: state.status,
            completed: state.completed,
            resolved_ambiguities: state.resolved_ambiguities,
            // A restored session has no files left to parse; nothing can be
            // added to this table, and the specs it would have fed already
            // carry their resolved constants.
            exported_constants: HashMap::new(),
            exported_helpers: HashMap::new(),
            all_helpers: HashMap::new(),
            all_constants: HashMap::new(),
        })
    }
}

// =============================================================================
// Errors
// =============================================================================

/// Errors from MCP operations.
#[derive(Debug, Clone)]
pub enum McpError {
    NotFound(String),
    ExtractionError(String),
    ValidationFailed(Vec<ValidationError>),
    IoError(String),
    InvalidChoice(usize, usize),
    SerializationError(String),
}

impl std::fmt::Display for McpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            McpError::NotFound(id) => write!(f, "Not found: {}", id),
            McpError::ExtractionError(e) => write!(f, "Extraction error: {}", e),
            McpError::ValidationFailed(errors) => {
                write!(f, "Validation failed with {} errors", errors.len())
            }
            McpError::IoError(e) => write!(f, "IO error: {}", e),
            McpError::InvalidChoice(choice, max) => {
                write!(f, "Invalid choice {} (max {})", choice, max - 1)
            }
            McpError::SerializationError(e) => write!(f, "Serialization error: {}", e),
        }
    }
}

impl std::error::Error for McpError {}

// =============================================================================
// Helper Functions for Parser Validation
// =============================================================================

/// Convert a ParseError to line and column numbers.
fn byte_offset_to_line_col(source: &str, error: &ParseError) -> (usize, usize) {
    // Extract byte offset from the error's span if available
    let byte_offset = match error {
        ParseError::UnexpectedToken { span, .. } => span.start,
        ParseError::DeprecatedRustSyntax { span, .. } => span.start,
        _ => 0,
    };

    // Convert byte offset to line/column
    let mut line = 1;
    let mut col = 1;
    let mut current_offset = 0;

    for ch in source.chars() {
        if current_offset >= byte_offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
        current_offset += ch.len_utf8();
    }

    (line, col)
}

/// Format a ParseError into a user-friendly message and optional suggestion.
fn format_parse_error(error: &ParseError) -> (String, Option<String>) {
    match error {
        ParseError::UnexpectedToken { expected, found, .. } => {
            let message = format!("Syntax error: expected {}, found {:?}", expected, found);
            (message, None)
        }
        ParseError::UnexpectedEof => {
            let message = "Unexpected end of file".to_string();
            let suggestion = Some("Check for unclosed braces or incomplete statements".to_string());
            (message, suggestion)
        }
        ParseError::InvalidNumber(s) => {
            let message = format!("Invalid number literal: {}", s);
            (message, None)
        }
        ParseError::Custom(msg) => {
            (msg.clone(), None)
        }
        ParseError::DeprecatedRustSyntax { rust, sigil, .. } => {
            let message = format!("Deprecated Rust syntax '{}' used", rust);
            let suggestion = Some(format!("Use Sigil's native syntax: {}", sigil));
            (message, suggestion)
        }
    }
}

/// Order constants so each comes after everything it references.
///
/// A plain Kahn's algorithm over "names this initialiser mentions". Anything
/// left over — a cycle, which JavaScript's module semantics permit and Sigil's
/// module scope does not — keeps its original position rather than being
/// dropped; the emission guard will refuse it and say so.
pub fn topo_sort_constants(items: Vec<ModuleConstantExtraction>) -> Vec<ModuleConstantExtraction> {
    use std::collections::HashSet;
    let names: HashSet<String> = items.iter().map(|c| c.name.clone()).collect();
    let deps: Vec<HashSet<String>> = items
        .iter()
        .map(|c| {
            c.init
                .split(|ch: char| !(ch.is_alphanumeric() || ch == '_' || ch == '$'))
                .filter(|w| !w.is_empty() && names.contains(*w) && *w != c.name)
                .map(|w| w.to_string())
                .collect()
        })
        .collect();

    let mut placed: HashSet<String> = HashSet::new();
    let mut out: Vec<ModuleConstantExtraction> = Vec::with_capacity(items.len());
    let mut remaining: Vec<usize> = (0..items.len()).collect();

    loop {
        let ready: Vec<usize> = remaining
            .iter()
            .copied()
            .filter(|&i| deps[i].iter().all(|d| placed.contains(d)))
            .collect();
        if ready.is_empty() {
            break;
        }
        for i in ready {
            placed.insert(items[i].name.clone());
            out.push(items[i].clone());
            remaining.retain(|&r| r != i);
        }
    }
    for i in remaining {
        out.push(items[i].clone());
    }
    out
}

#[cfg(test)]
mod constant_to_helper_tests {
    use super::*;
    use crate::migrate::react::extraction::{
        HelperFunctionExtraction, ModuleConstantExtraction, SourceLocation,
    };

    fn helper(name: &str, source: &str, exported: bool) -> HelperFunctionExtraction {
        HelperFunctionExtraction {
            name: name.to_string(),
            location: SourceLocation { start_line: 0, start_col: 0, end_line: 0, end_col: 0 },
            exported,
            is_async: false,
            is_generator: false,
            parameters: Vec::new(),
            return_type: None,
            is_pure: true,
            side_effects: Vec::new(),
            calls: Vec::new(),
            used_by: Vec::new(),
            source: source.to_string(),
            returns_expression: None,
        }
    }

    fn constant(name: &str, init: &str) -> ModuleConstantExtraction {
        ModuleConstantExtraction {
            name: name.to_string(),
            exported: true,
            type_annotation: None,
            init: init.to_string(),
            entries: Vec::new(),
        }
    }

    /// A helper reached only from a module constant's initialiser.
    ///
    /// `BUILTIN_THEMES` is exported and emitted; `daemoniorumTheme` is called
    /// only from inside its object literal, is in nobody's `module_scope`, and
    /// no helper body names it. Before the constant->helper edge it was never
    /// carried, and the call compiled to a constant 0.
    #[test]
    fn helper_named_only_by_a_constant_initialiser_is_reached() {
        let mut all = HashMap::new();
        all.insert(
            "daemoniorumTheme".to_string(),
            helper("daemoniorumTheme", "function daemoniorumTheme(mode) { return pickAccent(mode) }", false),
        );
        // …and transitively, what that helper itself calls.
        all.insert(
            "pickAccent".to_string(),
            helper("pickAccent", "function pickAccent(m) { return m }", false),
        );
        let exported: HashMap<String, HelperFunctionExtraction> = HashMap::new();

        let c = constant(
            "BUILTIN_THEMES",
            r#"{ "daemoniorum": { modes: { dark: daemoniorumTheme("dark") } } }"#,
        );

        // The seeding this fix added: scan the constant's initialiser for names
        // that are helpers, then expand transitively.
        let mut seeds = Vec::new();
        for word in c
            .init
            .split(|ch: char| !(ch.is_alphanumeric() || ch == '_' || ch == '$'))
        {
            if !word.is_empty() && all.contains_key(word) {
                seeds.push(word.to_string());
            }
        }
        assert_eq!(seeds, vec!["daemoniorumTheme".to_string()], "constant initialiser must seed the helper");

        let reached = MigrationSession::reachable_helpers_from(
            seeds,
            std::collections::HashSet::new(),
            &exported,
            &all,
        );
        let names: std::collections::HashSet<&str> =
            reached.iter().map(|h| h.name.as_str()).collect();

        assert!(names.contains("daemoniorumTheme"), "helper named by the constant");
        assert!(names.contains("pickAccent"), "and what it calls, transitively");
    }
}
