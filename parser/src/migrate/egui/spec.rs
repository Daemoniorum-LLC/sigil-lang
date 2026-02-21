//! Migration spec generation for egui → Qliphoth.
//!
//! Takes an `EguiExtraction` and produces an `EguiMigrationSpec` with field
//! names matching the React tool's `ComponentMigrationSpec` where they overlap.
//!
//! # Field name alignment with React tool
//!
//! | This file              | React `spec.rs`              |
//! |------------------------|------------------------------|
//! | `EguiMigrationSpec`    | `ComponentMigrationSpec`     |
//! | `id`                   | `id`                         |
//! | `name`                 | `name`                       |
//! | `source`               | `source` (ComponentSource)   |
//! | `target`               | `target` (TargetInfo)        |
//! | `recommendations`      | `recommendations`            |
//! | `ambiguities`          | `ambiguities`                |
//! | `complexity`           | `complexity`                 |
//! | `complexity_factors`   | `complexity_factors`         |
//! | `status`               | `status`                     |
//! | `automation_score`     | (egui-specific addition)     |
//! | `EguiMigrationManifest`| `MigrationSpec`              |

use serde::{Deserialize, Serialize};
use std::path::Path;

use super::extraction::{EguiExtraction, StructExtraction};
use super::patterns::AmbiguityKind;

// =============================================================================
// Shared enums (aligned with React tool)
// =============================================================================

/// Target pattern for the generated Sigil code.
/// Matches React tool `TargetPattern`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TargetPattern {
    /// Stateful struct with show/render method → actor.
    Actor,
    /// Pure helper struct or free function → rite.
    Function,
}

/// Complexity rating. Matches React tool `Complexity`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Complexity {
    Simple,
    Moderate,
    Complex,
}

/// Migration lifecycle status. Matches React tool `MigrationStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MigrationStatus {
    Pending,
    InProgress,
    Completed,
    Blocked,
}

// =============================================================================
// Spec sub-types
// =============================================================================

/// Source location info. Matches React tool `ComponentSource`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EguiSource {
    /// Absolute path to the source file.
    pub path: String,
    /// Full source text.
    pub code: String,
}

/// Target location info. Matches React tool `TargetInfo`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EguiTarget {
    /// Suggested output path relative to workspace, e.g. `"sigil/src/notifications.sigil"`.
    pub suggested_path: String,
    pub pattern: TargetPattern,
}

/// A recommended state field. Matches React tool `StateFieldRecommendation` shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EguiStateField {
    /// Field name (snake_case).
    pub name: String,
    /// Stringified Rust type, e.g. `"Vec<NotificationEntry>"`.
    pub field_type: String,
    /// Evidentiality marker: `"!"` = required, `"?"` = optional.
    pub evidentiality: String,
}

/// A recommended Msg variant. Matches React tool `MessageRecommendation` shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EguiMessage {
    /// Variant name (PascalCase), e.g. `"Dismiss"`.
    pub name: String,
    /// Optional payload type string, e.g. `"usize"` or `null`.
    pub payload: Option<String>,
    /// Brief description of what this message does.
    pub description: String,
}

/// Recommendations for the target actor. Matches React tool `Recommendations`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EguiRecommendations {
    pub state_fields: Vec<EguiStateField>,
    pub messages: Vec<EguiMessage>,
}

/// A single ambiguity in the source that requires manual review.
/// Matches the JSON format from the plan file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EguiAmbiguity {
    /// Marker string, e.g. `"CANVAS_TO_SVG"`.
    pub kind: String,
    /// Approximate 1-based line number (0 = unknown).
    pub line: u32,
    /// Short source snippet for context.
    pub snippet: String,
}

// =============================================================================
// Top-level spec (one per struct/actor)
// =============================================================================

/// Migration spec for a single egui struct/actor.
/// Field names mirror React tool `ComponentMigrationSpec`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EguiMigrationSpec {
    /// Unique ID: `"<relative_path>:<StructName>"`.
    pub id: String,
    /// Struct name, e.g. `"Notifications"`.
    pub name: String,
    pub source: EguiSource,
    pub target: EguiTarget,
    pub recommendations: EguiRecommendations,
    pub ambiguities: Vec<EguiAmbiguity>,
    /// Fraction of the migration that can be automated (0.0–1.0).
    /// Computed as `1.0 - (ambiguity_count / total_pattern_count).clamp(0,1)`.
    pub automation_score: f32,
    pub complexity: Complexity,
    pub complexity_factors: Vec<String>,
    pub status: MigrationStatus,
}

// =============================================================================
// Manifest (one per run / directory)
// =============================================================================

/// Top-level migration manifest for a directory scan.
/// Matches React tool `MigrationSpec`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EguiMigrationManifest {
    pub version: String,
    pub generated_at: String,
    pub source_root: String,
    pub components: Vec<EguiMigrationSpec>,
    pub state: ManifestState,
}

/// Progress counters. Matches React tool `MigrationState`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestState {
    pub total_components: usize,
    pub completed: usize,
    pub in_progress: usize,
    pub blocked: usize,
    pub last_updated: String,
}

// =============================================================================
// Spec builder
// =============================================================================

/// Build an `EguiMigrationSpec` from an `EguiExtraction` and a specific struct.
pub fn build_spec(
    extraction: &EguiExtraction,
    struct_def: &StructExtraction,
    source_root: &Path,
) -> EguiMigrationSpec {
    let relative_path = &extraction.file.relative_path;
    let name = &struct_def.name;
    let id = format!("{}:{}", relative_path, name);

    // --- Source ---
    let source = EguiSource {
        path: extraction.file.path.to_string_lossy().into_owned(),
        code: extraction.file.source.clone(),
    };

    // --- Target ---
    let stem = extraction.file.path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");

    // Determine if there's a show/render/view method for this struct
    let has_view = extraction.impls.iter()
        .filter(|imp| &imp.type_name == name)
        .any(|imp| imp.methods.iter().any(|m| m.is_view));

    let pattern = if has_view || !struct_def.fields.is_empty() {
        TargetPattern::Actor
    } else {
        TargetPattern::Function
    };

    let target = EguiTarget {
        suggested_path: format!("sigil/src/{}.sigil", stem),
        pattern,
    };

    // --- State fields (from struct fields) ---
    let state_fields: Vec<EguiStateField> = struct_def.fields.iter().map(|f| {
        let evidentiality = if f.field_type.starts_with("Option<") {
            "?".to_string()
        } else {
            "!".to_string()
        };
        EguiStateField {
            name: f.name.clone(),
            field_type: f.field_type.clone(),
            evidentiality,
        }
    }).collect();

    // --- Collect ambiguities from all view methods ---
    let mut raw_ambiguities: Vec<(AmbiguityKind, u32, String)> = Vec::new();
    let mut total_patterns: usize = 0;

    for imp in extraction.impls.iter().filter(|i| &i.type_name == name) {
        for method in imp.methods.iter().filter(|m| m.is_view) {
            total_patterns += method.body_patterns.len();
            for amb in &method.ambiguities {
                raw_ambiguities.push((amb.kind, amb.line, amb.snippet.clone()));
            }
        }
    }

    let ambiguities: Vec<EguiAmbiguity> = raw_ambiguities.iter().map(|(kind, line, snippet)| {
        // Use serde_json to get SCREAMING_SNAKE_CASE (e.g. "CANVAS_TO_SVG" not "CANVASTOSVG")
        let kind_str = serde_json::to_string(kind)
            .unwrap_or_default()
            .trim_matches('"')
            .to_string();
        EguiAmbiguity {
            kind: kind_str,
            line: *line,
            snippet: snippet.clone(),
        }
    }).collect();

    // --- Messages (infer from click/change patterns) ---
    let messages = infer_messages(extraction, name);

    // --- Automation score ---
    let amb_count = ambiguities.len() as f32;
    let total = (total_patterns + ambiguities.len()) as f32;
    let automation_score = if total == 0.0 {
        1.0
    } else {
        (1.0 - (amb_count / total)).clamp(0.0, 1.0)
    };

    // --- Complexity ---
    let (complexity, complexity_factors) = rate_complexity(
        struct_def.fields.len(),
        ambiguities.len(),
        total_patterns,
    );

    EguiMigrationSpec {
        id,
        name: name.clone(),
        source,
        target,
        recommendations: EguiRecommendations { state_fields, messages },
        ambiguities,
        automation_score,
        complexity,
        complexity_factors,
        status: MigrationStatus::Pending,
    }
}

/// Infer Msg variants from detected button/checkbox/text-edit patterns.
fn infer_messages(extraction: &EguiExtraction, type_name: &str) -> Vec<EguiMessage> {
    let mut messages = Vec::new();

    for imp in extraction.impls.iter().filter(|i| i.type_name == type_name) {
        for method in &imp.methods {
            for pattern in &method.body_patterns {
                match pattern.kind.as_str() {
                    "button" | "small_button" => {
                        let label = pattern.args.first()
                            .map(|s| pascal_case_from_label(s))
                            .unwrap_or_else(|| "Clicked".to_string());
                        if !messages.iter().any(|m: &EguiMessage| m.name == label) {
                            messages.push(EguiMessage {
                                name: label.clone(),
                                payload: None,
                                description: format!("Button '{}' clicked", label),
                            });
                        }
                    }
                    "text_input" | "textarea" => {
                        let field = pattern.args.first()
                            .map(|s| strip_self_prefix(s))
                            .unwrap_or_else(|| "value".to_string());
                        let msg_name = format!("Set{}", to_pascal_case(&field));
                        if !messages.iter().any(|m: &EguiMessage| m.name == msg_name) {
                            messages.push(EguiMessage {
                                name: msg_name.clone(),
                                payload: Some("String".to_string()),
                                description: format!("Update {}", field),
                            });
                        }
                    }
                    "checkbox" => {
                        let field = pattern.args.first()
                            .map(|s| strip_self_prefix(s))
                            .unwrap_or_else(|| "checked".to_string());
                        let msg_name = format!("Toggle{}", to_pascal_case(&field));
                        if !messages.iter().any(|m: &EguiMessage| m.name == msg_name) {
                            messages.push(EguiMessage {
                                name: msg_name.clone(),
                                payload: None,
                                description: format!("Toggle {}", field),
                            });
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    messages
}

fn rate_complexity(
    field_count: usize,
    ambiguity_count: usize,
    pattern_count: usize,
) -> (Complexity, Vec<String>) {
    let mut factors = Vec::new();

    if field_count > 10 {
        factors.push(format!("{} state fields", field_count));
    }
    if ambiguity_count > 3 {
        factors.push(format!("{} unresolved ambiguities", ambiguity_count));
    }
    if pattern_count > 20 {
        factors.push(format!("{} UI patterns", pattern_count));
    }

    let complexity = if ambiguity_count > 5 || field_count > 15 || pattern_count > 30 {
        Complexity::Complex
    } else if ambiguity_count > 1 || field_count > 5 || pattern_count > 8 {
        Complexity::Moderate
    } else {
        Complexity::Simple
    };

    (complexity, factors)
}

// =============================================================================
// Manifest builder
// =============================================================================

pub fn build_manifest(
    components: Vec<EguiMigrationSpec>,
    source_root: &Path,
) -> EguiMigrationManifest {
    let total = components.len();
    let completed = components.iter().filter(|c| matches!(c.status, MigrationStatus::Completed)).count();
    let in_progress = components.iter().filter(|c| matches!(c.status, MigrationStatus::InProgress)).count();
    let blocked = components.iter().filter(|c| matches!(c.status, MigrationStatus::Blocked)).count();

    EguiMigrationManifest {
        version: "1.0.0".to_string(),
        generated_at: chrono_now(),
        source_root: source_root.to_string_lossy().into_owned(),
        components,
        state: ManifestState {
            total_components: total,
            completed,
            in_progress,
            blocked,
            last_updated: chrono_now(),
        },
    }
}

fn chrono_now() -> String {
    // Avoids chrono dependency — uses a placeholder that the CLI can override.
    "2026-01-01T00:00:00Z".to_string()
}

// =============================================================================
// String helpers
// =============================================================================

fn pascal_case_from_label(label: &str) -> String {
    // "\"Save\"" → "Save"; "\"hello world\"" → "HelloWorld"
    let trimmed = label.trim_matches('"');
    to_pascal_case(trimmed)
}

fn to_pascal_case(s: &str) -> String {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(|w| {
            let mut chars = w.chars();
            match chars.next() {
                None => String::new(),
                Some(c) => c.to_uppercase().to_string() + &chars.as_str().to_lowercase(),
            }
        })
        .collect()
}

fn strip_self_prefix(s: &str) -> String {
    s.trim_start_matches("&mut ")
        .trim_start_matches("&")
        .trim_start_matches("self.")
        .to_string()
}
