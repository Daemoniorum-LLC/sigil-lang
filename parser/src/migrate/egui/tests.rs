//! Integration tests for the egui → Qliphoth migration tool.
//!
//! Uses a synthetic egui source fixture that covers:
//! - Struct with named fields (public + private)
//! - impl block with a `show()` view method
//! - Button click pattern (automatable)
//! - Checkbox pattern (automatable)
//! - `ui.painter()` (ambiguous → CANVAS_TO_SVG)
//! - `ctx.request_repaint()` (no-op)
//!
//! Verification goals (matching plan file Section 6):
//! 1. JSON spec keys match React tool's ComponentMigrationSpec
//! 2. automation_score is computed correctly
//! 3. Generated `.sigil` text contains `actor`, `invoke qliphoth·prelude·*;`, Msg enum
//! 4. Ambiguity markers appear in generated output

use super::extraction::extract_source;
use super::generator::generate_sigil;
use super::spec::build_spec;
use super::patterns::{classify_method_call, AmbiguityKind};

use std::path::Path;

// =============================================================================
// Fixture source
// =============================================================================

/// Synthetic egui component that exercises all pattern categories.
const FIXTURE_SRC: &str = r#"
use egui;

/// Toast notification panel.
pub struct Notifications {
    pub history: Vec<String>,
    pub visible: bool,
    max_visible: usize,
}

impl Notifications {
    pub fn new() -> Self {
        Self {
            history: vec![],
            visible: true,
            max_visible: 5,
        }
    }

    pub fn show(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.heading("Notifications");
        ui.separator();

        if ui.button("Clear All").clicked() {
            self.history.clear();
        }

        ui.checkbox(&mut self.visible, "Show panel");

        // Ambiguous — custom canvas drawing
        let painter = ui.painter();

        // No-op in Qliphoth
        ctx.request_repaint();
    }
}
"#;

const VIRTUAL_PATH: &str = "test/notifications.rs";

// =============================================================================
// Helpers
// =============================================================================

fn fixture_extraction() -> super::extraction::EguiExtraction {
    extract_source(FIXTURE_SRC, Path::new(VIRTUAL_PATH))
        .expect("extract fixture")
}

fn fixture_spec() -> super::spec::EguiMigrationSpec {
    let ex = fixture_extraction();
    let s = ex.structs.iter()
        .find(|s| s.name == "Notifications")
        .expect("Notifications struct not found in extraction");
    build_spec(&ex, s, Path::new("test"))
}

// =============================================================================
// extraction tests
// =============================================================================

#[test]
fn extraction_finds_notifications_struct() {
    let ex = fixture_extraction();
    let names: Vec<&str> = ex.structs.iter().map(|s| s.name.as_str()).collect();
    assert!(names.contains(&"Notifications"), "structs: {:?}", names);
}

#[test]
fn extraction_struct_has_fields() {
    let ex = fixture_extraction();
    let notif = ex.structs.iter().find(|s| s.name == "Notifications").unwrap();
    let field_names: Vec<&str> = notif.fields.iter().map(|f| f.name.as_str()).collect();
    assert!(field_names.contains(&"history"), "fields: {:?}", field_names);
    assert!(field_names.contains(&"visible"), "fields: {:?}", field_names);
    assert!(field_names.contains(&"max_visible"), "fields: {:?}", field_names);
}

#[test]
fn extraction_finds_show_method() {
    let ex = fixture_extraction();
    let imp = ex.impls.iter().find(|i| i.type_name == "Notifications").unwrap();
    let method_names: Vec<&str> = imp.methods.iter().map(|m| m.name.as_str()).collect();
    assert!(method_names.contains(&"show"), "methods: {:?}", method_names);
}

#[test]
fn extraction_detects_button_pattern() {
    let ex = fixture_extraction();
    let imp = ex.impls.iter().find(|i| i.type_name == "Notifications").unwrap();
    let show = imp.methods.iter().find(|m| m.name == "show").unwrap();
    let kinds: Vec<&str> = show.body_patterns.iter().map(|p| p.kind.as_str()).collect();
    assert!(kinds.contains(&"button"), "patterns: {:?}", kinds);
}

#[test]
fn extraction_detects_canvas_ambiguity() {
    let ex = fixture_extraction();
    let imp = ex.impls.iter().find(|i| i.type_name == "Notifications").unwrap();
    let show = imp.methods.iter().find(|m| m.name == "show").unwrap();
    let amb_kinds: Vec<_> = show.ambiguities.iter().map(|a| a.kind).collect();
    assert!(
        amb_kinds.contains(&AmbiguityKind::CanvasToSvg),
        "ambiguities: {:?}", amb_kinds
    );
}

#[test]
fn extraction_show_is_view() {
    let ex = fixture_extraction();
    let imp = ex.impls.iter().find(|i| i.type_name == "Notifications").unwrap();
    let show = imp.methods.iter().find(|m| m.name == "show").unwrap();
    assert!(show.is_view, "show() should be flagged as is_view");
}

// =============================================================================
// spec tests
// =============================================================================

#[test]
fn spec_id_contains_struct_name() {
    let spec = fixture_spec();
    assert!(spec.id.contains("Notifications"), "id: {}", spec.id);
}

#[test]
fn spec_name_is_notifications() {
    let spec = fixture_spec();
    assert_eq!(spec.name, "Notifications");
}

#[test]
fn spec_has_state_fields() {
    let spec = fixture_spec();
    assert!(!spec.recommendations.state_fields.is_empty(), "expected state fields");
    let field_names: Vec<&str> = spec.recommendations.state_fields.iter()
        .map(|f| f.name.as_str()).collect();
    assert!(field_names.contains(&"history"), "fields: {:?}", field_names);
}

#[test]
fn spec_has_canvas_ambiguity() {
    let spec = fixture_spec();
    assert!(!spec.ambiguities.is_empty(), "expected at least one ambiguity");
    let kinds: Vec<&str> = spec.ambiguities.iter().map(|a| a.kind.as_str()).collect();
    assert!(kinds.contains(&"CANVAS_TO_SVG"), "ambiguity kinds: {:?}", kinds);
}

#[test]
fn spec_automation_score_below_one() {
    let spec = fixture_spec();
    assert!(spec.automation_score < 1.0, "score: {}", spec.automation_score);
    assert!(spec.automation_score >= 0.0, "score: {}", spec.automation_score);
}

#[test]
fn spec_status_is_pending() {
    use super::spec::MigrationStatus;
    let spec = fixture_spec();
    assert!(matches!(spec.status, MigrationStatus::Pending));
}

#[test]
fn spec_infers_messages_from_button() {
    let spec = fixture_spec();
    // "Clear All" button → at least one message inferred
    assert!(!spec.recommendations.messages.is_empty(), "expected inferred messages");
}

/// Verify JSON serialization produces keys matching the React tool's
/// ComponentMigrationSpec format.
#[test]
fn spec_json_keys_match_react_tool_format() {
    let spec = fixture_spec();
    let json = serde_json::to_string(&spec).expect("serialize");
    let val: serde_json::Value = serde_json::from_str(&json).expect("parse");
    let obj = val.as_object().expect("object");

    for key in &["id", "name", "source", "target", "recommendations", "ambiguities",
                 "automation_score", "complexity", "complexity_factors", "status"] {
        assert!(obj.contains_key(*key), "missing key: {}", key);
    }
}

// =============================================================================
// generator tests
// =============================================================================

#[test]
fn generator_produces_invoke_header() {
    let spec = fixture_spec();
    let gen = generate_sigil(&spec);
    assert!(
        gen.code.contains("invoke qliphoth\u{00B7}prelude\u{00B7}*;"),
        "missing invoke header in:\n{}", gen.code
    );
}

#[test]
fn generator_produces_actor_block() {
    let spec = fixture_spec();
    let gen = generate_sigil(&spec);
    assert!(gen.code.contains("actor Notifications {"), "missing actor block in:\n{}", gen.code);
}

#[test]
fn generator_produces_msg_enum() {
    let spec = fixture_spec();
    let gen = generate_sigil(&spec);
    assert!(gen.code.contains("enum Msg {"), "missing enum Msg");
}

#[test]
fn generator_produces_view_rite() {
    let spec = fixture_spec();
    let gen = generate_sigil(&spec);
    assert!(gen.code.contains("rite view() -> VNode!"), "missing view rite");
}

#[test]
fn generator_embeds_ambiguity_markers() {
    let spec = fixture_spec();
    let gen = generate_sigil(&spec);
    assert!(
        gen.code.contains("CANVAS_TO_SVG"),
        "expected CANVAS_TO_SVG marker in output:\n{}", gen.code
    );
}

#[test]
fn generator_actor_name_is_set() {
    let spec = fixture_spec();
    let gen = generate_sigil(&spec);
    assert_eq!(gen.actor_name, "Notifications");
}

// =============================================================================
// pattern library unit tests
// =============================================================================

#[test]
fn patterns_label_maps_to_ok() {
    assert!(classify_method_call("label", &["\"hello\"".into()]).is_ok());
}

#[test]
fn patterns_painter_is_canvas_to_svg() {
    assert_eq!(
        classify_method_call("painter", &[]),
        Err(AmbiguityKind::CanvasToSvg)
    );
}

#[test]
fn patterns_button_maps_to_ok() {
    assert_eq!(
        classify_method_call("button", &["\"Save\"".into()]),
        Ok("button".to_string())
    );
}

#[test]
fn patterns_request_repaint_is_noop() {
    assert_eq!(
        classify_method_call("request_repaint", &[]),
        Ok("noop_repaint".to_string())
    );
}

#[test]
fn patterns_scroll_is_scroll_area() {
    assert_eq!(
        classify_method_call("scroll_area", &[]),
        Err(AmbiguityKind::ScrollAreaWidget)
    );
}

#[test]
fn patterns_unknown_is_generic_widget() {
    assert_eq!(
        classify_method_call("frobnicate", &[]),
        Err(AmbiguityKind::GenericWidget)
    );
}
