//! Pattern library: egui API calls → Qliphoth VNode mappings.
//!
//! Each entry in the pattern library describes a single egui method call
//! that can be automatically translated to Sigil/Qliphoth.  Calls that
//! cannot be automatically mapped return an `AmbiguityKind` instead.
//!
//! # Pattern table (from plan)
//!
//! | egui                          | Sigil/Qliphoth                                         |
//! |-------------------------------|--------------------------------------------------------|
//! | `ui.label(text)`              | `p()·text(text)·to_vnode()`                            |
//! | `ui.heading(text)`            | `h3()·text(text)·to_vnode()`                           |
//! | `ui.separator()`              | `hr()·to_vnode()`                                      |
//! | `ui.button(text)`             | `button()·text(text)·to_vnode()`                       |
//! | `ui.checkbox(&mut v, lbl)`    | `input()·attr("type","checkbox")·attr("checked",self.v)` |
//! | `ui.text_edit_singleline(&mut v)` | `input()·attr("type","text")·attr("value",self.v)` |
//! | `ui.horizontal(\|ui\| {...})` | `div()·style("display:flex;flex-direction:row")`       |
//! | `ui.vertical(\|ui\| {...})`   | `div()·style("display:flex;flex-direction:column")`    |
//! | `ctx.request_repaint()`       | DELETE (no retained-mode repaint in Qliphoth)          |
//!
//! # Ambiguity markers
//!
//! | egui                   | `AmbiguityKind`    |
//! |------------------------|--------------------|
//! | `ui.painter().*`       | `CanvasToSvg`      |
//! | `egui::plot::*`        | `ChartComponent`   |
//! | `ui.with_layout(…)`    | `LayoutSystem`     |
//! | `ui.allocate_ui(…)`    | `CustomSizing`     |
//! | `ScrollArea::*`        | `ScrollArea`       |
//! | `.on_hover_text(…)`    | `Tooltip`          |

use serde::{Deserialize, Serialize};

// =============================================================================
// AmbiguityKind
// =============================================================================

/// Categories of egui calls that cannot be automatically mapped.
/// These generate `??MARKER??` annotations in the Sigil output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AmbiguityKind {
    /// `ui.painter().*` — custom canvas drawing; no direct VNode equivalent.
    CanvasToSvg,
    /// `egui::plot::*` — chart/plot widgets.
    ChartComponent,
    /// `ui.with_layout(…)` — complex layout directives.
    LayoutSystem,
    /// `ui.allocate_ui(…)` / `ui.allocate_exact_size(…)` — manual size allocation.
    CustomSizing,
    /// `ScrollArea::*` — scroll containers.
    ScrollAreaWidget,
    /// `.on_hover_text(…)` — tooltip; no standard VNode attr yet.
    Tooltip,
    /// `egui::Window::*` / `egui::Area::*` — floating overlay positioning.
    FloatingOverlay,
    /// `ui.add(widget)` — generic widget add; type not statically known.
    GenericWidget,
    /// `egui::RichText::*` — styled text with per-span formatting.
    RichText,
    /// `ui.image(…)` — image rendering.
    ImageWidget,
}

impl AmbiguityKind {
    /// The `??MARKER??` string to embed in generated Sigil output.
    pub fn marker(self) -> &'static str {
        match self {
            Self::CanvasToSvg       => "??CANVAS_TO_SVG??",
            Self::ChartComponent    => "??CHART_COMPONENT??",
            Self::LayoutSystem      => "??LAYOUT_SYSTEM??",
            Self::CustomSizing      => "??CUSTOM_SIZING??",
            Self::ScrollAreaWidget  => "??SCROLL_AREA??",
            Self::Tooltip           => "??TOOLTIP??",
            Self::FloatingOverlay   => "??FLOATING_OVERLAY??",
            Self::GenericWidget     => "??GENERIC_WIDGET??",
            Self::RichText          => "??RICH_TEXT??",
            Self::ImageWidget       => "??IMAGE_WIDGET??",
        }
    }
}

// =============================================================================
// Pattern classification
// =============================================================================

/// Classify an egui method call by name and argument snippets.
///
/// Returns `Ok(kind_string)` for automatable mappings, or
/// `Err(AmbiguityKind)` for patterns that need manual review.
///
/// The `kind_string` is stored in `DetectedPattern::kind` and consumed
/// by the generator to emit the correct VNode builder chain.
pub fn classify_method_call(
    method: &str,
    args: &[String],
) -> Result<String, AmbiguityKind> {
    match method {
        // --- Direct text elements ---
        "label"             => Ok("label".to_string()),
        "heading"           => Ok("heading".to_string()),
        "small"             => Ok("small".to_string()),
        "monospace"         => Ok("monospace".to_string()),
        "code"              => Ok("code".to_string()),
        "strong"            => Ok("strong".to_string()),
        "weak"              => Ok("weak".to_string()),

        // --- Form elements ---
        "button"            => Ok("button".to_string()),
        "small_button"      => Ok("small_button".to_string()),
        "checkbox"          => Ok("checkbox".to_string()),
        "radio"             => Ok("radio".to_string()),
        "text_edit_singleline" => Ok("text_input".to_string()),
        "text_edit_multiline"  => Ok("textarea".to_string()),

        // --- Layout ---
        "separator"         => Ok("separator".to_string()),
        "horizontal"        => Ok("horizontal".to_string()),
        "vertical"          => Ok("vertical".to_string()),
        "add_space" | "spacing" => Ok("spacer".to_string()),

        // --- Control flow helpers ---
        "collapsing"        => Ok("collapsible".to_string()),
        "group"             => Ok("group".to_string()),

        // --- Deleted patterns (no-ops in Qliphoth) ---
        "request_repaint"   => Ok("noop_repaint".to_string()),
        "ctx"               => Ok("noop_ctx".to_string()),

        // --- Ambiguities ---
        "painter"           => Err(AmbiguityKind::CanvasToSvg),
        "with_layout"       => Err(AmbiguityKind::LayoutSystem),
        "allocate_ui" | "allocate_exact_size" | "allocate_response"
                            => Err(AmbiguityKind::CustomSizing),
        "on_hover_text" | "on_hover_ui"
                            => Err(AmbiguityKind::Tooltip),
        "image"             => Err(AmbiguityKind::ImageWidget),
        "add" if args.iter().any(|a| a.contains("widget") || a.contains("Widget"))
                            => Err(AmbiguityKind::GenericWidget),

        _ if method.starts_with("plot") || method.contains("Plot")
                            => Err(AmbiguityKind::ChartComponent),
        _ if method.contains("scroll") || method.contains("Scroll")
                            => Err(AmbiguityKind::ScrollAreaWidget),
        _ if method.contains("window") || method.contains("Window")
            || method.contains("area") || method.contains("Area")
                            => Err(AmbiguityKind::FloatingOverlay),
        _ if method.contains("rich") || method.contains("Rich")
                            => Err(AmbiguityKind::RichText),

        // Unknown method — treat as generic widget ambiguity
        _                   => Err(AmbiguityKind::GenericWidget),
    }
}

// =============================================================================
// VNode code generation helpers (used by generator.rs)
// =============================================================================

/// Generate the Sigil VNode builder expression for a classified pattern.
///
/// Returns `None` for no-op patterns (e.g. `request_repaint`).
pub fn vnode_for_pattern(kind: &str, args: &[String]) -> Option<String> {
    let first = args.first().map(|s| s.as_str()).unwrap_or("\"\"");
    let second = args.get(1).map(|s| s.as_str()).unwrap_or("\"\"");

    match kind {
        "label"       => Some(format!("p()·text({})·to_vnode()", first)),
        "heading"     => Some(format!("h3()·text({})·to_vnode()", first)),
        "small"       => Some(format!("small()·text({})·to_vnode()", first)),
        "monospace"   => Some(format!("code()·text({})·to_vnode()", first)),
        "code"        => Some(format!("code()·text({})·to_vnode()", first)),
        "strong"      => Some(format!("strong()·text({})·to_vnode()", first)),
        "weak"        => Some(format!(
            "span()·style(\"color:#969696;\")·text({})·to_vnode()", first
        )),

        "button" | "small_button" => Some(format!(
            "button()·on_click(|_| Msg::TODO)·text({})·to_vnode()", first
        )),
        "checkbox"    => Some(format!(
            "input()·attr(\"type\",\"checkbox\")·attr(\"checked\",self.{})·to_vnode()",
            strip_ampersand_mut(first)
        )),
        "radio"       => Some(format!(
            "input()·attr(\"type\",\"radio\")·attr(\"value\",self.{})·to_vnode()",
            strip_ampersand_mut(first)
        )),
        "text_input"  => Some(format!(
            "input()·attr(\"type\",\"text\")·attr(\"value\",self.{})·on_change(|v| Msg::TODO(v))·to_vnode()",
            strip_ampersand_mut(first)
        )),
        "textarea"    => Some(format!(
            "textarea()·attr(\"value\",self.{})·on_change(|v| Msg::TODO(v))·to_vnode()",
            strip_ampersand_mut(first)
        )),

        "separator"   => Some("hr()·to_vnode()".to_string()),
        "horizontal"  => Some(
            "div()·style(\"display:flex;flex-direction:row;\")·to_vnode() // ??INLINE_CHILDREN??".to_string()
        ),
        "vertical"    => Some(
            "div()·style(\"display:flex;flex-direction:column;\")·to_vnode() // ??INLINE_CHILDREN??".to_string()
        ),
        "spacer"      => Some("div()·style(\"flex-shrink:0;\")·to_vnode()".to_string()),
        "collapsible" => Some(format!(
            "details()·child(summary()·text({})·to_vnode())·to_vnode() // ??COLLAPSING??", first
        )),
        "group"       => Some(
            "div()·style(\"border:1px solid rgba(108,108,118,0.2);border-radius:4px;padding:8px;\")·to_vnode()".to_string()
        ),

        // No-ops
        "noop_repaint" | "noop_ctx" => None,

        _ => Some(format!(
            "// ??GENERIC_WIDGET?? — egui .{}({}) has no direct mapping",
            kind, first
        )),
    }
}

/// Strip `&mut ` / `&` prefix from an argument string (e.g. `"&mut self.count"` → `"self.count"`).
fn strip_ampersand_mut(s: &str) -> &str {
    let s = s.trim();
    let s = s.strip_prefix("&mut ").unwrap_or(s);
    let s = s.strip_prefix('&').unwrap_or(s);
    s
}

// =============================================================================
// Color mapping
// =============================================================================

/// Convert `egui::Color32::from_rgb(r, g, b)` args to a CSS string.
pub fn color32_to_css(r: u8, g: u8, b: u8) -> String {
    format!("rgb({},{},{})", r, g, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_label_mapping() {
        assert_eq!(classify_method_call("label", &["\"hello\"".into()]), Ok("label".into()));
    }

    #[test]
    fn test_painter_is_ambiguous() {
        assert_eq!(classify_method_call("painter", &[]), Err(AmbiguityKind::CanvasToSvg));
    }

    #[test]
    fn test_vnode_label() {
        let v = vnode_for_pattern("label", &["\"hello\"".into()]);
        assert_eq!(v, Some("p()·text(\"hello\")·to_vnode()".into()));
    }

    #[test]
    fn test_noop_repaint() {
        let v = vnode_for_pattern("noop_repaint", &[]);
        assert_eq!(v, None);
    }
}
