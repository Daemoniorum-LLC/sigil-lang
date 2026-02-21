//! Pattern library: egui API calls → Qliphoth VNode mappings.
//!
//! Each entry in the pattern library describes a single egui method call
//! that can be automatically translated to Sigil/Qliphoth.  Calls that
//! cannot be automatically mapped return an `AmbiguityKind` instead.
//!
//! # Pattern table
//!
//! ## Text / display
//! | egui                          | Sigil/Qliphoth                                       |
//! |-------------------------------|------------------------------------------------------|
//! | `ui.label(text)`              | `p()·text(text)·to_vnode()`                          |
//! | `ui.heading(text)`            | `h3()·text(text)·to_vnode()`                         |
//! | `ui.colored_label(color,text)`| `span()·style("color:…")·text(text)·to_vnode()`     |
//! | `ui.strong(text)`             | `strong()·text(text)·to_vnode()`                     |
//! | `ui.weak(text)`               | `span()·style("color:#969696")·text(text)·to_vnode()`|
//! | `ui.separator()`              | `hr()·to_vnode()`                                    |
//!
//! ## Form / interactive
//! | egui                                | Sigil/Qliphoth                                    |
//! |-------------------------------------|---------------------------------------------------|
//! | `ui.button(text)`                   | `button()·on_click(…)·text(text)·to_vnode()`      |
//! | `ui.checkbox(&mut v, lbl)`          | `input(type=checkbox)·attr("checked",self.v)`     |
//! | `ui.radio(val, text)`               | `input(type=radio)·attr("value",val)`             |
//! | `ui.selectable_label(selected,text)`| `li()·on_click(…)·text(text)·to_vnode()`          |
//! | `ui.selectable_value(&mut v,k,text)`| `option()·attr("selected",v==k)·text(text)`       |
//! | `ui.text_edit_singleline(&mut v)`   | `input(type=text)·attr("value",self.v)`           |
//! | `ui.text_edit_multiline(&mut v)`    | `textarea()·attr("value",self.v)`                 |
//! | `ui.spinner()`                      | `span()·class("spinner")·to_vnode()`              |
//!
//! ## Layout
//! | egui                            | Sigil/Qliphoth                                        |
//! |---------------------------------|-------------------------------------------------------|
//! | `ui.horizontal(\|ui\| …)`       | `div(display:flex;flex-direction:row)`                |
//! | `ui.vertical(\|ui\| …)`         | `div(display:flex;flex-direction:column)`             |
//! | `ui.horizontal_wrapped(\|ui\|…)`| `div(display:flex;flex-wrap:wrap)`                    |
//! | `ui.vertical_centered(\|ui\|…)` | `div(display:flex;align-items:center;…column)`        |
//! | `ui.centered_and_justified(…)`  | `div(display:flex;justify-content:center;…)`          |
//! | `ui.indent(id, \|ui\| …)`       | `div(padding-left:1em)`                               |
//! | `ui.group(\|ui\| …)`            | `div` with border styling                             |
//! | `ui.collapsing(text, \|ui\|…)`  | `details()·child(summary()·text(text))`               |
//! | `ui.menu_button(text,\|ui\|…)`  | `button(class=menu-button)` + ??INLINE_CHILDREN??     |
//! | `ui.add_space(n)` / `spacing()` | `div(flex-shrink:0)`                                  |
//!
//! ## No-ops (deleted in Qliphoth)
//! | egui                         | Qliphoth         |
//! |------------------------------|------------------|
//! | `ctx.request_repaint()`      | DELETE           |
//! | `ui.visuals().*`             | DELETE (CSS vars)|
//! | `ctx.input(\|i\|…)`          | DELETE (Msg dispatch handles input) |
//! | `.clicked()` / `.changed()`  | DELETE (widget call site returns; msg handles it) |
//! | `ctx.output_mut(\|o\|…)`     | DELETE           |
//! | `ctx.memory_mut(\|m\|…)`     | DELETE           |
//! | `ctx.options_mut(\|o\|…)`    | DELETE           |
//! | `ui.set_max_width(…)` etc.   | CSS constraint   |
//!
//! # Ambiguity markers
//!
//! | egui                   | `AmbiguityKind`    |
//! |------------------------|---------------------|
//! | `ui.painter().*`       | `CanvasToSvg`       |
//! | `egui::plot::*`        | `ChartComponent`    |
//! | `ui.with_layout(…)`    | `LayoutSystem`      |
//! | `ui.allocate_ui(…)`    | `CustomSizing`      |
//! | `ScrollArea::*`        | `ScrollAreaWidget`  |
//! | `.on_hover_text(…)`    | `Tooltip`           |
//! | `egui::RichText::*`    | `RichText`          |
//! | `ui.image(…)`          | `ImageWidget`       |
//! | `ctx.input(\|i\|…)`    | `InputHandler`      |

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
    /// `ctx.input(|i| …)` — keyboard / pointer input; handled by Msg dispatch in actors.
    InputHandler,
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
            Self::InputHandler      => "??INPUT_HANDLER??",
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
        "colored_label"     => Ok("colored_label".to_string()),

        // --- Form elements ---
        "button"            => Ok("button".to_string()),
        "small_button"      => Ok("small_button".to_string()),
        "checkbox"          => Ok("checkbox".to_string()),
        "radio"             => Ok("radio".to_string()),
        "text_edit_singleline" => Ok("text_input".to_string()),
        "text_edit_multiline"  => Ok("textarea".to_string()),

        // Selectable items (list selection patterns)
        "selectable_label"  => Ok("selectable_label".to_string()),
        "selectable_value"  => Ok("selectable_value".to_string()),

        // Loading indicator
        "spinner"           => Ok("spinner".to_string()),

        // Slider / numeric input — egui DragValue, Slider
        "drag_value" | "slider" => Ok("number_input".to_string()),

        // ComboBox is complex (opens a popup), flag as layout ambiguity
        "combo_box"         => Err(AmbiguityKind::LayoutSystem),

        // --- Layout ---
        "separator"         => Ok("separator".to_string()),
        "horizontal"        => Ok("horizontal".to_string()),
        "vertical"          => Ok("vertical".to_string()),
        "horizontal_wrapped"=> Ok("horizontal_wrapped".to_string()),
        "vertical_centered" => Ok("vertical_centered".to_string()),
        "centered_and_justified" => Ok("centered_and_justified".to_string()),
        "indent"            => Ok("indent".to_string()),
        "add_space" | "spacing" => Ok("spacer".to_string()),

        // Menu
        "menu_button"       => Ok("menu_button".to_string()),

        // --- Control flow helpers ---
        "collapsing"        => Ok("collapsible".to_string()),
        "group"             => Ok("group".to_string()),

        // Conditional widget wrapper (add_enabled, add_visible)
        "add_enabled" | "add_visible" => Ok("conditional_widget".to_string()),

        // --- Response methods — no-ops in Qliphoth ---
        // In egui, widgets return a Response; in Qliphoth actors handle events via Msg.
        // These calls appear as chained method calls (e.g. `.clicked()`) and should
        // be deleted — the event is dispatched by the Qliphoth runtime instead.
        "clicked" | "double_clicked" | "triple_clicked"
        | "secondary_clicked" | "middle_clicked"
        | "changed" | "lost_focus" | "gained_focus"
        | "hovered" | "highlighted" | "enabled"
        | "is_pointer_button_down_on" | "drag_started"
        | "drag_released" | "dragged"           => Ok("noop_response".to_string()),

        // --- Context / style mutations — no-ops in Qliphoth (use CSS / theme system) ---
        "visuals" | "weak_text_color" | "text_color"
        | "style" | "spacing_mut" | "options_mut"
        | "set_visuals" | "set_style"           => Ok("noop_style".to_string()),

        // Egui context mutations deleted in Qliphoth
        "output_mut" | "memory_mut"             => Ok("noop_ctx_mutation".to_string()),

        // Sizing constraints — translate to CSS; no explicit VNode builder needed
        "set_max_width" | "set_min_width" | "set_width"
        | "set_max_height" | "set_min_height" | "set_height"
        | "set_clip_rect"                       => Ok("noop_sizing".to_string()),

        // Context queries — deleted (no retained-mode layout queries in Qliphoth)
        "is_rect_visible" | "make_persistent_id"
        | "available_width" | "available_height"
        | "available_size" | "available_size_before_wrap"
        | "available_rect_before_wrap" | "screen_rect"
        | "max_rect" | "cursor"                 => Ok("noop_ctx_query".to_string()),

        // Cursor icon — map to CSS cursor property (no VNode needed at call site)
        "set_cursor_icon"                       => Ok("noop_cursor".to_string()),

        // Close menu — event handled by Msg dispatch
        "close_menu" | "close"                  => Ok("noop_close".to_string()),

        // Grid row terminators
        "end_row"                               => Ok("noop_end_row".to_string()),

        // egui::Grid — no direct VNode; needs manual migration to table/grid
        "show" if args.is_empty()               => Ok("noop_show".to_string()),

        // --- Deleted patterns (no-ops in Qliphoth) ---
        "request_repaint"   => Ok("noop_repaint".to_string()),
        "ctx"               => Ok("noop_ctx".to_string()),

        // --- Ambiguities ---
        "painter"           => Err(AmbiguityKind::CanvasToSvg),
        // Direct painter drawing calls (when called on a stored painter ref)
        "rect_filled" | "rect_stroke" | "line_segment" | "line"
        | "circle_filled" | "circle_stroke" | "text" | "galley"
        | "arrow" | "debug_rect"                => Err(AmbiguityKind::CanvasToSvg),

        "fonts"                                 => Err(AmbiguityKind::RichText),

        "with_layout"       => Err(AmbiguityKind::LayoutSystem),

        "allocate_ui" | "allocate_exact_size" | "allocate_response"
        | "allocate_space" | "allocate_painter" | "allocate_rect"
        | "put"                                 => Err(AmbiguityKind::CustomSizing),

        "on_hover_text" | "on_hover_ui"         => Err(AmbiguityKind::Tooltip),

        "image"                                 => Err(AmbiguityKind::ImageWidget),

        // ctx.input(|i| …) — keyboard/pointer input; needs manual Msg wiring
        "input"                                 => Err(AmbiguityKind::InputHandler),

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
        _                                       => Err(AmbiguityKind::GenericWidget),
    }
}

// =============================================================================
// VNode code generation helpers (used by generator.rs)
// =============================================================================

/// Generate the Sigil VNode builder expression for a classified pattern.
///
/// Returns `None` for no-op patterns (e.g. `request_repaint`, response methods).
pub fn vnode_for_pattern(kind: &str, args: &[String]) -> Option<String> {
    let first  = args.first().map(|s| s.as_str()).unwrap_or("\"\"");
    let second = args.get(1).map(|s| s.as_str()).unwrap_or("\"\"");

    match kind {
        // --- Text ---
        "label"       => Some(format!("p()·text({})·to_vnode()", first)),
        "heading"     => Some(format!("h3()·text({})·to_vnode()", first)),
        "small"       => Some(format!("small()·text({})·to_vnode()", first)),
        "monospace"   => Some(format!("code()·text({})·to_vnode()", first)),
        "code"        => Some(format!("code()·text({})·to_vnode()", first)),
        "strong"      => Some(format!("strong()·text({})·to_vnode()", first)),
        "weak"        => Some(format!(
            "span()·style(\"color:#969696;\")·text({})·to_vnode()", first
        )),
        // colored_label(color, text) — color is first arg, text is second
        "colored_label" => Some(format!(
            "span()·style(\"color:??COLOR??\")·text({})·to_vnode() // ??COLOR??={}", second, first
        )),

        // --- Form ---
        "button" | "small_button" => Some(format!(
            "button()·on_click(|_| Msg::TODO)·text({})·to_vnode()", first
        )),
        "checkbox" => Some(format!(
            "input()·attr(\"type\",\"checkbox\")·attr(\"checked\",self.{})·to_vnode()",
            strip_ampersand_mut(first)
        )),
        "radio" => Some(format!(
            "input()·attr(\"type\",\"radio\")·attr(\"value\",self.{})·to_vnode()",
            strip_ampersand_mut(first)
        )),
        "text_input" => Some(format!(
            "input()·attr(\"type\",\"text\")·attr(\"value\",self.{})·on_change(|v| Msg::TODO(v))·to_vnode()",
            strip_ampersand_mut(first)
        )),
        "textarea" => Some(format!(
            "textarea()·attr(\"value\",self.{})·on_change(|v| Msg::TODO(v))·to_vnode()",
            strip_ampersand_mut(first)
        )),
        "selectable_label" => Some(format!(
            "li()·on_click(|_| Msg::TODO)·attr(\"aria-selected\",{})·text({})·to_vnode()",
            first, second
        )),
        "selectable_value" => Some(format!(
            "option()·attr(\"selected\",self.{}=={})·text({})·to_vnode() // ??SELECTABLE??",
            strip_ampersand_mut(first), second,
            args.get(2).map(|s| s.as_str()).unwrap_or("\"\"")
        )),
        "number_input" => Some(format!(
            "input()·attr(\"type\",\"number\")·attr(\"value\",self.{})·on_change(|v| Msg::TODO(v))·to_vnode()",
            strip_ampersand_mut(first)
        )),
        "spinner" => Some("span()·class(\"spinner\")·to_vnode()".to_string()),

        // --- Layout ---
        "separator" => Some("hr()·to_vnode()".to_string()),
        "horizontal" => Some(
            "div()·style(\"display:flex;flex-direction:row;\")·to_vnode() // ??INLINE_CHILDREN??".to_string()
        ),
        "vertical" => Some(
            "div()·style(\"display:flex;flex-direction:column;\")·to_vnode() // ??INLINE_CHILDREN??".to_string()
        ),
        "horizontal_wrapped" => Some(
            "div()·style(\"display:flex;flex-wrap:wrap;\")·to_vnode() // ??INLINE_CHILDREN??".to_string()
        ),
        "vertical_centered" => Some(
            "div()·style(\"display:flex;flex-direction:column;align-items:center;\")·to_vnode() // ??INLINE_CHILDREN??".to_string()
        ),
        "centered_and_justified" => Some(
            "div()·style(\"display:flex;justify-content:center;align-items:center;\")·to_vnode() // ??INLINE_CHILDREN??".to_string()
        ),
        "indent" => Some(
            "div()·style(\"padding-left:1em;\")·to_vnode() // ??INLINE_CHILDREN??".to_string()
        ),
        "spacer" => Some("div()·style(\"flex-shrink:0;\")·to_vnode()".to_string()),
        "collapsible" => Some(format!(
            "details()·child(summary()·text({})·to_vnode())·to_vnode() // ??COLLAPSING??", first
        )),
        "group" => Some(
            "div()·style(\"border:1px solid rgba(108,108,118,0.2);border-radius:4px;padding:8px;\")·to_vnode()".to_string()
        ),
        "menu_button" => Some(format!(
            "button()·class(\"menu-button\")·text({})·to_vnode() // ??INLINE_CHILDREN??", first
        )),
        "conditional_widget" => Some(
            "// ??ADD_ENABLED?? — use ·attr(\"disabled\",!condition) on the child widget".to_string()
        ),

        // No-ops — all return None (deleted in Qliphoth)
        s if s.starts_with("noop_") => None,

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

    #[test]
    fn test_selectable_label() {
        assert_eq!(
            classify_method_call("selectable_label", &["true".into(), "\"Item\"".into()]),
            Ok("selectable_label".into())
        );
    }

    #[test]
    fn test_selectable_value() {
        assert_eq!(
            classify_method_call("selectable_value", &["&mut self.selected".into(), "0".into(), "\"First\"".into()]),
            Ok("selectable_value".into())
        );
    }

    #[test]
    fn test_spinner() {
        assert_eq!(classify_method_call("spinner", &[]), Ok("spinner".into()));
    }

    #[test]
    fn test_clicked_is_noop() {
        assert_eq!(classify_method_call("clicked", &[]), Ok("noop_response".into()));
    }

    #[test]
    fn test_changed_is_noop() {
        assert_eq!(classify_method_call("changed", &[]), Ok("noop_response".into()));
    }

    #[test]
    fn test_visuals_is_noop() {
        assert_eq!(classify_method_call("visuals", &[]), Ok("noop_style".into()));
    }

    #[test]
    fn test_input_is_input_handler() {
        assert_eq!(classify_method_call("input", &[]), Err(AmbiguityKind::InputHandler));
    }

    #[test]
    fn test_horizontal_wrapped() {
        assert_eq!(classify_method_call("horizontal_wrapped", &[]), Ok("horizontal_wrapped".into()));
    }

    #[test]
    fn test_noop_response_vnode_is_none() {
        assert_eq!(vnode_for_pattern("noop_response", &[]), None);
    }

    #[test]
    fn test_noop_style_vnode_is_none() {
        assert_eq!(vnode_for_pattern("noop_style", &[]), None);
    }

    #[test]
    fn test_spinner_vnode() {
        assert_eq!(
            vnode_for_pattern("spinner", &[]),
            Some("span()·class(\"spinner\")·to_vnode()".into())
        );
    }

    #[test]
    fn test_rect_filled_is_canvas() {
        assert_eq!(classify_method_call("rect_filled", &[]), Err(AmbiguityKind::CanvasToSvg));
    }
}
