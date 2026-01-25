//! Code Formatter for Sigil Language
//!
//! Provides consistent code formatting with configurable style options.
//!
//! Usage:
//!   sigil fmt <file>           Format a single file
//!   sigil fmt <dir>            Format all .sg/.sigil files in directory
//!   sigil fmt --check <path>   Check formatting without modifying
//!   sigil fmt --stdin          Read from stdin, write to stdout

use std::fs;
use std::io::{self, Read};
use std::path::Path;

/// Formatting configuration
#[derive(Debug, Clone)]
pub struct FormatConfig {
    /// Indentation width (spaces)
    pub indent_width: usize,
    /// Use tabs instead of spaces
    pub use_tabs: bool,
    /// Maximum line width
    pub max_line_width: usize,
    /// Add trailing commas
    pub trailing_commas: bool,
    /// Space after colons in type annotations
    pub space_after_colon: bool,
    /// Space around binary operators
    pub space_around_ops: bool,
}

impl Default for FormatConfig {
    fn default() -> Self {
        Self {
            indent_width: 4,
            use_tabs: false,
            max_line_width: 100,
            trailing_commas: true,
            space_after_colon: true,
            space_around_ops: true,
        }
    }
}

impl FormatConfig {
    /// Load config from sigil.toml or .sigilfmt.toml
    pub fn load() -> Self {
        // Try to load from config files
        if let Ok(content) = fs::read_to_string("sigil.toml") {
            if let Ok(parsed) = toml::from_str::<toml::Value>(&content) {
                if let Some(fmt) = parsed.get("fmt") {
                    return Self::from_toml(fmt);
                }
            }
        }

        if let Ok(content) = fs::read_to_string(".sigilfmt.toml") {
            if let Ok(parsed) = toml::from_str::<toml::Value>(&content) {
                return Self::from_toml(&parsed);
            }
        }

        Self::default()
    }

    fn from_toml(value: &toml::Value) -> Self {
        let mut config = Self::default();

        if let Some(width) = value.get("indent_width").and_then(|v| v.as_integer()) {
            config.indent_width = width as usize;
        }
        if let Some(tabs) = value.get("use_tabs").and_then(|v| v.as_bool()) {
            config.use_tabs = tabs;
        }
        if let Some(width) = value.get("max_line_width").and_then(|v| v.as_integer()) {
            config.max_line_width = width as usize;
        }
        if let Some(trailing) = value.get("trailing_commas").and_then(|v| v.as_bool()) {
            config.trailing_commas = trailing;
        }
        if let Some(space) = value.get("space_after_colon").and_then(|v| v.as_bool()) {
            config.space_after_colon = space;
        }
        if let Some(space) = value.get("space_around_ops").and_then(|v| v.as_bool()) {
            config.space_around_ops = space;
        }

        config
    }
}

/// Line-based code formatter
pub struct Formatter {
    config: FormatConfig,
}

impl Formatter {
    pub fn new(config: FormatConfig) -> Self {
        Self { config }
    }

    /// Format source code string
    pub fn format_source(&self, source: &str) -> Result<String, String> {
        let mut output = String::new();
        let mut indent_level: i32 = 0;

        for line in source.lines() {
            let trimmed = line.trim();

            // Skip empty lines but preserve them
            if trimmed.is_empty() {
                output.push('\n');
                continue;
            }

            // Handle comment lines
            if trimmed.starts_with("//") {
                output.push_str(&self.make_indent(indent_level));
                output.push_str(trimmed);
                output.push('\n');
                continue;
            }

            // Adjust indent for closing braces at start of line
            let starts_with_close =
                trimmed.starts_with('}') || trimmed.starts_with(')') || trimmed.starts_with(']');

            if starts_with_close && indent_level > 0 {
                indent_level -= 1;
            }

            // Format the line
            let formatted_line = self.format_line(trimmed);

            // Write with proper indentation
            output.push_str(&self.make_indent(indent_level));
            output.push_str(&formatted_line);
            output.push('\n');

            // Count braces for next line's indentation
            let mut depth_change: i32 = 0;
            let mut in_string = false;
            let mut in_char = false;
            let mut prev_char = '\0';

            for ch in trimmed.chars() {
                // Track string/char literals
                if ch == '"' && prev_char != '\\' && !in_char {
                    in_string = !in_string;
                } else if ch == '\'' && prev_char != '\\' && !in_string {
                    in_char = !in_char;
                }

                // Count braces outside of strings
                if !in_string && !in_char {
                    match ch {
                        '{' | '(' | '[' => depth_change += 1,
                        '}' | ')' | ']' => {
                            // Only decrement if this isn't at start (already handled)
                            if !starts_with_close || depth_change > 0 {
                                depth_change -= 1;
                            }
                        }
                        _ => {}
                    }
                }

                prev_char = ch;
            }

            indent_level += depth_change;
            if indent_level < 0 {
                indent_level = 0;
            }
        }

        // Ensure file ends with newline
        if !output.ends_with('\n') {
            output.push('\n');
        }

        // Remove trailing whitespace from each line
        let cleaned: String = output
            .lines()
            .map(|line| line.trim_end())
            .collect::<Vec<_>>()
            .join("\n");

        Ok(if cleaned.is_empty() {
            String::new()
        } else {
            cleaned + "\n"
        })
    }

    fn make_indent(&self, level: i32) -> String {
        if level <= 0 {
            return String::new();
        }
        let level = level as usize;
        if self.config.use_tabs {
            "\t".repeat(level)
        } else {
            " ".repeat(level * self.config.indent_width)
        }
    }

    fn format_line(&self, line: &str) -> String {
        let mut result = String::new();
        let mut chars = line.chars().peekable();
        let mut in_string = false;
        let mut in_char = false;
        let mut prev_char = '\0';
        let mut last_was_space = false;

        while let Some(ch) = chars.next() {
            // Track string/char literals
            if ch == '"' && prev_char != '\\' && !in_char {
                in_string = !in_string;
            } else if ch == '\'' && prev_char != '\\' && !in_string {
                in_char = !in_char;
            }

            // Inside strings/chars, preserve exactly
            if in_string || in_char {
                result.push(ch);
                prev_char = ch;
                last_was_space = false;
                continue;
            }

            // Normalize whitespace
            if ch.is_whitespace() {
                if !last_was_space && !result.is_empty() {
                    result.push(' ');
                    last_was_space = true;
                }
                prev_char = ch;
                continue;
            }

            last_was_space = false;

            // Handle operators with spacing
            if self.config.space_around_ops {
                match ch {
                    '+' | '-' | '*' | '/' | '%' | '=' | '<' | '>' | '!' | '&' | '|' | '^' => {
                        // Check for compound operators
                        let next = chars.peek().copied();
                        let is_compound = matches!(
                            (ch, next),
                            ('+', Some('+'))
                                | ('-', Some('-'))
                                | ('*', Some('*'))
                                | ('/', Some('/'))
                                | ('=', Some('='))
                                | ('!', Some('='))
                                | ('<', Some('='))
                                | ('>', Some('='))
                                | ('<', Some('<'))
                                | ('>', Some('>'))
                                | ('&', Some('&'))
                                | ('|', Some('|'))
                                | ('|', Some('>'))
                                | ('-', Some('>'))
                                | ('=', Some('>'))
                        );

                        // Don't add space before unary operators
                        let is_unary = prev_char == '('
                            || prev_char == '['
                            || prev_char == ','
                            || prev_char == '='
                            || prev_char == '<'
                            || prev_char == '>'
                            || prev_char == '{'
                            || prev_char == '\0'
                            || result.is_empty();

                        // Special case: don't space around :: or ->
                        if ch == ':' && next == Some(':') {
                            result.push(ch);
                            prev_char = ch;
                            continue;
                        }

                        if ch == '-' && next == Some('>') {
                            // Return type arrow
                            if !result.ends_with(' ') {
                                result.push(' ');
                            }
                            result.push('-');
                            result.push(chars.next().unwrap());
                            result.push(' ');
                            prev_char = '>';
                            continue;
                        }

                        if !is_unary {
                            if !result.ends_with(' ') {
                                result.push(' ');
                            }
                        }

                        result.push(ch);

                        if is_compound {
                            result.push(chars.next().unwrap());
                        }

                        // Add space after binary operators
                        if !is_unary && !matches!(next, Some('=') | Some('>') | Some('<')) {
                            result.push(' ');
                            last_was_space = true;
                        }

                        prev_char = ch;
                        continue;
                    }
                    _ => {}
                }
            }

            // Handle colons with optional spacing
            if ch == ':' {
                let next = chars.peek().copied();
                if next == Some(':') {
                    // Path separator ::
                    result.push(':');
                    result.push(chars.next().unwrap());
                    prev_char = ':';
                    continue;
                }

                result.push(':');
                if self.config.space_after_colon && next != Some(':') {
                    result.push(' ');
                    last_was_space = true;
                }
                prev_char = ':';
                continue;
            }

            // Handle commas
            if ch == ',' {
                result.push(',');
                result.push(' ');
                last_was_space = true;
                prev_char = ch;
                continue;
            }

            // Handle semicolons
            if ch == ';' {
                // Remove trailing space before semicolon
                if result.ends_with(' ') {
                    result.pop();
                }
                result.push(';');
                prev_char = ch;
                continue;
            }

            // Handle opening braces
            if ch == '{' {
                // Add space before { if not already present
                if !result.is_empty() && !result.ends_with(' ') && !result.ends_with('(') {
                    result.push(' ');
                }
                result.push('{');
                prev_char = ch;
                continue;
            }

            result.push(ch);
            prev_char = ch;
        }

        // Trim trailing whitespace
        result.trim_end().to_string()
    }
}

/// Format a file in place
pub fn format_file(path: &Path, config: &FormatConfig) -> Result<bool, String> {
    let source = fs::read_to_string(path).map_err(|e| format!("Failed to read file: {}", e))?;

    let formatter = Formatter::new(config.clone());
    let formatted = formatter.format_source(&source)?;

    if formatted == source {
        return Ok(false); // No changes
    }

    fs::write(path, &formatted).map_err(|e| format!("Failed to write file: {}", e))?;

    Ok(true)
}

/// Check if a file is formatted
pub fn check_file(path: &Path, config: &FormatConfig) -> Result<bool, String> {
    let source = fs::read_to_string(path).map_err(|e| format!("Failed to read file: {}", e))?;

    let formatter = Formatter::new(config.clone());
    let formatted = formatter.format_source(&source)?;

    Ok(formatted == source)
}

/// Format source from stdin
pub fn format_stdin(config: &FormatConfig) -> Result<String, String> {
    let mut source = String::new();
    io::stdin()
        .read_to_string(&mut source)
        .map_err(|e| format!("Failed to read stdin: {}", e))?;

    let formatter = Formatter::new(config.clone());
    formatter.format_source(&source)
}

/// Format all Sigil files in a directory
pub fn format_directory(
    dir: &Path,
    config: &FormatConfig,
    check_only: bool,
) -> Result<FormatResult, String> {
    let mut result = FormatResult::default();

    for entry in walkdir::WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            let path = e.path();
            path.is_file()
                && (path
                    .extension()
                    .map_or(false, |ext| ext == "sg" || ext == "sigil"))
        })
    {
        let path = entry.path();
        result.total += 1;

        if check_only {
            match check_file(path, config) {
                Ok(true) => result.formatted += 1,
                Ok(false) => {
                    result.unformatted.push(path.to_path_buf());
                }
                Err(e) => {
                    result.errors.push((path.to_path_buf(), e));
                }
            }
        } else {
            match format_file(path, config) {
                Ok(true) => {
                    result.formatted += 1;
                    result.changed.push(path.to_path_buf());
                }
                Ok(false) => result.formatted += 1,
                Err(e) => {
                    result.errors.push((path.to_path_buf(), e));
                }
            }
        }
    }

    Ok(result)
}

/// Result of formatting operation
#[derive(Debug, Default)]
pub struct FormatResult {
    pub total: usize,
    pub formatted: usize,
    pub changed: Vec<std::path::PathBuf>,
    pub unformatted: Vec<std::path::PathBuf>,
    pub errors: Vec<(std::path::PathBuf, String)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_formatting() {
        let config = FormatConfig::default();
        let formatter = Formatter::new(config);

        let input = "rite main(){≔ x=1+2;}";
        let formatted = formatter.format_source(input).unwrap();
        assert!(formatted.contains("rite main()"));
    }

    #[test]
    fn test_indentation() {
        let config = FormatConfig::default();
        let formatter = Formatter::new(config);

        let input = "rite main() {\n≔ x = 1;\n}";
        let formatted = formatter.format_source(input).unwrap();
        assert!(formatted.contains("    ≔ x")); // 4 spaces indent
    }

    #[test]
    fn test_preserves_strings() {
        let config = FormatConfig::default();
        let formatter = Formatter::new(config);

        let input = r#"≔ s = "hello   world";"#;
        let formatted = formatter.format_source(input).unwrap();
        assert!(formatted.contains("\"hello   world\""));
    }
}
