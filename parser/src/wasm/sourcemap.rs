//! Source Map Generation for Sigil WASM
//!
//! Generates source maps that map WASM bytecode offsets back to Sigil source locations.
//! This enables debugging and meaningful error messages in browser devtools.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::span::Span;

/// Line and column position in source code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct SourceLocation {
    /// 1-based line number
    pub line: u32,
    /// 0-based column number (in UTF-8 bytes)
    pub column: u32,
}

impl SourceLocation {
    pub fn new(line: u32, column: u32) -> Self {
        Self { line, column }
    }
}

/// Maps byte offsets in source code to line/column positions.
#[derive(Debug, Clone)]
pub struct SourceLineMap {
    /// Byte offset of each line start (0-indexed)
    line_starts: Vec<usize>,
    /// Total source length
    source_len: usize,
}

impl SourceLineMap {
    /// Create a new line map from source text.
    pub fn new(source: &str) -> Self {
        let mut line_starts = vec![0]; // Line 1 starts at offset 0

        for (offset, ch) in source.char_indices() {
            if ch == '\n' {
                line_starts.push(offset + 1);
            }
        }

        Self {
            line_starts,
            source_len: source.len(),
        }
    }

    /// Convert a byte offset to line/column.
    pub fn offset_to_location(&self, offset: usize) -> SourceLocation {
        if offset >= self.source_len {
            // Return last line, column 0
            return SourceLocation::new(self.line_starts.len() as u32, 0);
        }

        // Binary search to find the line
        let line_idx = match self.line_starts.binary_search(&offset) {
            Ok(idx) => idx,                    // Exact match at line start
            Err(idx) => idx.saturating_sub(1), // Between line starts
        };

        let line = line_idx as u32 + 1; // 1-based
        let column = (offset - self.line_starts[line_idx]) as u32; // 0-based

        SourceLocation::new(line, column)
    }

    /// Convert a Span to start/end locations.
    pub fn span_to_range(&self, span: Span) -> (SourceLocation, SourceLocation) {
        (
            self.offset_to_location(span.start),
            self.offset_to_location(span.end),
        )
    }

    /// Get total number of lines.
    pub fn line_count(&self) -> usize {
        self.line_starts.len()
    }
}

/// A source mapping entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceMapping {
    /// Function name
    pub function: String,
    /// Instruction offset within the function
    pub instruction_offset: u32,
    /// Source location
    pub location: SourceLocation,
    /// Optional end location (for spans)
    pub end_location: Option<SourceLocation>,
}

/// Complete source map for a WASM module.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceMap {
    /// Version (always 1 for now)
    pub version: u32,
    /// Source file name
    pub file: String,
    /// Original source code (optional, for inline source)
    pub source_content: Option<String>,
    /// Function mappings
    pub functions: HashMap<String, FunctionSourceMap>,
}

/// Source map for a single function.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FunctionSourceMap {
    /// Function name
    pub name: String,
    /// Start location in source
    pub start: SourceLocation,
    /// End location in source
    pub end: SourceLocation,
    /// Instruction-to-source mappings
    pub mappings: Vec<InstructionMapping>,
}

/// Maps a single WASM instruction to source location.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstructionMapping {
    /// Instruction index within function
    pub instruction_idx: u32,
    /// Source line (1-based)
    pub line: u32,
    /// Source column (0-based)
    pub column: u32,
}

impl SourceMap {
    /// Create a new empty source map.
    pub fn new(file: impl Into<String>) -> Self {
        Self {
            version: 1,
            file: file.into(),
            source_content: None,
            functions: HashMap::new(),
        }
    }

    /// Create with inline source content.
    pub fn with_source(file: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            version: 1,
            file: file.into(),
            source_content: Some(source.into()),
            functions: HashMap::new(),
        }
    }

    /// Add a function mapping.
    pub fn add_function(&mut self, func_map: FunctionSourceMap) {
        self.functions.insert(func_map.name.clone(), func_map);
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(self).unwrap_or_default()
    }

    /// Serialize to compact JSON (for embedding in WASM).
    pub fn to_compact_json(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }

    /// Create WASM custom section bytes.
    /// Returns bytes for a "sigil_sourcemap" custom section.
    pub fn to_custom_section(&self) -> Vec<u8> {
        let json = self.to_compact_json();
        let name = b"sigil_sourcemap";

        // Custom section format: name_len + name + content
        let mut bytes = Vec::new();

        // Name (LEB128 length + bytes)
        bytes.push(name.len() as u8);
        bytes.extend_from_slice(name);

        // Content (the JSON)
        bytes.extend_from_slice(json.as_bytes());

        bytes
    }
}

/// Builder for tracking source mappings during compilation.
#[derive(Debug)]
pub struct SourceMapBuilder {
    /// Source file name
    file: String,
    /// Line map for the source
    line_map: SourceLineMap,
    /// Original source (for inline embedding)
    source: Option<String>,
    /// Current function being tracked
    current_function: Option<FunctionSourceMap>,
    /// Completed function maps
    functions: Vec<FunctionSourceMap>,
}

impl SourceMapBuilder {
    /// Create a new builder from source code.
    pub fn new(file: impl Into<String>, source: &str) -> Self {
        Self {
            file: file.into(),
            line_map: SourceLineMap::new(source),
            source: Some(source.to_string()),
            current_function: None,
            functions: Vec::new(),
        }
    }

    /// Start tracking a new function.
    pub fn begin_function(&mut self, name: impl Into<String>, span: Span) {
        let (start, end) = self.line_map.span_to_range(span);
        self.current_function = Some(FunctionSourceMap {
            name: name.into(),
            start,
            end,
            mappings: Vec::new(),
        });
    }

    /// Add an instruction mapping for the current function.
    pub fn add_instruction(&mut self, instruction_idx: u32, span: Span) {
        if let Some(ref mut func) = self.current_function {
            let loc = self.line_map.offset_to_location(span.start);
            func.mappings.push(InstructionMapping {
                instruction_idx,
                line: loc.line,
                column: loc.column,
            });
        }
    }

    /// End the current function.
    pub fn end_function(&mut self) {
        if let Some(func) = self.current_function.take() {
            self.functions.push(func);
        }
    }

    /// Build the final source map.
    pub fn build(self) -> SourceMap {
        let mut map = if let Some(source) = self.source {
            SourceMap::with_source(self.file, source)
        } else {
            SourceMap::new(self.file)
        };

        for func in self.functions {
            map.add_function(func);
        }

        map
    }

    /// Get the line map for offset calculations.
    pub fn line_map(&self) -> &SourceLineMap {
        &self.line_map
    }

    /// Convert an offset to a location.
    pub fn offset_to_location(&self, offset: usize) -> SourceLocation {
        self.line_map.offset_to_location(offset)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_map_simple() {
        let source = "line1\nline2\nline3";
        let map = SourceLineMap::new(source);

        assert_eq!(map.line_count(), 3);
        assert_eq!(map.offset_to_location(0), SourceLocation::new(1, 0));
        assert_eq!(map.offset_to_location(3), SourceLocation::new(1, 3));
        assert_eq!(map.offset_to_location(6), SourceLocation::new(2, 0));
        assert_eq!(map.offset_to_location(12), SourceLocation::new(3, 0));
    }

    #[test]
    fn test_line_map_empty_lines() {
        let source = "a\n\nb";
        let map = SourceLineMap::new(source);

        assert_eq!(map.line_count(), 3);
        assert_eq!(map.offset_to_location(0), SourceLocation::new(1, 0)); // 'a'
        assert_eq!(map.offset_to_location(2), SourceLocation::new(2, 0)); // empty line
        assert_eq!(map.offset_to_location(3), SourceLocation::new(3, 0)); // 'b'
    }

    #[test]
    fn test_source_map_json() {
        let mut map = SourceMap::new("test.sigil");
        map.add_function(FunctionSourceMap {
            name: "main".to_string(),
            start: SourceLocation::new(1, 0),
            end: SourceLocation::new(5, 1),
            mappings: vec![
                InstructionMapping {
                    instruction_idx: 0,
                    line: 2,
                    column: 4,
                },
                InstructionMapping {
                    instruction_idx: 1,
                    line: 3,
                    column: 4,
                },
            ],
        });

        let json = map.to_json();
        assert!(json.contains("\"main\""));
        assert!(json.contains("\"version\": 1"));
    }

    #[test]
    fn test_builder() {
        let source = "pub fn main() {\n    print(1);\n}";
        let mut builder = SourceMapBuilder::new("test.sigil", source);

        builder.begin_function("main", Span::new(0, source.len()));
        builder.add_instruction(0, Span::new(16, 25)); // print(1)
        builder.end_function();

        let map = builder.build();
        assert_eq!(map.functions.len(), 1);
        assert!(map.functions.contains_key("main"));
    }
}
