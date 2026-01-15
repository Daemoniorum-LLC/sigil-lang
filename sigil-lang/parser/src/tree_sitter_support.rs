//! Tree-sitter integration for Sigil
//!
//! Provides syntax parsing capabilities for multiple programming languages
//! using tree-sitter grammars. This enables Samael and other AI agents to
//! perform real syntax analysis on source code.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use tree_sitter::{Language, Parser, Tree, Node};

/// Supported languages for tree-sitter parsing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TSLanguage {
    Rust,
    Python,
    JavaScript,
    TypeScript,
    TypeScriptTsx,
    Go,
    C,
    Cpp,
    Java,
    Json,
    Css,
    Bash,
}

impl TSLanguage {
    /// Get the tree-sitter Language for this enum variant
    pub fn get_language(&self) -> Language {
        match self {
            TSLanguage::Rust => tree_sitter_rust::language(),
            TSLanguage::Python => tree_sitter_python::language(),
            TSLanguage::JavaScript => tree_sitter_javascript::language(),
            TSLanguage::TypeScript => tree_sitter_typescript::language_typescript(),
            TSLanguage::TypeScriptTsx => tree_sitter_typescript::language_tsx(),
            TSLanguage::Go => tree_sitter_go::language(),
            TSLanguage::C => tree_sitter_c::language(),
            TSLanguage::Cpp => tree_sitter_cpp::language(),
            TSLanguage::Java => tree_sitter_java::language(),
            TSLanguage::Json => tree_sitter_json::language(),
            TSLanguage::Css => tree_sitter_css::language(),
            TSLanguage::Bash => tree_sitter_bash::language(),
        }
    }

    /// Parse a string to get the language enum
    /// Handles formats like "Rust", "rust", "Language::Rust", etc.
    pub fn from_str(s: &str) -> Option<Self> {
        // Handle enum-style strings like "Language::Rust"
        let name = s.rsplit("::").next().unwrap_or(s);

        match name.to_lowercase().as_str() {
            "rust" | "rs" | "sigil" => Some(TSLanguage::Rust),
            "python" | "py" => Some(TSLanguage::Python),
            "javascript" | "js" => Some(TSLanguage::JavaScript),
            "typescript" | "ts" => Some(TSLanguage::TypeScript),
            "tsx" | "typescripttsx" => Some(TSLanguage::TypeScriptTsx),
            "go" | "golang" => Some(TSLanguage::Go),
            "c" => Some(TSLanguage::C),
            "cpp" | "c++" | "cxx" => Some(TSLanguage::Cpp),
            "java" => Some(TSLanguage::Java),
            "json" => Some(TSLanguage::Json),
            "css" => Some(TSLanguage::Css),
            "bash" | "sh" | "shell" => Some(TSLanguage::Bash),
            // Languages not yet supported - return None
            "html" | "htm" => None,  // tree-sitter-html uses incompatible version
            "kotlin" | "kt" => None,
            "yaml" | "yml" => None,
            "toml" => None,
            "sql" => None,
            "markdown" | "md" => None,
            _ => None,
        }
    }

    /// Get the canonical name for this language
    pub fn name(&self) -> &'static str {
        match self {
            TSLanguage::Rust => "Rust",
            TSLanguage::Python => "Python",
            TSLanguage::JavaScript => "JavaScript",
            TSLanguage::TypeScript => "TypeScript",
            TSLanguage::TypeScriptTsx => "TypeScriptTsx",
            TSLanguage::Go => "Go",
            TSLanguage::C => "C",
            TSLanguage::Cpp => "Cpp",
            TSLanguage::Java => "Java",
            TSLanguage::Json => "Json",
            TSLanguage::Css => "Css",
            TSLanguage::Bash => "Bash",
        }
    }
}

/// Tree-sitter parser wrapper
pub struct TSParser {
    parser: Parser,
    language: TSLanguage,
}

impl TSParser {
    /// Create a new parser for the given language
    pub fn new(language: TSLanguage) -> Result<Self, String> {
        let mut parser = Parser::new();
        parser.set_language(language.get_language())
            .map_err(|e| format!("Failed to set language: {:?}", e))?;

        Ok(TSParser { parser, language })
    }

    /// Parse source code and return a tree
    pub fn parse(&mut self, source: &str) -> Result<TSTree, String> {
        self.parser.parse(source, None)
            .map(|tree| TSTree {
                tree,
                source: source.to_string(),
                language: self.language,
            })
            .ok_or_else(|| "Failed to parse source code".to_string())
    }

    /// Get the language this parser is configured for
    pub fn language(&self) -> TSLanguage {
        self.language
    }
}

/// Wrapper for a parsed syntax tree
pub struct TSTree {
    tree: Tree,
    source: String,
    language: TSLanguage,
}

impl TSTree {
    /// Get the root node of the tree
    pub fn root_node(&self) -> Node {
        self.tree.root_node()
    }

    /// Get the source code that was parsed
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Get the language of this tree
    pub fn language(&self) -> TSLanguage {
        self.language
    }
}

/// Parse source code with a given language
pub fn parse_source(language_str: &str, source: &str) -> Result<TSTree, String> {
    let language = TSLanguage::from_str(language_str)
        .ok_or_else(|| format!("Unsupported language: {}", language_str))?;

    let mut parser = TSParser::new(language)?;
    parser.parse(source)
}

/// Convert a tree-sitter Node to interpreter Value fields
/// Returns a HashMap suitable for creating a SyntaxNode struct
pub fn node_to_value(node: &Node) -> HashMap<String, crate::interpreter::Value> {
    use crate::interpreter::Value;

    let mut fields = HashMap::new();

    // Basic node info
    fields.insert("kind".to_string(), Value::String(Rc::new(node.kind().to_string())));
    fields.insert("is_named".to_string(), Value::Bool(node.is_named()));
    fields.insert("is_error".to_string(), Value::Bool(node.is_error()));
    fields.insert("is_missing".to_string(), Value::Bool(node.is_missing()));

    // Position info
    let start = node.start_position();
    let end = node.end_position();

    let mut start_fields = HashMap::new();
    start_fields.insert("row".to_string(), Value::Int(start.row as i64));
    start_fields.insert("column".to_string(), Value::Int(start.column as i64));
    fields.insert("start".to_string(), Value::Struct {
        name: "Position".to_string(),
        fields: Rc::new(RefCell::new(start_fields)),
    });

    let mut end_fields = HashMap::new();
    end_fields.insert("row".to_string(), Value::Int(end.row as i64));
    end_fields.insert("column".to_string(), Value::Int(end.column as i64));
    fields.insert("end".to_string(), Value::Struct {
        name: "Position".to_string(),
        fields: Rc::new(RefCell::new(end_fields)),
    });

    // Byte range
    fields.insert("start_byte".to_string(), Value::Int(node.start_byte() as i64));
    fields.insert("end_byte".to_string(), Value::Int(node.end_byte() as i64));

    // Child count
    fields.insert("child_count".to_string(), Value::Int(node.child_count() as i64));
    fields.insert("named_child_count".to_string(), Value::Int(node.named_child_count() as i64));

    // Children (recursively converted)
    let children: Vec<Value> = (0..node.child_count())
        .filter_map(|i| node.child(i))
        .map(|child| Value::Struct {
            name: "SyntaxNode".to_string(),
            fields: Rc::new(RefCell::new(node_to_value(&child))),
        })
        .collect();
    fields.insert("children".to_string(), Value::Array(Rc::new(RefCell::new(children))));

    // Named children only
    let named_children: Vec<Value> = (0..node.named_child_count())
        .filter_map(|i| node.named_child(i))
        .map(|child| Value::Struct {
            name: "SyntaxNode".to_string(),
            fields: Rc::new(RefCell::new(node_to_value(&child))),
        })
        .collect();
    fields.insert("named_children".to_string(), Value::Array(Rc::new(RefCell::new(named_children))));

    fields
}

/// Get the text content of a node from source
pub fn node_text<'a>(node: &Node, source: &'a str) -> &'a str {
    &source[node.start_byte()..node.end_byte()]
}

/// List all supported languages
pub fn supported_languages() -> Vec<&'static str> {
    vec![
        "Rust", "Python", "JavaScript", "TypeScript", "TypeScriptTsx",
        "Go", "C", "Cpp", "Java", "Json", "Css", "Bash",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_rust() {
        let source = r#"
fn main() {
    println!("Hello, world!");
}
"#;
        let tree = parse_source("rust", source).unwrap();
        let root = tree.root_node();
        assert_eq!(root.kind(), "source_file");
        assert!(root.child_count() > 0);
    }

    #[test]
    fn test_parse_python() {
        let source = r#"
def greet(name):
    print(f"Hello, {name}!")

greet("world")
"#;
        let tree = parse_source("python", source).unwrap();
        let root = tree.root_node();
        assert_eq!(root.kind(), "module");
    }

    #[test]
    fn test_parse_javascript() {
        let source = r#"
function greet(name) {
    console.log(`Hello, ${name}!`);
}
greet("world");
"#;
        let tree = parse_source("javascript", source).unwrap();
        let root = tree.root_node();
        assert_eq!(root.kind(), "program");
    }

    #[test]
    fn test_language_from_str() {
        assert_eq!(TSLanguage::from_str("rust"), Some(TSLanguage::Rust));
        assert_eq!(TSLanguage::from_str("Rust"), Some(TSLanguage::Rust));
        assert_eq!(TSLanguage::from_str("Language::Rust"), Some(TSLanguage::Rust));
        assert_eq!(TSLanguage::from_str("python"), Some(TSLanguage::Python));
        assert_eq!(TSLanguage::from_str("py"), Some(TSLanguage::Python));
        assert_eq!(TSLanguage::from_str("unknown"), None);
    }

    #[test]
    fn test_unsupported_languages() {
        assert!(TSLanguage::from_str("kotlin").is_none());
        assert!(TSLanguage::from_str("yaml").is_none());
        assert!(TSLanguage::from_str("toml").is_none());
    }
}
