//! Macro expansion for WASM compilation.
//!
//! Handles Sigil macros like `format!`, `html!`, `vec!`, etc.
//! by expanding them into equivalent WASM runtime calls.

use wasm_encoder::Instruction;

use super::error::{WasmError, WasmResult};
use super::WasmCompiler;
use crate::parser::Parser;

impl WasmCompiler {
    /// Compile a macro invocation.
    /// Returns Ok(true) if the macro was handled, Ok(false) if not.
    pub fn compile_macro(&mut self, macro_name: &str, tokens: &str) -> WasmResult<bool> {
        match macro_name {
            "format" => {
                self.compile_format_macro(tokens)?;
                Ok(true)
            }
            "html" => {
                self.compile_html_macro(tokens)?;
                Ok(true)
            }
            "vec" => {
                self.compile_vec_macro(tokens)?;
                Ok(true)
            }
            "concat" => {
                self.compile_concat_macro(tokens)?;
                Ok(true)
            }
            "stringify" => {
                self.compile_stringify_macro(tokens)?;
                Ok(true)
            }
            "console_log" | "console_warn" | "console_error" => {
                self.compile_console_log_macro(macro_name, tokens)?;
                Ok(true)
            }
            "print" | "println" | "eprint" | "eprintln" => {
                self.compile_print_macro(macro_name, tokens)?;
                Ok(true)
            }
            "matches" => {
                self.compile_matches_macro(tokens)?;
                Ok(true)
            }
            "assert" | "assert_eq" | "assert_ne" => {
                // In WASM, assertions are no-ops for now (could call panic import)
                // Push true/unit for expression result
                let func = self.current_function_mut()
                    .ok_or_else(|| WasmError::internal("not in function context"))?;
                func.push(Instruction::I64Const(0)); // unit value
                Ok(true)
            }
            // Handled elsewhere as special cases
            "unreachable" | "panic" | "todo" | "unimplemented" => Ok(false),
            "debug_assert" | "debug_assert_eq" | "debug_assert_ne" => Ok(false),
            _ => Ok(false),
        }
    }

    /// Compile format! macro: format!("template {}", arg1, arg2, ...)
    /// Generates string concatenation via runtime imports.
    fn compile_format_macro(&mut self, tokens: &str) -> WasmResult<()> {
        let tokens = tokens.trim();

        // Empty format! -> empty string
        if tokens.is_empty() {
            let offset = self.add_string("");
            let func = self.current_function_mut()
                .ok_or_else(|| WasmError::internal("not in function context"))?;
            func.push(Instruction::I32Const(offset as i32));
            func.push(Instruction::I64ExtendI32U);
            return Ok(());
        }

        // Handle raw strings: RawStringDelimited("content") format from parser
        // or traditional r#"..."# syntax
        let (format_str, args_str) = if tokens.starts_with("RawStringDelimited(") {
            self.parse_raw_string_delimited(tokens)?
        } else if tokens.starts_with("r ") || tokens.starts_with("r#") {
            self.parse_raw_format_string(tokens)?
        } else if tokens.starts_with('"') {
            // Regular string
            self.parse_format_string(tokens)?
        } else {
            return Err(WasmError::parse("format! requires a string literal"));
        };

        // Parse arguments - separate named (name = value) from positional
        let raw_args = self.parse_macro_args(args_str)?;
        let mut named_args: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        let mut positional_args: Vec<String> = Vec::new();

        for arg in raw_args {
            // Check if this is a named argument (name = value)
            if let Some(eq_pos) = arg.find('=') {
                // Make sure it's not == or <= or >= etc
                let before_eq = arg[..eq_pos].chars().last();
                let after_eq = arg.get(eq_pos + 1..eq_pos + 2).and_then(|s| s.chars().next());
                if before_eq != Some('!') && before_eq != Some('<') && before_eq != Some('>')
                   && before_eq != Some('=') && after_eq != Some('=') {
                    let name = arg[..eq_pos].trim().to_string();
                    let value = arg[eq_pos + 1..].trim().to_string();
                    named_args.insert(name, value);
                    continue;
                }
            }
            positional_args.push(arg);
        }

        // Split format string by {} placeholders and interleave with args
        let parts = self.split_format_string(&format_str)?;

        // Generate code: concat all parts together
        // Start with first literal part (or empty string if starts with {})
        let first_literal = parts.literals.first().map(|s| s.as_str()).unwrap_or("");
        let offset = self.add_string(first_literal);
        let func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        func.push(Instruction::I32Const(offset as i32));
        func.push(Instruction::I64ExtendI32U);

        // Process each placeholder and its following literal
        let mut positional_idx = 0;
        for (i, spec) in parts.format_specs.iter().enumerate() {
            // Determine which argument to use for this placeholder
            let arg_expr = if spec.is_empty() || spec.starts_with(':') {
                // Positional placeholder {} or {:spec}
                let arg = positional_args.get(positional_idx)
                    .ok_or_else(|| WasmError::parse(&format!(
                        "format! has more positional placeholders than arguments"
                    )))?;
                positional_idx += 1;
                arg.clone()
            } else {
                // Named placeholder {name} or {name:spec}
                let name = spec.split(':').next().unwrap_or(spec).trim();
                // Check if it's actually a positional number like {0} or {1}
                if name.chars().all(|c| c.is_ascii_digit()) {
                    // Positional index like {0}, {1}
                    let idx: usize = name.parse().unwrap_or(0);
                    positional_args.get(idx)
                        .cloned()
                        .ok_or_else(|| WasmError::parse(&format!(
                            "format! argument index {} out of range", idx
                        )))?
                } else {
                    named_args.get(name)
                        .cloned()
                        .ok_or_else(|| WasmError::parse(&format!(
                            "format! missing named argument: {}", name
                        )))?
                }
            };

            // Convert argument to string
            self.compile_arg_to_string(&arg_expr, &Some(spec.as_str()))?;

            // Concat current result with argument string
            self.emit_string_concat()?;

            // If there's a literal after this placeholder, add it
            if i + 1 < parts.literals.len() {
                let literal = &parts.literals[i + 1];
                if !literal.is_empty() {
                    let offset = self.add_string(literal);
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::I32Const(offset as i32));
                    func.push(Instruction::I64ExtendI32U);
                    self.emit_string_concat()?;
                }
            }
        }

        Ok(())
    }

    /// Parse a raw format string (r#"..."# or r"...") and return (content, remaining_args)
    fn parse_raw_format_string<'a>(&self, tokens: &'a str) -> WasmResult<(String, &'a str)> {
        // Raw strings are tokenized with spaces: r # " content " # or r " content "
        let tokens = tokens.trim();

        // Skip 'r'
        let rest = if tokens.starts_with("r ") {
            &tokens[2..]
        } else if tokens.starts_with("r#") {
            &tokens[1..]
        } else {
            return Err(WasmError::parse("expected raw string"));
        };

        // Count # delimiters
        let hash_count = rest.chars().take_while(|c| *c == '#' || *c == ' ')
            .filter(|c| *c == '#')
            .count();

        // Find the opening quote
        let quote_start = rest.find('"')
            .ok_or_else(|| WasmError::parse("expected \" in raw string"))?;

        // Find the closing delimiter: " followed by hash_count # characters
        let content_start = quote_start + 1;
        let rest_after_quote = &rest[content_start..];

        // Build closing pattern: " # # ... (with possible spaces)
        let mut end_pos = None;
        let mut i = 0;
        while i < rest_after_quote.len() {
            // Safety: ensure we're at a character boundary
            if !rest_after_quote.is_char_boundary(i) {
                i += 1;
                continue;
            }
            if rest_after_quote[i..].starts_with('"') {
                // Check for matching # count after the quote
                let after_quote = &rest_after_quote[i + 1..];
                let mut hashes_found = 0;
                let mut j = 0;
                while j < after_quote.len() {
                    if !after_quote.is_char_boundary(j) {
                        j += 1;
                        continue;
                    }
                    let c = after_quote[j..].chars().next().unwrap_or(' ');
                    if c == '#' {
                        hashes_found += 1;
                    } else if c == ' ' {
                        // Skip spaces between # marks
                    } else {
                        break;
                    }
                    j += 1;
                }
                if hashes_found == hash_count {
                    end_pos = Some(i);
                    break;
                }
            }
            i += 1;
        }

        let content_end = end_pos.ok_or_else(|| WasmError::parse("unclosed raw string"))?;
        let content = rest_after_quote[..content_end].to_string();

        // Find where arguments start (after closing " # # ...)
        let total_consumed = quote_start + 1 + content_end + 1; // r + hashes + " + content + "
        // Skip closing hashes
        let mut args_start = total_consumed;
        while args_start < rest.len() {
            let c = rest.chars().nth(args_start).unwrap_or(' ');
            if c == '#' || c == ' ' {
                args_start += 1;
            } else {
                break;
            }
        }

        let args_str = if args_start < rest.len() {
            rest[args_start..].trim_start_matches(',').trim()
        } else {
            ""
        };

        Ok((content, args_str))
    }

    /// Parse a RawStringDelimited("content") format from the parser
    /// Returns (content, remaining_args_str)
    fn parse_raw_string_delimited<'a>(&self, tokens: &'a str) -> WasmResult<(String, &'a str)> {
        // Format: RawStringDelimited("content") , args...
        let prefix = "RawStringDelimited(";
        if !tokens.starts_with(prefix) {
            return Err(WasmError::parse("expected RawStringDelimited"));
        }

        // Find the inner string content
        let after_prefix = &tokens[prefix.len()..];
        if !after_prefix.starts_with('"') {
            return Err(WasmError::parse("expected \" after RawStringDelimited("));
        }

        // Find the closing ") - the inner string is quoted
        let mut in_escape = false;
        let mut content_end = 0;
        for (i, c) in after_prefix[1..].char_indices() {
            if in_escape {
                in_escape = false;
            } else if c == '\\' {
                in_escape = true;
            } else if c == '"' {
                content_end = i + 1;
                break;
            }
        }

        if content_end == 0 {
            return Err(WasmError::parse("unclosed string in RawStringDelimited"));
        }

        let content = after_prefix[1..content_end].to_string();

        // Skip past the closing )
        let after_content = &after_prefix[content_end + 1..];
        let rest = after_content.trim_start();
        let rest = if rest.starts_with(')') { &rest[1..] } else { rest };
        let args_str = rest.trim_start_matches(',').trim();

        Ok((content, args_str))
    }

    /// Parse a format string and return (format_str_content, remaining_args_str)
    fn parse_format_string<'a>(&self, tokens: &'a str) -> WasmResult<(String, &'a str)> {
        let mut in_escape = false;
        let mut format_end = 1;

        for (i, c) in tokens[1..].char_indices() {
            if in_escape {
                in_escape = false;
            } else if c == '\\' {
                in_escape = true;
            } else if c == '"' {
                format_end = i + 2;
                break;
            }
        }

        let format_str = tokens[1..format_end-1].to_string();
        let args_str = if format_end < tokens.len() {
            tokens[format_end..].trim_start_matches(',').trim()
        } else {
            ""
        };

        Ok((format_str, args_str))
    }

    /// Parse macro arguments, respecting parentheses/brackets/braces nesting.
    pub(crate) fn parse_macro_args(&self, args_str: &str) -> WasmResult<Vec<String>> {
        if args_str.is_empty() {
            return Ok(Vec::new());
        }

        let mut args = Vec::new();
        let mut depth = 0;
        let mut current_arg = String::new();

        for c in args_str.chars() {
            match c {
                '(' | '[' | '{' => {
                    depth += 1;
                    current_arg.push(c);
                }
                ')' | ']' | '}' => {
                    depth -= 1;
                    current_arg.push(c);
                }
                ',' if depth == 0 => {
                    let arg = current_arg.trim().to_string();
                    if !arg.is_empty() {
                        args.push(arg);
                    }
                    current_arg.clear();
                }
                _ => current_arg.push(c),
            }
        }

        // Don't forget last argument
        let arg = current_arg.trim().to_string();
        if !arg.is_empty() {
            args.push(arg);
        }

        Ok(args)
    }

    /// Split a format string into literal parts and count placeholders.
    fn split_format_string(&self, format_str: &str) -> WasmResult<FormatParts> {
        let mut literals = Vec::new();
        let mut format_specs = Vec::new();
        let mut current = String::new();
        let mut in_placeholder = false;
        let mut placeholder_content = String::new();
        let mut chars = format_str.chars().peekable();

        while let Some(c) = chars.next() {
            if in_placeholder {
                if c == '}' {
                    // End of placeholder — record the spec and return to literal
                    // accumulation mode.  Do NOT push `current` here; the text
                    // between this `}` and the next `{` belongs to the NEXT
                    // inter-placeholder literal and will be saved when that `{`
                    // is reached (or at end-of-string below).
                    in_placeholder = false;
                    format_specs.push(placeholder_content.clone());
                    placeholder_content.clear();
                } else {
                    placeholder_content.push(c);
                }
            } else if c == '{' {
                // Check for escaped {{
                if chars.peek() == Some(&'{') {
                    chars.next();
                    current.push('{');
                } else {
                    // Start of placeholder — always save the current literal
                    // (even when empty) so that literals[i] is the text that
                    // precedes format_specs[i] for every i.  The old guard
                    // `if literals.is_empty()` caused the pre-placeholder text
                    // to be skipped for the 2nd and later placeholders, and the
                    // stale `current` was pushed by the previous `}` instead,
                    // producing an off-by-one shift in the literals vector.
                    literals.push(current.clone());
                    current.clear();
                    in_placeholder = true;
                }
            } else if c == '}' {
                // Check for escaped }}
                if chars.peek() == Some(&'}') {
                    chars.next();
                    current.push('}');
                } else {
                    return Err(WasmError::parse("unmatched } in format string"));
                }
            } else {
                current.push(c);
            }
        }

        if in_placeholder {
            return Err(WasmError::parse("unclosed { in format string"));
        }

        // Always push the trailing literal (text after the last placeholder,
        // or the entire string when there are no placeholders).  This keeps
        // the invariant literals.len() == format_specs.len() + 1, which
        // compile_format_macro relies on.
        literals.push(current);

        let placeholders = format_specs.len();
        Ok(FormatParts {
            literals,
            format_specs,
            placeholders,
        })
    }

    /// Compile an argument expression and convert to string.
    fn compile_arg_to_string(&mut self, arg_expr: &str, format_spec: &Option<&str>) -> WasmResult<()> {
        // Parse and compile the argument expression
        let mut parser = Parser::new(arg_expr);
        let expr = parser.parse_expr()
            .map_err(|e| WasmError::parse(&format!("in format! argument: {}", e)))?;

        self.compile_expr(&expr)?;

        // Get string::from_int or string::from_float import index
        let from_int_idx = self.imports.get_func("string_from_int")
            .ok_or_else(|| WasmError::internal("string::from_int import not found"))?;
        let from_float_idx = self.imports.get_func("string_from_float")
            .ok_or_else(|| WasmError::internal("string::from_float import not found"))?;

        // Determine conversion based on format spec or default to int
        let is_float = format_spec
            .map(|s| s.contains('.') || s.contains('e') || s.contains('E'))
            .unwrap_or(false);

        let func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        if is_float {
            // Convert f64 bits (stored as i64) back to f64, then to string
            func.push(Instruction::F64ReinterpretI64);
            func.push(Instruction::Call(from_float_idx));
            // from_float returns i32, extend to i64 for uniform representation
            func.push(Instruction::I64ExtendI32U);
        } else {
            // Convert i64 to string
            func.push(Instruction::Call(from_int_idx));
            // from_int returns i32, extend to i64 for uniform representation
            func.push(Instruction::I64ExtendI32U);
        }

        Ok(())
    }

    /// Emit string concatenation: concat(a, b) -> result
    /// Assumes two strings are on the stack as i64 (uniform representation).
    /// Wraps them to i32 for the call, then extends result back to i64.
    fn emit_string_concat(&mut self) -> WasmResult<()> {
        use wasm_encoder::ValType;

        let concat_idx = self.imports.get_func("string_concat")
            .ok_or_else(|| WasmError::internal("string::concat import not found"))?;

        let func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Stack: [... str_a (i64), str_b (i64)]
        // Wrap str_b to i32 and store in temp
        func.push(Instruction::I32WrapI64);
        let str_b_local = func.alloc_local("__concat_b".to_string(), ValType::I32);
        func.push(Instruction::LocalSet(str_b_local));

        // Wrap str_a to i32
        func.push(Instruction::I32WrapI64);

        // Push str_b back
        func.push(Instruction::LocalGet(str_b_local));

        // Call concat(str_a: i32, str_b: i32) -> i32
        func.push(Instruction::Call(concat_idx));

        // Extend result to i64
        func.push(Instruction::I64ExtendI32U);

        Ok(())
    }

    /// Compile html! macro for VDOM generation.
    fn compile_html_macro(&mut self, tokens: &str) -> WasmResult<()> {
        let tokens = tokens.trim();

        // The macro tokens include the outer braces, strip them
        let tokens = if tokens.starts_with('{') && tokens.ends_with('}') {
            tokens[1..tokens.len()-1].trim()
        } else {
            tokens
        };

        // Normalize HTML tokens - the parser inserts spaces between operators
        // Remove spaces around angle brackets and slashes for HTML parsing
        let tokens = self.normalize_html_tokens(tokens);

        // Handle fragment: <></>
        if tokens == "<></>" || tokens.is_empty() {
            return self.compile_html_fragment();
        }

        // Parse HTML-like syntax
        let node = self.parse_html_tokens(&tokens)?;
        self.compile_html_node(&node)
    }

    /// Normalize HTML tokens by removing spurious spaces from tokenization.
    fn normalize_html_tokens(&self, tokens: &str) -> String {
        let mut result = String::new();
        let mut chars = tokens.chars().peekable();

        while let Some(c) = chars.next() {
            match c {
                '<' => {
                    result.push('<');
                    // Skip following spaces
                    while chars.peek() == Some(&' ') {
                        chars.next();
                    }
                }
                '/' => {
                    // Check if we're in a closing tag context
                    if result.ends_with('<') || result.ends_with(' ') {
                        // Trim trailing space before /
                        while result.ends_with(' ') {
                            result.pop();
                        }
                    }
                    result.push('/');
                    // Skip following spaces
                    while chars.peek() == Some(&' ') {
                        chars.next();
                    }
                }
                '>' => {
                    // Trim trailing space before >
                    while result.ends_with(' ') && !result.ends_with("= ") {
                        result.pop();
                    }
                    result.push('>');
                }
                ' ' => {
                    // Don't add space after < or /
                    if !result.ends_with('<') && !result.ends_with('/') {
                        // Don't add multiple spaces
                        if !result.ends_with(' ') {
                            result.push(' ');
                        }
                    }
                }
                _ => result.push(c),
            }
        }

        result
    }

    /// Compile an empty fragment.
    fn compile_html_fragment(&mut self) -> WasmResult<()> {
        let create_fragment_idx = self.imports.get_func("vdom_create_fragment")
            .ok_or_else(|| WasmError::internal("vdom::create_fragment import not found"))?;

        let func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        func.push(Instruction::Call(create_fragment_idx));

        Ok(())
    }

    /// Parse HTML-like tokens into a tree structure.
    fn parse_html_tokens(&self, tokens: &str) -> WasmResult<HtmlNode> {
        let tokens = tokens.trim();

        // Fragment
        if tokens.starts_with("<><") || tokens == "<></>" {
            return Ok(HtmlNode::Fragment { children: Vec::new() });
        }

        // Element: <tag ...>...</tag> or <tag ... />
        if tokens.starts_with('<') {
            return self.parse_html_element(tokens);
        }

        // Text node or expression
        if tokens.starts_with('{') && tokens.ends_with('}') {
            // Expression: {expr}
            let expr_str = &tokens[1..tokens.len()-1];
            return Ok(HtmlNode::Expression(expr_str.to_string()));
        }

        // Plain text
        Ok(HtmlNode::Text(tokens.to_string()))
    }

    /// Parse an HTML element.
    fn parse_html_element(&self, tokens: &str) -> WasmResult<HtmlNode> {
        // Very simplified parser - real implementation would be more robust
        let tokens = tokens.trim();

        if !tokens.starts_with('<') {
            return Err(WasmError::parse("expected < at start of element"));
        }

        // Handle fragment: <></>
        if tokens.starts_with("<>") {
            // Find </>
            if let Some(close_pos) = tokens.find("</>") {
                let children_str = &tokens[2..close_pos];
                let children = if children_str.trim().is_empty() {
                    Vec::new()
                } else {
                    self.parse_html_children(children_str)?
                };
                return Ok(HtmlNode::Fragment { children });
            }
            return Err(WasmError::parse("unclosed fragment, expected </>"));
        }

        // Find tag name
        let tag_end = tokens[1..].find(|c: char| c.is_whitespace() || c == '>' || c == '/')
            .map(|i| i + 1)
            .unwrap_or(tokens.len());
        let tag = tokens[1..tag_end].to_string();

        // Self-closing tag?
        if tokens.ends_with("/>") {
            let attrs_str = &tokens[tag_end..tokens.len()-2].trim();
            let attributes = self.parse_html_attributes(attrs_str)?;
            return Ok(HtmlNode::Element {
                tag,
                attributes,
                children: Vec::new(),
            });
        }

        // Find closing > of opening tag
        let open_end = tokens.find('>').ok_or_else(|| WasmError::parse("unclosed tag"))?;
        let attrs_str = &tokens[tag_end..open_end].trim();
        let attributes = self.parse_html_attributes(attrs_str)?;

        // Find closing tag
        let close_tag = format!("</{}>", tag);
        let close_start = tokens.rfind(&close_tag)
            .ok_or_else(|| WasmError::parse(&format!("missing closing tag: {}", close_tag)))?;

        // Parse children
        let children_str = &tokens[open_end+1..close_start];
        let children = self.parse_html_children(children_str)?;

        Ok(HtmlNode::Element {
            tag,
            attributes,
            children,
        })
    }

    /// Parse HTML attributes from a string.
    fn parse_html_attributes(&self, attrs_str: &str) -> WasmResult<Vec<HtmlAttribute>> {
        let mut attributes = Vec::new();
        let mut remaining = attrs_str.trim();

        while !remaining.is_empty() {
            // Find attribute name
            let name_end = remaining.find(|c: char| c == '=' || c.is_whitespace())
                .unwrap_or(remaining.len());

            if name_end == 0 {
                remaining = remaining.trim_start();
                continue;
            }

            let name = remaining[..name_end].to_string();
            remaining = remaining[name_end..].trim_start();

            // Check for =
            if remaining.starts_with('=') {
                remaining = remaining[1..].trim_start();

                // Parse value
                if remaining.starts_with('{') {
                    // Expression value
                    let depth_result = self.find_closing_brace(remaining);
                    let end = depth_result?;
                    let value = remaining[1..end].to_string();
                    attributes.push(HtmlAttribute {
                        name,
                        value: HtmlAttrValue::Expression(value),
                    });
                    remaining = remaining[end+1..].trim_start();
                } else if remaining.starts_with('"') {
                    // String value
                    let end = remaining[1..].find('"')
                        .map(|i| i + 1)
                        .ok_or_else(|| WasmError::parse("unclosed attribute string"))?;
                    let value = remaining[1..end].to_string();
                    attributes.push(HtmlAttribute {
                        name,
                        value: HtmlAttrValue::String(value),
                    });
                    remaining = remaining[end+1..].trim_start();
                } else {
                    // Unquoted value (take until whitespace)
                    let end = remaining.find(|c: char| c.is_whitespace()).unwrap_or(remaining.len());
                    let value = remaining[..end].to_string();
                    attributes.push(HtmlAttribute {
                        name,
                        value: HtmlAttrValue::String(value),
                    });
                    remaining = remaining[end..].trim_start();
                }
            } else {
                // Boolean attribute
                attributes.push(HtmlAttribute {
                    name,
                    value: HtmlAttrValue::Boolean,
                });
            }
        }

        Ok(attributes)
    }

    /// Find closing brace, respecting nesting.
    fn find_closing_brace(&self, s: &str) -> WasmResult<usize> {
        let mut depth = 0;
        for (i, c) in s.char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(i);
                    }
                }
                _ => {}
            }
        }
        Err(WasmError::parse("unclosed brace in attribute"))
    }

    /// Parse HTML children from content string.
    fn parse_html_children(&self, content: &str) -> WasmResult<Vec<HtmlNode>> {
        let content = content.trim();
        if content.is_empty() {
            return Ok(Vec::new());
        }

        let mut children = Vec::new();
        let mut remaining = content;

        while !remaining.is_empty() {
            remaining = remaining.trim_start();
            if remaining.is_empty() {
                break;
            }

            if remaining.starts_with('<') {
                // Element or fragment
                if remaining.starts_with("</") {
                    // Closing tag - shouldn't happen here
                    break;
                }

                // Find end of this element
                let end = self.find_element_end(remaining)?;
                let element_str = &remaining[..end];
                children.push(self.parse_html_tokens(element_str)?);
                remaining = &remaining[end..];
            } else if remaining.starts_with('{') {
                // Expression
                let end = self.find_closing_brace(remaining)?;
                let expr_str = &remaining[1..end];
                children.push(HtmlNode::Expression(expr_str.to_string()));
                remaining = &remaining[end+1..];
            } else {
                // Text - take until < or {
                let end = remaining.find(|c: char| c == '<' || c == '{')
                    .unwrap_or(remaining.len());
                let text = remaining[..end].trim();
                if !text.is_empty() {
                    children.push(HtmlNode::Text(text.to_string()));
                }
                remaining = &remaining[end..];
            }
        }

        Ok(children)
    }

    /// Find the end of an HTML element.
    fn find_element_end(&self, s: &str) -> WasmResult<usize> {
        // Handle fragment: <>...</>
        if s.starts_with("<>") {
            let mut depth = 0;
            let mut i = 0;
            while i < s.len() {
                // Safety: ensure we're at a character boundary
                if !s.is_char_boundary(i) {
                    i += 1;
                    continue;
                }
                if s[i..].starts_with("<>") {
                    depth += 1;
                    i += 2;
                } else if s[i..].starts_with("</>") {
                    depth -= 1;
                    if depth == 0 {
                        return Ok(i + 3);
                    }
                    i += 3;
                } else {
                    i += 1;
                }
            }
            return Err(WasmError::parse("unclosed fragment: <>"));
        }

        // Self-closing?
        if let Some(end) = s.find("/>") {
            // Make sure it's not inside a string
            let before = &s[..end];
            if before.matches('"').count() % 2 == 0 {
                return Ok(end + 2);
            }
        }

        // Find tag name
        let tag_end = s[1..].find(|c: char| c.is_whitespace() || c == '>' || c == '/')
            .map(|i| i + 1)
            .ok_or_else(|| WasmError::parse("invalid tag"))?;
        let tag = &s[1..tag_end];

        // Empty tag check (shouldn't happen after fragment handling)
        if tag.is_empty() {
            return Err(WasmError::parse("empty tag name"));
        }

        // Find matching closing tag
        let close_tag = format!("</{}>", tag);
        let mut depth = 0;
        let open_tag_start = format!("<{}", tag);

        let mut i = 0;
        while i < s.len() {
            // Safety: ensure we're at a character boundary
            if !s.is_char_boundary(i) {
                i += 1;
                continue;
            }
            if s[i..].starts_with(&open_tag_start) {
                // Check if it's actually an opening tag (followed by space, >, or /)
                let next_char_pos = i + open_tag_start.len();
                if next_char_pos < s.len() && s.is_char_boundary(next_char_pos) {
                    let next_char = s[next_char_pos..].chars().next().unwrap_or(' ');
                    if next_char.is_whitespace() || next_char == '>' || next_char == '/' {
                        depth += 1;
                    }
                }
            } else if s[i..].starts_with(&close_tag) {
                depth -= 1;
                if depth == 0 {
                    return Ok(i + close_tag.len());
                }
            }
            i += 1;
        }

        Err(WasmError::parse(&format!("unclosed element: <{}>", tag)))
    }

    /// Compile an HTML node to VDOM calls.
    fn compile_html_node(&mut self, node: &HtmlNode) -> WasmResult<()> {
        match node {
            HtmlNode::Fragment { children } => {
                self.compile_html_fragment()?;

                // Extend i32 result to i64 for uniform storage
                {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::I64ExtendI32U);
                }

                let fragment_local = self.allocate_temp_local()?;

                // Store the fragment (now i64)
                {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::LocalSet(fragment_local));
                }

                // Append children to fragment
                for child in children {
                    {
                        let func = self.current_function_mut().unwrap();
                        func.push(Instruction::LocalGet(fragment_local));
                        func.push(Instruction::I32WrapI64);  // wrap to i32 for append
                    }
                    self.compile_html_node(child)?;
                    // Child result is i32 from create_vnode/create_text_vnode
                    self.emit_append_vnode_child()?;
                }

                // Return fragment as i32
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::LocalGet(fragment_local));
                func.push(Instruction::I32WrapI64);
                Ok(())
            }

            HtmlNode::Element { tag, attributes, children } => {
                // Create vnode
                let tag_offset = self.add_string(tag);
                let create_vnode_idx = self.imports.get_func("vdom_create_vnode")
                    .ok_or_else(|| WasmError::internal("vdom::create_vnode import not found"))?;

                {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::I64Const(tag_offset as i64));  // string ref as i64
                    func.push(Instruction::Call(create_vnode_idx));
                    // Extend i32 result to i64 for uniform storage
                    func.push(Instruction::I64ExtendI32U);
                }

                // Save vnode to local (i64)
                let vnode_local = self.allocate_temp_local()?;
                {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::LocalSet(vnode_local));
                }

                // Set attributes (import expects vnode as i32)
                for attr in attributes {
                    {
                        let func = self.current_function_mut().unwrap();
                        func.push(Instruction::LocalGet(vnode_local));
                        func.push(Instruction::I32WrapI64);  // wrap to i32 for set_vnode_prop
                    }
                    self.compile_html_attribute(attr)?;
                }

                // Append children
                for child in children {
                    {
                        let func = self.current_function_mut().unwrap();
                        func.push(Instruction::LocalGet(vnode_local));
                        func.push(Instruction::I32WrapI64);  // wrap parent to i32
                    }
                    self.compile_html_node(child)?;
                    // Child result is i32 from create_vnode/create_text_vnode
                    self.emit_append_vnode_child()?;
                }

                // Return vnode as i32 (for html! the result is used directly)
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::LocalGet(vnode_local));
                func.push(Instruction::I32WrapI64);
                Ok(())
            }

            HtmlNode::Text(text) => {
                let text_offset = self.add_string(text);
                let create_text_vnode_idx = self.imports.get_func("vdom_create_text_vnode")
                    .ok_or_else(|| WasmError::internal("vdom::create_text_vnode import not found"))?;

                let func = self.current_function_mut().unwrap();
                func.push(Instruction::I64Const(text_offset as i64));  // string ref as i64
                func.push(Instruction::Call(create_text_vnode_idx));
                // Result is i32, leave as-is for append_vnode_child
                Ok(())
            }

            HtmlNode::Expression(expr_str) => {
                // Check for conditional: if cond { html!{...} } else { html!{...} }
                if expr_str.trim().starts_with("if ") {
                    return self.compile_html_conditional(expr_str);
                }

                // Regular expression - compile and assume it returns a vnode
                let mut parser = Parser::new(expr_str);
                let expr = parser.parse_expr()
                    .map_err(|e| WasmError::parse(&format!("in html! expression: {}", e)))?;
                self.compile_expr(&expr)
            }
        }
    }

    /// Compile an HTML attribute.
    /// Stack on entry: [i32_vnode]
    /// Import signatures:
    ///   set_vnode_str_prop(vnodeId: i32, nameStrRef: i64, valueStrRef: i64)
    ///   set_vnode_prop(vnodeId: i32, nameStrRef: i64, value: i64)
    fn compile_html_attribute(&mut self, attr: &HtmlAttribute) -> WasmResult<()> {
        let name_offset = self.add_string(&attr.name);

        match &attr.value {
            HtmlAttrValue::String(s) => {
                let value_offset = self.add_string(s);
                let set_str_prop_idx = self.imports.get_func("vdom_set_vnode_str_prop")
                    .ok_or_else(|| WasmError::internal("vdom::set_vnode_str_prop import not found"))?;

                let func = self.current_function_mut().unwrap();
                // vnode i32 already on stack
                func.push(Instruction::I64Const(name_offset as i64));   // name as i64
                func.push(Instruction::I64Const(value_offset as i64));  // value as i64
                func.push(Instruction::Call(set_str_prop_idx));
            }
            HtmlAttrValue::Expression(expr_str) => {
                let set_prop_idx = self.imports.get_func("vdom_set_vnode_prop")
                    .ok_or_else(|| WasmError::internal("vdom::set_vnode_prop import not found"))?;

                let func = self.current_function_mut().unwrap();
                // vnode i32 already on stack
                func.push(Instruction::I64Const(name_offset as i64));  // name as i64
                drop(func);

                // Compile expression (produces i64)
                let mut parser = Parser::new(expr_str);
                let expr = parser.parse_expr()
                    .map_err(|e| WasmError::parse(&format!("in html! attribute: {}", e)))?;
                self.compile_expr(&expr)?;

                let func = self.current_function_mut().unwrap();
                func.push(Instruction::Call(set_prop_idx));
            }
            HtmlAttrValue::Boolean => {
                let set_prop_idx = self.imports.get_func("vdom_set_vnode_prop")
                    .ok_or_else(|| WasmError::internal("vdom::set_vnode_prop import not found"))?;

                let func = self.current_function_mut().unwrap();
                // vnode i32 already on stack
                func.push(Instruction::I64Const(name_offset as i64));  // name as i64
                func.push(Instruction::I64Const(1)); // true as i64
                func.push(Instruction::Call(set_prop_idx));
            }
        }

        Ok(())
    }

    /// Compile conditional in html!: {if cond { html!{a} } else { html!{b} }}
    fn compile_html_conditional(&mut self, expr_str: &str) -> WasmResult<()> {
        // Parse as regular if expression
        let mut parser = Parser::new(expr_str);
        let expr = parser.parse_expr()
            .map_err(|e| WasmError::parse(&format!("in html! conditional: {}", e)))?;
        self.compile_expr(&expr)
    }

    /// Emit vdom::append_vnode_child call.
    fn emit_append_vnode_child(&mut self) -> WasmResult<()> {
        let append_idx = self.imports.get_func("vdom_append_vnode_child")
            .ok_or_else(|| WasmError::internal("vdom::append_vnode_child import not found"))?;

        let func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        func.push(Instruction::Call(append_idx));

        Ok(())
    }

    /// Allocate a temporary local variable.
    fn allocate_temp_local(&mut self) -> WasmResult<u32> {
        let func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Add an i64 local for temporary storage
        let local_idx = func.params.len() as u32 + func.local_types.len() as u32;
        func.local_types.push(wasm_encoder::ValType::I64);

        Ok(local_idx)
    }

    /// Compile vec! macro.
    fn compile_vec_macro(&mut self, tokens: &str) -> WasmResult<()> {
        let args = self.parse_macro_args(tokens)?;

        // Create new array
        let array_new_idx = self.imports.get_func("morpheme_array_new")
            .ok_or_else(|| WasmError::internal("morpheme::array_new import not found"))?;
        let array_push_idx = self.imports.get_func("morpheme_array_push")
            .ok_or_else(|| WasmError::internal("morpheme::array_push import not found"))?;

        let func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        func.push(Instruction::Call(array_new_idx));
        // array_new returns i32, extend to i64 for Sigil's uniform type system
        func.push(Instruction::I64ExtendI32U);

        // Push each element
        for arg in args {
            // Save array reference (i64)
            let array_local = self.allocate_temp_local()?;
            let func = self.current_function_mut().unwrap();
            func.push(Instruction::LocalSet(array_local));
            // Push array ref wrapped to i32 for array_push
            func.push(Instruction::LocalGet(array_local));
            func.push(Instruction::I32WrapI64);
            drop(func);

            // Compile element (leaves i64 on stack)
            let mut parser = Parser::new(&arg);
            let expr = parser.parse_expr()
                .map_err(|e| WasmError::parse(&format!("in vec! element: {}", e)))?;
            self.compile_expr(&expr)?;

            // Push to array: array_push(i32 arr, i64 elem)
            let func = self.current_function_mut().unwrap();
            func.push(Instruction::Call(array_push_idx));
            // Leave array ref (i64) on stack for next iteration or return
            func.push(Instruction::LocalGet(array_local));
        }

        Ok(())
    }

    /// Compile concat! macro.
    fn compile_concat_macro(&mut self, tokens: &str) -> WasmResult<()> {
        let args = self.parse_macro_args(tokens)?;

        if args.is_empty() {
            let offset = self.add_string("");
            let func = self.current_function_mut()
                .ok_or_else(|| WasmError::internal("not in function context"))?;
            func.push(Instruction::I32Const(offset as i32));
            func.push(Instruction::I64ExtendI32U);
            return Ok(());
        }

        // Concatenate string literals at compile time
        let mut result = String::new();
        for arg in args {
            let arg = arg.trim();
            if arg.starts_with('"') && arg.ends_with('"') {
                result.push_str(&arg[1..arg.len()-1]);
            } else {
                // Non-literal - can't concatenate at compile time
                return Err(WasmError::unsupported("concat! with non-literal arguments"));
            }
        }

        let offset = self.add_string(&result);
        let func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        func.push(Instruction::I32Const(offset as i32));
        func.push(Instruction::I64ExtendI32U);

        Ok(())
    }

    /// Compile stringify! macro.
    fn compile_stringify_macro(&mut self, tokens: &str) -> WasmResult<()> {
        // stringify! just returns the tokens as a string literal
        let offset = self.add_string(tokens.trim());
        let func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        func.push(Instruction::I32Const(offset as i32));
        func.push(Instruction::I64ExtendI32U);

        Ok(())
    }

    /// Compile console_log!/console_warn!/console_error! macros.
    /// These compile similarly to format! but call the appropriate console import.
    fn compile_console_log_macro(&mut self, macro_name: &str, tokens: &str) -> WasmResult<()> {
        let tokens = tokens.trim();

        // Determine which import to use based on macro name
        let import_name = match macro_name {
            "console_warn" => "console_warn",
            "console_error" => "console_error",
            _ => "console_log",
        };

        // If empty, log empty string
        if tokens.is_empty() {
            let offset = self.add_string("");
            let func = self.current_function_mut()
                .ok_or_else(|| WasmError::internal("not in function context"))?;
            func.push(Instruction::I32Const(offset as i32));
            func.push(Instruction::I64ExtendI32U);
        } else {
            // Use format! compilation to build the string
            self.compile_format_macro(tokens)?;
        }

        // Call the appropriate console function
        let console_fn_idx = self.imports.get_func(import_name)
            .ok_or_else(|| WasmError::internal(&format!("{} import not found", import_name)))?;

        let func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        func.push(Instruction::Call(console_fn_idx));

        // Console functions return void, push unit value for expression result
        func.push(Instruction::I64Const(0));

        Ok(())
    }

    /// Compile print!/println!/eprint!/eprintln! macros.
    /// These are aliases for console_log (browser environment).
    fn compile_print_macro(&mut self, macro_name: &str, tokens: &str) -> WasmResult<()> {
        // In WASM/browser, all print variants use console.log
        // The 'e' variants (eprint/eprintln) could use console.error
        let console_type = if macro_name.starts_with('e') {
            "console_error"
        } else {
            "console_log"
        };
        self.compile_console_log_macro(console_type, tokens)
    }

    /// Compile matches!(expr, pattern) macro.
    /// Expands to: match expr { pattern => true, _ => false }
    fn compile_matches_macro(&mut self, tokens: &str) -> WasmResult<()> {
        let tokens = tokens.trim();

        // Parse: expr, pattern (possibly with guard)
        // Simple parsing: find first comma at depth 0
        let args = self.parse_macro_args(tokens)?;
        if args.len() < 2 {
            return Err(WasmError::parse("matches! requires at least 2 arguments: expr, pattern"));
        }

        let expr_str = &args[0];
        let pattern_str = &args[1];

        // Parse the expression
        let mut parser = Parser::new(expr_str);
        let expr = parser.parse_expr()
            .map_err(|e| WasmError::parse(&format!("in matches! expression: {}", e)))?;

        // For enum variant matching like `ConnectionState::Connected`:
        // Check if expression's enum discriminant matches the pattern's discriminant
        // This is a simplified implementation that works for simple enum variants

        // Compile the expression (this puts the enum value on the stack)
        self.compile_expr(&expr)?;

        // Extract the enum discriminant from the value
        // For tagged enums, discriminant is typically stored in the low bits
        // We'll use a simple approach: extract discriminant and compare to expected value

        // Parse the pattern to get the expected discriminant
        let discriminant = self.get_enum_discriminant_from_pattern(pattern_str)?;

        // Get discriminant from the value on stack
        // Enum values are stored as: (tag << 32) | data or similar
        // For now, assume the value itself IS the discriminant for unit variants
        let func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;

        // Compare: value == expected_discriminant
        func.push(Instruction::I64Const(discriminant));
        func.push(Instruction::I64Eq);

        // Result is already on stack as i32 (0 or 1)
        // Extend to i64 for uniform value representation
        func.push(Instruction::I64ExtendI32U);

        Ok(())
    }

    /// Extract the expected discriminant value from a pattern string.
    /// For patterns like "ConnectionState::Connected", returns the variant index.
    fn get_enum_discriminant_from_pattern(&self, pattern: &str) -> WasmResult<i64> {
        let pattern = pattern.trim();

        // Parse as path: SomeEnum::Variant
        if let Some(variant_pos) = pattern.rfind("::") {
            let enum_name = &pattern[..variant_pos];
            let variant_name = &pattern[variant_pos + 2..];

            // Look up the enum layout
            if let Some(layout) = self.enum_layouts.get(enum_name) {
                // Find the variant index
                for (idx, (name, _, _)) in layout.variants.iter().enumerate() {
                    if name == variant_name {
                        return Ok(idx as i64);
                    }
                }
                return Err(WasmError::parse(&format!(
                    "unknown variant '{}' in enum '{}'", variant_name, enum_name
                )));
            }

            // Enum not found in layouts - might be referenced by short name
            // Try looking up just the last segment of the enum path
            let short_enum_name = enum_name.rsplit("::").next().unwrap_or(enum_name);
            if let Some(layout) = self.enum_layouts.get(short_enum_name) {
                for (idx, (name, _, _)) in layout.variants.iter().enumerate() {
                    if name == variant_name {
                        return Ok(idx as i64);
                    }
                }
            }

            // If enum not found, return 0 as default (Connected is typically first)
            // This is a fallback for when enum definitions aren't available
            // In a full implementation, we'd track all enum definitions
            return Ok(0);
        }

        // Simple pattern (not a path) - assume it's a bool or number
        if pattern == "true" {
            return Ok(1);
        } else if pattern == "false" {
            return Ok(0);
        } else if let Ok(n) = pattern.parse::<i64>() {
            return Ok(n);
        }

        // Unknown pattern - return 0 as fallback
        Ok(0)
    }
}

/// Format string parsing result.
struct FormatParts {
    /// Literal string parts between placeholders.
    literals: Vec<String>,
    /// Format specifiers for each placeholder (contents of {}).
    format_specs: Vec<String>,
    /// Number of placeholders.
    placeholders: usize,
}

/// HTML node representation.
#[derive(Debug)]
enum HtmlNode {
    Element {
        tag: String,
        attributes: Vec<HtmlAttribute>,
        children: Vec<HtmlNode>,
    },
    Fragment {
        children: Vec<HtmlNode>,
    },
    Text(String),
    Expression(String),
}

/// HTML attribute.
#[derive(Debug)]
struct HtmlAttribute {
    name: String,
    value: HtmlAttrValue,
}

/// HTML attribute value.
#[derive(Debug)]
enum HtmlAttrValue {
    String(String),
    Expression(String),
    Boolean,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wasm::literals::create_test_compiler_with_function;

    #[test]
    fn test_parse_format_string_simple() {
        let compiler = WasmCompiler::new();
        let (format_str, args) = compiler.parse_format_string("\"hello {}\"").unwrap();
        assert_eq!(format_str, "hello {}");
        assert_eq!(args, "");
    }

    #[test]
    fn test_parse_format_string_with_args() {
        let compiler = WasmCompiler::new();
        let (format_str, args) = compiler.parse_format_string("\"hello {}\", world").unwrap();
        assert_eq!(format_str, "hello {}");
        assert_eq!(args, "world");
    }

    #[test]
    fn test_split_format_string() {
        let compiler = WasmCompiler::new();
        let parts = compiler.split_format_string("hello {} world").unwrap();
        assert_eq!(parts.literals.len(), 2);
        assert_eq!(parts.literals[0], "hello ");
        assert_eq!(parts.literals[1], " world");
        assert_eq!(parts.placeholders, 1);
    }

    #[test]
    fn test_parse_macro_args() {
        let compiler = WasmCompiler::new();
        let args = compiler.parse_macro_args("a, b, c").unwrap();
        assert_eq!(args, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_parse_macro_args_nested() {
        let compiler = WasmCompiler::new();
        let args = compiler.parse_macro_args("foo(a, b), bar").unwrap();
        assert_eq!(args, vec!["foo(a, b)", "bar"]);
    }
}
