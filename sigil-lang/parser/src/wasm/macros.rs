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

        // Parse the format string (first quoted string)
        if !tokens.starts_with('"') {
            return Err(WasmError::parse("format! requires a string literal"));
        }

        // Find end of format string
        let (format_str, args_str) = self.parse_format_string(tokens)?;

        // Parse arguments
        let args = self.parse_macro_args(args_str)?;

        // Split format string by {} placeholders and interleave with args
        let parts = self.split_format_string(&format_str)?;

        if parts.placeholders > args.len() {
            return Err(WasmError::parse(&format!(
                "format! has {} placeholders but only {} arguments",
                parts.placeholders, args.len()
            )));
        }

        // Generate code: concat all parts together
        // Start with first literal part (or empty string if starts with {})
        let first_literal = parts.literals.first().map(|s| s.as_str()).unwrap_or("");
        let offset = self.add_string(first_literal);
        let func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        func.push(Instruction::I32Const(offset as i32));
        func.push(Instruction::I64ExtendI32U);

        // Process each argument and its following literal
        for (i, arg_expr) in args.iter().enumerate() {
            // Convert argument to string
            self.compile_arg_to_string(arg_expr, &parts.format_specs.get(i).map(|s| s.as_str()))?;

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
                    // End of placeholder
                    in_placeholder = false;
                    format_specs.push(placeholder_content.clone());
                    placeholder_content.clear();
                    literals.push(current.clone());
                    current.clear();
                } else {
                    placeholder_content.push(c);
                }
            } else if c == '{' {
                // Check for escaped {{
                if chars.peek() == Some(&'{') {
                    chars.next();
                    current.push('{');
                } else {
                    // Start of placeholder - save current literal first
                    if literals.is_empty() {
                        literals.push(current.clone());
                        current.clear();
                    }
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

        // Push any remaining literal
        if !current.is_empty() || literals.is_empty() {
            literals.push(current);
        }

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
        } else {
            // Convert i64 to string
            func.push(Instruction::Call(from_int_idx));
        }

        Ok(())
    }

    /// Emit string concatenation: concat(a, b) -> result
    /// Assumes two strings are on the stack.
    fn emit_string_concat(&mut self) -> WasmResult<()> {
        let concat_idx = self.imports.get_func("string_concat")
            .ok_or_else(|| WasmError::internal("string::concat import not found"))?;

        let func = self.current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        func.push(Instruction::Call(concat_idx));

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
            if s[i..].starts_with(&open_tag_start) {
                // Check if it's actually an opening tag (followed by space, >, or /)
                let next_char_pos = i + open_tag_start.len();
                if next_char_pos < s.len() {
                    let next_char = s.chars().nth(next_char_pos).unwrap_or(' ');
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
                let fragment_local = self.allocate_temp_local()?;

                // Store the fragment
                {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::LocalSet(fragment_local));
                }

                // Append children to fragment
                for child in children {
                    {
                        let func = self.current_function_mut().unwrap();
                        func.push(Instruction::LocalGet(fragment_local));
                    }
                    self.compile_html_node(child)?;
                    self.emit_append_vnode_child()?;
                }

                // Return fragment
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::LocalGet(fragment_local));
                Ok(())
            }

            HtmlNode::Element { tag, attributes, children } => {
                // Create vnode
                let tag_offset = self.add_string(tag);
                let create_vnode_idx = self.imports.get_func("vdom_create_vnode")
                    .ok_or_else(|| WasmError::internal("vdom::create_vnode import not found"))?;

                {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::I32Const(tag_offset as i32));
                    func.push(Instruction::I64ExtendI32U);
                    func.push(Instruction::Call(create_vnode_idx));
                }

                // Save vnode to local
                let vnode_local = self.allocate_temp_local()?;
                {
                    let func = self.current_function_mut().unwrap();
                    func.push(Instruction::LocalSet(vnode_local));
                }

                // Set attributes
                for attr in attributes {
                    {
                        let func = self.current_function_mut().unwrap();
                        func.push(Instruction::LocalGet(vnode_local));
                    }
                    self.compile_html_attribute(attr)?;
                }

                // Append children
                for child in children {
                    {
                        let func = self.current_function_mut().unwrap();
                        func.push(Instruction::LocalGet(vnode_local));
                    }
                    self.compile_html_node(child)?;
                    self.emit_append_vnode_child()?;
                }

                // Return vnode
                let func = self.current_function_mut().unwrap();
                func.push(Instruction::LocalGet(vnode_local));
                Ok(())
            }

            HtmlNode::Text(text) => {
                let text_offset = self.add_string(text);
                let create_text_vnode_idx = self.imports.get_func("vdom_create_text_vnode")
                    .ok_or_else(|| WasmError::internal("vdom::create_text_vnode import not found"))?;

                let func = self.current_function_mut().unwrap();
                func.push(Instruction::I32Const(text_offset as i32));
                func.push(Instruction::I64ExtendI32U);
                func.push(Instruction::Call(create_text_vnode_idx));
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
    fn compile_html_attribute(&mut self, attr: &HtmlAttribute) -> WasmResult<()> {
        let name_offset = self.add_string(&attr.name);

        match &attr.value {
            HtmlAttrValue::String(s) => {
                let value_offset = self.add_string(s);
                let set_str_prop_idx = self.imports.get_func("vdom_set_vnode_str_prop")
                    .ok_or_else(|| WasmError::internal("vdom::set_vnode_str_prop import not found"))?;

                let func = self.current_function_mut().unwrap();
                func.push(Instruction::I32Const(name_offset as i32));
                func.push(Instruction::I64ExtendI32U);
                func.push(Instruction::I32Const(value_offset as i32));
                func.push(Instruction::I64ExtendI32U);
                func.push(Instruction::Call(set_str_prop_idx));
            }
            HtmlAttrValue::Expression(expr_str) => {
                let set_prop_idx = self.imports.get_func("vdom_set_vnode_prop")
                    .ok_or_else(|| WasmError::internal("vdom::set_vnode_prop import not found"))?;

                let func = self.current_function_mut().unwrap();
                func.push(Instruction::I32Const(name_offset as i32));
                func.push(Instruction::I64ExtendI32U);
                drop(func);

                // Compile expression
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
                func.push(Instruction::I32Const(name_offset as i32));
                func.push(Instruction::I64ExtendI32U);
                func.push(Instruction::I64Const(1)); // true
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

        // Push each element
        for arg in args {
            // Save array reference
            let array_local = self.allocate_temp_local()?;
            let func = self.current_function_mut().unwrap();
            func.push(Instruction::LocalSet(array_local));
            func.push(Instruction::LocalGet(array_local));
            drop(func);

            // Compile element
            let mut parser = Parser::new(&arg);
            let expr = parser.parse_expr()
                .map_err(|e| WasmError::parse(&format!("in vec! element: {}", e)))?;
            self.compile_expr(&expr)?;

            // Push to array
            let func = self.current_function_mut().unwrap();
            func.push(Instruction::Call(array_push_idx));
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
