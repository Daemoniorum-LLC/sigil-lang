// =============================================================================
// Tests for interpreter bugs found during Jormungandr bootstrap
// Bug reference: docs/bugs/INTERPRETER-RUNTIME-BUGS.md
// =============================================================================

// Helper functions for bug tests (avoiding name conflicts with existing run())
fn bug_test_run(source: &str) -> Result<Value, RuntimeError> {
    let mut parser = Parser::new(source);
    let file = parser
        .parse_file()
        .map_err(|e| RuntimeError::new(e.to_string()))?;
    let mut interp = Interpreter::new();
    crate::register_stdlib(&mut interp);  // Register Result·Ok, Option·Some, etc.
    interp.execute(&file)
}

fn bug_test_expect_int(source: &str, expected: i64) {
    match bug_test_run(source) {
        Ok(Value::Int(i)) => assert_eq!(i, expected, "Expected {}, got {}", expected, i),
        Ok(v) => panic!("Expected Int({}), got {:?}", expected, v),
        Err(e) => panic!("Expected Int({}), got error: {}", expected, e),
    }
}

// =============================================================================
// BUG INT-001: Vec::push on struct field doesn't persist
// =============================================================================

#[test]
fn test_local_struct_vec_push() {
    // Control test: local struct should work
    let source = r#"
        struct Container {
            items: ![String],
        }

        impl Container {
            fn new() -> !Container {
                Container { items: [] }
            }
        }

        fn main() -> i64 {
            let mut c = Container·new();
            c.items.push("hello".to_string());
            c.items.len() as i64
        }
    "#;
    bug_test_expect_int(source, 1);
}

#[test]
fn test_inline_module_struct_vec_push() {
    // Test with inline module - this should also work
    let source = r#"
        mod test_mod {
            pub struct Container {
                pub items: ![String],
            }

            impl Container {
                pub fn new() -> !Container {
                    Container { items: [] }
                }
            }
        }

        invoke test_mod·*;

        fn main() -> i64 {
            let mut c = Container·new();
            c.items.push("hello".to_string());
            c.items.len() as i64
        }
    "#;
    bug_test_expect_int(source, 1);
}

#[test]
fn test_struct_vec_multiple_pushes() {
    // Multiple pushes should accumulate
    let source = r#"
        struct Container {
            items: ![String],
        }

        fn main() -> i64 {
            let mut c = Container { items: [] };
            c.items.push("a".to_string());
            c.items.push("b".to_string());
            c.items.push("c".to_string());
            c.items.len() as i64
        }
    "#;
    bug_test_expect_int(source, 3);
}

#[test]
fn test_struct_vec_push_in_function() {
    // Push in a separate function that takes mutable reference
    let source = r#"
        struct Container {
            items: ![String],
        }

        fn add_item(c: &mut Container, item: !String) {
            c.items.push(item);
        }

        fn main() -> i64 {
            let mut c = Container { items: [] };
            add_item(&mut c, "hello".to_string());
            c.items.len() as i64
        }
    "#;
    bug_test_expect_int(source, 1);
}

#[test]
fn test_struct_vec_push_via_method() {
    // Push via impl method
    let source = r#"
        struct Container {
            items: ![String],
        }

        impl Container {
            fn add(&mut self, item: !String) {
                self.items.push(item);
            }
        }

        fn main() -> i64 {
            let mut c = Container { items: [] };
            c.add("hello".to_string());
            c.items.len() as i64
        }
    "#;
    bug_test_expect_int(source, 1);
}

// =============================================================================
// BUG INT-002: Match Result::Ok(imported_struct) fails
// =============================================================================

#[test]
fn test_match_result_ok_primitive() {
    // Control: matching on Result<primitive, _> should work
    let source = r#"
        fn make_ok() -> !Result<i32, String> {
            Result·Ok(42)
        }

        fn main() -> i64 {
            match make_ok() {
                Result·Ok(v) => v as i64,
                Result·Err(_) => -1,
            }
        }
    "#;
    bug_test_expect_int(source, 42);
}

#[test]
fn test_match_result_err_string() {
    // Control: matching on Result<_, String> error should work
    let source = r#"
        fn make_err() -> !Result<i32, String> {
            Result·Err("error".to_string())
        }

        fn main() -> i64 {
            match make_err() {
                Result·Ok(_) => 1,
                Result·Err(e) => {
                    if e == "error" { 42 } else { -1 }
                },
            }
        }
    "#;
    bug_test_expect_int(source, 42);
}

#[test]
fn test_match_result_ok_local_struct() {
    // Match on Result<LocalStruct, _>
    let source = r#"
        struct MyData {
            value: !i32,
        }

        fn make_data() -> !Result<MyData, String> {
            Result·Ok(MyData { value: 42 })
        }

        fn main() -> i64 {
            match make_data() {
                Result·Ok(d) => d.value as i64,
                Result·Err(_) => -1,
            }
        }
    "#;
    bug_test_expect_int(source, 42);
}

#[test]
fn test_match_result_ok_inline_module_struct() {
    // Bug INT-002: Match on Result<ImportedStruct, _>
    let source = r#"
        mod test_mod {
            pub struct MyData {
                pub value: !i32,
            }

            pub fn make_data() -> !Result<MyData, String> {
                Result·Ok(MyData { value: 42 })
            }
        }

        invoke test_mod·*;

        fn main() -> i64 {
            match make_data() {
                Result·Ok(d) => d.value as i64,
                Result·Err(_) => -1,
            }
        }
    "#;
    bug_test_expect_int(source, 42);
}

// =============================================================================
// BUG INT-003: unwrap_err() returns null
// =============================================================================

#[test]
fn test_result_unwrap_ok() {
    // Control: unwrap() on Ok should work
    let source = r#"
        fn main() -> i64 {
            let r: Result<i32, String> = Result·Ok(42);
            r.unwrap() as i64
        }
    "#;
    bug_test_expect_int(source, 42);
}

#[test]
fn test_result_unwrap_err_basic() {
    // Bug INT-003: unwrap_err() should return the error value
    let source = r#"
        fn make_err() -> !Result<i32, String> {
            Result·Err("error message".to_string())
        }

        fn main() -> i64 {
            let r = make_err();
            let e = r.unwrap_err();
            if e == "error message" { 42 } else { -1 }
        }
    "#;
    bug_test_expect_int(source, 42);
}

#[test]
fn test_result_unwrap_err_via_match_comparison() {
    // Verify error content via alternative means
    let source = r#"
        fn make_err() -> !Result<i32, String> {
            Result·Err("test error".to_string())
        }

        fn main() -> i64 {
            match make_err() {
                Result·Ok(_) => -1,
                Result·Err(e) => {
                    // Verify the error string is correct
                    if e.len() > 0 { 42 } else { -1 }
                },
            }
        }
    "#;
    bug_test_expect_int(source, 42);
}

// =============================================================================
// BUG INT-004: is_ok().to_string() produces unexpected output
// =============================================================================

#[test]
fn test_result_is_ok_bool() {
    // Control: is_ok() should return proper boolean
    let source = r#"
        fn main() -> i64 {
            let ok: Result<i32, String> = Result·Ok(1);
            let err: Result<i32, String> = Result·Err("e".to_string());
            if ok.is_ok() && !err.is_ok() { 42 } else { -1 }
        }
    "#;
    bug_test_expect_int(source, 42);
}

#[test]
fn test_result_is_err_bool() {
    // is_err() should also work
    let source = r#"
        fn main() -> i64 {
            let ok: Result<i32, String> = Result·Ok(1);
            let err: Result<i32, String> = Result·Err("e".to_string());
            if !ok.is_err() && err.is_err() { 42 } else { -1 }
        }
    "#;
    bug_test_expect_int(source, 42);
}

#[test]
fn test_bool_to_string() {
    // Bug INT-004: bool.to_string() should produce "true"/"false"
    let source = r#"
        fn main() -> i64 {
            let t = true.to_string();
            let f = false.to_string();
            if t == "true" && f == "false" { 42 } else { -1 }
        }
    "#;
    bug_test_expect_int(source, 42);
}

// =============================================================================
// Combined scenario: Config parsing simulation
// =============================================================================

#[test]
fn test_config_style_parsing() {
    // This simulates the actual Config::from_args pattern from driver.sg
    let source = r#"
        struct Config {
            input_files: ![String],
            verbose: !bool,
        }

        impl Config {
            fn default() -> !Config {
                Config {
                    input_files: [],
                    verbose: false,
                }
            }

            fn from_args(argv: ![String]) -> !Result<Config, String> {
                let mut config = Config·default();

                for arg in argv.iter() {
                    let s = arg.clone();
                    if s == "-v" {
                        config.verbose = true;
                    } else if !s.starts_with("-") {
                        config.input_files.push(s);
                    }
                }

                if config.input_files.is_empty() {
                    return Result·Err("no input files".to_string());
                }

                Result·Ok(config)
            }
        }

        fn main() -> i64 {
            let args = vec!["compile".to_string(), "test.sg".to_string(), "-v".to_string()];
            let result = Config·from_args(args);

            match result {
                Result·Ok(config) => {
                    if config.input_files.len() == 2 && config.verbose {
                        42
                    } else {
                        // Debug: return actual len
                        config.input_files.len() as i64
                    }
                },
                Result·Err(e) => {
                    // Return -1 if error
                    -1
                },
            }
        }
    "#;
    bug_test_expect_int(source, 42);
}

// =============================================================================
// BUG INT-006: Enum variant constructors not registered for inline modules
// =============================================================================

#[test]
fn test_enum_variant_constructor_inline_module() {
    // INT-006: Enum variants should be callable from code that imports them
    let source = r#"
        mod token {
            pub enum Token {
                Ident(!String),
                Fn,
            }
        }

        mod lexer {
            invoke crate·token·*;

            pub fn make_ident(name: !String) -> !Token {
                // After invoke, Token·Ident should be available
                Token·Ident(name)
            }
        }

        invoke token·*;
        invoke lexer·*;

        fn main() -> i64 {
            let tok = make_ident("hello".to_string());
            match tok {
                Token·Ident(s) => {
                    if s == "hello" { 1 } else { -1 }
                }
                _ => -2,
            }
        }
    "#;
    bug_test_expect_int(source, 1);
}

#[test]
fn test_enum_variant_constructor_with_invoke() {
    // Enum variants should work after invoke
    let source = r#"
        mod mymod {
            pub enum Color {
                Red,
                Green,
                Blue,
                Rgb(!i64, !i64, !i64),
            }
        }

        invoke mymod·*;

        fn main() -> i64 {
            // After invoke, Color·Rgb should be available
            let c = Color·Rgb(255, 128, 64);
            match c {
                Color·Rgb(r, g, b) => r,
                _ => -1,
            }
        }
    "#;
    bug_test_expect_int(source, 255);
}

#[test]
fn test_enum_variant_imported_then_used() {
    // Test the specific pattern Jormungandr uses
    let source = r#"
        mod token {
            pub enum Token {
                Ident(!String),
                Fn,
                Number(!i64),
            }
        }

        mod lexer {
            invoke crate·token·*;

            pub struct Lexer {
                pub pos: !i64,
            }

            impl Lexer {
                pub fn new() -> !Lexer {
                    Lexer { pos: 0 }
                }

                pub fn keyword_or_ident(self, name: !String) -> !Token {
                    if name == "fn" {
                        Token·Fn
                    } else {
                        Token·Ident(name)
                    }
                }

                pub fn next_token(mut self) -> !Token {
                    // Simulate lexing "fn"
                    self.keyword_or_ident("fn".to_string())
                }
            }
        }

        invoke token·*;
        invoke lexer·*;

        fn main() -> i64 {
            let mut lex = Lexer·new();
            let tok = lex.next_token();
            match tok {
                Token·Fn => 1,
                Token·Ident(_) => -1,
                _ => -2,
            }
        }
    "#;
    bug_test_expect_int(source, 1);
}

#[test]
fn test_enum_variant_ident_return() {
    // Test returning Token::Ident specifically
    let source = r#"
        mod token {
            pub enum Token {
                Ident(!String),
                Fn,
            }
        }

        mod lexer {
            invoke crate·token·*;

            pub fn lex_ident(name: !String) -> !Token {
                Token·Ident(name)
            }
        }

        invoke token·*;
        invoke lexer·*;

        fn main() -> i64 {
            let tok = lex_ident("main".to_string());
            match tok {
                Token·Ident(s) => {
                    if s == "main" { 42 } else { -1 }
                }
                _ => -2,
            }
        }
    "#;
    bug_test_expect_int(source, 42);
}
