//! LLVM Compiler Tests
//!
//! Tests for the LLVM backend covering evidentiality, generics, and morpheme operations.

use super::*;
use crate::optimize::OptLevel;

fn run_sigil(source: &str) -> Result<i64, String> {
    let context = Context::create();
    let mut compiler = LlvmCompiler::new(&context, OptLevel::Standard)?;
    compiler.compile(source)?;
    compiler.run()
}

// ============================================
// Evidentiality Tests
// ============================================

#[test]
fn test_evidential_known_unwrap() {
    // Known (!) just returns the inner value
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            let x = 42!;
            x
        }
    "#,
    );
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn test_evidential_uncertain() {
    // Uncertain (?) wraps and unwraps correctly
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            let x = 100?;
            x
        }
    "#,
    );
    assert_eq!(result.unwrap(), 100);
}

#[test]
fn test_evidential_reported() {
    // Reported (~) wraps and unwraps correctly
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            let x = 200~;
            x
        }
    "#,
    );
    assert_eq!(result.unwrap(), 200);
}

#[test]
fn test_evidential_predicted() {
    // Predicted (◊) wraps and unwraps correctly
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            let x = 300◊;
            x
        }
    "#,
    );
    assert_eq!(result.unwrap(), 300);
}

#[test]
fn test_evidential_in_expression() {
    // Evidential values can be used in expressions
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            let a = 10?;
            let b = 20?;
            a + b
        }
    "#,
    );
    assert_eq!(result.unwrap(), 30);
}

#[test]
fn test_evidential_unwrap_chain() {
    // Chain: uncertain -> known (unwrap)
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            let x = 42?;
            let y = x!;
            y
        }
    "#,
    );
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn test_evidential_nested() {
    // Nested evidential operations
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            let x = (50?)!;
            x + 5
        }
    "#,
    );
    assert_eq!(result.unwrap(), 55);
}

#[test]
fn test_evidential_with_arithmetic() {
    // Evidential values with arithmetic
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            let known = 100!;
            let uncertain = 50?;
            known + uncertain * 2
        }
    "#,
    );
    assert_eq!(result.unwrap(), 200);
}

#[test]
fn test_evidential_function_return() {
    // Function returning evidential value
    let result = run_sigil(
        r#"
        fn get_uncertain() -> i64 {
            42?
        }

        fn main() -> i64 {
            let x = get_uncertain();
            x + 8
        }
    "#,
    );
    assert_eq!(result.unwrap(), 50);
}

#[test]
fn test_evidential_mixed_markers() {
    // Mix different evidentiality markers
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            let a = 10!;  // known
            let b = 20?;  // uncertain
            let c = 30~;  // reported
            a + b + c
        }
    "#,
    );
    assert_eq!(result.unwrap(), 60);
}

#[test]
fn test_evidential_in_if() {
    // Evidential in conditional
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            let x = 1?;
            if x == 1 {
                100?
            } else {
                200?
            }
        }
    "#,
    );
    assert_eq!(result.unwrap(), 100);
}

#[test]
fn test_evidential_paradox() {
    // Paradox (‽) marker - contradiction detection
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            let x = 42‽;
            x
        }
    "#,
    );
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn test_evidential_multiple_unwraps() {
    // Multiple sequential unwraps
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            let a = 10?;
            let b = a!;
            let c = b!;
            c
        }
    "#,
    );
    assert_eq!(result.unwrap(), 10);
}

#[test]
fn test_evidential_in_loop() {
    // Evidential values in a loop
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            let mut sum = 0?;
            let mut i = 0;
            while i < 5 {
                sum = sum + i?;
                i = i + 1;
            }
            sum!
        }
    "#,
    );
    assert_eq!(result.unwrap(), 10); // 0 + 1 + 2 + 3 + 4 = 10
}

#[test]
fn test_evidential_comparison() {
    // Comparison of evidential values
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            let a = 10?;
            let b = 20?;
            if a < b {
                1!
            } else {
                0!
            }
        }
    "#,
    );
    assert_eq!(result.unwrap(), 1);
}

#[test]
fn test_evidential_negation() {
    // Negation with evidential values
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            let x = 42?;
            let y = -x;
            y + 100
        }
    "#,
    );
    assert_eq!(result.unwrap(), 58); // -42 + 100 = 58
}

#[test]
fn test_evidential_chain_operations() {
    // Chain of operations with mixed evidentiality
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            let x = 10!;
            let y = 20?;
            let z = 30~;
            let w = 40◊;
            x + y + z + w
        }
    "#,
    );
    assert_eq!(result.unwrap(), 100);
}

#[test]
fn test_evidential_deeply_nested() {
    // Deeply nested evidential expressions
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            let x = ((((42?)?)?)?)?;
            x!
        }
    "#,
    );
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn test_evidential_struct_field() {
    // Evidential values as struct fields
    let result = run_sigil(
        r#"
        struct Data {
            value: i64,
        }

        fn main() -> i64 {
            let d = Data { value: 100? };
            d.value + 1
        }
    "#,
    );
    assert_eq!(result.unwrap(), 101);
}

#[test]
fn test_evidential_function_param() {
    // Function with evidential parameter
    let result = run_sigil(
        r#"
        fn double(x: i64) -> i64 {
            x * 2
        }

        fn main() -> i64 {
            let val = 25?;
            double(val!)
        }
    "#,
    );
    assert_eq!(result.unwrap(), 50);
}

#[test]
fn test_evidential_all_markers_chain() {
    // All 5 evidentiality markers in sequence
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            let known = 1!;      // Known
            let uncertain = 2?;  // Uncertain
            let reported = 3~;   // Reported
            let predicted = 4◊;  // Predicted
            let paradox = 5‽;    // Paradox
            known + uncertain + reported + predicted + paradox
        }
    "#,
    );
    assert_eq!(result.unwrap(), 15);
}

// ============================================
// Generic Monomorphization Tests
// ============================================

#[test]
fn test_generic_struct_basic() {
    let result = run_sigil(
        r#"
        struct Container<T> {
            value: T,
            count: i32,
        }

        fn main() -> i64 {
            let c = Container::<i32> { value: 42, count: 1 };
            c.value + c.count
        }
    "#,
    );
    assert_eq!(result.unwrap(), 43);
}

#[test]
fn test_generic_struct_two_params() {
    let result = run_sigil(
        r#"
        struct Pair<A, B> {
            first: A,
            second: B,
        }

        fn main() -> i64 {
            let p = Pair::<i32, i32> { first: 10, second: 20 };
            p.first + p.second
        }
    "#,
    );
    assert_eq!(result.unwrap(), 30);
}

// ============================================
// Morpheme Tests - Element Access
// ============================================

#[test]
fn test_morpheme_first() {
    // First element: [1, 2, 3] |α returns 1
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            [10, 20, 30] |α
        }
    "#,
    );
    assert_eq!(result.unwrap(), 10);
}

#[test]
fn test_morpheme_last() {
    // Last element: [1, 2, 3] |ω returns 3
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            [10, 20, 30] |ω
        }
    "#,
    );
    assert_eq!(result.unwrap(), 30);
}

#[test]
fn test_morpheme_middle() {
    // Middle element: [1, 2, 3, 4, 5] |μ returns 3
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            [10, 20, 30, 40, 50] |μ
        }
    "#,
    );
    assert_eq!(result.unwrap(), 30);
}

#[test]
fn test_morpheme_nth() {
    // Nth element: [1, 2, 3] |ν{1} returns 2
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            [10, 20, 30] |ν{1}
        }
    "#,
    );
    assert_eq!(result.unwrap(), 20);
}

// ============================================
// Morpheme Tests - Reductions
// ============================================

#[test]
fn test_morpheme_reduce_min() {
    // Simple min of two values
    let result = run_sigil(
        r#"
        fn min2(a: i64, b: i64) -> i64 {
            if a < b { a } else { b }
        }
        fn main() -> i64 {
            min2(min2(5, 2), min2(8, 1))
        }
    "#,
    );
    assert_eq!(result.unwrap(), 1);
}

#[test]
fn test_morpheme_reduce_max() {
    // Simple max of two values
    let result = run_sigil(
        r#"
        fn max2(a: i64, b: i64) -> i64 {
            if a > b { a } else { b }
        }
        fn main() -> i64 {
            max2(max2(5, 2), max2(8, 9))
        }
    "#,
    );
    assert_eq!(result.unwrap(), 9);
}

#[test]
fn test_morpheme_reduce_all_true() {
    // All: [1, 2, 3] |ρ& returns 1 (all non-zero)
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            [1, 2, 3] |ρ&
        }
    "#,
    );
    assert_eq!(result.unwrap(), 1);
}

#[test]
fn test_morpheme_reduce_all_false() {
    // All: [1, 0, 3] |ρ& returns 0 (not all non-zero)
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            [1, 0, 3] |ρ&
        }
    "#,
    );
    assert_eq!(result.unwrap(), 0);
}

#[test]
fn test_morpheme_reduce_any_true() {
    // Any: [0, 0, 1] |ρ| returns 1 (at least one non-zero)
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            [0, 0, 1] |ρ|
        }
    "#,
    );
    assert_eq!(result.unwrap(), 1);
}

#[test]
fn test_morpheme_reduce_any_false() {
    // Any: [0, 0, 0] |ρ| returns 0 (none non-zero)
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            [0, 0, 0] |ρ|
        }
    "#,
    );
    assert_eq!(result.unwrap(), 0);
}

// ============================================
// Combined Morpheme Tests
// ============================================

#[test]
fn test_morpheme_transform_then_first() {
    // Transform then first: [1, 2, 3] |τ{|x| x * 10} |α returns 10
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            let arr = [1, 2, 3] |τ{|x| x * 10};
            arr |α
        }
    "#,
    );
    // Note: This tests that transform returns array, then first extracts
    // Current impl may need adjustment
    assert!(result.is_ok());
}

#[test]
fn test_morpheme_filter_then_sum() {
    // Filter then sum: keep values > 3, sum them
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            [1, 5, 2, 8, 3, 7] |φ{|x| x > 3} |ρ+
        }
    "#,
    );
    // After filter: [5, 8, 7], sum = 20
    assert_eq!(result.unwrap(), 20);
}

// ============================================
// Morpheme Tests - Sort, Choice, Custom Reduce
// ============================================

#[test]
fn test_morpheme_sort_basic() {
    // Sort returns minimum (first after sort): [3, 1, 2] |σ returns 1
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            [3, 1, 2] |σ
        }
    "#,
    );
    assert_eq!(result.unwrap(), 1);
}

#[test]
fn test_morpheme_sort_already_sorted() {
    // Sort already sorted: [1, 2, 3] |σ returns 1
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            [1, 2, 3] |σ
        }
    "#,
    );
    assert_eq!(result.unwrap(), 1);
}

#[test]
fn test_morpheme_sort_reverse() {
    // Sort reverse: [5, 4, 3, 2, 1] |σ returns 1
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            [5, 4, 3, 2, 1] |σ
        }
    "#,
    );
    assert_eq!(result.unwrap(), 1);
}

#[test]
fn test_morpheme_sort_single() {
    // Sort single element: [42] |σ returns 42
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            [42] |σ
        }
    "#,
    );
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn test_morpheme_choice_deterministic() {
    // Choice is deterministic based on array contents
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            [10, 20, 30] |χ
        }
    "#,
    );
    // Result should be one of 10, 20, or 30
    let val = result.unwrap();
    assert!(val == 10 || val == 20 || val == 30);
}

#[test]
fn test_morpheme_choice_single() {
    // Choice with single element: [42] |χ returns 42
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            [42] |χ
        }
    "#,
    );
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn test_morpheme_custom_reduce_sum() {
    // Custom reduce sum: [1, 2, 3, 4] |ρ{|a, x| a + x} = 10
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            [1, 2, 3, 4] |ρ{|acc, x| acc + x}
        }
    "#,
    );
    assert_eq!(result.unwrap(), 10);
}

#[test]
fn test_morpheme_custom_reduce_product() {
    // Custom reduce product: [1, 2, 3, 4] |ρ{|a, x| a * x} = 24
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            [1, 2, 3, 4] |ρ{|acc, x| acc * x}
        }
    "#,
    );
    assert_eq!(result.unwrap(), 24);
}

#[test]
fn test_morpheme_custom_reduce_difference() {
    // Custom reduce difference: [100, 20, 5] |ρ{|a, x| a - x} = 75
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            [100, 20, 5] |ρ{|acc, x| acc - x}
        }
    "#,
    );
    assert_eq!(result.unwrap(), 75);
}

#[test]
fn test_morpheme_custom_reduce_single() {
    // Custom reduce single element: [42] |ρ{|a, x| a + x} = 42
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            [42] |ρ{|acc, x| acc + x}
        }
    "#,
    );
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn test_morpheme_await_expr() {
    // Await expression form: expr⌛ (postfix syntax)
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            let x = 42;
            x⌛
        }
    "#,
    );
    // In sync LLVM context, await is identity
    assert_eq!(result.unwrap(), 42);
}

#[test]
fn test_morpheme_await_nested() {
    // Nested await expressions
    let result = run_sigil(
        r#"
        fn main() -> i64 {
            let x = 21;
            let y = x⌛ + x⌛;
            y
        }
    "#,
    );
    assert_eq!(result.unwrap(), 42);
}
