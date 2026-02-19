//! Specification tests for async state machine transformation.
//!
//! These tests define correct behavior per Agent-TDD methodology.
//! They serve as executable specifications for the transformation.

use super::*;
use crate::ast::*;

/// Helper to create a simple identifier path.
fn ident_path(name: &str) -> Expr {
    Expr::Path(TypePath {
        segments: vec![PathSegment {
            ident: Ident {
                name: name.to_string(),
                evidentiality: None,
                affect: None,
                span: crate::span::Span::default(),
            },
            generics: None,
        }],
    })
}

/// Helper to create a call expression.
fn call(func_name: &str, args: Vec<Expr>) -> Expr {
    Expr::Call {
        func: Box::new(ident_path(func_name)),
        args,
    }
}

/// Helper to create an await expression.
fn await_expr(inner: Expr) -> Expr {
    Expr::Await {
        expr: Box::new(inner),
        evidentiality: None,
    }
}

/// Helper to create an integer literal.
fn int_lit(value: i64) -> Expr {
    Expr::Literal(Literal::Int {
        value: value.to_string(),
        base: NumBase::Decimal,
        suffix: None,
    })
}

/// Helper to create a boolean literal.
fn bool_lit(value: bool) -> Expr {
    Expr::Literal(Literal::Bool(value))
}

/// Helper to create a while expression.
fn while_expr(condition: Expr, body_stmts: Vec<Stmt>, body_expr: Option<Expr>) -> Expr {
    Expr::While {
        label: None,
        condition: Box::new(condition),
        body: Block {
            stmts: body_stmts,
            expr: body_expr.map(Box::new),
        },
    }
}

/// Helper to create a loop expression (infinite loop).
fn loop_expr(body_stmts: Vec<Stmt>, body_expr: Option<Expr>) -> Expr {
    Expr::Loop {
        label: None,
        body: Block {
            stmts: body_stmts,
            expr: body_expr.map(Box::new),
        },
    }
}

/// Helper to create a break expression.
fn break_expr(value: Option<Expr>) -> Expr {
    Expr::Break {
        label: None,
        value: value.map(Box::new),
    }
}

/// Helper to create an if expression.
fn if_expr(condition: Expr, then_stmts: Vec<Stmt>, then_expr: Option<Expr>,
           else_stmts: Option<Vec<Stmt>>, else_expr: Option<Expr>) -> Expr {
    let else_branch = if else_stmts.is_some() || else_expr.is_some() {
        Some(Box::new(Expr::Block(Block {
            stmts: else_stmts.unwrap_or_default(),
            expr: else_expr.map(Box::new),
        })))
    } else {
        None
    };

    Expr::If {
        condition: Box::new(condition),
        then_branch: Block {
            stmts: then_stmts,
            expr: then_expr.map(Box::new),
        },
        else_branch,
    }
}

/// Helper to create a let statement.
fn let_stmt(name: &str, init: Expr) -> Stmt {
    Stmt::Let {
        pattern: Pattern::Ident {
            mutable: false,
            name: Ident {
                name: name.to_string(),
                evidentiality: None,
                affect: None,
                span: crate::span::Span::default(),
            },
            evidentiality: None,
        },
        ty: None,
        init: Some(init),
    }
}

/// Helper to create a simple async function.
fn make_async_fn(name: &str, stmts: Vec<Stmt>, trailing_expr: Option<Expr>) -> Function {
    Function {
        doc_comments: Vec::new(),
        visibility: Visibility::Private,
        is_async: true,
        is_const: false,
        is_unsafe: false,
        attrs: FunctionAttrs::default(),
        name: Ident {
            name: name.to_string(),
            evidentiality: None,
            affect: None,
            span: crate::span::Span::default(),
        },
        aspect: None,
        generics: None,
        params: Vec::new(),
        return_type: Some(TypeExpr::Path(TypePath {
            segments: vec![PathSegment {
                ident: Ident {
                    name: "i64".to_string(),
                    evidentiality: None,
                    affect: None,
                    span: crate::span::Span::default(),
                },
                generics: None,
            }],
        })),
        where_clause: None,
        body: Some(Block {
            stmts,
            expr: trailing_expr.map(Box::new),
        }),
    }
}

// =============================================================================
// SPECIFICATION TESTS: IR Types
// =============================================================================

mod ir_spec {
    use super::*;

    #[test]
    fn spec_state_machine_ir_starts_empty() {
        let ir = StateMachineIR::new(
            "test".to_string(),
            vec![],
            None,
        );

        assert!(ir.states.is_empty());
        assert!(ir.locals.is_empty());
        assert_eq!(ir.next_state_idx(), 0);
    }

    #[test]
    fn spec_add_state_assigns_sequential_indices() {
        let mut ir = StateMachineIR::new(
            "test".to_string(),
            vec![],
            None,
        );

        let idx0 = ir.add_state(State::entry());
        let idx1 = ir.add_state(State::resume(1));
        let idx2 = ir.add_state(State::resume(2));

        assert_eq!(idx0, 0);
        assert_eq!(idx1, 1);
        assert_eq!(idx2, 2);
    }

    #[test]
    fn spec_entry_state_has_correct_flags() {
        let entry = State::entry();

        assert_eq!(entry.index, 0);
        assert!(entry.is_entry);
        assert!(!entry.is_resume);
        assert!(entry.resume_binding.is_none());
    }

    #[test]
    fn spec_resume_state_has_correct_flags() {
        let resume = State::resume(1);

        assert_eq!(resume.index, 1);
        assert!(!resume.is_entry);
        assert!(resume.is_resume);
    }

    #[test]
    fn spec_state_exit_await_tracks_targets() {
        let exit = StateExit::Await {
            promise: ident_path("fetch"),
            next_state: 2,
            saved_locals: vec!["x".to_string()],
        };

        assert_eq!(exit.target_states(), vec![2]);
        assert!(exit.is_await());
        assert!(!exit.is_return());
    }

    #[test]
    fn spec_state_exit_branch_tracks_both_targets() {
        let exit = StateExit::Branch {
            condition: ident_path("cond"),
            then_state: 1,
            else_state: 2,
        };

        let targets = exit.target_states();
        assert!(targets.contains(&1));
        assert!(targets.contains(&2));
    }

    #[test]
    fn spec_state_exit_return_has_no_targets() {
        let exit = StateExit::Return {
            value: int_lit(42),
        };

        assert!(exit.target_states().is_empty());
        assert!(exit.is_return());
    }

    #[test]
    fn spec_frame_layout_assigns_sequential_offsets() {
        let mut layout = FrameLayout::new();

        layout.add_local("a");
        layout.add_local("b");
        layout.add_local("c");

        // State is at 0, locals start at 8
        assert_eq!(layout.get_offset("a"), Some(8));
        assert_eq!(layout.get_offset("b"), Some(16));
        assert_eq!(layout.get_offset("c"), Some(24));
        assert_eq!(layout.total_size, 32);
    }

    #[test]
    fn spec_local_decl_liveness() {
        let local = LocalDecl {
            name: "x".to_string(),
            ty: None,
            defined_in_state: 1,
            live_until_state: 3,
        };

        assert!(!local.is_live_in(0));
        assert!(local.is_live_in(1));
        assert!(local.is_live_in(2));
        assert!(local.is_live_in(3));
        assert!(!local.is_live_in(4));
    }
}

// =============================================================================
// SPECIFICATION TESTS: Validation
// =============================================================================

mod validation_spec {
    use super::*;

    #[test]
    fn spec_valid_ir_passes_validation() {
        let mut ir = StateMachineIR::new(
            "test".to_string(),
            vec![],
            None,
        );

        let mut entry = State::entry();
        entry.exit = StateExit::Return {
            value: int_lit(42),
        };
        ir.add_state(entry);

        assert!(ir.validate().is_ok());
    }

    #[test]
    fn spec_non_entry_state_0_fails_validation() {
        let mut ir = StateMachineIR::new(
            "test".to_string(),
            vec![],
            None,
        );

        // Create state 0 without is_entry flag
        let mut bad_entry = State::resume(0);
        bad_entry.exit = StateExit::Return {
            value: int_lit(42),
        };
        ir.add_state(bad_entry);

        let result = ir.validate();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.contains("is_entry")));
    }

    #[test]
    fn spec_dangling_state_reference_fails_validation() {
        let mut ir = StateMachineIR::new(
            "test".to_string(),
            vec![],
            None,
        );

        let mut entry = State::entry();
        // Reference non-existent state 99
        entry.exit = StateExit::Goto { target: 99 };
        ir.add_state(entry);

        let result = ir.validate();
        assert!(result.is_err());
        let errors = result.unwrap_err();
        assert!(errors.iter().any(|e| e.contains("non-existent state")));
    }
}

// =============================================================================
// SPECIFICATION TESTS: Transformation - Phase 1 (Straight-line)
// =============================================================================

mod transform_phase1_spec {
    use super::*;

    #[test]
    fn spec_no_await_creates_single_state() {
        // async rite simple() -> i64 { 42 }
        let func = make_async_fn(
            "simple",
            vec![],
            Some(int_lit(42)),
        );

        let ir = transform_async_function(&func).expect("Transform failed");

        assert_eq!(ir.states.len(), 1);
        assert!(ir.states[0].is_entry);
        assert!(matches!(ir.states[0].exit, StateExit::Return { .. }));
    }

    #[test]
    fn spec_single_await_creates_two_states() {
        // async rite fetch_one() -> i64 {
        //     let x = fetch()|await;
        //     x
        // }
        let func = make_async_fn(
            "fetch_one",
            vec![
                let_stmt("x", await_expr(call("fetch", vec![]))),
            ],
            Some(ident_path("x")),
        );

        let ir = transform_async_function(&func).expect("Transform failed");

        assert_eq!(ir.states.len(), 2);

        // State 0: entry, exits with await
        assert!(ir.states[0].is_entry);
        assert!(ir.states[0].exit.is_await());

        // State 1: resume, exits with return
        assert!(ir.states[1].is_resume);
        assert!(ir.states[1].exit.is_return());
    }

    #[test]
    fn spec_two_awaits_creates_three_states() {
        // async rite fetch_two() -> i64 {
        //     let a = fetch(1)|await;
        //     let b = fetch(2)|await;
        //     a + b  -- simplified to just 'a' for test
        // }
        let func = make_async_fn(
            "fetch_two",
            vec![
                let_stmt("a", await_expr(call("fetch", vec![int_lit(1)]))),
                let_stmt("b", await_expr(call("fetch", vec![int_lit(2)]))),
            ],
            Some(ident_path("a")),
        );

        let ir = transform_async_function(&func).expect("Transform failed");

        assert_eq!(ir.states.len(), 3);

        // State 0: entry -> await
        assert!(ir.states[0].is_entry);
        assert!(ir.states[0].exit.is_await());

        // State 1: resume -> await
        assert!(ir.states[1].is_resume);
        assert!(ir.states[1].exit.is_await());

        // State 2: resume -> return
        assert!(ir.states[2].is_resume);
        assert!(ir.states[2].exit.is_return());
    }

    #[test]
    fn spec_await_result_bound_in_resume_state() {
        // async rite with_binding() -> i64 {
        //     let result = compute()|await;
        //     result
        // }
        let func = make_async_fn(
            "with_binding",
            vec![
                let_stmt("result", await_expr(call("compute", vec![]))),
            ],
            Some(ident_path("result")),
        );

        let ir = transform_async_function(&func).expect("Transform failed");

        // The resume state should have the binding
        assert_eq!(ir.states[1].resume_binding, Some("result".to_string()));
    }

    #[test]
    fn spec_locals_declared_before_await_are_saved() {
        // async rite with_locals() -> i64 {
        //     let x = 10;
        //     let y = fetch()|await;
        //     x + y  -- simplified
        // }
        let func = make_async_fn(
            "with_locals",
            vec![
                let_stmt("x", int_lit(10)),
                let_stmt("y", await_expr(call("fetch", vec![]))),
            ],
            Some(ident_path("x")),
        );

        let ir = transform_async_function(&func).expect("Transform failed");

        // Check that state 0's await saves 'x'
        if let StateExit::Await { saved_locals, .. } = &ir.states[0].exit {
            assert!(saved_locals.contains(&"x".to_string()));
        } else {
            panic!("Expected Await exit");
        }
    }

    #[test]
    fn spec_frame_layout_includes_all_locals() {
        // async rite multi_locals() -> i64 {
        //     let a = 1;
        //     let b = fetch()|await;
        //     let c = 3;
        //     a + b + c  -- simplified
        // }
        let func = make_async_fn(
            "multi_locals",
            vec![
                let_stmt("a", int_lit(1)),
                let_stmt("b", await_expr(call("fetch", vec![]))),
                let_stmt("c", int_lit(3)),
            ],
            Some(ident_path("a")),
        );

        let ir = transform_async_function(&func).expect("Transform failed");

        assert!(ir.frame_layout.has_offset("a"));
        assert!(ir.frame_layout.has_offset("b"));
        assert!(ir.frame_layout.has_offset("c"));
    }

    #[test]
    fn spec_validated_ir_has_no_unreachable_exits() {
        // async rite complete() -> i64 {
        //     let x = fetch()|await;
        //     x
        // }
        let func = make_async_fn(
            "complete",
            vec![
                let_stmt("x", await_expr(call("fetch", vec![]))),
            ],
            Some(ident_path("x")),
        );

        let ir = transform_async_function(&func).expect("Transform failed");

        for state in &ir.states {
            assert!(
                !matches!(state.exit, StateExit::Unreachable),
                "State {} has Unreachable exit",
                state.index
            );
        }
    }
}

// =============================================================================
// SPECIFICATION TESTS: needs_state_machine detection
// =============================================================================

mod detection_spec {
    use super::*;

    #[test]
    fn spec_non_async_function_does_not_need_sm() {
        let mut func = make_async_fn(
            "sync_fn",
            vec![],
            Some(int_lit(42)),
        );
        func.is_async = false;

        assert!(!needs_state_machine(&func));
    }

    #[test]
    fn spec_async_no_await_does_not_need_sm() {
        let func = make_async_fn(
            "no_await",
            vec![],
            Some(int_lit(42)),
        );

        assert!(!needs_state_machine(&func));
    }

    #[test]
    fn spec_async_single_await_does_not_need_sm() {
        // Single await can use simpler Asyncify approach
        let func = make_async_fn(
            "one_await",
            vec![],
            Some(await_expr(call("fetch", vec![]))),
        );

        assert!(!needs_state_machine(&func));
    }

    #[test]
    fn spec_async_multiple_awaits_needs_sm() {
        let func = make_async_fn(
            "multi_await",
            vec![
                let_stmt("a", await_expr(call("fetch", vec![int_lit(1)]))),
                let_stmt("b", await_expr(call("fetch", vec![int_lit(2)]))),
            ],
            Some(ident_path("a")),
        );

        assert!(needs_state_machine(&func));
    }
}

// =============================================================================
// SPECIFICATION TESTS: Error handling
// =============================================================================

mod error_spec {
    use super::*;

    #[test]
    fn spec_function_without_body_returns_error() {
        let func = Function {
            doc_comments: Vec::new(),
            visibility: Visibility::Private,
            is_async: true,
            is_const: false,
            is_unsafe: false,
            attrs: FunctionAttrs::default(),
            name: Ident {
                name: "no_body".to_string(),
                evidentiality: None,
                affect: None,
                span: crate::span::Span::default(),
            },
            aspect: None,
            generics: None,
            params: Vec::new(),
            return_type: None,
            where_clause: None,
            body: None,
        };

        let result = transform_async_function(&func);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err().kind,
            TransformErrorKind::NoBody
        ));
    }
}

// =============================================================================
// PROPERTY TESTS
// =============================================================================

mod property_spec {
    use super::*;

    #[test]
    fn property_state_count_equals_await_count_plus_one() {
        // For straight-line code: states = awaits + 1
        let test_cases = [
            // (stmts, trailing_expr, expected_awaits, expected_states)
            (vec![], Some(int_lit(1)), 0, 1),
            (vec![], Some(await_expr(call("f", vec![]))), 1, 2),
            (vec![
                let_stmt("a", await_expr(call("f", vec![]))),
                let_stmt("b", await_expr(call("g", vec![]))),
            ], Some(ident_path("a")), 2, 3),
        ];

        for (stmts, trailing, expected_awaits, expected_states) in test_cases {
            let func = make_async_fn("test", stmts, trailing);
            let ir = transform_async_function(&func).expect("Transform failed");

            // Count actual await exits
            let await_count = ir.states.iter()
                .filter(|s| s.exit.is_await())
                .count();

            assert_eq!(await_count, expected_awaits);
            assert_eq!(ir.states.len(), expected_states);
        }
    }

    #[test]
    fn property_all_states_reachable_from_entry() {
        let func = make_async_fn(
            "reachable",
            vec![
                let_stmt("a", await_expr(call("f", vec![]))),
                let_stmt("b", await_expr(call("g", vec![]))),
                let_stmt("c", await_expr(call("h", vec![]))),
            ],
            Some(ident_path("a")),
        );

        let ir = transform_async_function(&func).expect("Transform failed");

        // BFS from state 0 should reach all states
        let mut visited = vec![false; ir.states.len()];
        let mut queue = vec![0u32];

        while let Some(idx) = queue.pop() {
            if visited[idx as usize] {
                continue;
            }
            visited[idx as usize] = true;

            for target in ir.states[idx as usize].exit.target_states() {
                if !visited[target as usize] {
                    queue.push(target);
                }
            }
        }

        for (i, v) in visited.iter().enumerate() {
            assert!(v, "State {} is not reachable from entry", i);
        }
    }
}

// =============================================================================
// SPECIFICATION TESTS: Transformation - Phase 2 (Conditionals)
// =============================================================================

mod transform_phase2_spec {
    use super::*;

    #[test]
    fn spec_if_without_await_stays_in_single_state() {
        // async rite simple_if() -> i64 {
        //     if true { 1 } else { 2 }
        // }
        let func = make_async_fn(
            "simple_if",
            vec![],
            Some(if_expr(
                bool_lit(true),
                vec![],
                Some(int_lit(1)),
                Some(vec![]),
                Some(int_lit(2)),
            )),
        );

        let ir = transform_async_function(&func).expect("Transform failed");

        // No await, so single state
        assert_eq!(ir.states.len(), 1);
        assert!(ir.states[0].exit.is_return());
    }

    #[test]
    fn spec_if_with_await_in_then_creates_branch() {
        // async rite await_in_then() -> i64 {
        //     if cond() {
        //         let x = fetch()|await;
        //         x
        //     } else {
        //         42
        //     }
        // }
        let func = make_async_fn(
            "await_in_then",
            vec![],
            Some(if_expr(
                call("cond", vec![]),
                vec![let_stmt("x", await_expr(call("fetch", vec![])))],
                Some(ident_path("x")),
                Some(vec![]),
                Some(int_lit(42)),
            )),
        );

        let ir = transform_async_function(&func).expect("Transform failed");

        // Should have: entry (branch) -> then-await -> then-resume -> join
        //                            -> else (no await) -> join
        // Minimum: entry + then-pre-await + then-post-await + else + join = 5 states
        // Or simplified: entry branches to then/else, both eventually return
        assert!(ir.states.len() >= 3, "Expected at least 3 states, got {}", ir.states.len());

        // Entry state should have a Branch exit
        assert!(
            matches!(ir.states[0].exit, StateExit::Branch { .. }),
            "Entry state should have Branch exit, got {:?}",
            ir.states[0].exit
        );
    }

    #[test]
    fn spec_if_with_await_in_both_branches() {
        // async rite await_in_both() -> i64 {
        //     if cond() {
        //         let x = fetch_a()|await;
        //         x
        //     } else {
        //         let y = fetch_b()|await;
        //         y
        //     }
        // }
        let func = make_async_fn(
            "await_in_both",
            vec![],
            Some(if_expr(
                call("cond", vec![]),
                vec![let_stmt("x", await_expr(call("fetch_a", vec![])))],
                Some(ident_path("x")),
                Some(vec![let_stmt("y", await_expr(call("fetch_b", vec![])))]),
                Some(ident_path("y")),
            )),
        );

        let ir = transform_async_function(&func).expect("Transform failed");

        // Should have states for:
        // - entry (Branch)
        // - then pre-await (Await)
        // - then post-await (Return or Goto)
        // - else pre-await (Await)
        // - else post-await (Return or Goto)
        // Possibly a join state
        assert!(ir.states.len() >= 5, "Expected at least 5 states, got {}", ir.states.len());

        // Entry should branch
        if let StateExit::Branch { then_state, else_state, .. } = &ir.states[0].exit {
            // Both targets should exist
            assert!(*then_state < ir.states.len() as u32);
            assert!(*else_state < ir.states.len() as u32);
            // They should be different
            assert_ne!(then_state, else_state);
        } else {
            panic!("Entry state should have Branch exit");
        }

        // Should have exactly 2 await exits (one per branch)
        let await_count = ir.states.iter()
            .filter(|s| s.exit.is_await())
            .count();
        assert_eq!(await_count, 2, "Expected 2 await points, got {}", await_count);
    }

    #[test]
    fn spec_if_branches_join_at_continuation() {
        // async rite join_after_if() -> i64 {
        //     let result = if cond() {
        //         fetch_a()|await
        //     } else {
        //         fetch_b()|await
        //     };
        //     process(result)  // This code runs after either branch
        // }
        let func = make_async_fn(
            "join_after_if",
            vec![
                let_stmt("result", if_expr(
                    call("cond", vec![]),
                    vec![],
                    Some(await_expr(call("fetch_a", vec![]))),
                    Some(vec![]),
                    Some(await_expr(call("fetch_b", vec![]))),
                )),
            ],
            Some(call("process", vec![ident_path("result")])),
        );

        let ir = transform_async_function(&func).expect("Transform failed");

        // Both branch continuations should eventually reach a join state
        // that executes process(result)
        // The IR should pass validation (all targets exist)
        assert!(ir.validate().is_ok(), "IR should be valid: {:?}", ir.validate());
    }

    #[test]
    fn spec_if_only_else_has_await() {
        // async rite await_in_else() -> i64 {
        //     if cond() {
        //         42
        //     } else {
        //         fetch()|await
        //     }
        // }
        let func = make_async_fn(
            "await_in_else",
            vec![],
            Some(if_expr(
                call("cond", vec![]),
                vec![],
                Some(int_lit(42)),
                Some(vec![]),
                Some(await_expr(call("fetch", vec![]))),
            )),
        );

        let ir = transform_async_function(&func).expect("Transform failed");

        // Should have: entry (branch) -> then (return)
        //                            -> else-await -> else-resume (return)
        assert!(ir.states.len() >= 3, "Expected at least 3 states, got {}", ir.states.len());

        // Exactly one await
        let await_count = ir.states.iter()
            .filter(|s| s.exit.is_await())
            .count();
        assert_eq!(await_count, 1);
    }

    #[test]
    fn spec_nested_if_with_await() {
        // async rite nested_if() -> i64 {
        //     if outer_cond() {
        //         if inner_cond() {
        //             fetch_a()|await
        //         } else {
        //             1
        //         }
        //     } else {
        //         fetch_b()|await
        //     }
        // }
        let func = make_async_fn(
            "nested_if",
            vec![],
            Some(if_expr(
                call("outer_cond", vec![]),
                vec![],
                Some(if_expr(
                    call("inner_cond", vec![]),
                    vec![],
                    Some(await_expr(call("fetch_a", vec![]))),
                    Some(vec![]),
                    Some(int_lit(1)),
                )),
                Some(vec![]),
                Some(await_expr(call("fetch_b", vec![]))),
            )),
        );

        let ir = transform_async_function(&func).expect("Transform failed");

        // Should handle nested conditionals with awaits
        // 2 awaits total (fetch_a and fetch_b)
        let await_count = ir.states.iter()
            .filter(|s| s.exit.is_await())
            .count();
        assert_eq!(await_count, 2, "Expected 2 await points for nested if");

        // IR should be valid
        assert!(ir.validate().is_ok());
    }

    #[test]
    fn spec_if_without_else_with_await() {
        // async rite no_else() -> i64 {
        //     if cond() {
        //         fetch()|await;
        //     }
        //     42
        // }
        let func = make_async_fn(
            "no_else",
            vec![
                Stmt::Semi(if_expr(
                    call("cond", vec![]),
                    vec![],
                    Some(await_expr(call("fetch", vec![]))),
                    None, // No else branch
                    None,
                )),
            ],
            Some(int_lit(42)),
        );

        let ir = transform_async_function(&func).expect("Transform failed");

        // Should have: entry (branch) -> then-await -> then-resume -> join
        //                            -> else (returns unit) -> join
        //              join -> return 42
        assert!(ir.states.len() >= 4, "Expected at least 4 states, got {}", ir.states.len());

        // Entry should branch
        assert!(
            matches!(ir.states[0].exit, StateExit::Branch { .. }),
            "Entry state should have Branch exit"
        );

        // Exactly one await (in then branch)
        let await_count = ir.states.iter()
            .filter(|s| s.exit.is_await())
            .count();
        assert_eq!(await_count, 1);

        // IR should be valid
        assert!(ir.validate().is_ok(), "IR should be valid: {:?}", ir.validate());
    }

    #[test]
    fn spec_if_statement_with_await_no_trailing_expr() {
        // async rite stmt_if() {
        //     if cond() {
        //         fetch_a()|await;
        //     } else {
        //         fetch_b()|await;
        //     }
        //     // No trailing expression - implicit unit return
        // }
        let func = Function {
            doc_comments: Vec::new(),
            visibility: Visibility::Private,
            is_async: true,
            is_const: false,
            is_unsafe: false,
            attrs: FunctionAttrs::default(),
            name: Ident {
                name: "stmt_if".to_string(),
                evidentiality: None,
                affect: None,
                span: crate::span::Span::default(),
            },
            aspect: None,
            generics: None,
            params: Vec::new(),
            return_type: None, // Returns unit
            where_clause: None,
            body: Some(Block {
                stmts: vec![
                    Stmt::Semi(if_expr(
                        call("cond", vec![]),
                        vec![],
                        Some(await_expr(call("fetch_a", vec![]))),
                        Some(vec![]),
                        Some(await_expr(call("fetch_b", vec![]))),
                    )),
                ],
                expr: None, // No trailing expression
            }),
        };

        let ir = transform_async_function(&func).expect("Transform failed");

        // Should handle if statement with no trailing expression
        // Entry branches, both branches have awaits, join returns unit
        assert!(ir.validate().is_ok(), "IR should be valid: {:?}", ir.validate());

        // Should have 2 await points
        let await_count = ir.states.iter()
            .filter(|s| s.exit.is_await())
            .count();
        assert_eq!(await_count, 2);
    }

    #[test]
    fn spec_if_with_code_before_and_after() {
        // async rite code_around_if() -> i64 {
        //     let before = setup();
        //     let x = if cond() {
        //         fetch()|await
        //     } else {
        //         0
        //     };
        //     cleanup(x)
        // }
        let func = make_async_fn(
            "code_around_if",
            vec![
                let_stmt("before", call("setup", vec![])),
                let_stmt("x", if_expr(
                    call("cond", vec![]),
                    vec![],
                    Some(await_expr(call("fetch", vec![]))),
                    Some(vec![]),
                    Some(int_lit(0)),
                )),
            ],
            Some(call("cleanup", vec![ident_path("x")])),
        );

        let ir = transform_async_function(&func).expect("Transform failed");

        // 'before' should be saved across the await
        if let Some(state) = ir.states.iter().find(|s| s.exit.is_await()) {
            if let StateExit::Await { saved_locals, .. } = &state.exit {
                assert!(
                    saved_locals.contains(&"before".to_string()),
                    "Local 'before' should be saved across await"
                );
            }
        }

        assert!(ir.validate().is_ok());
    }
}

// =============================================================================
// SPECIFICATION TESTS: Transformation - Phase 3 (Loops)
// =============================================================================

mod transform_phase3_spec {
    use super::*;

    #[test]
    fn spec_while_without_await_stays_in_single_state() {
        // async rite simple_while() -> i64 {
        //     while has_more() {
        //         process();
        //     }
        //     42
        // }
        let func = make_async_fn(
            "simple_while",
            vec![
                Stmt::Semi(while_expr(
                    call("has_more", vec![]),
                    vec![Stmt::Semi(call("process", vec![]))],
                    None,
                )),
            ],
            Some(int_lit(42)),
        );

        let ir = transform_async_function(&func).expect("Transform failed");

        // No await, so single state with the while embedded in body
        assert_eq!(ir.states.len(), 1);
        assert!(ir.states[0].exit.is_return());
    }

    #[test]
    fn spec_while_with_await_creates_loop_head() {
        // async rite while_await() -> i64 {
        //     while has_more() {
        //         let item = fetch_next()|await;
        //         process(item);
        //     }
        //     done()
        // }
        let func = make_async_fn(
            "while_await",
            vec![
                Stmt::Semi(while_expr(
                    call("has_more", vec![]),
                    vec![
                        let_stmt("item", await_expr(call("fetch_next", vec![]))),
                        Stmt::Semi(call("process", vec![ident_path("item")])),
                    ],
                    None,
                )),
            ],
            Some(call("done", vec![])),
        );

        let ir = transform_async_function(&func).expect("Transform failed");

        // Should have:
        // State 0 (entry): Goto -> loop head
        // State 1 (loop head): LoopHead { condition, body_state: 2, exit_state: 4 }
        // State 2 (body, pre-await): Await
        // State 3 (body, post-await): Goto -> loop head
        // State 4 (after loop): Return
        assert!(ir.states.len() >= 4, "Expected at least 4 states, got {}", ir.states.len());

        // Should have exactly one await
        let await_count = ir.states.iter()
            .filter(|s| s.exit.is_await())
            .count();
        assert_eq!(await_count, 1, "Expected 1 await point");

        // Should have a LoopHead exit
        let has_loop_head = ir.states.iter()
            .any(|s| matches!(s.exit, StateExit::LoopHead { .. }));
        assert!(has_loop_head, "Expected a LoopHead exit");

        // IR should be valid
        assert!(ir.validate().is_ok(), "IR should be valid: {:?}", ir.validate());
    }

    #[test]
    fn spec_loop_body_returns_to_head() {
        // async rite loop_back() -> i64 {
        //     while true {
        //         let x = fetch()|await;
        //         if done(x) { break; }
        //     }
        //     42
        // }
        // For now, simplify to just: while condition { await }
        let func = make_async_fn(
            "loop_back",
            vec![
                Stmt::Semi(while_expr(
                    call("condition", vec![]),
                    vec![
                        let_stmt("x", await_expr(call("fetch", vec![]))),
                    ],
                    None,
                )),
            ],
            Some(int_lit(42)),
        );

        let ir = transform_async_function(&func).expect("Transform failed");

        // Find the post-await state (resume state after the await)
        let resume_states: Vec<_> = ir.states.iter()
            .filter(|s| s.is_resume && s.resume_binding.is_some())
            .collect();

        assert!(!resume_states.is_empty(), "Should have a resume state with binding");

        // The resume state should Goto back to the loop head
        // Find the loop head state
        let loop_head_idx = ir.states.iter()
            .position(|s| matches!(s.exit, StateExit::LoopHead { .. }));

        if let Some(head_idx) = loop_head_idx {
            // At least one state should goto back to the loop head
            let gotos_to_head = ir.states.iter()
                .filter(|s| matches!(&s.exit, StateExit::Goto { target } if *target == head_idx as u32))
                .count();
            assert!(gotos_to_head > 0, "Expected at least one Goto back to loop head");
        }

        assert!(ir.validate().is_ok());
    }

    #[test]
    fn spec_infinite_loop_with_break() {
        // async rite loop_break() -> i64 {
        //     loop {
        //         let x = fetch()|await;
        //         if done(x) {
        //             break;  // break without value
        //         }
        //     }
        //     42
        // }
        let func = make_async_fn(
            "loop_break",
            vec![
                Stmt::Semi(loop_expr(
                    vec![
                        let_stmt("x", await_expr(call("fetch", vec![]))),
                    ],
                    Some(break_expr(None)),  // break without value
                )),
            ],
            Some(int_lit(42)),
        );

        let ir = transform_async_function(&func).expect("Transform failed");

        // Should have a LoopHead with no condition (infinite loop)
        let infinite_loop = ir.states.iter()
            .find(|s| matches!(&s.exit, StateExit::LoopHead { condition: None, .. }));
        assert!(infinite_loop.is_some(), "Expected infinite loop (LoopHead with condition: None)");

        // Should have await
        let await_count = ir.states.iter()
            .filter(|s| s.exit.is_await())
            .count();
        assert_eq!(await_count, 1);

        // Break should Goto exit state
        let break_goto = ir.states.iter()
            .any(|s| matches!(s.exit, StateExit::Goto { .. }));
        assert!(break_goto, "Break should create Goto exit");

        assert!(ir.validate().is_ok(), "IR should be valid: {:?}", ir.validate());
    }

    #[test]
    fn spec_multiple_awaits_in_loop_body() {
        // async rite multi_await_loop() -> i64 {
        //     while condition() {
        //         let a = fetch_a()|await;
        //         let b = fetch_b()|await;
        //         process(a, b);
        //     }
        //     42
        // }
        let func = make_async_fn(
            "multi_await_loop",
            vec![
                Stmt::Semi(while_expr(
                    call("condition", vec![]),
                    vec![
                        let_stmt("a", await_expr(call("fetch_a", vec![]))),
                        let_stmt("b", await_expr(call("fetch_b", vec![]))),
                        Stmt::Semi(call("process", vec![ident_path("a"), ident_path("b")])),
                    ],
                    None,
                )),
            ],
            Some(int_lit(42)),
        );

        let ir = transform_async_function(&func).expect("Transform failed");

        // Should have 2 await points
        let await_count = ir.states.iter()
            .filter(|s| s.exit.is_await())
            .count();
        assert_eq!(await_count, 2, "Expected 2 await points in loop body");

        // Both 'a' and 'b' should be declared
        assert!(ir.has_local("a"), "Local 'a' should be declared");
        assert!(ir.has_local("b"), "Local 'b' should be declared");

        assert!(ir.validate().is_ok());
    }

    #[test]
    fn spec_nested_loop_with_await() {
        // async rite nested_loop() -> i64 {
        //     while outer() {
        //         while inner() {
        //             let x = fetch()|await;
        //         }
        //     }
        //     42
        // }
        let func = make_async_fn(
            "nested_loop",
            vec![
                Stmt::Semi(while_expr(
                    call("outer", vec![]),
                    vec![
                        Stmt::Semi(while_expr(
                            call("inner", vec![]),
                            vec![
                                let_stmt("x", await_expr(call("fetch", vec![]))),
                            ],
                            None,
                        )),
                    ],
                    None,
                )),
            ],
            Some(int_lit(42)),
        );

        let ir = transform_async_function(&func).expect("Transform failed");

        // Should have 2 LoopHead exits (outer and inner)
        let loop_head_count = ir.states.iter()
            .filter(|s| matches!(s.exit, StateExit::LoopHead { .. }))
            .count();
        assert_eq!(loop_head_count, 2, "Expected 2 loop heads for nested loops");

        // One await
        let await_count = ir.states.iter()
            .filter(|s| s.exit.is_await())
            .count();
        assert_eq!(await_count, 1);

        assert!(ir.validate().is_ok());
    }

    #[test]
    fn spec_loop_with_code_before_and_after() {
        // async rite code_around_loop() -> i64 {
        //     let setup_val = setup();
        //     while has_more() {
        //         let x = fetch()|await;
        //     }
        //     cleanup(setup_val)
        // }
        let func = make_async_fn(
            "code_around_loop",
            vec![
                let_stmt("setup_val", call("setup", vec![])),
                Stmt::Semi(while_expr(
                    call("has_more", vec![]),
                    vec![
                        let_stmt("x", await_expr(call("fetch", vec![]))),
                    ],
                    None,
                )),
            ],
            Some(call("cleanup", vec![ident_path("setup_val")])),
        );

        let ir = transform_async_function(&func).expect("Transform failed");

        // setup_val should be saved across awaits in the loop
        if let Some(state) = ir.states.iter().find(|s| s.exit.is_await()) {
            if let StateExit::Await { saved_locals, .. } = &state.exit {
                assert!(
                    saved_locals.contains(&"setup_val".to_string()),
                    "Local 'setup_val' should be saved across await in loop"
                );
            }
        }

        assert!(ir.validate().is_ok());
    }

    #[test]
    fn spec_while_condition_re_evaluated_each_iteration() {
        // This is a semantic test - the loop head state should contain the
        // condition expression, not a precomputed value
        let func = make_async_fn(
            "condition_check",
            vec![
                Stmt::Semi(while_expr(
                    call("check_condition", vec![]),
                    vec![
                        let_stmt("x", await_expr(call("fetch", vec![]))),
                    ],
                    None,
                )),
            ],
            Some(int_lit(0)),
        );

        let ir = transform_async_function(&func).expect("Transform failed");

        // Find the LoopHead
        let loop_head = ir.states.iter()
            .find(|s| matches!(s.exit, StateExit::LoopHead { .. }));

        assert!(loop_head.is_some(), "Should have a loop head state");

        if let Some(state) = loop_head {
            if let StateExit::LoopHead { condition, .. } = &state.exit {
                assert!(condition.is_some(), "While loop should have a condition");
            }
        }

        assert!(ir.validate().is_ok());
    }

    #[test]
    fn spec_continue_returns_to_loop_head() {
        // async rite with_continue() -> i64 {
        //     while condition() {
        //         let x = fetch()|await;
        //         if skip(x) {
        //             continue;
        //         }
        //         process(x);
        //     }
        //     42
        // }
        // Simplified: while { await; continue; }
        let func = make_async_fn(
            "with_continue",
            vec![
                Stmt::Semi(while_expr(
                    call("condition", vec![]),
                    vec![
                        let_stmt("x", await_expr(call("fetch", vec![]))),
                    ],
                    Some(continue_expr()),
                )),
            ],
            Some(int_lit(42)),
        );

        let ir = transform_async_function(&func).expect("Transform failed");

        // Find the loop head
        let loop_head_idx = ir.states.iter()
            .position(|s| matches!(s.exit, StateExit::LoopHead { .. }));

        assert!(loop_head_idx.is_some(), "Should have a loop head");

        // Continue should Goto back to loop head
        let head_idx = loop_head_idx.unwrap() as u32;
        let gotos_to_head = ir.states.iter()
            .filter(|s| matches!(&s.exit, StateExit::Goto { target } if *target == head_idx))
            .count();

        // Should have at least one Goto to head (from continue)
        assert!(gotos_to_head >= 1, "Continue should Goto loop head");

        assert!(ir.validate().is_ok());
    }

    #[test]
    fn spec_break_with_value_creates_synthetic_local() {
        // async rite break_value() -> i64 {
        //     loop {
        //         let x = fetch()|await;
        //         break x;  // break with value creates synthetic local
        //     }
        // }
        let func = make_async_fn(
            "break_value",
            vec![],
            Some(loop_expr(
                vec![
                    let_stmt("x", await_expr(call("fetch", vec![]))),
                ],
                Some(break_expr(Some(ident_path("x")))),
            )),
        );

        let result = transform_async_function(&func);
        assert!(result.is_ok(), "break with value should succeed");

        let ir = result.unwrap();

        // Check that a synthetic local was created for the break value
        let has_synthetic = ir.locals.iter().any(|local| local.name.starts_with("__break_value_"));
        assert!(has_synthetic, "Should create synthetic local for break value");

        assert!(ir.validate().is_ok());
    }

    #[test]
    fn spec_for_loop_with_await_returns_error() {
        // async rite for_await() -> i64 {
        //     for item in items {
        //         let x = fetch(item)|await;
        //     }
        //     42
        // }
        let func = make_async_fn(
            "for_await",
            vec![
                Stmt::Semi(for_expr(
                    "item",
                    ident_path("items"),
                    vec![
                        let_stmt("x", await_expr(call("fetch", vec![ident_path("item")]))),
                    ],
                )),
            ],
            Some(int_lit(42)),
        );

        let result = transform_async_function(&func);
        assert!(result.is_err(), "for loop with await should return error");

        let err = result.unwrap_err();
        assert!(
            err.message.contains("For loops"),
            "Error should mention 'For loops', got: {}",
            err.message
        );
    }
}

/// Helper to create a continue expression.
fn continue_expr() -> Expr {
    Expr::Continue { label: None }
}

/// Helper to create a for expression.
fn for_expr(binding: &str, iter: Expr, body_stmts: Vec<Stmt>) -> Expr {
    Expr::For {
        label: None,
        pattern: Pattern::Ident {
            mutable: false,
            name: crate::ast::Ident {
                name: binding.to_string(),
                evidentiality: None,
                affect: None,
                span: crate::span::Span::default(),
            },
            evidentiality: None,
        },
        iter: Box::new(iter),
        body: Block {
            stmts: body_stmts,
            expr: None,
        },
    }
}
