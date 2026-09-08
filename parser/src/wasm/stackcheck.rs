//! A stack checker over the backend's own instruction lists.
//!
//! `wasmparser` says a module is invalid and gives a byte offset into the
//! encoded binary. That is the right answer to "is this loadable" and the wrong
//! one to "what did I emit wrong": the offset names nothing a person can act on,
//! and the constructs that produce these bugs reproduce only in combination, so
//! narrowing them by deleting source lines produces programs that fail for a
//! different reason than the original.
//!
//! This walks a `CompiledFunction` before it is encoded and reports the first
//! instruction whose operands do not line up — by function name, by index, with
//! the surrounding instructions and the modelled stack. That is enough to
//! recognise the construct in the generated Sigil without bisecting for it.
//!
//! It models the subset the backend emits. Anything past a branch inside a frame
//! is not tracked (WebAssembly's stack is polymorphic after one, and guessing
//! there would produce false reports), so it can miss a bug; it should not
//! invent one.

use wasm_encoder::{BlockType, Instruction, ValType};

use super::types::CompiledFunction;

/// What went wrong, and enough context to recognise it.
pub struct StackReport {
    pub function: String,
    pub index: usize,
    pub message: String,
    /// Instructions around the failure, rendered, with the failing one marked.
    pub window: Vec<String>,
    /// The modelled operand stack just before the failing instruction.
    pub stack: Vec<String>,
}

impl StackReport {
    pub fn render(&self) -> String {
        let mut out = format!(
            "  in `{}`, instruction #{}: {}\n",
            self.function, self.index, self.message
        );
        out.push_str(&format!(
            "    stack before: [{}]\n",
            if self.stack.is_empty() {
                "empty".to_string()
            } else {
                self.stack.join(", ")
            }
        ));
        for line in &self.window {
            out.push_str("    ");
            out.push_str(line);
            out.push('\n');
        }
        out
    }
}

struct Frame {
    /// Stack height on entry, below this frame's own values.
    height: usize,
    /// What the frame leaves behind at its `end`.
    results: Vec<ValType>,
    /// A branch was taken out of this frame, so its stack is polymorphic.
    unreachable: bool,
}

fn ty(t: ValType) -> &'static str {
    match t {
        ValType::I32 => "i32",
        ValType::I64 => "i64",
        ValType::F32 => "f32",
        ValType::F64 => "f64",
        _ => "?",
    }
}

fn block_results(
    bt: &BlockType,
    type_sig: &dyn Fn(u32) -> Option<(Vec<ValType>, Vec<ValType>)>,
) -> Vec<ValType> {
    match bt {
        BlockType::Empty => Vec::new(),
        BlockType::Result(t) => vec![*t],
        BlockType::FunctionType(i) => type_sig(*i).map(|(_, r)| r).unwrap_or_default(),
    }
}

/// Check one function. `None` means nothing was found in the modelled subset.
pub fn check_function(
    func: &CompiledFunction,
    repairs: &mut Vec<(usize, Instruction<'static>)>,
    call_sig: &dyn Fn(u32) -> Option<(Vec<ValType>, Vec<ValType>)>,
    type_sig: &dyn Fn(u32) -> Option<(Vec<ValType>, Vec<ValType>)>,
    global_ty: &dyn Fn(u32) -> Option<ValType>,
) -> Option<StackReport> {
    let params: Vec<ValType> = func.params.iter().map(|(_, t)| *t).collect();
    let nparams = params.len();
    let local_ty = |idx: u32| -> Option<ValType> {
        let i = idx as usize;
        if i < nparams {
            params.get(i).copied()
        } else {
            func.local_types.get(i - nparams).copied()
        }
    };

    // Each entry is the value's type and the index of the instruction that
    // pushed it — which is where a conversion for it belongs.
    let mut stack: Vec<(ValType, usize)> = Vec::new();
    repairs.clear();
    let mut frames: Vec<Frame> = vec![Frame {
        height: 0,
        results: func.results.clone(),
        unreachable: false,
    }];

    // How much of the instruction stream to show. Six lines names most
    // constructs; SIGIL_WASM_WINDOW widens it when they do not.
    let before: usize = std::env::var("SIGIL_WASM_WINDOW")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(6);

    let report = |index: usize, message: String, stack: &[(ValType, usize)]| -> StackReport {
        let lo = index.saturating_sub(before);
        let hi = (index + 3).min(func.instructions.len());
        let window = (lo..hi)
            .map(|i| {
                format!(
                    "{} {:>4}  {:?}",
                    if i == index { "->" } else { "  " },
                    i,
                    func.instructions[i]
                )
            })
            .collect();
        StackReport {
            function: func.name.clone(),
            index,
            message,
            window,
            stack: stack.iter().map(|(t, _)| ty(*t).to_string()).collect(),
        }
    };

    for (i, instr) in func.instructions.iter().enumerate() {
        let poly = frames.last().map(|f| f.unreachable).unwrap_or(false);

        match instr {
            Instruction::Block(bt) | Instruction::Loop(bt) => {
                frames.push(Frame {
                    height: stack.len(),
                    results: block_results(bt, type_sig),
                    unreachable: poly,
                });
                continue;
            }
            Instruction::If(bt) => {
                if !poly && stack.pop().is_none() {
                    return Some(report(i, "`if` with no condition on the stack".into(), &stack));
                }
                frames.push(Frame {
                    height: stack.len(),
                    results: block_results(bt, type_sig),
                    unreachable: poly,
                });
                continue;
            }
            Instruction::Else => {
                // The then-arm has to leave exactly what the block declares,
                // the same as the else-arm does at `end`. Skipping this check
                // is what let six modules through the checker while
                // `wasmparser` rejected them: the arm that was wrong was always
                // the first one.
                if let Some(frame) = frames.last() {
                    if !frame.unreachable {
                        let want = frame.height + frame.results.len();
                        if stack.len() != want {
                            let msg = if stack.len() > want {
                                format!(
                                    "{} value(s) left on the stack at the end of the `then` \
                                     arm (expected {}, found {})",
                                    stack.len() - want,
                                    frame.results.len(),
                                    stack.len() - frame.height
                                )
                            } else {
                                format!(
                                    "`then` arm ends short: expected {} value(s), found {}",
                                    frame.results.len(),
                                    stack.len().saturating_sub(frame.height)
                                )
                            };
                            return Some(report(i, msg, &stack));
                        }
                    }
                }
                if let Some(frame) = frames.last_mut() {
                    stack.truncate(frame.height);
                    frame.unreachable = false;
                }
                continue;
            }
            Instruction::End => {
                let Some(frame) = frames.pop() else {
                    return Some(report(i, "`end` with no open block".into(), &stack));
                };
                if !frame.unreachable {
                    let want = frame.height + frame.results.len();
                    if stack.len() != want {
                        let msg = if stack.len() > want {
                            format!(
                                "{} value(s) left on the stack at the end of this block \
                                 (expected {}, found {})",
                                stack.len() - want,
                                frame.results.len(),
                                stack.len() - frame.height
                            )
                        } else {
                            format!(
                                "block ends short: expected {} value(s), found {}",
                                frame.results.len(),
                                stack.len().saturating_sub(frame.height)
                            )
                        };
                        return Some(report(i, msg, &stack));
                    }
                }
                stack.truncate(frame.height);
                stack.extend(frame.results.iter().map(|t| (*t, i)));
                if frames.is_empty() {
                    // Function body finished.
                    if i + 1 != func.instructions.len() {
                        return Some(report(
                            i + 1,
                            "instructions after the end of the function body".into(),
                            &stack,
                        ));
                    }
                }
                continue;
            }
            Instruction::Br(_) | Instruction::BrTable(_, _) | Instruction::Return
            | Instruction::Unreachable => {
                if let Some(frame) = frames.last_mut() {
                    frame.unreachable = true;
                }
                continue;
            }
            Instruction::BrIf(_) => {
                if !poly {
                    stack.pop();
                }
                continue;
            }
            _ => {}
        }

        if poly {
            continue;
        }

        let (pops, pushes) = match super::coerce::stack_effect(
            instr, call_sig, type_sig, &local_ty, global_ty,
        ) {
            Some(e) => e,
            // Not modelled: stop tracking rather than report something
            // invented — and say so, because an unmodelled instruction is a
            // hole in this tool, not a property of the program.
            None => {
                if std::env::var("SIGIL_WASM_UNMODELLED").is_ok() {
                    eprintln!("  unmodelled in `{}` #{}: {:?}", func.name, i, instr);
                }
                if let Some(frame) = frames.last_mut() {
                    frame.unreachable = true;
                }
                continue;
            }
        };

        let floor = frames.last().map(|f| f.height).unwrap_or(0);
        if stack.len() < floor + pops {
            return Some(report(
                i,
                format!(
                    "needs {} operand(s), {} available in this block",
                    pops,
                    stack.len() - floor
                ),
                &stack,
            ));
        }
        let wants = super::coerce::operand_types(
            instr, call_sig, type_sig, &local_ty, global_ty,
        );
        for k in 0..pops {
            let (got, from) = stack.pop().unwrap();
            let Some(want) = wants.as_ref().and_then(|v| v.get(v.len() - 1 - k).copied()) else {
                continue;
            };
            if got == want {
                continue;
            }
            // A conversion inserted just after the instruction that pushed the
            // value fixes it wherever that instruction is — including when it is
            // the `end` of a block, which is exactly the case a backward scan
            // over the operand sequence cannot reach.
            match super::coerce::conversion_for(got, want) {
                Some(conv) => repairs.push((from + 1, conv)),
                None => {
                    return Some(report(
                        i,
                        format!(
                            "operand {} is {}, but this instruction takes {}",
                            pops - k,
                            ty(got),
                            ty(want)
                        ),
                        &stack,
                    ))
                }
            }
        }
        if let Some(t) = pushes {
            stack.push((t, i));
        }
    }

    None
}
