//! Coerce call operands to the types their callee declares.
//!
//! The backend represents every Sigil value as an i64 on the WASM stack, but its
//! own imports are declared with real types — `array_push(i32, i64)`,
//! `math_round(f64) -> f64`, `string_slice(i32, i64, i64)`. There are 66
//! hand-rolled import call sites and no shared coercion between them, so whether
//! a module validated was decided per site: **not one of the 86 modules the
//! Lares client compiled to was valid WebAssembly** (S46), and the failures were
//! spread across a dozen different signatures.
//!
//! This is the shared mechanism. After a function is compiled, walk its
//! instructions; at each `call`, find where each operand begins by stepping
//! backwards through the operand stack effects, and splice in a conversion where
//! the value produced does not match the parameter declared.
//!
//! It is deliberately conservative. An operand sequence containing control flow
//! is left alone rather than guessed at — a partial repair that is always right
//! is worth more than a total one that is sometimes wrong.

use std::collections::HashMap;
use wasm_encoder::{Instruction, ValType};

/// What one instruction does to the operand stack: how many values it consumes,
/// and the type of the single value it produces (if any).
pub fn stack_effect(
    instr: &Instruction<'static>,
    call_sig: &dyn Fn(u32) -> Option<(Vec<ValType>, Vec<ValType>)>,
    type_sig: &dyn Fn(u32) -> Option<(Vec<ValType>, Vec<ValType>)>,
    local_ty: &dyn Fn(u32) -> Option<ValType>,
    global_ty: &dyn Fn(u32) -> Option<ValType>,
) -> Option<(usize, Option<ValType>)> {
    use Instruction as I;
    use ValType::*;
    Some(match instr {
        I::I32Const(_) => (0, Some(I32)),
        I::I64Const(_) => (0, Some(I64)),
        I::F32Const(_) => (0, Some(F32)),
        I::F64Const(_) => (0, Some(F64)),

        I::LocalGet(i) => (0, Some(local_ty(*i)?)),
        I::LocalTee(i) => (1, Some(local_ty(*i)?)),
        I::LocalSet(_) => (1, None),
        I::GlobalGet(i) => (0, Some(global_ty(*i)?)),
        I::GlobalSet(_) => (1, None),

        I::I32Load(_) => (1, Some(I32)),
        I::I64Load(_) => (1, Some(I64)),
        I::I32Store(_) | I::I64Store(_) => (2, None),

        I::Drop => (1, None),
        I::Select => (3, None), // type depends on operands; treat as opaque

        // Conversions
        I::I32WrapI64 => (1, Some(I32)),
        I::I64ExtendI32S | I::I64ExtendI32U => (1, Some(I64)),
        I::F64ConvertI32S | I::F64ConvertI32U | I::F64ConvertI64S | I::F64ConvertI64U => {
            (1, Some(F64))
        }
        I::F32ConvertI32S | I::F32ConvertI32U | I::F32ConvertI64S | I::F32ConvertI64U => {
            (1, Some(F32))
        }
        I::I32TruncF32S | I::I32TruncF32U | I::I32TruncF64S | I::I32TruncF64U => (1, Some(I32)),
        I::I64TruncF32S | I::I64TruncF32U | I::I64TruncF64S | I::I64TruncF64U => (1, Some(I64)),
        I::F32DemoteF64 => (1, Some(F32)),
        I::F64PromoteF32 => (1, Some(F64)),
        I::I32ReinterpretF32 => (1, Some(I32)),
        I::I64ReinterpretF64 => (1, Some(I64)),
        I::F32ReinterpretI32 => (1, Some(F32)),
        I::F64ReinterpretI64 => (1, Some(F64)),

        // i32 arithmetic / comparison
        I::I32Add | I::I32Sub | I::I32Mul | I::I32DivS | I::I32DivU | I::I32RemS | I::I32RemU
        | I::I32And | I::I32Or | I::I32Xor | I::I32Shl | I::I32ShrS | I::I32ShrU => (2, Some(I32)),
        I::I32Clz | I::I32Ctz | I::I32Popcnt | I::I32Eqz => (1, Some(I32)),
        I::I32Eq | I::I32Ne | I::I32LtS | I::I32LtU | I::I32GtS | I::I32GtU | I::I32LeS
        | I::I32LeU | I::I32GeS | I::I32GeU => (2, Some(I32)),

        // i64 arithmetic; comparisons yield i32, as WebAssembly defines them
        I::I64Add | I::I64Sub | I::I64Mul | I::I64DivS | I::I64DivU | I::I64RemS | I::I64RemU
        | I::I64And | I::I64Or | I::I64Xor | I::I64Shl | I::I64ShrS | I::I64ShrU => (2, Some(I64)),
        I::I64Clz | I::I64Ctz | I::I64Popcnt => (1, Some(I64)),
        I::I64Eqz => (1, Some(I32)),
        I::I64Eq | I::I64Ne | I::I64LtS | I::I64LtU | I::I64GtS | I::I64GtU | I::I64LeS
        | I::I64LeU | I::I64GeS | I::I64GeU => (2, Some(I32)),

        // floats
        I::F64Add | I::F64Sub | I::F64Mul | I::F64Div => (2, Some(F64)),
        I::F64Abs | I::F64Neg | I::F64Ceil | I::F64Floor | I::F64Sqrt => (1, Some(F64)),
        I::F64Eq | I::F64Ne | I::F64Lt | I::F64Gt | I::F64Le | I::F64Ge => (2, Some(I32)),
        I::F32Add | I::F32Sub | I::F32Mul | I::F32Div => (2, Some(F32)),
        I::F32Abs | I::F32Neg | I::F32Ceil | I::F32Floor | I::F32Sqrt => (1, Some(F32)),
        I::F32Eq | I::F32Ne | I::F32Lt | I::F32Gt | I::F32Le | I::F32Ge => (2, Some(I32)),

        I::Call(idx) => {
            let (params, results) = call_sig(*idx)?;
            if results.len() > 1 {
                return None;
            }
            (params.len(), results.first().copied())
        }
        I::CallIndirect { type_index, .. } => {
            let (params, results) = type_sig(*type_index)?;
            if results.len() > 1 {
                return None;
            }
            (params.len() + 1, results.first().copied())
        }

        // Everything else — control flow above all — is where this stops.
        _ => return None,
    })
}


/// The operand types an instruction requires, when they are fixed by the
/// instruction itself. `None` means "not modelled" — those are left alone.
pub fn operand_types(
    instr: &Instruction<'static>,
    call_sig: &dyn Fn(u32) -> Option<(Vec<ValType>, Vec<ValType>)>,
    type_sig: &dyn Fn(u32) -> Option<(Vec<ValType>, Vec<ValType>)>,
    local_ty: &dyn Fn(u32) -> Option<ValType>,
    global_ty: &dyn Fn(u32) -> Option<ValType>,
) -> Option<Vec<ValType>> {
    use Instruction as I;
    use ValType::*;
    Some(match instr {
        I::Call(idx) => call_sig(*idx)?.0,
        // The table index is the last operand and is always i32; the ones
        // before it come from the signature the call names.
        I::CallIndirect { type_index, .. } => {
            let mut p = type_sig(*type_index)?.0;
            p.push(I32);
            p
        }
        I::LocalSet(i) | I::LocalTee(i) => vec![local_ty(*i)?],
        I::GlobalSet(i) => vec![global_ty(*i)?],

        I::I64Store(_) => vec![I32, I64],
        I::I32Store(_) => vec![I32, I32],
        I::I64Load(_) | I::I32Load(_) => vec![I32],

        I::I32WrapI64 => vec![I64],
        I::I64ExtendI32S | I::I64ExtendI32U => vec![I32],
        I::F64ConvertI64S | I::F64ConvertI64U => vec![I64],
        I::F64ConvertI32S | I::F64ConvertI32U => vec![I32],
        I::I64TruncF64S | I::I64TruncF64U | I::I32TruncF64S | I::I32TruncF64U => vec![F64],

        I::I64Add | I::I64Sub | I::I64Mul | I::I64DivS | I::I64DivU | I::I64RemS | I::I64RemU
        | I::I64And | I::I64Or | I::I64Xor | I::I64Shl | I::I64ShrS | I::I64ShrU | I::I64Eq
        | I::I64Ne | I::I64LtS | I::I64LtU | I::I64GtS | I::I64GtU | I::I64LeS | I::I64LeU
        | I::I64GeS | I::I64GeU => vec![I64, I64],
        I::I64Clz | I::I64Ctz | I::I64Popcnt | I::I64Eqz => vec![I64],

        I::I32Add | I::I32Sub | I::I32Mul | I::I32DivS | I::I32DivU | I::I32RemS | I::I32RemU
        | I::I32And | I::I32Or | I::I32Xor | I::I32Shl | I::I32ShrS | I::I32ShrU | I::I32Eq
        | I::I32Ne | I::I32LtS | I::I32LtU | I::I32GtS | I::I32GtU | I::I32LeS | I::I32LeU
        | I::I32GeS | I::I32GeU => vec![I32, I32],
        I::I32Clz | I::I32Ctz | I::I32Popcnt | I::I32Eqz => vec![I32],

        I::F64Add | I::F64Sub | I::F64Mul | I::F64Div | I::F64Eq | I::F64Ne | I::F64Lt
        | I::F64Gt | I::F64Le | I::F64Ge => vec![F64, F64],
        I::F64Abs | I::F64Neg | I::F64Ceil | I::F64Floor | I::F64Sqrt => vec![F64],

        _ => return None,
    })
}

/// The instruction that turns `from` into `to`, if one exists.
pub fn conversion_for(from: ValType, to: ValType) -> Option<Instruction<'static>> {
    conversion(from, to)
}

fn conversion(from: ValType, to: ValType) -> Option<Instruction<'static>> {
    use ValType::*;
    Some(match (from, to) {
        (I64, I32) => Instruction::I32WrapI64,
        (I32, I64) => Instruction::I64ExtendI32U,
        (I64, F64) => Instruction::F64ConvertI64S,
        (I32, F64) => Instruction::F64ConvertI32S,
        (F64, I64) => Instruction::I64TruncF64S,
        (F64, I32) => Instruction::I32TruncF64S,
        (I64, F32) => Instruction::F32ConvertI64S,
        (I32, F32) => Instruction::F32ConvertI32S,
        (F32, I64) => Instruction::I64TruncF32S,
        (F32, I32) => Instruction::I32TruncF32S,
        (F32, F64) => Instruction::F64PromoteF32,
        (F64, F32) => Instruction::F32DemoteF64,
        _ => return None,
    })
}


/// Bring every host-call result back to the i64 the rest of the stack uses.
///
/// The imports are declared with real types — `math_min(f64, f64) -> f64`,
/// `array_len(i32) -> i32` — and their results were left as they came. A
/// `math_min(…)` in one arm of a `⎇` against an i64 in the other is a branch
/// type mismatch the operand repair cannot see, because it does not cross
/// control flow.
///
/// Safe to do unconditionally *because* `repair_operand_types` runs after it: a
/// consumer that genuinely wants the original type gets a conversion back. The
/// pair is the convention, not either half.
pub fn normalise_call_results(
    instructions: &mut Vec<Instruction<'static>>,
    import_count: u32,
    call_sig: &dyn Fn(u32) -> Option<(Vec<ValType>, Vec<ValType>)>,
) -> usize {
    let mut inserts: Vec<(usize, Instruction<'static>)> = Vec::new();
    for pos in 0..instructions.len() {
        let Instruction::Call(idx) = instructions[pos] else {
            continue;
        };
        // Host imports only. A user function's result type is already whatever
        // the backend chose for it.
        if idx >= import_count {
            continue;
        }
        let Some((_, results)) = call_sig(idx) else {
            continue;
        };
        let [got] = results[..] else { continue };
        if got == ValType::I64 {
            continue;
        }
        let Some(conv) = conversion(got, ValType::I64) else {
            continue;
        };
        // Already converted by the emitting site. `Instruction` has no
        // `PartialEq`, so compare what it does rather than what it is.
        let already = instructions
            .get(pos + 1)
            .and_then(|next| stack_effect(next, call_sig, &|_| None, &|_| None, &|_| None))
            .map(|(pops, pushed)| pops == 1 && pushed == Some(ValType::I64))
            .unwrap_or(false);
        if already {
            continue;
        }
        inserts.push((pos + 1, conv));
    }
    let count = inserts.len();
    for (at, instr) in inserts.into_iter().rev() {
        instructions.insert(at, instr);
    }
    count
}

/// Insert operand conversions so every `call` agrees with its callee.
///
/// Returns the number of conversions inserted.
pub fn repair_operand_types(
    instructions: &mut Vec<Instruction<'static>>,
    results: &[ValType],
    call_sig: &dyn Fn(u32) -> Option<(Vec<ValType>, Vec<ValType>)>,
    type_sig: &dyn Fn(u32) -> Option<(Vec<ValType>, Vec<ValType>)>,
    local_ty: &dyn Fn(u32) -> Option<ValType>,
    global_ty: &dyn Fn(u32) -> Option<ValType>,
) -> usize {
    // Collect insertions first: splicing while scanning would invalidate the
    // indices the scan is built on.
    let mut inserts: Vec<(usize, Instruction<'static>)> = Vec::new();

    for pos in 0..instructions.len() {
        let Some(params) = operand_types(&instructions[pos], call_sig, type_sig, local_ty, global_ty) else {
            continue;
        };
        if params.is_empty() {
            continue;
        }

        // Step backwards over the operands. `boundary[k]` ends up as the index
        // just past the sub-sequence that produced parameter k, which is where a
        // conversion for it belongs.
        let mut produced: Vec<(usize, ValType)> = Vec::new();
        let mut i = pos;
        let mut ok = true;
        for _ in 0..params.len() {
            if i == 0 {
                ok = false;
                break;
            }
            // Consume one whole operand: the instruction at i-1 produces it,
            // and whatever that instruction pops must be consumed too.
            let mut need = 1usize;
            let mut top: Option<ValType> = None;
            while need > 0 {
                if i == 0 {
                    ok = false;
                    break;
                }
                i -= 1;
                let Some((pops, pushed)) =
                    stack_effect(&instructions[i], call_sig, type_sig, local_ty, global_ty)
                else {
                    ok = false;
                    break;
                };
                match pushed {
                    Some(t) => {
                        if top.is_none() {
                            top = Some(t);
                        }
                        need -= 1;
                    }
                    // An instruction that produces nothing cannot be part of an
                    // operand sequence read backwards.
                    None => {
                        ok = false;
                        break;
                    }
                }
                need += pops;
            }
            if !ok {
                break;
            }
            match top {
                Some(t) => produced.push((i, t)),
                None => {
                    ok = false;
                    break;
                }
            }
        }
        if !ok || produced.len() != params.len() {
            continue;
        }

        // `produced` is in reverse parameter order, and each entry's index is
        // where that operand's sub-sequence STARTS. The conversion goes after it
        // ends, which is where the next operand starts — or at the call itself
        // for the last parameter.
        let mut ends: Vec<usize> = Vec::with_capacity(produced.len());
        for k in 0..produced.len() {
            ends.push(if k == 0 { pos } else { produced[k - 1].0 });
        }
        for (k, (_, got)) in produced.iter().enumerate() {
            let want = params[params.len() - 1 - k];
            if *got != want {
                if let Some(conv) = conversion(*got, want) {
                    inserts.push((ends[k], conv));
                }
            }
        }
    }

    // The value a function falls through with has to match its declared result.
    // A call returning f64 (`math_round`) or i32 (`array_len`) in tail position
    // reported "type error in fallthru".
    if let Some(&want) = results.first() {
        // A function body ends with `End`; the value it falls through with is
        // produced by the instruction before it.
        let mut at = instructions.len();
        while at > 0 && matches!(instructions[at - 1], Instruction::End) {
            at -= 1;
        }
        if at > 0 {
            let last = &instructions[at - 1];
            if !matches!(last, Instruction::Return | Instruction::Unreachable) {
                if let Some((_, Some(got))) = stack_effect(last, call_sig, type_sig, local_ty, global_ty)
                {
                    if got != want {
                        if let Some(conv) = conversion(got, want) {
                            inserts.push((at, conv));
                        }
                    }
                }
            }
        }
    }

    if inserts.is_empty() {
        return 0;
    }

    // Apply back to front so earlier indices stay valid. Ties keep their
    // relative order, which matters when two operands convert at the same point.
    inserts.sort_by(|a, b| b.0.cmp(&a.0));
    let count = inserts.len();
    let mut grouped: HashMap<usize, Vec<Instruction<'static>>> = HashMap::new();
    for (at, instr) in inserts {
        grouped.entry(at).or_default().push(instr);
    }
    let mut points: Vec<usize> = grouped.keys().copied().collect();
    points.sort_unstable_by(|a, b| b.cmp(a));
    for at in points {
        let instrs = grouped.remove(&at).unwrap();
        for instr in instrs.into_iter().rev() {
            instructions.insert(at, instr);
        }
    }
    count
}
