//! The one way this backend touches a Vec or a HashMap.
//!
//! S57: there used to be two array representations and the backend mixed them.
//! Array literals, `∀ x ∈ xs`, indexing and the `Pattern::Slice` binding read a
//! **linear-memory** layout — a 4-byte length followed by 8-byte slots — while
//! `push`, `filter`, `map`, `len`, `sort` and `join` used the **host-side**
//! `morpheme` array imports, where a Vec is an id in a JS map. A Vec built by
//! one was not readable by the other, and `HashMap` had only the host form, so
//! `∀ (k, v) ∈ attrs` could not iterate one at all.
//!
//! Host-side wins, for four reasons. Every collection *operation* was already
//! there; the rest of the value model already is (vnodes, JSON, DOM nodes, maps
//! are all host ids); growth is free, where a memory Vec needs a real allocator
//! and this backend has a bump pointer and no `free`; and `HashMap` has no
//! memory layout at all, so choosing memory means writing a hash map in
//! WebAssembly.
//!
//! Everything below leaves the uniform i64 on the stack that the rest of the
//! backend expects, and takes the same.

use wasm_encoder::{Instruction, ValType};

use super::error::{WasmError, WasmResult};
use super::WasmCompiler;

impl WasmCompiler {
    fn collection_import(&self, name: &str) -> WasmResult<u32> {
        self.imports
            .get_func(name)
            .ok_or_else(|| WasmError::internal(format!("{} import not found", name)))
    }

    /// A new empty array. Leaves its handle as i64.
    pub(crate) fn emit_array_new(&mut self) -> WasmResult<()> {
        let idx = self.collection_import("array_new")?;
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        func.push(Instruction::Call(idx));
        func.push(Instruction::I64ExtendI32U);
        Ok(())
    }

    /// A new empty map. Leaves its handle as i64.
    pub(crate) fn emit_map_new(&mut self) -> WasmResult<()> {
        let idx = self.collection_import("map_new")?;
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        func.push(Instruction::Call(idx));
        func.push(Instruction::I64ExtendI32U);
        Ok(())
    }

    /// A new empty set. Leaves its handle as i64.
    pub(crate) fn emit_set_new(&mut self) -> WasmResult<()> {
        let idx = self.collection_import("set_new")?;
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        func.push(Instruction::Call(idx));
        func.push(Instruction::I64ExtendI32U);
        Ok(())
    }

    /// `HashSet·from(xs)`. Stack: `[… src]` → `[… set]`.
    pub(crate) fn emit_set_from(&mut self) -> WasmResult<()> {
        let idx = self.collection_import("set_from")?;
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::Call(idx));
        func.push(Instruction::I64ExtendI32U);
        Ok(())
    }

    /// `HashMap·from(entries)`. Stack: `[… src]` → `[… map]`.
    pub(crate) fn emit_map_from(&mut self) -> WasmResult<()> {
        let idx = self.collection_import("map_from")?;
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::Call(idx));
        func.push(Instruction::I64ExtendI32U);
        Ok(())
    }

    /// Stack: `[… arr]` → `[… len]`.
    pub(crate) fn emit_array_len(&mut self) -> WasmResult<()> {
        let idx = self.collection_import("array_len")?;
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::Call(idx));
        func.push(Instruction::I64ExtendI32U);
        Ok(())
    }

    /// Stack: `[… arr idx]` → `[… value]`.
    pub(crate) fn emit_array_get(&mut self) -> WasmResult<()> {
        let idx = self.collection_import("array_get")?;
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        // Both parameters are i32; the index is on top.
        func.push(Instruction::I32WrapI64);
        let i = func.alloc_local("__coll_i".to_string(), ValType::I32);
        func.push(Instruction::LocalSet(i));
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::LocalGet(i));
        func.push(Instruction::Call(idx));
        Ok(())
    }

    /// Stack: `[… arr idx value]` → `[…]`.
    pub(crate) fn emit_array_set(&mut self) -> WasmResult<()> {
        let idx = self.collection_import("array_set")?;
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        let v = func.alloc_local("__coll_v".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(v));
        func.push(Instruction::I32WrapI64);
        let i = func.alloc_local("__coll_i".to_string(), ValType::I32);
        func.push(Instruction::LocalSet(i));
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::LocalGet(i));
        func.push(Instruction::LocalGet(v));
        func.push(Instruction::Call(idx));
        Ok(())
    }

    /// Stack: `[… arr value]` → `[…]`.
    pub(crate) fn emit_array_push(&mut self) -> WasmResult<()> {
        let idx = self.collection_import("array_push")?;
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        let v = func.alloc_local("__coll_v".to_string(), ValType::I64);
        func.push(Instruction::LocalSet(v));
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::LocalGet(v));
        func.push(Instruction::Call(idx));
        Ok(())
    }

    /// Stack: `[… collection]` → `[… array]`.
    ///
    /// What `∀` iterates. A map becomes its entries — an array of two-element
    /// arrays — and an array is already itself, so `∀ x ∈ xs` and
    /// `∀ (k, v) ∈ m` are the same loop over different contents.
    pub(crate) fn emit_iterable(&mut self) -> WasmResult<()> {
        let idx = self.collection_import("iter_of")?;
        let func = self
            .current_function_mut()
            .ok_or_else(|| WasmError::internal("not in function context"))?;
        func.push(Instruction::I32WrapI64);
        func.push(Instruction::Call(idx));
        func.push(Instruction::I64ExtendI32U);
        Ok(())
    }
}
