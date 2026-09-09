//! State Machine Intermediate Representation
//!
//! Core types for representing async functions as explicit state machines.
//! These types are backend-independent and can be compiled to WASM, LLVM, or interpreted.

use crate::ast::{Expr, Stmt, TypeExpr};
use std::collections::HashMap;

/// The state machine intermediate representation for an async function.
#[derive(Debug, Clone)]
pub struct StateMachineIR {
    /// Original function name
    pub name: String,
    /// Original parameters (name, type)
    pub params: Vec<(String, TypeExpr)>,
    /// Return type
    pub result_type: Option<TypeExpr>,
    /// All states in the machine
    pub states: Vec<State>,
    /// All locals declared across all states
    pub locals: Vec<LocalDecl>,
    /// Memory layout for suspension frame
    pub frame_layout: FrameLayout,
}

impl StateMachineIR {
    /// Create a new state machine IR for a function.
    pub fn new(name: String, params: Vec<(String, TypeExpr)>, result_type: Option<TypeExpr>) -> Self {
        Self {
            name,
            params,
            result_type,
            states: Vec::new(),
            locals: Vec::new(),
            frame_layout: FrameLayout::new(),
        }
    }

    /// Add a new state and return its index.
    pub fn add_state(&mut self, state: State) -> u32 {
        let idx = self.states.len() as u32;
        self.states.push(state);
        idx
    }

    /// Get the next state index that would be assigned.
    pub fn next_state_idx(&self) -> u32 {
        self.states.len() as u32
    }

    /// Declare a local variable.
    pub fn declare_local(&mut self, name: String, ty: Option<TypeExpr>, defined_in_state: u32) {
        self.locals.push(LocalDecl {
            name: name.clone(),
            ty,
            defined_in_state,
            live_until_state: defined_in_state, // Updated during liveness analysis
        });
        self.frame_layout.add_local(&name);
    }

    /// Declare a local variable only if not already declared.
    /// Returns true if the local was newly declared, false if it already existed.
    pub fn declare_local_if_new(&mut self, name: String, ty: Option<TypeExpr>, defined_in_state: u32) -> bool {
        if self.locals.iter().any(|l| l.name == name) {
            return false;
        }
        self.declare_local(name, ty, defined_in_state);
        true
    }

    /// Check if a local has been declared.
    pub fn has_local(&self, name: &str) -> bool {
        self.locals.iter().any(|l| l.name == name)
    }

    /// Get a state by index.
    pub fn get_state(&self, idx: u32) -> Option<&State> {
        self.states.get(idx as usize)
    }

    /// Get a mutable state by index.
    pub fn get_state_mut(&mut self, idx: u32) -> Option<&mut State> {
        self.states.get_mut(idx as usize)
    }

    /// Validate all invariants. Returns errors if any are violated.
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();

        // INV-1: State indices are contiguous
        for (i, state) in self.states.iter().enumerate() {
            if state.index != i as u32 {
                errors.push(format!(
                    "State at position {} has index {}, expected {}",
                    i, state.index, i
                ));
            }
        }

        // INV-2: Entry/resume flags
        if let Some(state0) = self.states.first() {
            if !state0.is_entry {
                errors.push("State 0 must have is_entry = true".to_string());
            }
            if state0.is_resume {
                errors.push("State 0 must have is_resume = false".to_string());
            }
        }

        for state in self.states.iter().skip(1) {
            if state.is_entry {
                errors.push(format!("State {} has is_entry = true, but only state 0 should", state.index));
            }
        }

        // INV-3: Resume binding consistency
        // Entry state must not have resume_binding
        if let Some(state0) = self.states.first() {
            if state0.resume_binding.is_some() {
                errors.push("State 0 (entry) must not have resume_binding".to_string());
            }
        }

        // Resume states that follow an Await should have resume_binding if the value is used
        // (Note: We only validate that entry state has no binding; the transformer ensures
        // resume states have appropriate bindings when needed)

        // INV-4: All exit targets exist
        for state in &self.states {
            for target in state.exit.target_states() {
                if target >= self.states.len() as u32 {
                    errors.push(format!(
                        "State {} references non-existent state {}",
                        state.index, target
                    ));
                }
            }
        }

        // Check for Unreachable exits (should be replaced during transformation)
        for state in &self.states {
            if matches!(state.exit, StateExit::Unreachable) {
                errors.push(format!(
                    "State {} has Unreachable exit - transformation incomplete",
                    state.index
                ));
            }
        }

        // Check for orphaned states (not reachable from entry)
        if !self.states.is_empty() {
            let mut reachable = vec![false; self.states.len()];
            let mut queue = vec![0u32]; // Start from entry state

            while let Some(idx) = queue.pop() {
                if reachable[idx as usize] {
                    continue;
                }
                reachable[idx as usize] = true;

                for target in self.states[idx as usize].exit.target_states() {
                    if (target as usize) < reachable.len() && !reachable[target as usize] {
                        queue.push(target);
                    }
                }
            }

            for (i, is_reachable) in reachable.iter().enumerate() {
                if !is_reachable {
                    errors.push(format!(
                        "State {} is not reachable from entry state",
                        i
                    ));
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// A single state in the state machine.
#[derive(Debug, Clone)]
pub struct State {
    /// State number (0 = entry)
    pub index: u32,
    /// Reachable from initial call
    pub is_entry: bool,
    /// Reachable from resume (post-await)
    pub is_resume: bool,
    /// Variable to bind the resume value to (if this state follows an await)
    pub resume_binding: Option<String>,
    /// Statements to execute in this state
    pub body: Vec<Stmt>,
    /// How this state exits
    pub exit: StateExit,
}

impl State {
    /// Create a new entry state (state 0).
    pub fn entry() -> Self {
        Self {
            index: 0,
            is_entry: true,
            is_resume: false,
            resume_binding: None,
            body: Vec::new(),
            exit: StateExit::Unreachable, // Must be set
        }
    }

    /// Create a new resume state (post-await).
    pub fn resume(index: u32) -> Self {
        Self {
            index,
            is_entry: false,
            is_resume: true,
            resume_binding: None,
            body: Vec::new(),
            exit: StateExit::Unreachable, // Must be set
        }
    }

    /// Create a new intermediate state (neither entry nor resume).
    /// Used for branch targets in conditionals.
    pub fn intermediate(index: u32) -> Self {
        Self {
            index,
            is_entry: false,
            is_resume: false,
            resume_binding: None,
            body: Vec::new(),
            exit: StateExit::Unreachable,
        }
    }
}

/// How a state exits.
#[derive(Debug, Clone)]
pub enum StateExit {
    /// Suspend at await, resume in next_state.
    Await {
        /// The promise expression to await
        promise: Expr,
        /// State to resume in when promise resolves
        next_state: u32,
        /// Locals to save before suspension
        saved_locals: Vec<String>,
    },

    /// Return final value, function complete.
    Return {
        /// The value to return
        value: Expr,
    },

    /// Unconditional transition to another state.
    Goto {
        /// Target state
        target: u32,
    },

    /// Conditional transition.
    Branch {
        /// Condition to evaluate
        condition: Expr,
        /// State if condition is true
        then_state: u32,
        /// State if condition is false
        else_state: u32,
    },

    /// Loop head (condition checked here).
    LoopHead {
        /// Condition (None for infinite loop)
        condition: Option<Expr>,
        /// State for loop body
        body_state: u32,
        /// State after loop exits
        exit_state: u32,
    },

    /// Placeholder - must be replaced before validation.
    Unreachable,
}

impl StateExit {
    /// Get all target state indices referenced by this exit.
    pub fn target_states(&self) -> Vec<u32> {
        match self {
            StateExit::Await { next_state, .. } => vec![*next_state],
            StateExit::Return { .. } => vec![],
            StateExit::Goto { target } => vec![*target],
            StateExit::Branch { then_state, else_state, .. } => vec![*then_state, *else_state],
            StateExit::LoopHead { body_state, exit_state, .. } => vec![*body_state, *exit_state],
            StateExit::Unreachable => vec![],
        }
    }

    /// Check if this is a suspension point (await).
    pub fn is_await(&self) -> bool {
        matches!(self, StateExit::Await { .. })
    }

    /// Check if this is a terminal exit (return).
    pub fn is_return(&self) -> bool {
        matches!(self, StateExit::Return { .. })
    }
}

/// Declaration of a local variable in the state machine.
#[derive(Debug, Clone)]
pub struct LocalDecl {
    /// Variable name
    pub name: String,
    /// Variable type (if known)
    pub ty: Option<TypeExpr>,
    /// First state where this variable is defined
    pub defined_in_state: u32,
    /// Last state where this variable is used
    pub live_until_state: u32,
}

impl LocalDecl {
    /// Check if this local is live in a given state.
    pub fn is_live_in(&self, state: u32) -> bool {
        state >= self.defined_in_state && state <= self.live_until_state
    }
}

/// Memory layout for the suspension frame.
#[derive(Debug, Clone)]
pub struct FrameLayout {
    /// Byte offset of the state field (always 0)
    pub state_offset: u32,
    /// Byte offset where locals begin
    pub locals_offset: u32,
    /// Map from local name to byte offset
    pub local_offsets: HashMap<String, u32>,
    /// Total frame size in bytes
    pub total_size: u32,
}

impl FrameLayout {
    /// Create a new frame layout.
    pub fn new() -> Self {
        Self {
            state_offset: 0,
            locals_offset: 8, // After state (i32) + padding (i32)
            local_offsets: HashMap::new(),
            total_size: 8,
        }
    }

    /// Add a local to the frame (all locals are 8 bytes / i64).
    pub fn add_local(&mut self, name: &str) {
        let offset = self.total_size;
        self.local_offsets.insert(name.to_string(), offset);
        self.total_size += 8;
    }

    /// Get the byte offset for a local.
    pub fn get_offset(&self, name: &str) -> Option<u32> {
        self.local_offsets.get(name).copied()
    }

    /// Check if a local has an offset in the frame.
    pub fn has_offset(&self, name: &str) -> bool {
        self.local_offsets.contains_key(name)
    }
}

impl Default for FrameLayout {
    fn default() -> Self {
        Self::new()
    }
}
