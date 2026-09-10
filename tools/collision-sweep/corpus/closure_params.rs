// #84/#89: a closure parameter is a binding, not a field. `this` binds
// cleanly in Sigil, so none of these may be reported -- develop carried this
// as a live false positive, found in cranelift-isle and rustc-demangle.

pub fn untyped() -> u32 {
    let f = |this| this;
    f(1)
}

pub fn typed() -> u32 {
    let f = |this: u32| this;
    f(2)
}

pub fn several() -> u32 {
    let f = |this, of| this + of;
    f(1, 2)
}

// #91: `mut` between the bar and the name. The byte before the name is the
// `t` of `mut`, so the untyped test does not see it, and there is no colon.
pub fn with_mut() -> u32 {
    let f = |mut tome| { tome += 1; tome };
    f(3)
}

pub fn with_mut_typed() -> u32 {
    let f = |mut tome: u32| { tome += 1; tome };
    f(4)
}
