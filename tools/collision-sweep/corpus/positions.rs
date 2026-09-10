// One name, two positions, opposite verdicts. `tome` is a valid Sigil field
// and an invalid binding; `this` is the other way round. A scanner with one
// verdict per word gets one of these wrong whichever way it decides.

pub struct Doc {
    pub tome: String,
    pub this: u32,
}

pub fn load(tome: &str) -> usize {
    tome.len()
}

pub fn read(this: u32) -> u32 {
    this
}
