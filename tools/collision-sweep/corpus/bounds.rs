// A `where` bound puts a name in front of a colon inside a trait body, which
// otherwise reads exactly like a field declaration.

pub trait Sized2 {
    fn probe(&self)
    where
        Self: Sized;
}

pub trait Bounded<T>
where
    T: Clone,
{
    fn get(&self) -> T;
}
