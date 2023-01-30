#[cfg(debug_assertions)]
macro_rules! cgt_assert {
    ($($arg:expr)*) => (assert!($($arg)*);)
}

#[cfg(not(debug_assertions))]
macro_rules! cgt_assert {
    ($($arg:expr)*) => {};
}
