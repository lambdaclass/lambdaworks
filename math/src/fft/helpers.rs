pub fn log2(n: usize) -> u64 {
    assert!(n.is_power_of_two(), "n must be a power of two");
    n.trailing_zeros() as u64
}
