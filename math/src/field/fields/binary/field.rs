use core::marker::PhantomData;
use core::ops::{Add, Mul, Neg, Sub};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TowerFieldElement {
    /// The underlying value
    pub value: u128,
    /// Number of levels in the tower
    pub num_levels: usize,
    /// Number of bits (2^num_levels)
    pub num_bits: usize,
}

impl TowerFieldElement {
    pub fn new(val: u128, num_levels: Option<usize>) -> Self {
        let levels = num_levels.unwrap_or(3);
        let bits = 1 << levels;
        let modulus = if bits >= 128 {
            u128::MAX
        } else {
            (1 << bits) - 1
        };
        Self {
            value: val & modulus,
            num_levels: levels,
            num_bits: bits,
        }
    }

    /// Returns true if the element is zero
    pub fn is_zero(&self) -> bool {
        self.value == 0
    }
}

impl Default for TowerFieldElement {
    fn default() -> Self {
        Self::new(0, None)
    }
}
