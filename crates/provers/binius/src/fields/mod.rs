//! Binary Field Implementations for Binius
//!
//! This module provides the binary field infrastructure needed for Binius,
//! leveraging the existing TowerFieldElement implementation.

pub mod tower {
    pub use lambdaworks_math::field::fields::binary::field::{BinaryFieldError, TowerFieldElement};

    pub type Tower = TowerFieldElement;

    pub fn new(value: u128, level: usize) -> Tower {
        TowerFieldElement::new(value, level)
    }

    pub fn zero() -> Tower {
        TowerFieldElement::zero()
    }

    pub fn one() -> Tower {
        TowerFieldElement::one()
    }

    pub fn inv(elem: &Tower) -> Result<Tower, BinaryFieldError> {
        elem.inv()
    }

    pub fn pow(elem: &Tower, exp: u32) -> Tower {
        elem.pow(exp)
    }

    pub fn is_zero(elem: &Tower) -> bool {
        elem.is_zero()
    }

    pub fn is_one(elem: &Tower) -> bool {
        elem.is_one()
    }

    pub fn value(elem: &Tower) -> u128 {
        elem.value()
    }

    pub fn num_level(elem: &Tower) -> usize {
        elem.num_level()
    }

    pub fn num_bits(elem: &Tower) -> usize {
        elem.num_bits()
    }

    pub fn split(elem: &Tower) -> (Tower, Tower) {
        elem.split()
    }

    pub fn join(high: &Tower, low: &Tower) -> Tower {
        high.join(low)
    }
}

/// Represents a specific level in the binary field tower
/// Level 0: GF(2) - 2 elements
/// Level 1: GF(2^2) - 4 elements
/// Level 2: GF(2^4) - 16 elements
/// Level 3: GF(2^8) - 256 elements
/// Level 4: GF(2^16) - 65,536 elements
/// Level 5: GF(2^32) - ~4 billion elements
/// Level 6: GF(2^64) - ~18 quintillion elements
/// Level 7: GF(2^128) - ~340 undecillion elements
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FieldLevel(pub usize);

impl FieldLevel {
    pub const GF2: Self = Self(0);
    pub const GF4: Self = Self(1);
    pub const GF16: Self = Self(2);
    pub const GF256: Self = Self(3);
    pub const GF65536: Self = Self(4);
    pub const GF2_32: Self = Self(5);
    pub const GF2_64: Self = Self(6);
    pub const GF2_128: Self = Self(7);

    pub fn num_bits(&self) -> usize {
        1 << self.0
    }

    pub fn order(&self) -> u128 {
        1u128 << self.num_bits()
    }
}

pub use tower::{BinaryFieldError, Tower, TowerFieldElement};
