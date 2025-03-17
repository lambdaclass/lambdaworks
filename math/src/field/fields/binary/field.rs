use core::marker::PhantomData;
use core::ops::{Add, Mul, Neg, Sub};

pub enum BinaryFieldError {
    /// Attempt to create an invlaid field element.
    InavlidElement,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct TowerFieldElement {
    /// The underlying value
    pub value: u128,
    /// Number of levels in the tower.
    /// It's the degree of the field extension that the element belongs to.
    pub num_level: usize,
    /// Number of bits (2^num_levels)
    /// It's the order of the field extension that the element belongs to.
    /// Is it better to have a function num_bits(){ 1<<num_levels }?
    pub num_bits: usize,
}

impl TowerFieldElement {
    // TODO: Choose which new() we want.
    pub fn new_2(value: u128, num_level: usize) -> Result<Self, BinaryFieldError> {
        let num_bits = 1 << num_level;
        // If val doesn't fit in the level given as num_level:
        if val > (1 << bits) - 1 {
            return Err(BinaryFieldError::InavlidElement);
        } else {
            Ok(Self {
                value,
                num_level,
                num_bits,
            })
        }
    }

    // Based on ingonyama.
    pub fn new(val: u128, num_level: usize) -> Self {
        let bits = 1 << num_level;
        let mask = if bits >= 128 {
            u128::MAX
        } else {
            // For example, if bits = 8. (1 << bits) - 1 = 11111111
            (1 << bits) - 1
        };
        Self {
            value: val & mask,
            num_level: level,
            num_bits: bits,
        }
    }

    /// Returns true if the element is zero
    pub fn is_zero(&self) -> bool {
        self.value == 0
    }

    /// Returns true if this element is one
    #[inline]
    pub fn is_one(&self) -> bool {
        self.value == 1
    }

    /// Returns the underlying value
    #[inline]
    pub fn value(&self) -> u128 {
        self.value
    }

    /// Returns number of levels in the tower
    #[inline]
    pub fn num_level(&self) -> usize {
        self.num_level
    }

    /// Returns number of bits (2^num_levels)
    #[inline]
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    // Equality check
    fn equals(&self, other: &Self) -> bool {
        self.val == other.get_val()
    }

    // Example:
    // Let's say we want to calculate a + b, with:
    // a: num_bits = 32, num_level = 5.
    // b: num_bits = 8, num_level = 3.
    // Then a + b = a_hi || (a_lo + b), where:
    // a_hi are the first 24 msb of a.
    // a_lo are the last 8 lsb of a.
    fn add_(&self, other: &Self) -> Self {
        if self.num_level > other.num_level {
            let mask = (1 << other.num_bits) - 1;
            // Lsb of a: get the last other.num_bits bits
            let low = self.value & mask;
            // Perform the addition (XOR) on the lower part
            let result_low = low ^ other.value;
            // Msb of a: get the remaining bits
            let high = self.value >> other.num_bits;
            // Combine the results
            let result_value = (high << other.num_bits) | result_low;

            Self {
                value: result_value,
                num_level: self.num_level,
                num_bits: self.num_bits,
            }
        } else if self.num_level < other.num_level {
            // If other is larger, delegate to its add method
            other.add(self)
        } else {
            // Same size, just XOR
            Self {
                value: self.value ^ other.value,
                num_level: self.num_level,
                num_bits: self.num_bits,
            }
        }
    }

    /// Returns binary string representation
    pub fn to_binary_string(&self) -> String {
        format!("{:0width$b}", self.value, width = self.num_bits)
    }

    /// Splits element into high and low parts
    pub fn split(&self) -> (Self, Self) {
        let half_bits = self.num_bits / 2;
        let mask = (1 << half_bits) - 1;
        let lo = self.value & mask;
        let hi = (self.value >> half_bits) & mask;

        (Self::new(hi, num_level), Self::new(lo, num_level))
    }

    /// Joins with another element as the low part
    pub fn join(&self, low: &Self) -> Self {
        let new_bits = self.num_bits * 2;
        let joined = (self.value << self.num_bits) | low.value;
        Self {
            value: joined,
            num_level: self.num_level + 1,
            num_bits: new_bits,
        }
    }
}

impl Default for TowerFieldElement {
    fn default() -> Self {
        Self::new(0, None)
    }
}
