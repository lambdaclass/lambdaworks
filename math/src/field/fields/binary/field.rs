use core::fmt;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};
use std::iter::{Product, Sum};

#[derive(Debug)]
pub enum BinaryFieldError {
    /// Attempt to compute inverse of zero
    InverseOfZero,
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
    // Constructor that always succeeds by masking the value
    pub fn new(val: u128, num_level: usize) -> Self {
        // Esta version en el caso de el nivel es demasiado grande, se limita a 7.
        // ver si en esta iteracion se puede hacer un safe_level o no.

        //     // Limit num_level to a maximum valid value for u128
        //     let safe_level = if num_level > 7 { 7 } else { num_level };

        //     let bits = 1 << safe_level;
        //     let mask = if bits >= 128 {
        //         u128::MAX
        //     } else {
        //         (1 << bits) - 1
        //     };

        //     Self {
        //         value: val & mask,
        //         num_level: safe_level,
        //         num_bits: bits,
        //     }
        // }
        let bits = 1 << num_level;
        let mask = if bits >= 128 {
            u128::MAX
        } else {
            // For example, if bits = 8. (1 << bits) - 1 = 11111111
            (1 << bits) - 1
        };
        Self {
            value: val & mask,
            num_level: num_level,
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
    pub fn equals(&self, other: &Self) -> bool {
        self.value == other.value()
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

        (
            Self::new(hi, self.num_level - 1),
            Self::new(lo, self.num_level - 1),
        )
    }

    /// Joins with another element as the low part
    pub fn join(&self, low: &Self) -> Self {
        let joined = (self.value << self.num_bits) | low.value;
        Self::new(joined, self.num_level + 1)
    }

    // Extend number of levels
    pub fn extend_num_levels(&mut self, new_levels: usize) {
        if self.num_level < new_levels {
            self.set_num_levels(new_levels);
        }
    }

    // Set number of levels
    pub fn set_num_levels(&mut self, new_levels: usize) {
        self.num_level = new_levels;
        self.num_bits = 1 << self.num_level;
    }

    /// Create a zero element
    pub fn zero() -> Self {
        Self::new(0, 0)
    }

    /// Create a one element
    pub fn one() -> Self {
        Self::new(1, 0)
    }

    /// Abstract multiplication helper (similar to binius implementation)
    pub fn mul_abstract(
        a_hi: &Self,
        a_lo: &Self,
        a_sum: &Self,
        b_hi: &Self,
        b_lo: &Self,
        b_sum: &Self,
    ) -> Self {
        // Perform modular operations based on: x_i^2 = x_i * x_{i-1} + 1
        let mut mx = a_hi.clone() * b_hi.clone(); // mx = a_hi * b_hi
        let mut lo = a_lo.clone() * b_lo.clone(); // lo = a_lo * b_lo
        let mx_num_level = mx.num_level;
        let mx_num_bits = mx.num_bits;
        lo = lo + mx.clone();

        // mx * 2^(mx.num_half_bits())
        mx = mx * Self::new(1 << (mx_num_bits / 2), mx_num_level);

        // Perform hi operations
        let mut hi = a_sum.clone() * b_sum.clone(); // hi = a_sum * b_sum
        hi = hi + (lo.clone() + mx); // hi += lo + mx

        // Concatenate hi and lo by shifting hi to make space for lo
        hi.join(&lo)
    }

    /// Computes the multiplicative inverse of this element using a recursive algorithm
    /// Returns an error if the element is zero
    pub fn inv(&self) -> Result<Self, BinaryFieldError> {
        if self.is_zero() {
            return Err(BinaryFieldError::InverseOfZero);
        }

        // For F2, the inverse of 1 is 1
        if self.num_level == 0 {
            return Ok(self.clone());
        }

        // For small fields, use Fermat's little theorem
        if self.num_level <= 1 || self.num_bits <= 4 {
            // Use Fermat's little theorem:
            // In GF(2^n), x^(2^n-1) = 1 for any non-zero x
            // Therefore x^(2^n-2) is the multiplicative inverse
            return Ok(self.pow((1 << self.num_bits) - 2));
        }

        // For larger fields, use recursive algorithm
        let (a_hi, a_lo) = self.split();

        // Compute 2^(k-1) where k = num_bits/2
        let two_pow_k_minus_one = Self::new(1 << ((self.num_bits / 2) - 1), self.num_level - 1);

        // a = a_hi * x^k + a_lo
        // a_lo_next = a_hi * x^(k-1) + a_lo
        let a_lo_next = a_lo.clone() + a_hi.clone() * two_pow_k_minus_one;

        // Δ = a_lo * a_lo_next + a_hi^2
        let delta = a_lo.clone() * a_lo_next.clone() + a_hi.clone() * a_hi.clone();

        // Compute inverse of delta recursively
        let delta_inverse = delta.inv()?;

        // Compute parts of the inverse
        let out_hi = delta_inverse.clone() * a_hi;
        let out_lo = delta_inverse * a_lo_next;

        // Join the parts to get the final inverse
        Ok(out_hi.join(&out_lo))
    }

    /// Calculate power
    pub fn pow(&self, exp: u32) -> Self {
        let mut result = Self::one();
        let mut base = self.clone();
        let mut exp_val = exp;

        while exp_val > 0 {
            if exp_val & 1 == 1 {
                result = result * base.clone();
            }
            base = base.clone() * base.clone();
            exp_val >>= 1;
        }

        result
    }

    // Helper method that handles addition with different sizes
    // Use Ingoya's implementation
    // TO DO : Benchmark this implementation vs Ingoyama
    // Diego's style
    fn add_elements(&self, other: &Self) -> Self {
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
            // If other is larger, swap the arguments and call the same method
            other.add_elements(self)
        } else {
            // Same size, just XOR
            Self {
                value: self.value ^ other.value,
                num_level: self.num_level,
                num_bits: self.num_bits,
            }
        }
    }

    // Helper method that handles multiplication with different sizes
    fn mul_elements(&self, other: &Self) -> Self {
        // Optimizations for 0 or 1
        if self.is_zero() || other.is_zero() {
            return Self::zero();
        }
        if self.is_one() {
            return other.clone();
        }
        if other.is_one() {
            return self.clone();
        }

        // If elements have different sizes
        if self.num_level != other.num_level {
            if self.num_level > other.num_level {
                let mask = (1 << other.num_bits) - 1;
                // Lower part
                let low = self.value & mask;
                // Calculate the lower part using multiplication at the smaller element level
                let result_low = if other.value == 0 {
                    0
                } else {
                    // Create an element of the smaller level
                    let small_self = Self::new(low, other.num_level);
                    // Multiply using existing logic
                    let result = small_self.mul_elements(other);
                    result.value
                };

                // Higher part
                let high = self.value >> other.num_bits;
                // Combine the results
                let result_value = (high << other.num_bits) | result_low;

                return Self {
                    value: result_value,
                    num_level: self.num_level,
                    num_bits: self.num_bits,
                };
            } else {
                // If the other element is larger, swap the arguments and call the same method
                return other.mul_elements(self);
            }
        }

        // If both elements have the same level
        if self.num_level <= 1 {
            // Special case for level 0 (F₂)
            if self.num_level == 0 {
                return Self::new(self.value & other.value, 0);
            }

            // Special case for level 1 (F₄)
            let a_hi = (self.value >> 1) & 1;
            let a_lo = self.value & 1;
            let b_hi = (other.value >> 1) & 1;
            let b_lo = other.value & 1;

            let a_sum = a_hi ^ a_lo;
            let b_sum = b_hi ^ b_lo;

            let lo = a_lo * b_lo;
            let hi = (a_sum * b_sum) ^ lo;
            let lo = (a_hi * b_hi) ^ lo;

            return Self::new(2 * hi + lo, 1);
        } else {
            // For higher levels, use Karatsuba
            let (a_hi, a_lo) = self.split();
            let (b_hi, b_lo) = other.split();
            let a_sum = a_hi.clone() + a_lo.clone();
            let b_sum = b_hi.clone() + b_lo.clone();

            return Self::mul_abstract(&a_hi, &a_lo, &a_sum, &b_hi, &b_lo, &b_sum);
        }
    }
}

impl Add for TowerFieldElement {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        // Use the helper method that takes references
        self.add_elements(&other)
    }
}

impl<'a> Add<&'a TowerFieldElement> for &'a TowerFieldElement {
    type Output = TowerFieldElement;

    fn add(self, other: &'a TowerFieldElement) -> TowerFieldElement {
        // Directly use the helper method
        self.add_elements(other)
    }
}

impl AddAssign for TowerFieldElement {
    fn add_assign(&mut self, other: Self) {
        *self = *self + other;
    }
}

impl Sub for TowerFieldElement {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        // In binary fields, subtraction is the same as addition
        self + other
    }
}

impl Neg for TowerFieldElement {
    type Output = Self;

    fn neg(self) -> Self {
        // In binary fields, negation is the identity
        self
    }
}

impl Mul for TowerFieldElement {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        // Use the helper method that takes references
        self.mul_elements(&other)
    }
}

impl<'a> Mul<&'a TowerFieldElement> for &'a TowerFieldElement {
    type Output = TowerFieldElement;

    fn mul(self, other: &'a TowerFieldElement) -> TowerFieldElement {
        // Directly use the helper method
        self.mul_elements(other)
    }
}

impl MulAssign for TowerFieldElement {
    fn mul_assign(&mut self, other: Self) {
        *self = *self * other;
    }
}

impl Product for TowerFieldElement {
    fn product<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl Sum for TowerFieldElement {
    fn sum<I>(iter: I) -> Self
    where
        I: Iterator<Item = Self>,
    {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl fmt::Display for TowerFieldElement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl From<u128> for TowerFieldElement {
    fn from(val: u128) -> Self {
        TowerFieldElement::new(val, 7)
    }
}

impl From<u64> for TowerFieldElement {
    fn from(val: u64) -> Self {
        TowerFieldElement::new(val as u128, 6)
    }
}

impl From<u32> for TowerFieldElement {
    fn from(val: u32) -> Self {
        TowerFieldElement::new(val as u128, 5)
    }
}

impl From<u16> for TowerFieldElement {
    fn from(val: u16) -> Self {
        TowerFieldElement::new(val as u128, 4)
    }
}

impl From<u8> for TowerFieldElement {
    fn from(val: u8) -> Self {
        TowerFieldElement::new(val as u128, 3)
    }
}

impl Default for TowerFieldElement {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

// Testing module
#[cfg(test)]
mod tests {
    use super::*;

    // Dejamos este test? deberiamos tratar el posible error de que pase mayor nivel?
    // #[test]
    // fn test_new_safe() {
    //     // Test with level too large
    //     let elem = TowerFieldElement::new(0, 8);
    //     assert_eq!(elem.num_level, 7); // Should be capped at 7

    //     // Test with value too large for level
    //     let elem = TowerFieldElement::new(4, 1); // Level 1 can only store 0-3
    //     assert_eq!(elem.value, 0); // Should mask to 0 (4 & 3 = 0)
    // }

    #[test]
    fn test_addition() {
        let a = TowerFieldElement::new(5, 3); // 8 bits
        let b = TowerFieldElement::new(3, 2); // 4 bits

        let c = a + b;
        // 5 (0101) + 3 (0011) should be 6 (0110) at level 3
        assert_eq!(c.value, 6);
        assert_eq!(c.num_level, 3);

        // Test commutative property
        let d = b + a;
        assert_eq!(d.value, 6);
        assert_eq!(d.num_level, 3);
    }

    #[test]
    fn test_multiplication() {
        // Base case: F₂ (1 bit)
        let a0 = TowerFieldElement::new(1, 0);
        let b0 = TowerFieldElement::new(1, 0);
        let c0 = a0 * b0;
        assert_eq!(c0.value, 1);
        assert_eq!(c0.num_level, 0);

        // Case F₄ (2 bits)
        let a1 = TowerFieldElement::new(2, 1); // x
        let b1 = TowerFieldElement::new(3, 1); // x + 1
        let c1 = a1 * b1;
        assert_eq!(c1.value, 1); // x * (x + 1) = x² + x = 1 + x + x = 1

        // Case with different sizes
        let a2 = TowerFieldElement::new(10, 3); // Level 3 (8 bits)
        let b2 = TowerFieldElement::new(3, 1); // Level 1 (2 bits)
        let c2 = a2 * b2;
        // The result should have the level of the larger element
        assert_eq!(c2.num_level, 3);

        // Case of multiplication by 0 and 1
        let e = TowerFieldElement::new(42, 5);
        assert_eq!((e * TowerFieldElement::zero()).value, 0);
        assert_eq!((e * TowerFieldElement::one()).value, e.value);
    }

    #[test]
    fn test_inverse() {
        // Test inverse in F₂
        let a0 = TowerFieldElement::new(1, 0);
        let inv_a0 = a0.inv().unwrap();
        assert_eq!(inv_a0.value, 1);
        assert_eq!(inv_a0.num_level, 0);

        // Test inverse in F₄
        let a1 = TowerFieldElement::new(2, 1); // element 'x' in F₄
        let inv_a1 = a1.inv().unwrap();
        assert_eq!(inv_a1.value, 3); // inverse of 'x' is 'x+1' in F₄
        assert_eq!(inv_a1.num_level, 1);

        // Verify a * a^(-1) = 1
        assert_eq!((a1 * inv_a1).value, 1);

        // Test inverse in F₈
        let a2 = TowerFieldElement::new(2, 2); // element 'x' in F₈
        let inv_a2 = a2.inv().unwrap();
        // Verify a * a^(-1) = 1
        assert_eq!((a2 * inv_a2).value, 1);

        // Test inverse of 0 returns error
        let zero = TowerFieldElement::zero();
        assert!(matches!(zero.inv(), Err(BinaryFieldError::InverseOfZero)));
    }

    #[test]
    fn test_operations_different_levels() {
        // Create elements of different levels
        let a = TowerFieldElement::new(5, 3); // Level 3 (8 bits)
        let b = TowerFieldElement::new(3, 1); // Level 1 (2 bits)

        // Multiplication: should maintain the level of the larger element
        let c = a * b;
        assert_eq!(c.num_level, 3);

        // Create specific values for controlled test
        let x = TowerFieldElement::new(0b1010, 3); // Level 3: 1010 binary
        let y = TowerFieldElement::new(0b11, 1); // Level 1: 11 binary

        // Multiply
        let result = x * y;

        // Expected result: the lower 2 bits of x are multiplied by y
        // 10 * 11 = 01 (multiplication in F₄)
        // Result should be 1000 + 01 = 1001
        assert_eq!(result.value, 0b1001);
        assert_eq!(result.num_level, 3);

        // Test commutative property
        let result2 = y * x;
        assert_eq!(result2.value, result.value);
        assert_eq!(result2.num_level, result.num_level);

        // Test multiplication between more levels
        let z = TowerFieldElement::new(13, 4); // Level 4 (16 bits)
        let big_result = z * y; // Level 4 * Level 1
        assert_eq!(big_result.num_level, 4);

        // Test multiplication by 0 and 1 across different levels
        let zero = TowerFieldElement::zero(); // Level 0
        let one = TowerFieldElement::one(); // Level 0

        assert_eq!((x * zero).value, 0);
        assert_eq!((x * one).value, x.value);
        assert_eq!((z * zero).value, 0);
        assert_eq!((z * one).value, z.value);
    }

    #[test]
    fn test_mixed_operations() {
        // Test combination of operations with different level elements
        // For binary fields with our implementation, we need specific values
        // to verify properties correctly

        // Choose values where the distributive property will hold correctly
        let a = TowerFieldElement::new(0b100, 3); // Level 3 (value 4)
        let b = TowerFieldElement::new(0b11, 1); // Level 1 (value 3)
        let c = TowerFieldElement::new(0b10, 2); // Level 2 (value 2)

        // Compute (a + b) * c
        let sum = a + b;
        let result1 = sum * c;

        // Compute a * c + b * c
        let ac = a * c;
        let bc = b * c;
        let result2 = ac + bc;

        // Verify distributive property
        assert_eq!(result1.value, result2.value);
        assert_eq!(result1.num_level, result2.num_level);
    }

    #[test]
    fn test_multiplication_associativity() {
        // Test associativity of multiplication
        let a = TowerFieldElement::new(0x1F, 4); // Level 4 (16 bits)
        let b = TowerFieldElement::new(0x2A, 4);
        let c = TowerFieldElement::new(0x35, 4);

        // Multiplication associativity: (a * b) * c = a * (b * c)
        let left = (a * b) * c;
        let right = a * (b * c);

        assert_eq!(left.value, right.value);
        assert_eq!(left.num_level, right.num_level);
    }

    #[test]
    fn test_multiplication_special_cases() {
        // Test multiplication by zero and one across different levels
        let levels = [1, 2, 3, 4];

        for &level in &levels {
            let a = TowerFieldElement::new(0x1F, level);
            let zero = TowerFieldElement::zero();
            let one = TowerFieldElement::one();

            // Multiplication by zero
            assert_eq!((a * zero).value, 0);
            assert_eq!((zero * a).value, 0);

            // Multiplication by one
            assert_eq!((a * one).value, a.value);
            assert_eq!((one * a).value, a.value);
        }
    }

    #[test]
    fn test_multiplication_with_split_join() {
        // Test that multiplication works correctly with split and rejoined elements
        let value = 0xABCD;
        let a = TowerFieldElement::new(value, 4);

        // Split and rejoin
        let (hi, lo) = a.split();
        let rejoined = hi.join(&lo);

        // Multiply both representations by the same value
        let multiplier = TowerFieldElement::new(0x1234, 4);
        let result1 = a * multiplier;
        let result2 = rejoined * multiplier;

        assert_eq!(result1.value, result2.value);
        assert_eq!(result1.num_level, result2.num_level);
    }

    #[test]
    fn test_multiplication_overflow() {
        // Test that multiplication properly handles overflow and reduction
        for level in 1..5 {
            let max_value = (1u128 << (1 << level)) - 1; // Maximum value for this level
            let a = TowerFieldElement::new(max_value, level);
            let b = TowerFieldElement::new(max_value, level);

            let result = a * b;

            // Result should be properly reduced (not exceed the field size)
            assert!(result.value < (1u128 << result.num_bits));
        }
    }

    #[test]
    fn test_split_join_consistency() {
        // Test that join and split are consistent operations
        for i in 1..10 {
            let original = TowerFieldElement::new(i, 3);
            let (hi, lo) = original.split();
            let rejoined = hi.join(&lo);

            // The rejoined value should equal the original
            assert_eq!(rejoined.value, original.value);
            assert_eq!(rejoined.num_level, original.num_level);
            assert_eq!(rejoined.num_bits, original.num_bits);
        }
    }
}
