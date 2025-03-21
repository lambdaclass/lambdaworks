use core::cmp::Ordering;
use core::fmt;
use core::iter::{Product, Sum};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub};

// Implementation of binary fields of the form GF(2^(2^n)) by constructing a tower of field extensions.
// The basic idea is to represent an element of the field as a polynomial with coefficients in GF(2) = {0, 1}.
// The coefficients of each polynomial are stored as bits in a `u128` integer.
// The tower structure is built recursively, with each level representing an extension of the previous field.

// For more details, see:
// - Lambdaclass blog post about the use of binary fields in SNARKs: https://blog.lambdaclass.com/snarks-on-binary-fields-binius/
// - Vitalik Buterin's Binius: https://vitalik.eth.limo/general/2024/04/29/binius.html

#[derive(Debug)]
pub enum BinaryFieldError {
    /// Attempt to compute inverse of zero
    InverseOfZero,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
/// An element in the tower of binary field extensions from level 0 to level 7.
///
/// Implements arithmetic in finite fields GF(2^(2^n)) where n is the level of the field extension in the tower.
///
/// The internal representation stores polynomial coefficients as bits in a u128 integer.
pub struct TowerFieldElement {
    /// The value of the element.
    /// The binary expression of this value represents the coefficients of the corresponding polynomial of the element.
    /// For example, if value = 0b1101, then p = xy + y + 1. If value = 0b0110, then p = y + x.
    pub value: u128,
    /// Number of the level in the tower.
    /// It tells us to which field extension the element belongs.
    /// It goes from 0 (representing the base field of two elements) to 7 (representing the field extension of 2^128 elements).
    pub num_level: usize,
}

impl TowerFieldElement {
    /// Constructor that always succeeds by masking the value if it is too big for the given
    /// num_level, and limiting the level so that is not greater than 7.
    pub fn new(val: u128, num_level: usize) -> Self {
        // Limit num_level to a maximum valid value for u128.
        let safe_level = if num_level > 7 { 7 } else { num_level };

        // The number of bits needed for the given level
        let bits = 1 << safe_level;
        let mask = if bits >= 128 {
            u128::MAX
        } else {
            (1 << bits) - 1
        };

        Self {
            // We take just the lsb of val that fit in the extension field we are.
            value: val & mask,
            num_level: safe_level,
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

    /// Returns level number in the tower.
    #[inline]
    pub fn num_level(&self) -> usize {
        self.num_level
    }

    /// Returns the number of bits needed for that level (2^num_levels).
    /// Note that the order of the extension field in that level is 2^num_bits.
    #[inline]
    pub fn num_bits(&self) -> usize {
        1 << self.num_level()
    }

    // Equality check
    pub fn equals(&self, other: &Self) -> bool {
        self.value == other.value()
    }

    /// Returns binary string representation
    #[cfg(feature = "std")]
    pub fn to_binary_string(&self) -> String {
        format!("{:0width$b}", self.value, width = self.num_bits())
    }

    /// Splits element into high and low parts.
    /// For example, if a = xy + y + x, then a = (x + 1)y + x and
    /// therefore, a_hi = x + 1 and a_lo = x.
    pub fn split(&self) -> (Self, Self) {
        let half_bits = self.num_bits() / 2;
        let mask = (1 << half_bits) - 1;
        let lo = self.value() & mask;
        let hi = (self.value() >> half_bits) & mask;

        (
            Self::new(hi, self.num_level() - 1),
            Self::new(lo, self.num_level() - 1),
        )
    }

    /// Joins the hi and low part making a new element of a bigger level.
    /// For example, if a_hi = x and a_low = 1
    /// then a = xy + 1.
    pub fn join(&self, low: &Self) -> Self {
        let joined = (self.value() << self.num_bits()) | low.value();
        Self::new(joined, self.num_level() + 1)
    }

    // It embeds an element in an extension changing the level number.
    pub fn extend_num_level(&mut self, new_level: usize) {
        if self.num_level() < new_level {
            self.num_level = new_level;
        }
    }

    /// Create a zero element
    pub fn zero() -> Self {
        Self::new(0, 0)
    }

    /// Create a one element
    pub fn one() -> Self {
        Self::new(1, 0)
    }

    /// Addition between elements of same or different levels.
    fn add_elements(&self, other: &Self) -> Self {
        let num_level = self.num_level().max(other.num_level());
        Self::new(self.value() ^ other.value(), num_level)
    }

    // Multiplies a and b in the following way:
    //
    // - If a and b are from the same level:
    // a = a_hi * x_n + a_lo
    // b = b_hi * x_n + b_lo
    // Then a * b = (b_hi * a_hi * x_{n-1} + b_hi * a_lo + a_hi * b_lo ) * x_n + b_hi * a_hi + a_lo * b_lo.
    // We calculate each product in the equation below using recursion.
    //
    // - if a's level is larger than b's level, we partition a until we have parts of the size of b and
    // multiply each part by b.
    fn mul(self, other: Self) -> Self {
        match self.num_level().cmp(&other.num_level()) {
            Ordering::Greater => {
                // We split a into two parts and call the same method to multiply each part by b.
                let (a_hi, a_lo) = self.split();
                // Join a_hi * b and a_lo * b.
                a_hi.mul(other).join(&a_lo.mul(other))
            }
            Ordering::Less => {
                // If b is larger than a, we swap the arguments and call the same method.
                other.mul(self)
            }
            Ordering::Equal => {
                // Base case:
                if self.num_level() == 0 {
                    // In the binary base field, multiplication is the same as AND operation.
                    return Self::new(self.value() & other.value(), 0);
                }

                // Split both elements into high and low parts
                let (a_high, a_low) = self.split();
                let (b_high, b_low) = other.split();

                // Step 1: Compute sub-products
                let low_product = a_low.mul(b_low); // a_low * b_low
                let high_product = a_high.mul(b_high); // a_high * b_high

                // Step 2: Get the polynomial x_{n-1} value
                let x_value = if self.num_level() == 1 {
                    Self::new(1, 0)
                } else {
                    Self::new(1 << (self.num_bits() / 4), self.num_level() - 1)
                };

                // Step 3: Compute high_product * x_{n-1}
                let shifted_high_product = high_product.mul(x_value);

                // Step 4: Karatsuba optimization for middle term
                // Instead of computing a_high*b_low + a_low*b_high directly,
                // we use (a_low+a_high)*(b_low+b_high) - low_product - high_product
                let sum_product = (a_low + a_high).mul(b_low + b_high);
                let middle_term = sum_product - low_product - high_product;

                // Step 5: Join the parts according to the tower field multiplication formula
                (shifted_high_product + middle_term).join(&(high_product + low_product))
            }
        }
    }

    /// Computes the multiplicative inverse using Fermat's little theorem.
    /// Returns an error if the element is zero.
    // Based on Ingoyama's implementation
    // https://github.com/ingonyama-zk/smallfield-super-sumcheck/blob/a8c61beef39bc0c10a8f68d25eeac0a7190a7289/src/tower_fields/binius.rs#L116C5-L116C6
    pub fn inv(&self) -> Result<Self, BinaryFieldError> {
        if self.is_zero() {
            return Err(BinaryFieldError::InverseOfZero);
        }
        if self.num_level() <= 1 || self.num_bits() <= 4 {
            let exponent = (1 << self.num_bits()) - 2;
            Ok(Self::pow(self, exponent as u32))
        } else {
            let (a_hi, a_lo) = self.split();
            let two_pow_k_minus_one = Self::new(1 << (self.num_bits() / 4), self.num_level() - 1);
            // a = a_hi * x^k + a_lo
            // a_lo_next = a_hi * x^(k-1) + a_lo
            let a_lo_next = a_lo + a_hi * two_pow_k_minus_one;

            // Δ = a_lo * a_lo_next + a_hi^2
            let delta = a_lo * a_lo_next + a_hi * a_hi;

            // Compute inverse of delta recursively
            let delta_inverse = delta.inv()?;

            // Compute parts of the inverse
            let out_hi = delta_inverse * a_hi;
            let out_lo = delta_inverse * a_lo_next;

            // Join the parts to get the final inverse
            Ok(out_hi.join(&out_lo))
        }
    }

    /// Calculate power.
    pub fn pow(&self, exp: u32) -> Self {
        let mut result = Self::one();
        let mut base = *self;
        let mut exp_val = exp;

        while exp_val > 0 {
            if exp_val & 1 == 1 {
                result *= base;
            }
            base = base * base;
            exp_val >>= 1;
        }

        result
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
#[allow(clippy::suspicious_arithmetic_impl)]
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
        self.mul(other)
    }
}

impl<'a> Mul<&'a TowerFieldElement> for &'a TowerFieldElement {
    type Output = TowerFieldElement;

    fn mul(self, other: &'a TowerFieldElement) -> TowerFieldElement {
        <TowerFieldElement as Mul<TowerFieldElement>>::mul(*self, *other)
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

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_new_safe() {
        // Test with level too large
        let elem = TowerFieldElement::new(0, 8);
        assert_eq!(elem.num_level, 7); // Should be capped at 7

        // Test with value too large for level
        let elem = TowerFieldElement::new(4, 1); // Level 1 can only store 0-3
        assert_eq!(elem.value, 0); // Should mask to 0 (100 & 11 = 00)
    }

    #[test]
    fn test_addition() {
        let a = TowerFieldElement::new(5, 9); // 8 bits
        let b = TowerFieldElement::new(3, 2); // 4 bits

        let c = a + b;
        // 5 (0101) + 3 (0011) should be 6 (0110) at level 3
        assert_eq!(c.value, 6);
        assert_eq!(c.num_level, 7);

        // Test commutative property
        let d = b + a;
        assert_eq!(d, c);
    }

    #[test]
    fn mul_in_level_0() {
        let a = TowerFieldElement::new(0, 0);
        let b = TowerFieldElement::new(1, 0);
        assert_eq!(a * a, a);
        assert_eq!(a * b, a);
        assert_eq!(b * b, b);
    }

    #[test]
    #[allow(clippy::zero_prefixed_literal)]
    fn mul_in_level_1() {
        let a = TowerFieldElement::new(00, 1); // 0
        let b = TowerFieldElement::new(01, 1); // 1
        let c = TowerFieldElement::new(10, 1); // x
        let d = TowerFieldElement::new(11, 1); // x + 1
        assert_eq!(a * a, a);
        assert_eq!(a * b, a);
        assert_eq!(b * c, c);
        assert_eq!(c * d, b);
    }

    #[test]
    fn mul_in_level_2() {
        let a = TowerFieldElement::new(0b0000, 2); // 0
        let b = TowerFieldElement::new(0b0001, 2); // 1
        let c = TowerFieldElement::new(0b0010, 2); // x
        let d = TowerFieldElement::new(0b0011, 2); // x + 1
        let e = TowerFieldElement::new(0b0100, 2); // y
        let f = TowerFieldElement::new(0b0101, 2); // y + 1
        let g = TowerFieldElement::new(0b0110, 2); // y + x
        let h = TowerFieldElement::new(0b0111, 2); // y + x + 1
        let i = TowerFieldElement::new(0b1000, 2); // yx
        let j = TowerFieldElement::new(0b1001, 2); // yx + 1
        let k = TowerFieldElement::new(0b1010, 2); // yx + x
        let l = TowerFieldElement::new(0b1011, 2); // yx + x + 1
        let n = TowerFieldElement::new(0b1100, 2); // yx + y
        let m = TowerFieldElement::new(0b1101, 2); // yx + y + 1
        let o = TowerFieldElement::new(0b1110, 2); // yx + y + x
        let p = TowerFieldElement::new(0b1111, 2); // yx + y + x + 1

        assert_eq!(a * p, a); // 0 * (yx + y + x + 1) = 0
        assert_eq!(a * l, a); // 0 * (yx + x + 1) = 0
        assert_eq!(b * m, m); // 1 * 1 = 1
        assert_eq!(c * e, i); // x * y = xy
        assert_eq!(c * c, d); // x * x = x + 1
        assert_eq!(g * h, n); //(y + x)(y + x + 1) = yx + y
        assert_eq!(k * j, b); // (yx + x)(yx + 1) = 1
        assert_eq!(j * f, d); // (yx + 1)(y + 1) = x + 1
        assert_eq!(e * e, j); // y * y = yx + 1
        assert_eq!(n * o, k); // (yx + y)(yx + y + x) = yx + x
    }

    #[test]
    fn mul_between_different_levels() {
        let a = TowerFieldElement::new(0b10, 1); // x
        let b = TowerFieldElement::new(0b0100, 2); // y
        let c = TowerFieldElement::new(0b1000, 2); // yx
        assert_eq!(a * b, c);
    }

    #[test]
    fn test_correct_level_mul() {
        let a = TowerFieldElement::new(0b1111, 5);
        let b = TowerFieldElement::new(0b1010, 2);
        assert_eq!((a * b).num_level, 5);
    }

    #[test]
    fn mul_is_asociative() {
        let a = TowerFieldElement::new(83, 7);
        let b = TowerFieldElement::new(31, 5);
        let c = TowerFieldElement::new(3, 2);
        let ab = a * b;
        let bc = b * c;
        assert_eq!(ab * c, a * bc);
    }

    #[test]
    fn mul_is_conmutative() {
        let a = TowerFieldElement::new(127, 7);
        let b = TowerFieldElement::new(6, 3);
        let ab = a * b;
        let ba = b * a;
        assert_eq!(ab, ba);
    }

    #[test]
    fn test_inverse() {
        let a0 = TowerFieldElement::new(1, 0);
        let inv_a0 = a0.inv().unwrap();
        assert_eq!(inv_a0.value, 1);
        assert_eq!(inv_a0.num_level, 0);

        let a1 = TowerFieldElement::new(2, 1);
        let inv_a1 = a1.inv().unwrap();
        assert_eq!(inv_a1.value, 3); // because 10 * 11 = 01.
        assert_eq!(inv_a1.num_level, 1);

        // Verify a * a^(-1) = 1
        let a2 = TowerFieldElement::new(15, 4);
        let inv_a2 = a2.inv().unwrap();
        let one = TowerFieldElement::new(1, 4);
        assert_eq!(a2 * inv_a2, one);

        let a3 = TowerFieldElement::new(30, 5);
        let inv_a3 = a3.inv().unwrap();
        let one = TowerFieldElement::new(1, 5);
        assert_eq!(a3 * inv_a3, one);

        let zero = TowerFieldElement::zero();
        assert!(matches!(zero.inv(), Err(BinaryFieldError::InverseOfZero)));
    }

    #[test]
    fn test_multiplication_overflow() {
        for level in 0..7 {
            let max_value = (1u128 << (1 << level)) - 1; // Maximum value for this level
            let a = TowerFieldElement::new(max_value, level);
            let b = TowerFieldElement::new(max_value, level);

            let result = a * b;

            // Result should be properly reduced
            assert!(result.value < (1u128 << result.num_bits()));
        }
    }

    #[test]
    fn test_split_join_consistency() {
        // Test that join and split are consistent operations
        for i in 0..20 {
            let original = TowerFieldElement::new(i, 3);
            let (hi, lo) = original.split();
            let rejoined = hi.join(&lo);

            assert_eq!(rejoined, original);
        }
    }
    #[cfg(feature = "std")]
    #[test]
    fn test_bin_representation() {
        let a = TowerFieldElement::new(0b1010, 5);
        assert_eq!(a.to_binary_string(), "00000000000000000000000000001010");
        let b = TowerFieldElement::new(0b1010, 4);
        assert_eq!(b.to_binary_string(), "0000000000001010");
    }

    // Strategy to generate a TowerFieldElement with a random level between 0 and 7.
    // For a given level:
    // - The number of bits is computed as 1 << level.
    // - For level 0, valid values are 0 to (1 << 1) - 1 = 1.
    // - For level > 0, valid values are 0 to (1 << (1 << level)) - 1.
    fn arb_tower_element_any() -> impl Strategy<Value = TowerFieldElement> {
        (0usize..=7)
            .prop_flat_map(|level| {
                let max_val = if level == 0 {
                    1
                } else if (1usize << level) >= 128 {
                    u128::MAX
                } else {
                    (1u128 << (1 << level)) - 1
                };
                (Just(level), 0u128..=max_val)
            })
            .prop_map(|(level, val)| TowerFieldElement::new(val, level))
    }

    #[cfg(feature = "std")]
    proptest! {
        // Test that multiplication is commutative:
        // For any two randomly generated elements, a * b should equal b * a.
        #[test]
        fn test_mul_commutative(a in arb_tower_element_any(), b in arb_tower_element_any()) {
            prop_assert_eq!(a * b, b * a);
        }

        // Test that multiplication is associative:
        // For any three randomly generated elements, (a * b) * c should equal a * (b * c).
        #[test]
        fn test_mul_associative(a in arb_tower_element_any(), b in arb_tower_element_any(), c in arb_tower_element_any()) {
            prop_assert_eq!((a * b) * c, a * (b * c));
        }
    }
}
