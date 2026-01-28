//! Comparison gadgets for equality and ordering checks.

use crate::dsl::builder::CircuitBuilder;
use crate::dsl::gadgets::{Gadget, GadgetError};
use crate::dsl::types::{BoolVar, FieldVar};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;

/// Checks if a value is zero.
///
/// Returns a boolean that is 1 if the input is zero, 0 otherwise.
///
/// # Implementation
///
/// Uses the inverse trick: if x ≠ 0, then x has an inverse, and x * inv = 1.
/// We output is_zero = 1 - x * inv.
///
/// Constraints:
/// - is_zero * x = 0 (if x ≠ 0, then is_zero = 0)
/// - is_zero + x * inv = 1 (if x = 0, then is_zero = 1)
pub struct IsZero;

impl<F: IsField> Gadget<F> for IsZero {
    type Input = FieldVar;
    type Output = BoolVar;

    fn synthesize(
        builder: &mut CircuitBuilder<F>,
        x: Self::Input,
    ) -> Result<Self::Output, GadgetError> {
        // Create auxiliary variables
        let is_zero = builder.new_variable();
        let inv = builder.new_variable();

        // Constraint 1: is_zero * x = 0
        // This ensures is_zero = 0 when x ≠ 0
        let is_zero_times_x = builder.mul(&is_zero, &x);
        builder.assert_zero(&is_zero_times_x);

        // Constraint 2: is_zero + x * inv = 1
        // This ensures is_zero = 1 when x = 0 (since then x * inv = 0)
        // When x ≠ 0, inv = 1/x, so x * inv = 1, and is_zero = 0
        let x_times_inv = builder.mul(&x, &inv);
        let sum = builder.add(&is_zero, &x_times_inv);
        builder.assert_eq_constant(&sum, FieldElement::one());

        // is_zero is boolean by construction
        Ok(BoolVar::new(is_zero.inner))
    }

    fn constraint_count() -> usize {
        3 // Two multiplication constraints + one equality
    }

    fn name() -> &'static str {
        "IsZero"
    }
}

/// Checks if two values are equal.
///
/// Returns a boolean that is 1 if a == b, 0 otherwise.
pub struct IsEqual;

impl<F: IsField> Gadget<F> for IsEqual {
    type Input = (FieldVar, FieldVar);
    type Output = BoolVar;

    fn synthesize(
        builder: &mut CircuitBuilder<F>,
        (a, b): Self::Input,
    ) -> Result<Self::Output, GadgetError> {
        // a == b iff a - b == 0
        let diff = builder.sub(&a, &b);
        IsZero::synthesize(builder, diff)
    }

    fn constraint_count() -> usize {
        // Subtraction + IsZero (2 mul + 1 add = 3)
        1 + 3
    }

    fn name() -> &'static str {
        "IsEqual"
    }
}

/// Less than comparison for bounded integers: a < b.
///
/// Assumes both inputs are in the range [0, 2^BITS).
///
/// # Implementation
///
/// a < b iff (2^BITS + a - b) has bit BITS equal to 0.
///
/// We compute diff = 2^BITS + a - b, then decompose to BITS+1 bits.
/// If a < b, then diff is in [1, 2^BITS), so bit BITS is 0.
/// If a >= b, then diff is in [2^BITS, 2^(BITS+1)), so bit BITS is 1.
///
/// The result is NOT(bit BITS).
pub struct LessThan<const BITS: usize>;

impl<F: IsField, const BITS: usize> Gadget<F> for LessThan<BITS> {
    type Input = (FieldVar, FieldVar);
    type Output = BoolVar;

    fn synthesize(
        builder: &mut CircuitBuilder<F>,
        (a, b): Self::Input,
    ) -> Result<Self::Output, GadgetError> {
        if BITS == 0 {
            return Err(GadgetError::InvalidInput(
                "BITS must be greater than 0".to_string(),
            ));
        }

        // Compute diff = 2^BITS + a - b
        // Use field exponentiation to avoid shift overflow when BITS >= 64.
        let offset = FieldElement::<F>::from(2u64).pow(BITS as u64);
        let a_plus_offset = builder.add_constant(&a, offset);
        let diff = builder.sub(&a_plus_offset, &b);

        // Decompose diff into BITS+1 bits using a runtime approach
        // We can't use RangeCheck::<{ BITS + 1 }> due to const generic limitations
        // Instead, we manually create the bits and constraints
        let num_bits = BITS + 1;

        let mut bits = Vec::with_capacity(num_bits);
        for _ in 0..num_bits {
            let bit = builder.new_variable();
            let bool_var = builder.assert_bool(&bit);
            bits.push(bool_var);
        }

        // Recompose: sum of bit_i * 2^i should equal diff
        let mut accumulated = builder.constant(FieldElement::<F>::zero());
        let two = FieldElement::<F>::from(2u64);
        let mut power_of_two = FieldElement::<F>::one();

        for bit in &bits {
            let scaled_bit = builder.mul_constant(bit, power_of_two.clone());
            accumulated = builder.add(&accumulated, &scaled_bit);
            power_of_two = &power_of_two * &two;
        }

        builder.assert_eq(&accumulated, &diff);

        // The top bit indicates a >= b, so result is NOT(top_bit)
        let top_bit = bits
            .last()
            .ok_or_else(|| GadgetError::SynthesisError("No bits in decomposition".to_string()))?;
        Ok(builder.not(top_bit))
    }

    fn constraint_count() -> usize {
        // (BITS+1) boolean constraints + (BITS+1) multiplications + 1 equality + NOT
        (BITS + 1) * 2 + 2
    }

    fn name() -> &'static str {
        "LessThan"
    }
}

/// Less than or equal comparison: a <= b.
///
/// Assumes both inputs are in the range [0, 2^BITS).
pub struct LessThanOrEqual<const BITS: usize>;

impl<F: IsField, const BITS: usize> Gadget<F> for LessThanOrEqual<BITS> {
    type Input = (FieldVar, FieldVar);
    type Output = BoolVar;

    fn synthesize(
        builder: &mut CircuitBuilder<F>,
        (a, b): Self::Input,
    ) -> Result<Self::Output, GadgetError> {
        // a <= b iff NOT(b < a)
        let b_lt_a = LessThan::<BITS>::synthesize(builder, (b, a))?;
        Ok(builder.not(&b_lt_a))
    }

    fn constraint_count() -> usize {
        // LessThan constraint count + 1 for NOT
        (BITS + 1) * 2 + 2 + 1
    }

    fn name() -> &'static str {
        "LessThanOrEqual"
    }
}

/// Greater than comparison: a > b.
///
/// Assumes both inputs are in the range [0, 2^BITS).
pub struct GreaterThan<const BITS: usize>;

impl<F: IsField, const BITS: usize> Gadget<F> for GreaterThan<BITS> {
    type Input = (FieldVar, FieldVar);
    type Output = BoolVar;

    fn synthesize(
        builder: &mut CircuitBuilder<F>,
        (a, b): Self::Input,
    ) -> Result<Self::Output, GadgetError> {
        // a > b iff b < a
        LessThan::<BITS>::synthesize(builder, (b, a))
    }

    fn constraint_count() -> usize {
        // Same as LessThan
        (BITS + 1) * 2 + 2
    }

    fn name() -> &'static str {
        "GreaterThan"
    }
}

/// Greater than or equal comparison: a >= b.
///
/// Assumes both inputs are in the range [0, 2^BITS).
pub struct GreaterThanOrEqual<const BITS: usize>;

impl<F: IsField, const BITS: usize> Gadget<F> for GreaterThanOrEqual<BITS> {
    type Input = (FieldVar, FieldVar);
    type Output = BoolVar;

    fn synthesize(
        builder: &mut CircuitBuilder<F>,
        (a, b): Self::Input,
    ) -> Result<Self::Output, GadgetError> {
        // a >= b iff NOT(a < b)
        let a_lt_b = LessThan::<BITS>::synthesize(builder, (a, b))?;
        Ok(builder.not(&a_lt_b))
    }

    fn constraint_count() -> usize {
        // LessThan constraint count + 1 for NOT
        (BITS + 1) * 2 + 2 + 1
    }

    fn name() -> &'static str {
        "GreaterThanOrEqual"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    type F = U64PrimeField<65537>;

    #[test]
    fn test_is_zero_gadget() {
        let mut builder = CircuitBuilder::<F>::new();

        let x = builder.public_input("x");
        let _is_zero = IsZero::synthesize(&mut builder, x).unwrap();
    }

    #[test]
    fn test_is_equal_gadget() {
        let mut builder = CircuitBuilder::<F>::new();

        let a = builder.public_input("a");
        let b = builder.public_input("b");
        let _is_eq = IsEqual::synthesize(&mut builder, (a, b)).unwrap();
    }

    #[test]
    fn test_less_than_gadget() {
        let mut builder = CircuitBuilder::<F>::new();

        let a = builder.public_input("a");
        let b = builder.public_input("b");

        // Test with 8-bit comparison
        let _result = LessThan::<8>::synthesize(&mut builder, (a, b)).unwrap();
    }

    #[test]
    fn test_less_than_64_bits_does_not_overflow() {
        let mut builder = CircuitBuilder::<F>::new();

        let a = builder.public_input("a");
        let b = builder.public_input("b");

        // Ensure large bit widths don't overflow during synthesis.
        let _result = LessThan::<64>::synthesize(&mut builder, (a, b)).unwrap();
    }

    #[test]
    fn test_comparison_constraint_counts() {
        // Use explicit type annotations to avoid inference issues
        assert!(<IsZero as Gadget<F>>::constraint_count() > 0);
        assert!(
            <IsEqual as Gadget<F>>::constraint_count() > <IsZero as Gadget<F>>::constraint_count()
        );
    }
}
