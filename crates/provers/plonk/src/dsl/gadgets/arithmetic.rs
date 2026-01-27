//! Arithmetic gadgets for range checks and bounded operations.

use crate::dsl::builder::CircuitBuilder;
use crate::dsl::gadgets::{Gadget, GadgetError};
use crate::dsl::types::{BoolVar, FieldVar};
use lambdaworks_math::field::element::FieldElement;
use lambdaworks_math::field::traits::IsField;

/// Range check gadget: asserts 0 <= x < 2^BITS.
///
/// This gadget decomposes the input into bits, constrains each bit to be
/// boolean (0 or 1), and verifies the recomposition equals the original value.
///
/// # Type Parameters
/// * `BITS` - The number of bits (range is [0, 2^BITS))
///
/// # Constraints
/// * BITS boolean constraints (one per bit)
/// * 1 equality constraint (recomposition check)
///
/// # Example
///
/// ```ignore
/// // Check that x is a valid u8 (0 <= x < 256)
/// RangeCheck::<8>::synthesize(&mut builder, x)?;
/// ```
pub struct RangeCheck<const BITS: usize>;

/// Input for RangeCheck gadget.
pub struct RangeCheckInput<F: IsField> {
    /// The value to range check
    pub value: FieldVar,
    /// Optional: provide bit hints for faster witness generation
    pub bit_hints: Option<Vec<FieldElement<F>>>,
}

impl<F: IsField> From<FieldVar> for RangeCheckInput<F> {
    fn from(value: FieldVar) -> Self {
        Self {
            value,
            bit_hints: None,
        }
    }
}

/// Output of RangeCheck gadget.
pub struct RangeCheckOutput {
    /// The bit decomposition (LSB first)
    pub bits: Vec<BoolVar>,
}

impl<F: IsField, const BITS: usize> Gadget<F> for RangeCheck<BITS> {
    type Input = RangeCheckInput<F>;
    type Output = RangeCheckOutput;

    fn synthesize(
        builder: &mut CircuitBuilder<F>,
        input: Self::Input,
    ) -> Result<Self::Output, GadgetError> {
        if BITS == 0 {
            return Err(GadgetError::InvalidInput(
                "BITS must be greater than 0".to_string(),
            ));
        }

        // Create variables for each bit
        let mut bits = Vec::with_capacity(BITS);
        for _ in 0..BITS {
            let bit = builder.new_variable();
            // Constrain to boolean: bit * (bit - 1) = 0
            let bool_var = builder.assert_bool(&bit);
            bits.push(bool_var);
        }

        // Recompose: sum of bit_i * 2^i should equal input
        // We build this incrementally to avoid large powers
        let mut accumulated = builder.constant(FieldElement::<F>::zero());
        let two = FieldElement::<F>::from(2u64);
        let mut power_of_two = FieldElement::<F>::one();

        for bit in &bits {
            // accumulated += bit * 2^i
            let scaled_bit = builder.mul_constant(bit, power_of_two.clone());
            accumulated = builder.add(&accumulated, &scaled_bit);
            power_of_two = &power_of_two * &two;
        }

        // Assert recomposition equals original value
        builder.assert_eq(&accumulated, &input.value);

        Ok(RangeCheckOutput { bits })
    }

    fn constraint_count() -> usize {
        // BITS boolean constraints + 1 equality check
        // Plus intermediate multiplication constraints
        BITS * 2 + 1
    }

    fn name() -> &'static str {
        "RangeCheck"
    }
}

/// Decomposes a field element into bits (LSB first).
///
/// Unlike RangeCheck, this doesn't add a recomposition constraint,
/// so it's useful when you need the bits for other purposes.
pub struct ToBits<const BITS: usize>;

impl<F: IsField, const BITS: usize> Gadget<F> for ToBits<BITS> {
    type Input = FieldVar;
    type Output = Vec<BoolVar>;

    fn synthesize(
        builder: &mut CircuitBuilder<F>,
        input: Self::Input,
    ) -> Result<Self::Output, GadgetError> {
        let range_check_result = RangeCheck::<BITS>::synthesize(
            builder,
            RangeCheckInput {
                value: input,
                bit_hints: None,
            },
        )?;
        Ok(range_check_result.bits)
    }

    fn constraint_count() -> usize {
        // Same as RangeCheck: BITS boolean constraints + 1 equality + BITS multiplications
        BITS * 2 + 1
    }

    fn name() -> &'static str {
        "ToBits"
    }
}

/// Recomposes a field element from bits (LSB first).
pub struct FromBits;

impl<F: IsField> Gadget<F> for FromBits {
    type Input = Vec<BoolVar>;
    type Output = FieldVar;

    fn synthesize(
        builder: &mut CircuitBuilder<F>,
        bits: Self::Input,
    ) -> Result<Self::Output, GadgetError> {
        if bits.is_empty() {
            return Err(GadgetError::InvalidInput("bits cannot be empty".to_string()));
        }

        let mut result = builder.constant(FieldElement::<F>::zero());
        let two = FieldElement::<F>::from(2u64);
        let mut power_of_two = FieldElement::<F>::one();

        for bit in &bits {
            let scaled_bit = builder.mul_constant(bit, power_of_two.clone());
            result = builder.add(&result, &scaled_bit);
            power_of_two = &power_of_two * &two;
        }

        Ok(result)
    }

    fn constraint_count() -> usize {
        // Depends on number of bits (computed at runtime)
        0 // Can't know statically
    }

    fn name() -> &'static str {
        "FromBits"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    type F = U64PrimeField<65537>;

    #[test]
    fn test_range_check_creates_bits() {
        let mut builder = CircuitBuilder::<F>::new();

        let x = builder.public_input("x");
        let result = RangeCheck::<8>::synthesize(&mut builder, x.into()).unwrap();

        assert_eq!(result.bits.len(), 8);
    }

    #[test]
    fn test_range_check_zero_bits_error() {
        let mut builder = CircuitBuilder::<F>::new();

        let x = builder.public_input("x");
        let result = RangeCheck::<0>::synthesize(&mut builder, x.into());

        assert!(result.is_err());
    }

    #[test]
    fn test_to_bits() {
        let mut builder = CircuitBuilder::<F>::new();

        let x = builder.public_input("x");
        let bits = ToBits::<4>::synthesize(&mut builder, x).unwrap();

        assert_eq!(bits.len(), 4);
    }

    #[test]
    fn test_from_bits() {
        let mut builder = CircuitBuilder::<F>::new();

        // Create some boolean variables
        let b0 = builder.private_input("b0");
        let b1 = builder.private_input("b1");
        let b0_bool = builder.assert_bool(&b0);
        let b1_bool = builder.assert_bool(&b1);

        let _result = FromBits::synthesize(&mut builder, vec![b0_bool, b1_bool]).unwrap();
    }

    #[test]
    fn test_from_bits_empty_error() {
        let mut builder = CircuitBuilder::<F>::new();

        let result = FromBits::synthesize(&mut builder, vec![]);
        assert!(result.is_err());
    }
}
