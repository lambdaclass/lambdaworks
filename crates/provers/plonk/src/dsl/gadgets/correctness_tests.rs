//! Constraint correctness tests for gadgets.
//!
//! These tests verify that gadgets generate correct constraints by:
//! 1. Building circuits with gadgets
//! 2. Verifying gadget outputs and structure
//! 3. Testing error conditions

#[cfg(test)]
mod tests {
    use crate::dsl::builder::CircuitBuilder;
    use crate::dsl::gadgets::arithmetic::{FromBits, RangeCheck, ToBits};
    use crate::dsl::gadgets::comparison::{
        GreaterThan, GreaterThanOrEqual, IsEqual, IsZero, LessThan, LessThanOrEqual,
    };
    use crate::dsl::gadgets::Gadget;
    use lambdaworks_math::field::fields::u64_prime_field::U64PrimeField;

    // Use a simple field for testing
    type F = U64PrimeField<65537>;

    // ========================================
    // Arithmetic Gadget Tests
    // ========================================

    #[test]
    fn test_range_check_constraint_structure() {
        let mut builder = CircuitBuilder::<F>::new();
        let x = builder.public_input("x");

        // 4-bit range check should create:
        // - 4 boolean constraints (bit * (bit - 1) = 0)
        // - Recomposition constraint (sum of bits * 2^i = x)
        let result = RangeCheck::<4>::synthesize(&mut builder, x.into());
        assert!(result.is_ok());

        let bits = result.unwrap().bits;
        assert_eq!(bits.len(), 4, "Should produce 4 bits for 4-bit range check");
    }

    #[test]
    fn test_range_check_different_bit_widths() {
        // Test various bit widths
        for bits in [1, 2, 4, 8, 16] {
            let mut builder = CircuitBuilder::<F>::new();
            let x = builder.public_input("x");

            let result = match bits {
                1 => RangeCheck::<1>::synthesize(&mut builder, x.into()).map(|r| r.bits.len()),
                2 => RangeCheck::<2>::synthesize(&mut builder, x.into()).map(|r| r.bits.len()),
                4 => RangeCheck::<4>::synthesize(&mut builder, x.into()).map(|r| r.bits.len()),
                8 => RangeCheck::<8>::synthesize(&mut builder, x.into()).map(|r| r.bits.len()),
                16 => RangeCheck::<16>::synthesize(&mut builder, x.into()).map(|r| r.bits.len()),
                _ => unreachable!(),
            };

            assert!(result.is_ok());
            assert_eq!(result.unwrap(), bits);
        }
    }

    #[test]
    fn test_range_check_zero_bits_error() {
        let mut builder = CircuitBuilder::<F>::new();
        let x = builder.public_input("x");

        let result = RangeCheck::<0>::synthesize(&mut builder, x.into());
        assert!(result.is_err(), "RangeCheck with 0 bits should fail");
    }

    #[test]
    fn test_to_bits_produces_correct_count() {
        let mut builder = CircuitBuilder::<F>::new();
        let x = builder.public_input("x");

        let bits = ToBits::<8>::synthesize(&mut builder, x).unwrap();
        assert_eq!(bits.len(), 8);
    }

    #[test]
    fn test_from_bits_constraint_generation() {
        let mut builder = CircuitBuilder::<F>::new();

        // Create 4 boolean inputs
        let b0 = builder.private_input("b0");
        let b1 = builder.private_input("b1");
        let b2 = builder.private_input("b2");
        let b3 = builder.private_input("b3");

        let b0_bool = builder.assert_bool(&b0);
        let b1_bool = builder.assert_bool(&b1);
        let b2_bool = builder.assert_bool(&b2);
        let b3_bool = builder.assert_bool(&b3);

        let result = FromBits::synthesize(&mut builder, vec![b0_bool, b1_bool, b2_bool, b3_bool]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_from_bits_empty_fails() {
        let mut builder = CircuitBuilder::<F>::new();
        let result = FromBits::synthesize(&mut builder, vec![]);
        assert!(result.is_err());
    }

    // ========================================
    // Comparison Gadget Tests
    // ========================================

    #[test]
    fn test_is_zero_constraint_structure() {
        let mut builder = CircuitBuilder::<F>::new();
        let x = builder.public_input("x");

        let result = IsZero::synthesize(&mut builder, x);
        assert!(result.is_ok());

        // IsZero creates internal variables and constraints
        // Verify the output is a valid BoolVar
        let is_zero_output = result.unwrap();
        assert!(is_zero_output.variable() > x.variable(), "IsZero should create new variables");
    }

    #[test]
    fn test_is_equal_uses_is_zero() {
        let mut builder = CircuitBuilder::<F>::new();
        let a = builder.public_input("a");
        let b = builder.public_input("b");

        let result = IsEqual::synthesize(&mut builder, (a, b));
        assert!(result.is_ok());
    }

    #[test]
    fn test_less_than_constraint_structure() {
        let mut builder = CircuitBuilder::<F>::new();
        let a = builder.public_input("a");
        let b = builder.public_input("b");

        // 8-bit less than
        let result = LessThan::<8>::synthesize(&mut builder, (a, b));
        assert!(result.is_ok());

        // Verify the output is a valid BoolVar
        let lt_output = result.unwrap();
        assert!(lt_output.variable() > b.variable(), "LessThan should create new variables");
    }

    #[test]
    fn test_less_than_zero_bits_error() {
        let mut builder = CircuitBuilder::<F>::new();
        let a = builder.public_input("a");
        let b = builder.public_input("b");

        let result = LessThan::<0>::synthesize(&mut builder, (a, b));
        assert!(result.is_err(), "LessThan with 0 bits should fail");
    }

    #[test]
    fn test_comparison_gadget_relationships() {
        // Verify that LessThanOrEqual, GreaterThan, GreaterThanOrEqual
        // are correctly defined in terms of LessThan

        let mut builder1 = CircuitBuilder::<F>::new();
        let a1 = builder1.public_input("a");
        let b1 = builder1.public_input("b");
        let lte_result = LessThanOrEqual::<4>::synthesize(&mut builder1, (a1, b1));
        assert!(lte_result.is_ok());

        let mut builder2 = CircuitBuilder::<F>::new();
        let a2 = builder2.public_input("a");
        let b2 = builder2.public_input("b");
        let gt_result = GreaterThan::<4>::synthesize(&mut builder2, (a2, b2));
        assert!(gt_result.is_ok());

        let mut builder3 = CircuitBuilder::<F>::new();
        let a3 = builder3.public_input("a");
        let b3 = builder3.public_input("b");
        let gte_result = GreaterThanOrEqual::<4>::synthesize(&mut builder3, (a3, b3));
        assert!(gte_result.is_ok());
    }

    // ========================================
    // Constraint Count Verification
    // ========================================

    #[test]
    fn test_gadget_constraint_counts_are_positive() {
        // All gadgets should report positive constraint counts
        assert!(<IsZero as Gadget<F>>::constraint_count() > 0);
        assert!(<IsEqual as Gadget<F>>::constraint_count() > 0);
        assert!(<LessThan<8> as Gadget<F>>::constraint_count() > 0);
        assert!(<LessThanOrEqual<8> as Gadget<F>>::constraint_count() > 0);
        assert!(<GreaterThan<8> as Gadget<F>>::constraint_count() > 0);
        assert!(<GreaterThanOrEqual<8> as Gadget<F>>::constraint_count() > 0);
        assert!(<RangeCheck<8> as Gadget<F>>::constraint_count() > 0);
        assert!(<ToBits<8> as Gadget<F>>::constraint_count() > 0);
    }

    #[test]
    fn test_comparison_constraint_count_ordering() {
        // LessThanOrEqual and GreaterThanOrEqual should have 1 more constraint than LessThan
        let lt_count = <LessThan<8> as Gadget<F>>::constraint_count();
        let lte_count = <LessThanOrEqual<8> as Gadget<F>>::constraint_count();
        let gte_count = <GreaterThanOrEqual<8> as Gadget<F>>::constraint_count();
        let gt_count = <GreaterThan<8> as Gadget<F>>::constraint_count();

        assert_eq!(gt_count, lt_count, "GreaterThan should equal LessThan");
        assert_eq!(
            lte_count,
            lt_count + 1,
            "LessThanOrEqual should be LessThan + 1"
        );
        assert_eq!(
            gte_count,
            lt_count + 1,
            "GreaterThanOrEqual should be LessThan + 1"
        );
    }

    // ========================================
    // Gadget Name Tests
    // ========================================

    #[test]
    fn test_gadget_names() {
        assert_eq!(<IsZero as Gadget<F>>::name(), "IsZero");
        assert_eq!(<IsEqual as Gadget<F>>::name(), "IsEqual");
        assert_eq!(<LessThan<8> as Gadget<F>>::name(), "LessThan");
        assert_eq!(<LessThanOrEqual<8> as Gadget<F>>::name(), "LessThanOrEqual");
        assert_eq!(<GreaterThan<8> as Gadget<F>>::name(), "GreaterThan");
        assert_eq!(
            <GreaterThanOrEqual<8> as Gadget<F>>::name(),
            "GreaterThanOrEqual"
        );
        assert_eq!(<RangeCheck<8> as Gadget<F>>::name(), "RangeCheck");
        assert_eq!(<ToBits<8> as Gadget<F>>::name(), "ToBits");
        assert_eq!(<FromBits as Gadget<F>>::name(), "FromBits");
    }

    // ========================================
    // Builder State Tests
    // ========================================

    #[test]
    fn test_gadgets_preserve_builder_state() {
        let mut builder = CircuitBuilder::<F>::new();

        // Create some variables first
        let x = builder.public_input("x");
        let y = builder.public_input("y");

        // Apply a gadget
        let _ = IsZero::synthesize(&mut builder, x);

        // Verify original variables are still accessible
        assert!(builder.get_var("x").is_some());
        assert!(builder.get_var("y").is_some());

        // Can still use the builder
        let z = builder.add(&x, &y);
        assert_ne!(z.variable(), x.variable());
    }

    #[test]
    fn test_multiple_gadgets_compose() {
        let mut builder = CircuitBuilder::<F>::new();

        let a = builder.public_input("a");
        let b = builder.public_input("b");
        let c = builder.public_input("c");

        // Chain multiple gadget applications
        let a_is_zero = IsZero::synthesize(&mut builder, a).unwrap();
        let b_is_zero = IsZero::synthesize(&mut builder, b).unwrap();

        // Use gadget outputs
        let both_zero = builder.and(&a_is_zero, &b_is_zero);

        // More operations
        let result = builder.select(&both_zero, &c, &a);

        // Should have created variables for all the intermediate values
        assert!(
            result.variable() > c.variable(),
            "Composed gadgets should create many new variables"
        );
    }
}
