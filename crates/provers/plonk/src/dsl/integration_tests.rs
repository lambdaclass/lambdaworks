//! End-to-end integration tests for the PLONK DSL.
//!
//! These tests verify that circuits built with the DSL correctly
//! prove and verify through the complete PLONK pipeline.

#[cfg(test)]
mod tests {
    use crate::dsl::builder::CircuitBuilder;
    use crate::dsl::gadgets::arithmetic::RangeCheck;
    use crate::dsl::gadgets::comparison::{IsEqual, IsZero};
    use crate::dsl::gadgets::Gadget;
    use crate::prover::Prover;
    use crate::setup::setup;
    use crate::test_utils::utils::{
        test_srs, TestRandomFieldGenerator, KZG, ORDER_R_MINUS_1_ROOT_UNITY,
    };
    use crate::verifier::Verifier;
    use lambdaworks_math::elliptic_curve::short_weierstrass::curves::bls12_381::default_types::{
        FrElement, FrField,
    };

    /// Helper to run the complete prove/verify cycle for a circuit.
    fn prove_and_verify(builder: &CircuitBuilder<FrField>, inputs: &[(&str, FrElement)]) -> bool {
        // Build common preprocessed input
        let cpi = builder
            .build_cpi(&ORDER_R_MINUS_1_ROOT_UNITY)
            .expect("Failed to build CPI");

        // Build witness
        let witness = builder
            .build_witness(inputs)
            .expect("Failed to build witness");

        // Extract public inputs in correct order
        let public_inputs = builder
            .extract_public_inputs(inputs)
            .expect("Failed to extract public inputs");

        // Setup
        let srs = test_srs(cpi.n);
        let kzg = KZG::new(srs);
        let vk = setup(&cpi, &kzg);

        // Prove
        let prover = Prover::new(kzg.clone(), TestRandomFieldGenerator);
        let proof = prover
            .prove(&witness, &public_inputs, &cpi, &vk)
            .expect("Failed to create proof");

        // Verify
        let verifier = Verifier::new(kzg);
        verifier.verify(&proof, &public_inputs, &cpi, &vk)
    }

    #[test]
    fn test_dsl_simple_multiplication() {
        // Circuit: x * e == y
        let mut builder = CircuitBuilder::<FrField>::new();

        let x = builder.public_input("x");
        let y = builder.public_input("y");
        let e = builder.private_input("e");

        let product = builder.mul(&x, &e);
        builder.assert_eq(&product, &y);

        // Test with x=4, e=3, y=12 (4 * 3 = 12)
        let result = prove_and_verify(
            &builder,
            &[
                ("x", FrElement::from(4u64)),
                ("y", FrElement::from(12u64)),
                ("e", FrElement::from(3u64)),
            ],
        );

        assert!(
            result,
            "Proof verification failed for simple multiplication"
        );
    }

    #[test]
    fn test_dsl_addition() {
        // Circuit: a + b == c
        let mut builder = CircuitBuilder::<FrField>::new();

        let a = builder.public_input("a");
        let b = builder.public_input("b");
        let c = builder.public_input("c");

        let sum = builder.add(&a, &b);
        builder.assert_eq(&sum, &c);

        // Test with a=5, b=7, c=12
        let result = prove_and_verify(
            &builder,
            &[
                ("a", FrElement::from(5u64)),
                ("b", FrElement::from(7u64)),
                ("c", FrElement::from(12u64)),
            ],
        );

        assert!(result, "Proof verification failed for addition");
    }

    #[test]
    fn test_dsl_subtraction() {
        // Circuit: a - b == c
        let mut builder = CircuitBuilder::<FrField>::new();

        let a = builder.public_input("a");
        let b = builder.public_input("b");
        let c = builder.public_input("c");

        let diff = builder.sub(&a, &b);
        builder.assert_eq(&diff, &c);

        // Test with a=10, b=3, c=7
        let result = prove_and_verify(
            &builder,
            &[
                ("a", FrElement::from(10u64)),
                ("b", FrElement::from(3u64)),
                ("c", FrElement::from(7u64)),
            ],
        );

        assert!(result, "Proof verification failed for subtraction");
    }

    #[test]
    fn test_dsl_boolean_constraint() {
        // Circuit: prove b is boolean and b * x == y
        let mut builder = CircuitBuilder::<FrField>::new();

        let b = builder.public_input("b");
        let x = builder.public_input("x");
        let y = builder.public_input("y");

        let b_bool = builder.assert_bool(&b);
        let product = builder.mul(&b_bool, &x);
        builder.assert_eq(&product, &y);

        // Test with b=1, x=42, y=42 (1 * 42 = 42)
        let result = prove_and_verify(
            &builder,
            &[
                ("b", FrElement::from(1u64)),
                ("x", FrElement::from(42u64)),
                ("y", FrElement::from(42u64)),
            ],
        );

        assert!(result, "Proof verification failed for boolean constraint");
    }

    #[test]
    fn test_dsl_select_mux() {
        // Circuit: result = if cond then a else b
        let mut builder = CircuitBuilder::<FrField>::new();

        let cond = builder.public_input("cond");
        let a = builder.public_input("a");
        let b = builder.public_input("b");
        let expected = builder.public_input("expected");

        let cond_bool = builder.assert_bool(&cond);
        let result = builder.select(&cond_bool, &a, &b);
        builder.assert_eq(&result, &expected);

        // Test with cond=1, a=100, b=200, expected=100
        let result1 = prove_and_verify(
            &builder,
            &[
                ("cond", FrElement::from(1u64)),
                ("a", FrElement::from(100u64)),
                ("b", FrElement::from(200u64)),
                ("expected", FrElement::from(100u64)),
            ],
        );
        assert!(result1, "Select with cond=1 failed");

        // Test with cond=0, a=100, b=200, expected=200
        let result0 = prove_and_verify(
            &builder,
            &[
                ("cond", FrElement::from(0u64)),
                ("a", FrElement::from(100u64)),
                ("b", FrElement::from(200u64)),
                ("expected", FrElement::from(200u64)),
            ],
        );
        assert!(result0, "Select with cond=0 failed");
    }

    #[test]
    fn test_dsl_chained_operations() {
        // Circuit: ((a * b) + c) * d == result
        let mut builder = CircuitBuilder::<FrField>::new();

        let a = builder.public_input("a");
        let b = builder.private_input("b");
        let c = builder.private_input("c");
        let d = builder.private_input("d");
        let result = builder.public_input("result");

        let ab = builder.mul(&a, &b);
        let ab_plus_c = builder.add(&ab, &c);
        let final_val = builder.mul(&ab_plus_c, &d);
        builder.assert_eq(&final_val, &result);

        // Test: ((2 * 3) + 4) * 5 = (6 + 4) * 5 = 10 * 5 = 50
        let verified = prove_and_verify(
            &builder,
            &[
                ("a", FrElement::from(2u64)),
                ("b", FrElement::from(3u64)),
                ("c", FrElement::from(4u64)),
                ("d", FrElement::from(5u64)),
                ("result", FrElement::from(50u64)),
            ],
        );

        assert!(verified, "Chained operations verification failed");
    }

    #[test]
    #[ignore = "Requires witness solver hints for internal gadget variables"]
    fn test_dsl_is_zero_gadget() {
        // Circuit: prove that a value is zero
        // NOTE: This test requires the constraint solver to compute internal
        // gadget variables (is_zero, inv). Needs hint support in gadgets.
        let mut builder = CircuitBuilder::<FrField>::new();

        let x = builder.public_input("x");
        let expected = builder.public_input("expected");

        let is_zero_result = IsZero::synthesize(&mut builder, x).expect("IsZero synthesis failed");
        builder.assert_eq(&is_zero_result, &expected);

        // Test with x=0, expected=1 (0 is zero)
        let result_zero = prove_and_verify(
            &builder,
            &[
                ("x", FrElement::from(0u64)),
                ("expected", FrElement::from(1u64)),
            ],
        );
        assert!(result_zero, "IsZero(0) should return 1");

        // Test with x=5, expected=0 (5 is not zero)
        let result_nonzero = prove_and_verify(
            &builder,
            &[
                ("x", FrElement::from(5u64)),
                ("expected", FrElement::from(0u64)),
            ],
        );
        assert!(result_nonzero, "IsZero(5) should return 0");
    }

    #[test]
    #[ignore = "Requires witness solver hints for internal gadget variables"]
    fn test_dsl_is_equal_gadget() {
        // Circuit: prove equality check
        // NOTE: Uses IsZero internally which needs hint support.
        let mut builder = CircuitBuilder::<FrField>::new();

        let a = builder.public_input("a");
        let b = builder.public_input("b");
        let expected = builder.public_input("expected");

        let is_eq = IsEqual::synthesize(&mut builder, (a, b)).expect("IsEqual synthesis failed");
        builder.assert_eq(&is_eq, &expected);

        // Test equal values: a=42, b=42, expected=1
        let result_equal = prove_and_verify(
            &builder,
            &[
                ("a", FrElement::from(42u64)),
                ("b", FrElement::from(42u64)),
                ("expected", FrElement::from(1u64)),
            ],
        );
        assert!(result_equal, "IsEqual(42, 42) should return 1");

        // Test unequal values: a=10, b=20, expected=0
        let result_unequal = prove_and_verify(
            &builder,
            &[
                ("a", FrElement::from(10u64)),
                ("b", FrElement::from(20u64)),
                ("expected", FrElement::from(0u64)),
            ],
        );
        assert!(result_unequal, "IsEqual(10, 20) should return 0");
    }

    #[test]
    #[ignore = "Requires witness solver hints for bit decomposition"]
    fn test_dsl_range_check_gadget() {
        // Circuit: prove value is in range [0, 2^8)
        // NOTE: Bit decomposition requires hints for individual bit values.
        let mut builder = CircuitBuilder::<FrField>::new();

        let x = builder.public_input("x");

        // Range check for 8 bits (0-255)
        let _bits = RangeCheck::<8>::synthesize(&mut builder, x.into())
            .expect("RangeCheck synthesis failed");

        // Test with valid value x=200 (within 0-255)
        let result = prove_and_verify(&builder, &[("x", FrElement::from(200u64))]);
        assert!(result, "RangeCheck for 200 (valid 8-bit) failed");
    }

    #[test]
    fn test_dsl_logical_operations() {
        // Circuit: test AND, OR, XOR
        let mut builder = CircuitBuilder::<FrField>::new();

        let a = builder.public_input("a");
        let b = builder.public_input("b");
        let and_expected = builder.public_input("and_result");
        let or_expected = builder.public_input("or_result");

        let a_bool = builder.assert_bool(&a);
        let b_bool = builder.assert_bool(&b);

        let and_result = builder.and(&a_bool, &b_bool);
        let or_result = builder.or(&a_bool, &b_bool);

        builder.assert_eq(&and_result, &and_expected);
        builder.assert_eq(&or_result, &or_expected);

        // Test: a=1, b=1 => AND=1, OR=1
        let result_11 = prove_and_verify(
            &builder,
            &[
                ("a", FrElement::from(1u64)),
                ("b", FrElement::from(1u64)),
                ("and_result", FrElement::from(1u64)),
                ("or_result", FrElement::from(1u64)),
            ],
        );
        assert!(result_11, "Logical ops with a=1, b=1 failed");

        // Test: a=1, b=0 => AND=0, OR=1
        let result_10 = prove_and_verify(
            &builder,
            &[
                ("a", FrElement::from(1u64)),
                ("b", FrElement::from(0u64)),
                ("and_result", FrElement::from(0u64)),
                ("or_result", FrElement::from(1u64)),
            ],
        );
        assert!(result_10, "Logical ops with a=1, b=0 failed");

        // Test: a=0, b=0 => AND=0, OR=0
        let result_00 = prove_and_verify(
            &builder,
            &[
                ("a", FrElement::from(0u64)),
                ("b", FrElement::from(0u64)),
                ("and_result", FrElement::from(0u64)),
                ("or_result", FrElement::from(0u64)),
            ],
        );
        assert!(result_00, "Logical ops with a=0, b=0 failed");
    }

    #[test]
    fn test_dsl_private_computation() {
        // Circuit proving knowledge of preimage: hash(secret) == public_hash
        // Simplified: secret^2 == public_value
        let mut builder = CircuitBuilder::<FrField>::new();

        let public_value = builder.public_input("public_value");
        let secret = builder.private_input("secret");

        let squared = builder.mul(&secret, &secret);
        builder.assert_eq(&squared, &public_value);

        // Prove knowledge of secret=7 such that 7^2 = 49
        let result = prove_and_verify(
            &builder,
            &[
                ("public_value", FrElement::from(49u64)),
                ("secret", FrElement::from(7u64)),
            ],
        );

        assert!(result, "Private computation proof failed");
    }
}
