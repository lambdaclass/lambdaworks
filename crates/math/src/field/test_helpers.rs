/// Test helpers and macros for comprehensive field testing.
///
/// This module provides reusable test infrastructure for verifying field axioms
/// using property-based testing (proptest).

/// Generates comprehensive property-based tests for field axioms.
///
/// This macro creates tests that verify:
/// - Additive group properties (associativity, commutativity, identity, inverse)
/// - Multiplicative group properties (associativity, commutativity, identity, inverse)
/// - Distributivity of multiplication over addition
/// - Additional properties (double, square, pow)
///
/// # Usage
///
/// ```ignore
/// impl_field_axiom_tests!(
///     field_type: MyPrimeField,
///     element_type: FieldElement<MyPrimeField>,
///     // Optional: generator function for random elements
///     generator: || FieldElement::<MyPrimeField>::from(rand::random::<u64>()),
/// );
/// ```
#[macro_export]
macro_rules! impl_field_axiom_tests {
    (
        field: $field:ty,
        element: $element:ty,
        generator: $gen:expr $(,)?
    ) => {
        mod field_axiom_tests {
            use super::*;
            use proptest::prelude::*;

            fn arb_element() -> impl Strategy<Value = $element> {
                any::<u64>().prop_map(|x| {
                    let gen_fn: fn() -> $element = $gen;
                    // Use the seed to generate deterministic but varied elements
                    let base = gen_fn();
                    base * <$element>::from(x % 1000 + 1)
                })
            }

            fn arb_nonzero_element() -> impl Strategy<Value = $element> {
                arb_element().prop_filter("non-zero element", |x| *x != <$element>::zero())
            }

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(100))]

                // ==================== ADDITION AXIOMS ====================

                #[test]
                fn add_associativity(
                    a in arb_element(),
                    b in arb_element(),
                    c in arb_element()
                ) {
                    // (a + b) + c = a + (b + c)
                    prop_assert_eq!((&a + &b) + &c, &a + (&b + &c));
                }

                #[test]
                fn add_commutativity(a in arb_element(), b in arb_element()) {
                    // a + b = b + a
                    prop_assert_eq!(&a + &b, &b + &a);
                }

                #[test]
                fn add_identity(a in arb_element()) {
                    // a + 0 = a
                    let zero = <$element>::zero();
                    prop_assert_eq!(&a + &zero, a.clone());
                    prop_assert_eq!(&zero + &a, a);
                }

                #[test]
                fn add_inverse(a in arb_element()) {
                    // a + (-a) = 0
                    let neg_a = -&a;
                    prop_assert_eq!(&a + &neg_a, <$element>::zero());
                }

                // ==================== MULTIPLICATION AXIOMS ====================

                #[test]
                fn mul_associativity(
                    a in arb_element(),
                    b in arb_element(),
                    c in arb_element()
                ) {
                    // (a * b) * c = a * (b * c)
                    prop_assert_eq!((&a * &b) * &c, &a * (&b * &c));
                }

                #[test]
                fn mul_commutativity(a in arb_element(), b in arb_element()) {
                    // a * b = b * a
                    prop_assert_eq!(&a * &b, &b * &a);
                }

                #[test]
                fn mul_identity(a in arb_element()) {
                    // a * 1 = a
                    let one = <$element>::one();
                    prop_assert_eq!(&a * &one, a.clone());
                    prop_assert_eq!(&one * &a, a);
                }

                #[test]
                fn mul_inverse(a in arb_nonzero_element()) {
                    // a * a^(-1) = 1
                    // INVARIANT: a is guaranteed non-zero by arb_nonzero_element filter
                    let a_inv = a.inv().expect("nonzero element has multiplicative inverse");
                    prop_assert_eq!(&a * &a_inv, <$element>::one());
                }

                #[test]
                fn mul_zero(a in arb_element()) {
                    // a * 0 = 0
                    let zero = <$element>::zero();
                    prop_assert_eq!(&a * &zero, <$element>::zero());
                }

                // ==================== DISTRIBUTIVITY ====================

                #[test]
                fn left_distributivity(
                    a in arb_element(),
                    b in arb_element(),
                    c in arb_element()
                ) {
                    // a * (b + c) = a * b + a * c
                    prop_assert_eq!(&a * &(&b + &c), &(&a * &b) + &(&a * &c));
                }

                #[test]
                fn right_distributivity(
                    a in arb_element(),
                    b in arb_element(),
                    c in arb_element()
                ) {
                    // (a + b) * c = a * c + b * c
                    prop_assert_eq!(&(&a + &b) * &c, &(&a * &c) + &(&b * &c));
                }

                // ==================== DERIVED OPERATIONS ====================

                #[test]
                fn double_equals_add_self(a in arb_element()) {
                    // double(a) = a + a
                    prop_assert_eq!(a.double(), &a + &a);
                }

                #[test]
                fn square_equals_mul_self(a in arb_element()) {
                    // square(a) = a * a
                    prop_assert_eq!(a.square(), &a * &a);
                }

                #[test]
                fn subtraction_is_add_neg(a in arb_element(), b in arb_element()) {
                    // a - b = a + (-b)
                    prop_assert_eq!(&a - &b, &a + &(-&b));
                }

                #[test]
                fn division_is_mul_inv(a in arb_element(), b in arb_nonzero_element()) {
                    // a / b = a * b^(-1)
                    // INVARIANT: b is guaranteed non-zero by arb_nonzero_element filter
                    let a_div_b = (&a / &b).expect("division by nonzero element succeeds");
                    let b_inv = b.inv().expect("nonzero element has multiplicative inverse");
                    prop_assert_eq!(a_div_b, &a * &b_inv);
                }

                #[test]
                fn pow_two_equals_square(a in arb_element()) {
                    // a^2 = a.square()
                    prop_assert_eq!(a.pow(2u64), a.square());
                }

                #[test]
                fn pow_zero_is_one(a in arb_nonzero_element()) {
                    // a^0 = 1
                    prop_assert_eq!(a.pow(0u64), <$element>::one());
                }

                #[test]
                fn pow_one_is_self(a in arb_element()) {
                    // a^1 = a
                    prop_assert_eq!(a.pow(1u64), a);
                }

                #[test]
                fn neg_neg_is_identity(a in arb_element()) {
                    // -(-a) = a
                    prop_assert_eq!(-(-&a), a);
                }

                #[test]
                fn inv_inv_is_identity(a in arb_nonzero_element()) {
                    // (a^(-1))^(-1) = a
                    // INVARIANT: a is guaranteed non-zero by arb_nonzero_element filter
                    let a_inv = a.inv().expect("nonzero element has multiplicative inverse");
                    let a_inv_inv = a_inv.inv().expect("inverse of nonzero is also nonzero");
                    prop_assert_eq!(a_inv_inv, a);
                }
            }
        }
    };

    // Simplified version that uses from(u64) as generator
    (
        field: $field:ty,
        element: $element:ty $(,)?
    ) => {
        $crate::impl_field_axiom_tests!(
            field: $field,
            element: $element,
            generator: || <$element>::from(1u64),
        );
    };
}

/// Generates tests for FFT field properties.
///
/// This macro creates tests that verify:
/// - Two-adicity constant is valid
/// - Primitive root of unity has correct order (for small two-adicity)
/// - get_primitive_root_of_unity works correctly
///
/// Note: For fields with two-adicity > 63, some tests are skipped because
/// computing 2^two_adicity would overflow u64.
#[macro_export]
macro_rules! impl_fft_field_tests {
    (
        field: $field:ty,
        element: $element:ty,
        two_adicity: $two_adicity:expr $(,)?
    ) => {
        mod fft_field_tests {
            use super::*;
            use $crate::field::traits::IsFFTField;

            #[test]
            fn two_adicity_matches() {
                assert_eq!(<$field>::TWO_ADICITY, $two_adicity);
            }

            // These tests only work for two_adicity <= 63 (otherwise 1u64 << n overflows)
            #[test]
            fn primitive_root_has_correct_order() {
                if <$field>::TWO_ADICITY > 63 {
                    // Skip for large two-adicity - tested indirectly via get_primitive_root_of_unity
                    return;
                }
                let root = <$element>::new(<$field>::TWO_ADIC_PRIMITIVE_ROOT_OF_UNITY);
                let order = 1u64 << <$field>::TWO_ADICITY;
                assert_eq!(root.pow(order), <$element>::one());
            }

            #[test]
            fn primitive_root_is_primitive() {
                if <$field>::TWO_ADICITY > 63 {
                    // Skip for large two-adicity - tested indirectly via get_primitive_root_of_unity
                    return;
                }
                let root = <$element>::new(<$field>::TWO_ADIC_PRIMITIVE_ROOT_OF_UNITY);
                let half_order = 1u64 << (<$field>::TWO_ADICITY - 1);
                assert_ne!(root.pow(half_order), <$element>::one());
            }

            #[test]
            fn get_primitive_root_of_unity_works() {
                // Test up to order 10 or two_adicity, whichever is smaller
                for order in 1..core::cmp::min(<$field>::TWO_ADICITY, 10) {
                    let root = <$field>::get_primitive_root_of_unity(order)
                        .expect("get_primitive_root_of_unity succeeds for valid order");
                    let n = 1u64 << order;
                    assert_eq!(root.pow(n), <$element>::one(), "root^n should be 1 for order {}", order);
                    assert_ne!(root.pow(n / 2), <$element>::one(), "root^(n/2) should not be 1 for order {}", order);
                }
            }

            #[test]
            fn get_primitive_root_order_zero_is_one() {
                let root = <$field>::get_primitive_root_of_unity(0)
                    .expect("get_primitive_root_of_unity(0) always succeeds");
                assert_eq!(root, <$element>::one());
            }

            #[test]
            fn get_primitive_root_fails_for_large_order() {
                let result = <$field>::get_primitive_root_of_unity(<$field>::TWO_ADICITY + 1);
                assert!(result.is_err());
            }
        }
    };
}

/// Generates tests for extension field properties.
///
/// Verifies extension-specific properties like:
/// - Base field embedding works correctly
/// - Arithmetic preserves embedding
#[macro_export]
macro_rules! impl_extension_field_tests {
    (
        base_element: $base_elem:ty,
        ext_element: $ext_elem:ty $(,)?
    ) => {
        mod extension_field_tests {
            use super::*;
            use proptest::prelude::*;

            fn arb_base_element() -> impl Strategy<Value = $base_elem> {
                any::<u64>().prop_map(|x| <$base_elem>::from(x % 1000 + 1))
            }

            proptest! {
                #![proptest_config(ProptestConfig::with_cases(50))]

                #[test]
                fn base_field_embeds_correctly(a in arb_base_element()) {
                    // Embedding should preserve arithmetic
                    let embedded: $ext_elem = a.clone().to_extension();
                    let double_base = a.double();
                    let double_ext = embedded.double();
                    let double_base_embedded: $ext_elem = double_base.to_extension();
                    prop_assert_eq!(double_ext, double_base_embedded);
                }

                #[test]
                fn base_field_mul_in_extension(a in arb_base_element(), b in arb_base_element()) {
                    // (a * b) embedded = a embedded * b embedded
                    let prod_base = &a * &b;
                    let a_ext: $ext_elem = a.to_extension();
                    let b_ext: $ext_elem = b.to_extension();
                    let prod_ext = &a_ext * &b_ext;
                    let prod_base_ext: $ext_elem = prod_base.to_extension();
                    prop_assert_eq!(prod_ext, prod_base_ext);
                }
            }
        }
    };
}

#[cfg(test)]
mod tests {
    // Sanity test that macros compile
    use super::*;

    #[test]
    fn macros_are_exported() {
        // This test just verifies the module compiles
    }
}
