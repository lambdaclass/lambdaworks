#![no_main]
//! Fuzz tests for BabyBear field.
//!
//! Tests verify:
//! - Field axioms (commutativity, associativity, distributivity)
//! - Arithmetic operations (add, sub, mul, inv, pow)
//! - Edge cases (0, 1, p-1, near-modulus values)
//! - Montgomery representation correctness

use lambdaworks_math::{
    field::{
        element::FieldElement, fields::fft_friendly::babybear::Babybear31PrimeField,
        traits::IsFFTField,
    },
    traits::ByteConversion,
};
use libfuzzer_sys::fuzz_target;

type FE = FieldElement<Babybear31PrimeField>;

// BabyBear prime: 2^31 - 2^27 + 1
const P: u64 = 2013265921;

fuzz_target!(|data: (u32, u32, u32, u8)| {
    let (a_val, b_val, c_val, op_selector) = data;

    // Create field elements
    let a = FE::from(a_val as u64);
    let b = FE::from(b_val as u64);
    let c = FE::from(c_val as u64);

    // ===== TEST 1: Addition commutativity =====
    // a + b = b + a
    assert_eq!(
        &a + &b,
        &b + &a,
        "Addition commutativity failed: a={}, b={}",
        a_val,
        b_val
    );

    // ===== TEST 2: Addition associativity =====
    // (a + b) + c = a + (b + c)
    assert_eq!(
        (&a + &b) + &c,
        &a + (&b + &c),
        "Addition associativity failed"
    );

    // ===== TEST 3: Multiplication commutativity =====
    // a * b = b * a
    assert_eq!(
        &a * &b,
        &b * &a,
        "Multiplication commutativity failed: a={}, b={}",
        a_val,
        b_val
    );

    // ===== TEST 4: Multiplication associativity =====
    // (a * b) * c = a * (b * c)
    assert_eq!(
        (&a * &b) * &c,
        &a * (&b * &c),
        "Multiplication associativity failed"
    );

    // ===== TEST 5: Distributivity =====
    // a * (b + c) = a * b + a * c
    assert_eq!(
        &a * (&b + &c),
        (&a * &b) + (&a * &c),
        "Distributivity failed: a={}, b={}, c={}",
        a_val,
        b_val,
        c_val
    );

    // ===== TEST 6: Additive identity =====
    // a + 0 = a
    assert_eq!(&a + &FE::zero(), a, "Additive identity failed");

    // ===== TEST 7: Multiplicative identity =====
    // a * 1 = a
    assert_eq!(&a * &FE::one(), a, "Multiplicative identity failed");

    // ===== TEST 8: Additive inverse =====
    // a + (-a) = 0
    assert_eq!(
        &a + &-&a,
        FE::zero(),
        "Additive inverse failed: a={}",
        a_val
    );

    // ===== TEST 9: Multiplicative inverse (if a != 0) =====
    // a * a^(-1) = 1 for a != 0
    if a != FE::zero() {
        match a.inv() {
            Ok(a_inv) => {
                assert_eq!(
                    &a * &a_inv,
                    FE::one(),
                    "Multiplicative inverse failed: a={}, a_inv representative={:?}",
                    a_val,
                    a_inv.representative()
                );
            }
            Err(_) => panic!("Inverse of non-zero element failed: a={}", a_val),
        }
    } else {
        // Inverse of zero should fail
        assert!(a.inv().is_err(), "Inverse of zero should error");
    }

    // ===== TEST 10: Subtraction consistency =====
    // a - b = a + (-b)
    assert_eq!(&a - &b, &a + &-&b, "Subtraction consistency failed");

    // ===== TEST 11: Multiplication by zero =====
    // a * 0 = 0
    assert_eq!(
        &a * &FE::zero(),
        FE::zero(),
        "Multiplication by zero failed"
    );

    // ===== TEST 12: Squaring =====
    // a^2 = a * a
    assert_eq!(a.square(), &a * &a, "Squaring failed: a={}", a_val);

    // ===== TEST 13: Doubling =====
    // 2*a = a + a
    assert_eq!(a.double(), &a + &a, "Doubling failed: a={}", a_val);

    // ===== TEST 14: Power consistency =====
    // Test small powers
    let exp = (op_selector % 10) as u64;
    let mut pow_result = FE::one();
    for _ in 0..exp {
        pow_result = &pow_result * &a;
    }
    assert_eq!(
        a.pow(exp),
        pow_result,
        "Power consistency failed: a={}, exp={}",
        a_val,
        exp
    );

    // ===== TEST 15: Fermat's little theorem =====
    // a^(p-1) = 1 for a != 0
    if a != FE::zero() && op_selector % 50 == 0 {
        // Only test occasionally (expensive)
        let a_pow_p_minus_1 = a.pow(P - 1);
        assert_eq!(
            a_pow_p_minus_1,
            FE::one(),
            "Fermat's little theorem failed: a={}, a^(p-1)={:?}",
            a_val,
            a_pow_p_minus_1.representative()
        );
    }

    // ===== TEST 16: Modular reduction correctness =====
    // Verify that representatives are in range [0, p)
    let a_rep = a.representative();
    assert!(a_rep < P as u32, "Representative out of range: {:?}", a_rep);

    // ===== TEST 17: Edge case - operations with p-1 =====
    let p_minus_1 = FE::from(P - 1);

    // (p-1) + 1 = 0 mod p
    assert_eq!(&p_minus_1 + &FE::one(), FE::zero(), "p-1 + 1 != 0");

    // (p-1) * (p-1) = 1 mod p (since -1 * -1 = 1)
    assert_eq!(&p_minus_1 * &p_minus_1, FE::one(), "(-1) * (-1) != 1");

    // ===== TEST 18: Division consistency =====
    // (a / b) * b = a for b != 0
    if b != FE::zero() {
        match &a / &b {
            Ok(quotient) => {
                assert_eq!(
                    &quotient * &b,
                    a,
                    "Division consistency failed: a={}, b={}",
                    a_val,
                    b_val
                );
            }
            Err(_) => panic!("Division failed unexpectedly: a={}, b={}", a_val, b_val),
        }
    }

    // ===== TEST 19: Negation is additive inverse =====
    // -(-a) = a
    assert_eq!(-&-&a, a, "Double negation failed: a={}", a_val);

    // ===== TEST 20: From/to bytes roundtrip =====
    let a_bytes = a.to_bytes_le();
    let a_from_bytes = FE::from_bytes_le(&a_bytes).expect("from_bytes_le failed");
    assert_eq!(
        a_from_bytes,
        a,
        "Bytes roundtrip failed: original={:?}, after={:?}",
        a.representative(),
        a_from_bytes.representative()
    );

    // ===== TEST 21: Two-adicity and root of unity =====
    // Verify the two-adic root of unity
    if op_selector == 0 {
        // Only test once per fuzz input
        let omega = FE::from(Babybear31PrimeField::TWO_ADIC_PRIMITVE_ROOT_OF_UNITY as u64);
        let two_adicity = Babybear31PrimeField::TWO_ADICITY;

        // omega^(2^n) = 1 where n is two-adicity (24 for BabyBear)
        let omega_pow = omega.pow(1u64 << two_adicity);
        assert_eq!(
            omega_pow,
            FE::one(),
            "Two-adic root of unity property failed: omega^(2^{}) != 1",
            two_adicity
        );

        // omega^(2^(n-1)) != 1 (primitive root check)
        let omega_pow_half = omega.pow(1u64 << (two_adicity - 1));
        assert_ne!(
            omega_pow_half,
            FE::one(),
            "Root of unity not primitive: omega^(2^{}) = 1",
            two_adicity - 1
        );
    }

    // ===== TEST 22: Batch operations consistency =====
    // Test that batch addition matches individual additions
    if op_selector % 20 == 0 {
        let elements = vec![a.clone(), b.clone(), c.clone()];
        let sum = elements.iter().fold(FE::zero(), |acc, x| &acc + x);
        let expected = &(&a + &b) + &c;
        assert_eq!(sum, expected, "Batch addition consistency failed");
    }

    // ===== TEST 23: Zero behavior =====
    let zero = FE::zero();
    assert_eq!(zero, FE::zero(), "Zero detection failed");
    assert_eq!(&zero + &zero, zero, "0 + 0 != 0");
    assert_eq!(&zero * &a, zero, "0 * a != 0");
    assert_eq!(-&zero, zero, "-0 != 0");

    // ===== TEST 24: One behavior =====
    let one = FE::one();
    assert_eq!(one, FE::one(), "One detection failed");
    assert_eq!(&one * &one, one, "1 * 1 != 1");
    assert_ne!(one, zero, "1 == 0");

    // ===== TEST 25: Multiplication is repeated addition =====
    // Verify for small multipliers
    let multiplier = (op_selector % 5) as u64 + 1;
    let mut repeated_add = FE::zero();
    for _ in 0..multiplier {
        repeated_add = &repeated_add + &a;
    }
    assert_eq!(
        &a * &FE::from(multiplier),
        repeated_add,
        "Multiplication != repeated addition: a={}, mult={}",
        a_val,
        multiplier
    );
});
