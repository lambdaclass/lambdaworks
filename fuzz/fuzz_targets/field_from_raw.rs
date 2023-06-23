#![no_main]

use libfuzzer_sys::fuzz_target;
use lambdaworks_math::field::{
    element::FieldElement, 
    fields::fft_friendly::stark_252_prime_field::Stark252PrimeField
};
use ibig::{modular::ModuloRing, UBig};
use lambdaworks_math::unsigned_integer::element::UnsignedInteger;

fuzz_target!(|values: (u64, u64)| {
    let (value_u64_a, value_u64_b) = values;

    // Basic checks against from(u64)
    let a_expected = FieldElement::<Stark252PrimeField>::from(value_u64_a);
    let b_expected = FieldElement::<Stark252PrimeField>::from(value_u64_b);

    let value_a = UnsignedInteger::from(value_u64_a);
    let value_b = UnsignedInteger::from(value_u64_b);

    let a = FieldElement::<Stark252PrimeField>::from_raw(&value_a);
    let b = FieldElement::<Stark252PrimeField>::from_raw(&value_b);

    let add = &a + &b;
    let expected_add = &a_expected + &b_expected;
    assert_eq!(&(add.to_string())[2..], &(expected_add.to_string())[2..]);

    let sub = &a - &b;
    let expected_sub = &a_expected - &b_expected;
    assert_eq!(&(sub.to_string())[2..], &(expected_sub.to_string())[2..]);

    let mul = &a * &b;
    let expected_mul = &a_expected * &b_expected;
    assert_eq!(&(mul.to_string())[2..], &(expected_mul.to_string())[2..]);

    // if !value_b.chars().all(|c| matches!(c, '0')) { 
    //     let div = &a / &b; 
    //     let expected_div = &a_expected / &b_expected;
    //     assert_eq!(&(div.to_string())[2..], expected_div.residue().in_radix(16).to_string());
    // }

    let pow = &a.pow(b.representative());
    let expected_pow = a_expected.pow(b_expected.representative());
    assert_eq!(&(pow.to_string())[2..], &(expected_pow.to_string())[2..]);

    for n in [&a, &b] {
        match n.sqrt() {
            Some((fst_sqrt, snd_sqrt)) => {
                assert_eq!(fst_sqrt.square(), snd_sqrt.square(), "Squared roots don't match each other");
                assert_eq!(n, &fst_sqrt.square(), "Squared roots don't match original number");
            }
            None => {}
        };
    }

    // Axioms soundness
    
    let one = FieldElement::<Stark252PrimeField>::one();
    let zero = FieldElement::<Stark252PrimeField>::zero();

    assert_eq!(&a + &zero, a, "Neutral add element a failed");
    assert_eq!(&b + &zero, b, "Neutral mul element b failed");
    assert_eq!(&a * &one, a, "Neutral add element a failed");
    assert_eq!(&b * &one, b, "Neutral mul element b failed");

    assert_eq!(&a + &b, &b + &a, "Commutative add property failed");
    assert_eq!(&a * &b, &b * &a, "Commutative mul property failed");

    let c = &a * &b;
    assert_eq!((&a + &b) + &c, &a + (&b + &c), "Associative add property failed");
    assert_eq!((&a * &b) * &c, &a * (&b * &c), "Associative mul property failed");

    assert_eq!(&a * (&b + &c), &a * &b + &a * &c, "Distributive property failed");

    assert_eq!(&a - &a, zero, "Inverse add a failed");
    assert_eq!(&b - &b, zero, "Inverse add b failed");

    if a != zero {
        assert_eq!(&a * a.inv(), one, "Inverse mul a failed");
    }
    if b != zero {
        assert_eq!(&b * b.inv(), one, "Inverse mul b failed");
    }
    
});

