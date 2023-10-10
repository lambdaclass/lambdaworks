#![no_main]

use libfuzzer_sys::fuzz_target;
use lambdaworks_math::field::{
    element::FieldElement, 
    fields::{
        mersenne31::field::Mersenne31Field,
        fft_friendly::u64_mersenne_montgomery_field::Mersenne31MontgomeryPrimeField,
    }
};

fuzz_target!(|values: (u32, u32)| {

    let (value_u32_a, value_u32_b) = values;
   
    let a =  FieldElement::<Mersenne31Field>::from(value_u32_a as u64);
    let b =  FieldElement::<Mersenne31Field>::from(value_u32_b as u64);

    let a_expected = FieldElement::<Mersenne31MontgomeryPrimeField>::from(value_u32_a as u64);
    let b_expected = FieldElement::<Mersenne31MontgomeryPrimeField>::from(value_u32_b as u64);

    let add_u32 = &a + &b;
    let addition = &a_expected + &b_expected;
    
    assert_eq!(add_u32.representative() as u64, addition.representative().limbs[0]);

    let sub_u32 = &a - &b;
    let substraction = &a_expected - &b_expected;
    assert_eq!(sub_u32.representative() as u64, substraction.representative().limbs[0]);
    
    let mul_u32 = &a * &b;
    let multiplication = &a_expected * &b_expected;
    assert_eq!(mul_u32.representative() as u64, multiplication.representative().limbs[0]);

    let pow = &a.pow(b.representative());
    let expected_pow = a_expected.pow(b_expected.representative());
    assert_eq!(pow.representative() as u64, expected_pow.representative().limbs[0]);
    
    if value_u32_b != 0 && b.inv().is_ok() && b_expected.inv().is_ok() {
        let div = &a / &b; 
        assert_eq!(&div * &b, a.clone());
        let expected_div = &a_expected / &b_expected;
        assert_eq!(div.representative() as u64, expected_div.representative().limbs[0]);
    }

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

    let one = FieldElement::<Mersenne31Field>::one();
    let zero = FieldElement::<Mersenne31Field>::zero();

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
        assert_eq!(&a * a.inv().unwrap(), one, "Inverse mul a failed");
    }
    if b != zero {
        assert_eq!(&b * b.inv().unwrap(), one, "Inverse mul b failed");
    }
});
