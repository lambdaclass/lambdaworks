#![no_main]

use libfuzzer_sys::fuzz_target;
use lambdaworks_math::field::{
    element::FieldElement, 
    fields::mersenne31::mersenne31::Mersenne31Field,
};
use ibig::{modular::ModuloRing, UBig};
fuzz_target!(|values: (u32, u32)| {

    let (value_u64_a, value_u64_b) = values;
    let mersenne_prime = 
        UBig::from(2u32^32 - 1u32);
   
    let ring = ModuloRing::new(&mersenne_prime);

    let a =  FieldElement::<Mersenne31Field>::from(value_u64_a as u64);
    let b =  FieldElement::<Mersenne31Field>::from(value_u64_b as u64);

    let a_expected = ring.from(value_u64_a);
    let b_expected = ring.from(value_u64_b);

    let add_u64 = &a + &b;
    let addition = &a_expected + &b_expected;
    
    assert_eq!(add_u64.to_string(), addition.residue().to_string());

    let sub_u64 = &a - &b;
    let substraction = &a_expected - &b_expected;
    assert_eq!(sub_u64.to_string(), substraction.residue().to_string());
    
    let mul_u64 = &a * &b;
    let multiplication = &a_expected * &b_expected;
    assert_eq!(mul_u64.to_string(), multiplication.residue().to_string());

    let pow = &a.pow(b.representative());
    let expected_pow = a_expected.pow(&b_expected.residue());
    assert_eq!(pow.to_string(), expected_pow.residue().to_string());
    
    if value_u64_b != 0 {
        
        let div = &a / &b; 
        assert_eq!(&div * &b, a.clone());
        let expected_div = &a_expected / &b_expected;
        assert_eq!(&(div.to_string())[2..], expected_div.residue().in_radix(16).to_string());
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
