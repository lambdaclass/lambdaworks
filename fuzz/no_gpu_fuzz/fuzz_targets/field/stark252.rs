#![no_main]

use libfuzzer_sys::fuzz_target;
use lambdaworks_math::field::{
    element::FieldElement, 
    fields::fft_friendly::stark_252_prime_field::Stark252PrimeField
};
use ibig::{modular::ModuloRing, UBig};
use lambdaworks_math::traits::ByteConversion;

fuzz_target!(|bytes: ([u8;32], [u8;32])| {

    let cairo_prime = 
        UBig::from_str_radix("800000000000011000000000000000000000000000000000000000000000001", 16).unwrap();
   
    let stark252_ring_prime = ModuloRing::new(&cairo_prime);

    let (bytes_a, bytes_b) = bytes;
    let a = FieldElement::from_bytes_be(&bytes_a).unwrap();
    let b = FieldElement::from_bytes_be(&bytes_b).unwrap();

    let a_hex = a.to_string()[2..].to_string();
    let b_hex = b.to_string()[2..].to_string();

    let a_ring = stark252_ring_prime.from(&UBig::from_str_radix(&a_hex, 16).unwrap());
    let b_ring = stark252_ring_prime.from(&UBig::from_str_radix(&b_hex, 16).unwrap());

    let add = &a + &b;
    let addition = &a_ring + &b_ring;
    
    assert_eq!(&(add.to_string())[2..], addition.residue().in_radix(16).to_string());

    let sub = &a - &b;
    let substraction = &a_ring - &b_ring;
    assert_eq!(&(sub.to_string())[2..], substraction.residue().in_radix(16).to_string());
    
    let mul = &a * &b;
    let multiplication = &a_ring * &b_ring;
    assert_eq!(&(mul.to_string())[2..], multiplication.residue().in_radix(16).to_string());

    let pow = &a.pow(b.canonical());
    let expected_pow = a_ring.pow(&b_ring.residue());
    assert_eq!(&(pow.to_string())[2..], expected_pow.residue().in_radix(16).to_string());
    
    if b != FieldElement::zero() {
        
        let div = &a / &b; 
        assert_eq!(&div * &b, a.clone());
        let expected_div = &a_ring / &b_ring;
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
        assert_eq!(&a * a.inv().unwrap(), one, "Inverse mul a failed");
    }
    if b != zero {
        assert_eq!(&b * b.inv().unwrap(), one, "Inverse mul b failed");
    }
});
