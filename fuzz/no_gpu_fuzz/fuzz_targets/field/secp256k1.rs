#![no_main]

use libfuzzer_sys::fuzz_target;
use lambdaworks_math::field::{
    element::FieldElement
};
use ibig::{modular::ModuloRing, UBig};
use lambdaworks_math::traits::ByteConversion;
use lambdaworks_math::field::fields::montgomery_backed_prime_fields::U256PrimeField;
use lambdaworks_math::unsigned_integer::element::U256;
use lambdaworks_math::field::fields::montgomery_backed_prime_fields::IsModulus;

#[derive(Clone, Debug, Hash, Copy)]
pub struct MontgomeryConfigSecpPrimeField;

impl IsModulus<U256> for MontgomeryConfigSecpPrimeField {
    const MODULUS: U256 =
        U256::from_hex_unchecked("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F");
}

pub type SecpPrimeField = U256PrimeField<MontgomeryConfigSecpPrimeField>;

fuzz_target!(|bytes: ([u8;32], [u8;32])| {

    let secp256k1_prime = 
        UBig::from_str_radix("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F", 16).unwrap();
   
    let secp256k1_ring_prime = ModuloRing::new(&secp256k1_prime);

    let (bytes_a, bytes_b) = bytes;
    let a = FieldElement::<SecpPrimeField>::from_bytes_be(&bytes_a).unwrap();
    let b = FieldElement::<SecpPrimeField>::from_bytes_be(&bytes_b).unwrap();

    let a_hex = a.to_string()[2..].to_string();
    let b_hex = b.to_string()[2..].to_string();

    let a_ring = secp256k1_ring_prime.from(&UBig::from_str_radix(&a_hex, 16).unwrap());
    let b_ring = secp256k1_ring_prime.from(&UBig::from_str_radix(&b_hex, 16).unwrap());

    let add = &a + &b;
    let addition = &a_ring + &b_ring;
    
    assert_eq!(&(add.to_string())[2..], addition.residue().in_radix(16).to_string());

    let sub = &a - &b;
    let substraction = &a_ring - &b_ring;
    assert_eq!(&(sub.to_string())[2..], substraction.residue().in_radix(16).to_string());
    
    let mul = &a * &b;
    let multiplication = &a_ring * &b_ring;
    assert_eq!(&(mul.to_string())[2..], multiplication.residue().in_radix(16).to_string());

    let pow = &a.pow(b.representative());
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

    let one = FieldElement::<SecpPrimeField>::one();
    let zero = FieldElement::<SecpPrimeField>::zero();

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
