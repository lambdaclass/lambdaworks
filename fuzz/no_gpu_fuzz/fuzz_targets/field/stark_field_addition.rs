#![no_main]

use libfuzzer_sys::fuzz_target;
use lambdaworks_math::field::{
    element::FieldElement, 
    fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};

use lambdaworks_math::traits::ByteConversion;

type FE = FieldElement<Stark252PrimeField>;

use ibig::{modular::ModuloRing, UBig};

fuzz_target!(|bytes: ([u8;32], [u8;32])| {
    let (bytes_a, bytes_b) = bytes;
    let a = FE::from_bytes_be(&bytes_a).unwrap();
    let b = FE::from_bytes_be(&bytes_b).unwrap();

    let a_hex = a.representative().to_string()[2..].to_string();
    let b_hex = b.representative().to_string()[2..].to_string();

    let c = a + &b;
    let c_hex = c.representative().to_string()[2..].to_string();

    let prime = 
    UBig::from_str_radix("800000000000011000000000000000000000000000000000000000000000001", 16).unwrap();
    let cairo_ring = ModuloRing::new(&prime);

    let a_ring = cairo_ring.from(&UBig::from_str_radix(&a_hex, 16).unwrap());
    let b_ring = cairo_ring.from(&UBig::from_str_radix(&b_hex, 16).unwrap());
    let expected_c = a_ring + &b_ring;
    let expected_c_hex = expected_c.residue().in_radix(16).to_string();

    assert_eq!(expected_c_hex, c_hex);
});

