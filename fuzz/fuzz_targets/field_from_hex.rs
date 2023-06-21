#![no_main]

use libfuzzer_sys::fuzz_target;
use lambdaworks_math::field::{
    element::FieldElement, 
    fields::fft_friendly::stark_252_prime_field::Stark252PrimeField
};
use ibig::{modular::ModuloRing, UBig};

fuzz_target!(|values: (String, String)| {
    let (value_a, value_b) = values;
    let cairo_prime = 
        UBig::from_str_radix("800000000000011000000000000000000000000000000000000000000000001", 16).unwrap();
    let a_parsed = UBig::from_str_radix(&value_a, 16);
    let b_parsed = UBig::from_str_radix(&value_b, 16);

    if value_a.len() < 64 && value_b.len() < 64 && 
       value_a.chars().all(|c| matches!(c, '0'..='9' | 'a'..='z' |'A'..='Z')) &&
       value_b.chars().all(|c| matches!(c, '0'..='9' | 'a'..='z' |'A'..='Z')) &&
       a_parsed.is_ok() && b_parsed.is_ok() && 
       a_parsed.as_ref().unwrap() < &cairo_prime && b_parsed.as_ref().unwrap() < &cairo_prime
    {

        let a = FieldElement::<Stark252PrimeField>::from_hex_unchecked(&value_a);
        let b = FieldElement::<Stark252PrimeField>::from_hex_unchecked(&value_b);

        let cairo_ring = ModuloRing::new(&cairo_prime);
        let a_expected = cairo_ring.from(a_parsed.unwrap());
        let b_expected = cairo_ring.from(b_parsed.unwrap());

        let add = &a + &b;
        let expected_add = &a_expected + &b_expected;
        assert_eq!(&(add.to_string())[2..], expected_add.residue().in_radix(16).to_string());

        let sub = &a - &b;
        let expected_sub = &a_expected - &b_expected;
        assert_eq!(&(sub.to_string())[2..], expected_sub.residue().in_radix(16).to_string());

        let mul = &a * &b;
        let expected_mul = &a_expected * &b_expected;
        assert_eq!(&(mul.to_string())[2..], expected_mul.residue().in_radix(16).to_string());

        if !value_b.chars().all(|c| matches!(c, '0')) { 
            let div = &a / &b; 
            let expected_div = &a_expected / &b_expected;
            assert_eq!(&(div.to_string())[2..], expected_div.residue().in_radix(16).to_string());
        }
    }
});

