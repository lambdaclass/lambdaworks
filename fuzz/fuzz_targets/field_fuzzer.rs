#![no_main]

use libfuzzer_sys::fuzz_target;
use lambdaworks_math::field::{
    element::FieldElement, 
    fields::fft_friendly::stark_252_prime_field::Stark252PrimeField
};
use num_traits::Pow;
use lambdaworks_math::unsigned_integer::element::UnsignedInteger;
use std::fmt::Display;
use ibig::{ibig, modular::ModuloRing, ubig, UBig};
use num_bigint::BigUint;

fuzz_target!(|values: (u64, u64)| {

    let (value_u64_a, value_u64_b) = values;

    let cairo_prime = 
        UBig::from_str_radix("800000000000011000000000000000000000000000000000000000000000001", 16).unwrap();
   
    let ring = ModuloRing::new(&cairo_prime);


    let a =  FieldElement::<Stark252PrimeField>::from(value_u64_a);
    let b =  FieldElement::<Stark252PrimeField>::from(value_u64_b);
    println!();

    let a_expected = ring.from(value_u64_a);
    let b_expected = ring.from(value_u64_b);

    let add_u64 = &a + &b;
    let addition = &a_expected + &b_expected;
    
    assert_eq!(&(add_u64.to_string())[2..], addition.residue().in_radix(16).to_string());

    let sub_u64 = &a - &b;
    let substraction = &a_expected - &b_expected;
    assert_eq!(&(sub_u64.to_string())[2..], substraction.residue().in_radix(16).to_string());
    
    let mul_u64 = &a * &b;
    let multiplication = &a_expected * &b_expected;
    assert_eq!(&(mul_u64.to_string())[2..], multiplication.residue().in_radix(16).to_string());

    let pow = &a.pow(b.representative());
    let expected_pow = a_expected.pow(&b_expected.residue());
    assert_eq!(&(pow.to_string())[2..], expected_pow.residue().in_radix(16).to_string());
    
    if value_u64_b != 0 {
        
        let div = &a / &b; 
        assert_eq!(&div * &b, a.clone());
        let expected_div = &a_expected / &b_expected;
        assert_eq!(&(div.to_string())[2..], expected_div.residue().in_radix(16).to_string());
    }
    
    
});
