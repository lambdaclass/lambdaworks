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

    // let c =  FieldElement::<Stark252PrimeField>::from(2u64);
    // let d =  FieldElement::<Stark252PrimeField>::from(5u64);
    // let e = c - d;
    // println!("resultado de sub {:?}", e.representative() );
    let d: UBig = "800000000000011000000000000000000000000000000000000000000000001".parse().unwrap();

    let ring = ModuloRing::new(&d);


    let a =  FieldElement::<Stark252PrimeField>::from(value_u64_a);
    let b =  FieldElement::<Stark252PrimeField>::from(value_u64_b);

    let add_u64 = &a + &b;
    let addition = ring.from(value_u64_a) + ring.from(value_u64_b);
    
    assert_eq!(&(add_u64.to_string())[2..], addition.residue().in_radix(16).to_string());

    let sub_u64 = &a - &b;
    let substraction = ring.from(value_u64_a) - ring.from(value_u64_b);
    assert_eq!(&(sub_u64.to_string())[2..], substraction.residue().in_radix(16).to_string());
    
    let mul_u64 = &a * &b;
    let multiplication = ring.from(value_u64_a) * ring.from(value_u64_b);
    assert_eq!(&(mul_u64.to_string())[2..], multiplication.residue().in_radix(16).to_string());

    let pow_u64 = &a.pow(value_u64_b.clone());
    // let pow = BigUint::from(value_u64_a).pow(value_u64_b);
    // assert_eq!(&(pow_u64.to_string())[2..], pow.residue().in_radix(16).to_string());
    
    if value_u64_b != 0 {
        let div_u64 = &a / &b;
        let division = ring.from(value_u64_a) / ring.from(value_u64_b);
        assert_eq!(&(div_u64.to_string())[2..], division.residue().in_radix(16).to_string());
    }
    
    
});
