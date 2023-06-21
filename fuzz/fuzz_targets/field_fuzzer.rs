#![no_main]

use libfuzzer_sys::fuzz_target;
use lambdaworks_math::field::{
    element::FieldElement, 
    fields::fft_friendly::stark_252_prime_field::Stark252PrimeField
};


fuzz_target!(|values: (u64, u64)| {

    let (value_u64_a, value_u64_b) = values;

    let a =  FieldElement::<Stark252PrimeField>::from(value_u64_a);
    let b =  FieldElement::<Stark252PrimeField>::from(value_u64_b);
    let mul_u64 = &a * &b;
    if value_u64_b != 0 {let div_u64 = &a / &b;}
    let add_u64 = &a + &b;
    let sub = &a - b;
    let pow_u64 = a.pow(value_u64_b);
});
