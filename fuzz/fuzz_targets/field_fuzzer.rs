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


    // let sub_u64 = a - b;
    // if value_u64_a.clone() > value_u64_b.clone() { 
    //     let substraction = value_u64_a.clone() - value_u64_b.clone();
    //     assert_eq!(sub_u64, FieldElement::<Stark252PrimeField>::from(substraction));
    // } else { 
    //     let substraction = value_u64_b.clone() - value_u64_a.clone();
    //     assert_eq!(sub_u64, -FieldElement::<Stark252PrimeField>::from(substraction));
    // }

    // let pow_u64 = &a.pow(value_u64_b.clone());
    // // let pow = value_u64_a.clone().pow(value_u64_b.clone());
    // // assert_eq!(pow_u64, FieldElement::<Stark252PrimeField>::from(pow));
    
    // let mul_u64 = &a * &b;
    // let mut multiplication = u128::from(value_u64_a.clone()) * u128::from(value_u64_b.clone());
    // let multiplication: u64 = multiplication as u64;
    // println!(" unsignedd{:?}", value_u64_a.clone());
    // println!(" unsigneddd{:?}", value_u64_b.clone());
    // println!(" multiplication{:?}", multiplication);
    // // multiplication fails result
    // // assert_eq!(mul_u64, FieldElement::<Stark252PrimeField>::from(multiplication ));
    // if value_u64_b != 0 {
    //     let division = value_u64_a.clone() / value_u64_b.clone();
    //     let div_u64 = &a / &b;
    //     // divition fails result
    //     // assert_eq!(div_u64, FieldElement::<Stark252PrimeField>::from(division));
    // }
    
    
});
