#![no_main]
use libfuzzer_sys::fuzz_target;
use lambdaworks_math::{
    fft::polynomial::FFTPoly,
    gpu::metal::fft::polynomial::{evaluate_fft_metal, interpolate_fft_metal}, 
    polynomial::Polynomial,
    field::{
        fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
        element::FieldElement
    },
    unsigned_integer::element::UnsignedInteger
};


fuzz_target!(|values: (Vec<[u64;4]>, Vec<[u64;4]>)| {
    println!("{:?}", values);
    let (mut input_raw, mut twiddles_raw) = values;
    let mut inputs = Vec::new();
    let mut twiddles = Vec::new();

    if input_raw.len() == 0 {
        input_raw.push([1u64;4]);
    }

    for i in 0..input_raw.len() {
        let input_value = UnsignedInteger::<4>::from_limbs(input_raw[i]);
        inputs.push(FieldElement::<Stark252PrimeField>::from_raw(&input_value))
    }

    if twiddles_raw.len() == 0 {
        twiddles_raw.push([1u64;4]);
    }

    for i in 0..twiddles_raw.len() {
        let twiddle_value = UnsignedInteger::<4>::from_limbs(twiddles_raw[i]);
        twiddles.push(FieldElement::<Stark252PrimeField>::from_raw(&twiddle_value))
    }

    let polinomial_inputs =  Polynomial { coefficients: (*inputs).to_vec() };

    let evaluated_fields_cpu = polinomial_inputs.evaluate_fft(1, None);
    let evaluated_fields_metal = evaluate_fft_metal(&inputs);

    match evaluated_fields_cpu {
        Ok(ref evaluated_fields_cpu) => assert_eq!(evaluated_fields_metal.unwrap(), *evaluated_fields_cpu),
        Err(_) => assert!(evaluated_fields_metal.is_err())
    };

    let interpolated_poly = Polynomial::interpolate_fft(&evaluated_fields_cpu.as_ref().unwrap());
    let interpolated_fields_metal = interpolate_fft_metal(&evaluated_fields_cpu.unwrap());


    //let interpolated_fields_metal = interpolate_fft_metal(&evaluated_fields_metal).unwrap();

    //assert_eq!(interpolated_fields_metal, polinomial_inputs);
});

