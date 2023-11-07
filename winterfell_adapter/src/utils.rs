use lambdaworks_math::field::fields::fft_friendly::stark_252_prime_field::Stark252PrimeField;
use stark_platinum_prover::fri::FieldElement;
use crate::field_element::field_element::AdapterFieldElement;


pub fn vec_field2adapter(input: &[FieldElement<Stark252PrimeField>]) -> Vec<AdapterFieldElement> {
    input.iter().map(|&e| AdapterFieldElement(e)).collect()
}

pub fn vec_adapter2field(input: &[AdapterFieldElement]) -> Vec<FieldElement<Stark252PrimeField>> {
    input.iter().map(|&e| e.0).collect()
}

pub fn matrix_field2adapter(input: &[Vec<FieldElement<Stark252PrimeField>>]) -> Vec<Vec<AdapterFieldElement>> {
    input.iter().map(|v| vec_field2adapter(&v)).collect()
}

pub fn matrix_adapter2field(input: &[Vec<AdapterFieldElement>]) -> Vec<Vec<FieldElement<Stark252PrimeField>>> {
    input.iter().map(|v| vec_adapter2field(&v)).collect()
}
