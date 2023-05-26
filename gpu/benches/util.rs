use lambdaworks_math::{
    field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    },
    polynomial::Polynomial,
    unsigned_integer::element::UnsignedInteger,
};
use rand::{distributions::Standard, prelude::Distribution, random, Rng};

pub type F = Stark252PrimeField;
pub type FE = FieldElement<F>;

pub fn rand_vec<T>(order: u64) -> Vec<T>
where
    Standard: Distribution<T>,
{
    let mut rng = rand::thread_rng();
    let mut result = Vec::with_capacity(1 << order);

    for _ in 0..result.capacity() {
        result.push(rng.gen());
    }

    result
}

pub fn rand_field_elements(order: u64) -> Vec<FE> {
    let mut result = Vec::with_capacity(1 << order);
    for _ in 0..result.capacity() {
        let rand_big = UnsignedInteger { limbs: random() };
        result.push(FE::new(rand_big));
    }
    result
}

pub fn rand_poly(order: u64) -> Polynomial<FE> {
    Polynomial::new(&rand_field_elements(order))
}
