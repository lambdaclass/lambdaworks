use lambdaworks_math::field::{
    element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};
use rand::{distributions::Standard, prelude::Distribution, Rng};

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
