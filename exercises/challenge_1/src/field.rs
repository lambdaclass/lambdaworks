use lambdaworks_math::field::{
    element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
};

pub type ChallengeElement = FieldElement<Stark252PrimeField>;
