use ark_ff::BigInt;
use ark_std::UniformRand;
use ark_test_curves::starknet_fp::Fq;
use lambdaworks_math::{
    field::{
        element::FieldElement, fields::fft_friendly::stark_252_prime_field::Stark252PrimeField,
    },
    unsigned_integer::element::UnsignedInteger,
};
use rand::SeedableRng;

/// Creates `amount` random elements
pub fn generate_random_elements(amount: u64) -> Vec<Fq> {
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(9001);
    let mut arkworks_vec = Vec::new();
    for _i in 0..amount {
        let a = Fq::rand(&mut rng);
        arkworks_vec.push(a);
    }

    arkworks_vec
}

pub fn to_lambdaworks_vec(arkworks_vec: &[Fq]) -> Vec<FieldElement<Stark252PrimeField>> {
    let mut lambdaworks_vec = Vec::new();
    for &arkworks_felt in arkworks_vec {
        let big_int: BigInt<4> = arkworks_felt.into();
        let mut limbs = big_int.0;
        limbs.reverse();

        let a: FieldElement<Stark252PrimeField> = FieldElement::from(&UnsignedInteger { limbs });

        assert_eq!(a.representative().limbs, limbs);

        lambdaworks_vec.push(a);
    }

    lambdaworks_vec
}
