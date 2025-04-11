use const_random::const_random;
use lambdaworks_math::{
    field::{
        element::FieldElement,
        fields::{
            mersenne31::{extensions::Degree2ExtensionField, field::Mersenne31Field},
            u64_prime_field::{U64FieldElement, U64PrimeField},
        },
    },
    polynomial::{
        dense_multilinear_poly::DenseMultilinearPolynomial,
        sparse_multilinear_poly::SparseMultilinearPolynomial, Polynomial,
    },
};
use rand::random;

// Mersenne prime numbers
// https://www.math.utah.edu/~pa/math/mersenne.html
const PRIMES: [u64; 39] = [
    13, 17, 19, 31, 61, 89, 107, 127, 521, 607, 1279, 2203, 2281, 3217, 4253, 4423, 9689, 9941,
    11213, 19937, 21701, 23209, 44497, 86243, 110503, 132049, 216091, 756839, 859433, 1257787,
    1398269, 2976221, 3021377, 6972593, 13466917, 20996011, 24036583, 25964951, 30402457,
];

const MODULUS: u64 = PRIMES[const_random!(usize) % PRIMES.len()];
pub type FE = U64FieldElement<MODULUS>;

#[allow(dead_code)]
#[inline(never)]
#[export_name = "u64_utils::rand_field_elements"]
pub fn rand_field_elements(order: u64) -> Vec<FE> {
    let mut result = Vec::with_capacity(1 << order);
    for _ in 0..result.capacity() {
        result.push(FE::new(random()));
    }
    result
}

#[allow(dead_code)]
#[inline(never)]
#[export_name = "u64_utils::rand_poly"]
pub fn rand_poly(order: u64) -> Polynomial<FE> {
    Polynomial::new(&rand_field_elements(order))
}
#[allow(dead_code)]
#[inline(never)]
#[export_name = "u64_utils::rand_complex_mersenne_field_elements"]
pub fn rand_complex_mersenne_field_elements(
    order: u32,
) -> Vec<FieldElement<Degree2ExtensionField>> {
    let mut result = Vec::with_capacity(1 << order);
    for _ in 0..result.capacity() {
        result.push(FieldElement::<Degree2ExtensionField>::new([
            FieldElement::<Mersenne31Field>::new(random()),
            FieldElement::<Mersenne31Field>::new(random()),
        ]));
    }
    result
}

#[allow(dead_code)]
#[inline(never)]
#[export_name = "u64_utils::rand_complex_mersenne_poly"]
pub fn rand_complex_mersenne_poly(order: u32) -> Polynomial<FieldElement<Degree2ExtensionField>> {
    Polynomial::new(&rand_complex_mersenne_field_elements(order))
}

#[allow(dead_code)]
#[inline(never)]
#[export_name = "u64_utils::rand_dense_multilinear_poly"]
pub fn rand_dense_multilinear_poly(
    order: u64,
) -> DenseMultilinearPolynomial<U64PrimeField<MODULUS>> {
    DenseMultilinearPolynomial::new(rand_field_elements(order))
}

#[allow(dead_code)]
#[inline(never)]
#[export_name = "u64_utils::rand_sparse_multilinear_poly"]
pub fn rand_sparse_multilinear_poly(
    num_vars: usize,
    order: u64,
) -> SparseMultilinearPolynomial<U64PrimeField<MODULUS>> {
    let evals = rand_field_elements(order)
        .into_iter()
        .map(|eval| (random(), eval))
        .collect::<Vec<_>>();
    SparseMultilinearPolynomial::new(num_vars, evals)
}
