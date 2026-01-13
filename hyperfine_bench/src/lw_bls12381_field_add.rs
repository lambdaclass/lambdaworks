use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::field_extension::BLS12381PrimeField,
    field::element::FieldElement,
};

fn main() {
    let a = FieldElement::<BLS12381PrimeField>::from(0xDEADBEEFu64);
    let b = FieldElement::<BLS12381PrimeField>::from(0xCAFEBABEu64);

    for _ in 0..100_000 {
        std::hint::black_box(std::hint::black_box(&a) + std::hint::black_box(&b));
    }
}
