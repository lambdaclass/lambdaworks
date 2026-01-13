use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bn_254::field_extension::BN254PrimeField,
    field::element::FieldElement,
};

fn main() {
    let a = FieldElement::<BN254PrimeField>::from(0xDEADBEEFu64);
    let b = FieldElement::<BN254PrimeField>::from(0xCAFEBABEu64);

    for _ in 0..100_000 {
        std::hint::black_box(std::hint::black_box(&a) - std::hint::black_box(&b));
    }
}
