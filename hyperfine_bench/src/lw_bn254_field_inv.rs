use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bn_254::field_extension::BN254PrimeField,
    field::element::FieldElement,
};

fn main() {
    let a = FieldElement::<BN254PrimeField>::from(0xDEADBEEFu64);

    for _ in 0..1_000 {
        std::hint::black_box(std::hint::black_box(&a).inv().unwrap());
    }
}
