use lambdaworks_math::{
    elliptic_curve::{
        short_weierstrass::curves::bls12_381::{
            curve::BLS12381Curve,
            default_types::{FrElement, FrField},
        },
        traits::IsEllipticCurve,
    },
    field::element::FieldElement,
    unsigned_integer::element::U256,
};
use rand::{Rng, SeedableRng};

pub type Curve = BLS12381Curve;

pub type F = FrField;

pub type FE = FrElement;

pub type CurvePoint = <BLS12381Curve as IsEllipticCurve>::PointRepresentation;

pub fn sample_field_elem() -> FE {
    let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(9001);
    FE::new(U256 {
        limbs: [
            rng.gen::<u64>(),
            rng.gen::<u64>(),
            rng.gen::<u64>(),
            rng.gen::<u64>(),
        ],
    })
}
