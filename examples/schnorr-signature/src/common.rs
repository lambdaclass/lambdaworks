use lambdaworks_math::{
    elliptic_curve::{
        short_weierstrass::curves::bn_254::{
            curve::BN254Curve,
            default_types::{FrElement, FrField},
        },
        traits::IsEllipticCurve,
    },
    unsigned_integer::element::U256,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

// We use the BN-254 Curve as the group. We could have used any other curve.
pub type Curve = BN254Curve;

// We use the finite field Fr where r is the number of elements that the curve has,
// i.e. r is the order of the group G form by the curve.
pub type F = FrField;

pub type FE = FrElement;

pub type CurvePoint = <BN254Curve as IsEllipticCurve>::PointRepresentation;

pub fn sample_field_elem(mut rng: ChaCha20Rng) -> FE {
    FE::new(U256 {
        limbs: [
            rng.gen::<u64>(),
            rng.gen::<u64>(),
            rng.gen::<u64>(),
            rng.gen::<u64>(),
        ],
    })
}
