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
use rand::Rng;
use rand_chacha::ChaCha20Rng;

/// We use the BN-254 Curve as the group. Any curve could be used.
pub type Curve = BN254Curve;

/// The scalar field Fr where r is the order of the curve group.
pub type F = FrField;

pub type FE = FrElement;

pub type CurvePoint = <BN254Curve as IsEllipticCurve>::PointRepresentation;

/// Sample a random field element using the provided RNG.
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
