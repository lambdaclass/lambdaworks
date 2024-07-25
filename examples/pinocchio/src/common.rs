use lambdaworks_math::{
    elliptic_curve::{
        short_weierstrass::curves::bls12_381::{
            curve::BLS12381Curve, default_types::FrElement, default_types::FrField,
            pairing::BLS12381AtePairing, twist::BLS12381TwistCurve,
        },
        traits::IsEllipticCurve,
    },
    unsigned_integer::element::U256,
};
use rand::{Rng, SeedableRng};

pub type Curve = BLS12381Curve;
pub type TwistedCurve = BLS12381TwistCurve;

pub type FE = FrElement;
pub type F = FrField;

pub type Pairing = BLS12381AtePairing;

pub type G1Point = <BLS12381Curve as IsEllipticCurve>::PointRepresentation;
pub type G2Point = <BLS12381TwistCurve as IsEllipticCurve>::PointRepresentation;

pub const ORDER_R_MINUS_1_ROOT_UNITY: FE = FE::from_hex_unchecked("7");

pub fn sample_fr_elem() -> FE {
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
