use lambdaworks_crypto::commitments::kzg::KateZaveruchaGoldberg;
use lambdaworks_math::unsigned_integer::element::U256;
use lambdaworks_math::{
    elliptic_curve::short_weierstrass::curves::bls12_381::{
        curve::BLS12381Curve,
        default_types::{FrElement as FE, FrField},
        pairing::BLS12381AtePairing,
        twist::BLS12381TwistCurve,
    },
    elliptic_curve::traits::{IsEllipticCurve, IsPairing},
    field::{element::FieldElement, traits::IsFFTField},
};
use rand::Rng;

pub type Curve = BLS12381Curve;
pub type TwistedCurve = BLS12381TwistCurve;

pub type FrElement = FE;

pub type Pairing = BLS12381AtePairing;

pub type KZG = KateZaveruchaGoldberg<FrField, Pairing>;

pub type G1Point = <BLS12381Curve as IsEllipticCurve>::PointRepresentation;
pub type G2Point = <BLS12381TwistCurve as IsEllipticCurve>::PointRepresentation;
pub type PairingOutput = FieldElement<<Pairing as IsPairing>::OutputField>;

pub fn sample_fr_elem() -> FrElement {
    let mut rng = rand::thread_rng();
    FrElement::new(U256 {
        limbs: [
            rng.gen::<u64>(),
            rng.gen::<u64>(),
            rng.gen::<u64>(),
            rng.gen::<u64>(),
        ],
    })
}

/// Generates a domain to interpolate: 1, omega, omega², ..., omega^size
pub fn generate_domain(number_of_gates: usize) -> Vec<FrElement> {
    let omega =
        FrField::get_primitive_root_of_unity(number_of_gates.trailing_zeros() as u64).unwrap();
    (1..number_of_gates).fold(vec![FieldElement::one()], |mut acc, _| {
        acc.push(acc.last().unwrap() * &omega);
        acc
    })
}
