//! Common types and utilities for Groth16.
//!
//! This module defines the concrete curve types used by the Groth16 implementation.
//! Currently configured for the BLS12-381 pairing-friendly curve.
//!
//! # Using a Different Curve
//!
//! To use Groth16 with a different pairing-friendly curve (e.g., BN254):
//!
//! 1. Change the imports to your target curve
//! 2. Update all the type aliases below to match
//! 3. Ensure your curve's scalar field has FFT-friendly properties
//!
//! The implementation requires:
//! - A pairing-friendly curve with types `Curve`, `TwistedCurve`, and `Pairing`
//! - A scalar field `FrElement` that supports FFT operations
//! - The scalar field must have a generator for the multiplicative group

use lambdaworks_math::{
    elliptic_curve::{
        short_weierstrass::curves::bls12_381::{
            curve::BLS12381Curve, default_types::FrElement as FE, default_types::FrField as FrF,
            pairing::BLS12381AtePairing, twist::BLS12381TwistCurve,
        },
        traits::{IsEllipticCurve, IsPairing},
    },
    field::element::FieldElement,
    unsigned_integer::element::U256,
};
use rand::{Rng, SeedableRng};

/// The main curve G1 (BLS12-381).
pub type Curve = BLS12381Curve;

/// The twist curve G2 (BLS12-381 twist).
pub type TwistedCurve = BLS12381TwistCurve;

/// Scalar field element (Fr of BLS12-381).
pub type FrElement = FE;

/// Scalar field type (Fr of BLS12-381).
pub type FrField = FrF;

/// The pairing operation (BLS12-381 ate pairing).
pub type Pairing = BLS12381AtePairing;

/// Point on the G1 curve (BLS12-381 base curve).
pub type G1Point = <BLS12381Curve as IsEllipticCurve>::PointRepresentation;

/// Point on the G2 curve (BLS12-381 twist curve).
pub type G2Point = <BLS12381TwistCurve as IsEllipticCurve>::PointRepresentation;

/// Output of the pairing operation (element in Fq12).
pub type PairingOutput = FieldElement<<Pairing as IsPairing>::OutputField>;

/// Generator of the multiplicative group of Fr.
///
/// The multiplicative group is obtained by taking powers of this element:
/// `{w^0, w, w^2, ..., w^{r-2}} = Fr \ {0}`
///
/// This is used for FFT operations on the scalar field.
pub const ORDER_R_MINUS_1_ROOT_UNITY: FrElement = FrElement::from_hex_unchecked("7");

/// Samples a random element in the scalar field Fr.
///
/// Uses `ChaCha20Rng` seeded from system entropy for cryptographic randomness.
/// This is used during proof generation for the blinding factors.
pub fn sample_fr_elem() -> FrElement {
    let mut rng = rand_chacha::ChaCha20Rng::from_entropy();
    FrElement::new(U256 { limbs: rng.gen() })
}
