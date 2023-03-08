use lambdaworks_math::{elliptic_curve::{short_weierstrass::curves::bls12_381::{pairing::BLS12381AtePairing}, traits::IsPairing}, cyclic_group::IsGroup};
use lambdaworks_crypto::commitments::kzg::{KateZaveruchaGoldberg, StructuredReferenceString};
use lambdaworks_math::{polynomial::Polynomial, field::{element::FieldElement, fields::montgomery_backed_prime_fields::{U256PrimeField, IsMontgomeryConfiguration}}, unsigned_integer::element::U256};

use crate::config::{FrElement, G1Point, G2Point, KZG};

pub struct Circuit;

impl Circuit {
    // TODO: This should execute the program and return the trace.
    pub fn get_program_trace_polynomials(&self) -> (Polynomial<FrElement>, Polynomial<FrElement>, Polynomial<FrElement>) {
        (
            Polynomial::new(&[FrElement::zero(), FrElement::one()]),
            Polynomial::new(&[FrElement::zero(), FrElement::one()]),
            Polynomial::new(&[FrElement::zero(), FrElement::one()])
        )
    }

    // TODO: This should interpolate the structure of the program into polynomials.
    pub fn get_common_preprocessed_input(&self) -> CommonPreprocessedInput {
        CommonPreprocessedInput {
            Qm: Polynomial::new(&[FrElement::zero(), FrElement::one()]),
            Ql: Polynomial::new(&[FrElement::zero(), FrElement::one()]),
            Qr: Polynomial::new(&[FrElement::zero(), FrElement::one()]),
            Qo: Polynomial::new(&[FrElement::zero(), FrElement::one()]),
            Qc: Polynomial::new(&[FrElement::zero(), FrElement::one()]),
            S1: Polynomial::new(&[FrElement::zero(), FrElement::one()]),
            S2: Polynomial::new(&[FrElement::zero(), FrElement::one()]),
            S3: Polynomial::new(&[FrElement::zero(), FrElement::one()]),
        }
    }
}

pub struct CommonPreprocessedInput {
    Qm: Polynomial<FrElement>,
    Ql: Polynomial<FrElement>,
    Qr: Polynomial<FrElement>,
    Qo: Polynomial<FrElement>,
    Qc: Polynomial<FrElement>,
    S1: Polynomial<FrElement>,
    S2: Polynomial<FrElement>,
    S3: Polynomial<FrElement>,
}

pub struct VerificationKey<G1Point> {
    Qm: G1Point,
    Ql: G1Point,
    Qr: G1Point,
    Qo: G1Point,
    Qc: G1Point,
    S1: G1Point,
    S2: G1Point,
    S3: G1Point,
}


#[allow(unused)]
fn setup<const MAXIMUM_DEGREE: usize, P: IsPairing> (
    srs: StructuredReferenceString<MAXIMUM_DEGREE, G1Point, G2Point>,
    circuit: Circuit
) -> VerificationKey<G1Point> {
    let kzg = KZG::new(srs);
    let common_input = circuit.get_common_preprocessed_input();

    VerificationKey {
        Qm: kzg.commit(&common_input.Qm),
        Ql: kzg.commit(&common_input.Ql),
        Qr: kzg.commit(&common_input.Qr),
        Qo: kzg.commit(&common_input.Qo),
        Qc: kzg.commit(&common_input.Qc),
        S1: kzg.commit(&common_input.S1),
        S2: kzg.commit(&common_input.S2),
        S3: kzg.commit(&common_input.S3),
    }
}
